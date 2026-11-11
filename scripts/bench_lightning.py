"""Benchmark: modelo base × LoRA Lightning (destilação de poucos steps), com mais de um
prompt. Carrega o pipeline em bf16 na CPU uma vez e, pra cada configuração, monta um
transformer novo (cópia do base + LoRA fundido, se houver), quantiza em fp8 (mesmo
Float8WeightOnlyConfig do servidor) e manda pra GPU -- assim não precisa recarregar os
54 GB do volume a cada configuração.

O LoRA é fundido ANTES da quantização: fundir num transformer já quantizado em fp8 pelo
torchao não é suportado de forma confiável.

Configuração do Lightning conforme o exemplo oficial (ModelTC/Qwen-Image-Lightning,
generate_with_diffusers.py): FlowMatchEulerDiscreteScheduler com shift=3 fixo e
true_cfg_scale=1.0 (sem CFG -> 1 passada por step em vez de 2).

Uso (no pod, GPU livre, na raiz do repo):
    PYTHONPATH=. /root/venv/bin/python scripts/bench_lightning.py entrada saida \
        --lora-dir /workspace/models/lightning

Saída: <saida>/<config>__<prompt>/<imagem>.jpg e <saida>/resultados.csv
"""

import argparse
import copy
import csv
import gc
import math
import os
import time
from pathlib import Path

import torch
from PIL import Image

from server import image_io

MODEL_DIR = os.environ.get('MODEL_DIR', '/workspace/models/Qwen-Image-Edit-2511')
MAX_SIDE = int(os.environ.get('MAX_SIDE', '1536'))
EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

PROMPTS = {
    # DEFAULT_PROMPT do Render em produção (todes_render é o gatilho do LoRA do Kontext,
    # não significa nada pro Qwen -- mantido pra comparar com o que o Render manda).
    'render': (
        'todes_render,Adds photorealism while keeping your colors and materials unchanged. '
        'Best for textured designs.'
    ),
    'estrito': (
        'Make this image look like a real photograph. Only change lighting, shadows, '
        'reflections and surface realism. Do not change any color, material or finish: '
        'white stays white, every surface keeps its exact original color and texture. '
        'Do not add or remove any object, window, door, plant or decoration. Keep the exact '
        'same furniture, layout and camera angle.'
    ),
}

LIGHTNING_SCHEDULER = {
    'base_image_seq_len': 256,
    'base_shift': math.log(3),
    'invert_sigmas': False,
    'max_image_seq_len': 8192,
    'max_shift': math.log(3),
    'num_train_timesteps': 1000,
    'shift': 1.0,
    'shift_terminal': None,
    'stochastic_sampling': False,
    'time_shift_type': 'exponential',
    'use_beta_sigmas': False,
    'use_dynamic_shifting': True,
    'use_exponential_sigmas': False,
    'use_karras_sigmas': False,
}


def configs(lora_dir: Path) -> list[dict]:
    return [
        {'nome': 'base_16steps', 'lora': None, 'steps': 16, 'cfg': 4.0},
        {'nome': 'base_24steps', 'lora': None, 'steps': 24, 'cfg': 4.0},
        {
            'nome': 'lightning4_4steps',
            'lora': lora_dir / 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors',
            'steps': 4,
            'cfg': 1.0,
        },
        {
            'nome': 'lightning8_8steps',
            'lora': lora_dir / 'Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors',
            'steps': 8,
            'cfg': 1.0,
        },
    ]


def main() -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--lora-dir', type=Path, default=Path('/workspace/models/lightning'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in EXTS)
    loaded = {p: image_io.limit_size(Image.open(p).convert('RGB'), MAX_SIDE) for p in images}

    t0 = time.monotonic()
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    base_transformer = pipe.transformer  # bf16, CPU -- nunca modificado
    base_scheduler = pipe.scheduler
    lightning_scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
    # Tudo menos o transformer fica na GPU o tempo todo.
    pipe.transformer = None
    pipe.to('cuda')
    print(f'Carga (bf16, CPU): {time.monotonic() - t0:.0f}s', flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'resultados.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'prompt', 'steps', 'cfg', 'imagem', 'largura', 'altura', 'segundos'])

        for cfg in configs(args.lora_dir):
            t = time.monotonic()
            transformer = copy.deepcopy(base_transformer)
            pipe.transformer = transformer
            if cfg['lora']:
                # Funde o LoRA no transformer bf16 (CPU) e descarta os adapters.
                pipe.load_lora_weights(str(cfg['lora']))
                pipe.fuse_lora()
                pipe.unload_lora_weights()
                pipe.scheduler = lightning_scheduler
            else:
                pipe.scheduler = base_scheduler
            quantize_(transformer, Float8WeightOnlyConfig())
            transformer.to('cuda')
            print(f"[{cfg['nome']}] preparo: {time.monotonic() - t:.0f}s", flush=True)

            # Aquecimento (não medido).
            pipe(
                image=[loaded[images[0]]], prompt='photo', negative_prompt=' ',
                true_cfg_scale=cfg['cfg'], guidance_scale=1.0, num_inference_steps=2,
            )

            for prompt_nome, prompt in PROMPTS.items():
                out_dir = args.output_dir / f"{cfg['nome']}__{prompt_nome}"
                out_dir.mkdir(exist_ok=True)
                for path in images:
                    img = loaded[path]
                    generator = torch.Generator('cuda').manual_seed(args.seed)
                    torch.cuda.synchronize()
                    t = time.monotonic()
                    with torch.inference_mode():
                        out = pipe(
                            image=[img],
                            prompt=prompt,
                            negative_prompt=' ',
                            true_cfg_scale=cfg['cfg'],
                            guidance_scale=1.0,
                            num_inference_steps=cfg['steps'],
                            generator=generator,
                        ).images[0]
                    torch.cuda.synchronize()
                    secs = time.monotonic() - t
                    out.save(out_dir / f'{path.stem}.jpg', quality=95)
                    writer.writerow([
                        cfg['nome'], prompt_nome, cfg['steps'], cfg['cfg'], path.name,
                        img.width, img.height, f'{secs:.1f}',
                    ])
                    f.flush()
                    print(f"[{cfg['nome']}] {prompt_nome} {path.name}: {secs:.1f}s", flush=True)

            pipe.transformer = None
            del transformer
            gc.collect()
            torch.cuda.empty_cache()

    print(f'Pronto: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
