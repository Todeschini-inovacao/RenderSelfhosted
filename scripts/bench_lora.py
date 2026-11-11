"""Compara checkpoints / forças / prompts de um LoRA treinado (ai-toolkit) no conjunto de
validação.

Carrega o pipeline bf16 uma vez; pra cada configuração copia o transformer, funde os LoRAs
(ANTES de quantizar em fp8, como no bench_lightning.py), quantiza e renderiza todas as
imagens com a mesma seed. A fusão e a quantização são feitas na GPU (o text encoder desce
pra CPU nesse meio-tempo pra caber), o que é bem mais rápido que na CPU.

Formato de cada configuração:
    base                                   sem LoRA
    lora.safetensors                       força 1.0
    lora.safetensors@0.7                   força 0.7
    a.safetensors+b.safetensors@0.8        vários LoRAs (a força vale pro último)
    ...#render,estrito                     prompts dessa config (padrão: --prompts)

Uso (no pod, GPU livre, na raiz do repo):
    PYTHONPATH=. /root/venv/bin/python scripts/bench_lora.py val/control saida \
        base ckpt_2500.safetensors@0.8 --prompts render estrito --steps 16 --cfg 4

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
    # DEFAULT_PROMPT do Render em produção (também é a legenda do treino).
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


def parse(spec: str, default_prompts: list[str]) -> dict:
    prompts = default_prompts
    if '#' in spec:
        spec, p = spec.split('#', 1)
        prompts = p.split(',')
    scale = 1.0
    if '@' in spec:
        spec, s = spec.rsplit('@', 1)
        scale = float(s)
    loras = [] if spec == 'base' else spec.split('+')
    names = []
    for p in loras:
        stem = Path(p).stem
        tail = stem.rsplit('_', 1)[-1]
        if 'lightning' in stem.lower():
            names.append('lightning')
        elif tail.isdigit():
            names.append(f'ckpt_{int(tail)}')
        elif tail.startswith('v') and tail[1:].isdigit():  # checkpoint final do ai-toolkit
            names.append('ckpt_final')
        else:
            names.append(stem)
    # Lightning (destilação de poucos steps): 4 ou 8 steps conforme o arquivo, sem CFG e
    # com o scheduler próprio -- igual ao bench_lightning.py.
    lightning = next((p for p in loras if 'lightning' in Path(p).stem.lower()), None)
    steps = None
    if lightning:
        steps = 8 if '8steps' in Path(lightning).stem.lower() else 4
        names = [f'lightning{steps}' if n == 'lightning' else n for n in names]
    nome = '+'.join(names) if names else 'base'
    if loras and scale != 1.0:
        nome += f'@{scale:g}'
    return {
        'nome': nome, 'loras': loras, 'scale': scale, 'prompts': prompts,
        'lightning': bool(lightning), 'steps': steps,
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


def main() -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--prompts', nargs='+', default=['render'], choices=list(PROMPTS))
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--cfg', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    configs = [parse(c, args.prompts) for c in args.configs]
    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in EXTS)
    loaded = {p: image_io.limit_size(Image.open(p).convert('RGB'), MAX_SIDE) for p in images}

    t0 = time.monotonic()
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    base_transformer = pipe.transformer  # bf16, CPU -- nunca modificado
    base_scheduler = pipe.scheduler
    lightning_scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
    pipe.transformer = None
    pipe.to('cuda')
    print(f'Carga (bf16, CPU): {time.monotonic() - t0:.0f}s', flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'resultados.csv'
    new = not csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(['config', 'prompt', 'steps', 'cfg', 'imagem', 'largura', 'altura', 'segundos'])

        for cfg in configs:
            t = time.monotonic()
            # Text encoder sai da GPU pra caber o transformer em bf16 (~41 GB) durante a fusão.
            pipe.text_encoder.to('cpu')
            torch.cuda.empty_cache()
            transformer = copy.deepcopy(base_transformer).to('cuda')
            pipe.transformer = transformer
            for i, lora in enumerate(cfg['loras']):
                pipe.load_lora_weights(lora)
                last = i == len(cfg['loras']) - 1
                pipe.fuse_lora(lora_scale=cfg['scale'] if last else 1.0)
                pipe.unload_lora_weights()
            quantize_(transformer, Float8WeightOnlyConfig())
            gc.collect()
            torch.cuda.empty_cache()
            pipe.text_encoder.to('cuda')
            pipe.scheduler = lightning_scheduler if cfg['lightning'] else base_scheduler
            steps = cfg['steps'] or args.steps
            true_cfg = 1.0 if cfg['lightning'] else args.cfg
            print(f"[{cfg['nome']}] preparo: {time.monotonic() - t:.0f}s", flush=True)

            for prompt_nome in cfg['prompts']:
                out_dir = args.output_dir / f"{cfg['nome']}__{prompt_nome}"
                out_dir.mkdir(exist_ok=True)
                for path in images:
                    img = loaded[path]
                    generator = torch.Generator('cuda').manual_seed(args.seed)
                    torch.cuda.synchronize()
                    t = time.monotonic()
                    with torch.inference_mode():
                        out = pipe(
                            image=[img], prompt=PROMPTS[prompt_nome], negative_prompt=' ',
                            true_cfg_scale=true_cfg, guidance_scale=1.0,
                            num_inference_steps=steps, generator=generator,
                        ).images[0]
                    torch.cuda.synchronize()
                    secs = time.monotonic() - t
                    out.save(out_dir / f'{path.stem}.jpg', quality=95)
                    writer.writerow([
                        cfg['nome'], prompt_nome, steps, true_cfg, path.name,
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
