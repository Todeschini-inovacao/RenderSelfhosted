"""Gera um transformer com os LoRAs já fundidos (bf16), pro servidor subir sem fundir.

Usa as mesmas variáveis do servidor (LIGHTNING_PATH, LORA_PATH, LORA_SCALE, MODEL_DIR) e
a mesma ordem de fusão (Lightning com força 1.0, depois o LoRA de estilo). Grava a pasta
no formato do diffusers + fused_info.json; depois é só subir o servidor com
TRANSFORMER_DIR=<pasta> (sem precisar de LORA_PATH/LIGHTNING_PATH).

Uso (no pod, GPU livre, na raiz do repo):
    LIGHTNING_PATH=/workspace/models/lightning/...-4steps-....safetensors \
    LORA_PATH=/workspace/lora/output/.../todes_render_qwen_edit_2511_v1.safetensors \
    LORA_SCALE=0.8 MODEL_DIR=/workspace/models/Qwen-Image-Edit-2511 \
    PYTHONPATH=. /root/venv/bin/python scripts/fuse_transformer.py /workspace/models/fused/<nome>
"""

import argparse
import json
import time
from pathlib import Path

import torch

from server.config import Settings
from server.qwen_pipeline import FUSED_INFO_FILE, fuse_loras, fusion_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    from diffusers import QwenImageEditPlusPipeline

    settings = Settings()
    loras = fusion_plan(settings)
    if not loras:
        raise SystemExit('Defina LORA_PATH e/ou LIGHTNING_PATH.')
    source = settings.MODEL_DIR or settings.MODEL_ID
    revision = None if settings.MODEL_DIR else settings.MODEL_REVISION

    t0 = time.monotonic()
    # Só o transformer importa; text encoder/VAE ficam na CPU e não são salvos.
    pipe = QwenImageEditPlusPipeline.from_pretrained(source, revision=revision, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        pipe.transformer.to('cuda')
    print(f'Carga: {time.monotonic() - t0:.0f}s', flush=True)

    t = time.monotonic()
    fuse_loras(pipe, loras)
    print(f'Fusão: {time.monotonic() - t:.0f}s', flush=True)

    t = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe.transformer.save_pretrained(args.output_dir, max_shard_size='5GB')
    info = {
        'model': source,
        'lightning': bool(settings.LIGHTNING_PATH),
        'loras': [{'path': p, 'scale': s} for p, s in loras],
    }
    (args.output_dir / FUSED_INFO_FILE).write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'Salvo em {args.output_dir} ({time.monotonic() - t:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
