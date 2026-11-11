"""Benchmark de número de steps: carrega o pipeline real uma vez e renderiza cada imagem
de entrada com cada valor de steps, medindo o tempo. Usa a mesma configuração do
servidor (server/config.py / .env / env vars: PRECISION, GUIDANCE, MAX_SIDE, MODEL_DIR...)
e o mesmo pré-processamento (limit_size), só variando NUM_STEPS.

Precisa da GPU livre -- pare o servidor antes (não cabem dois modelos na VRAM).

Uso (no pod, na raiz do repo):
    MOCK=false /root/venv/bin/python scripts/bench_steps.py pasta_entrada pasta_saida \
        --steps 4 8 16 24 --prompt "..."

Saída: <pasta_saida>/steps_<N>/<imagem>.jpg e <pasta_saida>/resultados.csv
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import torch
from PIL import Image

from server import image_io
from server.config import settings
from server.qwen_pipeline import QwenImageEditPipeline

DEFAULT_PROMPT = (
    'Transform this 3D render into a photorealistic interior photograph. Keep the exact same '
    'furniture, layout, camera angle, materials, wood tones and colors. Realistic lighting, '
    'natural shadows and reflections.'
)
EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--steps', type=int, nargs='+', default=[4, 8, 16, 24])
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f'Nenhuma imagem em {args.input_dir}')

    pipe = QwenImageEditPipeline(settings)
    t0 = time.monotonic()
    pipe.load()
    print(f'Carga: {time.monotonic() - t0:.0f}s (precisão {pipe.precision})', flush=True)

    loaded = {p: image_io.limit_size(Image.open(p).convert('RGB'), settings.MAX_SIDE) for p in images}

    # Aquecimento (não medido): a primeira chamada inclui inicialização de kernels.
    settings.NUM_STEPS = 2
    pipe.generate(loaded[images[0]], args.prompt, 'interior')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'resultados.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['steps', 'imagem', 'largura', 'altura', 'segundos'])
        for steps in args.steps:
            settings.NUM_STEPS = steps
            out_dir = args.output_dir / f'steps_{steps}'
            out_dir.mkdir(exist_ok=True)
            for path in images:
                img = loaded[path]
                torch.cuda.synchronize()
                t = time.monotonic()
                out = pipe.generate(img, args.prompt, 'interior')
                torch.cuda.synchronize()
                secs = time.monotonic() - t
                out.save(out_dir / f'{path.stem}.jpg', quality=95)
                writer.writerow([steps, path.name, img.width, img.height, f'{secs:.1f}'])
                f.flush()
                print(f'steps={steps:>2} {path.name} {img.width}x{img.height}: {secs:.1f}s', flush=True)

    print(f'Pronto: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
