"""Baixa os pesos dos upscalers (server/upscale.py) e confere o SHA-256 -- roda no build da
imagem do serverless (docker/Dockerfile.serverless). Os .pth são pickle: o hash fixo garante
que é exatamente o arquivo publicado pelos autores (Real-ESRGAN, SwinIR), nada trocado depois.

Uso: python scripts/download_upscalers.py <pasta destino>
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

# nome do arquivo -> (URL oficial do release, SHA-256 conferido em 2026-11-11)
WEIGHTS = {
    'RealESRGAN_x2plus.pth': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        '49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb',
    ),
    'RealESRGAN_x4plus.pth': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1',
    ),
    '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth': (
        'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'
        '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
        '99adfa91350a84c99e946c1eb3d8fce34bc28f57d807b09dc8fe40a316328c0a',
    ),
}


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'upscalers')
    out.mkdir(parents=True, exist_ok=True)
    for name, (url, sha256) in WEIGHTS.items():
        dest = out / name
        h = hashlib.sha256()
        req = urllib.request.Request(url, headers={'User-Agent': 'render-local-build'})
        with urllib.request.urlopen(req, timeout=300) as resp, dest.open('wb') as f:
            while chunk := resp.read(1 << 20):
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != sha256:
            dest.unlink()
            raise SystemExit(f'SHA-256 diferente em {name}: {h.hexdigest()} (esperado {sha256})')
        print(f'{name}: ok ({dest.stat().st_size / 1e6:.0f} MB)', flush=True)


if __name__ == '__main__':
    main()
