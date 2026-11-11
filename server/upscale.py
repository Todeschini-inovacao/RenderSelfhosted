"""Pós-processamento: devolve o render no tamanho exato da imagem original.

O Qwen-Image-Edit gera sempre ~1 MP (ex.: 1376x768 pra uma entrada 1920x1080). Pra entregar
no tamanho da original sem borrar, a saída passa por um upscaler de super-resolução (que
recria nitidez) e depois é reduzida com Lanczos pro tamanho exato da original:

    original 1920x1080 -> Qwen 1376x768 -> upscaler 2x 2752x1536 -> Lanczos 1920x1080

Se a original for menor que o render, pula o upscaler e só reduz. Os modelos são carregados
pelo spandrel (detecta a arquitetura pelo .pth) a partir de UPSCALER_DIR -- baixados no
build por scripts/download_upscalers.py.
"""

import logging
import threading
from pathlib import Path

from PIL import Image

from server.config import Settings

logger = logging.getLogger('render-local')

# nome (UPSCALER / campo "upscaler" da requisição) -> arquivo em UPSCALER_DIR
UPSCALERS = {
    'realesrgan_x2': 'RealESRGAN_x2plus.pth',
    'realesrgan_x4': 'RealESRGAN_x4plus.pth',
    'swinir_x4': '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
}
NAMES = {'none', *UPSCALERS}

_models: dict = {}
_lock = threading.Lock()


def _load(name: str, settings: Settings):
    with _lock:
        if name not in _models:
            import torch
            from spandrel import ModelLoader

            path = Path(settings.UPSCALER_DIR) / UPSCALERS[name]
            model = ModelLoader().load_from_file(str(path)).eval().to(_device())
            if model.supports_half and _device() == 'cuda':
                model = model.half()
            _models[name] = model
            logger.info('Upscaler %s carregado (%sx, fp16: %s)', name, model.scale, model.supports_half)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return _models[name]


def _device() -> str:
    import torch

    return 'cuda' if torch.cuda.is_available() else 'cpu'


def preload(name: str, settings: Settings) -> None:
    if name != 'none':
        _load(name, settings)


def _upscale(img: Image.Image, name: str, settings: Settings) -> Image.Image:
    import numpy as np
    import torch

    model = _load(name, settings)
    dtype = torch.float16 if model.supports_half and _device() == 'cuda' else torch.float32
    x = torch.from_numpy(np.asarray(img.convert('RGB')).copy()).to(_device())
    x = x.permute(2, 0, 1).unsqueeze(0).to(dtype) / 255.0
    with torch.inference_mode():
        y = _tiled(model, x, model.scale, settings.UPSCALE_TILE)
    y = (y.clamp(0, 1) * 255.0).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(y)


def _tiled(model, x, scale: int, tile: int, pad: int = 32):
    """Roda em blocos (com margem, recortada depois) pra limitar a VRAM -- o Qwen continua
    ocupando a maior parte da GPU."""
    _, _, h, w = x.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return model(x)
    out = x.new_zeros(1, 3, h * scale, w * scale)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
            yb, xb = min(y1 + pad, h), min(x1 + pad, w)
            o = model(x[:, :, ya:yb, xa:xb])
            oy, ox = (y0 - ya) * scale, (x0 - xa) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = o[
                :, :, oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale
            ]
    return out


def finalize(img: Image.Image, original_size: tuple[int, int], name: str, settings: Settings) -> Image.Image:
    """Entrega no tamanho da original (OUTPUT_SIZE=original) ou como o modelo gerou."""
    if settings.OUTPUT_SIZE != 'original' or img.size == original_size:
        return img
    tw, th = original_size
    if name != 'none' and (tw > img.width or th > img.height):
        img = _upscale(img, name, settings)
    return img.resize((tw, th), Image.LANCZOS)
