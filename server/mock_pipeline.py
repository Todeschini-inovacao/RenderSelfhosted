"""Pipeline falso para validar a integração sem GPU: devolve a entrada com uma faixa "MOCK"."""

import time

from PIL import Image, ImageDraw, ImageFont


class MockPipeline:
    name = 'mock'
    precision = 'none'
    vram_gb = None

    def __init__(self, delay_s: float = 2.0):
        self.delay_s = delay_s

    def load(self) -> None:
        pass

    def generate(self, image: Image.Image, prompt: str, scene_type: str) -> Image.Image:
        time.sleep(self.delay_s)

        out = image.convert('RGBA')
        overlay = Image.new('RGBA', out.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        band_h = max(40, out.height // 10)
        draw.rectangle([(0, 0), (out.width, band_h)], fill=(220, 30, 30, 170))
        font = ImageFont.load_default(size=int(band_h * 0.6))
        draw.text(
            (band_h // 3, band_h // 2),
            f'MOCK · {scene_type}',
            fill=(255, 255, 255, 255),
            font=font,
            anchor='lm',
        )
        return Image.alpha_composite(out, overlay).convert('RGB')
