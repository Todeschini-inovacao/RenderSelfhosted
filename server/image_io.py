"""Conversões entre o que chega na API (data URI, base64 puro ou URL) e PIL."""

import base64
import binascii
import io
import uuid
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024

# outputFormat do Render -> (formato PIL, extensão, mime)
FORMATS = {
    'jpg': ('JPEG', 'jpg', 'image/jpeg'),
    'jpeg': ('JPEG', 'jpg', 'image/jpeg'),
    'png': ('PNG', 'png', 'image/png'),
    'webp': ('WEBP', 'webp', 'image/webp'),
}


class InvalidImageError(ValueError):
    pass


def _open(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f'Conteúdo não é uma imagem válida: {exc}') from exc
    return img.convert('RGB')


async def load_image(value: str) -> Image.Image:
    """Aceita http(s) URL, data URI ou base64 puro."""
    value = value.strip()
    if value.startswith(('http://', 'https://')):
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            try:
                resp = await client.get(value)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise InvalidImageError(f'Falha ao baixar imagem: {exc}') from exc
        if len(resp.content) > MAX_DOWNLOAD_BYTES:
            raise InvalidImageError('Imagem maior que o limite de 30 MB')
        return _open(resp.content)

    if value.startswith('data:'):
        value = value.split(',', 1)[1] if ',' in value else ''
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError('Base64 inválido') from exc
    return _open(data)


def limit_size(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def _encode(img: Image.Image, output_format: str) -> tuple[bytes, str, str]:
    pil_format, ext, mime = FORMATS.get(output_format.lower(), FORMATS['png'])
    buf = io.BytesIO()
    save_kwargs = {'quality': 95} if pil_format in ('JPEG', 'WEBP') else {}
    img.save(buf, format=pil_format, **save_kwargs)
    return buf.getvalue(), ext, mime


def to_data_url(img: Image.Image, output_format: str) -> str:
    data, _, mime = _encode(img, output_format)
    return f'data:{mime};base64,{base64.b64encode(data).decode("ascii")}'


def save_output(img: Image.Image, output_format: str, output_dir: str) -> str:
    """Salva em disco e devolve o nome do arquivo."""
    data, ext, _ = _encode(img, output_format)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    name = f'{uuid.uuid4().hex}.{ext}'
    (Path(output_dir) / name).write_bytes(data)
    return name
