"""Upload do resultado pra um bucket S3-compatível (AWS S3, Cloudflare R2, ...) --
RETURN_MODE=s3. Usado no RunPod Serverless, onde o worker é efêmero e não pode servir a
imagem por URL própria (ver docker/Dockerfile.serverless).

Credenciais pelas variáveis padrão do boto3: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
"""

import uuid
from functools import lru_cache

from PIL import Image

from server import image_io
from server.config import Settings


@lru_cache(maxsize=1)
def _client(endpoint_url: str | None, region: str | None):
    import boto3

    return boto3.client('s3', endpoint_url=endpoint_url or None, region_name=region or None)


def upload(img: Image.Image, output_format: str, settings: Settings) -> str:
    """Sobe a imagem e devolve a URL: pública (S3_PUBLIC_BASE_URL) ou pré-assinada."""
    data, ext, mime = image_io.encode(img, output_format)
    key = f'{settings.S3_PREFIX}{uuid.uuid4().hex}.{ext}'
    client = _client(settings.S3_ENDPOINT_URL, settings.S3_REGION)
    client.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=data, ContentType=mime)
    if settings.S3_PUBLIC_BASE_URL:
        return f'{settings.S3_PUBLIC_BASE_URL.rstrip("/")}/{key}'
    return client.generate_presigned_url(
        'get_object',
        Params={'Bucket': settings.S3_BUCKET, 'Key': key},
        ExpiresIn=settings.S3_PRESIGN_EXPIRES,
    )
