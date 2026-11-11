"""Upload do resultado pra um storage externo -- usado no RunPod Serverless, onde o worker
é efêmero e não pode servir a imagem por URL própria (ver docker/Dockerfile.serverless).

- RETURN_MODE=s3: bucket S3-compatível (AWS S3, Cloudflare R2, ...). Credenciais pelas
  variáveis padrão do boto3: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
- RETURN_MODE=azure: Azure Blob Storage, via AZURE_STORAGE_CONNECTION_STRING.
"""

import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from PIL import Image

from server import image_io
from server.config import Settings


def upload(img: Image.Image, output_format: str, settings: Settings) -> str:
    """Sobe a imagem pro storage do RETURN_MODE e devolve a URL."""
    data, ext, mime = image_io.encode(img, output_format)
    name = f'{uuid.uuid4().hex}.{ext}'
    if settings.RETURN_MODE == 'azure':
        return _upload_azure(data, f'{settings.AZURE_PREFIX}{name}', mime, settings)
    return _upload_s3(data, f'{settings.S3_PREFIX}{name}', mime, settings)


# --- S3 -------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _client(endpoint_url: str | None, region: str | None):
    import boto3

    return boto3.client('s3', endpoint_url=endpoint_url or None, region_name=region or None)


def _upload_s3(data: bytes, key: str, mime: str, settings: Settings) -> str:
    """URL pública (S3_PUBLIC_BASE_URL) ou pré-assinada."""
    client = _client(settings.S3_ENDPOINT_URL, settings.S3_REGION)
    client.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=data, ContentType=mime)
    if settings.S3_PUBLIC_BASE_URL:
        return f'{settings.S3_PUBLIC_BASE_URL.rstrip("/")}/{key}'
    return client.generate_presigned_url(
        'get_object',
        Params={'Bucket': settings.S3_BUCKET, 'Key': key},
        ExpiresIn=settings.S3_PRESIGN_EXPIRES,
    )


# --- Azure Blob -----------------------------------------------------------------------


@lru_cache(maxsize=1)
def _container(connection_string: str, container: str):
    from azure.storage.blob import BlobServiceClient

    return BlobServiceClient.from_connection_string(connection_string).get_container_client(
        container
    )


def _upload_azure(data: bytes, blob_name: str, mime: str, settings: Settings) -> str:
    """URL pública (AZURE_PUBLIC_BASE_URL) ou URL do blob com SAS de leitura."""
    from azure.storage.blob import BlobSasPermissions, ContentSettings, generate_blob_sas

    container = _container(settings.AZURE_STORAGE_CONNECTION_STRING, settings.AZURE_CONTAINER)
    blob = container.get_blob_client(blob_name)
    blob.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type=mime))
    if settings.AZURE_PUBLIC_BASE_URL:
        return f'{settings.AZURE_PUBLIC_BASE_URL.rstrip("/")}/{blob_name}'
    account_key = blob.credential.account_key if blob.credential else None
    if not account_key:
        raise RuntimeError(
            'Sem AZURE_PUBLIC_BASE_URL, a connection string precisa de AccountKey pra gerar o SAS'
        )
    sas = generate_blob_sas(
        account_name=blob.account_name,
        container_name=settings.AZURE_CONTAINER,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(days=settings.AZURE_SAS_DAYS),
    )
    return f'{blob.url}?{sas}'
