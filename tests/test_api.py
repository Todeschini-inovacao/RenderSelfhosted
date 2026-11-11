import base64
import io
import os
import time

os.environ['MOCK'] = 'true'
os.environ['MOCK_DELAY_S'] = '0'

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server.config import settings
from server.main import app


def _png_b64(size=(64, 48), color=(10, 120, 200)) -> str:
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'OUTPUT_DIR', str(tmp_path))
    monkeypatch.setattr(settings, 'API_KEY', None)
    monkeypatch.setattr(settings, 'RETURN_MODE', 'url')
    with TestClient(app) as c:
        for _ in range(50):
            if c.get('/health').json()['status'] == 'ready':
                break
            time.sleep(0.05)
        yield c


def test_health_ready(client):
    body = client.get('/health').json()
    assert body['status'] == 'ready'
    assert body['model'] == 'mock'


def test_render_data_uri_returns_url(client, tmp_path):
    payload = {
        'sceneType': 'interior',
        'outputFormat': 'jpg',
        'image': f'data:image/png;base64,{_png_b64()}',
        'prompt': 'teste',
    }
    resp = client.post('/render', json=payload)
    assert resp.status_code == 200, resp.text
    url = resp.json()['image']
    name = url.rsplit('/', 1)[1]
    assert name.endswith('.jpg')
    assert (tmp_path / name).exists()


def test_render_plain_base64_returns_data_uri(client, monkeypatch):
    monkeypatch.setattr(settings, 'RETURN_MODE', 'base64')
    resp = client.post('/render', json={'image': _png_b64(), 'prompt': 'x', 'outputFormat': 'png'})
    assert resp.status_code == 200, resp.text
    image = resp.json()['image']
    assert image.startswith('data:image/png;base64,')
    decoded = Image.open(io.BytesIO(base64.b64decode(image.split(',', 1)[1])))
    assert decoded.size == (64, 48)


def test_render_resizes_large_input(client, monkeypatch):
    monkeypatch.setattr(settings, 'MAX_SIDE', 32)
    monkeypatch.setattr(settings, 'RETURN_MODE', 'base64')
    resp = client.post('/render', json={'image': _png_b64(), 'prompt': 'x', 'outputFormat': 'png'})
    image = resp.json()['image']
    decoded = Image.open(io.BytesIO(base64.b64decode(image.split(',', 1)[1])))
    assert max(decoded.size) == 32


def test_render_invalid_image(client):
    resp = client.post('/render', json={'image': 'nao-e-base64!!', 'prompt': 'x'})
    assert resp.status_code == 400


def test_render_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, 'API_KEY', 'segredo')
    payload = {'image': _png_b64(), 'prompt': 'x'}
    assert client.post('/render', json=payload).status_code == 401
    ok = client.post('/render', json=payload, headers={'Authorization': 'Bearer segredo'})
    assert ok.status_code == 200


def test_ping_ready(client):
    resp = client.get('/ping')
    assert resp.status_code == 200
    assert resp.json() == {'status': 'healthy'}


def test_ping_loading_and_error(client, monkeypatch):
    from server.main import state

    monkeypatch.setattr(state, 'status', 'loading')
    assert client.get('/ping').status_code == 204
    monkeypatch.setattr(state, 'status', 'error')
    monkeypatch.setattr(state, 'error', 'falhou')
    resp = client.get('/ping')
    assert resp.status_code == 503
    assert resp.json()['error'] == 'falhou'


def test_render_s3_returns_bucket_url(client, monkeypatch):
    from server import storage

    puts = []

    class FakeS3:
        def put_object(self, **kw):
            puts.append(kw)

    monkeypatch.setattr(storage, '_client', lambda *a: FakeS3())
    monkeypatch.setattr(settings, 'RETURN_MODE', 's3')
    monkeypatch.setattr(settings, 'S3_BUCKET', 'renders-bucket')
    monkeypatch.setattr(settings, 'S3_PUBLIC_BASE_URL', 'https://cdn.exemplo.com/')
    resp = client.post('/render', json={'image': _png_b64(), 'prompt': 'x', 'outputFormat': 'jpg'})
    assert resp.status_code == 200, resp.text
    url = resp.json()['image']
    assert url.startswith('https://cdn.exemplo.com/renders/') and url.endswith('.jpg')
    assert puts[0]['Bucket'] == 'renders-bucket'
    assert puts[0]['ContentType'] == 'image/jpeg'
    assert url.endswith(puts[0]['Key'])


def test_s3_mode_requires_bucket():
    from server.config import Settings

    with pytest.raises(ValueError):
        Settings(RETURN_MODE='s3', S3_BUCKET=None)


# Connection string no formato real (conta/chave fictícias, chave base64 válida).
_AZ_CONN = (
    'DefaultEndpointsProtocol=https;AccountName=contateste;'
    'AccountKey=' + base64.b64encode(b'chave-ficticia-de-teste').decode() + ';'
    'EndpointSuffix=core.windows.net'
)


def _azure_settings(monkeypatch, public_url=None):
    from azure.storage.blob import BlobClient

    uploads = []
    monkeypatch.setattr(
        BlobClient, 'upload_blob', lambda self, data, **kw: uploads.append((self.blob_name, kw))
    )
    monkeypatch.setattr(settings, 'RETURN_MODE', 'azure')
    monkeypatch.setattr(settings, 'AZURE_STORAGE_CONNECTION_STRING', _AZ_CONN)
    monkeypatch.setattr(settings, 'AZURE_CONTAINER', 'renders')
    monkeypatch.setattr(settings, 'AZURE_PUBLIC_BASE_URL', public_url)
    return uploads


def test_render_azure_returns_sas_url(client, monkeypatch):
    uploads = _azure_settings(monkeypatch)
    resp = client.post('/render', json={'image': _png_b64(), 'prompt': 'x', 'outputFormat': 'png'})
    assert resp.status_code == 200, resp.text
    url = resp.json()['image']
    assert url.startswith('https://contateste.blob.core.windows.net/renders/renders/')
    assert '.png?' in url and 'sig=' in url and 'sp=r' in url
    assert uploads[0][1]['content_settings'].content_type == 'image/png'


def test_render_azure_public_url(client, monkeypatch):
    uploads = _azure_settings(monkeypatch, public_url='https://cdn.exemplo.com/renders')
    resp = client.post('/render', json={'image': _png_b64(), 'prompt': 'x', 'outputFormat': 'jpg'})
    assert resp.status_code == 200, resp.text
    assert resp.json()['image'] == f'https://cdn.exemplo.com/renders/{uploads[0][0]}'


def test_azure_mode_requires_connection():
    from server.config import Settings

    with pytest.raises(ValueError):
        Settings(RETURN_MODE='azure', AZURE_CONTAINER='x')
