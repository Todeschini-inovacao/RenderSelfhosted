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
