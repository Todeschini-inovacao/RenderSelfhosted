"""Testa o servidor de renderização (local ou no pod do RunPod): GET /health e
POST /render com uma imagem de entrada. Só usa a biblioteca padrão do Python --
não precisa da venv do projeto, roda em qualquer máquina com Python 3.10+.

Uso:
    python scripts/test_render.py caminho/imagem.jpg \
        --url https://<pod-id>-8000.proxy.runpod.net \
        --api-key <mesmo valor do API_KEY do servidor> \
        --prompt "mesma sala, de dia, com iluminação natural"

--url e --api-key também podem vir das variáveis de ambiente RENDER_URL e API_KEY.
"""

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


# O proxy do RunPod fica atrás do Cloudflare, que bloqueia (HTTP 403 / erro 1010) o
# User-Agent padrão do urllib por parecer bot -- manda um de navegador normal.
_BROWSER_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
}


def _get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', 'replace')
        raise SystemExit(f'Erro HTTP {exc.code} em {url}: {detail}')
    except urllib.error.URLError as exc:
        raise SystemExit(f'Falha ao conectar em {url}: {exc.reason}')


def _post_json(url: str, headers: dict, body: bytes, timeout: float) -> dict:
    req = urllib.request.Request(url, data=body, method='POST', headers={**_BROWSER_HEADERS, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', 'replace')
        raise SystemExit(f'Erro HTTP {exc.code} em {url}: {detail}')
    except urllib.error.URLError as exc:
        raise SystemExit(f'Falha ao conectar em {url}: {exc.reason}')


def check_health(base_url: str) -> None:
    print('== GET /health')
    data = _get_json(f'{base_url}/health', timeout=15.0)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    if data.get('status') != 'ready':
        raise SystemExit(f"Servidor não está pronto (status={data.get('status')!r}).")


def render(
    base_url: str,
    api_key: str | None,
    image_path: Path,
    prompt: str,
    scene_type: str,
    output_format: str,
) -> dict:
    print(f'== POST /render ({image_path.name})')
    mime = mimetypes.guess_type(image_path.name)[0] or 'image/jpeg'
    image_b64 = base64.b64encode(image_path.read_bytes()).decode('ascii')
    payload = json.dumps(
        {
            'image': f'data:{mime};base64,{image_b64}',
            'prompt': prompt,
            'sceneType': scene_type,
            'outputFormat': output_format,
        }
    ).encode('utf-8')

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    t0 = time.monotonic()
    data = _post_json(f'{base_url}/render', headers, payload, timeout=180.0)
    wall_s = time.monotonic() - t0
    print(f"Resposta em {wall_s:.1f}s (servidor reportou {data['elapsed_ms']} ms), modelo: {data['model']}")
    return data


def save_result(image_field: str, out_path: Path) -> None:
    if image_field.startswith('data:'):
        _, b64_data = image_field.split(',', 1)
        out_path.write_bytes(base64.b64decode(b64_data))
        print(f'Salvo em {out_path}')
        return

    print(f'Resultado é uma URL (RETURN_MODE != base64): {image_field}')
    try:
        req = urllib.request.Request(image_field, headers=_BROWSER_HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            out_path.write_bytes(resp.read())
        print(f'Baixado e salvo em {out_path}')
    except Exception as exc:
        print(f'Não consegui baixar automaticamente ({exc}); abra a URL manualmente.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', type=Path, help='Caminho da imagem de entrada')
    parser.add_argument('--url', default=None, help='Base URL do servidor (ou env RENDER_URL)')
    parser.add_argument('--api-key', default=None, help='API key (ou env API_KEY)')
    parser.add_argument(
        '--prompt',
        default='Render fotorrealista da mesma cena, iluminação natural de dia',
        help='Prompt de geração',
    )
    parser.add_argument('--scene-type', default='interior', help='sceneType enviado na requisição')
    parser.add_argument('--output-format', default='jpg', choices=['jpg', 'jpeg', 'png', 'webp'])
    parser.add_argument('--out', type=Path, default=None, help='Onde salvar o resultado (default: <imagem>_render.<ext>)')
    parser.add_argument('--skip-health', action='store_true', help='Pula a checagem de /health antes de renderizar')
    args = parser.parse_args()

    base_url = (args.url or os.environ.get('RENDER_URL') or '').rstrip('/')
    if not base_url:
        parser.error('Informe --url ou defina a variável de ambiente RENDER_URL')
    api_key = args.api_key or os.environ.get('API_KEY')

    if not args.image.is_file():
        parser.error(f'Imagem não encontrada: {args.image}')

    if not args.skip_health:
        check_health(base_url)

    result = render(base_url, api_key, args.image, args.prompt, args.scene_type, args.output_format)

    out_path = args.out or args.image.with_name(f'{args.image.stem}_render.{args.output_format}')
    save_result(result['image'], out_path)


if __name__ == '__main__':
    main()
