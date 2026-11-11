"""Baixa os pesos do modelo pro cache do HuggingFace, pra embutir na imagem Docker
(decisão D3 do SPEC.md: pesos embutidos, "só ligar" -- sem precisar de internet/token
em runtime). Roda uma vez no build da imagem (ver docker/Dockerfile).

Mitigação do risco "licença de checkpoint muda entre versões" (SPEC.md seção 6):
fixa a revisão (commit hash) do repositório no HuggingFace -- `main` pode avançar pra
uma versão com licença diferente sem aviso. Revisão abaixo confirmada em 2026-10-06
(huggingface.co/api/models/Qwen/Qwen-Image-Edit-2511 -> lastModified 2025-12-23,
licença Apache 2.0, repositório não-gated).

Uso: HF_TOKEN=<token com acesso de leitura> python scripts/download_models.py
No Dockerfile, o token entra via `--mount=type=secret` (nunca fica gravado na imagem).

Com MODEL_DIR definido, baixa como arquivos comuns nessa pasta em vez do cache do HF.
O cache usa symlinks (snapshots/ -> blobs/), e o volume /workspace do RunPod (geesefs,
em cima de S3) não guarda symlinks: depois de um restart do pod eles viram arquivos de
0 bytes e o modelo deixa de carregar. Ver scripts/runpod_setup.sh.
"""

import os
import sys

MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen-Image-Edit-2511')
MODEL_REVISION = os.environ.get('MODEL_REVISION', '6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9')
MODEL_DIR = os.environ.get('MODEL_DIR') or None


def main() -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get('HF_TOKEN')
    destino = MODEL_DIR or 'o cache do HuggingFace'
    print(f'Baixando {MODEL_ID}@{MODEL_REVISION} para {destino}...', flush=True)

    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        token=token,
        local_dir=MODEL_DIR,
        # .safetensors + configs/tokenizer bastam -- sem .bin/.ckpt duplicados nem
        # imagens de exemplo do repositório (ver `siblings` da API do HF pro que sobra).
        # *.jinja: chat_template.jinja do processor, exigido no load do QwenImageEditPlusPipeline.
        allow_patterns=['*.safetensors', '*.json', '*.txt', 'tokenizer/**', '*.model', '*.jinja'],
    )

    print('Download concluído.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # noqa: BLE001 -- script de build, falha deve abortar a imagem
        print(f'Falha ao baixar o modelo: {exc}', file=sys.stderr)
        sys.exit(1)
