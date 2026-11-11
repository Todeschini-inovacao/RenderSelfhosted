"""Monta e publica o modelo de produção num repo PRIVADO do HuggingFace, pro "Model
caching" do RunPod Serverless (docker/Dockerfile.serverless).

O repo fica com o layout de um pipeline diffusers completo:
    model_index.json, scheduler/, tokenizer/, processor/, vae/, text_encoder/  <- base
    transformer/  <- transformer com LoRAs fundidos e em fp8 (quantize_transformer.py)
O transformer fp8 é pickle (.bin do torchao): o repo precisa ser privado e só carregado
pelos nossos workers.

Uso (no pod, na raiz do repo; HF_TOKEN com permissão de escrita):
    HF_TOKEN=hf_... PYTHONPATH=. /root/venv/bin/python scripts/publish_model.py \
        <org-ou-usuario>/<nome-do-repo> \
        --base /workspace/models/Qwen-Image-Edit-2511 \
        --transformer /workspace/models/fused/todes_v1_0.8_lightning4_fp8
(se houver cópias em /root, passe elas: hardlink em vez de copiar 35 GB)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from server.qwen_pipeline import FUSED_INFO_FILE

BASE_PARTS = ['model_index.json', 'scheduler', 'tokenizer', 'processor', 'vae', 'text_encoder']


def _link_or_copy(src: str, dst: str) -> None:
    # Hardlink quando estão no mesmo disco (instantâneo, sem ocupar espaço); senão copia.
    # Symlink não serve: o upload do HF não segue links de pasta em todas as versões.
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage(base: Path, transformer: Path, out: Path) -> None:
    """Monta a pasta no layout do repo. Com base/transformer no mesmo disco de `out`
    (ex.: cópias em /root), usa hardlinks e não ocupa espaço extra."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for part in BASE_PARTS:
        src = base / part
        if src.is_dir():
            shutil.copytree(src, out / part, copy_function=_link_or_copy)
        else:
            _link_or_copy(str(src), str(out / part))
    shutil.copytree(transformer, out / 'transformer', copy_function=_link_or_copy)
    info = json.loads((transformer / FUSED_INFO_FILE).read_text(encoding='utf-8'))
    if info.get('quantization') != 'fp8_weight_only':
        raise SystemExit(f'{transformer} não é um transformer fp8 de quantize_transformer.py')
    (out / 'README.md').write_text(
        '# Render Local -- modelo de produção (privado)\n\n'
        'Qwen-Image-Edit-2511 (Apache 2.0) com LoRAs fundidos no transformer, em fp8 '
        '(torchao). Gerado por scripts/publish_model.py.\n\n'
        f'```json\n{json.dumps(info, indent=2)}\n```\n',
        encoding='utf-8',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('repo_id')
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--transformer', type=Path, required=True)
    parser.add_argument('--stage-dir', type=Path, default=Path('/root/publish_stage'))
    args = parser.parse_args()

    from huggingface_hub import HfApi

    stage(args.base, args.transformer, args.stage_dir)
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi(token=os.environ.get('HF_TOKEN'))
    # Token fine-grained restrito a um repo já criado (privado) não tem permissão de criar
    # repo -- só cria se ainda não existir.
    try:
        info = api.repo_info(args.repo_id)
    except RepositoryNotFoundError:
        api.create_repo(args.repo_id, private=True)
        info = api.repo_info(args.repo_id)
    if not info.private:
        raise SystemExit(f'{args.repo_id} existe e é PÚBLICO -- não publico pesos nele.')
    # upload_large_folder: retomável e em paralelo (35 GB).
    api.upload_large_folder(repo_id=args.repo_id, repo_type='model', folder_path=args.stage_dir)
    print(f'Publicado: https://huggingface.co/{args.repo_id}', flush=True)


if __name__ == '__main__':
    main()
