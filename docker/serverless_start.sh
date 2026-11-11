#!/usr/bin/env bash
# Entrypoint do worker RunPod Serverless (docker/Dockerfile.serverless).
#
# Acha os pesos do MODEL_REPO (repo do HuggingFace montado por scripts/publish_model.py):
#   1. no cache do RunPod ("Model caching" do endpoint): disco local, boot rápido;
#   2. senão, baixa pra /models (precisa de HF_TOKEN se o repo for privado; mais lento).
# Depois sobe o servidor na porta $PORT (o RunPod checa $PORT_HEALTH/ping).
set -euo pipefail

: "${MODEL_REPO:?Defina MODEL_REPO (ex.: org/todes-render-qwen-edit-fp8) nas env vars do endpoint}"
MODEL_REVISION=${MODEL_REVISION:-main}
CACHE_HUB=${RUNPOD_HF_CACHE:-/runpod-volume/huggingface-cache/hub}

cache_dir="$CACHE_HUB/models--${MODEL_REPO//\//--}"
snapshot=""
if [ -d "$cache_dir/snapshots" ]; then
    # Revisão exata se o refs/ existir; senão o primeiro snapshot do cache.
    if [ -f "$cache_dir/refs/$MODEL_REVISION" ]; then
        snapshot="$cache_dir/snapshots/$(cat "$cache_dir/refs/$MODEL_REVISION")"
    else
        snapshot=$(ls -d "$cache_dir"/snapshots/*/ 2>/dev/null | head -1 || true)
    fi
fi

if [ -n "$snapshot" ] && [ -f "$snapshot/model_index.json" ]; then
    echo "Pesos no cache do RunPod: $snapshot"
else
    echo "Pesos fora do cache do RunPod -- baixando $MODEL_REPO@$MODEL_REVISION para /models"
    python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='/models')
EOF
    snapshot=/models
fi

export MODEL_DIR="${snapshot%/}"
export TRANSFORMER_DIR="$MODEL_DIR/transformer"
export HF_HUB_OFFLINE=1
mkdir -p "$OUTPUT_DIR"

exec python -m uvicorn server.main:app --host 0.0.0.0 --port "${PORT:-80}"
