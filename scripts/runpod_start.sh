#!/usr/bin/env bash
# Sobe o servidor com o pipeline real num pod do RunPod (rodar scripts/runpod_setup.sh
# antes). Variáveis de ambiente sobrescrevem o .env (se existir) -- ex.: PRECISION=nf4.
#
# Uso (dentro do pod, na raiz do repo):
#   export API_KEY=<segredo>   # obrigatório: o proxy do RunPod deixa a porta pública
#   bash scripts/runpod_start.sh
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
VENV=${VENV:-$WORKSPACE/venv}
export HF_HOME=${HF_HOME:-$WORKSPACE/hf_cache}

cd "$(dirname "$0")/.."

if [ -z "${API_KEY:-}" ]; then
    echo "Defina API_KEY: pelo proxy do RunPod o servidor fica acessível pela internet." >&2
    echo "No Render, use o mesmo valor em CUSTOM_API_KEY." >&2
    exit 1
fi

# RUNPOD_POD_ID é definido pelo RunPod em todo pod; a porta 8000 precisa estar em
# "Expose HTTP Ports" na configuração do pod.
if [ -z "${PUBLIC_BASE_URL:-}" ]; then
    if [ -z "${RUNPOD_POD_ID:-}" ]; then
        echo "RUNPOD_POD_ID não definido -- defina PUBLIC_BASE_URL manualmente." >&2
        exit 1
    fi
    PUBLIC_BASE_URL="https://${RUNPOD_POD_ID}-8000.proxy.runpod.net"
fi

export API_KEY PUBLIC_BASE_URL
export MOCK=false
export HF_HUB_OFFLINE=1
export OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE/outputs}

echo "URL pública: $PUBLIC_BASE_URL"
echo "No Render: CUSTOM_API_URL=$PUBLIC_BASE_URL/render"
exec "$VENV/bin/python" -m uvicorn server.main:app --host 0.0.0.0 --port 8000
