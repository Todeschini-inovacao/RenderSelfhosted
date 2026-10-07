#!/usr/bin/env bash
# Prepara um pod do RunPod pra rodar o pipeline real direto (sem Docker -- o pod já é um
# container, não dá pra rodar `docker build` dentro dele). Passo a passo completo no
# README, seção "Teste na nuvem (RunPod)".
#
# Idempotente: pode rodar de novo a cada pod novo. Tudo que é pesado (venv, pesos do
# modelo) fica em /workspace, que é o volume persistente do pod -- num pod reiniciado
# com o mesmo volume, só confere e segue.
#
# Uso (dentro do pod, na raiz do repo):
#   export HF_TOKEN=hf_...     # ou configurado como secret/env var do pod no RunPod
#   bash scripts/runpod_setup.sh
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
VENV=${VENV:-$WORKSPACE/venv}
export HF_HOME=${HF_HOME:-$WORKSPACE/hf_cache}

# torch com CUDA 13.0: a RTX 5090 (Blackwell, sm_120) precisa de build CUDA 12.8+. Mesma
# versão de torch da base do docker/Dockerfile.
TORCH_INDEX=https://download.pytorch.org/whl/cu130
TORCH_VERSION=2.14.1

cd "$(dirname "$0")/.."

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if [ ! -x "$VENV/bin/python" ]; then
    echo "== Criando venv em $VENV"
    python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"

echo "== Instalando torch $TORCH_VERSION (cu130) e dependências"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "torch==$TORCH_VERSION" torchvision --index-url "$TORCH_INDEX"
"$PY" -m pip install -r requirements-gpu.txt

echo "== Conferindo suporte da GPU no torch"
"$PY" - <<'EOF'
import torch

assert torch.cuda.is_available(), 'torch não enxerga a GPU'
major, minor = torch.cuda.get_device_capability(0)
arch = f'sm_{major}{minor}'
props = torch.cuda.get_device_properties(0)
print(f'{props.name}: {props.total_memory / 1024**3:.1f} GB, {arch}, torch {torch.__version__}')
assert arch in torch.cuda.get_arch_list(), f'este build do torch não tem kernels pra {arch}'
EOF

echo "== Baixando pesos para $HF_HOME (~60 GB na primeira vez)"
"$PY" scripts/download_models.py

echo "== Pronto. Para subir o servidor: bash scripts/runpod_start.sh"
