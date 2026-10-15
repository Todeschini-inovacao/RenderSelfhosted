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

# Mesma versão de torch da base do docker/Dockerfile. Builds publicados pra 2.14.1: cu126
# e cu130. cu130 exige driver 580+ e é o único que roda em Blackwell (RTX 50xx/PRO 6000,
# sm_120); cu126 roda com driver 560+ em GPUs até Ada/Hopper (4090, 6000 Ada, L40S, H100).
# O check de arquitetura mais abaixo falha com mensagem clara se a combinação não servir.
TORCH_VERSION=2.14.1

cd "$(dirname "$0")/.."

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ -z "${TORCH_CUDA:-}" ]; then
    if [ "$DRIVER_MAJOR" -ge 580 ]; then TORCH_CUDA=cu130; else TORCH_CUDA=cu126; fi
fi
TORCH_INDEX=https://download.pytorch.org/whl/$TORCH_CUDA
echo "Driver $DRIVER_MAJOR -> torch $TORCH_VERSION+$TORCH_CUDA"

if [ ! -x "$VENV/bin/python" ]; then
    echo "== Criando venv em $VENV"
    python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"

echo "== Instalando torch $TORCH_VERSION ($TORCH_CUDA) e dependências"
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
# get_arch_list() nem sempre lista a arch exata (ex.: builds oficiais do torch não trazem
# sm_89/Ada explicitamente, só sm_86/Ampere -- mas o kernel sm_86 roda em GPUs Ada por
# compatibilidade binária). Em vez de exigir match exato, testa uma conta real na GPU.
if arch not in torch.cuda.get_arch_list():
    print(f'aviso: {arch} não está em get_arch_list() ({torch.cuda.get_arch_list()}); testando execução real')
try:
    x = torch.randn(256, 256, device='cuda')
    (x @ x).sum().item()
except RuntimeError as e:
    raise AssertionError(f'GPU não executa kernels deste build do torch ({arch}): {e}') from e
EOF

echo "== Baixando pesos para $HF_HOME (~60 GB na primeira vez)"
"$PY" scripts/download_models.py

echo "== Pronto. Para subir o servidor: bash scripts/runpod_start.sh"
