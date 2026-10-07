# syntax=docker/dockerfile:1.7
#
# Base com CUDA 13.0 -- precisa de driver NVIDIA 580+ no host. CUDA 12.8+ é obrigatório
# pra GPUs Blackwell (RTX 50xx, sm_120): o torch das tags -cuda12.6-* não tem kernels pra
# elas ("no kernel image is available"). Tags disponíveis pra 2.14.1: cuda12.6, cuda13.0,
# cuda13.2 (https://hub.docker.com/r/pytorch/pytorch/tags). Só voltar pra cuda12.6 se a
# máquina tiver GPU pré-Blackwell (ex.: 4090) com driver antigo -- confirmar com
# `nvidia-smi` (SPEC.md E0).
#
# Variante "runtime" (não "devel"): não tem nvcc, só as libs de CUDA pra rodar. O torch
# e o torchao (pra fp8) trazem seus próprios binários CUDA via pip -- se o torchao
# precisar compilar algum kernel Triton em runtime e isso falhar na GPU real, trocar
# pra uma tag "-devel" é o primeiro passo de diagnóstico (PENDENTE validar, SPEC E3).
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

WORKDIR /app

COPY requirements.txt requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY server ./server
COPY scripts ./scripts

ARG MODEL_ID=Qwen/Qwen-Image-Edit-2511
ARG MODEL_REVISION=6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9
ENV MODEL_ID=${MODEL_ID} \
    MODEL_REVISION=${MODEL_REVISION} \
    HF_HOME=/app/.hf_cache

# Baixa os pesos durante o build (decisão D3 do SPEC.md: embutidos na imagem, "só
# ligar" -- sem precisar de internet/token depois). O token só existe durante este RUN,
# via secret do BuildKit -- nunca é gravado numa layer da imagem.
#   docker build --secret id=hf_token,src=./.hf_token.txt -f docker/Dockerfile .
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) python scripts/download_models.py

# Só depois do download -- offline teria quebrado o RUN acima.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

# Formato confirmado rodando o servidor mock local (Starlette/FastAPI usa separators
# compactos, sem espaço): {"status":"ready",...}
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if b'\"status\":\"ready\"' in urllib.request.urlopen('http://localhost:8000/health').read() else 1)"

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
