# Render Local — inferência de imagens na própria GPU

Objetivo: testar a geração dos renders **localmente, com placa de vídeo própria**, em vez da API do fal.ai usada hoje pelo projeto [Render](../Render). Tudo empacotado num container Docker para só ligar na máquina de teste.

> Status (2026-10-06): servidor **mock** pronto (E1–E2); pipeline real com Qwen-Image-Edit-2511 e Docker escritos (E3–E4), ainda sem testar numa GPU de verdade. VRAM da máquina de teste confirmada em 24 GB+ (D1) — modelo principal decidido: **Qwen-Image-Edit-2511** (D2). Plano completo em [SPEC.md](SPEC.md).
> Pendência: a máquina física chegar, pra validar E3/E4 e seguir pro benchmark (E6).
> Alternativa em andamento (2026-10-07): validar E3 numa RTX 5090 alugada no RunPod, ver [Teste na nuvem](#teste-na-nuvem-runpod-rtx-5090-32-gb).

## Como rodar (modo mock, sem GPU)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
copy .env.example .env          # ajuste PUBLIC_BASE_URL para o IP desta máquina
.venv\Scripts\python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

- `GET http://localhost:8000/health` → `"status": "ready"`
- Testes: `.venv\Scripts\python -m pytest`

No `.env` do Render:
```
RENDERING_API_PROVIDER=custom
CUSTOM_API_URL=http://<ip-desta-maquina>:8000/render
CUSTOM_API_KEY=            # mesmo valor de API_KEY, se usar
```
O render volta com a imagem de entrada e uma faixa vermelha "MOCK".

## Como rodar (modo real, GPU + Docker) — quando a máquina chegar

Pré-requisitos no host (SPEC.md E0): driver NVIDIA atualizado, Docker Desktop
(WSL2) ou Docker + `nvidia-container-toolkit`, e confirmar que `docker run --rm
--gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi` funciona.

```powershell
# 1. Token de leitura do HuggingFace (conta própria, não precisa aceitar termos --
#    o repositório Qwen/Qwen-Image-Edit-2511 não é gated), num arquivo não versionado:
"hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" | Out-File -Encoding ascii -NoNewline .hf_token.txt

# 2. .env com os ajustes desta máquina (PUBLIC_BASE_URL principalmente)
copy .env.example .env

# 3. Build (baixa ~20B de pesos durante o build -- demora e usa bastante disco, ver D3)
docker compose -f docker/docker-compose.yml build

# 4. Subir
docker compose -f docker/docker-compose.yml up -d

# 5. Esperar /health = "ready" (o modelo carrega em background no startup)
curl http://localhost:8000/health
```

No `.env` do Render, igual ao modo mock, só que agora o servidor responde com a
renderização real em vez da faixa "MOCK".

**Ainda não validado em GPU real** (sem máquina disponível até agora) — ver a lista
de pendências da etapa E3 em [SPEC.md](SPEC.md#5-etapas) antes de assumir que vai
funcionar de primeira. Problemas mais prováveis: estimativa de VRAM/precisão errada
pra a GPU real, ou a imagem `-runtime` faltando alguma dependência de build do
torchao para fp8 (nesse caso, trocar a tag base no `docker/Dockerfile` por uma
`-devel`).

## Teste na nuvem (RunPod, RTX 5090 32 GB)

Valida a etapa E3 (pipeline real) sem esperar a máquina física. Roda **sem Docker**: o
pod do RunPod já é um container, então não dá para rodar `docker build` lá dentro. Para
testar a E4 (a imagem Docker), use uma VM com Docker.

Com 32 GB, o `PRECISION=auto` escolhe **fp8 com CPU offload** (`CPU_OFFLOAD=auto`). O
transformer quantizado e o text encoder em bf16 não cabem juntos na GPU, então cada um
sobe quando é usado.

1. **Criar o pod**
   - GPU RTX 5090.
   - Template oficial PyTorch (qualquer um com Python 3.10+; o script instala o torch certo).
   - **Volume de ~150 GB** montado em `/workspace`. Os pesos (~60 GB) e o venv ficam
     nele. Com *Network Volume*, o download sobrevive a parar e recriar o pod.
   - *Expose HTTP Ports*: `8000`.
   - Em *Environment Variables*, `HF_TOKEN` (de preferência como RunPod Secret) e `API_KEY`.
2. **Levar o código para o pod.** O repo ainda não tem remote: crie o repositório
   privado no GitHub (E1) e faça `git clone` com um token, ou envie com
   `runpodctl send`. Não envie o `.env` nem o `.hf_token.txt`.
3. **Instalar e baixar os pesos** (no terminal do pod, na raiz do repo):
   ```bash
   bash scripts/runpod_setup.sh
   ```
   O script confere que o torch tem kernels para a GPU (sm_120) antes de baixar os pesos.
4. **Subir o servidor**:
   ```bash
   bash scripts/runpod_start.sh
   ```
   Ele mostra a URL pública (`https://<pod-id>-8000.proxy.runpod.net`). Espere o
   `/health` responder `"status": "ready"`: a quantização na CPU leva alguns minutos.
5. **No `.env` do Render**:
   ```
   RENDERING_API_PROVIDER=custom
   CUSTOM_API_URL=https://<pod-id>-8000.proxy.runpod.net/render
   CUSTOM_API_KEY=<mesmo valor de API_KEY>
   ```
   A URL é pública, então o upscale do fal também alcança o resultado. Isso resolve a
   limitação do fluxo PPTX descrita no SPEC.

**Atenção:**
- O proxy HTTP do RunPod corta requisições com mais de **~100 s**. Se um render
  passar disso, reduza `NUM_STEPS`/`MAX_SIDE` ou exponha a porta como TCP (IP público)
  em vez de HTTP.
- **Pare o pod** quando terminar. O volume continua cobrando, bem menos que a GPU.

---

## 1. O que o Render usa hoje no fal.ai

| Uso | Endpoint fal | Alternativa local |
|---|---|---|
| Render principal | `fal-ai/flux-kontext-lora` (`app/services/architecture_rendering.py`, `_render_with_falai`) | FLUX.1 Kontext [dev] / Qwen-Image-Edit / FLUX.2 klein 4B |
| LoRAs | `FAL_LORA_PATH` e `FAL_LORA_PATH_V2` (`.safetensors`) | Baixar e carregar local (só servem no Kontext) |
| Upscale | `fal-ai/seedvr/upscale/image`, `fal-ai/aura-sr` | SeedVR2 (pesado), AuraSR (leve) — ambos abertos |
| Análise de imagem | `fal-ai/any-llm/vision` (`app/services/image_analysis.py`) | VLM local (ex.: Qwen2.5-VL) ou manter no fal |

Parâmetros atuais (`app/core/config.py`): steps 30, guidance 2.5, png, `acceleration`, `resolution_mode: auto`.

### Atalho: provider `custom` já existe
`_render_with_custom_api` em `architecture_rendering.py` faz `POST` em `CUSTOM_API_URL` com:

```json
{ "sceneType": "...", "outputFormat": "jpg", "image": "<base64 ou url>", "prompt": "..." }
```

e aceita resposta com `image`, `imageUrl`, `url` ou `output`. Se o servidor local seguir esse contrato, dá pra testar **sem mudar código no Render**, só no `.env`:

```
RENDERING_API_PROVIDER=custom
CUSTOM_API_URL=http://<ip-da-maquina>:8000/render
```

Limitações: o `custom` não envia LoRA/steps/guidance (ficam em env vars do container) e tem timeout de 180 s.

---

## 2. Arquitetura do container

- Servidor **FastAPI + diffusers**:
  - carrega pipeline + LoRA **uma vez** no startup;
  - `POST /render` no contrato do provider `custom`;
  - `GET /health` (pronto quando o modelo terminar de carregar);
  - lock para processar **1 imagem por vez** na GPU;
  - detecta a VRAM no startup e escolhe bf16 completo ou quantizado (fp8/NF4).
- **Dockerfile**: base `pytorch/pytorch` com CUDA + `diffusers`, `transformers`, `accelerate`, `peft`, `bitsandbytes`.
- **Pesos do modelo dentro da imagem** (decisão preferida, "só ligar"):
  - imagem ~35 GB, funciona offline, token do HuggingFace só no build;
  - alternativa: baixar no 1º start num volume (imagem ~8–10 GB, precisa token em cada máquina).
- Rodar: `docker run --gpus all -p 8000:8000 render-local`
- Distribuir: registry privado (Azure Container Registry) ou `docker save` → `.tar` → `docker load`.
- **Nunca publicar a imagem em registry público** (licença dos pesos).

### O que NÃO vai no container (pré-requisitos do host)
- Driver NVIDIA atualizado.
- Docker Desktop com WSL2 (Windows) ou Docker + `nvidia-container-toolkit` (Linux).
- Teste: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
- Só NVIDIA (AMD exigiria outra imagem).

---

## 3. Passos

1. **Host**: driver NVIDIA, Docker, testar GPU no container.
2. **HuggingFace**: conta, aceitar termos do modelo escolhido, gerar token.
3. **LoRAs**: baixar os `.safetensors` de `FAL_LORA_PATH` / `_V2` (só se for usar Kontext).
4. **Servidor + Dockerfile + compose** (≈150 linhas de Python).
5. **Ligar no Render** via provider `custom` (`.env`), ajustar timeout se precisar.
6. **Testar**: mesmo lote no fal e local, comparar qualidade, tempo e Delta E.
7. **Depois, se valer**: provider `local` dedicado (LoRA, steps, v2), endpoints `/upscale`, VLM local.

Estimativa: ~1 dia para o container gerando; +1–2 dias para integração completa com upscale.

---

## 4. VRAM e desempenho (FLUX Kontext dev, ~12B)

| VRAM | Como roda |
|---|---|
| 24 GB+ (3090/4090/5090) | bf16 completo |
| 12–16 GB | quantizado (fp8 / NF4 / GGUF), leve perda de qualidade |
| < 12 GB | CPU offload, bem lento |

Referência: 4090, 1024 px, 30 steps ≈ 20–40 s por imagem (fal tende a ser mais rápido).

---

## 5. Licenças — modelos para uso comercial

| Modelo | Licença | Notas |
|---|---|---|
| **Qwen-Image-Edit** (1.0 / 2509 / 2511 / 2.0) | **Apache 2.0** ✅ | Melhor preservação entre os abertos. ~20B → 40 GB+ bf16, ~20–24 GB quantizado. Suporta LoRA (fal treina) |
| **FLUX.2 [klein] 4B** | **Apache 2.0** ✅ | Leve (~13 GB), rápido, menos fiel em detalhe fino |
| FLUX.2 [klein] 9B | Não comercial ❌ | |
| **Qwen-Image 2.1** | **Só pesquisa** ❌ | Lançado set/2026 — cuidado, licença mudou |
| FLUX.1 Kontext [dev] | Não comercial; licença comercial BFL **~US$ 999/mês** | Mantém os LoRAs atuais |

Obs.: usar Kontext **via API do fal** (como hoje) já é uso comercial coberto. A licença só pesa ao rodar os pesos por conta própria.

---

## 6. Requisito: não alterar acabamentos, puxadores e cores

Nenhum modelo garante isso sozinho (todos regeneram a imagem inteira). Estratégia:

1. **LoRA treinado nos produtos** — LoRAs do Kontext não migram; retreinar com os mesmos pares.
2. **Travar geometria** — ControlNet canny/depth ou guidance menor.
3. **Máscara + recomposição** (a garantia real) — gerar o render e colar de volta os pixels originais dos móveis via máscara, deixando a IA só com luz/ambiente. Se a entrada vem do 3D, a máscara pode sair do próprio 3D.
4. **Medir Delta E** nas regiões dos móveis (entrada × saída) e rejeitar renders que mudarem cor.

---

## 7. Recomendação

1. Container com **Qwen-Image-Edit 2511** quantizado (se a GPU tiver 24 GB); senão **FLUX.2 klein 4B**.
2. Lote de imagens reais: local × fal (Kontext + LoRA), comparar com Delta E.
3. Se o Qwen chegar perto, treinar LoRA nele com os pares existentes.
4. Com requisito rígido de cor/acabamento, implementar a recomposição por máscara independente do modelo.
5. Se só o Kontext + LoRA servir: comparar US$ 999/mês + hardware × custo atual do fal (dashboard de custos do Render).

---

## Fontes
- [Qwen Image Edit vs FLUX Klein: commercial image editing](https://www.virse.ai/blog/qwen-image-edit-vs-flux-klein-commercial-image-editing)
- [Qwen-Image-2.1: 7B Open Weights You Cannot Ship](https://cellcog.ai/blog/qwen-image-2-1/)
- [Qwen/Qwen-Image-2.1 · License discussion](https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/6)
- [Flux (text-to-image model) – Wikipedia](https://en.wikipedia.org/wiki/Flux_(text-to-image_model))
- [Flux 2 and Flux Kontext Explained](https://invideo.io/blog/flux-ai-image-generator/)
- [BFL – Self-serve dev license overview & pricing](https://help.bfl.ai/articles/9272590838-self-serve-dev-license-overview-pricing)
- [BFL – Open Weights Licensing](https://bfl.ai/licensing)
- [FLUX.1-Kontext-dev on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- [Instruction Image Editing 2026: Kontext vs Qwen vs Step1X](https://builderai.tools/blog/ai-image-editing-flux-kontext-qwen-image-edit-step1x)
