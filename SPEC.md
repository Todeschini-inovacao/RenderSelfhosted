# SPEC — Render Local (inferência em GPU via Docker)

> Contexto, pesquisa de modelos e licenças: ver [README.md](README.md).
> Este documento é o **plano de execução**: etapas em ordem, com tarefas e critério de aceite.

- **Status:** E1, E2, E3 e E4 concluídas no código (pipeline real + Docker escritos, sem GPU pra testar ainda); falta E0 (a máquina física) pra validar e seguir pra E5/E6
- **Última atualização:** 2026-10-07

---

## 1. Objetivo

Ter um container Docker que, ligado numa máquina com GPU NVIDIA, exponha uma API HTTP de renderização compatível com o provider `custom` do projeto Render, permitindo comparar qualidade, tempo e custo contra o fal.ai **sem alterar o código do Render**.

### Escopo
- Servidor de inferência (FastAPI) com um modelo de edição de imagem de licença comercial.
- Empacotamento em Docker, com pesos embutidos na imagem ("só ligar").
- Integração com o Render via `RENDERING_API_PROVIDER=custom`.
- Script de benchmark local × fal (tempo + diferença de cor nos móveis).

### Fora de escopo (por enquanto)
- Substituir o fal.ai em produção.
- Upscale e análise de imagem locais (etapas opcionais no fim).
- Escalar para várias GPUs / fila distribuída.

### Critérios de sucesso do projeto
- Render local gerado a partir do fluxo normal do Render (via `custom`).
- Tempo por imagem ≤ 60 s na GPU de teste.
- Diferença de cor (Delta E 2000) média nas regiões de móveis ≤ 2,0 (imperceptível a olho) — com recomposição por máscara se necessário.
- Qualidade visual aprovada em comparação lado a lado com o fal.

---

## 2. Decisões em aberto

| # | Decisão | Opções | Default proposto | Bloqueia |
|---|---|---|---|---|
| D1 | GPU da máquina de teste | modelo / VRAM | **VRAM confirmada: 24 GB+** (modelo exato da placa ainda em aberto) | E3 |
| D2 | Modelo principal | Qwen-Image-Edit 2511 · FLUX.2 klein 4B · Kontext dev (licença paga) | **Confirmado: Qwen-Image-Edit-2511**, pela regra acima (VRAM ≥ 24 GB) | E3 |
| D3 | Pesos na imagem ou volume | embutido (~35 GB) · download no 1º start | embutido | E4 |
| D4 | Distribuição da imagem | Azure Container Registry · `docker save` (.tar) | ACR | E9 |
| D5 | Lote de imagens de teste | quais cenas/produtos | ~20 imagens reais, cobrindo marcas Domo/Inspira/Galeria | E6 |

> Confirmar sempre a licença do checkpoint exato no HuggingFace antes de usar (ex.: Qwen-Image 2.1 é só pesquisa; FLUX.2 klein 9B é não comercial).

---

## 3. Estrutura proposta do repositório

```
Render-Local/
├── README.md              # contexto e pesquisa
├── SPEC.md                # este plano
├── .env.example
├── .gitignore             # ignora models/, outputs/, .env
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── server/
│   ├── main.py            # FastAPI: /render, /health
│   ├── config.py          # env vars
│   ├── pipeline.py        # carga do modelo, escolha por VRAM, geração
│   ├── mock_pipeline.py   # devolve a própria imagem (dev sem GPU)
│   └── image_io.py        # base64/URL <-> PIL
├── scripts/
│   ├── download_models.py # baixa pesos (usado no build)
│   └── benchmark.py       # local × fal, tempo + Delta E
├── samples/               # imagens de teste (+ máscaras)
└── outputs/               # resultados do benchmark
```

---

## 4. Contrato da API

Compatível com `_render_with_custom_api` do Render (`app/services/architecture_rendering.py`).

### `POST /render`
Request:
```json
{
  "sceneType": "interior",
  "outputFormat": "jpg",
  "image": "data:image/png;base64,... | https://...",
  "prompt": "..."
}
```
Response `200`:
```json
{ "image": "http://<PUBLIC_BASE_URL>/outputs/<id>.jpg", "elapsed_ms": 23150, "model": "qwen-image-edit-2511" }
```
Por padrão (`RETURN_MODE=url`) a imagem é salva em `OUTPUT_DIR` e servida em `/outputs/`. Motivo: o Render grava o resultado como `rendered_url` no banco, então um data URI de vários MB iria parar lá. `RETURN_MODE=base64` devolve `data:image/...;base64,...`.

> **Limitação conhecida:** no fluxo PPTX (`render_pptx.py`) o Render envia `rendered_url` para o upscale do **fal.ai**, que não alcança um IP local. Para testar, usar o fluxo de imagem avulsa (`render_image.py`) ou expor o servidor publicamente (ex.: túnel), ou implementar `/upscale` local (E10).
Erros: `400` imagem inválida · `503` modelo ainda carregando · `500` falha na geração (com `detail`).

Auth opcional: se `API_KEY` definido, exigir `Authorization: Bearer <API_KEY>` (o Render já envia `CUSTOM_API_KEY` assim).

### `GET /health`
```json
{ "status": "loading | ready | error", "model": "...", "precision": "bf16 | fp8 | nf4", "vram_gb": 24.0 }
```

### Variáveis de ambiente do container
| Var | Default | Descrição |
|---|---|---|
| `MODEL_ID` | definido em D2 | checkpoint HF |
| `PRECISION` | `auto` | `auto` escolhe por VRAM; ou `bf16` / `fp8` / `nf4` |
| `NUM_STEPS` | `30` | passos de inferência |
| `GUIDANCE` | `2.5` | guidance scale |
| `LORA_PATH` | vazio | `.safetensors` opcional |
| `LORA_SCALE` | `1.0` | |
| `MAX_SIDE` | `1536` | redimensiona entrada maior que isso |
| `API_KEY` | vazio | auth opcional |
| `MOCK` | `true` | usa `mock_pipeline` (sem GPU) |
| `MOCK_DELAY_S` | `2` | latência simulada do mock |
| `OUTPUT_DIR` | `outputs` | onde os renders são salvos |
| `RETURN_MODE` | `url` | `url` ou `base64` |
| `PUBLIC_BASE_URL` | `http://localhost:8000` | base das URLs devolvidas; usar o IP/host alcançável pelo Render |

---

## 5. Etapas

### E0 — Preparar a máquina de teste
- [ ] Anotar GPU e VRAM (`nvidia-smi`) → fecha **D1**
- [ ] Atualizar driver NVIDIA
- [ ] Instalar Docker Desktop (WSL2) ou Docker + `nvidia-container-toolkit` (Linux)
- [ ] `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
- [ ] Conta HuggingFace + token (read) + aceitar termos do modelo, se houver

**Aceite:** `nvidia-smi` funciona dentro de um container.

### E1 — Esqueleto do repositório
- [x] `git init`, `.gitignore`, `.env.example`
- [x] Estrutura inicial (`server/`, `tests/`); `docker/`, `scripts/`, `samples/` entram nas próximas etapas
- [ ] Primeiro commit
- [ ] Repositório privado no GitHub (org Todeschini-inovacao?)

**Aceite:** repo versionado com a estrutura vazia.

### E2 — Servidor com pipeline mock (não precisa de GPU) ✅
- [x] `server/main.py` com `/render` e `/health` no contrato da seção 4
- [x] `image_io.py`: aceitar base64 (com/sem prefixo `data:`) e URL
- [x] `mock_pipeline.py`: devolve a imagem de entrada com uma faixa "MOCK"
- [x] Lock para 1 geração por vez
- [x] Testes (`tests/test_api.py`, 6 casos)
- [x] Validado chamando `_render_with_custom_api` do próprio Render contra o servidor (2026-09-23)
- [ ] Teste pelo frontend/worker do Render (fluxo de imagem avulsa)

**Aceite:** um render disparado pelo Render volta com a imagem mock. Valida o contrato **antes** de mexer com GPU.

### E3 — Pipeline real
- [x] `server/qwen_pipeline.py`: carrega `Qwen/Qwen-Image-Edit-2511` com `diffusers` — classe confirmada no model card oficial: `QwenImageEditPlusPipeline`, disponível desde `diffusers==0.41.0` estável (não precisa da instalação via git do card)
- [x] Detecção de VRAM e escolha de precisão (`auto`): ≥64GB bf16 completo (pesos ~58GB), 18–64GB fp8 (torchao, só quantiza o `transformer`; exige compute capability 8.9+, senão nf4), <18GB nf4 (bitsandbytes) — limiares de partida, recalibrar no benchmark (E6)
- [x] Carregar LoRA opcional (`load_lora_weights` + `fuse_lora`, mesma env var `LORA_PATH`/`LORA_SCALE` de antes — mas ver nota: LoRA do Kontext não migra pro Qwen, isso só serve se/quando E8 treinar um novo)
- [x] Revisão do checkpoint fixada (`MODEL_REVISION`, mitiga o risco da seção 6) — confirmada em 2026-10-06 via API do HF (`6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`, Apache 2.0, não-gated)
- [x] Carregar modelo no startup em background; `/health` = `loading` até pronto — já coberto pelo `main.py` da E2, sem mudança: `build_pipeline()` agora devolve `QwenImageEditPipeline` quando `MOCK=false`
- [x] Ajustes pra RTX 5090 32GB (2026-10-07): base Docker e torch com CUDA 13.0 (Blackwell/sm_120 exige 12.8+); ramo fp8 carrega na CPU e usa `enable_model_cpu_offload()` abaixo de 40GB (`CPU_OFFLOAD=auto`), porque transformer fp8 + text encoder bf16 (~37GB) não cabem em 32GB
- [x] Scripts pra validar num pod do RunPod sem Docker: `scripts/runpod_setup.sh` e `scripts/runpod_start.sh` (README, "Teste na nuvem")
- [ ] **PENDENTE (precisa de GPU — física ou RunPod):** rodar direto na máquina (sem Docker) com algumas imagens de `samples/` — nada disso foi testado em hardware real ainda. Em especial, confirmar: (a) a combinação quantização + `device_map`/`.to("cuda")` nos ramos fp8/nf4 (ver comentários em `qwen_pipeline.py`); (b) se a imagem `-runtime` (sem nvcc) basta pro torchao/Triton em fp8, ou se precisa trocar pra `-devel`.

**Aceite:** gera imagem coerente em ≤ 60 s na GPU de teste, sem estourar VRAM.

### E4 — Docker
- [x] `docker/Dockerfile` (base `pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime` — trocar a tag se o driver da máquina de teste não acompanhar CUDA 12.6, ver comentário no arquivo)
- [x] `scripts/download_models.py` executado no build (`--mount=type=secret` para o `HF_TOKEN`, sem gravar o token na imagem)
- [x] `HF_HUB_OFFLINE=1` em runtime (só depois do download, senão quebra o próprio download)
- [x] `docker/docker-compose.yml` com GPU, porta 8000, `restart: unless-stopped`, healthcheck em `/health` — validado com `docker compose config` (sintaxe), não com build real (sem Docker daemon/GPU neste ambiente)
- [x] Documentar no README (seção "Como rodar (modo real, GPU + Docker)")

**Aceite:** numa máquina limpa (só driver + Docker), `docker run --gpus all -p 8000:8000 render-local` sobe e responde `ready` sem internet. **Não testado ainda** — falta a máquina (E0).

### E5 — Integração com o Render
- [ ] `.env` do Render: `RENDERING_API_PROVIDER=custom`, `CUSTOM_API_URL=http://<ip>:8000/render`
- [ ] Verificar timeout de 180 s em `_render_with_custom_api`; aumentar se necessário
- [ ] Testar fluxo completo pelo frontend/worker do Render

**Aceite:** job completo no Render usando o container, sem mudanças de código (ou só o timeout).

### E6 — Benchmark local × fal
- [ ] Montar lote de teste (D5) em `samples/`, com máscara dos móveis por imagem (PNG preto/branco)
- [ ] `scripts/benchmark.py`: para cada imagem, gerar no fal (Kontext + LoRA atual) e local; salvar em `outputs/`
- [ ] Medir: tempo, Delta E 2000 médio/máx na região da máscara (entrada × saída)
- [ ] Gerar relatório (CSV + página HTML lado a lado)

**Aceite:** relatório com números para decidir se vale seguir.

### E7 — Preservação de acabamentos (recomposição por máscara)
- [ ] Pós-processamento: colar pixels originais dos móveis sobre o render usando a máscara, com borda suavizada (feather)
- [ ] Ajuste de luz: transferir luminância/sombra do render para a região colada, mantendo a crominância original
- [ ] Definir de onde vem a máscara em produção (export do 3D? segmentação automática, ex.: SAM?)
- [ ] Opcional: ControlNet canny/depth para travar geometria
- [ ] Re-rodar E6 com recomposição ligada

**Aceite:** Delta E médio ≤ 2,0 nos móveis e visual aprovado.

### E8 — LoRA no modelo novo (se E6 indicar necessidade)
- [ ] Reunir os pares (entrada → render aprovado) usados no LoRA do Kontext
- [ ] Treinar LoRA para o modelo de D2 (fal trainer ou ai-toolkit local)
- [ ] Re-rodar E6 com o LoRA

**Aceite:** qualidade igual ou superior ao Kontext + LoRA no lote de teste.

### E9 — Distribuição
- [ ] Publicar imagem no registry privado (D4) ou gerar `.tar` com `docker save`
- [ ] Instruções de 3 passos no README para outra máquina
- [ ] **Nunca** publicar em registry público

**Aceite:** outra pessoa sobe o container numa segunda máquina só seguindo o README.

### E10 — Opcionais
- [ ] `POST /upscale` com AuraSR (e SeedVR2 se a VRAM permitir)
- [ ] Análise de imagem local (VLM) para substituir `fal-ai/any-llm/vision`
- [ ] Provider `local` dedicado no Render (envia LoRA, steps, guidance e suporta a variante v2)
- [ ] Comparativo de custo: hardware/energia × gasto atual no fal (dashboard de custos do Render)

---

## 6. Riscos

| Risco | Mitigação |
|---|---|
| VRAM insuficiente para Qwen-Image-Edit | Quantização fp8/NF4 ou fallback para FLUX.2 klein 4B |
| Qualidade abaixo do Kontext + LoRA | E8 (treinar LoRA) e E7 (recomposição) |
| Mudança de cor/acabamento | E7 é obrigatório se o requisito for rígido; Delta E como gate |
| Licença de checkpoint muda entre versões | Fixar revisão (commit hash) do HF no `download_models.py` e registrar a licença |
| Imagem Docker muito grande (~35 GB) | Aceitável para teste; alternativa D3 = volume |
| Timeout do provider `custom` (180 s) | Aumentar timeout ou reduzir steps/resolução |

---

## 7. Ordem sugerida para começar

1. **E0 + D1** — descobrir a GPU (define todo o resto).
2. **E1 + E2** — dá pra fazer já, **sem GPU**, e valida a integração com o Render.
3. **E3 → E4 → E5** — modelo real no container.
4. **E6** — números para decidir; E7/E8 conforme o resultado.
