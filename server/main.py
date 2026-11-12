"""API de renderização local, compatível com o provider `custom` do Render.

Rodar: uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from server import image_io, storage, upscale
from server.config import resolve_prompt, settings
from server.pipeline import Pipeline, build_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('render-local')


class _State:
    status: Literal['loading', 'ready', 'error'] = 'loading'
    error: str | None = None
    pipeline: Pipeline | None = None
    # A GPU processa uma imagem por vez; demais requisições aguardam na fila
    lock = asyncio.Lock()


state = _State()


async def _load_pipeline() -> None:
    try:
        pipeline = build_pipeline(settings)
        logger.info('Carregando pipeline %s...', pipeline.name)
        await asyncio.to_thread(pipeline.load)
        if not settings.MOCK:
            # Carrega o upscaler padrão já no boot (a 1ª requisição não paga o load).
            await asyncio.to_thread(upscale.preload, settings.UPSCALER, settings)
        state.pipeline = pipeline
        state.status = 'ready'
        logger.info('Pipeline %s pronto (precisão: %s)', pipeline.name, pipeline.precision)
    except Exception as exc:
        state.status = 'error'
        state.error = str(exc)
        logger.exception('Falha ao carregar pipeline')


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.status, state.error, state.pipeline = 'loading', None, None
    state.lock = asyncio.Lock()
    task = asyncio.create_task(_load_pipeline())
    yield
    task.cancel()


app = FastAPI(title='Render Local', lifespan=lifespan)
Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
app.mount('/outputs', StaticFiles(directory=settings.OUTPUT_DIR), name='outputs')


class RenderRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image: str
    prompt: str
    scene_type: str = Field('interior', alias='sceneType')
    output_format: str = Field('jpg', alias='outputFormat')
    # Opcional, pra testes/comparação: sobrescreve o UPSCALER do servidor nesta requisição.
    # O Render não manda -- usa o padrão do servidor.
    upscaler: str | None = None
    # Opcional: semente fixa (mesma entrada + mesma seed = mesmo render). Sem ela, aleatório.
    seed: int | None = None


class RenderResponse(BaseModel):
    image: str
    elapsed_ms: int  # total no servidor (fila + geração + pós + upload)
    model: str
    # Detalhamento do elapsed_ms, pra investigar lentidão:
    queue_ms: int  # esperando a GPU (outras requisições na frente, neste worker)
    generate_ms: int  # geração no Qwen
    post_ms: int  # upscale + redimensionamento pro tamanho original
    upload_ms: int  # salvar/enviar o resultado (Azure/S3/disco/base64)


# Status pra requisição cujo cliente desistiu antes de chegar a vez dela na GPU
# (convenção do nginx, "client closed request"). Ninguém lê essa resposta.
CLIENT_CLOSED_REQUEST = 499


def _check_auth(authorization: str | None) -> None:
    if not settings.API_KEY:
        return
    if authorization != f'Bearer {settings.API_KEY}':
        raise HTTPException(status_code=401, detail='API key inválida')


@app.get('/ping')
async def ping():
    """Health check do RunPod Serverless (endpoint load balancing): 204 = inicializando,
    200 = pronto pra receber tráfego, outro código = worker com defeito (sai do pool)."""
    if state.status == 'ready':
        return {'status': 'healthy'}
    if state.status == 'loading':
        return Response(status_code=204)
    return JSONResponse({'status': 'error', 'error': state.error}, status_code=503)


@app.get('/health')
async def health():
    pipeline = state.pipeline
    return {
        'status': state.status,
        'error': state.error,
        'model': pipeline.name if pipeline else None,
        'precision': pipeline.precision if pipeline else None,
        'vram_gb': pipeline.vram_gb if pipeline else None,
        # O que está fundido no transformer (QwenImageEditPipeline.fused); o mock não tem.
        'loras': [
            {'name': Path(f['path']).name, 'scale': f['scale']}
            for f in getattr(pipeline, 'fused', [])
        ],
        'lightning': getattr(pipeline, 'lightning', False),
        'transformer_dir': settings.TRANSFORMER_DIR,
        'prompt_mode': settings.PROMPT_MODE,
        'steps': settings.NUM_STEPS,
        'output_size': settings.OUTPUT_SIZE,
        'upscaler': settings.UPSCALER,
        'busy': state.lock.locked(),
    }


@app.post('/render', response_model=RenderResponse)
async def render(
    req: RenderRequest, request: Request, authorization: str | None = Header(default=None)
):
    _check_auth(authorization)

    if state.status != 'ready' or state.pipeline is None:
        raise HTTPException(status_code=503, detail=f'Modelo não está pronto ({state.status})')

    upscaler = req.upscaler or settings.UPSCALER
    if upscaler not in upscale.NAMES:
        raise HTTPException(
            status_code=400, detail=f'upscaler inválido: {upscaler!r} (opções: {sorted(upscale.NAMES)})'
        )

    try:
        image = await image_io.load_image(req.image)
    except image_io.InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    original_size = image.size  # a saída volta neste tamanho (OUTPUT_SIZE=original)
    image = image_io.limit_size(image, settings.MAX_SIDE)

    t0 = time.monotonic()
    async with state.lock:
        t_start = time.monotonic()
        # Se o cliente desistiu enquanto esperava na fila (ex.: timeout do Render), não
        # gasta GPU com um render que ninguém vai receber -- libera a vez pro próximo.
        if await request.is_disconnected():
            logger.warning(
                'Cliente desistiu após %d ms na fila; render descartado',
                int((t_start - t0) * 1000),
            )
            return Response(status_code=CLIENT_CLOSED_REQUEST)
        try:
            output = await asyncio.to_thread(
                state.pipeline.generate,
                image, resolve_prompt(req.prompt, settings), req.scene_type, req.seed,
            )
            t_gen = time.monotonic()
            model_size = output.size
            output = await asyncio.to_thread(
                upscale.finalize, output, original_size, upscaler, settings
            )
        except Exception as exc:
            logger.exception('Falha na geração')
            raise HTTPException(status_code=500, detail=f'Falha na geração: {exc}')
    t_post = time.monotonic()
    queue_ms = int((t_start - t0) * 1000)
    generate_ms = int((t_gen - t_start) * 1000)
    post_ms = int((t_post - t_gen) * 1000)
    logger.info(
        'Fila %d ms + geração %d ms (%dx%d) + pós %d ms (%s -> %dx%d)',
        queue_ms, generate_ms, *model_size, post_ms, upscaler, output.width, output.height,
    )

    if settings.RETURN_MODE == 'base64':
        result = image_io.to_data_url(output, req.output_format)
    elif settings.RETURN_MODE in ('s3', 'azure'):
        try:
            result = await asyncio.to_thread(storage.upload, output, req.output_format, settings)
        except Exception as exc:
            logger.exception('Falha no upload pro bucket')
            # 500, não 502: o Cloudflare na frente do RunPod Serverless troca o corpo de
            # respostas 502 por um "error code: 502" genérico e o detalhe se perde.
            raise HTTPException(status_code=500, detail=f'Falha no upload do resultado: {exc}')
    else:
        name = image_io.save_output(output, req.output_format, settings.OUTPUT_DIR)
        result = f'{settings.PUBLIC_BASE_URL.rstrip("/")}/outputs/{name}'
    t_end = time.monotonic()
    upload_ms = int((t_end - t_post) * 1000)
    elapsed_ms = int((t_end - t0) * 1000)

    logger.info(
        'Render ok: %s %dx%d em %d ms (fila %d, geração %d, pós %d, upload %d)',
        req.scene_type, output.width, output.height, elapsed_ms,
        queue_ms, generate_ms, post_ms, upload_ms,
    )
    return RenderResponse(
        image=result, elapsed_ms=elapsed_ms, model=state.pipeline.name,
        queue_ms=queue_ms, generate_ms=generate_ms, post_ms=post_ms, upload_ms=upload_ms,
    )
