"""API de renderização local, compatível com o provider `custom` do Render.

Rodar: uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from server import image_io
from server.config import settings
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


class RenderResponse(BaseModel):
    image: str
    elapsed_ms: int
    model: str


def _check_auth(authorization: str | None) -> None:
    if not settings.API_KEY:
        return
    if authorization != f'Bearer {settings.API_KEY}':
        raise HTTPException(status_code=401, detail='API key inválida')


@app.get('/health')
async def health():
    pipeline = state.pipeline
    return {
        'status': state.status,
        'error': state.error,
        'model': pipeline.name if pipeline else None,
        'precision': pipeline.precision if pipeline else None,
        'vram_gb': pipeline.vram_gb if pipeline else None,
        'busy': state.lock.locked(),
    }


@app.post('/render', response_model=RenderResponse)
async def render(req: RenderRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)

    if state.status != 'ready' or state.pipeline is None:
        raise HTTPException(status_code=503, detail=f'Modelo não está pronto ({state.status})')

    try:
        image = await image_io.load_image(req.image)
    except image_io.InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    image = image_io.limit_size(image, settings.MAX_SIDE)

    t0 = time.monotonic()
    async with state.lock:
        try:
            output = await asyncio.to_thread(
                state.pipeline.generate, image, req.prompt, req.scene_type
            )
        except Exception as exc:
            logger.exception('Falha na geração')
            raise HTTPException(status_code=500, detail=f'Falha na geração: {exc}')
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if settings.RETURN_MODE == 'base64':
        result = image_io.to_data_url(output, req.output_format)
    else:
        name = image_io.save_output(output, req.output_format, settings.OUTPUT_DIR)
        result = f'{settings.PUBLIC_BASE_URL.rstrip("/")}/outputs/{name}'

    logger.info(
        'Render ok: %s %dx%d em %d ms', req.scene_type, output.width, output.height, elapsed_ms
    )
    return RenderResponse(image=result, elapsed_ms=elapsed_ms, model=state.pipeline.name)
