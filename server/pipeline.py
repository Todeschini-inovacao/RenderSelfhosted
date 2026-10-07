"""Interface comum dos pipelines de geração e seleção por configuração."""

from typing import Protocol

from PIL import Image

from server.config import Settings


class Pipeline(Protocol):
    name: str
    precision: str
    vram_gb: float | None

    def load(self) -> None:
        """Carrega pesos. Pode demorar minutos; roda fora do event loop."""

    def generate(self, image: Image.Image, prompt: str, scene_type: str) -> Image.Image:
        """Gera o render. Bloqueante; roda fora do event loop."""


def build_pipeline(settings: Settings) -> Pipeline:
    if settings.MOCK:
        from server.mock_pipeline import MockPipeline

        return MockPipeline(delay_s=settings.MOCK_DELAY_S)

    from server.qwen_pipeline import QwenImageEditPipeline

    return QwenImageEditPipeline(settings)
