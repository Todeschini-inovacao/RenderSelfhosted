from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # Pipeline
    MOCK: bool = True
    MOCK_DELAY_S: float = 2.0
    MODEL_ID: str = 'Qwen/Qwen-Image-Edit-2511'
    MODEL_REVISION: str = '6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9'
    # auto escolhe pela VRAM detectada (ver server/qwen_pipeline.py); fp8/nf4 = quantizado.
    PRECISION: Literal['auto', 'bf16', 'fp8', 'nf4'] = 'auto'
    # Só afeta PRECISION=fp8: auto liga o CPU offload abaixo de 40GB de VRAM (fp8 inteiro
    # na GPU pede ~37GB + ativações). Ver server/qwen_pipeline.py.
    CPU_OFFLOAD: Literal['auto', 'on', 'off'] = 'auto'
    NUM_STEPS: int = 30
    # Mapeado para `true_cfg_scale` do Qwen-Image-Edit (não `guidance_scale`, que fica
    # fixo em 1.0 -- exigido pelo checkpoint). 4.0 é o default do exemplo oficial do
    # model card; 2.5 era o valor calibrado pro FLUX Kontext do fal.ai, não se aplica
    # aqui. Recalibrar no benchmark (SPEC.md E6).
    GUIDANCE: float = 4.0
    LORA_PATH: str | None = None
    LORA_SCALE: float = 1.0
    MAX_SIDE: int = 1536

    # API
    API_KEY: str | None = None

    # Saída
    OUTPUT_DIR: str = 'outputs'
    RETURN_MODE: Literal['url', 'base64'] = 'url'
    PUBLIC_BASE_URL: str = 'http://localhost:8000'


settings = Settings()
