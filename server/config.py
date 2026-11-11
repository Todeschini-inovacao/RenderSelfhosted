from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # Pipeline
    MOCK: bool = True
    MOCK_DELAY_S: float = 2.0
    MODEL_ID: str = 'Qwen/Qwen-Image-Edit-2511'
    MODEL_REVISION: str = '6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9'
    # Pasta local com os arquivos do modelo (baixados com scripts/download_models.py e
    # MODEL_DIR definido). Se definido, é usado no lugar de MODEL_ID + cache do HF.
    MODEL_DIR: str | None = None
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
    # LoRA fundido no transformer no load (antes da quantização fp8). LORA_SCALE = força.
    LORA_PATH: str | None = None
    LORA_SCALE: float = 1.0
    # LoRA Lightning (destilação de poucos steps, lightx2v): fundido com força 1.0 antes do
    # LORA_PATH. Com ele o servidor usa o scheduler próprio do Lightning e true_cfg_scale=1
    # (GUIDANCE é ignorado); NUM_STEPS deve ser 4 ou 8, conforme o arquivo.
    LIGHTNING_PATH: str | None = None
    # Pasta com um transformer já fundido (gerado por scripts/fuse_transformer.py). Se
    # definido, LORA_PATH/LIGHTNING_PATH não são fundidos de novo -- o que foi fundido vem
    # do fused_info.json da pasta. Corta o tempo de subida.
    TRANSFORMER_DIR: str | None = None
    # request = usa o prompt que vem na requisição; strict = ignora e usa STRICT_PROMPT
    # (o que deu a cor mais fiel nos testes do LoRA -- ver scripts/bench_lora.py).
    PROMPT_MODE: Literal['request', 'strict'] = 'request'
    MAX_SIDE: int = 1536

    # API
    API_KEY: str | None = None

    # Saída
    OUTPUT_DIR: str = 'outputs'
    # url = salva em OUTPUT_DIR e serve em /outputs; base64 = data URI na resposta;
    # s3 = sobe pro bucket (S3_*) e devolve a URL de lá -- o modo do Serverless.
    RETURN_MODE: Literal['url', 'base64', 's3'] = 'url'
    PUBLIC_BASE_URL: str = 'http://localhost:8000'

    # Bucket S3-compatível (RETURN_MODE=s3). Cloudflare R2: S3_ENDPOINT_URL=
    # https://<account_id>.r2.cloudflarestorage.com e S3_REGION=auto. Credenciais em
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
    S3_BUCKET: str | None = None
    S3_ENDPOINT_URL: str | None = None
    S3_REGION: str | None = None
    S3_PREFIX: str = 'renders/'
    # URL pública do bucket (domínio próprio ou r2.dev). Sem ela, devolve URL
    # pré-assinada, que expira (máx. 7 dias) -- ruim se o Render guarda a URL no banco.
    S3_PUBLIC_BASE_URL: str | None = None
    S3_PRESIGN_EXPIRES: int = 7 * 24 * 3600

    @model_validator(mode='after')
    def _check_s3(self):
        if self.RETURN_MODE == 's3' and not self.S3_BUCKET:
            raise ValueError('RETURN_MODE=s3 exige S3_BUCKET')
        return self


STRICT_PROMPT = (
    'Make this image look like a real photograph. Only change lighting, shadows, '
    'reflections and surface realism. Do not change any color, material or finish: '
    'white stays white, every surface keeps its exact original color and texture. '
    'Do not add or remove any object, window, door, plant or decoration. Keep the exact '
    'same furniture, layout and camera angle.'
)


def resolve_prompt(request_prompt: str, settings: 'Settings') -> str:
    return STRICT_PROMPT if settings.PROMPT_MODE == 'strict' else request_prompt


settings = Settings()
