"""Pipeline real: Qwen-Image-Edit via diffusers.

Classe confirmada no model card oficial (huggingface.co/Qwen/Qwen-Image-Edit-2511):
`QwenImageEditPlusPipeline`, disponível desde diffusers==0.41.0 -- não precisa do
`pip install git+https://github.com/huggingface/diffusers` que o card sugere (isso é
só cautela genérica de lançamento, antes da classe virar release estável).

VRAM por precisão: bf16 completo pede ~64GB (pesos ~58GB + ativações); entre ~18GB e
64GB, quantizar -- fp8 (torchao) preserva mais qualidade que nf4, mas exige GPU com
compute capability 8.9+ (Ada/RTX 40xx, Hopper, Blackwell/RTX 50xx); nf4 (bitsandbytes)
funciona em qualquer GPU CUDA. Os limiares abaixo são ponto de partida -- ajustar depois
do benchmark (SPEC.md E6) com a GPU real.

PENDENTE (SPEC.md E3): a combinação quantização + `device_map`/`pipe.to("cuda")` varia
por precisão (comentada em cada ramo abaixo) -- precisa validar na GPU real, ainda não
testado nesta sessão (sem GPU disponível aqui).
"""

import json
import logging
import math
from pathlib import Path

import torch
from PIL import Image

from server.config import Settings

logger = logging.getLogger('render-local')

# Pontos de partida -- recalibrar no benchmark (E6) com a GPU real.
# bf16 completo: transformer ~40GB + text encoder ~16GB + VAE = ~58GB + ativações.
_VRAM_BF16_GB = 64.0
_VRAM_FP8_GB = 18.0
# fp8 do torchao exige compute capability 8.9+ (Ada, Hopper, Blackwell); Ampere (A40,
# A6000, 3090) cai pra nf4 no auto.
_FP8_MIN_CAPABILITY = (8, 9)
# fp8 só quantiza o transformer (~20GB); o text encoder (Qwen2.5-VL 7B, ~16GB bf16) e o
# VAE continuam em bf16 -- tudo junto na GPU dá ~37GB + ativações. Abaixo disso (ex.: RTX
# 5090 32GB, 4090 24GB), CPU offload: cada componente sobe pra GPU só quando é usado.
_VRAM_FP8_NO_OFFLOAD_GB = 40.0
# Com LoRA em fp8: a partir daqui o transformer bf16 (~41 GB) cabe na GPU pra fundir e
# quantizar lá (minutos a menos que na CPU).
_VRAM_FUSE_ON_GPU_GB = 44.0

# Scheduler do Lightning (lightx2v/Qwen-Image-Lightning): shift fixo log(3) com dynamic
# shifting. Mesmo config usado em scripts/bench_lightning.py e scripts/bench_lora.py.
_LIGHTNING_SCHEDULER = {
    'base_image_seq_len': 256,
    'base_shift': math.log(3),
    'invert_sigmas': False,
    'max_image_seq_len': 8192,
    'max_shift': math.log(3),
    'num_train_timesteps': 1000,
    'shift': 1.0,
    'shift_terminal': None,
    'stochastic_sampling': False,
    'time_shift_type': 'exponential',
    'use_beta_sigmas': False,
    'use_dynamic_shifting': True,
    'use_exponential_sigmas': False,
    'use_karras_sigmas': False,
}

FUSED_INFO_FILE = 'fused_info.json'


def fusion_plan(settings: Settings) -> list[tuple[str, float]]:
    """LoRAs a fundir, na ordem: Lightning (força 1.0) e depois o LoRA de estilo."""
    loras = []
    if settings.LIGHTNING_PATH:
        loras.append((settings.LIGHTNING_PATH, 1.0))
    if settings.LORA_PATH:
        loras.append((settings.LORA_PATH, settings.LORA_SCALE))
    return loras


def fuse_loras(pipe, loras: list[tuple[str, float]]) -> None:
    for path, scale in loras:
        logger.info('Fundindo LoRA %s (força %s)', path, scale)
        pipe.load_lora_weights(path)
        pipe.fuse_lora(lora_scale=scale)
        pipe.unload_lora_weights()


def _detect_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def _supports_fp8() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= _FP8_MIN_CAPABILITY


def _resolve_precision(requested: str, vram_gb: float | None, fp8_ok: bool) -> str:
    if requested != 'auto':
        return requested
    if vram_gb is None:
        raise RuntimeError('PRECISION=auto exige GPU CUDA disponível para detectar a VRAM.')
    if vram_gb >= _VRAM_BF16_GB:
        return 'bf16'
    if vram_gb >= _VRAM_FP8_GB and fp8_ok:
        return 'fp8'
    return 'nf4'


def _resolve_cpu_offload(requested: str, vram_gb: float | None) -> bool:
    if requested != 'auto':
        return requested == 'on'
    return vram_gb is None or vram_gb < _VRAM_FP8_NO_OFFLOAD_GB


class QwenImageEditPipeline:
    """Carrega Qwen-Image-Edit e gera renders a partir de uma imagem de referência."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self.name = settings.MODEL_ID
        self.precision = 'unresolved'
        self.vram_gb: float | None = None
        # O que está fundido no transformer (LoRAs + força) e se é Lightning -- vem das
        # settings ou, com TRANSFORMER_DIR, do fused_info.json da pasta.
        self.fused: list[dict] = []
        self.lightning = False
        self._pipe = None

    def load(self) -> None:
        from diffusers import QwenImageEditPlusPipeline

        # MODEL_DIR (pasta local com os arquivos do repo) tem prioridade sobre MODEL_ID; numa
        # pasta local a revisão não se aplica -- quem fixa a revisão é o download.
        model_source = self._settings.MODEL_DIR or self._settings.MODEL_ID
        revision = None if self._settings.MODEL_DIR else self._settings.MODEL_REVISION
        self.vram_gb = _detect_vram_gb()
        self.precision = _resolve_precision(self._settings.PRECISION, self.vram_gb, _supports_fp8())
        logger.info(
            'Carregando %s (VRAM detectada: %s, precisão: %s)',
            model_source,
            f'{self.vram_gb:.1f} GB' if self.vram_gb else 'desconhecida',
            self.precision,
        )

        # Transformer pré-fundido (TRANSFORMER_DIR) ou LoRAs a fundir agora no load.
        transformer_dir = self._settings.TRANSFORMER_DIR
        # Transformer já salvo em fp8 (scripts/quantize_transformer.py): carrega direto,
        # sem quantizar de novo.
        prequantized = False
        if transformer_dir:
            info = json.loads((Path(transformer_dir) / FUSED_INFO_FILE).read_text(encoding='utf-8'))
            self.fused = info['loras']
            self.lightning = bool(info['lightning'])
            prequantized = info.get('quantization') == 'fp8_weight_only'
            if prequantized and self.precision != 'fp8':
                raise ValueError(
                    f'{transformer_dir} está em fp8; use PRECISION=fp8 (resolvido: {self.precision}).'
                )
            loras: list[tuple[str, float]] = []
            logger.info('Transformer pré-fundido: %s (%s)', transformer_dir, self.fused)
        else:
            loras = fusion_plan(self._settings)
            self.fused = [{'path': p, 'scale': s} for p, s in loras]
            self.lightning = bool(self._settings.LIGHTNING_PATH)

        def load_transformer_bf16():
            from diffusers import QwenImageTransformer2DModel

            return QwenImageTransformer2DModel.from_pretrained(
                transformer_dir, torch_dtype=torch.bfloat16
            )

        if self.precision == 'bf16':
            extra = {'transformer': load_transformer_bf16()} if transformer_dir else {}
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                model_source,
                revision=revision,
                torch_dtype=torch.bfloat16,
                **extra,
            )
            pipe.to('cuda')
            fuse_loras(pipe, loras)

        elif self.precision == 'fp8':
            # Padrão oficial de quantização por componente (docs/quantization/torchao.md):
            # só o transformer (a parte grande, ~20B) é quantizado; text encoder e VAE
            # ficam em bf16. Sem device_map, a quantização acontece na CPU durante o load
            # (mais lento, mas sem o pico de VRAM de ter bf16 + fp8 ao mesmo tempo na
            # GPU) e depois o pipeline vai pra GPU inteiro ou com CPU offload -- os dois
            # caminhos são os indicados na mesma doc.
            from diffusers import PipelineQuantizationConfig, TorchAoConfig
            from torchao.quantization import Float8WeightOnlyConfig, quantize_

            if prequantized:
                # Pesos já em fp8 do torchao (pickle -- só pastas geradas por nós).
                from diffusers import QwenImageTransformer2DModel

                transformer = QwenImageTransformer2DModel.from_pretrained(
                    transformer_dir, torch_dtype=torch.bfloat16, use_safetensors=False
                )
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    model_source, revision=revision, torch_dtype=torch.bfloat16,
                    transformer=transformer,
                )
            elif loras or transformer_dir:
                # Com LoRA: carrega em bf16, funde o LoRA e só então quantiza -- fundir
                # num transformer já quantizado pelo torchao não é confiável. Com VRAM
                # pra caber o transformer bf16 (~41 GB), funde/quantiza na GPU (bem mais
                # rápido); o text encoder ainda está na CPU nesse ponto. Com
                # TRANSFORMER_DIR o transformer já vem fundido -- só quantiza.
                extra = {'transformer': load_transformer_bf16()} if transformer_dir else {}
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    model_source, revision=revision, torch_dtype=torch.bfloat16, **extra
                )
                if self.vram_gb and self.vram_gb >= _VRAM_FUSE_ON_GPU_GB:
                    pipe.transformer.to('cuda')
                fuse_loras(pipe, loras)
                quantize_(pipe.transformer, Float8WeightOnlyConfig())
                torch.cuda.empty_cache()
            else:
                quant_config = PipelineQuantizationConfig(
                    quant_mapping={'transformer': TorchAoConfig(Float8WeightOnlyConfig())}
                )
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    model_source,
                    revision=revision,
                    quantization_config=quant_config,
                    torch_dtype=torch.bfloat16,
                )
            if _resolve_cpu_offload(self._settings.CPU_OFFLOAD, self.vram_gb):
                logger.info('fp8 com CPU offload (VRAM abaixo de %.0f GB)', _VRAM_FP8_NO_OFFLOAD_GB)
                pipe.enable_model_cpu_offload()
            else:
                pipe.to('cuda')

        elif self.precision == 'nf4':
            # bitsandbytes só quantiza o transformer (passado separado pro pipeline,
            # igual ao padrão oficial pra Flux em docs/quantization/bitsandbytes.md).
            # Depois de carregado com device_map="auto", NÃO chamar pipe.to("cuda") de
            # novo -- quebra com módulos já quantizados em 4bit.
            from diffusers import AutoModel, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            transformer_kwargs = (
                {'pretrained_model_name_or_path': transformer_dir}
                if transformer_dir
                else {
                    'pretrained_model_name_or_path': model_source,
                    'revision': revision,
                    'subfolder': 'transformer',
                }
            )
            transformer = AutoModel.from_pretrained(
                **transformer_kwargs,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
            )
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                model_source,
                revision=revision,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
            # Não testado: fundir LoRA em transformer nf4 (bitsandbytes).
            fuse_loras(pipe, loras)

        else:
            raise ValueError(f'Precisão desconhecida: {self.precision}')

        if self.lightning:
            from diffusers import FlowMatchEulerDiscreteScheduler

            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(_LIGHTNING_SCHEDULER)
            logger.info('Lightning: scheduler próprio, sem CFG, %s steps', self._settings.NUM_STEPS)

        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    def generate(self, image: Image.Image, prompt: str, scene_type: str) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError('Pipeline não carregado -- chame load() primeiro.')

        # scene_type não é usado pelo Qwen-Image-Edit (o fal.ai também só usava isso pra
        # registro, não mandava pro modelo) -- mantido só pra bater com o Protocol/contrato
        # da API (ver server/pipeline.py e o POST /render em server/main.py).
        # true_cfg_scale é o knob real de CFG (equivalente ao FAL_GUIDANCE_SCALE do Render);
        # guidance_scale=1.0 fixo é exigido pelo checkpoint (guidance-distilled), igual ao
        # exemplo oficial do model card. Lightning é destilado sem CFG: true_cfg_scale=1
        # (e o pipeline pula o passo negativo, metade do custo por step).
        with torch.inference_mode():
            output = self._pipe(
                image=[image],
                prompt=prompt,
                negative_prompt=' ',
                true_cfg_scale=1.0 if self.lightning else self._settings.GUIDANCE,
                guidance_scale=1.0,
                num_inference_steps=self._settings.NUM_STEPS,
                num_images_per_prompt=1,
            )
        return output.images[0]
