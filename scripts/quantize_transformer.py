"""Converte um transformer fundido (saída de scripts/fuse_transformer.py, bf16 ~39 GB) para
fp8 weight-only do torchao (~20 GB) e salva -- o servidor sobe lendo metade dos bytes e sem
o passo de quantizar.

O torchao serializa com pickle (safe_serialization=False): só carregue pastas geradas por
você mesmo.

Uso (no pod, na raiz do repo; roda na CPU, não precisa da GPU livre):
    PYTHONPATH=. /root/venv/bin/python scripts/quantize_transformer.py \
        /workspace/models/fused/<nome> /workspace/models/fused/<nome>_fp8 [--verify]
"""

import argparse
import json
import time
from pathlib import Path

import torch

from server.qwen_pipeline import FUSED_INFO_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--verify', action='store_true', help='Recarrega a saída e compara com o original')
    args = parser.parse_args()

    from diffusers import QwenImageTransformer2DModel, TorchAoConfig
    from torchao.quantization import Float8WeightOnlyConfig

    t = time.monotonic()
    model = QwenImageTransformer2DModel.from_pretrained(
        args.input_dir,
        quantization_config=TorchAoConfig(Float8WeightOnlyConfig()),
        torch_dtype=torch.bfloat16,
    )
    print(f'Carga + quantização: {time.monotonic() - t:.0f}s', flush=True)

    t = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=False, max_shard_size='5GB')
    info = json.loads((args.input_dir / FUSED_INFO_FILE).read_text(encoding='utf-8'))
    info['quantization'] = 'fp8_weight_only'
    (args.output_dir / FUSED_INFO_FILE).write_text(json.dumps(info, indent=2), encoding='utf-8')
    size = sum(f.stat().st_size for f in args.output_dir.iterdir()) / 1024**3
    print(f'Salvo em {args.output_dir}: {size:.1f} GB ({time.monotonic() - t:.0f}s)', flush=True)

    if args.verify:
        t = time.monotonic()
        loaded = QwenImageTransformer2DModel.from_pretrained(
            args.output_dir, torch_dtype=torch.bfloat16, use_safetensors=False
        )
        print(f'Recarga: {time.monotonic() - t:.0f}s', flush=True)
        a = dict(model.named_parameters())
        b = dict(loaded.named_parameters())
        assert a.keys() == b.keys(), 'parâmetros diferentes'
        name = next(n for n in a if n.endswith('.weight') and a[n].dim() == 2)
        same = torch.equal(a[name].dequantize(), b[name].dequantize())
        print(f'Tipo: {type(b[name]).__name__}; {name} idêntico: {same}', flush=True)


if __name__ == '__main__':
    main()
