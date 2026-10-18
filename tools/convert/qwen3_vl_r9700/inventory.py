"""R9700 artifact inventory for the Qwen3-VL text encoder (T2I + v2v).

Piece 7a/7d: direct-BF16 1:1. The text stack (398) plus the vision ViT (351).
Quantization (FP8/Q4) is layered on later.
"""

from __future__ import annotations

from collections import Counter

from tools.convert.qwen3.common.inventory import (
    BF16,
    DIRECT_FORMATS,
    TensorSpec,
    tensor_spec,
)

from . import source_inventory

MODEL_ID = source_inventory.MODEL_ID
WEIGHTS_ID = "r9700-bf16-candidate"
TARGET_KEY = "qwen3_vl_r9700"
RECIPE_ID = "r9700-bf16-v0"


def _candidate_tensor(spec: TensorSpec) -> TensorSpec:
    numeric_format = spec.format if spec.format in DIRECT_FORMATS else BF16
    return tensor_spec(spec.name, spec.shape, numeric_format)


TENSOR_SPECS: tuple[TensorSpec, ...] = tuple(
    _candidate_tensor(spec) for spec in source_inventory.TENSOR_SPECS
)
OBJECT_SPECS = TENSOR_SPECS  # no frontend resources yet
FORMAT_COUNTS: dict[str, int] = dict(Counter(spec.format for spec in TENSOR_SPECS))


def validate_inventory() -> None:
    if len(TENSOR_SPECS) != 749:
        raise ValueError(f"text-encoder candidate inventory count is not 749, got {len(TENSOR_SPECS)}")
    if FORMAT_COUNTS != {BF16: 749}:
        raise ValueError(f"text-encoder candidate must be all-BF16, got {FORMAT_COUNTS}")


validate_inventory()
