"""Provisional R9700 artifact inventory for Qwen-Image-2.1-Turbo (DiT).

Piece 1: direct-BF16 1:1. The quantization candidate (FP8/Q4) is added in Piece 2.
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
TARGET_KEY = "qwen_image_21_r9700"
RECIPE_ID = "r9700-bf16-v0"


def _candidate_tensor(spec: TensorSpec) -> TensorSpec:
    numeric_format = spec.format if spec.format in DIRECT_FORMATS else BF16
    return tensor_spec(spec.name, spec.shape, numeric_format)


TENSOR_SPECS: tuple[TensorSpec, ...] = tuple(
    _candidate_tensor(spec) for spec in source_inventory.TENSOR_SPECS
)
OBJECT_SPECS = TENSOR_SPECS  # Piece 2: no frontend resources yet
FORMAT_COUNTS: dict[str, int] = dict(Counter(spec.format for spec in TENSOR_SPECS))


def validate_inventory() -> None:
    if len(TENSOR_SPECS) != 297:
        raise ValueError(f"DiT candidate inventory count is not 297, got {len(TENSOR_SPECS)}")
    if FORMAT_COUNTS != {BF16: 297}:
        raise ValueError(f"Piece-1 DiT candidate must be all-BF16, got {FORMAT_COUNTS}")


validate_inventory()
