"""Neutral logical source inventory for the Qwen-Image-2.1-Turbo DiT (transformer).

Piece 1: a direct-BF16, 1:1 inventory of the 297-tensor DiT. Quantization and
packing (gate/up concat, QKV fusion) are added in Piece 2.
"""

from __future__ import annotations

from tools.convert.qwen3.common.inventory import (
    BF16,
    FORMAT_NAMES,
    TensorSpec,
    tensor_spec,
)

MODEL_ID = "qwen-image-2.1-turbo"

NUM_LAYERS = 32
HIDDEN = 4096
HEAD_DIM = 128
INTERMEDIATE = 12288
LATENT_CHANNELS = 64

_tensor = tensor_spec


def _build_block_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for layer in range(NUM_LAYERS):
        prefix = f"dit/layers/{layer}/"
        specs.extend(
            (
                _tensor(prefix + "attention/query", (HIDDEN, HIDDEN), BF16),
                _tensor(prefix + "attention/key", (HIDDEN, HIDDEN), BF16),
                _tensor(prefix + "attention/value", (HIDDEN, HIDDEN), BF16),
                _tensor(prefix + "attention/output", (HIDDEN, HIDDEN), BF16),
                _tensor(prefix + "attention/query_norm", (HEAD_DIM,), BF16),
                _tensor(prefix + "attention/key_norm", (HEAD_DIM,), BF16),
                _tensor(prefix + "mlp/gate", (INTERMEDIATE, HIDDEN), BF16),
                _tensor(prefix + "mlp/up", (INTERMEDIATE, HIDDEN), BF16),
                _tensor(prefix + "mlp/down", (HIDDEN, INTERMEDIATE), BF16),
            )
        )
    return tuple(specs)


def _build_io_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("dit/img_in", (HIDDEN, LATENT_CHANNELS), BF16),
        _tensor("dit/txt_in/in_layer", (HIDDEN, HIDDEN), BF16),
        _tensor("dit/txt_in/out_layer", (HIDDEN, HIDDEN), BF16),
        _tensor("dit/txt_in/text_norm", (HIDDEN,), BF16),
        _tensor("dit/time_embed/linear_1", (HIDDEN, 256), BF16),
        _tensor("dit/time_embed/linear_2", (HIDDEN, HIDDEN), BF16),
        _tensor("dit/modulation", (4 * HIDDEN, HIDDEN), BF16),
        _tensor("dit/norm_out", (HIDDEN, HIDDEN), BF16),
        _tensor("dit/proj_out", (LATENT_CHANNELS, HIDDEN), BF16),
    )


BLOCK_TENSOR_SPECS = _build_block_specs()
IO_TENSOR_SPECS = _build_io_specs()
TENSOR_SPECS = IO_TENSOR_SPECS + BLOCK_TENSOR_SPECS

FORMAT_COUNTS = {
    fmt: sum(spec.format == fmt for spec in TENSOR_SPECS) for fmt in FORMAT_NAMES
}


def validate_inventory() -> None:
    if len(TENSOR_SPECS) != 297:
        raise ValueError(f"DiT inventory count is not 297, got {len(TENSOR_SPECS)}")
    if FORMAT_COUNTS.get(BF16) != 297:
        raise ValueError(f"Piece-1 DiT inventory must be all-BF16, got {FORMAT_COUNTS}")


validate_inventory()
