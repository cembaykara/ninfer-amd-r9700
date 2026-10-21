"""Neutral logical source inventory for the Qwen-Image-2.1 VAE decoder.

Piece 7e: a direct-BF16, 1:1 inventory of the VAE decoder (the part that decodes the
DiT latent -> image). The encoder is dropped (we only decode). Conv weights are kept
4D [C_out, C_in, kH, kW]; norms are 1D gamma.
"""

from __future__ import annotations

from tools.convert.qwen3.common.inventory import (
    BF16,
    FORMAT_NAMES,
    TensorSpec,
    tensor_spec,
)

MODEL_ID = "qwen-image-21-vae-decoder"

# decoder channel dims (output per up_block) and input channels
UP_OUT = [1152, 1152, 576, 288, 144]
UP_IN = [1152, 1152, 1152, 576, 288]
NUM_RESNETS = 3
MID_DIM = 1152
LATENT = 64
IMAGE_CH = 4

_tensor = tensor_spec


def _resnet_specs(prefix: str, cin: int, cout: int, shortcut: bool) -> tuple[TensorSpec, ...]:
    specs = (
        _tensor(prefix + "conv1", (cout, cin, 3, 3), BF16),
        _tensor(prefix + "conv1_b", (cout,), BF16),
        _tensor(prefix + "conv2", (cout, cout, 3, 3), BF16),
        _tensor(prefix + "conv2_b", (cout,), BF16),
        _tensor(prefix + "norm1", (cin, 1, 1, 1), BF16),
        _tensor(prefix + "norm2", (cout, 1, 1, 1), BF16),
    )
    if shortcut:
        specs = specs + (
            _tensor(prefix + "shortcut", (cout, cin, 1, 1), BF16),
            _tensor(prefix + "shortcut_b", (cout,), BF16),
        )
    return specs


def _build_up_block_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for i in range(len(UP_OUT)):
        cin, cout = UP_IN[i], UP_OUT[i]
        base = f"vae/up_blocks/{i}/"
        for j in range(NUM_RESNETS):
            specs.extend(_resnet_specs(base + f"resnets/{j}/", cin if j == 0 else cout, cout,
                                       shortcut=(j == 0 and cin != cout)))
        # upsampler: resample on blocks 0-3, time_conv only on blocks 0-2, none on block 4
        if i < 4:
            specs.extend(
                (
                    _tensor(base + "upsampler/resample", (cout, cout, 3, 3), BF16),
                    _tensor(base + "upsampler/resample_b", (cout,), BF16),
                )
            )
        if i < 3:
            specs.extend(
                (
                    _tensor(base + "upsampler/time_conv", (2 * cout, cout, 1, 1), BF16),
                    _tensor(base + "upsampler/time_conv_b", (2 * cout,), BF16),
                )
            )
    return tuple(specs)


def _build_mid_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for j in range(2):
        specs.extend(_resnet_specs(f"vae/mid_block/resnets/{j}/", MID_DIM, MID_DIM, shortcut=False))
    specs.extend(
        (
            _tensor("vae/mid_block/attn/to_qkv", (3 * MID_DIM, MID_DIM, 1, 1), BF16),
            _tensor("vae/mid_block/attn/to_qkv_b", (3 * MID_DIM,), BF16),
            _tensor("vae/mid_block/attn/proj", (MID_DIM, MID_DIM, 1, 1), BF16),
            _tensor("vae/mid_block/attn/proj_b", (MID_DIM,), BF16),
            _tensor("vae/mid_block/attn/norm", (MID_DIM, 1, 1), BF16),
        )
    )
    return tuple(specs)


def _build_io_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("vae/conv_in", (UP_OUT[0], LATENT, 3, 3), BF16),
        _tensor("vae/conv_in_b", (UP_OUT[0],), BF16),
        _tensor("vae/conv_out", (IMAGE_CH, UP_OUT[-1], 3, 3), BF16),
        _tensor("vae/conv_out_b", (IMAGE_CH,), BF16),
        _tensor("vae/norm_out", (UP_OUT[-1], 1, 1, 1), BF16),
        _tensor("vae/post_quant", (LATENT, LATENT, 1, 1), BF16),
        _tensor("vae/post_quant_b", (LATENT,), BF16),
    )


IO_SPECS = _build_io_specs()
UP_SPECS = _build_up_block_specs()
MID_SPECS = _build_mid_specs()
TENSOR_SPECS = IO_SPECS + UP_SPECS + MID_SPECS

FORMAT_COUNTS = {fmt: sum(s.format == fmt for s in TENSOR_SPECS) for fmt in FORMAT_NAMES}


def validate_inventory() -> None:
    if len(TENSOR_SPECS) != 134:
        raise ValueError(f"VAE decoder inventory count is not 134, got {len(TENSOR_SPECS)}")
    if FORMAT_COUNTS.get(BF16) != 134:
        raise ValueError(f"VAE decoder inventory must be all-BF16, got {FORMAT_COUNTS}")


validate_inventory()
