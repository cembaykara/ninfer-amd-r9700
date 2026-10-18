"""Neutral logical source inventory for the Qwen3-VL text encoder (T2I + v2v).

Piece 7a/7d: a direct-BF16, 1:1 inventory of the 398-tensor text stack plus the
351-tensor vision (ViT) encoder. The lm_head is dropped. v2v needs the ViT to
encode the input image; T2I ignores it.
"""

from __future__ import annotations

from tools.convert.qwen3.common.inventory import (
    BF16,
    FORMAT_NAMES,
    TensorSpec,
    tensor_spec,
)

MODEL_ID = "qwen3-vl-text-encoder"

# --- text stack ---
NUM_LAYERS = 36
HIDDEN = 4096
HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # 1024
INTERMEDIATE = 12288
VOCAB = 151936

# --- vision (ViT) ---
VIT_DEPTH = 27
VIT_HIDDEN = 1152
VIT_INTERMEDIATE = 4304
VIT_NUM_HEADS = 16
VIT_HEAD_DIM = VIT_HIDDEN // VIT_NUM_HEADS  # 72
VIT_PATCH = 16
VIT_TEMPORAL_PATCH = 2
VIT_IN_CHANNELS = 3
VIT_POS_EMBED = 2304  # 48x48 grid
VIT_MERGE = 2  # spatial_merge_size
VIT_MERGED = VIT_HIDDEN * (VIT_MERGE ** 2)  # 4608
VIT_OUT = 4096  # out_hidden_size
DEEPSTACK_INDEXES = (8, 16, 24)

_tensor = tensor_spec


def _build_block_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for layer in range(NUM_LAYERS):
        prefix = f"encoder/layers/{layer}/"
        specs.extend(
            (
                _tensor(prefix + "attention/query", (HIDDEN, HIDDEN), BF16),
                _tensor(prefix + "attention/key", (KV_DIM, HIDDEN), BF16),
                _tensor(prefix + "attention/value", (KV_DIM, HIDDEN), BF16),
                _tensor(prefix + "attention/output", (HIDDEN, HIDDEN), BF16),
                _tensor(prefix + "attention/query_norm", (HEAD_DIM,), BF16),
                _tensor(prefix + "attention/key_norm", (HEAD_DIM,), BF16),
                _tensor(prefix + "mlp/gate", (INTERMEDIATE, HIDDEN), BF16),
                _tensor(prefix + "mlp/up", (INTERMEDIATE, HIDDEN), BF16),
                _tensor(prefix + "mlp/down", (HIDDEN, INTERMEDIATE), BF16),
                _tensor(prefix + "input_layernorm", (HIDDEN,), BF16),
                _tensor(prefix + "post_attention_layernorm", (HIDDEN,), BF16),
            )
        )
    return tuple(specs)


def _build_io_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("encoder/embed_tokens", (VOCAB, HIDDEN), BF16),
        _tensor("encoder/final_norm", (HIDDEN,), BF16),
    )


def _build_vision_block_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = []
    for layer in range(VIT_DEPTH):
        prefix = f"visual/blocks/{layer}/"
        specs.extend(
            (
                _tensor(prefix + "attn/qkv", (3 * VIT_HIDDEN, VIT_HIDDEN), BF16),
                _tensor(prefix + "attn/qkv_b", (3 * VIT_HIDDEN,), BF16),
                _tensor(prefix + "attn/proj", (VIT_HIDDEN, VIT_HIDDEN), BF16),
                _tensor(prefix + "attn/proj_b", (VIT_HIDDEN,), BF16),
                _tensor(prefix + "mlp/fc1", (VIT_INTERMEDIATE, VIT_HIDDEN), BF16),
                _tensor(prefix + "mlp/fc1_b", (VIT_INTERMEDIATE,), BF16),
                _tensor(prefix + "mlp/fc2", (VIT_HIDDEN, VIT_INTERMEDIATE), BF16),
                _tensor(prefix + "mlp/fc2_b", (VIT_HIDDEN,), BF16),
                _tensor(prefix + "norm1", (VIT_HIDDEN,), BF16),
                _tensor(prefix + "norm1_b", (VIT_HIDDEN,), BF16),
                _tensor(prefix + "norm2", (VIT_HIDDEN,), BF16),
                _tensor(prefix + "norm2_b", (VIT_HIDDEN,), BF16),
            )
        )
    return tuple(specs)


def _build_vision_io_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor("visual/patch_embed", (VIT_HIDDEN, VIT_IN_CHANNELS, VIT_TEMPORAL_PATCH, VIT_PATCH, VIT_PATCH), BF16),
        _tensor("visual/patch_embed_b", (VIT_HIDDEN,), BF16),
        _tensor("visual/pos_embed", (VIT_POS_EMBED, VIT_HIDDEN), BF16),
        # main merger (use_postshuffle_norm=False): norm on VIT_HIDDEN
        _tensor("visual/merger/norm", (VIT_HIDDEN,), BF16),
        _tensor("visual/merger/norm_b", (VIT_HIDDEN,), BF16),
        _tensor("visual/merger/fc1", (VIT_MERGED, VIT_MERGED), BF16),
        _tensor("visual/merger/fc1_b", (VIT_MERGED,), BF16),
        _tensor("visual/merger/fc2", (VIT_OUT, VIT_MERGED), BF16),
        _tensor("visual/merger/fc2_b", (VIT_OUT,), BF16),
    ]
    # deepstack mergers (use_postshuffle_norm=True): norm on VIT_MERGED
    for i in range(len(DEEPSTACK_INDEXES)):
        specs.extend(
            (
                _tensor(f"visual/deepstack/{i}/norm", (VIT_MERGED,), BF16),
                _tensor(f"visual/deepstack/{i}/norm_b", (VIT_MERGED,), BF16),
                _tensor(f"visual/deepstack/{i}/fc1", (VIT_MERGED, VIT_MERGED), BF16),
                _tensor(f"visual/deepstack/{i}/fc1_b", (VIT_MERGED,), BF16),
                _tensor(f"visual/deepstack/{i}/fc2", (VIT_OUT, VIT_MERGED), BF16),
                _tensor(f"visual/deepstack/{i}/fc2_b", (VIT_OUT,), BF16),
            )
        )
    return tuple(specs)


BLOCK_TENSOR_SPECS = _build_block_specs()
IO_TENSOR_SPECS = _build_io_specs()
VISION_BLOCK_SPECS = _build_vision_block_specs()
VISION_IO_SPECS = _build_vision_io_specs()
TENSOR_SPECS = IO_TENSOR_SPECS + BLOCK_TENSOR_SPECS + VISION_IO_SPECS + VISION_BLOCK_SPECS

FORMAT_COUNTS = {
    fmt: sum(spec.format == fmt for spec in TENSOR_SPECS) for fmt in FORMAT_NAMES
}

TOTAL = 749  # 398 text + 351 vision


def validate_inventory() -> None:
    if len(TENSOR_SPECS) != TOTAL:
        raise ValueError(f"text-encoder inventory count is not {TOTAL}, got {len(TENSOR_SPECS)}")
    if FORMAT_COUNTS.get(BF16) != TOTAL:
        raise ValueError(f"text-encoder inventory must be all-BF16, got {FORMAT_COUNTS}")


validate_inventory()
