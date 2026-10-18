"""Hugging Face source recipe for the Qwen3-VL text encoder (T2I + v2v).

Piece 7a/7d: a direct 1:1 BF16 mapping from the Qwen3-VL text stack and the
vision (ViT) encoder to the logical inventory. The source is a single
model.safetensors (not an indexed multi-shard).
"""

from __future__ import annotations

from pathlib import Path

from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3.common.recipe import (
    SourcePreflight,
    SourceTensor,
    TensorRecipe,
    preflight_source_reader,
    source,
    source_requirements as _recipe_source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import source_inventory as inventory

WEIGHTS_FILENAME = "model.safetensors"


def _build_block_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = []
    for layer in range(inventory.NUM_LAYERS):
        src = f"model.language_model.layers.{layer}."
        obj = f"encoder/layers/{layer}/"
        recipes.extend(
            (
                TensorRecipe(obj + "attention/query", source(src + "self_attn.q_proj.weight", (4096, 4096))),
                TensorRecipe(obj + "attention/key", source(src + "self_attn.k_proj.weight", (1024, 4096))),
                TensorRecipe(obj + "attention/value", source(src + "self_attn.v_proj.weight", (1024, 4096))),
                TensorRecipe(obj + "attention/output", source(src + "self_attn.o_proj.weight", (4096, 4096))),
                TensorRecipe(obj + "attention/query_norm", source(src + "self_attn.q_norm.weight", (128,))),
                TensorRecipe(obj + "attention/key_norm", source(src + "self_attn.k_norm.weight", (128,))),
                TensorRecipe(obj + "mlp/gate", source(src + "mlp.gate_proj.weight", (12288, 4096))),
                TensorRecipe(obj + "mlp/up", source(src + "mlp.up_proj.weight", (12288, 4096))),
                TensorRecipe(obj + "mlp/down", source(src + "mlp.down_proj.weight", (4096, 12288))),
                TensorRecipe(obj + "input_layernorm", source(src + "input_layernorm.weight", (4096,))),
                TensorRecipe(obj + "post_attention_layernorm", source(src + "post_attention_layernorm.weight", (4096,))),
            )
        )
    return tuple(recipes)


def _build_io_recipes() -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe("encoder/embed_tokens", source("model.language_model.embed_tokens.weight", (151936, 4096))),
        TensorRecipe("encoder/final_norm", source("model.language_model.norm.weight", (4096,))),
    )


def _build_vision_block_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = []
    for layer in range(inventory.VIT_DEPTH):
        src = f"model.visual.blocks.{layer}."
        obj = f"visual/blocks/{layer}/"
        recipes.extend(
            (
                TensorRecipe(obj + "attn/qkv", source(src + "attn.qkv.weight", (3456, 1152))),
                TensorRecipe(obj + "attn/qkv_b", source(src + "attn.qkv.bias", (3456,))),
                TensorRecipe(obj + "attn/proj", source(src + "attn.proj.weight", (1152, 1152))),
                TensorRecipe(obj + "attn/proj_b", source(src + "attn.proj.bias", (1152,))),
                TensorRecipe(obj + "mlp/fc1", source(src + "mlp.linear_fc1.weight", (4304, 1152))),
                TensorRecipe(obj + "mlp/fc1_b", source(src + "mlp.linear_fc1.bias", (4304,))),
                TensorRecipe(obj + "mlp/fc2", source(src + "mlp.linear_fc2.weight", (1152, 4304))),
                TensorRecipe(obj + "mlp/fc2_b", source(src + "mlp.linear_fc2.bias", (1152,))),
                TensorRecipe(obj + "norm1", source(src + "norm1.weight", (1152,))),
                TensorRecipe(obj + "norm1_b", source(src + "norm1.bias", (1152,))),
                TensorRecipe(obj + "norm2", source(src + "norm2.weight", (1152,))),
                TensorRecipe(obj + "norm2_b", source(src + "norm2.bias", (1152,))),
            )
        )
    return tuple(recipes)


def _build_vision_io_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe("visual/patch_embed", source("model.visual.patch_embed.proj.weight", (1152, 3, 2, 16, 16))),
        TensorRecipe("visual/patch_embed_b", source("model.visual.patch_embed.proj.bias", (1152,))),
        TensorRecipe("visual/pos_embed", source("model.visual.pos_embed.weight", (2304, 1152))),
        TensorRecipe("visual/merger/norm", source("model.visual.merger.norm.weight", (1152,))),
        TensorRecipe("visual/merger/norm_b", source("model.visual.merger.norm.bias", (1152,))),
        TensorRecipe("visual/merger/fc1", source("model.visual.merger.linear_fc1.weight", (4608, 4608))),
        TensorRecipe("visual/merger/fc1_b", source("model.visual.merger.linear_fc1.bias", (4608,))),
        TensorRecipe("visual/merger/fc2", source("model.visual.merger.linear_fc2.weight", (4096, 4608))),
        TensorRecipe("visual/merger/fc2_b", source("model.visual.merger.linear_fc2.bias", (4096,))),
    ]
    for i in range(len(inventory.DEEPSTACK_INDEXES)):
        src = f"model.visual.deepstack_merger_list.{i}."
        obj = f"visual/deepstack/{i}/"
        recipes.extend(
            (
                TensorRecipe(obj + "norm", source(src + "norm.weight", (4608,))),
                TensorRecipe(obj + "norm_b", source(src + "norm.bias", (4608,))),
                TensorRecipe(obj + "fc1", source(src + "linear_fc1.weight", (4608, 4608))),
                TensorRecipe(obj + "fc1_b", source(src + "linear_fc1.bias", (4608,))),
                TensorRecipe(obj + "fc2", source(src + "linear_fc2.weight", (4096, 4608))),
                TensorRecipe(obj + "fc2_b", source(src + "linear_fc2.bias", (4096,))),
            )
        )
    return tuple(recipes)


RECIPE_SPECS = (
    _build_io_recipes()
    + _build_block_recipes()
    + _build_vision_io_recipes()
    + _build_vision_block_recipes()
)
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)


def source_requirements() -> dict[str, SourceTensor]:
    return _recipe_source_requirements(RECIPE_SPECS)


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    with ShardReader.from_file(Path(model_dir) / WEIGHTS_FILENAME) as reader:
        return preflight_source_reader(reader, RECIPE_SPECS)


validate_recipe_coverage()
