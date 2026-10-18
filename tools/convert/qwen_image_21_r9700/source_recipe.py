"""Hugging Face source recipe for the Qwen-Image-2.1-Turbo DiT (transformer).

Piece 1: a direct 1:1 BF16 mapping from the diffusers transformer checkpoint to
the logical DiT inventory. No packing/quantization yet.
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

INDEX_FILENAME = "diffusion_pytorch_model.safetensors.index.json"


def _build_block_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = []
    for layer in range(inventory.NUM_LAYERS):
        src = f"transformer_blocks.{layer}."
        obj = f"dit/layers/{layer}/"
        recipes.extend(
            (
                TensorRecipe(obj + "attention/query", source(src + "attn.to_q.weight", (4096, 4096))),
                TensorRecipe(obj + "attention/key", source(src + "attn.to_k.weight", (4096, 4096))),
                TensorRecipe(obj + "attention/value", source(src + "attn.to_v.weight", (4096, 4096))),
                TensorRecipe(obj + "attention/output", source(src + "attn.to_out.0.weight", (4096, 4096))),
                TensorRecipe(obj + "attention/query_norm", source(src + "attn.norm_q.weight", (128,))),
                TensorRecipe(obj + "attention/key_norm", source(src + "attn.norm_k.weight", (128,))),
                TensorRecipe(obj + "mlp/gate", source(src + "img_mlp.gate_layer.weight", (12288, 4096))),
                TensorRecipe(obj + "mlp/up", source(src + "img_mlp.proj.weight", (12288, 4096))),
                TensorRecipe(obj + "mlp/down", source(src + "img_mlp.out.weight", (4096, 12288))),
            )
        )
    return tuple(recipes)


def _build_io_recipes() -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe("dit/img_in", source("img_in.weight", (4096, 64))),
        TensorRecipe("dit/txt_in/in_layer", source("txt_in.in_layer.weight", (4096, 4096))),
        TensorRecipe("dit/txt_in/out_layer", source("txt_in.out_layer.weight", (4096, 4096))),
        TensorRecipe("dit/txt_in/text_norm", source("txt_in.text_norm.weight", (4096,))),
        TensorRecipe(
            "dit/time_embed/linear_1",
            source("time_text_embed.timestep_embedder.linear_1.weight", (4096, 256)),
        ),
        TensorRecipe(
            "dit/time_embed/linear_2",
            source("time_text_embed.timestep_embedder.linear_2.weight", (4096, 4096)),
        ),
        TensorRecipe("dit/modulation", source("modulation.1.weight", (16384, 4096))),
        TensorRecipe("dit/norm_out", source("norm_out.linear.weight", (4096, 4096))),
        TensorRecipe("dit/proj_out", source("proj_out.weight", (64, 4096))),
    )


RECIPE_SPECS = _build_io_recipes() + _build_block_recipes()
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)


def source_requirements() -> dict[str, SourceTensor]:
    return _recipe_source_requirements(RECIPE_SPECS)


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    with ShardReader(model_dir, INDEX_FILENAME) as reader:
        return preflight_source_reader(reader, RECIPE_SPECS)


validate_recipe_coverage()
