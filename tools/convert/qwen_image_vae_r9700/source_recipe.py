"""Hugging Face source recipe for the Qwen-Image-2.1 VAE decoder.

Piece 7e: a direct 1:1 BF16 mapping from the VAE decoder tensors to the logical inventory.
The source is a single diffusion_pytorch_model.safetensors.
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

WEIGHTS_FILENAME = "diffusion_pytorch_model.safetensors"


def _resnet_recipes(prefix_src: str, prefix_obj: str, cin: int, cout: int, shortcut: bool) -> tuple[TensorRecipe, ...]:
    recipes = (
        TensorRecipe(prefix_obj + "conv1", source(prefix_src + "conv1.weight", (cout, cin, 3, 3))),
        TensorRecipe(prefix_obj + "conv1_b", source(prefix_src + "conv1.bias", (cout,))),
        TensorRecipe(prefix_obj + "conv2", source(prefix_src + "conv2.weight", (cout, cout, 3, 3))),
        TensorRecipe(prefix_obj + "conv2_b", source(prefix_src + "conv2.bias", (cout,))),
        TensorRecipe(prefix_obj + "norm1", source(prefix_src + "norm1.gamma", (cin, 1, 1, 1))),
        TensorRecipe(prefix_obj + "norm2", source(prefix_src + "norm2.gamma", (cout, 1, 1, 1))),
    )
    if shortcut:
        recipes = recipes + (
            TensorRecipe(prefix_obj + "shortcut", source(prefix_src + "conv_shortcut.weight", (cout, cin, 1, 1))),
            TensorRecipe(prefix_obj + "shortcut_b", source(prefix_src + "conv_shortcut.bias", (cout,))),
        )
    return recipes


def _build_up_block_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = []
    for i in range(len(inventory.UP_OUT)):
        cin, cout = inventory.UP_IN[i], inventory.UP_OUT[i]
        src_base = f"decoder.up_blocks.{i}."
        obj_base = f"vae/up_blocks/{i}/"
        for j in range(inventory.NUM_RESNETS):
            recipes.extend(_resnet_recipes(
                src_base + f"resnets.{j}.", obj_base + f"resnets/{j}/",
                cin if j == 0 else cout, cout, shortcut=(j == 0 and cin != cout)))
        if i < 4:
            recipes.extend(
                (
                    TensorRecipe(obj_base + "upsampler/resample", source(src_base + "upsampler.resample.1.weight", (cout, cout, 3, 3))),
                    TensorRecipe(obj_base + "upsampler/resample_b", source(src_base + "upsampler.resample.1.bias", (cout,))),
                )
            )
        if i < 3:
            recipes.extend(
                (
                    TensorRecipe(obj_base + "upsampler/time_conv", source(src_base + "upsampler.time_conv.weight", (2 * cout, cout, 1, 1))),
                    TensorRecipe(obj_base + "upsampler/time_conv_b", source(src_base + "upsampler.time_conv.bias", (2 * cout,))),
                )
            )
    return tuple(recipes)


def _build_mid_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = []
    for j in range(2):
        recipes.extend(_resnet_recipes(f"decoder.mid_block.resnets.{j}.", f"vae/mid_block/resnets/{j}/",
                                       inventory.MID_DIM, inventory.MID_DIM, shortcut=False))
    recipes.extend(
        (
            TensorRecipe("vae/mid_block/attn/to_qkv", source("decoder.mid_block.attentions.0.to_qkv.weight", (3 * inventory.MID_DIM, inventory.MID_DIM, 1, 1))),
            TensorRecipe("vae/mid_block/attn/to_qkv_b", source("decoder.mid_block.attentions.0.to_qkv.bias", (3 * inventory.MID_DIM,))),
            TensorRecipe("vae/mid_block/attn/proj", source("decoder.mid_block.attentions.0.proj.weight", (inventory.MID_DIM, inventory.MID_DIM, 1, 1))),
            TensorRecipe("vae/mid_block/attn/proj_b", source("decoder.mid_block.attentions.0.proj.bias", (inventory.MID_DIM,))),
            TensorRecipe("vae/mid_block/attn/norm", source("decoder.mid_block.attentions.0.norm.gamma", (inventory.MID_DIM, 1, 1))),
        )
    )
    return tuple(recipes)


def _build_io_recipes() -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe("vae/conv_in", source("decoder.conv_in.weight", (1152, 64, 3, 3))),
        TensorRecipe("vae/conv_in_b", source("decoder.conv_in.bias", (1152,))),
        TensorRecipe("vae/conv_out", source("decoder.conv_out.weight", (4, 144, 3, 3))),
        TensorRecipe("vae/conv_out_b", source("decoder.conv_out.bias", (4,))),
        TensorRecipe("vae/norm_out", source("decoder.norm_out.gamma", (144, 1, 1, 1))),
        TensorRecipe("vae/post_quant", source("post_quant_conv.weight", (64, 64, 1, 1))),
        TensorRecipe("vae/post_quant_b", source("post_quant_conv.bias", (64,))),
    )


RECIPE_SPECS = _build_io_recipes() + _build_up_block_recipes() + _build_mid_recipes()
RECIPES_BY_NAME = {r.object_name: r for r in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)


def source_requirements() -> dict[str, SourceTensor]:
    return _recipe_source_requirements(RECIPE_SPECS)


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    with ShardReader.from_file(Path(model_dir) / WEIGHTS_FILENAME) as reader:
        return preflight_source_reader(reader, RECIPE_SPECS)


validate_recipe_coverage()
