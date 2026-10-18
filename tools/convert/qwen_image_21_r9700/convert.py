"""Build the Qwen-Image-2.1-Turbo DiT .ninfer artifact (Piece 2: BF16 direct).

Reads the diffusers transformer checkpoint through the 1:1 source recipe and
writes a BF16 .ninfer artifact. FP8/Q4 quantization is layered on in Piece 2b.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3.common import conversion as family_conversion
from tools.convert.qwen3.common.recipe import materialize_recipe

from . import inventory, source_recipe


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device,
) -> Path:
    """Convert the diffusers DiT checkpoint into a BF16 .ninfer artifact."""
    output = Path(out_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    started = time.perf_counter()
    resolved_device = pick_device(device)

    inventory.validate_inventory()
    source_preflight = source_recipe.preflight_sources(model_dir)
    object_plan = family_conversion.build_object_plan(inventory.OBJECT_SPECS, {})
    identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)

    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        ShardReader(model_dir, source_recipe.INDEX_FILENAME) as reader,
        ArtifactWriter(output, identity, object_plan.specs) as writer,
    ):
        if writer.objects != object_plan.objects:
            raise RuntimeError("writer object plan differs from preflight")
        for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
            tensor = materialize_recipe(source_recipe.RECIPES_BY_NAME[spec.name], reader)
            if tuple(tensor.shape) != spec.shape:
                raise ValueError(f"{spec.name}: materialized {tuple(tensor.shape)} != {spec.shape}")
            payload = family_conversion.encode_tensor_payload(tensor, spec, resolved_device)
            writer.write(spec.name, payload)
            del tensor, payload
            print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)

    stats = family_conversion.object_statistics(object_plan.objects)
    print(
        f"done in {time.perf_counter() - started:.1f}s: "
        f"{stats['count']} objects, {stats['object_bytes'] / 1e9:.2f} GB",
        flush=True,
    )
    return output
