import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools.artifact import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec, encode_q4_n16k16
from tools.parity.qwen3_8_27b.vision import (
    metrics,
    publish_json_create_only,
    source_provenance,
)
from tools.reference.qwen3_8_27b.bindings import PhysicalBlock
from tools.reference.qwen3_8_27b.weights import WeightStore


class VisionReportHelpersTest(unittest.TestCase):
    def test_streams_tiled_vision_weights_from_artifact(self) -> None:
        # Reproduces the production Vision reference's missing residency assignment.
        # Signed codes and represented scales define the exact mathematical output.
        shape = (16, 128)
        codes = ((torch.arange(16 * 128).reshape(16, 2, 64) * 7 + 3) % 16 - 8).to(torch.int8)
        scales = ((torch.arange(32).reshape(16, 2) % 7 + 1) / 8).half()
        expected = (codes.float() * scales.float().unsqueeze(-1)).reshape(shape).bfloat16()
        name = "vision/layers/0/attention/qkv"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vision.ninfer"
            with ArtifactWriter(
                path,
                ArtifactIdentity("qwen3.8-27b", "r9700-lq4"),
                (TensorSpec(name, shape, "Q4G64_F16S", "r9700-q4g64-n16-k16-v1"),),
            ) as writer:
                writer.write(name, (encode_q4_n16k16(codes, scales, shape),))
            with Artifact.open(path) as artifact:
                block = PhysicalBlock(0, artifact.objects[0], "vision")
                binding = mock.Mock()
                binding.blocks_for.return_value = (block,)
                binding.payload.side_effect = lambda value: artifact.payload(value.descriptor.name)
                store = WeightStore(binding, "cpu", capacity=1, text=False, vision=True)
                try:
                    self.assertTrue(
                        torch.equal(
                            store.tensor(block).view(torch.uint16), expected.view(torch.uint16)
                        )
                    )
                finally:
                    store.close()

    def test_source_provenance_binds_exact_index_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "a.safetensors").write_bytes(b"a")
            (root / "b.safetensors").write_bytes(b"b")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "b.safetensors", "y": "a.safetensors"}}),
                encoding="utf-8",
            )
            with mock.patch(
                "tools.parity.qwen3_8_27b.vision.validate_checkpoint_files",
                return_value={"x": "b.safetensors", "y": "a.safetensors"},
            ):
                result = source_provenance(root)
            self.assertEqual(list(result["shards"]), ["a.safetensors", "b.safetensors"])
            self.assertEqual(result["shards"]["a.safetensors"], {"bytes": 1})

    def test_source_provenance_rejects_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "missing.safetensors"}}), encoding="utf-8"
            )
            with (
                mock.patch(
                    "tools.parity.qwen3_8_27b.vision.validate_checkpoint_files",
                    side_effect=ValueError("missing shards"),
                ),
                self.assertRaisesRegex(ValueError, "missing shards"),
            ):
                source_provenance(root)

    def test_metrics_reject_shape_and_nonfinite(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            metrics(torch.zeros(2), torch.zeros(3))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            metrics(torch.tensor([float("nan")]), torch.zeros(1))

    def test_metrics_returns_finite_exact_result(self) -> None:
        result = metrics(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
        self.assertEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["cosine"], 1.0, places=6)

    def test_report_publication_rejects_dangling_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            output.symlink_to(root / "missing")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                publish_json_create_only(output, {"status": "complete"})
            self.assertTrue(output.is_symlink())

    def test_report_publication_is_create_only_and_durable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            value = {"status": "complete"}
            publish_json_create_only(output, value)
            self.assertEqual(json.loads(output.read_text()), value)
            inode = os.stat(output, follow_symlinks=False).st_ino
            with self.assertRaisesRegex(ValueError, "overwrite"):
                publish_json_create_only(output, {"status": "replacement"})
            self.assertEqual(os.stat(output, follow_symlinks=False).st_ino, inode)


if __name__ == "__main__":
    unittest.main()
