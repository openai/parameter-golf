from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from spectral_flood_walk_v3 import DeepFloorModel, V3Config, resolve_device, train_and_evaluate


class SpectralFloodWalkV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = resolve_device("cpu")

    def _write_fixture(self) -> Path:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = Path(tmpdir.name) / "enwik8"
        data = np.arange(16384, dtype=np.uint8)
        path.write_bytes(data.tobytes())
        return path

    def _base_config(self, *, cross_token_mode: str) -> V3Config:
        return V3Config(
            enwik8_path=str(self._write_fixture()),
            device="cpu",
            seed=123,
            seq_len=16,
            stride=8,
            batch_size=2,
            train_steps=2,
            eval_batches=2,
            report_every=1,
            recurrent_dim=32,
            recurrent_heads=4,
            num_distinct_blocks=2,
            view_count=2,
            view_combination="average",
            cross_token_mode=cross_token_mode,
            train_recurrence_steps=4,
            eval_recurrence_steps=6,
            train_floor_interval=2,
            floor_min_interval=1,
            floor_max_interval=3,
            floor_threshold=0.01,
            cache_dataset_on_device=False,
        )

    def test_floor_forward_emits_logits_and_floor_metadata(self) -> None:
        cfg = self._base_config(cross_token_mode="floor")
        model = DeepFloorModel(cfg).to(self.device)
        inputs = torch.randint(0, 256, (2, cfg.seq_len), device=self.device)

        logits, metadata = model(inputs, adaptive_floor=True, recurrence_steps=5, return_metadata=True)

        self.assertEqual(tuple(logits.shape), (2, cfg.seq_len, cfg.vocab_size))
        self.assertEqual(len(metadata["floor_steps"]), cfg.view_count)
        self.assertTrue(all(isinstance(steps, list) for steps in metadata["floor_steps"]))

    def test_fused_forward_emits_logits_without_floor_events(self) -> None:
        cfg = self._base_config(cross_token_mode="fused")
        model = DeepFloorModel(cfg).to(self.device)
        inputs = torch.randint(0, 256, (2, cfg.seq_len), device=self.device)

        logits, metadata = model(inputs, adaptive_floor=True, recurrence_steps=5, return_metadata=True)

        self.assertEqual(tuple(logits.shape), (2, cfg.seq_len, cfg.vocab_size))
        self.assertEqual(metadata["floor_counts"], [0, 0])

    def test_artifact_estimate_grows_with_views(self) -> None:
        low_cfg = self._base_config(cross_token_mode="floor")
        high_cfg = self._base_config(cross_token_mode="floor")
        high_cfg.view_count = 4

        low_model = DeepFloorModel(low_cfg)
        high_model = DeepFloorModel(high_cfg)

        self.assertGreater(high_model.estimate_artifact_bytes(), low_model.estimate_artifact_bytes())

    def test_floor_training_smoke(self) -> None:
        cfg = self._base_config(cross_token_mode="floor")

        result = train_and_evaluate(cfg)

        self.assertEqual(result["config"]["cross_token_mode"], "floor")
        self.assertGreater(len(result["train"]["history"]), 0)
        self.assertIn("bpb", result["val"])
        self.assertLess(result["artifact"]["estimated_mb"], 16.0)

    def test_fused_training_smoke(self) -> None:
        cfg = self._base_config(cross_token_mode="fused")
        cfg.kernel_feature_map = "identity"

        result = train_and_evaluate(cfg)

        self.assertEqual(result["config"]["cross_token_mode"], "fused")
        self.assertGreater(len(result["train"]["history"]), 0)
        self.assertIn("bpb", result["test"])
        self.assertLess(result["artifact"]["estimated_mb"], 16.0)


if __name__ == "__main__":
    unittest.main()
