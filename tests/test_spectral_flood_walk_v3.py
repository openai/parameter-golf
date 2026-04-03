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
        cfg.norm_interval_k = 2
        model = DeepFloorModel(cfg).to(self.device)
        inputs = torch.randint(0, 256, (2, cfg.seq_len), device=self.device)

        logits, metadata = model(inputs, adaptive_floor=True, recurrence_steps=5, return_metadata=True)

        self.assertEqual(tuple(logits.shape), (2, cfg.seq_len, cfg.vocab_size))
        self.assertEqual(len(metadata["floor_steps"]), cfg.view_count)
        self.assertTrue(all(isinstance(steps, list) for steps in metadata["floor_steps"]))
        self.assertEqual(metadata["state_norm_counts"], [2, 2])

    def test_fused_forward_emits_logits_without_floor_events(self) -> None:
        cfg = self._base_config(cross_token_mode="fused")
        cfg.norm_interval_k = 2
        model = DeepFloorModel(cfg).to(self.device)
        inputs = torch.randint(0, 256, (2, cfg.seq_len), device=self.device)

        logits, metadata = model(inputs, adaptive_floor=True, recurrence_steps=5, return_metadata=True)

        self.assertEqual(tuple(logits.shape), (2, cfg.seq_len, cfg.vocab_size))
        self.assertEqual(metadata["floor_counts"], [0, 0])
        self.assertEqual(metadata["state_norm_counts"], [2, 2])
        self.assertEqual(metadata["accumulator_norm_counts"], [2, 2])

    def test_project_view_combination_emits_logits(self) -> None:
        cfg = self._base_config(cross_token_mode="floor")
        cfg.view_combination = "project"
        model = DeepFloorModel(cfg).to(self.device)
        inputs = torch.randint(0, 256, (2, cfg.seq_len), device=self.device)

        logits = model(inputs, adaptive_floor=False, recurrence_steps=4)

        self.assertEqual(tuple(logits.shape), (2, cfg.seq_len, cfg.vocab_size))

    def test_artifact_estimate_grows_with_views(self) -> None:
        low_cfg = self._base_config(cross_token_mode="floor")
        high_cfg = self._base_config(cross_token_mode="floor")
        high_cfg.view_count = 4

        low_model = DeepFloorModel(low_cfg)
        high_model = DeepFloorModel(high_cfg)

        self.assertGreater(high_model.estimate_artifact_bytes(), low_model.estimate_artifact_bytes())

    def test_floor_training_smoke(self) -> None:
        cfg = self._base_config(cross_token_mode="floor")
        cfg.jacobian_lambda = 0.02

        result = train_and_evaluate(cfg)

        self.assertEqual(result["config"]["cross_token_mode"], "floor")
        self.assertGreater(len(result["train"]["history"]), 0)
        self.assertIn("bpb", result["val"])
        self.assertLess(result["artifact"]["estimated_mb"], 16.0)
        self.assertIn("jacobian_proxy_loss", result["train"]["history"][0])
        self.assertGreaterEqual(
            result["train"]["history"][0]["loss"],
            result["train"]["history"][0]["ce_loss"],
        )

    def test_fused_training_smoke(self) -> None:
        cfg = self._base_config(cross_token_mode="fused")
        cfg.kernel_feature_map = "identity"

        result = train_and_evaluate(cfg)

        self.assertEqual(result["config"]["cross_token_mode"], "fused")
        self.assertGreater(len(result["train"]["history"]), 0)
        self.assertIn("bpb", result["test"])
        self.assertLess(result["artifact"]["estimated_mb"], 16.0)

    def test_stochastic_rounding_changes_logits(self) -> None:
        base_cfg = self._base_config(cross_token_mode="fused")
        base_cfg.quantization = "ternary"
        noisy_cfg = self._base_config(cross_token_mode="fused")
        noisy_cfg.quantization = "ternary"
        noisy_cfg.stochastic_round_p = 1.0

        base_model = DeepFloorModel(base_cfg).to(self.device)
        noisy_model = DeepFloorModel(noisy_cfg).to(self.device)
        noisy_model.load_state_dict(base_model.state_dict())
        inputs = torch.randint(0, 256, (2, base_cfg.seq_len), device=self.device)

        torch.manual_seed(0)
        clean_logits = base_model(inputs, adaptive_floor=True, recurrence_steps=4)
        torch.manual_seed(0)
        noisy_logits = noisy_model(inputs, adaptive_floor=True, recurrence_steps=4)

        self.assertFalse(torch.allclose(clean_logits, noisy_logits))

    def test_training_forward_reports_tbptt_detaches(self) -> None:
        cfg = self._base_config(cross_token_mode="fused")
        cfg.tbptt_chunk = 2
        model = DeepFloorModel(cfg).to(self.device)
        model.train()
        inputs = torch.randint(0, 256, (2, cfg.seq_len), device=self.device)

        _, metadata = model(inputs, adaptive_floor=False, recurrence_steps=5, tbptt_chunk=2, return_metadata=True)

        self.assertEqual(metadata["tbptt_detach_counts"], [2, 2])

    def test_recurrent_block_param_count_matches_qkvo(self) -> None:
        cfg = self._base_config(cross_token_mode="floor")
        model = DeepFloorModel(cfg)
        block = model.blocks[0]
        # QKV+O at d: 4 × d × d weight params (norms have no weight params in RMSNorm)
        weight_params = sum(p.numel() for p in block.parameters() if p.dim() >= 2)
        self.assertEqual(weight_params, 4 * cfg.recurrent_dim * cfg.recurrent_dim)

    def test_cosine_lr_schedule_decays_during_training(self) -> None:
        cfg = self._base_config(cross_token_mode="floor")
        cfg.train_steps = 8
        cfg.warmup_steps = 2
        cfg.min_lr_scale = 0.1
        result = train_and_evaluate(cfg)
        history = result["train"]["history"]
        # After warmup, LR should decay
        self.assertIn("lr", history[-1])
        self.assertLess(history[-1]["lr"], history[2]["lr"])

    def test_jacobian_proxy_loss_samples_valid_depth(self) -> None:
        cfg = self._base_config(cross_token_mode="floor")
        model = DeepFloorModel(cfg).to(self.device)
        inputs = torch.randint(0, 256, (2, cfg.seq_len), device=self.device)

        penalty, gain, probe_step = model.jacobian_proxy_loss(inputs, recurrence_steps=5)

        self.assertGreaterEqual(probe_step, 0)
        self.assertLess(probe_step, 5)
        self.assertTrue(torch.isfinite(penalty))
        self.assertTrue(torch.isfinite(gain))


if __name__ == "__main__":
    unittest.main()
