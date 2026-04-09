from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from spectral_flood_walk_v0 import build_sentencepiece_luts, export_quantized_model_npz
from spectral_flood_walk_v1 import (
    ProductKeyMemory,
    SpectralFloodWalkV1A,
    V1Config,
    batch_from_starts,
    build_lm_starts,
    estimate_v1a_sizes,
    evaluate_v1a,
)


class SpectralFloodWalkV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = V1Config(
            batch_size=2,
            train_steps=4,
            eval_batches=2,
            seq_len=8,
            stride=4,
            embed_dim=32,
            num_layers=4,
            num_heads=4,
            ff_mult=2,
            pos_buckets=16,
            semantic_layers="1,3",
            pk_num_subkeys=8,
            pk_key_dim=4,
            pk_topk_sub=2,
            pk_topk_final=3,
            pk_code_dim=8,
        )
        self.model = SpectralFloodWalkV1A(self.cfg, vocab_size=64)

    def test_product_key_memory_returns_expected_shape(self) -> None:
        memory = ProductKeyMemory(
            dim=32,
            num_subkeys=8,
            key_dim=4,
            code_dim=8,
            topk_sub=2,
            topk_final=3,
        )
        x = torch.randn(2, 5, 32)
        out, stats = memory(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIn("selected_score_mean", stats)
        self.assertTrue(torch.isfinite(stats["selected_score_mean"]))

    def test_forward_emits_normalized_probabilities(self) -> None:
        inputs = torch.randint(0, 64, (2, self.cfg.seq_len))
        targets = torch.randint(0, 64, (2, self.cfg.seq_len))
        out = self.model(inputs, targets)
        self.assertIsNotNone(out["loss"])
        probs = torch.softmax(out["logits"], dim=-1)
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        self.assertIsNotNone(out["semantic_score_mean"])

    def test_size_estimate_tracks_semantic_memory(self) -> None:
        sizes = estimate_v1a_sizes(self.cfg, vocab_size=1024)
        self.assertGreater(sizes["compact_model_bytes_estimate"], 0.0)
        self.assertGreater(sizes["expanded_semantic_bytes_estimate"], 0.0)
        cfg_no_sem = V1Config(use_semantic_memory=False, semantic_layers="")
        sizes_no_sem = estimate_v1a_sizes(cfg_no_sem, vocab_size=1024)
        self.assertEqual(sizes_no_sem["expanded_semantic_bytes_estimate"], 0.0)

    def test_batching_and_eval_report_val_bpb(self) -> None:
        tokens = torch.randint(0, 64, (96,), dtype=torch.long)
        starts = build_lm_starts(int(tokens.numel()), self.cfg.seq_len, self.cfg.stride)
        inputs, targets = batch_from_starts(tokens, starts[: self.cfg.batch_size], self.cfg.seq_len, torch.device("cpu"))
        self.assertEqual(inputs.shape, (self.cfg.batch_size, self.cfg.seq_len))
        self.assertEqual(targets.shape, (self.cfg.batch_size, self.cfg.seq_len))
        tokenizer_path = Path(__file__).resolve().parents[1] / "data" / "tokenizers" / "fineweb_1024_bpe.model"
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            tokenizer_path, 1024, torch.device("cpu")
        )
        metrics = evaluate_v1a(
            self.model,
            tokens,
            starts,
            self.cfg,
            torch.device("cpu"),
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        self.assertIn("val_bpb", metrics)
        self.assertGreater(metrics["val_bpb"], 0.0)

    def test_quantized_export_writes_model(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "model_v1a_int8.npz"
            size = export_quantized_model_npz(self.model, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(size, 0)


if __name__ == "__main__":
    unittest.main()
