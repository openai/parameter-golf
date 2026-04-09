from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend

from spectral_flood_walk_v0 import build_sentencepiece_luts
from spectral_flood_walk_v2a import (
    ResidualBasis,
    ResidualRouter,
    ResidualTables,
    StrongTransformerLM,
    V2Config,
    attention_backend_override,
    build_eval_score_mask,
    build_lm_starts,
    evaluate_mode,
)


class SpectralFloodWalkV2ATests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = V2Config(
            batch_size=2,
            train_steps=2,
            calibration_steps=2,
            eval_batches=2,
            seq_len=8,
            stride=4,
            model_dim=32,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            residual_rank=8,
            residual_orders="1,2,3",
            residual_table_size=128,
            spine_variant="xsa",
            xsa_last_n=2,
        )
        self.device = torch.device("cpu")
        self.model = StrongTransformerLM(self.cfg, vocab_size=64).to(self.device)
        self.basis = ResidualBasis(64, self.cfg.residual_rank, len(self.cfg.residual_order_ids), 0.01).to(self.device)
        self.router = ResidualRouter(self.cfg.residual_order_ids, self.cfg.residual_table_size, self.device)
        self.tables = ResidualTables(
            orders=self.cfg.residual_order_ids,
            table_size=self.cfg.residual_table_size,
            rank=self.cfg.residual_rank,
            device=self.device,
            decay=self.cfg.residual_decay,
        )

    def test_router_changes_with_suffix(self) -> None:
        tokens_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        tokens_b = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 9]], dtype=torch.long)
        ids_a = self.router.context_ids(tokens_a)
        ids_b = self.router.context_ids(tokens_b)
        self.assertTrue(torch.equal(ids_a[:, :-1, 0], ids_b[:, :-1, 0]))
        self.assertFalse(torch.equal(ids_a[:, -1, :], ids_b[:, -1, :]))

    def test_residual_tables_update_and_stats(self) -> None:
        inputs = torch.randint(0, 64, (2, self.cfg.seq_len), dtype=torch.long)
        ids = self.router.context_ids(inputs)
        direction = torch.randn(2, self.cfg.seq_len, self.cfg.residual_rank)
        score_mask = torch.tensor(
            [
                [True, True, True, True, False, False, False, False],
                [False, False, False, False, True, True, True, True],
            ],
            dtype=torch.bool,
        )
        self.tables.update(ids, direction, self.basis.order_scales().detach(), step_size=0.1, score_mask=score_mask)
        self.tables.note_reads(ids)
        coeffs, stats = self.tables.lookup(ids)
        table_stats = self.tables.stats()
        self.assertEqual(coeffs.shape, (2, self.cfg.seq_len, len(self.cfg.residual_order_ids), self.cfg.residual_rank))
        self.assertGreater(stats["mean_coeff_norm"], 0.0)
        self.assertGreater(table_stats["resident_mb"], 0.0)
        self.assertGreater(table_stats["mean_non_empty_slots"], 0.0)

    def test_eval_modes_report_bpb_and_residual_stats(self) -> None:
        tokens = torch.randint(0, 64, (96,), dtype=torch.long)
        starts = build_lm_starts(int(tokens.numel()), self.cfg.seq_len, self.cfg.stride)
        tokenizer_path = Path(__file__).resolve().parents[1] / "data" / "tokenizers" / "fineweb_1024_bpe.model"
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            tokenizer_path, 1024, self.device
        )
        for _ in range(2):
            batch_inputs = tokens[: self.cfg.seq_len].unsqueeze(0).repeat(2, 1)
            ids = self.router.context_ids(batch_inputs)
            direction = torch.randn(2, self.cfg.seq_len, self.cfg.residual_rank)
            self.tables.update(ids, direction, self.basis.order_scales().detach(), step_size=0.05)
        context = evaluate_mode(
            mode="context",
            model=self.model,
            basis=None,
            tables=None,
            router=None,
            tokens=tokens,
            starts=starts,
            cfg=self.cfg,
            device=self.device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        online = evaluate_mode(
            mode="online_residual",
            model=self.model,
            basis=self.basis,
            tables=self.tables,
            router=self.router,
            tokens=tokens,
            starts=starts,
            cfg=self.cfg,
            device=self.device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        self.assertGreater(context["val_bpb"], 0.0)
        self.assertGreater(online["val_bpb"], 0.0)
        self.assertIn("residual_tables", online)
        residual_stats = online["residual_tables"]
        assert isinstance(residual_stats, dict)
        self.assertIn("mean_hits", residual_stats)
        self.assertGreaterEqual(float(residual_stats["mean_hits"]), 0.0)

    def test_eval_score_mask_only_counts_new_tokens(self) -> None:
        mask = build_eval_score_mask([0, 4, 8], seq_len=8, stride=4, device=self.device)
        self.assertEqual(mask.shape, (3, 8))
        self.assertEqual(int(mask[0].sum().item()), 8)
        self.assertEqual(int(mask[1].sum().item()), 4)
        self.assertTrue(torch.equal(mask[1], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)))

    def test_attention_backend_override_uses_math_under_vmap(self) -> None:
        def detect(tensor: torch.Tensor) -> torch.Tensor:
            backend = attention_backend_override(tensor)
            return torch.tensor(
                1 if backend == SDPBackend.MATH else 0,
                dtype=torch.int64,
                device=tensor.device,
            )

        flags = torch.vmap(detect)(torch.randn(3, 4))
        self.assertTrue(torch.equal(flags.cpu(), torch.ones((3,), dtype=torch.int64)))
        self.assertIsNone(attention_backend_override(torch.randn(4)))


if __name__ == "__main__":
    unittest.main()
