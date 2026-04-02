from __future__ import annotations

import unittest
from pathlib import Path

import torch

from spectral_flood_walk_v0 import build_sentencepiece_luts
from spectral_flood_walk_v2b import (
    PersistentHiddenMemory,
    ResidualRouter,
    StrongTransformerLM,
    V2bConfig,
    build_eval_score_mask,
    build_lm_starts,
    evaluate_mode,
    hidden_cross_entropy_gradient,
)


class SpectralFloodWalkV2BTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = V2bConfig(
            batch_size=2,
            train_steps=2,
            eval_batches=2,
            seq_len=8,
            stride=4,
            model_dim=32,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            spine_variant="xsa",
            xsa_last_n=2,
            memory_orders="1,2",
            memory_table_size=128,
            memory_min_read_count=1.0,
            maintenance_passes=1,
            maintenance_max_slots=16,
        )
        self.device = torch.device("cpu")
        self.model = StrongTransformerLM(self.cfg, vocab_size=64).to(self.device)
        self.router = ResidualRouter(self.cfg.memory_order_ids, self.cfg.memory_table_size, self.device)
        self.memory = PersistentHiddenMemory(
            orders=self.cfg.memory_order_ids,
            table_size=self.cfg.memory_table_size,
            dim=self.cfg.model_dim,
            device=self.device,
            order_scales=self.cfg.memory_order_scale_values,
            decay=self.cfg.memory_decay,
            ema_decay=self.cfg.memory_ema_decay,
            read_scale=self.cfg.memory_read_scale,
            min_read_count=self.cfg.memory_min_read_count,
            max_delta_norm=self.cfg.memory_max_delta_norm,
            maintenance_passes=self.cfg.maintenance_passes,
            maintenance_blend=self.cfg.maintenance_blend,
            maintenance_max_slots=self.cfg.maintenance_max_slots,
            maintenance_metric=self.cfg.maintenance_metric,
            maintenance_use_grad=self.cfg.maintenance_use_grad,
        )

    def test_hidden_cross_entropy_gradient_matches_hidden_shape(self) -> None:
        inputs = torch.randint(0, 64, (2, self.cfg.seq_len), dtype=torch.long)
        targets = torch.randint(0, 64, (2, self.cfg.seq_len), dtype=torch.long)
        out = self.model(inputs)
        hidden = out["hidden"]
        logits = out["logits"]
        assert hidden is not None
        assert logits is not None
        grad = hidden_cross_entropy_gradient(self.model, logits, targets)
        self.assertEqual(tuple(grad.shape), tuple(hidden.shape))
        self.assertGreater(float(grad.abs().sum().item()), 0.0)

    def test_hidden_memory_respects_score_mask(self) -> None:
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
        context_ids = self.router.context_ids(inputs)
        hidden_grad = torch.randn(1, self.cfg.seq_len, self.cfg.model_dim)
        zero_mask = torch.zeros((1, self.cfg.seq_len), dtype=torch.bool)
        update_stats = self.memory.update(context_ids, hidden_grad, step_size=0.1, score_mask=zero_mask)
        self.assertEqual(update_stats["updated_slots"], 0.0)
        self.assertEqual(float(self.memory.delta.abs().sum().item()), 0.0)

    def test_min_read_count_delays_memory_reads(self) -> None:
        cfg = V2bConfig(
            seq_len=4,
            model_dim=16,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            memory_orders="1",
            memory_table_size=32,
            memory_min_read_count=2.0,
            maintenance_passes=0,
        )
        memory = PersistentHiddenMemory(
            orders=cfg.memory_order_ids,
            table_size=cfg.memory_table_size,
            dim=cfg.model_dim,
            device=self.device,
            order_scales=cfg.memory_order_scale_values,
            decay=cfg.memory_decay,
            ema_decay=cfg.memory_ema_decay,
            read_scale=cfg.memory_read_scale,
            min_read_count=cfg.memory_min_read_count,
            max_delta_norm=cfg.memory_max_delta_norm,
            maintenance_passes=cfg.maintenance_passes,
            maintenance_blend=cfg.maintenance_blend,
            maintenance_max_slots=cfg.maintenance_max_slots,
            maintenance_metric=cfg.maintenance_metric,
            maintenance_use_grad=cfg.maintenance_use_grad,
        )
        router = ResidualRouter(cfg.memory_order_ids, cfg.memory_table_size, self.device)
        inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        context_ids = router.context_ids(inputs)
        hidden_grad = torch.ones((1, 4, cfg.model_dim), dtype=torch.float32)
        full_mask = torch.ones((1, 4), dtype=torch.bool)

        memory.update(context_ids, hidden_grad, step_size=0.1, score_mask=full_mask)
        first_lookup, _, _ = memory.lookup(context_ids)
        self.assertEqual(float(first_lookup.abs().sum().item()), 0.0)

        memory.update(context_ids, hidden_grad, step_size=0.1, score_mask=full_mask)
        second_lookup, _, _ = memory.lookup(context_ids)
        self.assertGreater(float(second_lookup.abs().sum().item()), 0.0)

    def test_online_eval_reports_memory_stats_and_flops(self) -> None:
        tokens = torch.randint(0, 64, (96,), dtype=torch.long)
        starts = build_lm_starts(int(tokens.numel()), self.cfg.seq_len, self.cfg.stride)
        tokenizer_path = Path(__file__).resolve().parents[1] / "data" / "tokenizers" / "fineweb_1024_bpe.model"
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            tokenizer_path, 1024, self.device
        )

        context = evaluate_mode(
            mode="context",
            model=self.model,
            memory=None,
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
            mode="online_persistent_hidden",
            model=self.model,
            memory=self.memory,
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
        self.assertIn("persistent_memory", online)
        self.assertGreater(float(online["memory_total_flops_estimate"]), 0.0)
        self.assertGreaterEqual(float(online["maintenance_slots"]), 0.0)

    def test_eval_score_mask_only_counts_new_tokens(self) -> None:
        mask = build_eval_score_mask([0, 4, 8], seq_len=8, stride=4, device=self.device)
        self.assertEqual(mask.shape, (3, 8))
        self.assertEqual(int(mask[0].sum().item()), 8)
        self.assertEqual(int(mask[1].sum().item()), 4)
        self.assertTrue(torch.equal(mask[1], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.bool)))


if __name__ == "__main__":
    unittest.main()
