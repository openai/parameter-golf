from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from spectral_flood_walk_v0 import (
    SeedBank,
    SpectralFloodWalkV0,
    V0Config,
    build_positions,
    build_sentencepiece_luts,
    evaluate,
    export_quantized_model_npz,
    fake_quantize_unit_int8,
    get_cuda_memory_stats,
    lr_scale_for_step,
    retrieval_dropout,
    summarize_history,
)


class SpectralFloodWalkV0Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = V0Config(
            batch_size=2,
            train_steps=4,
            bank_capacity=4,
            bank_min_entries=2,
            bank_writes_per_step=2,
            topk=4,
            prefix_len=8,
            chunk_size=4,
            embed_dim=16,
            expert_hidden=16,
            fused_dim=32,
            runtime_dim=24,
            code_dim=8,
            query_dim=8,
            reader_heads=4,
        )
        self.model = SpectralFloodWalkV0(self.cfg, vocab_size=32)
        self.prefix = torch.randint(0, 32, (2, self.cfg.prefix_len))
        self.chunk_targets = torch.randint(0, 32, (2, self.cfg.chunk_size))
        self.next_chunk = torch.randint(0, 32, (2, self.cfg.chunk_size))
        self.chunk_inputs = torch.cat([self.prefix[:, -1:], self.chunk_targets[:, :-1]], dim=1)

    def test_fake_quant_straight_through_preserves_gradients(self) -> None:
        x = torch.randn(3, 8, requires_grad=True)
        x_st, q = fake_quantize_unit_int8(x)
        self.assertEqual(q.dtype, torch.int8)
        loss = x_st.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_seed_bank_retains_highest_priority_entries(self) -> None:
        bank = SeedBank(model=self.model, capacity=3)
        keys = torch.randint(-127, 128, (5, self.cfg.query_dim), dtype=torch.int8)
        codes = torch.randn(5, self.cfg.code_dim)
        runtime_states = torch.randn(5, self.cfg.runtime_dim)
        expert_ids = torch.tensor([0, 1, 2, 3, 4]) % self.cfg.num_experts
        pos = torch.arange(5, dtype=torch.long)
        scores = torch.tensor([0.1, 0.9, 0.4, 0.8, 0.2])
        written = bank.add_entries(keys, codes, runtime_states, expert_ids, pos, scores, step=3, max_new_entries=None)
        self.assertEqual(written, 5)
        self.assertEqual(bank.size, 3)
        kept = sorted(float(x) for x in bank.scores.cpu())
        for observed, expected in zip(kept, [0.4, 0.8, 0.9]):
            self.assertAlmostEqual(observed, expected, places=5)

    def test_note_reads_updates_read_mass(self) -> None:
        bank = SeedBank(
            model=self.model,
            capacity=4,
            keys=torch.randint(-127, 128, (4, self.cfg.query_dim), dtype=torch.int8),
            codes=torch.randn(4, self.cfg.code_dim),
            expert_ids=torch.zeros(4, dtype=torch.long),
            pos_buckets=torch.zeros(4, dtype=torch.long),
            scores=torch.ones(4),
        )
        idx = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        attn = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], dtype=torch.float32)
        bank.note_reads(idx, attn, ema=0.5)
        self.assertGreater(float(bank.read_mass[1].item()), 0.0)
        self.assertGreater(float(bank.read_mass[2].item()), 0.0)
        self.assertGreater(bank.resident_bytes(), 0)

    def test_seed_bank_exports_npz(self) -> None:
        bank = SeedBank(
            model=self.model,
            capacity=4,
            keys=torch.randint(-127, 128, (4, self.cfg.query_dim), dtype=torch.int8),
            codes=torch.randn(4, self.cfg.code_dim),
            expert_ids=torch.zeros(4, dtype=torch.long),
            pos_buckets=torch.arange(4, dtype=torch.long),
            scores=torch.linspace(0.1, 0.4, 4),
        )
        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "seed_pool.npz"
            size = bank.export_npz(out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(size, 0)
            reloaded = SeedBank.load_npz(out_path, model=self.model, device=torch.device("cpu"))
            self.assertEqual(reloaded.store_mode, "codes")
            self.assertEqual(reloaded.size, bank.size)
            self.assertEqual(reloaded.codes.shape, bank.codes.shape)

    def test_runtime_state_bank_retrieves_and_exports(self) -> None:
        cfg = V0Config(
            batch_size=2,
            topk=2,
            prefix_len=8,
            chunk_size=4,
            embed_dim=16,
            expert_hidden=16,
            fused_dim=32,
            runtime_dim=24,
            code_dim=8,
            query_dim=8,
            reader_heads=4,
            bank_store_mode="runtime",
            bank_runtime_dtype="fp16",
        )
        model = SpectralFloodWalkV0(cfg, vocab_size=32)
        keys = torch.randint(-127, 128, (3, cfg.query_dim), dtype=torch.int8)
        runtime_states = torch.randn(3, cfg.runtime_dim)
        bank = SeedBank(
            model=model,
            capacity=4,
            store_mode="runtime",
            runtime_states=runtime_states,
            keys=keys,
            expert_ids=torch.zeros(3, dtype=torch.long),
            pos_buckets=torch.arange(3, dtype=torch.long),
            scores=torch.linspace(0.1, 0.3, 3),
        )
        queries = torch.randn(2, cfg.query_dim)
        states, expert_ids, pos, idx = bank.retrieve(queries, topk=2)
        self.assertEqual(states.shape, (2, 2, cfg.runtime_dim))
        self.assertEqual(expert_ids.shape, (2, 2))
        self.assertEqual(pos.shape, (2, 2))
        self.assertEqual(idx.shape, (2, 2))
        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "runtime_seed_pool.npz"
            bank.export_npz(out_path)
            payload = dict(np.load(out_path, allow_pickle=True))
            self.assertIn("runtime_states", payload)
            self.assertNotIn("codes", payload)
            self.assertEqual(payload["store_mode"][0], "runtime")
            reloaded = SeedBank.load_npz(out_path, model=model, device=torch.device("cpu"))
            self.assertEqual(reloaded.store_mode, "runtime")
            self.assertEqual(reloaded.runtime_states.shape, bank.runtime_states.shape)

    def test_quantized_model_export_writes_npz(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "model_int8.npz"
            size = export_quantized_model_npz(self.model, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(size, 0)

    def test_forward_batch_matches_forward(self) -> None:
        bank = SeedBank(
            model=self.model,
            capacity=2,
            keys=torch.randint(-127, 128, (2, self.cfg.query_dim), dtype=torch.int8),
            codes=torch.randn(2, self.cfg.code_dim),
            expert_ids=torch.zeros(2, dtype=torch.long),
            pos_buckets=torch.zeros(2, dtype=torch.long),
            scores=torch.ones(2),
        )
        out_forward = self.model(
            prefix_tokens=self.prefix,
            chunk_inputs=self.chunk_inputs,
            chunk_targets=self.chunk_targets,
            next_chunk_tokens=self.next_chunk,
            bank=bank,
            retrieval_enabled=True,
        )
        out_batch = self.model.forward_batch(
            self.prefix,
            self.chunk_inputs,
            self.chunk_targets,
            self.next_chunk,
            bank,
            retrieval_enabled=True,
        )
        self.assertTrue(torch.allclose(out_forward["loss"], out_batch["loss"]))

    def test_forward_emits_normalized_probabilities(self) -> None:
        fused, _ = self.model.encode_prefix(self.prefix)
        _, q_int8 = self.model.query_from_fused(fused)
        target_state = self.model.encode_target(self.chunk_targets, self.next_chunk)
        target_norms = target_state.norm(dim=-1)
        self.assertTrue(torch.allclose(target_norms, torch.ones_like(target_norms), atol=1e-4))
        codes = self.model.code_from_target(target_state)
        expanded = self.model.expand_codes(codes)
        expanded_norms = expanded.norm(dim=-1)
        self.assertTrue(torch.allclose(expanded_norms, torch.ones_like(expanded_norms), atol=1e-4))
        bank = SeedBank(
            model=self.model,
            capacity=2,
            keys=q_int8.detach(),
            codes=codes.detach(),
            expert_ids=torch.zeros(q_int8.size(0), dtype=torch.long),
            pos_buckets=torch.zeros(q_int8.size(0), dtype=torch.long),
            scores=torch.ones(q_int8.size(0)),
        )
        out = self.model.forward_batch(
            self.prefix,
            self.chunk_inputs,
            self.chunk_targets,
            self.next_chunk,
            bank,
            retrieval_enabled=True,
        )
        probs = torch.softmax(out["logits"], dim=-1)
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
        self.assertEqual(out["retrieved_indices"].shape[-1], min(self.cfg.topk, bank.size))

    def test_eval_reports_val_bpb(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        tokenizer_path = repo_root / "data" / "tokenizers" / "fineweb_1024_bpe.model"
        train_tokens = torch.randint(0, 32, (96,), dtype=torch.long)
        positions = build_positions(int(train_tokens.numel()), self.cfg.prefix_len, self.cfg.chunk_size, self.cfg.chunk_size)
        fused, _ = self.model.encode_prefix(self.prefix)
        _, q_int8 = self.model.query_from_fused(fused)
        target_state = self.model.encode_target(self.chunk_targets, self.next_chunk)
        codes = self.model.code_from_target(target_state)
        bank = SeedBank(
            model=self.model,
            capacity=2,
            keys=q_int8.detach(),
            codes=codes.detach(),
            expert_ids=torch.zeros(q_int8.size(0), dtype=torch.long),
            pos_buckets=torch.zeros(q_int8.size(0), dtype=torch.long),
            scores=torch.ones(q_int8.size(0)),
        )
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            tokenizer_path, 1024, torch.device("cpu")
        )
        metrics = evaluate(
            self.model,
            train_tokens,
            positions,
            self.cfg,
            torch.device("cpu"),
            bank,
            True,
            False,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        self.assertIn("val_bpb", metrics)
        self.assertGreater(metrics["val_bpb"], 0.0)
        self.assertGreater(metrics["bytes"], 0.0)
        self.assertEqual(metrics["bank_resident_bytes"], float(bank.resident_bytes()))
        self.assertIsNone(get_cuda_memory_stats(torch.device("cpu")))

    def test_quantized_export_sanitizes_nan_weights(self) -> None:
        with TemporaryDirectory() as tmpdir:
            self.model.lm_head.weight.data.fill_(float("nan"))
            out_path = Path(tmpdir) / "model_nan_safe.npz"
            size = export_quantized_model_npz(self.model, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(size, 0)

    def test_eval_online_append_grows_eval_bank_without_mutating_seed_bank(self) -> None:
        cfg = V0Config(
            batch_size=2,
            train_steps=4,
            bank_capacity=8,
            bank_min_entries=1,
            bank_writes_per_step=2,
            topk=2,
            prefix_len=8,
            chunk_size=4,
            embed_dim=16,
            expert_hidden=16,
            fused_dim=32,
            runtime_dim=24,
            code_dim=8,
            query_dim=8,
            reader_heads=4,
            bank_store_mode="runtime",
            eval_online_append=True,
            eval_append_writes_per_batch=2,
            eval_samples=4,
        )
        model = SpectralFloodWalkV0(cfg, vocab_size=32)
        tokenizer_path = Path(__file__).resolve().parents[1] / "data" / "tokenizers" / "fineweb_1024_bpe.model"
        tokens = torch.randint(0, 32, (96,), dtype=torch.long)
        positions = build_positions(int(tokens.numel()), cfg.prefix_len, cfg.chunk_size, cfg.chunk_size)
        seed_bank = SeedBank(
            model=model,
            capacity=8,
            store_mode="runtime",
            keys=torch.randint(-127, 128, (2, cfg.query_dim), dtype=torch.int8),
            runtime_states=torch.randn(2, cfg.runtime_dim),
            expert_ids=torch.zeros(2, dtype=torch.long),
            pos_buckets=torch.zeros(2, dtype=torch.long),
            scores=torch.ones(2),
        )
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            tokenizer_path, 1024, torch.device("cpu")
        )
        metrics = evaluate(
            model,
            tokens,
            positions,
            cfg,
            torch.device("cpu"),
            seed_bank,
            True,
            True,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        self.assertEqual(seed_bank.size, 2)
        self.assertEqual(metrics["bank_initial_size"], 2.0)
        self.assertGreater(metrics["bank_final_size"], 2.0)
        self.assertGreater(metrics["appended_entries"], 0.0)
        self.assertEqual(metrics["online_append"], 1.0)

    def test_dropout_schedule_lr_schedule_and_history_summary(self) -> None:
        start = retrieval_dropout(0, self.cfg)
        end = retrieval_dropout(self.cfg.train_steps - 1, self.cfg)
        self.assertGreater(start, end)
        self.assertLess(lr_scale_for_step(0, self.cfg), 1.0)
        self.assertEqual(lr_scale_for_step(self.cfg.warmup_steps, self.cfg), 1.0)
        summary = summarize_history(
            [
                {
                    "loss": 1.0,
                    "loss_nll": 0.8,
                    "loss_ret": 0.1,
                    "loss_recon": 0.05,
                    "loss_gate": 0.05,
                    "step_ms": 10.0,
                    "dropout": 1.0,
                    "lr_scale": 0.5,
                    "bank_size": 2.0,
                    "candidates_written": 2.0,
                    "avg_candidate_score": 0.7,
                    "avg_read_mass": 0.1,
                    "retrieval_used": 0.0,
                },
                {
                    "loss": 0.5,
                    "loss_nll": 0.4,
                    "loss_ret": 0.05,
                    "loss_recon": 0.03,
                    "loss_gate": 0.02,
                    "step_ms": 12.0,
                    "dropout": 0.5,
                    "lr_scale": 1.0,
                    "bank_size": 4.0,
                    "candidates_written": 1.0,
                    "avg_candidate_score": 0.9,
                    "avg_read_mass": 0.3,
                    "retrieval_used": 1.0,
                },
            ]
        )
        self.assertAlmostEqual(summary["mean_loss"], 0.75)
        self.assertAlmostEqual(summary["mean_bank_size"], 3.0)
        self.assertEqual(summary["num_records"], 2.0)


if __name__ == "__main__":
    unittest.main()
