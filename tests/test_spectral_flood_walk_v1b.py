from __future__ import annotations

import unittest
from pathlib import Path

import torch

from spectral_flood_walk_v0 import build_sentencepiece_luts
from spectral_flood_walk_v1 import SpectralFloodWalkV1A
from spectral_flood_walk_v1b import LocalEpisodicMemory, V1BConfig, evaluate_mode


class SpectralFloodWalkV1BTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = V1BConfig(
            batch_size=2,
            train_steps=2,
            eval_batches=2,
            seq_len=8,
            stride=4,
            embed_dim=32,
            num_layers=3,
            num_heads=4,
            ff_mult=2,
            pos_buckets=16,
            episodic_bucket_count=8,
            episodic_max_entries=32,
            episodic_topk=4,
            maintenance_every=1,
            maintenance_budget_buckets=4,
            maintenance_source_limit=8,
            summary_per_bucket=2,
            merge_similarity=0.8,
        )
        self.model = SpectralFloodWalkV1A(self.cfg.controller_config(), vocab_size=64)

    def test_local_memory_refines_bucket_summaries(self) -> None:
        memory = LocalEpisodicMemory(self.cfg, torch.device("cpu"), vocab_size=64)
        keys = F_normalize(torch.tensor([[1.0] * 32, [1.0] * 32, [-1.0] * 32]))
        values = F_normalize(torch.tensor([[0.5] * 32, [0.5] * 32, [-0.5] * 32]))
        surprise = torch.tensor([0.2, 0.3, 0.4])
        bucket_ids = torch.tensor([0, 0, 1], dtype=torch.long)
        appended = memory.append(keys, values, surprise, age=0, bucket_ids=bucket_ids)
        self.assertEqual(appended, 3)
        refine_stats = memory.refine_dirty()
        self.assertGreaterEqual(refine_stats["processed_buckets"], 1.0)
        stats = memory.snapshot_stats()
        self.assertGreater(stats["raw_entry_count"], 0.0)
        self.assertGreaterEqual(stats["summary_entry_count"], 1.0)

    def test_local_memory_router_spreads_random_keys(self) -> None:
        memory = LocalEpisodicMemory(self.cfg, torch.device("cpu"), vocab_size=64)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(123)
        token_windows = torch.randint(0, 64, (32, self.cfg.seq_len), generator=gen)
        bucket_ids = memory.bucket_ids_for_tokens(token_windows)
        self.assertGreater(bucket_ids.unique().numel(), 1)

    def test_eval_modes_emit_memory_stats_and_bpb(self) -> None:
        tokens = torch.randint(0, 64, (96,), dtype=torch.long)
        starts = list(range(0, 96 - self.cfg.seq_len - 1, self.cfg.stride))
        tokenizer_path = Path(__file__).resolve().parents[1] / "data" / "tokenizers" / "fineweb_1024_bpe.model"
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            tokenizer_path, 1024, torch.device("cpu")
        )
        controller = evaluate_mode(
            self.model,
            tokens,
            starts,
            self.cfg,
            torch.device("cpu"),
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            "controller",
        )
        refined = evaluate_mode(
            self.model,
            tokens,
            starts,
            self.cfg,
            torch.device("cpu"),
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            "refined",
        )
        self.assertGreater(controller["val_bpb"], 0.0)
        self.assertGreater(refined["val_bpb"], 0.0)
        self.assertIn("memory", refined)
        self.assertIn("retrieval", refined)


def F_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=-1, eps=1e-6)


if __name__ == "__main__":
    unittest.main()
