from __future__ import annotations

import unittest

import numpy as np

from frontier_cache import CausalCacheConfig, ScoreFirstCausalCache
from frontier_eval import (
    assign_windows_to_score_chunks,
    commit_position_range,
    plan_distributed_window_shard,
    sliding_window_starts,
)


def _make_cache() -> ScoreFirstCausalCache:
    return ScoreFirstCausalCache(
        CausalCacheConfig(
            mode="ppm",
            max_order=2,
            alpha=0.30,
            min_count=1,
            buckets=1024,
            mixing="dirichlet",
            count_smoothing=4.0,
            alpha_min=0.10,
            alpha_max=0.50,
            entropy_center=3.5,
            entropy_slope=2.0,
            order_entropy_centers={7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5},
        )
    )


class FrontierEvalPlanTest(unittest.TestCase):
    def test_distributed_cache_shard_matches_single_rank_scores(self) -> None:
        token_stream = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1], dtype=np.int64)
        total_tokens = len(token_stream) - 1
        seq_len = 4
        stride = 2
        model_probs_by_window = {
            0: np.array([0.20, 0.20, 0.20, 0.20], dtype=np.float64),
            2: np.array([0.30, 0.30], dtype=np.float64),
            4: np.array([0.40, 0.40], dtype=np.float64),
            6: np.array([0.50, 0.50], dtype=np.float64),
        }
        window_starts = sliding_window_starts(total_tokens, seq_len, stride, require_full_stride=False)
        single_cache = _make_cache()
        single_scores: dict[int, float] = {}
        for ws in window_starts:
            end = min(ws + seq_len, total_tokens)
            wlen = end - ws
            score_offset = 0 if ws == 0 else max(wlen - stride, 0)
            positions = np.arange(ws + score_offset + 1, ws + wlen + 1, dtype=np.int64)
            mixed = single_cache.score_segment(token_stream, positions, model_probs_by_window[ws])
            single_cache.commit_segment(token_stream, positions)
            single_scores.update({int(pos): float(prob) for pos, prob in zip(positions, mixed)})

        distributed_scores: dict[int, float] = {}
        world_size = 2
        score_chunks = assign_windows_to_score_chunks(window_starts, total_tokens, seq_len, stride, chunk_tokens=4)
        rank_caches = [_make_cache() for _ in range(world_size)]
        for windows in score_chunks:
            for rank in range(world_size):
                plan = plan_distributed_window_shard(windows, total_tokens, seq_len, stride, rank, world_size)
                commit_position_range(rank_caches[rank], token_stream, plan.prefix_start, plan.prefix_end, block_size=8)
                for ws in plan.windows:
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    score_offset = 0 if ws == 0 else max(wlen - stride, 0)
                    positions = np.arange(ws + score_offset + 1, ws + wlen + 1, dtype=np.int64)
                    mixed = rank_caches[rank].score_segment(token_stream, positions, model_probs_by_window[ws])
                    rank_caches[rank].commit_segment(token_stream, positions)
                    distributed_scores.update({int(pos): float(prob) for pos, prob in zip(positions, mixed)})
                commit_position_range(rank_caches[rank], token_stream, plan.suffix_start, plan.suffix_end, block_size=8)

        self.assertEqual(sorted(distributed_scores), sorted(single_scores))
        for position, single_prob in single_scores.items():
            self.assertAlmostEqual(distributed_scores[position], single_prob, places=12)

    def test_single_rank_plan_has_no_prefix_or_suffix(self) -> None:
        total_tokens = 8
        seq_len = 4
        stride = 2
        windows = sliding_window_starts(total_tokens, seq_len, stride, require_full_stride=False)
        plan = plan_distributed_window_shard(windows, total_tokens, seq_len, stride, rank=0, world_size=1)
        self.assertEqual(plan.windows, windows)
        self.assertIsNone(plan.prefix_end)
        self.assertIsNone(plan.suffix_start)


if __name__ == "__main__":
    unittest.main()
