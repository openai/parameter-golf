from __future__ import annotations

import unittest

import numpy as np

from frontier_cache import CausalCacheConfig, ScoreFirstCausalCache


class FrontierCacheTest(unittest.TestCase):
    def make_cache(self, **overrides: object) -> ScoreFirstCausalCache:
        defaults: dict[str, object] = {
            "mode": "ppm",
            "max_order": 3,
            "alpha": 0.30,
            "min_count": 1,
            "buckets": 1024,
            "mixing": "dirichlet",
            "count_smoothing": 2.0,
            "alpha_min": 0.10,
            "alpha_max": 0.50,
            "entropy_center": 3.5,
            "entropy_slope": 2.0,
            "order_entropy_centers": {7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5},
        }
        defaults.update(overrides)
        config = CausalCacheConfig(
            **defaults,
        )
        return ScoreFirstCausalCache(config)

    def test_score_commit_order_is_enforced(self) -> None:
        cache = self.make_cache()
        token_stream = np.array([1, 2, 1, 2, 1, 2], dtype=np.int64)
        positions = np.array([1, 2, 3], dtype=np.int64)
        model_probs = np.array([0.2, 0.2, 0.2], dtype=np.float64)

        cache.score_segment(token_stream, positions, model_probs)
        with self.assertRaisesRegex(RuntimeError, "commit the pending segment before scoring again"):
            cache.score_segment(token_stream, positions, model_probs)

        cache.commit_segment(token_stream, positions)
        with self.assertRaisesRegex(RuntimeError, "cannot commit before score_segment"):
            cache.commit_segment(token_stream, positions)

    def test_score_is_independent_of_future_suffix(self) -> None:
        token_stream_a = np.array([1, 2, 1, 2, 1, 2, 7, 7], dtype=np.int64)
        token_stream_b = np.array([1, 2, 1, 2, 1, 2, 9, 9], dtype=np.int64)
        history_positions = np.array([1, 2, 3, 4], dtype=np.int64)
        score_positions = np.array([5], dtype=np.int64)
        history_probs = np.full(history_positions.shape, 0.25, dtype=np.float64)
        model_prob = np.array([0.1], dtype=np.float64)

        cache_a = self.make_cache(max_order=2)
        cache_a.score_segment(token_stream_a, history_positions, history_probs)
        cache_a.commit_segment(token_stream_a, history_positions)
        score_a = cache_a.score_segment(token_stream_a, score_positions, model_prob)

        cache_b = self.make_cache(max_order=2)
        cache_b.score_segment(token_stream_b, history_positions, history_probs)
        cache_b.commit_segment(token_stream_b, history_positions)
        score_b = cache_b.score_segment(token_stream_b, score_positions, model_prob)

        self.assertAlmostEqual(float(score_a[0]), float(score_b[0]), places=12)

    def test_dirichlet_mixing_uses_single_pass_posterior_predictive(self) -> None:
        cache = self.make_cache(max_order=2, count_smoothing=2.0)
        token_stream = np.array([1, 2, 1, 3, 1, 2, 1, 2], dtype=np.int64)
        history_positions = np.array([1, 3, 5], dtype=np.int64)
        score_positions = np.array([7], dtype=np.int64)
        history_probs = np.full(history_positions.shape, 0.25, dtype=np.float64)
        model_prob = np.array([0.1], dtype=np.float64)

        cache.score_segment(token_stream, history_positions, history_probs)
        cache.commit_segment(token_stream, history_positions)
        mixed = cache.score_segment(token_stream, score_positions, model_prob)

        expected = (2.0 + 2.0 * 0.1) / (3.0 + 2.0)
        self.assertAlmostEqual(float(mixed[0]), expected, places=12)


if __name__ == "__main__":
    unittest.main()
