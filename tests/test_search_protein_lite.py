from __future__ import annotations

import unittest

from search.config import ParamSpec
from search.protein_lite import ProteinLite


class ProteinLiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = {
            "ITERATIONS": ParamSpec(distribution="int_uniform", min=500, max=5000, round_to=250),
            "MATRIX_LR": ParamSpec(distribution="log_uniform", min=0.01, max=0.08),
        }

    def _make_optimizer(self) -> ProteinLite:
        return ProteinLite(
            self.specs,
            seed=7,
            warm_start_suggestions=2,
            candidate_samples=16,
            max_observations=50,
            suggestions_per_center=8,
            prune_pareto=True,
            gp_alpha=1e-6,
            target_cost_ratios=(0.25, 0.5, 0.75, 1.0),
        )

    def test_resume_keeps_next_suggestion_deterministic(self):
        optimizer_a = self._make_optimizer()
        optimizer_b = self._make_optimizer()
        params0, _ = optimizer_a.suggest(0)
        params1, _ = optimizer_a.suggest(1)
        optimizer_a.observe(params0, score=-1.40, cost=100.0)
        optimizer_a.observe(params1, score=-1.35, cost=200.0)

        optimizer_b.observe(params0, score=-1.40, cost=100.0)
        optimizer_b.observe(params1, score=-1.35, cost=200.0)
        next_a, _ = optimizer_a.suggest(2)
        next_b, _ = optimizer_b.suggest(2)
        self.assertEqual(next_a, next_b)

    def test_lower_bpb_means_higher_score(self):
        optimizer = self._make_optimizer()
        params, _ = optimizer.suggest(0)
        optimizer.observe(params, score=-1.50, cost=100.0)
        optimizer.observe(params, score=-1.40, cost=120.0)
        best_score = max(obs.output for obs in optimizer.observations)
        self.assertAlmostEqual(best_score, -1.40)


if __name__ == "__main__":
    unittest.main()

