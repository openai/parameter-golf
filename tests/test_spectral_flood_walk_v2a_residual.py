from __future__ import annotations

import unittest

import torch

from spectral_flood_walk_v2a_residual import (
    ResidualBasis,
    ResidualRouter,
    ResidualTables,
    build_eval_score_mask,
    parse_orders_csv,
    probability_sum_deviation,
)


class ResidualHelpersTest(unittest.TestCase):
    def test_parse_orders_csv(self) -> None:
        self.assertEqual(parse_orders_csv("1,2,4"), (1, 2, 4))
        self.assertEqual(parse_orders_csv(""), ())

    def test_build_eval_score_mask_only_scores_new_tail(self) -> None:
        mask = build_eval_score_mask([0, 64], seq_len=8, stride=3, device=torch.device("cpu"))
        self.assertEqual(mask[0].sum().item(), 8)
        self.assertEqual(mask[1].sum().item(), 3)
        self.assertTrue(mask[1, -3:].all().item())
        self.assertFalse(mask[1, :-3].any().item())

    def test_probability_sum_deviation_stays_zero_after_residual(self) -> None:
        logits = torch.randn(2, 5, 16)
        delta = torch.randn(2, 5, 16)
        self.assertLess(probability_sum_deviation(logits, delta), 1e-6)

    def test_basis_from_embedding_svd_has_expected_shape(self) -> None:
        basis = ResidualBasis.from_embedding_svd(torch.randn(32, 12), rank=6, num_orders=4)
        self.assertEqual(tuple(basis.basis.shape), (32, 6))
        self.assertEqual(tuple(basis.order_scales().shape), (4,))

    def test_residual_table_update_respects_score_mask(self) -> None:
        device = torch.device("cpu")
        basis = ResidualBasis.from_embedding_svd(torch.randn(16, 8), rank=4, num_orders=2)
        router = ResidualRouter((1, 2), table_size=32, device=device)
        tables = ResidualTables(orders=(1, 2), table_size=32, rank=4, device=device, decay=1.0)
        inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        context_ids = router.context_ids(inputs)
        probs = torch.softmax(torch.randn(1, 4, 16), dim=-1)
        targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        direction = basis.projected_direction(probs, targets)
        zero_mask = torch.zeros((1, 4), dtype=torch.bool)
        tables.update(context_ids, direction, basis.order_scales(), step_size=0.1, score_mask=zero_mask)
        self.assertEqual(float(tables.coeffs.abs().sum().item()), 0.0)
        full_mask = torch.ones((1, 4), dtype=torch.bool)
        tables.update(context_ids, direction, basis.order_scales(), step_size=0.1, score_mask=full_mask)
        self.assertGreater(float(tables.coeffs.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
