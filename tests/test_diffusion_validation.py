from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sentencepiece as spm

from diffusion_objectives import (
    beta_schedule_from_mask_rates,
    build_mask_rates,
    corruption_rng,
    corrupt_batch_np,
    posterior_clean_probability,
    posterior_mask_probability,
    stratified_timesteps,
)
from validation_common import build_sentencepiece_luts, count_batch_bytes


class DiffusionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            num_diffusion_steps=4,
            mask_schedule="cosine",
            min_mask_rate=0.0,
            max_mask_rate=1.0,
        )

    def test_beta_schedule_reconstructs_cumulative_mask_rates(self) -> None:
        mask_rates = build_mask_rates(8, "cosine", 0.0, 1.0)
        betas = beta_schedule_from_mask_rates(mask_rates)
        reconstructed = np.zeros_like(mask_rates)
        for t in range(1, mask_rates.size):
            reconstructed[t] = reconstructed[t - 1] + (1.0 - reconstructed[t - 1]) * betas[t]
        np.testing.assert_allclose(reconstructed, mask_rates, atol=1e-6, rtol=1e-6)

    def test_absorbing_posterior_probabilities_match_closed_form(self) -> None:
        mask_rates = np.array([0.0, 0.2, 0.5, 1.0], dtype=np.float32)
        timesteps = np.array([1, 2, 3], dtype=np.int32)
        clean_prob = posterior_clean_probability(mask_rates, timesteps)
        mask_prob = posterior_mask_probability(mask_rates, timesteps)
        expected_clean = np.array([(0.2 - 0.0) / 0.2, (0.5 - 0.2) / 0.5, (1.0 - 0.5) / 1.0], dtype=np.float32)
        expected_mask = np.array([0.0 / 0.2, 0.2 / 0.5, 0.5 / 1.0], dtype=np.float32)
        np.testing.assert_allclose(clean_prob, expected_clean, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(mask_prob, expected_mask, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(clean_prob + mask_prob, np.ones_like(clean_prob), atol=1e-6, rtol=1e-6)

    def test_masked_state_kl_matches_closed_form_weighted_nll(self) -> None:
        mask_rates = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float64)
        clean_token_probs = np.array([0.7, 0.8, 0.9], dtype=np.float64)
        brute_force = 0.0
        closed_form = 0.0
        for t in range(1, mask_rates.size):
            m_t = float(mask_rates[t])
            m_prev = float(mask_rates[t - 1])
            alpha = (m_t - m_prev) / m_t
            p_clean = float(clean_token_probs[t - 1])
            p_theta = {"clean": alpha * p_clean, "mask": 1.0 - alpha}
            q = {"clean": alpha, "mask": 1.0 - alpha}
            kl_masked = sum(q[name] * math.log(q[name] / p_theta[name]) for name in ("clean", "mask") if q[name] > 0)
            brute_force += m_t * kl_masked
            closed_form += (m_t - m_prev) * (-math.log(p_clean))
        self.assertAlmostEqual(brute_force, closed_form, places=8)

    def test_better_stub_denoiser_gets_better_exact_elbo(self) -> None:
        mask_rates = np.array([0.0, 0.3, 0.6, 1.0], dtype=np.float64)
        better_probs = np.array([0.8, 0.85, 0.9], dtype=np.float64)
        worse_probs = np.array([0.55, 0.6, 0.65], dtype=np.float64)

        def exact_elbo(clean_probs: np.ndarray) -> float:
            return sum(
                (float(mask_rates[t]) - float(mask_rates[t - 1])) * (-math.log(float(clean_probs[t - 1])))
                for t in range(1, mask_rates.size)
            )

        self.assertLess(exact_elbo(better_probs), exact_elbo(worse_probs))

    def test_corruption_is_deterministic_for_fixed_seed(self) -> None:
        clean = np.arange(24, dtype=np.int32).reshape(3, 8) % 7 + 1
        timesteps = stratified_timesteps(clean.shape[0], self.args.num_diffusion_steps, offset=11)
        rng1 = corruption_rng(12345, 2, 0, 0)
        rng2 = corruption_rng(12345, 2, 0, 0)
        out1 = corrupt_batch_np(clean, self.args, rng1, 0, timesteps=timesteps)
        out2 = corrupt_batch_np(clean, self.args, rng2, 0, timesteps=timesteps)
        for a, b in zip(out1[:3], out2[:3], strict=True):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(out1[3], out2[3])

    def test_eval_corruption_allows_zero_mask_rows(self) -> None:
        tiny_args = SimpleNamespace(
            num_diffusion_steps=4,
            mask_schedule="linear",
            min_mask_rate=0.0,
            max_mask_rate=1.0,
        )
        clean = np.arange(8, dtype=np.int32).reshape(1, 8) + 1
        rng = np.random.default_rng(0)
        corrupted, _, loss_mask, _ = corrupt_batch_np(
            clean,
            tiny_args,
            rng,
            0,
            timesteps=np.array([1], dtype=np.int32),
            mask_rates=np.array([0.0, 0.0, 0.5, 0.75, 1.0], dtype=np.float32),
            ensure_masked_token=False,
        )
        np.testing.assert_array_equal(corrupted, clean)
        np.testing.assert_array_equal(loss_mask, np.zeros_like(loss_mask))

    def test_stratified_timesteps_cover_full_cycle(self) -> None:
        steps = stratified_timesteps(batch_size=10, num_diffusion_steps=4, offset=3)
        self.assertEqual(set(steps.tolist()), {1, 2, 3, 4})
        self.assertEqual(steps[0], 4)
        self.assertEqual(steps[1], 1)

    def test_byte_count_matches_incremental_decode_growth(self) -> None:
        tokenizer_path = Path("data/tokenizers/fineweb_1024_bpe.model")
        if not tokenizer_path.is_file():
            self.skipTest("local sentencepiece tokenizer is unavailable")
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, int(sp.vocab_size()))
        ids = sp.encode("The quick brown fox jumps over the lazy dog.")
        self.assertGreaterEqual(len(ids), 3)
        calc = count_batch_bytes(
            np.asarray(ids[:-1], dtype=np.int32),
            np.asarray(ids[1:], dtype=np.int32),
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        manual = 0
        for end in range(1, len(ids)):
            before = sp.decode(ids[:end]).encode("utf-8")
            after = sp.decode(ids[: end + 1]).encode("utf-8")
            manual += len(after) - len(before)
        self.assertEqual(calc, float(manual))


if __name__ == "__main__":
    unittest.main()
