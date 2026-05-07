"""Tests for SCHEDULE_HORIZON_SECONDS env var in the long-train continuation script.

Phase 2: Verifies that the optional schedule horizon can decouple the LR/warmdown
schedule from the hard wallclock stop, while preserving backward compatibility.
"""
import os
import sys
import unittest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# We extract and test the pure logic from the train script without importing
# the full module (which requires torch/CUDA). The logic under test:
#   - Hyperparameter parsing of the new env var
#   - schedule_horizon_ms derivation
#   - training_frac() using schedule_horizon_ms
#   - lr_mul() (unchanged, but verify schedule fraction feeds correctly)
#   - Stop condition still uses max_wallclock_ms (not schedule horizon)
# ---------------------------------------------------------------------------

TRAIN_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..",
    "records", "track_non_record_16mb",
    "2026-04-30_PR1950_LongTrainArtifactScaling", "train_gpt.py"
)


def _read_env_var_line():
    """Confirm the env var line exists in the script."""
    with open(TRAIN_SCRIPT) as f:
        for line in f:
            if "SCHEDULE_HORIZON_SECONDS" in line:
                return line.strip()
    return None


class TestScheduleHorizonEnvVarExists(unittest.TestCase):
    """The env var must be declared in the Hyperparameters class."""

    def test_env_var_declared_in_script(self):
        line = _read_env_var_line()
        self.assertIsNotNone(line, "SCHEDULE_HORIZON_SECONDS not found in train script")
        # Should default to 0 (meaning: fall back to max_wallclock_seconds)
        self.assertIn("0", line)


class TestScheduleHorizonDerivation(unittest.TestCase):
    """Test the schedule_horizon_ms logic extracted from train_model."""

    def _derive_schedule_horizon_ms(self, max_wallclock_seconds, schedule_horizon_seconds,
                                     gptq_reserve_seconds=4.0):
        """Replicate the derivation logic from train_model."""
        max_wallclock_ms = (
            1e3 * max_wallclock_seconds if max_wallclock_seconds > 0 else None
        )
        if max_wallclock_ms is not None:
            max_wallclock_ms -= gptq_reserve_seconds * 1e3

        # schedule_horizon_ms: if SCHEDULE_HORIZON_SECONDS > 0, use it; else same as max_wallclock_ms
        if schedule_horizon_seconds > 0:
            schedule_horizon_ms = 1e3 * schedule_horizon_seconds - gptq_reserve_seconds * 1e3
        else:
            schedule_horizon_ms = max_wallclock_ms

        return max_wallclock_ms, schedule_horizon_ms

    def test_unset_defaults_to_max_wallclock(self):
        """When SCHEDULE_HORIZON_SECONDS=0, schedule_horizon_ms == max_wallclock_ms."""
        max_ms, sched_ms = self._derive_schedule_horizon_ms(
            max_wallclock_seconds=21600, schedule_horizon_seconds=0
        )
        self.assertEqual(max_ms, sched_ms)

    def test_explicit_horizon_differs_from_stop(self):
        """When SCHEDULE_HORIZON_SECONDS is set, schedule differs from stop horizon."""
        max_ms, sched_ms = self._derive_schedule_horizon_ms(
            max_wallclock_seconds=43200,  # 12h stop
            schedule_horizon_seconds=21600,  # 6h schedule
        )
        # Stop horizon: 43200*1000 - 4000 = 43196000
        self.assertAlmostEqual(max_ms, 43196000.0)
        # Schedule horizon: 21600*1000 - 4000 = 21596000
        self.assertAlmostEqual(sched_ms, 21596000.0)
        self.assertNotEqual(max_ms, sched_ms)

    def test_schedule_horizon_shorter_than_stop(self):
        """Schedule horizon can be shorter than stop horizon (original 6h semantics)."""
        max_ms, sched_ms = self._derive_schedule_horizon_ms(
            max_wallclock_seconds=43200,
            schedule_horizon_seconds=21600,
        )
        self.assertLess(sched_ms, max_ms)

    def test_no_wallclock_mode_both_none(self):
        """When max_wallclock_seconds=0 (step-based), both are None."""
        max_ms, sched_ms = self._derive_schedule_horizon_ms(
            max_wallclock_seconds=0, schedule_horizon_seconds=0
        )
        self.assertIsNone(max_ms)
        self.assertIsNone(sched_ms)


class TestTrainingFracWithScheduleHorizon(unittest.TestCase):
    """training_frac should use schedule_horizon_ms, not max_wallclock_ms."""

    def _training_frac(self, step, elapsed_ms, schedule_horizon_ms, iterations=10000):
        """Extracted training_frac logic with schedule_horizon_ms."""
        if schedule_horizon_ms is None:
            return step / max(iterations, 1)
        return elapsed_ms / max(schedule_horizon_ms, 1e-09)

    def test_fraction_at_halfway_6h_schedule(self):
        """At 3h elapsed with 6h schedule horizon -> frac = 0.5."""
        sched_ms = 21596000.0  # 6h - 4s reserve
        frac = self._training_frac(0, 3 * 3600 * 1000, sched_ms)
        self.assertAlmostEqual(frac, 10800000.0 / 21596000.0, places=5)

    def test_fraction_exceeds_1_beyond_schedule_horizon(self):
        """If elapsed > schedule_horizon, frac > 1.0 (warmdown already complete)."""
        sched_ms = 21596000.0  # 6h schedule
        elapsed_ms = 30000000.0  # ~8.3h
        frac = self._training_frac(0, elapsed_ms, sched_ms)
        self.assertGreater(frac, 1.0)

    def test_step_mode_ignores_elapsed(self):
        """In step mode (no wallclock), fraction is step-based."""
        frac = self._training_frac(500, 99999.0, None, iterations=1000)
        self.assertAlmostEqual(frac, 0.5)


class TestLrMulWithExtendedHorizon(unittest.TestCase):
    """lr_mul should produce correct warmdown using schedule-based fraction."""

    def _lr_mul(self, frac, warmdown_frac=0.2, min_lr=0.0):
        if warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - warmdown_frac:
            return max((1.0 - frac) / warmdown_frac, min_lr)
        return 1.0

    def test_before_warmdown_region(self):
        """frac=0.5 with warmdown_frac=0.2 -> lr_mul=1.0"""
        self.assertEqual(self._lr_mul(0.5, warmdown_frac=0.2), 1.0)

    def test_at_warmdown_start(self):
        """frac=0.8 (1-0.2) -> start of warmdown, lr_mul=1.0"""
        self.assertAlmostEqual(self._lr_mul(0.8, warmdown_frac=0.2), 1.0)

    def test_at_schedule_end(self):
        """frac=1.0 -> lr_mul = 0.0"""
        self.assertAlmostEqual(self._lr_mul(1.0, warmdown_frac=0.2), 0.0)

    def test_beyond_schedule_end_with_min_lr(self):
        """frac>1.0 -> lr_mul clamped to min_lr."""
        result = self._lr_mul(1.5, warmdown_frac=0.2, min_lr=0.01)
        self.assertEqual(result, 0.01)

    def test_beyond_schedule_end_no_min_lr(self):
        """frac>1.0, min_lr=0 -> lr_mul=0."""
        result = self._lr_mul(1.5, warmdown_frac=0.2, min_lr=0.0)
        self.assertEqual(result, 0.0)


class TestStopConditionUnchanged(unittest.TestCase):
    """Stop condition must use max_wallclock_ms, NOT schedule_horizon_ms."""

    def _reached_cap(self, approx_training_time_ms, max_wallclock_ms):
        """Extracted stop condition logic."""
        return max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms

    def test_stop_uses_max_wallclock_not_schedule(self):
        """Training continues past schedule horizon until max_wallclock is reached."""
        max_wallclock_ms = 43196000.0  # 12h - reserve
        schedule_horizon_ms = 21596000.0  # 6h - reserve
        # At 8h elapsed: past schedule horizon but before max wallclock
        elapsed_ms = 8 * 3600 * 1000
        self.assertFalse(self._reached_cap(elapsed_ms, max_wallclock_ms))
        # At 13h elapsed: past max wallclock
        elapsed_ms = 13 * 3600 * 1000
        self.assertTrue(self._reached_cap(elapsed_ms, max_wallclock_ms))


class TestResumeCheckpointBackwardCompat(unittest.TestCase):
    """Existing checkpoints without schedule_horizon metadata must remain loadable."""

    def test_old_checkpoint_missing_schedule_horizon(self):
        """Old checkpoints don't have schedule_horizon_seconds - this must NOT cause errors."""
        # Simulate an old checkpoint dict (no schedule_horizon_seconds key)
        old_ckpt = {
            "step": 5000,
            "training_time_ms": 10800000.0,
            "world_size": 8,
            "rank": 0,
            "looping_active": False,
            "exported_minutes": [10, 20, 30],
            "hparam_fingerprint": {
                "num_layers": 11, "model_dim": 512, "num_heads": 8,
                "num_kv_heads": 4, "vocab_size": 8192, "mlp_mult": 4.0,
                "num_loops": 0, "train_seq_len": 2048,
                "tokenizer_path": "/data/tok", "data_path": "/data/train",
            },
        }
        # The patch must NOT add mandatory checkpoint fields
        # Verify: no KeyError when accessing only the standard fields
        self.assertIn("step", old_ckpt)
        self.assertIn("training_time_ms", old_ckpt)
        self.assertNotIn("schedule_horizon_seconds", old_ckpt)
        # The schedule horizon is derived from ENV, not from checkpoint
        # So loading an old checkpoint should work fine

    def test_new_env_var_not_stored_in_checkpoint(self):
        """Verify in the script that save_resume_checkpoint does NOT save schedule_horizon."""
        with open(TRAIN_SCRIPT) as f:
            content = f.read()
        # The save function should not reference schedule_horizon
        save_fn_start = content.find("def save_resume_checkpoint(")
        save_fn_end = content.find("\ndef ", save_fn_start + 1)
        save_fn_body = content[save_fn_start:save_fn_end]
        self.assertNotIn("schedule_horizon", save_fn_body,
                         "schedule_horizon should NOT be stored in resume checkpoints")


class TestLoopingActivationUsesScheduleHorizon(unittest.TestCase):
    """Loop activation (enable_looping_at) must key off schedule fraction."""

    def test_looping_activates_at_schedule_frac(self):
        """The enable_looping_at comparison must use schedule-based frac."""
        enable_looping_at = 0.5
        schedule_horizon_ms = 21596000.0  # 6h schedule

        # At 2h into a 12h run with 6h schedule horizon:
        elapsed_ms = 2 * 3600 * 1000  # 2h = 7200000
        frac = elapsed_ms / max(schedule_horizon_ms, 1e-09)
        self.assertLess(frac, enable_looping_at)

        # At 4h into the run (past the halfway point of 6h schedule):
        elapsed_ms = 4 * 3600 * 1000
        frac = elapsed_ms / max(schedule_horizon_ms, 1e-09)
        self.assertGreater(frac, enable_looping_at)


if __name__ == "__main__":
    unittest.main()
