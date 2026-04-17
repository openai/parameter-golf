from __future__ import annotations

import unittest

from auto_tune_launcher import (
    HardwareInfo,
    build_batch_candidates,
    build_signature,
    classify_failure,
    compile_ladder,
    dedupe_preserve_order,
    signature_key,
)


class AutoTuneLauncherTests(unittest.TestCase):
    def test_signature_is_stable(self) -> None:
        hw = HardwareInfo(
            gpu_names=["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090"],
            gpu_count=2,
            min_free_mb=24126,
            min_total_mb=24576,
            driver_version="590.48.01",
            cuda_version="13.1",
            nproc=2,
        )
        env = {
            "NUM_LAYERS": "11",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "TRAIN_SEQ_LEN": "1024",
        }
        sig1 = build_signature(hw, "competition", env)
        sig2 = build_signature(hw, "competition", dict(env))
        self.assertEqual(sig1, sig2)
        self.assertEqual(signature_key(sig1), signature_key(sig2))

    def test_batch_candidates_preserve_repo_backoff_values(self) -> None:
        env = {}
        values = build_batch_candidates(49152, env, "competition")
        self.assertEqual(values[0], 49152)
        self.assertIn(16000, values)
        self.assertIn(10128, values)
        self.assertIn(8192, values)

    def test_batch_candidates_respect_explicit_override(self) -> None:
        env = {"AUTO_TUNE_BATCH_CANDIDATES": "16000,10128,8192"}
        self.assertEqual(build_batch_candidates(49152, env, "competition"), [16000, 10128, 8192])

    def test_classify_oom(self) -> None:
        failure, reason = classify_failure(
            "[rank0]: torch.OutOfMemoryError: CUDA out of memory",
            timed_out=False,
            step_lines=0,
            stage="train",
        )
        self.assertEqual(failure, "oom")
        self.assertTrue("OutOfMemoryError" in reason or "CUDA out of memory" in reason)

    def test_classify_compile_backend_assert(self) -> None:
        failure, reason = classify_failure(
            "AssertionError: cannot extract sympy expressions from {'OUT_ptr': ...}",
            timed_out=False,
            step_lines=0,
            stage="train",
        )
        self.assertEqual(failure, "compile_backend")
        self.assertEqual(reason, "cannot extract sympy expressions")

    def test_classify_ddp_unused_grad(self) -> None:
        failure, reason = classify_failure(
            "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.",
            timed_out=False,
            step_lines=0,
            stage="train",
        )
        self.assertEqual(failure, "ddp_unused_grad")
        self.assertEqual(reason, "ddp_unused_grad")

    def test_classify_timeout_compile_bound(self) -> None:
        failure, reason = classify_failure(
            "",
            timed_out=True,
            step_lines=0,
            stage="train",
        )
        self.assertEqual(failure, "timeout_compile_bound")
        self.assertIn("timeout", reason)

    def test_compile_ladder_diagnostic_is_eager_only(self) -> None:
        self.assertEqual(compile_ladder("diagnostic"), [("none", "full", 0)])

    def test_compile_ladder_competition_keeps_max_autotune(self) -> None:
        ladder = compile_ladder("competition")
        self.assertEqual(ladder[0], ("max-autotune", "blocks", 4))
        self.assertEqual(ladder[-1], ("none", "full", 0))

    def test_dedupe_preserve_order(self) -> None:
        self.assertEqual(dedupe_preserve_order([4, 2, 4, 1, 2]), [4, 2, 1])


if __name__ == "__main__":
    unittest.main()
