from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from search.run_search import build_run_observations, resolve_run_log_path
from search.log_parser import ParsedRunLog, QuantizedMetric, SlidingWindowMetric, StepValidation
from search.runner import PreparedRun


class RunSearchTests(unittest.TestCase):
    def test_final_observation_uses_sliding_window_objective_when_present(self):
        parsed = ParsedRunLog(
            log_path="/tmp/run.log",
            step_validations=(
                StepValidation(
                    step=500,
                    total_steps=500,
                    val_loss=2.90,
                    val_bpb=1.70,
                    train_time_ms=100_000,
                ),
            ),
            terminal_validation=StepValidation(
                step=500,
                total_steps=500,
                val_loss=2.90,
                val_bpb=1.70,
                train_time_ms=100_000,
            ),
            roundtrip_int6=QuantizedMetric(
                label="roundtrip_int6+zstd22",
                bits=6,
                val_loss=2.905,
                val_bpb=1.705,
                eval_time_ms=1_000,
            ),
            roundtrip_int8=None,
            sliding_window_int6=SlidingWindowMetric(
                label="sliding_window_int6",
                val_loss=2.89,
                val_bpb=1.69,
                seq_len=2048,
                stride=256,
                eval_time_ms=10_000,
            ),
            int6_artifact_bytes=14_500_000,
            int8_artifact_bytes=None,
            has_nan=False,
            stopped_early=False,
            oversize_int6=False,
            status="completed",
            failure_reason=None,
        )
        observations = build_run_observations(
            parsed,
            params={"ITERATIONS": 500},
            run_id="demo_0000",
            run_index=0,
            prior_runs=[],
            default_proxy_int6_gap=0.01,
        )
        self.assertEqual(observations[-1]["source"], "sliding_window_int6")
        self.assertAlmostEqual(observations[-1]["target_bpb"], 1.69)
        self.assertAlmostEqual(observations[-1]["score"], -1.69)

    def test_resolve_run_log_path_falls_back_to_workdir_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            fallback = base / "logs" / "demo_0000.txt"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text("ok\n", encoding="utf-8")
            prepared = PreparedRun(
                run_id="demo_0000",
                run_env={},
                argv=["/bin/true"],
                display_command="true",
                log_path=base / "artifacts" / "workload_logs" / "demo_0000.txt",
                stdout_path=base / "stdout" / "demo_0000.stdout",
                status_path=base / "run_status" / "demo_0000.json",
            )
            resolved = resolve_run_log_path(prepared, workdir=base)
            self.assertEqual(resolved, fallback.resolve())


if __name__ == "__main__":
    unittest.main()
