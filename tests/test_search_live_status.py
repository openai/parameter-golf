from __future__ import annotations

import unittest
from pathlib import Path

from search.run_search import initialize_live_status, update_live_status_from_line
from search.runner import PreparedRun


class LiveStatusTests(unittest.TestCase):
    def _prepared(self) -> PreparedRun:
        base = Path("/tmp/parameter-golf-tests")
        return PreparedRun(
            run_id="demo_0000",
            run_env={},
            argv=["/bin/true"],
            display_command="true",
            log_path=base / "logs" / "demo_0000.txt",
            stdout_path=base / "stdout" / "demo_0000.stdout",
            status_path=base / "run_status" / "demo_0000.json",
        )

    def test_tracks_training_quant_and_sliding_progress(self):
        status = initialize_live_status(
            prepared=self._prepared(),
            run_index=0,
            params={"ITERATIONS": 2500},
            suggestion_info={},
        )

        update_live_status_from_line(
            status,
            "step:1500/2500 val_loss:2.7429 val_bpb:1.6245 train_time:311069ms step_avg:207.38ms",
        )
        self.assertEqual(status["phase"], "training")
        self.assertEqual(status["latest_step_validation"]["step"], 1500)
        self.assertAlmostEqual(status["latest_step_validation"]["val_bpb"], 1.6245)

        update_live_status_from_line(
            status,
            "roundtrip_int6+zstd22 val_loss:2.4093 val_bpb:1.4269 eval_time:100002ms",
        )
        self.assertEqual(status["phase"], "quant_eval")
        self.assertAlmostEqual(status["latest_roundtrip"]["int6+zstd22"]["val_bpb"], 1.4269)

        update_live_status_from_line(
            status,
            "sliding_window:start total_windows:242265 rank0_windows:242265 world_size:1 seq_len:2048 stride:256 log_every:2048",
        )
        self.assertEqual(status["phase"], "sliding_eval")
        self.assertEqual(status["sliding_window"]["total_windows"], 242265)
        self.assertEqual(status["sliding_window"]["log_every"], 2048)

        update_live_status_from_line(
            status,
            "sliding_window:progress rank0_windows:8192/242265 approx_global_windows:8192/242265 approx_pct:3.4 scored_tokens:2097152 elapsed:74.0s windows_per_sec:110.7 scored_tokens_per_sec:28339.9",
        )
        progress = status["sliding_window"]["progress"]
        self.assertEqual(status["phase"], "sliding_eval")
        self.assertEqual(progress["rank0_windows_done"], 8192)
        self.assertIn("eta_s", progress)

        update_live_status_from_line(
            status,
            "sliding_window val_loss:4.1525 val_bpb:2.4594 seq_len:2048 stride:256 eval_time:2215161ms",
        )
        self.assertEqual(status["phase"], "completed")
        self.assertAlmostEqual(status["sliding_window"]["final_metric"]["val_bpb"], 2.4594)


if __name__ == "__main__":
    unittest.main()
