from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from search.log_parser import parse_run_log


class LogParserTests(unittest.TestCase):
    def _parse(self, text: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.log"
            path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
            return parse_run_log(path)

    def test_parses_quantized_run(self):
        parsed = self._parse(
            """
            step:250/500 val_loss:3.1000 val_bpb:1.8200 train_time:50000ms step_avg:200.00ms
            step:500/500 val_loss:2.9000 val_bpb:1.7000 train_time:100000ms step_avg:200.00ms
            roundtrip_int8+zstd22 val_loss:2.9010 val_bpb:1.7010 eval_time:1000ms
            roundtrip_int6+zstd22 val_loss:2.9050 val_bpb:1.7050 eval_time:1000ms
            quant_summary int8_bpb:1.7010 int6_bpb:1.7050 int8_sz:15000000 int6_sz:14500000
            """
        )
        self.assertEqual(parsed.status, "completed")
        self.assertEqual(parsed.terminal_validation.step, 500)
        self.assertAlmostEqual(parsed.roundtrip_int6.val_bpb, 1.7050)
        self.assertEqual(parsed.int6_artifact_bytes, 14_500_000)

    def test_marks_missing_int6_as_failure(self):
        parsed = self._parse(
            """
            step:1000/1000 val_loss:2.6656 val_bpb:1.5787 train_time:182312ms step_avg:182.31ms
            final_int8_zlib_roundtrip val_loss:2.6671 val_bpb:1.5796 eval_time:85635ms
            """
        )
        self.assertEqual(parsed.status, "failed")
        self.assertIn("missing final int6 metric", parsed.failure_reason)

    def test_marks_oversize_run(self):
        parsed = self._parse(
            """
            step:5000/5000 val_loss:2.3537 val_bpb:1.3940 train_time:1022947ms step_avg:204.59ms
            roundtrip_int6+zstd22 val_loss:2.3571 val_bpb:1.3960 eval_time:99807ms
            quant_summary int8_bpb:1.3941 int6_bpb:1.3960 int8_sz:19903925 int6_sz:16500000
            """
        )
        self.assertEqual(parsed.status, "oversize")
        self.assertTrue(parsed.oversize_int6)

    def test_parses_sliding_window_metric_and_prefers_it_as_objective(self):
        parsed = self._parse(
            """
            step:1000/1000 val_loss:2.6656 val_bpb:1.5787 train_time:182312ms step_avg:182.31ms
            roundtrip_int6+zstd22 val_loss:2.6671 val_bpb:1.5796 eval_time:85635ms
            quant_summary int8_bpb:1.5789 int6_bpb:1.5796 int8_sz:14500000 int6_sz:14000000
            sliding_window val_loss:2.6500 val_bpb:1.5701 seq_len:2048 stride:256 eval_time:99999ms
            """
        )
        self.assertIsNotNone(parsed.sliding_window_int6)
        self.assertAlmostEqual(parsed.sliding_window_int6.val_bpb, 1.5701)
        self.assertEqual(parsed.sliding_window_int6.stride, 256)
        self.assertEqual(parsed.objective_source(), "sliding_window_int6")
        self.assertAlmostEqual(parsed.objective_bpb(), 1.5701)


if __name__ == "__main__":
    unittest.main()
