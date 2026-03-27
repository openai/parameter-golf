from __future__ import annotations

import unittest

from research.submission_readiness import (
    build_legality_note,
    build_submission_readiness_report,
    sync_result_submission_fields,
    sync_summary_submission_fields,
)


class SubmissionReadinessTest(unittest.TestCase):
    def test_sync_summary_promotes_canonical_metric_and_preserves_training_best(self) -> None:
        result = sync_result_submission_fields(
            {
                "metrics": {
                    "named_evals_exact": {
                        "final_int6_sliding_window": {"val_loss": 1.2, "val_bpb": 0.94086717},
                        "legal_ttt": {"val_loss": 1.0, "val_bpb": 0.77418915},
                    },
                },
            }
        )
        summary = sync_summary_submission_fields(
            {
                "best_val_loss": 2.0,
                "best_val_bpb": 2.2145,
                "status": "completed",
            },
            result,
        )
        self.assertEqual(summary["training_best_val_bpb"], 2.2145)
        self.assertEqual(summary["final_submission_metric_label"], "legal_ttt_exact")
        self.assertEqual(summary["best_val_bpb"], 0.77418915)

    def test_readiness_report_and_legality_note_use_official_metric(self) -> None:
        result = sync_result_submission_fields(
            {
                "status": "completed",
                "preset": "sota_plus_ppm_entropy_fixed",
                "git_commit": "deadbeef",
                "submission_budget_estimate_bytes": 4_794_942,
                "metrics": {
                    "named_evals_exact": {
                        "legal_ttt": {"val_loss": 1.0, "val_bpb": 0.77418915},
                    },
                },
            }
        )
        note, _text = build_legality_note(result)
        report = build_submission_readiness_report(
            result,
            {
                "best_val_bpb": 0.77418915,
                "final_submission_metric_label": "legal_ttt_exact",
                "final_submission_bpb": 0.77418915,
                "train_time_ms": 300_000,
                "max_wallclock_seconds": 600.0,
            },
            note,
            {
                "artifact_bytes_measured": 4_794_942,
                "code_bytes_measured": 117_470,
                "exported_bytes_measured": 4_677_472,
            },
        )
        self.assertEqual(note["official_submission_metric_label"], "legal_ttt_exact")
        self.assertEqual(report["completion_status"], "completed")
        self.assertEqual(report["git_commit"], "deadbeef")
        self.assertEqual(report["final_submission_bpb"], 0.77418915)
        self.assertEqual(report["exported_model_bytes"], 4_677_472)
        self.assertEqual(report["code_bytes"], 117_470)
        self.assertEqual(report["artifact_bytes"], 4_794_942)
        self.assertEqual(report["total_artifact_bytes"], 4_794_942)
        self.assertEqual(report["official_submission_metric_label"], "legal_ttt_exact")
        self.assertTrue(report["byte_budget_constraint_appears_satisfied"])
        self.assertTrue(report["wall_clock_constraint_appears_satisfied"])
        self.assertTrue(report["consistent"])


if __name__ == "__main__":
    unittest.main()
