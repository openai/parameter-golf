from __future__ import annotations

import unittest

from research.submission_metrics import (
    canonical_submission_eval,
    canonical_submission_fields,
    canonical_submission_fields_for_status,
    metric_payload_by_label,
)


class SubmissionMetricsTest(unittest.TestCase):
    def test_training_best_only_falls_back_to_last_val(self) -> None:
        metrics = {
            "last_val": {
                "val_loss": 1.234,
                "val_bpb": 2.2145,
            },
        }
        label, payload = canonical_submission_eval(metrics)
        self.assertEqual(label, "last_val")
        self.assertEqual(payload, metrics["last_val"])
        self.assertEqual(
            canonical_submission_fields(metrics),
            {
                "final_submission_metric_label": "last_val",
                "official_submission_metric_label": "last_val",
                "final_submission_loss": 1.234,
                "final_submission_bpb": 2.2145,
            },
        )

    def test_final_sliding_window_exact_beats_other_exact_fallbacks(self) -> None:
        metrics = {
            "named_evals_exact": {
                "final_int6_roundtrip": {"val_loss": 1.5, "val_bpb": 1.20},
                "final_int6_sliding_window": {"val_loss": 1.2, "val_bpb": 0.94086717},
                "final_int6_sliding_window_s64": {"val_loss": 1.1, "val_bpb": 0.9300},
            },
        }
        label, payload = canonical_submission_eval(metrics)
        self.assertEqual(label, "final_int6_sliding_window_exact")
        self.assertEqual(payload, metrics["named_evals_exact"]["final_int6_sliding_window"])

    def test_legal_ttt_exact_is_canonical_when_present(self) -> None:
        metrics = {
            "named_evals": {
                "legal_ttt": {"val_loss": 1.0, "val_bpb": 0.77418915, "eval_time_ms": 1234},
            },
            "named_evals_exact": {
                "final_int6_sliding_window": {"val_loss": 1.2, "val_bpb": 0.94086717},
                "legal_ttt": {"val_loss": 1.0, "val_bpb": 0.77418915},
            },
        }
        label, payload = canonical_submission_eval(metrics)
        self.assertEqual(label, "legal_ttt_exact")
        self.assertEqual(payload["val_loss"], metrics["named_evals_exact"]["legal_ttt"]["val_loss"])
        self.assertEqual(payload["val_bpb"], metrics["named_evals_exact"]["legal_ttt"]["val_bpb"])
        self.assertEqual(payload["eval_time_ms"], 1234)
        self.assertEqual(
            canonical_submission_fields(metrics),
            {
                "final_submission_metric_label": "legal_ttt_exact",
                "official_submission_metric_label": "legal_ttt_exact",
                "final_submission_loss": 1.0,
                "final_submission_bpb": 0.77418915,
            },
        )
        merged_payload = metric_payload_by_label(metrics, "legal_ttt_exact")
        self.assertEqual(merged_payload["eval_time_ms"], 1234)

    def test_track_aware_prequant_metric_preferred_when_requested(self) -> None:
        metrics = {
            "named_evals_exact": {
                "prequant_ttt": {"val_loss": 0.9, "val_bpb": 0.77},
                "legal_ttt": {"val_loss": 1.0, "val_bpb": 0.78},
            },
        }
        label, payload = canonical_submission_eval(metrics, track="prequant_ttt")
        self.assertEqual(label, "prequant_ttt_exact")
        self.assertEqual(payload["val_bpb"], 0.77)
        self.assertEqual(
            canonical_submission_fields(metrics, track="prequant_ttt"),
            {
                "final_submission_metric_label": "prequant_ttt_exact",
                "official_submission_metric_label": "prequant_ttt_exact",
                "final_submission_loss": 0.9,
                "final_submission_bpb": 0.77,
            },
        )

    def test_incomplete_run_has_no_official_submission_metric(self) -> None:
        metrics = {
            "named_evals_exact": {
                "final_int6_roundtrip": {"val_loss": 1.5, "val_bpb": 1.20},
            },
        }
        self.assertEqual(
            canonical_submission_fields_for_status(metrics, status="failed"),
            {
                "final_submission_metric_label": None,
                "official_submission_metric_label": None,
                "final_submission_loss": None,
                "final_submission_bpb": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
