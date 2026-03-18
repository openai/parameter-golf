from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.code_mutations import planned_code_mutation, preview_code_mutation
from harness.common import canonical_code_mutation
from harness.log_parser import parse_log
from harness.planner import plan_next_experiment
from harness.preflight import run_preflight
from harness.runner import _build_command
from harness.runner import run_experiment


REPO_ROOT = Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    def test_parse_record_log(self) -> None:
        metrics = parse_log(REPO_ROOT / "records/track_10min_16mb/2026-03-17_NaiveBaseline/train.log")
        self.assertAlmostEqual(metrics["final_roundtrip_val_bpb"], 1.2243657)
        self.assertEqual(metrics["stop_step"], 13780)
        self.assertEqual(metrics["serialized_model_int8_zlib_bytes"], 15815847)

    def test_parse_nonrecord_log(self) -> None:
        metrics = parse_log(
            REPO_ROOT / "records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/train.log"
        )
        self.assertAlmostEqual(metrics["final_roundtrip_val_bpb"], 1.20737944)
        self.assertAlmostEqual(metrics["last_pre_quant_val_bpb"], 1.1749)
        self.assertTrue(metrics["metrics_valid"])

    def test_parse_non_log_source_is_not_metrics_valid(self) -> None:
        metrics = parse_log(REPO_ROOT / "harness" / "planner.py")
        self.assertFalse(metrics["metrics_valid"])
        self.assertIn("no_train_or_val_lines_detected", metrics["parse_warnings"])

    def test_parse_source_file_is_not_treated_as_valid_log(self) -> None:
        metrics = parse_log(REPO_ROOT / "train_gpt.py")
        self.assertFalse(metrics["metrics_valid"])
        self.assertIn("no_train_or_val_lines_detected", metrics["parse_warnings"])

    def test_planner_avoids_duplicate_profile_envs(self) -> None:
        history = [
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "env": {
                        "DATA_PATH": "/tmp/data",
                        "TOKENIZER_PATH": "/tmp/tok.model",
                        "VOCAB_SIZE": "1024",
                        "ITERATIONS": "200",
                        "TRAIN_BATCH_TOKENS": "8192",
                        "TRAIN_SEQ_LEN": "1024",
                        "VAL_LOSS_EVERY": "0",
                        "VAL_BATCH_SIZE": "8192",
                        "TRAIN_LOG_EVERY": "25",
                        "GRAD_ACCUM_STEPS": "8",
                        "OUT_DIR": ".",
                    },
                },
                "result": {"status": "failed"},
                "metrics": {},
            }
        ]
        spec = plan_next_experiment(history, "mlx_smoke")
        self.assertNotEqual(spec["env"], history[0]["spec"]["env"])

    def test_planner_ignores_completed_runs_without_planner_eligible_trust(self) -> None:
        history = [
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "experiment_id": "bad_completed_run",
                    "env": {"ITERATIONS": "999"},
                },
                "result": {
                    "status": "completed",
                    "run_state": "completed_invalid_metrics",
                    "planner_eligible": False,
                },
                "metrics": {
                    "metrics_valid": False,
                    "final_roundtrip_val_bpb": 1.0,
                },
            }
        ]
        spec = plan_next_experiment(history, "mlx_smoke")
        self.assertEqual(spec["mutation_name"], "baseline")
        self.assertIsNone(spec["parent_experiment_id"])

    def test_preflight_reports_subset_local_dataset(self) -> None:
        spec = plan_next_experiment([], "mlx_smoke")
        preflight = run_preflight(spec)
        self.assertTrue(preflight["can_launch"])
        self.assertEqual(preflight["comparability"], "subset-only")
        self.assertGreaterEqual(preflight["actual_train_shards"], 1)
        self.assertEqual(spec["hypothesis_family"], "baseline_capture")
        self.assertEqual(spec["hypothesis_stage"], "scout")
        self.assertEqual(spec["funnel_stage"], "scout")
        self.assertEqual(spec["promotion_target_stage"], "confirm")

    def test_record_profile_requires_challenge_ready(self) -> None:
        spec = plan_next_experiment([], "torch_record_8gpu")
        preflight = run_preflight(spec)
        self.assertTrue(spec["require_challenge_ready"])
        self.assertFalse(preflight["challenge_ready"])
        self.assertFalse(preflight["launch_policy_ok"])
        self.assertFalse(preflight["ready_for_execution"])

    def test_torch_command_uses_preferred_python(self) -> None:
        spec = plan_next_experiment([], "torch_record_8gpu")
        command = _build_command(spec)
        self.assertEqual(command[0], str(REPO_ROOT / ".venv" / "bin" / "python"))
        self.assertEqual(command[1:3], ["-m", "torch.distributed.run"])

    def test_preview_code_mutation_changes_quantization_policy(self) -> None:
        mutation = planned_code_mutation(REPO_ROOT / "train_gpt.py", "quant_scale_fp32")
        preview = preview_code_mutation(REPO_ROOT / "train_gpt.py", mutation)
        self.assertIn("INT8_PER_ROW_SCALE_DTYPE = torch.float32", preview["mutated_text"])
        self.assertEqual(preview["code_mutation"]["name"], "quant_scale_fp32")
        self.assertGreaterEqual(len(preview["code_mutation"]["applied"]), 1)
        self.assertEqual(preview["code_mutation"]["source_hash"], mutation["source_hash"])

    def test_code_mutation_signature_includes_source_hash(self) -> None:
        mutation = planned_code_mutation(REPO_ROOT / "train_gpt.py", "quant_scale_fp32")
        changed_source = dict(mutation)
        changed_source["source_hash"] = "different_source_hash"
        changed_source["signature"] = canonical_code_mutation(changed_source)
        self.assertNotEqual(mutation["signature"], changed_source["signature"])

    def test_planner_dedupes_same_code_mutation_signature(self) -> None:
        prior = plan_next_experiment([], "mlx_smoke", manual_code_mutation="quant_clip_tighter")
        history = [
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "env": prior["env"],
                    "code_mutation": prior["code_mutation"],
                },
                "result": {"status": "completed"},
                "metrics": {
                    "final_roundtrip_val_bpb": 1.3,
                    "quant_gap_bpb": 0.03,
                },
            }
        ]
        spec = plan_next_experiment(history, "mlx_smoke")
        self.assertNotEqual((spec.get("code_mutation") or {}).get("name"), "quant_clip_tighter")

    def test_blocked_run_does_not_consume_candidate_signature(self) -> None:
        eligible_anchor = plan_next_experiment([], "mlx_smoke")
        history = [
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "env": eligible_anchor["env"],
                    "code_mutation": None,
                },
                "result": {
                    "status": "completed",
                    "run_state": "completed",
                    "planner_eligible": True,
                },
                "metrics": {
                    "metrics_valid": True,
                    "final_roundtrip_val_bpb": 1.3,
                    "last_pre_quant_val_bpb": 1.27,
                    "quant_gap_bpb": 0.03,
                },
            },
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "env": eligible_anchor["env"],
                    "code_mutation": planned_code_mutation(REPO_ROOT / "train_gpt_mlx.py", "quant_clip_tighter"),
                },
                "result": {
                    "status": "blocked",
                    "run_state": "blocked",
                    "planner_eligible": False,
                },
                "metrics": {"metrics_valid": False},
            },
        ]
        spec = plan_next_experiment(history, "mlx_smoke")
        self.assertEqual((spec.get("code_mutation") or {}).get("name"), "quant_clip_tighter")

    def test_planner_carries_forward_successful_parent_code_mutation_for_env_followups(self) -> None:
        prior = plan_next_experiment([], "mlx_smoke", manual_code_mutation="quant_clip_tighter")
        history = [
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "experiment_id": "good_code_parent",
                    "env": prior["env"],
                    "code_mutation": prior["code_mutation"],
                    "hypothesis_id": prior["hypothesis_id"],
                    "lineage_id": prior["lineage_id"],
                },
                "result": {
                    "status": "completed",
                    "run_state": "completed",
                    "planner_eligible": True,
                },
                "metrics": {
                    "metrics_valid": True,
                    "final_roundtrip_val_bpb": 1.25,
                    "last_pre_quant_val_bpb": 1.24,
                    "quant_gap_bpb": 0.01,
                },
            }
        ]
        spec = plan_next_experiment(history, "mlx_smoke")
        self.assertEqual(spec["mutation_kind"], "env")
        self.assertTrue(spec["inherits_parent_code_mutation"])
        self.assertEqual((spec.get("code_mutation") or {}).get("name"), "quant_clip_tighter")

    def test_winner_run_triggers_confirm_followup(self) -> None:
        baseline = plan_next_experiment([], "mlx_smoke")
        winner = plan_next_experiment([], "mlx_smoke", manual_code_mutation="quant_clip_tighter")
        history = [
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "experiment_id": "baseline_anchor",
                    "env": baseline["env"],
                    "code_mutation": baseline["code_mutation"],
                    "hypothesis_id": baseline["hypothesis_id"],
                    "lineage_id": baseline["lineage_id"],
                },
                "result": {
                    "status": "completed",
                    "run_state": "completed",
                    "planner_eligible": True,
                },
                "metrics": {
                    "metrics_valid": True,
                    "final_roundtrip_val_bpb": 1.24,
                    "last_pre_quant_val_bpb": 1.23,
                    "quant_gap_bpb": 0.02,
                },
            },
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "experiment_id": "winner_run",
                    "parent_experiment_id": "baseline_anchor",
                    "env": winner["env"],
                    "code_mutation": winner["code_mutation"],
                    "mutation_name": winner["mutation_name"],
                    "mutation_kind": winner["mutation_kind"],
                    "objective": winner["objective"],
                    "hypothesis_id": winner["hypothesis_id"],
                    "lineage_id": winner["lineage_id"],
                },
                "result": {
                    "status": "completed",
                    "run_state": "completed",
                    "planner_eligible": True,
                },
                "metrics": {
                    "metrics_valid": True,
                    "final_roundtrip_val_bpb": 1.2365,
                    "last_pre_quant_val_bpb": 1.232,
                    "quant_gap_bpb": 0.016,
                },
            },
        ]
        spec = plan_next_experiment(history, "mlx_smoke")
        self.assertEqual(spec["followup_reason"], "confirm_winner")
        self.assertEqual(spec["followup_of_experiment_id"], "winner_run")
        self.assertEqual(spec["funnel_stage"], "confirm")

    def test_confirmed_winner_triggers_one_neighbor_probe(self) -> None:
        baseline = plan_next_experiment([], "mlx_smoke")
        winner = plan_next_experiment([], "mlx_smoke", manual_code_mutation="quant_clip_tighter")
        history = [
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "experiment_id": "baseline_anchor",
                    "env": baseline["env"],
                    "code_mutation": baseline["code_mutation"],
                    "hypothesis_id": baseline["hypothesis_id"],
                    "lineage_id": baseline["lineage_id"],
                },
                "result": {"status": "completed", "run_state": "completed", "planner_eligible": True},
                "metrics": {
                    "metrics_valid": True,
                    "final_roundtrip_val_bpb": 1.24,
                    "last_pre_quant_val_bpb": 1.23,
                    "quant_gap_bpb": 0.02,
                },
            },
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "experiment_id": "winner_run",
                    "parent_experiment_id": "baseline_anchor",
                    "env": winner["env"],
                    "code_mutation": winner["code_mutation"],
                    "mutation_name": winner["mutation_name"],
                    "mutation_kind": winner["mutation_kind"],
                    "objective": winner["objective"],
                    "hypothesis_id": winner["hypothesis_id"],
                    "lineage_id": winner["lineage_id"],
                },
                "result": {"status": "completed", "run_state": "completed", "planner_eligible": True},
                "metrics": {
                    "metrics_valid": True,
                    "final_roundtrip_val_bpb": 1.2365,
                    "last_pre_quant_val_bpb": 1.232,
                    "quant_gap_bpb": 0.016,
                },
            },
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "experiment_id": "winner_confirm",
                    "parent_experiment_id": "winner_run",
                    "followup_reason": "confirm_winner",
                    "env": winner["env"],
                    "code_mutation": winner["code_mutation"],
                    "mutation_name": winner["mutation_name"],
                    "mutation_kind": winner["mutation_kind"],
                    "objective": winner["objective"],
                    "hypothesis_id": f"{winner['hypothesis_id']}::confirm",
                    "lineage_id": winner["lineage_id"],
                },
                "result": {"status": "completed", "run_state": "completed", "planner_eligible": True},
                "metrics": {
                    "metrics_valid": True,
                    "final_roundtrip_val_bpb": 1.234,
                    "last_pre_quant_val_bpb": 1.231,
                    "quant_gap_bpb": 0.015,
                },
            },
        ]
        spec = plan_next_experiment(history, "mlx_smoke")
        self.assertEqual(spec["followup_reason"], "neighbor_probe")
        self.assertEqual(spec["followup_of_experiment_id"], "winner_confirm")
        self.assertEqual((spec.get("code_mutation") or {}).get("name"), "quant_clip_tighter")

    def test_blocked_run_does_not_trigger_confirm_followup(self) -> None:
        baseline = plan_next_experiment([], "mlx_smoke")
        winner = plan_next_experiment([], "mlx_smoke", manual_code_mutation="quant_clip_tighter")
        history = [
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "experiment_id": "baseline_anchor",
                    "env": baseline["env"],
                    "code_mutation": baseline["code_mutation"],
                    "hypothesis_id": baseline["hypothesis_id"],
                    "lineage_id": baseline["lineage_id"],
                },
                "result": {"status": "completed", "run_state": "completed", "planner_eligible": True},
                "metrics": {
                    "metrics_valid": True,
                    "final_roundtrip_val_bpb": 1.24,
                    "last_pre_quant_val_bpb": 1.23,
                    "quant_gap_bpb": 0.02,
                },
            },
            {
                "spec": {
                    "profile": "mlx_smoke",
                    "experiment_id": "blocked_winner_like_run",
                    "parent_experiment_id": "baseline_anchor",
                    "env": winner["env"],
                    "code_mutation": winner["code_mutation"],
                    "mutation_name": winner["mutation_name"],
                    "mutation_kind": winner["mutation_kind"],
                    "objective": winner["objective"],
                    "hypothesis_id": winner["hypothesis_id"],
                    "lineage_id": winner["lineage_id"],
                },
                "result": {"status": "blocked", "run_state": "blocked", "planner_eligible": False},
                "metrics": {
                    "metrics_valid": False,
                    "final_roundtrip_val_bpb": 1.236,
                    "quant_gap_bpb": 0.015,
                },
            },
        ]
        spec = plan_next_experiment(history, "mlx_smoke")
        self.assertNotEqual(spec.get("followup_reason"), "confirm_winner")

    def test_dry_run_materializes_mutated_script(self) -> None:
        spec = plan_next_experiment([], "mlx_smoke", manual_code_mutation="quant_clip_tighter")
        spec["experiment_id"] = "test_code_mutation_dry_run"
        record = run_experiment(spec, dry_run=True)
        materialized = Path(record["result"]["materialized_script_path"])
        self.assertTrue(materialized.is_file())
        self.assertIn("INT8_CLIP_PERCENTILE = 99.99992", materialized.read_text(encoding="utf-8"))
        self.assertIn(str(materialized), record["result"]["command"])
        self.assertEqual(record["metrics"]["code_bytes_delta"], 0)
        self.assertFalse(record["result"]["planner_eligible"])
        self.assertEqual(record["result"]["run_state"], "dry_run")

    def test_invalid_code_mutation_blocks_instead_of_raising(self) -> None:
        spec = plan_next_experiment([], "mlx_smoke")
        with tempfile.TemporaryDirectory() as tmpdir:
            broken_script = Path(tmpdir) / "broken_train_gpt_mlx.py"
            broken_script.write_text("print('hello')\n", encoding="utf-8")
            spec["script"] = str(broken_script)
            spec["code_mutation"] = {
                "name": "quant_clip_tighter",
                "family": "quantization",
                "params": {"int8_clip_percentile": 99.99992},
                "script_basename": broken_script.name,
                "source_hash": "placeholder",
            }
            record = run_experiment(spec, dry_run=False)
        self.assertEqual(record["result"]["run_state"], "blocked")
        self.assertEqual(record["result"]["failure_reason"], "preflight_failed")
        self.assertFalse(record["result"]["planner_eligible"])

    def test_planner_adds_hypothesis_metadata_for_manual_code_mutation(self) -> None:
        spec = plan_next_experiment([], "mlx_smoke", manual_code_mutation="quant_clip_tighter")
        self.assertEqual(spec["hypothesis_family"], "quant_gap_reduction")
        self.assertEqual(spec["risk_level"], "medium")
        self.assertIn("final roundtrip quality", spec["expected_upside"])
        self.assertTrue(spec["hypothesis_id"].startswith(spec["lineage_id"]))

    def test_planner_adds_hypothesis_metadata(self) -> None:
        spec = plan_next_experiment([], "mlx_smoke", manual_code_mutation="quant_clip_tighter")
        self.assertEqual(spec["hypothesis_family"], "quant_gap_reduction")
        self.assertIn("final roundtrip quality", spec["expected_upside"])
        self.assertIn("final roundtrip improves", spec["promotion_rule"])
        self.assertTrue(spec["hypothesis_id"].startswith(spec["lineage_id"]))


if __name__ == "__main__":
    unittest.main()
