from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_evolutionary_matrix import build_matrix, render_shell, select_runs, shared_model_flags
from tools.run_evolutionary_matrix import build_queue_plan, detect_gpu_slots
from tools.summarize_evolutionary_runs import summarize_results


class EvolutionaryMatrixToolTests(unittest.TestCase):
    def test_matrix_contains_multiple_stages_and_filters(self) -> None:
        runs = build_matrix()
        self.assertGreaterEqual(len(runs), 40)
        self.assertTrue(any(run.stage == "5-committee-frontier" for run in runs))

        throughput_core = select_runs(runs, stages=("0-throughput",), include_tags=("core",))
        self.assertTrue(any(run.name == "throughput_fp16_standard" for run in throughput_core))
        self.assertTrue(all(run.stage == "0-throughput" for run in throughput_core))

        deep_no_xsa = select_runs(runs, include_tags=("deep",), exclude_tags=("xsa",))
        self.assertTrue(deep_no_xsa)
        self.assertTrue(all("xsa" not in run.tags for run in deep_no_xsa))

    def test_render_shell_includes_selected_runs(self) -> None:
        runs = select_runs(build_matrix(), names=("throughput_fp16_standard", "viability_full_all"))
        shell = render_shell(
            runs,
            python_bin="python3.11",
            script_path="tools/evolutionary_benchmark.py",
            output_dir="runs/evolutionary",
            enwik8_path="/tmp/enwik8",
        )

        self.assertIn("#!/usr/bin/env bash", shell)
        self.assertIn("throughput_fp16_standard", shell)
        self.assertIn("viability_full_all", shell)
        self.assertIn("--output-json runs/evolutionary/throughput_fp16_standard.json", shell)

    def test_shared_model_flags_expose_qk_gain(self) -> None:
        flags = shared_model_flags(qk_gain_init=4.0, spine_variant="xsa", xsa_last_n=2)

        self.assertIn("--qk-gain-init", flags)
        self.assertIn("4.0", flags)
        self.assertIn("--xsa-last-n", flags)

    def test_build_queue_plan_and_gpu_slot_parsing(self) -> None:
        runs = select_runs(build_matrix(), names=("throughput_fp16_standard", "viability_full_all"))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            log_dir = Path(tmpdir) / "logs"
            plan = build_queue_plan(runs=runs, output_dir=output_dir, log_dir=log_dir)

            self.assertEqual(len(plan), 2)
            self.assertTrue(str(plan[0].output_path).endswith("throughput_fp16_standard.json"))
            self.assertTrue(str(plan[1].log_path).endswith("viability_full_all.log"))

        self.assertEqual(detect_gpu_slots("0,2,5"), ("0", "2", "5"))

    def test_summarize_results_flattens_all_sections(self) -> None:
        throughput = {
            "config": {"dtype": "fp16", "spine_variant": "plain"},
            "model_param_mb_estimate": 15.8,
            "results": [
                {
                    "population_size": 64,
                    "status": "ok",
                    "batch_eval_s": 0.25,
                    "models_per_second": 256.0,
                    "estimated_population_param_mb": 1024.0,
                    "cuda_memory_after": {"max_reserved_bytes": float(8 * 1024**3)},
                }
            ],
        }
        viability = {
            "member_train_mode": "parallel_vmap",
            "members": [
                {"train": {"train_seconds_target": 30.0}, "eval": {"bpb": 1.30}},
                {"train": {"train_seconds_target": 30.0}, "eval": {"bpb": 1.28}},
            ],
            "member_ensemble_results": [
                {"topk": 2, "val_bpb": 1.20, "test_bpb": 1.22},
            ],
            "crossover": {
                "delta_overlap": {
                    "50": {
                        "summary": {
                            "num_trials": 28.0,
                            "fraction_between_parents": 0.75,
                            "fraction_collapse": 0.10,
                            "fraction_improves_best_parent": 0.05,
                        }
                    }
                }
            },
        }
        evolution = {
            "base": {"val": {"bpb": 1.25}},
            "evolution": {
                "crossover_strategy": "delta_overlap",
                "crossover_percentile": 50.0,
                "population_size": 16,
                "generations": 8,
                "history": [{"best_bpb": 1.21}, {"best_bpb": 1.19}],
                "final_best_val_bpb": 1.18,
                "final_best_test_bpb": 1.20,
                "ensemble_results": [
                    {"topk": 4, "val_bpb": 1.16, "test_bpb": 1.18},
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "throughput.json").write_text(json.dumps(throughput), encoding="utf-8")
            (root / "viability.json").write_text(json.dumps(viability), encoding="utf-8")
            (root / "evolution.json").write_text(json.dumps(evolution), encoding="utf-8")

            summary = summarize_results([str(root)])

        self.assertEqual(len(summary["throughput"]), 1)
        self.assertEqual(summary["throughput"][0]["population"], 64)
        self.assertEqual(summary["viability"][0]["strategy"], "delta_overlap")
        self.assertEqual(summary["viability"][0]["member_train_mode"], "parallel_vmap")
        self.assertEqual(summary["viability"][0]["copies"], 2)
        self.assertAlmostEqual(summary["viability"][0]["branch_seconds"], 60.0, places=6)
        self.assertAlmostEqual(summary["viability"][0]["ensemble_gain_vs_best"], 0.08, places=6)
        self.assertAlmostEqual(summary["evolution"][0]["gain_vs_base"], 0.07, places=6)
        self.assertAlmostEqual(summary["evolution"][0]["ensemble_gain_vs_best"], 0.02, places=6)

    def test_summarize_results_supports_committee_schedule_rows(self) -> None:
        committee = {
            "member_train_mode": "parallel_vmap_staged",
            "committee_schedule": {
                "stage_copies": [2, 4],
                "stage_train_seconds": [10.0, 5.0],
                "stage_signature": "2x10->4x5",
            },
            "members": [
                {"train": {"train_seconds_target": 15.0}, "eval": {"bpb": 1.40}},
                {"train": {"train_seconds_target": 15.0}, "eval": {"bpb": 1.32}},
                {"train": {"train_seconds_target": 15.0}, "eval": {"bpb": 1.35}},
                {"train": {"train_seconds_target": 15.0}, "eval": {"bpb": 1.34}},
            ],
            "member_ensemble_results": [
                {"topk": 4, "val_bpb": 1.25, "test_bpb": 1.27},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "committee.json").write_text(json.dumps(committee), encoding="utf-8")
            summary = summarize_results([str(root)])

        self.assertEqual(summary["viability"][0]["strategy"], "committee_schedule")
        self.assertEqual(summary["viability"][0]["schedule"], "2x10->4x5")
        self.assertAlmostEqual(summary["viability"][0]["branch_seconds"], 40.0, places=6)
        self.assertAlmostEqual(summary["viability"][0]["ensemble_gain_vs_best"], 0.07, places=6)

    def test_summarize_results_supports_committee_adaptive_rows(self) -> None:
        adaptive = {
            "member_train_mode": "parallel_vmap_adaptive",
            "committee_adaptive": {
                "initial_copies": 2,
                "round_train_seconds": [10.0, 5.0],
                "round_signature": "10s->5s",
            },
            "rounds": [
                {"copies": 2, "train_seconds_target": 10.0},
                {"copies": 4, "train_seconds_target": 5.0},
            ],
            "members": [
                {"train": {"train_seconds_target": 15.0}, "eval": {"bpb": 1.33}},
                {"train": {"train_seconds_target": 15.0}, "eval": {"bpb": 1.29}},
                {"train": {"train_seconds_target": 5.0}, "eval": {"bpb": 1.35}},
                {"train": {"train_seconds_target": 5.0}, "eval": {"bpb": 1.31}},
            ],
            "member_ensemble_results": [
                {"topk": 4, "val_bpb": 1.24, "test_bpb": 1.25},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "committee_adaptive.json").write_text(json.dumps(adaptive), encoding="utf-8")
            summary = summarize_results([str(root)])

        self.assertEqual(summary["viability"][0]["strategy"], "committee_adaptive")
        self.assertEqual(summary["viability"][0]["schedule"], "10s->5s")
        self.assertAlmostEqual(summary["viability"][0]["branch_seconds"], 40.0, places=6)
        self.assertAlmostEqual(summary["viability"][0]["ensemble_gain_vs_best"], 0.05, places=6)

    def test_summarize_results_supports_recipe_evolution_rows(self) -> None:
        recipe = {
            "recipe_evolution": {
                "population_size": 6,
                "generations": 3,
                "recipe_profile": "frontier",
            },
            "history": [
                {"best_bpb": 2.30},
                {"best_bpb": 2.18},
                {"best_bpb": 2.09},
            ],
            "best": {
                "val": {"bpb": 2.09},
                "test": {"bpb": 2.11},
            },
            "confirm_results": [
                {"val": {"bpb": 2.03}, "test": {"bpb": 2.05}},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
            summary = summarize_results([str(root)])

        self.assertEqual(summary["evolution"][0]["strategy"], "recipe_evolution")
        self.assertEqual(summary["evolution"][0]["population"], 6)
        self.assertEqual(summary["evolution"][0]["generations"], 3)
        self.assertAlmostEqual(summary["evolution"][0]["best_history_bpb"], 2.09, places=6)
        self.assertAlmostEqual(summary["evolution"][0]["ensemble_gain_vs_best"], 0.06, places=6)

    def test_summarize_results_supports_committee_compressibility_rows(self) -> None:
        payload = {
            "committee_schedule": {
                "stage_copies": [2, 4, 8],
                "stage_train_seconds": [120.0, 30.0, 15.0],
                "stage_signature": "2x120->4x30->8x15",
            },
            "committee_compressibility": {
                "member_count": 8,
            },
            "best_by_budget": [
                {
                    "strategy": "shared_union_sparse",
                    "budget_fraction": 1.0,
                    "val_bpb": 2.08,
                    "test_bpb": 2.10,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "compressibility.json").write_text(json.dumps(payload), encoding="utf-8")
            summary = summarize_results([str(root)])

        self.assertEqual(summary["viability"][0]["member_train_mode"], "committee_compressibility")
        self.assertEqual(summary["viability"][0]["schedule"], "2x120->4x30->8x15")
        self.assertEqual(summary["viability"][0]["copies"], 8)
        self.assertAlmostEqual(summary["viability"][0]["branch_seconds"], 480.0, places=6)


if __name__ == "__main__":
    unittest.main()
