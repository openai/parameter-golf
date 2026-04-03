from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_v2b_matrix import build_matrix, load_matrix_file
from tools.summarize_v2b_runs import load_row, render_table, resolve_result_path


class V2BToolsTests(unittest.TestCase):
    def test_matrix_names_are_unique_and_shell_commands_include_outputs(self) -> None:
        runs = build_matrix()
        names = [run.name for run in runs]
        self.assertEqual(len(names), len(set(names)))
        self.assertGreaterEqual(len(runs), 8)
        shell = runs[0].shell_command("python3.11", "spectral_flood_walk_v2b.py", "runs")
        self.assertIn("spectral_flood_walk_v2b.py", shell)
        self.assertIn("--output-json", shell)
        self.assertIn("--model-artifact-path", shell)

    def test_summarize_v2b_row_parses_new_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_a"
            run_dir.mkdir()
            result_path = run_dir / "result.json"
            payload = {
                "config": {
                    "seq_len": 512,
                    "model_dim": 448,
                    "num_layers": 9,
                    "memory_table_size": 65536,
                    "memory_min_read_count": 2.0,
                    "maintenance_passes": 2,
                    "maintenance_mode": "replay",
                    "maintenance_step_size": 0.05,
                    "maintenance_max_slots": 128,
                    "maintenance_metric": "hits",
                    "maintenance_grad_mix": 0.25,
                    "maintenance_replay_depth": 3,
                    "maintenance_replay_candidates": 64,
                    "maintenance_use_grad": False,
                },
                "eval_context": {"val_bpb": 2.40},
                "eval_online_persistent_hidden": {
                    "val_bpb": 2.31,
                    "memory_lookup_flops_estimate": 1.5e9,
                    "memory_update_flops_estimate": 0.4e9,
                    "memory_maintenance_flops_estimate": 2.0e9,
                    "active_slots_mean": 0.7,
                    "readable_slots_mean": 0.55,
                    "delta_norm_mean": 0.8,
                    "persistent_memory": {
                        "readable_fraction": 0.12,
                        "replay_fraction": 0.08,
                        "mean_loss_ema": 1.7,
                        "resident_mb": 96.0,
                    },
                },
                "eval_delta_online_bpb": -0.09,
                "model_artifact_bytes": 12_000_000,
            }
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            resolved = resolve_result_path(str(run_dir))
            self.assertEqual(resolved, result_path)

            row = load_row(result_path)
            self.assertEqual(row["run"], "run_a")
            self.assertEqual(row["maint_metric"], "hits")
            self.assertEqual(row["maint_mode"], "replay")
            self.assertEqual(row["replay_depth"], 3)
            self.assertAlmostEqual(float(row["delta_online"]), -0.09)
            self.assertAlmostEqual(float(row["total_gflop"]), 3.9)
            self.assertAlmostEqual(float(row["loss_ema"]), 1.7)

            table = render_table([row])
            self.assertIn("delta_online", table)
            self.assertIn("run_a", table)
            self.assertIn("maint_mode", table)
            self.assertIn("replay_depth", table)

    def test_matrix_json_grid_expands_axes_and_applies_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_path = Path(tmpdir) / "matrix.json"
            payload = {
                "stage": "search",
                "intent": "Generated search",
                "name_prefix": "grid",
                "axes": [
                    {
                        "name": "gate",
                        "values": [
                            {"token": "g1", "flags": ["--memory-min-read-count", "1"]},
                            {"token": "g4", "flags": ["--memory-min-read-count", "4"]},
                        ],
                    },
                    {
                        "name": "maint",
                        "values": [
                            {"token": "off", "flags": ["--maintenance-passes", "0"]},
                            {"token": "on", "flags": ["--maintenance-passes", "2"]},
                        ],
                    },
                    {
                        "name": "metric",
                        "values": [
                            {"token": "counts", "flags": ["--maintenance-metric", "counts"]},
                            {"token": "hits", "flags": ["--maintenance-metric", "hits"]},
                        ],
                    },
                ],
                "exclude": [{"maint": "off", "metric": "hits"}],
            }
            matrix_path.write_text(json.dumps(payload), encoding="utf-8")

            runs = load_matrix_file(matrix_path)
            self.assertEqual(len(runs), 6)
            self.assertEqual(runs[0].stage, "search")
            names = {run.name for run in runs}
            self.assertIn("grid_g1_off_counts", names)
            self.assertIn("grid_g4_on_hits", names)
            self.assertNotIn("grid_g1_off_hits", names)
            shell = next(run for run in runs if run.name == "grid_g4_on_hits").shell_command(
                "python3", "spectral_flood_walk_v2b.py", "runs"
            )
            self.assertIn("--memory-min-read-count", shell)
            self.assertIn("--maintenance-metric", shell)


if __name__ == "__main__":
    unittest.main()
