from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_v2b_matrix import build_matrix
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
                    "maintenance_max_slots": 128,
                    "maintenance_metric": "hits",
                    "maintenance_use_grad": True,
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
            self.assertAlmostEqual(float(row["delta_online"]), -0.09)
            self.assertAlmostEqual(float(row["total_gflop"]), 3.9)

            table = render_table([row])
            self.assertIn("delta_online", table)
            self.assertIn("run_a", table)


if __name__ == "__main__":
    unittest.main()
