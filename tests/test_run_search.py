from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autoresearch import run_search


class RunSearchPersistenceTests(unittest.TestCase):
    def test_make_result_keeps_description_outside_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            script_path = root / "train_gpt.py"
            log_path = root / "logs" / "autoresearch" / "trial.log"
            script_path.write_text("print('hi')\n", encoding="utf-8")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")

            with mock.patch.object(run_search, "ROOT", root):
                result = run_search.make_result(
                    run_id="ar_test_001",
                    backend="mlx",
                    mode="evolution",
                    status="ok",
                    val_bpb=1.23,
                    val_loss=4.56,
                    total_bytes=1234,
                    script_path=script_path,
                    quantized_model_bytes=1200,
                    model_params=42,
                    elapsed_seconds=1.5,
                    log_path=log_path,
                    preset="balanced",
                    code_mutation="",
                    parents=["seed_a", "seed_b"],
                    config={"MODEL_DIM": "512"},
                    description="parent-selected candidate",
                )

            self.assertEqual("parent-selected candidate", result.description)
            self.assertEqual({"MODEL_DIM": "512"}, result.config)

    def test_append_result_writes_parents_and_script_path_to_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_dir = root / "logs" / "autoresearch"
            trials_dir = log_dir / "trials"
            workbench_dir = log_dir / "workbench"
            results_tsv = log_dir / "results.tsv"
            best_json = log_dir / "best_config.json"
            script_path = root / "logs" / "autoresearch" / "workbench" / "candidate.py"
            log_path = root / "logs" / "autoresearch" / "trial.log"

            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('candidate')\n", encoding="utf-8")
            log_path.write_text("", encoding="utf-8")

            patches = [
                mock.patch.object(run_search, "ROOT", root),
                mock.patch.object(run_search, "LOG_DIR", log_dir),
                mock.patch.object(run_search, "TRIALS_DIR", trials_dir),
                mock.patch.object(run_search, "WORKBENCH_DIR", workbench_dir),
                mock.patch.object(run_search, "RESULTS_TSV", results_tsv),
                mock.patch.object(run_search, "BEST_JSON", best_json),
            ]

            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                run_search.ensure_dirs()
                result = run_search.make_result(
                    run_id="ar_test_002",
                    backend="mlx",
                    mode="code",
                    status="ok",
                    val_bpb=1.11,
                    val_loss=2.22,
                    total_bytes=3333,
                    script_path=script_path,
                    quantized_model_bytes=3000,
                    model_params=123,
                    elapsed_seconds=9.0,
                    log_path=log_path,
                    preset="",
                    code_mutation="gelu_mlp",
                    parents=["parent_one", "parent_two"],
                    config={"MODEL_DIM": "448"},
                    description="mutation candidate",
                )
                run_search.append_result(result)

                lines = results_tsv.read_text(encoding="utf-8").strip().splitlines()

            self.assertEqual(
                "run_id\tbackend\tmode\tval_bpb\tval_loss\ttotal_bytes\tstatus\tpreset\tcode_mutation\tparents\ttrain_script_path\tdescription",
                lines[0],
            )
            self.assertIn("parent_one,parent_two", lines[1])
            self.assertIn("logs/autoresearch/workbench/candidate.py", lines[1])
            self.assertTrue((trials_dir / "ar_test_002.json").exists())


if __name__ == "__main__":
    unittest.main()
