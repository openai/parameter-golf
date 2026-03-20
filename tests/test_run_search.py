from __future__ import annotations

import random
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

    def test_mlx_search_stays_in_local_batch_token_range(self) -> None:
        mlx_batch_tokens = [
            int(token)
            for token in run_search.SEARCH_CHOICES["mlx"]["TRAIN_BATCH_TOKENS"]
        ]

        self.assertTrue(all(token <= 65_536 for token in mlx_batch_tokens))
        self.assertEqual("32768", run_search.PRESETS["mlx"]["baseline"]["TRAIN_BATCH_TOKENS"])

    def test_parse_metrics_cuda_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "train_gpt.py"
            script_path.write_text("print('x')\n", encoding="utf-8")
            log_text = "\n".join(
                [
                    "model_params:123456",
                    "Serialized model int8+zlib: 555 bytes",
                    "Total submission size int8+zlib: 777 bytes",
                    "final_int8_zlib_roundtrip_exact val_loss:4.20000000 val_bpb:1.25000000",
                ]
            )
            val_loss, val_bpb, total_bytes, quantized_model_bytes, model_params = run_search.parse_metrics(
                log_text, "cuda", script_path
            )
        self.assertEqual(4.2, val_loss)
        self.assertEqual(1.25, val_bpb)
        self.assertEqual(777, total_bytes)
        self.assertEqual(555, quantized_model_bytes)
        self.assertEqual(123456, model_params)

    def test_parse_metrics_mlx_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "train_gpt_mlx.py"
            script_path.write_text("print('x')\n", encoding="utf-8")
            log_text = "\n".join(
                [
                    "model_params:98765",
                    "serialized_model_int8_zlib:654 bytes",
                    "final_int8_zlib_roundtrip_exact val_loss:3.10000000 val_bpb:1.10000000",
                ]
            )
            val_loss, val_bpb, total_bytes, quantized_model_bytes, model_params = run_search.parse_metrics(
                log_text, "mlx", script_path
            )
        self.assertEqual(3.1, val_loss)
        self.assertEqual(1.1, val_bpb)
        self.assertEqual(654 + len("print('x')\n".encode("utf-8")), total_bytes)
        self.assertEqual(654, quantized_model_bytes)
        self.assertEqual(98765, model_params)

    def test_parse_metrics_missing_final_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "train_gpt.py"
            script_path.write_text("print('x')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing final validation metrics"):
                run_search.parse_metrics("Serialized model int8+zlib: 123 bytes", "cuda", script_path)

    def test_parse_metrics_missing_cuda_total_size_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "train_gpt.py"
            script_path.write_text("print('x')\n", encoding="utf-8")
            log_text = "\n".join(
                [
                    "Serialized model int8+zlib: 123 bytes",
                    "final_int8_zlib_roundtrip_exact val_loss:3.0 val_bpb:1.0",
                ]
            )
            with self.assertRaisesRegex(ValueError, "missing cuda total submission size metric"):
                run_search.parse_metrics(log_text, "cuda", script_path)

    def test_apply_code_mutation_missing_target_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing mutation target"):
            run_search.apply_code_mutation("print('unchanged')\n", "cuda", "gelu_mlp")

    def test_find_under_limit_candidate_retries_and_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "train_gpt.py"
            script_path.write_text("print('x')\n", encoding="utf-8")
            cfg = run_search.normalize_config("mlx", run_search.PRESETS["mlx"]["baseline"])
            with mock.patch.object(
                run_search, "estimated_total_bytes", return_value=run_search.ARTIFACT_LIMIT_BYTES + 1
            ):
                with self.assertRaisesRegex(RuntimeError, "unable to generate under-limit candidate"):
                    run_search.find_under_limit_candidate(
                        backend="mlx",
                        mode="random",
                        base_cfg=cfg,
                        script_path=script_path,
                        build_candidate=lambda: dict(cfg),
                        max_attempts=3,
                    )

    def test_find_under_limit_code_candidate_retries_and_raises(self) -> None:
        cfg = run_search.normalize_config("mlx", run_search.PRESETS["mlx"]["baseline"])
        with (
            mock.patch.object(run_search, "mutate_config", return_value=dict(cfg)),
            mock.patch.object(run_search, "create_candidate_script", return_value=Path("/tmp/fake.py")),
            mock.patch.object(run_search, "estimated_total_bytes", return_value=run_search.ARTIFACT_LIMIT_BYTES + 1),
        ):
            with self.assertRaisesRegex(RuntimeError, "unable to generate under-limit code candidate"):
                run_search.find_under_limit_code_candidate(
                    backend="mlx",
                    mode="code",
                    base_cfg=cfg,
                    code_mutation="plain_logits",
                    index=0,
                    rng=random.Random(1337),
                    max_attempts=2,
                )


if __name__ == "__main__":
    unittest.main()
