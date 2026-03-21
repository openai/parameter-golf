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

    def test_legacy_best_and_trial_artifacts_still_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_dir = root / "logs" / "autoresearch"
            trials_dir = log_dir / "trials"
            workbench_dir = log_dir / "workbench"
            results_tsv = log_dir / "results.tsv"
            best_json = log_dir / "best_config.json"
            script_path = root / "train_gpt_mlx.py"

            log_dir.mkdir(parents=True, exist_ok=True)
            trials_dir.mkdir(parents=True, exist_ok=True)
            workbench_dir.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('legacy')\n", encoding="utf-8")

            best_json.write_text(
                """
                {
                  "run_id": "legacy_best",
                  "backend": "mlx",
                  "val_bpb": 1.23,
                  "val_loss": 4.56
                }
                """.strip(),
                encoding="utf-8",
            )
            (trials_dir / "legacy_trial.json").write_text(
                """
                {
                  "run_id": "legacy_trial",
                  "backend": "mlx",
                  "status": "ok",
                  "val_bpb": 1.11,
                  "val_loss": 2.22
                }
                """.strip(),
                encoding="utf-8",
            )

            patches = [
                mock.patch.object(run_search, "ROOT", root),
                mock.patch.object(run_search, "LOG_DIR", log_dir),
                mock.patch.object(run_search, "TRIALS_DIR", trials_dir),
                mock.patch.object(run_search, "WORKBENCH_DIR", workbench_dir),
                mock.patch.object(run_search, "RESULTS_TSV", results_tsv),
                mock.patch.object(run_search, "BEST_JSON", best_json),
            ]

            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                best = run_search.load_best()
                population = run_search.load_population("mlx")

            self.assertIsNotNone(best)
            assert best is not None
            self.assertEqual("legacy_best", best.run_id)
            self.assertEqual([], best.parents)
            self.assertEqual("", best.description)
            self.assertEqual(run_search.PRESETS["mlx"]["baseline"], best.config)
            self.assertEqual(1, len(population))
            self.assertEqual("legacy_trial", population[0].run_id)
            self.assertEqual(run_search.PRESETS["mlx"]["baseline"], population[0].config)

    def test_mlx_search_stays_in_local_batch_token_range(self) -> None:
        mlx_batch_tokens = [
            int(token)
            for token in run_search.SEARCH_CHOICES["mlx"]["TRAIN_BATCH_TOKENS"]
        ]

        self.assertTrue(all(token <= 65_536 for token in mlx_batch_tokens))
        self.assertEqual("8192", run_search.PRESETS["mlx"]["baseline"]["TRAIN_BATCH_TOKENS"])
        self.assertEqual("384", run_search.PRESETS["mlx"]["small_fast"]["MODEL_DIM"])
        self.assertEqual("512", run_search.PRESETS["mlx"]["small_fast"]["TRAIN_SEQ_LEN"])
        self.assertEqual("4096", run_search.PRESETS["mlx"]["small_fast"]["TRAIN_BATCH_TOKENS"])
        self.assertEqual("120", run_search.PRESETS["mlx"]["small_fast"]["MAX_WALLCLOCK_SECONDS"])
        self.assertEqual("2", run_search.PRESETS["mlx"]["micro_smoke"]["WARMUP_STEPS"])
        self.assertEqual("256", run_search.PRESETS["mlx"]["micro_smoke"]["VAL_EVAL_MAX_SEQS"])
        self.assertEqual("128", run_search.PRESETS["mlx"]["micro_smoke"]["MODEL_DIM"])
        self.assertEqual("64", run_search.PRESETS["mlx"]["micro_smoke"]["TRAIN_SEQ_LEN"])
        self.assertEqual("64", run_search.PRESETS["mlx"]["micro_smoke"]["TRAIN_BATCH_TOKENS"])
        self.assertEqual("1", run_search.PRESETS["mlx"]["micro_smoke"]["GRAD_ACCUM_STEPS"])

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

    def test_parse_metrics_mlx_combined_parent_child_log_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "train_gpt_mlx.py"
            script_path.write_text("print('x')\n", encoding="utf-8")
            log_text = "\n".join(
                [
                    "val_eval:truncated max_seqs:64",
                    "serialized_model_int8_zlib:700 bytes",
                    "logs/mlx_contract_check.txt",
                    "mlx_validate_only:starting quantized roundtrip evaluation",
                    "serialized_model_int8_zlib:654 bytes",
                    "final_int8_zlib_roundtrip val_loss:3.2 val_bpb:1.2 eval_time:123ms",
                    "final_int8_zlib_roundtrip_exact val_loss:3.10000000 val_bpb:1.10000000",
                ]
            )
            _, _, total_bytes, quantized_model_bytes, _ = run_search.parse_metrics(
                log_text, "mlx", script_path
            )
        self.assertEqual(654, quantized_model_bytes)
        self.assertEqual(654 + len("print('x')\n".encode("utf-8")), total_bytes)

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

    def test_mlx_code_mutations_match_current_training_script(self) -> None:
        script_text = Path("train_gpt_mlx.py").read_text(encoding="utf-8")
        for mutation_name in run_search.CODE_MUTATIONS["mlx"]:
            with self.subTest(mutation_name=mutation_name):
                mutated = run_search.apply_code_mutation(script_text, "mlx", mutation_name)
                self.assertNotEqual(script_text, mutated)

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

    def test_build_command_and_run_trial_use_unbuffered_output_for_mlx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_dir = root / "logs" / "autoresearch"
            trials_dir = log_dir / "trials"
            workbench_dir = log_dir / "workbench"
            results_tsv = log_dir / "results.tsv"
            best_json = log_dir / "best_config.json"
            script_path = root / "train_gpt_mlx.py"
            script_path.write_text("print('candidate')\n", encoding="utf-8")

            patches = [
                mock.patch.object(run_search, "ROOT", root),
                mock.patch.object(run_search, "LOG_DIR", log_dir),
                mock.patch.object(run_search, "TRIALS_DIR", trials_dir),
                mock.patch.object(run_search, "WORKBENCH_DIR", workbench_dir),
                mock.patch.object(run_search, "RESULTS_TSV", results_tsv),
                mock.patch.object(run_search, "BEST_JSON", best_json),
                mock.patch.object(run_search, "TRAIN_MLX", script_path),
                mock.patch.object(run_search, "TRAIN_CUDA", root / "train_gpt.py"),
            ]

            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                run_search.ensure_dirs()
                command = run_search.build_command("mlx", script_path, 1)
                self.assertEqual(["uv", "run", "python3", "-u", str(script_path)], command)

                captured_env: dict[str, str] = {}

                def fake_run(command, cwd, env, stdout, stderr, text):  # noqa: ANN001
                    captured_env.update(env)
                    class Completed:
                        returncode = 0
                    stdout.write(
                        "model_params:1\n"
                        "serialized_model_int8_zlib:1 bytes\n"
                        "final_int8_zlib_roundtrip_exact val_loss:1.0 val_bpb:1.0\n"
                    )
                    return Completed()

                with mock.patch.object(run_search.subprocess, "run", side_effect=fake_run):
                    result = run_search.run_trial(
                        index=0,
                        backend="mlx",
                        mode="preset",
                        cfg=run_search.normalize_config("mlx", run_search.PRESETS["mlx"]["small_fast"]),
                        nproc=1,
                        description="preset:small_fast",
                        preset="small_fast",
                    )

            self.assertEqual("1", captured_env["PYTHONUNBUFFERED"])
            self.assertEqual("ok", result.status)


if __name__ == "__main__":
    unittest.main()
