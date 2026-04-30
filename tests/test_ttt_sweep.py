#!/usr/bin/env python3
"""Tests for scripts/run_longtrain_ttt_sweep.py."""

import csv
import json
import os
import sys
import unittest

# Ensure repo root is on path so we can import the sweep module
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import run_longtrain_ttt_sweep as sweep


class TestVariantDefinitions(unittest.TestCase):
    """Test that all 7 variants are properly defined."""

    def test_variant_count(self):
        self.assertEqual(len(sweep.VARIANTS), 7)

    def test_all_variants_have_env(self):
        for vid, cfg in sweep.VARIANTS.items():
            self.assertIn("env", cfg, "variant %s missing 'env'" % vid)
            self.assertIsInstance(cfg["env"], dict)

    def test_all_variants_have_description(self):
        for vid, cfg in sweep.VARIANTS.items():
            self.assertIn("description", cfg, "variant %s missing 'description'" % vid)
            self.assertTrue(len(cfg["description"]) > 0)

    def test_optional_flag_only_on_v6(self):
        for vid, cfg in sweep.VARIANTS.items():
            if vid == "v6_prefix3000_phase4_optional":
                self.assertTrue(cfg.get("optional", False))
            else:
                self.assertFalse(cfg.get("optional", False),
                                 "variant %s should not be optional" % vid)

    def test_required_env_keys_present(self):
        required_keys = {
            "TTT_LORA_RANK", "TTT_LORA_ALPHA", "TTT_LORA_LR",
            "TTT_BATCH_SIZE", "TTT_CHUNK_SIZE", "GLOBAL_TTT_EPOCHS",
            "GLOBAL_TTT_CHUNK_TOKENS", "GLOBAL_TTT_BATCH_SEQS",
            "GLOBAL_TTT_WARMUP_START_LR", "GLOBAL_TTT_WARMUP_CHUNKS",
            "PHASED_TTT_PREFIX_DOCS", "PHASED_TTT_NUM_PHASES",
            "TTT_WARM_START_A",
        }
        for vid, cfg in sweep.VARIANTS.items():
            missing = required_keys - set(cfg["env"].keys())
            self.assertEqual(missing, set(),
                             "variant %s missing keys: %s" % (vid, missing))

    def test_all_env_values_are_strings(self):
        for vid, cfg in sweep.VARIANTS.items():
            for k, v in cfg["env"].items():
                self.assertIsInstance(v, str,
                                     "variant %s key %s: expected str, got %s"
                                     % (vid, k, type(v).__name__))


class TestBuildVariantEnv(unittest.TestCase):
    """Test build_variant_env produces correct merged environment."""

    def test_fixed_env_present(self):
        vid = "v0_control_pr1979"
        cfg = sweep.VARIANTS[vid]
        env = sweep.build_variant_env(
            vid, cfg, "/fake/model.ptz", "/fake/output",
            "train_gpt.py", "/fake/data", "/fake/tok.model")
        for k, v in sweep.FIXED_TTT_ENV.items():
            self.assertEqual(env[k], v,
                             "fixed key %s: expected %s, got %s" % (k, v, env.get(k)))

    def test_variant_overrides_present(self):
        vid = "v2_rank128_lr3e4"
        cfg = sweep.VARIANTS[vid]
        env = sweep.build_variant_env(
            vid, cfg, "/fake/model.ptz", "/fake/output",
            "train_gpt.py", "/fake/data", "/fake/tok.model")
        self.assertEqual(env["TTT_LORA_RANK"], "128")
        self.assertEqual(env["TTT_LORA_LR"], "0.0003")

    def test_artifact_path_set(self):
        vid = "v0_control_pr1979"
        cfg = sweep.VARIANTS[vid]
        env = sweep.build_variant_env(
            vid, cfg, "/my/artifact.ptz", "/out",
            "train_gpt.py", None, None)
        self.assertEqual(env["LOAD_QUANTIZED_MODEL_PATH"], "/my/artifact.ptz")

    def test_output_json_path(self):
        vid = "v1_rank128_alpha192"
        cfg = sweep.VARIANTS[vid]
        env = sweep.build_variant_env(
            vid, cfg, "/a.ptz", "/sweep_out",
            "train_gpt.py", None, None)
        expected = os.path.join("/sweep_out", vid, "ttt_eval_summary.json")
        self.assertEqual(env["TTT_EVAL_OUTPUT_JSON"], expected)

    def test_output_dir_per_variant(self):
        vid = "v3_local_batch_chunk"
        cfg = sweep.VARIANTS[vid]
        env = sweep.build_variant_env(
            vid, cfg, "/a.ptz", "/sweep_out",
            "train_gpt.py", None, None)
        self.assertEqual(env["OUTPUT_DIR"], os.path.join("/sweep_out", vid))

    def test_eval_only_always_set(self):
        for vid, cfg in sweep.VARIANTS.items():
            env = sweep.build_variant_env(
                vid, cfg, "/a.ptz", "/o", "train_gpt.py", None, None)
            self.assertEqual(env.get("TTT_EVAL_ONLY"), "1",
                             "variant %s must have TTT_EVAL_ONLY=1" % vid)

    def test_no_missing_keys_vs_fixed_and_variant(self):
        """Every key from FIXED + variant env must appear in merged env."""
        for vid, cfg in sweep.VARIANTS.items():
            env = sweep.build_variant_env(
                vid, cfg, "/a.ptz", "/o", "t.py", None, None)
            for k in sweep.FIXED_TTT_ENV:
                self.assertIn(k, env, "variant %s missing fixed key %s" % (vid, k))
            for k in cfg["env"]:
                self.assertIn(k, env, "variant %s missing variant key %s" % (vid, k))


class TestSelectVariants(unittest.TestCase):
    """Test variant selection logic."""

    def test_default_excludes_optional(self):
        selected = sweep.select_variants(None, include_optional=False)
        ids = [vid for vid, _ in selected]
        self.assertNotIn("v6_prefix3000_phase4_optional", ids)
        self.assertEqual(len(selected), 6)

    def test_include_optional(self):
        selected = sweep.select_variants(None, include_optional=True)
        ids = [vid for vid, _ in selected]
        self.assertIn("v6_prefix3000_phase4_optional", ids)
        self.assertEqual(len(selected), 7)

    def test_filter_specific(self):
        selected = sweep.select_variants("v0_control_pr1979,v5_prefix3000",
                                         include_optional=False)
        ids = [vid for vid, _ in selected]
        self.assertEqual(ids, ["v0_control_pr1979", "v5_prefix3000"])

    def test_filter_includes_optional_explicitly(self):
        selected = sweep.select_variants("v6_prefix3000_phase4_optional",
                                         include_optional=False)
        ids = [vid for vid, _ in selected]
        self.assertEqual(ids, ["v6_prefix3000_phase4_optional"])


class TestManifestGeneration(unittest.TestCase):
    """Test manifest JSON generation."""

    def setUp(self):
        self.out_dir = os.path.join(REPO_ROOT, "_test_sweep_manifest_tmp")
        os.makedirs(self.out_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

    def test_manifest_written(self):
        variants = sweep.select_variants(None, include_optional=True)
        path = sweep.generate_variant_manifest(
            variants, "/fake/model.ptz", self.out_dir)
        self.assertTrue(os.path.exists(path))

        with open(path) as f:
            manifest = json.load(f)
        self.assertEqual(len(manifest["variants"]), 7)
        self.assertEqual(manifest["artifact_path"], "/fake/model.ptz")
        self.assertIn("fixed_env", manifest)
        self.assertIn("generated_at", manifest)

    def test_manifest_variant_structure(self):
        variants = [("v0_control_pr1979", sweep.VARIANTS["v0_control_pr1979"])]
        path = sweep.generate_variant_manifest(variants, "/a.ptz", self.out_dir)
        with open(path) as f:
            manifest = json.load(f)
        v0 = manifest["variants"]["v0_control_pr1979"]
        self.assertIn("description", v0)
        self.assertIn("env_overrides", v0)
        self.assertIn("optional", v0)


class TestAggregateResults(unittest.TestCase):
    """Test CSV + JSON aggregation from per-variant result dicts."""

    def setUp(self):
        self.out_dir = os.path.join(REPO_ROOT, "_test_sweep_aggregate_tmp")
        os.makedirs(self.out_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

    def _make_results(self):
        return [
            {
                "variant_id": "v0_control_pr1979",
                "description": "baseline",
                "quantized_bpb_fixed": 1.04944,
                "post_ttt_bpb": 1.03988,
                "ttt_gain_bpb": 0.00956,
                "eval_seconds": 540.0,
                "total_wallclock_seconds": 600.0,
                "docs_evaluated": 5000,
                "tokens_evaluated": 12345678,
                "prefix_docs": 2000,
                "phases": 3,
                "peak_memory_mib": 65000,
                "status": "success",
                "error": None,
            },
            {
                "variant_id": "v1_rank128_alpha192",
                "description": "higher rank",
                "quantized_bpb_fixed": 1.04944,
                "post_ttt_bpb": 1.03500,
                "ttt_gain_bpb": 0.01444,
                "eval_seconds": 580.0,
                "total_wallclock_seconds": 620.0,
                "docs_evaluated": 5000,
                "tokens_evaluated": 12345678,
                "prefix_docs": 2000,
                "phases": 3,
                "peak_memory_mib": 70000,
                "status": "success",
                "error": None,
            },
            {
                "variant_id": "v2_rank128_lr3e4",
                "description": "timeout example",
                "quantized_bpb_fixed": None,
                "post_ttt_bpb": None,
                "ttt_gain_bpb": None,
                "eval_seconds": None,
                "total_wallclock_seconds": 1200.0,
                "docs_evaluated": None,
                "tokens_evaluated": None,
                "prefix_docs": 2000,
                "phases": 3,
                "peak_memory_mib": None,
                "status": "timeout",
                "error": "exceeded 20 min timeout",
            },
        ]

    def test_csv_written(self):
        results = self._make_results()
        csv_path, _ = sweep.aggregate_results(self.out_dir, results)
        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["variant_id"], "v0_control_pr1979")
        self.assertEqual(rows[0]["status"], "success")

    def test_csv_columns(self):
        results = self._make_results()
        csv_path, _ = sweep.aggregate_results(self.out_dir, results)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        for field in sweep.RESULT_FIELDS:
            self.assertIn(field, fieldnames, "CSV missing field: %s" % field)

    def test_summary_best_variant(self):
        results = self._make_results()
        _, summary_path = sweep.aggregate_results(self.out_dir, results)
        with open(summary_path) as f:
            summary = json.load(f)
        self.assertEqual(summary["total_variants"], 3)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["timed_out"], 1)
        best = summary["best_variant"]
        self.assertIsNotNone(best)
        self.assertEqual(best["variant_id"], "v1_rank128_alpha192")
        self.assertAlmostEqual(best["post_ttt_bpb"], 1.03500, places=5)

    def test_summary_no_successful(self):
        results = [self._make_results()[2]]  # only the timeout
        _, summary_path = sweep.aggregate_results(self.out_dir, results)
        with open(summary_path) as f:
            summary = json.load(f)
        self.assertIsNone(summary["best_variant"])


class TestDryRun(unittest.TestCase):
    """Test dry-run produces expected output."""

    def test_dry_run_no_crash(self):
        """dry_run should print without errors."""
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            variants = sweep.select_variants(None, include_optional=False)
            sweep.dry_run(variants, "/fake/model.ptz", "/fake/output",
                          8, 20, "train_gpt.py", None, None)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        self.assertIn("DRY RUN", output)
        self.assertIn("v0_control_pr1979", output)
        self.assertIn("6 variants", output)
        self.assertIn("LOAD_QUANTIZED_MODEL_PATH", output)

    def test_dry_run_with_optional(self):
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            variants = sweep.select_variants(None, include_optional=True)
            sweep.dry_run(variants, "/fake/model.ptz", "/fake/output",
                          8, 20, "train_gpt.py", None, None)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        self.assertIn("7 variants", output)
        self.assertIn("v6_prefix3000_phase4_optional", output)
        self.assertIn("[OPTIONAL]", output)


class TestEmitPodCommand(unittest.TestCase):
    """Test pod command generation."""

    def test_emit_contains_all_variants(self):
        variants = sweep.select_variants(None, include_optional=False)
        script = sweep.emit_pod_command(
            variants, "/root/model.ptz", "/root/sweep_out",
            8, 20, "train_gpt.py", None, None)
        self.assertIn("#!/bin/bash", script)
        self.assertIn("set -euo pipefail", script)
        for vid, _ in variants:
            self.assertIn(vid, script)
        self.assertIn("torchrun", script)
        self.assertIn("TTT_EVAL_ONLY", script)


if __name__ == "__main__":
    unittest.main()
