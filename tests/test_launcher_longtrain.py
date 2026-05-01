"""Tests for run_longtrain_scaling.py CLI args and command building."""

import argparse
import os
import sys
import unittest
from pathlib import Path

# Ensure scripts/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

# We can't fully import run_longtrain_scaling without the runpod deps being
# available, so we mock them at the module level before importing.
import types

# Stub modules that require network / external deps
for mod_name in ("runpod_http_rehearsal", "runpod_safe"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Provide stub symbols expected by run_longtrain_scaling
_rhr = sys.modules["runpod_http_rehearsal"]
_rhr.main = lambda: None
_rhr.build_bundle_b64 = lambda **kw: ""
_rhr.build_boot_command = lambda cmd: ""
_rhr.build_launcher_state = lambda **kw: {}
_rhr.write_launcher_state = lambda *a, **kw: None
_rhr.record_launcher_exception = lambda *a, **kw: None
_rhr.terminate_pod_with_launcher_state = lambda *a, **kw: None
_rhr.wait_http_proxy = lambda *a, **kw: None
_rhr.wait_startup_readiness_and_maybe_download_status = lambda *a, **kw: None
_rhr.download_file = lambda *a, **kw: None
_rhr.H100_COST_PER_GPU_HR = 3.50
_rhr.HTTP_TERMINAL_STATUSES = ("DONE", "FAIL", "TIMEOUT")

_rs = sys.modules["runpod_safe"]
_rs.UA = "test"
_rs._make_ssl_ctx = lambda: None
_rs.balance = lambda: (100.0, "USD")
_rs.create_pod = lambda **kw: {"id": "test", "costPerHr": 28.0}
_rs.wait_runtime = lambda pid: {"uptimeInSeconds": 0}
_rs.terminate_and_wait = lambda pid: None

import run_longtrain_scaling as launcher

# Real snapshot directory for integration-style tests
REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "results" / "8h_longtrain_final" / "resume_snapshot_step_36452"


class TestParseExportMinutes(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(launcher.parse_export_minutes("10,20,30"), [10, 20, 30])

    def test_sorting(self):
        self.assertEqual(launcher.parse_export_minutes("30,10,20"), [10, 20, 30])

    def test_spaces(self):
        self.assertEqual(launcher.parse_export_minutes(" 5 , 10 , 15 "), [5, 10, 15])


class TestDurationHoursDefaults(unittest.TestCase):
    """--duration-hours should auto-apply 4h defaults."""

    def _parse(self, *extra_args):
        """Parse args with --dry-run (won't launch) plus extras."""
        old_argv = sys.argv
        try:
            sys.argv = ["prog", "--dry-run"] + list(extra_args)
            parser = self._build_parser()
            args = parser.parse_args()
            # Replicate the duration-hours default logic from main()
            if args.duration_hours is not None:
                h = args.duration_hours
                if args.max_wallclock == launcher.DEFAULT_MAX_WALLCLOCK:
                    args.max_wallclock = h * 3600
                if args.max_minutes == launcher.DEFAULT_MAX_MINUTES:
                    args.max_minutes = h * 60 + 60
                if args.export_minutes == launcher.DEFAULT_EXPORT_MINUTES:
                    args.export_minutes = launcher.DEFAULT_4H_EXPORT_MINUTES
                if args.resume_save_minutes is None:
                    args.resume_save_minutes = launcher.DEFAULT_4H_RESUME_SAVE_MINUTES
                if args.iterations is None:
                    args.iterations = launcher.DEFAULT_4H_ITERATIONS
            return args
        finally:
            sys.argv = old_argv

    def _build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=launcher.DEFAULT_SEED)
        parser.add_argument("--max-minutes", type=int, default=launcher.DEFAULT_MAX_MINUTES)
        parser.add_argument("--max-wallclock", type=int, default=launcher.DEFAULT_MAX_WALLCLOCK)
        parser.add_argument("--export-minutes", default=launcher.DEFAULT_EXPORT_MINUTES)
        parser.add_argument("--export-mode", default=launcher.DEFAULT_EXPORT_MODE)
        parser.add_argument("--train-script", default=None)
        parser.add_argument("--results-dir", default=None)
        parser.add_argument("--download-checkpoints", action="store_true")
        parser.add_argument("--duration-hours", type=int, default=None)
        parser.add_argument("--iterations", type=int, default=None)
        parser.add_argument("--enable-resume", action="store_true")
        parser.add_argument("--resume-save-minutes", default=None)
        parser.add_argument("--resume-from", default=None)
        parser.add_argument("--resume-keep-last", type=int, default=3)
        parser.add_argument("--run-ttt-sweep-after-train", action="store_true")
        parser.add_argument("--ttt-sweep-variants", default=None)
        parser.add_argument("--ttt-max-minutes-per-variant", type=int, default=20)
        parser.add_argument("--dry-run", action="store_true")
        return parser

    def test_4h_wallclock(self):
        args = self._parse("--duration-hours", "4")
        self.assertEqual(args.max_wallclock, 14400)

    def test_4h_max_minutes(self):
        args = self._parse("--duration-hours", "4")
        self.assertEqual(args.max_minutes, 300)

    def test_4h_iterations_default(self):
        args = self._parse("--duration-hours", "4")
        self.assertEqual(args.iterations, launcher.DEFAULT_4H_ITERATIONS)

    def test_4h_resume_save_minutes(self):
        args = self._parse("--duration-hours", "4")
        self.assertEqual(args.resume_save_minutes, launcher.DEFAULT_4H_RESUME_SAVE_MINUTES)

    def test_4h_export_minutes(self):
        args = self._parse("--duration-hours", "4")
        self.assertEqual(args.export_minutes, launcher.DEFAULT_4H_EXPORT_MINUTES)

    def test_manual_override_takes_priority(self):
        args = self._parse("--duration-hours", "4", "--max-wallclock", "7200")
        self.assertEqual(args.max_wallclock, 7200)

    def test_iterations_override(self):
        args = self._parse("--duration-hours", "4", "--iterations", "50000")
        self.assertEqual(args.iterations, 50000)


class TestBuildSeedCmdResume(unittest.TestCase):
    """Resume env vars should appear in the shell command."""

    def _make_args(self, **overrides):
        defaults = dict(
            seed=42, export_minutes="10,20", max_wallclock=3600,
            export_mode="light", enable_resume=False,
            resume_save_minutes=None, resume_keep_last=3,
            resume_from=None, iterations=None,
            run_ttt_sweep_after_train=False,
            ttt_sweep_variants=None, ttt_max_minutes_per_variant=20,
            prequant_only=False, resume_decompose_only=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_resume_by_default(self):
        args = self._make_args()
        cmd = launcher.build_seed_cmd(args)
        self.assertNotIn("RESUME_ENABLED=1", cmd)
        self.assertNotIn("RESUME_DIR=", cmd)

    def test_resume_enabled(self):
        args = self._make_args(enable_resume=True, resume_save_minutes="30,60,90")
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("RESUME_ENABLED=1", cmd)
        self.assertIn("RESUME_DIR=/root/rehearsal_out/seed42/resume", cmd)
        self.assertIn("RESUME_SAVE_MINUTES=30,60,90", cmd)
        self.assertIn("RESUME_KEEP_LAST=3", cmd)

    def test_resume_from(self):
        args = self._make_args(resume_from="/some/path/ckpt.pt")
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("RESUME_FROM=/some/path/ckpt.pt", cmd)

    def test_iterations_in_env(self):
        args = self._make_args(iterations=100000)
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("ITERATIONS=100000", cmd)

    def test_no_iterations_by_default(self):
        args = self._make_args()
        cmd = launcher.build_seed_cmd(args)
        self.assertNotIn("ITERATIONS=", cmd)

    def test_prequant_only(self):
        args = self._make_args(prequant_only=True)
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("PREQUANT_ONLY=1", cmd)
        self.assertIn(
            "PREQUANT_EVAL_OUTPUT_JSON=/root/rehearsal_out/seed42/prequant_eval_summary.json",
            cmd,
        )
        self.assertIn(
            "cp /root/rehearsal_out/seed42/prequant_eval_summary.json "
            "/root/rehearsal_out/prequant_eval_summary.json",
            cmd,
        )

    def test_resume_decompose_only(self):
        args = self._make_args(resume_decompose_only=True)
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("RESUME_DECOMPOSE_ONLY=1", cmd)
        self.assertIn(
            "RESUME_DECOMPOSE_OUTPUT_JSON=/root/rehearsal_out/seed42/resume_stage_decomposition.json",
            cmd,
        )
        self.assertIn(
            "RESUME_DECOMPOSE_BATCH_JSONL=/root/rehearsal_out/seed42/resume_stage_batch_deltas.jsonl",
            cmd,
        )
        self.assertIn(
            "cp /root/rehearsal_out/seed42/resume_stage_decomposition.json "
            "/root/rehearsal_out/resume_stage_decomposition.json",
            cmd,
        )
        self.assertIn(
            "cp /root/rehearsal_out/seed42/resume_stage_batch_deltas.jsonl "
            "/root/rehearsal_out/resume_stage_batch_deltas.jsonl",
            cmd,
        )
        self.assertIn(
            "cp /root/rehearsal_out/seed42/ttt_eval_summary.json "
            "/root/rehearsal_out/ttt_eval_summary.json",
            cmd,
        )

    def test_resume_decompose_only_uses_eval_download_script(self):
        args = self._make_args(resume_decompose_only=True)
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("CaseOps eval data ready", cmd)
        self.assertNotIn("Expected >=39 train shards", cmd)


class TestBuildSeedCmdTTTSweep(unittest.TestCase):
    """TTT sweep command should be appended when flag is set."""

    def _make_args(self, **overrides):
        defaults = dict(
            seed=42, export_minutes="10,20", max_wallclock=3600,
            export_mode="light", enable_resume=False,
            resume_save_minutes=None, resume_keep_last=3,
            resume_from=None, iterations=None,
            run_ttt_sweep_after_train=False,
            ttt_sweep_variants=None, ttt_max_minutes_per_variant=20,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_no_sweep_by_default(self):
        args = self._make_args()
        cmd = launcher.build_seed_cmd(args)
        self.assertNotIn("run_longtrain_ttt_sweep", cmd)

    def test_sweep_enabled(self):
        args = self._make_args(run_ttt_sweep_after_train=True)
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("run_longtrain_ttt_sweep.py", cmd)
        self.assertIn("--max-minutes-per-variant 20", cmd)
        self.assertNotIn("--variants", cmd)

    def test_sweep_with_variants(self):
        args = self._make_args(run_ttt_sweep_after_train=True, ttt_sweep_variants="v1,v2")
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("--variants v1,v2", cmd)

    def test_sweep_copies_results(self):
        args = self._make_args(run_ttt_sweep_after_train=True)
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("ttt_sweep_manifest.json", cmd)
        self.assertIn("ttt_sweep_results.csv", cmd)
        self.assertIn("ttt_sweep_summary.json", cmd)


class TestBuildSweepOnlyCmd(unittest.TestCase):
    def test_uses_eval_download_script(self):
        args = argparse.Namespace(
            ttt_max_minutes_per_variant=20,
            ttt_sweep_variants=None,
        )
        cmd = launcher.build_sweep_only_cmd(args)
        self.assertIn("CaseOps eval data ready", cmd)
        self.assertNotIn("Expected >=39 train shards", cmd)


class TestBuildDownloadCaseOpsScript(unittest.TestCase):
    def test_full_mode_downloads_train_and_val(self):
        script = launcher.build_download_caseops_script("full")
        self.assertIn(
            "datasets/datasets/{}/".format(launcher.CASEOPS_DATASET_DIR),
            script,
        )
        self.assertIn("Expected >=39 train shards", script)

    def test_eval_mode_downloads_val_only(self):
        script = launcher.build_download_caseops_script("eval")
        self.assertIn(
            "datasets/datasets/{}/fineweb_val_*".format(launcher.CASEOPS_DATASET_DIR),
            script,
        )
        self.assertIn("CaseOps eval data ready", script)
        self.assertNotIn("Expected >=39 train shards", script)


class TestBuildDownloadList(unittest.TestCase):
    def test_no_sweep(self):
        files = launcher.build_download_list(42, "10,20")
        self.assertNotIn("ttt_sweep/ttt_sweep_manifest.json", files)

    def test_with_sweep(self):
        files = launcher.build_download_list(42, "10,20", include_ttt_sweep=True)
        self.assertIn("ttt_sweep/ttt_sweep_manifest.json", files)
        self.assertIn("ttt_sweep/ttt_sweep_results.csv", files)
        self.assertIn("ttt_sweep/ttt_sweep_summary.json", files)

    def test_always_has_base_files(self):
        files = launcher.build_download_list(42, "10,20")
        self.assertIn("status.txt", files)
        self.assertIn("seed42_log.txt", files)
        self.assertIn("final_model.int6.ptz", files)
        self.assertIn("scaling_results.csv", files)
        self.assertIn("checkpoint_10min.json", files)
        self.assertIn("final_model.int6.10min.ptz", files)

    def test_prequant_only_downloads_summary_and_skips_final_artifact(self):
        files = launcher.build_download_list(42, "360", prequant_only=True)
        self.assertIn("prequant_eval_summary.json", files)
        self.assertNotIn("final_model.int6.ptz", files)
        self.assertNotIn("scaling_results.csv", files)

    def test_resume_decompose_only_downloads_stage_outputs(self):
        files = launcher.build_download_list(42, "360", resume_decompose_only=True)
        self.assertIn("resume_stage_decomposition.json", files)
        self.assertIn("resume_stage_batch_deltas.jsonl", files)
        self.assertIn("ttt_eval_summary.json", files)
        self.assertIn("final_model.int6.ptz", files)
        self.assertNotIn("checkpoint_360min.json", files)


class TestBuildSweepDownloadList(unittest.TestCase):
    def test_default_excludes_optional_variant(self):
        files = launcher.build_sweep_download_list()
        self.assertNotIn(
            "ttt_sweep/v6_prefix3000_phase4_optional/variant_result.json", files
        )

    def test_requested_variant_limits_downloads(self):
        files = launcher.build_sweep_download_list("v0_control_pr1979")
        self.assertIn("ttt_sweep/v0_control_pr1979/variant_result.json", files)
        self.assertNotIn(
            "ttt_sweep/v1_rank128_alpha192/variant_result.json", files
        )
        self.assertNotIn(
            "ttt_sweep/v_sliding_window_control/sliding_eval_summary.json", files
        )

    def test_sliding_variant_downloads_summary(self):
        files = launcher.build_sweep_download_list("v_sliding_window_control")
        self.assertIn(
            "ttt_sweep/v_sliding_window_control/sliding_eval_summary.json", files
        )


class TestDefaultConstants(unittest.TestCase):
    """Verify 4-hour default constants are defined."""

    def test_4h_constants_exist(self):
        self.assertEqual(launcher.DEFAULT_4H_MAX_WALLCLOCK, 14400)
        self.assertEqual(launcher.DEFAULT_4H_MAX_MINUTES, 360)
        self.assertEqual(launcher.DEFAULT_4H_EXPORT_MINUTES, "60,120,180,240")
        self.assertEqual(launcher.DEFAULT_4H_RESUME_SAVE_MINUTES, "30,60,90,120,150,180,210,240")
        self.assertEqual(launcher.DEFAULT_4H_ITERATIONS, 100000)


# ---------------------------------------------------------------------------
# Phase 3: Resumed 6h-horizon continuation — 4-GPU-only safety + labeling
# ---------------------------------------------------------------------------

class TestContinuationGPUControl(unittest.TestCase):
    """--continuation-label forces --num-gpus=4 and rejects other GPU counts."""

    def _parse(self, *extra_args):
        old_argv = sys.argv
        try:
            sys.argv = ["prog", "--dry-run"] + list(extra_args)
            # Use the launcher's own parser builder
            args = launcher.build_arg_parser().parse_args()
            launcher.apply_post_parse_defaults(args)
            return args
        finally:
            sys.argv = old_argv

    def test_num_gpus_default_is_8(self):
        args = self._parse()
        self.assertEqual(args.num_gpus, 8)

    def test_num_gpus_explicit_4(self):
        args = self._parse("--num-gpus", "4")
        self.assertEqual(args.num_gpus, 4)

    def test_continuation_label_forces_4_gpus(self):
        args = self._parse(
            "--continuation-label", "resumed_6h_horizon",
            "--resume-from", "/some/path",
        )
        self.assertEqual(args.num_gpus, 4)

    def test_continuation_label_rejects_8_gpus(self):
        """Explicitly requesting 8 GPUs with a continuation label should error."""
        with self.assertRaises(SystemExit):
            self._parse(
                "--continuation-label", "resumed_6h_horizon",
                "--num-gpus", "8",
                "--resume-from", "/some/path",
            )

    def test_continuation_label_rejects_8_gpus_equals_form(self):
        """--num-gpus=8 (equals form) must also be rejected."""
        with self.assertRaises(SystemExit):
            self._parse(
                "--continuation-label", "resumed_6h_horizon",
                "--num-gpus=8",
                "--resume-from", "/some/path",
            )

    def test_continuation_label_allows_4_gpus_explicit(self):
        args = self._parse(
            "--continuation-label", "resumed_6h_horizon",
            "--num-gpus", "4",
            "--resume-from", "/some/path",
        )
        self.assertEqual(args.num_gpus, 4)


class TestContinuationResumePathWiring(unittest.TestCase):
    """Resume path from captured snapshot flows into the seed command with on-pod rewrite."""

    def _make_args(self, **overrides):
        defaults = dict(
            seed=42, export_minutes="60,120,180,240", max_wallclock=21600,
            export_mode="light", enable_resume=True,
            resume_save_minutes="30,60,90,120,150,180,210,240,270,300,330,360",
            resume_keep_last=3,
            resume_from=str(SNAPSHOT_DIR),
            iterations=100000,
            run_ttt_sweep_after_train=False,
            ttt_sweep_variants=None, ttt_max_minutes_per_variant=20,
            num_gpus=4,
            continuation_label="resumed_6h_horizon",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_resume_from_rewritten_to_onpod_manifest(self):
        """When resume_from is a local dir with continuation_label, RESUME_FROM uses on-pod path."""
        args = self._make_args()
        cmd = launcher.build_seed_cmd(args)
        expected = "RESUME_FROM={}/resume_manifest.json".format(
            launcher.ONPOD_RESUME_SNAPSHOT_PATH)
        self.assertIn(expected, cmd)

    def test_resume_from_not_rewritten_without_label(self):
        """Without continuation_label, RESUME_FROM passes through as-is."""
        args = self._make_args(continuation_label=None)
        cmd = launcher.build_seed_cmd(args)
        # Local dir path passes through (non-continuation legacy behavior)
        self.assertIn("RESUME_FROM={}".format(SNAPSHOT_DIR), cmd)

    def test_resume_enabled_in_seed_cmd(self):
        args = self._make_args()
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("RESUME_ENABLED=1", cmd)


class TestContinuationLabelInDryRun(unittest.TestCase):
    """Dry-run output should reflect 4 GPUs and the continuation label."""

    def _parse(self, *extra_args):
        old_argv = sys.argv
        try:
            sys.argv = ["prog", "--dry-run"] + list(extra_args)
            args = launcher.build_arg_parser().parse_args()
            launcher.apply_post_parse_defaults(args)
            return args
        finally:
            sys.argv = old_argv

    def test_dry_run_shows_4_gpus(self):
        import io
        from contextlib import redirect_stdout
        args = self._parse(
            "--continuation-label", "resumed_6h_horizon",
            "--num-gpus", "4",
            "--resume-from", "/some/path",
        )
        # build_dry_run_summary should show 4 GPUs
        summary = launcher.build_dry_run_summary(args)
        self.assertIn("GPUs: 4", summary)
        self.assertNotIn("GPUs: 8", summary)

    def test_dry_run_shows_continuation_label(self):
        args = self._parse(
            "--continuation-label", "resumed_6h_horizon",
            "--num-gpus", "4",
            "--resume-from", "/some/path",
        )
        summary = launcher.build_dry_run_summary(args)
        self.assertIn("resumed_6h_horizon", summary)

    def test_dry_run_cost_uses_actual_gpu_count(self):
        args = self._parse(
            "--continuation-label", "resumed_6h_horizon",
            "--num-gpus", "4",
            "--resume-from", "/some/path",
            "--max-minutes", "420",
        )
        summary = launcher.build_dry_run_summary(args)
        # Cost for 4 GPUs × $3.50/hr × 7h = $98.00
        expected_cost = 4 * launcher.H100_COST_PER_GPU_HR * (420 / 60.0)
        self.assertIn("Est cost: ${:.2f}".format(expected_cost), summary)


class TestContinuationPodNaming(unittest.TestCase):
    """Pod name should include continuation label when set."""

    def _parse(self, *extra_args):
        old_argv = sys.argv
        try:
            sys.argv = ["prog", "--dry-run"] + list(extra_args)
            args = launcher.build_arg_parser().parse_args()
            launcher.apply_post_parse_defaults(args)
            return args
        finally:
            sys.argv = old_argv

    def test_pod_name_includes_label(self):
        args = self._parse(
            "--continuation-label", "resumed_6h_horizon",
            "--num-gpus", "4",
            "--resume-from", "/some/path",
        )
        pod_name = launcher.build_pod_name(args)
        self.assertIn("resumed-6h-horizon", pod_name)

    def test_pod_name_default(self):
        args = self._parse()
        pod_name = launcher.build_pod_name(args)
        self.assertEqual(pod_name, "pgolf-longtrain-scaling")


class TestScheduleHorizon(unittest.TestCase):
    """--schedule-horizon passes SCHEDULE_HORIZON_SECONDS into the seed command."""

    def _parse(self, *extra_args):
        old_argv = sys.argv
        try:
            sys.argv = ["prog", "--dry-run"] + list(extra_args)
            args = launcher.build_arg_parser().parse_args()
            launcher.apply_post_parse_defaults(args)
            return args
        finally:
            sys.argv = old_argv

    def _make_args(self, **overrides):
        defaults = dict(
            seed=42, export_minutes="60,120,180,240", max_wallclock=21600,
            export_mode="light", enable_resume=True,
            resume_save_minutes="30,60,90,120,150,180,210,240,270,300,330,360",
            resume_keep_last=3, resume_from="/some/path",
            iterations=100000,
            run_ttt_sweep_after_train=False,
            ttt_sweep_variants=None, ttt_max_minutes_per_variant=20,
            num_gpus=4, continuation_label="resumed_6h_horizon",
            schedule_horizon=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_arg_default_is_none(self):
        args = self._parse()
        self.assertIsNone(args.schedule_horizon)

    def test_arg_parses_value(self):
        args = self._parse("--schedule-horizon", "21600")
        self.assertEqual(args.schedule_horizon, 21600)

    def test_env_emitted_in_seed_cmd(self):
        args = self._make_args(schedule_horizon=21600)
        cmd = launcher.build_seed_cmd(args)
        self.assertIn("SCHEDULE_HORIZON_SECONDS=21600", cmd)

    def test_env_not_emitted_when_none(self):
        args = self._make_args(schedule_horizon=None)
        cmd = launcher.build_seed_cmd(args)
        self.assertNotIn("SCHEDULE_HORIZON_SECONDS", cmd)

    def test_dry_run_summary_includes_horizon(self):
        args = self._parse(
            "--schedule-horizon", "21600",
            "--continuation-label", "resumed_6h_horizon",
            "--resume-from", "/some/path",
        )
        summary = launcher.build_dry_run_summary(args)
        self.assertIn("Schedule horizon: 21600s", summary)

    def test_dry_run_summary_omits_when_unset(self):
        args = self._parse()
        summary = launcher.build_dry_run_summary(args)
        self.assertNotIn("Schedule horizon", summary)


class TestContinuationSSHUploadWiring(unittest.TestCase):
    """SSH upload specs are built from the real local snapshot directory."""

    def test_build_resume_ssh_uploads_real_snapshot(self):
        """Verifies SSH upload specs for the actual captured snapshot."""
        if not SNAPSHOT_DIR.exists():
            self.skipTest("Snapshot directory not available")
        specs = launcher.build_resume_ssh_uploads(str(SNAPSHOT_DIR))
        # Should have manifest + 4 rank files + stdout_tail.txt = 6 files
        self.assertGreaterEqual(len(specs), 5)
        # Each spec is "local_path:arcname"
        arc_names = [s.split(":", 1)[1] for s in specs]
        self.assertIn("resume_snapshot/resume_manifest.json", arc_names)
        self.assertIn("resume_snapshot/resume_rank0_step36452.pt", arc_names)
        self.assertIn("resume_snapshot/resume_rank3_step36452.pt", arc_names)

    def test_build_resume_ssh_uploads_missing_dir(self):
        """Non-existent directory raises SystemExit."""
        with self.assertRaises(SystemExit):
            launcher.build_resume_ssh_uploads("/nonexistent/path/xyz")

    def test_run_standard_wires_ssh_upload_for_continuation(self):
        """run_standard appends --ssh-upload args for continuation with local snapshot."""
        if not SNAPSHOT_DIR.exists():
            self.skipTest("Snapshot directory not available")
        args = argparse.Namespace(
            seed=42, num_gpus=4, max_minutes=420,
            continuation_label="resumed_6h_horizon",
            resume_from=str(SNAPSHOT_DIR),
            export_minutes="60,120,180,240,300,360",
            run_ttt_sweep_after_train=False,
            results_dir=None,
        )
        # Capture sys.argv as set by run_standard (http_main is stubbed)
        captured_argv = []
        original_http_main = launcher.http_main
        def capture_main():
            captured_argv.extend(sys.argv)
        launcher.http_main = capture_main
        try:
            launcher.run_standard(args, "echo test", ["status.txt"], "train_gpt.py")
        finally:
            launcher.http_main = original_http_main
        # Verify SSH upload flags present
        self.assertIn("--ssh-upload", captured_argv)
        # Find all ssh-upload values
        ssh_uploads = []
        for i, v in enumerate(captured_argv):
            if v == "--ssh-upload" and i + 1 < len(captured_argv):
                ssh_uploads.append(captured_argv[i + 1])
        # Must include the manifest
        arcs = [s.split(":", 1)[1] for s in ssh_uploads]
        self.assertIn("resume_snapshot/resume_manifest.json", arcs)
        # Must include rank files
        self.assertTrue(any("resume_rank0" in a for a in arcs))

    def test_seed_cmd_resume_from_rewritten_for_real_snapshot(self):
        """build_seed_cmd rewrites RESUME_FROM to on-pod manifest path for real snapshot."""
        if not SNAPSHOT_DIR.exists():
            self.skipTest("Snapshot directory not available")
        args = argparse.Namespace(
            seed=42, export_minutes="60,120,180,240,300,360",
            max_wallclock=21600, export_mode="light",
            enable_resume=True,
            resume_save_minutes="30,60,90,120,150,180,210,240,270,300,330,360",
            resume_keep_last=3,
            resume_from=str(SNAPSHOT_DIR),
            iterations=100000,
            run_ttt_sweep_after_train=False,
            ttt_sweep_variants=None, ttt_max_minutes_per_variant=20,
            num_gpus=4,
            continuation_label="resumed_6h_horizon",
        )
        cmd = launcher.build_seed_cmd(args)
        # Should point to the on-pod manifest, not the local path
        self.assertIn(
            "RESUME_FROM=/root/rehearsal_src/resume_snapshot/resume_manifest.json",
            cmd,
        )
        self.assertNotIn(str(SNAPSHOT_DIR), cmd)


if __name__ == "__main__":
    unittest.main()
