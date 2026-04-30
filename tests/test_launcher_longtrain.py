"""Tests for run_longtrain_scaling.py CLI args and command building."""

import argparse
import os
import sys
import unittest

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


class TestDefaultConstants(unittest.TestCase):
    """Verify 4-hour default constants are defined."""

    def test_4h_constants_exist(self):
        self.assertEqual(launcher.DEFAULT_4H_MAX_WALLCLOCK, 14400)
        self.assertEqual(launcher.DEFAULT_4H_MAX_MINUTES, 360)
        self.assertEqual(launcher.DEFAULT_4H_EXPORT_MINUTES, "60,120,180,240")
        self.assertEqual(launcher.DEFAULT_4H_RESUME_SAVE_MINUTES, "30,60,90,120,150,180,210,240")
        self.assertEqual(launcher.DEFAULT_4H_ITERATIONS, 100000)


if __name__ == "__main__":
    unittest.main()
