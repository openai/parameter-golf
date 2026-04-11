"""Tests for scripts/causal/experiment_runner.py — Paired Seed Ablation (C5)."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

from scripts.causal.experiment_runner import (
    _classify_condition,
    load_config,
    parse_last_val_bpb,
    run_condition,
    validate_config,
)


# ===== Fixtures =============================================================

def _write_config(path, script="train_gpt_mlx.py", env_overrides=None, description="test"):
    cfg = {
        "script": script,
        "env_overrides": env_overrides or {"ITERATIONS": "10"},
        "description": description,
    }
    Path(path).write_text(json.dumps(cfg))
    return cfg


# ===== 1. Config validation ==================================================

class TestConfigValidation:
    def test_reject_missing_script_field(self, tmp_path):
        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text(json.dumps({"env_overrides": {"A": "1"}}))
        with pytest.raises((ValueError, KeyError)):
            validate_config(load_config(str(cfg_path)))

    def test_reject_non_string_env_overrides(self, tmp_path):
        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text(json.dumps({
            "script": "train_gpt_mlx.py",
            "env_overrides": {"ITERATIONS": 10},  # int, not str
        }))
        with pytest.raises((ValueError, TypeError)):
            validate_config(load_config(str(cfg_path)))

    def test_reject_missing_env_overrides(self, tmp_path):
        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text(json.dumps({"script": "train_gpt_mlx.py"}))
        with pytest.raises((ValueError, KeyError)):
            validate_config(load_config(str(cfg_path)))

    def test_valid_config_accepted(self, tmp_path):
        cfg_path = tmp_path / "good.json"
        _write_config(cfg_path)
        cfg = validate_config(load_config(str(cfg_path)))
        assert cfg["script"] == "train_gpt_mlx.py"


# ===== 2. stdout BPB parsing =================================================

class TestStdoutBpbParsing:
    def test_parses_last_occurrence(self):
        stdout = (
            "step 100 | val_bpb:1.5000\n"
            "step 200 | val_bpb:1.3000\n"
            "step 300 | val_bpb:1.2345\n"
        )
        assert parse_last_val_bpb(stdout) == pytest.approx(1.2345)

    def test_single_occurrence(self):
        stdout = "training done val_bpb:2.0001\n"
        assert parse_last_val_bpb(stdout) == pytest.approx(2.0001)

    def test_embedded_in_other_text(self):
        stdout = "INFO blah blah val_bpb:0.9876 more text\nfinal line\n"
        assert parse_last_val_bpb(stdout) == pytest.approx(0.9876)


# ===== 3. Partial failure ====================================================

class TestPartialFailure:
    def test_one_of_three_seeds_fails(self, tmp_path):
        """When 1 of 3 seeds fails, results should have n=2 and reduced_power flag."""
        # Create a mock script that fails on seed=137 and succeeds otherwise
        mock_script = tmp_path / "mock_train.py"
        mock_script.write_text(
            "import os, sys\n"
            "seed = os.environ.get('SEED', '0')\n"
            "if seed == '137':\n"
            "    sys.exit(1)\n"
            "print(f'val_bpb:1.{seed}')\n"
        )
        cfg = {
            "script": str(mock_script),
            "env_overrides": {},
            "description": "partial failure test",
        }
        results = run_condition(cfg, seeds=[42, 137, 256], timeout=30)
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        assert len(successful) == 2
        assert len(failed) == 1
        assert failed[0]["seed"] == 137

        # Verify reduced_power classification
        status = _classify_condition(results)
        assert status["status"] == "reduced_power"
        assert status["reduced_power"] is True
        assert status["n_successful"] == 2
        assert status["n_failed"] == 1

    def test_all_seeds_fail(self, tmp_path):
        mock_script = tmp_path / "mock_fail.py"
        mock_script.write_text("import sys; sys.exit(1)\n")
        cfg = {
            "script": str(mock_script),
            "env_overrides": {},
            "description": "total failure test",
        }
        results = run_condition(cfg, seeds=[42, 137, 256], timeout=30)
        assert all("error" in r for r in results)

        # Verify failed classification
        status = _classify_condition(results)
        assert status["status"] == "failed"


# ===== 4. Timeout handling ===================================================

class TestTimeoutHandling:
    def test_subprocess_timeout(self, tmp_path):
        """Subprocess that exceeds timeout should be captured as an error."""
        mock_script = tmp_path / "mock_slow.py"
        mock_script.write_text("import time; time.sleep(60)\n")
        cfg = {
            "script": str(mock_script),
            "env_overrides": {},
            "description": "timeout test",
        }
        results = run_condition(cfg, seeds=[42], timeout=2)
        assert len(results) == 1
        assert "error" in results[0]
        assert "timeout" in results[0]["error"].lower() or "timed out" in results[0]["error"].lower()


# ===== 5. Truncated stdout ===================================================

class TestTruncatedStdout:
    def test_no_val_bpb_line(self):
        """When no val_bpb line is found, parse_last_val_bpb returns None."""
        stdout = "training started\nstep 100\ntraining done\n"
        result = parse_last_val_bpb(stdout)
        assert result is None

    def test_empty_stdout(self):
        result = parse_last_val_bpb("")
        assert result is None

    def test_run_with_no_bpb_output(self, tmp_path):
        """Subprocess that outputs no val_bpb line produces result with error."""
        mock_script = tmp_path / "mock_no_bpb.py"
        mock_script.write_text("print('no metrics here')\n")
        cfg = {
            "script": str(mock_script),
            "env_overrides": {},
            "description": "no bpb test",
        }
        results = run_condition(cfg, seeds=[42], timeout=30)
        assert len(results) == 1
        # Should have error or val_bpb should be None
        r = results[0]
        assert r.get("val_bpb") is None or "error" in r


# ===== 6. Integration dry-run with mock script ================================

class TestIntegrationDryRun:
    def test_full_pipeline_with_mock(self, tmp_path):
        """Run the full pipeline with a trivial mock script."""
        mock_script = tmp_path / "mock_train.py"
        mock_script.write_text(
            "import os\n"
            "seed = os.environ.get('SEED', '0')\n"
            "print(f'step 100 | val_bpb:1.{seed}')\n"
            "print(f'step 200 | val_bpb:1.23{seed[-1]}')\n"
        )
        treatment_cfg_path = tmp_path / "treatment.json"
        control_cfg_path = tmp_path / "control.json"
        output_path = tmp_path / "raw_runs.json"

        _write_config(treatment_cfg_path, script=str(mock_script),
                      env_overrides={"ITERATIONS": "10"}, description="treatment")
        _write_config(control_cfg_path, script=str(mock_script),
                      env_overrides={"ITERATIONS": "10"}, description="control")

        from scripts.causal.experiment_runner import main

        main([
            "--treatment", str(treatment_cfg_path),
            "--control", str(control_cfg_path),
            "--output", str(output_path),
            "--seeds", "42,137,256",
            "--platform", "mlx",
        ])

        assert output_path.exists()
        data = json.loads(output_path.read_text())

        # Validate schema
        assert data["platform"] == "mlx"
        assert data["seeds"] == [42, 137, 256]
        assert "treatment" in data
        assert "control" in data
        assert "config" in data["treatment"]
        assert "results" in data["treatment"]
        assert len(data["treatment"]["results"]) == 3
        assert len(data["control"]["results"]) == 3

        # Each result should have seed and val_bpb
        for r in data["treatment"]["results"]:
            assert "seed" in r
            assert "val_bpb" in r
            assert isinstance(r["val_bpb"], float)
