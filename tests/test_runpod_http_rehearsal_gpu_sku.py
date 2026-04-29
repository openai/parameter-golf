"""Tests for --gpu-sku flag additions in runpod_http_rehearsal.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Parser-level tests — use --help or controlled argument parsing to verify
# argparse accepts/rejects SKUs without network calls.
# ---------------------------------------------------------------------------

def test_parser_accepts_gpu_sku_a100_1x(monkeypatch):
    """--gpu-sku a100-1x is a valid choice and --help exits 0."""
    monkeypatch.setattr(
        "sys.argv",
        ["rhr", "--gpu-sku", "a100-1x", "--max-minutes", "1", "--help"],
    )
    import runpod_http_rehearsal  # noqa: F401 – ensure importable
    with pytest.raises(SystemExit) as exc_info:
        runpod_http_rehearsal.main()
    # argparse --help always exits with code 0
    assert exc_info.value.code == 0


def test_parser_rejects_unknown_sku(monkeypatch, capsys):
    """--gpu-sku with an unknown value causes argparse to exit with code 2."""
    monkeypatch.setattr(
        "sys.argv",
        ["rhr", "--gpu-sku", "v100-1x", "--max-minutes", "1"],
    )
    import runpod_http_rehearsal
    with pytest.raises(SystemExit) as exc_info:
        runpod_http_rehearsal.main()
    assert exc_info.value.code == 2


def test_parser_gpus_sku_mismatch_errors(monkeypatch, capsys):
    """--gpus 1 --gpu-sku a100-2x conflicts (a100-2x requires 2 GPUs) → error."""
    monkeypatch.setattr(
        "sys.argv",
        ["rhr", "--gpu-sku", "a100-2x", "--gpus", "1", "--max-minutes", "1"],
    )
    import runpod_http_rehearsal
    # balance() would be called next; patch it to avoid network but let the
    # mismatch check in main() fire first.
    monkeypatch.setattr(runpod_http_rehearsal, "balance", lambda: (100.0, 0.0))
    with pytest.raises(SystemExit) as exc_info:
        runpod_http_rehearsal.main()
    # Must be a non-zero exit
    assert exc_info.value.code not in (0, None)


# ---------------------------------------------------------------------------
# Launcher state test — mock all network calls and verify the written
# launcher_state.json contains cost_per_hr, gpu_sku, and gpu_type_id.
# ---------------------------------------------------------------------------

def test_launcher_state_includes_cost_per_hr(monkeypatch, tmp_path):
    """After pod creation, cost_per_hr and gpu_sku appear in launcher_state.json."""
    import runpod_http_rehearsal

    # Satisfy _require_api_key without touching network.
    monkeypatch.setenv("RUNPOD_API_KEY", "test")

    # Patch network-touching functions in the module's namespace.
    monkeypatch.setattr(runpod_http_rehearsal, "balance", lambda: (100.0, 0.0))

    fake_pod = {"id": "x", "costPerHr": 1.99, "machineId": "m"}
    monkeypatch.setattr(runpod_http_rehearsal, "create_pod", lambda **kw: fake_pod)

    fake_runtime = {"uptimeInSeconds": 5, "ports": []}
    monkeypatch.setattr(runpod_http_rehearsal, "wait_runtime", lambda pod_id, timeout=600: fake_runtime)

    monkeypatch.setattr(
        runpod_http_rehearsal,
        "wait_startup_readiness_and_maybe_download_status",
        lambda *a, **kw: "RUNNING",
    )
    monkeypatch.setattr(
        runpod_http_rehearsal, "wait_http_proxy", lambda *a, **kw: "DONE"
    )
    monkeypatch.setattr(
        runpod_http_rehearsal, "download_file", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        runpod_http_rehearsal, "terminate_and_wait", lambda pod_id, **kw: True
    )

    results_dir = tmp_path / "test_results"
    results_dir.mkdir()

    monkeypatch.setattr(
        "sys.argv",
        [
            "rhr",
            "--gpu-sku", "a100-1x",
            "--max-minutes", "1",
            "--results-dir", str(results_dir),
        ],
    )

    runpod_http_rehearsal.main()

    state_path = results_dir / "launcher_state.json"
    assert state_path.exists(), "launcher_state.json was not written"
    state = json.loads(state_path.read_text())

    assert state.get("cost_per_hr") == 1.99, "cost_per_hr not stamped correctly"
    assert state.get("gpu_sku") == "a100-1x", "gpu_sku not stamped correctly"
    assert state.get("gpu_type_id") == "NVIDIA A100-SXM4-80GB", "gpu_type_id not stamped correctly"
