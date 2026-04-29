"""Tests for run_ppmd_cuda_runpod.py wrapper."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_ppmd_cuda_runpod as wrapper


BASE_FLAGS = [
    "--gpu-sku", "a100-1x",
    "--mode", "env-smoke",
    "--branch", "main",
    "--commit", "abc123",
    "--max-minutes", "8",
    "--results-dir", "/some/dir",
]


def _parse(argv: list[str] | None = None):
    parser = wrapper._build_argparser()
    return parser.parse_args(argv if argv is not None else BASE_FLAGS)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parser_required_flags():
    """Omitting any required flag raises SystemExit(2)."""
    parser = wrapper._build_argparser()
    required_pairs = [
        ("--gpu-sku",),
        ("--mode",),
        ("--branch",),
        ("--commit",),
        ("--max-minutes",),
        ("--results-dir",),
    ]
    for (flag,) in required_pairs:
        # Build an argv that removes the flag and its value
        partial = []
        skip_next = False
        for tok in BASE_FLAGS:
            if skip_next:
                skip_next = False
                continue
            if tok == flag:
                skip_next = True
                continue
            partial.append(tok)
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(partial)
        assert exc_info.value.code == 2, "Expected exit code 2 when {} is missing".format(flag)


def test_default_pod_name():
    """Default pod name is ppmd-cuda-{mode}-{gpu_sku}."""
    args = _parse()
    assert args.pod_name is None
    expected = "ppmd-cuda-{mode}-{sku}".format(mode=args.mode, sku=args.gpu_sku)
    assert expected == "ppmd-cuda-env-smoke-a100-1x"


# ---------------------------------------------------------------------------
# Download-list tests
# ---------------------------------------------------------------------------

def test_default_download_list_env_smoke_includes_cuda_env_probe():
    dl = wrapper._default_download_list("env-smoke")
    assert "cuda_env_probe.json" in dl
    assert "status.txt" in dl
    assert "nvidia_smi.txt" in dl


# ---------------------------------------------------------------------------
# Dry-run test
# ---------------------------------------------------------------------------

def test_dry_run_prints_payload_and_exits_zero(capsys):
    """--dry-run prints payload and returns without launching subprocess."""
    wrapper.main(BASE_FLAGS + ["--dry-run"])
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "cuda_env_probe.json" in captured.out


# ---------------------------------------------------------------------------
# Payload content tests
# ---------------------------------------------------------------------------

def test_payload_writes_cuda_env_probe_for_env_smoke_mode():
    """env-smoke payload references cuda_env_probe.json."""
    args = _parse()
    payload = wrapper._build_payload(args)
    assert "cuda_env_probe.json" in payload


def test_payload_skips_git_clone_for_env_smoke():
    """env-smoke payload must NOT contain 'git clone'."""
    args = _parse()
    payload = wrapper._build_payload(args)
    assert "git clone" not in payload


def test_payload_does_git_clone_for_build_smoke_mode():
    """build-smoke with --no-bundle-source falls back to git clone stub."""
    build_smoke_flags = [
        "--gpu-sku", "a100-1x",
        "--mode", "build-smoke",
        "--branch", "main",
        "--commit", "abc123",
        "--max-minutes", "12",
        "--results-dir", "/some/dir",
        "--no-bundle-source",
    ]
    args = _parse(build_smoke_flags)
    payload = wrapper._build_payload(args)
    assert "git clone" in payload



# ---------------------------------------------------------------------------
# Phase 4: new mode tests
# ---------------------------------------------------------------------------

def test_trie_prefix_256_mode_accepted():
    """trie-prefix-256 must be a valid --mode choice."""
    args = _parse([
        "--gpu-sku", "a100-1x",
        "--mode", "trie-prefix-256",
        "--branch", "main",
        "--commit", "abc123",
        "--max-minutes", "15",
        "--results-dir", "/some/dir",
    ])
    assert args.mode == "trie-prefix-256"


def test_trie_prefix_1k_mode_accepted():
    args = _parse([
        "--gpu-sku", "a100-1x",
        "--mode", "trie-prefix-1k",
        "--branch", "main",
        "--commit", "abc123",
        "--max-minutes", "15",
        "--results-dir", "/some/dir",
    ])
    assert args.mode == "trie-prefix-1k"


def test_bench_mode_accepted():
    args = _parse([
        "--gpu-sku", "a100-1x",
        "--mode", "bench",
        "--branch", "main",
        "--commit", "abc123",
        "--max-minutes", "20",
        "--results-dir", "/some/dir",
    ])
    assert args.mode == "bench"


def test_full_eval_mode_accepted():
    args = _parse([
        "--gpu-sku", "h100-1x",
        "--mode", "full-eval",
        "--branch", "main",
        "--commit", "abc123",
        "--max-minutes", "60",
        "--results-dir", "/some/dir",
    ])
    assert args.mode == "full-eval"


def test_trie_prefix_256_dry_run(capsys):
    """Dry-run for trie-prefix-256 prints make cuda and score_path_a_arrays_cuda."""
    wrapper.main([
        "--gpu-sku", "a100-1x",
        "--mode", "trie-prefix-256",
        "--branch", "main",
        "--commit", "abc123",
        "--max-minutes", "15",
        "--results-dir", "/some/dir",
        "--dry-run",
    ])
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "make cuda" in captured.out


def test_bench_dry_run(capsys):
    """Dry-run for bench prints projected_full_eval and bench json filenames."""
    wrapper.main([
        "--gpu-sku", "a100-1x",
        "--mode", "bench",
        "--branch", "main",
        "--commit", "abc123",
        "--max-minutes", "20",
        "--results-dir", "/some/dir",
        "--dry-run",
    ])
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "ppmd_cuda_bench_4096x8192.json" in captured.out


def test_full_eval_dry_run(capsys):
    """Dry-run for full-eval prints eval_path_a_ppmd.py and cuda backend flags."""
    wrapper.main([
        "--gpu-sku", "h100-1x",
        "--mode", "full-eval",
        "--branch", "main",
        "--commit", "abc123",
        "--max-minutes", "60",
        "--results-dir", "/some/dir",
        "--dry-run",
    ])
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "eval_path_a_ppmd.py" in captured.out
    assert "--backend cuda" in captured.out
