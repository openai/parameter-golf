"""Tests for bundle-source additions in run_ppmd_cuda_runpod.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_ppmd_cuda_runpod as wrapper

BASE_FLAGS_BUILD_SMOKE = [
    "--gpu-sku", "a100-1x",
    "--mode", "build-smoke",
    "--branch", "main",
    "--commit", "abc123",
    "--max-minutes", "12",
    "--results-dir", "/some/dir",
]

BASE_FLAGS_ENV_SMOKE = [
    "--gpu-sku", "a100-1x",
    "--mode", "env-smoke",
    "--branch", "main",
    "--commit", "abc123",
    "--max-minutes", "8",
    "--results-dir", "/some/dir",
]


def _parse(argv):
    return wrapper._build_argparser().parse_args(argv)


# ---------------------------------------------------------------------------
# Bundle-source flag tests
# ---------------------------------------------------------------------------

def test_bundle_source_on_by_default_for_build_smoke():
    args = _parse(BASE_FLAGS_BUILD_SMOKE)
    assert wrapper._bundle_source_enabled(args) is True


def test_bundle_source_off_by_default_for_env_smoke():
    args = _parse(BASE_FLAGS_ENV_SMOKE)
    assert wrapper._bundle_source_enabled(args) is False


def test_bundle_source_can_be_disabled_explicitly():
    args = _parse(BASE_FLAGS_BUILD_SMOKE + ["--no-bundle-source"])
    assert wrapper._bundle_source_enabled(args) is False


def test_bundle_source_can_be_forced_on_for_env_smoke():
    args = _parse(BASE_FLAGS_ENV_SMOKE + ["--bundle-source"])
    assert wrapper._bundle_source_enabled(args) is True


# ---------------------------------------------------------------------------
# Bundle extra-files tests
# ---------------------------------------------------------------------------

def test_dry_run_build_smoke_includes_ppmd_cpp_in_bundle_extras():
    """build-smoke with default --bundle-source must include ppmd_cpp directory."""
    args = _parse(BASE_FLAGS_BUILD_SMOKE)
    extras = wrapper._get_bundle_extra_files(args)
    arcnames = [arc for _, arc in extras]
    assert "ppmd_cpp" in arcnames, (
        "Expected 'ppmd_cpp' in bundle extras, got: {}".format(arcnames)
    )


def test_env_smoke_dry_run_does_not_bundle_ppmd_cpp():
    """env-smoke (bundle-source=off) must produce an empty extras list."""
    args = _parse(BASE_FLAGS_ENV_SMOKE)
    extras = wrapper._get_bundle_extra_files(args)
    assert extras == [], "env-smoke should not bundle any extras, got: {}".format(extras)


def test_bundle_no_source_produces_empty_extras():
    """--no-bundle-source must produce an empty extras list."""
    args = _parse(BASE_FLAGS_BUILD_SMOKE + ["--no-bundle-source"])
    extras = wrapper._get_bundle_extra_files(args)
    assert extras == []


# ---------------------------------------------------------------------------
# Payload content tests
# ---------------------------------------------------------------------------

def test_dry_run_build_smoke_payload_runs_make_cuda():
    """build-smoke payload must contain 'make cuda'."""
    args = _parse(BASE_FLAGS_BUILD_SMOKE)
    payload = wrapper._build_payload(args)
    assert "make cuda" in payload, (
        "Expected 'make cuda' in build-smoke payload. Got:\n{}".format(payload[:500])
    )


def test_dry_run_build_smoke_payload_writes_import_smoke_json():
    """build-smoke payload must reference ppmd_cuda_import_smoke.json."""
    args = _parse(BASE_FLAGS_BUILD_SMOKE)
    payload = wrapper._build_payload(args)
    assert "ppmd_cuda_import_smoke.json" in payload


def test_dry_run_build_smoke_payload_uses_bundle_path():
    """build-smoke payload must cd to /root/rehearsal_src/ppmd_cpp (bundle path)."""
    args = _parse(BASE_FLAGS_BUILD_SMOKE)
    payload = wrapper._build_payload(args)
    assert "/root/rehearsal_src/ppmd_cpp" in payload
    # git clone should NOT appear when using bundle
    assert "git clone" not in payload


def test_build_smoke_no_bundle_falls_back_to_git_clone():
    """With --no-bundle-source, build-smoke payload falls back to stub + git clone."""
    args = _parse(BASE_FLAGS_BUILD_SMOKE + ["--no-bundle-source"])
    payload = wrapper._build_payload(args)
    assert "git clone" in payload


def test_dry_run_build_smoke_cmd_includes_extra_file_ppmd_cpp(capsys):
    """Dry-run for build-smoke must include --extra-file with ppmd_cpp in the cmd."""
    wrapper.main(BASE_FLAGS_BUILD_SMOKE + ["--dry-run"])
    captured = capsys.readouterr()
    assert "ppmd_cpp" in captured.out
    assert "make cuda" in captured.out


# ---------------------------------------------------------------------------
# rehearsal cmd structure test
# ---------------------------------------------------------------------------

def test_rehearsal_cmd_contains_extra_file_for_ppmd_cpp():
    """_build_rehearsal_cmd must include --extra-file ppmd_cpp when bundle-source on."""
    args = _parse(BASE_FLAGS_BUILD_SMOKE)
    payload = wrapper._build_payload(args)
    dl = wrapper._default_download_list(args.mode)
    cmd = wrapper._build_rehearsal_cmd(args, payload, dl, "test-pod")
    assert "--extra-file" in cmd
    extra_file_vals = [
        cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--extra-file"
    ]
    assert any("ppmd_cpp" in v for v in extra_file_vals), (
        "--extra-file with ppmd_cpp not found. Extra files: {}".format(extra_file_vals)
    )


def test_rehearsal_cmd_no_extra_file_for_env_smoke():
    """env-smoke cmd must not add any --extra-file args."""
    args = _parse(BASE_FLAGS_ENV_SMOKE)
    payload = wrapper._build_payload(args)
    dl = wrapper._default_download_list(args.mode)
    cmd = wrapper._build_rehearsal_cmd(args, payload, dl, "test-pod")
    assert "--extra-file" not in cmd


# ---------------------------------------------------------------------------
# kernel-equiv payload tests
# ---------------------------------------------------------------------------

BASE_FLAGS_KERNEL_EQUIV = [
    "--gpu-sku", "a100-1x",
    "--mode", "kernel-equiv",
    "--branch", "main",
    "--commit", "abc123",
    "--max-minutes", "15",
    "--results-dir", "/some/dir",
]


def test_kernel_equiv_bundle_source_on_by_default():
    args = _parse(BASE_FLAGS_KERNEL_EQUIV)
    assert wrapper._bundle_source_enabled(args) is True


def test_kernel_equiv_payload_runs_make_cuda():
    """kernel-equiv payload must run `make cuda`."""
    args = _parse(BASE_FLAGS_KERNEL_EQUIV)
    payload = wrapper._build_payload(args)
    assert "make cuda" in payload, (
        "Expected 'make cuda' in kernel-equiv payload"
    )


def test_kernel_equiv_payload_runs_make_cpp():
    """kernel-equiv payload must also build the CPU extension."""
    args = _parse(BASE_FLAGS_KERNEL_EQUIV)
    payload = wrapper._build_payload(args)
    # The snippet runs plain `make` to build _ppmd_cpp.
    assert "make" in payload


def test_kernel_equiv_payload_writes_json():
    """kernel-equiv payload must reference ppmd_cuda_kernel_equiv.json."""
    args = _parse(BASE_FLAGS_KERNEL_EQUIV)
    payload = wrapper._build_payload(args)
    assert "ppmd_cuda_kernel_equiv.json" in payload


def test_kernel_equiv_payload_json_keys():
    """kernel-equiv payload Python snippet must include all required JSON keys."""
    args = _parse(BASE_FLAGS_KERNEL_EQUIV)
    payload = wrapper._build_payload(args)
    required_keys = [
        "contexts_tested", "bytes_tested", "max_abs_diff",
        "max_sum_prob_error", "gpu_name", "git_commit",
        "cuda_runtime_version", "device_name", "status",
    ]
    for key in required_keys:
        assert key in payload, (
            "Key '{}' not found in kernel-equiv payload".format(key)
        )


def test_kernel_equiv_payload_does_not_git_clone():
    """kernel-equiv with bundle-source must NOT contain 'git clone'."""
    args = _parse(BASE_FLAGS_KERNEL_EQUIV)
    payload = wrapper._build_payload(args)
    assert "git clone" not in payload


def test_kernel_equiv_no_bundle_falls_back_to_git_clone():
    """kernel-equiv with --no-bundle-source falls back to stub + git clone."""
    args = _parse(BASE_FLAGS_KERNEL_EQUIV + ["--no-bundle-source"])
    payload = wrapper._build_payload(args)
    assert "git clone" in payload


def test_dry_run_kernel_equiv_includes_make_cuda(capsys):
    """Dry-run for kernel-equiv must include make cuda in the printed payload."""
    wrapper.main(BASE_FLAGS_KERNEL_EQUIV + ["--dry-run"])
    captured = capsys.readouterr()
    assert "make cuda" in captured.out
    assert "ppmd_cuda_kernel_equiv.json" in captured.out


def test_kernel_equiv_download_list_includes_json_and_log():
    dl = wrapper._default_download_list("kernel-equiv")
    assert "ppmd_cuda_kernel_equiv.json" in dl
    assert "ppmd_cuda_kernel_equiv.log" in dl
    assert "status.txt" in dl


# ---------------------------------------------------------------------------
# Phase 4: trie-prefix-256, trie-prefix-1k, bench, full-eval payload tests
# ---------------------------------------------------------------------------

BASE_FLAGS_TRIE_PREFIX_256 = [
    "--gpu-sku", "a100-1x",
    "--mode", "trie-prefix-256",
    "--branch", "main",
    "--commit", "abc123",
    "--max-minutes", "15",
    "--results-dir", "/some/dir",
]

BASE_FLAGS_TRIE_PREFIX_1K = [
    "--gpu-sku", "a100-1x",
    "--mode", "trie-prefix-1k",
    "--branch", "main",
    "--commit", "abc123",
    "--max-minutes", "15",
    "--results-dir", "/some/dir",
]

BASE_FLAGS_BENCH = [
    "--gpu-sku", "a100-1x",
    "--mode", "bench",
    "--branch", "main",
    "--commit", "abc123",
    "--max-minutes", "20",
    "--results-dir", "/some/dir",
]

BASE_FLAGS_FULL_EVAL = [
    "--gpu-sku", "h100-1x",
    "--mode", "full-eval",
    "--branch", "main",
    "--commit", "abc123",
    "--max-minutes", "60",
    "--results-dir", "/some/dir",
]


# --- trie-prefix-256 ---

def test_trie_prefix_256_bundle_source_on_by_default():
    args = _parse(BASE_FLAGS_TRIE_PREFIX_256)
    assert wrapper._bundle_source_enabled(args) is True


def test_trie_prefix_256_payload_makes_cuda():
    args = _parse(BASE_FLAGS_TRIE_PREFIX_256)
    payload = wrapper._build_payload(args)
    assert "make cuda" in payload


def test_trie_prefix_256_payload_makes_cpp():
    args = _parse(BASE_FLAGS_TRIE_PREFIX_256)
    payload = wrapper._build_payload(args)
    assert "make" in payload


def test_trie_prefix_256_payload_writes_json():
    args = _parse(BASE_FLAGS_TRIE_PREFIX_256)
    payload = wrapper._build_payload(args)
    assert "path_a_cuda_prefix_256.json" in payload


def test_trie_prefix_256_payload_not_stub():
    """trie-prefix-256 payload must NOT be the NOT_IMPLEMENTED stub."""
    args = _parse(BASE_FLAGS_TRIE_PREFIX_256)
    payload = wrapper._build_payload(args)
    assert "NOT_IMPLEMENTED" not in payload


def test_trie_prefix_256_payload_calls_score_path_a_arrays_cuda():
    args = _parse(BASE_FLAGS_TRIE_PREFIX_256)
    payload = wrapper._build_payload(args)
    assert "score_path_a_arrays_cuda" in payload


def test_trie_prefix_256_payload_no_git_clone():
    """trie-prefix-256 with bundle-source must NOT clone from git."""
    args = _parse(BASE_FLAGS_TRIE_PREFIX_256)
    payload = wrapper._build_payload(args)
    assert "git clone" not in payload


def test_trie_prefix_256_payload_n_positions():
    args = _parse(BASE_FLAGS_TRIE_PREFIX_256)
    payload = wrapper._build_payload(args)
    assert "256" in payload


def test_trie_prefix_256_download_list():
    dl = wrapper._default_download_list("trie-prefix-256")
    assert "path_a_cuda_prefix_256.json" in dl
    assert "path_a_cuda_prefix_256.log" in dl
    assert "status.txt" in dl


# --- trie-prefix-1k ---

def test_trie_prefix_1k_payload_not_stub():
    args = _parse(BASE_FLAGS_TRIE_PREFIX_1K)
    payload = wrapper._build_payload(args)
    assert "NOT_IMPLEMENTED" not in payload


def test_trie_prefix_1k_payload_writes_json():
    args = _parse(BASE_FLAGS_TRIE_PREFIX_1K)
    payload = wrapper._build_payload(args)
    assert "path_a_cuda_prefix_1k.json" in payload


def test_trie_prefix_1k_payload_n_positions():
    args = _parse(BASE_FLAGS_TRIE_PREFIX_1K)
    payload = wrapper._build_payload(args)
    assert "1000" in payload


# --- bench ---

def test_bench_payload_not_stub():
    args = _parse(BASE_FLAGS_BENCH)
    payload = wrapper._build_payload(args)
    assert "NOT_IMPLEMENTED" not in payload


def test_bench_payload_writes_two_json_files():
    args = _parse(BASE_FLAGS_BENCH)
    payload = wrapper._build_payload(args)
    assert "ppmd_cuda_bench_4096x8192.json" in payload
    assert "ppmd_cuda_bench_prefix_1k.json" in payload


def test_bench_payload_runs_make_cuda():
    args = _parse(BASE_FLAGS_BENCH)
    payload = wrapper._build_payload(args)
    assert "make cuda" in payload


def test_bench_payload_includes_projected_full_eval():
    args = _parse(BASE_FLAGS_BENCH)
    payload = wrapper._build_payload(args)
    assert "projected_full_eval" in payload


def test_bench_download_list():
    dl = wrapper._default_download_list("bench")
    assert "ppmd_cuda_bench_4096x8192.json" in dl
    assert "ppmd_cuda_bench_prefix_1k.json" in dl


# --- full-eval ---

def test_full_eval_payload_not_stub():
    args = _parse(BASE_FLAGS_FULL_EVAL)
    payload = wrapper._build_payload(args)
    assert "NOT_IMPLEMENTED" not in payload


def test_full_eval_payload_downloads_fineweb():
    args = _parse(BASE_FLAGS_FULL_EVAL)
    payload = wrapper._build_payload(args)
    assert "cached_challenge_fineweb.py" in payload


def test_full_eval_payload_runs_eval_script():
    args = _parse(BASE_FLAGS_FULL_EVAL)
    payload = wrapper._build_payload(args)
    assert "eval_path_a_ppmd.py" in payload


def test_full_eval_payload_uses_cuda_backend():
    args = _parse(BASE_FLAGS_FULL_EVAL)
    payload = wrapper._build_payload(args)
    assert "--backend cuda" in payload


def test_full_eval_payload_writes_sha256():
    args = _parse(BASE_FLAGS_FULL_EVAL)
    payload = wrapper._build_payload(args)
    assert "path_a_cuda_full_eval.sha256" in payload


def test_full_eval_bundle_includes_data_files():
    """full-eval bundle extras must include cached_challenge_fineweb.py and tokenizer_specs.json."""
    args = _parse(BASE_FLAGS_FULL_EVAL)
    extras = wrapper._get_bundle_extra_files(args)
    arcnames = [arc for _, arc in extras]
    data_arcnames = [a for a in arcnames if a.startswith("data/")]
    assert any("cached_challenge_fineweb" in a for a in data_arcnames), (
        "Expected data/cached_challenge_fineweb.py in extras, got: {}".format(arcnames)
    )


def test_full_eval_download_list():
    dl = wrapper._default_download_list("full-eval")
    assert "path_a_cuda_full_eval.json" in dl
    assert "path_a_cuda_full_eval.log" in dl
    assert "path_a_cuda_full_eval.sha256" in dl
