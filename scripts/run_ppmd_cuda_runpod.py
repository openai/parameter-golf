#!/usr/bin/env python3
"""Wrapper for running PPM-D CUDA RunPod rehearsal stages.

Generates stage-specific pod payloads and invokes runpod_http_rehearsal.py.
Does NOT launch any pod directly — delegates everything to the rehearsal
launcher which handles bundle, artifact retrieval, and teardown.

Usage (dry-run, no pod launched):
    python3 scripts/run_ppmd_cuda_runpod.py \\
        --gpu-sku a100-1x --mode env-smoke --branch main --commit HEAD \\
        --max-minutes 8 --results-dir results/ppmd_cuda_runpod/01_env_smoke_a100_1x \\
        --dry-run
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import hashlib
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

sys.path.insert(0, str(SCRIPTS_DIR))
from runpod_safe import GPU_SKU_TABLE  # noqa: E402

DEFAULT_DOCKER_IMAGE = "matotezitanka/proteus-pytorch:community"
DEFAULT_REPO_URL = "https://github.com/Christopher-Lee-McClendon/parameter-golf.git"
DEFAULT_RUNTIME_TIMEOUT_SEC = 600

# Modes where source bundling is ON by default (git clone skipped for build-smoke/kernel-equiv).
BUNDLE_SOURCE_MODES = frozenset({
    "build-smoke", "kernel-equiv",
    "trie-prefix-256", "trie-prefix-1k",
    "bench", "full-eval",
})
# Modes where git clone is skipped when bundle-source is active.
BUNDLE_SKIP_CLONE_MODES = frozenset({"build-smoke", "kernel-equiv"})

COMMON_ARTIFACTS = [
    "status.txt",
    "pgolf_exit_code.txt",
    "overall_exit_code.txt",
    "pgolf_stdout.txt",
    "http_server.log",
    "launcher_state.json",
    "nvidia_smi.txt",
    "python_version.txt",
    "git_rev.txt",
    "pip_freeze.txt",
    "nvcc_version.txt",
]

STAGE_ARTIFACTS = {
    "env-smoke": ["cuda_env_probe.json"],
    "build-smoke": ["ppmd_cuda_build.log", "ppmd_cuda_import_smoke.json", "ppmd_cuda_build_manifest.txt"],
    "kernel-equiv": ["ppmd_cuda_kernel_equiv.json", "ppmd_cuda_kernel_equiv.log"],
    "trie-prefix-256": ["path_a_cuda_prefix_256.json", "path_a_cuda_prefix_256.log"],
    "trie-prefix-1k": ["path_a_cuda_prefix_1k.json", "path_a_cuda_prefix_1k.log"],
    "bench": ["ppmd_cuda_bench_4096x8192.json", "ppmd_cuda_bench_prefix_1k.json"],
    "full-eval": ["path_a_cuda_full_eval.json", "path_a_cuda_full_eval.log", "path_a_cuda_full_eval.sha256"],
}

# Stage-specific primary JSON artifact (for NOT_IMPLEMENTED stub)
STAGE_PRIMARY_JSON = {
    "build-smoke": "ppmd_cuda_import_smoke.json",
    "kernel-equiv": "ppmd_cuda_kernel_equiv.json",
    "trie-prefix-256": "path_a_cuda_prefix_256.json",
    "trie-prefix-1k": "path_a_cuda_prefix_1k.json",
    "bench": "ppmd_cuda_bench_4096x8192.json",
    "full-eval": "path_a_cuda_full_eval.json",
}


def _default_download_list(mode: str) -> list[str]:
    return COMMON_ARTIFACTS + STAGE_ARTIFACTS.get(mode, [])


def _env_capture_snippet() -> str:
    return textwrap.dedent("""\
        nvidia-smi > /root/rehearsal_out/nvidia_smi.txt 2>&1 || true
        python3 --version > /root/rehearsal_out/python_version.txt 2>&1 || true
        nvcc --version > /root/rehearsal_out/nvcc_version.txt 2>&1 || true
        pip freeze > /root/rehearsal_out/pip_freeze.txt 2>&1 || true
    """).rstrip()


def _cuda_env_probe_snippet(stage: str, gpu_sku: str, branch: str, commit: str) -> str:
    """Inline python3 -c snippet that writes cuda_env_probe.json."""
    py_code = (
        "import json, datetime, sys\n"
        "try:\n"
        "    import torch\n"
        "    obj = {\n"
        "        'device_count': torch.cuda.device_count(),\n"
        "        'cuda_available': torch.cuda.is_available(),\n"
        "        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n"
        "        'torch_version': torch.__version__,\n"
        "        'cuda_runtime_version': torch.version.cuda,\n"
        "    }\n"
        "except Exception as e:\n"
        "    obj = {'error': str(e), 'cuda_available': False, 'device_count': 0}\n"
        "obj.update({\n"
        "    'stage': " + repr(stage) + ",\n"
        "    'lane': " + repr(gpu_sku) + ",\n"
        "    'gpu_sku': " + repr(gpu_sku) + ",\n"
        "    'branch': " + repr(branch) + ",\n"
        "    'commit': " + repr(commit) + ",\n"
        "    'status': 'OK',\n"
        "    'on_pod_timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',\n"
        "})\n"
        "sys.stdout.write(json.dumps(obj, indent=2))\n"
    )
    return "python3 -c {} > /root/rehearsal_out/cuda_env_probe.json 2>&1 || true".format(
        shlex.quote(py_code)
    )


def _git_clone_snippet(branch: str, commit: str, repo_url: str) -> str:
    return textwrap.dedent("""\
        git clone --depth 1 --branch {branch} {repo_url} /root/rehearsal_src/parameter_golf2 \\
          || git clone {repo_url} /root/rehearsal_src/parameter_golf2
        cd /root/rehearsal_src/parameter_golf2
        git checkout {commit} || true
        git rev-parse HEAD > /root/rehearsal_out/git_rev.txt 2>&1 || true
    """).rstrip().format(
        branch=shlex.quote(branch),
        commit=shlex.quote(commit),
        repo_url=shlex.quote(repo_url),
    )


def _stage_stub_json_snippet(stage: str, gpu_sku: str, branch: str, commit: str, json_file: str) -> str:
    """Write a NOT_IMPLEMENTED skeleton stage JSON for stages where CUDA code doesn't exist yet."""
    py_code = (
        "import json, datetime, sys\n"
        "obj = {\n"
        "    'stage': " + repr(stage) + ",\n"
        "    'lane': " + repr(gpu_sku) + ",\n"
        "    'gpu_sku': " + repr(gpu_sku) + ",\n"
        "    'branch': " + repr(branch) + ",\n"
        "    'commit': " + repr(commit) + ",\n"
        "    'status': 'NOT_IMPLEMENTED',\n"
        "    'message': 'CUDA backend not yet implemented; stub written by run_ppmd_cuda_runpod.py',\n"
        "    'on_pod_timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',\n"
        "}\n"
        "with open('/root/rehearsal_out/" + json_file + "', 'w') as _f:\n"
        "    json.dump(obj, _f, indent=2)\n"
    )
    return "python3 -c {}".format(shlex.quote(py_code))


def _bundle_source_enabled(args: argparse.Namespace) -> bool:
    """Return True if source bundling is active for the given args."""
    if args.bundle_source is not None:
        return args.bundle_source
    return args.mode in BUNDLE_SOURCE_MODES


def _get_bundle_extra_files(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return (local_path, arcname) pairs to pass as --extra-file to rehearsal.

    Mode-aware to keep total env-var bytes well under the RunPod GraphQL
    request limit (~80 KB total env). Only include what the on-pod payload
    actually imports/runs.
    """
    if not _bundle_source_enabled(args):
        return []
    extras: list[tuple[str, str]] = []
    ppmd_cpp_dir = REPO_ROOT / "scripts" / "ppmd_cpp"
    if ppmd_cpp_dir.exists():
        extras.append((str(ppmd_cpp_dir), "ppmd_cpp"))

    # build-smoke and kernel-equiv only need the C++/CUDA source tree (and
    # for kernel-equiv the cuda smoke test). Eval scripts + full test suite
    # would push the bundle past RunPod's env-var size limit.
    if args.mode == "build-smoke":
        return extras
    if args.mode == "kernel-equiv":
        smoke = REPO_ROOT / "tests" / "test_ppmd_cpp_cuda_smoke.py"
        if smoke.exists():
            extras.append((str(smoke), smoke.name))
        return extras

    # Heavier stages need eval_path_a_ppmd.py only when they actually invoke
    # it. trie-prefix and bench inline their own scoring through
    # `_ppmd_cuda.cuda.score_path_a_arrays_cuda(...)` so they need only
    # the ppmd_cpp directory. full-eval invokes the full eval pipeline.
    if args.mode in ("trie-prefix-256", "trie-prefix-1k", "bench"):
        return extras

    p = SCRIPTS_DIR / "eval_path_a_ppmd.py"
    if p.exists():
        extras.append((str(p), "eval_path_a_ppmd.py"))

    # full-eval additionally needs data download helpers for on-pod FineWeb fetch.
    if args.mode == "full-eval":
        for name in ("cached_challenge_fineweb.py", "tokenizer_specs.json"):
            p = REPO_ROOT / "data" / name
            if p.exists():
                extras.append((str(p), f"data/{name}"))

    return extras


def _build_smoke_snippet(gpu_sku: str, branch: str, commit: str) -> str:
    """Return the shell snippet that builds _ppmd_cuda and runs import smoke."""
    import_py = (
        "import sys, json, os\n"
        "sys.path.insert(0, '/root/rehearsal_src/ppmd_cpp')\n"
        "build_ec = int(os.environ.get('BUILD_EC', '1'))\n"
        "try:\n"
        "    import _ppmd_cuda\n"
        "    out = {\n"
        "        'imported': True,\n"
        "        'module_path': _ppmd_cuda.__file__,\n"
        "        'version': _ppmd_cuda.version(),\n"
        "        'cuda_available': _ppmd_cuda.cuda.available(),\n"
        "        'cuda_device_count': _ppmd_cuda.cuda.device_count(),\n"
        "        'cuda_runtime_version': _ppmd_cuda.cuda.runtime_version(),\n"
        "        'cuda_driver_version': _ppmd_cuda.cuda.driver_version(),\n"
        "        'cuda_device_name': (_ppmd_cuda.cuda.device_name(0)\n"
        "                             if _ppmd_cuda.cuda.device_count() > 0 else ''),\n"
        "    }\n"
        "except Exception as e:\n"
        "    out = {'imported': False, 'error': str(e)}\n"
        "out['status'] = 'OK' if (build_ec == 0 and out.get('imported')) else 'FAIL'\n"
        "out['build_exit_code'] = build_ec\n"
        "out.update({\n"
        "    'stage': " + repr("build-smoke") + ",\n"
        "    'gpu_sku': " + repr(gpu_sku) + ",\n"
        "    'branch': " + repr(branch) + ",\n"
        "    'commit': " + repr(commit) + ",\n"
        "})\n"
        "import datetime\n"
        "out['on_pod_timestamp_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'\n"
        "open('/root/rehearsal_out/ppmd_cuda_import_smoke.json', 'w').write(\n"
        "    json.dumps(out, indent=2))\n"
    )
    return textwrap.dedent("""\
        printf '%s\\n' {commit_q} > /root/rehearsal_out/git_rev.txt
        pip install --quiet --break-system-packages pybind11 >> /root/rehearsal_out/ppmd_cuda_build.log 2>&1 || true
        cd /root/rehearsal_src/ppmd_cpp
        set +e
        PYTHON=$(which python3) make cuda 2>&1 | tee -a /root/rehearsal_out/ppmd_cuda_build.log
        BUILD_EC=${{PIPESTATUS[0]}}
        set -e
        ls _ppmd_cuda*.so 2>/dev/null > /root/rehearsal_out/ppmd_cuda_build_manifest.txt
        sha256sum _ppmd_cuda*.so 2>/dev/null >> /root/rehearsal_out/ppmd_cuda_build_manifest.txt || true
        export BUILD_EC
        python3 -c {py_q}
        exit $BUILD_EC
    """).rstrip().format(
        commit_q=shlex.quote("bundle:{}".format(commit)),
        py_q=shlex.quote(import_py),
    )


def _kernel_equiv_snippet(gpu_sku: str, branch: str, commit: str) -> str:
    """Return the shell snippet that builds both extensions and runs the
    inline equivalence test, writing ppmd_cuda_kernel_equiv.json."""
    equiv_py = (
        "import sys, json, random, datetime\n"
        "import numpy as np\n"
        "sys.path.insert(0, '/root/rehearsal_src/ppmd_cpp')\n"
        "out = {\n"
        "    'stage': 'kernel-equiv',\n"
        "    'gpu_sku': " + repr(gpu_sku) + ",\n"
        "    'branch': " + repr(branch) + ",\n"
        "    'git_commit': " + repr(commit) + ",\n"
        "    'on_pod_timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',\n"
        "}\n"
        "try:\n"
        "    import _ppmd_cuda\n"
        "    cuda_dev = _ppmd_cuda.cuda.device_count()\n"
        "    out['cuda_device_count'] = cuda_dev\n"
        "    out['cuda_runtime_version'] = _ppmd_cuda.cuda.runtime_version()\n"
        "    out['device_name'] = _ppmd_cuda.cuda.device_name(0) if cuda_dev > 0 else ''\n"
        "    out['gpu_name'] = out['device_name']\n"
        "    if cuda_dev == 0:\n"
        "        out.update({'status': 'SKIP', 'reason': 'no CUDA device'})\n"
        "        sys.stdout.write(json.dumps(out, indent=2))\n"
        "        sys.exit(0)\n"
        "    rng = random.Random(42)\n"
        "    state = _ppmd_cuda.PPMDState(order=5)\n"
        "    payload = bytes(rng.randrange(256) for _ in range(20000))\n"
        "    state.update_bytes(payload)\n"
        "    windows = []\n"
        "    for i in range(200):\n"
        "        wlen = rng.choice([0, 1, 2, 3, 4, 5, 6])\n"
        "        windows.append(bytes(rng.randrange(256) for _ in range(wlen)))\n"
        "    cpu = np.zeros((len(windows), 256), dtype=np.float64)\n"
        "    for i, w in enumerate(windows):\n"
        "        v = state.clone_virtual()\n"
        "        for b in w:\n"
        "            v = v.fork_and_update(b)\n"
        "        cpu[i, :] = list(v.byte_probs())\n"
        "    cuda_probs = np.asarray(_ppmd_cuda.cuda.byte_probs_batched(state, windows))\n"
        "    diff = np.abs(cuda_probs - cpu)\n"
        "    max_abs_diff = float(diff.max())\n"
        "    sums = cuda_probs.sum(axis=1)\n"
        "    max_sum_err = float(np.abs(sums - 1.0).max())\n"
        "    status = 'OK' if (max_abs_diff <= 1e-15 and max_sum_err <= 1e-12) else 'FAIL'\n"
        "    out.update({\n"
        "        'contexts_tested': len(windows),\n"
        "        'bytes_tested': len(windows) * 256,\n"
        "        'max_abs_diff': max_abs_diff,\n"
        "        'max_sum_prob_error': max_sum_err,\n"
        "        'status': status,\n"
        "    })\n"
        "except Exception as e:\n"
        "    import traceback\n"
        "    out.update({'status': 'FAIL', 'error': str(e),\n"
        "                'traceback': traceback.format_exc()})\n"
        "with open('/root/rehearsal_out/ppmd_cuda_kernel_equiv.json', 'w') as _f:\n"
        "    json.dump(out, _f, indent=2)\n"
        "print(json.dumps(out, indent=2))\n"
        "if out.get('status') not in ('OK', 'SKIP'):\n"
        "    sys.exit(1)\n"
    )
    return textwrap.dedent("""\
        printf '%s\\n' {commit_q} > /root/rehearsal_out/git_rev.txt
        pip install --quiet --break-system-packages pybind11 numpy \\
            >> /root/rehearsal_out/ppmd_cuda_kernel_equiv.log 2>&1 || true
        cd /root/rehearsal_src/ppmd_cpp
        set +e
        PYTHON=$(which python3) make cuda 2>&1 | tee -a /root/rehearsal_out/ppmd_cuda_kernel_equiv.log
        CUDA_BUILD_EC=${{PIPESTATUS[0]}}
        PYTHON=$(which python3) make 2>&1 | tee -a /root/rehearsal_out/ppmd_cuda_kernel_equiv.log
        set -e
        python3 -c {py_q} 2>&1 | tee -a /root/rehearsal_out/ppmd_cuda_kernel_equiv.log
    """).rstrip().format(
        commit_q=shlex.quote("bundle:{}".format(commit)),
        py_q=shlex.quote(equiv_py),
    )


def _trie_prefix_snippet(gpu_sku: str, branch: str, commit: str, n_positions: int) -> str:
    """Build snippet that runs synthetic Path A CUDA scoring and compares with CPP."""
    suffix = "1k" if n_positions == 1000 else str(n_positions)
    stage = f"trie-prefix-{suffix}"
    json_file = f"path_a_cuda_prefix_{suffix}.json"
    log_file = f"path_a_cuda_prefix_{suffix}.log"

    py_code = (
        "import sys, json, random, datetime, traceback\n"
        "import numpy as np\n"
        "sys.path.insert(0, '/root/rehearsal_src/ppmd_cpp')\n"
        "out = {\n"
        "    'stage': " + repr(stage) + ",\n"
        "    'gpu_sku': " + repr(gpu_sku) + ",\n"
        "    'branch': " + repr(branch) + ",\n"
        "    'git_commit': " + repr(commit) + ",\n"
        "    'n_positions': " + repr(n_positions) + ",\n"
        "    'on_pod_timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',\n"
        "}\n"
        "try:\n"
        "    import _ppmd_cuda, _ppmd_cpp\n"
        "    cuda_dev = _ppmd_cuda.cuda.device_count()\n"
        "    out['cuda_device_count'] = cuda_dev\n"
        "    out['device_name'] = _ppmd_cuda.cuda.device_name(0) if cuda_dev > 0 else ''\n"
        "    if cuda_dev == 0:\n"
        "        out.update({'status': 'SKIP', 'reason': 'no CUDA device'})\n"
        "        sys.exit(0)\n"
        "    rng = random.Random(42)\n"
        "    # Build PPM state with 20K random bytes.\n"
        "    state = _ppmd_cuda.PPMDState(order=5)\n"
        "    state.update_bytes(bytes(rng.randrange(256) for _ in range(20000)))\n"
        "    # Synthetic vocab: 300 tokens, byte seqs length 1-4.\n"
        "    vocab_size = 300\n"
        "    token_bytes = [bytes([rng.randrange(256) for _ in range(rng.randint(1,4))]) for _ in range(vocab_size)]\n"
        "    bnd_flat = b''.join(token_bytes)\n"
        "    bnd_off = np.zeros(vocab_size + 1, dtype=np.int32)\n"
        "    for i, tb in enumerate(token_bytes): bnd_off[i+1] = bnd_off[i] + len(tb)\n"
        "    nbnd_flat = bnd_flat; nbnd_off = bnd_off.copy()\n"
        "    emit = np.ones(vocab_size, dtype=np.uint8)\n"
        "    isb = np.zeros(vocab_size, dtype=np.uint8)\n"
        "    n = " + repr(n_positions) + "\n"
        "    target_ids = np.array([rng.randrange(vocab_size) for _ in range(n)], dtype=np.int32)\n"
        "    prev_ids   = np.full(n, -1, dtype=np.int32)\n"
        "    nll_nats   = np.full(n, float(np.log(vocab_size)), dtype=np.float64)\n"
        "    hp = {'order':5,'lambda_hi':0.9,'lambda_lo':0.05,'conf_threshold':0.9,'update_after_score':True}\n"
        "    import time\n"
        "    t0 = time.perf_counter()\n"
        "    cuda_out = _ppmd_cuda.cuda.score_path_a_arrays_cuda(\n"
        "        target_ids, prev_ids, nll_nats,\n"
        "        np.frombuffer(bnd_flat, dtype=np.uint8), bnd_off,\n"
        "        np.frombuffer(nbnd_flat, dtype=np.uint8), nbnd_off,\n"
        "        emit, isb, hp)\n"
        "    cuda_elapsed = time.perf_counter() - t0\n"
        "    t0 = time.perf_counter()\n"
        "    cpp_out = _ppmd_cpp.score_path_a_arrays(\n"
        "        target_ids, prev_ids, nll_nats,\n"
        "        np.frombuffer(bnd_flat, dtype=np.uint8), bnd_off,\n"
        "        np.frombuffer(nbnd_flat, dtype=np.uint8), nbnd_off,\n"
        "        emit, isb, hp)\n"
        "    cpp_elapsed = time.perf_counter() - t0\n"
        "    equiv_diff = abs(float(cuda_out['bpb']) - float(cpp_out['bpb']))\n"
        "    status = 'OK' if equiv_diff <= 1e-10 else 'FAIL'\n"
        "    out.update({\n"
        "        'cuda_bpb': float(cuda_out['bpb']),\n"
        "        'cpp_bpb': float(cpp_out['bpb']),\n"
        "        'equiv_bpb_diff': equiv_diff,\n"
        "        'cuda_elapsed_sec': cuda_elapsed,\n"
        "        'cpp_elapsed_sec': cpp_elapsed,\n"
        "        'positions': int(cuda_out['positions']),\n"
        "        'status': status,\n"
        "    })\n"
        "except Exception as e:\n"
        "    out.update({'status': 'FAIL', 'error': str(e), 'traceback': traceback.format_exc()})\n"
        "with open('/root/rehearsal_out/" + json_file + "', 'w') as _f:\n"
        "    json.dump(out, _f, indent=2)\n"
        "print(json.dumps(out, indent=2))\n"
        "if out.get('status') not in ('OK', 'SKIP'):\n"
        "    sys.exit(1)\n"
    )
    return textwrap.dedent("""\
        printf '%s\\n' {commit_q} > /root/rehearsal_out/git_rev.txt
        pip install --quiet --break-system-packages pybind11 numpy \\
            >> /root/rehearsal_out/{log_file} 2>&1 || true
        cd /root/rehearsal_src/ppmd_cpp
        set +e
        PYTHON=$(which python3) make cuda 2>&1 | tee -a /root/rehearsal_out/{log_file}
        PYTHON=$(which python3) make 2>&1 | tee -a /root/rehearsal_out/{log_file}
        set -e
        python3 -c {py_q} 2>&1 | tee -a /root/rehearsal_out/{log_file}
    """).rstrip().format(
        commit_q=shlex.quote("bundle:{}".format(commit)),
        log_file=log_file,
        py_q=shlex.quote(py_code),
    )


def _bench_snippet(gpu_sku: str, branch: str, commit: str) -> str:
    """Build snippet that runs byte-prob bench + trie-prefix-1k throughput bench."""
    bench_py = (
        "import sys, json, time, random, datetime, traceback\n"
        "import numpy as np\n"
        "sys.path.insert(0, '/root/rehearsal_src/ppmd_cpp')\n"
        "base = {\n"
        "    'gpu_sku': " + repr(gpu_sku) + ",\n"
        "    'branch': " + repr(branch) + ",\n"
        "    'git_commit': " + repr(commit) + ",\n"
        "    'on_pod_timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',\n"
        "}\n"
        "def run_byte_prob_bench():\n"
        "    out = {**base, 'stage': 'bench-byte-prob-4096x8192'}\n"
        "    try:\n"
        "        import _ppmd_cuda\n"
        "        if _ppmd_cuda.cuda.device_count() == 0:\n"
        "            out['status'] = 'SKIP'; return out\n"
        "        rng = random.Random(42)\n"
        "        state = _ppmd_cuda.PPMDState(order=5)\n"
        "        state.update_bytes(bytes(rng.randrange(256) for _ in range(20000)))\n"
        "        windows = [bytes(rng.randrange(256) for _ in range(rng.randint(0,6))) for _ in range(4096)]\n"
        "        # Warmup\n"
        "        for _ in range(3): _ppmd_cuda.cuda.byte_probs_batched(state, windows[:64])\n"
        "        t0 = time.perf_counter()\n"
        "        for _ in range(5): result = _ppmd_cuda.cuda.byte_probs_batched(state, windows)\n"
        "        elapsed = (time.perf_counter() - t0) / 5\n"
        "        n_probs = 4096 * 256\n"
        "        out.update({'elapsed_sec': elapsed, 'n_windows': 4096, 'n_probs': n_probs,\n"
        "                    'probs_per_sec': n_probs / elapsed, 'status': 'OK',\n"
        "                    'device_name': _ppmd_cuda.cuda.device_name(0),\n"
        "                    'projected_full_eval_byte_prob_sec': 8192 * elapsed / 4096})\n"
        "    except Exception as e:\n"
        "        out.update({'status': 'FAIL', 'error': str(e), 'traceback': traceback.format_exc()})\n"
        "    return out\n"
        "def run_trie_prefix_bench():\n"
        "    out = {**base, 'stage': 'bench-trie-prefix-1k'}\n"
        "    try:\n"
        "        import _ppmd_cuda, _ppmd_cpp\n"
        "        if _ppmd_cuda.cuda.device_count() == 0:\n"
        "            out['status'] = 'SKIP'; return out\n"
        "        rng = random.Random(99)\n"
        "        state = _ppmd_cuda.PPMDState(order=5)\n"
        "        state.update_bytes(bytes(rng.randrange(256) for _ in range(20000)))\n"
        "        vocab_size = 300\n"
        "        token_bytes = [bytes([rng.randrange(256) for _ in range(rng.randint(1,4))]) for _ in range(vocab_size)]\n"
        "        bnd_flat = b''.join(token_bytes)\n"
        "        bnd_off = np.zeros(vocab_size+1, dtype=np.int32)\n"
        "        for i, tb in enumerate(token_bytes): bnd_off[i+1] = bnd_off[i]+len(tb)\n"
        "        emit = np.ones(vocab_size, dtype=np.uint8)\n"
        "        isb = np.zeros(vocab_size, dtype=np.uint8)\n"
        "        n = 1000\n"
        "        tids = np.array([rng.randrange(vocab_size) for _ in range(n)], dtype=np.int32)\n"
        "        pids = np.full(n, -1, dtype=np.int32)\n"
        "        nlls = np.full(n, float(np.log(vocab_size)), dtype=np.float64)\n"
        "        hp = {'order':5,'lambda_hi':0.9,'lambda_lo':0.05,'conf_threshold':0.9,'update_after_score':True}\n"
        "        bflt = np.frombuffer(bnd_flat, dtype=np.uint8)\n"
        "        # Warmup\n"
        "        _ppmd_cuda.cuda.score_path_a_arrays_cuda(tids[:50],pids[:50],nlls[:50],bflt,bnd_off,bflt,bnd_off,emit,isb,hp)\n"
        "        t0 = time.perf_counter()\n"
        "        for _ in range(3): _ppmd_cuda.cuda.score_path_a_arrays_cuda(tids,pids,nlls,bflt,bnd_off,bflt,bnd_off,emit,isb,hp)\n"
        "        elapsed = (time.perf_counter() - t0) / 3\n"
        "        out.update({'elapsed_sec': elapsed, 'n_positions': n,\n"
        "                    'positions_per_sec': n / elapsed,\n"
        "                    'status': 'OK',\n"
        "                    'device_name': _ppmd_cuda.cuda.device_name(0),\n"
        "                    'projected_full_eval_trie_prefix_sec': elapsed})\n"
        "    except Exception as e:\n"
        "        out.update({'status': 'FAIL', 'error': str(e), 'traceback': traceback.format_exc()})\n"
        "    return out\n"
        "r1 = run_byte_prob_bench()\n"
        "r2 = run_trie_prefix_bench()\n"
        "with open('/root/rehearsal_out/ppmd_cuda_bench_4096x8192.json', 'w') as f: json.dump(r1, f, indent=2)\n"
        "with open('/root/rehearsal_out/ppmd_cuda_bench_prefix_1k.json', 'w') as f: json.dump(r2, f, indent=2)\n"
        "print('bench-byte-prob:', json.dumps(r1, indent=2))\n"
        "print('bench-trie-prefix:', json.dumps(r2, indent=2))\n"
        "overall = 'OK' if r1.get('status') in ('OK','SKIP') and r2.get('status') in ('OK','SKIP') else 'FAIL'\n"
        "if overall != 'OK': sys.exit(1)\n"
    )
    return textwrap.dedent("""\
        printf '%s\\n' {commit_q} > /root/rehearsal_out/git_rev.txt
        pip install --quiet --break-system-packages pybind11 numpy \\
            >> /root/rehearsal_out/ppmd_cuda_bench.log 2>&1 || true
        cd /root/rehearsal_src/ppmd_cpp
        set +e
        PYTHON=$(which python3) make cuda 2>&1 | tee -a /root/rehearsal_out/ppmd_cuda_bench.log
        PYTHON=$(which python3) make 2>&1 | tee -a /root/rehearsal_out/ppmd_cuda_bench.log
        set -e
        python3 -c {py_q} 2>&1 | tee -a /root/rehearsal_out/ppmd_cuda_bench.log
    """).rstrip().format(
        commit_q=shlex.quote("bundle:{}".format(commit)),
        py_q=shlex.quote(bench_py),
    )


def _full_eval_snippet(gpu_sku: str, branch: str, commit: str, positions: int | None) -> str:
    """Build snippet that downloads FineWeb val data and runs full Path A CUDA eval (bundle path)."""
    positions_arg = f"--max-positions {positions}" if positions else ""
    eval_py_path = "/root/rehearsal_src/eval_path_a_ppmd.py"

    setup_cmd = textwrap.dedent("""\
        printf '%s\\n' {commit_q} > /root/rehearsal_out/git_rev.txt
        pip install --quiet --break-system-packages pybind11 numpy sentencepiece \\
            >> /root/rehearsal_out/path_a_cuda_full_eval.log 2>&1 || true
        cd /root/rehearsal_src/ppmd_cpp
        set +e
        PYTHON=$(which python3) make cuda 2>&1 | tee -a /root/rehearsal_out/path_a_cuda_full_eval.log
        BUILD_CUDA_EC=${{PIPESTATUS[0]}}
        PYTHON=$(which python3) make 2>&1 | tee -a /root/rehearsal_out/path_a_cuda_full_eval.log
        BUILD_CPP_EC=${{PIPESTATUS[0]}}
        set -e
        cd /root/rehearsal_src
        # Download FineWeb SP8192 validation split
        MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \\
            python3 data/cached_challenge_fineweb.py --variant sp8192 \\
            >> /root/rehearsal_out/path_a_cuda_full_eval.log 2>&1 || true
        # Run Path A CUDA eval
        PYTHONPATH=/root/rehearsal_src/ppmd_cpp:/root/rehearsal_src \\
        python3 {eval_py_path} \\
            --backend cuda \\
            --backend-equiv-check 256 \\
            {positions_arg} \\
            --output /root/rehearsal_out/path_a_cuda_full_eval.json \\
            2>&1 | tee -a /root/rehearsal_out/path_a_cuda_full_eval.log
        EVAL_EC=${{PIPESTATUS[0]}}
        # Compute sha256 of output JSON
        sha256sum /root/rehearsal_out/path_a_cuda_full_eval.json \\
            > /root/rehearsal_out/path_a_cuda_full_eval.sha256 2>&1 || true
        exit $EVAL_EC
    """).rstrip().format(
        commit_q=shlex.quote("bundle:{}".format(commit)),
        eval_py_path=eval_py_path,
        positions_arg=positions_arg,
    )
    return setup_cmd


def _full_eval_clone_snippet(gpu_sku: str, branch: str, commit: str, positions: int | None) -> str:
    """Build snippet that runs the FULL neural+PPM-D Path A eval using a cloned repo.

    Requires:
      - The clone contains scripts/ppmd_cpp (CUDA backend), scripts/eval_path_a_ppmd.py,
        results/exp_1876_ppmd/train_gpt_merged.py, results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz,
        and data/cached_challenge_fineweb.py with data/tokenizer_specs.json.
      - The pod has CUDA + torch + FA3 already installed (matotezitanka/proteus-pytorch:community).
    """
    positions_arg = f"--max-positions {positions}" if positions else ""
    setup_cmd = textwrap.dedent("""\
        REPO=/root/rehearsal_src/parameter_golf2
        cd $REPO
        pip install --quiet --break-system-packages pybind11 numpy sentencepiece \\
            >> /root/rehearsal_out/path_a_cuda_full_eval.log 2>&1 || true
        cd $REPO/scripts/ppmd_cpp
        set +e
        PYTHON=$(which python3) make cuda 2>&1 | tee -a /root/rehearsal_out/path_a_cuda_full_eval.log
        BUILD_CUDA_EC=${{PIPESTATUS[0]}}
        PYTHON=$(which python3) make 2>&1 | tee -a /root/rehearsal_out/path_a_cuda_full_eval.log
        BUILD_CPP_EC=${{PIPESTATUS[0]}}
        set -e
        cd $REPO
        # Download FineWeb SP8192 validation split
        MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \\
            python3 data/cached_challenge_fineweb.py --variant sp8192 \\
            >> /root/rehearsal_out/path_a_cuda_full_eval.log 2>&1 || true
        # Run full Path A CUDA eval (neural NLL + PPM-D scoring)
        cd $REPO/scripts
        PYTHONPATH=$REPO/scripts/ppmd_cpp:$REPO/scripts:$REPO \\
        python3 $REPO/scripts/eval_path_a_ppmd.py \\
            --backend cuda \\
            --backend-equiv-check 64 \\
            --allow-slow-python-full-eval \\
            {positions_arg} \\
            --output /root/rehearsal_out/path_a_cuda_full_eval.json \\
            2>&1 | tee -a /root/rehearsal_out/path_a_cuda_full_eval.log
        EVAL_EC=${{PIPESTATUS[0]}}
        sha256sum /root/rehearsal_out/path_a_cuda_full_eval.json \\
            > /root/rehearsal_out/path_a_cuda_full_eval.sha256 2>&1 || true
        exit $EVAL_EC
    """).rstrip().format(
        positions_arg=positions_arg,
    )
    return setup_cmd


def _build_payload(args: argparse.Namespace) -> str:
    mode = args.mode
    branch = args.branch
    commit = args.commit
    gpu_sku = args.gpu_sku
    repo_url = args.repo_url
    use_bundle = _bundle_source_enabled(args)

    lines = [
        "set -euo pipefail",
        "mkdir -p /root/rehearsal_src /root/rehearsal_out",
        "",
        _env_capture_snippet(),
        "",
    ]

    if mode == "env-smoke":
        # Skip git clone for env-smoke; write a placeholder git_rev.txt
        lines.append(
            "printf '%s\\n' {} > /root/rehearsal_out/git_rev.txt".format(
                shlex.quote("{} (skipped)".format(commit))
            )
        )
        lines.append("")
        lines.append(_cuda_env_probe_snippet(mode, gpu_sku, branch, commit))
    elif mode == "build-smoke" and use_bundle:
        lines.append(_build_smoke_snippet(gpu_sku, branch, commit))
    elif mode == "kernel-equiv" and use_bundle:
        lines.append(_kernel_equiv_snippet(gpu_sku, branch, commit))
    elif mode == "trie-prefix-256" and use_bundle:
        lines.append(_trie_prefix_snippet(gpu_sku, branch, commit, n_positions=256))
    elif mode == "trie-prefix-1k" and use_bundle:
        lines.append(_trie_prefix_snippet(gpu_sku, branch, commit, n_positions=1000))
    elif mode == "bench" and use_bundle:
        lines.append(_bench_snippet(gpu_sku, branch, commit))
    elif mode == "full-eval" and use_bundle:
        positions = getattr(args, "prefix_positions", None)
        lines.append(_full_eval_snippet(gpu_sku, branch, commit, positions=positions))
    elif mode == "full-eval" and not use_bundle:
        # Clone-based full eval path: fetch repo (with model + train_gpt_merged.py),
        # build CUDA backend, download data, run full neural+PPM-D eval.
        positions = getattr(args, "prefix_positions", None)
        lines.append(_git_clone_snippet(branch, commit, repo_url))
        lines.append("")
        lines.append(_full_eval_clone_snippet(gpu_sku, branch, commit, positions=positions))
    else:
        # Fallback: git clone + checkout + stage stub (for non-bundle runs)
        lines.append(_git_clone_snippet(branch, commit, repo_url))
        lines.append("")
        primary_json = STAGE_PRIMARY_JSON.get(mode, "{}_stage.json".format(mode.replace("-", "_")))
        lines.append(_stage_stub_json_snippet(mode, gpu_sku, branch, commit, primary_json))

    return "\n".join(lines)


def _build_rehearsal_cmd(
    args: argparse.Namespace,
    payload: str,
    download_list: list[str],
    pod_name: str,
) -> list[str]:
    """Build the subprocess command for runpod_http_rehearsal.py."""
    rehearsal_script = str(SCRIPTS_DIR / "runpod_http_rehearsal.py")
    cmd = [
        sys.executable, rehearsal_script,
        "--gpu-sku", args.gpu_sku,
        "--max-minutes", str(args.max_minutes),
        "--pod-name", pod_name,
        "--results-dir", args.results_dir,
        "--cmd", payload,
        "--docker-image", args.docker_image,
        "--runtime-timeout-sec", str(args.runtime_timeout_sec),
        "--download",
    ] + download_list
    for local_path, arcname in _get_bundle_extra_files(args):
        cmd.extend(["--extra-file", "{}:{}".format(local_path, arcname)])
    return cmd


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPM-D CUDA RunPod rehearsal wrapper — generates stage payloads.",
    )
    parser.add_argument(
        "--gpu-sku", required=True, choices=list(GPU_SKU_TABLE.keys()),
        help="GPU SKU lane selector",
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["env-smoke", "build-smoke", "kernel-equiv",
                 "trie-prefix-256", "trie-prefix-1k", "bench", "full-eval"],
        help="Stage selector",
    )
    parser.add_argument("--branch", required=True, help="Git branch to check out on-pod")
    parser.add_argument("--commit", required=True, help="Exact commit SHA to check out on-pod")
    parser.add_argument("--max-minutes", required=True, type=int, help="Pod wallclock cap in minutes")
    parser.add_argument("--results-dir", required=True, help="Local retrieval directory on this HPC")
    parser.add_argument(
        "--pod-name", default=None,
        help="Pod display name (default: ppmd-cuda-{mode}-{gpu_sku})",
    )
    parser.add_argument(
        "--runtime-timeout-sec", type=int, default=DEFAULT_RUNTIME_TIMEOUT_SEC,
        help="Seconds to wait for RunPod runtime startup (default: {})".format(DEFAULT_RUNTIME_TIMEOUT_SEC),
    )
    parser.add_argument(
        "--docker-image", default=DEFAULT_DOCKER_IMAGE,
        help="Docker image (default: {})".format(DEFAULT_DOCKER_IMAGE),
    )
    parser.add_argument(
        "--prefix-positions", type=int, default=None,
        help="Prefix length for trie-prefix modes",
    )
    parser.add_argument(
        "--download", nargs="*", default=None,
        help="Override artifact download list (default: mode-specific)",
    )
    parser.add_argument(
        "--repo-url", default=DEFAULT_REPO_URL,
        help="Git repo URL for on-pod clone (default: {})".format(DEFAULT_REPO_URL),
    )
    parser.add_argument(
        "--bundle-source", action=argparse.BooleanOptionalAction, default=None,
        help="Bundle ppmd_cpp source into pod bundle (default: ON for non-env-smoke modes)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print generated payload and subprocess command; do not launch",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    pod_name = args.pod_name or "ppmd-cuda-{mode}-{sku}".format(
        mode=args.mode, sku=args.gpu_sku
    )
    download_list = args.download if args.download is not None else _default_download_list(args.mode)
    payload = _build_payload(args)
    cmd = _build_rehearsal_cmd(args, payload, download_list, pod_name)

    if args.dry_run:
        payload_sha = hashlib.sha256(payload.encode()).hexdigest()
        extra_files = _get_bundle_extra_files(args)
        print("=== DRY RUN ===")
        print("Pod name   :", pod_name)
        print("GPU SKU    :", args.gpu_sku)
        print("Mode       :", args.mode)
        print("Branch     :", args.branch)
        print("Commit     :", args.commit)
        print("Max minutes:", args.max_minutes)
        print("Results dir:", args.results_dir)
        print("Bundle src :", _bundle_source_enabled(args))
        if extra_files:
            print("Bundle extras:")
            for lp, arc in extra_files:
                print("  {}  ->  {}".format(lp, arc))
        print("Payload SHA:", payload_sha)
        print("")
        print("--- PAYLOAD ---")
        print(payload)
        print("")
        print("--- SUBPROCESS CMD ---")
        print(" ".join(shlex.quote(c) for c in cmd))
        return

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
