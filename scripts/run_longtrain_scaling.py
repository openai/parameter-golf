#!/usr/bin/env python3
"""Launch long-train artifact scaling experiment on 8×H100.

Trains for 1 hour (3600s) with checkpoints exported at configurable intervals
(default: 10, 20, 30, 45, 60 minutes).  Each checkpoint export takes ~130s
(GPTQ + lrzip compression), so total wallclock is ~80-85 minutes for training
+ exports, plus ~15-20 minutes for final TTT eval.  Total: ~100-110 minutes.

Supports HTTP-based telemetry: with --download-checkpoints, polls every 2
minutes for new checkpoint files and downloads them as they appear.

Usage:
    python scripts/run_longtrain_scaling.py
    python scripts/run_longtrain_scaling.py --download-checkpoints
    python scripts/run_longtrain_scaling.py --seed 314 --max-minutes 140
    python scripts/run_longtrain_scaling.py --dry-run
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from runpod_http_rehearsal import (
    main as http_main,
    build_bundle_b64,
    build_boot_command,
    build_launcher_state,
    write_launcher_state,
    record_launcher_exception,
    terminate_pod_with_launcher_state,
    wait_http_proxy,
    wait_startup_readiness_and_maybe_download_status,
    download_file,
    H100_COST_PER_GPU_HR,
    HTTP_TERMINAL_STATUSES,
)
from runpod_safe import (
    UA, _make_ssl_ctx, balance, create_pod, wait_runtime, terminate_and_wait,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASEOPS_REPO = "romeerp/parameter-golf-caseops-v1"
CASEOPS_DATASET_DIR = "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
CASEOPS_TOKENIZER = "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"

DEFAULT_SEED = 42
DEFAULT_MAX_MINUTES = 130
DEFAULT_MAX_WALLCLOCK = 3600
DEFAULT_EXPORT_MINUTES = "10,20,30,45,60"
DEFAULT_EXPORT_MODE = "light"
SEED_TIMEOUT_MIN = 120
POLL_INTERVAL_SEC = 120  # 2 minutes

# 4-hour duration mode defaults
DEFAULT_4H_MAX_WALLCLOCK = 14400
DEFAULT_4H_MAX_MINUTES = 360  # 6 hours total pod time (4h train + GPTQ + TTT)
DEFAULT_4H_EXPORT_MINUTES = "60,120,180,240"
DEFAULT_4H_RESUME_SAVE_MINUTES = "30,60,90,120,150,180,210,240"
DEFAULT_4H_ITERATIONS = 100000

# On-pod directory where resume snapshot files land via SSH upload
ONPOD_RESUME_SNAPSHOT_DIR = "resume_snapshot"
ONPOD_RESUME_SNAPSHOT_PATH = "/root/rehearsal_src/" + ONPOD_RESUME_SNAPSHOT_DIR


def build_resume_ssh_uploads(local_snapshot_dir):
    """Build --ssh-upload specs for all files in a local resume snapshot directory.

    Returns a list of strings suitable for appending to sys.argv as
    --ssh-upload arguments. Each file lands at
    /root/rehearsal_src/resume_snapshot/<filename> on-pod.

    Raises SystemExit if the directory or manifest is missing.
    """
    snap = Path(local_snapshot_dir)
    if not snap.is_dir():
        raise SystemExit(
            "ERROR: --resume-from directory does not exist: {}".format(local_snapshot_dir)
        )
    manifest = snap / "resume_manifest.json"
    if not manifest.exists():
        raise SystemExit(
            "ERROR: resume_manifest.json not found in: {}".format(local_snapshot_dir)
        )
    specs = []
    for f in sorted(snap.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            arc = "{}/{}".format(ONPOD_RESUME_SNAPSHOT_DIR, f.name)
            specs.append("{}:{}".format(str(f), arc))
    return specs


def parse_export_minutes(s):
    """Parse comma-separated minute values into a sorted list of ints."""
    return sorted(int(x.strip()) for x in s.split(","))


def _shell_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


def build_download_caseops_script():
    """Python script to download CaseOps data on-pod using snapshot_download."""
    return f'''
import os, time
from huggingface_hub import snapshot_download

REPO = "{CASEOPS_REPO}"
LOCAL_ROOT = "/root/caseops_data"

t0 = time.time()
snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    local_dir=LOCAL_ROOT,
    allow_patterns=[
        "datasets/datasets/{CASEOPS_DATASET_DIR}/*",
        "datasets/tokenizers/{CASEOPS_TOKENIZER}",
    ],
    max_workers=8,
)
elapsed = time.time() - t0
data_dir = os.path.join(LOCAL_ROOT, "datasets", "datasets", "{CASEOPS_DATASET_DIR}")
tok_path = os.path.join(LOCAL_ROOT, "datasets", "tokenizers", "{CASEOPS_TOKENIZER}")
n_train = len([f for f in os.listdir(data_dir) if f.startswith("fineweb_train_")])
n_val = len([f for f in os.listdir(data_dir) if f.startswith("fineweb_val_")])
assert os.path.isfile(tok_path), f"Tokenizer not found: {{tok_path}}"
assert n_train >= 39, f"Expected >=39 train shards, found {{n_train}}"
assert n_val >= 1, f"Expected >=1 val shard, found {{n_val}}"
print(f"CaseOps data ready: {{n_train}} train + {{n_val}} val shards in {{elapsed:.0f}}s")
print(f"DATA_DIR: {{data_dir}}")
print(f"TOK: {{tok_path}}")
'''


def build_seed_cmd(args):
    """Build the shell command to run on-pod."""
    seed = args.seed
    export_minutes = args.export_minutes
    max_wallclock = args.max_wallclock
    export_mode = args.export_mode

    download_script = build_download_caseops_script()
    data_path = f"/root/caseops_data/datasets/datasets/{CASEOPS_DATASET_DIR}"
    tok_path = f"/root/caseops_data/datasets/tokenizers/{CASEOPS_TOKENIZER}"
    artifact_dir = f"/root/rehearsal_out/seed{seed}"

    parts = []
    parts.append("cd /root/rehearsal_src")

    # Install deps including lrzip for pergroup compressor
    parts.append(
        "apt-get update -qq && apt-get install -y -qq lrzip 2>&1 | tail -3"
    )
    parts.append(
        "pip install --break-system-packages -r requirements.txt brotli python-minifier 2>&1 | tail -5"
    )
    parts.append("hash -r && which pyminify && which lrzip")

    # Preflight: verify critical imports + lrzip
    parts.append(
        'python3 -c "import brotli, sentencepiece, numpy, torch; '
        'from flash_attn_interface import flash_attn_func; '
        'import subprocess; subprocess.run([\\\"pyminify\\\", \\\"--help\\\"], capture_output=True, check=True); '
        'subprocess.run([\\\"lrzip\\\", \\\"--help\\\"], capture_output=True, check=True); '
        "print('Preflight OK (incl. lrzip)')\""
    )

    # Download CaseOps data
    parts.append(f"python3 -c {_shell_quote(download_script)}")

    # Create artifact dir
    parts.append(f"mkdir -p {artifact_dir}")

    # Warmup sleep
    parts.append(
        f"echo 'Sleeping 10s before seed {seed} training (long-train scaling)...'"
    )
    parts.append("sleep 10")

    # All environment variables for the long-train scaling experiment
    env = (
        f"SEED={seed} "
        f"CASEOPS_ENABLED=1 "
        f"PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 "
        f"MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=12.0 "
        f"MLP_CLIP_SIGMAS=12.0 "
        f"EMBED_BITS=7 EMBED_CLIP_SIGMAS=12.0 "
        f"MATRIX_LR=0.026 "
        f"MIN_LR=0.1 "
        f"FUSED_CE_ENABLED=1 "
        f"SPARSE_ATTN_GATE_ENABLED=1 "
        f"SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 "
        f"LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 "
        f"LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 "
        f"TTT_WARM_START_A=1 "
        f"GPTQ_RESERVE_SECONDS=5.5 GPTQ_CALIBRATION_BATCHES=16 "
        f"EMBED_WD=0.06 COMPRESSOR=pergroup "
        f"NON_RECORD_LONGTRAIN=1 "
        f"MAX_WALLCLOCK_SECONDS={max_wallclock} "
        f"LONGTRAIN_EXPORT_MINUTES={export_minutes} "
        f"EXPORT_MODE={export_mode} "
        f"DATA_PATH={data_path} "
        f"TOKENIZER_PATH={tok_path} "
        f"ARTIFACT_DIR={artifact_dir} "
        f"RUN_ID=train_seed{seed}"
    )

    # Resume env vars
    if getattr(args, "enable_resume", False):
        resume_dir = f"/root/rehearsal_out/seed{seed}/resume"
        env += f" RESUME_ENABLED=1 RESUME_DIR={resume_dir}"
        if getattr(args, "resume_save_minutes", None):
            env += f" RESUME_SAVE_MINUTES={args.resume_save_minutes}"
        env += f" RESUME_KEEP_LAST={getattr(args, 'resume_keep_last', 3)}"
    if getattr(args, "resume_from", None):
        # For continuation runs with SSH upload, rewrite to on-pod path
        resume_from_path = args.resume_from
        if getattr(args, "continuation_label", None) and Path(resume_from_path).is_dir():
            resume_from_path = ONPOD_RESUME_SNAPSHOT_PATH + "/resume_manifest.json"
        env += f" RESUME_FROM={resume_from_path}"

    # Iterations override
    if getattr(args, "iterations", None) is not None:
        env += f" ITERATIONS={args.iterations}"

    # Schedule horizon for continuation runs (Phase 2 patch)
    if getattr(args, "schedule_horizon", None) is not None:
        env += f" SCHEDULE_HORIZON_SECONDS={args.schedule_horizon}"

    # Compute per-seed timeout from training wallclock + buffer for GPTQ/eval
    # Training itself: max_wallclock seconds
    # Plus: 4 checkpoint exports × ~150s each + final GPTQ ~150s + TTT eval ~600s
    # Plus: data download ~120s + startup ~60s
    seed_timeout_min = max(SEED_TIMEOUT_MIN, (max_wallclock // 60) + 60)

    # Run training with timeout; export PATH so pyminify/lrzip are findable
    # Use nvidia-smi to auto-detect GPU count for flexibility across 4/8 GPU configs
    # Unset PGOLF_BUNDLE env vars to prevent large env from confusing NCCL/torch distributed
    # Set NCCL_SHM_DISABLE=1 to work around corrupted /dev/shm on some RunPod community machines
    parts.append(
        f"timeout {seed_timeout_min}m bash -c "
        f"'export PATH=/usr/local/bin:/usr/bin:/root/.local/bin:$PATH && "
        f"unset PGOLF_BUNDLE_B64 PGOLF_BUNDLE_PARTS $(env | grep -o \"PGOLF_BUNDLE_PART_[0-9]*\" | tr \"\\n\" \" \") 2>/dev/null; "
        f"export NCCL_SHM_DISABLE=1 && "
        f"NGPUS=$(nvidia-smi -L | wc -l) && echo \"Detected $NGPUS GPUs\" && "
        f"{env} torchrun --standalone --nproc_per_node=$NGPUS train_gpt.py'; "
        f"echo $? > /root/rehearsal_out/seed{seed}_exit.txt"
    )

    # Copy training log
    parts.append(
        f"cp {artifact_dir}/train_seed{seed}.txt "
        f"/root/rehearsal_out/seed{seed}_log.txt 2>/dev/null || true"
    )

    # TTT sweep after training (if enabled)
    if getattr(args, "run_ttt_sweep_after_train", False):
        ttt_max_min = getattr(args, "ttt_max_minutes_per_variant", 20)
        sweep_cmd = (
            f"python3 scripts/run_longtrain_ttt_sweep.py "
            f"--artifact {artifact_dir}/final_model.int6.ptz "
            f"--output-dir {artifact_dir}/ttt_sweep "
            f"--train-script train_gpt.py "
            f"--data-path {data_path} "
            f"--tokenizer-path {tok_path} "
            f"--ngpus $(nvidia-smi -L | wc -l) "
            f"--max-minutes-per-variant {ttt_max_min}"
        )
        ttt_variants = getattr(args, "ttt_sweep_variants", None)
        if ttt_variants:
            sweep_cmd += f" --variants {ttt_variants}"
        parts.append(f"echo '=== RUNNING TTT SWEEP ===' && {sweep_cmd}")
        # Copy sweep results to rehearsal_out for HTTP serving
        parts.append(
            f"mkdir -p /root/rehearsal_out/ttt_sweep && "
            f"cp {artifact_dir}/ttt_sweep/ttt_sweep_manifest.json "
            f"/root/rehearsal_out/ttt_sweep/ttt_sweep_manifest.json 2>/dev/null || true && "
            f"cp {artifact_dir}/ttt_sweep/ttt_sweep_results.csv "
            f"/root/rehearsal_out/ttt_sweep/ttt_sweep_results.csv 2>/dev/null || true && "
            f"cp {artifact_dir}/ttt_sweep/ttt_sweep_summary.json "
            f"/root/rehearsal_out/ttt_sweep/ttt_sweep_summary.json 2>/dev/null || true"
        )

    # Copy checkpoint JSONs and .ptz files to rehearsal_out root for HTTP serving
    # JSONs are in artifact_dir root, .ptz files are in ckpt_Xmin/ subdirectories
    minutes_list = parse_export_minutes(export_minutes)
    for m in minutes_list:
        parts.append(
            f"cp {artifact_dir}/checkpoint_{m}min.json "
            f"/root/rehearsal_out/checkpoint_{m}min.json 2>/dev/null || true"
        )
        parts.append(
            f"cp {artifact_dir}/ckpt_{m}min/final_model.int6.{m}min.ptz "
            f"/root/rehearsal_out/final_model.int6.{m}min.ptz 2>/dev/null || true"
        )

    # Copy final model and scaling results
    parts.append(
        f"cp {artifact_dir}/final_model.int6.ptz "
        f"/root/rehearsal_out/final_model.int6.ptz 2>/dev/null || true"
    )
    parts.append(
        f"cp {artifact_dir}/scaling_results.csv "
        f"/root/rehearsal_out/scaling_results.csv 2>/dev/null || true"
    )

    # Summary
    parts.append("echo '=== LONGTRAIN SCALING SUMMARY ==='")
    parts.append(
        f"echo 'Seed {seed} exit:' && "
        f"cat /root/rehearsal_out/seed{seed}_exit.txt 2>/dev/null || echo 'unknown'"
    )
    parts.append(
        f"echo 'Seed {seed} log tail:' && "
        f"tail -50 /root/rehearsal_out/seed{seed}_log.txt 2>/dev/null || echo 'no log'"
    )
    parts.append(f"ls -la {artifact_dir}/ 2>/dev/null || true")
    parts.append("ls -la /root/rehearsal_out/")

    return " && ".join(parts)


def build_sweep_only_cmd(args):
    """Build command for TTT-sweep-only pod (no training).

    The artifact is uploaded via HTTP to /root/rehearsal_src/artifact/final_model.int6.ptz.
    """
    download_script = build_download_caseops_script()
    data_path = f"/root/caseops_data/datasets/datasets/{CASEOPS_DATASET_DIR}"
    tok_path = f"/root/caseops_data/datasets/tokenizers/{CASEOPS_TOKENIZER}"
    sweep_output = "/root/rehearsal_out/ttt_sweep"
    artifact_on_pod = "/root/rehearsal_src/artifact/final_model.int6.ptz"

    parts = []
    parts.append("cd /root/rehearsal_src")

    # Install deps
    parts.append(
        "apt-get update -qq && apt-get install -y -qq lrzip 2>&1 | tail -3"
    )
    parts.append(
        "pip install --break-system-packages -r requirements.txt brotli python-minifier 2>&1 | tail -5"
    )
    parts.append("hash -r")

    # Preflight
    parts.append(
        'python3 -c "import brotli, sentencepiece, numpy, torch; '
        'from flash_attn_interface import flash_attn_func; '
        "print('Preflight OK')\""
    )

    # Download CaseOps data
    parts.append(f"python3 -c {_shell_quote(download_script)}")

    # Verify artifact was uploaded
    parts.append(f"ls -la {artifact_on_pod}")

    # Clean env for distributed training
    parts.append(
        'unset PGOLF_BUNDLE_B64 PGOLF_BUNDLE_PARTS '
        '$(env | grep -o "PGOLF_BUNDLE_PART_[0-9]*" | tr "\\n" " ") 2>/dev/null; '
        'export NCCL_SHM_DISABLE=1'
    )

    # Run TTT sweep
    ttt_max_min = getattr(args, "ttt_max_minutes_per_variant", 20)
    sweep_cmd = (
        f"python3 scripts/run_longtrain_ttt_sweep.py "
        f"--artifact {artifact_on_pod} "
        f"--output-dir {sweep_output} "
        f"--train-script train_gpt.py "
        f"--data-path {data_path} "
        f"--tokenizer-path {tok_path} "
        f"--ngpus $(nvidia-smi -L | wc -l) "
        f"--max-minutes-per-variant {ttt_max_min}"
    )
    ttt_variants = getattr(args, "ttt_sweep_variants", None)
    if ttt_variants:
        sweep_cmd += f" --variants {ttt_variants}"
    # Include optional variant if include-optional flag is set
    sweep_cmd += " --include-optional"

    parts.append(f"echo '=== RUNNING TTT SWEEP (sweep-only mode) ===' && {sweep_cmd}")

    # List outputs
    parts.append(f"ls -la {sweep_output}/ 2>/dev/null || true")

    return " && ".join(parts)


def build_download_list(seed, export_minutes_str, include_ttt_sweep=False):
    """Build list of files to download from the pod after completion."""
    files = ["status.txt", "pgolf_exit_code.txt", "pgolf_stdout.txt"]
    files.append(f"seed{seed}_log.txt")
    files.append(f"seed{seed}_exit.txt")

    for m in parse_export_minutes(export_minutes_str):
        files.append(f"checkpoint_{m}min.json")
        files.append(f"final_model.int6.{m}min.ptz")

    files.append("final_model.int6.ptz")
    files.append("scaling_results.csv")

    if include_ttt_sweep:
        files.append("ttt_sweep/ttt_sweep_manifest.json")
        files.append("ttt_sweep/ttt_sweep_results.csv")
        files.append("ttt_sweep/ttt_sweep_summary.json")

    return files


def build_monitor_file_list(seed, export_minutes_str):
    """Checkpoint files to poll for during training (in artifact subdirectory).

    The HTTP server serves /root/rehearsal_out/, so files written by the
    training script to ARTIFACT_DIR=/root/rehearsal_out/seed<N>/ are
    accessible at seed<N>/<filename> through the proxy.
    JSONs are in artifact_dir root; .ptz files are in ckpt_Xmin/ subdirs.
    """
    files = []
    for m in parse_export_minutes(export_minutes_str):
        files.append(f"seed{seed}/checkpoint_{m}min.json")
        files.append(f"seed{seed}/ckpt_{m}min/final_model.int6.{m}min.ptz")
    return files


# ---------------------------------------------------------------------------
# Standard run (no monitoring) — proven sys.argv + http_main() pattern
# ---------------------------------------------------------------------------

def run_standard(args, cmd, download_files, train_script):
    """Standard run: delegates to http_main() via sys.argv (same as run_1934_repro.py)."""
    sys.argv = [
        "runpod_http_rehearsal.py",
        "--gpus", str(args.num_gpus),
        "--max-minutes", str(args.max_minutes),
        "--pod-name", build_pod_name(args),
        "--train-script", train_script,
        "--cmd", cmd,
        "--download",
    ] + download_files

    if args.results_dir:
        sys.argv.extend(["--results-dir", str(args.results_dir)])

    # Bundle TTT sweep script if sweep is requested
    if getattr(args, "run_ttt_sweep_after_train", False):
        sweep_script = REPO_ROOT / "scripts" / "run_longtrain_ttt_sweep.py"
        if sweep_script.exists():
            sys.argv.extend(["--extra-file", "{}:scripts/run_longtrain_ttt_sweep.py".format(sweep_script)])

    # Wire SSH upload for continuation resume snapshots
    if getattr(args, "continuation_label", None) and getattr(args, "resume_from", None):
        snap_dir = args.resume_from
        if Path(snap_dir).is_dir():
            ssh_specs = build_resume_ssh_uploads(snap_dir)
            for spec in ssh_specs:
                sys.argv.extend(["--ssh-upload", spec])

    http_main()


# ---------------------------------------------------------------------------
# Monitored run — polls for checkpoint files during training
# ---------------------------------------------------------------------------

def _check_terminal_status(pod_id):
    """Non-blocking check for terminal status via HTTP proxy. Returns status or None."""
    url = "https://{}-30000.proxy.runpod.net/status.txt".format(pod_id)
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", UA)
        with urllib.request.urlopen(req, timeout=15, context=_make_ssl_ctx()) as r:
            body = r.read().decode("utf-8", errors="replace").strip()
        if body in HTTP_TERMINAL_STATUSES:
            return body
    except Exception:
        pass
    return None


def _monitor_download_log_tail(pod_id, seed, out_dir):
    """Best-effort download of partial training log for progress reporting."""
    log_name = "seed{s}/train_seed{s}.txt".format(s=seed)
    try:
        log_path = download_file(
            pod_id, 30000, log_name, out_dir,
            optional=True, local_name="seed{}_log_partial.txt".format(seed),
        )
        if log_path:
            with open(log_path, "r", errors="replace") as f:
                lines = f.readlines()
            if lines:
                return lines[-1].strip()
    except Exception:
        pass
    return None


def run_with_monitoring(args, cmd, download_files, train_script):
    """Run with periodic checkpoint download during training."""
    max_minutes = args.max_minutes
    seed = args.seed

    # --- balance check ---
    cost_est = args.num_gpus * H100_COST_PER_GPU_HR * max_minutes / 60.0
    bal, _ = balance()
    print("Balance: ${:.2f}  Est cost: ${:.2f}  ({} GPUs, {} min)".format(
        bal, cost_est, args.num_gpus, max_minutes))
    if bal < cost_est * 1.05:
        raise SystemExit(
            "ERROR: Insufficient balance (need >= 1.05× est cost = ${:.2f})".format(cost_est * 1.05)
        )

    # --- build bundle ---
    ts = Path(train_script) if train_script else None
    extra_files = []
    if getattr(args, "run_ttt_sweep_after_train", False):
        sweep_script = os.path.join(REPO_ROOT, "scripts", "run_longtrain_ttt_sweep.py")
        if os.path.exists(sweep_script):
            extra_files.append((sweep_script, "scripts/run_longtrain_ttt_sweep.py"))
    bundle_b64 = build_bundle_b64(train_script=ts, extra_files=extra_files or None)

    CHUNK_SIZE = 32 * 1024
    chunk_env = {"PGOLF_MAX_MINUTES": str(max_minutes)}
    if len(bundle_b64) <= CHUNK_SIZE:
        chunk_env["PGOLF_BUNDLE_B64"] = bundle_b64
        chunk_env["PGOLF_BUNDLE_PARTS"] = "0"
    else:
        n_parts = (len(bundle_b64) + CHUNK_SIZE - 1) // CHUNK_SIZE
        chunk_env["PGOLF_BUNDLE_PARTS"] = str(n_parts)
        for i in range(n_parts):
            chunk_env["PGOLF_BUNDLE_PART_{:03d}".format(i)] = (
                bundle_b64[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
            )
        print("Bundle chunked: {} bytes -> {} parts of {} bytes".format(
            len(bundle_b64), n_parts, CHUNK_SIZE))

    docker_args = build_boot_command(cmd)
    hard_deadline_sec = max_minutes * 60 + 120

    pod_id = None
    out_dir = None
    launcher_state = None
    original_exc = None

    try:
        # Try multiple GPU types and cloud configurations
        # Filter to only the requested GPU count to prevent accidental mismatch
        pod = None
        all_gpu_types = [
            ("NVIDIA H100 80GB HBM3", "H100 SXM", 8),
            ("NVIDIA H100 NVL", "H100 NVL", 8),
            ("NVIDIA H200", "H200 SXM", 8),
            ("NVIDIA H100 80GB HBM3", "H100 SXM", 4),
            ("NVIDIA H100 NVL", "H100 NVL", 4),
        ]
        gpu_types = [(t, l, c) for t, l, c in all_gpu_types if c == args.num_gpus]
        cloud_types = ["COMMUNITY", "SECURE"]
        actual_gpus = args.num_gpus
        pod_name = build_pod_name(args)
        for gpu_type_id, gpu_label, gpu_count in gpu_types:
            for cloud_type in cloud_types:
                try:
                    pod = create_pod(
                        name=pod_name,
                        gpus=gpu_count,
                        max_minutes=max_minutes,
                        docker_args=docker_args,
                        extra_env=chunk_env,
                        ports="30000/http,22/tcp",
                        start_ssh=False,
                        deadline_sec=hard_deadline_sec,
                        cloud_type=cloud_type,
                        gpu_type_id=gpu_type_id,
                    )
                    actual_gpus = gpu_count
                    print("Pod created: {}×{} on {} cloud".format(gpu_count, gpu_label, cloud_type))
                    break
                except RuntimeError as e:
                    if "SUPPLY_CONSTRAINT" in str(e) or "no longer any instances" in str(e):
                        print("No {}×{} on {} cloud, trying next...".format(gpu_count, gpu_label, cloud_type))
                        continue
                    raise
            if pod is not None:
                break
        if pod is None:
            raise RuntimeError("No suitable GPU configuration available")
        pod_id = pod["id"]
        out_dir = (
            Path(args.results_dir) if args.results_dir
            else Path(REPO_ROOT) / "results" / "pod_{}_longtrain".format(pod_id)
        )

        launcher_state = build_launcher_state(
            launcher="run_longtrain_scaling",
            pod_id=pod_id,
            pod_name=build_pod_name(args),
            gpus=args.num_gpus,
            max_minutes=max_minutes,
            results_dir=out_dir,
            hard_deadline_sec=hard_deadline_sec,
            bundle_b64=bundle_b64,
            command=cmd,
            docker_args=docker_args,
        )
        launcher_state["cost_per_hr"] = pod.get("costPerHr")
        write_launcher_state(out_dir, launcher_state)

        print("Pod: {}  ${}/hr  name={}".format(
            pod_id, pod.get("costPerHr", "?"), pod_name))

        rt = wait_runtime(pod_id)
        print("Pod RUNNING (uptime={}s)".format(rt["uptimeInSeconds"]))

        wait_startup_readiness_and_maybe_download_status(pod_id, 30000, out_dir)

        # --- monitoring loop ---
        monitor_files = build_monitor_file_list(seed, args.export_minutes)
        downloaded_set = set()
        terminal_timeout = max(180, max_minutes * 60 + 60)
        deadline = time.time() + terminal_timeout

        print("Monitoring: polling every {}s for {} checkpoint files (deadline in {}s)".format(
            POLL_INTERVAL_SEC, len(monitor_files), terminal_timeout))

        terminal_status = None
        while time.time() < deadline:
            # Check for terminal status first
            terminal_status = _check_terminal_status(pod_id)
            if terminal_status is not None:
                print("Terminal status reached: {}".format(terminal_status))
                break

            # Poll for new checkpoint files in the artifact subdirectory
            newly_downloaded = 0
            for fname in monitor_files:
                if fname in downloaded_set:
                    continue
                try:
                    path = download_file(pod_id, 30000, fname, out_dir, optional=True)
                    if path:
                        downloaded_set.add(fname)
                        newly_downloaded += 1
                        print("  [MONITOR] Downloaded: {} ({} bytes)".format(
                            path.name, path.stat().st_size))
                except Exception:
                    pass

            # Best-effort log tail for progress
            tail = _monitor_download_log_tail(pod_id, seed, out_dir)
            remaining_min = (deadline - time.time()) / 60.0
            status_line = "  [MONITOR] {}/{} checkpoint files, {:.0f}min remaining".format(
                len(downloaded_set), len(monitor_files), remaining_min)
            if tail:
                status_line += " | log: {}".format(tail[:120])
            print(status_line)

            time.sleep(POLL_INTERVAL_SEC)
        else:
            raise RuntimeError(
                "HTTP endpoint did not reach terminal status within {}s".format(terminal_timeout)
            )

        # --- download final artifacts ---
        print("Downloading final artifacts...")
        for name in download_files:
            optional = name.endswith(
                (".ptz", ".pt", "_log.txt", "_exit.txt", ".json", ".csv")
            )
            path = download_file(pod_id, 30000, name, out_dir, optional=optional)
            if path:
                print("  {} ({})".format(path.name, path.stat().st_size))
            else:
                print("  {} (not found, skipped)".format(name))

    except BaseException as exc:
        original_exc = exc
        if pod_id is not None and out_dir is not None and launcher_state is not None:
            try:
                record_launcher_exception(out_dir, launcher_state, exc)
            except BaseException as state_exc:
                print(
                    "WARNING: failed to record launcher exception for pod {}: {}".format(
                        pod_id, state_exc.__class__.__name__
                    ),
                    file=sys.stderr,
                )
        raise
    finally:
        if pod_id is not None and out_dir is not None and launcher_state is not None:
            print("Terminating pod {}...".format(pod_id))
            try:
                terminate_pod_with_launcher_state(
                    out_dir, launcher_state, pod_id, terminate_and_wait,
                    original_exc=original_exc,
                )
            except BaseException as cleanup_exc:
                if original_exc is None:
                    raise
                print(
                    "WARNING: failed during cleanup for pod {} after {}: {}".format(
                        pod_id, original_exc.__class__.__name__,
                        cleanup_exc.__class__.__name__,
                    ),
                    file=sys.stderr,
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_arg_parser():
    """Build the argument parser (extracted for testability)."""
    parser = argparse.ArgumentParser(
        description="Long-train artifact scaling experiment: training with periodic checkpoint exports"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Training seed (default: {})".format(DEFAULT_SEED),
    )
    parser.add_argument(
        "--max-minutes", type=int, default=DEFAULT_MAX_MINUTES,
        help="Pod wallclock limit in minutes (default: {})".format(DEFAULT_MAX_MINUTES),
    )
    parser.add_argument(
        "--max-wallclock", type=int, default=DEFAULT_MAX_WALLCLOCK,
        help="MAX_WALLCLOCK_SECONDS for training (default: {})".format(DEFAULT_MAX_WALLCLOCK),
    )
    parser.add_argument(
        "--export-minutes", default=DEFAULT_EXPORT_MINUTES,
        help="Comma-separated checkpoint export times in minutes (default: {})".format(
            DEFAULT_EXPORT_MINUTES),
    )
    parser.add_argument(
        "--export-mode", default=DEFAULT_EXPORT_MODE,
        help="Export mode for checkpoints (default: {})".format(DEFAULT_EXPORT_MODE),
    )
    parser.add_argument(
        "--train-script", default=None,
        help="Override train_gpt.py path (default: repo root train_gpt.py)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Override results directory",
    )
    parser.add_argument(
        "--download-checkpoints", action="store_true",
        help="Enable periodic polling and download of checkpoint files during training",
    )
    parser.add_argument(
        "--duration-hours", type=int, default=None,
        help="Training duration in hours (auto-sets wallclock, max-minutes, export, resume defaults)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override default ITERATIONS env var",
    )
    parser.add_argument(
        "--enable-resume", action="store_true",
        help="Enable checkpoint resume (RESUME_ENABLED=1)",
    )
    parser.add_argument(
        "--resume-save-minutes", default=None,
        help="Comma-separated resume checkpoint save times in minutes",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Path for RESUME_FROM env var (resume from a prior checkpoint)",
    )
    parser.add_argument(
        "--resume-keep-last", type=int, default=3,
        help="RESUME_KEEP_LAST: number of resume checkpoints to keep (default: 3)",
    )
    parser.add_argument(
        "--run-ttt-sweep-after-train", action="store_true",
        help="Run TTT sweep on final artifact after training completes",
    )
    parser.add_argument(
        "--ttt-sweep-variants", default=None,
        help="Comma-separated TTT variant IDs for sweep (default: all)",
    )
    parser.add_argument(
        "--ttt-max-minutes-per-variant", type=int, default=20,
        help="Timeout per TTT variant in minutes (default: 20)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8,
        help="Number of GPUs for the pod (default: 8). Continuation runs require 4.",
    )
    parser.add_argument(
        "--continuation-label", default=None,
        help="Label for resumed continuation runs (e.g. 'resumed_6h_horizon'). "
             "Forces --num-gpus=4 if not explicitly set to 4.",
    )
    parser.add_argument(
        "--schedule-horizon", type=int, default=None,
        help="SCHEDULE_HORIZON_SECONDS env var for the train script (seconds). "
             "Sets the LR schedule horizon for continuation runs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print command and settings but don't launch pod",
    )
    parser.add_argument(
        "--sweep-only-artifact", default=None,
        help="Skip training; run TTT sweep only on this local .ptz artifact. "
             "Uploads artifact to pod via HTTP and runs the sweep.",
    )
    return parser


def apply_post_parse_defaults(args):
    """Apply derived defaults after parsing (extracted for testability).

    Raises SystemExit if continuation-label + num-gpus conflict detected.
    """
    # --- Continuation label safety gate ---
    if args.continuation_label is not None:
        # Detect if user explicitly set --num-gpus
        explicitly_set = any(a == "--num-gpus" or a.startswith("--num-gpus=") for a in sys.argv)
        if not explicitly_set:
            # Default to 4 for continuations
            args.num_gpus = 4
        elif args.num_gpus != 4:
            # User explicitly set something other than 4; reject
            raise SystemExit(
                "ERROR: --continuation-label requires --num-gpus=4 "
                "(got {}). Refusing to launch on {} GPUs for a resumed "
                "continuation.".format(args.num_gpus, args.num_gpus)
            )

    # Apply duration-hours defaults when set
    if args.duration_hours is not None:
        h = args.duration_hours
        if args.max_wallclock == DEFAULT_MAX_WALLCLOCK:
            args.max_wallclock = h * 3600
        if args.max_minutes == DEFAULT_MAX_MINUTES:
            args.max_minutes = h * 60 + 60
        if args.export_minutes == DEFAULT_EXPORT_MINUTES:
            args.export_minutes = DEFAULT_4H_EXPORT_MINUTES
        if args.resume_save_minutes is None:
            args.resume_save_minutes = DEFAULT_4H_RESUME_SAVE_MINUTES
        if args.iterations is None:
            args.iterations = DEFAULT_4H_ITERATIONS

    # If TTT sweep is enabled and user hasn't explicitly overridden max-minutes,
    # add sweep time to pod budget automatically
    if getattr(args, "run_ttt_sweep_after_train", False):
        ttt_max_min = getattr(args, "ttt_max_minutes_per_variant", 20)
        num_variants = 6  # default (optional excluded)
        if getattr(args, "ttt_sweep_variants", None):
            num_variants = len(args.ttt_sweep_variants.split(","))
        sweep_budget_min = num_variants * ttt_max_min + 15
        # Only auto-inflate if user relied on defaults
        if "--max-minutes" not in sys.argv:
            args.max_minutes = args.max_minutes + sweep_budget_min

    return args


def build_pod_name(args):
    """Build pod name, incorporating continuation label if present."""
    base = "pgolf-longtrain-scaling"
    if args.continuation_label:
        # Replace underscores with hyphens for pod-name-friendliness
        label = args.continuation_label.replace("_", "-")
        return "{}-{}".format(base, label)
    return base


def build_dry_run_summary(args):
    """Build dry-run summary string reflecting actual GPU count and label."""
    lines = []
    lines.append("=== SETTINGS ===")
    if args.continuation_label:
        lines.append("Continuation: {} (resumed, NOT a fresh run)".format(
            args.continuation_label))
    lines.append("Seed: {}".format(args.seed))
    lines.append("Max pod minutes: {}".format(args.max_minutes))
    lines.append("MAX_WALLCLOCK_SECONDS: {}".format(args.max_wallclock))
    if hasattr(args, 'export_minutes'):
        lines.append("Export minutes: {}".format(args.export_minutes))
    lines.append("Export mode: {}".format(args.export_mode))
    if args.duration_hours is not None:
        lines.append("Duration hours: {}".format(args.duration_hours))
    if args.iterations is not None:
        lines.append("Iterations: {}".format(args.iterations))
    if getattr(args, "schedule_horizon", None) is not None:
        lines.append("Schedule horizon: {}s".format(args.schedule_horizon))
    lines.append("Resume enabled: {}".format(args.enable_resume))
    if args.resume_from:
        lines.append("Resume from: {}".format(args.resume_from))
    lines.append("GPUs: {}".format(args.num_gpus))
    hrs = args.max_minutes / 60.0
    lines.append("Est cost: ${:.2f}".format(args.num_gpus * H100_COST_PER_GPU_HR * hrs))
    lines.append("Pod name: {}".format(build_pod_name(args)))
    return "\n".join(lines)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    apply_post_parse_defaults(args)

    # --- Sweep-only mode ---
    if args.sweep_only_artifact:
        artifact_path = os.path.abspath(args.sweep_only_artifact)
        if not os.path.exists(artifact_path):
            raise SystemExit("ERROR: --sweep-only-artifact not found: {}".format(artifact_path))

        cmd = build_sweep_only_cmd(args)
        # Build comprehensive download list for sweep results
        variant_ids = [
            "v_sliding_window_control",
            "v0_control_pr1979", "v1_rank128_alpha192", "v2_rank128_lr3e4",
            "v3_local_batch_chunk", "v4_global2_largechunk", "v5_prefix3000",
            "v6_prefix3000_phase4_optional",
        ]
        download_files = [
            "status.txt", "pgolf_exit_code.txt", "pgolf_stdout.txt",
            "ttt_sweep/ttt_sweep_manifest.json",
            "ttt_sweep/ttt_sweep_results.csv",
            "ttt_sweep/ttt_sweep_summary.json",
        ]
        for vid in variant_ids:
            download_files.append(f"ttt_sweep/{vid}/variant_result.json")
            download_files.append(f"ttt_sweep/{vid}/eval.log")
            download_files.append(f"ttt_sweep/{vid}/sliding_eval_summary.json")

        if args.dry_run:
            print("=== SWEEP-ONLY MODE ===")
            print("Artifact: {} ({:.1f} MB)".format(
                artifact_path, os.path.getsize(artifact_path) / 1048576))
            print("GPUs: {}".format(args.num_gpus))
            print("Max minutes: {}".format(args.max_minutes))
            print("TTT variants: {}".format(args.ttt_sweep_variants or "all + optional"))
            print("TTT max min/variant: {}".format(args.ttt_max_minutes_per_variant))
            print("\n=== POD COMMAND ===")
            print(cmd)
            print("\nFiles to download:")
            for f in download_files:
                print("  {}".format(f))
            return

        # Build sys.argv for http_main
        train_script = args.train_script or os.path.join(
            REPO_ROOT, "records", "track_non_record_16mb",
            "2026-04-30_PR1950_LongTrainArtifactScaling", "train_gpt.py"
        )
        sys.argv = [
            "runpod_http_rehearsal.py",
            "--gpus", str(args.num_gpus),
            "--max-minutes", str(args.max_minutes),
            "--pod-name", "pgolf-ttt-sweep",
            "--train-script", train_script,
            "--cmd", cmd,
            "--download",
        ] + download_files

        # Bundle the sweep script
        sweep_script = os.path.join(REPO_ROOT, "scripts", "run_longtrain_ttt_sweep.py")
        if os.path.exists(sweep_script):
            sys.argv.extend(["--extra-file", "{}:scripts/run_longtrain_ttt_sweep.py".format(sweep_script)])

        # Upload the artifact via HTTP
        sys.argv.extend(["--ssh-upload", "{}:artifact/final_model.int6.ptz".format(artifact_path)])

        if args.results_dir:
            sys.argv.extend(["--results-dir", str(args.results_dir)])

        http_main()
        return

    # --- Standard training mode ---
    # Resolve train script — default to the long-train modified version
    _longtrain_default = os.path.join(
        REPO_ROOT, "records", "track_non_record_16mb",
        "2026-04-30_PR1950_LongTrainArtifactScaling", "train_gpt.py"
    )
    train_script = args.train_script or _longtrain_default
    if not os.path.exists(train_script):
        raise SystemExit("ERROR: train script not found: {}".format(train_script))

    cmd = build_seed_cmd(args)
    download_files = build_download_list(
        args.seed, args.export_minutes,
        include_ttt_sweep=args.run_ttt_sweep_after_train,
    )

    if args.dry_run:
        print("=== POD COMMAND ===")
        print(cmd)
        print()
        print(build_dry_run_summary(args))
        print("Train script: {}".format(train_script))
        print("Per-seed timeout: {} min".format(max(SEED_TIMEOUT_MIN, (args.max_wallclock // 60) + 60)))
        print("Download checkpoints (monitoring): {}".format(args.download_checkpoints))
        if args.enable_resume or args.resume_save_minutes:
            print("Resume save minutes: {}".format(
                parse_export_minutes(args.resume_save_minutes) if args.resume_save_minutes else "N/A"))
            print("Resume keep last: {}".format(args.resume_keep_last))
        print("TTT sweep after train: {}".format(args.run_ttt_sweep_after_train))
        if args.run_ttt_sweep_after_train:
            print("TTT variants: {}".format(args.ttt_sweep_variants or "all"))
            print("TTT max minutes per variant: {}".format(args.ttt_max_minutes_per_variant))
        print("\nFiles to download ({}):".format(len(download_files)))
        for f in download_files:
            print("  {}".format(f))
        if args.download_checkpoints:
            monitor_files = build_monitor_file_list(args.seed, args.export_minutes)
            print("\nFiles to monitor during training ({}):".format(len(monitor_files)))
            for f in monitor_files:
                print("  {}".format(f))
        return

    if args.download_checkpoints:
        run_with_monitoring(args, cmd, download_files, train_script)
    else:
        run_standard(args, cmd, download_files, train_script)


if __name__ == "__main__":
    main()
