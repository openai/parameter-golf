"""
wave46_clip_prune.py — Int6 + 25% pruning + 99th percentile clipping.

Goal: fit int6 quality (1.0677 BPB base) under 16MB by:
1. Tighter clip percentile: 99.0 (vs 99.99984) -- more weights cluster near int6 boundaries
2. Higher pruning: 25% (vs 15%) -- zeros compress extremely well in zstd
Both levers push artifact size down without changing architecture.
Uses the 1.0677 base script (best architecture: SwiGLU + U-Net + XSA4 + BigramHash).

Artifacts and results are written to Modal Volume 'parameter-golf-data'.
Job runs fully on Modal infrastructure regardless of what happens locally.

Usage:
  Launch:   modal run modal/wave31_detached.py --tag wave31-warmdown6k
  Retrieve: modal run modal/wave31_detached.py::retrieve --tag wave31-warmdown6k
  Status:   modal app logs <app-id>
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

# ── App + Volume ──────────────────────────────────────────────────────────────
app = modal.App("parameter-golf-wave46")

# Persistent volume — survives client disconnects and app stops
results_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=False)

VOLUME_RESULTS_DIR = "/vol/results"
VOLUME_ARTIFACTS_DIR = "/vol/artifacts"

# ── Image ─────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.6",
        "numpy",
        "requests",
        "zstandard",
        "huggingface_hub",
        "datasets",
        "sentencepiece",
        "tqdm",
    )
    .apt_install("git")
    .run_commands(
        "git clone --depth=1 https://github.com/openai/parameter-golf.git /opt/pg-repo",
        "cd /opt/pg-repo && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80",
    )
)


# ── GPU Function ──────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="H100:8",
    timeout=3600,
    retries=3,
    cloud="gcp",
    volumes={"/vol": results_vol},
)
def run_experiment(train_script: str, run_tag: str, env_overrides: dict | None = None) -> dict:
    workdir = "/tmp/parameter-golf"
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(VOLUME_RESULTS_DIR, exist_ok=True)
    os.makedirs(VOLUME_ARTIFACTS_DIR, exist_ok=True)

    # Write training script
    script_path = os.path.join(workdir, "train_gpt.py")
    with open(script_path, "w") as f:
        f.write(train_script)

    # Symlink data
    data_base = os.path.join(workdir, "data")
    os.makedirs(data_base, exist_ok=True)
    for name in ["datasets", "tokenizers"]:
        link = os.path.join(data_base, name)
        if not os.path.exists(link):
            os.symlink(f"/opt/pg-repo/data/{name}", link)

    dataset_path = os.path.join(data_base, "datasets", "fineweb10B_sp1024")
    tokenizer_path = os.path.join(data_base, "tokenizers", "fineweb_1024_bpe.model")

    print(f"Starting 8xH100 run: {run_tag}")
    start = time.time()

    log_path = os.path.join(workdir, "train_output.log")
    env_parts = [
        f'DATA_PATH="{dataset_path}"',
        f'TOKENIZER_PATH="{tokenizer_path}"',
        "TORCH_SHOW_CPP_STACKTRACES=1",
        "PYTHONFAULTHANDLER=1",
    ]
    if env_overrides:
        for k, v in env_overrides.items():
            env_parts.append(f'{k}="{v}"')
    env_str = " ".join(env_parts)

    cmd = f"cd {workdir} && {env_str} torchrun --nproc_per_node=8 {script_path} > {log_path} 2>&1"
    exit_status = os.system(cmd)
    return_code = exit_status >> 8

    # Read log
    full_log = ""
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            full_log = f.read()

    elapsed = time.time() - start

    # Parse val_bpb from log
    val_bpb = None
    for line in reversed(full_log.split("\n")):
        if "val_bpb" in line:
            try:
                val_str = line.split("val_bpb")[-1].strip().lstrip(":").strip()
                val_bpb = float(val_str.split()[0])
                break
            except (ValueError, IndexError):
                pass

    log_tail = "\n".join(full_log.split("\n")[-120:])
    status = "crash" if return_code != 0 else ("success" if val_bpb else "no_metric")

    # Find and read artifact
    artifact_bytes = None
    artifact_size = 0
    artifact_fname = None
    for fname in ["final_model.ptz", "final_model.int6.ptz", "final_model.int5.ptz"]:
        artifact_path_local = os.path.join(workdir, fname)
        if os.path.exists(artifact_path_local):
            with open(artifact_path_local, "rb") as af:
                artifact_bytes = af.read()
            artifact_size = len(artifact_bytes)
            artifact_fname = fname
            print(f"Artifact found: {fname} = {artifact_size:,} bytes ({artifact_size / 1024 / 1024:.2f} MB)")
            break

    # ── Save artifact to Volume (survives client disconnect) ──────────────────
    if artifact_bytes:
        vol_artifact_path = os.path.join(VOLUME_ARTIFACTS_DIR, f"{run_tag}_model.ptz")
        with open(vol_artifact_path, "wb") as af:
            af.write(artifact_bytes)
        print(f"Artifact saved to volume: {vol_artifact_path}")
        results_vol.commit()

    # ── Build and save result JSON to Volume ──────────────────────────────────
    result = {
        "tag": run_tag,
        "val_bpb": val_bpb,
        "status": status,
        "elapsed": elapsed,
        "artifact_size": artifact_size,
        "artifact_mb": round(artifact_size / 1024 / 1024, 2),
        "artifact_fname": artifact_fname,
        "return_code": return_code,
        "timestamp": datetime.utcnow().isoformat(),
        "env_overrides": env_overrides or {},
        "log_tail": log_tail,
    }

    result_path = os.path.join(VOLUME_RESULTS_DIR, f"{run_tag}_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()

    # ── Stdout summary (visible in modal app logs) ────────────────────────────
    print(f"RESULT_JSON|{json.dumps({k: v for k, v in result.items() if k != 'log_tail'})}")
    print(f"Result: val_bpb={val_bpb} | status={status} | time={elapsed:.0f}s | artifact={artifact_size / 1024 / 1024:.2f}MB")

    if val_bpb and val_bpb < 1.1194:
        print(f"*** NEW #1: {val_bpb:.8f} BPB -- beats current #1 (1.1194) ***")
    if artifact_size > 0 and artifact_size <= 16 * 1024 * 1024:
        print(f"*** ARTIFACT VALID: {artifact_size / 1024 / 1024:.2f} MB <= 16.00 MB ***")
    elif artifact_size > 16 * 1024 * 1024:
        print(f"!!! ARTIFACT TOO LARGE: {artifact_size / 1024 / 1024:.2f} MB > 16.00 MB !!!")

    print(f"\nLast 120 lines of training log:\n{log_tail}")

    return result


# ── Retrieve Function (pull artifact + result from Volume) ────────────────────
@app.function(
    image=modal.Image.debian_slim(python_version="3.10"),
    volumes={"/vol": results_vol},
)
def fetch_from_volume(run_tag: str) -> dict:
    """Read result JSON and artifact bytes from the volume."""
    result_path = os.path.join(VOLUME_RESULTS_DIR, f"{run_tag}_result.json")
    artifact_path = os.path.join(VOLUME_ARTIFACTS_DIR, f"{run_tag}_model.ptz")

    result = {}
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            result = json.load(f)
        print(f"Result loaded from volume: {result_path}")
    else:
        print(f"No result file found at {result_path}")

    artifact_bytes = None
    artifact_size = 0
    if os.path.exists(artifact_path):
        with open(artifact_path, "rb") as af:
            artifact_bytes = af.read()
        artifact_size = len(artifact_bytes)
        print(f"Artifact loaded: {artifact_size:,} bytes ({artifact_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"No artifact found at {artifact_path}")

    return {
        "result": result,
        "artifact_bytes": artifact_bytes,
        "artifact_size": artifact_size,
    }


# ── Local Entrypoints ─────────────────────────────────────────────────────────
MODAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(MODAL_DIR, "..")
BEST_SCRIPTS_DIR = os.path.join(REPO_DIR, "best_scripts")
SCRIPT_PATH = os.path.join(BEST_SCRIPTS_DIR, "train_gpt_8x_bpb1.0677_w6-adamw-ttt.py")


@app.local_entrypoint()
def main(tag: str = "wave46-int6-clip99-prune25", env: str = "", no_poll: bool = False):
    """Launch a detached training run. Safe to Ctrl+C after launch."""
    with open(SCRIPT_PATH, "r") as f:
        script = f.read()

    env_overrides = {}
    if env:
        for pair in env.split(","):
            k, v = pair.split("=", 1)
            env_overrides[k.strip()] = v.strip()

    # Default wave32 env (warmdown=6000, AdamW TTT, score-first order, quant fixes)
    default_env = {
        "MATRIX_LR": "0.025",
        "SCALAR_LR": "0.025",
        "MUON_WD": "0.0",
        "ADAM_WD": "0.0",
        "GRAD_CLIP_NORM": "0.0",
        "MUON_MOMENTUM": "0.95",
        "WARMDOWN_ITERS": "6000",
        # TTT: AdamW, MLP-only, 1 epoch (proven safe, adds ~0.004 BPB)
        "TTT_ENABLED": "1",
        "TTT_USE_ADAMW": "1",
        "TTT_ADAMW_LR": "0.0004",
        "TTT_ADAMW_WD": "0.0",
        "TTT_MLP_ONLY": "1",
        "TTT_EPOCHS": "1",
        "TTT_FREEZE_BLOCKS": "0",
        # wave46: int6 + tighter clip + higher pruning to fit under 16MB
        # 99th percentile clip (vs 99.99984): more weights at boundaries = better zstd
        # 25% pruning (vs 15%): zeros compress ~10x vs live weights in zstd
        # int6 base: 1.0677 BPB -- 0.065 better than current int5 submission
        "INT6_CLIP_PERCENTILE": "99.0",
        "PRUNE_PCT": "0.25",
        # Keep original architecture from 1.0677 (MLP_HIDDEN=1792, BIGRAM_BUCKETS=8192)
    }
    # env_overrides from CLI take precedence
    merged_env = {**default_env, **env_overrides}

    print(f"\n{'=' * 60}")
    print(f"  Launching DETACHED run: {tag}")
    print(f"  Env: {merged_env}")
    print(f"  Artifact will be saved to Modal Volume 'parameter-golf-data'")
    print(f"  Safe to Ctrl+C -- job runs on Modal regardless")
    print(f"{'=' * 60}\n")

    # spawn() + handle.get() with --detach flag: job survives local disconnect
    handle = run_experiment.spawn(script, tag, merged_env)
    print(f"Job spawned: {handle.object_id}")
    print(f"Monitor: modal app logs (look for RESULT_JSON)")
    print(f"Retrieve: modal run modal/wave31_detached.py::retrieve --tag {tag}")
    print("\nWaiting for result (safe to Ctrl+C -- job runs on Modal)...")

    start_poll = time.time()
    try:
        result = handle.get(timeout=3600)
        elapsed = time.time() - start_poll

        bpb = result.get("val_bpb")
        status = result.get("status")
        artifact_size = result.get("artifact_size", 0)
        artifact_mb = artifact_size / 1024 / 1024

        print(f"\n{'=' * 60}")
        print(f"  DONE: val_bpb={bpb} | status={status} | {elapsed:.0f}s")
        print(f"  Artifact: {artifact_mb:.2f} MB")
        if bpb and bpb < 1.1194:
            print(f"  *** NEW #1: {bpb:.8f} BPB ***")
        if artifact_size > 0 and artifact_size <= 16 * 1024 * 1024:
            print(f"  *** ARTIFACT VALID ***")
        print(f"{'=' * 60}\n")

        # Save artifact locally
        artifact_bytes = result.get("artifact_bytes")
        if artifact_bytes:
            artifact_dir = os.path.join(REPO_DIR, "artifacts")
            os.makedirs(artifact_dir, exist_ok=True)
            out_path = os.path.join(artifact_dir, f"{tag}_model.ptz")
            with open(out_path, "wb") as af:
                af.write(artifact_bytes)
            print(f"Artifact saved locally: {out_path} ({artifact_size:,} bytes)")

        # Log to JSONL
        entry = {
            "timestamp": datetime.now().isoformat(),
            "tag": tag,
            "val_bpb": bpb,
            "status": status,
            "elapsed": elapsed,
            "env": merged_env,
            "artifact_size": artifact_size,
        }
        results_path = os.path.join(REPO_DIR, "wave7b_results.jsonl")
        with open(results_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    except KeyboardInterrupt:
        print(f"\nLocal client detached. Job may still be running on Modal.")
        print(f"When done, retrieve with:")
        print(f"  modal run modal/wave31_detached.py::retrieve --tag {tag}")


@app.local_entrypoint()
def retrieve(tag: str = "wave31-warmdown6k"):
    """Retrieve artifact + result from Modal Volume after a completed run."""
    print(f"Fetching results for tag: {tag}")

    data = fetch_from_volume.remote(tag)
    result = data.get("result", {})
    artifact_bytes = data.get("artifact_bytes")
    artifact_size = data.get("artifact_size", 0)

    if result:
        bpb = result.get("val_bpb")
        status = result.get("status")
        print(f"\nResult: val_bpb={bpb} | status={status} | artifact={artifact_size / 1024 / 1024:.2f}MB")
        if bpb and bpb < 1.1194:
            print(f"*** NEW #1: {bpb:.8f} BPB ***")
    else:
        print("No result found in volume -- run may not have completed yet.")
        return

    if artifact_bytes:
        artifact_dir = os.path.join(REPO_DIR, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        out_path = os.path.join(artifact_dir, f"{tag}_model.ptz")
        with open(out_path, "wb") as af:
            af.write(artifact_bytes)
        print(f"Artifact saved: {out_path} ({artifact_size:,} bytes, {artifact_size / 1024 / 1024:.2f} MB)")
        if artifact_size <= 16 * 1024 * 1024:
            print(f"*** ARTIFACT FITS: {artifact_size / 1024 / 1024:.2f} MB <= 16.00 MB ***")
        else:
            print(f"!!! TOO LARGE: {artifact_size / 1024 / 1024:.2f} MB ***")
    else:
        print("No artifact found in volume.")
