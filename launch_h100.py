#!/usr/bin/env python3
"""
Launch H100 on RunPod, run training, collect results, auto-terminate.

Two modes:
  --dev     1xH100 (~$3-4/hr) for iteration and debugging
  --submit  8xH100 (~$24/hr) for final submission validation

Usage:
    export RUNPOD_API_KEY=your_key_here
    python3 launch_h100.py --dev          # cheap dev runs
    python3 launch_h100.py --submit       # final 8xH100 run
"""

import argparse
import os
import sys
import time

import runpod

# ── Config ──────────────────────────────────────────────────────────────
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("Set RUNPOD_API_KEY env var first.")
    print("Get it from: https://www.runpod.io/console/user/settings")
    sys.exit(1)

runpod.api_key = RUNPOD_API_KEY

IMAGE = "runpod/parameter-golf:latest"
GPU_TYPE = "NVIDIA H100 80GB HBM3"
CLOUD_TYPE = "ALL"  # cheapest option

# Git branch to use
GIT_BRANCH = os.environ.get("GIT_BRANCH", "combined-approach")

# Env overrides for the training run (tweak these between runs)
TRAIN_ENV = {
    "USE_RANDOM_ADAPTERS": "0",
    "ADAPTER_RANK": "128",
}

def build_setup_script(train_shards):
    return f"""
set -ex
cd /workspace

# Check what's pre-installed in the parameter-golf template
python3 -c "import torch; print(f'PyTorch {{torch.__version__}}')" || true
python3 -c "import flash_attn; print(f'FlashAttn {{flash_attn.__version__}}')" 2>/dev/null || echo "No flash_attn"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# Clone repo (remove stale non-git dir from template if present)
if [ -d parameter-golf ] && [ ! -d parameter-golf/.git ]; then
    rm -rf parameter-golf
fi
if [ ! -d parameter-golf ]; then
    git clone https://github.com/dljr-github/parameter-golf.git
fi
cd parameter-golf
git fetch origin {GIT_BRANCH} && git checkout {GIT_BRANCH} && git reset --hard origin/{GIT_BRANCH}

# Install deps (some may already be in the template)
pip install -q sentencepiece brotli numpy huggingface-hub 2>/dev/null || true

# Download SP8192 dataset ({train_shards} train shards)
if [ ! -f data/datasets/fineweb10B_sp8192/fineweb_val_000000.bin ]; then
    rm -f data/manifest.json
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \\
    python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards {train_shards}
else
    echo "SP8192 dataset already cached"
fi

echo "=== SETUP COMPLETE ==="
ls -la data/datasets/fineweb10B_sp8192/
ls -la data/tokenizers/fineweb_8192_bpe.*
nvidia-smi
"""


def build_train_script(gpu_count, env_overrides=None, log_name="run_h100.log"):
    env = dict(TRAIN_ENV)
    if env_overrides:
        env.update(env_overrides)
    env_exports = " ".join(f'{k}={v}' for k, v in env.items())
    if gpu_count == 1:
        run_cmd = f"{env_exports} python3 train_gpt.py"
    else:
        run_cmd = f"{env_exports} torchrun --nproc_per_node={gpu_count} train_gpt.py"
    return f"""
set -ex
cd /workspace/parameter-golf

# Run training
{run_cmd} 2>&1 | tee {log_name}

echo "=== TRAINING COMPLETE ==="
echo "--- Key results ---"
grep "final_int8_zlib_roundtrip_exact\\|quantized_sliding\\|quantized_ttt\\|quantized " {log_name} || true
grep "peak memory" {log_name} || true
grep "Serialized model" {log_name} || true
grep "depth_recurrence" {log_name} || true
grep "Total submission" {log_name} || true
"""


def wait_for_ssh(pod_id, timeout=600):
    """Poll until SSH port is exposed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id)
        runtime = pod.get("runtime")
        if runtime and runtime.get("ports"):
            for p in runtime["ports"]:
                if p.get("privatePort") == 22:
                    return p["ip"], p["publicPort"]
        status = pod.get("desiredStatus", "?")
        print(f"  Waiting for pod... status={status}", end="\r")
        time.sleep(5)
    raise TimeoutError(f"Pod {pod_id} SSH not ready after {timeout}s")


def main():
    parser = argparse.ArgumentParser(description="Launch H100 training on RunPod")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true", help="1xH100 dev run (~$3-4/hr)")
    mode.add_argument("--submit", action="store_true", help="8xH100 submission run (~$24/hr)")
    parser.add_argument("--keep", action="store_true", help="Don't terminate pod after run (for debugging)")
    args = parser.parse_args()

    if args.dev:
        gpu_count = 1
        train_shards = 10  # ~1.25B tokens, fine for single-GPU 10-min run
        pod_name = "pgolf-dev-1xh100"
        rate_est = 4.0  # $/hr estimate
    else:
        gpu_count = 8
        train_shards = 20  # ~2.5B tokens
        pod_name = "pgolf-submit-8xh100"
        rate_est = 24.0

    # 1. Check SSH keys
    ssh_dir = os.path.expanduser("~/.runpod/ssh/")
    if not os.path.isdir(ssh_dir) or not any(f.endswith(".pub") for f in os.listdir(ssh_dir)):
        print("No RunPod SSH keys found. Generating...")
        from runpod.cli.groups.ssh.functions import generate_ssh_key_pair
        generate_ssh_key_pair("runpod_key")
        print(f"Generated keys in {ssh_dir}")

    # 2. Create pod
    print(f"\nCreating {gpu_count}x H100 pod ({CLOUD_TYPE} cloud, ~${rate_est:.0f}/hr)...")
    pod = runpod.create_pod(
        name=pod_name,
        image_name=IMAGE,
        gpu_type_id=GPU_TYPE,
        gpu_count=gpu_count,
        cloud_type=CLOUD_TYPE,
        support_public_ip=True,
        start_ssh=True,
        container_disk_in_gb=50,
        volume_in_gb=100,
        volume_mount_path="/runpod-volume",
        ports="22/tcp",
    )
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    print(f"Dashboard: https://www.runpod.io/console/pods/{pod_id}")

    start_time = time.time()

    try:
        # 3. Wait for SSH
        print("\nWaiting for SSH...")
        ssh_ip, ssh_port = wait_for_ssh(pod_id)
        elapsed = time.time() - start_time
        print(f"\nSSH ready at {ssh_ip}:{ssh_port} ({elapsed:.0f}s)")

        # 4. Run setup + training
        from runpod.cli.utils.ssh_cmd import SSHConnection
        with SSHConnection(pod_id) as ssh:
            print(f"\n=== Setup (clone, deps, {train_shards} train shards) ===")
            ssh.run_commands([build_setup_script(train_shards)])

            # Submission run: Random adapters r304, MLP=3, no EMA, no TTT
            print(f"\n=== Submission: Random Adapters r304 ({gpu_count}x H100, 10 min) ===")
            ssh.run_commands([build_train_script(gpu_count,
                env_overrides={
                    "USE_RANDOM_ADAPTERS": "1",
                    "ADAPTER_RANK": "304",
                    "MLP_MULT": "3",
                },
                log_name="run_submission.log")])

            # 5. Download results
            print("\n=== Downloading results ===")
            os.makedirs("h100_results", exist_ok=True)
            ssh.get_file("/workspace/parameter-golf/run_submission.log", "h100_results/run_submission.log")

        elapsed = time.time() - start_time
        print(f"\nTotal pod time: {elapsed/60:.1f} min")
        est_cost = (elapsed / 3600) * rate_est
        print(f"Estimated cost: ~${est_cost:.2f}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 6. Auto-terminate to stop billing
        if args.keep:
            print(f"\n--keep: Pod {pod_id} left running. REMEMBER TO TERMINATE MANUALLY.")
        else:
            print(f"\nTerminating pod {pod_id}...")
            runpod.terminate_pod(pod_id)
            print("Pod terminated. Billing stopped.")

    # 7. Print results summary for both runs
    for label, log_path in [("SUBMISSION", "h100_results/run_submission.log"),
                            ("STANDARD (no adapters)", "h100_results/run_standard.log"),
                            ("RANDOM ADAPTERS", "h100_results/run_adapters.log")]:
        if not os.path.exists(log_path):
            continue
        print("\n" + "=" * 60)
        print(f"RESULTS: {label}")
        print("=" * 60)
        with open(log_path) as f:
            for line in f:
                if any(k in line for k in [
                    "quantized ", "quantized_sliding", "quantized_ttt",
                    "pre_quant_post_ema",
                    "peak memory",
                    "Serialized model",
                    "depth_recurrence",
                    "Total submission",
                    "stopping_early",
                    "model_params",
                ]):
                    print(f"  {line.rstrip()}")
        print("\n  Validation BPB progression:")
        with open(log_path) as f:
            for line in f:
                if "val_bpb" in line and "step:" in line:
                    print(f"    {line.rstrip()}")


if __name__ == "__main__":
    main()
