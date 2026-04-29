#!/usr/bin/env python3
"""Run #1851 seed reproduction on 8×H100 via HTTP-bootstrap pod.

Downloads CaseOps data from romeerp/parameter-golf-caseops-v1, then runs
train+eval for each requested seed with per-seed timeout protection.

Usage:
    python scripts/run_1851_seeds.py --seeds 314 1234 --max-minutes 75
    python scripts/run_1851_seeds.py --seeds 314 --max-minutes 40  # single seed
"""

import argparse
import os
import sys

# Add scripts/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from runpod_http_rehearsal import main as http_main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASEOPS_REPO = "romeerp/parameter-golf-caseops-v1"
CASEOPS_DATASET_DIR = "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
CASEOPS_TOKENIZER = "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
# Per-seed timeout: 40 min covers 10 min train + ~25 min TTT (3 phases) + buffer
SEED_TIMEOUT_MIN = 40


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
# Verify key files exist
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


def build_seed_cmd(seeds):
    """Build the shell command to run on-pod for given seeds."""
    download_script = build_download_caseops_script()

    data_path = f"/root/caseops_data/datasets/datasets/{CASEOPS_DATASET_DIR}"
    tok_path = f"/root/caseops_data/datasets/tokenizers/{CASEOPS_TOKENIZER}"

    parts = []
    parts.append("cd /root/rehearsal_src")
    # Install deps including brotli (used by 1851 compressor) and pyminify (code minification)
    parts.append(
        "pip install --break-system-packages -r requirements.txt brotli python-minifier 2>&1 | tail -5"
    )
    # Refresh PATH hash and verify pyminify is actually findable
    parts.append("hash -r && which pyminify")
    # Preflight: verify critical imports before spending time on data download
    parts.append(
        'python3 -c "import brotli, sentencepiece, numpy, torch; '
        'from flash_attn_interface import flash_attn_func; '
        'import subprocess; subprocess.run([\\\"pyminify\\\", \\\"--help\\\"], capture_output=True, check=True); '
        'print(\'Preflight OK\')"'
    )
    # Download CaseOps data (parallel via snapshot_download)
    parts.append(f"python3 -c {_shell_quote(download_script)}")

    for seed in seeds:
        artifact_dir = f"/root/rehearsal_out/seed{seed}"
        parts.append(f"mkdir -p {artifact_dir}")
        env = (
            f"SEED={seed} "
            f"CASEOPS_ENABLED=1 "
            f"EMBED_BITS=7 "
            f"SMEAR_GATE_ENABLED=1 "
            f"SPARSE_ATTN_GATE_ENABLED=1 "
            f"MIN_LR=0.1 "
            f"EMBED_CLIP_SIGMAS=15.0 "
            f"MLP_CLIP_SIGMAS=12.0 "
            f"GPTQ_RESERVE_SECONDS=0.5 "
            f"PHASED_TTT_NUM_PHASES=3 "
            f"DATA_PATH={data_path} "
            f"TOKENIZER_PATH={tok_path} "
            f"ARTIFACT_DIR={artifact_dir} "
            f"RUN_ID=train_seed{seed}"
        )
        # Wrap in timeout; export PATH so pyminify is findable in torchrun subprocesses
        parts.append(
            f"timeout {SEED_TIMEOUT_MIN}m bash -c 'export PATH=/usr/local/bin:/usr/bin:/root/.local/bin:$PATH && {env} torchrun --standalone --nproc_per_node=8 train_gpt.py'; "
            f"echo $? > /root/rehearsal_out/seed{seed}_exit.txt"
        )
        # Copy key artifacts to rehearsal_out root with seed prefix
        parts.append(
            f"cp {artifact_dir}/train_seed{seed}.txt /root/rehearsal_out/seed{seed}_log.txt 2>/dev/null || true"
        )
        parts.append(
            f"cp {artifact_dir}/final_model.int6.ptz /root/rehearsal_out/seed{seed}_model.ptz 2>/dev/null || true"
        )
        parts.append(f"ls -la {artifact_dir}/ 2>/dev/null || true")

    # Summary
    parts.append("echo '=== SEED RUN SUMMARY ==='")
    for seed in seeds:
        parts.append(
            f"echo 'Seed {seed} exit:' && cat /root/rehearsal_out/seed{seed}_exit.txt 2>/dev/null || echo 'unknown'"
        )
        parts.append(
            f"echo 'Seed {seed} tail:' && tail -30 /root/rehearsal_out/seed{seed}_log.txt 2>/dev/null || echo 'no log'"
        )
    parts.append("ls -la /root/rehearsal_out/")

    return " && ".join(parts)


def _shell_quote(s):
    """Quote a string for shell embedding."""
    return "'" + s.replace("'", "'\\''") + "'"


def main():
    parser = argparse.ArgumentParser(description="Run #1851 seed reproduction on 8×H100")
    parser.add_argument("--seeds", type=int, nargs="+", default=[314, 1234],
                        help="Seeds to run (default: 314 1234)")
    parser.add_argument("--max-minutes", type=int, default=100,
                        help="Pod wallclock limit in minutes (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print command but don't launch pod")
    args = parser.parse_args()

    cmd = build_seed_cmd(args.seeds)

    if args.dry_run:
        print("=== POD COMMAND ===")
        print(cmd)
        print(f"\n=== SETTINGS ===")
        print(f"Seeds: {args.seeds}")
        print(f"Max minutes: {args.max_minutes}")
        print(f"Per-seed timeout: {SEED_TIMEOUT_MIN} min")
        print(f"GPUs: 8")
        print(f"Est cost: ${8 * 2.99 * args.max_minutes / 60:.2f}")
        return

    # Build download list — seed logs/models are optional (may not exist if seed failed)
    download_files = ["status.txt", "pgolf_exit_code.txt", "pgolf_stdout.txt"]
    for seed in args.seeds:
        download_files.append(f"seed{seed}_log.txt")
        download_files.append(f"seed{seed}_model.ptz")
        download_files.append(f"seed{seed}_exit.txt")

    # Construct sys.argv for http_main
    sys.argv = [
        "runpod_http_rehearsal.py",
        "--gpus", "8",
        "--max-minutes", str(args.max_minutes),
        "--pod-name", "pgolf-1851-seeds",
        "--train-script", os.path.join(REPO_ROOT, "results", "1851_decoded.py"),
        "--cmd", cmd,
        "--download",
    ] + download_files

    http_main()


if __name__ == "__main__":
    main()
