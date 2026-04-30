#!/usr/bin/env python3
"""
Run 3 seeds of train_gpt.py on 8xH100, stream logs to stdout + per-seed log files,
parse final metrics, and update submission.json with seeds / seed_results /
val_bpb / val_bpb_std.

Usage (from repo root):
    # Full 3-seed submission run (writes submission.json):
    python3 records/track_10min_16mb/2026-03-25_DenseFormer_LeakyRelu2_VRL_GradClip/run_3seeds.py

    # Single-seed dry run (no submission.json write — for verifying schedule, etc):
    python3 records/.../run_3seeds.py --once
    python3 records/.../run_3seeds.py --once --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

SEEDS = [1337, 42, 7]

SUBMISSION_DIR = Path(__file__).resolve().parent
SCRIPT = SUBMISSION_DIR / "train_gpt.py"
LOG_DIR = SUBMISSION_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_JSON = SUBMISSION_DIR / "submission.json"

# Schedule retuned for 8xH100: at ~71ms/step the 600s cap fits ~8400 steps.
# Setting ITERATIONS=8300 lets cosine warmdown (last 1200 steps) actually run
# instead of being cut off mid-peak-LR.
ENV_OVERRIDES = {
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "GRAD_CLIP_NORM": "0.3",
    "VRL": "1",
    "XSA_LAYERS": "4",
    "VAL_LOSS_EVERY": "200",
    "ITERATIONS": "8300",
    "WARMDOWN_ITERS": "1200",
}

PAT_FINAL_EXACT = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(\d+\.\d+) val_bpb:(\d+\.\d+)"
)
PAT_FINAL = re.compile(
    r"final_int8_zlib_roundtrip val_loss:(\d+\.\d+) val_bpb:(\d+\.\d+) eval_time:(\d+)ms"
)
PAT_TRAIN_STEP = re.compile(
    r"step:(\d+)/\d+ train_loss:[\d.]+ train_time:(\d+)ms step_avg:([\d.]+)ms"
)
PAT_VAL_STEP = re.compile(
    r"step:(\d+)/\d+ val_loss:(\d+\.\d+) val_bpb:(\d+\.\d+)"
)
PAT_QUANT_BYTES = re.compile(r"Serialized model int8\+zlib: (\d+) bytes")
PAT_CODE_BYTES = re.compile(r"Code size: (\d+) bytes")
PAT_TOTAL_BYTES = re.compile(r"Total submission size int8\+zlib: (\d+) bytes")


def run_seed(seed: int) -> Path:
    log_path = LOG_DIR / f"seed{seed}.log"
    banner = f"=== Seed {seed} → {log_path} ==="
    print(f"\n{banner}\n", flush=True)

    env = os.environ.copy()
    env.update(ENV_OVERRIDES)
    env["SEED"] = str(seed)
    env["RUN_ID"] = f"denseformer_xsa_vrl_seed{seed}"

    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", str(SCRIPT)]
    t0 = time.time()
    with log_path.open("w") as f:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
        proc.wait()
    elapsed = time.time() - t0
    print(f"\n=== Seed {seed} done in {elapsed:.1f}s, exit={proc.returncode} ===\n", flush=True)
    if proc.returncode != 0:
        sys.exit(f"seed {seed} failed (exit {proc.returncode}); see {log_path}")
    return log_path


def parse_log(path: Path) -> dict:
    text = path.read_text()

    m = PAT_FINAL_EXACT.search(text)
    if not m:
        sys.exit(f"no 'final_int8_zlib_roundtrip_exact' line in {path}")
    val_loss = float(m.group(1))
    val_bpb = float(m.group(2))

    m_round = PAT_FINAL.search(text)
    eval_time_seconds = int(m_round.group(3)) / 1000.0 if m_round else None

    last_train = None
    for m_step in PAT_TRAIN_STEP.finditer(text):
        last_train = m_step
    step_stop = int(last_train.group(1)) if last_train else None
    wallclock_seconds = int(last_train.group(2)) / 1000.0 if last_train else None
    step_avg_ms = float(last_train.group(3)) if last_train else None

    pre_quant_val_loss = None
    pre_quant_val_bpb = None
    head = text.split("final_int8_zlib_roundtrip", 1)[0]
    for m_val in PAT_VAL_STEP.finditer(head):
        pre_quant_val_loss = float(m_val.group(2))
        pre_quant_val_bpb = float(m_val.group(3))

    quant_bytes = int(PAT_QUANT_BYTES.search(text).group(1)) if PAT_QUANT_BYTES.search(text) else None
    code_bytes = int(PAT_CODE_BYTES.search(text).group(1)) if PAT_CODE_BYTES.search(text) else None
    total_bytes = int(PAT_TOTAL_BYTES.search(text).group(1)) if PAT_TOTAL_BYTES.search(text) else None

    return {
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "step_stop": step_stop,
        "wallclock_seconds": wallclock_seconds,
        "eval_time_seconds": eval_time_seconds,
        "step_avg_ms": step_avg_ms,
        "pre_quant_val_loss": pre_quant_val_loss,
        "pre_quant_val_bpb": pre_quant_val_bpb,
        "bytes_model_int8_zlib": quant_bytes,
        "bytes_code": code_bytes,
        "bytes_total": total_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single seed for verification. Does NOT touch submission.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to use for --once. Defaults to SEEDS[0] (1337).",
    )
    args = parser.parse_args()

    if not SCRIPT.exists():
        sys.exit(f"train script not found: {SCRIPT}")

    if args.once:
        seed = args.seed if args.seed is not None else SEEDS[0]
        log = run_seed(seed)
        parsed = parse_log(log)
        print(f"\n--- Seed {seed} (verification run) ---")
        print(json.dumps(parsed, indent=2))
        print(
            "\nschedule sanity-check: step_stop should be close to ITERATIONS "
            f"({ENV_OVERRIDES.get('ITERATIONS', '<default>')}).\n"
            "If it's far below, training was wallclock-capped before warmdown finished — "
            "lower ITERATIONS further. If it's well above the cap window, raise it.\n"
            "submission.json was NOT modified."
        )
        return

    results: dict[str, dict] = {}
    for seed in SEEDS:
        log = run_seed(seed)
        results[str(seed)] = parse_log(log)
        print(
            f"\n--- Seed {seed} parsed: val_bpb={results[str(seed)]['val_bpb']:.4f} "
            f"step_stop={results[str(seed)]['step_stop']} ---\n",
            flush=True,
        )

    bpbs = [r["val_bpb"] for r in results.values()]
    losses = [r["val_loss"] for r in results.values()]
    pre_bpbs = [r["pre_quant_val_bpb"] for r in results.values() if r["pre_quant_val_bpb"] is not None]
    pre_losses = [r["pre_quant_val_loss"] for r in results.values() if r["pre_quant_val_loss"] is not None]

    data = json.loads(SUBMISSION_JSON.read_text())
    data["seeds"] = SEEDS
    data["seed_results"] = results
    data["val_bpb"] = round(statistics.mean(bpbs), 8)
    data["val_loss"] = round(statistics.mean(losses), 8)
    data["val_bpb_std"] = round(statistics.stdev(bpbs), 8) if len(bpbs) > 1 else 0.0
    data["val_loss_std"] = round(statistics.stdev(losses), 8) if len(losses) > 1 else 0.0
    if pre_bpbs:
        data["pre_quant_val_bpb"] = round(statistics.mean(pre_bpbs), 8)
    if pre_losses:
        data["pre_quant_val_loss"] = round(statistics.mean(pre_losses), 8)

    r0 = results[str(SEEDS[0])]
    data["step_stop"] = r0["step_stop"]
    data["wallclock_seconds"] = r0["wallclock_seconds"]
    data["eval_time_seconds"] = r0["eval_time_seconds"]
    data["bytes_total"] = r0["bytes_total"]
    data["bytes_model_int8_zlib"] = r0["bytes_model_int8_zlib"]
    data["bytes_code"] = r0["bytes_code"]
    data["status"] = "verified_8xH100"

    SUBMISSION_JSON.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\n=== submission.json updated → {SUBMISSION_JSON} ===\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
