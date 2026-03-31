#!/usr/bin/env python3
"""
QK_SLOT single-GPU ablation runner.

Cases:
  baseline         — QK_GAIN=1.5 (default), SLOT=0  → post_ema_bpb + sliding_bpb (no SLOT)
  qk_gain4         — QK_GAIN=4.0, SLOT=0             → post_ema_bpb + sliding_bpb (no SLOT)
  slot_only        — QK_GAIN=1.5, SLOT=1             → post_ema_bpb + sliding_bpb+SLOT
  qk_gain4_slot    — QK_GAIN=4.0, SLOT=1             → post_ema_bpb + sliding_bpb+SLOT (cross-corr)

Cross-correlation: if slot and qk_gain4 are additive, qk_gain4_slot delta should equal
(qk_gain4 delta) + (slot_only delta). Any divergence means interaction.

QK_GAIN signal: compare post_ema_bpb between baseline and qk_gain4 (training-side only).
SLOT signal: compare sliding_bpb with/without SLOT on the same training config.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Case:
    name: str
    env: dict[str, str]
    note: str


CASES = [
    Case(
        name="baseline",
        env={},
        note="Control: QK_GAIN=1.5 (default), no SLOT",
    ),
    Case(
        name="qk_gain4",
        env={"QK_GAIN_INIT": "4.0"},
        note="QK_GAIN_INIT=4.0 only — measure training-side delta",
    ),
    Case(
        name="slot_only",
        env={"SLOT_ENABLED": "1"},
        note="SLOT enabled on default QK_GAIN — measure eval-side delta",
    ),
    Case(
        name="qk_gain4_slot",
        env={"QK_GAIN_INIT": "4.0", "SLOT_ENABLED": "1"},
        note="Both combined — cross-correlation: should ≈ sum of individual deltas",
    ),
]

BASE_ENV = {
    "ITERATIONS": "1200",
    "WARMDOWN_ITERS": "0",
    "TRAIN_BATCH_TOKENS": "786432",
    "TRAIN_SEQ_LEN": "2048",
    "MAX_WALLCLOCK_SECONDS": "0",
    "VAL_LOSS_EVERY": "1200",
    "TRAIN_LOG_EVERY": "400",
    "COMPILE_ENABLED": "1",
    "COMPILE_FULLGRAPH": "1",
    "SKIP_GPTQ": "1",
    "LOADER_MODE": "coprime",
    "COPRIME_MAX_LOADED_SHARDS": "1",
    "COPRIME_SHARDS_PER_BATCH": "1",
    "COPRIME_SHARD_HOLD_STEPS": "64",
    "COMPLEMENT_ALPHA": "0",
    "XSA_LAST_N": "11",
    "BIGRAM_VOCAB_SIZE": "2048",
    "ROPE_DIMS": "16",
    "SWA_EVERY": "50",
    "MTP_NUM_HEADS": "0",
    "TRIGRAM": "0",
    "NGRAM_EVAL_ORDER": "0",
    "CUBRIC_CADENCE": "0",
    "NGRAM_ENTROPY_SHIFT": "0",
    "SKIP_FINAL_EVAL": "0",       # run sliding window eval
    "EVAL_STRIDE": "64",
    "POST_EMA_DIAGNOSTIC": "1",
    "SLOT_STEPS": "8",
    "SLOT_LR": "0.005",
    "SLOT_MAX_WINDOWS": "512",    # ~1M tokens — fast on single GPU, sufficient signal
}


def parse_log(log_text: str) -> dict[str, str]:
    results: dict[str, str] = {}
    patterns = {
        "step_avg_ms": r"step_avg:(\d+\.\d+)ms",
        "post_ema_bpb": r"DIAGNOSTIC post_ema val_loss:\S+ val_bpb:(\S+)",
        "sliding_bpb": r"final_sliding_window(?:\+slot\S*)? val_loss:\S+ val_bpb:(\S+)",
        "sliding_bpb_exact": r"final_sliding_window(?:\+slot\S*)?_exact val_loss:\S+ val_bpb:(\S+)",
    }
    for key, pat in patterns.items():
        matches = re.findall(pat, log_text)
        if matches:
            results[key] = matches[-1]
    return results


def run_case(
    case: Case,
    train_script: Path,
    repo_root: Path,
    log_dir: Path,
    torchrun_bin: str,
    nproc: int,
    seed: int,
    dry_run: bool,
) -> dict:
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(case.env)
    env["SEED"] = str(seed)
    env["DATA_PATH"] = env.get("DATA_PATH", str(repo_root / "data" / "datasets" / "fineweb10B_sp1024"))
    env["TOKENIZER_PATH"] = env.get("TOKENIZER_PATH", str(repo_root / "data" / "tokenizers" / "fineweb_1024_bpe.model"))

    hopper = repo_root / "flash-attention" / "hopper"
    if hopper.is_dir():
        env["PYTHONPATH"] = f"{hopper}:{env.get('PYTHONPATH', '')}"

    log_file = log_dir / f"{case.name}_s{seed}.log"
    cmd = [torchrun_bin, "--standalone", f"--nproc_per_node={nproc}", str(train_script)]

    slot_info = f"QK_GAIN={case.env.get('QK_GAIN_INIT', '1.5')} SLOT={case.env.get('SLOT_ENABLED', '0')}"
    print(f"\n{'='*60}")
    print(f"CASE: {case.name}  ({slot_info})")
    print(f"note: {case.note}")
    print(f"log:  {log_file}")
    print(f"{'='*60}")

    if dry_run:
        return {"name": case.name, "note": case.note, "slot_info": slot_info,
                "post_ema_bpb": "DRY", "sliding_bpb": "DRY", "step_avg_ms": "DRY", "log": str(log_file)}

    t0 = time.perf_counter()
    with log_file.open("w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        rc = proc.wait()
    elapsed = time.perf_counter() - t0
    print(f"\n[{case.name}] finished in {elapsed:.0f}s rc={rc}")

    log_text = log_file.read_text()
    parsed = parse_log(log_text)

    return {
        "name": case.name,
        "note": case.note,
        "slot_info": slot_info,
        "rc": rc,
        "elapsed_s": f"{elapsed:.0f}",
        "post_ema_bpb": parsed.get("post_ema_bpb", "N/A"),
        "sliding_bpb": parsed.get("sliding_bpb", "N/A"),
        "sliding_bpb_exact": parsed.get("sliding_bpb_exact", "N/A"),
        "step_avg_ms": parsed.get("step_avg_ms", "N/A"),
        "log": str(log_file),
    }


def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*80}")
    print("QK_SLOT ABLATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'case':<20} {'qk/slot':<22} {'post_ema_bpb':<16} {'sliding_bpb':<16} {'step_ms':<10}")
    print("-" * 84)

    base_post = base_slide = None
    for r in results:
        if r["name"] == "baseline":
            try:
                base_post = float(r["post_ema_bpb"])
                base_slide = float(r["sliding_bpb"])
            except (ValueError, TypeError):
                pass

    for r in results:
        try:
            p = float(r["post_ema_bpb"])
            dp = f"({p - base_post:+.4f})" if base_post and r["name"] != "baseline" else ""
        except (ValueError, TypeError):
            p, dp = r["post_ema_bpb"], ""
        try:
            s = float(r["sliding_bpb"])
            ds = f"({s - base_slide:+.4f})" if base_slide and r["name"] != "baseline" else ""
        except (ValueError, TypeError):
            s, ds = r["sliding_bpb"], ""
        post_str = f"{p:.6f}{dp}" if isinstance(p, float) else str(p)
        slide_str = f"{s:.6f}{ds}" if isinstance(s, float) else str(s)
        print(f"{r['name']:<20} {r['slot_info']:<22} {post_str:<16} {slide_str:<16} {r.get('step_avg_ms','N/A'):<10}")

    # Cross-correlation check
    vals: dict[str, float] = {}
    for r in results:
        try:
            vals[r["name"]] = float(r["sliding_bpb"])
        except (ValueError, TypeError):
            pass
    if all(k in vals for k in ("baseline", "qk_gain4", "slot_only", "qk_gain4_slot")):
        qk_delta = vals["qk_gain4"] - vals["baseline"]
        slot_delta = vals["slot_only"] - vals["baseline"]
        combo_delta = vals["qk_gain4_slot"] - vals["baseline"]
        additive_prediction = qk_delta + slot_delta
        interaction = combo_delta - additive_prediction
        print(f"\nCROSS-CORRELATION (sliding_bpb):")
        print(f"  QK_GAIN delta:        {qk_delta:+.4f}")
        print(f"  SLOT delta:           {slot_delta:+.4f}")
        print(f"  Sum (predicted):      {additive_prediction:+.4f}")
        print(f"  Actual combo:         {combo_delta:+.4f}")
        print(f"  Interaction residual: {interaction:+.4f}  ({'compatible' if abs(interaction) < 0.002 else 'INTERACTION DETECTED'})")
    print(f"{'='*80}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="QK_SLOT single-GPU ablation runner")
    ap.add_argument("--seed", type=int, default=444)
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--torchrun", default="torchrun")
    ap.add_argument("--cases", nargs="+",
                    choices=[c.name for c in CASES] + ["all"],
                    default=["all"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    train_script = script_dir / "train_gpt.py"
    log_dir = script_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not train_script.is_file():
        raise SystemExit(f"ERROR: missing {train_script}")

    selected = CASES if "all" in args.cases else [c for c in CASES if c.name in args.cases]

    print(f"QK_SLOT Ablation  seed={args.seed}  nproc={args.nproc}  cases={[c.name for c in selected]}")
    print(f"SLOT_MAX_WINDOWS=512 (~1M tokens, fast single-GPU proxy)")

    results = []
    for case in selected:
        r = run_case(case, train_script, repo_root, log_dir,
                     args.torchrun, args.nproc, args.seed, args.dry_run)
        results.append(r)
        print_summary(results)

    csv_path = log_dir / f"summary_s{args.seed}_{int(time.time())}.csv"
    if results and not args.dry_run:
        fieldnames = list(results[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
