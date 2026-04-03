#!/usr/bin/env python3
"""
Arch+Schedule sweep — 9 cases vs Rascal II baseline.
All cases: MAX_WALLCLOCK_SECONDS=600, NPROC=4, seed=444.
QAT and SWA both fire on 4xGPU within 600s (~2650s and ~2800s respectively).

Cases (one variable vs baseline each):
  baseline     — control (exact sota_now.sh env)
  rope_32      — ROPE_DIMS 16→32
  bigram_3072  — BIGRAM_VOCAB_SIZE 2048→3072  (competition target)
  bigram_4096  — BIGRAM_VOCAB_SIZE 2048→4096  (watch size gate)
  qat_early    — LATE_QAT_THRESHOLD 0.15→0.25 (QAT fires earlier, ~2420 steps)
  qat_late     — LATE_QAT_THRESHOLD 0.15→0.05 (QAT fires later, ~3120 steps)
  swa_dense    — SWA_EVERY 50→10 (more snapshots)
  gptq         — SKIP_GPTQ=0 (full Hessian GPTQ, training-data calib, 30s reserve)
  warmdown_4k  — WARMDOWN_ITERS 3500→4000

Key metrics:
  post_ema_bpb      — float32 model quality (POST_EMA_DIAGNOSTIC=1)
  sliding_bpb       — final sliding window score (the race metric)
  int6_bpb          — after int6+zstd quantization
  quant_gap         — int6_bpb - post_ema_bpb (lower = QAT working)
  size_bytes        — serialized int6+zstd size (must stay ≤ 16MB)
  qat_step          — which step QAT fired
  swa_start_step    — which step SWA started
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
    post_only: bool = False  # if True: skip training, load checkpoint from a prior case


CASES = [
    Case(
        name="baseline",
        env={},
        note="Control — exact SOTA env (Rascal II)",
    ),
    Case(
        name="rope_32",
        env={"ROPE_DIMS": "32"},
        note="ROPE_DIMS 16→32 — more positional coverage",
    ),
    Case(
        name="bigram_3072",
        env={"BIGRAM_VOCAB_SIZE": "3072"},
        note="BIGRAM_VOCAB_SIZE 2048→3072 — competition target (PR #1019 uses 3072)",
    ),
    Case(
        name="bigram_4096",
        env={"BIGRAM_VOCAB_SIZE": "4096"},
        note="BIGRAM_VOCAB_SIZE 2048→4096 — WATCH SIZE GATE",
    ),
    Case(
        name="qat_early",
        env={"LATE_QAT_THRESHOLD": "0.25"},
        note="LATE_QAT_THRESHOLD 0.15→0.25 — QAT fires earlier (~step 2420)",
    ),
    Case(
        name="qat_late",
        env={"LATE_QAT_THRESHOLD": "0.05"},
        note="LATE_QAT_THRESHOLD 0.15→0.05 — QAT fires later (~step 3120)",
    ),
    Case(
        name="swa_dense",
        env={"SWA_EVERY": "10"},
        note="SWA_EVERY 50→10 — more weight snapshots",
    ),
    Case(
        name="gptq",
        env={"SKIP_GPTQ": "0"},
        note="SKIP_GPTQ 1→0 — full Hessian GPTQ on baseline checkpoint (no retraining)",
        post_only=True,
    ),
    Case(
        name="gptq_full",
        env={"SKIP_GPTQ": "0"},
        note="SKIP_GPTQ 1→0 — full training + GPTQ (30s reserve, ~170 fewer steps on 4xGPU)",
    ),
    Case(
        name="warmdown_4k",
        env={"WARMDOWN_ITERS": "4000"},
        note="WARMDOWN_ITERS 3500→4000 — longer warmdown, matches competition leaders",
    ),
]

# Exact env from sota_now.sh — baseline for all cases
BASE_ENV = {
    "MAX_WALLCLOCK_SECONDS": "600",
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
    "LATE_QAT_THRESHOLD": "0.15",  # explicit — this is what we're varying
    "POST_EMA_DIAGNOSTIC": "1",    # need post_ema_bpb for quant gap analysis
    "EVAL_STRIDE": "64",
    "SKIP_FINAL_EVAL": "0",
}


def parse_log(log_text: str) -> dict[str, str]:
    results: dict[str, str] = {}
    patterns = {
        "step_avg_ms":   r"step:500/\S+ \S+ \S+ step_avg:(\S+)ms",
        "post_ema_bpb":  r"DIAGNOSTIC post_ema val_loss:\S+ val_bpb:(\S+)",
        "sliding_bpb":   r"final_sliding_window_exact val_loss:\S+ val_bpb:(\S+)",
        "int6_bpb":      r"final_int6_roundtrip_exact val_loss:\S+ val_bpb:(\S+)",
        "size_bytes":    r"Total submission size int6\+zstd: (\d+) bytes",
        "qat_step":      r"late_qat:enabled step:(\d+)",
        "swa_start":     r"swa:start step:(\d+)",
        "total_steps":   r"stopping_early: wallclock_cap \S+ step:(\d+)/",
    }
    for key, pat in patterns.items():
        m = re.search(pat, log_text)
        if m:
            results[key] = m.group(1)
    # fallback: last step line
    if "total_steps" not in results:
        m = re.search(r"step:(\d+)/\d+ val_loss", log_text)
        if m:
            results["total_steps"] = m.group(1)
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
    checkpoint_path: Path | None = None,
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

    if case.post_only:
        if checkpoint_path is None or not checkpoint_path.is_file():
            print(f"[{case.name}] SKIP — no checkpoint available (baseline must run first)")
            return {"name": case.name, "note": case.note, "rc": "SKIP",
                    "post_ema_bpb": "SKIP", "sliding_bpb": "SKIP", "int6_bpb": "SKIP",
                    "size_bytes": "SKIP", "quant_gap": "SKIP", "qat_step": "SKIP",
                    "swa_start": "SKIP", "total_steps": "0", "step_avg_ms": "SKIP",
                    "log": "SKIP"}
        env["SKIP_TRAIN"] = "1"
        env["LOAD_CHECKPOINT"] = str(checkpoint_path)

    log_file = log_dir / f"{case.name}_s{seed}.log"
    cmd = [torchrun_bin, "--standalone", f"--nproc_per_node={nproc}", str(train_script)]
    changed = {k: v for k, v in case.env.items()}
    mode_tag = " [POST-TRAIN: reuses baseline checkpoint]" if case.post_only else ""
    print(f"\n{'='*70}")
    print(f"CASE: {case.name}{mode_tag}")
    print(f"note: {case.note}")
    print(f"diff: {changed if changed else '(none — baseline)'}")
    print(f"log:  {log_file}")
    print(f"{'='*70}")

    if dry_run:
        return {"name": case.name, "note": case.note,
                "post_ema_bpb": "DRY", "sliding_bpb": "DRY",
                "int6_bpb": "DRY", "size_bytes": "DRY",
                "qat_step": "DRY", "swa_start": "DRY", "step_avg_ms": "DRY",
                "log": str(log_file)}

    t0 = time.perf_counter()
    with log_file.open("w") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(repo_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
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
        "rc": rc,
        "elapsed_s": f"{elapsed:.0f}",
        "post_ema_bpb": parsed.get("post_ema_bpb", "N/A"),
        "sliding_bpb": parsed.get("sliding_bpb", "N/A"),
        "int6_bpb": parsed.get("int6_bpb", "N/A"),
        "size_bytes": parsed.get("size_bytes", "N/A"),
        "quant_gap": _gap(parsed.get("post_ema_bpb"), parsed.get("int6_bpb")),
        "qat_step": parsed.get("qat_step", "N/A"),
        "swa_start": parsed.get("swa_start", "N/A"),
        "total_steps": parsed.get("total_steps", "N/A"),
        "step_avg_ms": parsed.get("step_avg_ms", "N/A"),
        "log": str(log_file),
    }


def _gap(a: str | None, b: str | None) -> str:
    try:
        return f"{float(b) - float(a):+.4f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "N/A"


def _delta(val: str, base: float | None, name: str) -> str:
    try:
        v = float(val)
        d = f" ({v - base:+.4f})" if base is not None and name != "baseline" else ""
        return f"{v:.6f}{d}"
    except (TypeError, ValueError):
        return str(val)


def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*100}")
    print("ARCH+SCHED SWEEP SUMMARY")
    print(f"{'='*100}")
    base_slide = base_post = base_int6 = None
    for r in results:
        if r["name"] == "baseline":
            try: base_post = float(r["post_ema_bpb"])
            except (TypeError, ValueError): pass
            try: base_slide = float(r["sliding_bpb"])
            except (TypeError, ValueError): pass
            try: base_int6 = float(r["int6_bpb"])
            except (TypeError, ValueError): pass

    header = f"{'case':<15} {'post_ema':<22} {'sliding':<22} {'int6':<22} {'quant_gap':<12} {'size_MB':<9} {'qat_step':<10} {'steps':<6}"
    print(header)
    print("-" * len(header))
    for r in results:
        size_mb = "N/A"
        try:
            size_mb = f"{int(r['size_bytes']) / 1e6:.2f}MB"
        except (TypeError, ValueError):
            pass
        print(
            f"{r['name']:<15} "
            f"{_delta(r['post_ema_bpb'], base_post, r['name']):<22} "
            f"{_delta(r['sliding_bpb'], base_slide, r['name']):<22} "
            f"{_delta(r['int6_bpb'], base_int6, r['name']):<22} "
            f"{r.get('quant_gap','N/A'):<12} "
            f"{size_mb:<9} "
            f"{r.get('qat_step','N/A'):<10} "
            f"{r.get('total_steps','N/A'):<6}"
        )
    print(f"{'='*100}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=444)
    ap.add_argument("--nproc", type=int, default=4)
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
    print(f"Arch+Sched Sweep  seed={args.seed}  nproc={args.nproc}  cases={[c.name for c in selected]}")
    print(f"MAX_WALLCLOCK_SECONDS=600 — QAT fires ~step 2800, SWA ~step 2650 on 4xH100")

    # Checkpoint saved after each full training run; post_only cases reuse it.
    # We use the baseline checkpoint so post_only cases test quant on the best model.
    saved_checkpoint: Path | None = None
    final_model_src = repo_root / "final_model.pt"

    results = []
    for case in selected:
        r = run_case(case, train_script, repo_root, log_dir,
                     args.torchrun, args.nproc, args.seed, args.dry_run,
                     checkpoint_path=saved_checkpoint)
        results.append(r)
        # After a full training run, snapshot the checkpoint for post_only cases.
        # Use the first successful training run (baseline if present, else first case).
        if not case.post_only and saved_checkpoint is None and not args.dry_run:
            if final_model_src.is_file() and r.get("rc") == 0:
                saved_checkpoint = log_dir / f"checkpoint_{case.name}_s{args.seed}.pt"
                import shutil
                shutil.copy2(str(final_model_src), str(saved_checkpoint))
                print(f"[checkpoint] saved {case.name} model → {saved_checkpoint}")
        print_summary(results)

    csv_path = log_dir / f"summary_s{args.seed}_{int(time.time())}.csv"
    if results and not args.dry_run:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
