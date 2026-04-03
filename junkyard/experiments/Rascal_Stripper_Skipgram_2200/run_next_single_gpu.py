#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path


def count_train_shards(data_path: Path) -> int:
    return len(list(data_path.glob("fineweb_train_*.bin")))


def parse_last_float(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return float(matches[-1])


def parse_metrics(log_file: Path) -> tuple[float | None, float | None]:
    text = log_file.read_text(encoding="utf-8", errors="replace")
    step_avg = parse_last_float(text, r"step:\d+/\d+ train_loss:[0-9.]+ train_time:\d+ms step_avg:([0-9.]+)ms")
    post_ema = parse_last_float(text, r"DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:([0-9.]+)")
    return step_avg, post_ema


def run_case(
    *,
    repo_root: Path,
    train_script: Path,
    log_file: Path,
    env: dict[str, str],
    torchrun_bin: str,
    nproc_per_node: int,
) -> int:
    cmd = [
        torchrun_bin,
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(train_script),
    ]
    print(f"cmd={' '.join(cmd)}")
    print(f"log={log_file}")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description="Next calibrated single-GPU RASCAL ablation pack")
    ap.add_argument("--nproc-per-node", type=int, default=1)
    ap.add_argument("--seed", type=int, default=444)
    ap.add_argument("--iterations", type=int, default=1200)
    ap.add_argument("--train-batch-tokens", type=int, default=786432)
    ap.add_argument("--min-train-shards", type=int, default=4)
    ap.add_argument("--torchrun-bin", type=str, default=os.environ.get("TORCHRUN_BIN", "torchrun"))
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    train_script = script_dir / "train_gpt_baseline.py"

    data_path = Path(os.environ.get("DATA_PATH", str(repo_root / "data" / "datasets" / "fineweb10B_sp1024")))
    tokenizer_path = Path(os.environ.get("TOKENIZER_PATH", str(repo_root / "data" / "tokenizers" / "fineweb_1024_bpe.model")))

    if not data_path.is_dir():
        raise SystemExit(f"ERROR: DATA_PATH does not exist: {data_path}")
    if not tokenizer_path.is_file():
        raise SystemExit(f"ERROR: TOKENIZER_PATH does not exist: {tokenizer_path}")

    shards = count_train_shards(data_path)
    if shards < args.min_train_shards:
        raise SystemExit(
            f"ERROR: need at least {args.min_train_shards} train shards for real loader-cache signal, found {shards}.\n"
            "Run: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4"
        )

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"rascal_next_single_gpu_{ts}"
    log_dir = script_dir / "logs" / run_tag
    csv_path = log_dir / "summary.csv"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("RASCAL NEXT SINGLE-GPU PACK")
    print(f"seed={args.seed} nproc={args.nproc_per_node}")
    print(f"iterations={args.iterations} train_batch_tokens={args.train_batch_tokens}")
    print(f"data_path={data_path} train_shards={shards}")
    print(f"tokenizer_path={tokenizer_path}")
    print(f"log_dir={log_dir}")
    print("cases=baseline,muon_ns4,loader_cache4,combo_ns4_cache4")
    print("============================================================")

    base_env = os.environ.copy()
    base_env.update(
        {
            "DATA_PATH": str(data_path),
            "TOKENIZER_PATH": str(tokenizer_path),
            "SEED": str(args.seed),
            "ITERATIONS": str(args.iterations),
            "WARMDOWN_ITERS": "0",
            "MAX_WALLCLOCK_SECONDS": "0",
            "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
            "TRAIN_SEQ_LEN": "2048",
            "EVAL_SEQ_LEN": "2048",
            "VAL_BATCH_SIZE": "131072",
            "VAL_LOSS_EVERY": str(args.iterations),
            "TRAIN_LOG_EVERY": "500",
            "SKIP_FINAL_EVAL": "1",
            "POST_EMA_DIAGNOSTIC": "1",
            "COMPILE_ENABLED": "1",
            "COMPILE_FULLGRAPH": "1",
            "LOADER_MODE": "coprime",
            "COPRIME_SHARDS_PER_BATCH": "1",
            "COPRIME_SHARD_HOLD_STEPS": "64",
            "XSA_LAST_N": "11",
            "BIGRAM_VOCAB_SIZE": "2048",
            "ROPE_DIMS": "16",
            "TRIGRAM": "0",
        }
    )

    cases = [
        ("baseline", {}),
        ("muon_ns4", {"MUON_BACKEND_STEPS": "4"}),
        ("loader_cache4", {"COPRIME_MAX_LOADED_SHARDS": "4"}),
        ("combo_ns4_cache4", {"MUON_BACKEND_STEPS": "4", "COPRIME_MAX_LOADED_SHARDS": "4"}),
    ]

    rows: list[dict[str, str]] = []
    baseline_step = None
    baseline_bpb = None

    for name, extra in cases:
        print("")
        print("------------------------------------------------------------")
        print(f"CASE: {name}")
        env = base_env.copy()
        env.update(extra)
        env["RUN_ID"] = f"next_{name}_s{args.seed}"
        env.setdefault("COPRIME_MAX_LOADED_SHARDS", "1")

        log_file = log_dir / f"{name}.log"
        rc = run_case(
            repo_root=repo_root,
            train_script=train_script,
            log_file=log_file,
            env=env,
            torchrun_bin=args.torchrun_bin,
            nproc_per_node=args.nproc_per_node,
        )
        if rc != 0:
            raise SystemExit(f"Run failed: {name} exit_code={rc}")

        step_avg, post_ema_bpb = parse_metrics(log_file)
        if step_avg is None or post_ema_bpb is None:
            raise SystemExit(f"Failed to parse metrics for {name}: {log_file}")

        if name == "baseline":
            baseline_step = step_avg
            baseline_bpb = post_ema_bpb

        d_step = step_avg - baseline_step if baseline_step is not None else 0.0
        d_bpb = post_ema_bpb - baseline_bpb if baseline_bpb is not None else 0.0
        rows.append(
            {
                "case": name,
                "step_avg_ms": f"{step_avg:.8f}",
                "post_ema_bpb": f"{post_ema_bpb:.8f}",
                "delta_step_vs_baseline_ms": f"{d_step:+.8f}",
                "delta_bpb_vs_baseline": f"{d_bpb:+.8f}",
                "logfile": str(log_file),
            }
        )
        print(
            f"RESULT case={name} step_avg_ms={step_avg:.8f} post_ema_bpb={post_ema_bpb:.8f} "
            f"delta_step={d_step:+.8f} delta_bpb={d_bpb:+.8f}"
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "step_avg_ms",
                "post_ema_bpb",
                "delta_step_vs_baseline_ms",
                "delta_bpb_vs_baseline",
                "logfile",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print("")
    print("============================================================")
    print("NEXT PACK COMPLETE")
    print(f"summary_csv={csv_path}")
    print("============================================================")
    for r in rows:
        print(
            f"{r['case']:<17} step_avg={r['step_avg_ms']} post_ema_bpb={r['post_ema_bpb']} "
            f"d_step={r['delta_step_vs_baseline_ms']} d_bpb={r['delta_bpb_vs_baseline']}"
        )


if __name__ == "__main__":
    main()

