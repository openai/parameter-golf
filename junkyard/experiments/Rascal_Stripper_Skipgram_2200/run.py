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


def parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("No seeds provided")
    return out


def parse_val_bpb(log_file: Path) -> float | None:
    text = log_file.read_text(encoding="utf-8", errors="replace")
    patterns = [
        r"DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:([0-9.]+)",
        r"step:\d+/\d+ val_loss:[0-9.]+ val_bpb:([0-9.]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return float(matches[-1])
    return None


def run_one(
    *,
    repo_root: Path,
    script: Path,
    log_file: Path,
    env: dict[str, str],
    torchrun_bin: str,
    nproc_per_node: int,
) -> int:
    cmd = [
        torchrun_bin,
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(script),
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
    ap = argparse.ArgumentParser(description="RASCAL Stripped skip-gram signal A/B (2200 steps)")
    ap.add_argument("--nproc-per-node", type=int, default=1)
    ap.add_argument("--seeds", type=str, default="444")
    ap.add_argument("--iterations", type=int, default=2200)
    ap.add_argument("--train-batch-tokens", type=int, default=131072)
    ap.add_argument("--skipgram-patterns", type=str, default="1,3,5;1,2,4")
    ap.add_argument("--skipgram-mix", type=float, default=1.0)
    ap.add_argument(
        "--mode",
        type=str,
        default="ab",
        choices=["ab", "calibrate"],
        help="ab=baseline+skipgram, calibrate=baseline+skipgram_low+skipgram_high",
    )
    ap.add_argument("--low-patterns", type=str, default="1,3")
    ap.add_argument("--low-mix", type=float, default=0.5)
    ap.add_argument("--high-patterns", type=str, default="1,3,5;1,2,4;1,4,8")
    ap.add_argument("--high-mix", type=float, default=1.5)
    ap.add_argument("--torchrun-bin", type=str, default=os.environ.get("TORCHRUN_BIN", "torchrun"))
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    seeds = parse_seeds(args.seeds)

    data_path = os.environ.get("DATA_PATH", str(repo_root / "data" / "datasets" / "fineweb10B_sp1024"))
    tokenizer_path = os.environ.get("TOKENIZER_PATH", str(repo_root / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
    if not Path(data_path).is_dir():
        raise SystemExit(f"ERROR: DATA_PATH does not exist: {data_path}")
    if not Path(tokenizer_path).is_file():
        raise SystemExit(f"ERROR: TOKENIZER_PATH does not exist: {tokenizer_path}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"rascal_stripped_skipgram2200_{ts}"
    log_dir = script_dir / "logs" / run_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "summary.csv"

    print("============================================================")
    print("RASCAL STRIPPED SKIP-GRAM SIGNAL A/B")
    print(f"seeds={seeds} nproc_per_node={args.nproc_per_node} mode={args.mode}")
    print(f"iterations={args.iterations} train_batch_tokens={args.train_batch_tokens}")
    print("warmdown_iters=0 skip_final_eval=1")
    print(f"data_path={data_path}")
    print(f"tokenizer_path={tokenizer_path}")
    print(f"log_dir={log_dir}")
    print("============================================================")

    base_env = os.environ.copy()
    base_env.update(
        {
            "DATA_PATH": data_path,
            "TOKENIZER_PATH": tokenizer_path,
            "ITERATIONS": str(args.iterations),
            "WARMDOWN_ITERS": "0",
            "MAX_WALLCLOCK_SECONDS": "0",
            "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
            "TRAIN_SEQ_LEN": "2048",
            "EVAL_SEQ_LEN": "2048",
            "VAL_BATCH_SIZE": "131072",
            "VAL_LOSS_EVERY": str(args.iterations),
            "TRAIN_LOG_EVERY": "100",
            "SKIP_FINAL_EVAL": "1",
            "POST_EMA_DIAGNOSTIC": "1",
            "COMPILE_ENABLED": "0",
            "LOADER_MODE": "coprime",
            "COPRIME_MAX_LOADED_SHARDS": "1",
            "COPRIME_SHARDS_PER_BATCH": "1",
            "COPRIME_SHARD_HOLD_STEPS": "64",
            "XSA_LAST_N": "11",
            "BIGRAM_VOCAB_SIZE": "2048",
            "ROPE_DIMS": "16",
            "TRIGRAM": "0",
        }
    )

    if args.mode == "ab":
        cases = [
            ("baseline", script_dir / "train_gpt_baseline.py", {}),
            (
                "skipgram",
                script_dir / "train_gpt_skipgram.py",
                {
                    "SKIPGRAM_ENABLE": "1",
                    "SKIPGRAM_PATTERNS": args.skipgram_patterns,
                    "SKIPGRAM_MIX": f"{args.skipgram_mix}",
                },
            ),
        ]
    else:
        cases = [
            ("baseline", script_dir / "train_gpt_baseline.py", {}),
            (
                "skipgram_low",
                script_dir / "train_gpt_skipgram.py",
                {
                    "SKIPGRAM_ENABLE": "1",
                    "SKIPGRAM_PATTERNS": args.low_patterns,
                    "SKIPGRAM_MIX": f"{args.low_mix}",
                },
            ),
            (
                "skipgram_high",
                script_dir / "train_gpt_skipgram.py",
                {
                    "SKIPGRAM_ENABLE": "1",
                    "SKIPGRAM_PATTERNS": args.high_patterns,
                    "SKIPGRAM_MIX": f"{args.high_mix}",
                },
            ),
        ]

    rows: list[dict[str, str]] = []
    baseline_by_seed: dict[int, float] = {}

    for seed in seeds:
        for name, script, extra in cases:
            log_file = log_dir / f"{name}_seed{seed}.log"
            env = base_env.copy()
            env.update(extra)
            env["SEED"] = str(seed)
            env["RUN_ID"] = f"ab2200_{name}_s{seed}"

            print("")
            print("------------------------------------------------------------")
            print(f"RUN seed={seed} variant={name}")
            rc = run_one(
                repo_root=repo_root,
                script=script,
                log_file=log_file,
                env=env,
                torchrun_bin=args.torchrun_bin,
                nproc_per_node=args.nproc_per_node,
            )
            if rc != 0:
                raise SystemExit(f"Run failed: seed={seed} variant={name} exit_code={rc}")

            bpb = parse_val_bpb(log_file)
            if bpb is None:
                raise SystemExit(f"Could not parse val_bpb from {log_file}")
            if name == "baseline":
                baseline_by_seed[seed] = bpb
            base = baseline_by_seed.get(seed)
            delta = bpb - base if base is not None else 0.0
            rows.append(
                {
                    "seed": str(seed),
                    "variant": name,
                    "val_bpb": f"{bpb:.8f}",
                    "delta_vs_baseline": f"{delta:+.8f}",
                    "logfile": str(log_file),
                }
            )
            print(f"RESULT seed={seed} variant={name} val_bpb={bpb:.8f} delta_vs_baseline={delta:+.8f}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "variant", "val_bpb", "delta_vs_baseline", "logfile"])
        w.writeheader()
        w.writerows(rows)

    print("")
    print("============================================================")
    print("A/B COMPLETE")
    print(f"summary_csv={csv_path}")
    print("============================================================")
    print("seed  variant   val_bpb     delta_vs_baseline")
    for r in rows:
        print(
            f"{r['seed']:>4}  {r['variant']:<8}  {r['val_bpb']:<10}  {r['delta_vs_baseline']:<18}"
        )


if __name__ == "__main__":
    main()
