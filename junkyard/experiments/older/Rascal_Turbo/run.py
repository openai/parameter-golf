#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_seeds(raw: str) -> list[int]:
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise SystemExit("FATAL: no seeds parsed from --seeds")
    return seeds


def parse_last_float(text: str, pattern: str) -> float | None:
    m = None
    for m in re.finditer(pattern, text):
        pass
    if m is None:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_last_int(text: str, pattern: str) -> int | None:
    m = None
    for m in re.finditer(pattern, text):
        pass
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def fmt(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.8f}"


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rascal_Turbo single launcher (preflight + seed loop + CSV summary)"
    )
    p.add_argument("--seeds", default=os.environ.get("SEEDS", "42,300,444"))
    p.add_argument(
        "--nproc-per-node",
        default=os.environ.get("NPROC_PER_NODE", "auto"),
        help="'auto' or explicit integer",
    )
    p.add_argument("--torchrun-bin", default=os.environ.get("TORCHRUN_BIN", "torchrun"))

    p.add_argument(
        "--mode",
        choices=["race", "signal"],
        default=os.environ.get("MODE", "race"),
        help="race=WR wallclock profile, signal=2000-step cheap check profile",
    )
    p.add_argument("--iterations", type=int, default=int(os.environ.get("ITERATIONS", "0")))
    p.add_argument("--warmdown-iters", type=int, default=int(os.environ.get("WARMDOWN_ITERS", "-1")))
    p.add_argument(
        "--train-batch-tokens",
        type=int,
        default=int(os.environ.get("TRAIN_BATCH_TOKENS", "0")),
    )
    p.add_argument(
        "--compile-enabled",
        type=int,
        choices=[0, 1],
        default=int(os.environ.get("COMPILE_ENABLED", "-1")),
    )
    p.add_argument(
        "--skip-final-eval",
        type=int,
        choices=[0, 1],
        default=int(os.environ.get("SKIP_FINAL_EVAL", "-1")),
    )
    p.add_argument(
        "--post-ema-diagnostic",
        type=int,
        choices=[0, 1],
        default=int(os.environ.get("POST_EMA_DIAGNOSTIC", "1")),
    )

    p.add_argument(
        "--max-wallclock-seconds",
        type=int,
        default=int(os.environ.get("MAX_WALLCLOCK_SECONDS", "0")),
    )
    p.add_argument(
        "--base-wallclock-seconds",
        type=int,
        default=int(os.environ.get("BASE_WALLCLOCK_SECONDS", "600")),
    )
    p.add_argument(
        "--equiv-world-size",
        type=int,
        default=int(os.environ.get("EQUIV_WORLD_SIZE", "8")),
    )

    p.add_argument(
        "--data-path",
        default=os.environ.get("DATA_PATH", str(repo_root / "data/datasets/fineweb10B_sp1024")),
    )
    p.add_argument(
        "--tokenizer-path",
        default=os.environ.get("TOKENIZER_PATH", str(repo_root / "data/tokenizers/fineweb_1024_bpe.model")),
    )
    p.add_argument("--eval-stride", type=int, default=int(os.environ.get("EVAL_STRIDE", "64")))
    p.add_argument("--run-tag", default=f"rascal_turbo_{time.strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--dry-run", action="store_true")
    return p


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FATAL: {message}")


def pick_mode_defaults(args: argparse.Namespace) -> None:
    if args.mode == "race":
        if args.iterations == 0:
            args.iterations = 20000
        if args.warmdown_iters < 0:
            args.warmdown_iters = 3500
        if args.train_batch_tokens == 0:
            args.train_batch_tokens = 786432
        if args.compile_enabled < 0:
            args.compile_enabled = 1
        if args.skip_final_eval < 0:
            args.skip_final_eval = 0
    else:
        if args.iterations == 0:
            args.iterations = 2000
        if args.warmdown_iters < 0:
            args.warmdown_iters = 0
        if args.train_batch_tokens == 0:
            args.train_batch_tokens = 131072
        if args.compile_enabled < 0:
            args.compile_enabled = 0
        if args.skip_final_eval < 0:
            args.skip_final_eval = 1


def preflight(args: argparse.Namespace, train_script: Path) -> tuple[int, int]:
    require(train_script.is_file(), f"missing train script: {train_script}")
    require(Path(args.data_path).is_dir(), f"DATA_PATH does not exist: {args.data_path}")
    require(Path(args.tokenizer_path).is_file(), f"TOKENIZER_PATH does not exist: {args.tokenizer_path}")
    require(shutil.which(args.torchrun_bin) is not None, f"torchrun not found: {args.torchrun_bin}")

    missing = [m for m in ("sentencepiece", "zstandard", "numpy") if importlib.util.find_spec(m) is None]
    require(not missing, f"missing python modules: {', '.join(missing)}")

    import torch

    require(torch.cuda.is_available(), "CUDA is not available")
    gpu_count = torch.cuda.device_count()
    require(gpu_count >= 1, "no visible CUDA devices")

    if args.nproc_per_node == "auto":
        nproc = gpu_count
    else:
        nproc = int(args.nproc_per_node)
    require(nproc >= 1, "nproc_per_node must be >= 1")
    require(nproc <= gpu_count, f"nproc_per_node={nproc} exceeds visible_gpus={gpu_count}")

    print("============================================================")
    print("RASCAL TURBO")
    print(f"mode={args.mode}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"visible_gpus={gpu_count} nproc_per_node={nproc}")
    print(f"torchrun={args.torchrun_bin}")
    print(f"data_path={args.data_path}")
    print(f"tokenizer_path={args.tokenizer_path}")
    print("============================================================")
    return gpu_count, nproc


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    train_script = script_dir / "train_gpt.py"

    parser = build_parser(repo_root)
    args = parser.parse_args()
    pick_mode_defaults(args)
    seeds = parse_seeds(args.seeds)

    _, nproc = preflight(args, train_script)
    if args.max_wallclock_seconds > 0:
        wallclock_seconds = args.max_wallclock_seconds
    else:
        wallclock_seconds = int(
            math.ceil((args.base_wallclock_seconds * args.equiv_world_size) / max(1, nproc))
        )

    log_dir = script_dir / "logs" / args.run_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = log_dir / "summary.csv"

    print(f"seeds={seeds}")
    print(f"iterations={args.iterations} warmdown_iters={args.warmdown_iters}")
    print(f"train_batch_tokens={args.train_batch_tokens}")
    print(f"wallclock_seconds={wallclock_seconds}")
    print(f"log_dir={log_dir}")

    rows: list[dict[str, str]] = []
    for seed in seeds:
        log_file = log_dir / f"seed_{seed}.log"
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": f"{repo_root / 'flash-attention/hopper'}:{env.get('PYTHONPATH', '')}",
                "DATA_PATH": args.data_path,
                "TOKENIZER_PATH": args.tokenizer_path,
                "SEED": str(seed),
                "ITERATIONS": str(args.iterations),
                "WARMDOWN_ITERS": str(args.warmdown_iters),
                "TRAIN_BATCH_TOKENS": str(args.train_batch_tokens),
                "MAX_WALLCLOCK_SECONDS": str(wallclock_seconds),
                "COMPILE_ENABLED": str(args.compile_enabled),
                "SKIP_FINAL_EVAL": str(args.skip_final_eval),
                "POST_EMA_DIAGNOSTIC": str(args.post_ema_diagnostic),
                "EVAL_STRIDE": str(args.eval_stride),
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
                "MUON_BACKEND_STEPS": os.environ.get("MUON_BACKEND_STEPS", "4"),
                "MUON_POST_NORM": os.environ.get("MUON_POST_NORM", "row_col"),
            }
        )

        cmd = [args.torchrun_bin, "--standalone", f"--nproc_per_node={nproc}", str(train_script)]
        print("\n------------------------------------------------------------")
        print(f"RUN seed={seed}")
        print("cmd=" + " ".join(cmd))
        print(f"log={log_file}")
        print("------------------------------------------------------------")

        if args.dry_run:
            continue

        with log_file.open("w", encoding="utf-8") as fh:
            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                fh.write(line)
            exit_code = proc.wait()
        if exit_code != 0:
            raise SystemExit(f"FATAL: seed {seed} failed (exit {exit_code}). log={log_file}")

        text = log_file.read_text(encoding="utf-8", errors="replace")
        post_ema_bpb = parse_last_float(text, r"DIAGNOSTIC post_ema .*?val_bpb:([0-9.]+)")
        int6_bpb = parse_last_float(text, r"final_int6_roundtrip_exact .*?val_bpb:([0-9.]+)")
        sliding_bpb = parse_last_float(text, r"final_sliding_window_exact .*?val_bpb:([0-9.]+)")
        size_bytes = parse_last_int(text, r"Total submission size int6\+.*?:\s*([0-9]+)\s+bytes")

        rows.append(
            {
                "seed": str(seed),
                "post_ema_bpb": fmt(post_ema_bpb),
                "final_int6_bpb": fmt(int6_bpb),
                "final_sliding_bpb": fmt(sliding_bpb),
                "total_size_bytes": fmt(size_bytes),
                "logfile": str(log_file),
            }
        )
        print(
            f"RESULT seed={seed} post_ema={fmt(post_ema_bpb)} int6={fmt(int6_bpb)} "
            f"sliding={fmt(sliding_bpb)} size={fmt(size_bytes)}"
        )

    if args.dry_run:
        print("\nDRY RUN complete")
        return 0

    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "seed",
                "post_ema_bpb",
                "final_int6_bpb",
                "final_sliding_bpb",
                "total_size_bytes",
                "logfile",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n============================================================")
    print("SUMMARY")
    print("============================================================")
    for row in rows:
        print(
            f"seed={row['seed']} post_ema={row['post_ema_bpb']} int6={row['final_int6_bpb']} "
            f"sliding={row['final_sliding_bpb']} size={row['total_size_bytes']}"
        )

    def as_floats(key: str) -> list[float]:
        out = []
        for row in rows:
            try:
                out.append(float(row[key]))
            except Exception:
                pass
        return out

    post_vals = as_floats("post_ema_bpb")
    int6_vals = as_floats("final_int6_bpb")
    sliding_vals = as_floats("final_sliding_bpb")
    size_vals = as_floats("total_size_bytes")

    print("\nAverages:")
    if post_vals:
        print(f"post_ema_bpb_mean={statistics.mean(post_vals):.8f}")
    if int6_vals:
        print(f"final_int6_bpb_mean={statistics.mean(int6_vals):.8f}")
    if sliding_vals:
        print(f"final_sliding_bpb_mean={statistics.mean(sliding_vals):.8f}")
    if size_vals:
        print(f"total_size_mean_bytes={statistics.mean(size_vals):.0f}")
        print(f"total_size_max_bytes={max(size_vals):.0f}")

    print(f"\nCSV={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
