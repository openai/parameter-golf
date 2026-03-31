#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cobra_harness as ch

COBRA_DIR = Path(__file__).resolve().parent
HARNESS = COBRA_DIR / "cobra_harness.py"
RE_LOGFILE = re.compile(r"^log_file\s*:\s*(.+)$", re.MULTILINE)


@dataclass
class RunRow:
    letter: str
    candidate: str
    seed: int
    log_path: Path
    base_bpb: float | None
    diag_bpb: float | None
    step: int | None
    train_ms: int | None
    peak_mib: int | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run economical A/B(/B/A) Cobra sequences and report deltas")
    p.add_argument("--a", default="c0_green1_anchor", help="Candidate name for arm A")
    p.add_argument("--b", required=True, help="Candidate name for arm B")
    p.add_argument(
        "--sequence",
        default="ABBA",
        help="Sequence pattern using letters A/B (default: ABBA)",
    )
    p.add_argument(
        "--seeds",
        default="1337,2045",
        help="Comma-separated seeds (default: 1337,2045)",
    )
    p.add_argument("--max-wallclock", type=float, default=180.0, help="Wallclock seconds per run")
    p.add_argument("--nproc", type=int, default=1, help="nproc_per_node (default: 1 for cheap proxy)")
    p.add_argument("--execute", action="store_true", help="Actually launch runs")
    return p.parse_args()


def parse_log_path(stdout_text: str) -> Path:
    m = RE_LOGFILE.search(stdout_text)
    if not m:
        raise RuntimeError("Could not find log_file path in harness output")
    return Path(m.group(1).strip())


def run_harness(candidate: str, seed: int, nproc: int, max_wallclock: float, execute: bool) -> Path:
    cmd = [
        sys.executable,
        str(HARNESS),
        "run",
        "--candidate",
        candidate,
        "--seed",
        str(seed),
        "--nproc",
        str(nproc),
        "--max-wallclock",
        str(max_wallclock),
    ]
    if execute:
        cmd.append("--execute")

    proc = subprocess.run(cmd, cwd=str(ch.ROOT), capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"Harness run failed for candidate={candidate} seed={seed} rc={proc.returncode}")
    return parse_log_path(proc.stdout)


def summarize(rows: List[RunRow]) -> None:
    print("\nA/B summary (lower base_bpb is better):")
    print("arm\truns\tmean_base_bpb\tmean_diag_bpb\tmean_step\tmean_train_ms")

    by_arm: Dict[str, List[RunRow]] = {"A": [], "B": []}
    for r in rows:
        by_arm[r.letter].append(r)

    arm_means: Dict[str, float] = {}
    for arm in ("A", "B"):
        bucket = by_arm[arm]
        base_vals = [r.base_bpb for r in bucket if r.base_bpb is not None]
        diag_vals = [r.diag_bpb for r in bucket if r.diag_bpb is not None]
        step_vals = [float(r.step) for r in bucket if r.step is not None]
        ms_vals = [float(r.train_ms) for r in bucket if r.train_ms is not None]

        mean_base = statistics.fmean(base_vals) if base_vals else float("nan")
        mean_diag = statistics.fmean(diag_vals) if diag_vals else float("nan")
        mean_step = statistics.fmean(step_vals) if step_vals else float("nan")
        mean_ms = statistics.fmean(ms_vals) if ms_vals else float("nan")
        arm_means[arm] = mean_base

        print(
            f"{arm}\t{len(bucket)}\t{mean_base:.8f}\t{mean_diag:.8f}\t{mean_step:.2f}\t{mean_ms:.0f}"
        )

    delta = arm_means["B"] - arm_means["A"]
    print(f"\nDelta (B - A) base_bpb: {delta:+.8f}")
    if delta < 0:
        print("Decision: B is better on the 1-GPU proxy.")
    else:
        print("Decision: A remains better on the 1-GPU proxy.")


def main() -> int:
    args = parse_args()
    seq = args.sequence.strip().upper()
    if not seq:
        raise ValueError("--sequence cannot be empty")
    if any(ch_ not in {"A", "B"} for ch_ in seq):
        raise ValueError("--sequence must contain only A and B")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")

    candidate_for = {"A": args.a, "B": args.b}
    rows: List[RunRow] = []

    print("Cobra economical A/B sequence")
    print(f"sequence       : {seq}")
    print(f"seeds          : {seeds}")
    print(f"arm A          : {args.a}")
    print(f"arm B          : {args.b}")
    print(f"nproc          : {args.nproc}")
    print(f"max_wallclock  : {args.max_wallclock}")
    print(f"execute        : {int(args.execute)}")
    print("")

    for seed in seeds:
        print(f"=== seed {seed} ===")
        for idx, letter in enumerate(seq, start=1):
            cand = candidate_for[letter]
            print(f"[{idx}/{len(seq)}] {letter} -> {cand}")
            log_path = run_harness(cand, seed, args.nproc, args.max_wallclock, args.execute)

            if not args.execute:
                continue
            if not log_path.exists():
                raise FileNotFoundError(log_path)

            parsed = ch.parse_log(log_path)
            rows.append(
                RunRow(
                    letter=letter,
                    candidate=cand,
                    seed=seed,
                    log_path=log_path,
                    base_bpb=parsed.get("base_bpb"),
                    diag_bpb=parsed.get("diag_bpb"),
                    step=parsed.get("step"),
                    train_ms=parsed.get("train_ms"),
                    peak_mib=parsed.get("peak_mib"),
                )
            )

    if not args.execute:
        print("\nDry-run only. Add --execute to launch runs and compute deltas.")
        return 0

    if not rows:
        print("No rows parsed; nothing to summarize.")
        return 1

    summarize(rows)
    print("\nPer-run rows:")
    print("seed\tarm\tcandidate\tbase_bpb\tdiag_bpb\tstep\ttrain_ms\tpeak_mib\tlog")
    for r in rows:
        print(
            "\t".join(
                [
                    str(r.seed),
                    r.letter,
                    r.candidate,
                    "-" if r.base_bpb is None else f"{r.base_bpb:.8f}",
                    "-" if r.diag_bpb is None else f"{r.diag_bpb:.8f}",
                    "-" if r.step is None else str(r.step),
                    "-" if r.train_ms is None else str(r.train_ms),
                    "-" if r.peak_mib is None else str(r.peak_mib),
                    str(r.log_path),
                ]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
