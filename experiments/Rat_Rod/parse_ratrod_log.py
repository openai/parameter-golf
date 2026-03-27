#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def last_float(pattern: re.Pattern[str], text: str) -> str:
    matches = pattern.findall(text)
    if not matches:
        return "-"
    if isinstance(matches[0], tuple):
        return str(matches[-1][-1])
    return str(matches[-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse Rat Rod log into one TSV row")
    p.add_argument("--log", required=True, help="Path to log file")
    p.add_argument("--sweep", required=True, help="Sweep name (e.g. warmdown, swa)")
    p.add_argument("--seed", required=True, help="Seed used for run")
    p.add_argument("--value", required=True, help="Sweep value used for run")
    p.add_argument("--ngram-order", type=int, default=9, help="Expected ngram order for final eval metric")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.log).read_text(encoding="utf-8", errors="replace")

    re_cap = re.compile(r"step:(\d+)/20000 val_loss:[0-9.]+ val_bpb:([0-9.]+)")
    re_diag = re.compile(r"DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:([0-9.]+)")
    re_slide = re.compile(r"final_sliding_window_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)")
    re_ng = re.compile(
        rf"final_sliding_window_ngram{args.ngram_order}(?:_partial)?_exact "
        r"val_loss:[0-9.]+ val_bpb:([0-9.]+)"
    )
    re_peak = re.compile(r"peak memory allocated: ([0-9]+) MiB reserved: ([0-9]+) MiB")

    cap_matches = re_cap.findall(text)
    cap_step = cap_matches[-1][0] if cap_matches else "-"
    cap_bpb = cap_matches[-1][1] if cap_matches else "-"
    diag_bpb = last_float(re_diag, text)
    slide_bpb = last_float(re_slide, text)
    ng_bpb = last_float(re_ng, text)

    peak_matches = re_peak.findall(text)
    peak_alloc = peak_matches[-1][0] if peak_matches else "-"

    row = [
        args.sweep,
        str(args.seed),
        str(args.value),
        cap_step,
        cap_bpb,
        diag_bpb,
        slide_bpb,
        ng_bpb,
        peak_alloc,
        args.log,
    ]
    print("\t".join(row))


if __name__ == "__main__":
    main()
