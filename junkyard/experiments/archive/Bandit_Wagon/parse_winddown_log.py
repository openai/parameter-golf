#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _last_group(pattern: re.Pattern[str], text: str, group: int = 1) -> str:
    matches = list(pattern.finditer(text))
    if not matches:
        return "-"
    return matches[-1].group(group)


def _last_cap(text: str) -> tuple[str, str]:
    cap_re = re.compile(r"step:(\d+)/(\d+)\s+val_loss:[0-9.eE+-]+\s+val_bpb:([0-9.eE+-]+)")
    matches = list(cap_re.finditer(text))
    if not matches:
        return "-", "-"
    m = matches[-1]
    return m.group(1), m.group(3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse Bandit Wagon winddown log into one TSV row")
    p.add_argument("--log", required=True, help="Path to log file")
    p.add_argument("--arm", required=True, help="Arm name")
    p.add_argument("--seed", required=True, help="Seed")
    p.add_argument("--meta", default="", help="Opaque metadata string (env summary)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.log).read_text(encoding="utf-8", errors="replace")

    diag_re = re.compile(r"DIAGNOSTIC post_ema val_loss:[0-9.eE+-]+ val_bpb:([0-9.eE+-]+)")
    roundtrip_re = re.compile(r"final_int6_roundtrip_exact val_loss:[0-9.eE+-]+ val_bpb:([0-9.eE+-]+)")
    sliding_re = re.compile(r"final_int6_sliding_window_exact val_loss:[0-9.eE+-]+ val_bpb:([0-9.eE+-]+)")
    peak_re = re.compile(r"peak memory allocated: (\d+) MiB reserved: (\d+) MiB")

    cap_step, cap_bpb = _last_cap(text)
    diag_bpb = _last_group(diag_re, text)
    roundtrip_bpb = _last_group(roundtrip_re, text)
    sliding_bpb = _last_group(sliding_re, text)
    peak_alloc = _last_group(peak_re, text, group=1)

    row = [
        args.arm,
        str(args.seed),
        cap_step,
        cap_bpb,
        diag_bpb,
        roundtrip_bpb,
        sliding_bpb,
        peak_alloc,
        args.meta,
        args.log,
    ]
    print("\t".join(row))


if __name__ == "__main__":
    main()
