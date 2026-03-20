#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

PAT = re.compile(r"(final_[^ ]+)_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)")


def parse_metric(path: Path):
    tag = None
    loss = None
    bpb = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = PAT.search(line)
        if m:
            tag = m.group(1)
            loss = float(m.group(2))
            bpb = float(m.group(3))
    if tag is None:
        raise ValueError(f"No final exact metric found in {path}")
    return tag, loss, bpb


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="logs/*.txt", help="log file glob")
    args = ap.parse_args()

    paths = sorted(Path().glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    rows = []
    for p in paths:
        tag, loss, bpb = parse_metric(p)
        rows.append((p, tag, loss, bpb))

    print("file\ttag\tval_loss\tval_bpb")
    for p, tag, loss, bpb in rows:
        print(f"{p}\t{tag}\t{loss:.8f}\t{bpb:.8f}")

    losses = [r[2] for r in rows]
    bpbs = [r[3] for r in rows]
    print()
    print(f"count={len(rows)}")
    print(f"mean_val_loss={mean(losses):.8f} std={std(losses):.8f}")
    print(f"mean_val_bpb={mean(bpbs):.8f} std={std(bpbs):.8f}")


if __name__ == "__main__":
    main()
