#!/usr/bin/env python3
"""Produce per-record unified diffs + stats against NaiveBaseline."""
import pathlib, difflib, subprocess

ROOT = pathlib.Path("/Users/william/Desktop/parameter-golf/sota_analysis")
SRC  = ROOT / "records_normalized"
OUT  = ROOT / "diffs"
OUT.mkdir(exist_ok=True)

baseline = (SRC / "2026-03-17_NaiveBaseline.py").read_text().splitlines(keepends=True)

summary = []
for f in sorted(SRC.glob("*.py")):
    if "NaiveBaseline" in f.name:
        continue
    record = f.read_text().splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        baseline, record,
        fromfile="baseline.py",
        tofile=f.name,
        n=3,
    ))
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    (OUT / (f.stem + ".diff")).write_text("".join(diff))
    summary.append((f.stem, added, removed, len(record)))

summary.sort(key=lambda t: t[0])
print(f"{'Record':65s} {'+':>6s} {'-':>6s} {'lines':>6s}")
for name, a, r, n in summary:
    print(f"{name:65s} {a:>6d} {r:>6d} {n:>6d}")
