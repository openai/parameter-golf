#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", re.IGNORECASE)


def parse_kv(line: str) -> dict[str, str]:
    return {k: v for (k, v) in KV_RE.findall(line)}


def is_runtime_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if "log0(" in s:
        return False
    return (
        "parity_matrix:" in s
        or "metric_surface:" in s
        or "export_parity:stage=" in s
    )


def coerce_num(s: str) -> float | None:
    if NUM_RE.match(s or "") is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def format_num(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.6f}"


def load_report(log_path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, str]]:
    stages: dict[str, dict[str, str]] = {}
    surfaces: dict[str, str] = {}
    summary: dict[str, str] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if not is_runtime_line(raw):
                continue
            line = raw.strip()
            if "export_parity:stage=" in line:
                kv = parse_kv(line)
                stage = kv.get("stage")
                if stage:
                    stages[stage] = kv
            elif "metric_surface:" in line:
                m = re.search(r"metric_surface:([A-Za-z0-9_]+)", line)
                if m:
                    surfaces[m.group(1)] = line
            elif "parity_matrix:summary" in line:
                summary = parse_kv(line)
    return (stages, surfaces, summary)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract parity matrix summary from a training log.")
    ap.add_argument("log_path", type=Path, help="Path to train log file")
    args = ap.parse_args()

    if not args.log_path.exists():
        raise SystemExit(f"Log not found: {args.log_path}")

    (stages, surfaces, summary) = load_report(args.log_path)

    train_bpb = coerce_num(summary.get("train_bpb_evalpath", ""))
    val_bpb = coerce_num(summary.get("val_bpb_evalpath", ""))
    train_ce_evalmode = coerce_num(summary.get("train_loss_evalmode", ""))
    pre_export_bpb = coerce_num(summary.get("pre_export_bpb", ""))
    post_export_bpb = coerce_num(summary.get("post_export_bpb", ""))

    stage_a = stages.get("A", {})
    stage_d = stages.get("D", {})
    stage_c = stages.get("C", {})

    print(f"log: {args.log_path}")
    print("matrix:")
    print(f"  train_bpb_evalpath = {format_num(train_bpb)}")
    print(f"  val_bpb_evalpath   = {format_num(val_bpb)}")
    print(f"  train_loss_evalmode= {format_num(train_ce_evalmode)}")
    print(f"  pre_export_bpb     = {format_num(pre_export_bpb)}")
    print(f"  post_export_bpb    = {format_num(post_export_bpb)}")
    print("export_stages:")
    print(f"  A ce={stage_a.get('ce', 'n/a')} bpb={stage_a.get('bpb', 'n/a')}")
    print(f"  C ce={stage_c.get('ce', 'n/a')} bpb={stage_c.get('bpb', 'n/a')}")
    print(f"  D ce={stage_d.get('ce', 'n/a')} bpb={stage_d.get('bpb', 'n/a')}")
    print("metric_surface:")
    for key in ("train_bpb_evalpath", "val_bpb_evalpath", "train_loss_evalmode", "pre_export", "post_export"):
        print(f"  {key}: {surfaces.get(key, 'n/a')}")

    if train_bpb is not None and val_bpb is not None:
        gap = val_bpb - train_bpb
        print(f"diagnostic: val_minus_train_bpb={gap:.6f}")
    if pre_export_bpb is not None and post_export_bpb is not None:
        drift = post_export_bpb - pre_export_bpb
        print(f"diagnostic: post_minus_pre_export_bpb={drift:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

