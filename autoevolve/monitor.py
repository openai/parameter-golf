#!/usr/bin/env python3
"""
Morning dashboard for the auto-evolve run.

Usage:
    python3 autoevolve/monitor.py              # Full dashboard
    python3 autoevolve/monitor.py --tail       # Follow latest experiment log live
    python3 autoevolve/monitor.py --summary    # One-line summary (for quick SSH check)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUTOEVOLVE_DIR = ROOT / "autoevolve"
RESULTS_FILE = AUTOEVOLVE_DIR / "results.tsv"
LOGS_DIR = AUTOEVOLVE_DIR / "logs"

LEADERBOARD_SOTA = 1.1428  # Current public #1 as of 2026-03-20 (thwu1)
COST_PER_HOUR = 2.49       # 1×H100 on RunPod (update if using different SKU)


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    rows = []
    lines = RESULTS_FILE.read_text().strip().split("\n")
    if len(lines) < 2:
        return []
    headers = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(headers):
            parts += [""] * (len(headers) - len(parts))
        rows.append(dict(zip(headers, parts)))
    return rows


def parse_bpb(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def runtime_str(start: datetime, end: datetime) -> str:
    d = end - start
    h = int(d.total_seconds() // 3600)
    m = int((d.total_seconds() % 3600) // 60)
    return f"{h}h {m:02d}m"


def sparkline(values: list[float], width: int = 40) -> str:
    """ASCII sparkline for BPB trend (lower = better, trend down is good)."""
    if not values:
        return "(no data)"
    mn, mx = min(values), max(values)
    if mx == mn:
        return "─" * len(values)
    chars = "▁▂▃▄▅▆▇█"
    # Invert: high BPB = tall bar (bad), low BPB = short bar (good)
    result = []
    for v in values[-width:]:
        idx = int((v - mn) / (mx - mn) * (len(chars) - 1))
        result.append(chars[idx])
    return "".join(result)


def color(text: str, code: str) -> str:
    """ANSI color if stdout is a terminal."""
    if not sys.stdout.isatty():
        return text
    codes = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "bold": "1", "dim": "2"}
    return f"\033[{codes.get(code, '0')}m{text}\033[0m"


def print_dashboard() -> None:
    results = load_results()

    now = datetime.now()
    print()
    print(color("=" * 70, "bold"))
    print(color(f"  AUTO-EVOLVE DASHBOARD  —  {now.strftime('%Y-%m-%d %H:%M:%S')}", "bold"))
    print(color("=" * 70, "bold"))

    if not results:
        print(color("\n  No experiments recorded yet.\n", "yellow"))
        return

    # ── Runtime & cost ─────────────────────────────────────────────────────
    try:
        start_ts = datetime.fromisoformat(results[0]["timestamp"])
        end_ts = datetime.fromisoformat(results[-1]["timestamp"])
        rt = runtime_str(start_ts, end_ts)
        elapsed_h = (end_ts - start_ts).total_seconds() / 3600
        cost = elapsed_h * COST_PER_HOUR
    except Exception:
        rt = "unknown"
        cost = 0.0

    total = len(results)
    kept = [r for r in results if r.get("status") == "keep"]
    discarded = [r for r in results if r.get("status") == "discard"]
    crashed = [r for r in results if r.get("status") in ("crash", "llm_error", "invalid", "parse_error")]

    bpbs_keep = [parse_bpb(r["val_bpb"]) for r in kept]
    bpbs_keep = [b for b in bpbs_keep if b is not None]
    best_bpb = min(bpbs_keep) if bpbs_keep else None

    print(f"\n  Runtime:     {color(rt, 'cyan')}   |   Est. cost: {color(f'${cost:.2f}', 'yellow')}")
    print(f"  Experiments: {total} total — {color(str(len(kept)), 'green')} kept, "
          f"{len(discarded)} discarded, {color(str(len(crashed)), 'red')} failed")

    if best_bpb is not None:
        gap = best_bpb - LEADERBOARD_SOTA
        gap_str = f"{gap:+.4f} vs public SOTA ({LEADERBOARD_SOTA})"
        bpb_color = "green" if gap < 0 else ("yellow" if gap < 0.005 else "red")
        print(f"  Best BPB:    {color(f'{best_bpb:.4f}', bpb_color)}  ({gap_str})")
        if gap < 0:
            print(color("  ★ BEATING PUBLIC SOTA!", "green"))
    else:
        print(f"  Best BPB:    {color('No improvements yet', 'yellow')}")

    # ── Progress chart ──────────────────────────────────────────────────────
    all_bpbs = [parse_bpb(r["val_bpb"]) for r in results]
    bpb_seq = [b for b in all_bpbs if b is not None]
    if bpb_seq:
        mn_b, mx_b = min(bpb_seq), max(bpb_seq)
        print(f"\n  BPB Trend (lower=better):  min={mn_b:.4f}  max={mx_b:.4f}")
        print(f"  {sparkline(bpb_seq, width=60)}")

    # ── Recent experiments ──────────────────────────────────────────────────
    print(f"\n  {'─' * 68}")
    print(f"  {'#':>4}  {'STATUS':8}  {'BPB':7}  DESCRIPTION")
    print(f"  {'─' * 68}")

    recent = results[-15:]
    for r in recent:
        itr = r.get("iteration", "?")
        status = r.get("status", "?")
        bpb = parse_bpb(r.get("val_bpb", ""))
        desc = r.get("description", "")[:50]
        bpb_s = f"{bpb:.4f}" if bpb else "  N/A "

        if status == "keep":
            s_color, b_color = "green", "green"
            sym = "▲ KEEP "
        elif status == "discard":
            s_color, b_color = "dim", "dim"
            sym = "  disc "
        elif status in ("crash", "llm_error", "parse_error"):
            s_color, b_color = "red", "red"
            sym = "✗ FAIL "
        elif status == "invalid":
            s_color, b_color = "yellow", "yellow"
            sym = "~ BAD  "
        elif status == "dry_run":
            s_color, b_color = "cyan", "dim"
            sym = "~ DRY  "
        else:
            s_color, b_color = "dim", "dim"
            sym = "  ?    "

        print(f"  {itr:>4}  {color(sym, s_color)}  {color(bpb_s, b_color)}  {desc}")

    # ── Failure analysis ────────────────────────────────────────────────────
    if crashed:
        print(f"\n  {'─' * 68}")
        print(color(f"  FAILURES ({len(crashed)} total):", "red"))
        for r in crashed[-5:]:
            print(f"    #{r.get('iteration','?'):>3}  [{r.get('status','?'):12}]  {r.get('description','')[:55]}")

    # ── Current script ──────────────────────────────────────────────────────
    best_script = AUTOEVOLVE_DIR / "best_train_gpt.py"
    if best_script.exists():
        mtime = datetime.fromtimestamp(best_script.stat().st_mtime)
        print(f"\n  Best script: autoevolve/best_train_gpt.py (updated {mtime.strftime('%H:%M:%S')})")

    # ── Currently running ───────────────────────────────────────────────────
    logs = sorted(LOGS_DIR.glob("exp_*.log")) if LOGS_DIR.exists() else []
    if logs:
        latest_log = logs[-1]
        latest_mtime = datetime.fromtimestamp(latest_log.stat().st_mtime)
        age_s = (now - latest_mtime).total_seconds()
        if age_s < 1800:  # Active within last 30 min
            print(color(f"\n  Active experiment: {latest_log.name} (last write {int(age_s)}s ago)", "cyan"))
        else:
            print(color(f"\n  Last experiment: {latest_log.name} ({int(age_s/60)} min ago — may have stopped)", "yellow"))

    print(f"\n  {'─' * 68}")
    print(f"  Log dir: {LOGS_DIR}")
    print(f"  Tail latest log: python3 autoevolve/monitor.py --tail")
    print()


def print_summary() -> None:
    """One-line summary for quick SSH check."""
    results = load_results()
    if not results:
        print("No experiments yet.")
        return
    kept = [r for r in results if r.get("status") == "keep"]
    bpbs = [parse_bpb(r["val_bpb"]) for r in kept if parse_bpb(r.get("val_bpb"))]
    best = min(bpbs) if bpbs else None
    total = len(results)
    crashes = sum(1 for r in results if r.get("status") in ("crash", "llm_error"))
    best_s = f"{best:.4f}" if best else "N/A"
    gap_s = f"({best - LEADERBOARD_SOTA:+.4f} vs SOTA)" if best else ""
    print(f"Iterations: {total} | Best BPB: {best_s} {gap_s} | Kept: {len(kept)} | Crashes: {crashes}")


def tail_latest() -> None:
    """Stream the latest experiment log."""
    if not LOGS_DIR.exists():
        print("No logs directory found.")
        return
    logs = sorted(LOGS_DIR.glob("exp_*.log"))
    if not logs:
        print("No log files yet.")
        return
    latest = logs[-1]
    print(f"=== Tailing {latest.name} (Ctrl+C to stop) ===\n")
    with open(latest) as f:
        # Print existing content
        content = f.read()
        print(content, end="", flush=True)
        # Follow new output
        try:
            while True:
                line = f.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n(stopped)")


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-evolve monitoring dashboard")
    p.add_argument("--tail", action="store_true", help="Tail the latest experiment log")
    p.add_argument("--summary", action="store_true", help="One-line summary")
    p.add_argument("--watch", type=int, default=0, metavar="SEC",
                   help="Refresh dashboard every N seconds (e.g. --watch 30)")
    args = p.parse_args()

    if args.tail:
        tail_latest()
    elif args.summary:
        print_summary()
    elif args.watch > 0:
        try:
            while True:
                os.system("clear")
                print_dashboard()
                print(f"  (refreshing every {args.watch}s — Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        print_dashboard()


if __name__ == "__main__":
    main()
