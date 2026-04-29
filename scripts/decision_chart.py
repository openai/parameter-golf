#!/usr/bin/env python3
"""Decision chart for parameter-golf campaign.

Reads logs/sweep/results.csv and emits logs/sweep/decision_chart.md
with 🟢 LOCK / 🟡 HOLD / 🔴 DROP / ⚫ OPEN flags per knob, plus a
gap tracker vs SOTA 1.0810 BPB.

Flag rules (per knob value):
  🟢 LOCK  : confirmed at >=500 steps with quant_bpb AND beats baseline by >=0.002
  🟡 HOLD  : 150-step only, beats baseline-median by >=0.002 BPB (pre_quant_bpb)
  🔴 DROP  : within +/-0.002 of baseline OR regresses
  ⚫ OPEN  : insufficient data (failed/timeout/no_bpb)

Usage:
    python scripts/decision_chart.py [--csv PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "logs" / "sweep" / "results.csv"
DEFAULT_OUT = ROOT / "logs" / "sweep" / "decision_chart.md"

SOTA_BPB = 1.0810
SUBMISSION_BPB = 1.14638

# Locked baseline identifiers: these overrides define the "baseline" config.
LOCKED_KEYS = {
    "QK_GAIN_INIT": "5.5",
    "WARMDOWN_FRAC": "0.64",
    "TTT_ENABLED": "1",
    "SLIDING_WINDOW_ENABLED": "1",
    "TTT_EPOCHS": "1",
    "EMA_DECAY": "0.995",
    "LOGIT_SOFTCAP": "20",
}

# Knobs we evaluate. Each maps knob-name -> group label for the chart.
TRACKED_KNOBS = [
    "QK_GAIN_INIT",
    "WARMDOWN_FRAC",
    "EMA_DECAY",
    "LOGIT_SOFTCAP",
    "MATRIX_LR",
    "SCALAR_LR",
    "TTT_LR",
    "TTT_CHUNK_TOKENS",
    "TTT_MOMENTUM",
    "ETLB_ENABLED",
    "NUM_LOOPS",
    "LOOP_START",
    "LOOP_END",
    "PARALLEL_RESIDUAL_START",
]

# Thresholds
DELTA_MEANINGFUL = 0.002     # BPB improvement to care about
BASELINE_MEDIAN_FALLBACK = 1.9570  # rough baseline if no clean baselines detected


def parse_overrides(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not s:
        return out
    for tok in s.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def knob_delta_from_locked(overrides: dict[str, str]) -> dict[str, str]:
    """Return overrides that differ from locked baseline."""
    return {
        k: v
        for k, v in overrides.items()
        if LOCKED_KEYS.get(k) != v and k not in ("SEED", "ITERATIONS", "TIMEOUT_SECS", "MAX_WALLCLOCK_SECONDS", "FAST_SMOKE")
    }


def is_baseline(overrides: dict[str, str]) -> bool:
    """A run is baseline if every locked key matches and there's no extra knob."""
    extras = knob_delta_from_locked(overrides)
    if extras:
        return False
    for k, v in LOCKED_KEYS.items():
        if overrides.get(k) != v:
            return False
    return True


def fmt_float(x, prec=4) -> str:
    if x is None or x == "":
        return "—"
    try:
        return f"{float(x):.{prec}f}"
    except (TypeError, ValueError):
        return "—"


def status_flag(best_pre: float | None, best_quant: float | None,
                best_iters: int, baseline_med: float) -> tuple[str, str]:
    """Return (emoji, verdict)."""
    if best_pre is None and best_quant is None:
        return "⚫", "OPEN"
    ref = best_pre if best_pre is not None else best_quant
    gain = baseline_med - ref  # positive = improvement
    if best_quant is not None and best_iters >= 500 and gain >= DELTA_MEANINGFUL:
        return "🟢", "LOCK"
    if gain >= DELTA_MEANINGFUL:
        return "🟡", "HOLD"
    if gain <= -DELTA_MEANINGFUL:
        return "🔴", "REGRESS"
    return "🔴", "DROP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = []
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["_overrides"] = parse_overrides(r.get("overrides", ""))
            try:
                r["_iters"] = int(r.get("iterations") or 0)
            except ValueError:
                r["_iters"] = 0
            def _flt(k):
                v = r.get(k)
                try:
                    return float(v) if v not in (None, "") else None
                except ValueError:
                    return None
            r["_pre"] = _flt("pre_quant_bpb")
            r["_quant"] = _flt("quant_bpb")
            r["_status"] = r.get("status", "")
            rows.append(r)

    # Baseline median (pre_quant_bpb from clean baseline runs, 150-step)
    baseline_pre = [
        r["_pre"] for r in rows
        if is_baseline(r["_overrides"]) and r["_status"] == "ok" and r["_pre"] is not None
        and r["_iters"] <= 200
    ]
    baseline_med = statistics.median(baseline_pre) if baseline_pre else BASELINE_MEDIAN_FALLBACK

    # Best-ever quant_bpb for gap tracker
    best_quant_ever = None
    best_quant_row = None
    for r in rows:
        if r["_quant"] is not None and r["_status"] == "ok":
            if best_quant_ever is None or r["_quant"] < best_quant_ever:
                best_quant_ever = r["_quant"]
                best_quant_row = r

    best_pre_ever = None
    best_pre_row = None
    for r in rows:
        if r["_pre"] is not None and r["_status"] == "ok":
            if best_pre_ever is None or r["_pre"] < best_pre_ever:
                best_pre_ever = r["_pre"]
                best_pre_row = r

    # Group: per knob -> per value -> list of rows
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        extras = knob_delta_from_locked(r["_overrides"])
        for knob in TRACKED_KNOBS:
            if knob in extras:
                groups[knob][extras[knob]].append(r)

    # Build markdown
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append(f"# Decision Chart — {now}")
    lines.append("")
    lines.append("Auto-generated by `scripts/decision_chart.py` from `logs/sweep/results.csv`.")
    lines.append("")
    lines.append("## Gap Tracker")
    lines.append("")
    lines.append(f"- **SOTA target**    : {SOTA_BPB:.4f} BPB")
    lines.append(f"- **Current submit** : {SUBMISSION_BPB:.4f} BPB  (gap vs SOTA: +{SUBMISSION_BPB - SOTA_BPB:.4f})")
    if best_quant_ever is not None:
        lines.append(f"- **Best quant_bpb** : {best_quant_ever:.4f}  ({best_quant_row['label']}, iters={best_quant_row['_iters']})  (gap vs SOTA: +{best_quant_ever - SOTA_BPB:.4f})")
    if best_pre_ever is not None:
        lines.append(f"- **Best pre_bpb**   : {best_pre_ever:.4f}  ({best_pre_row['label']}, iters={best_pre_row['_iters']})")
    lines.append(f"- **Baseline median (150-step, clean)** : {baseline_med:.4f}  (n={len(baseline_pre)})")
    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append("- 🟢 **LOCK**    — verified at ≥500 steps with quant_bpb AND beats baseline by ≥0.002")
    lines.append("- 🟡 **HOLD**    — 150-step only; beats baseline by ≥0.002 (pre_bpb); needs 500-step validation")
    lines.append("- 🔴 **DROP**    — within ±0.002 of baseline or regresses")
    lines.append("- ⚫ **OPEN**    — no data / all runs failed")
    lines.append("")
    lines.append("## Per-Knob Status")
    lines.append("")

    for knob in TRACKED_KNOBS:
        entries = groups.get(knob, {})
        if not entries:
            lines.append(f"### {knob}  ⚫ OPEN")
            lines.append("")
            lines.append("_No runs._")
            lines.append("")
            continue

        lines.append(f"### {knob}")
        lines.append("")
        lines.append("| Value | Runs | Best pre_bpb | Best quant_bpb | Max iters | Δ vs baseline | Status |")
        lines.append("|---|---|---|---|---|---|---|")

        # Sort values numerically if possible
        def _k(v):
            try:
                return (0, float(v))
            except ValueError:
                return (1, v)
        for val in sorted(entries.keys(), key=_k):
            runs = entries[val]
            ok_runs = [r for r in runs if r["_status"] == "ok"]
            best_pre = min((r["_pre"] for r in ok_runs if r["_pre"] is not None), default=None)
            best_quant = min((r["_quant"] for r in ok_runs if r["_quant"] is not None), default=None)
            max_iters = max((r["_iters"] for r in ok_runs), default=0)
            if best_pre is None and not ok_runs:
                emoji, verdict = "⚫", "OPEN"
                delta_str = "—"
            else:
                emoji, verdict = status_flag(best_pre, best_quant, max_iters, baseline_med)
                ref = best_pre if best_pre is not None else best_quant
                delta = ref - baseline_med if ref is not None else None
                delta_str = f"{delta:+.4f}" if delta is not None else "—"
            n_total = len(runs)
            n_ok = len(ok_runs)
            lines.append(
                f"| `{val}` | {n_ok}/{n_total} ok | {fmt_float(best_pre)} | "
                f"{fmt_float(best_quant)} | {max_iters} | {delta_str} | {emoji} {verdict} |"
            )
        lines.append("")

    # Recommended locks (knobs where at least one value is LOCK/HOLD)
    lines.append("## Recommended Next Actions")
    lines.append("")
    locks: list[str] = []
    holds: list[str] = []
    for knob in TRACKED_KNOBS:
        entries = groups.get(knob, {})
        for val, runs in entries.items():
            ok_runs = [r for r in runs if r["_status"] == "ok"]
            if not ok_runs:
                continue
            best_pre = min((r["_pre"] for r in ok_runs if r["_pre"] is not None), default=None)
            best_quant = min((r["_quant"] for r in ok_runs if r["_quant"] is not None), default=None)
            max_iters = max((r["_iters"] for r in ok_runs), default=0)
            emoji, verdict = status_flag(best_pre, best_quant, max_iters, baseline_med)
            ref = best_pre if best_pre is not None else best_quant
            if ref is None:
                continue
            delta = ref - baseline_med
            if verdict == "LOCK":
                locks.append(f"- 🟢 `{knob}={val}` — quant_bpb={fmt_float(best_quant)} @ {max_iters} steps (Δ {delta:+.4f})")
            elif verdict == "HOLD":
                holds.append(f"- 🟡 `{knob}={val}` — pre_bpb={fmt_float(best_pre)} (Δ {delta:+.4f}) → needs 500-step validation")

    if locks:
        lines.append("### Lock these into final config")
        lines.extend(locks)
        lines.append("")
    if holds:
        lines.append("### Validate at 500 steps (sweep_21 territory)")
        lines.extend(holds)
        lines.append("")
    if not locks and not holds:
        lines.append("_No LOCK/HOLD candidates yet — keep exploring._")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Wrote {args.out}")
    print(f"Baseline median pre_bpb: {baseline_med:.4f} (n={len(baseline_pre)})")
    if best_quant_ever is not None:
        print(f"Best quant_bpb: {best_quant_ever:.4f} ({best_quant_row['label']})")
    print(f"LOCK: {len(locks)}  HOLD: {len(holds)}")


if __name__ == "__main__":
    main()
