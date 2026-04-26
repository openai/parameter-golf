"""Aggregate Parameter Golf Sprint 002 ablation results into a publishable table.

Reads `results/runs.jsonl` (the schema written by `record_run.py`) and prints:
  - Per-row n, BPB mean ± std, val_loss mean ± std, train_s/eval_s/artifact_bytes
  - Welch's t-test of each non-baseline row vs the baseline row on BPB
    (t, df, p two-sided), uncorrected
  - Effect size: Δ_bpb = row_mean - baseline_mean (positive = worse / more bits)

This is the paper-table generator. Output is reproducible from the JSONL alone
— no scipy / numpy dependency, just stdlib `math`. That makes it portable across
SturdivantAI Lab projects (GhostWrench, MoCo Lantern) where the same ablation
shape will recur.

Usage:
    python aggregate.py                                  # plain-text table to stdout
    python aggregate.py --markdown                       # GFM table for paper / hero.md
    python aggregate.py --json                           # machine-readable dump
    python aggregate.py --baseline B0 --rows B0,A1,A2    # subset
    python aggregate.py --input results/runs.jsonl       # custom path
    python aggregate.py --metric val_loss                # rank by val_loss instead of bpb

Statistical notes for the paper:
  - Welch's t (unequal variance), two-sided. Welch-Satterthwaite df.
  - p values uncorrected. With 8 ablation rows, applying Bonferroni gives
    α' = 0.05/8 = 0.00625 if you want family-wise error control across the matrix.
  - n=1 rows produce NaN std and NaN t (test undefined); reporting still emits
    the row so you can spot under-seeded ablations at a glance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO / "results" / "runs.jsonl"

# Render rows in the canonical hero.md Section 4 order.
ROW_ORDER = ["B0", "A1", "A2", "A3", "A4", "A5", "A6", "C1", "C2"]


# ---------------------------------------------------------------------------
# Statistics (stdlib-only)
# ---------------------------------------------------------------------------
def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std_sample(xs: list[float]) -> float:
    """Sample standard deviation (Bessel n-1). NaN for n<2 (single seed)."""
    if len(xs) < 2:
        return float("nan")
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def welch_t(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Welch's t-test (unequal variance). Returns (t, df, p_two_sided).

    Returns (nan, nan, nan) if either group has n<2.
    Degenerate equal-variance-zero case handled explicitly.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return (float("nan"), float("nan"), float("nan"))
    mu_a, mu_b = _mean(a), _mean(b)
    s2_a = sum((x - mu_a) ** 2 for x in a) / (n_a - 1)
    s2_b = sum((x - mu_b) ** 2 for x in b) / (n_b - 1)
    if s2_a == 0.0 and s2_b == 0.0:
        # Both groups perfectly homogeneous (will happen if seeds collapse).
        if mu_a == mu_b:
            return (0.0, float(n_a + n_b - 2), 1.0)
        return (float("inf") if mu_a > mu_b else float("-inf"),
                float(n_a + n_b - 2), 0.0)
    se = math.sqrt(s2_a / n_a + s2_b / n_b)
    t = (mu_a - mu_b) / se
    num = (s2_a / n_a + s2_b / n_b) ** 2
    den = ((s2_a / n_a) ** 2) / (n_a - 1) + ((s2_b / n_b) ** 2) / (n_b - 1)
    df = num / den if den > 0 else float(n_a + n_b - 2)
    # Two-sided p via regularized incomplete beta:
    #   p = I_x(df/2, 1/2),  x = df / (df + t^2).
    x = df / (df + t * t)
    p = _betainc_reg(df / 2.0, 0.5, x)
    return (t, df, p)


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b). NR 6.4 continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Lentz continued fraction for the incomplete beta."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


# ---------------------------------------------------------------------------
# Data loading + summarization
# ---------------------------------------------------------------------------
def load_rows(path: Path, row_filter: set[str] | None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARN: skipping malformed line {i}: {e}", file=sys.stderr)
                continue
            if row_filter and obj.get("row") not in row_filter:
                continue
            rows.append(obj)
    return rows


def group_by_row(rows: Iterable[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[r["row"]].append(r)
    return dict(g)


def summarize(rows: list[dict], metric: str = "bpb") -> dict:
    metric_vals = [r[metric] for r in rows]
    return {
        "n": len(rows),
        "metric": metric,
        "metric_mean": _mean(metric_vals),
        "metric_std": _std_sample(metric_vals),
        "metric_seeds": metric_vals,
        "bpb_mean": _mean([r["bpb"] for r in rows]),
        "bpb_std": _std_sample([r["bpb"] for r in rows]),
        "bpb_seeds": [r["bpb"] for r in rows],
        "val_loss_mean": _mean([r["val_loss"] for r in rows]),
        "val_loss_std": _std_sample([r["val_loss"] for r in rows]),
        "train_s_mean": _mean([r["train_s"] for r in rows]),
        "train_s_std": _std_sample([r["train_s"] for r in rows]),
        "eval_s_mean": _mean([r["eval_s"] for r in rows]),
        "eval_s_std": _std_sample([r["eval_s"] for r in rows]),
        "artifact_bytes_mean": _mean([r["artifact_bytes"] for r in rows]),
        "artifact_bytes_std": _std_sample([r["artifact_bytes"] for r in rows]),
        "stopped_early_any": any(r.get("stopped_early") for r in rows),
        "config_hashes": sorted({r.get("config_hash", "") for r in rows}),
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def _fmt_pm(mu: float, sd: float, prec: int = 4) -> str:
    if math.isnan(sd):
        return f"{mu:.{prec}f} (n=1)"
    return f"{mu:.{prec}f} ± {sd:.{prec}f}"


def _ordered_rows(summaries: dict[str, dict]) -> list[str]:
    """ROW_ORDER first; anything unknown appended sorted."""
    known = [r for r in ROW_ORDER if r in summaries]
    extra = sorted(r for r in summaries if r not in ROW_ORDER)
    return known + extra


def render_text(summaries: dict[str, dict], baseline: str, metric: str) -> str:
    base = summaries.get(baseline)
    base_seeds = base["metric_seeds"] if base else None

    lines: list[str] = []
    lines.append(f"Parameter Golf Sprint 002 — aggregate "
                 f"({len(summaries)} rows, baseline={baseline}, metric={metric})")
    lines.append("=" * 96)
    lines.append(f"{'row':<4} {'n':>2} {metric+' mean ± std':<24} "
                 f"{'Δ vs '+baseline:>10} {'t':>8} {'df':>6} {'p':>10} {'early':>5}")
    lines.append("-" * 96)
    for row in _ordered_rows(summaries):
        s = summaries[row]
        if row == baseline or base_seeds is None:
            delta_s = t_s = df_s = p_s = "—"
        else:
            delta = s["metric_mean"] - base["metric_mean"]
            t, df, p = welch_t(s["metric_seeds"], base_seeds)
            delta_s = f"{delta:+.4f}"
            t_s = "—" if math.isnan(t) else f"{t:+.2f}"
            df_s = "—" if math.isnan(df) else f"{df:.1f}"
            p_s = "—" if math.isnan(p) else f"{p:.4f}"
        early = "Y" if s["stopped_early_any"] else " "
        lines.append(
            f"{row:<4} {s['n']:>2} {_fmt_pm(s['metric_mean'], s['metric_std']):<24} "
            f"{delta_s:>10} {t_s:>8} {df_s:>6} {p_s:>10} {early:>5}"
        )
    lines.append("")
    lines.append("Wallclock & artifact:")
    lines.append(f"{'row':<4} {'train_s mean ± std':<24} {'eval_s mean ± std':<24} "
                 f"{'bytes mean ± std':>28}")
    lines.append("-" * 84)
    for row in _ordered_rows(summaries):
        s = summaries[row]
        ts = _fmt_pm(s["train_s_mean"], s["train_s_std"], prec=1)
        es = _fmt_pm(s["eval_s_mean"], s["eval_s_std"], prec=1)
        if math.isnan(s["artifact_bytes_std"]):
            bs = f"{s['artifact_bytes_mean']:,.0f} (n=1)"
        else:
            bs = f"{s['artifact_bytes_mean']:,.0f} ± {s['artifact_bytes_std']:,.0f}"
        lines.append(f"{row:<4} {ts:<24} {es:<24} {bs:>28}")
    return "\n".join(lines)


def render_markdown(summaries: dict[str, dict], baseline: str, metric: str) -> str:
    base = summaries.get(baseline)
    base_seeds = base["metric_seeds"] if base else None
    lines = [
        f"| row | n | {metric} mean ± std | Δ vs {baseline} | t | df | p | train_s | eval_s | bytes |",
        "|-----|---|----------------|----------|---|----|---|---------|--------|-------|",
    ]
    for row in _ordered_rows(summaries):
        s = summaries[row]
        if row == baseline or base_seeds is None:
            delta_s = t_s = df_s = p_s = "—"
        else:
            delta = s["metric_mean"] - base["metric_mean"]
            t, df, p = welch_t(s["metric_seeds"], base_seeds)
            delta_s = f"{delta:+.4f}"
            t_s = "—" if math.isnan(t) else f"{t:+.2f}"
            df_s = "—" if math.isnan(df) else f"{df:.1f}"
            p_s = "—" if math.isnan(p) else f"{p:.4f}"
        ts_s = (f"{s['train_s_mean']:.0f} ± {s['train_s_std']:.0f}"
                if not math.isnan(s["train_s_std"]) else f"{s['train_s_mean']:.0f}")
        es_s = (f"{s['eval_s_mean']:.0f} ± {s['eval_s_std']:.0f}"
                if not math.isnan(s["eval_s_std"]) else f"{s['eval_s_mean']:.0f}")
        lines.append(
            f"| {row} | {s['n']} | {_fmt_pm(s['metric_mean'], s['metric_std'])} | "
            f"{delta_s} | {t_s} | {df_s} | {p_s} | {ts_s} | {es_s} | "
            f"{s['artifact_bytes_mean']:,.0f} |"
        )
    return "\n".join(lines)


def _json_safe(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def render_json(summaries: dict[str, dict], baseline: str, metric: str) -> str:
    base = summaries.get(baseline)
    base_seeds = base["metric_seeds"] if base else None
    out: dict[str, dict] = {}
    for row in _ordered_rows(summaries):
        s = summaries[row]
        d = {k: _json_safe(v) if isinstance(v, float) else v for k, v in s.items()}
        if row == baseline or base_seeds is None:
            d["welch_t_vs_baseline"] = None
        else:
            t, df, p = welch_t(s["metric_seeds"], base_seeds)
            d["welch_t_vs_baseline"] = {
                "metric": metric,
                "baseline": baseline,
                "delta": s["metric_mean"] - base["metric_mean"],
                "t": _json_safe(t),
                "df": _json_safe(df),
                "p_two_sided": _json_safe(p),
            }
        out[row] = d
    return json.dumps({"baseline": baseline, "metric": metric, "rows": out},
                      indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help=f"jsonl path (default: {DEFAULT_INPUT.relative_to(REPO)})")
    ap.add_argument("--baseline", default="B0",
                    help="baseline row for Δ + Welch's t (default: B0)")
    ap.add_argument("--metric", default="bpb",
                    choices=["bpb", "val_loss"],
                    help="metric to rank/test on (default: bpb)")
    ap.add_argument("--rows", default=None,
                    help="comma-separated row whitelist (default: all)")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--markdown", action="store_true",
                     help="emit GFM table for paper / hero.md")
    fmt.add_argument("--json", action="store_true",
                     help="emit machine-readable JSON dump")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"FATAL: {args.input} not found. Run training first "
              "(./launch_b0.sh ... → record_run.py).", file=sys.stderr)
        return 2

    row_filter = set(args.rows.split(",")) if args.rows else None
    raw = load_rows(args.input, row_filter)
    if not raw:
        print(f"FATAL: no rows loaded from {args.input}", file=sys.stderr)
        return 2

    grouped = group_by_row(raw)
    summaries = {r: summarize(rs, metric=args.metric) for r, rs in grouped.items()}

    if args.markdown:
        print(render_markdown(summaries, args.baseline, args.metric))
    elif args.json:
        print(render_json(summaries, args.baseline, args.metric))
    else:
        print(render_text(summaries, args.baseline, args.metric))
    return 0


if __name__ == "__main__":
    sys.exit(main())
