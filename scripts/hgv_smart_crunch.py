#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


TARGET_SEEDS = [42, 314, 1337]
ARM_KEYS = [
    "MATRIX_LR",
    "SCALAR_LR",
    "LOGIT_SOFTCAP",
    "QK_GAIN_INIT",
    "EMA_DECAY",
    "TTT_ENABLED",
    "SLIDING_WINDOW_ENABLED",
]

EXPORT_DEFAULTS = {
    "SCALAR_LR": "0.030",
    "MATRIX_LR": "0.010",
}


def parse_overrides(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in (s or "").split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def overrides_signature(s: str) -> tuple[tuple[str, str], ...]:
    skip = {"RUN_ID", "NOTES", "TIMEOUT_SECS", "MAX_WALLCLOCK_SECONDS", "FAST_SMOKE"}
    parsed = parse_overrides(s)
    items = [(k, v) for k, v in parsed.items() if k not in skip and v != ""]
    items.sort()
    return tuple(items)


def to_float(v: str | None) -> float | None:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def to_int(v: str | None) -> int | None:
    try:
        if v in (None, ""):
            return None
        return int(v)
    except Exception:
        return None


def parse_ts(v: str | None) -> datetime | None:
    if not v:
        return None
    try:
        return datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def age_weight(ts: datetime | None, now: datetime) -> float:
    if ts is None:
        return 0.75
    hours = max((now - ts).total_seconds() / 3600.0, 0.0)
    # Half-life about 72h: keeps historical signal but favors recent stability.
    return 0.55 + 0.45 * math.exp(-hours / 72.0)


def status_penalty(status: str, has_quant: bool) -> float:
    s = (status or "").lower()
    if has_quant and s in ("ok", "timeout"):
        return 0.0
    if s == "ok":
        return 0.006
    if s == "timeout":
        return 0.010
    if s in ("error", "killed", "no_bpb"):
        return 0.020
    return 0.012


def canonical_arm(overrides: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple((k, overrides.get(k, "")) for k in ARM_KEYS)


def arm_str(arm: tuple[tuple[str, str], ...]) -> str:
    return " ".join(f"{k}={v}" for k, v in arm if v != "")


def rebuild_overrides(template: dict[str, str], seed: int, arm: tuple[tuple[str, str], ...]) -> str:
    skip = {
        "SEED",
        "TIMEOUT_SECS",
        "MAX_WALLCLOCK_SECONDS",
        "FAST_SMOKE",
        "RUN_ID",
        "NOTES",
    }
    out = {k: v for k, v in template.items() if k not in skip and v != ""}
    arm_map = dict(arm)
    for k, default_v in EXPORT_DEFAULTS.items():
        if out.get(k, "") == "":
            arm_v = arm_map.get(k, "")
            out[k] = arm_v if arm_v != "" else default_v

    toks = [f"{k}={v}" for k, v in out.items()]
    toks.append(f"SEED={seed}")
    return " ".join(toks)


@dataclass
class RowEval:
    effective_bpb: float
    quant_bpb: float | None
    pre_bpb: float | None
    label: str
    status: str
    seed: int | None
    iterations: int
    timestamp: datetime | None
    overrides: dict[str, str]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--recommend-out", type=Path, default=None)
    ap.add_argument("--sweep-out", type=Path, default=None)
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    rows = list(csv.DictReader(args.csv.open()))

    scored: list[RowEval] = []
    status_counts: dict[str, int] = {}
    for r in rows:
        status = (r.get("status") or "").strip() or "blank"
        status_counts[status] = status_counts.get(status, 0) + 1

        q = to_float(r.get("quant_bpb"))
        p = to_float(r.get("pre_quant_bpb"))
        metric = q if q is not None else p
        if metric is None:
            continue

        ts = parse_ts(r.get("timestamp"))
        iters = to_int(r.get("iterations")) or 0
        seed = to_int(r.get("seed"))
        ovr = parse_overrides(r.get("overrides") or "")
        iters_pen = 0.0 if iters >= 1000 else (0.005 if iters >= 500 else 0.010)
        eff = metric + status_penalty(status, q is not None) + iters_pen
        # Older datapoints get slightly downweighted, i.e., effectively penalized.
        eff += (1.0 - age_weight(ts, now)) * 0.010

        scored.append(
            RowEval(
                effective_bpb=eff,
                quant_bpb=q,
                pre_bpb=p,
                label=r.get("label") or "",
                status=status,
                seed=seed,
                iterations=iters,
                timestamp=ts,
                overrides=ovr,
            )
        )

    scored.sort(key=lambda x: x.effective_bpb)
    best_quant = sorted((s for s in scored if s.quant_bpb is not None), key=lambda x: x.quant_bpb or 9e9)

    by_arm: dict[tuple[tuple[str, str], ...], list[RowEval]] = {}
    for s in scored:
        arm = canonical_arm(s.overrides)
        if not any(v for _, v in arm):
            continue
        by_arm.setdefault(arm, []).append(s)

    arm_rank = []
    for arm, vals in by_arm.items():
        vals_sorted = sorted(vals, key=lambda x: x.effective_bpb)
        top = vals_sorted[0]
        quants = [v.quant_bpb for v in vals if v.quant_bpb is not None]
        failish = [v for v in vals if v.status in ("error", "killed", "no_bpb")]
        timeout_no_quant = [v for v in vals if v.status == "timeout" and v.quant_bpb is None]
        n = len(vals)
        fail_rate = len(failish) / n
        timeout_risk = len(timeout_no_quant) / n
        var_pen = 0.0
        if len(quants) >= 2:
            var_pen = statistics.pstdev(quants) * 0.35
        elif len(quants) == 1:
            var_pen = 0.003
        else:
            var_pen = 0.008
        arm_score = top.effective_bpb + 0.008 * fail_rate + 0.006 * timeout_risk + var_pen
        arm_rank.append((arm_score, arm, vals_sorted, quants, fail_rate, timeout_risk))

    arm_rank.sort(key=lambda x: x[0])

    print(f"rows {len(rows)}")
    print("best_quant_top5")
    for s in best_quant[:5]:
        print(
            f"{(s.quant_bpb or 9e9):.8f}\t{s.label}\tseed={s.seed}\tstatus={s.status}\titers={s.iterations}"
        )

    print("status_counts")
    for k in sorted(status_counts):
        print(f"{k} {status_counts[k]}")

    print("smart_arm_rank_top8")
    for score, arm, vals_sorted, quants, fail_rate, timeout_risk in arm_rank[:8]:
        best = vals_sorted[0]
        qbest = min(quants) if quants else None
        print(
            f"score={score:.8f}\tbest_metric={best.effective_bpb:.8f}\t"
            f"best_quant={qbest if qbest is not None else 'NA'}\t"
            f"n={len(vals_sorted)}\tfail_rate={fail_rate:.2f}\ttimeout_no_quant={timeout_risk:.2f}\t"
            f"arm={arm_str(arm)}"
        )

    if args.recommend_out is not None:
        lines = []
        lines.append("label\toverrides\treason")
        sweep_lines = []
        sweep_lines.append("# hgv smart auto-sweep")
        sweep_lines.append("# generated from full historical results with reliability and recency weighting")
        sweep_lines.append("# format: LABEL<TAB>KEY=VAL KEY=VAL ...")
        used_labels = set()
        used_override_sigs = set()
        for score, arm, vals_sorted, quants, fail_rate, timeout_risk in arm_rank[:6]:
            best = vals_sorted[0]
            seen_seeds = {v.seed for v in vals_sorted if v.seed is not None and v.quant_bpb is not None}
            candidate_seeds = [seed for seed in TARGET_SEEDS if seed not in seen_seeds]
            repeat_best = False
            if not candidate_seeds:
                fallback_seed = best.seed if best.seed is not None else TARGET_SEEDS[0]
                candidate_seeds = [fallback_seed]
                repeat_best = True

            for seed in candidate_seeds:
                arm_key = arm_str(arm)
                digest = hashlib.md5(f"{arm_key}|{seed}".encode("utf-8")).hexdigest()[:8]
                label = f"hgv_smart_{seed}_{digest}"
                if label in used_labels:
                    continue
                overrides = rebuild_overrides(best.overrides, seed, arm)
                sig = overrides_signature(overrides)
                if sig in used_override_sigs:
                    continue
                used_labels.add(label)
                used_override_sigs.add(sig)
                reason = (
                    f"arm_score={score:.6f};seen_quant_seeds={sorted(seen_seeds)};"
                    f"fail_rate={fail_rate:.2f};timeout_no_quant={timeout_risk:.2f};"
                    f"repeat_best_seed={1 if repeat_best else 0}"
                )
                lines.append(f"{label}\t{overrides}\t{reason}")
                sweep_lines.append(f"{label}\t{overrides}")
                break

        args.recommend_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if args.sweep_out is not None:
            args.sweep_out.write_text("\n".join(sweep_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
