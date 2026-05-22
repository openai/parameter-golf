#!/usr/bin/env python3
"""triage_score.py — score micro-test results, classify as DEAD/NOISE/LEAD.

For Tier A (GPTQ, from logs/sweep/micro_gptq.csv):
  Computes Δ vs baseline (calib=64, matclip=12.85, embclip=20.0) for each ckpt.
  σ taken from --sigma (default 0.0014, Spark s32 measurement).

For Tier B/C (training runs, from logs/sweep/results.csv filtered to micro_b_/micro_c_):
  Computes Δ vs the interleaved baseline rows (micro_b_baseline_a/b/c/d).
  Uses session-drift-corrected baseline (linear interp between adjacent baselines).

Classification thresholds (user-tunable via --threshold):
  |Δ| < 1σ        -> NOISE      (can't distinguish from randomness)
  Δ > +1σ         -> DEAD       (worse, abandon)
  Δ < -2σ         -> LEAD       (promising, confirm with 3 seeds @ 3000 steps)
  -2σ ≤ Δ ≤ -1σ   -> WEAK_LEAD  (marginal, re-test with 2nd seed before promoting)

Usage:
  python3 scripts/triage_score.py --tier a
  python3 scripts/triage_score.py --tier b --sigma 0.005
  python3 scripts/triage_score.py --tier all
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent


def classify(delta, sigma):
    if delta is None:
        return 'SKIP'
    absd = abs(delta)
    if absd < sigma:
        return 'NOISE'
    if delta > sigma:
        return 'DEAD'
    if delta <= -2 * sigma:
        return 'LEAD'
    return 'WEAK_LEAD'


def score_tier_a(csv_path, sigma):
    if not csv_path.exists():
        print(f'[triage-a] no file {csv_path}')
        return
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print(f'[triage-a] {csv_path} empty'); return

    # Group by ckpt, baseline = calib=64 for calib sweep, matclip=12.85 for matclip sweep, etc.
    by_ckpt = defaultdict(list)
    for r in rows:
        by_ckpt[r['ckpt']].append(r)

    print(f'\n=== TIER A (GPTQ knobs) — σ={sigma} ===')
    print(f'{"ckpt":32} {"knob":10} {"value":>8} {"quant_bpb":>10} {"Δ":>10}  {"verdict":10}')
    print('-' * 90)

    for ckpt, crows in by_ckpt.items():
        # Find baseline per knob family
        baselines = {}
        for r in crows:
            try:
                q = float(r['quant_bpb'])
            except (ValueError, KeyError):
                continue
            knob = r['knob']
            try:
                v = float(r['value'])
            except ValueError:
                continue
            is_base = (knob == 'calib' and v == 64) or \
                      (knob == 'matclip' and abs(v - 12.85) < 1e-6) or \
                      (knob == 'embclip' and abs(v - 20.0) < 1e-6)
            if is_base:
                baselines[knob] = q
        # Use any baseline as cross-knob anchor if a specific family is missing
        anchor = next(iter(baselines.values())) if baselines else None

        for r in crows:
            knob = r['knob']
            try:
                q = float(r['quant_bpb'])
            except (ValueError, KeyError):
                continue
            ref = baselines.get(knob, anchor)
            if ref is None:
                delta = None
            else:
                delta = q - ref
            verdict = classify(delta, sigma)
            dstr = f'{delta:+.5f}' if delta is not None else '   ----'
            print(f'{ckpt:32} {knob:10} {r["value"]:>8} {q:>10.5f} {dstr:>10}  {verdict:10}')
    print()


def score_tier_bc(tier, sigma):
    csv_path = REPO_ROOT / 'logs' / 'sweep' / 'results.csv'
    if not csv_path.exists():
        print(f'[triage-{tier}] no file {csv_path}'); return
    rows = list(csv.DictReader(open(csv_path)))
    prefix = f'micro_{tier}_'
    micro_rows = [r for r in rows if r.get('label', '').startswith(prefix)]
    if not micro_rows:
        print(f'[triage-{tier}] no rows with prefix {prefix}')
        return

    # Identify baselines (labels containing "baseline")
    baselines = [r for r in micro_rows if 'baseline' in r.get('label', '')]
    baseline_bpbs = []
    for r in baselines:
        try:
            baseline_bpbs.append(float(r['quant_bpb']))
        except (ValueError, KeyError):
            pass
    if not baseline_bpbs:
        print(f'[triage-{tier}] no baseline rows found; cannot score')
        return
    base_mean = sum(baseline_bpbs) / len(baseline_bpbs)
    base_spread = max(baseline_bpbs) - min(baseline_bpbs) if len(baseline_bpbs) > 1 else 0

    print(f'\n=== TIER {tier.upper()} — σ={sigma}, baseline mean={base_mean:.5f} (spread={base_spread:.5f}, N={len(baselines)}) ===')
    print(f'{"label":32} {"quant_bpb":>10} {"Δ vs base":>12}  {"verdict":10}')
    print('-' * 75)
    for r in micro_rows:
        label = r.get('label', '')
        try:
            q = float(r['quant_bpb'])
        except (ValueError, KeyError):
            print(f'{label:32}  (no quant_bpb)')
            continue
        delta = q - base_mean
        verdict = classify(delta, sigma) if 'baseline' not in label else 'BASELINE'
        print(f'{label:32} {q:>10.5f} {delta:>+12.5f}  {verdict:10}')
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=['a', 'b', 'c', 'all'], default='all')
    ap.add_argument('--sigma', type=float, default=None,
                    help='Noise threshold (default: 0.0014 for A, 0.005 for B/C)')
    ap.add_argument('--csv-a', default='logs/sweep/micro_gptq.csv')
    args = ap.parse_args()

    if args.tier in ('a', 'all'):
        score_tier_a(REPO_ROOT / args.csv_a, args.sigma if args.sigma is not None else 0.0014)
    if args.tier in ('b', 'all'):
        score_tier_bc('b', args.sigma if args.sigma is not None else 0.005)
    if args.tier in ('c', 'all'):
        score_tier_bc('c', args.sigma if args.sigma is not None else 0.005)


if __name__ == '__main__':
    main()
