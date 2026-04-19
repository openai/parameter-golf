#!/usr/bin/env python3
"""Windowed weight-delta over all consecutive checkpoint pairs in a run dir.

Adapted from runs/005-weight-delta/weight_delta.py. Produces:
  - per-window per-layer rel-movement (Frobenius norm of delta / norm of prev weight)
  - per-step-normalized (divide by (end_step - start_step))
  - LR-normalized (divide further by mean LR across the window)
  - loop-vs-non-loop aggregate per window

Output: one CSV per derived quantity + a combined 'delta_matrix.csv' with
  columns = [start_step, end_step, lr_mul_mid, <per-layer...>, loop_mean, nonloop_mean]

Usage:
  python windowed_weight_delta.py /path/to/run/checkpoints --outdir /path/to/analysis
"""
import argparse, csv, re, sys, time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from lr_schedule import lr_mul_at_mid_window

LOOP_LAYERS = {3, 4, 5}
N_LAYERS = 12  # SOTA arch: 12 blocks

RE_CKPT_STEP = re.compile(r"step(\d+)\.pt$")
RE_LAYER     = re.compile(r"blocks\.(\d+)\.")

def find_event_checkpoints(ckpt_dir: Path):
    """Return sorted list of (step, path) for explicit-event (ckpt_event_*) checkpoints only.
    Auto-emitted phase-boundary checkpoints (warmdown_start, pre_recurrence, final_*)
    are intentionally skipped here — they'd create uneven windows. Caller can add them back."""
    out = []
    for p in sorted(ckpt_dir.iterdir()):
        if not p.name.startswith("ckpt_event_step"): continue
        if m := RE_CKPT_STEP.search(p.name):
            out.append((int(m.group(1)), p))
    out.sort()
    return out

def layer_of(name: str) -> int:
    if m := RE_LAYER.search(name):
        return int(m.group(1))
    return -1  # global (embeddings, head, final norm, etc.)

def load_sd(path: Path):
    t0 = time.perf_counter()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    print(f"  loaded {path.name} ({len(sd)} params) in {time.perf_counter()-t0:.1f}s", flush=True)
    return sd

def window_deltas(prev_sd: dict, cur_sd: dict):
    """Return {layer: (sum_sq_delta, sum_sq_prev, numel)} aggregated over all params in layer."""
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for name, cur in cur_sd.items():
        if name not in prev_sd: continue
        prev = prev_sd[name]
        if cur.shape != prev.shape: continue
        if not cur.is_floating_point(): continue
        delta = (cur.float() - prev.float())
        layer = layer_of(name)
        agg[layer][0] += float((delta * delta).sum())
        agg[layer][1] += float((prev.float() * prev.float()).sum())
        agg[layer][2] += prev.numel()
    return agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=4550)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ckpts = find_event_checkpoints(args.ckpt_dir)
    print(f"found {len(ckpts)} event checkpoints: {[s for s, _ in ckpts]}")
    if len(ckpts) < 2:
        sys.exit("need >= 2 checkpoints for windowed analysis")

    rows = []
    prev_sd = load_sd(ckpts[0][1])
    for i in range(1, len(ckpts)):
        s_prev, _ = ckpts[i-1]
        s_cur, p_cur = ckpts[i]
        print(f"window {s_prev} -> {s_cur}")
        cur_sd = load_sd(p_cur)
        agg = window_deltas(prev_sd, cur_sd)

        n_steps = s_cur - s_prev
        lr_mid = lr_mul_at_mid_window(s_prev, s_cur, args.iterations)

        row = {"start_step": s_prev, "end_step": s_cur, "n_steps": n_steps, "lr_mul_mid": lr_mid}
        # per-layer rel-movement
        for L in range(-1, N_LAYERS):
            if L in agg:
                sq_delta, sq_prev, n = agg[L]
                rel = (sq_delta / max(sq_prev, 1e-12)) ** 0.5
                row[f"rel_layer_{L}"] = rel
                row[f"rel_per_step_layer_{L}"] = rel / n_steps
                row[f"lr_norm_layer_{L}"] = (rel / n_steps) / max(lr_mid, 1e-6)
            else:
                row[f"rel_layer_{L}"] = ""
                row[f"rel_per_step_layer_{L}"] = ""
                row[f"lr_norm_layer_{L}"] = ""

        # loop vs non-loop aggregate (weighted by numel)
        loop_sq_d, loop_sq_p = 0.0, 0.0
        nonloop_sq_d, nonloop_sq_p = 0.0, 0.0
        for L, (sq_d, sq_p, _) in agg.items():
            if L in LOOP_LAYERS:
                loop_sq_d += sq_d; loop_sq_p += sq_p
            elif 0 <= L < N_LAYERS:
                nonloop_sq_d += sq_d; nonloop_sq_p += sq_p
        loop_rel = (loop_sq_d / max(loop_sq_p, 1e-12)) ** 0.5
        nonloop_rel = (nonloop_sq_d / max(nonloop_sq_p, 1e-12)) ** 0.5
        row["loop_rel"] = loop_rel
        row["nonloop_rel"] = nonloop_rel
        row["loop_per_step"] = loop_rel / n_steps
        row["nonloop_per_step"] = nonloop_rel / n_steps
        row["loop_over_nonloop"] = loop_rel / max(nonloop_rel, 1e-12)

        rows.append(row)
        prev_sd = cur_sd  # slide window

    # Write combined CSV
    all_keys = sorted({k for r in rows for k in r}, key=lambda k: (
        0 if k in ("start_step", "end_step", "n_steps", "lr_mul_mid") else
        1 if k.startswith("rel_layer_") else
        2 if k.startswith("rel_per_step_") else
        3 if k.startswith("lr_norm_") else 4, k))
    with (args.outdir / "delta_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_keys})

    print(f"wrote delta_matrix.csv with {len(rows)} windows to {args.outdir}")

if __name__ == "__main__":
    main()
