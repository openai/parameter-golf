#!/usr/bin/env python3
"""Plots for spec 006 dense-dynamics artifacts.

Inputs: the CSVs produced by parse_train_log.py and windowed_weight_delta.py.
Outputs: PNGs in the same analysis/ dir.

Plots produced:
  1. loss_curves.png      — train (5-step) + val (100-step), linear & log panels
  2. grad_norms.png       — per-layer grad-norm heatmap over time
  3. per_layer_movement.png — rel-movement per step per layer heatmap
  4. lr_normalized_movement.png — same, LR-normalized
  5. loop_differential.png — loop/non-loop ratio over time
  6. train_val_gap.png    — train_loss - val_loss over time

All skeletons. Will need adjustments once real log formats are verified.
"""
import argparse, csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def load_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))

def to_float(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan

def plot_loss_curves(indir: Path, outdir: Path):
    train = load_csv(indir / "train_loss.csv")
    val = load_csv(indir / "val_loss.csv")
    t_steps = [int(r["step"]) for r in train]
    t_loss  = [float(r["train_loss"]) for r in train]
    v_steps = [int(r["step"]) for r in val]
    v_loss  = [float(r["val_loss"]) for r in val]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    for ax, yscale in [(ax1, "log"), (ax2, "linear")]:
        ax.plot(t_steps, t_loss, label="train_loss (5-step)", lw=1.2, alpha=0.8, color="#1f77b4")
        ax.plot(v_steps, v_loss, "o-", label="val_loss (100-step)", ms=4, color="#d62728")
        ax.set_yscale(yscale); ax.grid(alpha=0.3); ax.legend()
        ax.set_xlabel("step"); ax.set_ylabel(f"loss ({yscale})")
    ax1.set_title("Spec 006 — train + val loss")
    if any(s >= 500 for s in t_steps):
        ax2.set_xlim(500, max(t_steps))
        zoom_losses = [l for s, l in zip(t_steps, t_loss) if s >= 500]
        if zoom_losses:
            ax2.set_ylim(min(zoom_losses) - 0.05, max(zoom_losses[:20]) + 0.05)
    plt.tight_layout()
    plt.savefig(outdir / "loss_curves.png", dpi=140, bbox_inches="tight")
    plt.close()

def plot_grad_heatmap(indir: Path, outdir: Path):
    rows = load_csv(indir / "grad_norms.csv")
    if not rows: return
    layer_cols = sorted([k for k in rows[0] if k.startswith("layer_")],
                        key=lambda k: int(k.split("_")[1]))
    steps = [int(r["step"]) for r in rows]
    mat = np.array([[to_float(r.get(lc, "")) for lc in layer_cols] for r in rows])

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(mat.T, aspect="auto", origin="lower",
                   extent=[min(steps), max(steps), 0, len(layer_cols)],
                   cmap="viridis")
    ax.set_xlabel("step"); ax.set_ylabel("layer")
    ax.set_title("Per-layer grad-norm over training")
    plt.colorbar(im, ax=ax, label="grad norm")
    plt.tight_layout(); plt.savefig(outdir / "grad_norms.png", dpi=140, bbox_inches="tight")
    plt.close()

def plot_movement_heatmap(indir: Path, outdir: Path, mode: str = "per_step"):
    """mode: 'per_step' or 'lr_norm'"""
    rows = load_csv(indir / "delta_matrix.csv")
    if not rows: return
    prefix = "rel_per_step_layer_" if mode == "per_step" else "lr_norm_layer_"
    layer_cols = sorted([k for k in rows[0] if k.startswith(prefix)],
                        key=lambda k: int(k.split("_")[-1]))
    mid_steps = [(int(r["start_step"]) + int(r["end_step"])) / 2 for r in rows]
    mat = np.array([[to_float(r.get(lc, "")) for lc in layer_cols] for r in rows])

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(mat.T, aspect="auto", origin="lower",
                   extent=[min(mid_steps), max(mid_steps), 0, len(layer_cols)],
                   cmap="magma")
    ax.set_xlabel("step (window midpoint)"); ax.set_ylabel("layer")
    ax.set_title(f"Per-layer weight movement — {mode}")
    plt.colorbar(im, ax=ax, label="rel-movement")
    name = "per_layer_movement.png" if mode == "per_step" else "lr_normalized_movement.png"
    plt.tight_layout(); plt.savefig(outdir / name, dpi=140, bbox_inches="tight")
    plt.close()

def plot_loop_differential(indir: Path, outdir: Path):
    rows = load_csv(indir / "delta_matrix.csv")
    if not rows: return
    x = [(int(r["start_step"]) + int(r["end_step"])) / 2 for r in rows]
    y = [to_float(r.get("loop_over_nonloop", "")) for r in rows]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, y, "o-", lw=1.5)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="parity")
    ax.set_xlabel("step"); ax.set_ylabel("loop / non-loop movement ratio")
    ax.set_title("Loop-layer movement differential over training")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(outdir / "loop_differential.png", dpi=140, bbox_inches="tight")
    plt.close()

def plot_train_val_gap(indir: Path, outdir: Path):
    train = {int(r["step"]): float(r["train_loss"]) for r in load_csv(indir / "train_loss.csv")}
    val_rows = load_csv(indir / "val_loss.csv")
    if not val_rows: return
    v_steps, gaps, trains_at_v, vals = [], [], [], []
    for r in val_rows:
        s = int(r["step"])
        v = float(r["val_loss"])
        # nearest train sample
        nearest = min(train.keys(), key=lambda k: abs(k - s), default=None)
        if nearest is None: continue
        t = train[nearest]
        v_steps.append(s); gaps.append(v - t); trains_at_v.append(t); vals.append(v)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(v_steps, trains_at_v, "o-", label="train (nearest)", color="#1f77b4")
    ax1.plot(v_steps, vals, "s-", label="val", color="#d62728")
    ax1.set_ylabel("loss"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(v_steps, gaps, "o-", color="#2ca02c")
    ax2.axhline(0, color="gray", ls="--", alpha=0.5)
    ax2.set_ylabel("val - train"); ax2.set_xlabel("step"); ax2.grid(alpha=0.3)
    ax1.set_title("Train-val gap over training")
    plt.tight_layout(); plt.savefig(outdir / "train_val_gap.png", dpi=140, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir", type=Path, help="dir with parsed CSVs")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    outdir = args.outdir or args.indir
    outdir.mkdir(parents=True, exist_ok=True)

    try: plot_loss_curves(args.indir, outdir)
    except Exception as e: print(f"loss_curves: {e}")
    try: plot_grad_heatmap(args.indir, outdir)
    except Exception as e: print(f"grad_heatmap: {e}")
    try: plot_movement_heatmap(args.indir, outdir, "per_step")
    except Exception as e: print(f"movement per_step: {e}")
    try: plot_movement_heatmap(args.indir, outdir, "lr_norm")
    except Exception as e: print(f"movement lr_norm: {e}")
    try: plot_loop_differential(args.indir, outdir)
    except Exception as e: print(f"loop_differential: {e}")
    try: plot_train_val_gap(args.indir, outdir)
    except Exception as e: print(f"train_val_gap: {e}")

    print(f"plots → {outdir}")

if __name__ == "__main__":
    main()
