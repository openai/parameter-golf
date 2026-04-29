#!/usr/bin/env python3
"""Generate the 5 article charts from the matcha JSONLs and parameter-golf .txt logs.

Usage:
    python plot.py [--out DIR]

Reads from ../data/matcha/lb_*.jsonl and ../data/pg_logs/lb_*.txt (relative to
this script). Writes 5 PNGs to --out (default: ../figures/).
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DEFAULT_OUT = HERE.parent / "figures"

CONFIGS = [
    # (run_id, label, color)
    ("lb_baseline",      "1. Baseline",        "tab:gray"),
    ("lb_slide64",       "2. SlidingWindow",   "tab:blue"),
    ("lb_lora_ttt",      "3. LoRA TTT",        "tab:green"),
    ("lb_11l_ema_gptq",  "4. 11L EMA+GPTQ",    "tab:orange"),
    ("lb_par_resid_dr",  "5. ParResid+MiniDR", "tab:red"),
]

# Match the LAST score-bearing line in each config's .txt log
SCORE_PATTERNS = [
    re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"),
    re.compile(r"final_int8_zlib_roundtrip\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"),
    re.compile(r"final_int8_ttt_lora\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"),
    re.compile(r"final_int6_sliding_window_exact\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"),
    re.compile(r"quantized_ttt_phased\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)"),
]


def parse_val_bpb(txt_path: Path):
    """Return the LAST val_bpb score line in the log — that's the leaderboard score."""
    if not txt_path.exists():
        return None
    last = None
    for line in txt_path.read_text().splitlines():
        for pat in SCORE_PATTERNS:
            m = pat.search(line)
            if m:
                last = float(m.group(2))
    return last


def load_runs():
    runs = {}
    for run_id, label, color in CONFIGS:
        jsonl = DATA_DIR / "matcha" / f"{run_id}.jsonl"
        txt = DATA_DIR / "pg_logs" / f"{run_id}.txt"

        records = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
        se = next(r for r in records if r.get("type") == "session_end")
        steps = [r for r in records if r.get("type") == "step"]
        bpb = parse_val_bpb(txt)
        runs[run_id] = {"label": label, "color": color, "bpb": bpb, "se": se, "steps": steps}
        print(f"{label:<22} val_bpb={bpb}  kWh={se['energy_wh']/1000:.3f}  dur={se['duration_s']:.0f}s")
    return runs


def chart_01_hero(runs, out_dir: Path):
    """val_bpb vs total kWh, with manual label offsets to dodge the #2/#3 overlap."""
    fig, ax = plt.subplots(figsize=(9, 6))
    label_offsets = {
        "lb_baseline":     ( 10,   8),
        "lb_slide64":      ( 10, -16),
        "lb_lora_ttt":     ( 10,  10),
        "lb_11l_ema_gptq": ( 10,   8),
        "lb_par_resid_dr": (-10, -18),
    }
    for run_id, r in runs.items():
        kwh = r["se"]["energy_wh"] / 1000
        ax.scatter(kwh, r["bpb"], s=200, color=r["color"], zorder=3,
                   edgecolors="black", linewidths=1)
        dx, dy = label_offsets[run_id]
        ha = "right" if dx < 0 else "left"
        ax.annotate(r["label"], (kwh, r["bpb"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=10, ha=ha)
    ax.set_xlabel("Total energy per run (kWh)")
    ax.set_ylabel("val_bpb (post-quant, leaderboard score)")
    ax.set_title("Same wallclock cap (600s on 8×H100 SXM5), 18% loss spread")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "01_hero_loss_vs_energy.png", dpi=150)
    plt.close()


def chart_02_loss_trajectory(runs, out_dir: Path):
    """Loss vs cumulative energy curves per config."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for run_id, r in runs.items():
        cum_wh, ys = 0.0, []
        for rec in r["steps"]:
            cum_wh += rec["energy_j"] / 3600
            m = rec.get("train_metrics", {})
            loss = m.get("train_loss")
            if loss is not None and rec["step"] > 5:
                ys.append((cum_wh, loss))
        if ys:
            xs, ls = zip(*ys)
            ax.plot(xs, ls, label=r["label"], color=r["color"], linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Cumulative energy (Wh)")
    ax.set_ylabel("train_loss (in-run)")
    ax.set_title("Loss trajectory per config — same wallclock budget, different shapes")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "02_loss_vs_cumulative_energy.png", dpi=150)
    plt.close()


def chart_03_efficiency(runs, out_dir: Path):
    """Wh per 0.001 BPB-drop vs baseline."""
    base = runs["lb_baseline"]
    labels, ratios, colors = [], [], []
    for run_id, r in runs.items():
        if run_id == "lb_baseline":
            continue
        db = r["bpb"] - base["bpb"]
        dw = r["se"]["energy_wh"] - base["se"]["energy_wh"]
        if abs(db) > 1e-6:
            labels.append(r["label"])
            ratios.append(-dw / db)
            colors.append(r["color"])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, ratios, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Wh per 0.001 BPB drop vs baseline")
    ax.set_title("Energy efficiency: how much each config spent (or saved) per BPB-improvement unit")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, ratios):
        label = f"{val:+.0f} Wh" + (" (free win!)" if val < 0 else "")
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3 if val > 0 else -12), textcoords="offset points",
                    ha="center", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "03_energy_efficiency.png", dpi=150)
    plt.close()


def chart_04_train_vs_post(runs, out_dir: Path):
    """Stacked bar: training-phase Wh + post-training Wh, with % annotation."""
    labels, train_wh, post_wh = [], [], []
    for run_id, r in runs.items():
        se = r["se"]
        train_step_energy = sum(rec["energy_j"] for rec in r["steps"]) / 3600
        post = se["energy_wh"] - train_step_energy
        labels.append(r["label"])
        train_wh.append(train_step_energy)
        post_wh.append(post)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    ax.bar(x, train_wh, label="Training phase", color="steelblue")
    ax.bar(x, post_wh, bottom=train_wh,
           label="Post-training (GPTQ + eval + serialize)", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Energy (Wh)")
    ax.set_title("Training vs post-training energy — leaderboards measure only the blue bar")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, (t, p) in enumerate(zip(train_wh, post_wh)):
        pct = p / (t + p) * 100
        ax.annotate(f"{pct:.0f}% post-train",
                    (i, t + p), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "04_train_vs_post_train_energy.png", dpi=150)
    plt.close()


def chart_05_per_gpu_deviation(runs, out_dir: Path):
    """Per-GPU deviation from per-config median — surfaces the GPU 2 straggle."""
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.15
    x = np.arange(8)
    for i, (run_id, r) in enumerate(runs.items()):
        energies = [g["energy_j"] for g in sorted(r["se"]["gpus"], key=lambda g: g["idx"])]
        median = sorted(energies)[len(energies) // 2]
        deviations_pct = [(e - median) / median * 100 for e in energies]
        ax.bar(x + i * width, deviations_pct, width, label=r["label"], color=r["color"])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"GPU {i}" for i in range(8)])
    ax.set_ylabel("Deviation from per-config median (%)")
    ax.set_title("Per-GPU energy deviation — GPU 2 is the laggard in 5/5 configs")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "05_per_gpu_deviation.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    print()
    chart_01_hero(runs, args.out)
    chart_02_loss_trajectory(runs, args.out)
    chart_03_efficiency(runs, args.out)
    chart_04_train_vs_post(runs, args.out)
    chart_05_per_gpu_deviation(runs, args.out)

    pngs = sorted(args.out.glob("*.png"))
    print(f"\nWrote {len(pngs)} PNGs to {args.out}/")
    for p in pngs:
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
