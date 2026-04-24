"""Plot training curves from train.log files.

Parses logs from both submissions, renders three panels:
- Left: val_bpb trajectory for the March submission
- Center: train_loss for the three seeds of the April submission
- Right: final val_bpb per seed (sliding and roundtrip)

Usage:
    python3 scripts/plot_curves.py

Output: assets/loss_curves.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

MARCH_LOG = ROOT / "records" / "track_10min_16mb" / "2026-03-21_MixedQuant_BigramHash_SWA" / "train.log"
APRIL_DIR = ROOT / "records" / "track_10min_16mb" / "2026-04-01_TurboMuon_EngramLite_Improved"

RE_TRAIN = re.compile(r"step:(\d+)/\d+ train_loss:([\d.]+)")
RE_VAL = re.compile(r"step:(\d+)/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)")
RE_SWA_START = re.compile(r"swa_started step:(\d+)|swa:start step:(\d+)")
RE_QAT_START = re.compile(r"late_qat:enabled step:(\d+)")
RE_FINAL_SLIDING = re.compile(r"final_int6_sliding_window_exact val_loss:[\d.]+ val_bpb:([\d.]+)")
RE_FINAL_ROUNDTRIP = re.compile(r"final_int6_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    train_steps, train_loss = [], []
    val_steps, val_bpb = [], []
    swa_start, qat_start = None, None
    final_sliding, final_roundtrip = None, None

    for line in text.splitlines():
        m = RE_TRAIN.match(line)
        if m:
            train_steps.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            continue
        m = RE_VAL.match(line)
        if m:
            val_steps.append(int(m.group(1)))
            val_bpb.append(float(m.group(2)))
            continue
        m = RE_SWA_START.search(line)
        if m:
            swa_start = int(m.group(1) or m.group(2))
            continue
        m = RE_QAT_START.search(line)
        if m:
            qat_start = int(m.group(1))
            continue
        m = RE_FINAL_SLIDING.search(line)
        if m:
            final_sliding = float(m.group(1))
            continue
        m = RE_FINAL_ROUNDTRIP.search(line)
        if m:
            final_roundtrip = float(m.group(1))

    return dict(
        train_steps=train_steps,
        train_loss=train_loss,
        val_steps=val_steps,
        val_bpb=val_bpb,
        swa_start=swa_start,
        qat_start=qat_start,
        final_sliding=final_sliding,
        final_roundtrip=final_roundtrip,
    )


def plot_march(ax, data: dict):
    vs = [s for s in data["val_steps"] if s >= 500]
    vb = [b for s, b in zip(data["val_steps"], data["val_bpb"]) if s >= 500]

    ax.plot(vs, vb, color="#c0392b", marker="o", ms=5, lw=1.8, label="val_bpb (pre-roundtrip)")

    ax.axhline(1.2244, color="#7f8c8d", ls=":", lw=1.2, label="baseline 1.2244")
    ax.axhline(1.2421, color="#2c3e50", ls="-", lw=1.2, alpha=0.7, label="final post-roundtrip 1.2421")

    if data["swa_start"]:
        ax.axvline(data["swa_start"], color="#2980b9", ls="--", lw=1.3, alpha=0.8)
        ax.annotate(f"SWA start\nstep {data['swa_start']}",
                    xy=(data["swa_start"], 1.40), fontsize=9, color="#2980b9",
                    ha="left", xytext=(data["swa_start"] + 100, 1.40))

    ax.set_title("2026-03-21. Mixed Quantization + BigramHash + SWA", fontsize=11, loc="left")
    ax.set_xlabel("training step")
    ax.set_ylabel("val_bpb")
    ax.set_ylim(1.15, 1.45)
    ax.set_xlim(0, 11500)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def plot_april_train(ax, seeds: dict):
    colors = {1337: "#c0392b", 42: "#27ae60", 2024: "#2980b9"}
    for seed, data in seeds.items():
        ax.plot(data["train_steps"], data["train_loss"],
                color=colors[seed], marker="o", ms=4, lw=1.3, alpha=0.8,
                label=f"seed {seed}")

    any_qat = next((d["qat_start"] for d in seeds.values() if d["qat_start"]), None)
    any_swa = next((d["swa_start"] for d in seeds.values() if d["swa_start"]), None)

    if any_swa:
        ax.axvline(any_swa, color="#16a085", ls="--", lw=1.2, alpha=0.7)
        ax.annotate(f"SWA start ~{any_swa}", xy=(any_swa, 2.4), fontsize=9,
                    color="#16a085", xytext=(any_swa - 2200, 2.4))
    if any_qat:
        ax.axvline(any_qat, color="#8e44ad", ls="--", lw=1.2, alpha=0.7)
        ax.annotate(f"Late QAT ~{any_qat}", xy=(any_qat, 2.2), fontsize=9,
                    color="#8e44ad", xytext=(any_qat + 100, 2.2))

    ax.set_title("2026-04-01. Turbo-Muon + EngramLite (3 seeds)", fontsize=11, loc="left")
    ax.set_xlabel("training step")
    ax.set_ylabel("train_loss")
    ax.set_ylim(1.80, 2.50)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)


def plot_april_finals(ax, seeds: dict):
    labels = [f"seed {s}" for s in seeds]
    sliding = [seeds[s]["final_sliding"] for s in seeds]
    roundtrip = [seeds[s]["final_roundtrip"] for s in seeds]

    x = list(range(len(labels)))
    w = 0.38
    x_left = [i - w / 2 for i in x]
    x_right = [i + w / 2 for i in x]

    bars_sl = ax.bar(x_left, sliding, width=w, color="#27ae60", label="sliding (final metric)")
    bars_rt = ax.bar(x_right, roundtrip, width=w, color="#e67e22", label="roundtrip")

    for rect, v in zip(bars_sl, sliding):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=8.5)
    for rect, v in zip(bars_rt, roundtrip):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=8.5)

    mean_sl = sum(sliding) / len(sliding)
    ax.axhline(mean_sl, color="#2c3e50", ls=":", lw=1.0, alpha=0.7)
    ax.text(len(labels) - 0.5, mean_sl + 0.005, f"mean {mean_sl:.4f}",
            fontsize=9, color="#2c3e50", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("val_bpb")
    ax.set_ylim(1.10, 1.20)
    ax.set_title("April finals: 3-seed verification", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def main() -> int:
    if not MARCH_LOG.exists():
        print(f"missing {MARCH_LOG}", file=sys.stderr)
        return 1

    march = parse_log(MARCH_LOG)
    april = {s: parse_log(APRIL_DIR / f"train_seed{s}.log") for s in (1337, 42, 2024)}
    for s, d in april.items():
        if not d["train_steps"]:
            print(f"empty log for seed {s}", file=sys.stderr)
            return 1

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.7]))
    plot_march(axes[0], march)
    plot_april_train(axes[1], april)
    plot_april_finals(axes[2], april)

    fig.suptitle("Parameter Golf. Training dynamics of both submissions.", fontsize=13, y=1.03)
    plt.tight_layout()

    out = ROOT / "assets" / "loss_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
