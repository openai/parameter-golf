import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = "/home/claude-user/ai-workspace/projects/parameter-golf/runs"

RUNS = {
    "008 (baseline, 8×H)": {
        "log": f"{BASE}/008-1736-reproduction/seed_42/train.log",
        "color": "#888888",
        "ls": "--",
    },
    "021e (diagonal α frozen, 8×H)": {
        "log": f"{BASE}/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh/seed_42/train.log",
        "color": "#2196F3",
        "ls": "-",
    },
    "025b (cross-layer frozen, 4×H)": {
        "log": f"{BASE}/025b-cross-layer-carry-frozen/seed_42/train.log",
        "color": "#FF5722",
        "ls": "-",
    },
    "026 (cross-layer frozen, 8×H)": {
        "log": f"{BASE}/026-cross-layer-carry-frozen-8xh/seed_42/train.log",
        "color": "#4CAF50",
        "ls": "-",
    },
}

def parse_log(path):
    steps, losses, val_steps, val_bpbs, loop_step = [], [], [], [], None
    with open(path) as f:
        for line in f:
            m = re.match(r"(\d+)/20000 train_loss: ([\d.]+)", line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
            m = re.match(r"(\d+)/20000 val_loss: [\d.]+ val_bpb: ([\d.]+)", line)
            if m:
                val_steps.append(int(m.group(1)))
                val_bpbs.append(float(m.group(2)))
            m = re.search(r"layer_loop:enabled step:(\d+)", line)
            if m:
                loop_step = int(m.group(1))
    return steps, losses, val_steps, val_bpbs, loop_step

TTT_POINTS = {
    "021e (diagonal α frozen, 8×H)":      (5200, 1.06622),
    "025b (cross-layer frozen, 4×H)":     None,
    "026 (cross-layer frozen, 8×H)":      (5200, 1.06582),
    "008 (baseline, 8×H)":               (5200, 1.06728),  # 009 spinquant baseline
}
SOTA_TTT = 1.06610  # #1736 measured (reference_1736_seed42)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
fig.suptitle("Training loss comparison: 008 vs 021e vs 025b vs 026", fontsize=13, fontweight="bold")

# --- top: full training loss ---
ax1.set_title("Train loss (smoothed 5-step rolling avg)", fontsize=11)
ax1.set_xlabel("Step")
ax1.set_ylabel("Train loss")
ax1.set_xlim(0, 5100)
ax1.set_ylim(2.25, 3.0)

for label, cfg in RUNS.items():
    steps, losses, _, _, loop_step = parse_log(cfg["log"])
    if len(steps) < 3:
        continue
    arr = np.array(losses)
    kernel = np.ones(5) / 5
    smoothed = np.convolve(arr, kernel, mode="same")
    smoothed[:2] = arr[:2]
    smoothed[-2:] = arr[-2:]
    ax1.plot(steps, smoothed, color=cfg["color"], ls=cfg["ls"], lw=1.8, label=label, alpha=0.9)
    if loop_step:
        ax1.axvline(loop_step, color=cfg["color"], ls=":", lw=1.2, alpha=0.6)

ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.axvspan(2100, 2240, alpha=0.06, color="orange")
ax1.text(2170, 2.98, "loop\nactivates", ha="center", va="top", fontsize=7.5, color="#b35900")

# --- bottom: val_bpb ZOOMED ---
ax2.set_title("Val BPB — zoomed in (training checkpoints + post-TTT results)", fontsize=11)
ax2.set_xlabel("Step  [post-TTT shown at step 5200]")
ax2.set_ylabel("Val BPB")
ax2.set_xlim(0, 5500)
ax2.set_ylim(1.060, 1.120)

for label, cfg in RUNS.items():
    _, _, val_steps, val_bpbs, _ = parse_log(cfg["log"])
    if not val_steps:
        continue
    ax2.plot(val_steps, val_bpbs, color=cfg["color"], ls=cfg["ls"],
             lw=2, marker="o", markersize=5, label=label)
    # TTT point
    ttt = TTT_POINTS.get(label)
    if ttt:
        ax2.plot(ttt[0], ttt[1], marker="*", markersize=14,
                 color=cfg["color"], zorder=5)
        suffix = " (spinquant)" if "008" in label else ""
        ax2.annotate(f"TTT: {ttt[1]:.5f}{suffix}", xy=ttt,
                     xytext=(ttt[0]+30, ttt[1]+0.0005),
                     fontsize=8, color=cfg["color"])

# #1736 measured TTT reference line
ax2.axhline(SOTA_TTT, color="black", ls="--", lw=1.2, alpha=0.7)
ax2.text(200, SOTA_TTT + 0.0003, f"#1736 measured post-TTT: {SOTA_TTT}", fontsize=8, color="black")

# divider between training and TTT region
ax2.axvline(5000, color="gray", ls=":", lw=1.0, alpha=0.5)
ax2.text(5010, 1.119, "↑ post-TTT\n   eval", fontsize=7.5, color="gray", va="top")

ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.3)

ax2.axvline(4000, color="black", ls="--", lw=0.8, alpha=0.4)
ax2.text(4020, 1.118, "step 4000", fontsize=7.5, va="top", alpha=0.6)

plt.tight_layout()
out = "/home/claude-user/ai-workspace/projects/parameter-golf/research/scripts/training_loss_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved to {out}")
