#!/usr/bin/env python3
"""Plot recur-alpha trajectories across specs 015 (α=0 init), 016 (α=1 init),
017 (α=1 init, full pipeline — same commit as 016).

Parses `recur_alpha: values=[[...]] grad_norm=...` lines paired with
`{step}/20000 train_loss:` lines from each run's train.log.

Layout per log line: [[pass2_L3, pass2_L4, pass2_L5], [pass3_L3, pass3_L4, pass3_L5]]

Output:
- research/scripts/alpha_trajectory.png — 6-panel plot, one per (pass, layer)
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

RUN_SPECS = [
    ("015", ROOT / "runs/015-recur-alpha/seed_42/train.log",    "#1f77b4", "015 (α=0 init)"),
    ("016", ROOT / "runs/016-recur-alpha-ones/seed_42/train.log", "#ff7f0e", "016 (α=1 init)"),
    ("017", ROOT / "runs/017-recur-alpha-full/seed_42/train.log", "#2ca02c", "017 (α=1 init, full pipeline)"),
]

STEP_RE = re.compile(r"^(\d+)/\d+ train_loss:", re.M)
ALPHA_RE = re.compile(r"recur_alpha: values=(\[\[.*?\]\])", re.M)


def parse_log(path):
    """Return list of (step, alpha_6d) tuples. Alpha is flat 6 floats: [p2L3, p2L4, p2L5, p3L3, p3L4, p3L5]."""
    text = path.read_text()
    lines = text.splitlines()
    steps, alphas = [], []
    cur_step = None
    for ln in lines:
        m = STEP_RE.match(ln)
        if m:
            cur_step = int(m.group(1))
            continue
        m = ALPHA_RE.search(ln)
        if m and cur_step is not None:
            nested = json.loads(m.group(1))
            flat = nested[0] + nested[1]
            steps.append(cur_step)
            alphas.append(flat)
            cur_step = None  # consume
    return np.array(steps), np.array(alphas)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    titles = [
        "pass-2, L3", "pass-2, L4", "pass-2, L5",
        "pass-3, L3", "pass-3, L4", "pass-3, L5",
    ]

    all_data = []
    for slug, path, color, label in RUN_SPECS:
        if not path.exists():
            print(f"WARN missing {path}")
            continue
        steps, alphas = parse_log(path)
        print(f"{slug}: {len(steps)} α snapshots, steps {steps[0]}–{steps[-1]}")
        all_data.append((slug, steps, alphas, color, label))

    for idx, ax in enumerate(axes.flat):
        for slug, steps, alphas, color, label in all_data:
            ax.plot(steps, alphas[:, idx], color=color, lw=1.8, label=label, alpha=0.85)
        ax.set_title(titles[idx], fontsize=11)
        ax.grid(alpha=0.3)
        ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.5)
        # Approx looping-activation step
        ax.axvline(2140, color="#888", ls="--", lw=0.7, alpha=0.5)

    axes[0, 0].set_ylabel("α", fontsize=11)
    axes[1, 0].set_ylabel("α", fontsize=11)
    for ax in axes[1, :]:
        ax.set_xlabel("training step", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, 1.02), frameon=False,
    )
    fig.suptitle(
        "Recur-alpha trajectories — specs 015, 016, 017 at seed 42 (same commit family)",
        fontsize=13, y=1.06,
    )

    plt.tight_layout()
    out = Path(__file__).parent / "alpha_trajectory.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    # Print quick summary of late-training plateau values
    print()
    print("Late-training α values (step >= 3500 mean):")
    header = "spec  |  p2L3  p2L4  p2L5  |  p3L3  p3L4  p3L5"
    print(header)
    print("-" * len(header))
    for slug, steps, alphas, color, label in all_data:
        mask = steps >= 3500
        if mask.any():
            late = alphas[mask].mean(axis=0)
            print(f"{slug}   |  {late[0]:.3f}  {late[1]:.3f}  {late[2]:.3f}  |  {late[3]:.3f}  {late[4]:.3f}  {late[5]:.3f}")


if __name__ == "__main__":
    main()
