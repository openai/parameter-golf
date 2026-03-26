"""Plot training loss curves from raw_runs.json files.

Reads step_losses from experiment results and produces treatment vs control
loss curve plots for visual comparison.

Usage:
  python scripts/causal/plot_losses.py \
    --input results/causal/cycle_1/*_raw_runs.json \
    --output results/causal/cycle_1/loss_curves.png
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _extract_curves(condition_data: dict) -> list[list[tuple[int, float]]]:
    """Extract (step, loss) curves from a condition's results."""
    curves = []
    for result in condition_data.get("results", []):
        step_losses = result.get("step_losses")
        if not step_losses:
            continue
        curve = [(s["step"], s["train_loss"]) for s in step_losses]
        curves.append(curve)
    return curves


def _compute_mean_std(curves: list[list[tuple[int, float]]]):
    """Compute per-step mean and std across seeds. Handles ragged arrays by truncating to shortest."""
    if not curves:
        return [], [], []
    min_len = min(len(c) for c in curves)
    steps = [curves[0][i][0] for i in range(min_len)]
    losses = np.array([[c[i][1] for i in range(min_len)] for c in curves])
    return steps, losses.mean(axis=0), losses.std(axis=0)


def plot_loss_curves(raw_runs: dict, output_path: str, title: str = "Training Loss Curves"):
    """Plot treatment vs control training loss curves.

    Args:
        raw_runs: Parsed raw_runs.json dict with treatment/control results containing step_losses.
        output_path: Path to save the PNG plot.
        title: Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    treatment = raw_runs.get("treatment", {})
    control = raw_runs.get("control", {})

    t_curves = _extract_curves(treatment)
    c_curves = _extract_curves(control)

    has_data = False

    # Plot treatment curves
    if t_curves:
        has_data = True
        for i, curve in enumerate(t_curves):
            steps, losses = zip(*curve)
            ax.plot(steps, losses, color="tab:blue", alpha=0.3, linewidth=0.8,
                    label="Treatment" if i == 0 else None)
        # Mean + std
        steps, mean, std = _compute_mean_std(t_curves)
        if steps:
            ax.plot(steps, mean, color="tab:blue", linewidth=2, label="Treatment (mean)")
            ax.fill_between(steps, mean - std, mean + std, color="tab:blue", alpha=0.1)

    # Plot control curves
    if c_curves:
        has_data = True
        for i, curve in enumerate(c_curves):
            steps, losses = zip(*curve)
            ax.plot(steps, losses, color="tab:red", alpha=0.3, linewidth=0.8,
                    label="Control" if i == 0 else None)
        # Mean + std
        steps, mean, std = _compute_mean_std(c_curves)
        if steps:
            ax.plot(steps, mean, color="tab:red", linewidth=2, label="Control (mean)")
            ax.fill_between(steps, mean - std, mean + std, color="tab:red", alpha=0.1)

    if not has_data:
        log.warning("No step_losses found in raw_runs — producing empty plot")
        ax.text(0.5, 0.5, "No step_losses data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss (nats)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Plot saved to %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument("--input", nargs="+", required=True, help="raw_runs.json file(s)")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Training Loss Curves", help="Plot title")
    args = parser.parse_args()

    for input_path in args.input:
        with open(input_path) as f:
            raw_runs = json.load(f)
        out = args.output if len(args.input) == 1 else args.output.replace(".png", f"_{Path(input_path).stem}.png")
        plot_loss_curves(raw_runs, out, title=args.title)


if __name__ == "__main__":
    main()
