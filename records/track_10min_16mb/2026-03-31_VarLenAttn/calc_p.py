"""One-sided Welch's t-test for val_loss, plus summary stats for val_bpb."""

import argparse
import glob
import os
import re
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DEFAULT_BASELINE = os.path.join(REPO_ROOT, "records", "track_10min_16mb",
                                "2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072", "*.log")


def extract_metrics(path: str) -> tuple[float, float]:
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    last = lines[-1]
    loss_match = re.search(r"val_loss:([\d.]+)", last)
    bpb_match = re.search(r"val_bpb:([\d.]+)", last)
    if not loss_match or not bpb_match:
        raise ValueError(f"no val_loss/val_bpb in final line of {path}: {last!r}")
    return float(loss_match.group(1)), float(bpb_match.group(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", nargs="+", default=None)
    p.add_argument("--logs", nargs="+")
    p.add_argument("--losses", nargs="+", type=float)
    p.add_argument("--threshold", type=float, default=0.005)
    args = p.parse_args()
    assert (args.logs is None) != (args.losses is None), "pass exactly one of --logs or --losses"

    baseline_paths = args.baseline or sorted(glob.glob(DEFAULT_BASELINE))
    assert baseline_paths, f"no baseline logs found matching {DEFAULT_BASELINE}"
    baseline_metrics = np.array([extract_metrics(f) for f in baseline_paths])
    baseline_loss = baseline_metrics[:, 0]
    baseline_bpb = baseline_metrics[:, 1]

    if args.logs:
        new_metrics = np.array([extract_metrics(f) for f in args.logs])
        new_loss = new_metrics[:, 0]
        new_bpb = new_metrics[:, 1]
    else:
        new_loss = np.array(args.losses)
        new_bpb = None

    print(f"baseline val_loss: {baseline_loss}  mean={baseline_loss.mean():.6f}")
    print(f"new      val_loss: {new_loss}  mean={new_loss.mean():.6f}")
    print(f"delta (baseline - new): {baseline_loss.mean() - new_loss.mean():.6f}")
    if new_bpb is not None:
        print(f"\nbaseline val_bpb:  {baseline_bpb}  mean={baseline_bpb.mean():.6f}")
        print(f"new      val_bpb:  {new_bpb}  mean={new_bpb.mean():.6f}")
        print(f"delta (baseline - new): {baseline_bpb.mean() - new_bpb.mean():.6f}")
    print(f"\nval delta loss threshold: {args.threshold}")

    # H0: μ_baseline - μ_new <= threshold  (improvement is at most threshold)
    # H1: μ_baseline - μ_new > threshold
    if len(new_loss) == 1:
        # One-sample t-test: is baseline - threshold significantly above the new value?
        t, p_two = stats.ttest_1samp(baseline_loss - args.threshold, new_loss[0])
        p_val = p_two / 2 if t > 0 else 1 - p_two / 2
    elif len(baseline_loss) == 1:
        t, p_two = stats.ttest_1samp(new_loss + args.threshold, baseline_loss[0])
        p_val = p_two / 2 if t < 0 else 1 - p_two / 2
    else:
        shifted = baseline_loss - args.threshold
        t, p_two = stats.ttest_ind(shifted, new_loss, equal_var=False)
        p_val = p_two / 2 if t > 0 else 1 - p_two / 2
    print(f"p-value (new is ≥{args.threshold} below baseline): {p_val:.6f}")


if __name__ == "__main__":
    main()
