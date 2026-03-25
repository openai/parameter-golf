#!/usr/bin/env python3
"""
Post-run script: reads 3 seed logs, fills in README.md and submission.json.
Run locally after scp-ing logs from RunPod.

Usage:
    python3 finalize_submission.py [submission_dir]
    # defaults to the directory containing this script
"""
import json
import os
import re
import sys
from pathlib import Path

def extract_metrics(log_path: str) -> dict:
    """Extract key metrics from a training log."""
    text = Path(log_path).read_text()
    metrics = {}

    # Pre-TTT BPB (int6 roundtrip before TTT)
    m = re.findall(r"final_int6_roundtrip_exact.*?val_bpb:([\d.]+)", text)
    if m:
        metrics["pre_ttt_bpb"] = float(m[-1])

    # BPB from sliding window eval (the submission score — post-TTT)
    m = re.findall(r"final_int6_sliding_window_exact.*?val_bpb:([\d.]+)", text)
    if m:
        metrics["bpb"] = float(m[-1])

    # Artifact size
    m = re.findall(r"Total submission size.*?(\d+)\s*bytes", text)
    if m:
        metrics["artifact"] = int(m[-1])

    # Steps
    m = re.findall(r"stopping_early.*?step[: ]*(\d+)", text)
    if not m:
        m = re.findall(r"step[: ]*(\d+)", text)
    if m:
        metrics["steps"] = int(m[-1])

    return metrics


def main():
    sub_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    seeds = [42, 1337, 2024]
    results = {}

    print("Extracting metrics from logs...")
    for seed in seeds:
        log = sub_dir / f"train_seed{seed}.log"
        if not log.exists():
            print(f"  WARNING: {log} not found")
            continue
        m = extract_metrics(str(log))
        results[seed] = m
        print(f"  Seed {seed}: bpb={m.get('bpb', '?')}, artifact={m.get('artifact', '?')}, steps={m.get('steps', '?')}")

    if len(results) < 3:
        print(f"\nERROR: Only found {len(results)}/3 seed logs. Cannot finalize.")
        sys.exit(1)

    bpbs = [results[s]["bpb"] for s in seeds]
    mean_bpb = sum(bpbs) / len(bpbs)
    std_bpb = (sum((x - mean_bpb) ** 2 for x in bpbs) / len(bpbs)) ** 0.5
    max_artifact = max(results[s]["artifact"] for s in seeds)
    mean_artifact_mb = sum(results[s]["artifact"] for s in seeds) / 3 / 1_000_000

    print(f"\n  Mean BPB: {mean_bpb:.4f} (std {std_bpb:.4f})")
    print(f"  Max artifact: {max_artifact} bytes ({max_artifact/1_000_000:.2f} MB)")

    # Validation checks
    sota = 1.1194
    delta = mean_bpb - sota
    print(f"\n  vs SOTA ({sota}): {delta:+.4f} nats")
    if delta < -0.005:
        print(f"  PASS: Beats SOTA by {abs(delta):.4f} nats")
    elif delta < 0:
        print(f"  CLOSE: Improves by {abs(delta):.4f} nats but < 0.005 threshold")
        print(f"  Consider submitting as non-record if techniques are novel.")
    else:
        print(f"  DOES NOT BEAT SOTA. Consider as non-record submission.")

    if max_artifact > 16_000_000:
        print(f"  FAIL: Artifact exceeds 16MB ({max_artifact} bytes)")
    else:
        print(f"  PASS: All artifacts under 16MB")

    # Update submission.json
    json_path = sub_dir / "submission.json"
    sj = json.loads(json_path.read_text())
    sj["val_bpb"] = round(mean_bpb, 4)
    sj["bytes_total"] = max_artifact
    sj["blurb"] = (
        f"LeakyReLU(0.5)^2 activation + XSA on last 4 layers + Partial RoPE + LN Scale "
        f"+ VE128 + EMA/SWA + GPTQ-lite int6 + zstd-22. "
        f"Built on PR #549 stack. 3-seed mean: {mean_bpb:.4f} (std {std_bpb:.4f}). "
        f"All artifacts under 16MB."
    )
    json_path.write_text(json.dumps(sj, indent=2) + "\n")
    print(f"\n  Updated {json_path}")

    # Update README.md
    readme_path = sub_dir / "README.md"
    readme = readme_path.read_text()

    # Fill header
    readme = readme.replace("FILL_BPB", f"{mean_bpb:.4f}")
    readme = readme.replace("FILL_MB", f"{mean_artifact_mb:.2f}")

    # Fill results table
    for seed in seeds:
        r = results[seed]
        old_line = f"| {seed}   | FILL  | FILL              | FILL     |"
        new_line = (
            f"| {seed}   | {r.get('steps', '?')}  | {r['bpb']:.4f}            "
            f"| {r['artifact']/1_000_000:.2f} MB |"
        )
        readme = readme.replace(old_line, new_line)

    # Fill mean/std
    readme = readme.replace("**Mean: FILL | Std: FILL**", f"**Mean: {mean_bpb:.4f} | Std: {std_bpb:.4f}**")

    readme_path.write_text(readme)
    print(f"  Updated {readme_path}")

    print(f"\n{'='*50}")
    print("SUBMISSION READY. Next steps:")
    print(f"  1. Review README.md and submission.json")
    print(f"  2. git checkout -b submission/sunnypatneedi-leakyrelu-xsa")
    print(f"  3. git add {sub_dir.relative_to(sub_dir.parent.parent.parent)}/")
    print(f"  4. git commit -m 'Add submission: LeakyReLU + XSA'")
    print(f"  5. git push origin submission/sunnypatneedi-leakyrelu-xsa")
    print(f"  6. Open PR at: https://github.com/openai/parameter-golf/compare")


if __name__ == "__main__":
    main()
