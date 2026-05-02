#!/usr/bin/env python3
"""Analyze long-train scaling results from checkpoint JSONs."""

import json, os, sys, csv, glob
from pathlib import Path


def find_checkpoint_jsons(results_dir):
    """Find all checkpoint_*min.json files."""
    pattern = os.path.join(results_dir, "checkpoint_*min.json")
    files = sorted(
        glob.glob(pattern),
        key=lambda f: int(Path(f).stem.split("_")[1].replace("min", "")),
    )
    return files


def analyze(results_dir, output_dir=None):
    if output_dir is None:
        output_dir = results_dir

    jsons = find_checkpoint_jsons(results_dir)
    if not jsons:
        print(f"No checkpoint JSONs found in {results_dir}")
        return

    rows = []
    for f in jsons:
        with open(f) as fh:
            rows.append(json.load(fh))

    rows.sort(key=lambda r: r.get("checkpoint_minute", 0))

    # Write CSV
    csv_path = os.path.join(output_dir, "scaling_results.csv")
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # Analysis
    baseline = rows[0]
    baseline_bytes = baseline.get("artifact_bytes", 0)

    summary = {
        "research_question": "Does longer training make PR #1950 model more compressible?",
        "baseline_artifact_bytes": baseline_bytes,
        "checkpoints": [],
        "recommendation": "",
    }

    for row in rows:
        minute = row.get("checkpoint_minute", 0)
        art_bytes = row.get("artifact_bytes", 0)
        delta = art_bytes - baseline_bytes if baseline_bytes else 0
        summary["checkpoints"].append(
            {
                "minute": minute,
                "artifact_bytes": art_bytes,
                "delta_vs_10min": delta,
                "train_steps": row.get("train_steps", 0),
                "pre_quant_bpb": row.get("pre_quant_bpb"),
                "quantized_bpb": row.get("quantized_bpb"),
                "post_ttt_bpb": row.get("post_ttt_bpb"),
            }
        )

    # Decision thresholds
    final = rows[-1]
    final_bytes = final.get("artifact_bytes", 0)
    final_delta = final_bytes - baseline_bytes
    final_bpb = final.get("quantized_bpb") or final.get("pre_quant_bpb")
    baseline_bpb = baseline.get("quantized_bpb") or baseline.get("pre_quant_bpb")

    bpb_improved = final_bpb and baseline_bpb and final_bpb < baseline_bpb

    if final_delta <= -300000 and bpb_improved:
        summary["recommendation"] = (
            "STRONG_POSITIVE: 300KB+ artifact shrink with BPB improvement. "
            "Recommend testing larger non-record model."
        )
    elif final_delta <= -50000:
        summary["recommendation"] = (
            "MODERATE_POSITIVE: 50-300KB artifact shrink. "
            "Report same-model scaling benefit."
        )
    elif final_delta > 0 and bpb_improved:
        summary["recommendation"] = (
            "QUALITY_ONLY: Longer training improves BPB but not compressibility."
        )
    else:
        summary["recommendation"] = "NEGATIVE: No clear benefit from longer training."

    # Write summary JSON
    summary_path = os.path.join(output_dir, "scaling_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Write markdown summary
    md_path = os.path.join(output_dir, "scaling_summary.md")
    with open(md_path, "w") as f:
        f.write("# Long-Train Artifact Scaling Results\n\n")
        f.write(f"## Recommendation: {summary['recommendation']}\n\n")
        f.write(f"Baseline (10min): {baseline_bytes:,} bytes\n\n")
        f.write("| Minute | Steps | Artifact Bytes | Δ vs 10min | BPB |\n")
        f.write("|--------|-------|---------------|------------|-----|\n")
        for cp in summary["checkpoints"]:
            bpb_str = (
                f"{cp['quantized_bpb']:.5f}"
                if cp.get("quantized_bpb")
                else (
                    f"{cp['pre_quant_bpb']:.5f}"
                    if cp.get("pre_quant_bpb")
                    else "N/A"
                )
            )
            f.write(
                f"| {cp['minute']} | {cp['train_steps']} "
                f"| {cp['artifact_bytes']:,} | {cp['delta_vs_10min']:+,} "
                f"| {bpb_str} |\n"
            )
        f.write(f"\n## Decision\n\n{summary['recommendation']}\n")

    print(f"Analysis written to: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Markdown: {md_path}")
    print(f"\nRecommendation: {summary['recommendation']}")
    return summary


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = sys.argv[2] if len(sys.argv) > 2 else results_dir
    analyze(results_dir, output_dir)
