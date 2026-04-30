#!/usr/bin/env python3
"""Generate larger-variant plan based on artifact scaling results."""
import json, sys, os


def generate_plan(summary_path, output_path):
    with open(summary_path) as f:
        summary = json.load(f)

    checkpoints = summary.get("checkpoints", [])
    if not checkpoints:
        print("No checkpoint data found")
        return

    baseline_bytes = summary.get("baseline_artifact_bytes", 0)
    final = checkpoints[-1]
    delta = final.get("delta_vs_10min", 0)
    budget_freed = -delta if delta < 0 else 0

    plan = f"""# Larger Variant Plan

## Based on Scaling Results
- Baseline artifact: {baseline_bytes:,} bytes
- Final artifact delta: {delta:+,} bytes
- Budget freed by longer training: {budget_freed:,} bytes
- 16 MB cap: 16,000,000 bytes
"""

    if budget_freed >= 300000:
        plan += """
## Candidates (budget_freed >= 300KB)

### A. LQER_TOP_K=4 (add 1 more low-rank correction tensor)
- Estimated cost: ~80-120KB per additional tensor
- Risk: minimal, well-tested mechanism

### B. LQER_TOP_K=5
- Estimated cost: ~160-240KB for 2 more tensors
- Risk: diminishing returns likely

### C. Slightly wider model (MODEL_DIM=520 or 528)
- Estimated cost: ~200-400KB depending on dim increase
- Risk: may need hyperparameter re-tuning

### D. Additional layer (NUM_LAYERS=12)
- Estimated cost: ~500KB+
- Risk: significant, requires looping adjustment
"""
    elif budget_freed >= 50000:
        plan += """
## Candidates (budget_freed 50-300KB)

### Only conservative variants recommended:
### A. LQER_TOP_K=4 (if budget allows)
- Estimated cost: ~80-120KB
"""
    else:
        plan += """
## No larger variant recommended
- Insufficient artifact budget freed by longer training
- Consider quality-only benefits (better BPB at same size)
"""

    with open(output_path, "w") as f:
        f.write(plan)
    print(f"Plan written to: {output_path}")


if __name__ == "__main__":
    summary_path = (
        sys.argv[1] if len(sys.argv) > 1 else "results/scaling_summary.json"
    )
    output_path = (
        sys.argv[2] if len(sys.argv) > 2 else "results/larger_variant_plan.md"
    )
    generate_plan(summary_path, output_path)
