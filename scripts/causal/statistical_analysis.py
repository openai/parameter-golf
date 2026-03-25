"""Statistical analysis of paired-seed ablation experiments (C6).

CLI: python scripts/causal/statistical_analysis.py \
       --input raw_runs.json [--input raw_runs_2.json ...] \
       --output ablation_results.json

Takes raw_runs.json from experiment_runner.py, computes paired differences,
bootstrap CIs, t-test p-values, Holm-Bonferroni correction, and decision gate.
Outputs per design I6 schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.causal.common import decision_gate, holm_bonferroni, paired_ttest


# ---------------------------------------------------------------------------
# Platform transfer
# ---------------------------------------------------------------------------

def compute_platform_transfer(
    mlx_effect: float,
    h100_effect: float,
    divergence_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute transfer coefficient between MLX and H100 effects.

    transfer_coefficient = h100_effect / mlx_effect (when mlx_effect != 0).
    divergence_flag is True when the coefficient is far from 1.0.
    """
    if mlx_effect == 0:
        coeff = float("inf") if h100_effect != 0 else 1.0
    else:
        coeff = h100_effect / mlx_effect

    return {
        "mlx_effect": mlx_effect,
        "h100_effect": h100_effect,
        "transfer_coefficient": coeff,
        "divergence_flag": abs(coeff - 1.0) > divergence_threshold,
    }


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _load_raw_runs(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_paired_bpbs(raw: dict) -> tuple[list[float], list[float], list[int], str]:
    """Extract matched treatment/control BPB pairs, skipping error results.

    Returns (treatment_bpbs, control_bpbs, valid_seeds, platform).
    """
    platform = raw.get("platform", "unknown")
    treatment_results = raw["treatment"]["results"]
    control_results = raw["control"]["results"]

    # Index control by seed for matching
    control_by_seed = {r["seed"]: r for r in control_results}

    treatment_bpbs: list[float] = []
    control_bpbs: list[float] = []
    valid_seeds: list[int] = []

    for tr in treatment_results:
        if "error" in tr:
            continue
        seed = tr["seed"]
        cr = control_by_seed.get(seed)
        if cr is None or "error" in cr:
            continue
        treatment_bpbs.append(tr["val_bpb"])
        control_bpbs.append(cr["val_bpb"])
        valid_seeds.append(seed)

    return treatment_bpbs, control_bpbs, valid_seeds, platform


def analyse_raw_runs(
    input_paths: str | list[str],
    output_path: str,
    mde: float = 0.002,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """Run full statistical analysis on one or more raw_runs.json files.

    Returns the ablation_results dict and writes it to output_path.
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []

    # Collect per-platform effects for transfer coefficient
    platform_effects: dict[str, float] = {}

    for i, path in enumerate(input_paths):
        raw = _load_raw_runs(path)
        treatment_bpbs, control_bpbs, valid_seeds, platform = _extract_paired_bpbs(raw)

        if len(treatment_bpbs) < 2:
            # Need at least 2 pairs for t-test
            comparisons.append({
                "name": f"comparison_{i}",
                "platform": platform,
                "n_seeds": len(treatment_bpbs),
                "mean_effect": 0.0,
                "ci_lo": 0.0,
                "ci_hi": 0.0,
                "p_value": 1.0,
                "p_value_corrected": 1.0,
                "decision": "null",
            })
            raw_p_values.append(1.0)
            continue

        mean_diff, ci_lo, ci_hi, p_value = paired_ttest(treatment_bpbs, control_bpbs)

        comparisons.append({
            "name": f"comparison_{i}",
            "platform": platform,
            "n_seeds": len(valid_seeds),
            "mean_effect": mean_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_value": p_value,
            "p_value_corrected": p_value,  # Updated after correction
            "decision": "",  # Updated after correction
        })
        raw_p_values.append(p_value)
        platform_effects[platform] = mean_diff

    # Apply Holm-Bonferroni correction across all comparisons
    if len(raw_p_values) > 0:
        _reject, adjusted = holm_bonferroni(raw_p_values, alpha=alpha)
        for j, comp in enumerate(comparisons):
            comp["p_value_corrected"] = adjusted[j]
            comp["decision"] = decision_gate(comp["mean_effect"], adjusted[j], mde=mde)

    # Platform transfer
    platform_transfer = None
    if "mlx" in platform_effects and "h100" in platform_effects:
        platform_transfer = compute_platform_transfer(
            platform_effects["mlx"], platform_effects["h100"]
        )

    result: dict[str, Any] = {
        "comparisons": comparisons,
        "correction_method": "holm-bonferroni",
        "alpha": alpha,
        "mde": mde,
        "platform_transfer": platform_transfer,
    }

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical analysis of ablation results")
    parser.add_argument(
        "--input", required=True, nargs="+",
        help="Path(s) to raw_runs.json file(s)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for ablation_results.json output",
    )
    parser.add_argument("--mde", type=float, default=0.002, help="Minimum detectable effect")
    parser.add_argument("--alpha", type=float, default=0.01, help="Significance level")

    args = parser.parse_args()
    result = analyse_raw_runs(args.input, args.output, mde=args.mde, alpha=args.alpha)

    n_confirmed = sum(1 for c in result["comparisons"] if c["decision"] == "confirmed")
    n_suggestive = sum(1 for c in result["comparisons"] if c["decision"] == "suggestive")
    n_null = sum(1 for c in result["comparisons"] if c["decision"] == "null")
    print(f"Analysis complete: {n_confirmed} confirmed, {n_suggestive} suggestive, {n_null} null")


if __name__ == "__main__":
    main()
