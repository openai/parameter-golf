"""Pre/post quantization gap analysis (C8).

CLI: python scripts/causal/quant_gap_analysis.py \
       --checkpoint <path> \
       --val-data <path> \
       --tokenizer <path> \
       --output results/causal/diagnostics/quant_report.json \
       [--largest-training-effect <float>]

Loads model, evaluates pre-quant BPB, quantizes/dequantizes, evaluates
post-quant BPB, computes gap and threshold check. Outputs per I7b schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Pure-computation helpers (testable without MLX/checkpoint)
# ---------------------------------------------------------------------------

def compute_quant_gap(pre_bpb: float, post_bpb: float) -> float:
    """Compute quantization gap: post - pre."""
    return post_bpb - pre_bpb


def check_threshold(gap: float, largest_effect: float | None) -> bool:
    """Check if gap exceeds 3x the largest training effect.

    Returns False if largest_effect is None (no prior data).
    """
    if largest_effect is None:
        return False
    return gap > 3.0 * largest_effect


# ---------------------------------------------------------------------------
# Full pipeline (requires MLX + checkpoint)
# ---------------------------------------------------------------------------

def run_quant_gap_analysis(
    checkpoint_path: str,
    val_data_path: str,
    tokenizer_path: str,
    output_path: str,
    largest_training_effect: float | None = None,
) -> dict[str, Any]:
    """Run full quant gap analysis. Requires MLX and a checkpoint."""
    import mlx.core as mx
    import numpy as np
    import train_gpt_mlx as tgm

    from scripts.causal.common import compute_bpb, load_model

    # Load model and tokenizer
    model, sp = load_model(checkpoint_path)

    # Load validation tokens
    val_tokens = np.fromfile(
        str(Path(val_data_path) / "val_000000.bin"), dtype=np.uint16
    )

    # Pre-quant BPB
    pre_bpb = compute_bpb(model, val_tokens, sp)

    # Quantize -> dequantize (round-trip)
    state = dict(model.parameters())
    flat_state = {}
    for k, v in state.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat_state[f"{k}.{k2}"] = v2
        else:
            flat_state[k] = v

    quantized = tgm.quantize_state_dict_int8(flat_state)
    dequantized = tgm.dequantize_state_dict_int8(quantized)

    # Load dequantized weights back
    model.load_weights(list(dequantized.items()))
    mx.eval(model.parameters())

    # Post-quant BPB
    post_bpb = compute_bpb(model, val_tokens, sp)

    gap = compute_quant_gap(pre_bpb, post_bpb)
    exceeds = check_threshold(gap, largest_training_effect)

    result: dict[str, Any] = {
        "pre_quant_bpb": pre_bpb,
        "post_quant_bpb": post_bpb,
        "quant_gap": gap,
        "largest_training_effect": largest_training_effect,
        "gap_exceeds_3x_threshold": exceeds,
        "restrict_to_post_quant": exceeds,
        "per_category_deltas": None,  # Optional: requires token_loss_decompose
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre/post quantization gap analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--val-data", required=True, help="Path to validation data directory")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer model")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--largest-training-effect", type=float, default=None,
        help="Largest confirmed training effect size (from prior ablation)",
    )

    args = parser.parse_args()
    result = run_quant_gap_analysis(
        args.checkpoint, args.val_data, args.tokenizer, args.output,
        largest_training_effect=args.largest_training_effect,
    )

    print(f"Pre-quant BPB:  {result['pre_quant_bpb']:.4f}")
    print(f"Post-quant BPB: {result['post_quant_bpb']:.4f}")
    print(f"Quant gap:      {result['quant_gap']:.4f}")
    print(f"Exceeds 3x:     {result['gap_exceeds_3x_threshold']}")


if __name__ == "__main__":
    main()
