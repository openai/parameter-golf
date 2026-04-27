"""Influence proxy: gradient inner product scoring for training shards.

TracIn-inspired single-checkpoint influence proxy. For each training shard,
computes the dot product of flattened validation gradients and shard gradients
to estimate data influence on validation loss.

Integrates R4.2 shard variance check (CV threshold).

CLI:
  python scripts/causal/influence_proxy.py \
    --checkpoint <path> \
    --train-data <dir> \
    --val-data <dir> \
    --output results/causal/diagnostics/influence_scores.json \
    [--max-shards 20]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Pure functions (testable without MLX)
# ---------------------------------------------------------------------------


def compute_dot_product(
    val_grad: dict[str, "np.ndarray | Any"],
    shard_grad: dict[str, "np.ndarray | Any"],
) -> float:
    """Compute dot product of two flat gradient dicts.

    Iterates shared keys, computes element-wise product, sums all.
    Works with both numpy arrays and mlx arrays (via .item() or float()).
    """
    total = 0.0
    for key in val_grad:
        if key in shard_grad:
            v = val_grad[key]
            s = shard_grad[key]
            # Support both numpy and mlx arrays
            if hasattr(v, 'flatten'):
                product = (v.flatten() * s.flatten()).sum()
            else:
                product = sum(a * b for a, b in zip(v, s))
            total += float(product)
    return total


def compute_cv(scores: list[float]) -> float:
    """Compute coefficient of variation = std / mean.

    Returns 0.0 if mean is zero.
    """
    arr = np.array(scores, dtype=np.float64)
    mean_val = float(np.mean(arr))
    if abs(mean_val) < 1e-15:
        return 0.0
    std_val = float(np.std(arr, ddof=0))
    return std_val / abs(mean_val)


def build_variance_check(scores: list[float]) -> dict:
    """Build the variance_check sub-object for the output schema.

    Returns dict with mean, std, cv, recommendation.
    """
    arr = np.array(scores, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=0))
    cv = compute_cv(scores)
    recommendation = "skip" if cv < 0.1 else "proceed"
    return {
        "mean": mean_val,
        "std": std_val,
        "cv": cv,
        "recommendation": recommendation,
    }


def build_output(
    checkpoint: str,
    scores: list[dict],
    variance_check: dict,
) -> dict:
    """Build the final output JSON object per I8 schema."""
    skipped = variance_check["recommendation"] == "skip"
    return {
        "checkpoint": checkpoint,
        "n_shards_scored": len(scores),
        "scores": scores,
        "variance_check": variance_check,
        "curriculum_skipped": skipped,
        "reason": "CV < 0.1" if skipped else None,
    }


# ---------------------------------------------------------------------------
# MLX-dependent functions (integration)
# ---------------------------------------------------------------------------


def _compute_gradient(model, tokens, seq_len: int = 1024):
    """Compute gradient of loss w.r.t. trainable params on a token batch.

    Uses plain (non-compiled) nn.value_and_grad.
    Returns flat gradient dict {param_name: mx.array}.
    """
    import mlx.core as mx
    import mlx.nn as nn

    n_tokens = len(tokens)
    n_seq = n_tokens // seq_len
    if n_seq < 1:
        raise ValueError(f"Need at least {seq_len} tokens, got {n_tokens}")

    batch = mx.array(tokens[: n_seq * seq_len].reshape(n_seq, seq_len))
    x = batch[:, :-1]
    y = batch[:, 1:]

    def loss_fn(model_inner):
        logits = model_inner(x)
        loss = nn.losses.cross_entropy(logits, y).mean()
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    _loss, grads = loss_and_grad_fn(model)

    # Flatten grad tree
    from mlx.utils import tree_flatten
    flat = dict(tree_flatten(grads))
    return flat


def main():
    parser = argparse.ArgumentParser(description="Influence proxy: gradient inner product scoring")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.safetensors)")
    parser.add_argument("--train-data", required=True, help="Directory with training shard files")
    parser.add_argument("--val-data", required=True, help="Directory with validation data")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--max-shards", type=int, default=20, help="Maximum shards to score")
    args = parser.parse_args()

    import mlx.core as mx

    # Add project root to path for imports
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.causal.common import load_model

    print(f"Loading model from {args.checkpoint}")
    model, _tokenizer = load_model(args.checkpoint)

    # Load validation data
    val_files = sorted(glob.glob(os.path.join(args.val_data, "val_*.bin")))
    if not val_files:
        raise FileNotFoundError(f"No val_*.bin files in {args.val_data}")
    val_tokens = np.fromfile(val_files[0], dtype=np.uint16)

    # Compute validation gradient (4 sequences of 1024 tokens = 4096 tokens)
    print("Computing validation gradient...")
    val_grad = _compute_gradient(model, val_tokens[:4096], seq_len=1024)
    mx.eval(val_grad)

    # Memory check after first gradient
    print(f"Val gradient computed. Keys: {len(val_grad)}")

    # Find training shards
    train_files = sorted(glob.glob(os.path.join(args.train_data, "train_*.bin")))
    if not train_files:
        raise FileNotFoundError(f"No train_*.bin files in {args.train_data}")

    n_shards = min(len(train_files), args.max_shards)
    print(f"Scoring {n_shards} of {len(train_files)} shards...")

    scores = []
    for i in range(n_shards):
        shard_path = train_files[i]
        shard_name = os.path.basename(shard_path)

        # Sample 4096 tokens (4 sequences of 1024)
        shard_tokens = np.fromfile(shard_path, dtype=np.uint16)[:4096]
        if len(shard_tokens) < 1024:
            print(f"  Skipping {shard_name}: too few tokens ({len(shard_tokens)})")
            continue

        shard_grad = _compute_gradient(model, shard_tokens, seq_len=1024)
        mx.eval(shard_grad)

        # Dot product
        score = compute_dot_product(val_grad, shard_grad)
        # HARD REQUIREMENT: mx.eval() after each computation
        # (already called on shard_grad above; score is a plain float)

        scores.append({"shard": shard_name, "influence_score": score})
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{n_shards}] {shard_name}: {score:.6f}")

    # Sort descending by influence score
    scores.sort(key=lambda x: x["influence_score"], reverse=True)

    # Variance check (R4.2)
    score_values = [s["influence_score"] for s in scores]
    variance_check = build_variance_check(score_values) if score_values else {
        "mean": 0.0, "std": 0.0, "cv": 0.0, "recommendation": "skip",
    }

    output = build_output(
        checkpoint=args.checkpoint,
        scores=scores,
        variance_check=variance_check,
    )

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nOutput written to {args.output}")
    print(f"Recommendation: {variance_check['recommendation']} (CV={variance_check['cv']:.4f})")


if __name__ == "__main__":
    main()
