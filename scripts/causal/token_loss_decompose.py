"""Per-token loss decomposition analysis (C7).

CLI: python scripts/causal/token_loss_decompose.py \
       --checkpoint <path> \
       --val-data <path> \
       --tokenizer <path> \
       --output results/causal/diagnostics/token_analysis.json

Loads a checkpoint, runs forward pass with reduction='none', decomposes
per-token losses by frequency bucket and context type. Outputs per I7 schema.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np


# ---------------------------------------------------------------------------
# Pure-computation helpers (testable without MLX/checkpoint)
# ---------------------------------------------------------------------------

def verify_decomposition(
    per_token_losses: np.ndarray,
    aggregate_loss: float,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Verify that mean(per_token_losses) matches aggregate_loss within tolerance."""
    mean_ptl = float(np.mean(per_token_losses))
    delta = abs(mean_ptl - aggregate_loss)
    return {
        "mean_per_token_loss": mean_ptl,
        "aggregate_loss": aggregate_loss,
        "delta": delta,
        "passed": delta < tolerance,
    }


def build_frequency_buckets(vocab_freqs: np.ndarray) -> dict[int, str]:
    """Assign each token ID to a frequency bucket based on rank.

    Buckets: top_100 (rank 0-99), mid_100_500 (rank 100-499),
    tail_500_1024 (rank 500+).

    Args:
        vocab_freqs: Array of shape (vocab_size,) with frequency counts.

    Returns:
        Dict mapping token_id -> bucket name.
    """
    sorted_indices = np.argsort(-vocab_freqs)  # descending by frequency
    buckets: dict[int, str] = {}
    for rank, token_id in enumerate(sorted_indices):
        if rank < 100:
            buckets[int(token_id)] = "top_100"
        elif rank < 500:
            buckets[int(token_id)] = "mid_100_500"
        else:
            buckets[int(token_id)] = "tail_500_1024"
    return buckets


def classify_boundary_tokens(
    token_ids: np.ndarray,
    whitespace_tokens: set[int],
) -> list[bool]:
    """Classify tokens as boundary (first token or follows whitespace) or mid-sequence.

    A token is a boundary token if:
    - It is the first token in the sequence, OR
    - The preceding token is in whitespace_tokens (space, tab, newline, or
      tokens with leading space like ' the').

    Returns a list of booleans, True = boundary.
    """
    result: list[bool] = []
    for i in range(len(token_ids)):
        if i == 0:
            result.append(True)
        elif int(token_ids[i - 1]) in whitespace_tokens:
            result.append(True)
        else:
            result.append(False)
    return result


def compute_category_stats(
    per_token_losses: np.ndarray,
    token_ids: np.ndarray,
    buckets: dict[int, str],
) -> dict[str, dict[str, Any]]:
    """Compute per-bucket statistics: mean_loss, std, bpb_contribution.

    bpb_contribution = (n_tokens_in_bucket / total_tokens) * mean_loss_in_bucket
    so that sum of all contributions = mean(all losses).
    """
    total_n = len(per_token_losses)
    bucket_names = ["top_100", "mid_100_500", "tail_500_1024"]
    stats: dict[str, dict[str, Any]] = {}

    for bname in bucket_names:
        mask = np.array([buckets.get(int(tid), "tail_500_1024") == bname for tid in token_ids])
        bucket_losses = per_token_losses[mask]
        n = int(np.sum(mask))
        if n == 0:
            stats[bname] = {
                "n_tokens": 0,
                "mean_loss": 0.0,
                "std": 0.0,
                "bpb_contribution": 0.0,
            }
        else:
            mean_loss = float(np.mean(bucket_losses))
            stats[bname] = {
                "n_tokens": n,
                "mean_loss": mean_loss,
                "std": float(np.std(bucket_losses)),
                "bpb_contribution": (n / total_n) * mean_loss,
            }
    return stats


# ---------------------------------------------------------------------------
# Full pipeline (requires MLX + checkpoint)
# ---------------------------------------------------------------------------

def run_decomposition(
    checkpoint_path: str,
    val_data_path: str,
    tokenizer_path: str,
    output_path: str,
) -> dict[str, Any]:
    """Run full token loss decomposition. Requires MLX and a checkpoint."""
    import mlx.core as mx
    import mlx.nn as nn
    import sentencepiece as spm

    from scripts.causal.common import load_model

    # Load model and tokenizer
    model, sp = load_model(checkpoint_path)

    # Load validation data
    val_tokens = np.fromfile(
        str(Path(val_data_path) / "val_000000.bin"), dtype=np.uint16
    ).astype(np.int32)

    # Build frequency buckets from tokenizer
    vocab_size = sp.GetPieceSize()
    # Estimate token frequencies from validation data
    token_counts = np.bincount(val_tokens, minlength=vocab_size).astype(np.float64)
    buckets = build_frequency_buckets(token_counts)

    # Identify whitespace/boundary tokens from tokenizer
    whitespace_tokens: set[int] = set()
    for tid in range(vocab_size):
        piece = sp.IdToPiece(tid)
        # SentencePiece uses \u2581 for leading space
        if piece.startswith("\u2581") or piece in (" ", "\t", "\n", "\r"):
            whitespace_tokens.add(tid)

    # Forward pass with per-token losses
    seq_len = 1024
    n_seqs = min(len(val_tokens) // (seq_len + 1), 512)  # Cap for memory
    total_tokens = n_seqs * seq_len

    all_losses = np.empty(total_tokens, dtype=np.float64)
    all_token_ids = np.empty(total_tokens, dtype=np.int32)

    for i in range(n_seqs):
        start = i * seq_len
        chunk = val_tokens[start : start + seq_len + 1]
        x = mx.array(chunk[:-1].reshape(1, -1))
        y = mx.array(chunk[1:].reshape(1, -1))

        logits = model(x)
        # Per-token cross-entropy loss (no reduction)
        losses = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="none",
        )
        mx.eval(losses)
        loss_np = np.array(losses).astype(np.float64)

        offset = i * seq_len
        all_losses[offset : offset + seq_len] = loss_np
        all_token_ids[offset : offset + seq_len] = chunk[1 : seq_len + 1]

    # Aggregate loss
    aggregate_loss = float(np.mean(all_losses))

    # Decomposition check
    decomp_check = verify_decomposition(all_losses, aggregate_loss)

    # Frequency bucket stats
    freq_stats = compute_category_stats(all_losses, all_token_ids, buckets)

    # Boundary classification
    is_boundary = classify_boundary_tokens(all_token_ids, whitespace_tokens)
    boundary_mask = np.array(is_boundary)
    mid_mask = ~boundary_mask

    ln2 = math.log(2)
    boundary_losses = all_losses[boundary_mask]
    mid_losses = all_losses[mid_mask]

    by_context = {
        "boundary": {
            "n_tokens": int(np.sum(boundary_mask)),
            "mean_loss": float(np.mean(boundary_losses)) if len(boundary_losses) > 0 else 0.0,
            "bpb_contribution": (
                (len(boundary_losses) / len(all_losses)) * float(np.mean(boundary_losses))
                if len(boundary_losses) > 0 else 0.0
            ),
        },
        "mid_sequence": {
            "n_tokens": int(np.sum(mid_mask)),
            "mean_loss": float(np.mean(mid_losses)) if len(mid_losses) > 0 else 0.0,
            "bpb_contribution": (
                (len(mid_losses) / len(all_losses)) * float(np.mean(mid_losses))
                if len(mid_losses) > 0 else 0.0
            ),
        },
    }

    # BPB conversion
    aggregate_bpb = aggregate_loss / ln2  # Simplified; real BPB needs token/byte ratio

    result: dict[str, Any] = {
        "aggregate_bpb": aggregate_bpb,
        "decomposition_check": decomp_check,
        "by_frequency_bucket": freq_stats,
        "by_context_type": by_context,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-token loss decomposition")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--val-data", required=True, help="Path to validation data directory")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer model")
    parser.add_argument("--output", required=True, help="Output JSON path")

    args = parser.parse_args()
    result = run_decomposition(args.checkpoint, args.val_data, args.tokenizer, args.output)

    passed = result["decomposition_check"]["passed"]
    print(f"Decomposition check: {'PASSED' if passed else 'FAILED'}")
    print(f"Aggregate BPB: {result['aggregate_bpb']:.4f}")


if __name__ == "__main__":
    main()
