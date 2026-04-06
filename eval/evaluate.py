#!/usr/bin/env python3
"""BPB evaluation script for Parameter Golf.

Loads a checkpoint and evaluates BPB on FineWeb validation data
with sliding window support.

Usage:
    python eval/evaluate.py --checkpoint experiments/checkpoints/my_model.pt
    python eval/evaluate.py --checkpoint experiments/checkpoints/my_model.pt --stride 64
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.tiny_gpt import TinyGPT
from models.tiny_gpt_shared import TinyGPTShared
from utils.data import load_fineweb_valid


def parse_args():
    p = argparse.ArgumentParser(description="Parameter Golf Evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--stride", type=int, default=0,
                   help="Sliding window stride. 0 = use full seq_len (no overlap)")
    p.add_argument("--max-eval-bytes", type=int, default=0,
                   help="Max bytes to evaluate. 0 = use all validation data")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for evaluation (sliding window is sequential)")
    p.add_argument("--save-csv", type=str, default="",
                   help="Optional CSV path to save results")
    return p.parse_args()


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module:
    """Reconstruct model from checkpoint args."""
    args = ckpt["args"]
    if args["model_type"] == "standard":
        model = TinyGPT(
            vocab_size=args["vocab_size"],
            d_model=args["d_model"],
            n_layers=args["n_layers"],
            n_heads=args["n_heads"],
            mlp_ratio=args["mlp_ratio"],
            max_seq_len=args["max_seq_len"],
            tie_embeddings=args["tie_embeddings"],
            parallel_residual=args.get("parallel_residual", False),
        )
    else:
        model = TinyGPTShared(
            vocab_size=args["vocab_size"],
            d_model=args["d_model"],
            n_unique_layers=args["n_unique_layers"],
            n_loops=args["n_loops"],
            n_heads=args["n_heads"],
            mlp_ratio=args["mlp_ratio"],
            max_seq_len=args["max_seq_len"],
            tie_embeddings=args["tie_embeddings"],
            parallel_residual=args.get("parallel_residual", False),
        )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


def evaluate_sliding_window(model, data: np.ndarray, seq_len: int,
                            stride: int, device: torch.device,
                            dtype_ctx, max_bytes: int = 0):
    """Evaluate BPB with sliding window.

    For each window, only score tokens beyond the overlap region
    (tokens that are new in this window). This gives near-full
    context for every scored token.
    """
    if stride <= 0:
        stride = seq_len

    if max_bytes > 0:
        data = data[:max_bytes + 1]

    n_bytes = len(data) - 1  # -1 because we need target for last token
    total_nll = 0.0
    total_scored = 0

    model.eval()
    start = time.time()

    pos = 0
    windows = 0
    with torch.no_grad():
        while pos + seq_len < len(data):
            chunk = data[pos: pos + seq_len + 1]
            x = torch.from_numpy(chunk[:-1].copy()).long().unsqueeze(0).to(device)
            y = torch.from_numpy(chunk[1:].copy()).long().unsqueeze(0).to(device)

            with dtype_ctx:
                logits, _ = model(x)

            # Only score tokens in the "new" region (after overlap)
            if pos == 0:
                score_start = 0
            else:
                score_start = seq_len - stride

            log_probs = F.log_softmax(logits[:, score_start:, :], dim=-1)
            targets = y[:, score_start:]

            # Gather the log probs for actual targets
            nll = F.nll_loss(
                log_probs.reshape(-1, log_probs.size(-1)),
                targets.reshape(-1),
                reduction="sum"
            )
            n_tokens = targets.numel()

            total_nll += nll.item()
            total_scored += n_tokens
            windows += 1

            pos += stride

    elapsed = time.time() - start
    avg_nll = total_nll / total_scored
    bpb = avg_nll / math.log(2)

    return {
        "bpb": bpb,
        "avg_nll": avg_nll,
        "total_scored_tokens": total_scored,
        "total_scored_bytes": total_scored,  # byte-level, so tokens = bytes
        "n_windows": windows,
        "stride": stride,
        "seq_len": seq_len,
        "eval_time_seconds": elapsed,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    from contextlib import nullcontext
    if device.type == "cuda":
        dtype_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        dtype_ctx = nullcontext()

    # Load checkpoint
    print(f"[eval] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, device)
    ckpt_args = ckpt["args"]
    seq_len = ckpt_args["seq_len"] if "seq_len" in ckpt_args else ckpt_args.get("max_seq_len", 512)

    print(f"[eval] Model: {ckpt_args['model_type']}, d={ckpt_args['d_model']}")
    print(f"[eval] Trained for {ckpt.get('step', '?')} steps, "
          f"train val_bpb={ckpt.get('val_bpb', '?')}")

    # Load validation data
    print("[eval] Loading validation data...")
    val_data = load_fineweb_valid()
    print(f"[eval] Validation data: {len(val_data):,} bytes")

    # Determine stride
    stride = args.stride if args.stride > 0 else seq_len

    # Evaluate
    print(f"[eval] Evaluating with seq_len={seq_len}, stride={stride}...")
    results = evaluate_sliding_window(
        model, val_data, seq_len, stride, device, dtype_ctx,
        max_bytes=args.max_eval_bytes,
    )

    print(f"\n{'='*50}")
    print(f"  BPB:              {results['bpb']:.4f}")
    print(f"  Avg NLL:          {results['avg_nll']:.4f}")
    print(f"  Scored tokens:    {results['total_scored_tokens']:,}")
    print(f"  Windows:          {results['n_windows']}")
    print(f"  Stride:           {results['stride']}")
    print(f"  Eval time:        {results['eval_time_seconds']:.1f}s")
    print(f"{'='*50}\n")

    # Save CSV if requested
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results.keys()))
            w.writeheader()
            w.writerow(results)
        print(f"[eval] Results saved to {csv_path}")

    return results


if __name__ == "__main__":
    main()
