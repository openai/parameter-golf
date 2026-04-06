#!/usr/bin/env python3
"""Multi-order N-gram Backoff Cache for byte-level LM eval.

Mixes neural model predictions with cached n-gram statistics.
Strictly causal — only uses previously seen bytes.

Usage as library:
    cache = NGramBackoffCache(NGramCacheConfig())
    for byte in history:
        cache.update(byte)
    mixed_logprobs = cache.mix_logits(neural_logits)

Usage from CLI:
    python eval/ngram_cache.py --checkpoint path/to/model.pt --cache-weight 0.3
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class NGramCacheConfig:
    vocab_size: int = 256
    max_order: int = 5        # use 1-gram through 5-gram
    alpha: float = 0.4        # add-alpha smoothing
    cache_weight: float = 0.3 # interpolation: (1-w)*neural + w*ngram


class NGramBackoffCache:
    def __init__(self, cfg: NGramCacheConfig):
        self.cfg = cfg
        self.history: list = []
        # counts[n][(ctx_tuple)] -> [count_per_byte] array
        self.counts = {n: defaultdict(lambda: [0] * cfg.vocab_size)
                       for n in range(1, cfg.max_order + 1)}

    def update(self, byte_val: int):
        """Add a byte to history and update all n-gram counts."""
        self.history.append(byte_val)
        for n in range(1, self.cfg.max_order + 1):
            if len(self.history) >= n:
                ctx = tuple(self.history[-(n):][:-1])  # context = last n-1 bytes
                self.counts[n][ctx][byte_val] += 1

    def update_batch(self, bytes_seq):
        """Add multiple bytes at once."""
        for b in bytes_seq:
            self.update(int(b))

    def get_ngram_probs(self) -> torch.Tensor:
        """Get interpolated n-gram probability distribution."""
        V = self.cfg.vocab_size
        probs = torch.full((V,), 1.0 / V)

        for n in range(1, self.cfg.max_order + 1):
            if len(self.history) < n - 1:
                continue
            ctx = tuple(self.history[-(n - 1):]) if n > 1 else ()
            counts = self.counts[n].get(ctx)
            if counts is None:
                continue
            total = sum(counts) + self.cfg.alpha * V
            if total > 0:
                ngram_probs = torch.tensor(
                    [(c + self.cfg.alpha) / total for c in counts])
                # Higher-order n-grams get more weight
                w = n / (self.cfg.max_order * 2)
                probs = (1 - w) * probs + w * ngram_probs

        return probs

    def mix_logits(self, neural_logits: torch.Tensor) -> torch.Tensor:
        """Mix neural logits with n-gram cache. Returns log-probs."""
        w = self.cfg.cache_weight
        neural_probs = torch.softmax(neural_logits, dim=-1)
        ngram_probs = self.get_ngram_probs().to(neural_logits.device)
        mixed = (1 - w) * neural_probs + w * ngram_probs
        return torch.log(mixed + 1e-10)

    def reset(self):
        """Clear all history and counts."""
        self.history.clear()
        self.counts = {n: defaultdict(lambda: [0] * self.cfg.vocab_size)
                       for n in range(1, self.cfg.max_order + 1)}


def evaluate_with_ngram(model, tokens, cfg: NGramCacheConfig,
                        seq_len: int = 512, device: str = "cuda",
                        max_bytes: int = 0, verbose: bool = True):
    """Evaluate model + ngram cache mixture on byte sequence."""
    if max_bytes > 0:
        tokens = tokens[:max_bytes]

    cache = NGramBackoffCache(cfg)
    model.eval()

    total_nll = 0.0
    total_scored = 0
    start_time = time.time()

    # Process token by token (slow but correct for n-gram cache)
    # For efficiency, process in windows but score per-token with cache
    T = len(tokens)
    pos = 0

    while pos + 1 < T:
        # Build a window for neural model
        win_start = max(0, pos - seq_len + 1)
        win_end = min(pos + 1, T)
        window = tokens[win_start:win_end].unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(window[:, :-1])

        # Score the last position
        neural_logits = logits[0, -1, :]  # logits for predicting tokens[pos]
        target = tokens[pos].item()

        # Mix with n-gram cache
        mixed_logprobs = cache.mix_logits(neural_logits)
        nll = -mixed_logprobs[target].item()
        total_nll += nll
        total_scored += 1

        # Update cache with this byte
        cache.update(target)
        pos += 1

        if verbose and total_scored % 10000 == 0:
            elapsed = time.time() - start_time
            bpb = (total_nll / total_scored) / math.log(2)
            print(f"  [ngram] scored {total_scored:,} | bpb {bpb:.4f} | {elapsed:.0f}s")

    avg_nll = total_nll / max(total_scored, 1)
    bpb = avg_nll / math.log(2)
    elapsed = time.time() - start_time

    return {
        "bpb": bpb,
        "avg_nll": avg_nll,
        "tokens_scored": total_scored,
        "eval_time": elapsed,
        "cache_weight": cfg.cache_weight,
        "max_order": cfg.max_order,
    }


def parse_args():
    p = argparse.ArgumentParser(description="N-gram Cache Evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cache-weight", type=float, default=0.3)
    p.add_argument("--max-order", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--max-eval-bytes", type=int, default=100000,
                   help="Max bytes to evaluate (default 100k, token-by-token is slow)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ngram] Device: {device}")

    from eval.evaluate import build_model_from_ckpt

    print(f"[ngram] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, device)
    ckpt_args = ckpt["args"]
    seq_len = ckpt_args.get("seq_len", ckpt_args.get("max_seq_len", 512))

    print(f"[ngram] Model: {ckpt_args['model_type']}, d={ckpt_args['d_model']}")

    print("[ngram] Loading validation data...")
    from utils.data import load_fineweb_valid
    val_data = load_fineweb_valid()
    tokens = torch.from_numpy(val_data.copy()).long()
    print(f"[ngram] Validation data: {len(tokens):,} bytes")

    cfg = NGramCacheConfig(
        max_order=args.max_order,
        alpha=args.alpha,
        cache_weight=args.cache_weight,
    )

    print(f"\n[ngram] Config: order={cfg.max_order}, alpha={cfg.alpha}, "
          f"weight={cfg.cache_weight}")

    results = evaluate_with_ngram(
        model, tokens, cfg, seq_len=seq_len, device=str(device),
        max_bytes=args.max_eval_bytes,
    )

    print(f"\n{'='*50}")
    print(f"  N-gram Cache Results")
    print(f"  BPB:            {results['bpb']:.4f}")
    print(f"  Tokens scored:  {results['tokens_scored']:,}")
    print(f"  Eval time:      {results['eval_time']:.1f}s")
    print(f"  Cache weight:   {results['cache_weight']}")
    print(f"  Max order:      {results['max_order']}")
    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    main()
