#!/usr/bin/env python3
"""Score-First Test-Time Training (TTT) for Parameter Golf.

Protocol:
1. Start with base model weights
2. For each chunk of eval tokens:
   a. Adapt: run K SGD steps on history (already-scored tokens)
   b. Score: compute loss on current chunk (this is the real eval)
   c. Append chunk to history
   d. Optionally reset weights to base

Usage:
    python eval/ttt.py --checkpoint path/to/model.pt --sgd-steps 4 --lr 5e-3
"""

import argparse
import copy
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.evaluate import build_model_from_ckpt
from utils.data import load_fineweb_valid


@dataclass
class TTTConfig:
    sgd_steps: int = 4           # adaptation steps per chunk
    lr: float = 5e-3             # SGD learning rate for adaptation
    weight_decay: float = 0.0
    adapt_on: str = "ln_only"    # "all" | "ln_only" | "head_only"
    reset_after_chunk: bool = False  # reset to base weights after each chunk
    chunk_size: int = 64         # tokens per eval chunk
    train_window: int = 512      # max history tokens for adaptation
    device: str = "cuda"


class ScoreFirstTTT:
    def __init__(self, model, cfg: TTTConfig):
        self.model = model
        self.cfg = cfg
        self.base_state = copy.deepcopy(model.state_dict())

    def get_adapt_params(self):
        """Select which parameters to adapt."""
        if self.cfg.adapt_on == "all":
            return [p for p in self.model.parameters() if p.requires_grad]
        elif self.cfg.adapt_on == "ln_only":
            return [p for n, p in self.model.named_parameters()
                    if "ln" in n or "norm" in n]
        elif self.cfg.adapt_on == "head_only":
            return [p for n, p in self.model.named_parameters()
                    if "head" in n]
        else:
            raise ValueError(f"Unknown adapt_on: {self.cfg.adapt_on}")

    def restore_base_weights(self):
        self.model.load_state_dict(self.base_state)

    @torch.no_grad()
    def score_chunk(self, tokens, start, end):
        """Score tokens[start:end] using model, return total NLL."""
        self.model.eval()
        ctx_start = max(0, start - self.cfg.train_window)
        seq = tokens[ctx_start:end].unsqueeze(0)
        logits, _ = self.model(seq[:, :-1])
        targets = seq[:, 1:]
        # Only score the chunk portion
        chunk_len = end - start
        chunk_offset = seq.size(1) - 1 - chunk_len
        chunk_logits = logits[:, chunk_offset:, :]
        chunk_targets = targets[:, chunk_offset:]
        loss = F.cross_entropy(chunk_logits.reshape(-1, chunk_logits.size(-1)),
                               chunk_targets.reshape(-1), reduction='sum')
        n_scored = chunk_targets.numel()
        return float(loss), n_scored

    def adapt_on_history(self, tokens, history_end):
        """Run K SGD steps on tokens[:history_end]."""
        self.model.train()
        params = self.get_adapt_params()
        if not params:
            return []

        opt = torch.optim.SGD(params, lr=self.cfg.lr,
                              weight_decay=self.cfg.weight_decay)

        start = max(0, history_end - self.cfg.train_window)
        seq = tokens[start:history_end].unsqueeze(0)

        if seq.size(1) < 2:
            return []

        losses = []
        for _ in range(self.cfg.sgd_steps):
            opt.zero_grad()
            logits, _ = self.model(seq[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   seq[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            losses.append(float(loss))

        return losses

    def evaluate(self, tokens, max_bytes: int = 0, verbose: bool = True):
        """Full score-first TTT evaluation."""
        self.restore_base_weights()
        if max_bytes > 0:
            tokens = tokens[:max_bytes]
        T = len(tokens)

        total_nll = 0.0
        total_scored = 0
        history_end = self.cfg.chunk_size  # skip first chunk (no history)

        # Score first chunk without adaptation
        nll, n = self.score_chunk(tokens, 0, history_end)
        total_nll += nll
        total_scored += n

        chunks_done = 1
        start_time = time.time()

        while history_end < T:
            chunk_end = min(history_end + self.cfg.chunk_size, T)

            # Adapt on history
            self.adapt_on_history(tokens, history_end)

            # Score new chunk
            nll, n = self.score_chunk(tokens, history_end, chunk_end)
            total_nll += nll
            total_scored += n

            history_end = chunk_end
            chunks_done += 1

            if self.cfg.reset_after_chunk:
                self.restore_base_weights()

            if verbose and chunks_done % 50 == 0:
                elapsed = time.time() - start_time
                running_bpb = (total_nll / max(total_scored, 1)) / math.log(2)
                print(f"  [ttt] chunk {chunks_done} | scored {total_scored:,} | "
                      f"bpb {running_bpb:.4f} | {elapsed:.0f}s")

        avg_nll = total_nll / max(total_scored, 1)
        bpb = avg_nll / math.log(2)
        elapsed = time.time() - start_time
        return {
            "bpb": bpb,
            "avg_nll": avg_nll,
            "tokens_scored": total_scored,
            "chunks": chunks_done,
            "eval_time": elapsed,
        }


def parse_args():
    p = argparse.ArgumentParser(description="Score-First TTT Evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sgd-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--adapt-on", choices=["all", "ln_only", "head_only"],
                   default="ln_only")
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--train-window", type=int, default=512)
    p.add_argument("--reset-after-chunk", action="store_true")
    p.add_argument("--max-eval-bytes", type=int, default=0,
                   help="Max bytes to evaluate (0 = all)")
    p.add_argument("--weight-decay", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ttt] Device: {device}")

    # Load checkpoint
    print(f"[ttt] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, device)
    ckpt_args = ckpt["args"]
    print(f"[ttt] Model: {ckpt_args['model_type']}, d={ckpt_args['d_model']}")
    print(f"[ttt] Trained for {ckpt.get('step', '?')} steps, "
          f"train val_bpb={ckpt.get('val_bpb', '?'):.4f}")

    # Load validation data
    print("[ttt] Loading validation data...")
    val_data = load_fineweb_valid()
    tokens = torch.from_numpy(val_data.copy()).long().to(device)
    print(f"[ttt] Validation data: {len(tokens):,} bytes")

    # Configure TTT
    cfg = TTTConfig(
        sgd_steps=args.sgd_steps,
        lr=args.lr,
        adapt_on=args.adapt_on,
        chunk_size=args.chunk_size,
        train_window=args.train_window,
        reset_after_chunk=args.reset_after_chunk,
        weight_decay=args.weight_decay,
        device=str(device),
    )

    print(f"\n[ttt] Config: sgd_steps={cfg.sgd_steps}, lr={cfg.lr}, "
          f"adapt_on={cfg.adapt_on}, chunk={cfg.chunk_size}, "
          f"window={cfg.train_window}")

    # Run TTT evaluation
    ttt = ScoreFirstTTT(model, cfg)
    results = ttt.evaluate(tokens, max_bytes=args.max_eval_bytes)

    print(f"\n{'='*50}")
    print(f"  TTT Results")
    print(f"  BPB:            {results['bpb']:.4f}")
    print(f"  Avg NLL:        {results['avg_nll']:.4f}")
    print(f"  Tokens scored:  {results['tokens_scored']:,}")
    print(f"  Chunks:         {results['chunks']}")
    print(f"  Eval time:      {results['eval_time']:.1f}s")
    print(f"{'='*50}\n")

    # Compare with base model (no TTT)
    print("[ttt] Evaluating base model without TTT for comparison...")
    ttt.restore_base_weights()
    base_nll, base_n = 0.0, 0
    model.eval()
    chunk = cfg.chunk_size
    with torch.no_grad():
        for i in range(0, min(len(tokens), args.max_eval_bytes or len(tokens)) - chunk, chunk):
            nll, n = ttt.score_chunk(tokens, i, i + chunk)
            base_nll += nll
            base_n += n
    if base_n > 0:
        base_bpb = (base_nll / base_n) / math.log(2)
        print(f"  Base BPB (no TTT): {base_bpb:.4f}")
        print(f"  TTT improvement:   {base_bpb - results['bpb']:.4f} BPB")

    return results


if __name__ == "__main__":
    main()
