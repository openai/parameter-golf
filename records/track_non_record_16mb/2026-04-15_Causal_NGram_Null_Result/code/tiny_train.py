"""
Tiny local training + eval pipeline.

Trains a small sp1024 LM on a fraction of the val shard (we don't have train
shards locally — downloading them would be ~8GB), then evaluates BPB with and
without a causal n-gram additive contribution on a held-out slice.

This is a SANITY MEASUREMENT, not a real competition run. Absolute BPB will be
much worse than the 1.08 competition SOTA because (1) the model is tiny, (2)
we're training on val data (cheating absolute but fine for relative delta),
(3) only a few hundred steps.

What it tells us: whether the n-gram additive contribution gives a POSITIVE
delta when stacked on a trained neural model, and how much. This is the last
cheap signal we can get without spending on a pod.

Device: MPS if available, else CPU. Seq_len 256, batch 16, 2L 128d model.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ngram_eval import CausalNGram


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 128, n_layers: int = 2,
                 n_heads: int = 4, seq_len: int = 256, mlp_mult: int = 4):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=n_heads, dim_feedforward=dim * mlp_mult,
                batch_first=True, dropout=0.0, activation='gelu',
                norm_first=True,
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Tie input+output embeddings for efficiency
        self.head.weight = self.tok_emb.weight

    def forward_logits(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device),
                           diagonal=1)
        for blk in self.blocks:
            x = blk(x, src_mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1), reduction='mean'
        )


# -----------------------------------------------------------------------------
# Data: we split the val shard into a TRAIN portion (first 80%) and a HELDOUT
# portion (last 20%) for eval. Training on part of val is a cheat for absolute
# numbers but fine for RELATIVE measurement (with vs without n-gram, same
# trained model, same held-out eval).
# -----------------------------------------------------------------------------

def load_tokens(path: Path) -> np.ndarray:
    header_bytes = 256 * 4
    return np.fromfile(path, dtype='<u2', offset=header_bytes)


def iter_batches(tokens: np.ndarray, seq_len: int, batch_size: int,
                  n_batches: int, rng: np.random.Generator):
    """Sample random windows from tokens. Yields (x, y) numpy arrays."""
    for _ in range(n_batches):
        starts = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
        x = np.stack([tokens[s:s + seq_len] for s in starts]).astype(np.int64)
        y = np.stack([tokens[s + 1:s + seq_len + 1] for s in starts]).astype(np.int64)
        yield x, y


# -----------------------------------------------------------------------------
# Eval: sliding-window with/without n-gram, on held-out tokens
# -----------------------------------------------------------------------------

def eval_sliding(model, tokens: np.ndarray, vocab_size: int, seq_len: int,
                 stride: int, device, alpha: float, ngram_order: int,
                 batch_seqs: int = 16, chunk_tokens: int = 4096,
                 ngram_enabled: bool = True):
    """Sliding-window eval with optional causal n-gram additive logit blend.
    Returns mean NLL (nats) across scored tokens (bits-per-TOKEN, not per byte).

    The n-gram is a CausalNGram from ngram_eval. We follow the same
    freeze/update-after-score pattern as eval_val_ttt_with_ngram.
    """
    model.eval()
    context_size = seq_len - stride
    total_tokens = len(tokens) - 1

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]
    num_chunks = max(1, (total_tokens + chunk_tokens - 1) // chunk_tokens)
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)

    if ngram_enabled:
        ng = CausalNGram(vocab_size=vocab_size, order=ngram_order, delta=0.5,
                          min_context_count=2)
        ng.freeze()
    else:
        ng = None

    nll_sum = 0.0
    n_scored = 0

    tokens_tensor = torch.from_numpy(tokens).long()

    with torch.no_grad():
        for ci in range(num_chunks):
            windows = chunk_windows[ci]
            if not windows:
                continue
            chunk_scored_positions = []
            for bi in range(0, len(windows), batch_seqs):
                batch_ws = windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk_tok = tokens_tensor[ws:we + 1]
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model.forward_logits(x_batch)
                if ngram_enabled and alpha != 0.0:
                    ngram_log_p = ng.batch_log_probs_torch(x_batch).to(logits.dtype)
                    blended = logits + alpha * ngram_log_p
                else:
                    blended = logits

                nll = F.cross_entropy(
                    blended.reshape(-1, blended.size(-1)).float(),
                    y_batch.reshape(-1), reduction='none'
                ).reshape(bsz, seq_len)

                # Move nll to CPU (float64) for stable accumulation — MPS
                # doesn't support float64 so we accumulate on CPU.
                nll_cpu = nll.detach().cpu().to(torch.float64)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored = nll_cpu[i, s:wlen]
                    nll_sum += float(scored.sum().item())
                    n_scored += (wlen - s)
                    if ngram_enabled:
                        toks = y_batch[i, s:wlen].cpu().numpy().astype(np.int64)
                        ctx_prefix = x_batch[i, :s].cpu().numpy().astype(np.int64)
                        chunk_scored_positions.append((int(ws + s), ctx_prefix, toks))

            if ngram_enabled:
                chunk_scored_positions.sort(key=lambda t: t[0])
                for gpos, ctx_prefix, toks in chunk_scored_positions:
                    running = list(int(x) for x in ctx_prefix[-(ng.K - 1):]) if ng.K > 1 else []
                    for tok in toks:
                        ng.add_token(tuple(running), int(tok))
                        if ng.K > 1:
                            running.append(int(tok))
                            if len(running) > ng.K - 1:
                                running = running[-(ng.K - 1):]
                ng.freeze()

    mean_nll = nll_sum / max(n_scored, 1)
    return {
        "nll_sum": nll_sum,
        "n_scored": n_scored,
        "mean_nll_nats": mean_nll,
        "bits_per_tok": mean_nll / math.log(2),
        "unique_ctx": ng.unique_contexts() if ngram_enabled else None,
    }


# -----------------------------------------------------------------------------
# Main: train then eval
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path,
                    default=Path("data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"))
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-stride", type=int, default=64)
    ap.add_argument("--eval-chunk-tokens", type=int, default=8192)
    ap.add_argument("--held-out-frac", type=float, default=0.2,
                    help="Fraction of val shard reserved for eval")
    ap.add_argument("--train-cap", type=int, default=4_000_000,
                    help="Cap tokens used for training (for speed)")
    ap.add_argument("--eval-cap", type=int, default=200_000,
                    help="Cap tokens used for eval")
    ap.add_argument("--orders", type=str, default="3,4,5")
    ap.add_argument("--alphas", type=str, default="0,0.1,0.2,0.3,0.5,0.7,1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results_tiny_train.json"))
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="Override vocab size (auto-detected from val path if not set)")
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}", file=sys.stderr)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.val}...", file=sys.stderr)
    tokens = load_tokens(args.val)
    print(f"  {len(tokens):,} tokens", file=sys.stderr)

    # Determine vocab size: CLI override > auto-detect from path > default 1024
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
    else:
        path_str = str(args.val)
        if "sp8192" in path_str:
            vocab_size = 8192
        elif "sp4096" in path_str:
            vocab_size = 4096
        elif "sp1024" in path_str:
            vocab_size = 1024
        else:
            vocab_size = 1024
    print(f"  vocab_size: {vocab_size}", file=sys.stderr)
    split = int(len(tokens) * (1 - args.held_out_frac))
    train_tokens = tokens[:split][:args.train_cap]
    eval_tokens = tokens[split:split + args.eval_cap]
    print(f"  train: {len(train_tokens):,}  eval: {len(eval_tokens):,}",
          file=sys.stderr)

    model = TinyGPT(vocab_size=vocab_size, dim=args.dim, n_layers=args.layers,
                    n_heads=args.heads, seq_len=args.seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params", file=sys.stderr)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             betas=(0.9, 0.95), weight_decay=0.01)

    # Training loop
    model.train()
    t0 = time.time()
    last_loss = None
    for step, (x_np, y_np) in enumerate(
        iter_batches(train_tokens, args.seq_len, args.batch, args.steps, rng)
    ):
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = loss.item()
        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  step {step + 1}/{args.steps}  loss={last_loss:.4f}  "
                  f"({elapsed:.0f}s, {(step + 1) / elapsed:.1f} steps/s)",
                  file=sys.stderr)

    train_time = time.time() - t0
    print(f"Training done: {train_time:.0f}s, final loss {last_loss:.4f}",
          file=sys.stderr)

    # Eval sweep: for each order in --orders and each alpha in --alphas, run
    # the eval and record BPT. Start with alpha=0 baseline for the reference.
    orders = [int(x) for x in args.orders.split(",")]
    alphas = sorted({float(x) for x in args.alphas.split(",")})

    results = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "device": str(device),
        "n_params": n_params,
        "train_tokens": int(len(train_tokens)),
        "eval_tokens": int(len(eval_tokens)),
        "train_time_s": train_time,
        "final_train_loss": last_loss,
        "baseline": None,
        "runs": [],
    }

    print("\n--- EVAL SWEEP ---", file=sys.stderr)
    # Baseline (no n-gram)
    t0 = time.time()
    base = eval_sliding(model, eval_tokens, vocab_size, args.seq_len,
                         args.eval_stride, device, alpha=0.0, ngram_order=3,
                         ngram_enabled=False,
                         chunk_tokens=args.eval_chunk_tokens)
    base_time = time.time() - t0
    base["eval_time_s"] = base_time
    results["baseline"] = base
    print(f"  BASELINE (no ngram): bpt={base['bits_per_tok']:.5f} "
          f"({base_time:.0f}s)", file=sys.stderr)

    for order in orders:
        for alpha in alphas:
            t0 = time.time()
            res = eval_sliding(model, eval_tokens, vocab_size, args.seq_len,
                                args.eval_stride, device, alpha=alpha,
                                ngram_order=order, ngram_enabled=True,
                                chunk_tokens=args.eval_chunk_tokens)
            et = time.time() - t0
            delta = res['bits_per_tok'] - base['bits_per_tok']
            res["eval_time_s"] = et
            res["order"] = order
            res["alpha"] = alpha
            res["delta_vs_baseline_bpt"] = delta
            results["runs"].append(res)
            print(f"  order={order} alpha={alpha:.2f}  "
                  f"bpt={res['bits_per_tok']:.5f}  delta={delta:+.5f}  "
                  f"({et:.0f}s)", file=sys.stderr)

    # Write summary
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.out}", file=sys.stderr)

    # Print a compact table
    print("\n=== SUMMARY ===")
    print(f"{'order':>5} {'alpha':>6} {'bits/tok':>10} {'delta':>10}")
    print(f"{'base':>5} {'---':>6} {base['bits_per_tok']:>10.5f} {0.0:>+10.5f}")
    for r in results["runs"]:
        print(f"{r['order']:>5} {r['alpha']:>6.2f} {r['bits_per_tok']:>10.5f} "
              f"{r['delta_vs_baseline_bpt']:>+10.5f}")


if __name__ == "__main__":
    main()
