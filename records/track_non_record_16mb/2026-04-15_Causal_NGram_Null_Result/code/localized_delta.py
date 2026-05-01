"""
Localized-delta analysis: break the n-gram BPB improvement down by where
the scored token is in its document (early vs late) and whether the context
was seen recently (<2048 back) or long-range (>2048 back).

This tells us WHERE the gain is coming from. If most of the delta is in the
"long-range" bucket, that's the signal that this technique is bringing in
information the neural model literally cannot see — which is what we want
(and what makes the delta robust at scale).

If most of the delta is in the "short-range" bucket, the delta will likely
vanish on a well-trained model (which already captures short-range via
attention), and we should pivot.

Implementation: a lightweight eval loop that uses a fixed frozen model and
tags each scored position with its doc-position and cache-range-class, then
computes per-bucket BPB with and without the n-gram contribution.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ngram_eval import CausalNGram
from tiny_train import TinyGPT, load_tokens, pick_device


def run_localized_analysis(val_path: Path, held_out_frac: float,
                             eval_cap: int, seq_len: int, stride: int,
                             chunk_tokens: int, dim: int, layers: int,
                             steps: int, batch: int, lr: float,
                             order: int, alpha: float, seed: int):
    """Train a tiny model, then run ONE eval pass with fine-grained per-position
    bucketing. Returns per-bucket nll sums and token counts for both the
    baseline and the n-gram blend."""
    device = pick_device()
    torch.manual_seed(seed)

    tokens = load_tokens(val_path)
    split = int(len(tokens) * (1 - held_out_frac))
    train_tokens = tokens[:split][:4_000_000]
    eval_tokens = tokens[split:split + eval_cap]
    vocab_size = 1024

    # --- Train ---
    model = TinyGPT(vocab_size=vocab_size, dim=dim, n_layers=layers,
                    n_heads=4, seq_len=seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                              weight_decay=0.01)
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(steps):
        starts = rng.integers(0, len(train_tokens) - seq_len - 1, size=batch)
        x = np.stack([train_tokens[s:s + seq_len] for s in starts]).astype(np.int64)
        y = np.stack([train_tokens[s + 1:s + seq_len + 1] for s in starts]).astype(np.int64)
        x_t = torch.from_numpy(x).to(device)
        y_t = torch.from_numpy(y).to(device)
        loss = model(x_t, y_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()

    # --- Eval with per-position bucketing ---
    # Buckets:
    #   Range: "in_window" (cache hit at distance ≤ seq_len-1) vs
    #          "out_of_window" (distance > seq_len-1) vs
    #          "no_hit" (cache miss, context unseen)
    #   Doc position: measured as position_in_doc in {0-2047, 2048-4095, 4096+}
    #
    # For each bucket, accumulate: sum_nll_baseline, sum_nll_blend, count.
    from collections import defaultdict
    buckets = defaultdict(lambda: {"n": 0, "nll_base": 0.0, "nll_blend": 0.0})

    context_size = seq_len - stride
    total_tokens = len(eval_tokens) - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]

    # Track global position -> doc boundaries
    bos = 1  # FineWeb uses BOS as doc separator
    doc_start_of = np.zeros(len(eval_tokens), dtype=np.int32)
    last_start = 0
    for i, t in enumerate(eval_tokens):
        if int(t) == bos:
            last_start = i + 1
        doc_start_of[i] = last_start

    ng = CausalNGram(vocab_size=vocab_size, order=order, delta=0.5,
                      min_context_count=2)
    ng.freeze()

    # We also keep a separate "first-seen distance" map: ctx_tuple -> last_pos
    from collections import defaultdict as _dd
    first_seen = {}  # ctx -> global position of FIRST observation (for "range" classification at scoring)

    tokens_cpu = torch.from_numpy(eval_tokens).long()
    num_chunks = max(1, (total_tokens + chunk_tokens - 1) // chunk_tokens)
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)

    with torch.no_grad():
        for ci in range(num_chunks):
            windows = chunk_windows[ci]
            if not windows:
                continue
            chunk_scored = []
            for bi in range(0, len(windows), batch):
                batch_ws = windows[bi:bi + batch]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk_tok = tokens_cpu[ws:we + 1]
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                x_batch_dev = x_batch.to(device)
                y_batch_dev = y_batch.to(device)

                logits = model.forward_logits(x_batch_dev)
                ngram_log_p = ng.batch_log_probs_torch(x_batch_dev).to(logits.dtype)
                blended = logits + alpha * ngram_log_p

                nll_base = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch_dev.reshape(-1), reduction='none'
                ).reshape(bsz, seq_len).detach().cpu().to(torch.float64)
                nll_blend = F.cross_entropy(
                    blended.reshape(-1, blended.size(-1)).float(),
                    y_batch_dev.reshape(-1), reduction='none'
                ).reshape(bsz, seq_len).detach().cpu().to(torch.float64)

                x_batch_np = x_batch.numpy().astype(np.int64)
                y_batch_np = y_batch.numpy().astype(np.int64)

                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    for t in range(s, wlen):
                        gpos = ws + t  # global position
                        # n-gram context for predicting y[t] is x[t-K+2:t+1]
                        ctx_start = max(0, t - (order - 1) + 1)
                        ctx_tail = tuple(int(x) for x in x_batch_np[i, ctx_start:t + 1])

                        # Range class: how far back was this context first seen?
                        if ctx_tail not in first_seen:
                            range_cls = "no_hit"
                        else:
                            dist = gpos - first_seen[ctx_tail]
                            if dist <= seq_len:
                                range_cls = "in_window"
                            else:
                                range_cls = "out_of_window"

                        # Doc-position class
                        doc_start = int(doc_start_of[gpos])
                        pos_in_doc = gpos - doc_start
                        if pos_in_doc < 2048:
                            dp_cls = "0-2047"
                        elif pos_in_doc < 4096:
                            dp_cls = "2048-4095"
                        else:
                            dp_cls = "4096+"

                        key = (range_cls, dp_cls)
                        b = buckets[key]
                        b["n"] += 1
                        b["nll_base"] += float(nll_base[i, t])
                        b["nll_blend"] += float(nll_blend[i, t])

                        # Record for post-scoring update
                        chunk_scored.append((gpos, ctx_tail, int(y_batch_np[i, t])))

            # Update n-gram AND first_seen after scoring the chunk
            chunk_scored.sort()
            for gpos, ctx_tail, tok in chunk_scored:
                # Update first_seen: we record the position where this context
                # was *already seen* before — which is any prior observation.
                # For simplicity we use the position where we STORE the context
                # (i.e., when we add this tok with THIS context).
                if ctx_tail not in first_seen:
                    first_seen[ctx_tail] = gpos
                # Update the n-gram
                running = list(ctx_tail)
                ng.add_token(tuple(running), tok)
            ng.freeze()

    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path,
                    default=Path("data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"))
    ap.add_argument("--eval-cap", type=int, default=80_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--chunk-tokens", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--held-out-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results_localized_delta.json"))
    args = ap.parse_args()

    print(f"Training (dim={args.dim}, layers={args.layers}, steps={args.steps}) then "
          f"running localized analysis @ order={args.order}, alpha={args.alpha}",
          file=sys.stderr)
    buckets = run_localized_analysis(
        val_path=args.val,
        held_out_frac=args.held_out_frac,
        eval_cap=args.eval_cap,
        seq_len=args.seq_len,
        stride=args.stride,
        chunk_tokens=args.chunk_tokens,
        dim=args.dim,
        layers=args.layers,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        order=args.order,
        alpha=args.alpha,
        seed=args.seed,
    )

    # Compute per-bucket deltas and totals
    print(f"\n{'range':>15} {'doc_pos':>10} {'N':>8} {'bpt_base':>10} {'bpt_blend':>11} {'delta':>10} {'delta_x_N':>12}")
    print("-" * 80)

    total_nll_base = 0.0
    total_nll_blend = 0.0
    total_n = 0

    # Sort by range then doc_pos
    for key in sorted(buckets.keys()):
        b = buckets[key]
        n = b["n"]
        if n == 0:
            continue
        bpt_base = b["nll_base"] / n / math.log(2)
        bpt_blend = b["nll_blend"] / n / math.log(2)
        delta = bpt_blend - bpt_base
        delta_weighted = delta * n
        print(f"{key[0]:>15} {key[1]:>10} {n:>8} "
              f"{bpt_base:>10.5f} {bpt_blend:>11.5f} {delta:>+10.5f} "
              f"{delta_weighted:>+12.2f}")
        total_nll_base += b["nll_base"]
        total_nll_blend += b["nll_blend"]
        total_n += n

    print("-" * 80)
    overall_bpt_base = total_nll_base / total_n / math.log(2)
    overall_bpt_blend = total_nll_blend / total_n / math.log(2)
    overall_delta = overall_bpt_blend - overall_bpt_base
    print(f"{'OVERALL':>15} {'':>10} {total_n:>8} {overall_bpt_base:>10.5f} "
          f"{overall_bpt_blend:>11.5f} {overall_delta:>+10.5f}")

    result = {
        "overall": {
            "n": total_n,
            "bpt_base": overall_bpt_base,
            "bpt_blend": overall_bpt_blend,
            "delta": overall_delta,
        },
        "buckets": {f"{k[0]}__{k[1]}": {
            "n": v["n"],
            "nll_base": v["nll_base"],
            "nll_blend": v["nll_blend"],
            "bpt_base": v["nll_base"] / v["n"] / math.log(2) if v["n"] else 0,
            "bpt_blend": v["nll_blend"] / v["n"] / math.log(2) if v["n"] else 0,
        } for k, v in buckets.items()},
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
