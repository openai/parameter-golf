"""
Score each training shard by model perplexity using a simple approach.

1. Score all shards with random model (inherent difficulty baseline)
2. Train 500 steps on shard 0
3. Score all shards again (what's still hard after partial training)
4. Rank by remaining loss

Usage (single GPU is fine):
    python3 analysis/score_shards_simple.py --data-dir ./data/datasets/fineweb10B_sp1024
"""

import argparse
import glob
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class MiniGPT(nn.Module):
    """Minimal GPT for shard scoring. Same architecture shape as competition model."""
    def __init__(self, vocab=1024, dim=512, layers=6, heads=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*3,
                                       batch_first=True, norm_first=True, dropout=0.0)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # tie embeddings

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for block in self.blocks:
            h = block(h, src_mask=mask, is_causal=True)
        return self.head(self.norm(h))


def load_shard(path, vocab_size=1024):
    tokens = np.fromfile(path, dtype=np.uint16).astype(np.int64)
    tokens = np.clip(tokens, 0, vocab_size - 1)
    return torch.from_numpy(tokens)


def score_shard(model, tokens, device, seq_len=1024, max_batches=50, batch_size=16):
    model.eval()
    n_seqs = len(tokens) // (seq_len + 1)
    if n_seqs == 0:
        return float('inf')

    step = max(1, n_seqs // (max_batches * batch_size))
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for bi in range(0, min(n_seqs, max_batches * batch_size), batch_size):
            batch_starts = [((bi + b) * step) * (seq_len + 1) for b in range(batch_size)
                           if (bi + b) * step < n_seqs]
            if not batch_starts:
                break
            x = torch.stack([tokens[s:s+seq_len].to(device) for s in batch_starts])
            y = torch.stack([tokens[s+1:s+seq_len+1].to(device) for s in batch_starts])
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   y.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()

    return total_loss / max(total_tokens, 1)


def train_steps(model, tokens, device, steps=500, seq_len=1024, batch_size=16, lr=0.001):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_seqs = len(tokens) // (seq_len + 1)

    for step in range(steps):
        idx = (step * batch_size) % n_seqs
        batch_starts = [(idx + b) * (seq_len + 1) for b in range(batch_size) if idx + b < n_seqs]
        if not batch_starts:
            continue
        x = torch.stack([tokens[s:s+seq_len].to(device) for s in batch_starts])
        y = torch.stack([tokens[s+1:s+seq_len+1].to(device) for s in batch_starts])

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"  Train step {step+1}/{steps}, loss: {loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_files = sorted(glob.glob(str(Path(args.data_dir) / "fineweb_train_*.bin")))
    val_files = sorted(glob.glob(str(Path(args.data_dir) / "fineweb_val_*.bin")))
    print(f"Found {len(train_files)} train shards, {len(val_files)} val shards")

    model = MiniGPT(vocab=1024, dim=512, layers=6, heads=8).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: Score with random model
    print(f"\n{'='*60}")
    print("PHASE 1: Random model scoring")
    print(f"{'='*60}")
    random_scores = {}
    for i, f in enumerate(train_files):
        tokens = load_shard(f)
        loss = score_shard(model, tokens, device)
        random_scores[i] = loss
        if (i + 1) % 10 == 0 or i == len(train_files) - 1:
            print(f"  [{i+1}/{len(train_files)}] shard {i}: loss={loss:.4f}")

    val_tokens = torch.cat([load_shard(f) for f in val_files])
    val_random = score_shard(model, val_tokens, device)
    print(f"  Val loss (random): {val_random:.4f}")

    # Phase 2: Train on shard 0
    print(f"\n{'='*60}")
    print(f"PHASE 2: Training {args.train_steps} steps on shard 0")
    print(f"{'='*60}")
    train_tokens = load_shard(train_files[0])
    train_steps(model, train_tokens, device, steps=args.train_steps)

    # Phase 3: Score with trained model
    print(f"\n{'='*60}")
    print("PHASE 3: Trained model scoring")
    print(f"{'='*60}")
    trained_scores = {}
    for i, f in enumerate(train_files):
        tokens = load_shard(f)
        loss = score_shard(model, tokens, device)
        trained_scores[i] = loss
        if (i + 1) % 10 == 0 or i == len(train_files) - 1:
            print(f"  [{i+1}/{len(train_files)}] shard {i}: loss={loss:.4f}")

    val_trained = score_shard(model, val_tokens, device)
    print(f"  Val loss (trained): {val_trained:.4f}")

    # Results
    print(f"\n{'='*60}")
    print("RESULTS: Shard ranking by remaining loss (highest = most to learn)")
    print(f"{'='*60}")
    print(f"{'Rank':>4} {'Shard':>6} {'Random':>10} {'Trained':>10} {'Remaining':>10} {'Learned':>10}")
    print("-" * 60)

    shards = [(i, random_scores[i], trained_scores[i],
               trained_scores[i], random_scores[i] - trained_scores[i])
              for i in range(len(train_files))]
    shards.sort(key=lambda x: -x[3])  # sort by remaining loss descending

    for rank, (idx, rand, trained, remaining, learned) in enumerate(shards):
        print(f"{rank+1:>4} {idx:>6} {rand:>10.4f} {trained:>10.4f} {remaining:>10.4f} {learned:>10.4f}")

    # Key metrics
    losses = [s[3] for s in shards]
    loss_range = max(losses) - min(losses)
    loss_std = np.std(losses)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Remaining loss range: {min(losses):.4f} — {max(losses):.4f} (range: {loss_range:.4f})")
    print(f"Remaining loss std:   {loss_std:.4f}")
    print(f"Val loss:             {val_trained:.4f}")
    print(f"")
    if loss_range < 0.01:
        print("VERDICT: Small range — shards are similar even at model-perplexity level.")
        print("Shard reordering unlikely to help significantly.")
    else:
        print(f"VERDICT: Range of {loss_range:.4f} — meaningful variation!")
        print("Shard reordering could improve BPB.")

    recommended = [s[0] for s in shards]  # already sorted by remaining loss desc
    print(f"\nRecommended order (hardest first): {recommended[:20]}...")
    print(f"Skip (easiest):                    {recommended[-10:]}")


if __name__ == "__main__":
    main()
