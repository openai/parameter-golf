"""
Stage 2b: Score chunks using a neural LM trained on val.

Uses the same GPT architecture as the competition model (9 layers, 512 dim)
trained on the 62M val tokens with AdamW. Then scores every 32K chunk
across all training shards by perplexity under this model.

This captures syntactic structure, topic coherence, and long-range patterns
that bigram models cannot see.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import time

# Import model architecture from train_gpt.py
from train_gpt import GPT, load_data_shard

DATA_DIR = Path("data/datasets/fineweb10B_sp1024")
OUTPUT_DIR = Path("experiments/data_order/stage2_chunk_level")
CHUNK_SIZE = 32768
SEQ_LEN = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Model config (same as competition defaults)
MODEL_CONFIG = dict(
    vocab_size=1024,
    num_layers=9,
    model_dim=512,
    num_heads=8,
    num_kv_heads=4,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.005,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
)

# Training config for proxy model
PROXY_TRAIN_STEPS = 500
PROXY_BATCH_SEQS = 32  # sequences per batch
PROXY_LR = 3e-4


def load_shard_tokens_np(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    assert int(header[0]) == 20240520 and int(header[1]) == 1
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    return np.fromfile(path, dtype="<u2", count=num_tokens, offset=offset)


def train_proxy_on_val(val_tokens, model, steps, batch_seqs, lr, seq_len):
    """Train the model on val tokens with AdamW."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Convert val tokens to tensor
    val_t = torch.from_numpy(val_tokens.astype(np.int64)).to(DEVICE)
    n_tokens = len(val_t)

    for step in range(steps):
        # Random batch of sequences
        starts = torch.randint(0, n_tokens - seq_len - 1, (batch_seqs,))
        x = torch.stack([val_t[s:s+seq_len] for s in starts])
        y = torch.stack([val_t[s+1:s+seq_len+1] for s in starts])

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            print(f"  Proxy train step {step}/{steps}: loss={loss.item():.4f}")

    model.eval()
    return model


@torch.no_grad()
def score_chunk_batch(model, tokens_np, chunk_size, seq_len, batch_size=8):
    """Score all chunks in a shard. Returns array of CE values."""
    n_chunks = len(tokens_np) // chunk_size
    if n_chunks == 0:
        return np.array([])

    ces = []
    for ci in range(n_chunks):
        start = ci * chunk_size
        chunk = tokens_np[start:start + chunk_size]
        chunk_t = torch.from_numpy(chunk.astype(np.int64)).to(DEVICE)

        # Score in sliding windows of seq_len
        n_seqs = (chunk_size - 1) // seq_len
        losses = []
        for si in range(0, n_seqs, batch_size):
            batch_end = min(si + batch_size, n_seqs)
            xs = []
            ys = []
            for bi in range(si, batch_end):
                s = bi * seq_len
                xs.append(chunk_t[s:s+seq_len])
                ys.append(chunk_t[s+1:s+seq_len+1])
            x = torch.stack(xs)
            y = torch.stack(ys)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            losses.append(loss.item())
        ces.append(float(np.mean(losses)))

    return np.array(ces)


if __name__ == "__main__":
    t_start = time.time()

    val_path = DATA_DIR / "fineweb_val_000000.bin"
    shard_paths = sorted(DATA_DIR.glob("fineweb_train_*.bin"))
    print(f"Shards: {len(shard_paths)}, chunk_size: {CHUNK_SIZE}")
    print(f"Device: {DEVICE}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load val
    val_tokens = load_shard_tokens_np(val_path)
    print(f"Val: {len(val_tokens):,} tokens")

    # Build and train proxy model
    print(f"\nTraining proxy model ({PROXY_TRAIN_STEPS} steps, lr={PROXY_LR})...")
    model = GPT(**MODEL_CONFIG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    t_train = time.time()
    model = train_proxy_on_val(val_tokens, model, PROXY_TRAIN_STEPS, PROXY_BATCH_SEQS, PROXY_LR, SEQ_LEN)
    print(f"  Proxy training done in {time.time()-t_train:.1f}s")

    # Eval proxy on val (sanity check)
    val_t = torch.from_numpy(val_tokens[:SEQ_LEN * 100 + 1].astype(np.int64)).to(DEVICE)
    x_val = val_t[:-1].reshape(-1, SEQ_LEN)[:32]
    y_val = val_t[1:].reshape(-1, SEQ_LEN)[:32]
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        val_loss = model(x_val, y_val).item()
    print(f"  Proxy val loss: {val_loss:.4f}")

    # Score all chunks
    print(f"\nScoring {len(shard_paths)} shards...")
    all_scores = []
    all_ce = []
    per_shard_stats = []

    for shard_idx, path in enumerate(shard_paths):
        tokens = load_shard_tokens_np(path)
        ces = score_chunk_batch(model, tokens, CHUNK_SIZE, SEQ_LEN)

        for chunk_idx, ce in enumerate(ces):
            all_scores.append((shard_idx, chunk_idx, float(ce)))
            all_ce.append(float(ce))

        per_shard_stats.append({
            "shard": path.stem,
            "n_chunks": len(ces),
            "mean_ce": float(ces.mean()),
            "std_ce": float(ces.std()),
            "min_ce": float(ces.min()),
            "max_ce": float(ces.max()),
        })

        if (shard_idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  {shard_idx+1}/{len(shard_paths)} shards, {len(all_ce)} chunks, {elapsed:.0f}s")

    all_ce = np.array(all_ce)
    print(f"\nScored {len(all_ce):,} chunks in {time.time()-t_start:.1f}s")

    # Distribution analysis
    print(f"\n=== Neural LM CE Distribution ===")
    print(f"  Total chunks: {len(all_ce):,}")
    print(f"  Mean:   {all_ce.mean():.4f}")
    print(f"  Std:    {all_ce.std():.4f}")
    print(f"  Min:    {all_ce.min():.4f}")
    print(f"  Max:    {all_ce.max():.4f}")
    print(f"  Range:  {all_ce.max() - all_ce.min():.4f}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:2d}:    {np.percentile(all_ce, p):.4f}")

    # Within vs between shard variance
    shard_means = np.array([s["mean_ce"] for s in per_shard_stats])
    between_var = shard_means.var()
    within_var = np.mean([s["std_ce"]**2 for s in per_shard_stats])
    print(f"\n  Between-shard variance: {between_var:.6f}")
    print(f"  Within-shard variance:  {within_var:.6f}")
    print(f"  Ratio (within/between): {within_var/between_var:.1f}x")

    # Compare with bigram scores if available
    bigram_path = OUTPUT_DIR / "chunk_scores.json"
    if bigram_path.exists():
        with open(bigram_path) as f:
            bi_data = json.load(f)
        bi_chunks = {(c[0], c[1]): c[2] for c in bi_data["selection_1gpu"]["chunks"]}
        # Compute Spearman between neural and bigram
        bi_ces = []
        neural_ces = []
        for s_idx, c_idx, ce in all_scores:
            bi_ce = bi_chunks.get((s_idx, c_idx))
            if bi_ce is not None:
                bi_ces.append(bi_ce)
                neural_ces.append(ce)
        if len(bi_ces) > 100:
            bi_r = np.argsort(np.argsort(bi_ces)).astype(float)
            ne_r = np.argsort(np.argsort(neural_ces)).astype(float)
            d = bi_r - ne_r
            n = len(bi_r)
            spearman = 1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1))
            print(f"\n  Spearman(bigram, neural) on overlap: {spearman:.4f}")

    # Selection
    all_scores.sort(key=lambda x: x[2])
    tokens_1gpu = 1836 * 524288
    chunks_1gpu = tokens_1gpu // CHUNK_SIZE

    selected_ces = np.array([s[2] for s in all_scores[:chunks_1gpu]])
    print(f"\n=== Selection (1×GPU, {chunks_1gpu:,} chunks) ===")
    print(f"  CE range: {selected_ces.min():.4f} – {selected_ces.max():.4f}")
    print(f"  CE mean:  {selected_ces.mean():.4f} (vs full {all_ce.mean():.4f})")
    print(f"  CE gain:  {all_ce.mean() - selected_ces.mean():.4f}")

    # Save
    with open(OUTPUT_DIR / "chunk_scores_neural.json", "w") as f:
        json.dump({
            "chunk_size": CHUNK_SIZE,
            "proxy_train_steps": PROXY_TRAIN_STEPS,
            "proxy_val_loss": round(val_loss, 4),
            "total_chunks": len(all_scores),
            "distribution": {
                "mean": round(float(all_ce.mean()), 4),
                "std": round(float(all_ce.std()), 4),
                "min": round(float(all_ce.min()), 4),
                "max": round(float(all_ce.max()), 4),
                "between_shard_var": round(float(between_var), 6),
                "within_shard_var": round(float(within_var), 6),
            },
            "per_shard_stats": per_shard_stats,
            "selection_1gpu": {
                "n_chunks": chunks_1gpu,
                "ce_range": [round(all_scores[0][2], 4), round(all_scores[chunks_1gpu-1][2], 4)],
                "chunks": [(s, c, round(ce, 4)) for s, c, ce in all_scores[:chunks_1gpu]],
            },
        }, f)

    print(f"\nSaved to {OUTPUT_DIR / 'chunk_scores_neural.json'}")
    print(f"Total time: {time.time()-t_start:.1f}s")
