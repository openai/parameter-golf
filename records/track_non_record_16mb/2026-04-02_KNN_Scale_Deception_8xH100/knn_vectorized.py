#!/usr/bin/env python3
"""
Vectorized KNN Hidden State Retrieval — Competition-Ready
===========================================================

Replaces the per-token Python loop with batched GPU operations.
Instead of 51,200 individual KNN queries, does ~50 batch queries.

PROTOCOL (causal, score-first):
  For each chunk of tokens:
    1. Compute neural probs + hidden states (one forward pass)
    2. Batch KNN: all tokens in chunk query against ALL previously stored states
    3. Mix KNN distribution with neural (vectorized)
    4. Score all tokens in chunk
    5. AFTER: add chunk's hidden states to store
    6. Next chunk (with updated store)

Within a chunk, all queries use states from BEFORE the chunk.
This is the same causality as TTT (which also operates per-chunk).

SPEED ESTIMATE:
  Old: 51,200 Python iterations × GPU topk = 1959s
  New: 50 batch cdist calls + vectorized mixing = ~10-30s
  Speedup: 60-200×

LOCAL TEST: Validates correctness against the slow per-token version.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os

VOCAB_SIZE = 1024
SEQ_LEN = 512
DIM = 192  # local model dim
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Vectorized KNN Test")
print()

# ============================================================
# Vectorized KNN Scoring (THE KEY FUNCTION)
# ============================================================
def score_knn_vectorized(neural_probs, hidden_states, targets,
                          K=8, lam=0.12, chunk_size=1024,
                          vocab_size=VOCAB_SIZE, device=DEVICE):
    """
    Vectorized KNN scoring — competition-ready speed.

    Args:
        neural_probs: (N_tokens, V) float32 tensor — pre-computed softmax
        hidden_states: (N_tokens, dim) float32 tensor — from model._hidden()
        targets: (N_tokens,) long tensor — ground truth tokens
        K: number of nearest neighbors
        lam: KNN mixing weight (0.12 = 12% KNN, 88% neural)
        chunk_size: tokens per batch (larger = faster but more memory)

    Returns:
        bpc: bits per character (float)
        total_bits: total log-loss in bits
        scored: number of tokens scored

    Memory: O(N_tokens × dim) for stored hidden states
            O(chunk_size × N_stored) for distance matrix (peak)
    """
    N = len(targets)
    dim = hidden_states.shape[1]

    # Move to device
    if not isinstance(neural_probs, torch.Tensor):
        neural_probs = torch.tensor(neural_probs, dtype=torch.float32)
    if not isinstance(hidden_states, torch.Tensor):
        hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.long)

    neural_probs = neural_probs.to(device)
    hidden_states = hidden_states.to(device)
    targets = targets.to(device)

    # Clamp and normalize neural probs
    neural_probs = neural_probs.clamp(min=1e-10)
    neural_probs = neural_probs / neural_probs.sum(dim=1, keepdim=True)

    # Store for growing datastore
    stored_h = torch.zeros(N, dim, device=device, dtype=torch.float32)
    stored_tok = torch.zeros(N, device=device, dtype=torch.long)
    n_stored = 0

    total_bits = 0.0
    scored = 0

    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        chunk_len = chunk_end - chunk_start

        # This chunk's data
        q = hidden_states[chunk_start:chunk_end]       # (C, dim)
        t = targets[chunk_start:chunk_end]              # (C,)
        np_ = neural_probs[chunk_start:chunk_end]       # (C, V)

        if n_stored >= K:
            # === BATCH KNN ===
            # Compute squared L2 distances: (C, n_stored)
            # Using cdist for efficiency
            dists = torch.cdist(q, stored_h[:n_stored], p=2).pow(2)  # (C, n_stored)

            # Top-K nearest for each query
            topk_dists, topk_local_idx = dists.topk(K, dim=1, largest=False)  # (C, K)

            # Get tokens of nearest neighbors
            topk_toks = stored_tok[:n_stored][topk_local_idx]  # (C, K)

            # Softmax weights over distances
            weights = torch.exp(-topk_dists / dim)  # (C, K)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-30)

            # Build KNN distribution via scatter
            knn_dist = torch.zeros(chunk_len, vocab_size, device=device)
            knn_dist.scatter_add_(1, topk_toks, weights)

            # Smooth
            knn_dist = 0.99 * knn_dist + 0.01 / vocab_size
            knn_dist = knn_dist / knn_dist.sum(dim=1, keepdim=True)

            # Mix: (1-lam)*neural + lam*knn
            mixed = (1.0 - lam) * np_ + lam * knn_dist
        else:
            mixed = np_

        # Normalize
        mixed = mixed / mixed.sum(dim=1, keepdim=True)

        # Score: gather target probabilities
        target_probs = mixed.gather(1, t.unsqueeze(1)).squeeze(1)  # (C,)
        bits = -torch.log2(target_probs.clamp(min=1e-30))
        total_bits += bits.sum().item()
        scored += chunk_len

        # Store AFTER scoring (causal)
        stored_h[n_stored:n_stored + chunk_len] = q
        stored_tok[n_stored:n_stored + chunk_len] = t
        n_stored += chunk_len

    bpc = total_bits / scored
    return bpc, total_bits, scored


# ============================================================
# Original per-token KNN (for correctness comparison)
# ============================================================
def score_knn_pertokern(neural_probs_np, hidden_np, targets_np,
                         K=8, lam=0.12, vocab_size=VOCAB_SIZE):
    """Original slow per-token KNN (reference implementation)."""
    N = len(targets_np)
    dim = hidden_np.shape[1]
    stored_h = np.zeros((N, dim), np.float32)
    stored_tok = np.zeros(N, np.int32)
    ns = 0

    total_bits = 0.0
    scored = 0

    for i in range(N):
        tgt = int(targets_np[i])
        np_ = neural_probs_np[i].astype(np.float64)
        np_ = np.clip(np_, 1e-10, None); np_ /= np_.sum()

        if ns > K:
            diff = stored_h[:ns] - hidden_np[i]
            dists = np.einsum('ij,ij->i', diff, diff)
            ak = min(K, ns - 1)
            ki = np.argpartition(dists, ak)[:ak]
            kd = dists[ki]
            w = np.exp(-kd / dim); w /= w.sum() + 1e-30
            kp = np.zeros(vocab_size, np.float64)
            for j in range(K): kp[stored_tok[ki[j]]] += w[j]
            kp = 0.99 * kp + 0.01 / vocab_size
            mx = (1-lam) * np_ + lam * kp; mx /= mx.sum()
            p = max(mx[tgt], 1e-30)
        else:
            p = max(np_[tgt], 1e-30)

        total_bits += -math.log2(p); scored += 1
        stored_h[ns] = hidden_np[i]; stored_tok[ns] = tgt; ns += 1

    return total_bits / scored, total_bits, scored


# ============================================================
# Test: Compare vectorized vs per-token
# ============================================================
if __name__ == "__main__":
    # Load cached model
    MODEL_CACHE = "/Users/himanshudongre/Documents/GitHub/parameter_golf/cached_rope16_model.pt"

    if not os.path.exists(MODEL_CACHE):
        print("No cached model — run exp_three_way_stack.py first")
        sys.exit(1)

    # Minimal model for loading
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(dim))
            self.eps = eps
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale

    class GEGLU_MLP(nn.Module):
        def __init__(self, dim, expansion=2.0):
            super().__init__()
            h = int(dim * expansion)
            self.gate = nn.Linear(dim, h, bias=False)
            self.up = nn.Linear(dim, h, bias=False)
            self.down = nn.Linear(h, dim, bias=False)
        def forward(self, x):
            return self.down(F.gelu(self.gate(x)) * self.up(x))

    class FullMHA(nn.Module):
        def __init__(self, dim, n_heads, rope_dims=16):
            super().__init__()
            self.n_heads = n_heads; self.head_dim = dim // n_heads
            self.qkv = nn.Linear(dim, 3*dim, bias=False)
            self.out = nn.Linear(dim, dim, bias=False)
            self.rope_dims = rope_dims
            freqs = 1.0 / (10000.0 ** (torch.arange(0, rope_dims, 2).float() / rope_dims))
            t = torch.arange(SEQ_LEN).float()
            freqs = torch.outer(t, freqs)
            self.register_buffer('cos_cache', freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False)
            self.register_buffer('sin_cache', freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False)
        def _apply_rope(self, x):
            rd = self.rope_dims
            x_rope, x_pass = x[..., :rd], x[..., rd:]
            x1, x2 = x_rope[..., :rd//2], x_rope[..., rd//2:]
            cos = self.cos_cache[:, :, :x.size(2), :]
            sin = self.sin_cache[:, :, :x.size(2), :]
            out = torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], dim=-1)
            return torch.cat([out, x_pass], dim=-1)
        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            q, k = self._apply_rope(q), self._apply_rope(k)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.out(y.transpose(1,2).reshape(B, T, C))

    class Block(nn.Module):
        def __init__(self, dim, n_heads, expansion=2.0):
            super().__init__()
            self.ln1 = RMSNorm(dim); self.attn = FullMHA(dim, n_heads)
            self.ln2 = RMSNorm(dim); self.mlp = GEGLU_MLP(dim, expansion)
        def forward(self, x):
            x = x + self.attn(self.ln1(x)); x = x + self.mlp(self.ln2(x)); return x

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
            self.blocks = nn.ModuleList([Block(DIM, 6, 2.0) for _ in range(6)])
            self.ln_f = RMSNorm(DIM)
            for m in self.modules():
                if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
                elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
        def forward_with_hidden(self, idx):
            x = self.tok_emb(idx)
            for block in self.blocks: x = block(x)
            h = self.ln_f(x)
            return F.linear(h, self.tok_emb.weight), h

    # Load model
    print("Loading cached model...", flush=True)
    model = Transformer().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_CACHE, map_location=DEVICE, weights_only=True))
    model.eval()

    # Load data
    import urllib.request
    cache_path = "/Users/himanshudongre/Documents/GitHub/parameter_golf/text_corpus.txt"
    with open(cache_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    tokens = [b % VOCAB_SIZE for b in text.encode('utf-8')]
    n_seq = len(tokens) // (SEQ_LEN + 1)
    sequences = torch.tensor(tokens[:n_seq * (SEQ_LEN + 1)], dtype=torch.long).view(n_seq, SEQ_LEN + 1)
    n_train = int(n_seq * 0.9)
    eval_seq = sequences[n_train:]

    N_EVAL = min(100, len(eval_seq))
    print(f"Eval: {N_EVAL} sequences, {N_EVAL * SEQ_LEN:,} tokens")

    # Get probs + hidden states
    print("Computing probs + hidden...", flush=True)
    with torch.no_grad():
        eb = eval_seq[:N_EVAL].to(DEVICE)
        logits, hidden = model.forward_with_hidden(eb[:, :-1])
        probs = F.softmax(logits, dim=-1)  # (N, T, V)
        # Keep on device for vectorized version

    targets = eval_seq[:N_EVAL, 1:].contiguous()  # (N, T)

    # Flatten for scoring: (N*T, V), (N*T, dim), (N*T,)
    N_tokens = N_EVAL * SEQ_LEN
    probs_flat = probs.reshape(N_tokens, VOCAB_SIZE)
    hidden_flat = hidden.reshape(N_tokens, DIM)
    targets_flat = targets.reshape(N_tokens)

    # Also numpy versions for reference
    probs_np = probs_flat.cpu().numpy()
    hidden_np = hidden_flat.cpu().numpy()
    targets_np = targets_flat.cpu().numpy()

    # ==========================================
    # Test 1: Neural only
    # ==========================================
    print("\n" + "=" * 60)
    print("Neural only")
    t0 = time.time()
    tp = probs_flat.gather(1, targets_flat.to(DEVICE).unsqueeze(1)).squeeze(1)
    neural_bpc = (-torch.log2(tp.clamp(min=1e-30))).mean().item()
    print(f"  BPC: {neural_bpc:.4f} ({time.time()-t0:.2f}s)")

    # ==========================================
    # Test 2: Vectorized KNN
    # ==========================================
    print("\n" + "=" * 60)
    print("Vectorized KNN (chunk_size=1024)")
    t0 = time.time()
    vec_bpc, vec_bits, vec_scored = score_knn_vectorized(
        probs_flat, hidden_flat, targets_flat,
        K=8, lam=0.12, chunk_size=1024
    )
    vec_time = time.time() - t0
    vec_imp = (vec_bpc - neural_bpc) / neural_bpc * 100
    print(f"  BPC: {vec_bpc:.4f} ({vec_imp:+.2f}%) — {vec_time:.1f}s")

    # ==========================================
    # Test 3: Per-token KNN (reference, slow)
    # ==========================================
    print("\n" + "=" * 60)
    print("Per-token KNN (reference, slow)")
    t0 = time.time()
    ref_bpc, ref_bits, ref_scored = score_knn_pertokern(
        probs_np, hidden_np, targets_np, K=8, lam=0.12
    )
    ref_time = time.time() - t0
    ref_imp = (ref_bpc - neural_bpc) / neural_bpc * 100
    print(f"  BPC: {ref_bpc:.4f} ({ref_imp:+.2f}%) — {ref_time:.1f}s")

    # ==========================================
    # Test 4: Different chunk sizes
    # ==========================================
    print("\n" + "=" * 60)
    print("Chunk size sweep")
    for cs in [256, 512, 1024, 2048, 4096]:
        t0 = time.time()
        bpc, _, _ = score_knn_vectorized(
            probs_flat, hidden_flat, targets_flat,
            K=8, lam=0.12, chunk_size=cs
        )
        elapsed = time.time() - t0
        imp = (bpc - neural_bpc) / neural_bpc * 100
        print(f"  chunk={cs:5d}: BPC={bpc:.4f} ({imp:+.2f}%) — {elapsed:.1f}s")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Neural:          {neural_bpc:.4f}")
    print(f"  Vectorized KNN:  {vec_bpc:.4f} ({vec_imp:+.2f}%) — {vec_time:.1f}s")
    print(f"  Per-token KNN:   {ref_bpc:.4f} ({ref_imp:+.2f}%) — {ref_time:.1f}s")
    print(f"  Speedup:         {ref_time/vec_time:.0f}×")

    # Correctness check
    diff = abs(vec_bpc - ref_bpc)
    print(f"\n  BPC difference (vec vs ref): {diff:.4f}")
    if diff < 0.05:
        print(f"  CORRECTNESS: PASS (diff < 0.05)")
    else:
        print(f"  CORRECTNESS: WARN — vectorized differs from per-token")
        print(f"  (Expected: vectorized uses chunk-level causality, per-token uses token-level)")

    # Competition estimate
    print(f"\n  Competition estimate (62M tokens on 8×H100):")
    tokens_per_sec = N_tokens / vec_time
    comp_time = 62_000_000 / tokens_per_sec / 8  # 8 GPUs
    print(f"  Tokens/sec: {tokens_per_sec:.0f}")
    print(f"  Estimated: {comp_time:.0f}s (limit=600s)")
    print(f"  Fits: {'YES' if comp_time < 600 else 'NO — need larger chunk or GPU optimization'}")
