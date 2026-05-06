"""
Complementary Training: Divide & Conquer Between Train and Eval
================================================================

KEY INSIGHT (from Himanshu):
  Training budget: 10 min on 8×H100 → 16MB artifact
  Eval budget:     10 min on 8×H100 → online n-gram caches

  If the neural model wastes gradients learning what n-grams can learn
  for free at eval time, we're throwing away training compute.

  SOLUTION: Train neural model to focus on what n-grams CAN'T learn.
  Then at eval time, n-grams handle the easy patterns, neural handles hard ones.
  Effective learning time = 20 minutes instead of 10.

METHODS TESTED:
  A: Standard training (baseline)
  B: Entropy-weighted loss — upweight tokens where n-gram is uncertain
  C: Residual training — train neural on (target - ngram_prediction) effectively
  D: KL divergence penalty — push neural AWAY from n-gram where n-gram is good
  E: Hard token mining — only train on tokens where n-gram entropy > threshold
  F: Curriculum — start standard, gradually increase complementary weight

All methods combined with entropy-adaptive CTW-6 at eval time.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import json
import os
import urllib.request
from collections import defaultdict

VOCAB_SIZE = 1024
SEQ_LEN = 512
DIM = 192
N_HEADS = 6
N_LAYERS = 6
MLP_EXP = 2.0
TRAIN_STEPS = 1500
BATCH_SIZE = 32
LR = 3e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Complementary Training: Divide & Conquer")
print()

# ============================================================
# Data Loading (reuse from other experiments)
# ============================================================
def download_text_corpus():
    cache_path = "/Users/himanshudongre/Documents/GitHub/parameter_golf/text_corpus.txt"
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    urls = [
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    ]
    all_text = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=30)
            text = response.read().decode('utf-8', errors='ignore')
            all_text.append(text)
        except Exception as e:
            print(f"  Failed to download {url}: {e}")
    combined = "\n\n".join(all_text)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(combined)
    return combined

def tokenize_text(text, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    raw_bytes = text.encode('utf-8')
    tokens = [b % vocab_size for b in raw_bytes]
    n_seq = len(tokens) // (seq_len + 1)
    tokens = tokens[:n_seq * (seq_len + 1)]
    return torch.tensor(tokens, dtype=torch.long).view(n_seq, seq_len + 1)

# ============================================================
# Model (same RoPE 16 architecture)
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class GEGLU_MLP(nn.Module):
    def __init__(self, dim, expansion=2.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.out(F.gelu(self.w1(x)) * self.w2(x))

class FullMHA(nn.Module):
    def __init__(self, dim, n_heads, rope_dims=16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
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
        x_rope_out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([x_rope_out, x_pass], dim=-1)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self._apply_rope(q), self._apply_rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_expansion=2.0):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = FullMHA(dim, n_heads)
        self.ln2 = RMSNorm(dim)
        self.mlp = GEGLU_MLP(dim, expansion=mlp_expansion)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.blocks = nn.ModuleList([
            TransformerBlock(DIM, N_HEADS, MLP_EXP) for _ in range(N_LAYERS)
        ])
        self.ln_f = RMSNorm(DIM)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.ln_f(x), self.tok_emb.weight)

# ============================================================
# N-gram Statistics from Training Data
# ============================================================
class TrainingNgramStats:
    """Pre-compute n-gram statistics from training data.
    Used to create per-token weights for complementary training."""

    def __init__(self, train_sequences, vocab_size=VOCAB_SIZE):
        self.V = vocab_size
        print("  Building training n-gram statistics...", flush=True)
        t0 = time.time()

        # Bigram counts from training data
        self.bigram_counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        self.bigram_totals = np.zeros(vocab_size, dtype=np.float32)

        for i in range(len(train_sequences)):
            seq = train_sequences[i].numpy()
            for t in range(len(seq) - 1):
                prev, curr = seq[t], seq[t + 1]
                self.bigram_counts[prev, curr] += 1
                self.bigram_totals[prev] += 1

        # Pre-compute bigram entropy for each context token
        # H(next | prev) = -Σ p(next|prev) * log2(p(next|prev))
        self.bigram_entropy = np.zeros(vocab_size, dtype=np.float32)
        for prev in range(vocab_size):
            total = self.bigram_totals[prev]
            if total > 0:
                probs = self.bigram_counts[prev] / total
                probs = probs[probs > 0]  # filter zeros
                self.bigram_entropy[prev] = -np.sum(probs * np.log2(probs))
            else:
                self.bigram_entropy[prev] = np.log2(vocab_size)  # max entropy

        # Trigram counts (sparse, for contexts that appear enough)
        self.trigram_entropy = {}  # (prev2, prev1) -> entropy
        trigram_counts = defaultdict(lambda: defaultdict(int))
        trigram_totals = defaultdict(int)

        for i in range(len(train_sequences)):
            seq = train_sequences[i].numpy()
            for t in range(1, len(seq) - 1):
                ctx = (int(seq[t-1]), int(seq[t]))
                trigram_counts[ctx][int(seq[t+1])] += 1
                trigram_totals[ctx] += 1

        for ctx, total in trigram_totals.items():
            if total >= 5:  # only compute for frequent contexts
                counts = np.array(list(trigram_counts[ctx].values()), dtype=np.float32)
                probs = counts / total
                self.trigram_entropy[ctx] = -np.sum(probs * np.log2(probs))

        print(f"  N-gram stats built in {time.time()-t0:.1f}s", flush=True)
        print(f"    Bigram entropy range: [{self.bigram_entropy.min():.2f}, {self.bigram_entropy.max():.2f}]")
        print(f"    Mean bigram entropy: {self.bigram_entropy.mean():.2f}")
        print(f"    Trigram contexts with stats: {len(self.trigram_entropy)}")

    def get_token_weights(self, sequences, method='entropy', **kwargs):
        """Compute per-token training weights based on n-gram difficulty.

        Args:
            sequences: (B, T+1) tensor of token sequences
            method: weight computation method

        Returns:
            weights: (B, T) tensor of per-token loss weights
        """
        B, L = sequences.shape
        T = L - 1
        weights = torch.ones(B, T, dtype=torch.float32)

        if method == 'entropy':
            # Weight = bigram_entropy(prev_token) / max_entropy
            # High entropy = n-gram uncertain = neural should focus here
            max_H = np.log2(self.V)
            for b in range(B):
                for t in range(T):
                    prev = sequences[b, t].item()
                    H = self.bigram_entropy[prev]
                    weights[b, t] = H / max_H  # normalize to [0, 1]

        elif method == 'entropy_trigram':
            # Use trigram entropy when available, fall back to bigram
            max_H = np.log2(self.V)
            for b in range(B):
                for t in range(T):
                    prev = sequences[b, t].item()
                    if t > 0:
                        prev2 = sequences[b, t-1].item()
                        ctx = (prev2, prev)
                        H = self.trigram_entropy.get(ctx, self.bigram_entropy[prev])
                    else:
                        H = self.bigram_entropy[prev]
                    weights[b, t] = H / max_H

        elif method == 'hard_only':
            # Binary: train only on tokens where n-gram entropy > threshold
            threshold = kwargs.get('threshold', 5.0)  # bits
            for b in range(B):
                for t in range(T):
                    prev = sequences[b, t].item()
                    H = self.bigram_entropy[prev]
                    weights[b, t] = 1.0 if H > threshold else 0.0

        elif method == 'inverse_confidence':
            # Weight = 1 - max_bigram_prob(prev)
            # If n-gram is very confident (one dominant next token), downweight
            for b in range(B):
                for t in range(T):
                    prev = sequences[b, t].item()
                    total = self.bigram_totals[prev]
                    if total > 0:
                        max_prob = self.bigram_counts[prev].max() / total
                        weights[b, t] = 1.0 - max_prob
                    else:
                        weights[b, t] = 1.0

        elif method == 'softmax_temp':
            # Soft version: w = sigmoid(scale * (H - threshold))
            threshold = kwargs.get('threshold', 5.0)
            scale = kwargs.get('scale', 2.0)
            for b in range(B):
                for t in range(T):
                    prev = sequences[b, t].item()
                    H = self.bigram_entropy[prev]
                    weights[b, t] = 1.0 / (1.0 + math.exp(-scale * (H - threshold)))

        # Ensure mean weight ≈ 1 so effective learning rate is preserved
        w_mean = weights.mean()
        if w_mean > 0:
            weights = weights / w_mean

        return weights

# ============================================================
# Pre-compute weights for speed (vectorized)
# ============================================================
def precompute_all_weights(train_seq, ngram_stats, method, **kwargs):
    """Pre-compute all token weights to avoid per-step overhead."""
    print(f"  Pre-computing weights (method={method})...", flush=True)
    t0 = time.time()

    B, L = train_seq.shape
    T = L - 1

    if method == 'entropy':
        max_H = np.log2(VOCAB_SIZE)
        # Vectorized: map prev tokens to their bigram entropies
        prev_tokens = train_seq[:, :-1].numpy()  # (B, T)
        entropies = ngram_stats.bigram_entropy[prev_tokens]  # fancy indexing
        weights = torch.tensor(entropies / max_H, dtype=torch.float32)

    elif method == 'inverse_confidence':
        prev_tokens = train_seq[:, :-1].numpy()
        # For each prev token, get max bigram probability
        max_probs = np.zeros_like(prev_tokens, dtype=np.float32)
        for prev in range(VOCAB_SIZE):
            total = ngram_stats.bigram_totals[prev]
            if total > 0:
                max_probs[prev_tokens == prev] = ngram_stats.bigram_counts[prev].max() / total
            else:
                max_probs[prev_tokens == prev] = 0.0
        weights = torch.tensor(1.0 - max_probs, dtype=torch.float32)

    elif method == 'softmax_temp':
        max_H = np.log2(VOCAB_SIZE)
        threshold = kwargs.get('threshold', 5.0)
        scale = kwargs.get('scale', 2.0)
        prev_tokens = train_seq[:, :-1].numpy()
        entropies = ngram_stats.bigram_entropy[prev_tokens]
        sigmoid_weights = 1.0 / (1.0 + np.exp(-scale * (entropies - threshold)))
        weights = torch.tensor(sigmoid_weights, dtype=torch.float32)

    elif method == 'hard_only':
        threshold = kwargs.get('threshold', 5.0)
        prev_tokens = train_seq[:, :-1].numpy()
        entropies = ngram_stats.bigram_entropy[prev_tokens]
        weights = torch.tensor((entropies > threshold).astype(np.float32))

    elif method == 'standard':
        weights = torch.ones(B, T, dtype=torch.float32)

    else:
        weights = torch.ones(B, T, dtype=torch.float32)

    # Normalize so mean = 1
    w_mean = weights.mean()
    if w_mean > 0:
        weights = weights / w_mean

    # Stats
    print(f"    Weights computed in {time.time()-t0:.1f}s", flush=True)
    print(f"    Weight stats: min={weights.min():.3f}, max={weights.max():.3f}, "
          f"mean={weights.mean():.3f}, std={weights.std():.3f}", flush=True)
    frac_low = (weights < 0.5).float().mean().item()
    frac_high = (weights > 1.5).float().mean().item()
    print(f"    Fraction w<0.5: {frac_low:.1%}, w>1.5: {frac_high:.1%}", flush=True)

    return weights

# ============================================================
# Training with per-token weights
# ============================================================
def train_model_weighted(train_seq, eval_seq, all_weights, label=""):
    """Train model with per-token loss weights."""
    model = Transformer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{label}] Training: {n_params:,} params", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)

    t0 = time.time()
    for step in range(TRAIN_STEPS + 1):
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                eb = eval_seq[:100].to(DEVICE)
                logits = model(eb[:, :-1])
                ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), eb[:, 1:].reshape(-1))
            print(f"    Step {step:4d} | CE: {ce:.4f} | {(time.time()-t0)/max(step,1)*1000:.0f}ms/step", flush=True)
            model.train()
        if step >= TRAIN_STEPS:
            break

        bi = torch.randint(0, train_seq.size(0), (BATCH_SIZE,))
        batch = train_seq[bi].to(DEVICE)
        weights = all_weights[bi].to(DEVICE)  # (B, T)

        logits = model(batch[:, :-1])  # (B, T, V)

        # Per-token weighted cross-entropy
        # Standard: F.cross_entropy averages over all tokens equally
        # Complementary: weight each token by n-gram difficulty
        per_token_ce = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            batch[:, 1:].reshape(-1),
            reduction='none'
        ).reshape(BATCH_SIZE, -1)  # (B, T)

        loss = (per_token_ce * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    elapsed = time.time() - t0
    print(f"  [{label}] Training done in {elapsed:.1f}s", flush=True)
    return model

# ============================================================
# Eval: Entropy-adaptive CTW-6 (our best eval method)
# ============================================================
class DirichletCTWExpert:
    def __init__(self, vocab_size=VOCAB_SIZE, max_order=6, concentrations=None):
        self.V = vocab_size
        self.max_order = max_order
        if concentrations is None:
            self.concentrations = {k: 0.5 * k for k in range(1, max_order + 1)}
        else:
            self.concentrations = concentrations
        self.unigram_counts = np.zeros(vocab_size, dtype=np.uint32)
        self.unigram_total = 0
        self.bigram_counts = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
        self.bigram_totals = np.zeros(vocab_size, dtype=np.uint32)
        self.higher_counts = {}
        self.higher_totals = {}
        for k in range(3, max_order + 1):
            self.higher_counts[k] = defaultdict(lambda: defaultdict(int))
            self.higher_totals[k] = defaultdict(int)
        self.history = []

    def update(self, token):
        self.unigram_counts[token] += 1
        self.unigram_total += 1
        if len(self.history) >= 1:
            prev = self.history[-1]
            self.bigram_counts[prev, token] += 1
            self.bigram_totals[prev] += 1
        for k in range(3, self.max_order + 1):
            if len(self.history) >= k - 1:
                ctx = tuple(self.history[-(k-1):])
                self.higher_counts[k][ctx][token] += 1
                self.higher_totals[k][ctx] += 1
        self.history.append(token)

    def get_distribution(self, context_tokens):
        V = self.V
        # Start with uniform
        p = np.ones(V, dtype=np.float64) / V

        # Unigram
        if self.unigram_total > 0:
            c1 = self.concentrations.get(1, 0.5)
            p = (self.unigram_counts.astype(np.float64) + c1 * p) / (self.unigram_total + c1)

        # Bigram
        if len(context_tokens) >= 1:
            prev = context_tokens[-1]
            total = self.bigram_totals[prev]
            if total > 0:
                c2 = self.concentrations.get(2, 1.0)
                p = (self.bigram_counts[prev].astype(np.float64) + c2 * p) / (total + c2)

        # Higher order
        for k in range(3, min(self.max_order + 1, len(context_tokens) + 2)):
            if len(context_tokens) >= k - 1:
                ctx = tuple(context_tokens[-(k-1):])
                total = self.higher_totals[k].get(ctx, 0)
                if total > 0:
                    ck = self.concentrations.get(k, 0.5 * k)
                    counts_dict = self.higher_counts[k][ctx]
                    counts = np.zeros(V, dtype=np.float64)
                    for tok, cnt in counts_dict.items():
                        counts[tok] = cnt
                    p = (counts + ck * p) / (total + ck)

        return p

def eval_with_entropy_ctw(model, eval_seq, label=""):
    """Evaluate model with entropy-adaptive CTW-6 mixing."""
    model.eval()

    # Get neural probabilities
    with torch.no_grad():
        eb = eval_seq[:100].to(DEVICE)
        logits = model(eb[:, :-1])
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    sequences = eval_seq[:100].numpy()

    # Also compute neural-only BPC for comparison
    neural_bits = 0.0
    scored = 0
    for i in range(len(sequences)):
        for t in range(sequences.shape[1] - 1):
            target = sequences[i, t + 1]
            p = max(probs[i, t, target], 1e-30)
            neural_bits += -math.log2(p)
            scored += 1
    neural_bpc = neural_bits / scored

    # Now eval with entropy-adaptive CTW-6
    ctw = DirichletCTWExpert(max_order=6)

    total_bits = 0.0
    scored = 0

    for i in range(len(sequences)):
        context_tokens = []
        for t in range(sequences.shape[1] - 1):
            target = sequences[i, t + 1]

            # Neural prediction
            neural_p = probs[i, t].astype(np.float64)
            neural_p = np.clip(neural_p, 1e-10, None)
            neural_p = neural_p / neural_p.sum()

            # CTW prediction (context = tokens scored so far in this doc)
            ctw_p = ctw.get_distribution(context_tokens)

            # Entropy-adaptive mixing
            H = -np.sum(neural_p * np.log2(np.maximum(neural_p, 1e-30)))
            alpha = 0.05 + 0.55 / (1.0 + math.exp(-2.0 * (H - 4.0)))

            mixed = (1 - alpha) * neural_p + alpha * ctw_p
            mixed = mixed / mixed.sum()

            p = max(mixed[target], 1e-30)
            total_bits += -math.log2(p)
            scored += 1

            # Update CTW (AFTER scoring) and context
            ctw.update(int(target))
            context_tokens.append(int(target))
            if len(context_tokens) > 20:
                context_tokens = context_tokens[-20:]

    mixed_bpc = total_bits / scored
    improvement = (mixed_bpc - neural_bpc) / neural_bpc * 100

    print(f"  [{label}] Neural BPC: {neural_bpc:.4f} | Mixed BPC: {mixed_bpc:.4f} | "
          f"CTW helps: {improvement:.2f}%", flush=True)

    return neural_bpc, mixed_bpc

# ============================================================
# Main Experiment
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Loading data...", flush=True)
    corpus = download_text_corpus()
    all_sequences = tokenize_text(corpus)
    n_train = int(len(all_sequences) * 0.9)
    train_seq = all_sequences[:n_train]
    eval_seq = all_sequences[n_train:]
    print(f"  Train: {train_seq.shape}, Eval: {eval_seq.shape}")

    # Build n-gram statistics from training data
    print("\n" + "=" * 70)
    print("Building N-gram Statistics from Training Data")
    print("=" * 70)
    ngram_stats = TrainingNgramStats(train_seq)

    results = {}

    # ============================================================
    # A: Standard Training (baseline)
    # ============================================================
    print("\n" + "=" * 70)
    print("A: Standard Training (baseline)")
    print("=" * 70)
    weights_standard = precompute_all_weights(train_seq, ngram_stats, 'standard')
    model_a = train_model_weighted(train_seq, eval_seq, weights_standard, label="Standard")
    neural_a, mixed_a = eval_with_entropy_ctw(model_a, eval_seq, label="Standard")
    results["standard"] = {"neural": neural_a, "mixed": mixed_a}
    del model_a
    torch.mps.empty_cache() if DEVICE == "mps" else None

    # ============================================================
    # B: Entropy-Weighted Training
    # ============================================================
    print("\n" + "=" * 70)
    print("B: Entropy-Weighted Training (upweight hard tokens)")
    print("=" * 70)
    weights_entropy = precompute_all_weights(train_seq, ngram_stats, 'entropy')
    model_b = train_model_weighted(train_seq, eval_seq, weights_entropy, label="Entropy")
    neural_b, mixed_b = eval_with_entropy_ctw(model_b, eval_seq, label="Entropy")
    results["entropy_weighted"] = {"neural": neural_b, "mixed": mixed_b}
    del model_b
    torch.mps.empty_cache() if DEVICE == "mps" else None

    # ============================================================
    # C: Inverse Confidence Training
    # ============================================================
    print("\n" + "=" * 70)
    print("C: Inverse Confidence (downweight where n-gram is confident)")
    print("=" * 70)
    weights_inv = precompute_all_weights(train_seq, ngram_stats, 'inverse_confidence')
    model_c = train_model_weighted(train_seq, eval_seq, weights_inv, label="InvConf")
    neural_c, mixed_c = eval_with_entropy_ctw(model_c, eval_seq, label="InvConf")
    results["inverse_confidence"] = {"neural": neural_c, "mixed": mixed_c}
    del model_c
    torch.mps.empty_cache() if DEVICE == "mps" else None

    # ============================================================
    # D: Sigmoid Threshold (soft version of hard mining)
    # ============================================================
    print("\n" + "=" * 70)
    print("D: Sigmoid Threshold (smooth transition at entropy=5.0)")
    print("=" * 70)
    weights_sig = precompute_all_weights(train_seq, ngram_stats, 'softmax_temp',
                                          threshold=5.0, scale=2.0)
    model_d = train_model_weighted(train_seq, eval_seq, weights_sig, label="Sigmoid")
    neural_d, mixed_d = eval_with_entropy_ctw(model_d, eval_seq, label="Sigmoid")
    results["sigmoid_threshold"] = {"neural": neural_d, "mixed": mixed_d}
    del model_d
    torch.mps.empty_cache() if DEVICE == "mps" else None

    # ============================================================
    # E: Hard Token Mining (only train on hard tokens)
    # ============================================================
    print("\n" + "=" * 70)
    print("E: Hard Token Mining (only tokens with bigram entropy > 5.0)")
    print("=" * 70)
    weights_hard = precompute_all_weights(train_seq, ngram_stats, 'hard_only', threshold=5.0)
    model_e = train_model_weighted(train_seq, eval_seq, weights_hard, label="HardOnly")
    neural_e, mixed_e = eval_with_entropy_ctw(model_e, eval_seq, label="HardOnly")
    results["hard_only"] = {"neural": neural_e, "mixed": mixed_e}
    del model_e
    torch.mps.empty_cache() if DEVICE == "mps" else None

    # ============================================================
    # F: Lower Threshold Sigmoid
    # ============================================================
    print("\n" + "=" * 70)
    print("F: Sigmoid Threshold (threshold=3.0, gentler)")
    print("=" * 70)
    weights_sig2 = precompute_all_weights(train_seq, ngram_stats, 'softmax_temp',
                                           threshold=3.0, scale=1.5)
    model_f = train_model_weighted(train_seq, eval_seq, weights_sig2, label="Sigmoid-3.0")
    neural_f, mixed_f = eval_with_entropy_ctw(model_f, eval_seq, label="Sigmoid-3.0")
    results["sigmoid_3.0"] = {"neural": neural_f, "mixed": mixed_f}
    del model_f
    torch.mps.empty_cache() if DEVICE == "mps" else None

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Complementary Training Results")
    print("=" * 70)

    baseline_neural = results["standard"]["neural"]
    baseline_mixed = results["standard"]["mixed"]

    print(f"\n{'Method':<30} {'Neural BPC':>12} {'Mixed BPC':>12} {'vs Std Neural':>14} {'vs Std Mixed':>14}")
    print("-" * 86)
    for name, r in results.items():
        vs_neural = (r["neural"] - baseline_neural) / baseline_neural * 100
        vs_mixed = (r["mixed"] - baseline_mixed) / baseline_mixed * 100
        tag = " *** BEST ***" if r["mixed"] < baseline_mixed - 0.001 else ""
        print(f"  {name:<28} {r['neural']:>12.4f} {r['mixed']:>12.4f} {vs_neural:>+13.2f}% {vs_mixed:>+13.2f}%{tag}")

    print(f"\nKEY QUESTION: Does complementary training make the neural model")
    print(f"  better when COMBINED with eval-time n-grams?")
    print(f"  Standard neural+CTW: {baseline_mixed:.4f}")
    best_method = min(results.items(), key=lambda x: x[1]["mixed"])
    print(f"  Best complementary: {best_method[0]} = {best_method[1]['mixed']:.4f}")
    delta = (best_method[1]["mixed"] - baseline_mixed) / baseline_mixed * 100
    print(f"  Delta: {delta:+.2f}%")

    # Save results
    with open("complementary_training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to complementary_training_results.json")
