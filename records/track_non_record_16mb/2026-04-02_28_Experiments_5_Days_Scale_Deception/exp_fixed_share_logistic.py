"""
Fixed-Share Logistic Mixer + Dense Trigram Cache
===================================================
The FOUNDATION of our compression system.

KEY INSIGHT: Eval-time memory is UNLIMITED (640GB GPU HBM available).
The 16MB limit is on the ARTIFACT only. At eval time, we can build:
  - Dense trigram table: 1024³ × 4B = 4GB (zero hash collisions!)
  - Sparse higher-order tables: ~1-5GB
  - Total: ~10GB of 640GB = 1.5% utilization

LEGALITY PROTOCOL (every step annotated):
  1. PREDICT: P_mix computed from expert distributions + current mixer weights
     Uses ONLY past information (already-scored tokens) ✓
  2. SCORE: -log2(P_mix(x_true)) is the official score ✓
  3. OBSERVE: see x_true ✓
  4. UPDATE: mixer weights (Fixed-Share), n-gram caches (all orders) ✓
     All updates happen AFTER scoring, same as n-gram cache updates

Two innovations vs our failed Bayesian attempt:
  1. Fixed-Share: prevents weight collapse by redistributing α fraction uniformly
  2. Logistic mixing: operates in log-odds space, amplifies confident experts
     P = sigmoid(Σ w_i * logit(p_i))  [PAQ/cmix style]

Experts:
  1. Neural model (RoPE 16, fixed after training)
  2. Dense bigram cache (online, exact, 4MB)
  3. Dense trigram cache (online, exact, 4GB — FITS in GPU memory!)
  4. Dirichlet CTW order 6 (online, sparse for orders 3-6)
  5. Dirichlet CTW order 12 (online, sparse for orders 7-12)
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
print(f"Fixed-Share Logistic Mixer + Dense Trigram Cache")
print()

# ============================================================
# Data Loading
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
            start = text.find("*** START OF")
            if start != -1: start = text.find("\n", start) + 1
            else: start = 0
            end = text.find("*** END OF")
            if end == -1: end = len(text)
            all_text.append(text[start:end])
        except Exception as e:
            print(f"  Failed: {e}", flush=True)
    corpus = "\n\n".join(all_text)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(corpus)
    return corpus

def tokenize_text(text, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN+1):
    tokens = []
    text_bytes = text.encode('utf-8', errors='ignore')
    for i in range(len(text_bytes)):
        byte_val = text_bytes[i]
        if i + 1 < len(text_bytes):
            bigram = (text_bytes[i] << 8) | text_bytes[i + 1]
            bigram_slot = 256 + (bigram % (vocab_size - 256))
            if bigram % 3 == 0:
                tokens.append(bigram_slot)
                continue
        tokens.append(byte_val)
    n_seq = len(tokens) // seq_len
    tokens = tokens[:n_seq * seq_len]
    return torch.tensor(tokens, dtype=torch.long).reshape(n_seq, seq_len)

# ============================================================
# Model (RoPE 16)
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale

class GEGLU_MLP(nn.Module):
    def __init__(self, dim, expansion=2.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))

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
# EXPERT 1: Dense Bigram Cache (4MB, exact, O(1))
# ============================================================
class DenseBigramCache:
    """Dense exact bigram table. 1024² × 4B = 4MB.
    LEGALITY: Built causally from scored tokens. update() called AFTER scoring."""
    def __init__(self, vocab_size=VOCAB_SIZE, smoothing=0.5):
        self.V = vocab_size
        self.smoothing = smoothing
        self.counts = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
        self.totals = np.zeros(vocab_size, dtype=np.uint32)

    def get_distribution(self, prev_token):
        """BEFORE scoring: predict next token given previous."""
        counts = self.counts[prev_token].astype(np.float64)
        total = self.totals[prev_token]
        c = self.smoothing
        uniform = np.ones(self.V, dtype=np.float64) / self.V
        if total == 0:
            return uniform
        return (counts + c * uniform) / (total + c)

    def update(self, prev_token, curr_token):
        """AFTER scoring: add observation."""
        self.counts[prev_token][curr_token] += 1
        self.totals[prev_token] += 1

# ============================================================
# EXPERT 2: Dense Trigram Cache (4GB on GPU, exact, O(1))
# ============================================================
class DenseTrigramCache:
    """Dense exact trigram table. 1024³ × 4B = 4GB.
    On 8×H100 (640GB HBM), this is trivial. On Mac Mini, we use uint16
    to keep it at 2GB, or skip if not enough RAM.

    LEGALITY: Built causally from scored tokens. update() called AFTER scoring.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, smoothing=1.0, use_compact=True):
        self.V = vocab_size
        self.smoothing = smoothing
        self.use_compact = use_compact

        if use_compact:
            # uint16 = 2 bytes per entry, max count 65535
            # 1024³ × 2B = 2GB — fits on Mac Mini with 16GB RAM (tight)
            try:
                self.counts = np.zeros((vocab_size, vocab_size, vocab_size), dtype=np.uint16)
                self.totals = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
                self.available = True
                print(f"    Dense trigram cache allocated: {self.counts.nbytes / 1e9:.1f}GB", flush=True)
            except MemoryError:
                print(f"    Dense trigram cache: not enough RAM, falling back to sparse", flush=True)
                self.available = False
                self.sparse_counts = defaultdict(lambda: defaultdict(int))
                self.sparse_totals = defaultdict(int)
        else:
            # Full uint32 = 4GB
            self.counts = np.zeros((vocab_size, vocab_size, vocab_size), dtype=np.uint32)
            self.totals = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
            self.available = True

    def get_distribution(self, prev2, prev1):
        """BEFORE scoring: predict next given 2-token context."""
        if self.available:
            counts = self.counts[prev2, prev1].astype(np.float64)
            total = self.totals[prev2, prev1]
        else:
            key = (prev2, prev1)
            total = self.sparse_totals[key]
            counts = np.zeros(self.V, dtype=np.float64)
            if key in self.sparse_counts:
                for tok, cnt in self.sparse_counts[key].items():
                    counts[tok] = cnt

        c = self.smoothing
        uniform = np.ones(self.V, dtype=np.float64) / self.V
        if total == 0:
            return uniform
        return (counts + c * uniform) / (total + c)

    def update(self, prev2, prev1, curr_token):
        """AFTER scoring: add observation."""
        if self.available:
            self.counts[prev2, prev1, curr_token] = min(
                self.counts[prev2, prev1, curr_token] + 1, 65535
            ) if self.use_compact else self.counts[prev2, prev1, curr_token] + 1
            self.totals[prev2, prev1] += 1
        else:
            key = (prev2, prev1)
            self.sparse_counts[key][curr_token] += 1
            self.sparse_totals[key] += 1

# ============================================================
# EXPERT 3: Dirichlet CTW (sparse higher-order, orders 1-K)
# ============================================================
class DirichletCTWExpert:
    """Dirichlet-smoothed CTW with backoff from order K down to uniform.
    Orders 1-2 use dense arrays. Orders 3+ use sparse dicts.
    LEGALITY: Built from scored tokens. update() called AFTER scoring."""
    def __init__(self, vocab_size=VOCAB_SIZE, max_order=6, concentrations=None):
        self.V = vocab_size
        self.max_order = max_order
        if concentrations is None:
            self.concentrations = {k: 0.5 * k for k in range(1, max_order + 1)}
        else:
            self.concentrations = concentrations
        self.unigram_counts = np.zeros(vocab_size, dtype=np.uint32)
        self.unigram_total = 0
        self.higher_counts = {}
        self.higher_totals = {}
        for k in range(3, max_order + 1):
            self.higher_counts[k] = defaultdict(lambda: defaultdict(int))
            self.higher_totals[k] = defaultdict(int)
        self.history = []

    def update(self, token):
        """AFTER scoring."""
        self.unigram_counts[token] += 1
        self.unigram_total += 1
        for k in range(3, self.max_order + 1):
            if len(self.history) >= k - 1:
                ctx = tuple(self.history[-(k-1):])
                self.higher_counts[k][ctx][token] += 1
                self.higher_totals[k][ctx] += 1
        self.history.append(token)

    def get_distribution(self, bigram_dist, trigram_dist, context_tokens):
        """BEFORE scoring. Takes bigram/trigram distributions from dense caches,
        applies higher-order backoff on top."""
        # Start with the trigram distribution as base (already includes bigram/unigram backoff)
        p = trigram_dist.copy()

        # Apply higher-order backoff (orders 3+)
        for k in range(3, min(self.max_order + 1, len(context_tokens) + 2)):
            if len(context_tokens) >= k - 1:
                ctx = tuple(context_tokens[-(k-1):])
                total = self.higher_totals[k].get(ctx, 0)
                if total > 0:
                    ck = self.concentrations.get(k, 0.5 * k)
                    counts_dict = self.higher_counts[k][ctx]
                    counts = np.zeros(self.V, dtype=np.float64)
                    for tok, cnt in counts_dict.items():
                        counts[tok] = cnt
                    p = (counts + ck * p) / (total + ck)
        return p

# ============================================================
# EXPERT 4: Error-Pattern Expert (NOVEL)
# ============================================================
class ErrorPatternExpert:
    """Predicts the neural model's systematic errors.

    Maintains a table: (prev_token, neural_argmax) → distribution over actual tokens.
    When the neural model predicts token X after token Y, this expert knows
    the historical distribution of what ACTUALLY followed in that scenario.

    LEGALITY: Built from scored tokens only. Uses neural prediction (computed before
    scoring) and actual token (observed after scoring) to update the table.
    Predictions use only past information.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, smoothing=1.0):
        self.V = vocab_size
        self.smoothing = smoothing
        # (prev_token, neural_argmax) → counts[actual_token]
        self.counts = np.zeros((vocab_size, vocab_size, vocab_size), dtype=np.uint16)
        self.totals = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
        self.available = False  # will check if memory allows

    def try_allocate(self):
        """Try to allocate the dense table. Falls back to sparse if OOM."""
        try:
            self.counts = np.zeros((self.V, self.V, self.V), dtype=np.uint16)
            self.totals = np.zeros((self.V, self.V), dtype=np.uint32)
            self.available = True
            print(f"    Error pattern expert allocated: {self.counts.nbytes / 1e9:.1f}GB", flush=True)
        except MemoryError:
            print(f"    Error pattern expert: using sparse fallback", flush=True)
            self.available = False
            self.sparse_counts = defaultdict(lambda: defaultdict(int))
            self.sparse_totals = defaultdict(int)

    def get_distribution(self, prev_token, neural_argmax):
        """BEFORE scoring: what actually follows when neural predicts argmax after prev?"""
        if self.available:
            counts = self.counts[prev_token, neural_argmax].astype(np.float64)
            total = self.totals[prev_token, neural_argmax]
        else:
            key = (prev_token, neural_argmax)
            total = self.sparse_totals.get(key, 0)
            counts = np.zeros(self.V, dtype=np.float64)
            if key in self.sparse_counts:
                for tok, cnt in self.sparse_counts[key].items():
                    counts[tok] = cnt

        c = self.smoothing
        uniform = np.ones(self.V, dtype=np.float64) / self.V
        if total == 0:
            return uniform
        return (counts + c * uniform) / (total + c)

    def update(self, prev_token, neural_argmax, actual_token):
        """AFTER scoring: record what actually happened."""
        if self.available:
            self.counts[prev_token, neural_argmax, actual_token] = min(
                self.counts[prev_token, neural_argmax, actual_token] + 1, 65535
            )
            self.totals[prev_token, neural_argmax] += 1
        else:
            key = (prev_token, neural_argmax)
            self.sparse_counts[key][actual_token] += 1
            self.sparse_totals[key] += 1

# ============================================================
# Fixed-Share Mixer (Linear + Log-Linear options)
# ============================================================
class FixedShareMixer:
    """
    Combines K expert distributions using Fixed-Share weight updates.

    TWO MIXING MODES:
    1. LINEAR: P_mix = Σ_k w_k * P_k  (standard weighted average)
       - Safe, guaranteed normalized, well-behaved
    2. LOG-LINEAR: P_mix ∝ Π_k P_k^{w_k}  (product-of-experts)
       - Amplifies agreement between experts (sharper distributions)
       - Better when experts make independent errors

    NOTE: PAQ-style logistic mixing is designed for BINARY prediction
    (per-bit in arithmetic coder). For multi-class (1024 vocab), it
    produces degenerate distributions. DON'T USE IT FOR TOKENS.

    FIXED-SHARE UPDATE (Herbster & Warmuth 1998):
        After scoring, update: w_k ← (1-α) * w_k * P_k(x_true) + α/K
        α controls how much weight is redistributed uniformly.
        Prevents any expert from reaching zero weight (fixes Bayesian collapse).

    LEGALITY: All updates AFTER scoring. Same protocol as n-gram cache.
    """
    def __init__(self, n_experts, alpha=0.01, mode='linear', expert_names=None):
        self.K = n_experts
        self.alpha = alpha  # Fixed-Share redistribution rate
        self.mode = mode  # 'linear' or 'loglinear'
        self.weights = np.ones(n_experts, dtype=np.float64) / n_experts
        self.expert_names = expert_names or [f"Expert_{i}" for i in range(n_experts)]
        self.total_tokens = 0

    def get_mixture(self, expert_distributions):
        """
        STEP 1 (BEFORE SCORING): Compute mixture distribution.
        Returns full normalized distribution over vocabulary.
        """
        K = len(expert_distributions)
        V = expert_distributions[0].shape[0]

        if self.mode == 'loglinear':
            # Log-linear: P ∝ Π_k P_k^{w_k}
            eps = 1e-30
            log_mixture = np.zeros(V, dtype=np.float64)
            for k in range(K):
                log_mixture += self.weights[k] * np.log(np.maximum(expert_distributions[k], eps))
            # Numerical stability: subtract max before exp
            log_mixture -= log_mixture.max()
            mixture = np.exp(log_mixture)
            mixture /= mixture.sum()
        else:
            # Linear: P = Σ_k w_k * P_k
            mixture = np.zeros(V, dtype=np.float64)
            for k in range(K):
                mixture += self.weights[k] * expert_distributions[k]
            # Ensure normalized (should already be, but safety)
            s = mixture.sum()
            if s > 0:
                mixture /= s

        return mixture

    def observe(self, expert_distributions, true_token):
        """
        STEP 3 (AFTER SCORING): Update expert weights via Fixed-Share.
        w_k ← (1-α) * w_k * P_k(x_true) / Z + α/K
        """
        # Multiplicative update: weight each expert by how well it predicted
        for k in range(self.K):
            p_k = max(expert_distributions[k][true_token], 1e-30)
            self.weights[k] *= p_k

        # Normalize
        w_sum = self.weights.sum()
        if w_sum > 0:
            self.weights /= w_sum

        # Fixed-Share redistribution: prevent any expert from dying
        self.weights = (1 - self.alpha) * self.weights + self.alpha / self.K

        self.total_tokens += 1

    def get_weights_summary(self):
        return {n: f"{w:.4f}" for n, w in zip(self.expert_names, self.weights)}

# ============================================================
# Training
# ============================================================
def train_model(train_seq, eval_seq):
    model = Transformer().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Training RoPE 16 model: {n_params:,} params", flush=True)
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
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    print(f"  Training done in {time.time()-t0:.1f}s", flush=True)
    return model

# ============================================================
# Evaluation Methods
# ============================================================

def eval_neural_only(probs, sequences):
    """Baseline: neural only."""
    total_bits = 0.0
    scored = 0
    for i in range(len(sequences)):
        for t in range(sequences.shape[1] - 1):
            target = sequences[i, t + 1]
            p = max(probs[i, t, target], 1e-30)
            total_bits += -math.log2(p)
            scored += 1
    return total_bits / scored

def eval_entropy_adaptive_ctw(probs, sequences, max_order=6):
    """Our current best: entropy-adaptive mixing with Dirichlet CTW."""
    V = VOCAB_SIZE
    ctw = DirichletCTWExpert(max_order=max_order)
    bigram_cache = DenseBigramCache()

    total_bits = 0.0
    scored = 0

    for i in range(len(sequences)):
        for t in range(sequences.shape[1] - 1):
            target = sequences[i, t + 1]
            neural_p = probs[i, t].astype(np.float64)
            neural_p = neural_p / neural_p.sum()

            # Entropy-adaptive alpha
            H = -np.sum(neural_p * np.log2(neural_p + 1e-30))
            alpha = 0.05 + 0.55 / (1.0 + np.exp(-2.0 * (H - 4.0)))

            # Get n-gram distribution
            context = list(sequences[i, max(0, t - max_order + 1):t + 1])
            prev = context[-1] if len(context) > 0 else 0
            bigram_p = bigram_cache.get_distribution(prev)

            # Use unigram backoff for CTW base
            if ctw.unigram_total > 0:
                c1 = ctw.concentrations.get(1, 0.5)
                uni_p = (ctw.unigram_counts.astype(np.float64) + c1 / V) / (ctw.unigram_total + c1)
            else:
                uni_p = np.ones(V) / V

            # CTW higher orders
            ngram_p = ctw.get_distribution(bigram_p, bigram_p, context)

            # Mix
            mixed = (1 - alpha) * neural_p + alpha * ngram_p
            p_token = max(mixed[target], 1e-30)
            total_bits += -math.log2(p_token)
            scored += 1

            # UPDATE AFTER SCORING
            bigram_cache.update(prev, target)
            ctw.update(target)

    return total_bits / scored

def eval_fixed_share(probs, sequences, use_trigram=False, use_error_expert=False,
                     max_ctw_order=6, alpha=0.01, mode='linear', label=""):
    """Fixed-Share Mixer with multiple experts."""
    V = VOCAB_SIZE

    # Set up experts
    expert_names = ["Neural", "Bigram", "CTW"]
    bigram_cache = DenseBigramCache()
    ctw = DirichletCTWExpert(max_order=max_ctw_order)

    trigram_cache = None
    error_expert = None

    if use_trigram:
        trigram_cache = DenseTrigramCache(use_compact=True)
        if not trigram_cache.available:
            # Fall back to sparse if dense doesn't fit
            pass
        expert_names.append("Trigram")

    if use_error_expert:
        error_expert = ErrorPatternExpert()
        error_expert.available = False  # force sparse on Mac Mini
        error_expert.sparse_counts = defaultdict(lambda: defaultdict(int))
        error_expert.sparse_totals = defaultdict(int)
        expert_names.append("ErrorPat")

    n_experts = len(expert_names)
    mixer = FixedShareMixer(n_experts, alpha=alpha, mode=mode, expert_names=expert_names)

    total_bits = 0.0
    scored = 0
    weight_snapshots = []

    for i in range(len(sequences)):
        for t in range(sequences.shape[1] - 1):
            target = sequences[i, t + 1]
            prev = sequences[i, t] if t > 0 else 0
            prev2 = sequences[i, t - 1] if t > 1 else 0

            # --- STEP 1: PREDICT (before scoring) ---
            neural_p = probs[i, t].astype(np.float64)
            neural_p = np.clip(neural_p, 1e-10, None)
            neural_p = neural_p / neural_p.sum()

            # Expert distributions
            expert_dists = [neural_p]

            # Bigram expert
            bigram_p = bigram_cache.get_distribution(sequences[i, t])
            expert_dists.append(bigram_p)

            # CTW expert (uses bigram as base for backoff)
            context = list(sequences[i, max(0, t - max_ctw_order + 1):t + 1])
            ctw_p = ctw.get_distribution(bigram_p, bigram_p, context)
            expert_dists.append(ctw_p)

            # Trigram expert (if enabled)
            if trigram_cache is not None:
                if t >= 2:
                    tri_p = trigram_cache.get_distribution(prev2, prev)
                else:
                    tri_p = bigram_p  # fall back to bigram for first 2 positions
                expert_dists.append(tri_p)

            # Error-pattern expert (if enabled)
            if error_expert is not None:
                neural_argmax = int(np.argmax(neural_p))
                err_p = error_expert.get_distribution(sequences[i, t], neural_argmax)
                expert_dists.append(err_p)

            # Compute mixture
            mixed = mixer.get_mixture(expert_dists)

            # --- STEP 2: SCORE ---
            p_token = max(mixed[target], 1e-30)
            total_bits += -math.log2(p_token)
            scored += 1

            # --- STEP 3: UPDATE (after scoring) ---
            mixer.observe(expert_dists, target)
            bigram_cache.update(sequences[i, t], target)
            ctw.update(target)

            if trigram_cache is not None and t >= 2:
                trigram_cache.update(prev2, prev, target)

            if error_expert is not None:
                neural_argmax = int(np.argmax(neural_p))
                error_expert.update(sequences[i, t], neural_argmax, target)

            # Weight snapshots
            if scored % 5000 == 0:
                weight_snapshots.append({
                    'scored': scored,
                    'weights': mixer.weights.copy(),
                    'bpc': total_bits / scored
                })

    bpc = total_bits / scored

    # Print weight evolution
    if weight_snapshots and label:
        print(f"    [{label}] Weight evolution:", flush=True)
        header = "      " + "".join(f"{n:>10}" for n in expert_names) + "     BPC"
        print(header, flush=True)
        for snap in weight_snapshots[::max(1, len(weight_snapshots)//5)]:
            row = "      "
            for w in snap['weights']:
                row += f"{w:10.4f}"
            row += f"   {snap['bpc']:.4f}"
            print(row, flush=True)

    return bpc

# ============================================================
# Main
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

    # Train neural model once (or load cached)
    MODEL_CACHE = "/Users/himanshudongre/Documents/GitHub/parameter_golf/cached_rope16_model.pt"
    PROBS_CACHE = "/Users/himanshudongre/Documents/GitHub/parameter_golf/cached_neural_probs.npz"

    if os.path.exists(MODEL_CACHE):
        print("\n" + "=" * 70)
        print("Loading CACHED RoPE 16 neural model")
        print("=" * 70)
        model = Transformer().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_CACHE, map_location=DEVICE, weights_only=True))
        print("  Loaded cached model!", flush=True)
    else:
        print("\n" + "=" * 70)
        print("Training RoPE 16 neural model")
        print("=" * 70)
        model = train_model(train_seq, eval_seq)
        torch.save(model.state_dict(), MODEL_CACHE)
        print(f"  Model cached to {MODEL_CACHE}", flush=True)

    # Get neural probabilities (reused by all eval methods)
    sequences = eval_seq[:100].numpy()
    if os.path.exists(PROBS_CACHE):
        print("\nLoading cached neural probabilities...", flush=True)
        data = np.load(PROBS_CACHE)
        probs = data['probs']
        print(f"  Probs shape: {probs.shape}")
    else:
        print("\nComputing neural probabilities...", flush=True)
        model.eval()
        with torch.no_grad():
            eb = eval_seq[:100].to(DEVICE)
            logits = model(eb[:, :-1])
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        np.savez_compressed(PROBS_CACHE, probs=probs)
        print(f"  Probs shape: {probs.shape} (cached)")

    results = {}

    # --- A: Neural only ---
    print("\n" + "=" * 70)
    print("A: Neural only (baseline)")
    print("=" * 70)
    bpc_a = eval_neural_only(probs, sequences)
    results["neural_only"] = bpc_a
    print(f"  BPC: {bpc_a:.4f}", flush=True)

    # --- B: Entropy-adaptive CTW-6 (our current best) ---
    print("\n" + "=" * 70)
    print("B: Entropy-adaptive CTW-6 (current best)")
    print("=" * 70)
    bpc_b = eval_entropy_adaptive_ctw(probs, sequences, max_order=6)
    results["entropy_ctw6"] = bpc_b
    print(f"  BPC: {bpc_b:.4f}", flush=True)

    # --- C: Fixed-Share Linear (Neural + Bigram + CTW-6) ---
    print("\n" + "=" * 70)
    print("C: Fixed-Share Linear (Neural + Bigram + CTW-6)")
    print("=" * 70)
    bpc_c = eval_fixed_share(probs, sequences, max_ctw_order=6,
                             alpha=0.01, mode='linear', label="FS-linear-3exp")
    results["fs_linear_3exp"] = bpc_c
    print(f"  BPC: {bpc_c:.4f}", flush=True)

    # --- C2: Fixed-Share Log-Linear (product-of-experts) ---
    print("\n" + "=" * 70)
    print("C2: Fixed-Share Log-Linear (Neural + Bigram + CTW-6)")
    print("=" * 70)
    bpc_c2 = eval_fixed_share(probs, sequences, max_ctw_order=6,
                              alpha=0.01, mode='loglinear', label="FS-loglinear-3exp")
    results["fs_loglinear_3exp"] = bpc_c2
    print(f"  BPC: {bpc_c2:.4f}", flush=True)

    # --- D: Fixed-Share + Dense Trigram ---
    print("\n" + "=" * 70)
    print("D: Fixed-Share Linear + Dense Trigram (4 experts)")
    print("=" * 70)
    bpc_d = eval_fixed_share(probs, sequences, use_trigram=True, max_ctw_order=6,
                             alpha=0.01, mode='linear', label="FS+Trigram")
    results["fs_trigram"] = bpc_d
    print(f"  BPC: {bpc_d:.4f}", flush=True)

    # --- E: Fixed-Share + Error-Pattern expert ---
    print("\n" + "=" * 70)
    print("E: Fixed-Share Linear + Error Pattern (4 experts)")
    print("=" * 70)
    bpc_e = eval_fixed_share(probs, sequences, use_error_expert=True, max_ctw_order=6,
                             alpha=0.01, mode='linear', label="FS+ErrorPat")
    results["fs_error"] = bpc_e
    print(f"  BPC: {bpc_e:.4f}", flush=True)

    # --- F: Fixed-Share FULL (all experts) ---
    print("\n" + "=" * 70)
    print("F: Fixed-Share Linear FULL (5 experts)")
    print("=" * 70)
    bpc_f = eval_fixed_share(probs, sequences, use_trigram=True, use_error_expert=True,
                             max_ctw_order=6, alpha=0.01, mode='linear', label="FS-FULL")
    results["fs_full"] = bpc_f
    print(f"  BPC: {bpc_f:.4f}", flush=True)

    # --- G: Fixed-Share with CTW-12 ---
    print("\n" + "=" * 70)
    print("G: Fixed-Share Linear (Neural + Bigram + CTW-12)")
    print("=" * 70)
    bpc_g = eval_fixed_share(probs, sequences, max_ctw_order=12,
                             alpha=0.01, mode='linear', label="FS+CTW12")
    results["fs_ctw12"] = bpc_g
    print(f"  BPC: {bpc_g:.4f}", flush=True)

    # --- H: Alpha sweep ---
    print("\n" + "=" * 70)
    print("H: Alpha sweep (Fixed-Share redistribution rate)")
    print("=" * 70)
    for alpha_val in [0.001, 0.005, 0.02, 0.05, 0.1, 0.2]:
        bpc_h = eval_fixed_share(probs, sequences, max_ctw_order=6,
                                 alpha=alpha_val, mode='linear', label=f"α={alpha_val}")
        results[f"fs_alpha_{alpha_val}"] = bpc_h
        print(f"  α={alpha_val}: BPC={bpc_h:.4f}", flush=True)

    # --- I: Log-linear with best alpha ---
    print("\n" + "=" * 70)
    print("I: Log-Linear mode sweep")
    print("=" * 70)
    for alpha_val in [0.01, 0.05, 0.1]:
        bpc_i = eval_fixed_share(probs, sequences, max_ctw_order=6,
                                 alpha=alpha_val, mode='loglinear', label=f"LL-α={alpha_val}")
        results[f"fs_loglinear_alpha_{alpha_val}"] = bpc_i
        print(f"  Log-Linear α={alpha_val}: BPC={bpc_i:.4f}", flush=True)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline = results["neural_only"]
    current_best = results["entropy_ctw6"]

    print(f"\n{'Method':<55} {'BPC':>8} {'vs Neural':>10} {'vs Current':>11}")
    print("-" * 88)
    for key, bpc in results.items():
        vs_neural = (bpc - baseline) / baseline * 100
        vs_current = (bpc - current_best) / current_best * 100
        label = key.replace("_", " ")
        print(f"  {label:<53} {bpc:8.4f} {vs_neural:+9.2f}% {vs_current:+10.2f}%")

    print("\nKEY QUESTIONS:")
    if results.get("fs_3experts", 999) < current_best:
        print(f"  ✓ Fixed-Share Logistic beats entropy-adaptive!")
    else:
        print(f"  ✗ Entropy-adaptive still better (but check full config)")

    if results.get("fs_trigram", 999) < results.get("fs_3experts", 999):
        print(f"  ✓ Dense trigram expert adds value!")
    else:
        print(f"  ✗ Dense trigram doesn't help (at this data size)")

    if results.get("fs_error", 999) < results.get("fs_3experts", 999):
        print(f"  ✓ Error-pattern expert adds value!")
    else:
        print(f"  ✗ Error-pattern doesn't help yet")

    # Save
    with open("/Users/himanshudongre/Documents/GitHub/parameter_golf/fixed_share_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to fixed_share_results.json")
