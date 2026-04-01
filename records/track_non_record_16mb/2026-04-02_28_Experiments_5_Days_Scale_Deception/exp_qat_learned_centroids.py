"""
Experiment: Learned Quantization Grid + QAT
=============================================
NOVEL: Nobody in the competition does end-to-end QAT with non-uniform centroids.

Standard approach (everyone else):
  1. Train float32 model
  2. Post-hoc GPTQ → snap to uniform int5 grid {-16..15} * scale

Our approach:
  1. Train WITH simulated quantization from step 0 (STE gradients)
  2. Quantization centroids are NON-UNIFORM and LEARNED jointly
  3. At checkpoint: model is already quantized, zero post-hoc degradation

Mathematical basis:
  - Lloyd-Max quantizer: for Gaussian weights, non-uniform centroids (denser near zero)
    reduce MSE by 20-30% vs uniform grid at same bit-width
  - STE (Straight-Through Estimator): gradient flows through quantization by
    treating the rounding step as identity in the backward pass

Tests:
  A. Float32 baseline (no quantization)
  B. Float32 + post-hoc uniform int5 (simulated GPTQ)
  C. QAT with uniform int5 grid (STE, fixed centroids)
  D. QAT with LEARNED non-uniform int5 centroids (our novel idea)
  E. QAT with NormalFloat-5 centroids (Gaussian-optimal, fixed)
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
N_BITS = 5  # int5 quantization
N_LEVELS = 2 ** N_BITS  # 32 levels

print(f"Device: {DEVICE}")
print(f"Quantization: int{N_BITS} ({N_LEVELS} levels)")
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
    """Simple byte-level tokenizer matching other experiments."""
    raw_bytes = text.encode('utf-8')
    tokens = [b % vocab_size for b in raw_bytes]
    n_seq = len(tokens) // seq_len
    tokens = tokens[:n_seq * seq_len]
    return torch.tensor(tokens, dtype=torch.long).reshape(n_seq, seq_len)

# ============================================================
# Quantization Functions
# ============================================================

def uniform_quantize(w, n_bits=N_BITS):
    """Uniform quantization: snap to nearest grid point. STE in backward."""
    n_levels = 2 ** n_bits
    # Per-channel scale: map weight range to [0, n_levels-1]
    w_min = w.min(dim=-1, keepdim=True).values
    w_max = w.max(dim=-1, keepdim=True).values
    scale = (w_max - w_min) / (n_levels - 1)
    scale = scale.clamp(min=1e-8)
    # Quantize
    w_norm = (w - w_min) / scale
    w_int = w_norm.round().clamp(0, n_levels - 1)
    # Dequantize
    w_q = w_int * scale + w_min
    # STE: forward uses quantized, backward uses identity
    return w + (w_q - w).detach()


def learned_centroid_quantize(w, centroids):
    """
    Non-uniform quantization with learned centroids.
    centroids: [n_levels] sorted tensor of quantization levels.
    Maps each weight to nearest centroid. STE for gradients.
    """
    # centroids: [n_levels], w: [out, in] or any shape
    # Per-channel scaling first
    w_flat = w.reshape(w.size(0), -1)
    w_min = w_flat.min(dim=-1, keepdim=True).values
    w_max = w_flat.max(dim=-1, keepdim=True).values
    w_range = (w_max - w_min).clamp(min=1e-8)

    # Normalize to [0, 1]
    w_norm = (w_flat - w_min) / w_range

    # Find nearest centroid for each weight (centroids are in [0, 1])
    # centroids: [n_levels], w_norm: [out, in]
    c = centroids.unsqueeze(0).unsqueeze(0)  # [1, 1, n_levels]
    w_exp = w_norm.unsqueeze(-1)  # [out, in, 1]
    dists = (w_exp - c).abs()  # [out, in, n_levels]
    idx = dists.argmin(dim=-1)  # [out, in]

    # Quantized values
    w_q_norm = centroids[idx]  # [out, in]

    # Denormalize
    w_q = w_q_norm * w_range + w_min
    w_q = w_q.reshape(w.shape)

    # STE
    return w + (w_q - w).detach()


def normalfloat_centroids(n_bits=N_BITS):
    """
    NormalFloat quantization centroids (from QLoRA's NF4, extended to NF5).
    Optimal for Gaussian-distributed weights.
    Centroids are placed at quantiles of N(0,1).
    """
    n_levels = 2 ** n_bits
    # Quantiles of standard normal, mapped to [0, 1]
    from scipy.stats import norm
    quantiles = np.array([norm.ppf((i + 0.5) / n_levels) for i in range(n_levels)])
    # Normalize to [0, 1]
    quantiles = (quantiles - quantiles.min()) / (quantiles.max() - quantiles.min())
    return torch.tensor(quantiles, dtype=torch.float32)


def uniform_centroids(n_bits=N_BITS):
    """Uniform grid centroids in [0, 1]."""
    n_levels = 2 ** n_bits
    return torch.linspace(0, 1, n_levels)


# ============================================================
# Model Components (same as integration test, with quantization hooks)
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale

class QuantLinear(nn.Module):
    """Linear layer with optional quantization simulation."""
    def __init__(self, in_features, out_features, bias=False,
                 quant_fn=None, centroids=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.quant_fn = quant_fn
        self.centroids = centroids  # external reference to shared centroids
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        w = self.weight
        if self.quant_fn is not None:
            if self.centroids is not None:
                w = self.quant_fn(w, self.centroids)
            else:
                w = self.quant_fn(w)
        return F.linear(x, w, self.bias)

class GEGLU_MLP(nn.Module):
    def __init__(self, dim, expansion=2.0, quant_fn=None, centroids=None):
        super().__init__()
        hidden = int(dim * expansion)
        self.gate = QuantLinear(dim, hidden, quant_fn=quant_fn, centroids=centroids)
        self.up = QuantLinear(dim, hidden, quant_fn=quant_fn, centroids=centroids)
        self.down = QuantLinear(hidden, dim, quant_fn=quant_fn, centroids=centroids)
    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))

class FullMHA(nn.Module):
    def __init__(self, dim, n_heads, rope_dims=0, quant_fn=None, centroids=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = QuantLinear(dim, 3 * dim, quant_fn=quant_fn, centroids=centroids)
        self.out = QuantLinear(dim, dim, quant_fn=quant_fn, centroids=centroids)
        self.rope_dims = rope_dims
        if rope_dims > 0:
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
        if self.rope_dims > 0:
            q = self._apply_rope(q)
            k = self._apply_rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out(y)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_expansion=2.0, rope_dims=0,
                 quant_fn=None, centroids=None):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = FullMHA(dim, n_heads, rope_dims=rope_dims,
                            quant_fn=quant_fn, centroids=centroids)
        self.ln2 = RMSNorm(dim)
        self.mlp = GEGLU_MLP(dim, expansion=mlp_expansion,
                             quant_fn=quant_fn, centroids=centroids)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, dim=DIM, n_heads=N_HEADS,
                 n_layers=N_LAYERS, seq_len=SEQ_LEN, mlp_expansion=MLP_EXP,
                 rope_dims=16, quant_fn=None, centroids=None):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_expansion, rope_dims=rope_dims,
                             quant_fn=quant_fn, centroids=centroids)
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(dim)

        for m in self.modules():
            if isinstance(m, (nn.Linear, QuantLinear)):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.ln_f(x), self.tok_emb.weight)

# ============================================================
# Post-hoc quantization (simulate GPTQ-like)
# ============================================================
def apply_posthoc_quantization(model, quant_fn, centroids=None):
    """Apply quantization to all Linear/QuantLinear weights post-hoc."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Linear, QuantLinear)):
                if centroids is not None:
                    m.weight.data = learned_centroid_quantize(m.weight.data, centroids)
                else:
                    m.weight.data = uniform_quantize(m.weight.data)

# ============================================================
# Training and Eval
# ============================================================
def eval_ce(model, eval_seq):
    model.eval()
    with torch.no_grad():
        eb = eval_seq[:100].to(DEVICE)
        logits = model(eb[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), eb[:, 1:].reshape(-1))
    return ce.item()

def train_and_eval(model, train_seq, eval_seq, label="", centroids_param=None):
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    extra = ""
    if centroids_param is not None:
        centroids_param = centroids_param.to(DEVICE)
        extra = f", Centroid params: {centroids_param.numel()}"
    print(f"  [{label}] Params: {n_params:,}{extra}", flush=True)

    # Collect all params including centroids
    all_params = list(model.parameters())
    param_groups = [{'params': all_params, 'lr': LR}]
    if centroids_param is not None and centroids_param.requires_grad:
        # Ensure centroids are on the right device as a leaf tensor
        if centroids_param.device != torch.device(DEVICE):
            centroids_param.data = centroids_param.data.to(DEVICE)
        param_groups.append({'params': [centroids_param], 'lr': LR * 10.0})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)

    t0 = time.time()
    best_ce = float('inf')

    for step in range(TRAIN_STEPS + 1):
        if step % 500 == 0:
            ce = eval_ce(model, eval_seq)
            ms = (time.time() - t0) / max(step, 1) * 1000
            c_str = ""
            if centroids_param is not None:
                c_vals = centroids_param.detach().cpu().numpy()
                c_str = f" | centroids: [{c_vals[0]:.3f}..{c_vals[15]:.3f}..{c_vals[-1]:.3f}]"
            print(f"    Step {step:4d} | CE: {ce:.4f} | {ms:.0f}ms/step{c_str}", flush=True)
            best_ce = min(best_ce, ce)
            model.train()

        if step >= TRAIN_STEPS:
            break

        bi = torch.randint(0, train_seq.size(0), (BATCH_SIZE,))
        batch = train_seq[bi].to(DEVICE)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) +
                                        ([centroids_param] if centroids_param is not None else []), 1.0)
        optimizer.step()
        # Keep centroids sorted
        if centroids_param is not None and centroids_param.requires_grad:
            with torch.no_grad():
                centroids_param.data = centroids_param.data.sort().values
        scheduler.step()

    elapsed = time.time() - t0
    final_ce = eval_ce(model, eval_seq)
    print(f"    Done. Best CE: {best_ce:.4f}, Final CE: {final_ce:.4f} in {elapsed:.1f}s\n", flush=True)
    return {"label": label, "best_ce": best_ce, "final_ce": final_ce, "params": n_params, "elapsed_s": elapsed}

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("EXPERIMENT: Learned Quantization Grid + QAT")
    print("=" * 70)

    text = download_text_corpus()
    all_data = tokenize_text(text)
    n_eval = min(100, all_data.size(0) // 10)
    eval_seq = all_data[:n_eval]
    train_seq = all_data[n_eval:]
    print(f"  Train: {train_seq.shape}, Eval: {eval_seq.shape}\n", flush=True)

    results = {}

    # A. Float32 baseline (no quantization)
    print("=" * 70)
    print("A. FLOAT32 BASELINE (no quantization, RoPE 16)")
    print("=" * 70)
    torch.manual_seed(42)
    model = Transformer(rope_dims=16)
    res = train_and_eval(model, train_seq, eval_seq, "float32_baseline")
    results["float32_baseline"] = res
    float_ce = res["best_ce"]

    # Now apply post-hoc uniform quantization and measure degradation
    print("  → Applying post-hoc uniform int5 quantization...")
    apply_posthoc_quantization(model, uniform_quantize)
    posthoc_ce = eval_ce(model, eval_seq)
    print(f"  → Post-hoc int5 CE: {posthoc_ce:.4f} (degradation: {(posthoc_ce-float_ce)/float_ce*100:+.2f}%)\n")
    results["float32_posthoc_int5"] = {"ce": posthoc_ce, "degradation_pct": (posthoc_ce-float_ce)/float_ce*100}
    del model; torch.mps.empty_cache() if DEVICE == "mps" else None

    # B. QAT with uniform int5 (STE, fixed centroids)
    print("=" * 70)
    print("B. QAT UNIFORM INT5 (train with quantization, fixed grid)")
    print("=" * 70)
    torch.manual_seed(42)
    model = Transformer(rope_dims=16, quant_fn=uniform_quantize)
    res = train_and_eval(model, train_seq, eval_seq, "qat_uniform_int5")
    results["qat_uniform_int5"] = res
    del model; torch.mps.empty_cache() if DEVICE == "mps" else None

    # C. QAT with NormalFloat-5 centroids (Gaussian-optimal, fixed)
    print("=" * 70)
    print("C. QAT NORMALFLOAT-5 (Gaussian-optimal centroids, fixed)")
    print("=" * 70)
    try:
        nf5_centroids = normalfloat_centroids(N_BITS).to(DEVICE)
    except ImportError:
        # scipy not available, compute manually
        # Approximate NF5: denser near 0.5 (zero), sparser at extremes
        x = torch.linspace(-3, 3, N_LEVELS)
        nf5_centroids = torch.sigmoid(x)  # S-curve, denser near center
        nf5_centroids = (nf5_centroids - nf5_centroids.min()) / (nf5_centroids.max() - nf5_centroids.min())
        nf5_centroids = nf5_centroids.to(DEVICE)

    print(f"  NF5 centroids: {nf5_centroids[:5].tolist()}...{nf5_centroids[-5:].tolist()}")
    torch.manual_seed(42)
    qfn_nf5 = lambda w, c=nf5_centroids: learned_centroid_quantize(w, c)
    model = Transformer(rope_dims=16, quant_fn=qfn_nf5)
    res = train_and_eval(model, train_seq, eval_seq, "qat_nf5_fixed")
    results["qat_nf5_fixed"] = res
    del model; torch.mps.empty_cache() if DEVICE == "mps" else None

    # D. QAT with LEARNED non-uniform centroids (OUR NOVEL IDEA)
    print("=" * 70)
    print("D. QAT LEARNED CENTROIDS ★ (non-uniform, trained jointly)")
    print("=" * 70)
    # Initialize from uniform, let them learn
    learned_c = nn.Parameter(torch.linspace(0, 1, N_LEVELS).to(DEVICE))
    qfn_learned = lambda w, c=learned_c: learned_centroid_quantize(w, c)
    torch.manual_seed(42)
    model = Transformer(rope_dims=16, quant_fn=qfn_learned)
    res = train_and_eval(model, train_seq, eval_seq, "qat_learned_centroids", centroids_param=learned_c)
    results["qat_learned_centroids"] = res

    # Print final learned centroids
    final_c = learned_c.detach().cpu().numpy()
    print(f"  Final learned centroids ({N_LEVELS} levels):")
    print(f"    {final_c.tolist()}")

    # Check centroid distribution: are they non-uniform?
    gaps = np.diff(final_c)
    print(f"  Gap stats: min={gaps.min():.4f}, max={gaps.max():.4f}, "
          f"std={gaps.std():.4f}, mean={gaps.mean():.4f}")
    print(f"  Non-uniformity ratio: {gaps.max()/gaps.min():.2f}x")
    del model; torch.mps.empty_cache() if DEVICE == "mps" else None

    # E. QAT with LEARNED centroids initialized from NF5
    print("=" * 70)
    print("E. QAT LEARNED CENTROIDS (init from NF5)")
    print("=" * 70)
    try:
        nf5_init = normalfloat_centroids(N_BITS)
    except ImportError:
        x = torch.linspace(-3, 3, N_LEVELS)
        nf5_init = torch.sigmoid(x)
        nf5_init = (nf5_init - nf5_init.min()) / (nf5_init.max() - nf5_init.min())

    learned_c2 = nn.Parameter(nf5_init.clone().to(DEVICE))
    qfn_learned2 = lambda w, c=learned_c2: learned_centroid_quantize(w, c)
    torch.manual_seed(42)
    model = Transformer(rope_dims=16, quant_fn=qfn_learned2)
    res = train_and_eval(model, train_seq, eval_seq, "qat_learned_from_nf5", centroids_param=learned_c2)
    results["qat_learned_from_nf5"] = res
    del model; torch.mps.empty_cache() if DEVICE == "mps" else None

    # ==============================================================
    # SUMMARY
    # ==============================================================
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  {'Method':<35s} {'Best CE':>10s} {'vs Float':>10s} {'vs PostQ':>10s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")

    posthoc_ce_val = results["float32_posthoc_int5"]["ce"]

    for name in ["float32_baseline", "qat_uniform_int5", "qat_nf5_fixed",
                 "qat_learned_centroids", "qat_learned_from_nf5"]:
        r = results[name]
        ce = r["best_ce"]
        vs_float = (ce - float_ce) / float_ce * 100
        vs_postq = (ce - posthoc_ce_val) / posthoc_ce_val * 100
        print(f"  {name:<35s} {ce:>10.4f} {vs_float:>+9.2f}% {vs_postq:>+9.2f}%")

    print(f"\n  Post-hoc int5 CE: {posthoc_ce_val:.4f} (degradation from float: "
          f"{results['float32_posthoc_int5']['degradation_pct']:+.2f}%)")

    print(f"\n  KEY QUESTION: Does QAT (train-with-quant) beat post-hoc quant?")
    best_qat = min(results["qat_uniform_int5"]["best_ce"],
                   results["qat_nf5_fixed"]["best_ce"],
                   results["qat_learned_centroids"]["best_ce"],
                   results["qat_learned_from_nf5"]["best_ce"])
    qat_vs_posthoc = (best_qat - posthoc_ce_val) / posthoc_ce_val * 100
    print(f"  Best QAT CE: {best_qat:.4f} vs Post-hoc: {posthoc_ce_val:.4f} ({qat_vs_posthoc:+.2f}%)")
    print(f"  {'YES — QAT wins!' if best_qat < posthoc_ce_val else 'NO — post-hoc is fine'}")

    # Save
    results_file = "/Users/himanshudongre/Documents/GitHub/parameter_golf/qat_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")
    print("\nDone!")

if __name__ == "__main__":
    main()
