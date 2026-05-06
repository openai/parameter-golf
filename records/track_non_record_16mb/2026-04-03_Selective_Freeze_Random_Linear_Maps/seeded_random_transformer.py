#!/usr/bin/env python3
"""
Seeded Random Transformer with VeRA Adapters
==============================================
TRACK B: "Learning Adapters on Random Linear Maps" — OpenAI Wishlist Item

Core idea: 90-95% of weights are FROZEN RANDOM (regenerated from seeds, 0 bytes in artifact).
Only small VeRA adapters are learned and stored. Enables 200M+ param model in 16MB.

Theory: Johnson-Lindenstrauss + VeRA (2023) + SeedLM (ICLR 2025)

Phase 1: Proof of Life
  PASS criterion: CE < 5.0 by step 500 (on vocab=1024)
  FAIL: Kill immediately
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import json
import os

VOCAB_SIZE = 1024
SEQ_LEN = 512
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Seeded Random Transformer — Phase 1: Proof of Life")
print()

# ============================================================
# Core: Seeded Random Weight Generation
# ============================================================
class SeededRandomLinear(nn.Module):
    """Linear layer with FROZEN random weights generated from a seed.

    The weights cost 0 bytes in the artifact — only the seed (an integer)
    is needed to regenerate them deterministically.

    At init: generate random weights from seed, freeze them.
    Forward: standard linear with frozen weights + optional adapter.
    """
    def __init__(self, in_features, out_features, seed, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed

        # Generate deterministic random weights
        rng = torch.Generator()
        rng.manual_seed(seed)
        # Kaiming-style initialization scaled by sqrt(fan_in)
        weight = torch.randn(out_features, in_features, generator=rng) / math.sqrt(in_features)

        # Register as buffer (not parameter — won't be trained or saved)
        self.register_buffer('weight', weight)

        if bias:
            self.register_buffer('bias_frozen', torch.zeros(out_features))
        else:
            self.bias_frozen = None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias_frozen)


# ============================================================
# VeRA Adapter: Learned scaling on frozen random matrices
# ============================================================
class VeRAAdapter(nn.Module):
    """Vector-based Random Matrix Adaptation (VeRA, 2023).

    Instead of learning full weight matrices, learn SCALING VECTORS
    on frozen random matrices:

    ΔW = diag(d_b) @ B0 @ diag(d_a) @ A0

    Where A0, B0 are frozen random (from seeds), d_a, d_b are learned vectors.

    Parameter cost: 2 * rank (vs LoRA's 2 * rank * dim)
    This is 10x more parameter-efficient than LoRA.
    """
    def __init__(self, in_features, out_features, rank, seed_a, seed_b, scale=1.0):
        super().__init__()
        self.rank = rank
        self.scale = scale / rank

        # Frozen random matrices (from seeds)
        rng_a = torch.Generator(); rng_a.manual_seed(seed_a)
        rng_b = torch.Generator(); rng_b.manual_seed(seed_b)

        A0 = torch.randn(rank, in_features, generator=rng_a) / math.sqrt(in_features)
        B0 = torch.randn(out_features, rank, generator=rng_b) / math.sqrt(rank)

        self.register_buffer('A0', A0)  # (rank, in)
        self.register_buffer('B0', B0)  # (out, rank)

        # LEARNED scaling vectors — the only trainable params
        self.d_a = nn.Parameter(torch.ones(rank))   # (rank,)
        self.d_b = nn.Parameter(torch.ones(rank))   # (rank,)

    def forward(self, x):
        # x: (..., in_features)
        # ΔW = B0 @ diag(d_b) @ diag(d_a) @ A0
        # Efficient: x @ A0.T @ diag(d_a * d_b) @ B0.T
        h = x @ self.A0.T           # (..., rank)
        h = h * (self.d_a * self.d_b)  # (..., rank) — element-wise scaling
        h = h @ self.B0.T           # (..., out)
        return h * self.scale


# ============================================================
# Adapted Linear: Frozen random + VeRA adapter
# ============================================================
class AdaptedLinear(nn.Module):
    """Frozen random linear + VeRA adapter in parallel.

    output = frozen_random(x) + vera_adapter(x)

    The frozen part provides the geometric structure (Johnson-Lindenstrauss).
    The adapter provides task-specific tuning.
    """
    def __init__(self, in_features, out_features, seed, adapter_rank=8,
                 adapter_scale=1.0):
        super().__init__()
        self.frozen = SeededRandomLinear(in_features, out_features, seed)
        self.adapter = VeRAAdapter(
            in_features, out_features, adapter_rank,
            seed_a=seed * 1000 + 1,  # different seeds for adapter
            seed_b=seed * 1000 + 2,
            scale=adapter_scale
        )

    def forward(self, x):
        return self.frozen(x) + self.adapter(x)


# ============================================================
# Transformer with Seeded Random Weights
# ============================================================
class RMSNorm(nn.Module):
    """Learned — these are cheap and critical for stability."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class SeededAttention(nn.Module):
    """Multi-head attention with frozen random Q,K,V,O projections + VeRA adapters."""
    def __init__(self, dim, n_heads, layer_seed, adapter_rank=8, rope_dims=16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope_dims = rope_dims

        # Frozen random projections + learned adapters
        self.qkv = AdaptedLinear(dim, 3 * dim, seed=layer_seed * 10 + 1,
                                  adapter_rank=adapter_rank)
        self.out = AdaptedLinear(dim, dim, seed=layer_seed * 10 + 2,
                                  adapter_rank=adapter_rank)

        # RoPE (standard, not random)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, rope_dims, 2).float() / rope_dims))
        t = torch.arange(SEQ_LEN).float()
        freqs = torch.outer(t, freqs)
        self.register_buffer('cos_cache', freqs.cos()[None, None], persistent=False)
        self.register_buffer('sin_cache', freqs.sin()[None, None], persistent=False)

    def _apply_rope(self, x):
        rd = self.rope_dims
        x_rope, x_pass = x[..., :rd], x[..., rd:]
        x1, x2 = x_rope[..., :rd//2], x_rope[..., rd//2:]
        T = x.size(2)
        cos = self.cos_cache[:, :, :T]; sin = self.sin_cache[:, :, :T]
        out = torch.cat([x1*cos - x2*sin, x2*cos + x1*sin], -1)
        return torch.cat([out, x_pass], -1)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self._apply_rope(q), self._apply_rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class SeededMLP(nn.Module):
    """GEGLU MLP with frozen random weights + VeRA adapters."""
    def __init__(self, dim, expansion, layer_seed, adapter_rank=8):
        super().__init__()
        hidden = int(dim * expansion)
        self.gate = AdaptedLinear(dim, hidden, seed=layer_seed * 10 + 3,
                                   adapter_rank=adapter_rank)
        self.up = AdaptedLinear(dim, hidden, seed=layer_seed * 10 + 4,
                                 adapter_rank=adapter_rank)
        self.down = AdaptedLinear(hidden, dim, seed=layer_seed * 10 + 5,
                                   adapter_rank=adapter_rank)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))


class SeededBlock(nn.Module):
    def __init__(self, dim, n_heads, expansion, layer_seed, adapter_rank=8):
        super().__init__()
        self.ln1 = RMSNorm(dim)  # LEARNED (cheap, critical)
        self.attn = SeededAttention(dim, n_heads, layer_seed, adapter_rank)
        self.ln2 = RMSNorm(dim)  # LEARNED
        self.mlp = SeededMLP(dim, expansion, layer_seed, adapter_rank)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SeededTransformer(nn.Module):
    """Full transformer with frozen random weights + VeRA adapters.

    Trainable: embeddings, layer norms, VeRA scaling vectors
    Frozen: all linear projections (regenerated from seeds)
    """
    def __init__(self, dim=256, n_layers=6, n_heads=4, expansion=2.0,
                 adapter_rank=8, base_seed=42):
        super().__init__()
        self.dim = dim

        # LEARNED: embeddings (must be learned for task-specific vocabulary)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, dim)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        # Blocks with frozen random weights + adapters
        self.blocks = nn.ModuleList([
            SeededBlock(dim, n_heads, expansion,
                       layer_seed=base_seed + i,
                       adapter_rank=adapter_rank)
            for i in range(n_layers)
        ])

        # LEARNED: final norm
        self.ln_f = RMSNorm(dim)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        # Weight-tied LM head (uses learned embedding)
        return F.linear(self.ln_f(x), self.tok_emb.weight)

    def count_params(self):
        """Count learned vs frozen parameters."""
        learned = 0
        frozen = 0
        for name, param in self.named_parameters():
            learned += param.numel()
        for name, buf in self.named_buffers():
            if 'weight' in name or 'A0' in name or 'B0' in name:
                frozen += buf.numel()
        return learned, frozen

    def artifact_size(self):
        """Estimate artifact size (only learned params)."""
        learned, _ = self.count_params()
        # At int8: 1 byte per param
        return learned * 1  # bytes


# ============================================================
# Data Loading
# ============================================================
def load_data():
    cache = "/Users/himanshudongre/Documents/GitHub/parameter_golf/text_corpus.txt"
    if not os.path.exists(cache):
        cache = "text_corpus.txt"
    with open(cache, 'r', errors='ignore') as f:
        text = f.read()
    tokens = [b % VOCAB_SIZE for b in text.encode('utf-8')]
    n = len(tokens) // (SEQ_LEN + 1)
    seqs = torch.tensor(tokens[:n*(SEQ_LEN+1)], dtype=torch.long).view(n, SEQ_LEN+1)
    nt = int(n * 0.9)
    return seqs[:nt], seqs[nt:]


# ============================================================
# Training
# ============================================================
def train_and_eval(model, train_seq, eval_seq, steps=3000, lr=3e-4, wd=0.1,
                   label="", early_stop_step=500, early_stop_ce=5.0):
    """Train with early stopping."""
    model = model.to(DEVICE)
    learned, frozen = model.count_params()
    artifact = model.artifact_size()

    print(f"  [{label}]")
    print(f"    Learned params: {learned:,} ({learned*4/1e6:.1f}MB FP32)")
    print(f"    Frozen params:  {frozen:,} ({frozen*4/1e6:.1f}MB FP32)")
    print(f"    Total params:   {learned+frozen:,}")
    print(f"    Frozen ratio:   {frozen/(learned+frozen)*100:.1f}%")
    print(f"    Artifact size:  {artifact/1024:.1f}KB (int8)")
    print(f"    Effective model: {(learned+frozen)*4/1e6:.1f}MB")

    # Only train learned parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    t0 = time.time()
    best_ce = float('inf')

    for step in range(steps + 1):
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                eb = eval_seq[:min(200, len(eval_seq))].to(DEVICE)
                logits = model(eb[:, :-1])
                ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), eb[:, 1:].reshape(-1)).item()
            best_ce = min(best_ce, ce)
            elapsed = time.time() - t0
            print(f"    Step {step:4d} | CE={ce:.4f} | Best={best_ce:.4f} | {elapsed:.0f}s", flush=True)

            # EARLY STOPPING
            if step == early_stop_step and ce > early_stop_ce:
                print(f"    EARLY STOP: CE={ce:.4f} > {early_stop_ce} at step {step}")
                print(f"    VERDICT: FAIL")
                return best_ce, "FAIL"

            model.train()

        if step >= steps:
            break

        bi = torch.randint(0, train_seq.size(0), (32,))
        batch = train_seq[bi].to(DEVICE)
        loss = F.cross_entropy(model(batch[:, :-1]).reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step(); scheduler.step()

    final_ce = best_ce
    elapsed = time.time() - t0
    print(f"    Final: CE={final_ce:.4f} in {elapsed:.0f}s")
    return final_ce, "PASS" if final_ce < early_stop_ce else "MARGINAL"


# ============================================================
# Fully Trained Baseline (for comparison)
# ============================================================
class FullyTrainedTransformer(nn.Module):
    """Standard transformer with ALL weights learned. Same architecture as
    SeededTransformer but nothing is frozen."""
    def __init__(self, dim=256, n_layers=6, n_heads=4, expansion=2.0):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, dim)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                'ln1': RMSNorm(dim),
                'qkv': nn.Linear(dim, 3*dim, bias=False),
                'out': nn.Linear(dim, dim, bias=False),
                'ln2': RMSNorm(dim),
                'gate': nn.Linear(dim, int(dim*expansion), bias=False),
                'up': nn.Linear(dim, int(dim*expansion), bias=False),
                'down': nn.Linear(int(dim*expansion), dim, bias=False),
            }))
        self.ln_f = RMSNorm(dim)
        self.n_heads = n_heads; self.head_dim = dim // n_heads
        rd = 16
        freqs = 1.0/(10000.0**(torch.arange(0,rd,2).float()/rd))
        f = torch.outer(torch.arange(SEQ_LEN).float(), freqs)
        self.register_buffer('cos', f.cos()[None,None], persistent=False)
        self.register_buffer('sin', f.sin()[None,None], persistent=False)
        self.rd = rd
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            # Attention
            h = block['ln1'](x)
            B, T, C = h.shape
            qkv = block['qkv'](h).reshape(B,T,3,self.n_heads,self.head_dim)
            q,k,v = qkv.unbind(2); q,k,v = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
            rd=self.rd; c=self.cos[:,:,:T]; s=self.sin[:,:,:T]
            def rope(t):
                r,p=t[...,:rd],t[...,rd:]; r1,r2=r[...,:rd//2],r[...,rd//2:]
                return torch.cat([torch.cat([r1*c-r2*s,r2*c+r1*s],-1),p],-1)
            q,k = rope(q),rope(k)
            y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
            x = x + block['out'](y.transpose(1,2).reshape(B,T,C))
            # MLP
            h = block['ln2'](x)
            x = x + block['down'](F.gelu(block['gate'](h)) * block['up'](h))
        return F.linear(self.ln_f(x), self.tok_emb.weight)

    def count_params(self):
        return sum(p.numel() for p in self.parameters()), 0

    def artifact_size(self):
        return sum(p.numel() for p in self.parameters()) * 1


# ============================================================
# Main: Phase 1 Experiments
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1: Proof of Life")
    print("=" * 70)

    train_seq, eval_seq = load_data()
    print(f"Train: {train_seq.shape}, Eval: {eval_seq.shape}")

    results = {}

    # ==========================================
    # A: Fully trained baseline (100% learned)
    # ==========================================
    print(f"\n{'='*60}")
    print("A: Fully Trained Baseline (100% learned)")
    print(f"{'='*60}")
    torch.manual_seed(42)
    baseline = FullyTrainedTransformer(dim=256, n_layers=6, n_heads=4, expansion=2.0)
    ce_a, status_a = train_and_eval(baseline, train_seq, eval_seq, steps=3000, label="baseline")
    results["A_baseline"] = {"ce": ce_a, "status": status_a}

    # ==========================================
    # B: 90% frozen + VeRA rank=8
    # ==========================================
    print(f"\n{'='*60}")
    print("B: Seeded Random (90% frozen) + VeRA rank=8")
    print(f"{'='*60}")
    torch.manual_seed(42)
    model_b = SeededTransformer(dim=256, n_layers=6, n_heads=4, expansion=2.0,
                                 adapter_rank=8, base_seed=42)
    ce_b, status_b = train_and_eval(model_b, train_seq, eval_seq, steps=3000, label="90%_frozen_r8")
    results["B_90frozen_r8"] = {"ce": ce_b, "status": status_b}

    # ==========================================
    # C: 90% frozen + VeRA rank=16
    # ==========================================
    print(f"\n{'='*60}")
    print("C: Seeded Random (90% frozen) + VeRA rank=16")
    print(f"{'='*60}")
    torch.manual_seed(42)
    model_c = SeededTransformer(dim=256, n_layers=6, n_heads=4, expansion=2.0,
                                 adapter_rank=16, base_seed=42)
    ce_c, status_c = train_and_eval(model_c, train_seq, eval_seq, steps=3000, label="90%_frozen_r16")
    results["C_90frozen_r16"] = {"ce": ce_c, "status": status_c}

    # ==========================================
    # D: 90% frozen + VeRA rank=32
    # ==========================================
    print(f"\n{'='*60}")
    print("D: Seeded Random (90% frozen) + VeRA rank=32")
    print(f"{'='*60}")
    torch.manual_seed(42)
    model_d = SeededTransformer(dim=256, n_layers=6, n_heads=4, expansion=2.0,
                                 adapter_rank=32, base_seed=42)
    ce_d, status_d = train_and_eval(model_d, train_seq, eval_seq, steps=3000, label="90%_frozen_r32")
    results["D_90frozen_r32"] = {"ce": ce_d, "status": status_d}

    # ==========================================
    # E: LARGER model — 12L 384d (more total params, same artifact)
    # ==========================================
    print(f"\n{'='*60}")
    print("E: Large Seeded (12L 384d, 90% frozen) + VeRA rank=16")
    print(f"{'='*60}")
    torch.manual_seed(42)
    model_e = SeededTransformer(dim=384, n_layers=12, n_heads=6, expansion=2.0,
                                 adapter_rank=16, base_seed=42)
    ce_e, status_e = train_and_eval(model_e, train_seq, eval_seq, steps=3000, label="large_90%_frozen")
    results["E_large_90frozen"] = {"ce": ce_e, "status": status_e}

    # ==========================================
    # Summary
    # ==========================================
    print(f"\n{'='*70}")
    print("PHASE 1 SUMMARY")
    print(f"{'='*70}")
    for name, res in results.items():
        print(f"  {name}: CE={res['ce']:.4f} [{res['status']}]")

    baseline_ce = results["A_baseline"]["ce"]
    best_seeded = min((v["ce"], k) for k, v in results.items() if k != "A_baseline")
    gap = (best_seeded[0] - baseline_ce) / baseline_ce * 100

    print(f"\n  Baseline (100% learned): {baseline_ce:.4f}")
    print(f"  Best seeded:             {best_seeded[0]:.4f} [{best_seeded[1]}]")
    print(f"  Gap: {gap:+.1f}%")

    if gap < 30:
        print(f"\n  VERDICT: PROMISING — within 30% of baseline. Proceed to Phase 2.")
    elif gap < 50:
        print(f"\n  VERDICT: MARGINAL — within 50%. Try higher rank or different adapter.")
    else:
        print(f"\n  VERDICT: POOR — >50% gap. Fundamental issue with frozen random approach.")

    with open("/Users/himanshudongre/Documents/GitHub/parameter_golf/results_random_adapters_phase1.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results. Finished: {time.strftime('%H:%M:%S')}")
