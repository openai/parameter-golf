#!/usr/bin/env python3
"""
Overnight April 4: Three Novel Architectures
=============================================
Testing ideas to break below 1.06 BPB ceiling.

IDEA A: Dual Model Ensemble in 16MB
  - Clark 11L (8.66MB) + 13L freeze (5.94MB) = 14.6MB + code (~72KB)
  - At eval: run both, average logits
  - Ensembles reduce variance → better BPB

IDEA B: Full Frozen MLP + Low-Rank Correction (Idea D from discussion)
  - Freeze BOTH fc AND proj in MLP (full MLP frozen from seeds)
  - Add learned low-rank correction: A(dim→rank) @ B(rank→dim)
  - Attention fully learned
  - Enables 45L 129M params in 7.7MB artifact

IDEA C: Progressive Freeze (train 300 steps, freeze fc, continue 2700 steps)
  - Gets trained-quality fc weights for regularization benefit
  - Avoids random fc convergence problem

All tested on FineWeb sp1024 (from network volume or Gutenberg fallback).
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, json, os, copy, glob
from pathlib import Path

VOCAB_SIZE = 1024; SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64 if DEVICE == "cuda" else 32
STEPS = 3000

print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}")
print(f"Overnight Apr 4 — Novel Architectures")
print(f"Started: {time.strftime('%H:%M:%S')}")

# ============================================================
# Data
# ============================================================
def load_data():
    # Try FineWeb first
    sp1024_dir = "/workspace/repo/data/datasets/fineweb10B_sp1024"
    if os.path.exists(sp1024_dir):
        HEADER = 256 * 4
        train_files = sorted(glob.glob(os.path.join(sp1024_dir, "fineweb_train_*.bin")))[:1]
        val_files = sorted(glob.glob(os.path.join(sp1024_dir, "fineweb_val_*.bin")))
        if train_files and val_files:
            train_data = torch.from_numpy(np.fromfile(train_files[0], dtype="<u2", offset=HEADER).astype(np.int64))
            val_data = torch.from_numpy(np.fromfile(val_files[0], dtype="<u2", offset=HEADER).astype(np.int64))
            nt = len(train_data) // (SEQ_LEN + 1)
            nv = min(5000, len(val_data) // (SEQ_LEN + 1))
            print(f"Using FineWeb sp1024: {nt} train, {nv} val sequences")
            return train_data[:nt*(SEQ_LEN+1)].reshape(nt, SEQ_LEN+1), val_data[:nv*(SEQ_LEN+1)].reshape(nv, SEQ_LEN+1)

    # Fallback: Gutenberg
    for cache in ["text_corpus.txt", "/Users/himanshudongre/Documents/GitHub/parameter_golf/text_corpus.txt"]:
        if os.path.exists(cache):
            with open(cache, 'r', errors='ignore') as f: text = f.read()
            tokens = [b % VOCAB_SIZE for b in text.encode('utf-8')]
            n = len(tokens) // (SEQ_LEN + 1)
            seqs = torch.tensor(tokens[:n*(SEQ_LEN+1)], dtype=torch.long).view(n, SEQ_LEN+1)
            nt = int(n * 0.9)
            print(f"Using Gutenberg: {nt} train, {n-nt} val sequences")
            return seqs[:nt], seqs[nt:]
    raise FileNotFoundError("No data found")

# ============================================================
# Model Components
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale

class FrozenLinear(nn.Module):
    """Frozen random weights from seed. NOT saved."""
    def __init__(self, in_f, out_f, seed):
        super().__init__()
        rng = torch.Generator(); rng.manual_seed(seed)
        self.register_buffer('weight', torch.randn(out_f, in_f, generator=rng) / math.sqrt(in_f), persistent=False)
        self.in_features = in_f; self.out_features = out_f
    def forward(self, x):
        return F.linear(x, self.weight)

class Attn(nn.Module):
    def __init__(self, dim, nh=6):
        super().__init__()
        self.nh = nh; self.hd = dim // nh; rd = 16
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)
        freqs = 1.0/(10000.0**(torch.arange(0,rd,2).float()/rd))
        f = torch.outer(torch.arange(SEQ_LEN).float(), freqs)
        self.register_buffer('cos', f.cos()[None,None], persistent=False)
        self.register_buffer('sin', f.sin()[None,None], persistent=False)
        self.rd = rd
    def forward(self, x):
        B,T,C = x.shape
        qkv = self.qkv(x).reshape(B,T,3,self.nh,self.hd)
        q,k,v = qkv.unbind(2); q,k,v = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
        rd=self.rd; c=self.cos[:,:,:T]; s=self.sin[:,:,:T]
        def rope(t):
            r,p=t[...,:rd],t[...,rd:]; r1,r2=r[...,:rd//2],r[...,rd//2:]
            return torch.cat([torch.cat([r1*c-r2*s,r2*c+r1*s],-1),p],-1)
        q,k=rope(q),rope(k)
        return self.out(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,C))

# ============================================================
# Standard MLP (fully learned)
# ============================================================
class StandardMLP(nn.Module):
    def __init__(self, dim, exp=2.0):
        super().__init__()
        h = int(dim * exp)
        self.gate = nn.Linear(dim, h, bias=False)
        self.up = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)
        for m in [self.gate, self.up, self.down]: nn.init.normal_(m.weight, std=0.02)
    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))

# ============================================================
# Selective Freeze MLP (gate+up frozen, down learned)
# ============================================================
class FreezeMLP(nn.Module):
    def __init__(self, dim, exp=2.0, layer_seed=0):
        super().__init__()
        h = int(dim * exp)
        self.gate = FrozenLinear(dim, h, layer_seed*10+3)
        self.up = FrozenLinear(dim, h, layer_seed*10+4)
        self.down = nn.Linear(h, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))

# ============================================================
# IDEA B: Full Frozen MLP + Low-Rank Correction
# ============================================================
class FrozenMLPWithCorrection(nn.Module):
    """Full MLP frozen (fc+proj from seeds) + learned low-rank correction."""
    def __init__(self, dim, exp=2.0, rank=64, layer_seed=0):
        super().__init__()
        h = int(dim * exp)
        # Frozen full MLP
        self.frozen_fc = FrozenLinear(dim, h, layer_seed*10+3)
        self.frozen_proj = FrozenLinear(h, dim, layer_seed*10+5)
        # Learned low-rank correction (parallel path)
        self.corr_A = nn.Linear(dim, rank, bias=False)
        self.corr_B = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.corr_A.weight, std=0.02)
        nn.init.zeros_(self.corr_B.weight)  # start at zero = no correction initially
    def forward(self, x):
        # Frozen path
        frozen_out = self.frozen_proj(F.gelu(self.frozen_fc(x)))
        # Learned correction path
        correction = self.corr_B(F.gelu(self.corr_A(x)))
        return frozen_out + correction

# ============================================================
# Block + LM
# ============================================================
class Block(nn.Module):
    def __init__(self, dim, nh=6, mlp=None):
        super().__init__()
        self.ln1 = RMSNorm(dim); self.attn = Attn(dim, nh)
        self.ln2 = RMSNorm(dim); self.mlp = mlp
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class LM(nn.Module):
    def __init__(self, dim, nl, nh, blocks):
        super().__init__()
        self.dim = dim
        self.tok_emb = nn.Embedding(VOCAB_SIZE, dim)
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = RMSNorm(dim)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
    def forward(self, idx):
        x = self.tok_emb(idx)
        for b in self.blocks: x = b(x)
        return F.linear(self.ln_f(x), self.tok_emb.weight)
    def count_params(self):
        learned = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        frozen += sum(b.numel() for _, b in self.named_buffers())
        return learned, frozen

# ============================================================
# Training
# ============================================================
def train_eval(model, train_seq, eval_seq, steps=STEPS, lr=3e-4, wd=0.1, label=""):
    model = model.to(DEVICE)
    learned, frozen = model.count_params()
    print(f"  [{label}] Learned={learned:,} Frozen={frozen:,} Artifact~={learned*0.22/1024:.0f}KB", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time(); best = 999.0
    for step in range(steps+1):
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                eb = eval_seq[:200].to(DEVICE)
                ce = F.cross_entropy(model(eb[:,:-1]).reshape(-1,VOCAB_SIZE), eb[:,1:].reshape(-1)).item()
            best = min(best, ce)
            print(f"    Step {step:4d} | CE={ce:.4f} | Best={best:.4f} | {time.time()-t0:.0f}s", flush=True)
            model.train()
        if step >= steps: break
        bi = torch.randint(0, train_seq.size(0), (BATCH_SIZE,))
        batch = train_seq[bi].to(DEVICE)
        loss = F.cross_entropy(model(batch[:,:-1]).reshape(-1,VOCAB_SIZE), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sch.step()
    print(f"    Final: Best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return best

# ============================================================
# Build models
# ============================================================
def build_baseline(dim=192, nl=6, nh=6, exp=2.0):
    blocks = [Block(dim, nh, StandardMLP(dim, exp)) for _ in range(nl)]
    return LM(dim, nl, nh, blocks)

def build_freeze_gate_up(dim=192, nl=6, nh=6, exp=2.0):
    blocks = [Block(dim, nh, FreezeMLP(dim, exp, layer_seed=42+i)) for i in range(nl)]
    return LM(dim, nl, nh, blocks)

def build_frozen_mlp_correction(dim=192, nl=6, nh=6, exp=2.0, rank=64):
    blocks = [Block(dim, nh, FrozenMLPWithCorrection(dim, exp, rank, layer_seed=42+i)) for i in range(nl)]
    return LM(dim, nl, nh, blocks)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    train_seq, eval_seq = load_data()
    results = {}

    # === BASELINE ===
    print(f"\n{'='*60}\nBaseline 6L 192d\n{'='*60}")
    torch.manual_seed(42)
    m = build_baseline(); ce = train_eval(m, train_seq, eval_seq, label="baseline_6L")
    results["baseline_6L"] = ce; del m

    # === IDEA A: Would be dual-model ensemble, but needs Clark's code.
    # Test the CONCEPT: train 2 different models, ensemble at eval.
    print(f"\n{'='*60}\nIdea A: Dual Model Ensemble\n{'='*60}")
    torch.manual_seed(42)
    m1 = build_baseline(); train_eval(m1, train_seq, eval_seq, label="ensemble_m1")
    torch.manual_seed(123)  # different seed = different model
    m2 = build_freeze_gate_up(); train_eval(m2, train_seq, eval_seq, label="ensemble_m2")
    # Ensemble eval
    m1.eval(); m2.eval(); m1.to(DEVICE); m2.to(DEVICE)
    total_bits = 0.0; scored = 0
    with torch.no_grad():
        for i in range(0, min(200, len(eval_seq)), 10):
            eb = eval_seq[i:i+10].to(DEVICE)
            logits1 = m1(eb[:,:-1])
            logits2 = m2(eb[:,:-1])
            # Average logits (log-space ensemble)
            ensemble_logits = (logits1 + logits2) / 2
            probs = F.softmax(ensemble_logits, dim=-1)
            targets = eb[:,1:]
            for b in range(probs.shape[0]):
                for t in range(probs.shape[1]):
                    p = max(float(probs[b,t,targets[b,t]]), 1e-30)
                    total_bits += -math.log2(p); scored += 1
    ensemble_bpc = total_bits / scored
    # Compare with individual
    m1_bits = 0.0; m1_scored = 0
    with torch.no_grad():
        for i in range(0, min(200, len(eval_seq)), 10):
            eb = eval_seq[i:i+10].to(DEVICE)
            probs = F.softmax(m1(eb[:,:-1]), dim=-1)
            targets = eb[:,1:]
            for b in range(probs.shape[0]):
                for t in range(probs.shape[1]):
                    p = max(float(probs[b,t,targets[b,t]]), 1e-30)
                    m1_bits += -math.log2(p); m1_scored += 1
    m1_bpc = m1_bits / m1_scored
    print(f"  Model 1 BPC: {m1_bpc:.4f}")
    print(f"  Ensemble BPC: {ensemble_bpc:.4f} ({(ensemble_bpc-m1_bpc)/m1_bpc*100:+.2f}%)")
    results["ensemble_bpc"] = ensemble_bpc
    results["ensemble_m1_bpc"] = m1_bpc
    del m1, m2

    # === IDEA B: Frozen MLP + Low-Rank Correction ===
    print(f"\n{'='*60}\nIdea B: Frozen MLP + Low-Rank Correction\n{'='*60}")
    for rank in [32, 64, 128]:
        torch.manual_seed(42)
        m = build_frozen_mlp_correction(rank=rank)
        ce = train_eval(m, train_seq, eval_seq, label=f"frozen_corr_r{rank}")
        results[f"idea_b_r{rank}"] = ce; del m

    # Idea B at larger scale
    print(f"\n{'='*60}\nIdea B Large: 12L 384d Frozen MLP + Correction rank=64\n{'='*60}")
    torch.manual_seed(42)
    m = build_frozen_mlp_correction(dim=384, nl=12, nh=6, rank=64)
    ce = train_eval(m, train_seq, eval_seq, label="frozen_corr_12L_384d")
    results["idea_b_12L_384d"] = ce; del m

    # === IDEA C: Progressive Freeze ===
    print(f"\n{'='*60}\nIdea C: Progressive Freeze (train 1000 steps, freeze fc, continue)\n{'='*60}")
    torch.manual_seed(42)
    m = build_baseline()
    m.to(DEVICE)
    # Phase 1: train everything for 1000 steps
    trainable = list(m.parameters())
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    m.train()
    t0 = time.time()
    for step in range(1000):
        bi = torch.randint(0, train_seq.size(0), (BATCH_SIZE,))
        batch = train_seq[bi].to(DEVICE)
        loss = F.cross_entropy(m(batch[:,:-1]).reshape(-1,VOCAB_SIZE), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        eb = eval_seq[:200].to(DEVICE)
        ce_phase1 = F.cross_entropy(m(eb[:,:-1]).reshape(-1,VOCAB_SIZE), eb[:,1:].reshape(-1)).item()
    print(f"  Phase 1 (1000 steps, all learned): CE={ce_phase1:.4f} ({time.time()-t0:.0f}s)")

    # Phase 2: freeze MLP gate+up, continue training
    for block in m.blocks:
        mlp = block.mlp
        mlp.gate.weight.requires_grad = False
        mlp.up.weight.requires_grad = False
    trainable2 = [p for p in m.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(trainable2, lr=1e-4, weight_decay=0.1)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STEPS-1000)
    m.train()
    best = ce_phase1
    for step in range(1000, STEPS+1):
        if step % 500 == 0:
            m.eval()
            with torch.no_grad():
                eb = eval_seq[:200].to(DEVICE)
                ce = F.cross_entropy(m(eb[:,:-1]).reshape(-1,VOCAB_SIZE), eb[:,1:].reshape(-1)).item()
            best = min(best, ce)
            print(f"    Step {step:4d} | CE={ce:.4f} | Best={best:.4f} | {time.time()-t0:.0f}s", flush=True)
            m.train()
        if step >= STEPS: break
        bi = torch.randint(0, train_seq.size(0), (BATCH_SIZE,))
        batch = train_seq[bi].to(DEVICE)
        loss = F.cross_entropy(m(batch[:,:-1]).reshape(-1,VOCAB_SIZE), batch[:,1:].reshape(-1))
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable2, 1.0)
        opt2.step(); sch2.step()
    print(f"    Progressive freeze final: Best={best:.4f}")
    results["idea_c_progressive"] = best
    del m

    # === SUMMARY ===
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    baseline = results["baseline_6L"]
    for k, v in results.items():
        if isinstance(v, float):
            pct = (v - baseline) / baseline * 100
            print(f"  {k:35s}: {v:.4f} ({pct:+.2f}%)")

    print(f"\n  KEY QUESTIONS:")
    print(f"  1. Does ensemble beat single? {results.get('ensemble_bpc',999):.4f} vs {results.get('ensemble_m1_bpc',999):.4f}")
    b64 = results.get('idea_b_r64', 999)
    print(f"  2. Does frozen+correction work? r64={b64:.4f} vs baseline={baseline:.4f} ({(b64-baseline)/baseline*100:+.1f}%)")
    prog = results.get('idea_c_progressive', 999)
    print(f"  3. Does progressive freeze help? {prog:.4f} vs baseline={baseline:.4f} ({(prog-baseline)/baseline*100:+.1f}%)")

    with open("results_overnight_apr4.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved. Finished: {time.strftime('%H:%M:%S')}")
