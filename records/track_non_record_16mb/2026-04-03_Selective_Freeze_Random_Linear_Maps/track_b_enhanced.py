#!/usr/bin/env python3
"""
Track B Enhanced: Freeze MLP Only, Learn Attention
====================================================
Key insight: MLPs are 64% of params doing memorization.
Attention is 32% doing reasoning. Freeze MLPs (random), learn attention.

Tests:
  1. Baseline (100% learned) — control
  2. Freeze MLP only (64% frozen, 36% learned)
  3. Freeze MLP + output proj (72% frozen)
  4. Same as #2 but LARGER model (12L 384d)
  5. Same as #2 but COMPETITION scale (12L 512d)
  6. 50% frozen / 50% learned (freeze MLP gate+up, learn down+attention)

Early stop: step 500, CE > 5.0 = FAIL
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import math, time, json, os, urllib.request

VOCAB_SIZE = 1024; SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64 if DEVICE == "cuda" else 32

print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}")
print(f"Track B Enhanced — Freeze MLP, Learn Attention")
print()

# ============================================================
# Data
# ============================================================
def load_data():
    cache = "text_corpus.txt"
    if not os.path.exists(cache):
        print("Downloading Gutenberg data...")
        urls = [
            "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
            "https://www.gutenberg.org/cache/epub/11/pg11.txt",
            "https://www.gutenberg.org/cache/epub/84/pg84.txt",
            "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        ]
        texts = []
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                texts.append(urllib.request.urlopen(req, timeout=30).read().decode('utf-8', errors='ignore'))
            except: pass
        with open(cache, 'w') as f: f.write("\n\n".join(texts))
    with open(cache, 'r', errors='ignore') as f: text = f.read()
    tokens = [b % VOCAB_SIZE for b in text.encode('utf-8')]
    n = len(tokens) // (SEQ_LEN + 1)
    seqs = torch.tensor(tokens[:n*(SEQ_LEN+1)], dtype=torch.long).view(n, SEQ_LEN+1)
    nt = int(n * 0.9)
    return seqs[:nt], seqs[nt:]

# ============================================================
# Seeded Random Linear (frozen, from seed)
# ============================================================
class FrozenLinear(nn.Module):
    def __init__(self, in_f, out_f, seed):
        super().__init__()
        rng = torch.Generator(); rng.manual_seed(seed)
        w = torch.randn(out_f, in_f, generator=rng) / math.sqrt(in_f)
        self.register_buffer('weight', w)
    def forward(self, x):
        return F.linear(x, self.weight)

# ============================================================
# Model with selective freezing
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale

class MLP(nn.Module):
    def __init__(self, dim, exp=2.0, freeze_mode="none", layer_seed=0):
        super().__init__()
        h = int(dim * exp)
        if freeze_mode == "all":
            self.gate = FrozenLinear(dim, h, layer_seed*10+3)
            self.up = FrozenLinear(dim, h, layer_seed*10+4)
            self.down = FrozenLinear(h, dim, layer_seed*10+5)
        elif freeze_mode == "gate_up":
            self.gate = FrozenLinear(dim, h, layer_seed*10+3)
            self.up = FrozenLinear(dim, h, layer_seed*10+4)
            self.down = nn.Linear(h, dim, bias=False)
            nn.init.normal_(self.down.weight, std=0.02)
        else:
            self.gate = nn.Linear(dim, h, bias=False)
            self.up = nn.Linear(dim, h, bias=False)
            self.down = nn.Linear(h, dim, bias=False)
            for m in [self.gate, self.up, self.down]: nn.init.normal_(m.weight, std=0.02)
    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))

class Attn(nn.Module):
    def __init__(self, dim, nh, freeze_out=False, layer_seed=0):
        super().__init__()
        self.nh = nh; self.hd = dim // nh; rd = 16
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        if freeze_out:
            self.out = FrozenLinear(dim, dim, layer_seed*10+2)
        else:
            self.out = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        if isinstance(self.out, nn.Linear): nn.init.normal_(self.out.weight, std=0.02)
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
        q,k = rope(q),rope(k)
        return self.out(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,C))

class Block(nn.Module):
    def __init__(self, dim, nh, exp=2.0, mlp_freeze="none", attn_freeze_out=False, layer_seed=0):
        super().__init__()
        self.ln1=RMSNorm(dim); self.attn=Attn(dim, nh, attn_freeze_out, layer_seed)
        self.ln2=RMSNorm(dim); self.mlp=MLP(dim, exp, mlp_freeze, layer_seed)
    def forward(self, x):
        x=x+self.attn(self.ln1(x)); x=x+self.mlp(self.ln2(x)); return x

class LM(nn.Module):
    def __init__(self, dim, nl, nh, exp=2.0, mlp_freeze="none", attn_freeze_out=False, base_seed=42):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB_SIZE, dim)
        self.blocks=nn.ModuleList([
            Block(dim, nh, exp, mlp_freeze, attn_freeze_out, base_seed+i)
            for i in range(nl)
        ])
        self.ln_f=RMSNorm(dim)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
    def forward(self, idx):
        x=self.tok_emb(idx)
        for b in self.blocks: x=b(x)
        return F.linear(self.ln_f(x), self.tok_emb.weight)

    def count_params(self):
        learned = sum(p.numel() for p in self.parameters())
        frozen = sum(b.numel() for n, b in self.named_buffers() if 'weight' in n)
        return learned, frozen

# ============================================================
# Training
# ============================================================
def train_eval(model, train_seq, eval_seq, steps=3000, lr=3e-4, wd=0.1, label=""):
    model = model.to(DEVICE)
    learned, frozen = model.count_params()
    total = learned + frozen
    artifact_kb = learned * 1 / 1024  # int8
    print(f"  [{label}] Learned={learned:,} Frozen={frozen:,} Total={total:,} "
          f"Frozen%={frozen/max(total,1)*100:.1f}% Artifact={artifact_kb:.0f}KB", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time(); best = 999.0

    for step in range(steps+1):
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                eb=eval_seq[:200].to(DEVICE)
                ce=F.cross_entropy(model(eb[:,:-1]).reshape(-1,VOCAB_SIZE),eb[:,1:].reshape(-1)).item()
            best=min(best,ce)
            print(f"    Step {step:4d} | CE={ce:.4f} | Best={best:.4f} | {time.time()-t0:.0f}s", flush=True)
            if step == 500 and ce > 5.0:
                print(f"    EARLY STOP: CE={ce:.4f} > 5.0"); return best, "FAIL"
            model.train()
        if step >= steps: break
        bi=torch.randint(0, train_seq.size(0), (BATCH_SIZE,))
        batch=train_seq[bi].to(DEVICE)
        loss=F.cross_entropy(model(batch[:,:-1]).reshape(-1,VOCAB_SIZE),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sch.step()

    print(f"    Final: Best CE={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return best, "PASS"

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    train_seq, eval_seq = load_data()
    print(f"Train: {train_seq.shape}, Eval: {eval_seq.shape}\n")
    results = {}

    configs = [
        # (label, dim, nl, nh, exp, mlp_freeze, attn_freeze_out)
        ("1_baseline_6L_192d",      192, 6, 6, 2.0, "none", False),
        ("2_freeze_mlp_6L_192d",    192, 6, 6, 2.0, "all",  False),
        ("3_freeze_mlp+out_6L_192d",192, 6, 6, 2.0, "all",  True),
        ("4_freeze_gate_up_6L_192d",192, 6, 6, 2.0, "gate_up", False),
        ("5_freeze_mlp_12L_384d",   384,12, 6, 2.0, "all",  False),
        ("6_freeze_mlp_12L_512d",   512,12, 8, 2.0, "all",  False),
        ("7_baseline_12L_384d",     384,12, 6, 2.0, "none", False),
        ("8_freeze_mlp_6L_192d_4x", 192, 6, 6, 4.0, "all",  False),
    ]

    for label, dim, nl, nh, exp, mlp_f, attn_fo in configs:
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        torch.manual_seed(42)
        model = LM(dim, nl, nh, exp, mlp_f, attn_fo)
        ce, status = train_eval(model, train_seq, eval_seq, steps=3000, label=label)
        results[label] = {"ce": ce, "status": status}
        del model; torch.cuda.empty_cache() if DEVICE=="cuda" else None

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    b1 = results.get("1_baseline_6L_192d", {}).get("ce", 999)
    b7 = results.get("7_baseline_12L_384d", {}).get("ce", 999)
    for label, r in results.items():
        base = b7 if "384d" in label or "512d" in label else b1
        gap = (r["ce"] - base) / base * 100 if base < 999 else 0
        print(f"  {label:35s}: CE={r['ce']:.4f} ({gap:+.1f}% vs baseline) [{r['status']}]")

    print(f"\n  KEY QUESTION: Does freeze-MLP close the gap vs full VeRA freeze?")
    vera_best = 2.3221  # from Phase 1 results
    freeze_mlp = results.get("2_freeze_mlp_6L_192d", {}).get("ce", 999)
    print(f"    Full VeRA freeze (93%): CE=2.3221")
    print(f"    Freeze MLP only (64%):  CE={freeze_mlp:.4f}")
    if freeze_mlp < vera_best:
        print(f"    YES — freeze-MLP is {(vera_best-freeze_mlp)/vera_best*100:.1f}% better!")
    else:
        print(f"    NO — freeze-MLP is worse or similar")

    print(f"\n  KEY QUESTION: Does larger frozen model beat smaller learned model?")
    large_frozen = results.get("5_freeze_mlp_12L_384d", {}).get("ce", 999)
    small_learned = results.get("1_baseline_6L_192d", {}).get("ce", 999)
    print(f"    Small learned (6L 192d, 4.2M):    CE={small_learned:.4f}")
    print(f"    Large frozen-MLP (12L 384d):       CE={large_frozen:.4f}")
    if large_frozen < small_learned:
        print(f"    YES — larger frozen model wins! This validates the approach.")
    else:
        print(f"    Gap: {(large_frozen-small_learned)/small_learned*100:+.1f}%")

    with open("results_track_b_enhanced.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved. Finished: {time.strftime('%H:%M:%S')}")
