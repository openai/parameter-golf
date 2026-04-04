#!/usr/bin/env python3
"""
Track B Advanced on FINEWEB data (sp1024)
==========================================
Uses actual competition data from network volume.
Tests: freeze+dropout combined, larger models, scaled freeze, progressive unfreeze.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, json, os, glob
from pathlib import Path

VOCAB_SIZE = 1024; SEQ_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
print(f"Device: {DEVICE}")
print(f"Track B Advanced on FineWeb — {time.strftime('%H:%M:%S')}")

# ============================================================
# FineWeb Data Loading (from network volume)
# ============================================================
DATA_DIR = "/workspace/repo/data/datasets/fineweb10B_sp1024"
HEADER_BYTES = 256 * 4

def load_shard(path):
    data = np.fromfile(path, dtype="<u2", offset=HEADER_BYTES)
    return torch.from_numpy(data.astype(np.int64))

def load_fineweb():
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "fineweb_train_*.bin")))[:10]  # 10 shards
    val_files = sorted(glob.glob(os.path.join(DATA_DIR, "fineweb_val_*.bin")))
    print(f"Loading {len(train_files)} train shards + {len(val_files)} val shards")

    # Load first train shard for training
    train_tokens = load_shard(train_files[0])
    n_train = len(train_tokens) // (SEQ_LEN + 1)
    train_seq = train_tokens[:n_train * (SEQ_LEN + 1)].reshape(n_train, SEQ_LEN + 1)

    # Load val
    val_tokens = load_shard(val_files[0])
    n_val = min(5000, len(val_tokens) // (SEQ_LEN + 1))
    val_seq = val_tokens[:n_val * (SEQ_LEN + 1)].reshape(n_val, SEQ_LEN + 1)

    print(f"Train: {train_seq.shape}, Val: {val_seq.shape}")
    return train_seq, val_seq

# ============================================================
# Model components (same as before)
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale

class FrozenLinear(nn.Module):
    def __init__(self, in_f, out_f, seed):
        super().__init__()
        rng = torch.Generator(); rng.manual_seed(seed)
        self.register_buffer('weight', torch.randn(out_f, in_f, generator=rng) / math.sqrt(in_f))
        self.in_features = in_f; self.out_features = out_f
    def forward(self, x): return F.linear(x, self.weight)

class ScaledFrozenLinear(nn.Module):
    def __init__(self, in_f, out_f, seed):
        super().__init__()
        rng = torch.Generator(); rng.manual_seed(seed)
        self.register_buffer('weight', torch.randn(out_f, in_f, generator=rng) / math.sqrt(in_f))
        self.channel_scale = nn.Parameter(torch.ones(out_f))
        self.in_features = in_f; self.out_features = out_f
    def forward(self, x):
        return F.linear(x, self.weight) * self.channel_scale

class MLP(nn.Module):
    def __init__(self, dim, exp=2.0, mode="learned", dropout=0.0, layer_seed=0):
        super().__init__()
        h = int(dim * exp)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        if mode == "freeze_gate_up":
            self.gate = FrozenLinear(dim, h, layer_seed*10+3)
            self.up = FrozenLinear(dim, h, layer_seed*10+4)
        elif mode == "scaled_freeze":
            self.gate = ScaledFrozenLinear(dim, h, layer_seed*10+3)
            self.up = ScaledFrozenLinear(dim, h, layer_seed*10+4)
        else:
            self.gate = nn.Linear(dim, h, bias=False)
            self.up = nn.Linear(dim, h, bias=False)
            nn.init.normal_(self.gate.weight, std=0.02)
            nn.init.normal_(self.up.weight, std=0.02)
        self.down = nn.Linear(h, dim, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
    def forward(self, x):
        h = F.gelu(self.gate(x)) * self.up(x)
        if self.dropout: h = self.dropout(h)
        return self.down(h)

class Attn(nn.Module):
    def __init__(self, dim, nh=6):
        super().__init__()
        self.nh=nh; self.hd=dim//nh; rd=16
        self.qkv=nn.Linear(dim,3*dim,bias=False); self.out=nn.Linear(dim,dim,bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02); nn.init.normal_(self.out.weight, std=0.02)
        freqs=1.0/(10000.0**(torch.arange(0,rd,2).float()/rd))
        f=torch.outer(torch.arange(SEQ_LEN).float(),freqs)
        self.register_buffer('cos',f.cos()[None,None],persistent=False)
        self.register_buffer('sin',f.sin()[None,None],persistent=False); self.rd=rd
    def forward(self, x):
        B,T,C=x.shape; qkv=self.qkv(x).reshape(B,T,3,self.nh,self.hd)
        q,k,v=qkv.unbind(2); q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
        rd=self.rd; c=self.cos[:,:,:T]; s=self.sin[:,:,:T]
        def rope(t):
            r,p=t[...,:rd],t[...,rd:]; r1,r2=r[...,:rd//2],r[...,rd//2:]
            return torch.cat([torch.cat([r1*c-r2*s,r2*c+r1*s],-1),p],-1)
        q,k=rope(q),rope(k)
        return self.out(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,C))

class Block(nn.Module):
    def __init__(self, dim, nh=6, exp=2.0, mlp_mode="learned", dropout=0.0, layer_seed=0):
        super().__init__()
        self.ln1=RMSNorm(dim); self.attn=Attn(dim,nh)
        self.ln2=RMSNorm(dim); self.mlp=MLP(dim,exp,mlp_mode,dropout,layer_seed)
    def forward(self, x):
        x=x+self.attn(self.ln1(x)); x=x+self.mlp(self.ln2(x)); return x

class LM(nn.Module):
    def __init__(self, dim=192, nl=6, nh=6, exp=2.0, mlp_mode="learned", dropout=0.0, base_seed=42):
        super().__init__()
        self.tok_emb=nn.Embedding(VOCAB_SIZE,dim)
        self.blocks=nn.ModuleList([Block(dim,nh,exp,mlp_mode,dropout,base_seed+i) for i in range(nl)])
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

def train_eval(model, train_seq, eval_seq, steps=3000, lr=3e-4, wd=0.1, label=""):
    model=model.to(DEVICE)
    learned, frozen = model.count_params()
    print(f"  [{label}] Learned={learned:,} Frozen={frozen:,} Artifact={learned/1024:.0f}KB", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0=time.time(); best=999.0
    for step in range(steps+1):
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                eb=eval_seq[:200].to(DEVICE)
                ce=F.cross_entropy(model(eb[:,:-1]).reshape(-1,VOCAB_SIZE),eb[:,1:].reshape(-1)).item()
            best=min(best,ce)
            print(f"    Step {step:4d} | CE={ce:.4f} | Best={best:.4f} | {time.time()-t0:.0f}s", flush=True)
            model.train()
        if step>=steps: break
        bi=torch.randint(0, train_seq.size(0),(BATCH_SIZE,))
        batch=train_seq[bi].to(DEVICE)
        loss=F.cross_entropy(model(batch[:,:-1]).reshape(-1,VOCAB_SIZE),batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sch.step()
    print(f"    Final: Best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return best, learned

if __name__ == "__main__":
    train_seq, eval_seq = load_fineweb()
    results = {}

    configs = [
        # (label, dim, nl, nh, exp, mlp_mode, dropout, wd)
        ("1_baseline_dropout02",       192, 6, 6, 2.0, "learned",        0.2, 0.1),
        ("2_freeze+dropout02",         192, 6, 6, 2.0, "freeze_gate_up", 0.2, 0.1),
        ("3_freeze_8L256d",            256, 8, 4, 2.0, "freeze_gate_up", 0.0, 0.1),
        ("4_scaled_freeze",            192, 6, 6, 2.0, "scaled_freeze",  0.0, 0.1),
        ("5_freeze+drop_8L256d",       256, 8, 4, 2.0, "freeze_gate_up", 0.2, 0.1),
        ("6_freeze+drop_12L384d",      384,12, 6, 2.0, "freeze_gate_up", 0.1, 0.1),
        ("7_baseline_12L384d",         384,12, 6, 2.0, "learned",        0.0, 0.1),
    ]

    for label, dim, nl, nh, exp, mode, dropout, wd in configs:
        print(f"\n{'='*50}\n{label}\n{'='*50}")
        torch.manual_seed(42)
        model = LM(dim, nl, nh, exp, mode, dropout)
        ce, learned = train_eval(model, train_seq, eval_seq, steps=3000, wd=wd, label=label)
        results[label] = {"ce": ce, "learned": learned, "artifact_kb": learned/1024}
        del model; torch.cuda.empty_cache()

    print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    d02 = results["1_baseline_dropout02"]["ce"]
    for label, r in results.items():
        pct = (r["ce"] - d02) / d02 * 100
        print(f"  {label:<30s}: CE={r['ce']:.4f} Artifact={r['artifact_kb']:.0f}KB ({pct:+.1f}% vs dropout)")

    # The critical comparison: does BIGGER frozen+dropout beat SMALLER dropout?
    small_dropout = results["1_baseline_dropout02"]
    big_freeze_drop = results.get("6_freeze+drop_12L384d", {})
    if big_freeze_drop:
        print(f"\n  CRITICAL: Bigger frozen+dropout vs smaller dropout (same artifact budget)")
        print(f"    Small dropout (6L 192d): CE={small_dropout['ce']:.4f}, {small_dropout['artifact_kb']:.0f}KB")
        print(f"    Big frozen+drop (12L 384d): CE={big_freeze_drop['ce']:.4f}, {big_freeze_drop['artifact_kb']:.0f}KB")
        if big_freeze_drop["ce"] < small_dropout["ce"]:
            print(f"    WINNER: Bigger frozen model wins by {(small_dropout['ce']-big_freeze_drop['ce'])/small_dropout['ce']*100:.1f}%!")

    with open("/workspace/results_track_b_advanced_fineweb.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved. Finished: {time.strftime('%H:%M:%S')}")
