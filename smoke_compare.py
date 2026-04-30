"""
A/B Smoke Test — compare baseline vs improved model on CPU.
Runs both configs for the same number of steps, reports BPB difference.

Usage: python smoke_compare.py [num_steps]
"""
import sys
import os
import time
import math
import glob
import numpy as np
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F
import sentencepiece as spm
from torch import nn

num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
seq_len = 256
batch_tokens = 2048
vocab_size = 1024
dim = 512

# ---- Data ----
data_path = "./data/datasets/fineweb10B_sp1024"
tokenizer_path = "./data/tokenizers/fineweb_1024_bpe.model"

def load_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=num_tokens, offset=256*4).astype(np.uint16))

train_files = sorted(glob.glob(os.path.join(data_path, "fineweb_train_*.bin")))
val_files = sorted(glob.glob(os.path.join(data_path, "fineweb_val_*.bin")))
train_tokens = load_shard(Path(train_files[0]))
val_tokens = torch.cat([load_shard(Path(f)) for f in val_files])

sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

def build_bpb_luts(sp, vs):
    sv = int(sp.vocab_size())
    ts = max(sv, vs)
    bb = np.zeros(ts, dtype=np.int16)
    hs = np.zeros(ts, dtype=np.bool_)
    ib = np.ones(ts, dtype=np.bool_)
    for t in range(sv):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
        ib[t] = False
        if sp.is_byte(t): bb[t] = 1; continue
        p = sp.id_to_piece(t)
        if p.startswith("\u2581"): hs[t] = True; p = p[1:]
        bb[t] = len(p.encode("utf-8"))
    return torch.tensor(bb, dtype=torch.int16), torch.tensor(hs, dtype=torch.bool), torch.tensor(ib, dtype=torch.bool)

bb_lut, hs_lut, ib_lut = build_bpb_luts(sp, vocab_size)

# ---- Shared components ----
class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, d, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0/(base**(torch.arange(0,d,2,dtype=torch.float32)/d)), persistent=False)
    def forward(self, T, dev, dt):
        t = torch.arange(T, device=dev, dtype=self.inv_freq.dtype)
        f = torch.outer(t, self.inv_freq.to(dev))
        return f.cos()[None,None,:,:].to(dt), f.sin()[None,None,:,:].to(dt)

def apply_rot(x, c, s):
    h = x.size(-1)//2
    x1,x2 = x[...,:h], x[...,h:]
    return torch.cat((x1*c+x2*s, x1*(-s)+x2*c), dim=-1)

# ---- MODEL A: Baseline (ReLU^2, no extras) ----
class AttnA(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh, self.nkv, self.hd = 8, 4, dim//8
        kvd = self.nkv * self.hd
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kvd, bias=False)
        self.c_v = nn.Linear(dim, kvd, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rot = Rotary(self.hd)
    def forward(self, x):
        B,T,C = x.shape
        q = self.c_q(x).reshape(B,T,self.nh,self.hd).transpose(1,2)
        k = self.c_k(x).reshape(B,T,self.nkv,self.hd).transpose(1,2)
        v = self.c_v(x).reshape(B,T,self.nkv,self.hd).transpose(1,2)
        q,k = F.rms_norm(q,(q.size(-1),)), F.rms_norm(k,(k.size(-1),))
        c,s = self.rot(T, x.device, q.dtype)
        q,k = apply_rot(q,c,s), apply_rot(k,c,s)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
        return self.proj(y.transpose(1,2).contiguous().reshape(B,T,C))

class MLPA(nn.Module):
    def __init__(self, mult=2):
        super().__init__()
        self.fc = nn.Linear(dim, dim*mult, bias=False)
        self.proj = nn.Linear(dim*mult, dim, bias=False)
    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())

class BlockA(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1,self.n2 = RMSNorm(), RMSNorm()
        self.attn, self.mlp = AttnA(), MLPA()
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

class ModelA(nn.Module):
    def __init__(self, n_layers=9):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([BlockA() for _ in range(n_layers)])
        self.norm = RMSNorm()
        nn.init.normal_(self.emb.weight, std=0.005)
    def forward(self, idx, tgt):
        x = F.rms_norm(self.emb(idx), (dim,))
        for b in self.blocks: x = b(x)
        logits = F.linear(self.norm(x), self.emb.weight)
        return F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))

# ---- MODEL B: Improved (LeakyReLU^2, SmearGate, OrthoInit, 3x MLP) ----
class SmearGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), 3.0))
    def forward(self, x):
        g = torch.sigmoid(self.gate)[None,None,:]
        xp = torch.cat([torch.zeros_like(x[:,:1,:]), x[:,:-1,:]], dim=1)
        return g * x + (1-g) * xp

class MLPB(nn.Module):
    def __init__(self, mult=3):
        super().__init__()
        self.fc = nn.Linear(dim, dim*mult, bias=False)
        self.proj = nn.Linear(dim*mult, dim, bias=False)
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), 0.5).square())

class BlockB(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1,self.n2 = RMSNorm(), RMSNorm()
        self.attn, self.mlp = AttnA(), MLPB()
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

class ModelB(nn.Module):
    def __init__(self, n_layers=9):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.smear = SmearGate()
        self.blocks = nn.ModuleList([BlockB() for _ in range(n_layers)])
        self.norm = RMSNorm()
        nn.init.normal_(self.emb.weight, std=0.005)
        # OrthoInit on all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.ndim == 2:
                nn.init.orthogonal_(m.weight)
    def forward(self, idx, tgt):
        x = self.emb(idx)
        x = self.smear(x)
        x = F.rms_norm(x, (dim,))
        for b in self.blocks: x = b(x)
        logits = F.linear(self.norm(x), self.emb.weight)
        return F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))

# ---- Evaluate ----
def evaluate(model, max_seqs=50):
    model.eval()
    usable = ((val_tokens.numel()-1)//seq_len)*seq_len
    tokens = val_tokens[:usable+1]
    n = min(usable//seq_len, max_seqs)
    ls, tc, bc = 0.0, 0, 0
    with torch.no_grad():
        for i in range(n):
            c = tokens[i*seq_len:i*seq_len+seq_len+1].long()
            x, y = c[:-1].unsqueeze(0), c[1:].unsqueeze(0)
            ls += model(x,y).item() * seq_len
            tc += seq_len
            tb = bb_lut[c[1:]].to(torch.int16)
            tb += (hs_lut[c[1:]] & ~ib_lut[c[:-1]]).to(torch.int16)
            bc += tb.sum().item()
    vl = ls/tc
    model.train()
    return vl, (vl/math.log(2.0)) * (tc/bc)

# ---- Train & Compare ----
def train_and_eval(name, model):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f"  {name}: {n_params:,} params")
    print(f"{'='*50}")

    pre_loss, pre_bpb = evaluate(model)
    print(f"  Pre-train:  val_bpb = {pre_bpb:.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    pos = 0
    t0 = time.time()
    for step in range(1, num_steps+1):
        if pos + batch_tokens + 1 > train_tokens.numel(): pos = 0
        c = train_tokens[pos:pos+batch_tokens+1].long()
        pos += batch_tokens
        x, y = c[:-1].reshape(-1,seq_len), c[1:].reshape(-1,seq_len)
        opt.zero_grad()
        loss = model(x, y)
        loss.backward()
        opt.step()
        if step <= 3 or step % 10 == 0:
            print(f"  step {step:3d}: loss={loss.item():.4f}")
    elapsed = time.time() - t0

    post_loss, post_bpb = evaluate(model)
    print(f"  Post-train: val_bpb = {post_bpb:.4f}  ({elapsed:.1f}s)")
    return post_bpb

print(f"=== A/B COMPARISON: {num_steps} steps, seq_len={seq_len} ===")
print(f"Data: {len(train_files)} train shards, {val_tokens.numel():,} val tokens")

bpb_a = train_and_eval("MODEL A: Baseline (ReLU^2, 2x MLP, 9L)", ModelA(9))
bpb_b = train_and_eval("MODEL B: Improved (LeakyReLU^2, 3x MLP, SmearGate, OrthoInit, 9L)", ModelB(9))

print(f"\n{'='*50}")
print(f"  RESULTS after {num_steps} steps:")
print(f"  Model A (baseline): {bpb_a:.4f} BPB")
print(f"  Model B (improved): {bpb_b:.4f} BPB")
print(f"  Difference: {bpb_a - bpb_b:.4f} BPB ({'B wins' if bpb_b < bpb_a else 'A wins'})")
print(f"{'='*50}")
