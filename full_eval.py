"""
Full evaluation script — matches competition's eval methodology exactly.
Evaluates on ALL 62M val tokens with proper BPB calculation.
Usage: CUDA_VISIBLE_DEVICES=0 python full_eval.py best_model_v2.pt
"""
import os, sys, time, math, glob, numpy as np
from pathlib import Path
import torch, torch.nn.functional as F, sentencepiece as spm
from torch import nn

model_path = sys.argv[1] if len(sys.argv) > 1 else 'best_model_v2.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim, sl, vs = 512, 1024, 1024

print(f'Full evaluation of {model_path}', flush=True)
print(f'Device: {device}', flush=True)

# Load val data
val_files = sorted(glob.glob('data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
val_tokens = torch.cat([torch.from_numpy(np.fromfile(Path(f), dtype='<u2', offset=256*4).astype(np.uint16)) for f in val_files])
print(f'Val tokens: {val_tokens.numel():,}', flush=True)

# BPB LUTs
sp = spm.SentencePieceProcessor(model_file='data/tokenizers/fineweb_1024_bpe.model')
sv = int(sp.vocab_size())
bb = np.zeros(max(sv,vs), dtype=np.int16)
hs = np.zeros(max(sv,vs), dtype=np.bool_)
ib = np.ones(max(sv,vs), dtype=np.bool_)
for t in range(sv):
    if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
    ib[t] = False
    if sp.is_byte(t): bb[t]=1; continue
    p = sp.id_to_piece(t)
    if p.startswith('\u2581'): hs[t]=True; p=p[1:]
    bb[t] = len(p.encode('utf-8'))
bb_l = torch.tensor(bb, dtype=torch.int16, device=device)
hs_l = torch.tensor(hs, dtype=torch.bool, device=device)
ib_l = torch.tensor(ib, dtype=torch.bool, device=device)

# Model (must match training architecture)
class RMSNorm(nn.Module):
    def __init__(self, d): super().__init__(); self.eps = 1e-6
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Block(nn.Module):
    def __init__(self, d, mm):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q = CastedLinear(d, d, bias=False)
        self.k = CastedLinear(d, d//2, bias=False)
        self.v = CastedLinear(d, d//2, bias=False)
        self.o = CastedLinear(d, d, bias=False)
        self.fc = CastedLinear(d, d*mm, bias=False)
        self.proj = CastedLinear(d*mm, d, bias=False)
        self.nh, self.hd = 8, d//8
        self.attn_scale = nn.Parameter(torch.ones(d))
        self.mlp_scale = nn.Parameter(torch.ones(d))
        self.q_gain = nn.Parameter(torch.full((8,), 1.5))
    def forward(self, x):
        B,T,C = x.shape; h = self.n1(x)
        q = self.q(h).reshape(B,T,self.nh,self.hd).transpose(1,2)
        k = self.k(h).reshape(B,T,self.nh//2,self.hd).transpose(1,2)
        v = self.v(h).reshape(B,T,self.nh//2,self.hd).transpose(1,2)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        q = q * self.q_gain[None,:,None,None]
        a = F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
        x = x + self.attn_scale * self.o(a.transpose(1,2).contiguous().reshape(B,T,C))
        x = x + self.mlp_scale * self.proj(torch.relu(self.fc(self.n2(x))).square())
        return x

class GPT(nn.Module):
    def __init__(self, nl, mm):
        super().__init__()
        self.emb = nn.Embedding(vs, dim)
        self.blocks = nn.ModuleList([Block(dim, mm) for _ in range(nl)])
        self.ln = RMSNorm(dim)
        self.n_enc = nl // 2
        self.n_dec = nl - self.n_enc
        self.skip_weights = nn.Parameter(torch.ones(min(self.n_enc, self.n_dec), dim))
    def forward(self, idx):
        x = F.rms_norm(self.emb(idx), (dim,))
        skips = []
        for i in range(self.n_enc):
            x = self.blocks[i](x); skips.append(x)
        for i in range(self.n_dec):
            if skips: x = x + self.skip_weights[i] * skips.pop()
            x = self.blocks[self.n_enc + i](x)
        logits = F.linear(self.ln(x), self.emb.weight)
        return 30.0 * torch.tanh(logits / 30.0)

# Load model
model = GPT(9, 2).to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()
params = sum(p.numel() for p in model.parameters())
print(f'Model: {params:,} params', flush=True)

# Temperature scaling (FREE BPB improvement)
TEMP = float(os.environ.get('EVAL_TEMP', '1.0'))
print(f'Temperature: {TEMP}', flush=True)

# Full evaluation
usable = ((val_tokens.numel() - 1) // sl) * sl
tokens = val_tokens[:usable + 1].to(device=device, dtype=torch.long)
total_seqs = usable // sl
print(f'Evaluating {total_seqs:,} sequences ({usable:,} tokens)...', flush=True)

loss_sum = 0.0
token_count = 0
byte_count = 0
t0 = time.time()

with torch.no_grad():
    for i in range(total_seqs):
        c = tokens[i*sl : i*sl + sl + 1]
        x = c[:-1].unsqueeze(0)
        y = c[1:]
        logits = model(x).squeeze(0) / TEMP  # temperature scaling
        loss = F.cross_entropy(logits.float(), y, reduction='sum').item()
        loss_sum += loss
        token_count += sl

        tb = bb_l[y].to(torch.int16)
        tb += (hs_l[y] & ~ib_l[c[:-1]]).to(torch.int16)
        byte_count += tb.sum().item()

        if (i+1) % 5000 == 0:
            elapsed = time.time() - t0
            curr_bpb = (loss_sum / token_count / math.log(2)) * (token_count / byte_count)
            print(f'  {i+1}/{total_seqs}: val_bpb={curr_bpb:.4f} ({elapsed:.0f}s)', flush=True)

val_loss = loss_sum / token_count
bpt = val_loss / math.log(2)
tpb = token_count / byte_count
val_bpb = bpt * tpb

print(f'\n=== FULL EVALUATION RESULT ===', flush=True)
print(f'val_loss: {val_loss:.6f}', flush=True)
print(f'val_bpb:  {val_bpb:.6f}', flush=True)
print(f'Target:   1.2244', flush=True)
print(f'Tokens:   {token_count:,}', flush=True)
print(f'Bytes:    {byte_count:,}', flush=True)
print(f'Time:     {time.time()-t0:.0f}s', flush=True)
