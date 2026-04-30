"""
Sliding Window Evaluation — the biggest FREE BPB improvement (~0.034).
Instead of evaluating with non-overlapping seq_len chunks, we slide a window
with stride=64 and only score the last 64 tokens per window. This gives
every scored token ~960 tokens of context instead of 0-1023 average.

Usage: CUDA_VISIBLE_DEVICES=0 EVAL_TEMP=0.90 python sliding_window_eval.py best_model_8B.pt
"""
import os, sys, time, math, glob, numpy as np
from pathlib import Path
import torch, torch.nn.functional as F, sentencepiece as spm
from torch import nn

model_path = sys.argv[1] if len(sys.argv) > 1 else 'best_model_8B.pt'
TEMP = float(os.environ.get('EVAL_TEMP', '1.0'))
STRIDE = int(os.environ.get('EVAL_STRIDE', '64'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim, sl, vs = 512, 1024, 1024

print(f'Sliding Window Eval: {model_path}', flush=True)
print(f'Temperature: {TEMP}, Stride: {STRIDE}', flush=True)
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

# Model (must match training architecture — supports both 9L and 11L)
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

# Auto-detect model size from state dict
state = torch.load(model_path, map_location='cpu')
n_blocks = sum(1 for k in state if k.startswith('blocks.') and k.endswith('.n1.eps'))
# Infer MLP mult from fc weight shape
fc_key = [k for k in state if 'fc.weight' in k][0]
mlp_mult = state[fc_key].shape[0] // dim
print(f'Detected: {n_blocks}L {mlp_mult}xMLP', flush=True)

model = GPT(n_blocks, mlp_mult).to(device)
model.load_state_dict(state, strict=False)
model.eval()
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params', flush=True)

# Sliding window evaluation
total_tokens = val_tokens.numel() - 1
positions = list(range(0, total_tokens - sl + 1, STRIDE))
print(f'Windows: {len(positions):,} (stride={STRIDE})', flush=True)

loss_sum = 0.0
token_count = 0
byte_count = 0
t0 = time.time()

with torch.no_grad():
    for wi, pos in enumerate(positions):
        chunk = val_tokens[pos : pos + sl + 1].to(device=device, dtype=torch.long)
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:]

        logits = model(x).squeeze(0) / TEMP
        per_token_loss = F.cross_entropy(logits.float(), y, reduction='none')

        # Only score the last STRIDE tokens
        scored = per_token_loss[-STRIDE:]
        loss_sum += scored.sum().item()
        token_count += STRIDE

        score_start = sl - STRIDE
        prev = chunk[score_start : score_start + STRIDE]
        tgt = chunk[score_start + 1 : score_start + STRIDE + 1]
        tb = bb_l[tgt].to(torch.int16)
        tb += (hs_l[tgt] & ~ib_l[prev]).to(torch.int16)
        byte_count += tb.sum().item()

        if (wi + 1) % 5000 == 0:
            elapsed = time.time() - t0
            curr_bpb = (loss_sum / token_count / math.log(2)) * (token_count / byte_count)
            print(f'  {wi+1}/{len(positions)}: val_bpb={curr_bpb:.4f} ({elapsed:.0f}s)', flush=True)

val_loss = loss_sum / token_count
bpt = val_loss / math.log(2)
tpb = token_count / byte_count
val_bpb = bpt * tpb

print(f'\n=== SLIDING WINDOW EVAL (stride={STRIDE}, T={TEMP}) ===', flush=True)
print(f'val_loss: {val_loss:.6f}', flush=True)
print(f'val_bpb:  {val_bpb:.6f}', flush=True)
print(f'Target:   1.2244', flush=True)
print(f'Time:     {time.time()-t0:.0f}s', flush=True)
print(f'Windows:  {len(positions):,}', flush=True)
