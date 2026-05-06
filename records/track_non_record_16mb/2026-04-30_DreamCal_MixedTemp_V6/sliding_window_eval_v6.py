"""
Sliding Window Evaluation for V6 (SP4096) models.
Slides a window with configurable stride, scoring only the last STRIDE tokens
per window. Every scored token gets (seq_len - stride) tokens of context.

Architecture matches train_v6.py exactly:
  BigramHash + SmearGate + Partial RoPE + XSA + LeakyReLU(0.5)^2 + ln_scale

Usage: python sliding_window_eval_v6.py best_model_v6_ema.pt [--gpu 0] [--stride 64] [--temp 1.0]
"""
import os, sys, time, math, glob, argparse, numpy as np
from pathlib import Path
import torch, torch.nn.functional as F, sentencepiece as spm
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('model', help='Model checkpoint path')
parser.add_argument('--gpu', type=int, default=0, help='GPU index (-1 for CPU)')
parser.add_argument('--stride', type=int, default=64, help='Sliding window stride (score last N tokens per window)')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature scaling')
parser.add_argument('--seq-len', type=int, default=512, help='Sequence length (context window)')
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
dim = 512
ROPE_DIMS = 16
BIGRAM_VOCAB = 3072
BIGRAM_DIM = 112

print(f'Sliding Window Eval V6: {args.model}', flush=True)
print(f'Device: {device}, Stride: {args.stride}, Temp: {args.temp}, SeqLen: {args.seq_len}', flush=True)

# Load val data (SP4096)
val_files = sorted(glob.glob('data/datasets/fineweb10B_sp4096/fineweb_val_*.bin'))
if not val_files:
    print('ERROR: No SP4096 val files found'); sys.exit(1)
val_tokens = torch.cat([torch.from_numpy(np.fromfile(Path(f), dtype='<u2', offset=256*4).astype(np.uint16)) for f in val_files])
print(f'Val tokens: {val_tokens.numel():,}', flush=True)

# BPB LUTs (SP4096)
sp_model = 'data/tokenizers/fineweb_4096_bpe.model'
if not os.path.exists(sp_model):
    for alt in ['data/sp4096.model', 'sp4096.model']:
        if os.path.exists(alt): sp_model = alt; break
sp = spm.SentencePieceProcessor(model_file=sp_model)
sv = int(sp.vocab_size())
vs = max(sv, 4096)

bb = np.zeros(vs, dtype=np.int16)
hs = np.zeros(vs, dtype=np.bool_)
ib = np.ones(vs, dtype=np.bool_)
for t in range(sv):
    if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
    ib[t] = False
    if sp.is_byte(t): bb[t] = 1; continue
    p = sp.id_to_piece(t)
    if p.startswith('\u2581'): hs[t] = True; p = p[1:]
    bb[t] = len(p.encode('utf-8'))
bb_l = torch.tensor(bb, dtype=torch.int16, device=device)
hs_l = torch.tensor(hs, dtype=torch.bool, device=device)
ib_l = torch.tensor(ib, dtype=torch.bool, device=device)

# --- RoPE (Partial: only first ROPE_DIMS of 64 head dims) ---
def build_rope(seq_len, head_dim, rope_dims, base=10000.0):
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = 1.0 / (base ** (torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims))
    angles = pos[:, None] * freqs[None, :]
    cos_cache = torch.cos(angles)
    sin_cache = torch.sin(angles)
    return cos_cache.to(device), sin_cache.to(device)

def apply_partial_rope(x, cos, sin, rope_dims):
    B, nh, T, hd = x.shape
    x_rope = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    x1 = x_rope[..., 0::2]
    x2 = x_rope[..., 1::2]
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    o1 = x1 * cos_t - x2 * sin_t
    o2 = x2 * cos_t + x1 * sin_t
    x_rotated = torch.stack([o1, o2], dim=-1).flatten(-2)
    return torch.cat([x_rotated, x_pass], dim=-1)

rope_cos, rope_sin = build_rope(args.seq_len, dim // 8, ROPE_DIMS)

# --- SmearGate ---
class SmearGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(d, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

# --- BigramHash ---
class BigramHash(nn.Module):
    def __init__(self, vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, bigram_dim)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        nn.init.zeros_(self.embed.weight)
        nn.init.zeros_(self.proj.weight)
    def forward(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        h = self.embed(out.long())
        h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

# --- Model ---
class RMSNorm(nn.Module):
    def __init__(self, d): super().__init__(); self.eps = 1e-6
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Block(nn.Module):
    def __init__(self, d, mm, layer_idx, n_layers, nh=8):
        super().__init__()
        self.layer_idx = layer_idx
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.q = CastedLinear(d, d, bias=False)
        self.k = CastedLinear(d, d//2, bias=False)
        self.v = CastedLinear(d, d//2, bias=False)
        self.o = CastedLinear(d, d, bias=False)
        self.fc = CastedLinear(d, d*mm, bias=False)
        self.proj = CastedLinear(d*mm, d, bias=False)
        self.nh, self.hd = nh, d // nh
        self.attn_scale = nn.Parameter(torch.ones(d))
        self.mlp_scale = nn.Parameter(torch.ones(d))
        self.q_gain = nn.Parameter(torch.full((nh,), 5.0))
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)

    def forward(self, x):
        B, T, C = x.shape
        h = self.n1(x) * self.ln_scale

        q = self.q(h).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).reshape(B, T, self.nh//2, self.hd).transpose(1, 2)
        v = self.v(h).reshape(B, T, self.nh//2, self.hd).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = q * self.q_gain[None, :, None, None]

        q = apply_partial_rope(q, rope_cos, rope_sin, ROPE_DIMS)
        k = apply_partial_rope(k, rope_cos, rope_sin, ROPE_DIMS)

        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        # XSA: project out self-value from attention output (GQA-aware)
        y = a.transpose(1, 2)
        v_t = v.transpose(1, 2)
        Hkv = v_t.size(2)
        group = self.nh // Hkv
        y_g = y.reshape(B, T, Hkv, group, self.hd)
        vn = F.normalize(v_t, dim=-1).unsqueeze(3)
        proj_xsa = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        y = (y_g - proj_xsa).reshape(B, T, self.nh, self.hd)

        attn_out = self.o(y.contiguous().reshape(B, T, C))
        x = x + self.attn_scale * attn_out

        h2 = self.n2(x) * self.ln_scale
        x = x + self.mlp_scale * self.proj(F.leaky_relu(self.fc(h2), negative_slope=0.5).square())
        return x

class GPT(nn.Module):
    def __init__(self, nl, mm):
        super().__init__()
        self.emb = nn.Embedding(vs, dim)
        self.bigram = BigramHash(BIGRAM_VOCAB, BIGRAM_DIM, dim)
        self.smear = SmearGate(dim)
        self.blocks = nn.ModuleList([Block(dim, mm, i, nl) for i in range(nl)])
        self.ln = RMSNorm(dim)
        self.n_enc = nl // 2
        self.n_dec = nl - self.n_enc
        self.skip_weights = nn.Parameter(torch.ones(min(self.n_enc, self.n_dec), dim))

    def forward(self, idx):
        x = F.rms_norm(self.emb(idx), (dim,))
        x = x + self.bigram(idx)
        x = self.smear(x)

        skips = []
        for i in range(self.n_enc):
            x = self.blocks[i](x); skips.append(x)
        for i in range(self.n_dec):
            if skips: x = x + self.skip_weights[i] * skips.pop()
            x = self.blocks[self.n_enc + i](x)
        logits = F.linear(self.ln(x), self.emb.weight)
        return 30.0 * torch.tanh(logits / 30.0)

# Auto-detect model architecture
state = torch.load(args.model, map_location='cpu')
block_nums = set(int(k.split('.')[1]) for k in state if k.startswith('blocks.'))
n_blocks = len(block_nums)
fc_key = [k for k in state if 'fc.weight' in k and 'blocks.' in k][0]
mlp_mult = state[fc_key].shape[0] // dim
emb_key = [k for k in state if k == 'emb.weight'][0]
vs = state[emb_key].shape[0]
print(f'Architecture: {n_blocks}L {mlp_mult}xMLP {dim}d, vocab={vs}', flush=True)

model = GPT(n_blocks, mlp_mult)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f'WARNING: Missing keys: {missing}', flush=True)
if unexpected:
    print(f'WARNING: Unexpected keys: {unexpected}', flush=True)
model = model.to(device)
if device.type == 'cuda':
    model = model.bfloat16()
else:
    model = model.float()
model.eval()
print(f'Params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

# Sliding window evaluation
sl = args.seq_len
stride = args.stride
total_tokens = val_tokens.numel() - 1
positions = list(range(0, total_tokens - sl + 1, stride))
print(f'Windows: {len(positions):,} (stride={stride}, context={sl - stride})', flush=True)

loss_sum = 0.0
token_count = 0
byte_count = 0
t0 = time.time()

with torch.no_grad():
    for wi, pos in enumerate(positions):
        chunk = val_tokens[pos : pos + sl + 1].to(device=device, dtype=torch.long)
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:]

        if device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x).squeeze(0) / args.temp
        else:
            logits = model(x).squeeze(0) / args.temp

        per_token_loss = F.cross_entropy(logits.float(), y, reduction='none')

        # Only score the last STRIDE tokens
        scored = per_token_loss[-stride:]
        loss_sum += scored.sum().item()
        token_count += stride

        # BPB: count bytes for scored tokens
        score_start = sl - stride
        prev = chunk[score_start : score_start + stride]
        tgt = chunk[score_start + 1 : score_start + stride + 1]
        tb = bb_l[tgt].to(torch.int16)
        tb += (hs_l[tgt] & ~ib_l[prev]).to(torch.int16)
        byte_count += tb.sum().item()

        if (wi + 1) % 5000 == 0:
            elapsed = time.time() - t0
            curr_bpb = (loss_sum / token_count / math.log(2)) * (token_count / byte_count)
            print(f'  [{wi+1}/{len(positions)}] bpb={curr_bpb:.4f} ({elapsed:.0f}s)', flush=True)

val_loss = loss_sum / token_count
bpt = val_loss / math.log(2)
tpb = token_count / byte_count
val_bpb = bpt * tpb

elapsed = time.time() - t0
print(f'\n{"="*60}', flush=True)
print(f'SLIDING WINDOW EVAL (stride={stride}, T={args.temp}, seq_len={sl})', flush=True)
print(f'  val_bpb:  {val_bpb:.6f}', flush=True)
print(f'  val_loss: {val_loss:.6f}', flush=True)
print(f'  Target:   1.0897 (pure train SOTA)', flush=True)
print(f'  Tokens scored: {token_count:,}', flush=True)
print(f'  Bytes:    {byte_count:,}', flush=True)
print(f'  Time:     {elapsed:.0f}s', flush=True)
print(f'{"="*60}', flush=True)
