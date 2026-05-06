"""
Depth-Recurrent Training — inspired by PR #1331 (1.0900 BPB SOTA).
Key idea: 11 physical layers with layers 3-5 looped N times = massive effective depth.
Int6 STE QAT so the model fits in 16MB after quantization.

Architecture: 11 physical layers, 2x MLP (20.7M params)
- Layers 0-2: encoder (unique, run once)
- Layers 3-5: recurrent core (looped LOOP_ITERS times)
- Layers 6-10: decoder (unique, run once)
- Effective depth: 3 + 3*LOOP_ITERS + 5 layers

At int6 + bit-packing: ~15.5MB artifact (fits 16MB budget!)

Usage: CUDA_VISIBLE_DEVICES=1 python train_depth_recurrent.py
  Env vars: STEPS=50000, LOOP_ITERS=6, QAT_START=0.15, N_LAYERS=11, MLP_MULT=2
"""
import os, time, math, glob, numpy as np, sys
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch, torch.nn.functional as F, sentencepiece as spm
from torch import nn, Tensor
torch.backends.cuda.matmul.allow_tf32 = True

STEPS = int(os.environ.get('STEPS', 50000))
LOOP_ITERS = int(os.environ.get('LOOP_ITERS', 2))   # PR #1331 uses 1 extra iteration (14 total from 11)
LOOP_START = int(os.environ.get('LOOP_START', 3))    # First looping layer
LOOP_END = int(os.environ.get('LOOP_END', 6))        # Last looping layer (exclusive)
RECUR_ACTIVATE_STEP = int(os.environ.get('RECUR_STEP', 3000))  # PR #1331: activate recurrence at step 3000
RECUR_WARMUP_STEPS = int(os.environ.get('RECUR_WARMUP', 20))   # Warmup for recurrence gates
QAT_START_FRAC = float(os.environ.get('QAT_START', '0.15'))    # Enable QAT when LR drops below this fraction
WEIGHT_DECAY = float(os.environ.get('WD', '0.095'))  # PR #1331: higher WD for compression
VOCAB_SIZE = int(os.environ.get('VOCAB_SIZE', '4096'))  # SP4096 by default (all top PRs use it)
dim, sl, vs = 512, 1024, VOCAB_SIZE
device = torch.device('cuda')

# --- Muon Optimizer ---
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

USE_MUONEQ_R = bool(int(os.environ.get('MUONEQ_R', '1')))  # MuonEq-R (arXiv:2603.28254) — row normalization

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, backend_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                # Weight decay (decoupled, applied before Muon update)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                g = p.grad
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)
                # MuonEq-R: row-normalize gradients before NS orthogonalization
                # This equalizes the scale across output dimensions (arXiv:2603.28254)
                if USE_MUONEQ_R and g.ndim == 2:
                    row_norms = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    g = g / row_norms
                g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g.to(p.dtype), alpha=-lr)

# --- Int6 STE QAT ---
_qat_active = False

class FakeInt6(torch.autograd.Function):
    """Fake-quantize to int6 range [-31,31]. Gradients pass through (STE)."""
    @staticmethod
    def forward(ctx, x):
        if x.ndim == 2:
            amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            s = amax / 31.0
            return (torch.clamp(torch.round(x / s), -31, 31) * s).to(x.dtype)
        amax = x.abs().max().clamp_min(1e-12)
        s = amax / 31.0
        return (torch.clamp(torch.round(x / s), -31, 31) * s).to(x.dtype)
    @staticmethod
    def backward(ctx, g):
        return g

def fake_int6(x):
    return FakeInt6.apply(x)

# --- Data (streaming) ---
print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
_data_variant = f'sp{VOCAB_SIZE}'
train_files = sorted(glob.glob(f'data/datasets/fineweb10B_{_data_variant}/fineweb_train_*.bin'))
print(f'Train: {len(train_files)} shards available', flush=True)

current_shard_idx = 0
def load_shard(idx):
    f = train_files[idx % len(train_files)]
    h = np.fromfile(f, dtype='<i4', count=256)
    return torch.from_numpy(np.fromfile(f, dtype='<u2', count=int(h[2]), offset=256*4).astype(np.uint16))

train_tokens = load_shard(0)
print(f'Loaded shard 0: {train_tokens.numel():,} tokens', flush=True)

val_files = sorted(glob.glob(f'data/datasets/fineweb10B_{_data_variant}/fineweb_val_*.bin'))
val_tokens = torch.cat([torch.from_numpy(np.fromfile(Path(f), dtype='<u2', offset=256*4).astype(np.uint16)) for f in val_files])

sp = spm.SentencePieceProcessor(model_file=f'data/tokenizers/fineweb_{VOCAB_SIZE}_bpe.model')
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

# Byte weights for BPB-aware training (novel #25: high-byte tokens matter more for BPB)
byte_weights = torch.tensor(bb, dtype=torch.float32, device=device)
byte_weights = byte_weights / byte_weights[byte_weights > 0].mean()  # normalize so mean=1
byte_weights = byte_weights.clamp(min=0.1)
USE_BYTE_WEIGHTED = bool(int(os.environ.get('BYTE_WEIGHTED', '1')))  # ON by default
FOCAL_GAMMA = float(os.environ.get('FOCAL_GAMMA', '0.0'))  # 0=off, 1-2=moderate focal loss
if USE_BYTE_WEIGHTED:
    print(f'Byte-weighted loss: ON (5-byte tokens get {byte_weights[byte_weights>1.5].mean():.1f}x weight)', flush=True)
if FOCAL_GAMMA > 0:
    print(f'Focal loss: gamma={FOCAL_GAMMA}', flush=True)

# --- Model ---
class RMSNorm(nn.Module):
    def __init__(self, d): super().__init__(); self.eps = 1e-6
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if _qat_active:
            w = fake_int6(w)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

PARALLEL_RESIDUAL_START = int(os.environ.get('PARALLEL_START', 5))  # PR #1334: parallel from layer 7
QK_GAIN_INIT = float(os.environ.get('QK_GAIN', '5.0'))  # PR #1334: 5.0 (much higher than default 1.5!)

class Block(nn.Module):
    def __init__(self, d, mm, layer_idx=0):
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
        self.q_gain = nn.Parameter(torch.full((8,), QK_GAIN_INIT))
        self.parallel = layer_idx >= PARALLEL_RESIDUAL_START  # PaLM-style parallel attn+MLP

    def forward(self, x):
        B,T,C = x.shape; h = self.n1(x)
        q = self.q(h).reshape(B,T,self.nh,self.hd).transpose(1,2)
        k = self.k(h).reshape(B,T,self.nh//2,self.hd).transpose(1,2)
        v = self.v(h).reshape(B,T,self.nh//2,self.hd).transpose(1,2)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        q = q * self.q_gain[None,:,None,None]
        a = F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
        attn_out = self.attn_scale * self.o(a.transpose(1,2).contiguous().reshape(B,T,C))
        if self.parallel:
            # Parallel residual: attn and MLP computed from SAME input (PaLM-style, -0.007 BPB)
            mlp_out = self.mlp_scale * self.proj(F.leaky_relu(self.fc(self.n2(x)), negative_slope=0.5).square())
            x = x + attn_out + mlp_out
        else:
            # Sequential residual (standard)
            x = x + attn_out
            x = x + self.mlp_scale * self.proj(F.leaky_relu(self.fc(self.n2(x)), negative_slope=0.5).square())
        return x

class DepthRecurrentGPT(nn.Module):
    """
    GPT with depth recurrence: layers [LOOP_START:LOOP_END] are looped LOOP_ITERS times.
    This gives massive effective depth with no additional parameters.

    Effective layers = (LOOP_START) + (LOOP_END-LOOP_START)*LOOP_ITERS + (nl-LOOP_END)
    For nl=11, loop 3-5, iters=6: 3 + 3*6 + 5 = 26 effective layers
    For nl=11, loop 3-5, iters=20: 3 + 3*20 + 5 = 68 effective layers (like PR #1331)
    """
    def __init__(self, nl, mm, loop_start, loop_end, loop_iters):
        super().__init__()
        self.emb = nn.Embedding(vs, dim)
        self.blocks = nn.ModuleList([Block(dim, mm, layer_idx=i) for i in range(nl)])
        self.ln = RMSNorm(dim)
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.loop_iters = loop_iters

        # Per-iteration learnable gate for loop layers (controls contribution per loop)
        n_loop = loop_end - loop_start
        self.loop_gates = nn.Parameter(torch.ones(loop_iters, n_loop))

        # U-Net skip connections on the non-looping layers
        n_enc = loop_start  # encoder = pre-loop layers
        n_dec = nl - loop_end  # decoder = post-loop layers
        n_skip = min(n_enc, n_dec)
        self.skip_weights = nn.Parameter(torch.ones(n_skip, dim))
        self.n_enc = n_enc
        self.n_dec = n_dec
        self.n_skip = n_skip

        # SVD embedding initialization (novel #18: bigram co-occurrence SVD)
        svd_path = 'data/svd_embeddings_512.npy'
        if os.path.exists(svd_path):
            import numpy as _np
            svd_emb = torch.from_numpy(_np.load(svd_path)).float()
            self.emb.weight.data[:svd_emb.shape[0], :svd_emb.shape[1]] = svd_emb
            print(f'Initialized embeddings from SVD ({svd_path})', flush=True)
        else:
            nn.init.normal_(self.emb.weight, std=0.005)

        # Recurrence is activated later in training (PR #1331: at step 3000)
        self.recurrence_active = False
        self.recurrence_scale = 1.0  # Warmup: ramps from 0 to 1

        eff_depth_no_recur = nl
        eff_depth_recur = loop_start + n_loop * (1 + loop_iters) + (nl - loop_end)
        print(f'DepthRecurrentGPT: {nl}L {mm}xMLP, loop [{loop_start}:{loop_end}] x{loop_iters}', flush=True)
        print(f'Effective depth: {eff_depth_no_recur} (before activation) / {eff_depth_recur} (after)', flush=True)
        print(f'Physical params: {sum(p.numel() for p in self.parameters()):,}', flush=True)

    def forward(self, idx, tgt=None):
        x = F.rms_norm(self.emb(idx), (dim,))

        # Phase 1: Encoder (pre-loop layers)
        skips = []
        for i in range(self.loop_start):
            x = self.blocks[i](x)
            if i < self.n_skip:
                skips.append(x)

        # Phase 2: Run loop layers once (always)
        for layer_idx in range(self.loop_start, self.loop_end):
            x = self.blocks[layer_idx](x)

        # Phase 2b: Recurrent core — extra iterations (only when recurrence is active)
        if self.recurrence_active:
            for loop_iter in range(self.loop_iters):
                for j, layer_idx in enumerate(range(self.loop_start, self.loop_end)):
                    gate = self.loop_gates[loop_iter, j] * self.recurrence_scale
                    residual = x
                    x = self.blocks[layer_idx](x)
                    x = residual + gate * (x - residual)

        # Phase 3: Decoder (post-loop layers) with U-Net skips
        for i in range(self.n_dec):
            layer_idx = self.loop_end + i
            if i < self.n_skip and skips:
                x = x + self.skip_weights[i] * skips.pop()
            x = self.blocks[layer_idx](x)

        logits = F.linear(self.ln(x), self.emb.weight)
        logits = 30.0 * torch.tanh(logits / 30.0)

        if tgt is not None:
            logits_flat = logits.float().view(-1, vs)
            tgt_flat = tgt.view(-1)
            if USE_BYTE_WEIGHTED or FOCAL_GAMMA > 0:
                per_token_loss = F.cross_entropy(logits_flat, tgt_flat, reduction='none')
                weights = torch.ones_like(per_token_loss)
                if USE_BYTE_WEIGHTED:
                    weights = weights * byte_weights[tgt_flat]
                if FOCAL_GAMMA > 0:
                    # Focal loss: down-weight easy tokens, up-weight hard ones
                    pt = torch.exp(-per_token_loss)  # probability of correct token
                    weights = weights * (1 - pt).pow(FOCAL_GAMMA)
                return (per_token_loss * weights).mean()
            return F.cross_entropy(logits_flat, tgt_flat)
        return logits

def eval_bpb(model, max_seqs=300):
    model.eval()
    usable = ((val_tokens.numel()-1)//sl)*sl
    toks = val_tokens[:usable+1].to(device=device, dtype=torch.long)
    n = min(usable//sl, max_seqs); ls=0.0; tc=0; bc=0
    with torch.no_grad():
        for i in range(n):
            c = toks[i*sl:i*sl+sl+1]
            ls += model(c[:-1].unsqueeze(0), c[1:].unsqueeze(0)).item()*sl; tc += sl
            tb = bb_l[c[1:]].to(torch.int16)
            tb += (hs_l[c[1:]] & ~ib_l[c[:-1]]).to(torch.int16)
            bc += tb.sum().item()
    model.train()
    return (ls/tc/math.log(2.0)) * (tc/bc)

# --- Setup ---
N_LAYERS = int(os.environ.get('N_LAYERS', 11))
MLP_MULT = int(os.environ.get('MLP_MULT', 2))

model = DepthRecurrentGPT(N_LAYERS, MLP_MULT, LOOP_START, LOOP_END, LOOP_ITERS).to(device)
params = sum(p.numel() for p in model.parameters())

# Param split: Muon for 2D block weights, Adam for rest
matrix_params = [p for n, p in model.named_parameters() if p.ndim == 2 and 'blocks.' in n]
other_params = [p for n, p in model.named_parameters() if p.ndim < 2 or 'blocks.' not in n]

muon_opt = Muon(matrix_params, lr=0.022, momentum=0.95, backend_steps=5, weight_decay=WEIGHT_DECAY)  # PR #1331
adam_opt = torch.optim.Adam(other_params, lr=0.01, weight_decay=WEIGHT_DECAY)

GRAD_ACCUM = 16
WARMDOWN_FRAC = float(os.environ.get('WARMDOWN_FRAC', '0.3'))  # Novel #43: 30% better than 20%
PEAK_MUON_LR = 0.022  # PR #1331: slightly higher
PEAK_ADAM_LR = 0.01
MB = 8
_recurrence_active = False

print(f'Muon params: {sum(p.numel() for p in matrix_params):,}', flush=True)
print(f'Adam params: {sum(p.numel() for p in other_params):,}', flush=True)
print(f'Steps: {STEPS}, Micro batch: {MB}, Grad accum: {GRAD_ACCUM}, Eff batch: {MB*GRAD_ACCUM} seqs = {MB*GRAD_ACCUM*sl:,} tok', flush=True)
print(f'Warmdown: last {WARMDOWN_FRAC*100:.0f}%, QAT activates at LR frac < {QAT_START_FRAC}', flush=True)

# Estimate artifact size
int6_bytes = params * 6 / 8
print(f'Estimated int6 artifact: {int6_bytes/1e6:.2f} MB (limit: 16 MB)', flush=True)

# --- Training ---
pos = 0; t0 = time.time(); best_bpb = 999
for step in range(1, STEPS+1):
    # Cosine warmdown + QAT activation
    lr_frac = 1.0
    if step > STEPS * (1 - WARMDOWN_FRAC):
        progress = (step - STEPS * (1 - WARMDOWN_FRAC)) / max(int(STEPS * WARMDOWN_FRAC), 1)
        lr_frac = 0.5 * (1 + math.cos(math.pi * progress))
        for g in muon_opt.param_groups: g['lr'] = PEAK_MUON_LR * lr_frac
        for g in adam_opt.param_groups: g['lr'] = PEAK_ADAM_LR * lr_frac

    # Recurrence activation (PR #1331: at step 3000 with 20-step warmup)
    if not _recurrence_active and step >= RECUR_ACTIVATE_STEP:
        _recurrence_active = True
        model.recurrence_active = True
        eff = LOOP_START + (LOOP_END - LOOP_START) * (1 + LOOP_ITERS) + (N_LAYERS - LOOP_END)
        print(f'  *** RECURRENCE ACTIVATED at step {step}: {eff} effective layers ***', flush=True)
    if model.recurrence_active:
        # Warmup: ramp recurrence scale from 0 to 1 over RECUR_WARMUP_STEPS
        warmup_progress = min(1.0, (step - RECUR_ACTIVATE_STEP) / max(RECUR_WARMUP_STEPS, 1))
        model.recurrence_scale = warmup_progress

    # Late QAT activation
    if not _qat_active and lr_frac < QAT_START_FRAC:
        _qat_active = True
        print(f'  *** QAT ACTIVATED at step {step} (lr_frac={lr_frac:.3f}) ***', flush=True)

    muon_opt.zero_grad(); adam_opt.zero_grad()
    accum_loss = 0.0
    for _ in range(GRAD_ACCUM):
        n = MB*sl+1
        if pos+n > train_tokens.numel():
            current_shard_idx = (current_shard_idx + 1) % len(train_files)
            train_tokens = load_shard(current_shard_idx)
            pos = 0
        c = train_tokens[pos:pos+n].to(device=device, dtype=torch.long); pos += MB*sl
        loss = model(c[:-1].reshape(MB, sl), c[1:].reshape(MB, sl)) / GRAD_ACCUM
        loss.backward()
        accum_loss += loss.item()
    muon_opt.step(); adam_opt.step()

    if step in (1, 5, 10, 50, 100, 200, 500) or step % 1000 == 0 or step == STEPS:
        elapsed = time.time() - t0
        print(f'step {step:6d}/{STEPS}: loss={accum_loss:.4f} lr_frac={lr_frac:.4f} ms/step={1000*elapsed/step:.0f} elapsed={elapsed/60:.1f}min', flush=True)

    if step in (500, 1000, 2000, 3000) or step % 5000 == 0 or step == STEPS:
        bpb = eval_bpb(model)
        print(f'  >>> val_bpb = {bpb:.4f} (target: 1.2244, best: {best_bpb:.4f}) <<<', flush=True)
        if bpb < best_bpb:
            best_bpb = bpb
            torch.save(model.state_dict(), 'best_depth_recurrent.pt')
            print(f'  New best! Saved.', flush=True)
        if step >= STEPS // 2:
            ckpt_path = f'ckpt_dr_step{step}_bpb{bpb:.4f}.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f'  Checkpoint: {ckpt_path}', flush=True)

print(f'\nFINAL: best_val_bpb = {best_bpb:.4f} | time: {(time.time()-t0)/60:.1f}min', flush=True)
print(f'Model: {N_LAYERS}L {MLP_MULT}xMLP, loop [{LOOP_START}:{LOOP_END}] x{LOOP_ITERS}', flush=True)
print(f'Effective depth: {LOOP_START + (LOOP_END-LOOP_START)*LOOP_ITERS + (N_LAYERS-LOOP_END)}', flush=True)

# IW-SWA
import glob as _glob
ckpts = sorted(_glob.glob('ckpt_dr_step*_bpb*.pt'))
if len(ckpts) >= 2:
    print(f'\nIW-SWA over {len(ckpts)} checkpoints...', flush=True)
    avg_state = None; total_weight = 0.0
    for cp in ckpts:
        bpb_str = cp.split('bpb')[1].replace('.pt', '')
        weight = 1.0 / float(bpb_str)
        sd = torch.load(cp, map_location='cpu')
        if avg_state is None:
            avg_state = {k: v.float() * weight for k, v in sd.items()}
        else:
            for k in avg_state: avg_state[k] += sd[k].float() * weight
        total_weight += weight
        print(f'  {cp}: weight={weight:.4f}', flush=True)
    for k in avg_state: avg_state[k] /= total_weight
    model.load_state_dict({k: v.to(model.emb.weight.dtype) for k, v in avg_state.items()})
    model.to(device)
    swa_bpb = eval_bpb(model)
    print(f'  IW-SWA val_bpb = {swa_bpb:.4f} (vs best single: {best_bpb:.4f})', flush=True)
    if swa_bpb < best_bpb:
        torch.save({k: v.to(torch.bfloat16) for k, v in avg_state.items()}, 'best_depth_recurrent_swa.pt')
        print(f'  *** IW-SWA BEATS single best! ***', flush=True)
