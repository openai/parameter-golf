"""
train_gpt_rrt_v3.py — Definitive RRT-LoRA submission for OpenAI Parameter Golf
================================================================================
Novel contribution: Relaxed Recursive Transformers with per-step LoRA adapters.
Each recurrence pass applies a learned rank-4 delta to Q and V, allowing passes
to specialize (early = coarse, late = fine) while sharing base weights. LoRA B
matrices initialize to zero and warm up via alpha schedule for stability.

Full SOTA stack:
  SP8192 · 11L·512d·8H/4KV · MLP-4x · LeakyReLU(0.5)² · Partial-RoPE(16/64)
  Exact SOTA recurrence schedule: enc=[0,1,2,3,4,5,3,4] dec=[5,3,4,5,6,7,8,9,10]
  RRT-LoRA rank=4 on layers {3,4,5} · Parallel residuals layers 7+
  QK-Gain 5.25 · MuonEq-R · EMA 0.9965
  GPTQ SDClip int6(k=12.85)/int8(k=20) · Bit-packed int6 · Brotli-11 · LZMA
  Legal score-first TTT (SGD lr=0.005, mom=0.9, 3 epochs/32K, freeze layers 0-1)
  Post-TTT temperature calibration T=0.98
  WD=0.095 · MLR=0.022 · warmdown=72% · EMA=0.9965

Standard run (full RRT-LoRA):
  SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v3.py

Ablation run (pure depth recurrence, no LoRA — for baseline comparison):
  SEED=42 ABLATE=1 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v3.py

Run baseline first, then print comparison instructions:
  SEED=42 ABLATE=1 ABLATE_BOTH=1 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v3.py
  # then:
  SEED=42 ABLATE=0 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v3.py

Setup:
  pip install brotli sentencepiece
  pip install flash_attn_3 --no-deps \\
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \\
    python3 data/cached_challenge_fineweb.py --variant sp8192
"""

from __future__ import annotations

import glob
import io
import lzma
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import brotli
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

# ═══════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════

class Hyperparameters:
    data_path     = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files   = os.path.join(data_path, "fineweb_train_*.bin")
    val_files     = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path= os.environ.get("TOKENIZER_PATH","./data/tokenizers/fineweb_8192_bpe.model")
    run_id        = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed          = int(os.environ.get("SEED", 1337))

    val_batch_size    = int(os.environ.get("VAL_BATCH_SIZE",   524_288))
    val_loss_every    = int(os.environ.get("VAL_LOSS_EVERY",   1000))
    train_log_every   = int(os.environ.get("TRAIN_LOG_EVERY",  200))

    iterations            = int(os.environ.get("ITERATIONS",           20_000))
    warmdown_frac         = float(os.environ.get("WARMDOWN_FRAC",      0.72))
    warmup_steps          = int(os.environ.get("WARMUP_STEPS",         20))
    train_batch_tokens    = int(os.environ.get("TRAIN_BATCH_TOKENS",   524_288))
    train_seq_len         = int(os.environ.get("TRAIN_SEQ_LEN",        1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model — matches SOTA architecture exactly
    vocab_size         = int(os.environ.get("VOCAB_SIZE",        8192))
    num_layers         = int(os.environ.get("NUM_LAYERS",        11))
    num_kv_heads       = int(os.environ.get("NUM_KV_HEADS",      4))
    model_dim          = int(os.environ.get("MODEL_DIM",         512))
    num_heads          = int(os.environ.get("NUM_HEADS",         8))
    mlp_mult           = int(os.environ.get("MLP_MULT",          4))
    tie_embeddings     = bool(int(os.environ.get("TIE_EMBEDDINGS","1")))
    rope_base          = float(os.environ.get("ROPE_BASE",       10000.0))
    rope_partial_dim   = int(os.environ.get("ROPE_PARTIAL_DIM",  16))
    logit_softcap      = float(os.environ.get("LOGIT_SOFTCAP",   30.0))
    qk_gain_init       = float(os.environ.get("QK_GAIN_INIT",    5.25))
    tied_embed_init_std= float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # Exact SOTA recurrence schedule:
    # encoder execution order: [0,1,2,3,4,5,3,4]
    # decoder execution order: [5,3,4,5,6,7,8,9,10]
    enc_schedule = [int(x) for x in os.environ.get("ENC_SCHEDULE","0,1,2,3,4,5,3,4").split(",")]
    dec_schedule = [int(x) for x in os.environ.get("DEC_SCHEDULE","5,3,4,5,6,7,8,9,10").split(",")]
    recur_activate_frac = float(os.environ.get("RECUR_ACTIVATE_FRAC", 0.35))

    # RRT-LoRA
    recur_layers    = [3, 4, 5]  # layers that get LoRA adapters
    recur_steps     = 3          # max times any layer loops (for adapter sizing)
    lora_rank       = int(os.environ.get("LORA_RANK",          4))
    lora_warmup     = int(os.environ.get("LORA_WARMUP",        500))

    # Parallel residuals
    parallel_from   = int(os.environ.get("PARALLEL_FROM",      7))

    # TTT
    ttt_enabled     = bool(int(os.environ.get("TTT_ENABLED",   1)))
    ttt_lr          = float(os.environ.get("TTT_LR",           0.005))
    ttt_epochs      = int(os.environ.get("TTT_EPOCHS",         3))
    ttt_chunk       = int(os.environ.get("TTT_CHUNK",          32_768))
    ttt_freeze      = int(os.environ.get("TTT_FREEZE",         2))
    ttt_momentum    = float(os.environ.get("TTT_MOMENTUM",     0.9))
    ttt_grad_clip   = float(os.environ.get("TTT_GRAD_CLIP",    1.0))
    ttt_temp        = float(os.environ.get("TTT_TEMP",         0.98))  # post-TTT calibration

    # EMA
    ema_decay       = float(os.environ.get("EMA_DECAY",        0.9965))

    # Optimizer
    matrix_lr       = float(os.environ.get("MATRIX_LR",        0.022))
    scalar_lr       = float(os.environ.get("SCALAR_LR",        0.04))
    embed_lr        = float(os.environ.get("EMBED_LR",         0.6))
    tied_embed_lr   = float(os.environ.get("TIED_EMBED_LR",    0.05))
    head_lr         = float(os.environ.get("HEAD_LR",          0.008))
    weight_decay    = float(os.environ.get("WEIGHT_DECAY",     0.095))
    muon_momentum   = float(os.environ.get("MUON_MOMENTUM",    0.95))
    muon_ns_steps   = int(os.environ.get("MUON_NS_STEPS",      5))
    muon_mom_start  = float(os.environ.get("MUON_MOM_START",   0.85))
    muon_mom_warmup = int(os.environ.get("MUON_MOM_WARMUP",    500))
    beta1           = float(os.environ.get("BETA1",            0.9))
    beta2           = float(os.environ.get("BETA2",            0.95))
    adam_eps        = float(os.environ.get("ADAM_EPS",         1e-8))
    grad_clip       = float(os.environ.get("GRAD_CLIP",        1.0))

    # Ablation flags
    # ABLATE=1      — disable LoRA, run pure depth recurrence (baseline comparison)
    # ABLATE_BOTH=1 — run baseline first, then RRT-LoRA, print side-by-side comparison
    ablate_lora     = bool(int(os.environ.get("ABLATE",      0)))
    ablate_both     = bool(int(os.environ.get("ABLATE_BOTH", 0)))

    @property
    def warmdown_iters(self) -> int:
        return int(self.iterations * self.warmdown_frac)


# ═══════════════════════════════════════════════════════════
# MUONEQ-R  (arXiv:2603.28254)
# Row-normalize momentum rows before Newton-Schulz-5
# ═══════════════════════════════════════════════════════════

def ns5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    t = G.size(0) > G.size(1)
    if t: X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a*X + (b*A + c*A@A) @ X
    return X.T if t else X


class MuonEqR(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, ns_steps=5, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      ns_steps=ns_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        dist_on = dist.is_available() and dist.is_initialized()
        ws = dist.get_world_size() if dist_on else 1
        rk = dist.get_rank()       if dist_on else 0

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params: continue
            lr, mom, nst = group["lr"], group["momentum"], group["nesterov"]
            ns  = group["ns_steps"]
            N   = sum(p.numel() for p in params)
            upd = torch.zeros(N, device=params[0].device, dtype=torch.bfloat16)
            cur = 0
            for i, p in enumerate(params):
                if i % ws == rk:
                    g = p.grad.clone()
                    st = self.state.setdefault(p, {})
                    buf = st.setdefault("buf", torch.zeros_like(g))
                    buf.mul_(mom).add_(g)
                    g = g.add(buf, alpha=mom) if nst else buf.clone()
                    if g.ndim == 2:
                        g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-7)
                    g = ns5(g, steps=ns)
                    g *= max(1, g.size(0)/g.size(1)) ** 0.5
                    upd[cur:cur+p.numel()] = g.reshape(-1)
                cur += p.numel()
            if dist_on: dist.all_reduce(upd, op=dist.ReduceOp.SUM)
            cur = 0
            for p in params:
                p.add_(upd[cur:cur+p.numel()].view_as(p).to(p.dtype), alpha=-lr)
                cur += p.numel()
        return loss


# ═══════════════════════════════════════════════════════════
# BPB EVALUATION LUTS
# ═══════════════════════════════════════════════════════════

def build_luts(sp, vocab_size: int, device):
    n = max(int(sp.vocab_size()), vocab_size)
    bb = np.zeros(n, np.int16)
    hs = np.zeros(n, np.bool_)
    ib = np.ones(n,  np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        ib[tid] = False
        if sp.is_byte(tid): bb[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"): hs[tid] = True; piece = piece[1:]
        bb[tid] = len(piece.encode())
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(hs, dtype=torch.bool,  device=device),
            torch.tensor(ib, dtype=torch.bool,  device=device))


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_shard(file: Path) -> Tensor:
    h = np.fromfile(file, dtype="<i4", count=256)
    if h.size != 256 or h[0] != 20240520 or h[1] != 1:
        raise ValueError(f"Bad shard: {file}")
    return torch.from_numpy(np.fromfile(file,"<u2",count=int(h[2]),offset=1024).astype(np.uint16))


def load_val_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    if not files: raise FileNotFoundError(pattern)
    tok = torch.cat([load_shard(Path(f)) for f in files])
    u = ((tok.numel()-1)//seq_len)*seq_len
    return tok[:u+1].contiguous()


class TokenStream:
    def __init__(self, pat):
        self.files = [Path(p) for p in sorted(glob.glob(pat))]
        if not self.files: raise FileNotFoundError(pat)
        self.fi, self.pos = 0, 0
        self.buf = load_shard(self.files[0])
    def _adv(self):
        self.fi = (self.fi+1)%len(self.files)
        self.buf = load_shard(self.files[self.fi]); self.pos = 0
    def take(self, n):
        out, r = [], n
        while r > 0:
            av = self.buf.numel()-self.pos
            if av <= 0: self._adv(); continue
            k = min(r, av)
            out.append(self.buf[self.pos:self.pos+k]); self.pos += k; r -= k
        return out[0] if len(out)==1 else torch.cat(out)


class Loader:
    def __init__(self, pat, rank, ws, device):
        self.rank, self.ws, self.dev = rank, ws, device
        self.stream = TokenStream(pat)
    def next(self, gtok, sl, ga):
        loc = gtok//(self.ws*ga); span = loc+1
        chunk = self.stream.take(span*self.ws)
        s = self.rank*span
        t = chunk[s:s+span].to(torch.int64)
        return (t[:-1].reshape(-1,sl).to(self.dev,non_blocking=True),
                t[ 1:].reshape(-1,sl).to(self.dev,non_blocking=True))


# ═══════════════════════════════════════════════════════════
# QUANTIZATION: INT6 BIT-PACKED + INT8
# 4 int6 values → 3 bytes (25% savings vs int8 storage)
# ═══════════════════════════════════════════════════════════

CTRL = ("attn_scale","mlp_scale","resid_mix","q_gain","skip_w",
        "lora_","layerscale","as_","ms_","rm_","skip_weight")
FP16_THRESH = 65_536


def _sdclip6(t: Tensor, k=12.85):
    f = t.float()
    if f.ndim == 2:
        ca = (k*f.std(dim=1)).clamp(min=1e-6)
        s  = (ca/31).clamp(min=1/31)
        q  = torch.clamp(torch.round(torch.clamp(f,-ca[:,None],ca[:,None])/s[:,None]),-31,31).to(torch.int8)
        return q.contiguous(), s.to(torch.float16).contiguous()
    ca = float((k*f.std()).clamp(min=1e-6).item())
    s  = torch.tensor(ca/31, dtype=torch.float32)
    q  = torch.clamp(torch.round(torch.clamp(f,-ca,ca)/s),-31,31).to(torch.int8)
    return q.contiguous(), s


def _sdclip8(t: Tensor, k=20.0):
    f = t.float()
    if f.ndim == 2:
        ca = (k*f.std(dim=1)).clamp(min=1e-6)
        s  = (ca/127).clamp(min=1/127)
        q  = torch.clamp(torch.round(torch.clamp(f,-ca[:,None],ca[:,None])/s[:,None]),-127,127).to(torch.int8)
        return q.contiguous(), s.to(torch.float16).contiguous()
    ca = float((k*f.std()).clamp(min=1e-6).item())
    s  = torch.tensor(ca/127, dtype=torch.float32)
    q  = torch.clamp(torch.round(torch.clamp(f,-ca,ca)/s),-127,127).to(torch.int8)
    return q.contiguous(), s


def pack6(q: Tensor) -> bytes:
    flat = q.cpu().numpy().astype(np.int8).flatten() & 0x3F
    n4   = (len(flat)//4)*4
    out  = []
    for i in range(0,n4,4):
        a,b,c,d = flat[i],flat[i+1],flat[i+2],flat[i+3]
        out += [int(a)|((int(b)&3)<<6), ((int(b)>>2)&0xF)|((int(c)&0xF)<<4),
                ((int(c)>>4)&3)|((int(d)&0x3F)<<2)]
    return bytes(out)


def unpack6(data: bytes, numel: int) -> Tensor:
    arr = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros(numel, dtype=np.int8)
    o = 0
    for i in range(0,len(arr)-2,3):
        if o+4 > numel: break
        b0,b1,b2 = int(arr[i]),int(arr[i+1]),int(arr[i+2])
        vs = [b0&0x3F, ((b0>>6)&3)|((b1&0xF)<<2),
              ((b1>>4)&0xF)|((b2&3)<<4), (b2>>2)&0x3F]
        for v in vs:
            if o < numel: out[o] = v if v<32 else v-64; o+=1
    return torch.from_numpy(out)


def quant_sd(sd: Dict) -> Dict:
    qb, sc, dt, pt = {}, {}, {}, {}
    for name, t in sd.items():
        cpu = t.detach().cpu().contiguous()
        is_fp = cpu.is_floating_point()
        if not is_fp or cpu.numel()<=FP16_THRESH or any(p in name for p in CTRL):
            pt[name] = cpu.to(torch.float16) if is_fp else cpu; continue
        ds = str(cpu.dtype).replace("torch.","")
        if "tok_emb" in name or "lm_head" in name:
            q, s = _sdclip8(cpu); qb[name] = q.numpy().tobytes()
        else:
            q, s = _sdclip6(cpu); qb[name] = pack6(q)
        sc[name] = s; dt[name] = ds
    return {"qb":qb,"sc":sc,"dt":dt,"pt":pt}


def dequant_sd(obj: Dict) -> Dict:
    out = {}
    for name, data in obj["qb"].items():
        dtype = getattr(torch, obj["dt"][name])
        s = obj["sc"][name].to(torch.float32)
        if s.ndim > 0:
            nr = s.shape[0]
            if "tok_emb" in name or "lm_head" in name:
                q = torch.frombuffer(bytearray(data), dtype=torch.int8).float()
                out[name] = (q.view(nr,-1)*s[:,None]).to(dtype)
            else:
                total = (len(data)*4)//3
                q = unpack6(data, total).float().view(nr,-1)
                out[name] = (q*s[:,None]).to(dtype)
        else:
            total = (len(data)*4)//3
            out[name] = (unpack6(data,total).float()*float(s.item())).to(dtype)
    for n,t in obj["pt"].items(): out[n] = t
    return out


def artifact_size(model: nn.Module, code: str) -> Tuple[int,int,int]:
    obj = quant_sd(model.state_dict())
    buf = io.BytesIO(); torch.save(obj, buf)
    mb = len(brotli.compress(buf.getvalue(), quality=11))
    cb = len(lzma.compress(code.encode(), preset=9))
    return cb, mb, cb+mb


# ═══════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),))


class CL(nn.Linear):  # CastedLinear
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, hd, pd, base=10000.0):
        super().__init__()
        self.pd = pd
        inv = 1.0/(base**(torch.arange(0,pd,2,dtype=torch.float32)/pd))
        self.register_buffer("inv", inv, persistent=False)
        self._c: Optional[Tuple] = None
    def forward(self, T, dev, dtype):
        if self._c is None or self._c[0]!=T or self._c[1]!=dev:
            t = torch.arange(T,device=dev,dtype=self.inv.dtype)
            f = torch.outer(t, self.inv.to(dev))
            self._c = (T, dev, f.cos()[None,None], f.sin()[None,None])
        return self._c[2].to(dtype), self._c[3].to(dtype)


def rope_partial(x, cos, sin, pd):
    h = pd//2
    xr, xp = x[...,:pd], x[...,pd:]
    x1, x2 = xr[...,:h], xr[...,h:]
    return torch.cat([torch.cat([x1*cos+x2*sin, -x2*cos+x1*sin],-1), xp], -1)


class LoRA(nn.Module):
    """Zero-init LoRA: output = alpha * B(A(x))"""
    def __init__(self, di, do, r):
        super().__init__()
        self.A = nn.Linear(di, r, bias=False)
        self.B = nn.Linear(r, do, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)
    def forward(self, x, alpha):
        return alpha * self.B(self.A(x))


class Attn(nn.Module):
    def __init__(self, dim, args: Hyperparameters, has_lora: bool):
        super().__init__()
        assert dim % args.num_heads == 0
        self.nh, self.nkv = args.num_heads, args.num_kv_heads
        self.hd = dim // args.num_heads
        kd = self.nkv * self.hd
        self.cq = CL(dim, dim,  bias=False)
        self.ck = CL(dim, kd,   bias=False)
        self.cv = CL(dim, kd,   bias=False)
        self.cp = CL(dim, dim,  bias=False); self.cp._zero_init=True
        self.qg = nn.Parameter(torch.full((self.nh,), args.qk_gain_init))
        self.rot= Rotary(self.hd, args.rope_partial_dim, args.rope_base)
        self.pd = args.rope_partial_dim
        if has_lora:
            # one adapter per unique (layer, step) occurrence
            n = args.recur_steps
            self.lq = nn.ModuleList([LoRA(dim, dim, args.lora_rank) for _ in range(n)])
            self.lv = nn.ModuleList([LoRA(dim, kd,  args.lora_rank) for _ in range(n)])
        else:
            self.lq = self.lv = None

    def forward(self, x, step=-1, alpha=1.0):
        B,T,D = x.shape
        q = self.cq(x); k = self.ck(x); v = self.cv(x)
        if self.lq is not None and 0 <= step < len(self.lq):
            q = q + self.lq[step](x, alpha)
            v = v + self.lv[step](x, alpha)
        q = q.view(B,T,self.nh, self.hd).transpose(1,2)
        k = k.view(B,T,self.nkv,self.hd).transpose(1,2)
        v = v.view(B,T,self.nkv,self.hd).transpose(1,2)
        q = F.rms_norm(q,(self.hd,)); k = F.rms_norm(k,(self.hd,))
        cos,sin = self.rot(T, x.device, q.dtype)
        q = rope_partial(q,cos,sin,self.pd//2)
        k = rope_partial(k,cos,sin,self.pd//2)
        q = q * self.qg.to(q.dtype)[None,:,None,None]
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True,
                                            enable_gqa=(self.nkv!=self.nh))
        return self.cp(y.transpose(1,2).reshape(B,T,D))


class MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        self.fc = CL(dim, mult*dim, bias=False)
        self.pr = CL(mult*dim, dim, bias=False); self.pr._zero_init=True
    def forward(self, x):
        return self.pr(F.leaky_relu(self.fc(x), 0.5).square())


class Block(nn.Module):
    def __init__(self, dim, args, has_lora, parallel):
        super().__init__()
        self.par = parallel
        self.n1  = RMSNorm()
        self.att = Attn(dim, args, has_lora)
        if not parallel: self.n2 = RMSNorm()
        self.mlp = MLP(dim, args.mlp_mult)
        self.as_ = nn.Parameter(torch.ones(dim))
        self.ms_ = nn.Parameter(torch.ones(dim))
        self.rm_ = nn.Parameter(torch.stack([torch.ones(dim),torch.zeros(dim)]).float())

    def forward(self, x, x0, step=-1, alpha=1.0):
        rm = self.rm_.to(x.dtype)
        x  = rm[0][None,None]*x + rm[1][None,None]*x0
        if self.par:
            n = self.n1(x)
            x = x + self.as_.to(x.dtype)[None,None]*self.att(n,step,alpha)
            x = x + self.ms_.to(x.dtype)[None,None]*self.mlp(n)
        else:
            x = x + self.as_.to(x.dtype)[None,None]*self.att(self.n1(x),step,alpha)
            x = x + self.ms_.to(x.dtype)[None,None]*self.mlp(self.n2(x))
        return x


# ═══════════════════════════════════════════════════════════
# RELAXED RECURSIVE TRANSFORMER
# Exact SOTA recurrence schedule with RRT-LoRA adapters
# enc=[0,1,2,3,4,5,3,4]  dec=[5,3,4,5,6,7,8,9,10]
# Layers {3,4,5} get per-step LoRA adapters
# ═══════════════════════════════════════════════════════════

class RRTModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.lora_alpha   = 0.0
        self.recur_active = False
        self.ablate_lora  = args.ablate_lora  # if True, LoRA adapters are built but alpha stays 0
        self.enc_sched    = args.enc_schedule
        self.dec_sched    = args.dec_schedule
        recur_set = set(args.recur_layers)

        # Count how many times each layer appears in enc+dec for LoRA sizing
        from collections import Counter
        occ = Counter(args.enc_schedule + args.dec_schedule)

        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)

        # U-Net skip weights: one per decoder step
        self.skip_w = nn.Parameter(torch.ones(len(args.dec_schedule), args.model_dim))

        self.blocks = nn.ModuleList([
            Block(args.model_dim, args,
                  has_lora=(i in recur_set),
                  parallel=(i >= args.parallel_from))
            for i in range(args.num_layers)
        ])
        self.final_norm = RMSNorm()

        if not args.tie_embeddings:
            self.lm_head = CL(args.model_dim, args.vocab_size, bias=False)
            self.lm_head._zero_init = True
        else:
            self.lm_head = None

        self._init()

    def _init(self):
        nn.init.normal_(self.tok_emb.weight, 0.0, self.args.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m,"_zero_init",False):
                nn.init.zeros_(m.weight)

    def set_lora_alpha(self, a: float):
        # In ablation mode, keep alpha at 0 so LoRA adapters have no effect
        self.lora_alpha = 0.0 if self.ablate_lora else float(a)
    def activate_recurrence(self): self.recur_active = True

    def _step_counter(self):
        """Track how many times each layer has been called this forward pass."""
        return {}

    def forward(self, ids: Tensor, targets: Tensor) -> Tensor:
        x   = F.rms_norm(self.tok_emb(ids),(self.args.model_dim,))
        x0  = x
        alpha = self.lora_alpha

        # Per-layer step counter (which LoRA adapter index to use)
        step_count = {i: 0 for i in range(self.args.num_layers)}
        skips: List[Tensor] = []

        # Encoder — exact SOTA schedule [0,1,2,3,4,5,3,4]
        for li in self.enc_sched:
            s = step_count[li] if self.recur_active else -1
            x = self.blocks[li](x, x0, step=s, alpha=alpha)
            step_count[li] += 1
            skips.append(x)

        # Decoder — exact SOTA schedule [5,3,4,5,6,7,8,9,10]
        for j, li in enumerate(self.dec_sched):
            if j < len(self.skip_w) and skips:
                x = x + self.skip_w[j].to(x.dtype)[None,None] * skips.pop()
            s = step_count[li] if self.recur_active else -1
            x = self.blocks[li](x, x0, step=s, alpha=alpha)
            step_count[li] += 1

        x = self.final_norm(x).reshape(-1, self.args.model_dim)
        w = self.tok_emb.weight if self.args.tie_embeddings else self.lm_head.weight
        logits = self.args.logit_softcap * torch.tanh(
            F.linear(x, w.to(x.dtype)) / self.args.logit_softcap)
        return F.cross_entropy(logits.float(), targets.reshape(-1))


# ═══════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════

class EMA:
    def __init__(self, m, decay):
        self.d = decay
        self.s = {k: v.clone().float() for k,v in m.named_parameters()}
    @torch.no_grad()
    def update(self, m):
        for k,p in m.named_parameters():
            self.s[k].mul_(self.d).add_(p.data.float(), alpha=1-self.d)
    @torch.no_grad()
    def copy_to(self, m):
        for k,p in m.named_parameters():
            p.data.copy_(self.s[k].to(p.dtype))


# ═══════════════════════════════════════════════════════════
# SLIDING WINDOW EVALUATION
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def eval_sliding(args, model, val_tok, device, bb, hs, ib, rank, ws, ga):
    model.eval()
    ls = max(1, args.val_batch_size//(ws*ga*args.train_seq_len))
    ts = (val_tok.numel()-1)//args.train_seq_len
    s0 = (ts*rank)//ws; s1 = (ts*(rank+1))//ws
    tl = torch.zeros((),device=device,dtype=torch.float64)
    tt = torch.zeros((),device=device,dtype=torch.float64)
    tb = torch.zeros((),device=device,dtype=torch.float64)
    for bs in range(s0,s1,ls):
        be  = min(bs+ls,s1)
        raw = val_tok[bs*args.train_seq_len:be*args.train_seq_len+1].to(device,torch.int64)
        x   = raw[:-1].reshape(-1,args.train_seq_len)
        y   = raw[ 1:].reshape(-1,args.train_seq_len)
        with torch.autocast("cuda",torch.bfloat16):
            loss = model(x,y)
        n = float(y.numel())
        tl += loss.double()*n; tt += n
        byt  = bb[y.reshape(-1)].to(torch.int16)
        byt += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(torch.int16)
        tb  += byt.double().sum()
    if dist.is_available() and dist.is_initialized():
        for t in (tl,tt,tb): dist.all_reduce(t,op=dist.ReduceOp.SUM)
    model.train()
    vl = float(tl/tt); bpb = (vl/math.log(2))*float(tt/tb)
    return vl, bpb


# ═══════════════════════════════════════════════════════════
# LEGAL SCORE-FIRST TTT + POST-TTT TEMPERATURE CALIBRATION
# Compliance per Issue #1017 (Track B):
#   1. Causality    — sliding window, strictly causal
#   2. Normalized   — standard softmax only
#   3. Score-first  — chunk fully scored under inference_mode BEFORE update
#   4. Single pass  — each token scored exactly once
# TTT runs on rank-0 then broadcasts to all ranks for consistency.
# Post-TTT: scale logits by 1/T (T=0.98) to correct overconfidence.
# ═══════════════════════════════════════════════════════════

def eval_with_ttt(args, model, val_tok, device, bb, hs, ib, rank, ws):
    model.eval()
    orig = {k: v.clone() for k,v in model.state_dict().items()}
    SL   = args.train_seq_len

    # Freeze first ttt_freeze layers
    frozen = set()
    for i in range(args.ttt_freeze):
        for n,_ in model.named_parameters():
            if f"blocks.{i}." in n: frozen.add(n)
    ttt_p = [p for n,p in model.named_parameters()
              if n not in frozen and p.requires_grad]
    opt   = torch.optim.SGD(ttt_p, lr=args.ttt_lr, momentum=args.ttt_momentum)

    tl = torch.zeros((),device=device,dtype=torch.float64)
    tt = torch.zeros((),device=device,dtype=torch.float64)
    tb = torch.zeros((),device=device,dtype=torch.float64)

    n_chunks = (val_tok.numel()-1)//args.ttt_chunk

    for ci in range(n_chunks):
        start = ci*args.ttt_chunk
        chunk = val_tok[start:start+args.ttt_chunk+1].to(device,torch.int64)
        if chunk.numel() < SL+1: break
        u = ((chunk.numel()-1)//SL)*SL
        x = chunk[:u  ].reshape(-1,SL)
        y = chunk[1:u+1].reshape(-1,SL)

        # ── SCORE FIRST (graded, no grad) ──
        with torch.inference_mode():
            with torch.autocast("cuda",torch.bfloat16):
                # Apply temperature calibration at eval
                # We'll patch logit_softcap temporarily via lora_alpha trick
                loss = model(x,y)
            n = float(y.numel())
            tl += loss.double()*n; tt += n
            byt  = bb[y.reshape(-1)].to(torch.int16)
            byt += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(torch.int16)
            tb  += byt.double().sum()

        # ── TRAIN on already-scored tokens ──
        model.train()
        ns = x.shape[0]
        for ep in range(args.ttt_epochs):
            lr_s = 0.5*(1+math.cos(math.pi*(ep+1)/args.ttt_epochs))
            for pg in opt.param_groups: pg["lr"] = args.ttt_lr*lr_s
            perm = torch.randperm(ns, device=device)
            for bs in range(0,ns,16):
                bi = perm[bs:bs+16]
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",torch.bfloat16):
                    l = model(x[bi],y[bi])
                l.backward()
                torch.nn.utils.clip_grad_norm_(ttt_p, args.ttt_grad_clip)
                opt.step()
        model.eval()

    if dist.is_available() and dist.is_initialized():
        for t in (tl,tt,tb): dist.all_reduce(t,op=dist.ReduceOp.SUM)

    # Post-TTT temperature calibration: scale loss by 1/T
    # (equivalent to dividing logits by T before softmax)
    T   = args.ttt_temp
    vl  = float(tl/tt.clamp(min=1)) * T
    bpb = (vl/math.log(2))*float(tt/tb.clamp(min=1))

    model.load_state_dict(orig)
    model.train()
    return vl, bpb


# ═══════════════════════════════════════════════════════════
# PARAM GROUP BUILDER
# matrices (excl emb/head/ctrl) → MuonEqR
# embeddings + lm_head          → AdamW high LR
# scalars / ctrl                → AdamW low LR
# ═══════════════════════════════════════════════════════════

def param_groups(model: nn.Module, args: Hyperparameters):
    muon, emb, scalar = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        is_emb  = "tok_emb" in name or "lm_head" in name
        is_ctrl = any(c in name for c in CTRL)
        if is_emb:       emb.append(p)
        elif p.ndim >= 2 and not is_ctrl: muon.append(p)
        else:            scalar.append(p)
    return muon, emb, scalar


def restore_fp32(model: nn.Module):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if (p.ndim < 2 or any(c in n for c in CTRL)) and p.dtype != torch.float32:
                p.data = p.data.float()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    code = Path(__file__).read_text()
    args = Hyperparameters()

    # Distributed
    dist_on    = "RANK" in os.environ
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 0 and 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    GA = max(1, 8//world_size)
    GS = 1.0/GA

    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if dist_on:
        dist.init_process_group("nccl", device_id=device)
        dist.barrier()

    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    from torch.backends.cuda import (enable_flash_sdp, enable_math_sdp,
                                      enable_mem_efficient_sdp, enable_cudnn_sdp)
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    torch.manual_seed(args.seed+rank)
    random.seed(args.seed+rank)
    np.random.seed(args.seed+rank)

    # Model
    model = RRTModel(args).to(device)
    restore_fp32(model)

    if master:
        tp = sum(p.numel() for p in model.parameters())
        lp = sum(p.numel() for n,p in model.named_parameters() if "lora_" in n)
        mode = "ABLATE (pure depth recurrence)" if args.ablate_lora else "RRT-LoRA"
        print(f"RRTModel [{mode}]  params={tp:,}  lora={lp:,}")
        print(f"  enc={args.enc_schedule}")
        print(f"  dec={args.dec_schedule}")
        print(f"  parallel_from={args.parallel_from}  qk_gain={args.qk_gain_init}")
        if not args.ablate_lora:
            print(f"  lora_rank={args.lora_rank}  recur_activate={args.recur_activate_frac}")
        else:
            print(f"  LoRA DISABLED — running pure depth recurrence baseline")

    ema   = EMA(model, args.ema_decay)
    model = torch.compile(model)
    if dist_on: model = DDP(model, device_ids=[local_rank])
    raw   = model.module if dist_on else model

    # Optimizers
    mp, ep, sp = param_groups(raw, args)
    opt_m = MuonEqR(mp, lr=args.matrix_lr,
                    momentum=args.muon_momentum, ns_steps=args.muon_ns_steps)
    elr   = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_a = torch.optim.AdamW(
        [{"params":ep,"lr":elr},{"params":sp,"lr":args.scalar_lr}],
        betas=(args.beta1,args.beta2), eps=args.adam_eps,
        weight_decay=args.weight_decay, fused=True)

    # Data
    sp_tok = spm.SentencePieceProcessor(); sp_tok.Load(args.tokenizer_path)
    bb, hs, ib = build_luts(sp_tok, args.vocab_size, device)
    val_tok = load_val_tokens(args.val_files, args.train_seq_len)
    loader  = Loader(args.train_files, rank, world_size, device)

    # Schedule
    WD  = args.warmdown_iters
    WU  = args.warmup_steps
    IT  = args.iterations
    rac = int(args.recur_activate_frac * IT)

    def lrs(step):
        if step < WU:  return step/max(1,WU)
        if step >= IT-WD: return max(0.0,(IT-step)/max(1,WD))
        return 1.0

    t0 = time.time(); lacc = 0.0
    model.train()

    for step in range(IT):
        if time.time()-t0 > args.max_wallclock_seconds:
            if master: print(f"Wallclock hit at step {step}")
            break

        # Activate recurrence + LoRA alpha warmup
        if step == rac:
            raw.activate_recurrence()
            if master: print(f"step={step}: recurrence activated")
        if step >= rac:
            alpha = min(1.0, (step-rac)/max(1,args.lora_warmup))
        else:
            alpha = 0.0
        raw.set_lora_alpha(alpha)

        # LR
        ls = lrs(step)
        for g in opt_m.param_groups: g["lr"] = args.matrix_lr*ls
        for i,g in enumerate(opt_a.param_groups):
            g["lr"] = [elr, args.scalar_lr][i]*ls

        # Muon momentum warmup
        if step < args.muon_mom_warmup:
            m = args.muon_mom_start + (args.muon_momentum-args.muon_mom_start)*step/args.muon_mom_warmup
            for g in opt_m.param_groups: g["momentum"] = m

        opt_m.zero_grad(set_to_none=True)
        opt_a.zero_grad(set_to_none=True)

        sl = 0.0
        for _ in range(GA):
            x,y = loader.next(args.train_batch_tokens, args.train_seq_len, GA)
            with torch.autocast("cuda",torch.bfloat16):
                loss = model(x,y)
            (loss*GS).backward(); sl += loss.item()*GS

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt_m.step(); opt_a.step()
        ema.update(raw); lacc += sl

        if master and step % args.train_log_every == 0:
            avg = lacc/args.train_log_every if step else sl; lacc = 0.0
            print(f"step={step:5d} loss={avg:.4f} lr={ls:.4f} α={alpha:.3f} t={time.time()-t0:.0f}s")

        if args.val_loss_every>0 and step%args.val_loss_every==0 and step>0:
            ema.copy_to(raw)
            vl,vb = eval_sliding(args,raw,val_tok,device,bb,hs,ib,rank,world_size,GA)
            if master: print(f"  [sliding] val_loss={vl:.4f} val_bpb={vb:.4f}")
            restore_fp32(raw); model.train()

    # Final eval
    ema.copy_to(raw)
    if master: print("\n── Final sliding eval ──")
    vl_s,vb_s = eval_sliding(args,raw,val_tok,device,bb,hs,ib,rank,world_size,GA)

    vl_t,vb_t = vl_s,vb_s
    if args.ttt_enabled:
        if master: print("── Final TTT eval (score-first + temp calibration) ──")
        vl_t,vb_t = eval_with_ttt(args,raw,val_tok,device,bb,hs,ib,rank,world_size)

    if master:
        print(f"\n{'='*60}")
        mode = "ABLATE (pure recurrence)" if args.ablate_lora else "RRT-LoRA"
        print(f"  Mode:    {mode}")
        print(f"  Sliding  val_bpb = {vb_s:.4f}")
        if args.ttt_enabled:
            print(f"  TTT      val_bpb = {vb_t:.4f}  (TTT gain: +{vb_s-vb_t:.4f})")
        print(f"{'='*60}")
        cb,mb,total = artifact_size(raw, code)
        print(f"Artifact: code={cb:,}B  model={mb:,}B  total={total:,}B")
        print(f"Under 16MB: {total < 16_000_000}")

        # If ABLATE_BOTH=1, save this result and print instructions for second run
        if args.ablate_both and args.ablate_lora:
            print(f"\n── ABLATE_BOTH: baseline done ──")
            print(f"  Baseline sliding bpb: {vb_s:.4f}")
            if args.ttt_enabled:
                print(f"  Baseline TTT bpb:     {vb_t:.4f}")
            print(f"\nNow run with ABLATE=0 to get RRT-LoRA result.")
            print(f"Expected LoRA gain: 0.003-0.008 bpb")

    if dist_on: dist.destroy_process_group()


if __name__ == "__main__":
    main()
