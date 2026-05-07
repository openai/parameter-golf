# 11L int6 relu2 3x, partial rope, gated xsa, cosine ln, ema, smeargate, bigramhash

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import zstandard
    _COMP = "zstd"
except ImportError:
    _COMP = "zlib"
    import zlib


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 30))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    embed_lr = float(os.environ.get("EMBED_LR", 0.3))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    late_qat = bool(int(os.environ.get("LATE_QAT", "1")))
    qat_threshold = float(os.environ.get("QAT_THRESHOLD", 0.1))


# muon + wd

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, weight_decay=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            bs, nesterov, wd = group["backend_steps"], group["nesterov"], group["weight_decay"]
            total = sum(int(p.numel()) for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "buf" not in state:
                        state["buf"] = torch.zeros_like(g)
                    buf = state["buf"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=bs)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# bpb eval

def build_sentencepiece_luts(sp, vocab_size, device):
    sv = int(sp.vocab_size())
    ts = max(sv, vocab_size)
    bb = np.zeros((ts,), dtype=np.int16)
    hs = np.zeros((ts,), dtype=np.bool_)
    ib = np.ones((ts,), dtype=np.bool_)
    for tid in range(sv):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        ib[tid] = False
        if sp.is_byte(tid):
            bb[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            hs[tid] = True
            piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(hs, dtype=torch.bool, device=device),
            torch.tensor(ib, dtype=torch.bool, device=device))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Val too short for seq_len={seq_len}")
    return tokens[:usable + 1]


def eval_val_sliding(args, base_model, rank, world_size, device, val_tokens,
                     bb_lut, hs_lut, ib_lut):
    sl = args.eval_seq_len
    stride = args.eval_stride
    bseqs = args.eval_batch_seqs
    total = val_tokens.numel() - 1
    starts = list(range(0, total - sl + 1, stride))
    if not starts or starts[-1] + sl < total:
        starts.append(max(total - sl, 0))
    my = starts[rank::world_size]
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my), bseqs):
            bws = my[bi:bi + bseqs]
            bs = len(bws)
            xb = torch.zeros(bs, sl, dtype=torch.int64, device=device)
            yb = torch.zeros(bs, sl, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(bws):
                we = min(ws + sl, total)
                wl = we - ws
                wlens.append(wl)
                chunk = val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                xb[i, :wl] = chunk[:-1]
                yb[i, :wl] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(xb)
            nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                  yb.reshape(-1), reduction="none").reshape(bs, sl)
            for i, ws in enumerate(bws):
                wl = wlens[i]
                sf = 0 if ws == 0 else max(wl - stride, 0)
                sc_nll = nll[i, sf:wl]
                sc_y = yb[i, sf:wl]
                sc_x = xb[i, max(sf - 1, 0):max(sf - 1, 0) + sc_nll.numel()]
                nll_sum += sc_nll.to(torch.float64).sum()
                tok_count += sc_nll.numel()
                tb = bb_lut[sc_y].to(torch.int16)
                if sc_x.numel() == sc_y.numel():
                    tb += (hs_lut[sc_y] & ~ib_lut[sc_x]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [nll_sum, tok_count, byte_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = nll_sum / tok_count
    bpt = vl.item() / math.log(2.0)
    tpb = tok_count.item() / byte_count.item()
    base_model.train()
    return float(vl.item()), float(bpt * tpb)


# int6 quantization

CTRL_PATTERNS = tuple(p for p in os.environ.get("CTRL_PATTERNS",
    "attn_scale,mlp_scale,resid_mix,q_gain,xsa_gate,smear,bigram_scale").split(",") if p)
INT6_CLIP = 31

def quantize_int6(t):
    t32 = t.float()
    if t32.ndim == 2:
        rm = t32.abs().amax(dim=1)
        s = (rm / INT6_CLIP).clamp_min(1e-12).to(torch.float16)
        q = torch.clamp(torch.round(t32 / s.float()[:, None]), -INT6_CLIP - 1, INT6_CLIP).to(torch.int8)
        return q, s
    am = t32.abs().max().item()
    s = torch.tensor(max(am / INT6_CLIP, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / s.float()), -INT6_CLIP - 1, INT6_CLIP).to(torch.int8)
    return q, s


def quantize_sd(sd, embed_name="tok_emb.weight"):
    r, m, st = {}, {}, {"params": 0, "bytes": 0}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        st["params"] += t.numel()
        if not t.is_floating_point():
            r[name], m[name] = t, "pass"
            st["bytes"] += t.numel() * t.element_size()
            continue
        if name == embed_name:
            r[name], m[name] = t.to(torch.float16).contiguous(), "fp16"
            st["bytes"] += t.numel() * 2
            continue
        if any(p in name for p in CTRL_PATTERNS):
            r[name], m[name] = t.float().contiguous(), "ctrl"
            st["bytes"] += t.numel() * 4
            continue
        if t.numel() <= 65536:
            r[name], m[name] = t.to(torch.float16).contiguous(), "fp16"
            st["bytes"] += t.numel() * 2
            continue
        q, s = quantize_int6(t)
        r[name + ".q"], r[name + ".s"] = q, s
        m[name] = "int6"
        st["bytes"] += q.numel() + s.numel() * 2
    return r, m, st


def dequantize_sd(r, m, template):
    out = {}
    for name, orig in template.items():
        info = m[name]
        if info in ("pass", "ctrl", "fp16"):
            t = r[name]
            out[name] = t.to(orig.dtype).contiguous() if t.dtype != orig.dtype else t.contiguous()
            continue
        q, s = r[name + ".q"], r[name + ".s"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig.dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(orig.dtype).contiguous()
    return out


def compress(data):
    if _COMP == "zstd":
        return zstandard.ZstdCompressor(level=22).compress(data)
    return zlib.compress(data, level=9)

def decompress(data):
    if _COMP == "zstd":
        return zstandard.ZstdDecompressor().decompress(data)
    return zlib.decompress(data)


# data loading

def load_data_shard(file):
    hb = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard: {file}")
    nt = int(header[2])
    if file.stat().st_size != hb + nt * np.dtype("<u2").itemsize:
        raise ValueError(f"Size mismatch: {file}")
    t = np.fromfile(file, dtype="<u2", count=nt, offset=hb)
    return torch.from_numpy(t.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.fi, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])
    def _advance(self):
        self.fi = (self.fi + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.fi])
        self.pos = 0
    def take(self, n):
        chunks, rem = [], n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.ws, self.dev = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, glob_tok, seq_len, ga):
        lt = glob_tok // (self.ws * ga)
        span = lt + 1
        chunk = self.stream.take(span * self.ws)
        s = self.rank * span
        local = chunk[s:s + span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.dev, non_blocking=True), y.to(self.dev, non_blocking=True)


# transformer

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                rm = w32.abs().amax(dim=1)
                s = (rm / INT6_CLIP).clamp_min(1e-12)
                wq = (torch.clamp(torch.round(w32 / s[:, None]), -INT6_CLIP - 1, INT6_CLIP) * s[:, None]).to(x.dtype)
            w = w + (wq - w).detach()
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


def restore_fp32(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CTRL_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()


class Rotary(nn.Module):
    def __init__(self, head_dim, rope_dims=0, base=10000.0):
        super().__init__()
        rd = rope_dims if rope_dims > 0 else head_dim
        self.rd = rd
        inv = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._sl, self._cos, self._sin = 0, None, None

    def forward(self, seq_len, device, dtype):
        if self._cos is None or self._sl != seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos = f.cos()[None, None, :, :]
            self._sin = f.sin()[None, None, :, :]
            self._sl = seq_len
        return self._cos.to(dtype=dtype), self._sin.to(dtype=dtype)


def apply_rope(x, cos, sin, rd):
    if rd < x.size(-1):
        xr, xp = x[..., :rd], x[..., rd:]
        h = rd // 2
        x1, x2 = xr[..., :h], xr[..., h:]
        xr = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((xr, xp), dim=-1)
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, rope_dims, qk_gain_init):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.nh, self.nkv = num_heads, num_kv_heads
        self.hd = dim // num_heads
        assert self.hd % 2 == 0
        kv_dim = num_kv_heads * self.hd
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.hd, rope_dims=rope_dims, base=rope_base)
        self.use_xsa = False
        # gated xsa strength
        self.xsa_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _xsa(self, y, v):
        # gated exclusive self-attn
        B, T, H, D = y.shape
        Hkv = v.size(2)
        g = H // Hkv
        yg = y.reshape(B, T, Hkv, g, D)
        vn = F.normalize(v, dim=-1).unsqueeze(3)
        proj = (yg * vn).sum(dim=-1, keepdim=True) * vn
        xsa_out = (yg - proj).reshape(B, T, H, D)
        alpha = torch.sigmoid(self.xsa_gate.to(dtype=y.dtype))
        return alpha * xsa_out + (1 - alpha) * y

    def forward(self, x):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        rd = self.rotary.rd
        q, k = apply_rope(q, cos, sin, rd), apply_rope(k, cos, sin, rd)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.nkv != self.nh))
        y = y.transpose(1, 2).contiguous()
        if self.use_xsa:
            vt = v.transpose(1, 2).contiguous()
            y = self._xsa(y, vt)
        return self.proj(y.reshape(B, T, C))


class MLP(nn.Module):
    # relu squared 3x
    def __init__(self, dim, mlp_mult):
        super().__init__()
        h = mlp_mult * dim
        self.fc = CastedLinear(dim, h, bias=False)
        self.proj = CastedLinear(h, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())


class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        xp = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * xp


class BigramHash(nn.Module):
    def __init__(self, bvs, bd, md):
        super().__init__()
        self.bvs = bvs
        self.embed = nn.Embedding(bvs, bd)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bd, md, bias=False) if bd != md else None
        if self.proj:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def _hash(self, t):
        ti = t.to(torch.int32)
        mod = self.bvs - 1
        out = torch.empty_like(ti)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * ti[..., 1:], 27191 * ti[..., :-1]) % mod
        return out.long()
    def forward(self, ids):
        h = self.embed(self._hash(ids))
        if self.proj:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, rope_dims,
                 qk_gain_init, layer_idx, num_layers, ln_scale):
        super().__init__()
        self.an = RMSNorm()
        self.mn = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, rope_dims, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        # cosine ln decay
        self.lns = math.cos(math.pi * layer_idx / (2 * num_layers)) if ln_scale else 1.0

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s = self.lns
        ao = self.attn(self.an(x) * s if s != 1.0 else self.an(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * ao
        mo = self.mlp(self.mn(x) * s if s != 1.0 else self.mn(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mo
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, rope_dims,
                 qk_gain_init, xsa_last_n, ln_scale, bigram_vocab_size, bigram_dim):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.smear = SmearGate(model_dim)
        self.bigram = BigramHash(bigram_vocab_size, bigram_dim, model_dim)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, rope_dims,
                  qk_gain_init, i, num_layers, ln_scale)
            for i in range(num_layers)
        ])
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        # ortho init
        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        nl = num_layers
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if getattr(mod, "_zero_init", False):
                    nn.init.zeros_(mod.weight)
                elif mod.weight.ndim == 2 and min(mod.weight.shape) >= 64:
                    nn.init.orthogonal_(mod.weight, gain=1.0)
                    if ".proj" in name:
                        with torch.no_grad():
                            mod.weight.mul_(1.0 / math.sqrt(2 * nl))

    def forward(self, input_ids, target_ids):
        return F.cross_entropy(self.forward_logits(input_ids).float(), target_ids.reshape(-1), reduction="mean")

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids)
        x = self.smear(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        for block in self.blocks:
            x = block(x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        if self.tie_embeddings:
            lp = F.linear(x, self.tok_emb.weight)
        else:
            lp = self.lm_head(x)
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)


def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assert world_size > 0 and 8 % world_size == 0
    ga = 8 // world_size
    gs = 1.0 / ga
    assert torch.cuda.is_available()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
    def log0(msg, console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)
    log0(code, console=False); log0("=" * 80, console=False)
    log0(f"Python {sys.version} PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    assert int(sp.vocab_size()) == args.vocab_size
    val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)
    bb_lut, hs_lut, ib_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val:{val_tokens.numel()-1} comp:{_COMP}")

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, rope_dims=args.rope_dims,
        qk_gain_init=args.qk_gain_init, xsa_last_n=args.xsa_last_n, ln_scale=args.ln_scale,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_fp32(model)
    compiled = torch.compile(model, dynamic=False, fullgraph=True)
    ddp = DDP(compiled, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled

    bnp = list(model.blocks.named_parameters())
    mat_p = [p for n, p in bnp if p.ndim == 2 and not any(pat in n for pat in CTRL_PATTERNS)]
    sca_p = [p for n, p in bnp if p.ndim < 2 or any(pat in n for pat in CTRL_PATTERNS)]
    tlr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.Adam([{"params": [model.tok_emb.weight], "lr": tlr, "base_lr": tlr, "weight_decay": args.adam_weight_decay}],
                               betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon(mat_p, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps, weight_decay=args.muon_weight_decay)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    all_sca = sca_p + list(model.smear.parameters())
    opt_sca = torch.optim.Adam([{"params": all_sca, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                               betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_bg = torch.optim.Adam([{"params": list(model.bigram.parameters()), "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                              betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opts = [opt_tok, opt_muon, opt_sca, opt_bg]
    if model.lm_head is not None:
        opt_h = torch.optim.Adam([{"params": [model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
                                 betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        opts.insert(1, opt_h)
    np_ = sum(p.numel() for p in model.parameters())
    log0(f"params:{np_} layers:{args.num_layers} dim:{args.model_dim} mlp:{args.mlp_mult}x")
    log0(f"rope_dims:{args.rope_dims} xsa:{args.xsa_last_n} ln_scale:{args.ln_scale}")
    log0(f"lr:{args.matrix_lr} wd:{args.muon_weight_decay} clip:{args.grad_clip_norm}")
    log0(f"batch:{args.train_batch_tokens} seq:{args.train_seq_len} eval:{args.eval_seq_len} stride:{args.eval_stride}")
    log0(f"ema:{args.ema_enabled}/{args.ema_decay} qat:{args.late_qat}/{args.qat_threshold} seed:{args.seed}")

    loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zg():
        for o in opts: o.zero_grad(set_to_none=True)
    max_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed):
        if args.warmdown_iters <= 0: return 1.0
        if max_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        sms = elapsed / max(step, 1)
        wms = args.warmdown_iters * sms
        rms = max(max_ms - elapsed, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

    # compile warmup
    if args.warmup_steps > 0:
        isd = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
        ios = [copy.deepcopy(o.state_dict()) for o in opts]
        ddp.train()
        for ws in range(args.warmup_steps):
            zg()
            for ms in range(ga):
                if distributed: ddp.require_backward_grad_sync = ms == ga - 1
                x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, ga)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): (ddp(x, y) * gs).backward()
            for o in opts: o.step()
            zg()
            if args.warmup_steps <= 30 or (ws + 1) % 10 == 0: log0(f"warmup:{ws+1}/{args.warmup_steps}")
        model.load_state_dict(isd, strict=True)
        for o, s in zip(opts, ios, strict=True): o.load_state_dict(s)
        zg()
        if distributed: ddp.require_backward_grad_sync = True
        loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ema state
    ema = None
    if args.ema_enabled:
        ema = {n: t.detach().float().clone() for n, t in model.state_dict().items()}

    ttms = 0.0
    stop = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step = 0
    while True:
        last = step == args.iterations or (stop is not None and step >= stop)
        if last or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize(); ttms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val_sliding(args, model, rank, world_size, device, val_tokens, bb_lut, hs_lut, ib_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} time:{ttms:.0f}ms avg:{ttms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last:
            if stop is not None and step < args.iterations:
                log0(f"stopping step:{step}/{args.iterations} time:{ttms:.0f}ms")
            break
        elapsed = ttms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed)
        # late qat
        if args.late_qat and scale < args.qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"qat_on step:{step}")
        zg()
        tl = torch.zeros((), device=device)
        for mi in range(ga):
            if distributed: ddp.require_backward_grad_sync = mi == ga - 1
            x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, ga)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = ddp(x, y)
            tl += loss.detach()
            (loss * gs).backward()
        tl /= ga
        mfrac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for g in opt_muon.param_groups: g["momentum"] = (1 - mfrac) * args.muon_momentum_warmup_start + mfrac * args.muon_momentum
        for o in opts:
            for g in o.param_groups: g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        for o in opts: o.step()
        zg()
        # ema update
        if ema is not None:
            d = args.ema_decay
            for n, t in model.state_dict().items():
                ema[n].mul_(d).add_(t.detach().float(), alpha=1.0 - d)
        step += 1
        atm = ttms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop is not None):
            log0(f"step:{step}/{args.iterations} loss:{tl.item():.4f} time:{atm:.0f}ms avg:{atm/step:.2f}ms")
        reached = max_ms is not None and atm >= max_ms
        if distributed and max_ms is not None:
            rt = torch.tensor(int(reached), device=device)
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
            reached = bool(rt.item())
        if stop is None and reached: stop = step

    log0(f"peak_mem:{torch.cuda.max_memory_allocated()//1024//1024}MiB")
    CastedLinear._qat_enabled = False

    # load ema
    if ema is not None:
        log0("loading ema weights")
        model.load_state_dict({n: t.to(model.state_dict()[n].dtype) for n, t in ema.items()}, strict=True)
        del ema

    if master:
        torch.save(model.state_dict(), "final_model.pt")
        cb = len(code.encode("utf-8"))
        log0(f"raw_model:{os.path.getsize('final_model.pt')} code:{cb}")
    tsd = model.state_dict()
    qr, qm, qs = quantize_sd(tsd)
    buf = io.BytesIO(); torch.save({"r": qr, "m": qm}, buf)
    blob = compress(buf.getvalue())
    if master:
        with open("final_model.int6.ptz", "wb") as f: f.write(blob)
        qfb = os.path.getsize("final_model.int6.ptz")
        cb = len(code.encode("utf-8"))
        log0(f"int6+{_COMP}:{qfb} code:{cb} total:{qfb+cb} params:{qs['params']}")
    if distributed: dist.barrier()
    with open("final_model.int6.ptz", "rb") as f: disk = f.read()
    loaded = torch.load(io.BytesIO(decompress(disk)), map_location="cpu")
    model.load_state_dict(dequantize_sd(loaded["r"], loaded["m"], tsd), strict=True)
    torch.cuda.synchronize(); tq = time.perf_counter()
    ql, qb = eval_val_sliding(args, model, rank, world_size, device, val_tokens, bb_lut, hs_lut, ib_lut)
    torch.cuda.synchronize()
    log0(f"final_int6_{_COMP}_roundtrip val_loss:{ql:.4f} val_bpb:{qb:.4f} eval:{1000*(time.perf_counter()-tq):.0f}ms")
    log0(f"final_int6_{_COMP}_roundtrip_exact val_loss:{ql:.8f} val_bpb:{qb:.8f}")
    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
