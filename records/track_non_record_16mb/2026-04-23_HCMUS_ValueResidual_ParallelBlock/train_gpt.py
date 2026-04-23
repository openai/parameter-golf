
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
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────

class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH",      "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID",         str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED",       "1337"))

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE",  "524288"))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY",  "1000"))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", "200"))

    iterations         = int(os.environ.get("ITERATIONS",        "20000"))
    warmdown_iters     = int(os.environ.get("WARMDOWN_ITERS",    "1200"))
    warmup_steps       = int(os.environ.get("WARMUP_STEPS",      "20"))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS","524288"))
    train_seq_len      = int(os.environ.get("TRAIN_SEQ_LEN",     "1024"))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "600.0"))
    qk_gain_init       = float(os.environ.get("QK_GAIN_INIT",   "1.5"))
    eval_stride        = int(os.environ.get("EVAL_STRIDE",       "64"))

    # Model shape — 11 layers (parallel block saves compute vs baseline 9)
    vocab_size    = int(os.environ.get("VOCAB_SIZE",    "1024"))
    num_layers    = int(os.environ.get("NUM_LAYERS",    "11"))   # +2 vs baseline
    num_kv_heads  = int(os.environ.get("NUM_KV_HEADS",  "4"))
    model_dim     = int(os.environ.get("MODEL_DIM",     "512"))
    num_heads     = int(os.environ.get("NUM_HEADS",     "8"))
    mlp_mult      = float(os.environ.get("MLP_MULT",   "2.0"))
    tie_embeddings      = bool(int(os.environ.get("TIE_EMBEDDINGS",    "1")))
    rope_base           = float(os.environ.get("ROPE_BASE",            "10000.0"))
    logit_softcap       = float(os.environ.get("LOGIT_SOFTCAP",        "30.0"))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD",  "0.005"))

    # New Variant C flags
    use_parallel_block = bool(int(os.environ.get("USE_PARALLEL_BLOCK", "1")))
    use_value_residual = bool(int(os.environ.get("USE_VALUE_RESIDUAL", "1")))
    stochastic_depth   = float(os.environ.get("STOCHASTIC_DEPTH",     "0.1"))

    # Optimizer
    embed_lr      = float(os.environ.get("EMBED_LR",      "0.6"))
    head_lr       = float(os.environ.get("HEAD_LR",       "0.008"))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", "0.05"))
    matrix_lr     = float(os.environ.get("MATRIX_LR",     "0.04"))
    scalar_lr     = float(os.environ.get("SCALAR_LR",     "0.04"))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", "0.95"))
    muon_backend_steps         = int(os.environ.get("MUON_BACKEND_STEPS",         "5"))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START","0.85"))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", "500"))
    muon_wd  = float(os.environ.get("MUON_WD",  "0.02"))
    adam_wd  = float(os.environ.get("ADAM_WD",  "0.01"))
    beta1    = float(os.environ.get("BETA1",    "0.9"))
    beta2    = float(os.environ.get("BETA2",    "0.95"))
    adam_eps = float(os.environ.get("ADAM_EPS", "1e-8"))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", "0.3"))

    # Feature flags
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay   = float(os.environ.get("EMA_DECAY",      "0.997"))
    xsa_last_n  = int(os.environ.get("XSA_LAST_N",       "4"))    # 4 vs baseline 3
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", "4096"))
    bigram_dim        = int(os.environ.get("BIGRAM_DIM",        "128"))


# ─────────────────────────────────────────────
# MUON OPTIMIZER
# ─────────────────────────────────────────────

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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank       = dist.get_rank()       if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd   = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ─────────────────────────────────────────────
# TOKENIZER + QUANTIZATION UTILS
# ─────────────────────────────────────────────

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab   = int(sp.vocab_size())
    table_size = max(sp_vocab, vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_space  = np.zeros((table_size,), dtype=np.bool_)
    is_bound   = np.ones( (table_size,), dtype=np.bool_)
    for i in range(sp_vocab):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i):
            continue
        is_bound[i] = False
        if sp.is_byte(i):
            base_bytes[i] = 1
            continue
        piece = sp.id_to_piece(i)
        if piece.startswith("▁"):
            has_space[i] = True
            piece = piece[1:]
        base_bytes[i] = len(piece.encode("utf-8"))
    return (torch.from_numpy(base_bytes).to(device),
            torch.from_numpy(has_space).to(device),
            torch.from_numpy(is_bound).to(device))


CONTROL_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain",
    "skip_weight", "smear", "vr_scale",
)
INT8_CLIP_Q   = 0.9999984
INT8_MAX_PASS = 65_536

def _quant_t(t: Tensor):
    t32 = t.float()
    if t32.ndim == 2:
        clip  = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
        t32   = torch.clamp(t32, -clip[:, None], clip[:, None])
        scale = (clip / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(t32 / scale[:, None]), -127, 127).to(torch.int8)
        return q, scale.to(torch.float16), "row"
    clip  = float(torch.quantile(t32.abs(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip / 127.0 if clip > 0 else 1.0)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip, clip) / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32), "scalar"

def quantize_state_dict_int8(sd: dict):
    quant, scales, dtypes, passthrough, modes = {}, {}, {}, {}, {}
    for name, t in sd.items():
        t = t.detach().cpu()
        if not t.is_floating_point() or t.numel() <= INT8_MAX_PASS or \
                any(p in name for p in CONTROL_PATTERNS):
            passthrough[name] = t.to(torch.float16) if t.is_floating_point() else t
            continue
        q, s, m = _quant_t(t)
        quant[name], scales[name] = q, s
        dtypes[name], modes[name] = str(t.dtype).split(".")[-1], m
    return {"quant": quant, "scales": scales, "dtypes": dtypes,
            "passthrough": passthrough, "modes": modes}

def dequantize_state_dict_int8(obj: dict):
    out = {}
    for name, q in obj["quant"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s, m  = obj["scales"][name], obj["modes"][name]
        out[name] = (q.float() * s.float().view(-1, 1)).to(dtype) if m == "row" \
                    else (q.float() * float(s.item())).to(dtype)
    for name, t in obj["passthrough"].items():
        out[name] = t
    return out


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    assert header[0] == 20240520, "bad magic"
    ntok   = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=ntok, offset=256 * 4)
    return torch.from_numpy(tokens.astype(np.int32))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files  = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(pattern)
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(pattern)
        random.shuffle(self.files)
        self._idx = 0; self._pos = 0; self._buf = None
        self._load_next()
    def _load_next(self):
        self._buf = load_data_shard(self.files[self._idx % len(self.files)])
        self._idx += 1; self._pos = 0
    def take(self, n: int) -> Tensor:
        parts, rem = [], n
        while rem > 0:
            avail    = self._buf.numel() - self._pos
            take_now = min(rem, avail)
            parts.append(self._buf[self._pos: self._pos + take_now])
            self._pos += take_now; rem -= take_now
            if self._pos >= self._buf.numel():
                self._load_next()
        return torch.cat(parts)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.ws = world_size; self.dev = device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum: int):
        per_rank = global_tokens // (self.ws * grad_accum)
        chunk = self.stream.take(per_rank + 1).to(torch.int64).to(self.dev, non_blocking=True)
        return chunk[:-1].reshape(-1, seq_len), chunk[1:].reshape(-1, seq_len)


# ─────────────────────────────────────────────
# TRANSFORMER MODULES
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__(); self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024):
        super().__init__()
        self.dim = dim; self.base = base; self.train_seq_len = train_seq_len
        self.register_buffer("inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)),
            persistent=False)
        self._sc = 0; self._cos = None; self._sin = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos is None or self._sc != seq_len or self._cos.device != device:
            if seq_len > self.train_seq_len:
                sc  = seq_len / self.train_seq_len
                nb  = self.base * (sc ** (self.dim / (self.dim - 2)))
                inv = 1.0 / (nb ** (torch.arange(0, self.dim, 2,
                             dtype=torch.float32, device=device) / self.dim))
            else:
                inv = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv.dtype)
            f = torch.outer(t, inv)
            self._cos = f.cos()[None, :, None, :]
            self._sin = f.sin()[None, :, None, :]
            self._sc  = seq_len
        return self._cos.to(dtype), self._sin.to(dtype)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    h  = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float,
                 use_value_residual: bool = False):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        self.group        = num_heads // num_kv_heads
        kv_dim = num_kv_heads * self.head_dim

        self.c_q  = CastedLinear(dim, dim,    bias=False)
        self.c_k  = CastedLinear(dim, kv_dim, bias=False)
        self.c_v  = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim,    bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.use_xsa = False

        # Value Residual: scale for blending v_prev into current v
        # init=0 → no effect at start, learned gradually
        self.use_vr = use_value_residual
        if use_value_residual:
            self.vr_scale = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))

    def _xsa(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        y_g = y.reshape(B, T, Hkv, self.group, D)
        vn  = F.normalize(v, dim=-1).unsqueeze(-2)
        return (y_g - (y_g * vn).sum(-1, keepdim=True) * vn).reshape(B, T, H, D)

    def forward(self, x: Tensor, v_prev: Tensor | None = None):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads,    self.head_dim)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim)

        # Value Residual: blend v_prev into v
        if self.use_vr and v_prev is not None:
            scale = torch.sigmoid(self.vr_scale.to(v.dtype))[None, None, None, :]
            v = v + scale * v_prev

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2).repeat_interleave(self.group, 1)
        v_t = v.transpose(1, 2).repeat_interleave(self.group, 1)
        y   = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True).transpose(1, 2)

        if self.use_xsa:
            y = self._xsa(y, v)

        return self.proj(y.reshape(B, T, C)), v  # return v for next layer


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g     = torch.sigmoid(self.gate.to(x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, vocab: int, dim: int, model_dim: int):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, dim); nn.init.zeros_(self.embed.weight)
        self.proj  = CastedLinear(dim, model_dim, bias=False) if dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def _hash(self, t: Tensor) -> Tensor:
        t   = t.to(torch.int32); mod = self.vocab - 1
        out = torch.empty_like(t)
        out[..., 0]  = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, ids: Tensor) -> Tensor:
        h = self.embed(self._hash(ids))
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(h.dtype)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    """Parallel Attention + MLP block with optional Stochastic Depth.

    Sequential (baseline): x = x + attn(norm(x)); x = x + mlp(norm(x))
    Parallel (this):        x = x + attn(norm(x)) + mlp(norm(x))
    → Both branches see the same pre-norm input
    → ~15% faster → more training steps in same wallclock
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: float, rope_base: float, qk_gain_init: float,
                 use_parallel: bool = True, drop_rate: float = 0.0,
                 use_value_residual: bool = False):
        super().__init__()
        self.norm       = RMSNorm()   # single norm for parallel block
        self.attn       = CausalSelfAttention(dim, num_heads, num_kv_heads,
                                               rope_base, qk_gain_init,
                                               use_value_residual=use_value_residual)
        self.mlp        = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())
        self.use_parallel = use_parallel
        self.drop_rate    = drop_rate  # stochastic depth probability

    def forward(self, x: Tensor, x0: Tensor,
                v_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # Stochastic Depth: skip layer randomly during training
        if self.training and self.drop_rate > 0 and torch.rand(1).item() < self.drop_rate:
            return x, v_prev  # pass through unchanged, propagate v_prev

        mix = self.resid_mix.to(dtype=x.dtype)
        x   = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        if self.use_parallel:
            # Parallel: single norm, both branches computed from same x
            x_norm = self.norm(x)
            attn_out, v_cur = self.attn(x_norm, v_prev)
            mlp_out  = self.mlp(x_norm)
            x = x + self.attn_scale.to(x.dtype)[None, None, :] * attn_out \
                  + self.mlp_scale.to(x.dtype)[None, None, :]  * mlp_out
        else:
            # Sequential fallback
            attn_out, v_cur = self.attn(self.norm(x), v_prev)
            x = x + self.attn_scale.to(x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(x.dtype)[None, None, :]  * self.mlp(self.norm(x))

        return x, v_cur


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: float,
                 tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 bigram_vocab: int = 0, bigram_dim: int = 128,
                 xsa_last_n: int = 0,
                 use_parallel_block: bool = True,
                 use_value_residual: bool = True,
                 stochastic_depth: float = 0.1):
        super().__init__()
        self.tie_embeddings      = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap       = logit_softcap
        self.use_value_residual  = use_value_residual

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram  = BigramHashEmbedding(bigram_vocab, bigram_dim, model_dim) \
                       if bigram_vocab > 0 else None
        self.smear   = SmearGate(model_dim)

        # U-Net skip connections
        self.n_enc = num_layers // 2
        self.n_dec = num_layers - self.n_enc
        n_skip     = min(self.n_enc, self.n_dec)
        self.skip_weights = nn.Parameter(torch.ones(n_skip, model_dim, dtype=torch.float32))

        # Stochastic depth: linearly increase drop rate from 0 to max across layers
        # (deepest layers drop most — they're most redundant)
        self.blocks = nn.ModuleList([
            Block(
                model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                use_parallel=use_parallel_block,
                drop_rate=stochastic_depth * i / max(num_layers - 1, 1),
                use_value_residual=use_value_residual,
            )
            for i in range(num_layers)
        ])

        self.final_norm = RMSNorm()
        self.lm_head    = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # XSA on last xsa_last_n layers
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, std=self.tied_embed_init_std)
        n = len(self.blocks)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "_zero_init", False):
                    nn.init.zeros_(m.weight)
                elif m.weight.ndim == 2 and min(m.weight.shape) >= 64:
                    nn.init.orthogonal_(m.weight)
                    if ".proj" in name:
                        with torch.no_grad():
                            m.weight.mul_(1.0 / math.sqrt(2 * n))

    def _run_forward(self, x: Tensor, x0: Tensor) -> Tensor:
        """Run all blocks with Value Residual chain and U-Net skips."""
        skips: list[Tensor] = []
        v_prev: Tensor | None = None  # Value Residual chain

        for i in range(self.n_enc):
            x, v_prev = self.blocks[i](x, x0, v_prev)
            skips.append(x)

        for i in range(self.n_dec):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x, v_prev = self.blocks[self.n_enc + i](x, x0, v_prev)

        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x  = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x  = F.rms_norm(x, (x.size(-1),))
        x  = self.smear(x)
        x0 = x
        x  = self._run_forward(x, x0)
        x  = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x.reshape(-1, x.size(-1)), self.tok_emb.weight)
        else:
            logits = self.lm_head(x.reshape(-1, x.size(-1)))
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1))

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x  = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x  = F.rms_norm(x, (x.size(-1),))
        x  = self.smear(x)
        x0 = x
        x  = self._run_forward(x, x0)
        x  = self.final_norm(x)
        if self.tie_embeddings:
            return self.logit_softcap * torch.tanh(
                F.linear(x, self.tok_emb.weight) / self.logit_softcap)
        return self.logit_softcap * torch.tanh(self.lm_head(x) / self.logit_softcap)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def eval_val(args, model, rank, world_size, device, grad_accum,
             val_tokens, bb, hs, ib):
    seq_len    = args.train_seq_len
    local_seqs = max(1, args.val_batch_size // (world_size * grad_accum * seq_len))
    total_seqs = (val_tokens.numel() - 1) // seq_len
    s = (total_seqs * rank)       // world_size
    e = (total_seqs * (rank + 1)) // world_size
    ls = tc = bc = torch.zeros((), device=device, dtype=torch.float64)
    ls, tc, bc = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(3)]
    model.eval()
    with torch.inference_mode():
        for bs in range(s, e, local_seqs):
            be  = min(bs + local_seqs, e)
            raw = val_tokens[bs*seq_len: be*seq_len+1].to(device=device, dtype=torch.int64)
            x, y = raw[:-1].reshape(-1, seq_len), raw[1:].reshape(-1, seq_len)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y).detach()
            n  = float(y.numel())
            ls += loss.double() * n; tc += n
            tb  = bb[y.reshape(-1)].to(torch.int16)
            tb += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(torch.int16)
            bc += tb.double().sum()
    if dist.is_available() and dist.is_initialized():
        for t in (ls, tc, bc):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    model.train()
    vl = (ls / tc).item()
    return vl, (vl / math.log(2.0)) * (tc / bc).item()


def eval_val_sliding(model, device, val_tokens, bb, hs, ib,
                     seq_len: int, stride: int, rank: int, world_size: int,
                     batch_seqs: int = 32):
    n_tok   = val_tokens.numel() - 1
    starts  = list(range(0, n_tok, stride))
    my_s    = (len(starts) * rank)       // world_size
    my_e    = (len(starts) * (rank + 1)) // world_size
    ls, tc, bc = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(3)]
    model.eval()
    with torch.inference_mode():
        for bi in range(my_s, my_e, batch_seqs):
            bws = starts[bi: bi + batch_seqs]; bsz = len(bws)
            xb  = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            yb  = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wls = []
            for i, ws in enumerate(bws):
                wl = min(ws + seq_len, n_tok) - ws; wls.append(wl)
                ch = val_tokens[ws: ws+wl+1].to(torch.int64).to(device)
                xb[i, :wl] = ch[:-1]; yb[i, :wl] = ch[1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(xb)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   yb.reshape(-1), reduction="none").reshape(bsz, seq_len)
            for i, ws in enumerate(bws):
                wl = wls[i]; si = 0 if ws == 0 else max(wl - stride, 0)
                sc = nll[i, si:wl].double()
                ls += sc.sum(); tc += wl - si
                tgt  = yb[i, si:wl]; prev = xb[i, si:wl]
                tb   = bb[tgt].double() + (hs[tgt] & ~ib[prev]).double()
                bc  += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in (ls, tc, bc):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    model.train()
    vl = (ls / tc).item()
    return vl, (vl / math.log(2.0)) * (tc / bc).item()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank        = int(os.environ.get("RANK",       "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum  = max(1, 8 // world_size)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt" if master else None

    def log0(msg: str, console: bool = True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a") as f: print(msg, file=f)

    log0(code, console=False)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"vocab mismatch: {sp.vocab_size()} != {args.vocab_size}")

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb, hs, ib = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_tokens:{val_tokens.numel()-1}")
    log0(f"variant:C parallel_block:{args.use_parallel_block} "
         f"value_residual:{args.use_value_residual} "
         f"stochastic_depth:{args.stochastic_depth}")
    log0(f"num_layers:{args.num_layers} model_dim:{args.model_dim} "
         f"xsa_last_n:{args.xsa_last_n} ema:{args.ema_enabled}")

    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        use_parallel_block=args.use_parallel_block,
        use_value_residual=args.use_value_residual,
        stochastic_depth=args.stochastic_depth,
    ).to(device).bfloat16()

    for nm, p in base_model.named_parameters():
        if p.ndim < 2 or any(x in nm for x in CONTROL_PATTERNS):
            p.data = p.data.float()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.weight.data = m.weight.data.float()

    model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) \
                       if distributed else base_model

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}  world_size:{world_size}  grad_accum:{grad_accum}")

    # Optimizers — only look at blocks params (excludes tok_emb, bigram, smear)
    block_named = list(base_model.blocks.named_parameters())
    matrix_params = [p for nm, p in block_named
                     if p.ndim == 2 and not any(x in nm for x in CONTROL_PATTERNS)]
    scalar_params  = [p for nm, p in block_named
                     if p.ndim < 2 or any(x in nm for x in CONTROL_PATTERNS)]
    # Add non-block scalar params manually
    scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)

    tok_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_pg = [{"params": [base_model.tok_emb.weight], "lr": tok_lr, "base_lr": tok_lr}]
    if base_model.bigram is not None:
        tok_pg.append({"params": [base_model.bigram.embed.weight],
                        "lr": tok_lr, "base_lr": tok_lr})

    opt_tok    = torch.optim.AdamW(tok_pg, betas=(args.beta1, args.beta2),
                                    eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    opt_muon   = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                       backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    opt_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_wd, fused=True)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizers = [opt_tok, opt_muon, opt_scalar]

    def zero_grad_all():
        for o in optimizers: o.zero_grad(set_to_none=True)

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_wc_ms    = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    # Warmup
    if args.warmup_steps > 0:
        init_sd  = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for micro in range(grad_accum):
                x, y = train_loader.next_batch(args.train_batch_tokens,
                                                args.train_seq_len, grad_accum)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss / grad_accum).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            if (ws+1) % 10 == 0 or ws+1 == args.warmup_steps:
                log0(f"warmup {ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opt): o.load_state_dict(s)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    ema_state: dict | None = None
    if args.ema_enabled:
        ema_state = {k: t.detach().float().clone() for k, t in base_model.state_dict().items()}
        log0("ema:initialized")

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0: return 1.0
        if max_wc_ms is None:
            wd_s = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                   if wd_s <= step < args.iterations else 1.0
        sm = elapsed_ms / max(step, 1)
        wm = args.warmdown_iters * sm
        rm = max(max_wc_ms - elapsed_ms, 0.0)
        return rm / max(wm, 1e-9) if rm <= wm else 1.0

    train_ms = 0.0; stop_at: int | None = None
    torch.cuda.synchronize(); t0 = time.perf_counter()

    for step in range(args.iterations + 1):
        last = (step == args.iterations) or (stop_at is not None and step >= stop_at)

        if last or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum,
                               val_tokens, bb, hs, ib)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"train_time:{train_ms:.0f}ms step_avg:{train_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        if last: break

        elapsed = train_ms + 1000.0 * (time.perf_counter() - t0)
        scale   = lr_mul(step, elapsed)
        frac    = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_m  = (1-frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups: g["momentum"] = muon_m
        for o in optimizers:
            for g in o.param_groups: g["lr"] = g["base_lr"] * scale

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro in range(grad_accum):
            if distributed:
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            x, y = train_loader.next_batch(args.train_batch_tokens,
                                            args.train_seq_len, grad_accum)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / grad_accum).backward()
        train_loss /= grad_accum

        if args.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers: o.step()
        zero_grad_all()

        if ema_state is not None:
            d = args.ema_decay
            with torch.no_grad():
                for nm, t in base_model.state_dict().items():
                    ema_state[nm].mul_(d).add_(t.detach().float(), alpha=1.0 - d)

        approx = train_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and \
                (step+1 <= 10 or (step+1) % args.train_log_every == 0):
            log0(f"step:{step+1}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx:.0f}ms step_avg:{approx/(step+1):.2f}ms")

        hit = max_wc_ms is not None and approx >= max_wc_ms
        if distributed and max_wc_ms is not None:
            t_ = torch.tensor(int(hit), device=device)
            dist.all_reduce(t_, op=dist.ReduceOp.MAX); hit = bool(t_.item())
        if stop_at is None and hit: stop_at = step + 1

    log0(f"peak_mem:{torch.cuda.max_memory_allocated()//1024//1024}MiB "
         f"reserved:{torch.cuda.max_memory_reserved()//1024//1024}MiB")

    # Apply EMA
    if ema_state is not None:
        log0("ema:applying averaged weights")
        avg = {k: v.to(dtype=base_model.state_dict()[k].dtype) for k, v in ema_state.items()}
        base_model.load_state_dict(avg, strict=True)

    # Int-8 + zlib
    if master:
        sd   = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
        qobj = quantize_state_dict_int8(sd)
        buf  = io.BytesIO(); torch.save(qobj, buf)
        blob = zlib.compress(buf.getvalue(), 9)
        with open("final_model.int8.ptz", "wb") as f: f.write(blob)
        cb = len(code.encode("utf-8"))
        log0(f"model_int8_zlib:{len(blob)} bytes  code:{cb} bytes  "
             f"total:{len(blob)+cb} bytes  limit:16000000")

    if distributed: dist.barrier()

    # Roundtrip validation
    with open("final_model.int8.ptz", "rb") as f:
        deq = dequantize_state_dict_int8(
            torch.load(io.BytesIO(zlib.decompress(f.read())),
                       map_location="cpu", weights_only=False))

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        use_parallel_block=args.use_parallel_block,
        use_value_residual=args.use_value_residual,
        stochastic_depth=0.0,  # no dropout at eval
    ).to(device).bfloat16()
    for nm, p in eval_model.named_parameters():
        if p.ndim < 2 or any(x in nm for x in CONTROL_PATTERNS):
            p.data = p.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear): m.weight.data = m.weight.data.float()
    eval_model.load_state_dict(deq, strict=True)

    vl, vb = eval_val(args, eval_model, rank, world_size, device, grad_accum,
                       val_tokens, bb, hs, ib)
    log0(f"final_int8_roundtrip val_loss:{vl:.8f} val_bpb:{vb:.8f}")

    if args.eval_stride > 0:
        sl, sb = eval_val_sliding(eval_model, device, val_tokens, bb, hs, ib,
                                   seq_len=args.train_seq_len, stride=args.eval_stride,
                                   rank=rank, world_size=world_size)
        log0(f"final_int8_sliding val_loss:{sl:.8f} val_bpb:{sb:.8f} "
             f"stride:{args.eval_stride}")

    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
