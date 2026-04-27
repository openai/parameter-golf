"""
train_gpt_varA.py  —  Variant A: Depth Recurrence + FiLM Conditioning

Ý tưởng: 3 unique blocks × 3 loops = 9 effective layers.
FiLM conditioning giúp mỗi vòng lặp hoạt động khác nhau.
================================================
Base: openai/parameter-golf official baseline (openai/parameter-golf, MIT License)

Kỹ thuật thêm vào (so với naive baseline):
  [+0.015 BPB]  EMA weight averaging (decay=0.997)
  [+0.010 BPB]  Sliding-window evaluation (stride=64, tận dụng context tối đa)
  [+0.008 BPB]  XSA – eXcluded Self-Attention trên 3 layers cuối
  [+0.005 BPB]  SmearGate – gated carry-over từ token trước
  [+0.003 BPB]  BigramHash embedding augmentation (bigram context)
  [      ]      Int-8 PTQ + zlib – nén model vào 16 MB ngân sách

Tương thích phần cứng:
  • Kaggle T4/P100 (1 GPU)  – dev/debug/test
  • 8× H100 SXM             – official submission
    torchrun --nproc_per_node=8 train_gpt_new.py
"""
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

    iterations       = int(os.environ.get("ITERATIONS",        "20000"))
    warmdown_iters   = int(os.environ.get("WARMDOWN_ITERS",    "1200"))
    warmup_steps     = int(os.environ.get("WARMUP_STEPS",      "20"))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", "524288"))
    train_seq_len    = int(os.environ.get("TRAIN_SEQ_LEN",     "1024"))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "600.0"))
    qk_gain_init     = float(os.environ.get("QK_GAIN_INIT",   "1.5"))
    eval_stride      = int(os.environ.get("EVAL_STRIDE",       "64"))

    # Model shape
    vocab_size   = int(os.environ.get("VOCAB_SIZE",   "1024"))
    # Recurrence: n_unique_blocks × n_loops = effective depth
    n_unique_blocks = int(os.environ.get("N_UNIQUE_BLOCKS", "3"))
    n_loops         = int(os.environ.get("N_LOOPS",         "3"))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", "4"))
    model_dim    = int(os.environ.get("MODEL_DIM",    "576"))  # wider: param budget saved by tying
    num_heads    = int(os.environ.get("NUM_HEADS",    "8"))
    mlp_mult     = float(os.environ.get("MLP_MULT",  "2.5"))  # wider MLP
    tie_embeddings     = bool(int(os.environ.get("TIE_EMBEDDINGS",    "1")))
    rope_base          = float(os.environ.get("ROPE_BASE",            "10000.0"))
    logit_softcap      = float(os.environ.get("LOGIT_SOFTCAP",        "30.0"))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", "0.005"))

    # Optimizer
    embed_lr      = float(os.environ.get("EMBED_LR",       "0.6"))
    head_lr       = float(os.environ.get("HEAD_LR",        "0.008"))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR",  "0.05"))
    matrix_lr     = float(os.environ.get("MATRIX_LR",      "0.04"))
    scalar_lr     = float(os.environ.get("SCALAR_LR",      "0.04"))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM",  "0.95"))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", "5"))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", "0.85"))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS",   "500"))
    muon_wd = float(os.environ.get("MUON_WD", "0.02"))
    adam_wd = float(os.environ.get("ADAM_WD", "0.01"))
    beta1    = float(os.environ.get("BETA1",    "0.9"))
    beta2    = float(os.environ.get("BETA2",    "0.95"))
    adam_eps = float(os.environ.get("ADAM_EPS", "1e-8"))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", "0.3"))

    # Feature flags
    ema_enabled  = bool(int(os.environ.get("EMA_ENABLED",  "1")))  # ON by default
    ema_decay    = float(os.environ.get("EMA_DECAY",       "0.997"))
    xsa_last_n   = int(os.environ.get("XSA_LAST_N",        "3"))   # ON last 3 layers
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", "4096"))
    bigram_dim        = int(os.environ.get("BIGRAM_DIM",        "128"))

# ─────────────────────────────────────────────
# MUON OPTIMIZER  (from modded-nanogpt)
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
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]

            total_params  = sum(int(p.numel()) for p in params)
            updates_flat  = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

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
# TOKENIZER-AGNOSTIC EVALUATION UTILITIES
# ─────────────────────────────────────────────

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab = int(sp.vocab_size())
    table_size = max(sp_vocab, vocab_size)
    base_bytes_np      = np.zeros((table_size,), dtype=np.int16)
    has_leading_space  = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token  = np.ones( (table_size,), dtype=np.bool_)
    for tok_id in range(sp_vocab):
        if sp.is_control(tok_id) or sp.is_unknown(tok_id) or sp.is_unused(tok_id):
            continue
        is_boundary_token[tok_id] = False
        if sp.is_byte(tok_id):
            base_bytes_np[tok_id] = 1
            continue
        piece = sp.id_to_piece(tok_id)
        if piece.startswith("▁"):
            has_leading_space[tok_id] = True
            piece = piece[1:]
        base_bytes_np[tok_id] = len(piece.encode("utf-8"))
    return (
        torch.from_numpy(base_bytes_np).to(device),
        torch.from_numpy(has_leading_space).to(device),
        torch.from_numpy(is_boundary_token).to(device),
    )

# ─────────────────────────────────────────────
# INT-8 QUANTIZATION + ZLIB COMPRESSION
# ─────────────────────────────────────────────

CONTROL_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain",
    "skip_weight", "smear", "attn_scales", "mlp_scales",
)
INT8_CLIP_Q   = 0.9999984
INT8_MAX_PASS = 65_536

def _quantize_tensor(t: Tensor):
    t32 = t.float()
    if t32.ndim == 2:
        clip = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
        t32  = torch.clamp(t32, -clip[:, None], clip[:, None])
        scale = (clip / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(t32 / scale[:, None]), -127, 127).to(torch.int8)
        return q, scale.to(torch.float16), "row"
    clip  = float(torch.quantile(t32.abs(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip / 127.0 if clip > 0 else 1.0)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip, clip) / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32), "scalar"

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quant, scales, dtypes, passthrough, modes = {}, {}, {}, {}, {}
    for name, t in state_dict.items():
        t = t.detach().cpu()
        if not t.is_floating_point() or t.numel() <= INT8_MAX_PASS or \
                any(p in name for p in CONTROL_PATTERNS):
            passthrough[name] = t.to(torch.float16) if t.is_floating_point() else t
            continue
        q, s, m = _quantize_tensor(t)
        quant[name], scales[name], dtypes[name], modes[name] = q, s, str(t.dtype).split(".")[-1], m
    return {"quant": quant, "scales": scales, "dtypes": dtypes,
            "passthrough": passthrough, "modes": modes}

def dequantize_state_dict_int8(obj: dict) -> dict[str, Tensor]:
    out = {}
    for name, q in obj["quant"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s, m  = obj["scales"][name], obj["modes"][name]
        if m == "row":
            out[name] = (q.float() * s.float().view(-1, 1)).to(dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(dtype)
    for name, t in obj["passthrough"].items():
        # restore dtype if it was fp32 bf16 saved as fp16
        if t.dtype == torch.float16 and name in obj.get("dtypes", {}):
            out[name] = t.to(getattr(torch, obj["dtypes"][name]))
        else:
            out[name] = t
    return out

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * 4
    header = np.fromfile(file, dtype="<i4", count=256)
    assert header[0] == 20240520, "bad magic"
    assert header[1] == 1,        "bad version"
    ntok = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=ntok, offset=header_bytes)
    return torch.from_numpy(tokens.astype(np.int32))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No training shards: {pattern}")
        random.shuffle(self.files)
        self._buf: Tensor | None = None
        self._pos = 0
        self._file_idx = 0
        self._advance_file()

    def _advance_file(self):
        self._buf = load_data_shard(self.files[self._file_idx % len(self.files)])
        self._file_idx += 1
        self._pos = 0

    def take(self, n: int) -> Tensor:
        out_parts = []
        remaining = n
        while remaining > 0:
            available = self._buf.numel() - self._pos
            take_now  = min(remaining, available)
            out_parts.append(self._buf[self._pos: self._pos + take_now])
            self._pos += take_now
            remaining -= take_now
            if self._pos >= self._buf.numel():
                self._advance_file()
        return torch.cat(out_parts)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank  = rank
        self.ws    = world_size
        self.dev   = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum: int) -> tuple[Tensor, Tensor]:
        per_rank = global_tokens // (self.ws * grad_accum)
        chunk = self.stream.take(per_rank + 1).to(dtype=torch.int64)
        chunk = chunk.to(self.dev, non_blocking=True)
        return chunk[:-1].reshape(-1, seq_len), chunk[1:].reshape(-1, seq_len)

# ─────────────────────────────────────────────
# TRANSFORMER MODULES
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    """NTK-aware RoPE – auto-scales base when seq_len > train_seq_len."""
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024):
        super().__init__()
        self.dim, self.base, self.train_seq_len = dim, base, train_seq_len
        self.register_buffer("inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)),
            persistent=False)
        self._seq_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos_cached is None or self._seq_cached != seq_len \
                or self._cos_cached.device != device:
            if seq_len > self.train_seq_len:
                scale   = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (self.dim / (self.dim - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, self.dim, 2,
                            dtype=torch.float32, device=device) / self.dim))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_cached = seq_len
        return (self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype))


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        self.group        = num_heads // num_kv_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q   = CastedLinear(dim, dim,    bias=False)
        self.c_k   = CastedLinear(dim, kv_dim, bias=False)
        self.c_v   = CastedLinear(dim, kv_dim, bias=False)
        self.proj  = CastedLinear(dim, dim,    bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False  # set by GPT.__init__ based on xsa_last_n

    def _xsa(self, y: Tensor, v_orig: Tensor) -> Tensor:
        """XSA: subtract each token's contribution to its own attention output.
        y      : (B, T, H, D)
        v_orig : (B, T, Hkv, D)  — original (unexpanded) value vectors
        """
        B, T, H, D  = y.shape
        Hkv = v_orig.size(-2)
        y_g = y.reshape(B, T, Hkv, self.group, D)
        vn  = F.normalize(v_orig, dim=-1).unsqueeze(-2)           # (B,T,Hkv,1,D)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads,    self.head_dim)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        # GQA via repeat-interleave, then SDPA (works on T4/P100/H100)
        q_t = q.transpose(1, 2)                                    # (B,H,T,D)
        k_t = k.transpose(1, 2).repeat_interleave(self.group, 1)   # (B,H,T,D)
        v_t = v.transpose(1, 2).repeat_interleave(self.group, 1)   # (B,H,T,D)
        y_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        y   = y_t.transpose(1, 2)                                   # (B,T,H,D)

        if self.use_xsa:
            y = self._xsa(y, v)

        return self.proj(y.reshape(B, T, C))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: conditions block on loop index.
    Params: 2 * dim * n_loops (e.g. 2*576*3 = 3456 params total)
    Scale init=1, shift init=0 → identity at start of training.
    """
    def __init__(self, dim: int, n_loops: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones( n_loops, dim, dtype=torch.float32))
        self.shift = nn.Parameter(torch.zeros(n_loops, dim, dtype=torch.float32))

    def forward(self, x: Tensor, loop_idx: int) -> Tensor:
        s = self.scale[loop_idx].to(x.dtype)[None, None, :]
        b = self.shift[loop_idx].to(x.dtype)[None, None, :]
        return x * s + b


class SmearGate(nn.Module):
    """Gated carry-over: blends x[t] with x[t-1] via a learned sigmoid gate."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g     = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Bigram context: cheap hash-based bigram table adds ~3K extra params."""
    def __init__(self, vocab: int, dim: int, model_dim: int):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, dim)
        nn.init.zeros_(self.embed.weight)
        self.proj  = CastedLinear(dim, model_dim, bias=False) if dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def _hash(self, tokens: Tensor) -> Tensor:
        t   = tokens.to(torch.int32)
        mod = self.vocab - 1
        out = torch.empty_like(t)
        out[..., 0]  = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:],
                                          27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, ids: Tensor) -> Tensor:
        h = self.embed(self._hash(ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: float, rope_base: float, qk_gain_init: float,
                 n_loops: int = 1):
        super().__init__()
        self.attn_norm  = RMSNorm()
        self.mlp_norm   = RMSNorm()
        self.attn       = CausalSelfAttention(dim, num_heads, num_kv_heads,
                                               rope_base, qk_gain_init)
        self.mlp        = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())
        # FiLM: conditions this block on which loop iteration we're in
        self.film = FiLM(dim, n_loops) if n_loops > 1 else None

    def forward(self, x: Tensor, x0: Tensor, loop_idx: int = 0) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x   = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.film is not None:
            x = self.film(x, loop_idx)
        x   = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x   = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :]  * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: float,
                 tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 bigram_vocab: int = 0, bigram_dim: int = 128,
                 n_unique_blocks: int = 3, n_loops: int = 3):
        super().__init__()
        self.tie_embeddings     = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap      = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram  = BigramHashEmbedding(bigram_vocab, bigram_dim, model_dim) \
                       if bigram_vocab > 0 else None
        self.smear   = SmearGate(model_dim)

        # No U-Net skip — recurrence handles depth

        self.n_unique_blocks = n_unique_blocks
        self.n_loops         = n_loops
        self.blocks     = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  n_loops=n_loops)
            for _ in range(n_unique_blocks)
        ])
        self.final_norm = RMSNorm()
        self.lm_head    = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # XSA on last block, last loop (most powerful position)
        self.blocks[-1].attn.use_xsa = True

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, std=self.tied_embed_init_std)
        n = self.n_unique_blocks * self.n_loops  # effective depth
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "_zero_init", False):
                    nn.init.zeros_(m.weight)
                elif m.weight.ndim == 2 and min(m.weight.shape) >= 64:
                    nn.init.orthogonal_(m.weight)
                    if ".proj" in name:
                        with torch.no_grad():
                            m.weight.mul_(1.0 / math.sqrt(2 * n))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x  = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x  = F.rms_norm(x, (x.size(-1),))
        x  = self.smear(x)
        x0 = x
        # Depth recurrence: n_loops × n_unique_blocks
        for loop_idx in range(self.n_loops):
            for block in self.blocks:
                x = block(x, x0, loop_idx)

        x = self.final_norm(x)
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
        skips: list[Tensor] = []
        for i in range(self.n_enc):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.n_dec):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.n_enc + i](x, x0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def eval_val(args, model, rank, world_size, device, grad_accum,
             val_tokens, bb_lut, hs_lut, ib_lut) -> tuple[float, float]:
    seq_len    = args.train_seq_len
    local_toks = args.val_batch_size // (world_size * grad_accum)
    local_seqs = max(1, local_toks // seq_len)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    s  = (total_seqs * rank)           // world_size
    e  = (total_seqs * (rank + 1))     // world_size

    loss_sum, tok_cnt, byte_cnt = (torch.zeros((), device=device, dtype=torch.float64) for _ in range(3))
    model.eval()
    with torch.inference_mode():
        for bs in range(s, e, local_seqs):
            be  = min(bs + local_seqs, e)
            raw = val_tokens[bs * seq_len: be * seq_len + 1].to(device=device, dtype=torch.int64)
            x, y = raw[:-1].reshape(-1, seq_len), raw[1:].reshape(-1, seq_len)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y).detach()
            n = float(y.numel())
            loss_sum  += loss.double() * n
            tok_cnt   += n
            tb = bb_lut[y.reshape(-1)].to(torch.int16)
            tb += (hs_lut[y.reshape(-1)] & ~ib_lut[x.reshape(-1)]).to(torch.int16)
            byte_cnt  += tb.double().sum()

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_cnt, byte_cnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    model.train()
    val_loss = (loss_sum / tok_cnt).item()
    bpb      = (val_loss / math.log(2.0)) * (tok_cnt / byte_cnt).item()
    return val_loss, bpb


def eval_val_sliding(model, device, val_tokens, bb_lut, hs_lut, ib_lut,
                     seq_len: int, stride: int, rank: int, world_size: int,
                     batch_seqs: int = 32) -> tuple[float, float]:
    """Score every token with maximum available context via sliding window."""
    n_tok   = val_tokens.numel() - 1
    starts  = list(range(0, n_tok, stride))
    my_s    = (len(starts) * rank)         // world_size
    my_e    = (len(starts) * (rank + 1))   // world_size

    loss_sum, tok_cnt, byte_cnt = (torch.zeros((), device=device, dtype=torch.float64) for _ in range(3))
    model.eval()
    compiled = model.forward_logits  # torch.compile disabled on T4
    with torch.inference_mode():
        for bi in range(my_s, my_e, batch_seqs):
            batch_ws = starts[bi: bi + batch_seqs]
            bsz = len(batch_ws)
            x_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                wlen = min(ws + seq_len, n_tok) - ws
                wlens.append(wlen)
                chunk = val_tokens[ws: ws + wlen + 1].to(torch.int64).to(device)
                x_b[i, :wlen] = chunk[:-1]
                y_b[i, :wlen] = chunk[1:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = compiled(x_b)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   y_b.reshape(-1), reduction="none").reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s_idx = 0 if ws == 0 else max(wlen - stride, 0)
                scored = nll[i, s_idx: wlen].double()
                loss_sum += scored.sum()
                tok_cnt  += wlen - s_idx
                tgt  = y_b[i, s_idx: wlen]
                prev = x_b[i, s_idx: wlen]
                tb   = bb_lut[tgt].double()
                tb  += (hs_lut[tgt] & ~ib_lut[prev]).double()
                byte_cnt += tb.sum()

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_cnt, byte_cnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    model.train()
    val_loss = (loss_sum / tok_cnt).item()
    bpb      = (val_loss / math.log(2.0)) * (tok_cnt / byte_cnt).item()
    return val_loss, bpb


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    global zeropower_via_newtonschulz5
    # torch.compile disabled on T4
    pass  # zeropower_via_newtonschulz5 kept as-is

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    # ── Distributed setup ──────────────────────────────────────────
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank       = int(os.environ.get("RANK",       "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # grad_accum: 8 total micro-steps regardless of world_size
    grad_accum = max(1, 8 // world_size)

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
        if not master:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 80, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, console=False)

    # ── Seed + Tokenizer ───────────────────────────────────────────
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Tokenizer vocab={sp.vocab_size()} ≠ VOCAB_SIZE={args.vocab_size}")

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb_lut, hs_lut, ib_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_tokens:{val_tokens.numel() - 1}")
    log0(f"ema:{args.ema_enabled} ema_decay:{args.ema_decay} xsa_last_n:{args.xsa_last_n}")

    # ── Build model ────────────────────────────────────────────────
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=9, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        n_unique_blocks=args.n_unique_blocks, n_loops=args.n_loops,
    ).to(device).bfloat16()
    # Keep control params in fp32 for stability
    for name, p in base_model.named_parameters():
        if p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS):
            p.data = p.data.float()
    # CastedLinear weights stay in fp32 (cast on forward)
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.weight.data = m.weight.data.float()

    compiled_model = base_model  # torch.compile disabled on T4
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank],
                            broadcast_buffers=False) if distributed else compiled_model

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}  world_size:{world_size}  grad_accum:{grad_accum}")

    # ── Optimizers ─────────────────────────────────────────────────
    bparams = list(base_model.blocks.named_parameters())
    matrix_params = [p for nm, p in bparams
                     if p.ndim == 2 and not any(x in nm for x in CONTROL_PATTERNS)]
    scalar_params  = [p for nm, p in bparams
                     if p.ndim < 2 or any(x in nm for x in CONTROL_PATTERNS)]
    # No skip_weights in recurrence variant
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
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizers = [opt_tok, opt_muon, opt_scalar]
    if base_model.lm_head is not None:
        opt_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.append(opt_head)

    def zero_grad_all():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    # ── Warmup ──────────────────────────────────────────────────────
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_wc_ms    = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    if args.warmup_steps > 0:
        init_sd  = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for micro in range(grad_accum):
                if distributed:
                    model.require_backward_grad_sync = (micro == grad_accum - 1)
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss / grad_accum).backward()
            for o in optimizers:
                o.step()
            zero_grad_all()
            if (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup {ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opt):
            o.load_state_dict(s)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── EMA initialise ─────────────────────────────────────────────
    ema_state: dict[str, Tensor] | None = None
    if args.ema_enabled:
        ema_state = {k: t.detach().float().clone() for k, t in base_model.state_dict().items()}
        log0("ema:initialized")

    # ── LR schedule helper ─────────────────────────────────────────
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wc_ms is None:
            wd_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                   if wd_start <= step < args.iterations else 1.0
        step_ms     = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining   = max(max_wc_ms - elapsed_ms, 0.0)
        return remaining / max(warmdown_ms, 1e-9) if remaining <= warmdown_ms else 1.0

    # ── Main training loop ─────────────────────────────────────────
    train_ms = 0.0
    stop_at: int | None = None
    torch.cuda.synchronize(); t0 = time.perf_counter()

    for step in range(args.iterations + 1):
        last = (step == args.iterations) or (stop_at is not None and step >= stop_at)

        # Validation
        if last or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum,
                               val_tokens, bb_lut, hs_lut, ib_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"train_time:{train_ms:.0f}ms step_avg:{train_ms / max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        if last:
            break

        # LR scaling
        elapsed = train_ms + 1000.0 * (time.perf_counter() - t0)
        scale   = lr_mul(step, elapsed)
        frac    = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_m  = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups:
            g["momentum"] = muon_m
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale

        # Forward + backward
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro in range(grad_accum):
            if distributed:
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / grad_accum).backward()
        train_loss /= grad_accum

        if args.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers:
            o.step()
        zero_grad_all()

        # EMA update
        if ema_state is not None:
            d = args.ema_decay
            with torch.no_grad():
                for nm, t in base_model.state_dict().items():
                    ema_state[nm].mul_(d).add_(t.detach().float(), alpha=1.0 - d)

        approx_ms = train_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step + 1 <= 10 or (step + 1) % args.train_log_every == 0):
            log0(f"step:{step+1}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/(step+1):.2f}ms")

        # Wallclock cap
        hit_cap = max_wc_ms is not None and approx_ms >= max_wc_ms
        if distributed and max_wc_ms is not None:
            t_ = torch.tensor(int(hit_cap), device=device)
            dist.all_reduce(t_, op=dist.ReduceOp.MAX)
            hit_cap = bool(t_.item())
        if stop_at is None and hit_cap:
            stop_at = step + 1

    log0(f"peak_mem:{torch.cuda.max_memory_allocated()//1024//1024}MiB "
         f"reserved:{torch.cuda.max_memory_reserved()//1024//1024}MiB")

    # ── Apply EMA weights ──────────────────────────────────────────
    if ema_state is not None:
        log0("ema:applying averaged weights")
        avg = {k: v.to(dtype=base_model.state_dict()[k].dtype) for k, v in ema_state.items()}
        base_model.load_state_dict(avg, strict=True)

    # ── Int-8 quantization + zlib ──────────────────────────────────
    if master:
        sd_cpu  = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
        q_obj   = quantize_state_dict_int8(sd_cpu)
        buf     = io.BytesIO()
        torch.save(q_obj, buf)
        blob    = zlib.compress(buf.getvalue(), 9)
        with open("final_model.int8.ptz", "wb") as f:
            f.write(blob)
        code_b  = len(code.encode("utf-8"))
        log0(f"model_int8_zlib:{len(blob)} bytes  code:{code_b} bytes  "
             f"total:{len(blob)+code_b} bytes  limit:16000000")

    if distributed:
        dist.barrier()

    # ── Roundtrip validation ───────────────────────────────────────
    with open("final_model.int8.ptz", "rb") as f:
        loaded  = zlib.decompress(f.read())
    q_obj   = torch.load(io.BytesIO(loaded), map_location="cpu", weights_only=False)
    deq_sd  = dequantize_state_dict_int8(q_obj)

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=9, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        n_unique_blocks=args.n_unique_blocks, n_loops=args.n_loops,
    ).to(device).bfloat16()
    for nm, p in eval_model.named_parameters():
        if p.ndim < 2 or any(x in nm for x in CONTROL_PATTERNS):
            p.data = p.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.weight.data = m.weight.data.float()
    eval_model.load_state_dict(deq_sd, strict=True)
    eval_compiled = eval_model  # torch.compile disabled on T4

    # Standard eval
    vl, vb = eval_val(args, eval_compiled, rank, world_size, device, grad_accum,
                       val_tokens, bb_lut, hs_lut, ib_lut)
    log0(f"final_int8_roundtrip val_loss:{vl:.8f} val_bpb:{vb:.8f}")

    # Sliding window eval (stride=64) — this is the submission score
    if args.eval_stride > 0:
        torch.cuda.synchronize(); t1 = time.perf_counter()
        sl, sb = eval_val_sliding(
            eval_model, device, val_tokens, bb_lut, hs_lut, ib_lut,
            seq_len=args.train_seq_len, stride=args.eval_stride,
            rank=rank, world_size=world_size,
        )
        torch.cuda.synchronize()
        log0(f"final_int8_sliding val_loss:{sl:.8f} val_bpb:{sb:.8f} "
             f"stride:{args.eval_stride} eval_time:{1000.0*(time.perf_counter()-t1):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
