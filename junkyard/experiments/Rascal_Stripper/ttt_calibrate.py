#!/usr/bin/env python3
"""
ttt_calibrate.py — TTT hyperparameter calibration for Rascal family checkpoints.

Loads a trained checkpoint, runs baseline sliding-window eval, then runs one TTT
configuration and reports the delta. Run multiple times with different env vars to sweep.

Usage:
  MODEL_PATH=/workspace/parameter-golf-lab/checkpoints/run_xyz/final_model.pt \
  TTT_LR=0.0001 TTT_EPOCHS=2 TTT_FREEZE_BLOCKS=2 TTT_CHUNK_TOKENS=32768 \
  torchrun --nproc_per_node=8 experiments/Rascal_Stripper/ttt_calibrate.py

Key env vars:
  MODEL_PATH          Path to final_model.pt (REQUIRED)
  NGRAM_MODE          "auto" | "bigram" | "engram"  (default: auto-detect from checkpoint)
  TTT_LR              TTT learning rate          (default: 0.0005)
  TTT_EPOCHS          Epochs per chunk           (default: 3)
  TTT_FREEZE_BLOCKS   Num last blocks to train   (default: 2)
  TTT_CHUNK_TOKENS    Tokens per TTT chunk       (default: 32768)
  TTT_TEMPERATURE     Softmax temp for scoring   (default: 0.98)
  EVAL_STRIDE         Sliding window stride      (default: 64)
  QAT_ENABLED         1 = enable QAT STE during TTT train phase (default: 1)
"""
from __future__ import annotations
import glob, math, os, time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None; tl = None

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    flash_attn_3_func = None


# ── Hyperparameters ────────────────────────────────────────────────────────────
class Hyperparameters:
    # Paths
    data_path         = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    val_files         = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path    = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    # Model arch (must match checkpoint)
    vocab_size        = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers        = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads      = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim         = int(os.environ.get("MODEL_DIM", 512))
    num_heads         = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult          = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings    = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base         = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap     = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init      = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    xsa_last_n        = int(os.environ.get("XSA_LAST_N", 11))
    rope_dims         = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale          = bool(int(os.environ.get("LN_SCALE", "1")))
    ve_enabled        = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim            = int(os.environ.get("VE_DIM", 128))
    ve_layers         = os.environ.get("VE_LAYERS", "9,10")
    attn_scale_init   = float(os.environ.get("ATTN_SCALE_INIT", 1.0))
    mlp_scale_init    = float(os.environ.get("MLP_SCALE_INIT", 1.0))
    resid_mix_x_init  = float(os.environ.get("RESID_MIX_X_INIT", 1.0))
    resid_mix_x0_init = float(os.environ.get("RESID_MIX_X0_INIT", 0.0))
    # BigramHashEmbedding params (turbomuon/baseline checkpoints)
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim        = int(os.environ.get("BIGRAM_DIM", 128))
    # EngramLite params (engramlite/combo checkpoints)
    ngram_buckets     = int(os.environ.get("NGRAM_BUCKETS", 8192))
    ngram_heads       = int(os.environ.get("NGRAM_HEADS", 2))
    ngram_orders      = int(os.environ.get("NGRAM_ORDERS", 2))
    ngram_dim_per_head = int(os.environ.get("NGRAM_DIM_PER_HEAD", 32))
    # Eval
    train_seq_len     = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len      = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride       = int(os.environ.get("EVAL_STRIDE", 64))
    compile_enabled   = bool(int(os.environ.get("COMPILE_ENABLED", "1")))
    compile_fullgraph = bool(int(os.environ.get("COMPILE_FULLGRAPH", "1")))
    compile_mode      = os.environ.get("COMPILE_MODE", "").strip()
    # TTT
    ttt_lr            = float(os.environ.get("TTT_LR", "0.0005"))
    ttt_epochs        = int(os.environ.get("TTT_EPOCHS", "3"))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", "2"))
    ttt_chunk_tokens  = int(os.environ.get("TTT_CHUNK_TOKENS", "32768"))
    ttt_temperature   = float(os.environ.get("TTT_TEMPERATURE", "0.98"))
    qat_enabled       = bool(int(os.environ.get("QAT_ENABLED", "1")))


# ── Utilities ──────────────────────────────────────────────────────────────────
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,"
        "skip_weight,skip_weights,smear,ve_layer_scales,ve_shared.scale,vr_lambda",
    ).split(",") if p
)

SHARD_HEADER_DTYPE  = np.dtype("<i4")
SHARD_TOKEN_DTYPE   = np.dtype("<u2")
SHARD_HEADER_WORDS  = 256
SHARD_MAGIC         = 20240520
SHARD_VERSION       = 1
SHARD_HEADER_BYTES  = SHARD_HEADER_WORDS * SHARD_HEADER_DTYPE.itemsize

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype=SHARD_HEADER_DTYPE, count=SHARD_HEADER_WORDS)
    if int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype=SHARD_TOKEN_DTYPE, count=num_tokens, offset=SHARD_HEADER_BYTES)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

def build_sentencepiece_luts(sp, vocab_size: int, device) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_token_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_leading_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def maybe_compile(fn, *, enabled: bool, fullgraph: bool, mode: str = ""):
    if not enabled:
        return fn
    kwargs = dict(dynamic=False, fullgraph=fullgraph)
    if mode:
        kwargs["mode"] = mode
    return torch.compile(fn, **kwargs)


# ── Triton MLP kernel (optional) ───────────────────────────────────────────────
if triton is not None:
    @triton.jit
    def _leaky_relu_sq_forward_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a = tl.where(x >= 0, x, 0.5 * x)
        tl.store(y_ptr + offsets, a * a, mask=mask)

    @triton.jit
    def _leaky_relu_sq_backward_kernel(x_ptr, grad_out_ptr, grad_in_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a = tl.where(x >= 0, x, 0.5 * x)
        slope = tl.where(x >= 0, 1.0, 0.5)
        tl.store(grad_in_ptr + offsets, grad_out * (2.0 * a * slope), mask=mask)

    class TritonLeakyReluSqFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: Tensor) -> Tensor:
            x_c = x.contiguous(); y = torch.empty_like(x_c)
            grid = lambda m: (triton.cdiv(x_c.numel(), m["BLOCK_SIZE"]),)
            _leaky_relu_sq_forward_kernel[grid](x_c, y, x_c.numel(), BLOCK_SIZE=1024)
            ctx.save_for_backward(x_c); return y
        @staticmethod
        def backward(ctx, grad_out: Tensor):
            (x,) = ctx.saved_tensors
            g = grad_out.contiguous(); gi = torch.empty_like(g)
            grid = lambda m: (triton.cdiv(g.numel(), m["BLOCK_SIZE"]),)
            _leaky_relu_sq_backward_kernel[grid](x, g, gi, g.numel(), BLOCK_SIZE=1024)
            return (gi,)

def leaky_relu_sq(x: Tensor, kernel_mode: str = "") -> Tensor:
    if kernel_mode == "triton_act" and triton is not None:
        return TritonLeakyReluSqFn.apply(x)
    return F.leaky_relu(x, negative_slope=0.5).square()


# ── Model ──────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps or 1e-6
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim; self.base = base; self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0; self._cos_cached = None; self._sin_cached = None
    def forward(self, seq_len: int, device, dtype) -> tuple[Tensor, Tensor]:
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                sc = seq_len / self.train_seq_len
                new_base = self.base * (sc ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads; self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, q_w, k_w, v_w, out_w, v_embed=None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None: v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if flash_attn_3_func is not None:
            qa, ka, va = q, k, v
            if qa.dtype not in (torch.float16, torch.bfloat16):
                qa, ka, va = qa.to(torch.bfloat16), ka.to(torch.bfloat16), va.to(torch.bfloat16)
            y = flash_attn_3_func(qa, ka, va, causal=True)
        else:
            qh = q.transpose(1, 2); kh = k.transpose(1, 2); vh = v.transpose(1, 2)
            if self.num_heads != self.num_kv_heads:
                r = self.num_heads // self.num_kv_heads
                kh = kh.repeat_interleave(r, dim=1); vh = vh.repeat_interleave(r, dim=1)
            y = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True).transpose(1, 2)
        if self.use_xsa: y = self._xsa_efficient(y, v)
        return F.linear(y.reshape(bsz, seqlen, dim), out_w.to(x.dtype))

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, model_dim: int, trigram: bool = False):
        super().__init__()
        self.bigram_vocab_size = vocab_size; self._trigram = trigram
        self.embed = nn.Embedding(vocab_size, dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(dim, model_dim, bias=False) if dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32); mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def trigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32); mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., :2] = mod
        out[..., 2:] = (36313 * t[..., 2:] ^ 27191 * t[..., 1:-1] ^ 51497 * t[..., :-2]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self._trigram: h = h + self.embed(self.trigram_hash(token_ids))
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class EngramLite(nn.Module):
    def __init__(self, num_buckets, num_heads, num_orders, dim_per_head, model_dim):
        super().__init__()
        self.num_buckets = num_buckets; self.num_heads = num_heads; self.num_orders = num_orders
        total_slots = num_orders * num_heads * num_buckets
        concat_dim = num_orders * num_heads * dim_per_head
        self.embed = nn.Embedding(total_slots, dim_per_head)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(concat_dim, model_dim, bias=False)
        self.proj._zero_init = True
        self.ngram_gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
    def forward(self, input_ids):
        B = self.num_buckets
        prev = F.pad(input_ids[:, :-1], (1, 0), value=0)
        bi_h0 = (prev * 1009 + input_ids) % B
        bi_h1 = ((prev * 2719 + 314159) ^ (input_ids * 3137)) % B
        indices = [bi_h0, bi_h1 + B]
        if self.num_orders >= 2:
            pp = F.pad(prev[:, :-1], (1, 0), value=0)
            tri_h0 = ((pp * 36313) ^ (prev * 27191) ^ (input_ids * 4903)) % B
            tri_h1 = ((pp * 7919) ^ (prev * 4391) ^ (input_ids * 6151)) % B
            off = 2 * B
            indices.extend([tri_h0 + off, tri_h1 + off + B])
        all_idx = torch.stack(indices, dim=-1)
        all_emb = self.embed(all_idx)
        flat = all_emb.reshape(*input_ids.shape, -1)
        out = self.proj(flat)
        gate = torch.sigmoid(self.ngram_gate.to(dtype=out.dtype))[None, None, :]
        return out * gate

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        self.kernel_mode = os.environ.get("MLP_KERNEL_MODE", "").strip().lower()
    def forward(self, x: Tensor, up_w, down_w) -> Tensor:
        x = F.linear(x, up_w.to(x.dtype))
        x = leaky_relu_sq(x, kernel_mode=self.kernel_mode)
        return F.linear(x, down_w.to(x.dtype))

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.full((dim,), float(os.environ.get("ATTN_SCALE_INIT", "1.0")), dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.full((dim,), float(os.environ.get("MLP_SCALE_INIT",  "1.0")), dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((
            torch.full((dim,), float(os.environ.get("RESID_MIX_X_INIT",  "1.0")), dtype=torch.float32),
            torch.full((dim,), float(os.environ.get("RESID_MIX_X0_INIT", "0.0")), dtype=torch.float32),
        )))
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w, v_embed=None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, q_w, k_w, v_w, out_w, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out

class GPT(nn.Module):
    def __init__(self, args: Hyperparameters, ngram_mode: str = "bigram"):
        super().__init__()
        n = args.num_layers
        d = args.model_dim
        self.tie_embeddings = args.tie_embeddings
        self.tied_embed_init_std = args.tied_embed_init_std
        self.logit_softcap = args.logit_softcap
        self.tok_emb = nn.Embedding(args.vocab_size, d)
        if ngram_mode == "engram":
            self.bigram = EngramLite(args.ngram_buckets, args.ngram_heads, args.ngram_orders, args.ngram_dim_per_head, d)
        else:
            self.bigram = BigramHashEmbedding(args.bigram_vocab_size, args.bigram_dim, d) if args.bigram_vocab_size > 0 else None
        self.smear = SmearGate(d)
        self.num_encoder_layers = n // 2
        self.num_decoder_layers = n - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, d, dtype=torch.float32))
        head_dim = d // args.num_heads; kv_dim = args.num_kv_heads * head_dim
        mlp_dim = int(args.mlp_mult * d)
        self.num_layers = n
        self.qo_bank      = nn.Parameter(torch.empty(2 * n, d, d))
        self.kv_bank      = nn.Parameter(torch.empty(2 * n, kv_dim, d))
        self.mlp_up_bank  = nn.Parameter(torch.empty(n, mlp_dim, d))
        self.mlp_down_bank = nn.Parameter(torch.empty(n, d, mlp_dim))
        self.blocks = nn.ModuleList([Block(d, args.num_heads, args.num_kv_heads, args.mlp_mult, args.rope_base, args.qk_gain_init, i, args.ln_scale) for i in range(n)])
        if args.rope_dims > 0:
            for block in self.blocks:
                block.attn.rope_dims = args.rope_dims
                block.attn.rotary = Rotary(head_dim, base=args.rope_base, train_seq_len=1024, rope_dims=args.rope_dims)
        self.ve_layer_indices = [int(x) for x in args.ve_layers.split(",") if x.strip()] if args.ve_enabled else []
        kv_dim_ve = args.num_kv_heads * head_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(args.vocab_size, args.ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None; self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()
        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else CastedLinear(d, args.vocab_size, bias=False)
        if args.xsa_last_n > 0:
            for i in range(max(0, n - args.xsa_last_n), n):
                self.blocks[i].attn.use_xsa = True
    def _get_ve(self, layer_idx, input_ids, ve_cache=None):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices: return None
        if ve_cache is not None and 've' not in ve_cache: ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def _body(self, input_ids):
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None: x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),)); x = self.smear(x); x0 = x
        skips = []; ve_cache = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, self.qo_bank[i], self.kv_bank[i], self.kv_bank[n+i], self.qo_bank[n+i], self.mlp_up_bank[i], self.mlp_down_bank[i], v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n+bi], self.qo_bank[n+bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi], v_embed=ve)
        return self.final_norm(x)
    def forward(self, input_ids, target_ids) -> Tensor:
        x = self._body(input_ids)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        lp = F.linear(x_flat, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(lp / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets)
    def forward_logits(self, input_ids) -> Tensor:
        x = self._body(input_ids)
        lp = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)


# ── Eval functions (identical to train_gpt_turbomuon.py) ─────────────────────
def eval_val_sliding(args, base_model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, stride, batch_seqs=32, eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    my_s = (len(window_starts) * rank) // world_size
    my_e = (len(window_starts) * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = maybe_compile(base_model.forward_logits, enabled=args.compile_enabled, fullgraph=args.compile_fullgraph)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]; bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens); wlen = end - ws; wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]; y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction="none").reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]; s = 0 if ws == 0 else max(wlen - stride, 0)
                loss_sum += nll[i, s:wlen].to(torch.float64).sum(); token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    base_model.train()
    return val_loss, val_bpb


def eval_val_sliding_ttt(args, base_model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, stride, eval_seq_len=None):
    """Legal score-first TTT: score each chunk FIRST, then train on it."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk_tokens = args.ttt_chunk_tokens
    ttt_epochs = args.ttt_epochs
    ttt_lr = args.ttt_lr
    ttt_freeze_blocks = args.ttt_freeze_blocks
    ttt_temp = args.ttt_temperature
    batch_seqs = 32
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + ttt_chunk_tokens - 1) // ttt_chunk_tokens
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens); wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        ci = min((ws + s) // ttt_chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)
    for p in base_model.parameters(): p.requires_grad_(False)
    num_blocks = len(base_model.blocks); ttt_params = []; seen_ids = set()
    for i in range(max(0, num_blocks - ttt_freeze_blocks), num_blocks):
        for p in base_model.blocks[i].parameters():
            if id(p) not in seen_ids: p.requires_grad_(True); ttt_params.append(p); seen_ids.add(id(p))
    for name, p in base_model.named_parameters():
        if ("norm" in name or "scale" in name or "lm_head" in name) and id(p) not in seen_ids:
            p.requires_grad_(True); ttt_params.append(p); seen_ids.add(id(p))
    optimizer = torch.optim.AdamW(ttt_params, lr=ttt_lr, weight_decay=0.0, betas=(0.9, 0.999))
    polyak_decay = 0.998
    polyak_state = {id(p): p.data.clone() for p in ttt_params}
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()
    if rank == 0:
        print(f"ttt:start chunks={num_chunks} chunk_tokens={ttt_chunk_tokens} lr={ttt_lr} epochs={ttt_epochs} freeze_last={ttt_freeze_blocks}", flush=True)
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows: continue
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]
        if ci > 0:
            saved = {id(p): p.data.clone() for p in ttt_params}
            for p in ttt_params: p.data.copy_(polyak_state[id(p)])
        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]; bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens); wlen = end - ws; wlens.append(wlen)
                    tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = tok[:-1]; y_batch[i, :wlen] = tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy((logits.float() / ttt_temp).reshape(-1, logits.size(-1)), y_batch.reshape(-1), reduction="none").reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]; s = 0 if ws == 0 else max(wlen - stride, 0)
                    loss_sum += nll[i, s:wlen].to(torch.float64).sum(); token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
        if ci > 0:
            for p in ttt_params: p.data.copy_(saved[id(p)])
        is_last = ci == num_chunks - 1
        if not is_last and ttt_epochs > 0:
            chunk_start = ci * ttt_chunk_tokens
            chunk_end = min((ci + 1) * ttt_chunk_tokens, total_tokens)
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                progress = min(ci / max(num_chunks * 0.3, 1), 1.0)
                cos_lr *= 1.0 + 2.0 * progress
                for pg in optimizer.param_groups: pg["lr"] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                base_model.train()
                for _ep in range(ttt_epochs):
                    for bs in range(my_seq_s, my_seq_e, batch_seqs):
                        be = min(bs + batch_seqs, my_seq_e)
                        start_tok = chunk_start + bs * seq_len
                        end_tok = chunk_start + be * seq_len + 1
                        if end_tok > val_tokens.numel(): continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len); y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            ttt_logits = base_model.forward_logits(x)
                            per_tok = F.cross_entropy(ttt_logits.reshape(-1, ttt_logits.size(-1)), y.reshape(-1), reduction="none").reshape(y.shape)
                            bw = base_bytes_lut[y].float()
                            bw += (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).float()
                            ttt_loss = (per_tok * bw).sum() / bw.sum()
                        ttt_loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        optimizer.step()
                        for p in ttt_params: polyak_state[id(p)].lerp_(p.data, 1.0 - polyak_decay)
        if rank == 0 and (ci % 20 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1))
            print(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} t={elapsed:.1f}s", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    for p in base_model.parameters(): p.requires_grad_(True)
    base_model.eval()
    if rank == 0:
        print(f"ttt:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} elapsed={time.perf_counter()-t0:.1f}s")
    return val_loss, val_bpb


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    model_path = os.environ.get("MODEL_PATH", "")
    if not model_path:
        raise ValueError("MODEL_PATH env var is required. Point it at a final_model.pt")

    args = Hyperparameters()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    def log0(msg: str):
        if master: print(msg, flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Auto-detect ngram mode from checkpoint keys
    ngram_mode = os.environ.get("NGRAM_MODE", "auto").lower()
    if ngram_mode == "auto":
        sd_keys = list(torch.load(model_path, map_location="cpu", weights_only=True).keys())
        if "bigram.ngram_gate" in sd_keys:
            ngram_mode = "engram"
            log0("NGRAM_MODE=auto → detected EngramLite checkpoint")
        else:
            ngram_mode = "bigram"
            log0("NGRAM_MODE=auto → detected BigramHashEmbedding checkpoint")

    # Tokenizer + LUTs
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    # Val tokens
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    log0(f"val_tokens: {val_tokens.numel()}")

    # Build model
    model = GPT(args, ngram_mode=ngram_mode).to(device).bfloat16()

    # Load checkpoint
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:   log0(f"WARNING missing keys: {missing[:5]}")
    if unexpected: log0(f"WARNING unexpected keys: {unexpected[:5]}")
    model.eval()
    log0(f"Loaded: {model_path}")

    # QAT: enable for TTT train phase (matches training regime)
    CastedLinear._qat_enabled = args.qat_enabled

    seq_len = args.eval_seq_len

    # ── Baseline sliding window ────────────────────────────────────────────────
    log0(f"\n{'='*56}")
    log0(f"  BASELINE sliding window  stride={args.eval_stride}")
    log0(f"{'='*56}")
    t0 = time.perf_counter()
    sw_loss, sw_bpb = eval_val_sliding(
        args, model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.eval_stride, eval_seq_len=seq_len,
    )
    log0(f"baseline_sliding val_loss:{sw_loss:.8f} val_bpb:{sw_bpb:.8f}  ({time.perf_counter()-t0:.1f}s)")

    # ── TTT ───────────────────────────────────────────────────────────────────
    log0(f"\n{'='*56}")
    log0(f"  TTT  lr={args.ttt_lr}  epochs={args.ttt_epochs}  freeze_blocks={args.ttt_freeze_blocks}")
    log0(f"       chunk_tokens={args.ttt_chunk_tokens}  temperature={args.ttt_temperature}")
    log0(f"{'='*56}")
    ttt_loss, ttt_bpb = eval_val_sliding_ttt(
        args, model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.eval_stride, eval_seq_len=seq_len,
    )
    delta = ttt_bpb - sw_bpb
    log0(f"\nttt_result val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
    log0(f"ttt_delta  bpb:{delta:+.8f}  ({'BETTER' if delta < 0 else 'WORSE'})")
    log0(f"\n{'='*56}")
    log0(f"  SUMMARY")
    log0(f"  baseline : {sw_bpb:.8f}")
    log0(f"  ttt      : {ttt_bpb:.8f}")
    log0(f"  delta    : {delta:+.8f}")
    log0(f"  config   : lr={args.ttt_lr} ep={args.ttt_epochs} freeze={args.ttt_freeze_blocks} chunk={args.ttt_chunk_tokens}")
    log0(f"{'='*56}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
