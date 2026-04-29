"""SP8192 + Parallel Residuals + Depth Recurrence + LoRA Score-First TTT + Mixed int4/int6/int8 + AWQ.

Single-file, torchrun-friendly entry point for the parameter-golf 10-min/16-MB track.

Run:
    pip install sentencepiece numpy
    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
        python3 data/cached_challenge_fineweb.py --variant sp8192
    SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py

Architecture (forked from PR #1394 SP8192 stack):
  * 16 transformer blocks split into three zones:
        pre-recurrent  [0, 3)        encoder (push U-Net skips)
        recurrent      [3, 7)        middle band, looped recurrence_count=3 times
        post-recurrent [7, 16)       decoder (pop skips in reverse)
    Effective forward passes per token = 3 + 4*3 + 9 = 24 (24 vs 16 raw layers).
    Stored block params unchanged from a 16-block model: the recurrent zone
    re-uses the same 4 blocks across all 3 loop iterations.
  * GPT-J style parallel residuals: attention and MLP both read the same
    post-resid_mix input and both write into the residual stream in a single
    fused update.
  * GQA attention with RoPE + per-head learnable QK-Gain + RMSNorm, relu^2 MLP,
    tied embeddings, logit_softcap=15.
  * Muon optimizer for matrix params, Adam for tok_emb and scalars.

LoRA score-first TTT:
  * At eval time, attach LoRA(rank=16) adapters on every linear in every block:
    attn.{c_q,c_k,c_v,proj}, mlp.{fc,proj}.
  * Walk val tokens in chunks of 16384. For each chunk: SCORE under no_grad first
    (numbers locked into val_bpb), THEN take 4 Adam steps on LoRA-only params on
    that same chunk. Each token is scored exactly once before any gradient sees
    it -- this is the legal score-first protocol.

Mixed-precision int4/int6/int8 + AWQ + zstd export:
  * tok_emb / lm_head             -> int8 per-row sym (k_sd=20)
  * any .attn.proj.weight         -> int6 per-row sym
  * blocks.0.* and blocks.<N-1>.* -> int6 per-row sym (first/last sensitivity)
  * every other 2D matrix         -> int4 ASYM per-row, with optional AWQ
                                      activation-aware in-channel scale
  * <= 65k float tensors / scalars -> fp16 / fp32 passthrough
  * LoRA params                   -> stripped (runtime-only)
  * Final blob is zstd(level=22) when `zstandard` is importable, with a
    transparent zlib(level=9) fallback otherwise. Round-trip-decoded model
    is then evaluated end-to-end with sliding eval + score-first TTT.

Adapted from the openai/parameter-golf reference train_gpt.py and the SP8192
stack from PR #1394 (@clarkkev). See README.md for full credits.
"""
from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Optional: zstd at level 22 typically saves ~3-6% over zlib(9) on this kind
# of payload (mixed int4/int6/int8 packed weight tables), which can buy back
# enough bytes to keep the artifact under 16,000,000 when the model is at the
# edge. We fall back to zlib transparently if `zstandard` is not installed in
# the eval environment, so the script is not a hard dependency.
try:
    import zstandard as _zstd  # type: ignore
    _HAS_ZSTD = True
except ImportError:  # pragma: no cover - zstd is optional
    _zstd = None
    _HAS_ZSTD = False


# --------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)
LORA_NAME_PATTERNS = (".lora_a", ".lora_b", "lora_A", "lora_B")


# --------------------------------------------------------------------
# HYPERPARAMETERS
# --------------------------------------------------------------------
def _env_int(key, default):
    v = os.environ.get(key)
    return int(v) if v is not None and v != "" else int(default)


def _env_float(key, default):
    v = os.environ.get(key)
    return float(v) if v is not None and v != "" else float(default)


def _env_bool(key, default):
    v = os.environ.get(key)
    if v is None or v == "":
        return bool(default)
    return v.lower() in ("1", "true", "yes", "on")


def _env_str(key, default):
    v = os.environ.get(key)
    return v if v is not None and v != "" else default


class Hyperparameters:
    """All hyperparameters live in this single class. Defaults match the run
    configuration baked into the submission. Overridable via env vars for
    convenience (SEED, ITERATIONS, MAX_WALLCLOCK_SECONDS, DATA_PATH,
    TOKENIZER_PATH, RUN_ID, TTT_ENABLED).
    """
    _script_dir = str(Path(__file__).resolve().parent)

    # ---- data + tokenizer ----
    run_id          = _env_str("RUN_ID", "ttt_recur_parres_sp8192_16L_int4awq")
    train_shards    = _env_int("TRAIN_SHARDS", 80)
    data_path       = _env_str(
        "DATA_PATH",
        os.path.join(_script_dir, "data/datasets/fineweb10B_sp8192"),
    )
    train_files     = os.path.join(data_path, "fineweb_train_*.bin")
    val_files       = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path  = _env_str(
        "TOKENIZER_PATH",
        os.path.join(_script_dir, "data/tokenizers/fineweb_8192_bpe.model"),
    )

    # ---- logging / seed ----
    seed             = _env_int("SEED", 1337)
    val_batch_size   = _env_int("VAL_BATCH_SIZE", 524288)
    val_loss_every   = _env_int("VAL_LOSS_EVERY", 1000)
    train_log_every  = _env_int("TRAIN_LOG_EVERY", 200)

    # ---- training ----
    iterations            = _env_int  ("ITERATIONS",            100000)
    warmdown_iters        = _env_int  ("WARMDOWN_ITERS",          1800)
    warmup_steps          = _env_int  ("WARMUP_STEPS",              20)
    train_batch_tokens    = _env_int  ("TRAIN_BATCH_TOKENS",    524288)
    train_seq_len         = _env_int  ("TRAIN_SEQ_LEN",           2048)
    max_wallclock_seconds = _env_float("MAX_WALLCLOCK_SECONDS",    0.0)
    grad_accum_steps      = _env_int  ("GRAD_ACCUM_STEPS",           1)

    # ---- architecture ----
    vocab_size       = _env_int  ("VOCAB_SIZE",        8192)
    num_layers       = _env_int  ("NUM_LAYERS",          16)
    model_dim        = _env_int  ("MODEL_DIM",          512)
    num_heads        = _env_int  ("NUM_HEADS",            8)
    num_kv_heads     = _env_int  ("NUM_KV_HEADS",         4)
    mlp_mult         = _env_int  ("MLP_MULT",             3)
    tie_embeddings   = _env_bool ("TIE_EMBEDDINGS",    True)
    logit_softcap    = _env_float("LOGIT_SOFTCAP",     15.0)
    rope_base        = _env_float("ROPE_BASE",      10000.0)
    qk_gain_init     = _env_float("QK_GAIN_INIT",       1.5)
    parallel_residuals    = _env_bool("PARALLEL_RESIDUALS",    True)
    recurrent_layer_start = _env_int ("RECURRENT_LAYER_START",     3)
    recurrent_layer_end   = _env_int ("RECURRENT_LAYER_END",       7)
    recurrence_count      = _env_int ("RECURRENCE_COUNT",          3)

    # ---- optimizer ----
    embed_lr            = _env_float("EMBED_LR",                  0.6)
    head_lr             = _env_float("HEAD_LR",                 0.008)
    tied_embed_lr       = _env_float("TIED_EMBED_LR",            0.05)
    tied_embed_init_std = _env_float("TIED_EMBED_INIT_STD",     0.005)
    matrix_lr           = _env_float("MATRIX_LR",               0.032)
    scalar_lr           = _env_float("SCALAR_LR",                0.04)
    muon_momentum       = _env_float("MUON_MOMENTUM",            0.97)
    muon_backend_steps  = _env_int  ("MUON_BACKEND_STEPS",          6)
    muon_momentum_warmup_start = _env_float("MUON_MOMENTUM_WARMUP_START", 0.85)
    muon_momentum_warmup_steps = _env_int  ("MUON_MOMENTUM_WARMUP_STEPS",  500)
    beta1               = _env_float("BETA1",                     0.9)
    beta2               = _env_float("BETA2",                    0.95)
    adam_eps            = _env_float("ADAM_EPS",                 1e-8)
    grad_clip_norm      = _env_float("GRAD_CLIP_NORM",            1.0)

    # ---- TTT ----
    ttt_enabled            = _env_bool ("TTT_ENABLED",             True)
    ttt_lora_rank          = _env_int  ("TTT_LORA_RANK",             16)
    ttt_lora_alpha         = _env_float("TTT_LORA_ALPHA",          16.0)
    ttt_lora_targets       = _env_str  ("TTT_LORA_TARGETS",
                                        "attn_proj+mlp_proj+attn_qkv+mlp_fc")
    ttt_chunk_tokens       = _env_int  ("TTT_CHUNK_TOKENS",       16384)
    ttt_steps_per_chunk    = _env_int  ("TTT_STEPS_PER_CHUNK",        4)
    ttt_lr                 = _env_float("TTT_LR",                  1e-3)
    ttt_lora_block_first   = _env_int  ("TTT_LORA_BLOCK_FIRST",       0)
    ttt_lora_block_last    = _env_int  ("TTT_LORA_BLOCK_LAST",       -1)

    # ---- quantization (mixed int4/int6/int8 + optional AWQ) ----
    quant_scheme           = _env_str  ("QUANT_SCHEME",  "mixed_int4_int6_int8")
    quant_clip_mode        = _env_str  ("QUANT_CLIP_MODE",            "std")
    quant_embed_k_sd       = _env_float("QUANT_EMBED_K_SD",            20.0)
    quant_matrix_k_sd      = _env_float("QUANT_MATRIX_K_SD",          12.85)
    quant_int4_k_sd        = _env_float("QUANT_INT4_K_SD",              8.0)
    quant_int4_clip_q      = _env_float("QUANT_INT4_CLIP_Q",        0.99985)
    quant_int6_first_last  = _env_bool ("QUANT_INT6_FIRST_LAST",       True)
    quant_int6_attn_proj   = _env_bool ("QUANT_INT6_ATTN_PROJ",        True)
    quant_awq_enabled      = _env_bool ("QUANT_AWQ_ENABLED",           True)
    quant_awq_alpha        = _env_float("QUANT_AWQ_ALPHA",              0.5)
    quant_awq_calib_chunks = _env_int  ("QUANT_AWQ_CALIB_CHUNKS",         4)
    quant_awq_eps          = _env_float("QUANT_AWQ_EPS",               1e-5)


# --------------------------------------------------------------------
# MUON OPTIMIZER
# --------------------------------------------------------------------
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
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

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
            lr = group["lr"]
            momentum = group["momentum"]
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
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# --------------------------------------------------------------------
# TRANSFORMER MODULES
# --------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute."""
    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        w = self.weight.to(x.dtype)
        out = F.linear(x, w, bias)
        # Optional LoRA delta. Attached at TTT time by attach_lora_adapters().
        # We register lora_A / lora_B as Parameters under this Linear module so
        # DDP / state_dict sees them naturally if/when present.
        if getattr(self, "_lora_active", False):
            A = self.lora_A.to(x.dtype)
            B = self.lora_B.to(x.dtype)
            out = out + F.linear(F.linear(x, A), B) * self._lora_scale
        return out


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(s in name for s in CONTROL_TENSOR_NAME_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim

        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    """relu^2 MLP from the original modded-nanogpt setup."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 parallel_residuals=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        # parallel_residuals: GPT-J style. When True, attn and mlp both
        # read from the same post-resid_mix input and BOTH writes are
        # added into the residual stream in a single update. When False
        # we fall back to the classic sequential (attn then mlp) form.
        self.parallel_residuals = bool(parallel_residuals)

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.parallel_residuals:
            # Both branches see the SAME input. Reduces serial depth
            # and tends to give ~0.01 nats on this benchmark.
            attn_out = self.attn(self.attn_norm(x))
            mlp_out  = self.mlp(self.mlp_norm(x))
            x = (
                x
                + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
                + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
            )
        else:
            attn_out = self.attn(self.attn_norm(x))
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {args.logit_softcap}")
        self.tie_embeddings = args.tie_embeddings
        self.tied_embed_init_std = args.tied_embed_init_std
        self.logit_softcap = args.logit_softcap

        # ---- depth-recurrence layout ----
        # Three zones over `num_layers`:
        #   [0, rec_start)              -> pre-recurrent (encoder side, push skips)
        #   [rec_start, rec_end)        -> recurrent middle (looped recurrence_count times,
        #                                  no skip ops inside the loop)
        #   [rec_end, num_layers)       -> post-recurrent (decoder side, pop skips reverse)
        # If rec_end == rec_start (default), recurrence is disabled and we fall back
        # to the classic 4-encoder / 5-decoder U-Net split for backward-compat.
        self.num_total_layers = int(args.num_layers)
        rec_start = int(args.recurrent_layer_start)
        rec_end   = int(args.recurrent_layer_end)
        rec_count = max(1, int(args.recurrence_count))
        if rec_end == rec_start:
            # Backward-compat: classic U-Net split, no looping.
            self.use_recurrence = False
            self.num_pre_layers   = self.num_total_layers // 2
            self.num_recur_layers = 0
            self.num_post_layers  = self.num_total_layers - self.num_pre_layers
            self.recurrence_count = 1
        else:
            if not (0 <= rec_start < rec_end <= self.num_total_layers):
                raise ValueError(
                    f"bad recurrent_layer range [{rec_start},{rec_end}) over {self.num_total_layers}"
                )
            self.use_recurrence = True
            self.num_pre_layers   = rec_start
            self.num_recur_layers = rec_end - rec_start
            self.num_post_layers  = self.num_total_layers - rec_end
            self.recurrence_count = rec_count
        self.num_skip_weights = min(self.num_pre_layers, self.num_post_layers)

        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.skip_weights = nn.Parameter(
            torch.ones(max(self.num_skip_weights, 1), args.model_dim, dtype=torch.float32)
        )
        # If we have zero skip pairs, mark skip_weights as unused at forward time.
        self._has_skips = self.num_skip_weights > 0

        self.blocks = nn.ModuleList([
            Block(
                dim=args.model_dim,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                mlp_mult=args.mlp_mult,
                rope_base=args.rope_base,
                qk_gain_init=args.qk_gain_init,
                parallel_residuals=getattr(args, "parallel_residuals", False),
            )
            for _ in range(self.num_total_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else CastedLinear(args.model_dim, args.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _pre_idx(self, i): return i
    def _recur_idx(self, j): return self.num_pre_layers + j
    def _post_idx(self, k): return self.num_pre_layers + self.num_recur_layers + k

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        # ---- pre-recurrent: encoder, pushes skips ----
        for i in range(self.num_pre_layers):
            x = self.blocks[self._pre_idx(i)](x, x0)
            if self._has_skips and i < self.num_skip_weights:
                skips.append(x)

        # ---- recurrent middle: loop the same blocks N times, no skip ops ----
        # Each pass re-runs the same Python sub-loop. resid_mix re-injects x0
        # every block, which keeps the loop a refinement pass (not a stack of
        # blind compositions). torch.compile traces this fine with a fixed
        # rec_count; we keep it as a plain Python int so dynamic=False works.
        for _ in range(self.recurrence_count):
            for j in range(self.num_recur_layers):
                x = self.blocks[self._recur_idx(j)](x, x0)

        # ---- post-recurrent: decoder, pops skips in reverse ----
        for k in range(self.num_post_layers):
            if self._has_skips and skips and k < self.num_skip_weights:
                x = x + self.skip_weights[k].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self._post_idx(k)](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        # AWQ calibration uses target_ids=None purely for activation capture.
        if target_ids is None:
            return logits
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# --------------------------------------------------------------------
# LoRA TTT helpers
# --------------------------------------------------------------------
def _iter_lora_target_linears(model, targets, block_first, block_last):
    """Yield (block_idx, name, CastedLinear) for matrices that should receive a LoRA adapter."""
    target_set = set(t.strip() for t in targets.split("+") if t.strip())
    if block_last < 0:
        block_last = model.num_total_layers - 1
    for bi, blk in enumerate(model.blocks):
        if bi < block_first or bi > block_last:
            continue
        if "attn_proj" in target_set:
            yield bi, "attn.proj", blk.attn.proj
        if "mlp_proj" in target_set:
            yield bi, "mlp.proj", blk.mlp.proj
        if "attn_qkv" in target_set:
            yield bi, "attn.c_q", blk.attn.c_q
            yield bi, "attn.c_k", blk.attn.c_k
            yield bi, "attn.c_v", blk.attn.c_v
        if "mlp_fc" in target_set:
            yield bi, "mlp.fc", blk.mlp.fc


def attach_lora_adapters(model, rank, alpha, targets, block_first=0, block_last=-1):
    """Attach (and zero-initialize the B side of) LoRA adapters on the chosen
    Linear modules. Adapters are registered as Parameters so they participate
    in autograd / DDP, but they're EXCLUDED from the int6/int8 export. After
    attach the model output is unchanged because B = 0."""
    scale = float(alpha) / float(rank)
    n_attached = 0
    for bi, _, lin in _iter_lora_target_linears(model, targets, block_first, block_last):
        in_features  = lin.in_features
        out_features = lin.out_features
        device = lin.weight.device
        # A: (rank, in)  random-small;  B: (out, rank)  zeros.
        A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=torch.float32))
        nn.init.normal_(A, mean=0.0, std=1.0 / math.sqrt(in_features))
        B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=torch.float32))
        # Register on the linear so state_dict naming is `<lin path>.lora_A`.
        lin.register_parameter("lora_A", A)
        lin.register_parameter("lora_B", B)
        lin._lora_active = True
        lin._lora_scale = scale
        n_attached += 1
    return n_attached


def detach_lora_adapters(model):
    """Remove all LoRA adapters from the model (in-place)."""
    n_removed = 0
    for m in model.modules():
        if isinstance(m, CastedLinear) and getattr(m, "_lora_active", False):
            if hasattr(m, "lora_A"):
                del m._parameters["lora_A"]
            if hasattr(m, "lora_B"):
                del m._parameters["lora_B"]
            m._lora_active = False
            if hasattr(m, "_lora_scale"):
                del m._lora_scale
            n_removed += 1
    return n_removed


def reset_lora_adapters(model):
    """Reset every attached LoRA adapter back to its init state (B=0, A small-random)."""
    n_reset = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, CastedLinear) and getattr(m, "_lora_active", False):
                if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                    in_features = m.lora_A.shape[1]
                    nn.init.normal_(m.lora_A, mean=0.0, std=1.0 / math.sqrt(in_features))
                    nn.init.zeros_(m.lora_B)
                    n_reset += 1
    return n_reset


def lora_parameters(model):
    """Yield only the LoRA parameters in the model."""
    for m in model.modules():
        if isinstance(m, CastedLinear) and getattr(m, "_lora_active", False):
            if hasattr(m, "lora_A"):
                yield m.lora_A
            if hasattr(m, "lora_B"):
                yield m.lora_B


def freeze_base_for_ttt(model):
    """Set requires_grad=False on every non-LoRA parameter."""
    lora_ids = {id(p) for p in lora_parameters(model)}
    n_frozen = 0
    for p in model.parameters():
        if id(p) in lora_ids:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
            n_frozen += 1
    return n_frozen


def unfreeze_base(model):
    for p in model.parameters():
        p.requires_grad_(True)


# --------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------
def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    n = int(header[2])
    arr = np.fromfile(file, dtype="<u2", count=n, offset=header_bytes)
    return torch.from_numpy(arr.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(pattern)
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start:start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    n = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros((n,), dtype=np.int16)
    has_leading_space = np.zeros((n,), dtype=np.bool_)
    is_boundary = np.ones((n,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(pattern)
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


# --------------------------------------------------------------------
# EVAL
# --------------------------------------------------------------------
def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_cnt = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_cnt = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    # NOTE: use no_grad() (not inference_mode) so any lazily-created
    # buffers (e.g. Rotary cos/sin caches) are NOT inference tensors.
    # Otherwise a subsequent score_first_ttt_eval()/backward() on the
    # same model crashes with: 'Inference tensors cannot be saved for
    # backward.'
    with torch.no_grad():
        for s in range(seq_start, seq_end, local_batch_seqs):
            e = min(s + local_batch_seqs, seq_end)
            raw_s, raw_e = s * args.train_seq_len, e * args.train_seq_len + 1
            local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y).detach()
            ntok = float(y.numel())
            val_loss_sum += loss.to(torch.float64) * ntok
            val_token_cnt += ntok
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_cnt += tb.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in (val_loss_sum, val_token_cnt, val_byte_cnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_cnt
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_cnt.item() / val_byte_cnt.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def score_first_ttt_eval(args, base_model, rank, world_size, device,
                         val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut,
                         lora_rank=None, lora_alpha=None, lora_targets=None,
                         ttt_chunk_tokens=None, ttt_steps_per_chunk=None, ttt_lr=None,
                         lora_block_first=None, lora_block_last=None):
    """Legal score-first TTT evaluation.

    Process each rank's slice of the val set in chunks of `ttt_chunk_tokens`. For
    each chunk:
      1. SCORE: forward pass under no_grad(), accumulate loss + byte counts. These
         numbers are LOCKED into val_bpb before any gradient sees the chunk.
      2. TTT:  take `ttt_steps_per_chunk` Adam steps on LoRA-only params using the
         same chunk as the training batch.
    All-reduces happen for both the loss accumulator and the LoRA gradients each
    TTT step (DDP-equivalent semantics, no DDP wrapper needed because we manually
    sync grads on the small LoRA buffers).

    Returns (val_loss, val_bpb) just like eval_val. base_model is unwrapped
    (i.e. NOT torch.compile / DDP). Re-attaches+resets LoRA adapters at the start.
    """
    lora_rank = lora_rank if lora_rank is not None else args.ttt_lora_rank
    lora_alpha = lora_alpha if lora_alpha is not None else args.ttt_lora_alpha
    lora_targets = lora_targets if lora_targets is not None else args.ttt_lora_targets
    ttt_chunk_tokens = ttt_chunk_tokens if ttt_chunk_tokens is not None else args.ttt_chunk_tokens
    ttt_steps_per_chunk = ttt_steps_per_chunk if ttt_steps_per_chunk is not None else args.ttt_steps_per_chunk
    ttt_lr = ttt_lr if ttt_lr is not None else args.ttt_lr
    lora_block_first = lora_block_first if lora_block_first is not None else args.ttt_lora_block_first
    lora_block_last = lora_block_last if lora_block_last is not None else args.ttt_lora_block_last

    # Always start from a clean LoRA state. If adapters were attached previously
    # (e.g. a prior TTT eval at training-time), strip them and re-attach.
    detach_lora_adapters(base_model)
    n_attached = attach_lora_adapters(base_model, rank=lora_rank, alpha=lora_alpha,
                                      targets=lora_targets,
                                      block_first=lora_block_first,
                                      block_last=lora_block_last)
    freeze_base_for_ttt(base_model)
    lora_params = list(lora_parameters(base_model))
    ttt_optim = torch.optim.Adam(lora_params, lr=ttt_lr, betas=(0.9, 0.95), eps=1e-8)

    seq_len = args.train_seq_len
    chunk_seqs = max(1, ttt_chunk_tokens // seq_len)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_cnt = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_cnt = torch.zeros((), device=device, dtype=torch.float64)

    distributed = dist.is_available() and dist.is_initialized()
    eval_batch_seqs = max(1, (args.val_batch_size // (world_size * 1)) // seq_len)

    if rank == 0:
        print(f"[TTT] adapters attached: {n_attached} on '{lora_targets}'  "
              f"chunk_seqs={chunk_seqs} ({chunk_seqs*seq_len} tok)  "
              f"steps/chunk={ttt_steps_per_chunk} lr={ttt_lr}", flush=True)

    base_model.eval()
    # Chunk-index counter for the in-chunk LR warmup. Because B=0 at
    # init, the first few chunks would otherwise eat a large step on
    # essentially-cold gradients. Linear ramp 0 -> ttt_lr over the first
    # `ttt_warmup_chunks` chunks. After that we run at full ttt_lr.
    chunk_idx = 0
    ttt_warmup_chunks = 100
    for s in range(seq_start, seq_end, chunk_seqs):
        e = min(s + chunk_seqs, seq_end)
        raw_s, raw_e = s * seq_len, e * seq_len + 1
        local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64, non_blocking=True)
        x_chunk = local[:-1].reshape(-1, seq_len)
        y_chunk = local[1:].reshape(-1, seq_len)

        # ---- 1. SCORE this chunk under no-grad. These numbers are LOCKED. ----
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                # Score in mini-batches if the chunk is large to control memory.
                bs = x_chunk.shape[0]
                step = max(1, eval_batch_seqs)
                chunk_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
                chunk_tok_cnt = 0
                for i in range(0, bs, step):
                    xi = x_chunk[i:i+step]
                    yi = y_chunk[i:i+step]
                    li = base_model(xi, yi).detach()
                    nt = yi.numel()
                    chunk_loss_sum += li.to(torch.float64) * nt
                    chunk_tok_cnt += nt
            val_loss_sum += chunk_loss_sum
            val_token_cnt += float(chunk_tok_cnt)
            prev_ids, tgt_ids = x_chunk.reshape(-1), y_chunk.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_cnt += tb.to(torch.float64).sum()

        # ---- 2. TTT: gradient steps on LoRA-only over the same chunk. ----
        # Linear LR warmup on chunk_idx. With B=0 init, ramping in over the
        # first 100 chunks keeps the LoRA from taking an oversized first kick.
        warm_mul = min(1.0, (chunk_idx + 1) / float(max(ttt_warmup_chunks, 1)))
        for g in ttt_optim.param_groups:
            g["lr"] = float(ttt_lr) * float(warm_mul)
        base_model.train()
        for ttt_step in range(ttt_steps_per_chunk):
            ttt_optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                ttt_loss = base_model(x_chunk, y_chunk)
            ttt_loss.backward()
            # Manually all-reduce LoRA grads across ranks (since we don't
            # have a DDP wrapper here; the LoRA tensors are tiny).
            if distributed:
                for p in lora_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)
            ttt_optim.step()
        base_model.eval()
        chunk_idx += 1

    if distributed:
        for t in (val_loss_sum, val_token_cnt, val_byte_cnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_cnt
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_cnt.item() / val_byte_cnt.item()
    val_bpb = bits_per_token * tokens_per_byte
    base_model.train()

    # Cleanup so subsequent calls start clean.
    detach_lora_adapters(base_model)
    unfreeze_base(base_model)

    return float(val_loss.item()), float(val_bpb)


# --------------------------------------------------------------------
# OPTIMIZERS
# --------------------------------------------------------------------
def build_optimizers(base_model, args):
    """Optimizer split matching reference train_gpt.py (LoRA params, if any, are
    excluded - they're TTT-only and never see training-time gradients).
      - Adam for tok_emb (+ lm_head if untied)
      - Muon for 2D matrix params in blocks
      - Adam for scalars / control params (and skip_weights)
    """
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2
        and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
        and not any(lp in name for lp in LORA_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS))
        and not any(lp in name for lp in LORA_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0 and base_model._has_skips:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizers = []

    optimizers.append(torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True))

    if base_model.lm_head is not None:
        optimizers.append(torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True))

    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizers.append(optimizer_muon)

    optimizers.append(torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True))

    return optimizers, optimizer_muon


# --------------------------------------------------------------------
# INT4 / INT6 / INT8 ZLIB EXPORT
# --------------------------------------------------------------------
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
INT6_CLIP_PERCENTILE = 99.95
INT6_CLIP_Q = INT6_CLIP_PERCENTILE / 100.0
INT6_PER_ROW_SCALE_DTYPE = torch.float16
INT4_CLIP_PERCENTILE = 99.985
INT4_CLIP_Q = INT4_CLIP_PERCENTILE / 100.0
INT4_PER_ROW_SCALE_DTYPE = torch.float16
INT4_PER_ROW_MIN_DTYPE   = torch.float16
INT4_AWQ_SCALE_DTYPE     = torch.float16
INT8_EMBEDDING_NAME_PATTERNS = ("tok_emb", "lm_head")
INT6_FALLBACK_ATTN_PROJ_PATTERN = ".attn.proj.weight"
QUANT_FORMAT_VERSION = "mixed_int4_int6_int8_v2"


def _tensor_nbytes(t):
    return int(t.numel()) * int(t.element_size())


def _keep_float_tensor(name, t, passthrough_orig_dtypes):
    if any(pat in name for pat in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
    return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()


def _row_clip_abs(t32, clip_mode, k_sd, clip_q):
    """Per-row clipping magnitude. clip_mode in {"percentile", "std"}."""
    if t32.numel() == 0:
        return torch.empty((t32.shape[0] if t32.ndim >= 1 else 0,), dtype=torch.float32)
    if t32.ndim != 2:
        if clip_mode == "std":
            return torch.tensor(float(k_sd) * float(t32.std().item()), dtype=torch.float32)
        return torch.tensor(float(torch.quantile(t32.abs().flatten(), float(clip_q)).item()), dtype=torch.float32)
    if clip_mode == "std":
        return (float(k_sd) * t32.std(dim=1)).clamp_min(1e-8)
    return torch.quantile(t32.abs(), float(clip_q), dim=1)


def _quantize_float_tensor_int8(t, *, clip_mode="percentile", k_sd=20.0):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = _row_clip_abs(t32, clip_mode, k_sd, INT8_CLIP_Q)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(_row_clip_abs(t32, clip_mode, k_sd, INT8_CLIP_Q))
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def _quantize_float_tensor_int6(t, *, clip_mode="percentile", k_sd=12.85):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = _row_clip_abs(t32, clip_mode, k_sd, INT6_CLIP_Q)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 31.0).clamp_min(1.0 / 31.0)
        q_signed = torch.clamp(torch.round(clipped / scale[:, None]), -31, 31).to(torch.int8).contiguous()
        return q_signed, scale.to(dtype=INT6_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(_row_clip_abs(t32, clip_mode, k_sd, INT6_CLIP_Q))
    scale = torch.tensor(clip_abs / 31.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q_signed = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -31, 31).to(torch.int8).contiguous()
    return q_signed, scale


def _pack_int6(q_int8):
    flat = q_int8.contiguous().view(-1).to(torch.int32)
    n = int(flat.numel())
    u = (flat + 32) & 0x3F
    pad = (-n) % 4
    if pad:
        u = torch.cat([u, torch.zeros(pad, dtype=torch.int32)])
    g = u.view(-1, 4)
    a, b, c, d = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    b0 = ((a << 2) | ((b >> 4) & 0x3)) & 0xFF
    b1 = (((b & 0xF) << 4) | ((c >> 2) & 0xF)) & 0xFF
    b2 = (((c & 0x3) << 6) | (d & 0x3F)) & 0xFF
    packed = torch.stack([b0, b1, b2], dim=1).view(-1).to(torch.uint8).contiguous()
    return packed, n


def _unpack_int6(packed, n):
    p = packed.contiguous().view(-1, 3).to(torch.int32)
    b0, b1, b2 = p[:, 0], p[:, 1], p[:, 2]
    a = (b0 >> 2) & 0x3F
    b = (((b0 & 0x3) << 4) | ((b1 >> 4) & 0xF)) & 0x3F
    c = (((b1 & 0xF) << 2) | ((b2 >> 6) & 0x3)) & 0x3F
    d = b2 & 0x3F
    out = torch.stack([a, b, c, d], dim=1).view(-1)[:n]
    return (out - 32).to(torch.int8).contiguous()


# ---------- int4 asymmetric (per-row scale + min, optional AWQ in-channel scale) ----------

def _row_minmax(t32, *, clip_mode, k_sd, clip_q):
    """Per-row (lo, hi) for asymmetric quant.

    clip_mode='std'     -> hi = mean + k_sd*std,  lo = mean - k_sd*std (per row)
    clip_mode='percentile' -> hi = quantile(row, clip_q), lo = quantile(row, 1-clip_q)
    """
    assert t32.ndim == 2
    if clip_mode == "std":
        mu  = t32.mean(dim=1)
        sd  = t32.std(dim=1).clamp_min(1e-8)
        k   = float(k_sd)
        return (mu - k * sd), (mu + k * sd)
    q_hi = float(clip_q)
    q_lo = 1.0 - q_hi
    hi = torch.quantile(t32, q_hi, dim=1)
    lo = torch.quantile(t32, q_lo, dim=1)
    # Guard against degenerate constant rows.
    same = (hi - lo).abs() < 1e-12
    if same.any():
        hi = torch.where(same, hi + 1e-6, hi)
        lo = torch.where(same, lo - 1e-6, lo)
    return lo, hi


def _quantize_float_tensor_int4_asym(t, *, clip_mode="std", k_sd=8.0,
                                     clip_q=INT4_CLIP_Q, awq_in_scale=None):
    """Asymmetric int4 [0..15] per-row.

    Returns (q_uint8 in [0..15], scale fp16 [out], min fp16 [out],
             awq_in_scale fp16 [in] or None).

    If awq_in_scale is provided, the weight is multiplied along axis=1 BEFORE
    quantization. Recovery: w_recovered[r,c] = (q[r,c]*scale[r] + min[r]) / awq[c].
    """
    if t.ndim != 2:
        raise ValueError(f"int4 asym path expects 2D weight, got shape {tuple(t.shape)}")
    t32 = t.float().contiguous()
    if awq_in_scale is not None:
        s_in = awq_in_scale.float().to(t32.device)
        if s_in.shape != (t32.shape[1],):
            raise ValueError(
                f"awq_in_scale shape {tuple(s_in.shape)} != in_features={t32.shape[1]}"
            )
        t32 = t32 * s_in[None, :]
    lo, hi = _row_minmax(t32, clip_mode=clip_mode, k_sd=k_sd, clip_q=clip_q)
    # Clip to (lo, hi) to keep outliers from eating the codebook.
    clipped = torch.maximum(torch.minimum(t32, hi[:, None]), lo[:, None])
    scale = ((hi - lo) / 15.0).clamp_min(1e-8)
    q = torch.clamp(torch.round((clipped - lo[:, None]) / scale[:, None]), 0, 15)
    q = q.to(torch.uint8).contiguous()
    return (
        q,
        scale.to(dtype=INT4_PER_ROW_SCALE_DTYPE).contiguous(),
        lo.to(dtype=INT4_PER_ROW_MIN_DTYPE).contiguous(),
        (None if awq_in_scale is None
         else awq_in_scale.to(dtype=INT4_AWQ_SCALE_DTYPE).contiguous()),
    )


def _pack_int4(q_uint8):
    """Pack two nibbles per byte (low nibble first). Returns (packed_uint8, n_elements)."""
    flat = q_uint8.contiguous().view(-1).to(torch.int32) & 0xF
    n = int(flat.numel())
    if n & 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int32)])
    g = flat.view(-1, 2)
    packed = ((g[:, 0] & 0xF) | ((g[:, 1] & 0xF) << 4)).to(torch.uint8).contiguous()
    return packed, n


def _unpack_int4(packed, n):
    p = packed.contiguous().view(-1).to(torch.int32)
    lo = (p & 0xF)
    hi = ((p >> 4) & 0xF)
    out = torch.stack([lo, hi], dim=1).view(-1)[:n]
    return out.to(torch.uint8).contiguous()


def _make_int6_fallback_predicate(num_layers, *,
                                  first_last_blocks=True, attn_proj=True):
    """Build the predicate that decides which 2D float matrices stay at int6
    instead of going to int4. Embeddings are routed to int8 separately."""
    first_pref = "blocks.0."
    last_pref  = f"blocks.{int(num_layers) - 1}."
    def is_int6(name):
        if first_last_blocks and (name.startswith(first_pref) or name.startswith(last_pref)):
            return True
        if attn_proj and (INT6_FALLBACK_ATTN_PROJ_PATTERN in name):
            return True
        return False
    return is_int6


def _select_quant_scheme(name, *, int6_fallback_pred=None):
    """Returns ("int8" | "int6" | "int4_asym", bits)."""
    if any(pat in name for pat in INT8_EMBEDDING_NAME_PATTERNS):
        return ("int8", 8)
    if int6_fallback_pred is not None and int6_fallback_pred(name):
        return ("int6", 6)
    return ("int4_asym", 4)


def _select_quant_bits(name, *, int6_fallback_pred=None):
    return _select_quant_scheme(name, int6_fallback_pred=int6_fallback_pred)[1]


def _strip_lora_from_state_dict(state_dict):
    """LoRA adapters are runtime-only - if any are present in the state_dict
    (e.g. a TTT eval ran before export), strip them."""
    return {k: v for k, v in state_dict.items()
            if not any(p in k for p in LORA_NAME_PATTERNS)}


def collect_awq_input_scales(model, calib_chunks, args, device, *,
                             alpha=0.5, eps=1e-5, int6_fallback_pred=None,
                             verbose=True):
    """Run a small calibration forward pass and return per-input-channel
    scales for every matrix the export will route to int4.

    Args:
        model: a freshly-instantiated GPT with the trained weights loaded
            (NOT the int-quantized one). Will be set to eval and used in
            no-grad mode.
        calib_chunks: iterable of int64 token tensors of shape [seq_len].
            Typically a few non-overlapping chunks of validation tokens.
        args: Hyperparameters (used for tokenizer-specific helpers in the
            forward path; we only need the model itself).
        device: torch.device for the calibration run.
        alpha: AWQ exponent. 0.5 is the AWQ paper default (geometric mean
            between activation-rms and an implicit weight-rms of 1.0 after
            the per-row normalization). 0.0 disables (returns None).

    Returns:
        dict[name -> tensor[in_features] fp32 on cpu], or None when alpha<=0.
        The returned scales are normalized per tensor so their geometric mean
        is 1.0 (keeps the per-row clip thresholds meaningful).
    """
    if alpha is None or float(alpha) <= 0.0:
        if verbose:
            print("[awq] disabled (alpha <= 0)")
        return None

    model.eval()
    sums = {}     # name -> running sum of x^2 [in]
    counts = {}   # name -> running count of (batch*seq) elements

    def _full_name(module):
        # Resolve "blocks.<i>.<sub>.<sub>...<linear>" using a cached lookup.
        return module._awq_full_name

    def _hook(module, inputs, output):
        x = inputs[0]
        if x.dim() == 3:
            xf = x.detach().float().reshape(-1, x.shape[-1])
        else:
            xf = x.detach().float().reshape(-1, x.shape[-1])
        s = (xf * xf).sum(dim=0)
        n = xf.shape[0]
        nm = _full_name(module)
        if nm in sums:
            sums[nm].add_(s)
            counts[nm] += n
        else:
            sums[nm] = s
            counts[nm] = n

    # Annotate every CastedLinear inside blocks.* with its full state-dict
    # parameter name (i.e. "blocks.<i>.<...>.weight") and only register a
    # hook on those that the export will route to int4.
    handles = []
    for full_mod_name, mod in model.named_modules():
        if not isinstance(mod, CastedLinear):
            continue
        if not full_mod_name.startswith("blocks."):
            continue
        weight_name = f"{full_mod_name}.weight"
        if int6_fallback_pred is not None and int6_fallback_pred(weight_name):
            continue
        mod._awq_full_name = weight_name
        handles.append(mod.register_forward_hook(_hook))

    if not handles:
        if verbose:
            print("[awq] no int4 linears found, skipping calibration")
        return {}

    if verbose:
        print(f"[awq] hooked {len(handles)} linears, running {len(list(calib_chunks))} chunks...")

    # Re-materialize chunks since we just consumed the iterator above for the count.
    calib_chunks = list(calib_chunks)

    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    with torch.no_grad():
        for ci, tokens in enumerate(calib_chunks):
            tokens = tokens.to(device=device, dtype=torch.int64).unsqueeze(0)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
                # AWQ calibration only needs the forward activations; pass
                # target_ids=None so the cross-entropy path is skipped.
                _ = model(tokens, target_ids=None)
            if verbose:
                print(f"[awq]   chunk {ci+1}/{len(calib_chunks)} done")

    for h in handles:
        h.remove()
    for mod_name, mod in model.named_modules():
        if hasattr(mod, "_awq_full_name"):
            del mod._awq_full_name

    # Convert sums -> per-channel L2-rms -> s_in = rms^alpha, then normalize so
    # geometric mean == 1 per tensor.
    scales = {}
    a = float(alpha)
    for name, s in sums.items():
        n = max(int(counts[name]), 1)
        rms = (s.cpu() / n).clamp_min(eps).sqrt()
        s_in = rms.pow(a)
        log_s = s_in.log()
        s_in = (s_in / log_s.mean().exp()).contiguous()
        scales[name] = s_in.to(dtype=torch.float32)
    if verbose:
        ks = sorted(scales.keys())
        if ks:
            print(f"[awq] computed {len(ks)} scales, e.g. {ks[0]} shape={tuple(scales[ks[0]].shape)}")
    return scales


def quantize_state_dict_int8(state_dict, *, clip_mode="percentile",
                             embed_k_sd=20.0, matrix_k_sd=12.85,
                             int4_k_sd=8.0, int4_clip_q=INT4_CLIP_Q,
                             int6_fallback_pred=None,
                             awq_scales=None):
    """Mixed-precision export quantizer.

    Per-tensor dispatch:
      tok_emb / lm_head            -> int8 per-row sym
      first/last block, attn.proj  -> int6 per-row sym  (int6_fallback_pred decides)
      every other 2D matrix        -> int4 ASYMMETRIC per-row, optionally
                                       pre-multiplied by AWQ in-channel scale
      <=65k float tensors          -> fp16/fp32 passthrough
      non-float tensors            -> raw passthrough

    awq_scales: optional dict[name -> tensor[in_features]]; only consumed for
    tensors that route to int4. Missing entries fall back to symmetric int4
    without AWQ.
    """
    state_dict = _strip_lora_from_state_dict(state_dict)
    quantized, scales, mins, awq, dtypes = {}, {}, {}, {}, {}
    passthrough, passthrough_orig_dtypes = {}, {}
    qmeta = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes",
         "int4_tensors", "int6_tensors", "int8_tensors", "awq_tensors"), 0,
    )
    awq_scales = awq_scales or {}

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += _tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += _tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = _keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += _tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        if t.ndim == 2:
            scheme, bits = _select_quant_scheme(name, int6_fallback_pred=int6_fallback_pred)
        else:
            scheme, bits = ("int8", 8)

        if scheme == "int4_asym":
            awq_in = awq_scales.get(name)
            q, s, lo, awq_stored = _quantize_float_tensor_int4_asym(
                t, clip_mode=clip_mode, k_sd=int4_k_sd, clip_q=int4_clip_q,
                awq_in_scale=awq_in,
            )
            packed, n_elem = _pack_int4(q)
            quantized[name] = packed
            scales[name]    = s
            mins[name]      = lo
            dtypes[name]    = str(t.dtype).removeprefix("torch.")
            meta = {"scheme": "per_row_asym", "axis": 0, "bits": 4, "format": "int4_asym",
                    "n_elements": n_elem, "shape": list(t.shape)}
            if awq_stored is not None:
                awq[name] = awq_stored
                meta["awq"] = True
                stats["awq_tensors"] += 1
            qmeta[name] = meta
            stats["int4_tensors"] += 1
            stats["int8_payload_bytes"] += (_tensor_nbytes(packed) + _tensor_nbytes(s)
                                            + _tensor_nbytes(lo)
                                            + (_tensor_nbytes(awq_stored) if awq_stored is not None else 0))
        elif scheme == "int6":
            q_signed, s = _quantize_float_tensor_int6(t, clip_mode=clip_mode, k_sd=matrix_k_sd)
            packed, n_elem = _pack_int6(q_signed)
            quantized[name] = packed
            scales[name] = s
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            qmeta[name] = {"scheme": "per_row", "axis": 0, "bits": 6, "format": "int6",
                           "n_elements": n_elem, "shape": list(t.shape)}
            stats["int6_tensors"] += 1
            stats["int8_payload_bytes"] += _tensor_nbytes(packed) + _tensor_nbytes(s)
        else:  # int8
            k_sd_pick = embed_k_sd if any(p in name for p in INT8_EMBEDDING_NAME_PATTERNS) else matrix_k_sd
            q, s = _quantize_float_tensor_int8(t, clip_mode=clip_mode, k_sd=k_sd_pick)
            if s.ndim > 0:
                qmeta[name] = {"scheme": "per_row", "axis": 0, "bits": 8, "format": "int8"}
            else:
                qmeta[name] = {"scheme": "per_tensor", "bits": 8, "format": "int8"}
            quantized[name] = q
            scales[name] = s
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["int8_tensors"] += 1
            stats["int8_payload_bytes"] += _tensor_nbytes(q) + _tensor_nbytes(s)
    obj = {
        "__quant_format__": QUANT_FORMAT_VERSION,
        "quantized": quantized,
        "scales": scales,
        "mins": mins,
        "awq": awq,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj):
    fmt = obj.get("__quant_format__", "")
    if fmt != QUANT_FORMAT_VERSION:
        raise ValueError(f"unknown quant format: {fmt!r}")
    out = {}
    qmeta = obj["qmeta"]
    mins  = obj.get("mins", {})
    awqs  = obj.get("awq",  {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta[name]
        bits = int(meta["bits"])
        scheme = meta.get("scheme", "per_row")
        if bits == 4:
            flat = _unpack_int4(q, int(meta["n_elements"]))
            q_int = flat.view(*meta["shape"])
            s_f = s.to(dtype=torch.float32).view(q_int.shape[0], 1)
            lo  = mins[name].to(dtype=torch.float32).view(q_int.shape[0], 1)
            w = q_int.to(dtype=torch.float32) * s_f + lo
            if name in awqs:
                awq_in = awqs[name].to(dtype=torch.float32).view(1, q_int.shape[1])
                w = w / awq_in.clamp_min(1e-8)
            out[name] = w.to(dtype=dtype).contiguous()
        elif bits == 6:
            flat = _unpack_int6(q, int(meta["n_elements"]))
            q_int = flat.view(*meta["shape"])
            s_f = s.to(dtype=torch.float32)
            out[name] = (q_int.float() * s_f.view(q_int.shape[0], *([1] * (q_int.ndim - 1)))).to(dtype=dtype).contiguous()
        else:  # int8
            q_int = q
            if scheme == "per_row":
                s_f = s.to(dtype=torch.float32)
                out[name] = (q_int.float() * s_f.view(q_int.shape[0], *([1] * (q_int.ndim - 1)))).to(dtype=dtype).contiguous()
            else:
                scale = float(s.item())
                out[name] = (q_int.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def export_state_dict_int8_zlib(state_dict, verbose=True, *,
                                clip_mode="percentile", embed_k_sd=20.0,
                                matrix_k_sd=12.85, int4_k_sd=8.0,
                                int4_clip_q=INT4_CLIP_Q,
                                int6_fallback_pred=None,
                                awq_scales=None):
    obj, stats = quantize_state_dict_int8(
        state_dict,
        clip_mode=clip_mode, embed_k_sd=embed_k_sd,
        matrix_k_sd=matrix_k_sd, int4_k_sd=int4_k_sd,
        int4_clip_q=int4_clip_q,
        int6_fallback_pred=int6_fallback_pred,
        awq_scales=awq_scales,
    )
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw = buf.getvalue()
    if _HAS_ZSTD:
        # zstd-22 = highest "regular" zstd level. It is slow to compress (one-shot
        # at end of training, fine) but cheap to decompress. Empirically gives a
        # few-percent shrink vs zlib(9) on packed int4/int6 tables.
        compressed = _zstd.ZstdCompressor(level=22).compress(raw)
    else:
        compressed = zlib.compress(raw, level=9)
    if verbose:
        comp_label = "zstd" if _HAS_ZSTD else "zlib"
        print(f"params: {stats['param_count']:,}  "
              f"int8: {stats['int8_tensors']}  int6: {stats['int6_tensors']}  "
              f"int4: {stats['int4_tensors']}  awq: {stats['awq_tensors']}  "
              f"raw_state: {len(raw)/1e6:.2f} MB  "
              f"{comp_label}: {len(compressed)/1e6:.2f} MB  "
              f"(clip_mode={clip_mode})")
    return compressed, stats


def load_state_dict_int8_zlib(compressed_bytes):
    # Try zstd first when available, then fall back to zlib so artifacts
    # produced before the zstd switch (or in environments without zstandard)
    # still load. zstd raises ZstdError on a non-zstd payload; zlib on a
    # non-zlib payload raises zlib.error. We catch both and re-raise with a
    # helpful message if neither worked.
    if _HAS_ZSTD:
        try:
            decompressed = _zstd.ZstdDecompressor().decompress(compressed_bytes)
        except _zstd.ZstdError:
            decompressed = zlib.decompress(compressed_bytes)
    else:
        decompressed = zlib.decompress(compressed_bytes)
    obj = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
    return dequantize_state_dict_int8(obj)


# --------------------------------------------------------------------
# DDP SETUP + WORKERS
# --------------------------------------------------------------------
def _setup_distributed():
    """Initialize torch.distributed for either torchrun or single-GPU mode.

    Returns (rank, world_size, local_rank, device).
    Idempotent; safe to call once per process.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (parameter-golf submission expects 8xH100).")

    try:
        import sys
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun: env vars provide rank / world / local_rank.
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    else:
        # single-process fallback (handy for local smoke testing).
        rank, world_size, local_rank = 0, 1, 0
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )
    # H100 cuDNN SDP is typically faster than FA2 for GQA head configs.
    enable_cudnn_sdp(True)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    # Inductor tuning: coordinate-descent finds better tile sizes on H100.
    import torch._inductor.config as _ind_cfg
    _ind_cfg.coordinate_descent_tuning = True

    return rank, world_size, local_rank, device


def _build_model(args, device):
    base = GPT(args).to(device).bfloat16()
    for m in base.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base)
    return base


def _apply_muon_momentum_warmup(optimizer_muon, args, step):
    ws = args.muon_momentum_warmup_steps
    if ws > 0 and step < ws:
        frac = step / ws
        mom = args.muon_momentum_warmup_start + frac * (args.muon_momentum - args.muon_momentum_warmup_start)
    else:
        mom = args.muon_momentum
    for g in optimizer_muon.param_groups:
        g["momentum"] = mom


def train_main(rank, world_size, device, args, sd_save_path):
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    grad_scale = 1.0 / grad_accum_steps

    random.seed(args.seed + rank); np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank); torch.cuda.manual_seed_all(args.seed + rank)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = _build_model(args, device)
    model = DDP(torch.compile(base_model, dynamic=False, fullgraph=True, mode="max-autotune"),
                device_ids=[device.index], broadcast_buffers=False)

    if rank == 0:
        n_params = sum(p.numel() for p in base_model.parameters())
        eff_passes = (base_model.num_pre_layers
                      + base_model.num_recur_layers * base_model.recurrence_count
                      + base_model.num_post_layers)
        print(f"params: {n_params:,}  effective_passes/token: {eff_passes}", flush=True)

    optimizers, optimizer_muon = build_optimizers(base_model, args)
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        if rank == 0:
            print(f"warmup ({args.warmup_steps} steps)...", flush=True)
        init_sd = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for _ in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wloss = model(x, y)
                (wloss * grad_scale).backward()
            _apply_muon_momentum_warmup(optimizer_muon, args, ws)
            for o in optimizers:
                o.step()
            zero_grad_all()
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opt):
            o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    if rank == 0:
        print(f"training: {args.iterations} iters, {args.train_batch_tokens:,} tok/step, {world_size} GPUs",
              flush=True)

    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                         val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut)
            if rank == 0:
                print(f"step:{step}/{args.iterations} val_bpb:{val_bpb:.4f} val_loss:{val_loss:.4f} "
                      f"time:{training_time_ms:.0f}ms avg:{training_time_ms/max(step,1):.2f}ms",
                      flush=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        zero_grad_all()
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                train_loss = model(x, y)
            (train_loss * grad_scale).backward()

        if args.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        torch.cuda.synchronize()
        elapsed_ms_now = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        mul = lr_mul(step, elapsed_ms_now)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mul
        _apply_muon_momentum_warmup(optimizer_muon, args, step)

        for opt in optimizers:
            opt.step()

        step += 1
        if rank == 0 and args.train_log_every > 0 and step % args.train_log_every == 0:
            print(f"step:{step}/{args.iterations} loss:{train_loss.item():.4f} "
                  f"lr_mul:{mul:.4f} time:{elapsed_ms_now:.0f}ms avg:{elapsed_ms_now/step:.2f}ms",
                  flush=True)

        if max_wallclock_ms is not None and stop_after_step is None:
            if elapsed_ms_now > max_wallclock_ms:
                stop_after_step = step

    # ---- final TTT eval (legal score-first), if enabled ----
    if args.ttt_enabled:
        torch.cuda.synchronize(); dist.barrier()
        if rank == 0:
            print("[final] running legal score-first TTT eval (pre-quant)...", flush=True)
        t_ttt = time.perf_counter()
        # Run TTT on the unwrapped base_model (no DDP/torch.compile around it).
        # Manual all-reduce of LoRA grads is done inside score_first_ttt_eval.
        ttt_loss, ttt_bpb = score_first_ttt_eval(
            args, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut,
        )
        torch.cuda.synchronize(); dist.barrier()
        if rank == 0:
            print(f"[final-prequant] TTT val_bpb:{ttt_bpb:.4f} val_loss:{ttt_loss:.4f} "
                  f"ttt_time:{1000*(time.perf_counter()-t_ttt):.0f}ms", flush=True)

    if rank == 0:
        print(f"done: {training_time_ms/1000:.1f}s, {step} steps", flush=True)
        # Strip any LoRA from the saved sd defensively (TTT detaches them, but
        # belt-and-suspenders if a user calls TTT twice).
        sd_to_save = _strip_lora_from_state_dict(base_model.state_dict())
        torch.save(sd_to_save, sd_save_path)

    # Wait for rank 0's save to land before any rank tries to read it.
    dist.barrier()
    return base_model, val_tokens, (base_bytes_lut, has_leading_space_lut, is_boundary_lut)


def export_and_verify(rank, world_size, device, args,
                      base_model, val_tokens, luts, sd_save_path):
    """Run the full mixed-precision export -> round-trip load -> sliding eval ->
    legal score-first TTT eval pipeline. Returns (val_bpb_no_ttt, val_bpb_ttt,
    compressed_bytes_len). Called after train_main on every rank."""
    base_bytes_lut, has_leading_space_lut, is_boundary_lut = luts
    if rank == 0:
        print("[export] loading trained sd from disk for export...", flush=True)

    # Re-load on rank 0; broadcast-equivalent semantics achieved by all ranks
    # loading from the same on-disk file (rank 0 wrote it inside train_main).
    trained_sd = torch.load(sd_save_path, map_location="cpu", weights_only=True)

    def fresh_model():
        m = GPT(args).to(device).bfloat16()
        for sub in m.modules():
            if isinstance(sub, CastedLinear):
                sub.float()
        restore_low_dim_params_to_fp32(m)
        return m

    int6_pred = _make_int6_fallback_predicate(
        args.num_layers,
        first_last_blocks=args.quant_int6_first_last,
        attn_proj=args.quant_int6_attn_proj,
    )

    awq_scales = None
    if args.quant_awq_enabled and args.quant_awq_alpha > 0.0:
        if rank == 0:
            print(f"[awq] running calibration "
                  f"({args.quant_awq_calib_chunks} chunks of {args.train_seq_len} tokens, "
                  f"alpha={args.quant_awq_alpha})...", flush=True)
        calib_model = fresh_model()
        calib_model.load_state_dict(trained_sd, strict=True)
        calib_model.eval()
        n_chunks = int(args.quant_awq_calib_chunks)
        chunk_len = int(args.train_seq_len)
        flat_val = val_tokens.reshape(-1)
        if flat_val.numel() < n_chunks * chunk_len:
            n_chunks = max(1, flat_val.numel() // chunk_len)
            if rank == 0:
                print(f"[awq] WARNING: reduced calib chunks to {n_chunks} "
                      f"due to limited val tokens", flush=True)
        chunks = [flat_val[i*chunk_len:(i+1)*chunk_len].to(torch.int64) for i in range(n_chunks)]
        awq_scales = collect_awq_input_scales(
            calib_model, chunks, args, device,
            alpha=args.quant_awq_alpha,
            eps=args.quant_awq_eps,
            int6_fallback_pred=int6_pred,
            verbose=(rank == 0),
        )
        del calib_model
        torch.cuda.empty_cache()

    compressed_bytes, stats = export_state_dict_int8_zlib(
        trained_sd, verbose=(rank == 0),
        clip_mode=args.quant_clip_mode,
        embed_k_sd=args.quant_embed_k_sd,
        matrix_k_sd=args.quant_matrix_k_sd,
        int4_k_sd=args.quant_int4_k_sd,
        int4_clip_q=args.quant_int4_clip_q,
        int6_fallback_pred=int6_pred,
        awq_scales=awq_scales,
    )

    artifact_path = os.path.join(os.path.dirname(sd_save_path), "artifact_int4_zlib.bin")
    if rank == 0:
        with open(artifact_path, "wb") as f:
            f.write(compressed_bytes)
        # Code byte size = this script. Spec says "code + compressed model".
        try:
            code_bytes = os.path.getsize(__file__)
        except Exception:
            code_bytes = 0
        total_bytes = code_bytes + len(compressed_bytes)
        print(f"code: {code_bytes:,} B  model: {len(compressed_bytes):,} B  "
              f"total: {total_bytes:,} B  (limit 16,000,000)  "
              f"fits={total_bytes<=16_000_000}", flush=True)

    # Round-trip the compressed bytes back into a fresh GPT.
    reloaded_sd = load_state_dict_int8_zlib(compressed_bytes)
    q_model = fresh_model()
    q_model.load_state_dict(reloaded_sd, strict=True)
    q_compiled = torch.compile(q_model, dynamic=False, fullgraph=True)

    grad_accum_steps = max(1, int(args.grad_accum_steps))

    torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val(
        args, q_compiled, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut,
    )
    torch.cuda.synchronize(); dist.barrier()
    if rank == 0:
        print(f"[no-TTT] val_loss: {val_loss:.6f}  val_bpb: {val_bpb:.6f}  "
              f"eval: {1000*(time.perf_counter()-t0):.0f}ms", flush=True)

    val_bpb_ttt = None
    if args.ttt_enabled:
        if rank == 0:
            print("[TTT] running legal score-first TTT eval on int4-roundtrip model...",
                  flush=True)
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        ttt_loss, ttt_bpb = score_first_ttt_eval(
            args, q_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut,
        )
        torch.cuda.synchronize(); dist.barrier()
        val_bpb_ttt = ttt_bpb
        if rank == 0:
            print(f"[TTT]     val_loss: {ttt_loss:.6f}  val_bpb: {ttt_bpb:.6f}  "
                  f"eval: {1000*(time.perf_counter()-t0):.0f}ms", flush=True)
            print(f"delta(no-TTT - TTT) val_bpb: {val_bpb - ttt_bpb:+.6f}", flush=True)

    return val_bpb, val_bpb_ttt, len(compressed_bytes)


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    rank, world_size, local_rank, device = _setup_distributed()
    args = Hyperparameters()

    if rank == 0:
        rec_n = args.recurrent_layer_end - args.recurrent_layer_start
        eff_passes = (args.recurrent_layer_start
                      + rec_n * args.recurrence_count
                      + (args.num_layers - args.recurrent_layer_end))
        print(f"=== {args.run_id} | seed={args.seed} | world={world_size} ===", flush=True)
        print(f"vocab: {args.vocab_size}  parallel_residuals: {args.parallel_residuals}", flush=True)
        print(f"model: {args.num_layers}L x {args.model_dim}d  "
              f"heads:{args.num_heads}/{args.num_kv_heads}  mlp_mult:{args.mlp_mult}", flush=True)
        print(f"recurrence: layers [{args.recurrent_layer_start},"
              f"{args.recurrent_layer_end}) x {args.recurrence_count} -> "
              f"{eff_passes} effective passes/token", flush=True)
        print(f"ttt: enabled={args.ttt_enabled} rank={args.ttt_lora_rank} "
              f"alpha={args.ttt_lora_alpha} steps/chunk={args.ttt_steps_per_chunk} "
              f"lr={args.ttt_lr}", flush=True)
        print(f"quant: scheme={args.quant_scheme} clip_mode={args.quant_clip_mode}", flush=True)
        print(f"  int8 embed k={args.quant_embed_k_sd}   "
              f"int6 matrix k={args.quant_matrix_k_sd}   "
              f"int4 matrix k={args.quant_int4_k_sd}", flush=True)
        print(f"  int6 fallback: first_last_blocks={args.quant_int6_first_last}  "
              f"attn_proj={args.quant_int6_attn_proj}", flush=True)
        print(f"  awq: enabled={args.quant_awq_enabled} alpha={args.quant_awq_alpha} "
              f"calib_chunks={args.quant_awq_calib_chunks}", flush=True)
        print(f"train: iters:{args.iterations}  seq_len:{args.train_seq_len}  "
              f"batch_tokens:{args.train_batch_tokens:,}", flush=True)

    sd_save_path = _env_str(
        "TRAINED_SD_PATH",
        os.path.join(tempfile.gettempdir(), f"trained_{args.run_id}_seed{args.seed}.pt"),
    )

    base_model, val_tokens, luts = train_main(
        rank, world_size, device, args, sd_save_path,
    )

    # Free DDP+torch.compile memory before the export pipeline materializes a
    # fresh model for AWQ calibration and a second one for the round-trip eval.
    torch.cuda.empty_cache()

    export_and_verify(
        rank, world_size, device, args,
        base_model, val_tokens, luts, sd_save_path,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
