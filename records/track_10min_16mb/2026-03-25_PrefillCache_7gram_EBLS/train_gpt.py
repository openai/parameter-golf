"""train_gpt_ngram_v1.py — 2026-03-25
Based on train_gpt_submission.py (1.1198 BPB, 3-seed validated).
NEW: Multi-order n-gram backoff cache with entropy-adaptive alpha (PR #715/#727 style).
Expected: ~1.03-1.07 BPB depending on config. Legality: score-first, backward-looking only."""
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
import lzma
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn_interface import flash_attn_func as flash_attn_3_func
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    clip_range = int(os.environ.get("CLIP_RANGE", 31))
    compressor = os.environ.get("COMPRESSOR", "lzma")
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    vrl = bool(int(os.environ.get("VRL", "1")))
    gptq_enabled = bool(int(os.environ.get("GPTQ_ENABLED", "1")))
    gptq_calib_batches = int(os.environ.get("GPTQ_CALIB_BATCHES", 256))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))
    gptq_damp_factor = float(os.environ.get("GPTQ_DAMP_FACTOR", "0.01"))
    gptq_calib_source = os.environ.get("GPTQ_CALIB_SOURCE", "val")
    swa_ema_blend = float(os.environ.get("SWA_EMA_BLEND", "0.5"))
    # N-gram cache (eval-time only, score-first compliant)
    ngram_cache = bool(int(os.environ.get("NGRAM_CACHE", "1")))
    ngram_order = int(os.environ.get("NGRAM_ORDER", "7"))
    ngram_min_order = int(os.environ.get("NGRAM_MIN_ORDER", "2"))
    ngram_alpha = float(os.environ.get("NGRAM_ALPHA", "0.40"))
    ngram_min_count = int(os.environ.get("NGRAM_MIN_COUNT", "2"))
    ngram_buckets = int(os.environ.get("NGRAM_BUCKETS", "4194304"))
    ngram_entropy = bool(int(os.environ.get("NGRAM_ENTROPY", "1")))
    ngram_ent_base = float(os.environ.get("NGRAM_ENT_BASE", "0.05"))
    ngram_ent_range = float(os.environ.get("NGRAM_ENT_RANGE", "0.55"))
    ngram_ent_scale = float(os.environ.get("NGRAM_ENT_SCALE", "2.0"))
    ngram_ent_thresh = float(os.environ.get("NGRAM_ENT_THRESH", "4.0"))
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
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
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
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss
def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )
def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]
def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,ve_layer_scales,ve_shared.scale,vrl_lambda",
    ).split(",")
    if pattern
)
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        shard_order = os.environ.get("SHARD_ORDER", "")
        if shard_order:
            order = [int(x) for x in shard_order.split(",")]
            reordered = [self.files[i] for i in order if i < len(self.files)]
            remaining = [f for i, f in enumerate(self.files) if i not in set(order)]
            self.files = reordered + remaining
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)
class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    _clip_range: int = 31
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('_soft_round_alpha', torch.tensor(1.0), persistent=False)
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            cr = CastedLinear._clip_range
            w32 = self.weight.float()
            row_max = w32.abs().amax(dim=1).detach()
            scale = (row_max / float(cr)).clamp_min(1.0 / float(cr))
            x_norm = w32 / scale[:, None]
            alpha = self._soft_round_alpha
            fl = x_norm.floor()
            r = x_norm - fl - 0.5
            tanh_half = torch.tanh(alpha * 0.5)
            q_soft = fl + 0.5 * torch.tanh(alpha * r) / (tanh_half + 1e-10) + 0.5
            q_soft = torch.clamp(q_soft, -cr, cr)
            w_q = (q_soft * scale[:, None]).to(x.dtype)
            w = w_q
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
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
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False
        self.use_vrl = False
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, v_embed: Tensor | None = None, q_delta: Tensor | None = None, v_delta: Tensor | None = None, v0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        if q_delta is not None:
            q = q + q_delta
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        if v_delta is not None:
            v = v + v_delta
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v
        if self.use_vrl and v0 is not None:
            lam = self.vrl_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y), raw_v
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())
class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, layer_idx: int = 0, ln_scale: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None, q_delta_fn=None, v_delta_fn=None, v0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = self.attn_norm(x_in) * self.ln_scale_factor
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        attn_out, raw_v = self.attn(n, v_embed=v_embed, q_delta=qd, v_delta=vd, v0=v0)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out, raw_v
class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 bigram_vocab_size: int = 0, bigram_dim: int = 128, xsa_last_n: int = 0,
                 rope_dims: int = 0, ln_scale: bool = False,
                 ve_enabled: bool = False, ve_dim: int = 128, ve_layers: str = "9,10",
                 use_vrl: bool = False):
        super().__init__()
        self.use_vrl = use_vrl
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, ln_scale=ln_scale)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        if use_vrl:
            for i, block in enumerate(self.blocks):
                if i > 0:
                    block.attn.use_vrl = True
                    block.attn.vrl_lambda = nn.Parameter(torch.tensor([0.01, 0.99], dtype=torch.float32))
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        v0 = None
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x, raw_v = self.blocks[i](x, x0, v_embed=ve, q_delta_fn=qd, v_delta_fn=vd, v0=v0)
            if i == 0 and self.use_vrl:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            qd = lora.q_loras[bi] if lora else None
            vd = lora.v_loras[bi] if lora else None
            x, _ = self.blocks[bi](x, x0, v_embed=ve, q_delta_fn=qd, v_delta_fn=vd, v0=v0)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits_proj = logits_proj + (lora.lm_head_lora(x).reshape(-1, logits_proj.size(-1)) if lora else 0)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if lora:
            bsz, sl, V = logits_proj.shape[0] // target_ids.shape[1], target_ids.shape[1], logits_proj.shape[-1]
            return F.cross_entropy(logits.float(), targets, reduction="none").reshape(bsz, sl)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
    def forward_logits(self, input_ids: Tensor, return_hidden: bool = False):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        v0 = None
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](x, x0, v_embed=ve, v0=v0)
            if i == 0 and self.use_vrl:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x, _ = self.blocks[bi](x, x0, v_embed=ve, v0=v0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if return_hidden:
            return logits, x
        return logits
def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window eval with optional multi-order n-gram backoff cache.
    Score-first: each token scored by model BEFORE its data enters the cache."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    # N-gram cache setup (multi-order backoff, PR #715/#727 style)
    use_ngram = args.ngram_cache
    if use_ngram:
        val_np = val_tokens.cpu().numpy()
        _n_orders = args.ngram_order - args.ngram_min_order + 1
        ctx_tables = [np.zeros((args.ngram_buckets,), dtype=np.uint32) for _ in range(_n_orders)]
        full_tables = [np.zeros((args.ngram_buckets,), dtype=np.uint32) for _ in range(_n_orders)]
        ng_mask = np.uint64(args.ngram_buckets - 1)
        ng_primes = np.array(
            [np.uint64(36313), np.uint64(27191), np.uint64(51647), np.uint64(81929),
             np.uint64(131071), np.uint64(175447), np.uint64(209591)], dtype=np.uint64)
        if rank == 0:
            print(f"ngram_cache:enabled orders={args.ngram_min_order}-{args.ngram_order} "
                  f"entropy={args.ngram_entropy} alpha={args.ngram_alpha} "
                  f"min_count={args.ngram_min_count} buckets={args.ngram_buckets}", flush=True)
        # Pre-fill cache with ALL tokens preceding this rank's windows (no NCCL needed)
        if rank > 0 and len(my_windows) > 0:
            ws0 = my_windows[0]
            pre_end = ws0 + max(seq_len - stride, 0)  # last target pos scored by preceding ranks
            t_pf = time.perf_counter()
            for oi in range(_n_orders):
                cw = args.ngram_min_order + oi - 1
                positions = np.arange(max(cw, 1), pre_end + 1, dtype=np.int64)
                if len(positions) == 0:
                    continue
                ctx_hash = np.zeros(len(positions), dtype=np.uint64)
                for k in range(cw):
                    ctx_hash ^= val_np[positions - (cw - k)].astype(np.uint64) * ng_primes[k % len(ng_primes)]
                ck = (ctx_hash & ng_mask).astype(np.int64)
                tgt = val_np[positions].astype(np.uint64)
                fk = ((ctx_hash ^ (tgt * ng_primes[cw % len(ng_primes)])) & ng_mask).astype(np.int64)
                np.add.at(ctx_tables[oi], ck, 1)
                np.add.at(full_tables[oi], fk, 1)
            print(f"ngram_prefill:rank{rank} pre-filled {pre_end} positions in {time.perf_counter()-t_pf:.1f}s", flush=True)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                seg_len = wlen - s
                if seg_len <= 0:
                    continue
                scored_nll = nll[i, s:wlen].to(torch.float64)
                if use_ngram:
                    seg_nll_np = scored_nll.cpu().numpy()
                    seg_model_p = np.exp(-seg_nll_np)
                    n_seg = len(seg_nll_np)
                    # Global positions of TARGET tokens being predicted
                    global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
                    # Entropy-adaptive alpha
                    if args.ngram_entropy:
                        with torch.no_grad():
                            lp = F.log_softmax(logits[i, s:wlen].float(), dim=-1)
                            seg_ent = -(lp.exp() * lp).sum(dim=-1).cpu().numpy()
                        alpha_per_tok = args.ngram_ent_base + args.ngram_ent_range / (
                            1.0 + np.exp(-args.ngram_ent_scale * (seg_ent - args.ngram_ent_thresh)))
                    # Precompute hashes for all orders
                    order_data = []
                    for oi in range(_n_orders):
                        ctx_w = args.ngram_min_order + oi - 1
                        valid = global_j >= ctx_w
                        if not valid.any():
                            order_data.append(None)
                            continue
                        v_idx = np.nonzero(valid)[0]
                        jv = global_j[v_idx]
                        ctx_hash = np.zeros(len(jv), dtype=np.uint64)
                        for k in range(ctx_w):
                            tok = val_np[jv - (ctx_w - k)].astype(np.uint64)
                            ctx_hash ^= tok * ng_primes[k % len(ng_primes)]
                        ctx_key = (ctx_hash & ng_mask).astype(np.int64)
                        tgt_np = val_np[jv].astype(np.uint64)
                        full_key = ((ctx_hash ^ (tgt_np * ng_primes[ctx_w % len(ng_primes)])) & ng_mask).astype(np.int64)
                        order_data.append((v_idx, ctx_key, full_key))
                    # Multi-order backoff: highest order first
                    best_p_ng = np.full(n_seg, -1.0)
                    for oi in range(_n_orders - 1, -1, -1):
                        if order_data[oi] is None:
                            continue
                        v_idx, ctx_key, full_key = order_data[oi]
                        ctx_counts = ctx_tables[oi][ctx_key].astype(np.float64)
                        full_counts = full_tables[oi][full_key].astype(np.float64)
                        has_match = ctx_counts >= float(args.ngram_min_count)
                        needs_fill = has_match & (best_p_ng[v_idx] < 0)
                        if needs_fill.any():
                            fill_idx = v_idx[needs_fill]
                            p = np.minimum(full_counts[needs_fill], ctx_counts[needs_fill]) / np.maximum(ctx_counts[needs_fill], 1.0)
                            best_p_ng[fill_idx] = np.clip(p, 0.0, 1.0)
                    # Mix model prob with n-gram
                    has_match = best_p_ng >= 0
                    if has_match.any():
                        if args.ngram_entropy:
                            alpha = alpha_per_tok[has_match]
                        else:
                            alpha = args.ngram_alpha
                        seg_model_p[has_match] = (1.0 - alpha) * seg_model_p[has_match] + alpha * best_p_ng[has_match]
                    seg_nll_np = -np.log(np.clip(seg_model_p, 1e-12, 1.0))
                    # Score-first: update ALL order tables AFTER scoring
                    for oi in range(_n_orders):
                        if order_data[oi] is None:
                            continue
                        v_idx, ctx_key, full_key = order_data[oi]
                        np.add.at(ctx_tables[oi], ctx_key, 1)
                        np.add.at(full_tables[oi], full_key, 1)
                    scored_nll = torch.from_numpy(seg_nll_np).to(dtype=torch.float64, device=device)
                loss_sum += scored_nll.sum()
                token_count += float(seg_len)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale
def collect_hessians(
    model: nn.Module, train_loader, args, device: torch.device,
    grad_accum_steps: int, num_batches: int = 256,
) -> dict[str, Tensor]:
    hessians: dict[str, Tensor] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear):
            pname = name + ".weight"
            cols = module.weight.shape[1]
            hessians[pname] = torch.zeros(cols, cols, dtype=torch.float32, device="cpu")
            def make_hook(pn):
                def hook_fn(mod, inp, out):
                    x = inp[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pn] += (x.T @ x).cpu()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(pname)))
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(num_batches):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            model(x, y)
    for h in hooks:
        h.remove()
    for pn in hessians:
        H = hessians[pn]
        H /= num_batches
        damp = args.gptq_damp_factor * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[pn] = H
    return hessians
def quantize_int6_gptq(
    weight: Tensor, hessian: Tensor, clip_range: int = 31, block_size: int = 128,
    damp_factor: float = 0.01,
) -> tuple[Tensor, Tensor]:
    t32 = weight.float()
    if t32.ndim != 2:
        return quantize_int6_per_row(t32, clip_range)
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = damp_factor * torch.mean(torch.diag(H))
    H[torch.arange(cols, device=H.device), torch.arange(cols, device=H.device)] += damp
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    try:
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except RuntimeError:
        H.diagonal().add_(damp * 10)
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    best_q, best_scale, best_err = None, None, float("inf")
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, inv_perm]
    return best_q, best_scale
def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str],
                         hessians: dict[str, Tensor] | None = None,
                         gptq_block_size: int = 128, gptq_damp_factor: float = 0.01,
                         clip_range: int = 31):
    num_layers_total = max(
        (int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
        default=0,
    ) + 1
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            H = hessians.get(name) if hessians else None
            if H is not None and t.ndim == 2:
                q, s = quantize_int6_gptq(t, H, clip_range=clip_range, block_size=gptq_block_size, damp_factor=gptq_damp_factor)
            else:
                q, s = quantize_int6_per_row(t, clip_range=clip_range)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta
def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out
def _make_gpt(args, device):
    a = args
    m = GPT(vocab_size=a.vocab_size, num_layers=a.num_layers, model_dim=a.model_dim, num_heads=a.num_heads, num_kv_heads=a.num_kv_heads, mlp_mult=a.mlp_mult, tie_embeddings=a.tie_embeddings, tied_embed_init_std=a.tied_embed_init_std, logit_softcap=a.logit_softcap, rope_base=a.rope_base, qk_gain_init=a.qk_gain_init, bigram_vocab_size=a.bigram_vocab_size, bigram_dim=a.bigram_dim, xsa_last_n=a.xsa_last_n, rope_dims=a.rope_dims, ln_scale=a.ln_scale, ve_enabled=a.ve_enabled, ve_dim=a.ve_dim, ve_layers=a.ve_layers, use_vrl=a.vrl).to(device).bfloat16()
    for mod in m.modules():
        if isinstance(mod, CastedLinear): mod.float()
    restore_low_dim_params_to_fp32(m)
    return m
def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    CastedLinear._qat_enabled = False
    CastedLinear._clip_range = args.clip_range
    log0(f"mixed_precision: clip_range={args.clip_range} ({'int5' if args.clip_range == 15 else 'int6'}) compressor={args.compressor}")
    base_model = _make_gpt(args, device)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named_params if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            matrix_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)
    ab = (args.beta1, args.beta2)
    optimizer_tok = torch.optim.AdamW(tok_params, betas=ab, eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=ab, eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}], betas=ab, eps=args.adam_eps, fused=True)
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    vrl_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_vrl]
    log0(f"VRL:{args.vrl} active_layers:{vrl_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} iterations:{args.iterations} warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")
    log0(f"seed:{args.seed}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    # Warmup phase
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = 0.997
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        if CastedLinear._qat_enabled and args.late_qat_threshold > 0:
            qat_progress = 1.0 - scale / args.late_qat_threshold
            qat_progress = max(0.0, min(1.0, qat_progress))
            new_alpha = 1.0 + 15.0 * qat_progress
            for m in base_model.modules():
                if isinstance(m, CastedLinear):
                    m._soft_round_alpha.fill_(new_alpha)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
    # Apply averaged weights: blend SWA (if available) with EMA
    if swa_state is not None and swa_count > 0:
        blend = args.swa_ema_blend
        log0(f"swa:applying {swa_count} snapshots, blending with EMA ({blend:.2f}/{1-blend:.2f})")
        swa_avg = {name: (t / swa_count).to(device) for name, t in swa_state.items()}
        current_state = base_model.state_dict()
        avg_state = {}
        for name in current_state:
            ema_w = ema_state[name].to(dtype=current_state[name].dtype)
            swa_w = swa_avg[name].to(dtype=current_state[name].dtype)
            avg_state[name] = blend * ema_w + (1 - blend) * swa_w
    else:
        log0("ema:applying EMA weights (no SWA snapshots)")
        current_state = base_model.state_dict()
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(args, compiled_model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms")
    export_sd = base_model.state_dict()
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    # GPTQ: collect Hessians for calibration-based quantization
    hessians = None
    if args.gptq_enabled:
        log0(f"gptq:collecting hessians batches={args.gptq_calib_batches} source={args.gptq_calib_source}")
        t_hess = time.perf_counter()
        calib_loader = DistributedTokenLoader(args.val_files if args.gptq_calib_source == "val" else args.train_files, rank, world_size, device)
        hessians = collect_hessians(base_model, calib_loader, args, device, grad_accum_steps, num_batches=args.gptq_calib_batches)
        log0(f"gptq:hessians collected layers={len(hessians)} time={time.perf_counter() - t_hess:.1f}s")
        del calib_loader
        torch.cuda.empty_cache()
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"}, hessians=hessians, gptq_block_size=args.gptq_block_size, gptq_damp_factor=args.gptq_damp_factor, clip_range=args.clip_range)
    target_bytes = 16_000_000
    code_bytes = len(code.encode("utf-8"))
    target_model_bytes = target_bytes - code_bytes - 5_000
    def _serialize_and_compress(qr, qm, fast=False):
        buf = io.BytesIO()
        torch.save({"w": qr, "m": qm}, buf)
        raw = buf.getvalue()
        if args.compressor == "zstd":
            import zstandard as zstd
            level = 10 if fast else 22
            return zstd.ZstdCompressor(level=level).compress(raw)
        preset = 6 if fast else (9 | lzma.PRESET_EXTREME)
        return lzma.compress(raw, preset=preset)
    test_blob = _serialize_and_compress(quant_result, quant_meta)
    log0(f"gptq:pre_prune artifact={len(test_blob)} target={target_model_bytes}")
    if len(test_blob) > target_model_bytes:
        total_params = sum(v.numel() for v in quant_result.values() if v.dtype == torch.int8)
        max_prune = max(1000, total_params // 50)
        log0(f"gptq:over by {len(test_blob) - target_model_bytes} bytes, max_prune={max_prune}")
        pc = []  # (cost, qkey, idx_tuple)
        for name, info in quant_meta.items():
            if not (isinstance(info, dict) and info.get("type") == "int6"):
                continue
            qk, q, s = name + ".q", quant_result[name + ".q"], quant_result[name + ".scale"]
            H = hessians.get(name) if hessians else None
            hd = torch.diag(H).float() if H is not None else None
            for idx in (q.abs() == 1).nonzero(as_tuple=False):
                r, c = idx[0].item(), (idx[1].item() if len(idx) > 1 else 0)
                sc = s[r].float().item() if s.ndim > 0 else s.float().item()
                pc.append((sc * sc * (hd[c].item() if hd is not None and c < len(hd) else 1.0), qk, tuple(idx.tolist())))
        pc.sort(key=lambda x: x[0])
        pc = pc[:max_prune]
        log0(f"gptq:pruning candidates={len(pc)}")
        def _try_prune(n):
            qr = {k: v.clone() for k, v in quant_result.items()}
            for i in range(n):
                qr[pc[i][1]][pc[i][2]] = 0
            return qr
        lo, hi, best_n = 0, len(pc), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid == 0: lo = 1; continue
            blob = _serialize_and_compress(_try_prune(mid), quant_meta, fast=True)
            if len(blob) <= int(target_model_bytes * 0.997): best_n = mid; hi = mid - 1
            else: lo = mid + 1
        if best_n > 0:
            final_blob = _serialize_and_compress(_try_prune(best_n), quant_meta)
            while len(final_blob) > target_model_bytes and best_n < len(pc):
                best_n = min(best_n + max(1, best_n // 10), len(pc))
                final_blob = _serialize_and_compress(_try_prune(best_n), quant_meta)
            for i in range(best_n):
                quant_result[pc[i][1]][pc[i][2]] = 0
            log0(f"gptq:pruned {best_n} values ({100*best_n/total_params:.2f}%)")
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    if master_process:
        torch.save({"quantized": quant_result, "meta": quant_meta}, "final_int6_model.pt")
        log0(f"Saved quantized model to final_int6_model.pt")
    quant_raw = quant_buf.getvalue()
    if args.compressor == "zstd":
        import zstandard as zstd
        quant_blob = zstd.ZstdCompressor(level=22).compress(quant_raw)
        comp_label = "zstd"
    else:
        quant_blob = lzma.compress(quant_raw, preset=9 | lzma.PRESET_EXTREME)
        comp_label = "lzma"
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        log0(f"Serialized model int{args.clip_range*2+1}+{comp_label}: {quant_file_bytes} bytes")
        log0(f"Total submission size: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    if args.compressor == "zstd":
        import zstandard as zstd
        decompressed = zstd.ZstdDecompressor().decompress(quant_blob_disk)
    else:
        decompressed = lzma.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    eval_model = _make_gpt(args, device)
    eval_model.load_state_dict(deq_state, strict=True)
    CastedLinear._qat_enabled = False
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, compiled_eval, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, eval_seq_len=effective_eval_seq_len)
    torch.cuda.synchronize()
    log0(f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(args, eval_model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, stride=args.eval_stride, eval_seq_len=sw_seq_len)
        torch.cuda.synchronize()
        log0(f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms")
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
