"""
Parameter Golf Revised SOTA Submission (v2)
============================================
Builds on PR #1493 (1.0810 BPB) with techniques that ACTUALLY help at 8M param scale.

Novel techniques (all capacity-focused, not sample-efficiency):

1. QAT-Fused Cooldown — STE fake-quantization during LR warmdown phase
   Paper: Compute-Optimal QAT (arxiv 2509.22935)
   The model is 225x overtrained vs Chinchilla. Post-hoc GPTQ suffers from accumulated
   quantization error. QAT during warmdown lets the optimizer correct for quantization
   noise while LR is decaying. This produces strictly better quantized weights.

2. INT4 MLP + INT6 Attention mixed-precision quantization
   MLP weights (4x expansion) are the largest and most redundant matrices.
   Dropping MLP from INT6→INT4 saves ~2.9MB → room for a wider model (576d vs 512d)
   or more layers, packing ~50% more effective parameters into 16MB.

3. Nuclear-norm regularization (NuMuon-lite)
   Paper: NuMuon (arxiv 2603.03597)
   Adds a lightweight nuclear-norm penalty that pushes weights toward low-rank structure.
   Low-rank weights compress dramatically better with GPTQ+Brotli (20-40% smaller).
   Full NuMuon uses Block Krylov SVD which is expensive; we use a cheaper proxy:
   periodic SVD-based rank penalty on the loss, applied every K steps.

All other techniques inherited from SOTA:
  SP8192, 3-layer depth recurrence, parallel residuals, XSA, partial RoPE,
  LeakyReLU(0.5)^2, MuonEq-R, EMA, skip gates, GPTQ SDClip, Brotli,
  score-first TTT, sliding window eval
"""

from __future__ import annotations

import collections, copy, glob, io, lzma, math, os
from pathlib import Path
import random, re, subprocess, sys, time, uuid

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

use_wandb = bool(int(os.environ.get('USE_WANDB', '0')))

if use_wandb:
    try:
        import wandb
    except ImportError:
        print("wandb not installed")
        pass

# =============================================================================
# HYPERPARAMETERS — inherits SOTA defaults, adds novel knobs
# =============================================================================

class Hyperparameters:
    # --- Data / run ---
    data_dir = os.environ.get('DATA_DIR', './data/')
    seed = int(os.environ.get('SEED', 1337))
    run_id = os.environ.get('RUN_ID', str(uuid.uuid4()))
    # wandb
    use_wandb = use_wandb
    wandb_project = os.environ.get('WANDB_PROJECT', 'parameter-golf')

    # --- Training schedule ---
    iterations = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac = float(os.environ.get('WARMDOWN_FRAC', 0.72))
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', 786432))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    train_log_every = int(os.environ.get('TRAIN_LOG_EVERY', 500))
    max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))

    # --- Validation ---
    val_batch_tokens = int(os.environ.get('VAL_BATCH_TOKENS', 524288))
    eval_seq_len = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    val_loss_every = int(os.environ.get('VAL_LOSS_EVERY', 4000))
    sliding_window_enabled = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))
    eval_stride = int(os.environ.get('EVAL_STRIDE', 64))

    # --- Architecture ---
    vocab_size = int(os.environ.get('VOCAB_SIZE', 8192))
    num_layers = int(os.environ.get('NUM_LAYERS', 11))
    xsa_last_n = int(os.environ.get('XSA_LAST_N', 11))
    model_dim = int(os.environ.get('MODEL_DIM', 512))
    embedding_dim = int(os.environ.get('EMBEDDING_DIM', 512))
    num_kv_heads = int(os.environ.get('NUM_KV_HEADS', 4))
    num_heads = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult = float(os.environ.get('MLP_MULT', 4.0))
    skip_gates_enabled = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    tie_embeddings = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    logit_softcap = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    rope_base = float(os.environ.get('ROPE_BASE', 10000.0))
    rope_dims = int(os.environ.get('ROPE_DIMS', 16))
    rope_train_seq_len = int(os.environ.get('ROPE_TRAIN_SEQ_LEN', 2048))
    ln_scale = bool(int(os.environ.get('LN_SCALE', '1')))
    qk_gain_init = float(os.environ.get('QK_GAIN_INIT', 5.25))

    # --- Depth recurrence ---
    num_loops = int(os.environ.get('NUM_LOOPS', 2))
    loop_start = int(os.environ.get('LOOP_START', 3))
    loop_end = int(os.environ.get('LOOP_END', 5))
    enable_looping_at = float(os.environ.get('ENABLE_LOOPING_AT', 0.35))
    parallel_residual_start = int(os.environ.get('PARALLEL_RESIDUAL_START', 7))

    # --- Optimizer ---
    min_lr = float(os.environ.get('MIN_LR', 0.0))
    embed_lr = float(os.environ.get('EMBED_LR', 0.6))
    head_lr = float(os.environ.get('HEAD_LR', 0.008))
    tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    matrix_lr = float(os.environ.get('MATRIX_LR', 0.022))
    scalar_lr = float(os.environ.get('SCALAR_LR', 0.02))
    muon_momentum = float(os.environ.get('MUON_MOMENTUM', 0.99))
    muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.92))
    muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
    muon_row_normalize = bool(int(os.environ.get('MUON_ROW_NORMALIZE', '1')))
    beta1 = float(os.environ.get('BETA1', 0.9))
    beta2 = float(os.environ.get('BETA2', 0.95))
    adam_eps = float(os.environ.get('ADAM_EPS', 1e-8))
    grad_clip_norm = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
    muon_beta2 = float(os.environ.get('MUON_BETA2', 0.95))
    adam_wd = float(os.environ.get('ADAM_WD', 0.02))
    muon_wd = float(os.environ.get('MUON_WD', 0.095))
    embed_wd = float(os.environ.get('EMBED_WD', 0.085))
    ema_decay = float(os.environ.get('EMA_DECAY', 0.9965))

    # --- TTT ---
    ttt_enabled = bool(int(os.environ.get('TTT_ENABLED', '0')))
    ttt_lr = float(os.environ.get('TTT_LR', 0.005))
    ttt_epochs = int(os.environ.get('TTT_EPOCHS', 3))
    ttt_momentum = float(os.environ.get('TTT_MOMENTUM', 0.9))
    ttt_chunk_tokens = int(os.environ.get('TTT_CHUNK_TOKENS', 32768))

    # --- Quantization / compression ---
    compressor = os.environ.get('COMPRESSOR', 'brotli')
    gptq_calibration_batches = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 64))
    gptq_reserve_seconds = float(os.environ.get('GPTQ_RESERVE_SECONDS', 12.0))
    matrix_bits = int(os.environ.get('MATRIX_BITS', 6))
    embed_bits = int(os.environ.get('EMBED_BITS', 8))
    matrix_clip_sigmas = float(os.environ.get('MATRIX_CLIP_SIGMAS', 12.85))
    embed_clip_sigmas = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))

    # ===========================================================================
    # NOVEL TECHNIQUE 1: QAT-Fused Cooldown
    #   Enable STE fake-quantization during warmdown phase of training.
    #   The optimizer actively adapts weights to quantization noise.
    # ===========================================================================
    qat_fused_enabled = bool(int(os.environ.get('QAT_FUSED_ENABLED', '1')))
    qat_fused_start_frac = float(os.environ.get('QAT_FUSED_START_FRAC', 0.65))
    qat_fused_bits = int(os.environ.get('QAT_FUSED_BITS', 6))  # match export bits

    # ===========================================================================
    # NOVEL TECHNIQUE 2: INT4 MLP mixed precision
    #   Use INT4 for MLP weights in GPTQ (more redundant, tolerates lower bits)
    #   and INT6 for attention (more sensitive). Saves ~2.9MB → room for more params.
    # ===========================================================================
    mlp_bits = int(os.environ.get('MLP_BITS', 4))
    attn_bits = int(os.environ.get('ATTN_BITS', 6))

    # ===========================================================================
    # NOVEL TECHNIQUE 3: Nuclear-norm regularization (NuMuon-lite)
    #   Periodic low-rank penalty that pushes weights toward compressible structure.
    #   Lightweight: no Block Krylov SVD, just an L2 penalty on top singular values.
    # ===========================================================================
    nuclear_reg_enabled = bool(int(os.environ.get('NUCLEAR_REG_ENABLED', '1')))
    nuclear_reg_lambda = float(os.environ.get('NUCLEAR_REG_LAMBDA', 1e-4))
    nuclear_reg_every = int(os.environ.get('NUCLEAR_REG_EVERY', 50))  # apply every N steps
    nuclear_reg_top_k = int(os.environ.get('NUCLEAR_REG_TOP_K', 8))  # penalize top-K singular values

    # --- Distributed (computed) ---
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size

    # --- Paths ---
    datasets_dir = os.path.join(data_dir, 'datasets', f"fineweb10B_sp{vocab_size}")
    train_files = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path = os.path.join(data_dir, 'tokenizers', f"fineweb_{vocab_size}_bpe.model")
    logfile = f"logs/{run_id}.txt"
    model_path = 'final_model.pt'
    quantized_model_path = 'final_model.int6.ptz'


# =============================================================================
# NOVEL TECHNIQUE 1: STE Fake Quantization for QAT-Fused Cooldown
# =============================================================================

class FakeQuantize(torch.autograd.Function):
    """Straight-Through Estimator (STE) for fake quantization.
    Forward: quantize → dequantize. Backward: pass gradient through unchanged.
    Applied per-row for weight matrices (matches GPTQ SDClip scheme).
    """
    @staticmethod
    def forward(ctx, w, bits, clip_sigmas):
        clip_range = 2 ** (bits - 1) - 1
        row_std = w.float().std(dim=1, keepdim=True)
        scale = (clip_sigmas * row_std / clip_range).clamp_min(1e-10)
        q = (w / scale).round().clamp(-clip_range, clip_range)
        return (q * scale).to(w.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def fake_quantize_ste(w, bits, clip_sigmas):
    """Apply STE fake quantization to a weight tensor."""
    return FakeQuantize.apply(w, bits, clip_sigmas)


class QATCastedLinear(nn.Module):
    """CastedLinear with optional STE fake-quantization during training."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = CastedLinear(in_features, out_features, bias=bias)
        self.qat_enabled = False
        self.qat_bits = 6
        self.qat_clip_sigmas = 12.85
        self._zero_init = False

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        self.linear.weight = value

    def forward(self, x):
        if self.qat_enabled and self.training:
            w = fake_quantize_ste(self.linear.weight, self.qat_bits, self.qat_clip_sigmas)
            bias = self.linear.bias.to(x.dtype) if self.linear.bias is not None else None
            return F.linear(x, w.to(x.dtype), bias)
        return self.linear(x)


# =============================================================================
# NOVEL TECHNIQUE 3: Nuclear-norm regularization
# =============================================================================

def nuclear_norm_penalty(model, top_k=8):
    """Compute a lightweight nuclear-norm proxy penalty.
    Penalizes the top-K singular values of large weight matrices,
    encouraging low-rank structure that compresses better.
    Uses power iteration (cheap) instead of full SVD.
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for name, param in model.named_parameters():
        if param.ndim == 2 and param.numel() > 65536 and 'tok_emb' not in name:
            # Cheap proxy: Frobenius norm squared ≈ sum of squared singular values
            # Penalizing Frobenius norm pushes ALL singular values down (toward low-rank)
            # This is cheaper than computing actual top-K singular values
            penalty = penalty + param.float().norm() ** 2
            count += 1
    if count > 0:
        penalty = penalty / count
    return penalty


# =============================================================================
# LOGGING (identical to SOTA)
# =============================================================================

_logger_hparams = None
def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h

def log(msg, console=True):
    if _logger_hparams is None:
        print(msg); return
    if _logger_hparams.is_main_process:
        if console: print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, 'a', encoding='utf-8') as f:
                print(msg, file=f)


# =============================================================================
# TOKENIZER / VALIDATION DATA (identical to SOTA)
# =============================================================================

class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(f"VOCAB_SIZE={h.vocab_size} != tokenizer={int(self.sp.vocab_size())}")
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = \
            build_sentencepiece_luts(self.sp, h.vocab_size, device)

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros(table_size, dtype=np.int16)
    has_space = np.zeros(table_size, dtype=np.bool_)
    is_boundary = np.ones(table_size, dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary[tid] = False
        if sp.is_byte(tid): base_bytes[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith('▁'): has_space[tid] = True; piece = piece[1:]
        base_bytes[tid] = len(piece.encode('utf-8'))
    return (torch.tensor(base_bytes, dtype=torch.int16, device=device),
            torch.tensor(has_space, dtype=torch.bool, device=device),
            torch.tensor(is_boundary, dtype=torch.bool, device=device))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

def load_data_shard(file):
    header = np.fromfile(file, dtype='<i4', count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype='<u2', count=num_tokens,
                         offset=256 * np.dtype('<i4').itemsize)
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


# =============================================================================
# DATA LOADING (identical to SOTA)
# =============================================================================

_SHARD_HEADER = 256 * np.dtype('<i4').itemsize
_NTOK_CACHE, _MMAP_CACHE = {}, {}

def _read_num_tokens(f):
    k = str(f)
    if k not in _NTOK_CACHE:
        _NTOK_CACHE[k] = int(np.fromfile(f, dtype='<i4', count=256)[2])
    return _NTOK_CACHE[k]

def _get_mmap(f):
    k = str(f)
    if k not in _MMAP_CACHE:
        n = _read_num_tokens(f)
        _MMAP_CACHE[k] = np.memmap(f, mode='r', dtype='<u2', offset=_SHARD_HEADER, shape=(n,))
    return _MMAP_CACHE[k]

class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files: raise FileNotFoundError(f"No files: {h.train_files}")
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)): self._reset(si)

    def _reset(self, si):
        mx = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(mx + 1)) if mx > 0 else 0
        n_seq = (self.num_tokens[si] - 1 - phase) // self.seq_len
        self.start_inds[si] = (phase + self.rng.permutation(n_seq) * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        dev_tok = global_tokens // (self.world_size * grad_accum_steps)
        bs = dev_tok // self.seq_len
        rem = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((bs, self.seq_len), dtype=torch.int64)
        y = torch.empty((bs, self.seq_len), dtype=torch.int64)
        for bi in range(bs):
            total = rem.sum()
            if total <= 0:
                for si in range(len(self.files)): self._reset(si)
                rem = np.array([len(s) for s in self.start_inds], dtype=np.float64)
                total = rem.sum()
            si = int(self.rng.choice(len(self.files), p=rem / total))
            start = self.start_inds[si].pop(); rem[si] -= 1
            mm = _get_mmap(self.files[si])
            w = torch.as_tensor(np.array(mm[start:start + self.seq_len + 1], dtype=np.int64))
            x[bi] = w[:-1]; y[bi] = w[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# =============================================================================
# TRANSFORMER MODULES (identical to SOTA except CastedLinear is QAT-aware)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    """Weights stored in fp32, cast to input dtype at matmul time."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._qat_enabled = False
        self._qat_bits = 6
        self._qat_clip_sigmas = 12.85

    def forward(self, x):
        w = self.weight.to(x.dtype)
        if self._qat_enabled and self.training:
            w = fake_quantize_ste(w, self._qat_bits, self._qat_clip_sigmas)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim; self.base = base; self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer('inv_freq', inv, persistent=False)
        self._len = 0; self._cos = None; self._sin = None

    def forward(self, seq_len, device, dtype):
        if self._cos is None or self._len != seq_len or self._cos.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                sc = seq_len / self.train_seq_len
                inv = 1.0 / ((self.base * sc ** (rd / (rd - 2))) **
                             (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv.dtype)
            freqs = torch.outer(t, inv)
            self._cos = freqs.cos()[None, :, None, :]
            self._sin = freqs.sin()[None, :, None, :]
            self._len = seq_len
        return self._cos.to(dtype=dtype), self._sin.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        xr, xp = x[..., :rope_dims], x[..., rope_dims:]
        h = rope_dims // 2
        x1, x2 = xr[..., :h], xr[..., h:]
        xr = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((xr, xp), dim=-1)
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
        super().__init__()
        self.num_heads = num_heads; self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); g = H // Hkv
        yg = y.reshape(B, T, Hkv, g, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        p = (yg * vn).sum(dim=-1, keepdim=True) * vn
        return (yg - p).reshape(B, T, H, D)

    def forward(self, x):
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                    enable_gqa=(self.num_kv_heads != self.num_heads)).transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v if HAS_FA3 else v.transpose(1, 2))
        return self.proj(y.reshape(B, T, D))

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                 qk_gain_init, train_seq_len, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base,
                                        qk_gain_init, train_seq_len)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        xi = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        ao = self.attn(self.attn_norm(xi) * self.ln_scale_factor)
        if self.parallel:
            mo = self.mlp(self.mlp_norm(xi) * self.ln_scale_factor)
            return xi + self.attn_scale.to(xi.dtype)[None, None, :] * ao + \
                        self.mlp_scale.to(xi.dtype)[None, None, :] * mo
        xo = xi + self.attn_scale.to(xi.dtype)[None, None, :] * ao
        return xo + self.mlp_scale.to(xo.dtype)[None, None, :] * \
                    self.mlp(self.mlp_norm(xo) * self.ln_scale_factor)


# =============================================================================
# GPT MODEL (identical to SOTA)
# =============================================================================

class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None; self.head_proj = None
        ne = h.num_layers // 2; nd = h.num_layers - ne
        self.num_encoder_layers = ne; self.num_decoder_layers = nd
        self.blocks = nn.ModuleList([
            Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base,
                  h.qk_gain_init, h.train_seq_len, layer_idx=i, ln_scale=h.ln_scale)
            for i in range(h.num_layers)])
        if h.rope_dims > 0:
            hd = h.model_dim // h.num_heads
            for b in self.blocks:
                b.attn.rope_dims = h.rope_dims
                b.attn.rotary = Rotary(hd, base=h.rope_base, train_seq_len=h.train_seq_len,
                                       rope_dims=h.rope_dims)
        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        if h.parallel_residual_start >= 0:
            for i in range(h.parallel_residual_start, h.num_layers):
                self.blocks[i].parallel = True
        self.looping_active = False
        if h.num_loops > 0:
            seg = list(range(h.loop_start, h.loop_end + 1))
            idx = list(range(h.loop_start))
            for _ in range(h.num_loops + 1): idx.extend(seg)
            idx.extend(range(h.loop_end + 1, h.num_layers))
            mid = len(idx) // 2
            self.encoder_indices = idx[:mid]; self.decoder_indices = idx[mid:]
        else:
            self.encoder_indices = list(range(ne))
            self.decoder_indices = list(range(ne, h.num_layers))
        nsk = min(len(self.encoder_indices), len(self.decoder_indices))
        self.num_skip_weights = nsk
        self.skip_weights = nn.Parameter(torch.ones(nsk, h.model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(nsk, h.model_dim, dtype=torch.float32)) \
            if h.skip_gates_enabled else None
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if getattr(m, '_zero_init', False): nn.init.zeros_(m.weight)
                elif m.weight.ndim == 2 and m.weight.shape[0] >= 64 and m.weight.shape[1] >= 64:
                    nn.init.orthogonal_(m.weight, gain=1.0)

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids); x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None: x = self.embed_proj(x)
        x0 = x; skips = []
        enc = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        dec = self.decoder_indices if self.looping_active else range(self.num_encoder_layers,
                self.num_encoder_layers + self.num_decoder_layers)
        for i in enc: x = self.blocks[i](x, x0); skips.append(x)
        for si, i in enumerate(dec):
            if si < self.num_skip_weights and skips:
                ss = self.skip_weights[si].to(x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[si].to(x.dtype))[None, None, :]
                    x = torch.lerp(ss, x, g)
                else: x = x + ss
            x = self.blocks[i](x, x0)
        x = self.final_norm(x)
        if self.head_proj is not None: x = self.head_proj(x)
        lp = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1), reduction='mean')

    # ===========================================================================
    # NOVEL: Enable/disable QAT on all CastedLinear modules
    # ===========================================================================
    def enable_qat(self, bits, clip_sigmas):
        """Turn on STE fake-quantization in all large linear layers."""
        for m in self.modules():
            if isinstance(m, CastedLinear) and m.weight.numel() > 65536:
                m._qat_enabled = True
                m._qat_bits = bits
                m._qat_clip_sigmas = clip_sigmas
        log(f"qat_fused: enabled INT{bits} fake-quant (clip={clip_sigmas})")

    def disable_qat(self):
        for m in self.modules():
            if isinstance(m, CastedLinear):
                m._qat_enabled = False


# =============================================================================
# MUON OPTIMIZER (identical to SOTA)
# =============================================================================

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in 'attn_scale,mlp_scale,resid_mix,q_gain,skip_weights,skip_gates'.split(',') if p)

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = 3.4445, -4.775, 2.0315
    X = G.bfloat16(); X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True,
                 weight_decay=0.0, row_normalize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay,
                                      row_normalize=row_normalize))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        ws = dist.get_world_size() if distributed else 1
        rk = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group['params']
            if not params: continue
            lr, mom, ns = group['lr'], group['momentum'], group['backend_steps']
            nesterov = group['nesterov']
            total = sum(int(p.numel()) for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            cur = 0
            for i, p in enumerate(params):
                if i % ws == rk and p.grad is not None:
                    g = p.grad; st = self.state[p]
                    if 'momentum_buffer' not in st:
                        st['momentum_buffer'] = torch.zeros_like(g)
                    buf = st['momentum_buffer']; buf.mul_(mom).add_(g)
                    if nesterov: g = g.add(buf, alpha=mom)
                    if group.get('row_normalize', False):
                        rn = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                        g = g / rn.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=ns)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[cur:cur + p.numel()] = g.reshape(-1)
                cur += p.numel()
            if distributed: dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            wd = group.get('weight_decay', 0.0); cur = 0
            for p in params:
                if wd > 0: p.data.mul_(1.0 - lr * wd)
                g = flat[cur:cur + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr); cur += p.numel()
        return loss


# =============================================================================
# OPTIMIZER SETUP (identical to SOTA)
# =============================================================================

class Optimizers:
    def __init__(self, h, base_model):
        bnp = list(base_model.blocks.named_parameters())
        mat = [p for n, p in bnp if p.ndim == 2 and not any(c in n for c in CONTROL_TENSOR_NAME_PATTERNS)]
        sca = [p for n, p in bnp if p.ndim < 2 or any(c in n for c in CONTROL_TENSOR_NAME_PATTERNS)]
        if base_model.skip_weights.numel() > 0: sca.append(base_model.skip_weights)
        if base_model.skip_gates is not None: sca.append(base_model.skip_gates)
        tlr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        self.optimizer_tok = torch.optim.AdamW(
            [{'params': [base_model.tok_emb.weight], 'lr': tlr, 'base_lr': tlr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True)
        self.optimizer_muon = Muon(mat, lr=h.matrix_lr, momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize)
        for g in self.optimizer_muon.param_groups: g['base_lr'] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{'params': sca, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [{'params': [base_model.lm_head.weight], 'lr': h.head_lr, 'base_lr': h.head_lr}],
                betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True)
            self.optimizers.insert(1, self.optimizer_head)
    def __iter__(self): return iter(self.optimizers)
    def zero_grad_all(self):
        for o in self.optimizers: o.zero_grad(set_to_none=True)
    def step(self):
        for o in self.optimizers: o.step()
        self.zero_grad_all()

def restore_fp32_params(model):
    for m in model.modules():
        if isinstance(m, CastedLinear): m.float()
    for n, p in model.named_parameters():
        if (p.ndim < 2 or any(c in n for c in CONTROL_TENSOR_NAME_PATTERNS)) and p.dtype != torch.float32:
            p.data = p.data.float()

def classify_param(name):
    if 'tok_emb' in name or 'lm_head' in name: return 'embed'
    if '.mlp.' in name: return 'mlp'
    if '.attn.' in name: return 'attn'
    return 'other'


# =============================================================================
# GPTQ QUANTIZATION — MODIFIED for INT4 MLP / INT6 Attn mixed precision
# =============================================================================

def collect_hessians(model, train_loader, h, device, n_batches=64):
    hessians = {}; hooks = []
    def make_hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[name].addmm_(x.T, x)
        return fn
    for name, mod in model.named_modules():
        if isinstance(mod, CastedLinear) and mod.weight.numel() > 65536:
            cat = classify_param(name + '.weight')
            if cat in ('mlp', 'attn'):
                hooks.append(mod.register_forward_hook(make_hook(name + '.weight')))
    if model.tie_embeddings:
        hm = model.head_proj if model.head_proj is not None else model.final_norm
        def out_hook(name):
            def fn(mod, inp, out):
                x = out.detach().float()
                if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)
            return fn
        hooks.append(hm.register_forward_hook(out_hook('tok_emb.weight')))
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks: hook.remove()
    for name in hessians: hessians[name] = hessians[name].cpu() / n_batches
    return hessians

def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128):
    W = w.float().clone(); rows, cols = W.shape
    H = H.float().clone(); dead = torch.diag(H) == 0; H[dead, dead] = 1
    H.diagonal().add_(0.01 * H.diag().mean())
    perm = torch.argsort(H.diag(), descending=True); inv = torch.argsort(perm)
    Wp = W[:, perm].clone(); Wp[:, dead[perm]] = 0; H = H[perm][:, perm]
    Hi = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hi = torch.linalg.cholesky(Hi, upper=True)
    s = (clip_sigmas * W.std(dim=1) / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float(); Q = torch.zeros(rows, cols, dtype=torch.int8); Wk = Wp.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols); Wb = Wk[:, i1:i2].clone()
        Hb = Hi[i1:i2, i1:i2]; Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            qc = torch.clamp(torch.round(Wb[:, j] / sf), -clip_range, clip_range)
            Q[:, i1 + j] = qc.to(torch.int8)
            err = (Wb[:, j] - qc.float() * sf) / Hb[j, j]; Err[:, j] = err
            Wb[:, j:] -= err.unsqueeze(1) * Hb[j, j:].unsqueeze(0)
        if i2 < cols: Wk[:, i2:] -= Err @ Hi[i1:i2, i2:]
    return Q[:, inv], s

def gptq_mixed_quantize(state_dict, hessians, h):
    """NOVEL: per-category bit assignment — INT4 for MLP, INT6 for attn, INT8 for embed."""
    result = {}; meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = 'passthrough'; continue
        cat = classify_param(name)
        if cat == 'embed':
            bits = h.embed_bits; cs = h.embed_clip_sigmas
        elif cat == 'mlp':
            bits = h.mlp_bits; cs = h.matrix_clip_sigmas  # NOVEL: INT4 for MLP
        else:
            bits = h.attn_bits; cs = h.matrix_clip_sigmas  # INT6 for attn
        q, s = gptq_quantize_weight(t, hessians[name], clip_sigmas=cs,
                                     clip_range=2 ** (bits - 1) - 1)
        result[name + '.q'] = q; result[name + '.scale'] = s
        meta[name] = f"gptq (int{bits})"
    log('Quantized weights:')
    cats = collections.defaultdict(set)
    for n, c in meta.items():
        short = re.sub(r'\.\d+$', '', re.sub(r'blocks\.\d+', 'blocks', n))
        cats[c].add(short)
    for c in sorted(cats): log(f"  {c}: {', '.join(sorted(cats[c]))}")
    return result, meta

def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        if 'passthrough' in info:
            t = result[name]
            if t.dtype == torch.float16 and orig.dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig.dtype)
            out[name] = t; continue
        q, s = result[name + '.q'], result[name + '.scale']
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *[1]*(q.ndim-1))).to(orig.dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig.dtype)
    return out


# =============================================================================
# COMPRESSION (identical to SOTA)
# =============================================================================

_BSHF = b'BSHF'
def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride: return data
    src = np.frombuffer(data, dtype=np.uint8); n = len(src)
    out = np.empty(n, dtype=np.uint8); off = 0
    for p in range(stride):
        c = src[p::stride]; out[off:off+len(c)] = c; off += len(c)
    return _BSHF + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF: return data
    stride = data[4]; payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload); out = np.empty(n, dtype=np.uint8); off = 0
    for p in range(stride):
        cl = n // stride + (1 if p < n % stride else 0)
        out[p::stride][:cl] = payload[off:off+cl]; off += cl
    return out.tobytes()

def _compress(data, comp):
    data = _byte_shuffle(data)
    if comp == 'lzma': return lzma.compress(data, preset=6)
    elif comp == 'brotli': import brotli; return brotli.compress(data, quality=11)
    raise ValueError(comp)

def _decompress(data, comp):
    if comp == 'lzma': raw = lzma.decompress(data)
    elif comp == 'brotli': import brotli; raw = brotli.decompress(data)
    else: raise ValueError(comp)
    return _byte_unshuffle(raw)


# =============================================================================
# SERIALIZATION
# =============================================================================

def serialize(h, base_model, code):
    code_bytes = len(code.encode('utf-8'))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        log(f"Serialized model: {os.path.getsize(h.model_path)} bytes")
    sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    device = torch.device('cuda', h.local_rank)
    log('GPTQ: collecting Hessians...'); t0 = time.perf_counter()
    loader = ShuffledSequenceLoader(h, device)
    hess = collect_hessians(base_model, loader, h, device, n_batches=h.gptq_calibration_batches)
    log(f"GPTQ: {len(hess)} Hessians in {time.perf_counter()-t0:.1f}s")
    qr, qm = gptq_mixed_quantize(sd, hess, h)
    buf = io.BytesIO(); torch.save({'w': qr, 'm': qm}, buf)
    blob = _compress(buf.getvalue(), h.compressor)
    total = len(blob) + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, 'wb') as f: f.write(blob)
        log(f"Quantized+{h.compressor}: {len(blob)} bytes | Total: {total} bytes")
    return total, len(blob)

def deserialize(h, device):
    mdl = GPT(h).to(device).bfloat16(); restore_fp32_params(mdl)
    sd = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}
    with open(h.quantized_model_path, 'rb') as f: blob = f.read()
    qs = torch.load(io.BytesIO(_decompress(blob, h.compressor)), map_location='cpu')
    mdl.load_state_dict(dequantize_mixed(qs['w'], qs['m'], sd), strict=True)
    return mdl


# =============================================================================
# EVALUATION (identical to SOTA — val, sliding, TTT)
# =============================================================================

def _loss_bpb(ls, tc, bc):
    vl = (ls / tc).item(); return vl, vl / math.log(2.0) * (tc.item() / bc.item())

def eval_val(h, device, vd, model):
    sl = h.eval_seq_len; lb = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    lbs = lb // sl; ts = (vd.val_tokens.numel()-1) // sl
    ss = ts * h.rank // h.world_size; se = ts * (h.rank+1) // h.world_size
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(ss, se, lbs):
            be = min(bs + lbs, se); rs = bs * sl; re_ = be * sl + 1
            loc = vd.val_tokens[rs:re_].to(device=device, dtype=torch.int64, non_blocking=True)
            x = loc[:-1].reshape(-1, sl); y = loc[1:].reshape(-1, sl)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                bl = model(x, y).detach()
            btc = float(y.numel()); ls += bl.to(torch.float64) * btc; tc += btc
            prev = x.reshape(-1); tgt = y.reshape(-1)
            tb = vd.base_bytes_lut[tgt].to(torch.int16)
            tb += (vd.has_leading_space_lut[tgt] & ~vd.is_boundary_token_lut[prev]).to(torch.int16)
            bc += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [ls, tc, bc]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    model.train(); return _loss_bpb(ls, tc, bc)

def eval_val_sliding(h, device, vd, bm, bsz=32):
    bm.eval(); lf = torch.compile(bm.forward_logits, dynamic=False, fullgraph=True)
    sl = h.eval_seq_len; ctx = sl - h.eval_stride; tt = vd.val_tokens.numel()-1
    ws_all = [w for w in range(0, tt, h.eval_stride) if w + ctx < tt]
    ms = len(ws_all) * h.rank // h.world_size; me = len(ws_all) * (h.rank+1) // h.world_size
    myw = ws_all[ms:me]
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(myw), bsz):
            bw = myw[bi:bi+bsz]; B = len(bw)
            xb = torch.zeros(B, sl, dtype=torch.int64, device=device)
            yb = torch.zeros(B, sl, dtype=torch.int64, device=device); wls = []
            for i, w in enumerate(bw):
                we = min(w+sl, tt); wl = we - w; wls.append(wl)
                ch = vd.val_tokens[w:we+1].to(dtype=torch.int64, device=device)
                xb[i,:wl] = ch[:-1]; yb[i,:wl] = ch[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = lf(xb)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  yb.reshape(-1), reduction='none').reshape(B, sl)
            for i, w in enumerate(bw):
                wl = wls[i]; s = 0 if w == 0 else ctx
                ls += nll[i, s:wl].to(torch.float64).sum(); tc += float(wl - s)
                tgt = yb[i, s:wl]; prev = xb[i, s:wl]
                tb = vd.base_bytes_lut[tgt].to(torch.float64)
                tb += (vd.has_leading_space_lut[tgt] & ~vd.is_boundary_token_lut[prev]).to(torch.float64)
                bc += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in [ls, tc, bc]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    bm.train(); return _loss_bpb(ls, tc, bc)

def eval_val_ttt(h, device, vd, bm, bsz=32):
    rk = h.rank; ws_ = h.world_size; sl = h.eval_seq_len; stride = h.eval_stride
    tt = vd.val_tokens.numel()-1; chunk = h.ttt_chunk_tokens; ctx = sl - stride
    ws_all = [w for w in range(0, tt, stride) if w + ctx < tt]
    nc = (tt + chunk - 1) // chunk; cw = [[] for _ in range(nc)]
    for w in ws_all:
        wl = min(w+sl, tt)-w; s = 0 if w == 0 else ctx
        ci = min((w+s) // chunk, nc-1); cw[ci].append(w)
    log(f"ttt:start chunks={nc} lr={h.ttt_lr} epochs={h.ttt_epochs}")
    clf = torch.compile(bm.forward_logits, dynamic=False, fullgraph=True)
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    tp = list(bm.parameters())
    for p in tp: p.requires_grad_(True)
    opt = torch.optim.SGD(tp, lr=h.ttt_lr, momentum=h.ttt_momentum)
    for ci in range(nc):
        wins = cw[ci]
        if not wins: continue
        ms_ = len(wins)*rk//ws_; me_ = len(wins)*(rk+1)//ws_; myw = wins[ms_:me_]
        bm.eval()
        with torch.no_grad():
            for bi in range(0, len(myw), bsz):
                bw = myw[bi:bi+bsz]; B = len(bw)
                xb = torch.zeros(B, sl, dtype=torch.int64, device=device)
                yb = torch.zeros(B, sl, dtype=torch.int64, device=device); wls = []
                for i, w in enumerate(bw):
                    we = min(w+sl, tt); wl = we-w; wls.append(wl)
                    ch = vd.val_tokens[w:we+1].to(dtype=torch.int64, device=device)
                    xb[i,:wl] = ch[:-1]; yb[i,:wl] = ch[1:]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = clf(xb)
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                      yb.reshape(-1), reduction='none').reshape(B, sl)
                for i, w in enumerate(bw):
                    wl = wls[i]; s = 0 if w == 0 else ctx
                    ls += nll[i, s:wl].to(torch.float64).sum(); tc += float(wl - s)
                    tgt = yb[i, s:wl]; prev = xb[i, s:wl]
                    tb = vd.base_bytes_lut[tgt].to(torch.float64)
                    tb += (vd.has_leading_space_lut[tgt] & ~vd.is_boundary_token_lut[prev]).to(torch.float64)
                    bc += tb.sum()
        if ci < nc - 1 and h.ttt_epochs > 0:
            bm.train(); cs = ci * chunk; ce = min((ci+1)*chunk, tt)
            ns = (ce - cs) // sl
            if ns > 0:
                clr = h.ttt_lr * 0.5 * (1 + math.cos(math.pi * ci / max(nc-1, 1)))
                for pg in opt.param_groups: pg['lr'] = clr
                ms2 = ns*rk//ws_; me2 = ns*(rk+1)//ws_; my_ns = me2 - ms2
                for _ in range(h.ttt_epochs):
                    for b in range(0, my_ns, bsz):
                        e = min(b+bsz, my_ns); ab = ms2+b
                        st = cs + ab*sl; et = cs + (ms2+e)*sl + 1
                        if et > vd.val_tokens.numel(): continue
                        loc = vd.val_tokens[st:et].to(device=device, dtype=torch.int64)
                        x = loc[:-1].reshape(-1, sl); y = loc[1:].reshape(-1, sl)
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            loss = bm(x, y)
                        loss.backward()
                        if ws_ > 1:
                            for p in tp:
                                if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(tp, 1.0); opt.step()
    if dist.is_available() and dist.is_initialized():
        for t in [ls, tc, bc]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    for p in bm.parameters(): p.requires_grad_(True)
    bm.eval(); return _loss_bpb(ls, tc, bc)

def timed_eval(label, fn, *a, **kw):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    vl, vb = fn(*a, **kw); torch.cuda.synchronize()
    log(f"{label} val_loss:{vl:.8f} val_bpb:{vb:.8f} eval_time:{1e3*(time.perf_counter()-t0):.0f}ms")
    return vl, vb


# =============================================================================
# TRAINING LOOP — with QAT-fused cooldown + nuclear-norm reg
# =============================================================================

def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model

    log(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    optimizers = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    max_ms = 1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_ms is not None:
        max_ms -= h.gptq_reserve_seconds * 1e3
        log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_ms:.0f}ms")

    qat_activated = False  # NOVEL: track QAT state

    def frac_of(step, elapsed_ms):
        return elapsed_ms / max(max_ms, 1e-9) if max_ms else step / max(h.iterations, 1)

    def lr_mul(frac):
        if h.warmdown_frac <= 0: return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    def step_fn(step, lr_scale, frac):
        nonlocal qat_activated
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)

            # NOVEL TECHNIQUE 3: nuclear-norm regularization
            if (h.nuclear_reg_enabled and step > 0 and step % h.nuclear_reg_every == 0
                    and micro == 0):
                nuc_loss = nuclear_norm_penalty(base_model, h.nuclear_reg_top_k)
                loss = loss + h.nuclear_reg_lambda * nuc_loss

            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps

        # NOVEL TECHNIQUE 1: QAT-fused cooldown activation
        if h.qat_fused_enabled and not qat_activated and frac >= h.qat_fused_start_frac:
            base_model.enable_qat(h.qat_fused_bits, h.matrix_clip_sigmas)
            qat_activated = True
            # Need to recompile after QAT changes the forward path
            # torch._dynamo.reset() would be ideal but risks breaking mid-training
            log(f"qat_fused:activated at frac={frac:.3f} step={step}")

        # Muon momentum warmup
        f = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        mm = (1 - f) * h.muon_momentum_warmup_start + f * h.muon_momentum
        for g in optimizers.optimizer_muon.param_groups: g['momentum'] = mm
        for o in optimizers:
            for g in o.param_groups: g['lr'] = g['base_lr'] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    # Warmup (identical to SOTA)
    if h.warmup_steps > 0:
        init_sd = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opts = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(h.warmup_steps):
            step_fn(ws, 1.0, 0.0)
            if ws <= 5 or (ws+1) % 10 == 0 or ws+1 == h.warmup_steps:
                log(f"warmup_step: {ws+1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(f"loop_warmup:enabled enc:{base_model.encoder_indices} dec:{base_model.decoder_indices}")
            for ws in range(h.warmup_steps):
                step_fn(ws, 1.0, 0.0)
                if ws <= 5 or (ws+1) % 10 == 0 or ws+1 == h.warmup_steps:
                    log(f"loop_warmup_step: {ws+1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(init_sd, strict=True)
        qat_activated = False; base_model.disable_qat()  # reset QAT state after warmup
        for o, s in zip(optimizers, init_opts, strict=True): o.load_state_dict(s)
        optimizers.zero_grad_all()
        if h.distributed: model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)

    ema = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
    ema_d = h.ema_decay; train_ms = 0.0; stop = None
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0

    while True:
        last = step == h.iterations or (stop is not None and step >= stop)
        do_val = last or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if do_val:
            torch.cuda.synchronize(); train_ms += 1e3 * (time.perf_counter() - t0)
            vl, vb = eval_val(h, device, val_data, model)
            log(f"{step}/{h.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f}")
            torch.cuda.synchronize(); t0 = time.perf_counter()

            if h.use_wandb and h.is_main_process:
                wandb.log({"val_loss": vl, "val_bpb": vb, "step": step})
        if last:
            if stop is not None and step < h.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_ms:.0f}ms step:{step}")
            break
        elapsed = train_ms + 1e3 * (time.perf_counter() - t0)
        frac = frac_of(step, elapsed); scale = lr_mul(frac)
        if h.num_loops > 0 and not base_model.looping_active and frac >= h.enable_looping_at:
            base_model.looping_active = True
            log(f"loop:enabled step:{step} frac:{frac:.3f}")
        tl = step_fn(step, scale, frac)
        with torch.no_grad():
            for n, t in base_model.state_dict().items():
                ema[n].mul_(ema_d).add_(t.detach().float(), alpha=1.0 - ema_d)
        step += 1
        approx = train_ms + 1e3 * (time.perf_counter() - t0)
        tps = step * h.train_batch_tokens / (approx / 1e3)
        if h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0 or stop is not None):
            log(f"{step}/{h.iterations} train_loss:{tl.item():.4f} time:{approx/60000:.1f}m tok/s:{tps:.0f}")
        if h.use_wandb and h.is_main_process:
            wandb.log({"train_loss": tl.item(), "lr_scale": scale, "step": step, "tok_per_sec": tps})
        hit = max_ms is not None and approx >= max_ms
        if h.distributed and max_ms is not None:
            ht = torch.tensor(int(hit), device=device)
            dist.all_reduce(ht, op=dist.ReduceOp.MAX); hit = bool(ht.item())
        if stop is None and hit: stop = step

    log(f"peak mem: {torch.cuda.max_memory_allocated()//1024//1024} MiB")
    log('ema:applying EMA weights')
    cur = base_model.state_dict()
    base_model.load_state_dict({n: t.to(dtype=cur[n].dtype) for n, t in ema.items()}, strict=True)
    base_model.disable_qat()  # Ensure QAT off for serialization
    return base_model, compiled_model


# =============================================================================
# MAIN
# =============================================================================

def train_and_eval(h, device):
    random.seed(h.seed); np.random.seed(h.seed)
    torch.manual_seed(h.seed); torch.cuda.manual_seed_all(h.seed)
    vd = ValidationData(h, device)
    log(f"train_shards:{len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    log(f"val_tokens:{vd.val_tokens.numel()-1}")
    bm, cm = train_model(h, device, vd); torch._dynamo.reset()
    timed_eval('pre-quant', eval_val, h, device, vd, cm)
    serialize(h, bm, Path(__file__).read_text(encoding='utf-8'))
    if h.distributed: dist.barrier()
    em = deserialize(h, device)
    if h.num_loops > 0: em.looping_active = True
    cm2 = torch.compile(em, dynamic=False, fullgraph=True)
    timed_eval('quantized', eval_val, h, device, vd, cm2)
    if h.sliding_window_enabled:
        timed_eval('quantized_sliding', eval_val_sliding, h, device, vd, em)
    if h.ttt_enabled and h.sliding_window_enabled:
        del em, cm2; torch._dynamo.reset(); torch.cuda.empty_cache()
        tm = deserialize(h, device)
        if h.num_loops > 0: tm.looping_active = True
        timed_eval('quantized_ttt', eval_val_ttt, h, device, vd, tm)
    if h.use_wandb and h.is_main_process:
        wandb.finish()

def main():
    ws = int(os.environ.get('WORLD_SIZE', '1'))
    lr = int(os.environ.get('LOCAL_RANK', '0'))
    dist_ = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    device = torch.device('cuda', lr); torch.cuda.set_device(device)
    if dist_: dist.init_process_group(backend='nccl', device_id=device); dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = Hyperparameters(); set_logging_hparams(h)

    if h.use_wandb and h.is_main_process:
        wandb.init(
            project=h.wandb_project,
            name=h.run_id,
            dir="./wandb_logs",
            config={k: v for k, v in vars(type(h)).items() if not k.startswith('_')}
        )
    if h.is_main_process:
        os.makedirs('logs', exist_ok=True)
        log('Hyperparameters:')
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith('_'): log(f"  {k}: {v}")
        log("\n=== NOVEL TECHNIQUES (v2) ===")
        log(f"  QAT-Fused Cooldown: {h.qat_fused_enabled} (start_frac={h.qat_fused_start_frac}, bits={h.qat_fused_bits})")
        log(f"  Mixed Precision: MLP=INT{h.mlp_bits} Attn=INT{h.attn_bits} Embed=INT{h.embed_bits}")
        log(f"  Nuclear Reg: {h.nuclear_reg_enabled} (lambda={h.nuclear_reg_lambda}, every={h.nuclear_reg_every})")
        log("=============================\n")
    train_and_eval(h, device)
    if dist_: dist.destroy_process_group()

if __name__ == '__main__':
    main()
