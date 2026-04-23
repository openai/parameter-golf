"""
Experiment: Deep Equilibrium Universal Transformer (DEQ-UT)
============================================================
"Universal transformer" is explicitly on OpenAI's wish list for parameter golf.

CONCEPT:
  Instead of N sequential transformer layers, we run ONE transformer block
  repeatedly until its hidden states converge to a fixed point:

      x* = f(x*, z)    where z = input embedding, f = transformer block

  The model has 1 physical layer but effectively infinite depth at convergence.
  Parameter count is ~1/11th of a normal model, spending all 16MB on fidelity.

ARCHITECTURE:
  - Single transformer Block (attn + MLP + norms) — ~2M unquantized params
  - Anderson acceleration: 5-history window for fast convergence
  - Phantom gradients for training: treat as if we ran K=4 steps, backprop through those
  - At eval: run until ||z_{t+1} - z_t||/||z_t|| < tol (up to max_iter=20)
  - Track per-step convergence stats for analysis

WHY IT MIGHT WORK:
  - Scaling law says more compute at inference beats more params at fixed budget
  - DEQ = infinite depth from 1 physical block → extreme L(N) efficiency
  - Compatible with all quantization/compression tricks (only 1 block to store)
  - Test-Time Training becomes very cheap (only 1 block to adapt)

KNOWN RISKS:
  - Fixed-point may not converge for all inputs (use fallback after max_iter)
  - Training instability: phantom gradient approximation can diverge
  - Slower per-step than standard transformer (root-finding overhead)
  - May need careful initialization (start with identity-ish residual connections)

TO RUN (1xH100, ablation mode):
  RUN_ID=deq_smoke \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  MAX_WALLCLOCK_SECONDS=0 \
  ITERATIONS=2000 \
  DEQ_MAX_ITER=8 \
  DEQ_PHANTOM_STEPS=4 \
  torchrun --standalone --nproc_per_node=1 experiments/train_gpt_deq.py
"""

import collections, copy, glob, io, lzma, math, os
from pathlib import Path
import random, re, subprocess, sys, time, uuid
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    HAS_FLASH3 = True
except ImportError:
    HAS_FLASH3 = False


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
class Hyperparameters:
    data_dir                 = os.environ.get('DATA_DIR', './data/')
    seed                     = int(os.environ.get('SEED', 1337))
    run_id                   = os.environ.get('RUN_ID', str(uuid.uuid4()))
    iterations               = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac            = float(os.environ.get('WARMDOWN_FRAC', 0.72))
    warmup_steps             = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens       = int(os.environ.get('TRAIN_BATCH_TOKENS', 786432))
    train_seq_len            = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    train_log_every          = int(os.environ.get('TRAIN_LOG_EVERY', 200))
    max_wallclock_seconds    = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600))
    val_batch_tokens         = int(os.environ.get('VAL_BATCH_TOKENS', 524288))
    eval_seq_len             = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    val_loss_every           = int(os.environ.get('VAL_LOSS_EVERY', 500))
    sliding_window_enabled   = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))
    # Model
    vocab_size               = int(os.environ.get('VOCAB_SIZE', 8192))
    model_dim                = int(os.environ.get('MODEL_DIM', 512))
    num_heads                = int(os.environ.get('NUM_HEADS', 8))
    num_kv_heads             = int(os.environ.get('NUM_KV_HEADS', 4))
    mlp_mult                 = float(os.environ.get('MLP_MULT', 4.0))
    rope_base                = float(os.environ.get('ROPE_BASE', 1e4))
    rope_dims                = int(os.environ.get('ROPE_DIMS', 16))
    logit_softcap            = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    qk_gain_init             = float(os.environ.get('QK_GAIN_INIT', 5.5))
    # DEQ-specific
    deq_max_iter_train       = int(os.environ.get('DEQ_MAX_ITER_TRAIN', 8))
    deq_max_iter_eval        = int(os.environ.get('DEQ_MAX_ITER_EVAL', 20))
    deq_phantom_steps        = int(os.environ.get('DEQ_PHANTOM_STEPS', 4))  # steps to unroll for backprop
    deq_tol                  = float(os.environ.get('DEQ_TOL', 1e-3))       # convergence threshold
    deq_anderson_history     = int(os.environ.get('DEQ_ANDERSON_HISTORY', 5))
    deq_anderson_beta        = float(os.environ.get('DEQ_ANDERSON_BETA', 1.0))
    # Number of "warm" pre-iteration passes before starting Anderson (helps stability)
    deq_warmup_iters         = int(os.environ.get('DEQ_WARMUP_ITERS', 2))
    # Optimizer
    min_lr                   = float(os.environ.get('MIN_LR', 0.0))
    tied_embed_lr            = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std      = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    matrix_lr                = float(os.environ.get('MATRIX_LR', 0.022))
    scalar_lr                = float(os.environ.get('SCALAR_LR', 0.02))
    muon_momentum            = float(os.environ.get('MUON_MOMENTUM', 0.99))
    muon_backend_steps       = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_row_normalize       = bool(int(os.environ.get('MUON_ROW_NORMALIZE', '1')))
    muon_wd                  = float(os.environ.get('MUON_WD', 0.095))
    embed_wd                 = float(os.environ.get('EMBED_WD', 0.085))
    beta1                    = float(os.environ.get('BETA1', 0.9))
    beta2                    = float(os.environ.get('BETA2', 0.95))
    adam_eps                 = float(os.environ.get('ADAM_EPS', 1e-8))
    adam_wd                  = float(os.environ.get('ADAM_WD', 0.02))
    grad_clip_norm           = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
    ema_decay                = float(os.environ.get('EMA_DECAY', 0.9965))
    eval_stride              = int(os.environ.get('EVAL_STRIDE', 64))
    # Quantization
    compressor               = os.environ.get('COMPRESSOR', 'brotli')
    gptq_calibration_batches = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 64))
    gptq_reserve_seconds     = float(os.environ.get('GPTQ_RESERVE_SECONDS', 12.0))
    matrix_bits              = int(os.environ.get('MATRIX_BITS', 6))
    embed_bits               = int(os.environ.get('EMBED_BITS', 8))
    matrix_clip_sigmas       = float(os.environ.get('MATRIX_CLIP_SIGMAS', 12.85))
    embed_clip_sigmas        = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))
    # Distributed
    distributed      = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank             = int(os.environ.get('RANK', '0'))
    world_size       = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank       = int(os.environ.get('LOCAL_RANK', '0'))
    is_main_process  = rank == 0
    grad_accum_steps = 8 // world_size
    # Derived
    datasets_dir         = os.path.join(data_dir, 'datasets', f'fineweb10B_sp{vocab_size}')
    train_files          = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files            = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path       = os.path.join(data_dir, 'tokenizers', f'fineweb_{vocab_size}_bpe.model')
    logfile              = f'logs/{run_id}.txt'
    model_path           = 'final_model.pt'
    quantized_model_path = 'final_model.int6.ptz'


_logger_hparams = None


def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h


def log(msg, console=True):
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, 'a', encoding='utf-8') as f:
                print(msg, file=f)


# ---------------------------------------------------------------------------
# Data loading (identical to main submission)
# ---------------------------------------------------------------------------
class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(f"VOCAB_SIZE mismatch")
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = \
            build_sentencepiece_luts(self.sp, h.vocab_size, device)


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert sp.piece_to_id('▁') != sp.unk_id()
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
        if piece.startswith('▁'):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode('utf-8'))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))


_SHARD_HEADER_BYTES = 256 * np.dtype('<i4').itemsize
_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}


def _read_num_tokens(file):
    key = str(file)
    if key in _SHARD_NTOKENS_CACHE:
        return _SHARD_NTOKENS_CACHE[key]
    header = np.fromfile(file, dtype='<i4', count=256)
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


def _get_shard_memmap(file):
    key = str(file)
    if key in _MMAP_CACHE:
        return _MMAP_CACHE[key]
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode='r', dtype='<u2', offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


def load_data_shard(file):
    header_bytes = 256 * np.dtype('<i4').itemsize
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype='<u2', count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    return tokens[:usable + 1]


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
                total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[start_ind:start_ind + self.seq_len + 1], dtype=np.int64))
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=2048, rope_dims=0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        self.base = base
        self.train_seq_len = train_seq_len
        inv_freq = 1.0 / base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cache = {}

    def forward(self, seq_len, device, dtype):
        key = (seq_len, device, dtype)
        if key not in self._cache:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cache[key] = (freqs.cos()[None, :, None, :].to(dtype),
                                freqs.sin()[None, :, None, :].to(dtype))
        return self._cache[key]


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        return torch.cat((torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1), x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, rope_dims, qk_gain_init, train_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = rope_dims
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len, rope_dims=rope_dims)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if HAS_FLASH3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            # Fallback to sdpa
            q = q.transpose(1, 2)
            k = k.transpose(1, 2).expand(-1, self.num_heads, -1, -1)
            v = v.transpose(1, 2).expand(-1, self.num_heads, -1, -1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
        return self.proj(y.reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class UniversalBlock(nn.Module):
    """Single transformer block used for all iterations of the DEQ loop."""
    def __init__(self, h):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(h.model_dim, h.num_heads, h.num_kv_heads,
                                        h.rope_base, h.rope_dims, h.qk_gain_init, h.train_seq_len)
        self.mlp = MLP(h.model_dim, h.mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(h.model_dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(h.model_dim, dtype=torch.float32))
        # Learnable mixing with input embedding (DEQ injection)
        self.input_gate = nn.Parameter(torch.zeros(h.model_dim, dtype=torch.float32))

    def forward(self, z, z0):
        """
        z:  current fixed-point iterate [B, T, D]
        z0: input embedding (injected at every step to maintain conditioning)
        """
        # Condition on input at every iteration via learned gating
        g = torch.sigmoid(self.input_gate.to(dtype=z.dtype))[None, None, :]
        x = z + g * z0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Anderson Acceleration for fixed-point finding
# ---------------------------------------------------------------------------
def anderson_step(f_history, x_history, beta=1.0):
    """
    Given history of function values f(x_i) and iterates x_i,
    compute the Anderson mixing step to accelerate convergence.

    Returns the next iterate.
    """
    m = len(f_history)
    if m == 1:
        # No history to mix; just return f(x)
        return f_history[0]

    # F matrix: columns are residuals f(x_i) - x_i
    F = torch.stack([f - x for f, x in zip(f_history, x_history)], dim=-1)  # [B*T*D, m]
    # Solve least squares: min ||F @ alpha||^2 s.t. sum(alpha) = 1
    B, T, D = f_history[0].shape
    F_flat = F.reshape(B * T * D, m)
    # Normal equations: (F^T F) alpha = 1 / (1^T (F^T F)^-1 1) * (F^T F)^-1 1
    try:
        FtF = F_flat.T @ F_flat + 1e-8 * torch.eye(m, device=F_flat.device, dtype=F_flat.dtype)
        ones = torch.ones(m, 1, device=F_flat.device, dtype=F_flat.dtype)
        alpha = torch.linalg.solve(FtF, ones)
        alpha = alpha / alpha.sum()
    except Exception:
        # Fallback to pure iteration if solve fails
        return f_history[-1]
    # Mix: beta * sum(alpha_i * f(x_i)) + (1-beta) * sum(alpha_i * x_i)
    x_stack = torch.stack(x_history, dim=-1)  # [B, T, D, m]
    f_stack = torch.stack(f_history, dim=-1)
    alpha_t = alpha.reshape(1, 1, 1, m)
    x_mix = (f_stack * alpha_t).sum(dim=-1)
    return beta * x_mix + (1 - beta) * (x_stack * alpha_t).sum(dim=-1)


# ---------------------------------------------------------------------------
# DEQ GPT Model
# ---------------------------------------------------------------------------
class DEQGPT(nn.Module):
    """
    Deep Equilibrium Universal Transformer.
    One physical block run until fixed-point convergence.

    Virtual layer count at inference: as deep as needed for convergence.
    Total parameter count: ~2M (1 block) + embeddings.
    """
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.block = UniversalBlock(h)
        self.final_norm = RMSNorm()
        # Lightweight encoder/decoder projections (4 layers each) to give the
        # DEQ loop a better inductive starting point and output structure
        self.pre_deq = nn.ModuleList([
            nn.Sequential(
                RMSNorm(),
                CastedLinear(h.model_dim, h.model_dim, bias=False)
            ) for _ in range(2)
        ])
        self.post_deq = nn.ModuleList([
            nn.Sequential(
                RMSNorm(),
                CastedLinear(h.model_dim, h.model_dim, bias=False)
            ) for _ in range(2)
        ])
        self.max_iter_train = h.deq_max_iter_train
        self.max_iter_eval = h.deq_max_iter_eval
        self.phantom_steps = h.deq_phantom_steps
        self.tol = h.deq_tol
        self.anderson_history = h.deq_anderson_history
        self.anderson_beta = h.deq_anderson_beta
        self.warmup_iters = h.deq_warmup_iters
        # Track convergence stats
        self.register_buffer('_iter_count', torch.zeros(1), persistent=False)
        self.register_buffer('_iter_total', torch.zeros(1), persistent=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, '_zero_init', False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64:
                    nn.init.orthogonal_(module.weight, gain=0.5)

    def _run_block(self, z, z0):
        """Run one iteration of the universal block."""
        return self.block(z, z0)

    @torch.no_grad()
    def _fixed_point_eval(self, z0):
        """
        Anderson-accelerated fixed-point finding at eval time.
        Runs until convergence or max_iter_eval.
        """
        z = z0.clone()
        f_history = []
        x_history = []
        iters_run = 0
        for i in range(self.max_iter_eval):
            z_new = self._run_block(z, z0)
            f_history.append(z_new)
            x_history.append(z)
            if len(f_history) > self.anderson_history:
                f_history.pop(0)
                x_history.pop(0)
            if len(f_history) > self.warmup_iters:
                z_next = anderson_step(f_history, x_history, beta=self.anderson_beta)
            else:
                z_next = z_new
            # Check convergence
            rel_change = (z_next - z).norm() / (z.norm() + 1e-8)
            z = z_next
            iters_run = i + 1
            if rel_change < self.tol:
                break
        self._iter_count += iters_run
        self._iter_total += 1.0
        return z

    def _phantom_grad(self, z0):
        """
        Training forward: phantom gradient approach.
        Run phantom_steps iterations with gradient tracking, as if that were
        the full fixed-point. This approximates the implicit gradient.
        """
        z = z0.detach().clone()
        # Warm start: a few no-grad iterations to get close to fixed point
        with torch.no_grad():
            for _ in range(max(0, self.max_iter_train - self.phantom_steps)):
                z = self._run_block(z, z0)
        # Final phantom_steps with gradient
        for _ in range(self.phantom_steps):
            z = self._run_block(z, z0)
        return z

    def forward_logits(self, input_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        # Pre-DEQ feature extraction
        z0 = x
        for layer in self.pre_deq:
            norm_out, proj = layer
            z0 = z0 + proj(norm_out(z0))
        # DEQ loop
        if self.training:
            z = self._phantom_grad(z0)
        else:
            z = self._fixed_point_eval(z0)
        # Post-DEQ refinement
        for layer in self.post_deq:
            norm_out, proj = layer
            z = z + proj(norm_out(z))
        z = self.final_norm(z)
        logits = F.linear(z, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1), reduction='mean')

    def log_convergence_stats(self):
        if self._iter_total > 0:
            avg_iters = (self._iter_count / self._iter_total).item()
            log(f"deq_convergence: avg_iters={avg_iters:.2f} over {int(self._iter_total.item())} calls")
        self._iter_count.zero_()
        self._iter_total.zero_()


# ---------------------------------------------------------------------------
# Optimizer — same MuonEq-R as main submission
# ---------------------------------------------------------------------------
@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
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
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True,
                 weight_decay=0.0, row_normalize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                     nesterov=nesterov, weight_decay=weight_decay,
                                     row_normalize=row_normalize))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        rank = dist.get_rank() if is_dist else 0
        for group in self.param_groups:
            params = group['params']
            if not params:
                continue
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=group['momentum'])
                    if group.get('row_normalize', False):
                        row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                        g = g / row_norms.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if is_dist:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                wd = group.get('weight_decay', 0.0)
                if wd > 0.0:
                    p.data.mul_(1.0 - group['lr'] * wd)
                p.add_(updates_flat[curr:curr + p.numel()].view_as(p).to(p.dtype), alpha=-group['lr'])
                curr += p.numel()
        return loss


CONTROL_PATTERNS = ('attn_scale', 'mlp_scale', 'q_gain', 'input_gate')


class DEQOptimizers:
    def __init__(self, h, model):
        named_params = list(model.named_parameters())
        matrix_params = [p for name, p in named_params
                         if p.ndim == 2 and not any(c in name for c in CONTROL_PATTERNS)]
        scalar_params = [p for name, p in named_params
                         if p.ndim < 2 or any(c in name for c in CONTROL_PATTERNS)]
        scalar_params = [p for p in scalar_params if p is not model.tok_emb.weight]
        self.optimizer_tok = torch.optim.AdamW(
            [{'params': [model.tok_emb.weight], 'lr': h.tied_embed_lr, 'base_lr': h.tied_embed_lr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True)
        self.optimizer_muon = Muon(matrix_params, lr=h.matrix_lr, momentum=h.muon_momentum,
                                   backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd,
                                   row_normalize=h.muon_row_normalize)
        for group in self.optimizer_muon.param_groups:
            group['base_lr'] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{'params': scalar_params, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self):
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_batch)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  y_batch.reshape(-1), reduction='none').reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                loss_sum += nll[i, s:wlen].to(torch.float64).sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.log_convergence_stats()
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms")
    return val_loss, val_bpb


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(h, device, val_data):
    base_model = DEQGPT(h).to(device).bfloat16()
    # fp32 for control tensors
    for name, param in base_model.named_parameters():
        if param.ndim < 2 or any(c in name for c in CONTROL_PATTERNS):
            param.data = param.data.float()
    compiled_model = torch.compile(base_model, dynamic=False)
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model
    total_params = sum(p.numel() for p in base_model.parameters())
    log(f"model_params: {total_params} ({total_params / 1e6:.2f}M)")
    log(f"deq: max_iter_train={h.deq_max_iter_train} phantom_steps={h.deq_phantom_steps} "
        f"max_iter_eval={h.deq_max_iter_eval} tol={h.deq_tol}")
    optimizers = DEQOptimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    max_wallclock_ms = 1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1e3

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-9)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = (step == h.iterations or (stop_after_step is not None and step >= stop_after_step))
        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = timed_eval('val_sliding', eval_val_sliding, h, device, val_data, base_model)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            break
        elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)

        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(h.ema_decay).add_(t.detach().float(), alpha=1.0 - h.ema_decay)
        step += 1
        approx_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        should_log = h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0)
        if should_log:
            tok_per_sec = step * h.train_batch_tokens / (approx_ms / 1e3)
            log(f"{step}/{h.iterations} train_loss:{train_loss.item():.4f} "
                f"time:{approx_ms / 60000:.1f}m tok/s:{tok_per_sec:.0f}")
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log('ema: applying EMA weights')
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    return base_model


def main():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs('logs', exist_ok=True)
        log(f"=== DEQ Universal Transformer ===")
        log(f"DEQ config: max_iter_train={h.deq_max_iter_train} phantom={h.deq_phantom_steps} "
            f"max_iter_eval={h.deq_max_iter_eval} anderson_history={h.deq_anderson_history}")
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    train_model(h, device, val_data)
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
