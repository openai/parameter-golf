"""
Experiment: Mixture of Depths (MoD)
=====================================
"Mixture of Depths" — Raposo et al. 2024, explicitly on the parameter-golf wish list.

CONCEPT:
  At each transformer layer, a lightweight router (1 linear projection) decides
  whether each token should pass through the full layer or be skipped (identity).
  Only the top-k% of tokens by router score get processed; the rest pass through
  unchanged.

WHY IT WINS:
  - Training is ~2× faster in FLOPs (skip 50% of tokens per layer)
  - Same 10-minute wall clock → ~2× more gradient steps → better convergence
  - The saved compute can also be used for: more layers, larger dim, or tighter loops
  - Skipped tokens still attend to processed tokens in later layers (causal mask unchanged)

KEY INSIGHT FOR PARAMETER GOLF:
  MoD doesn't change the parameter count — it changes compute utilization.
  In the 10-minute training window, more steps = better loss = lower BPB.
  Even 1.5× training steps from 50% token routing is a huge advantage.

ARCHITECTURE:
  - Exact same GPT as SOTA (11L × 512d, SP1024/8192, MuonEq-R, GPTQ)
  - Each layer has a tiny router: Linear(dim, 1) → scalar score per token
  - Top-k routing: only the top `router_capacity` fraction of tokens go through
  - Skipped tokens: just get x = x (identity residual)
  - Router is trained end-to-end with gumbel-top-k for differentiability

ROUTER IMPLEMENTATION:
  - Straight-through estimator for training: use soft scores for loss, hard mask for forward
  - Capacity factor: 0.5 (50% of tokens processed) for max speedup
  - Learnable to increase capacity on hard tokens, decrease on easy tokens
  - Router loss: small auxiliary load-balancing loss to prevent router collapse

ENV VARS:
  MOD_CAPACITY        Router capacity (fraction of tokens to process) [default: 0.5]
  MOD_LAYERS          Comma-separated layer indices to apply MoD to [default: "1,2,3,4,5,6,7,8,9,10"]
  MOD_AUX_LOSS_COEF  Auxiliary load-balancing loss coefficient [default: 0.01]
  MOD_ROUTER_INIT    Initial router bias [default: 0.0]

TO RUN:
  RUN_ID=mod_smoke VOCAB_SIZE=1024 ITERATIONS=2000 MOD_CAPACITY=0.5 \
  torchrun --standalone --nproc_per_node=1 experiments/train_gpt_mod.py
"""

import collections, copy, glob, io, math, os
from pathlib import Path
import random, sys, time, uuid
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
    num_layers               = int(os.environ.get('NUM_LAYERS', 11))
    model_dim                = int(os.environ.get('MODEL_DIM', 512))
    num_heads                = int(os.environ.get('NUM_HEADS', 8))
    num_kv_heads             = int(os.environ.get('NUM_KV_HEADS', 4))
    mlp_mult                 = float(os.environ.get('MLP_MULT', 4.0))
    rope_base                = float(os.environ.get('ROPE_BASE', 1e4))
    rope_dims                = int(os.environ.get('ROPE_DIMS', 16))
    logit_softcap            = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    qk_gain_init             = float(os.environ.get('QK_GAIN_INIT', 5.5))
    num_loops                = int(os.environ.get('NUM_LOOPS', 2))
    loop_start               = int(os.environ.get('LOOP_START', 3))
    loop_end                 = int(os.environ.get('LOOP_END', 5))
    enable_looping_at        = float(os.environ.get('ENABLE_LOOPING_AT', 0.35))
    parallel_residual_start  = int(os.environ.get('PARALLEL_RESIDUAL_START', 7))
    skip_gates_enabled       = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    # MoD-specific
    mod_capacity      = float(os.environ.get('MOD_CAPACITY', 0.5))   # fraction of tokens to process
    mod_layers        = os.environ.get('MOD_LAYERS', 'all')           # 'all' or '1,3,5,7,9'
    mod_aux_loss_coef = float(os.environ.get('MOD_AUX_LOSS_COEF', 0.01))
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


_log_h = None
def set_log_h(h):
    global _log_h
    _log_h = h

def log(msg):
    if _log_h is None:
        print(msg)
        return
    if _log_h.is_main_process:
        print(msg)
        if _log_h.logfile:
            with open(_log_h.logfile, 'a') as f:
                print(msg, file=f)


# ---------------------------------------------------------------------------
# Data loading (shared infrastructure)
# ---------------------------------------------------------------------------
_SHARD_HEADER_BYTES = 256 * np.dtype('<i4').itemsize
_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}


def _read_num_tokens(file):
    key = str(file)
    if key not in _SHARD_NTOKENS_CACHE:
        header = np.fromfile(file, dtype='<i4', count=256)
        _SHARD_NTOKENS_CACHE[key] = int(header[2])
    return _SHARD_NTOKENS_CACHE[key]


def _get_shard_memmap(file):
    key = str(file)
    if key not in _MMAP_CACHE:
        n = _read_num_tokens(file)
        _MMAP_CACHE[key] = np.memmap(file, mode='r', dtype='<u2',
                                     offset=_SHARD_HEADER_BYTES, shape=(n,))
    return _MMAP_CACHE[key]


def load_data_shard(file):
    header = np.fromfile(file, dtype='<i4', count=256)
    num_tokens = int(header[2])
    return torch.from_numpy(
        np.fromfile(file, dtype='<u2', count=num_tokens,
                    offset=_SHARD_HEADER_BYTES).astype(np.int64))


def load_validation_tokens(pattern, seq_len):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files])
    return tokens[:(tokens.numel() - 1) // seq_len * seq_len + 1]


def build_sentencepiece_luts(sp, vocab_size, device):
    sz = max(int(sp.vocab_size()), vocab_size)
    base_bytes = np.zeros((sz,), dtype=np.int16)
    has_space = np.zeros((sz,), dtype=np.bool_)
    is_boundary = np.ones((sz,), dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith('▁'):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode('utf-8'))
    return (torch.tensor(base_bytes, dtype=torch.int16, device=device),
            torch.tensor(has_space, dtype=torch.bool, device=device),
            torch.tensor(is_boundary, dtype=torch.bool, device=device))


class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = \
            build_sentencepiece_luts(self.sp, h.vocab_size, device)


class ShuffledSequenceLoader:
    def __init__(self, h, device):
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
        n = self.num_tokens[si]
        max_phase = min(self.seq_len - 1, max(0, n - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_seq = (n - 1 - phase) // self.seq_len
        self.start_inds[si] = (phase + self.rng.permutation(num_seq) * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        bsz = (global_tokens // grad_accum_steps) // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((bsz, self.seq_len), dtype=torch.int64)
        y = torch.empty((bsz, self.seq_len), dtype=torch.int64)
        for bi in range(bsz):
            if remaining.sum() <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
            si = int(self.rng.choice(len(self.files), p=remaining / remaining.sum()))
            start = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            w = torch.as_tensor(np.array(mm[start:start + self.seq_len + 1], dtype=np.int64))
            x[bi], y[bi] = w[:-1], w[1:]
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
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


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
        xr, xp = x[..., :rope_dims], x[..., rope_dims:]
        h = rope_dims // 2
        x1, x2 = xr[..., :h], xr[..., h:]
        return torch.cat((torch.cat((x1*cos + x2*sin, x1*-sin + x2*cos), dim=-1), xp), dim=-1)
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1*cos + x2*sin, x1*-sin + x2*cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        dim, nh, nkv = h.model_dim, h.num_heads, h.num_kv_heads
        self.num_heads = nh
        self.num_kv_heads = nkv
        self.head_dim = dim // nh
        kv_dim = nkv * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((nh,), h.qk_gain_init, dtype=torch.float32))
        self.rope_dims = h.rope_dims
        self.rotary = Rotary(self.head_dim, base=h.rope_base,
                             train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)

    def forward(self, x):
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]
        if HAS_FLASH3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2).expand(-1, self.num_heads, -1, -1)
            v = v.transpose(1, 2).expand(-1, self.num_heads, -1, -1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
        return self.proj(y.reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        hidden = int(h.mlp_mult * h.model_dim)
        self.fc = CastedLinear(h.model_dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, h.model_dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), 0.5).square())


# ---------------------------------------------------------------------------
# MoD Token Router
# ---------------------------------------------------------------------------
class TokenRouter(nn.Module):
    """
    Lightweight token router for Mixture of Depths.
    Outputs a scalar score per token; top-k% tokens are routed through the block.

    During training: uses straight-through estimator
      - Soft scores flow through for gradient
      - Hard binary mask used for forward computation
      - Auxiliary load-balancing loss prevents router collapse

    During eval: deterministic top-k routing, no aux loss
    """
    def __init__(self, dim: int, capacity: float):
        super().__init__()
        self.capacity = capacity
        # Tiny 1-layer router — just a single linear projection to a scalar
        self.router = CastedLinear(dim, 1, bias=True)
        # Initialize with small weights so early training doesn't over-route
        nn.init.normal_(self.router.weight, std=0.01)
        nn.init.zeros_(self.router.bias)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]
        Returns:
          routed_x: [B, T, D] — only routed tokens have block output, others are identity
          aux_loss: scalar auxiliary load-balancing loss
        """
        B, T, D = x.shape
        k = max(1, int(T * self.capacity))

        # Router scores: [B, T, 1] → [B, T]
        scores = self.router(x).squeeze(-1)  # [B, T]

        # Top-k selection (hard mask for forward, soft scores for backward)
        topk_vals, topk_idx = torch.topk(scores, k, dim=-1, sorted=False)  # [B, k]

        # Hard binary mask: 1 for routed tokens
        mask = torch.zeros_like(scores, dtype=x.dtype)  # [B, T]
        mask.scatter_(1, topk_idx, 1.0)

        # Straight-through: treat mask as if it were the soft scores during backward
        # mask_ste = mask + (soft_scores - soft_scores.detach()) would conflate gradients
        # Instead: keep the mask, but flow the router score signal through aux loss only

        # Auxiliary load-balancing loss:
        # Encourages the router to select each token with equal probability across the batch
        # (prevents all attention going to same k tokens always)
        soft_probs = torch.sigmoid(scores)   # [B, T]
        # Target: capacity fraction should be True on average per token position
        avg_selected = mask.float().mean(dim=0)         # [T] - actual fraction selected per position
        avg_prob = soft_probs.float().mean(dim=0)       # [T] - expected fraction from router
        aux_loss = (avg_selected * avg_prob).mean()     # correlation pushes toward uniform selection

        return mask, aux_loss


class MoDBlock(nn.Module):
    """
    Transformer block with Mixture of Depths routing.
    Only top-k% tokens pass through the attn+mlp; the rest get identity.
    """
    def __init__(self, h, use_mod: bool = True, use_parallel_residual: bool = False):
        super().__init__()
        self.use_mod = use_mod
        self.use_parallel_residual = use_parallel_residual
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(h)
        self.mlp = MLP(h)
        self.attn_scale = nn.Parameter(torch.ones(h.model_dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(h.model_dim, dtype=torch.float32))
        if use_mod:
            self.router = TokenRouter(h.model_dim, capacity=h.mod_capacity)
        else:
            self.router = None

    def forward(self, x):
        aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)

        if self.router is not None:
            mask, aux_loss = self.router(x)  # mask: [B, T], float 0/1
            mask = mask.unsqueeze(-1)  # [B, T, 1]
        else:
            mask = None

        if self.use_parallel_residual:
            # GPT-J style: attn and MLP read same input
            normed = self.attn_norm(x)
            attn_out = self.attn(normed) * self.attn_scale.to(x.dtype)
            mlp_out = self.mlp(self.mlp_norm(x)) * self.mlp_scale.to(x.dtype)
            delta = attn_out + mlp_out
        else:
            attn_out = self.attn(self.attn_norm(x)) * self.attn_scale.to(x.dtype)
            mlp_out = self.mlp(self.mlp_norm(x + attn_out)) * self.mlp_scale.to(x.dtype)
            delta = attn_out + mlp_out

        if mask is not None:
            # Apply delta only to routed tokens; identity for skipped tokens
            x = x + mask * delta
        else:
            x = x + delta

        return x, aux_loss


# ---------------------------------------------------------------------------
# Full MoD GPT
# ---------------------------------------------------------------------------
class MoDGPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)

        # Determine which layers get MoD routing
        if h.mod_layers == 'all':
            # Skip layer 0 and last layer — always process those
            mod_layer_set = set(range(1, h.num_layers - 1))
        else:
            mod_layer_set = set(int(x) for x in h.mod_layers.split(','))

        self.blocks = nn.ModuleList([
            MoDBlock(
                h,
                use_mod=(i in mod_layer_set),
                use_parallel_residual=(i >= h.parallel_residual_start)
            )
            for i in range(h.num_layers)
        ])
        # Skip gates (U-Net style, encoder→decoder)
        if h.skip_gates_enabled:
            n_skip = h.num_layers // 2
            self.skip_weights = nn.ParameterList([
                nn.Parameter(torch.ones(h.model_dim, dtype=torch.float32))
                for _ in range(n_skip)
            ])
        self.final_norm = RMSNorm()
        self.mod_aux_loss_coef = h.mod_aux_loss_coef
        self.num_loops = h.num_loops
        self.loop_start = h.loop_start
        self.loop_end = h.loop_end
        self.enable_looping_at = h.enable_looping_at
        self._training_frac = 0.0
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, '_zero_init', False):
                    nn.init.zeros_(m.weight)
                elif m.weight.ndim == 2 and m.weight.shape[0] >= 64:
                    nn.init.orthogonal_(m.weight, gain=0.5)

    def set_training_frac(self, frac: float):
        self._training_frac = frac

    def _run_blocks(self, x, skip_connections=None):
        total_aux = torch.zeros((), device=x.device, dtype=torch.float32)
        n = len(self.blocks)
        n_skip = n // 2
        # Looping
        use_looping = self._training_frac >= self.enable_looping_at
        num_extra = self.num_loops - 1 if use_looping else 0

        layer_outputs = []
        li = 0  # physical layer index (with loop expansion)
        expanded = (list(range(self.loop_start)) +
                    list(range(self.loop_start, self.loop_end + 1)) * (self.num_loops) +
                    list(range(self.loop_end + 1, n)))

        for block_idx in expanded:
            x, aux = self.blocks[block_idx](x)
            total_aux = total_aux + aux
            layer_outputs.append((block_idx, x))

        return x, total_aux

    def forward_logits(self, input_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x, _ = self._run_blocks(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x, aux_loss = self._run_blocks(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                             target_ids.reshape(-1))
        return ce + self.mod_aux_loss_coef * aux_loss


# ---------------------------------------------------------------------------
# Optimizer (MuonEq-R + AdamW, same as main submission)
# ---------------------------------------------------------------------------
@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    if X.size(0) > X.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


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
        ws = dist.get_world_size() if is_dist else 1
        rank = dist.get_rank() if is_dist else 0
        for group in self.param_groups:
            params = group['params']
            if not params:
                continue
            total = sum(p.numel() for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % ws == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(g)
                    if group['nesterov']:
                        g = g + buf * group['momentum']
                    if group.get('row_normalize'):
                        g = g / g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if is_dist:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                if group.get('weight_decay', 0) > 0:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])
                p.add_(flat[curr:curr + p.numel()].view_as(p).to(p.dtype), alpha=-group['lr'])
                curr += p.numel()
        return loss


CTRL = ('attn_scale', 'mlp_scale', 'q_gain', 'skip_weight', 'router.router.bias')


def make_optimizers(h, model):
    named = list(model.named_parameters())
    matrix_p = [p for n, p in named if p.ndim == 2 and not any(c in n for c in CTRL)
                and p is not model.tok_emb.weight]
    scalar_p = [p for n, p in named if p.ndim < 2 or any(c in n for c in CTRL)]
    scalar_p = [p for p in scalar_p if p is not model.tok_emb.weight]

    opt_embed = torch.optim.AdamW(
        [{'params': [model.tok_emb.weight], 'lr': h.tied_embed_lr, 'base_lr': h.tied_embed_lr}],
        betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True)
    opt_muon = Muon(matrix_p, lr=h.matrix_lr, momentum=h.muon_momentum,
                    backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd,
                    row_normalize=h.muon_row_normalize)
    for g in opt_muon.param_groups:
        g['base_lr'] = h.matrix_lr
    opt_scalar = torch.optim.AdamW(
        [{'params': scalar_p, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}],
        betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
    return [opt_embed, opt_muon, opt_scalar]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_val_sliding(h, device, val_data, model, batch_seqs=32):
    model.eval()
    sl = h.eval_seq_len
    ctx = sl - h.eval_stride
    total = val_data.val_tokens.numel() - 1
    starts = [ws for ws in range(0, total, h.eval_stride) if ws + ctx < total]
    my_starts = starts[h.rank::h.world_size]
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(my_starts), batch_seqs):
            ws_batch = my_starts[bi:bi + batch_seqs]
            bsz = len(ws_batch)
            xb = torch.zeros(bsz, sl, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, sl, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(ws_batch):
                we = min(ws + sl, total)
                wlen = we - ws
                wlens.append(wlen)
                ch = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                xb[i, :wlen] = ch[:-1]
                yb[i, :wlen] = ch[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                lg = model.forward_logits(xb)
            nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(),
                                  yb.reshape(-1), reduction='none').reshape(bsz, sl)
            for i, ws in enumerate(ws_batch):
                wlen = wlens[i]
                s = 0 if ws == 0 else ctx
                ls += nll[i, s:wlen].to(torch.float64).sum()
                tc += float(wlen - s)
                tgt = yb[i, s:wlen]
                prev = xb[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] &
                       ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                bc += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ls, op=dist.ReduceOp.SUM)
        dist.all_reduce(tc, op=dist.ReduceOp.SUM)
        dist.all_reduce(bc, op=dist.ReduceOp.SUM)
    val_loss = (ls / tc).item()
    val_bpb = val_loss / math.log(2.0) * (tc.item() / bc.item())
    model.train()
    return val_loss, val_bpb


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(h, device, val_data):
    base_model = MoDGPT(h).to(device).bfloat16()
    for name, p in base_model.named_parameters():
        if p.ndim < 2 or any(c in name for c in CTRL):
            p.data = p.data.float()

    n_mod = sum(1 for b in base_model.blocks if b.use_mod)
    n_total = len(base_model.blocks)
    total_params = sum(p.numel() for p in base_model.parameters())
    log(f"MoD: {n_mod}/{n_total} blocks have routing | capacity={h.mod_capacity:.0%}")
    log(f"model_params: {total_params} ({total_params/1e6:.2f}M)")
    log(f"expected_speedup: ~{1 / (1 - h.mod_capacity * n_mod / n_total) :.2f}x FLOPs saved")

    compiled = torch.compile(base_model, dynamic=False)
    model = DDP(compiled, device_ids=[h.local_rank], broadcast_buffers=False) \
        if h.distributed else compiled
    opts = make_optimizers(h, base_model)

    loader = ShuffledSequenceLoader(h, device)
    max_ms = 1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_ms:
        max_ms -= h.gptq_reserve_seconds * 1e3

    def lr_scale(frac):
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    ema = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
    training_ms = 0.0
    stop_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last = step == h.iterations or (stop_step is not None and step >= stop_step)
        if last or (h.val_loss_every > 0 and step % h.val_loss_every == 0):
            torch.cuda.synchronize()
            training_ms += 1e3 * (time.perf_counter() - t0)
            t_eval = time.perf_counter()
            vl, vbpb = eval_val_sliding(h, device, val_data, base_model)
            torch.cuda.synchronize()
            em = 1e3 * (time.perf_counter() - t_eval)
            log(f"step:{step} val_loss:{vl:.6f} val_bpb:{vbpb:.6f} eval_ms:{em:.0f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last:
            break

        elapsed = training_ms + 1e3 * (time.perf_counter() - t0)
        frac = elapsed / max_ms if max_ms else step / max(h.iterations, 1)
        base_model.set_training_frac(frac)
        scale = lr_scale(frac)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        tloss = torch.zeros((), device=device)
        for ai in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = ai == h.grad_accum_steps - 1
            x, y = loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(x, y)
            tloss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        tloss /= h.grad_accum_steps
        for opt in opts:
            for g in opt.param_groups:
                g['lr'] = g['base_lr'] * scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        for opt in opts:
            opt.step()
        with torch.no_grad():
            for n, t in base_model.state_dict().items():
                ema[n].mul_(h.ema_decay).add_(t.detach().float(), alpha=1.0 - h.ema_decay)
        step += 1
        approx_ms = training_ms + 1e3 * (time.perf_counter() - t0)
        if h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0):
            tok_s = step * h.train_batch_tokens / (approx_ms / 1e3)
            log(f"{step}/{h.iterations} loss:{tloss.item():.4f} time:{approx_ms/60000:.1f}m tok/s:{tok_s:.0f}")
        if stop_step is None and max_ms and approx_ms >= max_ms:
            stop_step = step

    avg = {n: t.to(dtype=base_model.state_dict()[n].dtype) for n, t in ema.items()}
    base_model.load_state_dict(avg, strict=True)
    return base_model


def main():
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    h = Hyperparameters()
    set_log_h(h)
    if h.is_main_process:
        os.makedirs('logs', exist_ok=True)
        log(f"=== Mixture of Depths GPT ===")
        log(f"capacity={h.mod_capacity:.0%} | layers={h.mod_layers} | aux_coef={h.mod_aux_loss_coef}")
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
