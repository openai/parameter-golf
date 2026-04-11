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
    _HAS_FA3 = True
except ImportError:
    flash_attn_3_func = None
    _HAS_FA3 = False


def _sdpa_attn(q, k, v, causal=True):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if k.size(1) < q.size(1):
        r = q.size(1) // k.size(1)
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)


# ---------------------------------------------------------------------------
# Test mode overrides
# ---------------------------------------------------------------------------
_TEST_MODE = os.environ.get("TEST_MODE", "full").lower()
_TEST_OVERRIDES = {
    "smoke": {
        "NUM_LAYERS": "5", "MODEL_DIM": "256", "NUM_HEADS": "4", "NUM_KV_HEADS": "2",
        "EMBEDDING_DIM": "256", "ITERATIONS": "100", "WARMUP_STEPS": "5",
        "WARMDOWN_FRAC": "0.667", "TRAIN_BATCH_TOKENS": str(512 * 16),
        "VAL_BATCH_TOKENS": str(512 * 16), "TRAIN_SEQ_LEN": "512", "EVAL_SEQ_LEN": "512",
        "VAL_LOSS_EVERY": "50", "TRAIN_LOG_EVERY": "10", "MAX_WALLCLOCK_SECONDS": "300",
        "GPTQ_CALIBRATION_BATCHES": "4", "GPTQ_RESERVE_SECONDS": "5.0", "EMA_DECAY": "0.95",
        "NUM_LOOPS": "0", "LOOP_START": "0", "LOOP_END": "0", "XSA_LAST_N": "5",
        "SLIDING_WINDOW_ENABLED": "0",
    },
    "small": {
        "NUM_LAYERS": "7", "MODEL_DIM": "384", "NUM_HEADS": "6", "NUM_KV_HEADS": "3",
        "EMBEDDING_DIM": "384", "ITERATIONS": "500", "WARMUP_STEPS": "10",
        "WARMDOWN_FRAC": "0.667", "TRAIN_BATCH_TOKENS": str(1024 * 16),
        "VAL_BATCH_TOKENS": str(1024 * 16), "TRAIN_SEQ_LEN": "1024", "EVAL_SEQ_LEN": "1024",
        "VAL_LOSS_EVERY": "250", "TRAIN_LOG_EVERY": "50", "MAX_WALLCLOCK_SECONDS": "600",
        "GPTQ_CALIBRATION_BATCHES": "8", "GPTQ_RESERVE_SECONDS": "8.0", "EMA_DECAY": "0.99",
        "NUM_LOOPS": "0", "LOOP_START": "0", "LOOP_END": "0", "XSA_LAST_N": "7",
        "SLIDING_WINDOW_ENABLED": "0",
    },
    "medium": {
        "NUM_LAYERS": "11", "MODEL_DIM": "512", "NUM_HEADS": "8", "NUM_KV_HEADS": "4",
        "EMBEDDING_DIM": "512", "ITERATIONS": "2000", "WARMUP_STEPS": "15",
        "WARMDOWN_FRAC": "0.667", "TRAIN_BATCH_TOKENS": str(2048 * 8),
        "VAL_BATCH_TOKENS": str(2048 * 8), "TRAIN_SEQ_LEN": "2048", "EVAL_SEQ_LEN": "2048",
        "VAL_LOSS_EVERY": "500", "TRAIN_LOG_EVERY": "100", "MAX_WALLCLOCK_SECONDS": "1200",
        "GPTQ_CALIBRATION_BATCHES": "16", "GPTQ_RESERVE_SECONDS": "10.0", "EMA_DECAY": "0.99",
        "NUM_LOOPS": "2", "LOOP_START": "3", "LOOP_END": "5", "ENABLE_LOOPING_AT": "0.35",
        "XSA_LAST_N": "11", "SLIDING_WINDOW_ENABLED": "1",
    },
    "full": {},
}
if _TEST_MODE in _TEST_OVERRIDES:
    for _k, _v in _TEST_OVERRIDES[_TEST_MODE].items():
        if _k not in os.environ:
            os.environ[_k] = _v

_USE_TORCH_COMPILE = _TEST_MODE not in ("smoke",)

# ---------------------------------------------------------------------------
# Novel feature toggles
# ---------------------------------------------------------------------------
_CAT_ENABLED = bool(int(os.environ.get("CAT_ENABLED", "0")))
_CAT_WEIGHT = float(os.environ.get("CAT_WEIGHT", "0.001"))
_CAT_BITS = int(os.environ.get("CAT_BITS", "6"))
_CAT_EVERY = int(os.environ.get("CAT_EVERY", "50"))

_SPARSITY_ENABLED = bool(int(os.environ.get("SPARSITY_ENABLED", "0")))
_SPARSITY_APPLY_TO = os.environ.get("SPARSITY_APPLY_TO", "mlp").split(",")


# ---------------------------------------------------------------------------
# Hyperparameters (defaults match top-1 submission where applicable)
# ---------------------------------------------------------------------------
class Hyperparameters():
    data_dir = os.environ.get('DATA_DIR', './data/')
    seed = int(os.environ.get('SEED', 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    iterations = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac = float(os.environ.get('WARMDOWN_FRAC', 0.72))
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', 2048 * 48 * 8))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    train_log_every = int(os.environ.get('TRAIN_LOG_EVERY', 500))
    max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))
    val_batch_tokens = int(os.environ.get('VAL_BATCH_TOKENS', 2048 * 32 * 8))
    eval_seq_len = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    val_loss_every = int(os.environ.get('VAL_LOSS_EVERY', 4000))
    sliding_window_enabled = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))

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

    # QK gain bumped to 5.25 (top-1 uses 5.25, top-2/3 use 5.0)
    qk_gain_init = float(os.environ.get('QK_GAIN_INIT', 5.25))

    # Recurrence: loop layers 3-5 (top-1 uses 3-5, baseline used 4-5)
    num_loops = int(os.environ.get('NUM_LOOPS', 2))
    loop_start = int(os.environ.get('LOOP_START', 3))
    loop_end = int(os.environ.get('LOOP_END', 5))
    enable_looping_at = float(os.environ.get('ENABLE_LOOPING_AT', 0.35))

    # Parallel residuals from layer 7+ (GPT-J style, top-1/2/4 use this)
    parallel_residual_start = int(os.environ.get('PARALLEL_RESIDUAL_START', 7))

    min_lr = float(os.environ.get('MIN_LR', 0.0))
    embed_lr = float(os.environ.get('EMBED_LR', 0.6))
    head_lr = float(os.environ.get('HEAD_LR', 0.008))
    tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))

    # Matrix LR and weight decay bumped to match top-1
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
    eval_stride = int(os.environ.get('EVAL_STRIDE', 64))
    muon_beta2 = float(os.environ.get('MUON_BETA2', 0.95))
    adam_wd = float(os.environ.get('ADAM_WD', 0.02))
    muon_wd = float(os.environ.get('MUON_WD', 0.095))
    embed_wd = float(os.environ.get('EMBED_WD', 0.085))
    ema_decay = float(os.environ.get('EMA_DECAY', 0.9965))
    compressor = os.environ.get('COMPRESSOR', 'brotli')
    gptq_calibration_batches = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 64))
    gptq_reserve_seconds = float(os.environ.get('GPTQ_RESERVE_SECONDS', 12.0))
    matrix_bits = int(os.environ.get('MATRIX_BITS', 6))
    embed_bits = int(os.environ.get('EMBED_BITS', 8))
    matrix_clip_sigmas = float(os.environ.get('MATRIX_CLIP_SIGMAS', 12.85))
    embed_clip_sigmas = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))

    # TTT (Test-Time Training) - score-first SGD from top submissions
    ttt_enabled = bool(int(os.environ.get('TTT_ENABLED', '1')))
    ttt_lr = float(os.environ.get('TTT_LR', 0.005))
    ttt_momentum = float(os.environ.get('TTT_MOMENTUM', 0.9))
    ttt_epochs = int(os.environ.get('TTT_EPOCHS', 3))
    ttt_chunk_tokens = int(os.environ.get('TTT_CHUNK_TOKENS', 32768))

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    datasets_dir = os.path.join(data_dir, 'datasets', f'fineweb10B_sp{vocab_size}')
    train_files = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path = os.path.join(data_dir, 'tokenizers', f'fineweb_{vocab_size}_bpe.model')
    logfile = f"logs/{run_id}.txt"
    model_path = "final_model.pt"
    quantized_model_path = "final_model.int6.ptz"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_logger_hparams = None


def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h


def log(msg, console=False):
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        assert int(self.sp.vocab_size()) == h.vocab_size
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = \
            build_sentencepiece_luts(self.sp, h.vocab_size, device)


def build_sentencepiece_luts(sp, vocab_size, device):
    sv = int(sp.vocab_size())
    assert sp.piece_to_id("\u2581") != sp.unk_id()
    base_bytes = torch.zeros(max(sv, vocab_size), dtype=torch.int32, device=device)
    has_leading_space = torch.zeros(max(sv, vocab_size), dtype=torch.bool, device=device)
    is_boundary = torch.zeros(max(sv, vocab_size), dtype=torch.bool, device=device)
    for t in range(sv):
        piece = sp.id_to_piece(t)
        raw = piece.replace("\u2581", " ").encode("utf-8")
        base_bytes[t] = len(raw)
        if piece.startswith("\u2581"):
            has_leading_space[t] = True
            is_boundary[t] = True
        elif sp.is_control(t) or sp.is_unknown(t):
            is_boundary[t] = True
    return base_bytes, has_leading_space, is_boundary


def load_validation_tokens(pattern, seq_len):
    files = sorted(glob.glob(pattern))
    assert files, f"No files: {pattern}"
    chunks = []
    for f in files:
        chunks.append(load_data_shard(Path(f)))
    tokens = torch.cat(chunks)
    total = tokens.numel()
    usable = (total - 1) // seq_len * seq_len + 1
    return tokens[:usable]


def load_data_shard(file):
    hb = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    assert header.size == 256 and int(header[0]) == 20240520 and int(header[1]) == 1
    n = int(header[2])
    assert file.stat().st_size == hb + n * np.dtype("<u2").itemsize
    t = np.fromfile(file, dtype="<u2", count=n, offset=hb)
    assert t.size == n
    return torch.from_numpy(t.astype(np.uint16, copy=False))


_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}


def _read_num_tokens(file):
    key = str(file)
    c = _SHARD_NTOKENS_CACHE.get(key)
    if c is not None:
        return c
    header = np.fromfile(file, dtype="<i4", count=256)
    assert header.size == 256 and int(header[0]) == 20240520 and int(header[1]) == 1
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


def _get_shard_memmap(file):
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        assert all_files, f"No files: {h.train_files}"
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        mp = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(mp + 1)) if mp > 0 else 0
        ns = (self.num_tokens[si] - 1 - phase) // self.seq_len
        self.start_inds[si] = (phase + self.rng.permutation(ns) * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        dt = global_tokens // (self.world_size * grad_accum_steps)
        dbs = dt // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((dbs, self.seq_len), dtype=torch.int64)
        y = torch.empty((dbs, self.seq_len), dtype=torch.int64)
        for bi in range(dbs):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
                total = remaining.sum()
            si = int(self.rng.choice(len(self.files), p=remaining / total))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            w = torch.as_tensor(np.array(mm[start_ind:start_ind + self.seq_len + 1], dtype=np.int64))
            x[bi] = w[:-1]
            y[bi] = w[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ---------------------------------------------------------------------------
# Model architecture
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
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)),
            persistent=False,
        )
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len or self._cos_cached.device != device):
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


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        xr, xp = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = xr[..., :half], xr[..., half:]
        return torch.cat((
            torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1),
            xp,
        ), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
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
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        g = H // Hkv
        y_g = y.reshape(B, T, Hkv, g, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        return (y_g - (y_g * vn).sum(dim=-1, keepdim=True) * vn).reshape(B, T, H, D)

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
        y = flash_attn_3_func(q, k, v, causal=True) if _HAS_FA3 else _sdpa_attn(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
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


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 train_seq_len, layer_idx=0, ln_scale=False, parallel=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = parallel

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)

        if self.parallel:
            # GPT-J style: attention and MLP both read from same pre-attention input
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out \
                        + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
        else:
            # Standard sequential: MLP reads post-attention output
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * \
                self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)

        return x_out


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
            self.embed_proj = None
            self.head_proj = None

        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers

        self.blocks = nn.ModuleList([
            Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base,
                  h.qk_gain_init, h.train_seq_len, layer_idx=i, ln_scale=h.ln_scale,
                  parallel=(h.parallel_residual_start >= 0 and i >= h.parallel_residual_start))
            for i in range(h.num_layers)
        ])

        if h.rope_dims > 0:
            hd = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(hd, base=h.rope_base, train_seq_len=h.train_seq_len,
                                           rope_dims=h.rope_dims)

        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True

        # Layer looping / recurrence
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            ne = len(all_indices) // 2
            self.encoder_indices = all_indices[:ne]
            self.decoder_indices = all_indices[ne:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))

        # Skip connections (U-Net style)
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32)
        )
        self.skip_gates = nn.Parameter(
            torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)
        ) if h.skip_gates_enabled else None

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (module.weight.ndim == 2 and module.weight.shape[0] >= 64
                      and module.weight.shape[1] >= 64):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x

        skips = []
        enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        dec_iter = self.decoder_indices if self.looping_active else range(
            self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers
        )

        for i in enc_iter:
            x = self.blocks[i](x, x0)
            skips.append(x)

        for skip_idx, i in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                ss = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(ss, x, g)
                else:
                    x = x + ss
            x = self.blocks[i](x, x0)

        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        lp = F.linear(x, self.tok_emb.weight.to(x.dtype)) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean"
        )
        return loss


# ---------------------------------------------------------------------------
# CAT (Compressor-Aware Training)
# ---------------------------------------------------------------------------
def cat_compression_loss(model, bits=6, temperature=0.1):
    dev = next(model.parameters()).device
    total = torch.tensor(0.0, device=dev)
    count = 0
    cr = 2 ** (bits - 1) - 1
    for name, p in model.named_parameters():
        if p.ndim < 2 or p.numel() <= 65536:
            continue
        w = p.float()
        row_std = w.std(dim=-1, keepdim=True).clamp_min(1e-10)
        scaled = w / row_std
        q = torch.round(scaled * (cr / 3.0))
        q = q.clamp(-cr, cr)
        dist = (scaled * (cr / 3.0) - q).abs()
        soft_round = torch.sigmoid((dist - 0.5) / temperature)
        total = total + soft_round.mean()
        count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# 2:4 Sparsity (Hessian-guided: uses GPTQ Hessians when available)
# ---------------------------------------------------------------------------
def apply_sparsity_to_state_dict(sd, apply_to, hessians=None):
    result = {}
    sparsified = 0
    hessian_guided = 0
    for name, tensor in sd.items():
        if tensor.ndim != 2 or tensor.numel() <= 65536:
            result[name] = tensor
            continue
        cat = classify_param(name)
        if cat not in apply_to:
            result[name] = tensor
            continue
        t = tensor.float()
        rows, cols = t.shape

        # Hessian-guided importance: |w_j| * sqrt(H_jj) instead of just |w_j|
        # This preserves weights that are small but important to the loss
        if hessians is not None and name in hessians:
            H = hessians[name].float()
            diag_H = torch.diag(H).clamp_min(1e-8)
            col_importance = torch.sqrt(diag_H)  # shape: (cols,)
            importance = t.abs() * col_importance.unsqueeze(0)  # (rows, cols)
            hessian_guided += 1
        else:
            importance = t.abs()

        pad = (4 - cols % 4) % 4
        if pad:
            t = F.pad(t, (0, pad))
            importance = F.pad(importance, (0, pad))
        t4 = t.reshape(rows, -1, 4)
        imp4 = importance.reshape(rows, -1, 4)
        _, idx = imp4.topk(2, dim=-1)
        mask = torch.zeros_like(t4, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        t4 = t4 * mask
        t_sparse = t4.reshape(rows, -1)
        if pad:
            t_sparse = t_sparse[:, :cols]
        result[name] = t_sparse.to(tensor.dtype)
        sparsified += 1
    log(f"sparsity: applied 2:4 to {sparsified} tensors ({hessian_guided} Hessian-guided)")
    return result


# ---------------------------------------------------------------------------
# Param classification and optimizer
# ---------------------------------------------------------------------------
def classify_param(name):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def _zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
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


if _USE_TORCH_COMPILE:
    zeropower_via_newtonschulz5 = torch.compile(_zeropower_via_newtonschulz5)
else:
    zeropower_via_newtonschulz5 = _zeropower_via_newtonschulz5


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True,
                 weight_decay=0.0, row_normalize=False):
        super().__init__(params, dict(
            lr=lr, momentum=momentum, backend_steps=backend_steps,
            nesterov=nesterov, weight_decay=weight_decay, row_normalize=row_normalize,
        ))

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
                    if group.get("row_normalize", False):
                        rn = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                        g = g / rn.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,skip_gates"
    ).split(",") if p
)


class Optimizers():
    def __init__(self, h, base_model):
        bnp = list(base_model.blocks.named_parameters())
        matrix_params = [
            p for name, p in bnp
            if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        scalar_params = [
            p for name, p in bnp
            if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)

        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        self.optimizer_tok = torch.optim.AdamW(
            [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True,
        )
        self.optimizer_muon = Muon(
            matrix_params, lr=h.matrix_lr, momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize,
        )
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True,
        )
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [{"params": [base_model.lm_head.weight], "lr": h.head_lr, "base_lr": h.head_lr}],
                betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True,
            )
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self):
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()


def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (param.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) \
                and param.dtype != torch.float32:
            param.data = param.data.float()


# ---------------------------------------------------------------------------
# GPTQ quantization
# ---------------------------------------------------------------------------
def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[name].addmm_(x.T, x)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            cat = classify_param(name + ".weight")
            if cat in ("mlp", "attn"):
                hooks.append(module.register_forward_hook(make_hook(name + ".weight")))

    if model.tie_embeddings:
        hm = model.head_proj if model.head_proj is not None else model.final_norm

        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)
            return hook_fn

        hooks.append(hm.register_forward_hook(make_output_hook("tok_emb.weight")))

    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians


def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128):
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]

    cholesky_ok = False
    for extra_damp in [0.0, 0.1, 1.0, 10.0]:
        try:
            Ht = H.clone()
            if extra_damp > 0:
                Ht.diagonal().add_(extra_damp * Ht.diag().mean())
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Ht))
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            cholesky_ok = True
            break
        except torch.linalg.LinAlgError:
            continue

    if not cholesky_ok:
        row_std = W_orig.std(dim=1)
        s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
        return torch.clamp(torch.round(W_orig / s.float().unsqueeze(1)), -clip_range, clip_range).to(torch.int8), s

    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()

    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]

    return Q[:, invperm], s


def _simple_quantize_weight(w, clip_sigmas=3.0, clip_range=63):
    orig_shape = w.shape
    W = w.float().reshape(w.shape[0], -1)
    row_std = W.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    Q = torch.clamp(torch.round(W / s.float().unsqueeze(1)), -clip_range, clip_range).to(torch.int8)
    return Q.reshape(orig_shape), s


def gptq_mixed_quantize(state_dict, hessians, h):
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        cs = h.embed_clip_sigmas if "tok_emb" in name else h.matrix_clip_sigmas
        bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
        clip_range = 2 ** (bits - 1) - 1
        if name in hessians:
            q, s = gptq_quantize_weight(t, hessians[name], clip_sigmas=cs, clip_range=clip_range)
            meta[name] = f"gptq (int{bits})"
        else:
            log(f"  no Hessian for {name}, using simple quantization")
            q, s = _simple_quantize_weight(t, clip_sigmas=cs, clip_range=clip_range)
            meta[name] = f"simple (int{bits})"
        result[name + ".q"] = q
        result[name + ".scale"] = s

    cats = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub(r'\.\d+$', '', re.sub(r'blocks\.\d+', 'blocks', name))
        cats[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(cats):
        log(f"  {cat}: {', '.join(sorted(cats[cat]))}")
    return result, meta


def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        od = orig.dtype
        if "passthrough" in info:
            t = result[name]
            out[name] = t.to(od) if t.dtype == torch.float16 and od in (torch.float32, torch.bfloat16) else t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(od)
        else:
            out[name] = (q.float() * float(s.item())).to(od)
    return out


# ---------------------------------------------------------------------------
# Compression (byte-shuffle + brotli)
# ---------------------------------------------------------------------------
_BSHF_MAGIC = b"BSHF"


def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[off:off + len(chunk)] = chunk
        off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()


def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    off = 0
    for pos in range(stride):
        cl = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:cl] = payload[off:off + cl]
        off += cl
    return out.tobytes()


def _compress(data, compressor):
    data = _byte_shuffle(data)
    if compressor == "lzma":
        return lzma.compress(data, preset=6)
    elif compressor == "brotli":
        import brotli
        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def _decompress(data, compressor):
    if compressor == "lzma":
        raw = lzma.decompress(data)
    elif compressor == "brotli":
        import brotli
        raw = brotli.decompress(data)
    else:
        raise ValueError(f"Unknown compressor: {compressor!r}")
    return _byte_unshuffle(raw)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def serialize(h, base_model, code):
    code_bytes = len(code.encode("utf-8"))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        log(f"Serialized model: {os.path.getsize(h.model_path)} bytes")
        log(f"Code size: {code_bytes} bytes")

    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    device = torch.device("cuda", h.local_rank)
    log("GPTQ:collecting Hessians...")
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(base_model, calib_loader, h, device,
                                n_calibration_batches=h.gptq_calibration_batches)
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t0:.1f}s")
    if _SPARSITY_ENABLED:
        log("Applying Hessian-guided 2:4 sparsity...")
        sd_cpu = apply_sparsity_to_state_dict(sd_cpu, _SPARSITY_APPLY_TO, hessians=hessians)

    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = _compress(quant_buf.getvalue(), h.compressor)
    qfb = len(quant_blob)
    bt = qfb + code_bytes

    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        log(f"Serialized quantized+{h.compressor}: {qfb} bytes", console=True)
        log(f"Total submission: {bt} bytes", console=True)
    return bt, qfb


def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    with open(h.quantized_model_path, "rb") as f:
        qbd = f.read()
    qs = torch.load(io.BytesIO(_decompress(qbd, h.compressor)), map_location="cpu")
    deq = dequantize_mixed(qs["w"], qs["m"], sd_cpu)
    eval_model.load_state_dict(deq, strict=True)
    return eval_model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _loss_bpb(loss_sum, token_count, byte_count):
    vl = (loss_sum / token_count).item()
    return vl, vl / math.log(2.0) * (token_count.item() / byte_count.item())


def eval_val(h, device, val_data, model):
    seq_len = h.eval_seq_len
    lbt = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    assert lbt >= seq_len, "VAL_BATCH_TOKENS too small"
    lbs = lbt // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    ss = (total_seqs * h.rank) // h.world_size
    se = (total_seqs * (h.rank + 1)) // h.world_size
    vls = torch.zeros((), device=device, dtype=torch.float64)
    vtc = torch.zeros((), device=device, dtype=torch.float64)
    vbc = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(ss, se, lbs):
            bse = min(bss + lbs, se)
            rs = bss * seq_len
            re_ = bse * seq_len + 1
            local = val_data.val_tokens[rs:re_].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                bl = model(x, y).detach()
            btc = float(y.numel())
            vls += bl.to(torch.float64) * btc
            vtc += btc
            prev = x.reshape(-1)
            tgt = y.reshape(-1)
            tb = val_data.base_bytes_lut[tgt].to(dtype=torch.int16)
            tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(dtype=torch.int16)
            vbc += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vls, op=dist.ReduceOp.SUM)
        dist.all_reduce(vtc, op=dist.ReduceOp.SUM)
        dist.all_reduce(vbc, op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(vls, vtc, vbc)


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True) \
        if _USE_TORCH_COMPILE else base_model.forward_logits
    seq_len = h.eval_seq_len
    ctx = seq_len - h.eval_stride
    tt = val_data.val_tokens.numel() - 1
    ws_list = [ws for ws in range(0, tt, h.eval_stride) if ws + ctx < tt]
    tw = len(ws_list)
    ms = (tw * h.rank) // h.world_size
    me = (tw * (h.rank + 1)) // h.world_size
    mw = ws_list[ms:me]
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(mw), batch_seqs):
            bws = mw[bi:bi + batch_seqs]
            bsz = len(bws)
            xb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(bws):
                we = min(ws + seq_len, tt)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                xb[i, :wlen] = chunk[:-1]
                yb[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(xb)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none"
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(bws):
                wlen = wlens[i]
                s = 0 if ws == 0 else ctx
                ls += nll[i, s:wlen].to(torch.float64).sum()
                tc += float(wlen - s)
                tgt = yb[i, s:wlen]
                prev = xb[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                bc += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ls, op=dist.ReduceOp.SUM)
        dist.all_reduce(tc, op=dist.ReduceOp.SUM)
        dist.all_reduce(bc, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(ls, tc, bc)


# ---------------------------------------------------------------------------
# TTT (Test-Time Training) - score-first chunk-based SGD from top submissions
# ---------------------------------------------------------------------------
def eval_val_ttt(h, device, val_data, base_model, batch_seqs=32):
    """Score-first TTT: score each chunk under no_grad, then train on non-final chunks via SGD."""
    rank = h.rank
    world_size = h.world_size
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride

    # Build sliding windows and assign to chunks
    window_starts = [ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log(f"ttt:start chunks={num_chunks} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs}")

    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True) \
        if _USE_TORCH_COMPILE else base_model.forward_logits

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    ttt_params = [p for p in base_model.parameters()]
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue

        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]

        # Phase 1: Score windows under no_grad (score-first approach)
        base_model.eval()
        with torch.no_grad():
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
                    chunk_tok = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction='none'
                ).reshape(bsz, seq_len)

                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt]
                           & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # Phase 2: Train on chunk tokens (skip last chunk)
        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and h.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = h.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = chunk_seqs * rank // world_size
                my_seq_e = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(h.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_data.val_tokens.numel():
                            continue
                        local = val_data.val_tokens[start_tok:end_tok].to(
                            device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        optimizer.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    return _loss_bpb(loss_sum, token_count, byte_count)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    vl, vb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    log(f"{label} val_loss:{vl:.8f} val_bpb:{vb:.8f} eval_time:{1000.0 * (time.perf_counter() - t0):.0f}ms",
        console=True)
    return vl, vb


def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) \
        if _USE_TORCH_COMPILE else base_model
    model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False) \
        if h.distributed else compiled_model

    af = []
    if _CAT_ENABLED:
        af.append(f"CAT(weight={_CAT_WEIGHT},bits={_CAT_BITS},every={_CAT_EVERY})")
    if _SPARSITY_ENABLED:
        af.append(f"2:4_Sparsity(apply_to={_SPARSITY_APPLY_TO})")
    if h.parallel_residual_start >= 0:
        af.append(f"ParallelResid(from_layer={h.parallel_residual_start})")
    if h.ttt_enabled:
        af.append(f"TTT(lr={h.ttt_lr},epochs={h.ttt_epochs},chunk={h.ttt_chunk_tokens})")
    log(f"novel_features: {', '.join(af)}" if af else "novel_features: none (baseline)", console=True)
    log(f"test_mode: {_TEST_MODE}", console=True)
    log(f"fa3_available: {_HAS_FA3}")
    log(f"torch_compile: {_USE_TORCH_COMPILE}")
    log(f"model_params:{sum(p.numel() for p in base_model.parameters())}", console=True)

    optimizers = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    max_wc_ms = 1000.0 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wc_ms is not None:
        max_wc_ms -= h.gptq_reserve_seconds * 1000.0
        log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wc_ms:.0f}ms")

    def training_frac(step, elapsed_ms):
        if max_wc_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wc_ms, 1e-9)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
                if _CAT_ENABLED and step % _CAT_EVERY == 0:
                    loss = loss + _CAT_WEIGHT * cat_compression_loss(base_model, bits=_CAT_BITS)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps

        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        mm = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = mm
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    # Warmup phase
    if h.warmup_steps > 0:
        ims = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        ios = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(h.warmup_steps):
            step_fn(ws, 1.0)
            if ws <= 5 or (ws + 1) % 10 == 0 or ws + 1 == h.warmup_steps:
                log(f"warmup_step: {ws + 1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
            for ws in range(h.warmup_steps):
                step_fn(ws, 1.0)
                if ws <= 5 or (ws + 1) % 10 == 0 or ws + 1 == h.warmup_steps:
                    log(f"loop_warmup_step: {ws + 1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(ims, strict=True)
        for opt, state in zip(optimizers, ios, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)

    # Main training loop
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = h.ema_decay
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(h, device, val_data, model)
            log(f"{step}/{h.iterations} val_loss: {vl:.4f} val_bpb: {vb:.4f}", console=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}",
                    console=True)
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)

        if h.num_loops > 0 and not base_model.looping_active and frac >= h.enable_looping_at:
            base_model.looping_active = True
            log(f"layer_loop:enabled step:{step} frac:{frac:.3f} "
                f"encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")

        train_loss = step_fn(step, scale)
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0
                                       or stop_after_step is not None):
            tps = step * h.train_batch_tokens / (approx_ms / 1000.0)
            log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} "
                f"train_time: {approx_ms / 60000:.1f}m tok/s: {tps:.0f}")

        reached_cap = max_wc_ms is not None and approx_ms >= max_wc_ms
        if h.distributed and max_wc_ms is not None:
            rct = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rct, op=dist.ReduceOp.MAX)
            reached_cap = bool(rct.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")
    log("ema:applying EMA weights")
    cs = base_model.state_dict()
    avg = {name: t.to(dtype=cs[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg, strict=True)
    return base_model, compiled_model


# ---------------------------------------------------------------------------
# Train + Eval pipeline
# ---------------------------------------------------------------------------
def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    base_model, compiled_model = train_model(h, device, val_data)
    if _USE_TORCH_COMPILE:
        torch._dynamo.reset()
    timed_eval("pre-quantization post-ema", eval_val, h, device, val_data, compiled_model)
    serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))

    if h.distributed:
        dist.barrier()
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True) \
        if _USE_TORCH_COMPILE else eval_model

    timed_eval("quantized", eval_val, h, device, val_data, compiled_model)

    if h.sliding_window_enabled:
        timed_eval("quantized_sliding_window", eval_val_sliding, h, device, val_data, eval_model)

    # TTT evaluation (score-first SGD adaptation at eval time)
    if h.ttt_enabled and h.sliding_window_enabled:
        timed_eval("quantized_ttt", eval_val_ttt, h, device, val_data, eval_model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    assert torch.cuda.is_available(), "CUDA required"
    assert world_size > 0 and 8 % world_size == 0

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    if _HAS_FA3:
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)
    else:
        enable_mem_efficient_sdp(True)
        enable_math_sdp(True)

    torch._dynamo.config.optimize_ddp = False
    h = Hyperparameters()
    set_logging_hparams(h)

    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:")
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}")
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log(subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        ).stdout, console=False)
        log("=" * 100, console=False)

    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
