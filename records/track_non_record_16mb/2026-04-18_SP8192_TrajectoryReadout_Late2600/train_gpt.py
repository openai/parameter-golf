import collections, copy, glob, io, lzma, math, os
from pathlib import Path
import random, re, subprocess, sys, time, uuid, numpy as np, sentencepiece as spm, torch, torch.distributed as dist, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn
from flash_attn_interface import flash_attn_func as flash_attn_3_func

torch._dynamo.config.cache_size_limit = max(
    getattr(torch._dynamo.config, "cache_size_limit", 8), 64
)


class Hyperparameters:
    data_dir = os.environ.get("DATA_DIR", "./data/")
    seed = int(os.environ.get("SEED", 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 6e2))
    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 524288))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    sliding_window_enabled = bool(
        int(os.environ.get("SLIDING_WINDOW_ENABLED", "1"))
    )
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    embedding_dim = int(os.environ.get("EMBEDDING_DIM", 512))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    skip_gates_enabled = bool(int(os.environ.get("SKIP_GATES_ENABLED", "1")))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 3e1))
    rope_base = float(os.environ.get("ROPE_BASE", 1e4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    rope_train_seq_len = int(os.environ.get("ROPE_TRAIN_SEQ_LEN", 2048))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0))
    qk_gain_depth_ramp = float(os.environ.get("QK_GAIN_DEPTH_RAMP", 0.0))
    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))
    enable_looping_at_step = int(
        os.environ.get("ENABLE_LOOPING_AT_STEP", -1)
    )
    parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", 7))
    enable_parallel_residual_at = float(
        os.environ.get("ENABLE_PARALLEL_RESIDUAL_AT", 0.0)
    )
    enable_parallel_residual_at_step = int(
        os.environ.get("ENABLE_PARALLEL_RESIDUAL_AT_STEP", -1)
    )
    recur_attn_gate = bool(int(os.environ.get("RECUR_ATTN_GATE", "0")))
    recur_attn_gate_scale = float(
        os.environ.get("RECUR_ATTN_GATE_SCALE", 0.5)
    )
    use_pass_readout = bool(int(os.environ.get("USE_PASS_READOUT", "0")))
    readout_groups = int(os.environ.get("READOUT_GROUPS", 16))
    readout_scale = float(os.environ.get("READOUT_SCALE", 0.35))
    token_route_enabled = bool(int(os.environ.get("TOKEN_ROUTE_ENABLED", "0")))
    token_route_topk_frac = float(
        os.environ.get("TOKEN_ROUTE_TOPK_FRAC", 0.25)
    )
    token_route_start_pass = int(
        os.environ.get("TOKEN_ROUTE_START_PASS", 1)
    )
    token_route_min_tokens = int(
        os.environ.get("TOKEN_ROUTE_MIN_TOKENS", 64)
    )
    token_route_mlp_mult = float(
        os.environ.get("TOKEN_ROUTE_MLP_MULT", 1.5)
    )
    shared_adapter_dim = int(os.environ.get("SHARED_ADAPTER_DIM", 0))
    shared_adapter_start = int(os.environ.get("SHARED_ADAPTER_START", 1))
    shared_adapter_end = int(os.environ.get("SHARED_ADAPTER_END", 9))
    shared_adapter_scale_init = float(
        os.environ.get("SHARED_ADAPTER_SCALE_INIT", 1.0)
    )
    aux_exit_layer = int(os.environ.get("AUX_EXIT_LAYER", -1))
    aux_exit_weight = float(os.environ.get("AUX_EXIT_WEIGHT", 0.0))
    aux_exit_distill_weight = float(
        os.environ.get("AUX_EXIT_DISTILL_WEIGHT", 0.0)
    )
    aux_exit_distill_temp = float(
        os.environ.get("AUX_EXIT_DISTILL_TEMP", 1.0)
    )
    min_lr = float(os.environ.get("MIN_LR", 0.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.022))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92)
    )
    muon_momentum_warmup_steps = int(
        os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500)
    )
    muon_row_normalize = bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1")))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-08))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    adam_wd = float(os.environ.get("ADAM_WD", 0.02))
    muon_wd = float(os.environ.get("MUON_WD", 0.095))
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.005))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_hard_window_fraction = float(
        os.environ.get("TTT_HARD_WINDOW_FRACTION", 1.0)
    )
    ttt_hard_window_min = int(os.environ.get("TTT_HARD_WINDOW_MIN", 0))
    ttt_param_mode = os.environ.get("TTT_PARAM_MODE", "full")
    ttt_anchor_l2 = float(os.environ.get("TTT_ANCHOR_L2", 0.0))
    ttt_prefix_chunk_ratio = float(
        os.environ.get("TTT_PREFIX_CHUNK_RATIO", 0.0)
    )
    ttt_prefix_epochs = int(os.environ.get("TTT_PREFIX_EPOCHS", -1))
    ttt_prefix_lr_scale = float(os.environ.get("TTT_PREFIX_LR_SCALE", 1.0))
    ttt_prefix_hard_window_fraction = float(
        os.environ.get("TTT_PREFIX_HARD_WINDOW_FRACTION", -1.0)
    )
    ttt_easy_chunk_ratio = float(os.environ.get("TTT_EASY_CHUNK_RATIO", 0.0))
    ttt_easy_chunk_epochs = int(os.environ.get("TTT_EASY_CHUNK_EPOCHS", 0))
    ttt_outlier_drop_fraction = float(
        os.environ.get("TTT_OUTLIER_DROP_FRACTION", 0.0)
    )
    ttt_score_weight_power = float(
        os.environ.get("TTT_SCORE_WEIGHT_POWER", 0.0)
    )
    ttt_blend_back = float(os.environ.get("TTT_BLEND_BACK", 0.0))
    ttt_reset_momentum = bool(int(os.environ.get("TTT_RESET_MOMENTUM", "0")))
    eval_only = bool(int(os.environ.get("EVAL_ONLY", "0")))
    etlb_enabled = bool(int(os.environ.get("ETLB_ENABLED", "0")))
    etlb_lr = float(os.environ.get("ETLB_LR", 0.05))
    etlb_steps = int(os.environ.get("ETLB_STEPS", 5))
    etlb_clip = float(os.environ.get("ETLB_CLIP", 3.0))
    compressor = os.environ.get("COMPRESSOR", "brotli")
    gptq_calibration_batches = int(
        os.environ.get("GPTQ_CALIBRATION_BATCHES", 64)
    )
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 12.0))
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 2e1))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    find_unused_parameters = bool(
        int(os.environ.get("FIND_UNUSED_PARAMETERS", "0"))
    )
    grad_accum_steps = 8 // world_size
    datasets_dir = os.path.join(
        data_dir, "datasets", f"fineweb10B_sp{vocab_size}"
    )
    train_files = os.path.join(datasets_dir, "fineweb_train_*.bin")
    val_files = os.path.join(datasets_dir, "fineweb_val_*.bin")
    tokenizer_path = os.path.join(
        data_dir, "tokenizers", f"fineweb_{vocab_size}_bpe.model"
    )
    logfile = f"logs/{run_id}.txt"
    model_path = os.environ.get("MODEL_PATH", "final_model.pt")
    quantized_model_path = os.environ.get(
        "QUANTIZED_MODEL_PATH", "final_model.int6.ptz"
    )


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
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)


class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}"
            )
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        (
            self.base_bytes_lut,
            self.has_leading_space_lut,
            self.is_boundary_token_lut,
        ) = build_sentencepiece_luts(self.sp, h.vocab_size, device)


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert (
        sp.piece_to_id("▁") != sp.unk_id()
    ), "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if (
            sp.is_control(token_id)
            or sp.is_unknown(token_id)
            or sp.is_unused(token_id)
        ):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(
            f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}"
        )
    return tokens[: usable + 1]


def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
    tokens_np = np.fromfile(
        file, dtype="<u2", count=num_tokens, offset=header_bytes
    )
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}


def _read_num_tokens(file):
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


def _get_shard_memmap(file):
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(
        file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,)
    )
    _MMAP_CACHE[key] = mm
    return mm


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(
                f"No files found for pattern: {h.train_files}"
            )
        self.files = all_files[h.rank :: h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(
            self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1)
        )
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array(
            [len(s) for s in self.start_inds], dtype=np.float64
        )
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array(
                    [len(s) for s in self.start_inds], dtype=np.float64
                )
                total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(
                np.array(
                    mm[start_ind : start_ind + self.seq_len + 1],
                    dtype=np.int64,
                )
            )
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


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
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.rope_dims, 2, dtype=torch.float32)
            / self.rope_dims
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (
                    torch.arange(0, rd, 2, dtype=torch.float32, device=device)
                    / rd
                )
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(
            dtype=dtype
        )


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        rope_base,
        qk_gain_init,
        train_seq_len,
    ):
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
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rope_dims = 0
        self.rotary = Rotary(
            self.head_dim, base=rope_base, train_seq_len=train_seq_len
        )
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        attn_dtype = (
            x.dtype
            if x.dtype in (torch.float16, torch.bfloat16)
            else torch.bfloat16
        )
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),)).to(dtype=attn_dtype)
        k = F.rms_norm(k, (k.size(-1),)).to(dtype=attn_dtype)
        v = v.to(dtype=attn_dtype)
        cos, sin = self.rotary(seqlen, x.device, attn_dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims).to(dtype=attn_dtype)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims).to(dtype=attn_dtype)
        q = q * self.q_gain.to(dtype=attn_dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class SharedAdapter(nn.Module):
    def __init__(self, dim, bottleneck):
        super().__init__()
        self.fc = CastedLinear(dim, bottleneck, bias=False)
        self.proj = CastedLinear(bottleneck, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(F.silu(self.fc(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_base,
        qk_gain_init,
        train_seq_len,
        layer_idx=0,
        ln_scale=False,
        recur_attn_gate=False,
        recur_attn_gate_scale=0.5,
        token_route_enabled=False,
        token_route_topk_frac=0.25,
        token_route_start_pass=1,
        token_route_min_tokens=64,
        token_route_mlp_mult=1.5,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            train_seq_len,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        self.recur_attn_delta = (
            nn.Parameter(torch.zeros(3, dim, dtype=torch.float32))
            if recur_attn_gate
            else None
        )
        self.recur_attn_gate_scale = recur_attn_gate_scale
        self.route_score = (
            nn.Parameter(torch.zeros(3, dim, dtype=torch.float32))
            if token_route_enabled
            else None
        )
        self.route_scale = (
            nn.Parameter(torch.ones(dim, dtype=torch.float32))
            if token_route_enabled
            else None
        )
        self.route_norm = RMSNorm() if token_route_enabled else None
        self.route_mlp = (
            MLP(dim, token_route_mlp_mult) if token_route_enabled else None
        )
        self.token_route_topk_frac = token_route_topk_frac
        self.token_route_start_pass = token_route_start_pass
        self.token_route_min_tokens = token_route_min_tokens
        self.ln_scale_factor = (
            1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        )
        self.parallel = False

    def forward(self, x, x0, pass_idx=0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_in = (self.attn_norm(x_in) * self.ln_scale_factor).to(
            dtype=x_in.dtype
        )
        attn_out = self.attn(attn_in)
        if self.recur_attn_delta is not None:
            pass_idx = min(max(pass_idx, 0), self.recur_attn_delta.size(0) - 1)
            attn_gate = 1.0 + self.recur_attn_gate_scale * torch.tanh(
                self.recur_attn_delta[pass_idx].to(dtype=attn_out.dtype)
            )[None, None, :]
            attn_out = attn_out * attn_gate
        if self.parallel:
            mlp_in = (self.mlp_norm(x_in) * self.ln_scale_factor).to(
                dtype=x_in.dtype
            )
            mlp_out = self.mlp(mlp_in)
            x_out = (
                x_in
                + self.attn_scale.to(dtype=x_in.dtype)[None, None, :]
                * attn_out
                + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
            )
        else:
            x_out = (
                x_in
                + self.attn_scale.to(dtype=x_in.dtype)[None, None, :]
                * attn_out
            )
            mlp_in = (self.mlp_norm(x_out) * self.ln_scale_factor).to(
                dtype=x_out.dtype
            )
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[
                None, None, :
            ] * self.mlp(mlp_in)
        if (
            self.route_mlp is not None
            and pass_idx >= self.token_route_start_pass
        ):
            bsz, seqlen, dim = x_out.shape
            k = max(
                1,
                min(
                    seqlen,
                    max(
                        int(math.ceil(seqlen * self.token_route_topk_frac)),
                        self.token_route_min_tokens,
                    ),
                ),
            )
            route_in = (self.route_norm(x_out) * self.ln_scale_factor).to(
                dtype=x_out.dtype
            )
            route_idx = min(max(pass_idx, 0), self.route_score.size(0) - 1)
            route_score = (
                route_in.float().square().mean(dim=-1)
                + 0.05
                * (
                    route_in
                    * self.route_score[route_idx].to(dtype=route_in.dtype)[
                        None, None, :
                    ]
                )
                .mean(dim=-1)
                .float()
            )
            top_idx = route_score.topk(k, dim=1, sorted=False).indices
            route_mask = torch.zeros(
                bsz, seqlen, 1, dtype=x_out.dtype, device=x_out.device
            )
            route_mask.scatter_(
                1,
                top_idx.unsqueeze(-1),
                torch.ones(
                    bsz, k, 1, dtype=x_out.dtype, device=x_out.device
                ),
            )
            routed = (
                self.route_scale.to(dtype=route_in.dtype)[None, None, :]
                * self.route_mlp(route_in)
            )
            x_out = x_out + routed * route_mask
        return x_out


class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(
                f"logit_softcap must be positive, got {h.logit_softcap}"
            )
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.num_loops = h.num_loops
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(
                h.embedding_dim, h.model_dim, bias=False
            )
            self.head_proj = CastedLinear(
                h.model_dim, h.embedding_dim, bias=False
            )
        else:
            self.embed_proj = None
            self.head_proj = None
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        den = max(h.num_layers - 1, 1)
        self.blocks = nn.ModuleList(
            [
                Block(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    h.mlp_mult,
                    h.rope_base,
                    h.qk_gain_init + h.qk_gain_depth_ramp * (i / den),
                    h.train_seq_len,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                    recur_attn_gate=(
                        h.recur_attn_gate and h.loop_start <= i <= h.loop_end
                    ),
                    recur_attn_gate_scale=h.recur_attn_gate_scale,
                    token_route_enabled=(
                        h.token_route_enabled
                        and h.loop_start <= i <= h.loop_end
                    ),
                    token_route_topk_frac=h.token_route_topk_frac,
                    token_route_start_pass=h.token_route_start_pass,
                    token_route_min_tokens=h.token_route_min_tokens,
                    token_route_mlp_mult=h.token_route_mlp_mult,
                )
                for i in range(h.num_layers)
            ]
        )
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(
                    head_dim,
                    base=h.rope_base,
                    train_seq_len=h.train_seq_len,
                    rope_dims=h.rope_dims,
                )
        self.shared_adapter = (
            SharedAdapter(h.model_dim, h.shared_adapter_dim)
            if h.shared_adapter_dim > 0
            else None
        )
        self.shared_adapter_start = h.shared_adapter_start
        self.shared_adapter_end = h.shared_adapter_end
        self.shared_adapter_scales = (
            nn.Parameter(
                torch.full(
                    (h.num_layers,),
                    h.shared_adapter_scale_init,
                    dtype=torch.float32,
                )
            )
            if self.shared_adapter is not None
            else None
        )
        self.readout_block_idx = h.loop_end
        if h.use_pass_readout and h.num_loops > 0:
            if h.model_dim % h.readout_groups != 0:
                raise ValueError(
                    f"READOUT_GROUPS={h.readout_groups} must divide MODEL_DIM={h.model_dim}"
                )
            self.readout_groups = h.readout_groups
            self.readout_group_size = h.model_dim // h.readout_groups
            self.readout_delta = nn.Parameter(
                torch.zeros(h.num_loops, h.readout_groups, dtype=torch.float32)
            )
            self.readout_scale = h.readout_scale
        else:
            self.readout_groups = 0
            self.readout_group_size = 0
            self.readout_delta = None
            self.readout_scale = 0.0
        self.aux_exit_layer = h.aux_exit_layer
        self.aux_exit_weight = h.aux_exit_weight
        self.aux_exit_distill_weight = h.aux_exit_distill_weight
        self.aux_exit_distill_temp = h.aux_exit_distill_temp
        self.final_norm = RMSNorm()
        self.lm_head = (
            None
            if h.tie_embeddings
            else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        self.parallel_residual_start = h.parallel_residual_start
        self.parallel_residual_active = False
        self.set_parallel_residuals(
            h.parallel_residual_start >= 0
            and (
                h.enable_parallel_residual_at_step == 0
                or h.enable_parallel_residual_at_step < 0
                and h.enable_parallel_residual_at <= 0
            )
        )
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(
                range(self.num_encoder_layers, h.num_layers)
            )
        visit_counts = [0] * h.num_layers
        self.encoder_pass_indices = []
        for idx in self.encoder_indices:
            self.encoder_pass_indices.append(visit_counts[idx])
            visit_counts[idx] += 1
        self.decoder_pass_indices = []
        for idx in self.decoder_indices:
            self.decoder_pass_indices.append(visit_counts[idx])
            visit_counts[idx] += 1
        self.num_skip_weights = min(
            len(self.encoder_indices), len(self.decoder_indices)
        )
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32)
        )
        self.skip_gates = (
            nn.Parameter(
                torch.zeros(
                    self.num_skip_weights, h.model_dim, dtype=torch.float32
                )
            )
            if h.skip_gates_enabled
            else None
        )
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(
                self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std
            )
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (
                    module.weight.ndim == 2
                    and module.weight.shape[0] >= 64
                    and module.weight.shape[1] >= 64
                ):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def set_parallel_residuals(self, enabled):
        active = bool(enabled and self.parallel_residual_start >= 0)
        for i, block in enumerate(self.blocks):
            block.parallel = active and i >= self.parallel_residual_start
        self.parallel_residual_active = active

    def _apply_shared_adapter(self, x, block_idx):
        if (
            self.shared_adapter is None
            or block_idx < self.shared_adapter_start
            or block_idx > self.shared_adapter_end
        ):
            return x
        adapter_in = F.rms_norm(x, (x.size(-1),))
        adapter_out = self.shared_adapter(adapter_in)
        scale = self.shared_adapter_scales[block_idx].to(dtype=x.dtype)
        return x + scale * adapter_out

    def _record_pass_state(self, pass_states, x, pass_idx):
        if (
            pass_states is None
            or pass_idx < 0
            or pass_idx >= len(pass_states)
        ):
            return
        pass_states[pass_idx] = x

    def _apply_pass_readout(self, pass_states):
        if (
            self.readout_delta is None
            or pass_states is None
            or not pass_states
            or pass_states[-1] is None
        ):
            return pass_states[-1] if pass_states else None
        final_state = pass_states[-1]
        correction = torch.zeros_like(final_state)
        for idx, prev_state in enumerate(pass_states[:-1]):
            if prev_state is None:
                continue
            coeff = torch.tanh(self.readout_delta[idx]).to(
                dtype=final_state.dtype
            )
            coeff = coeff.repeat_interleave(self.readout_group_size)[
                None, None, :
            ]
            correction = correction + coeff * (prev_state - final_state)
        return final_state + self.readout_scale * correction

    def _project_logits(self, x):
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(
            logits_proj / self.logit_softcap
        )

    def forward_logits(self, input_ids, return_aux=False):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        pass_states = (
            [None] * (self.num_loops + 1)
            if self.readout_delta is not None
            else None
        )
        skips = []
        aux_hidden = None
        if self.looping_active:
            enc_iter = zip(
                self.encoder_indices,
                self.encoder_pass_indices,
            )
            dec_iter = zip(
                self.decoder_indices,
                self.decoder_pass_indices,
            )
        else:
            enc_iter = (
                (i, 0) for i in range(self.num_encoder_layers)
            )
            dec_iter = (
                (i, 0)
                for i in range(
                    self.num_encoder_layers,
                    self.num_encoder_layers + self.num_decoder_layers,
                )
            )
        for i, pass_idx in enc_iter:
            x = self.blocks[i](x, x0, pass_idx=pass_idx)
            x = self._apply_shared_adapter(x, i)
            if (
                return_aux
                and i == self.aux_exit_layer
                and aux_hidden is None
            ):
                aux_hidden = x
            if i == self.readout_block_idx:
                self._record_pass_state(pass_states, x, pass_idx)
            skips.append(x)
        for skip_idx, (i, pass_idx) in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = (
                    self.skip_weights[skip_idx].to(dtype=x.dtype)[
                        None, None, :
                    ]
                    * skips.pop()
                )
                if self.skip_gates is not None:
                    g = torch.sigmoid(
                        self.skip_gates[skip_idx].to(dtype=x.dtype)
                    )[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            x = self.blocks[i](x, x0, pass_idx=pass_idx)
            x = self._apply_shared_adapter(x, i)
            if (
                return_aux
                and i == self.aux_exit_layer
                and aux_hidden is None
            ):
                aux_hidden = x
            if i == self.readout_block_idx:
                self._record_pass_state(pass_states, x, pass_idx)
                if pass_idx == self.num_loops and pass_states is not None:
                    x = self._apply_pass_readout(pass_states)
        logits = self._project_logits(x)
        if return_aux and aux_hidden is not None:
            return logits, self._project_logits(aux_hidden)
        return logits

    def forward(self, input_ids, target_ids):
        if (
            (
                self.aux_exit_weight > 0.0
                or self.aux_exit_distill_weight > 0.0
            )
            and self.aux_exit_layer >= 0
        ):
            logits, aux_logits = self.forward_logits(input_ids, return_aux=True)
        else:
            logits = self.forward_logits(input_ids)
            aux_logits = None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )
        if aux_logits is not None:
            if self.aux_exit_weight > 0.0:
                aux_loss = F.cross_entropy(
                    aux_logits.reshape(-1, aux_logits.size(-1)).float(),
                    target_ids.reshape(-1),
                    reduction="mean",
                )
                loss = loss + self.aux_exit_weight * aux_loss
            if self.aux_exit_distill_weight > 0.0:
                temp = self.aux_exit_distill_temp
                student = (
                    aux_logits.reshape(-1, aux_logits.size(-1)).float() / temp
                )
                teacher = (
                    logits.detach().reshape(-1, logits.size(-1)).float()
                    / temp
                )
                aux_kl = F.kl_div(
                    F.log_softmax(student, dim=-1),
                    F.softmax(teacher, dim=-1),
                    reduction="batchmean",
                ) * (temp**2)
                loss = loss + self.aux_exit_distill_weight * aux_kl
        return loss


def classify_param(name):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name or ".route_mlp." in name:
        return "mlp"
    if ".attn." in name or ".proj." in name and ".mlp." not in name:
        return "attn"
    return "other"


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-07):
    a, b, c = 3.4445, -4.775, 2.0315
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
    def __init__(
        self,
        params,
        lr,
        momentum,
        backend_steps,
        nesterov=True,
        weight_decay=0.0,
        row_normalize=False,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
            ),
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
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )
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
                        row_norms = (
                            g.float()
                            .norm(dim=-1, keepdim=True)
                            .clamp_min(1e-07)
                        )
                        g = g / row_norms.to(g.dtype)
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
                g = (
                    updates_flat[curr : curr + p.numel()]
                    .view_as(p)
                    .to(dtype=p.dtype)
                )
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,recur_attn_delta,readout_delta,q_gain,route_score,route_scale,shared_adapter_scales,skip_weight,skip_weights,skip_gates",
    ).split(",")
    if pattern
)


class Optimizers:
    def __init__(self, h, base_model):
        block_named_params = list(base_model.blocks.named_parameters())
        adapter_named_params = (
            list(base_model.shared_adapter.named_parameters())
            if base_model.shared_adapter is not None
            else []
        )
        matrix_params = [
            p
            for (name, p) in block_named_params
            if p.ndim == 2
            and not any(
                pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS
            )
        ]
        matrix_params.extend(
            [
                p
                for (_, p) in adapter_named_params
                if p.ndim == 2
            ]
        )
        scalar_params = [
            p
            for (name, p) in block_named_params
            if p.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if base_model.shared_adapter_scales is not None:
            scalar_params.append(base_model.shared_adapter_scales)
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if (
            base_model.skip_gates is not None
            and base_model.skip_gates.numel() > 0
        ):
            scalar_params.append(base_model.skip_gates)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [
            {
                "params": [base_model.tok_emb.weight],
                "lr": token_lr,
                "base_lr": token_lr,
            }
        ]
        self.optimizer_tok = torch.optim.AdamW(
            tok_params,
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.embed_wd,
            fused=True,
        )
        self.optimizer_muon = Muon(
            matrix_params,
            lr=h.matrix_lr,
            momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps,
            weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize,
        )
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [
                {
                    "params": scalar_params,
                    "lr": h.scalar_lr,
                    "base_lr": h.scalar_lr,
                }
            ],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.adam_wd,
            fused=True,
        )
        self.optimizers = [
            self.optimizer_tok,
            self.optimizer_muon,
            self.optimizer_scalar,
        ]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [
                    {
                        "params": [base_model.lm_head.weight],
                        "lr": h.head_lr,
                        "base_lr": h.head_lr,
                    }
                ],
                betas=(h.beta1, h.beta2),
                eps=h.adam_eps,
                fused=True,
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
        if (
            param.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ) and param.dtype != torch.float32:
            param.data = param.data.float()


def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
            hessians[name].addmm_(x.T, x)

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            cat = classify_param(name + ".weight")
            if cat in ("mlp", "attn"):
                hooks.append(
                    module.register_forward_hook(make_hook(name + ".weight"))
                )
    if model.tie_embeddings:
        hook_module = (
            model.head_proj
            if model.head_proj is not None
            else model.final_norm
        )

        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1],
                        x.shape[1],
                        dtype=torch.float32,
                        device=device,
                    )
                hessians[name].addmm_(x.T, x)

            return hook_fn

        hooks.append(
            hook_module.register_forward_hook(
                make_output_hook("tok_emb.weight")
            )
        )
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(
                h.train_batch_tokens, h.grad_accum_steps
            )
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
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
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
            q_col = torch.clamp(
                torch.round(w_col / sf), -clip_range, clip_range
            )
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s


CONTROL_INT8_SUBSTRINGS = (
    "attn_scale",
    "attn_scales",
    "mlp_scale",
    "mlp_scales",
    "resid_mix",
    "resid_mixes",
    "recur_attn_delta",
    "readout_delta",
    "q_gain",
    "route_score",
    "route_scale",
    "shared_adapter_scales",
    "skip_weight",
    "skip_weights",
    "skip_gates",
)


def is_control_int8_tensor(name, tensor):
    return tensor.is_floating_point() and any(
        part in name for part in CONTROL_INT8_SUBSTRINGS
    )


def quantize_control_int8(tensor):
    t = tensor.detach().cpu().float().contiguous()
    if t.ndim <= 1:
        scale = t.abs().max().clamp_min(1e-8) / 127.0
        q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8)
        return q, scale.to(torch.float16)
    scale = t.flatten(1).abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.clamp(
        torch.round(t / scale.view(t.shape[0], *([1] * (t.ndim - 1)))),
        -127,
        127,
    ).to(torch.int8)
    return q, scale.to(torch.float16)


def gptq_mixed_quantize(state_dict, hessians, h):
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if is_control_int8_tensor(name, t):
            q, s = quantize_control_int8(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = "control (int8)"
            continue
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        cs = h.embed_clip_sigmas if "tok_emb" in name else h.matrix_clip_sigmas
        bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
        q, s = gptq_quantize_weight(
            t, hessians[name], clip_sigmas=cs, clip_range=2 ** (bits - 1) - 1
        )
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
    categories = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub("\\.\\d+$", "", re.sub("blocks\\.\\d+", "blocks", name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return result, meta


def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info == "control (int8)":
            q, s = result[name + ".q"], result[name + ".scale"]
            if s.ndim > 0:
                out[name] = (
                    q.float()
                    * s.float().view(q.shape[0], *[1] * (q.ndim - 1))
                ).to(orig_dtype)
            else:
                out[name] = (q.float() * float(s.item())).to(orig_dtype)
            continue
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (
                q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))
            ).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


_BSHF_MAGIC = b"BSHF"


def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off : dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
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
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off : src_off + chunk_len]
        src_off += chunk_len
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
    raw = _byte_unshuffle(raw)
    return raw


def serialize(h, base_model, code):
    code_bytes = len(code.encode("utf-8"))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size: {code_bytes} bytes")
    sd_cpu = {
        k: v.detach().cpu() for (k, v) in base_model.state_dict().items()
    }
    device = torch.device("cuda", h.local_rank)
    log("GPTQ:collecting Hessians from calibration data...")
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(
        base_model,
        calib_loader,
        h,
        device,
        n_calibration_batches=h.gptq_calibration_batches,
    )
    log(
        f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s"
    )
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        log(
            f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes"
        )
        log(
            f"Total submission size quantized+{h.compressor}: {bytes_total} bytes"
        )
    return bytes_total, quant_file_bytes


def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    sd_cpu = {
        k: v.detach().cpu() for (k, v) in eval_model.state_dict().items()
    }
    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob_disk, h.compressor)),
        map_location="cpu",
    )
    deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model


def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = (
        val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    )
    return val_loss, val_bpb


def eval_val(h, device, val_data, model):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (
        h.world_size * h.grad_accum_steps
    )
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(
                dtype=torch.int16
            )
            token_bytes += (
                val_data.has_leading_space_lut[tgt_ids]
                & ~val_data.is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    logits_fn = torch.compile(
        base_model.forward_logits, dynamic=False, fullgraph=True
    )
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [
        ws
        for ws in range(0, total_tokens, h.eval_stride)
        if ws + context_size < total_tokens
    ]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi : bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(
                bsz, seq_len, dtype=torch.int64, device=device
            )
            y_batch = torch.zeros(
                bsz, seq_len, dtype=torch.int64, device=device
            )
            wlens = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws : we + 1].to(
                    dtype=torch.int64, device=device
                )
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
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
                tb += (
                    val_data.has_leading_space_lut[tgt]
                    & ~val_data.is_boundary_token_lut[prev]
                ).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


def eval_val_ttt(h, device, val_data, base_model, batch_seqs=32):
    rank = h.rank
    world_size = h.world_size
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride
    window_starts = [
        ws
        for ws in range(0, total_tokens, stride)
        if ws + context_size < total_tokens
    ]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    hard_frac = max(0.0, min(h.ttt_hard_window_fraction, 1.0))
    compiled_logits = torch.compile(
        base_model.forward_logits, dynamic=False, fullgraph=True
    )
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ttt_mode, named_ttt_params = _select_ttt_named_params(h, base_model)
    ttt_params = [p for (_, p) in named_ttt_params]
    ttt_param_count = sum(int(p.numel()) for p in ttt_params)
    if ttt_param_count == 0:
        raise ValueError(
            f"TTT_PARAM_MODE={h.ttt_param_mode} selected no trainable parameters"
        )
    ttt_base_params = (
        [p.detach().clone() for p in ttt_params]
        if h.ttt_anchor_l2 > 0 or h.ttt_blend_back > 0
        else None
    )
    easy_chunk_ratio = max(0.0, min(h.ttt_easy_chunk_ratio, 1.0))
    easy_chunk_epochs = max(0, min(h.ttt_easy_chunk_epochs, h.ttt_epochs))
    prefix_chunk_ratio = max(0.0, min(h.ttt_prefix_chunk_ratio, 1.0))
    prefix_chunk_count = min(
        num_chunks,
        int(math.ceil(num_chunks * prefix_chunk_ratio)),
    )
    prefix_epochs = (
        h.ttt_epochs
        if h.ttt_prefix_epochs < 0
        else max(0, h.ttt_prefix_epochs)
    )
    prefix_lr_scale = max(0.0, h.ttt_prefix_lr_scale)
    prefix_hard_frac = (
        hard_frac
        if h.ttt_prefix_hard_window_fraction < 0.0
        else max(0.0, min(h.ttt_prefix_hard_window_fraction, 1.0))
    )
    outlier_drop_frac = max(0.0, min(h.ttt_outlier_drop_fraction, 0.95))
    score_weight_power = max(0.0, h.ttt_score_weight_power)
    blend_back = max(0.0, min(h.ttt_blend_back, 1.0))
    log(
        f"ttt:start chunks={num_chunks} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs} hard_window_fraction={hard_frac:.3f} param_mode={ttt_mode} trainable={ttt_param_count} anchor_l2={h.ttt_anchor_l2:g} prefix_chunk_ratio={prefix_chunk_ratio:.3f} prefix_chunk_count={prefix_chunk_count} prefix_epochs={prefix_epochs} prefix_lr_scale={prefix_lr_scale:.3f} prefix_hard_window_fraction={prefix_hard_frac:.3f} easy_chunk_ratio={easy_chunk_ratio:.3f} easy_chunk_epochs={easy_chunk_epochs} outlier_drop_frac={outlier_drop_frac:.3f} score_weight_power={score_weight_power:.3f} blend_back={blend_back:.3f} reset_momentum={int(h.ttt_reset_momentum)}"
    )
    for p in base_model.parameters():
        p.requires_grad_(False)
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(
        ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum
    )

    def restore_ttt_params():
        if blend_back <= 0.0 or ttt_base_params is None:
            return
        with torch.no_grad():
            for p, p0 in zip(ttt_params, ttt_base_params):
                p.lerp_(p0, blend_back)
                if h.ttt_reset_momentum:
                    state = optimizer.state.get(p)
                    if state is not None and "momentum_buffer" in state:
                        state["momentum_buffer"].zero_()

    running_chunk_loss = None
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]
        local_window_scores = []
        local_chunk_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        local_chunk_token_count = torch.zeros(
            (), device=device, dtype=torch.float64
        )
        base_model.eval()
        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi : bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(
                    bsz, seq_len, dtype=torch.int64, device=device
                )
                y_batch = torch.zeros(
                    bsz, seq_len, dtype=torch.int64, device=device
                )
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk_tok = val_data.val_tokens[ws : we + 1].to(
                        dtype=torch.int64, device=device
                    )
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    local_chunk_loss_sum += scored_nll.sum()
                    local_chunk_token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (
                        val_data.has_leading_space_lut[tgt]
                        & ~val_data.is_boundary_token_lut[prev]
                    ).to(torch.float64)
                    byte_count += tb.sum()
                    local_window_scores.append(
                        (float(scored_nll.mean().item()), ws)
                    )
        if world_size > 1:
            dist.all_reduce(local_chunk_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_chunk_token_count, op=dist.ReduceOp.SUM)
        chunk_mean_loss = float(
            local_chunk_loss_sum
            / local_chunk_token_count.clamp_min(1.0)
        )
        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and h.ttt_epochs > 0:
            base_model.train()
            cos_lr = (
                h.ttt_lr
                * 0.5
                * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
            )
            in_prefix_phase = ci < prefix_chunk_count
            phase_lr = cos_lr * (
                prefix_lr_scale if in_prefix_phase else 1.0
            )
            phase_hard_frac = (
                prefix_hard_frac if in_prefix_phase else hard_frac
            )
            for pg in optimizer.param_groups:
                pg["lr"] = phase_lr
            chunk_epochs = prefix_epochs if in_prefix_phase else h.ttt_epochs
            if (
                not in_prefix_phase
                and easy_chunk_ratio > 0.0
                and easy_chunk_epochs < h.ttt_epochs
                and running_chunk_loss is not None
                and chunk_mean_loss < running_chunk_loss * easy_chunk_ratio
            ):
                chunk_epochs = easy_chunk_epochs
            if h.is_main_process and (
                ci == 0 or ci == prefix_chunk_count
            ):
                phase_name = "prefix" if in_prefix_phase else "suffix"
                log(
                    f"ttt:phase chunk={ci+1}/{num_chunks} phase={phase_name} epochs={chunk_epochs} lr={phase_lr:.6f} hard_window_fraction={phase_hard_frac:.3f}"
                )
            if 0.0 < phase_hard_frac < 1.0:
                gathered = (
                    [None for _ in range(world_size)]
                    if world_size > 1
                    else None
                )
                if world_size > 1:
                    dist.all_gather_object(gathered, local_window_scores)
                    global_window_scores = [
                        item for sub in gathered for item in sub
                    ]
                else:
                    global_window_scores = local_window_scores
                global_window_scores.sort(key=lambda x: x[0], reverse=True)
                if outlier_drop_frac > 0.0 and len(global_window_scores) > 1:
                    drop_count = min(
                        int(math.floor(len(global_window_scores) * outlier_drop_frac)),
                        len(global_window_scores) - 1,
                    )
                    candidate_scores = global_window_scores[drop_count:]
                else:
                    candidate_scores = global_window_scores
                selected = max(
                    int(math.ceil(len(candidate_scores) * phase_hard_frac)),
                    h.ttt_hard_window_min,
                )
                if world_size > 1:
                    selected = max(selected, world_size)
                    selected = min(selected, len(candidate_scores))
                    selected -= selected % world_size
                    if selected == 0:
                        selected = min(len(candidate_scores), world_size)
                else:
                    selected = min(selected, len(candidate_scores))
                selected_pairs = candidate_scores[:selected]
                selected_windows = sorted(ws for (_, ws) in selected_pairs)
                selected_score_map = {ws: score for (score, ws) in selected_pairs}
                if h.is_main_process and (
                    ci == 0 or ci == prefix_chunk_count
                ):
                    log(
                        f"ttt:hard_windows chunk={ci+1}/{num_chunks} selected={selected}/{len(candidate_scores)} dropped={len(global_window_scores) - len(candidate_scores)}"
                    )
                my_count = (
                    len(selected_windows) // world_size
                    if world_size > 1
                    else len(selected_windows)
                )
                my_selected = (
                    selected_windows[rank * my_count : (rank + 1) * my_count]
                    if world_size > 1
                    else selected_windows
                )
                for _ep in range(chunk_epochs):
                    for bi in range(0, len(my_selected), batch_seqs):
                        batch_ws = my_selected[bi : bi + batch_seqs]
                        bsz = len(batch_ws)
                        x_batch = torch.zeros(
                            bsz, seq_len, dtype=torch.int64, device=device
                        )
                        y_batch = torch.zeros(
                            bsz, seq_len, dtype=torch.int64, device=device
                        )
                        loss_mask = torch.zeros(
                            bsz, seq_len, dtype=torch.bool, device=device
                        )
                        for i, ws in enumerate(batch_ws):
                            we = min(ws + seq_len, total_tokens)
                            wlen = we - ws
                            chunk_tok = val_data.val_tokens[ws : we + 1].to(
                                device=device, dtype=torch.int64
                            )
                            x_batch[i, :wlen] = chunk_tok[:-1]
                            y_batch[i, :wlen] = chunk_tok[1:]
                            s = 0 if ws == 0 else context_size
                            loss_mask[i, s:wlen] = True
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ):
                            logits = base_model.forward_logits(x_batch)
                        nll = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)).float(),
                            y_batch.reshape(-1),
                            reduction="none",
                        ).reshape(bsz, seq_len)
                        seq_loss = (
                            nll.masked_fill(~loss_mask, 0.0).sum(dim=1)
                            / loss_mask.sum(dim=1).clamp_min(1)
                        )
                        if score_weight_power > 0.0:
                            batch_scores = torch.tensor(
                                [selected_score_map[ws] for ws in batch_ws],
                                device=device,
                                dtype=torch.float32,
                            )
                            weights = (
                                batch_scores
                                / batch_scores.mean().clamp_min(1e-6)
                            ).clamp_min(1e-3).pow(score_weight_power)
                            loss = (seq_loss * weights).sum() / weights.sum()
                        else:
                            loss = seq_loss.mean()
                        if h.ttt_anchor_l2 > 0 and ttt_base_params is not None:
                            anchor_loss = torch.zeros(
                                (), device=device, dtype=torch.float32
                            )
                            for p, p0 in zip(ttt_params, ttt_base_params):
                                anchor_loss += (
                                    (p.float() - p0.float()).square().mean()
                                )
                            loss = loss + h.ttt_anchor_l2 * anchor_loss
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(
                                        p.grad, op=dist.ReduceOp.AVG
                                    )
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        optimizer.step()
                restore_ttt_params()
            else:
                chunk_seqs = (chunk_end - chunk_start) // seq_len
                if chunk_seqs > 0:
                    my_seq_s = chunk_seqs * rank // world_size
                    my_seq_e = chunk_seqs * (rank + 1) // world_size
                    my_chunk_seqs = my_seq_e - my_seq_s
                    for _ep in range(chunk_epochs):
                        for bs in range(0, my_chunk_seqs, batch_seqs):
                            be = min(bs + batch_seqs, my_chunk_seqs)
                            actual_bs = my_seq_s + bs
                            start_tok = chunk_start + actual_bs * seq_len
                            end_tok = (
                                chunk_start + (my_seq_s + be) * seq_len + 1
                            )
                            if end_tok > val_data.val_tokens.numel():
                                continue
                            local = val_data.val_tokens[start_tok:end_tok].to(
                                device=device, dtype=torch.int64
                            )
                            x = local[:-1].reshape(-1, seq_len)
                            y = local[1:].reshape(-1, seq_len)
                            optimizer.zero_grad(set_to_none=True)
                            with torch.autocast(
                                device_type="cuda", dtype=torch.bfloat16
                            ):
                                loss = base_model(x, y)
                            if h.ttt_anchor_l2 > 0 and ttt_base_params is not None:
                                anchor_loss = torch.zeros(
                                    (), device=device, dtype=torch.float32
                                )
                                for p, p0 in zip(ttt_params, ttt_base_params):
                                    anchor_loss += (
                                        (p.float() - p0.float())
                                        .square()
                                        .mean()
                                    )
                                loss = loss + h.ttt_anchor_l2 * anchor_loss
                            loss.backward()
                            if world_size > 1:
                                for p in ttt_params:
                                    if p.grad is not None:
                                        dist.all_reduce(
                                            p.grad, op=dist.ReduceOp.AVG
                                        )
                            torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                            optimizer.step()
                    restore_ttt_params()
        running_chunk_loss = (
            chunk_mean_loss
            if running_chunk_loss is None
            else 0.9 * running_chunk_loss + 0.1 * chunk_mean_loss
        )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    return _loss_bpb(loss_sum, token_count, byte_count)


def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    log(
        f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms"
    )
    return val_loss, val_bpb


def _is_q_only_ttt_param(name):
    return ".attn.c_q." in name or ".attn.q_gain" in name


def _is_readout_ttt_param(name):
    return "readout_delta" in name


def _select_ttt_named_params(h, base_model):
    ttt_mode = h.ttt_param_mode.lower()
    named_ttt_params = list(base_model.named_parameters())
    if ttt_mode == "full":
        return ttt_mode, named_ttt_params
    if ttt_mode == "control":
        return ttt_mode, [
            (name, p)
            for (name, p) in named_ttt_params
            if p.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
    loop_prefixes = tuple(
        f"blocks.{i}." for i in range(h.loop_start, h.loop_end + 1)
    )
    if ttt_mode == "recur_control":
        return ttt_mode, [
            (name, p)
            for (name, p) in named_ttt_params
            if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            and (
                any(prefix in name for prefix in loop_prefixes)
                or "skip_weights" in name
                or "skip_gates" in name
            )
        ]
    if ttt_mode == "q_only":
        return ttt_mode, [
            (name, p)
            for (name, p) in named_ttt_params
            if _is_q_only_ttt_param(name)
        ]
    if ttt_mode == "readout_only":
        return ttt_mode, [
            (name, p)
            for (name, p) in named_ttt_params
            if _is_readout_ttt_param(name)
        ]
    if ttt_mode == "loop_q_only":
        return ttt_mode, [
            (name, p)
            for (name, p) in named_ttt_params
            if _is_q_only_ttt_param(name)
            and any(prefix in name for prefix in loop_prefixes)
        ]
    if ttt_mode == "loop_readout":
        return ttt_mode, [
            (name, p)
            for (name, p) in named_ttt_params
            if _is_readout_ttt_param(name)
            or (
                any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
                and any(prefix in name for prefix in loop_prefixes)
            )
        ]
    raise ValueError(f"Unsupported TTT_PARAM_MODE={h.ttt_param_mode}")


def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        model = DDP(
            compiled_model,
            device_ids=[h.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=h.find_unused_parameters,
        )
    else:
        model = compiled_model
    log(f"model_params:{sum(p.numel()for p in base_model.parameters())}")
    optimizers = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    max_wallclock_ms = (
        1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    )
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1e3
        log(
            f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms"
        )

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-09)

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
                model.require_backward_grad_sync = (
                    micro_step == h.grad_accum_steps - 1
                )
            x, y = train_loader.next_batch(
                h.train_batch_tokens, h.grad_accum_steps
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = (
            min(step / h.muon_momentum_warmup_steps, 1.0)
            if h.muon_momentum_warmup_steps > 0
            else 1.0
        )
        muon_momentum = (
            1 - frac
        ) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                base_model.parameters(), h.grad_clip_norm
            )
        optimizers.step()
        return train_loss

    if h.warmup_steps > 0:
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for (name, tensor) in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if (
                warmup_step <= 5
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == h.warmup_steps
            ):
                log(f"warmup_step: {warmup_step+1}/{h.warmup_steps}")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(
                f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if (
                    warmup_step <= 5
                    or (warmup_step + 1) % 10 == 0
                    or warmup_step + 1 == h.warmup_steps
                ):
                    log(f"loop_warmup_step: {warmup_step+1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(
            optimizers, initial_optimizer_states, strict=True
        ):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)
        ema_state = {
            name: t.detach().float().clone()
            for (name, t) in base_model.state_dict().items()
        }
        ema_decay = h.ema_decay
        training_time_ms = 0.0
        stop_after_step = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step = 0
        while True:
            last_step = (
                step == h.iterations
                or stop_after_step is not None
                and step >= stop_after_step
            )
            should_validate = (
                last_step
                or h.val_loss_every > 0
                and step % h.val_loss_every == 0
            )
            if should_validate:
                torch.cuda.synchronize()
                training_time_ms += 1e3 * (time.perf_counter() - t0)
                val_loss, val_bpb = eval_val(h, device, val_data, base_model)
                log(
                    f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}"
                )
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            if last_step:
                if stop_after_step is not None and step < h.iterations:
                    log(
                        f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}"
                    )
                break
            elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
            frac = training_frac(step, elapsed_ms)
            scale = lr_mul(frac)
            loop_ready = (
                h.enable_looping_at_step >= 0
                and step >= h.enable_looping_at_step
                or h.enable_looping_at_step < 0
                and frac >= h.enable_looping_at
            )
            if (
                h.num_loops > 0
                and not base_model.looping_active
                and loop_ready
            ):
                base_model.looping_active = True
                log(
                    f"layer_loop:enabled step:{step} frac:{frac:.3f} trigger_step:{h.enable_looping_at_step} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
                )
            par_ready = (
                h.enable_parallel_residual_at_step >= 0
                and step >= h.enable_parallel_residual_at_step
                or h.enable_parallel_residual_at_step < 0
                and frac >= h.enable_parallel_residual_at
            )
            if (
                h.parallel_residual_start >= 0
                and not base_model.parallel_residual_active
                and par_ready
            ):
                base_model.set_parallel_residuals(True)
                log(
                    f"parallel_residual:enabled step:{step} frac:{frac:.3f} trigger_step:{h.enable_parallel_residual_at_step} start:{h.parallel_residual_start}"
                )
            train_loss = step_fn(step, scale)
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(ema_decay).add_(
                        t.detach().float(), alpha=1.0 - ema_decay
                    )
            step += 1
            approx_training_time_ms = training_time_ms + 1e3 * (
                time.perf_counter() - t0
            )
            should_log_train = h.train_log_every > 0 and (
                step <= 5
                or step % h.train_log_every == 0
                or stop_after_step is not None
            )
            if should_log_train:
                tok_per_sec = (
                    step
                    * h.train_batch_tokens
                    / (approx_training_time_ms / 1e3)
                )
                log(
                    f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}"
                )
            reached_cap = (
                max_wallclock_ms is not None
                and approx_training_time_ms >= max_wallclock_ms
            )
            if h.distributed and max_wallclock_ms is not None:
                reached_cap_tensor = torch.tensor(
                    int(reached_cap), device=device
                )
                dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
                reached_cap = bool(reached_cap_tensor.item())
            if stop_after_step is None and reached_cap:
                stop_after_step = step
    log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB"
    )
    log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {
        name: t.to(dtype=current_state[name].dtype)
        for (name, t) in ema_state.items()
    }
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model


def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    log(
        f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}"
    )
    log(f"val_tokens: {val_data.val_tokens.numel()-1}")
    if not h.eval_only:
        base_model, compiled_model = train_model(h, device, val_data)
        torch._dynamo.reset()
        timed_eval(
            "pre-quantization post-ema",
            eval_val,
            h,
            device,
            val_data,
            base_model,
        )
        serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
    else:
        log(f"eval_only: quantized_model_path={h.quantized_model_path}")
    if h.distributed:
        dist.barrier()
    eval_model = deserialize(h, device)
    if h.parallel_residual_start >= 0:
        eval_model.set_parallel_residuals(True)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval("quantized", eval_val, h, device, val_data, compiled_model)
    if h.sliding_window_enabled:
        timed_eval(
            "quantized_sliding_window",
            eval_val_sliding,
            h,
            device,
            val_data,
            eval_model,
        )
    if h.ttt_enabled and h.sliding_window_enabled:
        del eval_model, compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.parallel_residual_start >= 0:
            ttt_model.set_parallel_residuals(True)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        timed_eval(
            "quantized_ttt", eval_val_ttt, h, device, val_data, ttt_model
        )
        del ttt_model
    if h.etlb_enabled and h.sliding_window_enabled:
        if "eval_model" not in dir():
            eval_model = deserialize(h, device)
            if h.parallel_residual_start >= 0:
                eval_model.set_parallel_residuals(True)
            if h.num_loops > 0:
                eval_model.looping_active = True
        timed_eval(
            "quantized_sliding_etlb",
            eval_val_sliding_etlb,
            h,
            device,
            val_data,
            eval_model,
        )


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:", console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log(
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            ).stdout,
            console=False,
        )
        log("=" * 100, console=False)
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
