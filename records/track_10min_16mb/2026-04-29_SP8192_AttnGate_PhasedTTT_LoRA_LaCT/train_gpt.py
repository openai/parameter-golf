import collections, copy, fcntl, glob, io, lzma, math, os, struct
from pathlib import Path
import random, re, subprocess, sys, time, uuid
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from flash_attn_interface import flash_attn_func as flash_attn_3_func


class Hyperparameters:
    # ── data ──────────────────────────────────────────────────────────────
    data_dir                    = os.environ.get('DATA_DIR', './data/')
    seed                        = int(os.environ.get('SEED', 1337))
    run_id                      = os.environ.get('RUN_ID', str(uuid.uuid4()))

    # ── training ──────────────────────────────────────────────────────────
    iterations                  = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac               = float(os.environ.get('WARMDOWN_FRAC', 0.72))
    warmup_steps                = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens          = int(os.environ.get('TRAIN_BATCH_TOKENS', 786432))
    train_seq_len               = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    train_log_every             = int(os.environ.get('TRAIN_LOG_EVERY', 500))
    max_wallclock_seconds       = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))

    # ── validation ────────────────────────────────────────────────────────
    val_batch_tokens            = int(os.environ.get('VAL_BATCH_TOKENS', 524288))
    eval_seq_len                = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    val_loss_every              = int(os.environ.get('VAL_LOSS_EVERY', 4000))
    sliding_window_enabled      = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))

    # ── tokeniser ─────────────────────────────────────────────────────────
    vocab_size                  = int(os.environ.get('VOCAB_SIZE', 8192))

    # ── architecture ──────────────────────────────────────────────────────
    num_layers                  = int(os.environ.get('NUM_LAYERS', 11))
    xsa_last_n                  = int(os.environ.get('XSA_LAST_N', 11))
    model_dim                   = int(os.environ.get('MODEL_DIM', 512))
    embedding_dim               = int(os.environ.get('EMBEDDING_DIM', 512))
    num_kv_heads                = int(os.environ.get('NUM_KV_HEADS', 4))
    num_heads                   = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult                    = float(os.environ.get('MLP_MULT', 4.0))
    skip_gates_enabled          = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    gated_attn_enabled          = bool(int(os.environ.get('GATED_ATTN_ENABLED', '1')))
    gated_attn_init_std         = float(os.environ.get('GATED_ATTN_INIT_STD', 0.01))
    gated_attn_quant_gate       = bool(int(os.environ.get('GATED_ATTN_QUANT_GATE', '1')))
    tie_embeddings              = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    logit_softcap               = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    rope_base                   = float(os.environ.get('ROPE_BASE', 10000.0))
    rope_dims                   = int(os.environ.get('ROPE_DIMS', 16))
    rope_train_seq_len          = int(os.environ.get('ROPE_TRAIN_SEQ_LEN', 2048))
    ln_scale                    = bool(int(os.environ.get('LN_SCALE', '1')))
    qk_gain_init                = float(os.environ.get('QK_GAIN_INIT', 5.25))

    # ── depth recurrence ──────────────────────────────────────────────────
    num_loops                   = int(os.environ.get('NUM_LOOPS', 2))
    loop_start                  = int(os.environ.get('LOOP_START', 3))
    loop_end                    = int(os.environ.get('LOOP_END', 5))
    enable_looping_at           = float(os.environ.get('ENABLE_LOOPING_AT', 0.35))
    parallel_residual_start     = int(os.environ.get('PARALLEL_RESIDUAL_START', 7))

    # ── optimiser ─────────────────────────────────────────────────────────
    min_lr                      = float(os.environ.get('MIN_LR', 0.0))
    embed_lr                    = float(os.environ.get('EMBED_LR', 0.6))
    head_lr                     = float(os.environ.get('HEAD_LR', 0.008))
    tied_embed_lr               = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std         = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    matrix_lr                   = float(os.environ.get('MATRIX_LR', 0.022))
    scalar_lr                   = float(os.environ.get('SCALAR_LR', 0.02))
    muon_momentum               = float(os.environ.get('MUON_MOMENTUM', 0.99))
    muon_backend_steps          = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_momentum_warmup_start  = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.92))
    muon_momentum_warmup_steps  = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
    muon_row_normalize          = bool(int(os.environ.get('MUON_ROW_NORMALIZE', '1')))
    beta1                       = float(os.environ.get('BETA1', 0.9))
    beta2                       = float(os.environ.get('BETA2', 0.95))
    adam_eps                    = float(os.environ.get('ADAM_EPS', 1e-8))
    grad_clip_norm              = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
    eval_stride                 = int(os.environ.get('EVAL_STRIDE', 64))
    muon_beta2                  = float(os.environ.get('MUON_BETA2', 0.95))
    adam_wd                     = float(os.environ.get('ADAM_WD', 0.02))
    muon_wd                     = float(os.environ.get('MUON_WD', 0.095))
    embed_wd                    = float(os.environ.get('EMBED_WD', 0.085))
    ema_decay                   = float(os.environ.get('EMA_DECAY', 0.9965))

    # ── single-phase / phased TTT ─────────────────────────────────────────
    ttt_enabled                 = bool(int(os.environ.get('TTT_ENABLED', '1')))
    ttt_lr                      = float(os.environ.get('TTT_LR', 0.005))
    ttt_momentum                = float(os.environ.get('TTT_MOMENTUM', 0.9))

    # ── LoRA / phased-TTT adapter settings ────────────────────────────────
    ttt_lora_enabled            = bool(int(os.environ.get('TTT_LORA_ENABLED', '0')))
    ttt_lora_rank               = int(os.environ.get('TTT_LORA_RANK', 96))
    ttt_lora_lr                 = float(os.environ.get('TTT_LORA_LR', 0.0001))
    ttt_chunk_size              = int(os.environ.get('TTT_CHUNK_SIZE', 48))
    ttt_batch_size              = int(os.environ.get('TTT_BATCH_SIZE', 64))
    ttt_lora_batch_size         = ttt_batch_size
    ttt_grad_steps              = int(os.environ.get('TTT_GRAD_STEPS', 1))
    ttt_weight_decay            = float(os.environ.get('TTT_WEIGHT_DECAY', 0.5))
    ttt_beta1                   = float(os.environ.get('TTT_BETA1', 0.0))
    ttt_lora_beta1              = ttt_beta1
    ttt_beta2                   = float(os.environ.get('TTT_BETA2', 0.999))
    ttt_lora_beta2              = ttt_beta2
    ttt_k_lora                  = bool(int(os.environ.get('TTT_K_LORA', '1')))
    ttt_mlp_lora                = bool(int(os.environ.get('TTT_MLP_LORA', '1')))
    ttt_o_lora                  = bool(int(os.environ.get('TTT_O_LORA', '1')))
    ttt_optimizer               = os.environ.get('TTT_OPTIMIZER', 'adam')
    ttt_lora_optimizer          = ttt_optimizer
    ttt_eval_batches            = os.environ.get('TTT_EVAL_BATCHES', '')
    phased_ttt_enabled          = bool(int(os.environ.get('PHASED_TTT_ENABLED', '1')))
    phased_ttt_prefix_docs      = int(os.environ.get('PHASED_TTT_PREFIX_DOCS', 2000))
    phased_ttt_num_phases       = int(os.environ.get('PHASED_TTT_NUM_PHASES', 4))
    global_ttt_lr               = float(os.environ.get('GLOBAL_TTT_LR', 0.001))
    global_ttt_momentum         = float(os.environ.get('GLOBAL_TTT_MOMENTUM', 0.9))
    global_ttt_epochs           = int(os.environ.get('GLOBAL_TTT_EPOCHS', 1))
    global_ttt_chunk_tokens     = int(os.environ.get('GLOBAL_TTT_CHUNK_TOKENS', 32768))
    global_ttt_batch_seqs       = int(os.environ.get('GLOBAL_TTT_BATCH_SEQS', 32))
    global_ttt_warmup_start_lr  = float(os.environ.get('GLOBAL_TTT_WARMUP_START_LR', 0.0))
    global_ttt_warmup_chunks    = int(os.environ.get('GLOBAL_TTT_WARMUP_CHUNKS', 0))
    global_ttt_grad_clip        = float(os.environ.get('GLOBAL_TTT_GRAD_CLIP', 1.0))
    global_ttt_respect_doc_boundaries = bool(
        int(os.environ.get('GLOBAL_TTT_RESPECT_DOC_BOUNDARIES', '1'))
    )
    ttt_eval_seq_len            = int(os.environ.get('TTT_EVAL_SEQ_LEN', 2048))
    val_doc_fraction            = float(os.environ.get('VAL_DOC_FRACTION', 1.0))

    # ── LaCT fast-weight adapter (optional additional eval) ───────────────
    lact_ttt_enabled            = bool(int(os.environ.get('LACT_TTT_ENABLED', '0')))
    lact_fast_weight            = os.environ.get('LACT_FAST_WEIGHT', 'swiglu').lower()
    lact_state_dim              = int(os.environ.get('LACT_STATE_DIM', 128))
    lact_scale                  = float(os.environ.get('LACT_SCALE', 0.08))
    lact_lr                     = float(os.environ.get('LACT_LR', 0.02))
    lact_momentum               = float(os.environ.get('LACT_MOMENTUM', 0.9))
    lact_epochs                 = int(os.environ.get('LACT_EPOCHS', 1))
    lact_chunk_tokens           = int(os.environ.get('LACT_CHUNK_TOKENS', 32768))
    lact_update                 = os.environ.get('LACT_UPDATE', 'muon').lower()
    lact_base_ttt               = bool(int(os.environ.get('LACT_BASE_TTT', '1')))
    lact_batch_seqs             = int(os.environ.get('LACT_BATCH_SEQS', 16))
    lact_grad_clip              = float(os.environ.get('LACT_GRAD_CLIP', 1.0))
    lact_init_std               = float(os.environ.get('LACT_INIT_STD', 0.02))
    lact_normalize              = bool(int(os.environ.get('LACT_NORMALIZE', '1')))

    # ── export / GPTQ ─────────────────────────────────────────────────────
    compressor                  = os.environ.get('COMPRESSOR', 'brotli')
    gptq_calibration_batches    = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 64))
    gptq_reserve_seconds        = float(os.environ.get('GPTQ_RESERVE_SECONDS', 12.0))
    matrix_bits                 = int(os.environ.get('MATRIX_BITS', 6))
    embed_bits                  = int(os.environ.get('EMBED_BITS', 8))
    matrix_clip_sigmas          = float(os.environ.get('MATRIX_CLIP_SIGMAS', 12.85))
    embed_clip_sigmas           = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))
    mlp_clip_sigmas             = float(os.environ.get('MLP_CLIP_SIGMAS', 12.0))
    attn_clip_sigmas            = float(os.environ.get('ATTN_CLIP_SIGMAS', 13.0))
    lqer_enabled                = bool(int(os.environ.get('LQER_ENABLED', '1')))
    lqer_rank                   = int(os.environ.get('LQER_RANK', 4))
    lqer_top_k                  = int(os.environ.get('LQER_TOP_K', 3))
    lqer_factor_bits            = int(os.environ.get('LQER_FACTOR_BITS', 4))
    lqer_asym_enabled           = bool(int(os.environ.get('LQER_ASYM_ENABLED', '1')))
    lqer_asym_group             = int(os.environ.get('LQER_ASYM_GROUP', '64'))

    # ── entropy allocator ─────────────────────────────────────────────────
    export_allocator            = os.environ.get('EXPORT_ALLOCATOR', 'mixed').lower()
    artifact_target_bytes       = int(os.environ.get('ARTIFACT_TARGET_BYTES', 16000000))
    allocator_group_cols        = int(os.environ.get('ALLOCATOR_GROUP_COLS', 128))
    allocator_matrix_bits       = tuple(int(x) for x in os.environ.get('ALLOCATOR_MATRIX_BITS', '5,6,7').split(',') if x)
    # mlp falls back to matrix_bits; attn is conservative (no 5-bit)
    allocator_mlp_bits          = tuple(int(x) for x in os.environ.get('ALLOCATOR_MLP_BITS', '').split(',') if x) or None
    allocator_attn_bits         = tuple(int(x) for x in os.environ.get('ALLOCATOR_ATTN_BITS', '6,7').split(',') if x) or None
    # embeddings: conservative 8-bit only
    allocator_embed_bits        = tuple(int(x) for x in os.environ.get('ALLOCATOR_EMBED_BITS', '8').split(',') if x)
    allocator_matrix_sigmas     = tuple(float(x) for x in os.environ.get('ALLOCATOR_MATRIX_SIGMAS', '10.5,12.85,15.0').split(',') if x)
    allocator_mlp_sigmas        = tuple(float(x) for x in os.environ.get('ALLOCATOR_MLP_SIGMAS', '').split(',') if x) or None
    allocator_attn_sigmas       = tuple(float(x) for x in os.environ.get('ALLOCATOR_ATTN_SIGMAS', '').split(',') if x) or None
    allocator_embed_sigmas      = tuple(float(x) for x in os.environ.get('ALLOCATOR_EMBED_SIGMAS', '16.0,20.0,24.0').split(',') if x)
    allocator_use_entropy_proxy = bool(int(os.environ.get('ALLOCATOR_USE_ENTROPY_PROXY', '1')))
    allocator_lambdas           = tuple(float(x) for x in os.environ.get('ALLOCATOR_LAMBDAS', '0,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5').split(',') if x)
    allocator_code_wrappers     = tuple(x for x in os.environ.get('ALLOCATOR_CODE_WRAPPERS', 'source,lzma_raw_b85_exec').split(',') if x)

    # ── distributed ───────────────────────────────────────────────────────
    distributed     = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank            = int(os.environ.get('RANK', '0'))
    world_size      = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank      = int(os.environ.get('LOCAL_RANK', '0'))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size

    # ── derived paths ─────────────────────────────────────────────────────
    datasets_dir         = os.path.join(data_dir, 'datasets', f'fineweb10B_sp{vocab_size}')
    train_files          = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files            = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path       = os.path.join(data_dir, 'tokenizers', f'fineweb_{vocab_size}_bpe.model')
    logfile              = f'logs/{run_id}.txt'
    model_path           = 'final_model.pt'
    quantized_model_path = 'final_model.int6.ptz'


BOS_ID = None  # discovered from tokenizer at first TTT eval call

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


# ── data loading ──────────────────────────────────────────────────────────────

class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(f'VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}')
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = \
            build_sentencepiece_luts(self.sp, h.vocab_size, device)


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert sp.piece_to_id('▁') != sp.unk_id(), \
        "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np          = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np   = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np   = np.ones((table_size,),  dtype=np.bool_)
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
    return (
        torch.tensor(base_bytes_np,        dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool,  device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool,  device=device),
    )


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f'No files found for pattern: {pattern}')
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f'Validation split is too short for seq_len={seq_len}')
    return tokens[:usable + 1]


def load_data_shard(file):
    header_bytes = 256 * np.dtype('<i4').itemsize
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f'Unexpected shard header for {file}')
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype('<u2').itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f'Shard size mismatch for {file}')
    tokens_np = np.fromfile(file, dtype='<u2', count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


_SHARD_HEADER_BYTES = 256 * np.dtype('<i4').itemsize
_SHARD_NTOKENS_CACHE: dict = {}
_MMAP_CACHE: dict = {}


def _read_num_tokens(file):
    key = str(file)
    if key in _SHARD_NTOKENS_CACHE:
        return _SHARD_NTOKENS_CACHE[key]
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f'Unexpected shard header for {file}')
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


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len    = h.train_seq_len
        self.device     = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f'No files found for pattern: {h.train_files}')
        self.files     = all_files[h.rank::h.world_size]
        self.rng       = np.random.Generator(np.random.PCG64(h.rank))
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
        device_tokens    = global_tokens // (self.world_size * grad_accum_steps)
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


# ── model ─────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        w    = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim           = dim
        self.base          = base
        self.train_seq_len = train_seq_len
        self.rope_dims     = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale    = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached    = freqs.cos()[None, :, None, :]
            self._sin_cached    = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


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
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len,
                 gated_attn=False, gated_attn_init_std=0.01):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError('model_dim must be divisible by num_heads')
        if num_heads % num_kv_heads != 0:
            raise ValueError('num_heads must be divisible by num_kv_heads')
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError('head_dim must be even for RoPE')
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q   = CastedLinear(dim,    dim,    bias=False)
        self.c_k   = CastedLinear(dim,    kv_dim, bias=False)
        self.c_v   = CastedLinear(dim,    kv_dim, bias=False)
        self.proj  = CastedLinear(dim,    dim,    bias=False)
        self.proj._zero_init = True
        self.q_gain   = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary    = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa   = False
        # GatedAttn (Qwen / arXiv:2505.06708): input-dependent per-head sigmoid gate
        self.gated_attn = gated_attn
        if gated_attn:
            W = torch.empty(num_heads, dim, dtype=torch.float32)
            nn.init.normal_(W, mean=0.0, std=gated_attn_init_std)
            self.attn_gate_w = nn.Parameter(W)

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv   = v.size(-2)
        group = H // Hkv
        y_g   = y.reshape(B, T, Hkv, group, D)
        vn    = F.normalize(v, dim=-1).unsqueeze(-2)
        proj  = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x, q_lora=None, k_lora=None, v_lora=None, o_lora=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        if q_lora is not None:
            q = q + q_lora.delta(x)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x)
        if k_lora is not None:
            k = k + k_lora.delta(x)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_lora is not None:
            v = v + v_lora.delta(x)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.gated_attn:
            g = torch.sigmoid(F.linear(x.contiguous(), self.attn_gate_w.to(x.dtype)))
            y = y * g[..., None]
        y_flat = y.reshape(bsz, seqlen, dim)
        out = self.proj(y_flat)
        if o_lora is not None:
            out = out + o_lora.delta(y_flat)
        return out


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden    = int(mlp_mult * dim)
        self.fc   = CastedLinear(dim,    hidden, bias=False)
        self.proj = CastedLinear(hidden, dim,    bias=False)
        self.proj._zero_init = True

    def forward(self, x, mlp_lora=None):
        h = self.fc(x)
        if mlp_lora is not None:
            h = h + mlp_lora.delta(x)
        return self.proj(F.leaky_relu(h, negative_slope=0.5).square())


class BatchedLinearLoRA(nn.Module):
    """Batched LoRA adapter (PR #1727): one A/B pair per document slot in the batch.

    A: (bsz, rank, in_features)  — uniform init
    B: (bsz, out_features, rank) — zero init so delta=0 at start
    forward(x) for x: (bsz, seq_len, in_features) → (bsz, seq_len, out_features)
    delta is an alias so existing call sites are unchanged.
    """
    def __init__(self, bsz, in_features, out_features, rank):
        super().__init__()
        self._bound = 1.0 / math.sqrt(in_features)
        self.A = nn.Parameter(
            torch.empty(bsz, rank, in_features).uniform_(-self._bound, self._bound))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))

    def reset(self):
        with torch.no_grad():
            self.A.uniform_(-self._bound, self._bound)
            self.B.zero_()

    def forward(self, x):
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    delta = forward  # alias for existing call sites in Block/MLP/GPT forward


class BatchedTTTLoRA(nn.Module):
    """Batched LoRA adapter container (PR #1727) — bsz document slots per layer."""
    def __init__(self, bsz, base_model, rank, k_lora=True, mlp_lora=True, o_lora=True):
        super().__init__()
        dim = base_model.tok_emb.embedding_dim
        n_layers = len(base_model.blocks)
        num_heads = base_model.blocks[0].attn.num_heads
        num_kv_heads = base_model.blocks[0].attn.num_kv_heads
        kv_dim = num_kv_heads * (dim // num_heads)
        mlp_hidden = base_model.blocks[0].mlp.fc.weight.shape[0]
        vocab_size = base_model.tok_emb.num_embeddings
        self.bsz = bsz
        self.q_loras   = nn.ModuleList([BatchedLinearLoRA(bsz, dim, dim,         rank) for _ in range(n_layers)])
        self.v_loras   = nn.ModuleList([BatchedLinearLoRA(bsz, dim, kv_dim,      rank) for _ in range(n_layers)])
        self.k_loras   = (nn.ModuleList([BatchedLinearLoRA(bsz, dim, kv_dim,     rank) for _ in range(n_layers)])
                          if k_lora else None)
        self.o_loras   = (nn.ModuleList([BatchedLinearLoRA(bsz, dim, dim,        rank) for _ in range(n_layers)])
                          if o_lora else None)
        self.mlp_loras = (nn.ModuleList([BatchedLinearLoRA(bsz, dim, mlp_hidden, rank) for _ in range(n_layers)])
                          if mlp_lora else None)
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab_size, rank)

    def reset(self):
        for l in self.q_loras:  l.reset()
        for l in self.v_loras:  l.reset()
        if self.k_loras:
            for l in self.k_loras:  l.reset()
        if self.o_loras:
            for l in self.o_loras:  l.reset()
        if self.mlp_loras:
            for l in self.mlp_loras: l.reset()
        self.lm_head_lora.reset()


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult,
                 rope_base, qk_gain_init, train_seq_len,
                 layer_idx=0, ln_scale=False,
                 gated_attn=False, gated_attn_init_std=0.01):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads,
                                        rope_base, qk_gain_init, train_seq_len,
                                        gated_attn=gated_attn,
                                        gated_attn_init_std=gated_attn_init_std)
        self.mlp  = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False

    def forward(self, x, x0, ttt_lora=None, layer_idx=0):
        mix  = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        norm_in = self.attn_norm(x_in) * self.ln_scale_factor
        if ttt_lora is not None:
            attn_out = self.attn(
                norm_in,
                q_lora=ttt_lora.q_loras[layer_idx],
                k_lora=ttt_lora.k_loras[layer_idx] if ttt_lora.k_loras is not None else None,
                v_lora=ttt_lora.v_loras[layer_idx],
                o_lora=ttt_lora.o_loras[layer_idx] if ttt_lora.o_loras is not None else None,
            )
        else:
            attn_out = self.attn(norm_in)
        mlp_lora_i = (ttt_lora.mlp_loras[layer_idx]
                      if ttt_lora is not None and ttt_lora.mlp_loras is not None else None)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor, mlp_lora=mlp_lora_i)
            x_out = (x_in
                     + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
                     + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :]  * mlp_out)
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * \
                    self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, mlp_lora=mlp_lora_i)
        return x_out


class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f'logit_softcap must be positive, got {h.logit_softcap}')
        self.tie_embeddings     = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap      = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj  = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None
            self.head_proj  = None
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList([
            Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult,
                  h.rope_base, h.qk_gain_init, h.train_seq_len,
                  layer_idx=i, ln_scale=h.ln_scale,
                  gated_attn=h.gated_attn_enabled,
                  gated_attn_init_std=h.gated_attn_init_std)
            for i in range(h.num_layers)
        ])
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary    = Rotary(head_dim, base=h.rope_base,
                                              train_seq_len=h.train_seq_len,
                                              rope_dims=h.rope_dims)
        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.xsa_last_n > 0:
            for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        if h.parallel_residual_start >= 0:
            for i in range(h.parallel_residual_start, h.num_layers):
                self.blocks[i].parallel = True
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg   = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32))
        self.skip_gates   = (nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32))
                             if h.skip_gates_enabled else None)
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, '_zero_init', False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def forward_hidden(self, input_ids, ttt_lora=None):
        x  = self.tok_emb(input_ids)
        x  = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
        dec_iter = self.decoder_indices if self.looping_active else range(self.num_encoder_layers,
                                                                          self.num_encoder_layers + self.num_decoder_layers)
        for i in enc_iter:
            x = self.blocks[i](x, x0, ttt_lora=ttt_lora, layer_idx=i)
            skips.append(x)
        for skip_idx, i in enumerate(dec_iter):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else:
                    x = x + scaled_skip
            x = self.blocks[i](x, x0, ttt_lora=ttt_lora, layer_idx=i)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        return x

    def logits_from_hidden(self, x):
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward_logits(self, input_ids, lact_adapter=None, ttt_lora=None):
        x = self.forward_hidden(input_ids, ttt_lora=ttt_lora)
        if lact_adapter is not None:
            x = x + lact_adapter(x)
        if ttt_lora is not None:
            if self.tie_embeddings:
                raw = F.linear(x, self.tok_emb.weight)
            else:
                raw = self.lm_head(x)
            raw = raw + ttt_lora.lm_head_lora.delta(x)
            return self.logit_softcap * torch.tanh(raw.float() / self.logit_softcap)
        return self.logits_from_hidden(x)

    def forward(self, input_ids, target_ids, ttt_lora=None):
        logits = self.forward_logits(input_ids, ttt_lora=ttt_lora)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1), reduction='mean')

    def forward_ttt(self, input_ids, target_ids, lora):
        """Per-token cross-entropy losses → (bsz, seqlen) for batched document TTT."""
        logits = self.forward_logits(input_ids, ttt_lora=lora)
        bsz, seqlen = target_ids.shape
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction='none',
        ).reshape(bsz, seqlen)


# ── LaCT fast-weight adapter ──────────────────────────────────────────────────

class LaCTFastWeightAdapter(nn.Module):
    """Lightweight fast-weight adapter applied to hidden states before the output head.

    Weights stay in GPU memory only during eval; they are NOT serialised into the
    submission artifact, so they add zero bytes to the 16 MB budget.
    Updated chunk-by-chunk on already-scored tokens (score-first, legal).
    """
    def __init__(self, dim, state_dim=128, kind='swiglu', scale=0.08, init_std=0.02):
        super().__init__()
        self.dim       = dim
        self.state_dim = state_dim
        self.kind      = kind
        self.scale     = scale
        if kind == 'swiglu':
            self.w1 = nn.Parameter(torch.randn(dim, state_dim) * init_std)
            self.w3 = nn.Parameter(torch.randn(dim, state_dim) * init_std)
            self.w2 = nn.Parameter(torch.zeros(state_dim, dim))
        elif kind == 'linear':
            self.w1 = nn.Parameter(torch.randn(dim, state_dim) * init_std)
            self.w2 = nn.Parameter(torch.zeros(state_dim, dim))
        else:
            raise ValueError(f'Unknown LACT_FAST_WEIGHT={kind!r}; use swiglu or linear')
        self._init_norms = {
            name: p.detach().float().norm(dim=0, keepdim=True).clamp_min(1e-6)
            for name, p in self.named_parameters()
        }

    def forward(self, x):
        if self.kind == 'swiglu':
            h = F.silu(x @ self.w1.to(x.dtype)) * (x @ self.w3.to(x.dtype))
            return self.scale * (h @ self.w2.to(x.dtype))
        h = x @ self.w1.to(x.dtype)
        return self.scale * (h @ self.w2.to(x.dtype))

    @torch.no_grad()
    def normalize_(self):
        for name, p in self.named_parameters():
            if name == 'w2':
                continue
            n = p.detach().float().norm(dim=0, keepdim=True).clamp_min(1e-6)
            p.mul_((self._init_norms[name].to(p.device) / n).to(p.dtype))


def build_lact_adapter(h, base_model, device):
    dim = base_model.tok_emb.weight.size(1)
    return LaCTFastWeightAdapter(
        dim, h.lact_state_dim, h.lact_fast_weight, h.lact_scale, h.lact_init_std
    ).to(device).bfloat16()


# ── Muon / Newton-Schulz ──────────────────────────────────────────────────────

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
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


def lact_muon_step(adapter, h, states, lr=None):
    step_lr = h.lact_lr if lr is None else lr
    with torch.no_grad():
        for p in adapter.parameters():
            if p.grad is None:
                continue
            buf = states.setdefault(p, torch.zeros_like(p))
            buf.mul_(h.lact_momentum).add_(p.grad)
            g = buf
            if g.ndim == 2:
                g = zeropower_via_newtonschulz5(g, steps=5)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
            p.add_(g.to(p.dtype), alpha=-step_lr)
            p.grad = None
    if h.lact_normalize:
        adapter.normalize_()


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps,
                 nesterov=True, weight_decay=0.0, row_normalize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                     backend_steps=backend_steps,
                                     nesterov=nesterov,
                                     weight_decay=weight_decay,
                                     row_normalize=row_normalize))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size   = dist.get_world_size() if distributed else 1
        rank         = dist.get_rank()       if distributed else 0
        for group in self.param_groups:
            params = group['params']
            if not params:
                continue
            lr            = group['lr']
            momentum      = group['momentum']
            backend_steps = group['backend_steps']
            nesterov      = group['nesterov']
            total_params  = sum(int(p.numel()) for p in params)
            updates_flat  = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g     = p.grad
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    if group.get('row_normalize', False):
                        row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                        g = g / row_norms.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd   = group.get('weight_decay', 0.0)
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
        'CONTROL_TENSOR_NAME_PATTERNS',
        'attn_scale,mlp_scale,resid_mix,q_gain,'
        'skip_weight,skip_weights,skip_gates'
    ).split(',') if p
)


class Optimizers:
    def __init__(self, h, base_model):
        block_named_params = list(base_model.blocks.named_parameters())
        matrix_params = [p for name, p in block_named_params
                         if p.ndim == 2
                         and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
        scalar_params  = [p for name, p in block_named_params
                          if p.ndim < 2
                          or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        token_lr  = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [{'params': [base_model.tok_emb.weight], 'lr': token_lr, 'base_lr': token_lr}]
        self.optimizer_tok    = torch.optim.AdamW(tok_params, betas=(h.beta1, h.beta2),
                                                   eps=h.adam_eps, weight_decay=h.embed_wd, fused=True)
        self.optimizer_muon   = Muon(matrix_params, lr=h.matrix_lr, momentum=h.muon_momentum,
                                     backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd,
                                     row_normalize=h.muon_row_normalize)
        for group in self.optimizer_muon.param_groups:
            group['base_lr'] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{'params': scalar_params, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [{'params': [base_model.lm_head.weight], 'lr': h.head_lr, 'base_lr': h.head_lr}],
                betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True)
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
        if (param.ndim < 2
                or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) \
                and param.dtype != torch.float32:
            param.data = param.data.float()


# ── Hessian collection + GPTQ ────────────────────────────────────────────────

def classify_param(name):
    if 'tok_emb' in name or 'lm_head' in name:
        return 'embed'
    if '.mlp.' in name:
        return 'mlp'
    if '.attn.' in name or ('.proj.' in name and '.mlp.' not in name):
        return 'attn'
    return 'other'


def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians: dict = {}
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
            cat = classify_param(name + '.weight')
            if cat in ('mlp', 'attn'):
                hooks.append(module.register_forward_hook(make_hook(name + '.weight')))
    if model.tie_embeddings:
        hook_module = model.head_proj if model.head_proj is not None else model.final_norm
        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)
            return hook_fn
        hooks.append(hook_module.register_forward_hook(make_output_hook('tok_emb.weight')))
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
    perm    = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm  = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    s  = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q  = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2       = min(i1 + block_size, cols)
        W_block  = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d     = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s


# ── entropy-constrained group allocator ──────────────────────────────────────

def _code_wrapper_sizes(code, h):
    sizes = {}
    if 'source' in h.allocator_code_wrappers:
        sizes['source'] = len(code.encode('utf-8'))
    if 'lzma_raw_b85_exec' in h.allocator_code_wrappers:
        import base64
        payload = lzma.compress(code.encode('utf-8'),
                                format=lzma.FORMAT_RAW,
                                filters=[{'id': lzma.FILTER_LZMA2}])
        wrapped = (
            "import lzma as L,base64 as B\n"
            "exec(L.decompress(B.b85decode("
            + repr(base64.b85encode(payload).decode('ascii'))
            + "),format=L.FORMAT_RAW,filters=[{'id':L.FILTER_LZMA2}]))"
        )
        sizes['lzma_raw_b85_exec'] = len(wrapped.encode('utf-8'))
    if not sizes:
        sizes['source'] = len(code.encode('utf-8'))
    return sizes


def _write_tsv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(keys) + '\n')
        for row in rows:
            f.write('\t'.join(str(row.get(k, '')) for k in keys) + '\n')


def _group_trace(H, c1, c2):
    if H is None:
        return 1.0
    d = H.diag().float()[c1:c2]
    return float(d.mean().clamp_min(1e-12).item())


def _group_error(W, W_hat, H, c1, c2):
    return _group_trace(H, c1, c2) * float((W.float() - W_hat.float()).pow(2).mean().item())


def _dequant_group(q, s):
    return q.float() * s.float().view(q.shape[0], 1)


def _entropy_proxy_bytes(q_int8):
    flat   = q_int8.reshape(-1).numpy().astype(np.int16) + 128
    counts = np.bincount(flat, minlength=256).astype(np.float64)
    counts = counts[counts > 0]
    n      = counts.sum()
    probs  = counts / n
    bits_per_sym = -float((probs * np.log2(probs)).sum())
    return float(bits_per_sym * n / 8.0) + 16.0 * q_int8.shape[0]


def _allocator_bits_for_name(name, h):
    if 'tok_emb' in name:
        return h.allocator_embed_bits
    cat = classify_param(name)
    if cat == 'mlp'  and h.allocator_mlp_bits  is not None:
        return h.allocator_mlp_bits
    if cat == 'attn' and h.allocator_attn_bits is not None:
        return h.allocator_attn_bits
    return h.allocator_matrix_bits


def _allocator_sigmas_for_name(name, h):
    if 'tok_emb' in name:
        return h.allocator_embed_sigmas
    cat = classify_param(name)
    if cat == 'mlp'  and h.allocator_mlp_sigmas  is not None:
        return h.allocator_mlp_sigmas
    if cat == 'attn' and h.allocator_attn_sigmas is not None:
        return h.allocator_attn_sigmas
    return h.allocator_matrix_sigmas


def _precompute_group_options(name, t, H, h):
    W          = t.detach().cpu().float().contiguous()
    rows, cols = W.shape
    group_cols = max(1, h.allocator_group_cols)
    groups     = []
    for c1 in range(0, cols, group_cols):
        c2  = min(c1 + group_cols, cols)
        Wg  = W[:, c1:c2].contiguous()
        Hg  = H[c1:c2, c1:c2].contiguous() if H is not None else torch.eye(c2 - c1)
        opts = []
        for bits in _allocator_bits_for_name(name, h):
            clip_range = 2 ** (bits - 1) - 1
            for sigma in _allocator_sigmas_for_name(name, h):
                q, s    = gptq_quantize_weight(Wg, Hg, clip_sigmas=sigma,
                                               clip_range=clip_range,
                                               block_size=min(128, c2 - c1))
                recon   = _dequant_group(q, s)
                err     = _group_error(Wg, recon, H, c1, c2)
                proxy_bits = (_entropy_proxy_bytes(q) * 8.0
                              if h.allocator_use_entropy_proxy
                              else float(bits * Wg.numel() + 16 * rows))
                opts.append({'bits': int(bits), 'sigma': float(sigma),
                             'q': q.contiguous(), 'scale': s.contiguous(),
                             'error': err, 'proxy_bits': proxy_bits})
        opts.sort(key=lambda x: (x['error'], x['proxy_bits']))
        groups.append(opts)
    return {'name': name, 'shape': tuple(W.shape), 'groups': groups, 'group_cols': group_cols}


def _quantize_gate_int8_row(w):
    """Symmetric int8-per-row quantization for small gate tensors (QuantGate)."""
    W = w.float().contiguous()
    row_max = W.abs().amax(dim=1).clamp_min(1e-10)
    s = (row_max / 127.0).to(torch.float16)
    q = torch.clamp(torch.round(W / s.float().view(-1, 1)), -127, 127).to(torch.int8)
    return q, s


def _lqer_pack(A, B, bits):
    rng = 2 ** (bits - 1) - 1
    sA = (A.abs().amax(dim=1).clamp_min(1e-10) / rng).to(torch.float16)
    sB = (B.abs().amax(dim=1).clamp_min(1e-10) / rng).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float().view(-1, 1)), -rng, rng).to(torch.int8)
    qB = torch.clamp(torch.round(B / sB.float().view(-1, 1)), -rng, rng).to(torch.int8)
    return qA, sA, qB, sB


def _lqer_pack_asym(A, B, g=64):
    sA = (A.abs().amax().clamp_min(1e-10) / 1.5).to(torch.float16)
    qA = torch.clamp(torch.round(A / sA.float()), -2, 1).to(torch.int8)
    Bf = B.reshape(-1, g)
    Bmax = Bf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    sB = (Bmax / 7.5).to(torch.float16).reshape(-1)
    qB = torch.clamp(torch.round(Bf / sB.float().reshape(-1, 1)), -8, 7).to(
        torch.int8
    ).reshape(B.shape)
    return qA, sA, qB, sB


def _build_allocator_obj(state_dict, large_entries, selection, h):
    result = {}
    meta   = {}
    score  = 0.0
    proxy_bits = 0.0
    counts = collections.Counter()
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if name not in large_entries:
            if (h.gated_attn_quant_gate and t.is_floating_point() and t.ndim == 2
                    and name.endswith('.attn_gate_w') and 1024 <= t.numel() <= 8192):
                gq, gs = _quantize_gate_int8_row(t)
                result[name + '.gq'] = gq
                result[name + '.gs'] = gs
                meta[name] = 'gate_int8_row'
            else:
                result[name] = t.to(torch.float16) if t.is_floating_point() else t
                meta[name]   = 'passthrough (float16)'
            continue
        entry      = large_entries[name]
        rows, cols = entry['shape']
        q_full     = torch.empty((rows, cols), dtype=torch.int8)
        scale_full = torch.empty((rows, len(entry['groups'])), dtype=torch.float16)
        bits_list  = []
        sigmas_list = []
        for gi, opts in enumerate(entry['groups']):
            opt = opts[selection[name][gi]]
            c1  = gi * entry['group_cols']
            c2  = min(c1 + entry['group_cols'], cols)
            q_full[:, c1:c2] = opt['q']
            scale_full[:, gi] = opt['scale']
            bits_list.append(opt['bits'])
            sigmas_list.append(round(opt['sigma'], 6))
            score      += opt['error']
            proxy_bits += opt['proxy_bits']
            counts[f"int{opt['bits']}"] += c2 - c1
        result[name + '.q']    = q_full.contiguous()
        result[name + '.scale'] = scale_full.contiguous()
        meta[name] = {'format': 'group_gptq_v1', 'group_cols': entry['group_cols'],
                      'bits': bits_list, 'sigmas': sigmas_list}
    return ({'w': result, 'm': meta,
             'allocator': {'format': 'entropy_constrained_group_gptq_v1',
                           'score': score, 'proxy_bits': proxy_bits,
                           'bit_columns': dict(counts)}},
            score, proxy_bits, dict(counts))


def _selection_for_lambda(large_entries, lam):
    selection = {}
    for name, entry in large_entries.items():
        selection[name] = [
            min(range(len(opts)), key=lambda i: opts[i]['error'] + lam * opts[i]['proxy_bits'])
            for opts in entry['groups']
        ]
    return selection


def gptq_entropy_allocator_quantize(state_dict, hessians, h, code):
    large_entries = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.ndim != 2 or t.numel() <= 65536:
            continue
        if name not in hessians:
            log(f'allocator:missing_hessian name:{name} fallback_passthrough')
            continue
        log(f'allocator:precompute name:{name} shape:{tuple(t.shape)} '
            f'groups:{math.ceil(t.shape[1] / max(1, h.allocator_group_cols))}')
        large_entries[name] = _precompute_group_options(name, t, hessians[name], h)
    code_sizes = _code_wrapper_sizes(code, h)
    rows  = []
    best  = None
    seen  = set()
    for wrapper, code_bytes in sorted(code_sizes.items(), key=lambda kv: kv[1]):
        for lam in h.allocator_lambdas:
            selection = _selection_for_lambda(large_entries, lam)
            key = (wrapper, tuple((n, tuple(v)) for n, v in sorted(selection.items())))
            if key in seen:
                continue
            seen.add(key)
            obj, score, pb, counts = _build_allocator_obj(state_dict, large_entries, selection, h)
            buf   = io.BytesIO()
            torch.save(obj, buf)
            raw   = buf.getvalue()
            blob  = _compress(raw, h.compressor)
            model_bytes = len(blob)
            total_bytes = model_bytes + code_bytes
            row = {'lambda': lam, 'wrapper': wrapper,
                   'code_bytes': code_bytes, 'model_bytes': model_bytes,
                   'total_bytes': total_bytes, 'score': f'{score:.9e}',
                   'proxy_bits': f'{pb:.0f}',
                   'bit_columns': ','.join(f'{k}:{v}' for k, v in sorted(counts.items()))}
            rows.append(row)
            valid = total_bytes <= h.artifact_target_bytes
            if (best is None
                    or (valid and (best[0] > h.artifact_target_bytes or score < best[1]))
                    or (best[0] > h.artifact_target_bytes and total_bytes < best[0])):
                best = (total_bytes, score, model_bytes, code_bytes, wrapper, lam, obj, raw, blob, row)
    if h.is_main_process:
        os.makedirs('logs', exist_ok=True)
        _write_tsv('logs/allocator_candidates.tsv', rows)
        log(f'allocator_candidates:logs/allocator_candidates.tsv candidates:{len(rows)}')
    if best is None:
        raise RuntimeError('allocator produced no candidates')
    total_bytes, score, model_bytes, code_bytes, wrapper, lam, obj, raw, blob, row = best
    obj['allocator'].update({
        'selected_lambda': lam, 'selected_wrapper': wrapper,
        'selected_code_bytes': code_bytes, 'selected_model_bytes': model_bytes,
        'selected_total_bytes': total_bytes, 'target_total_bytes': h.artifact_target_bytes,
    })
    log(f'allocator_selected lambda:{lam:g} wrapper:{wrapper} score:{score:.9e} '
        f'model_bytes:{model_bytes} code_bytes:{code_bytes} '
        f'total_bytes:{total_bytes} target_bytes:{h.artifact_target_bytes}')
    return obj, {'compressed_bytes': model_bytes, 'raw_bytes': len(raw),
                 'total_bytes': total_bytes, 'code_bytes': code_bytes,
                 'wrapper': wrapper, 'score': score, 'blob': blob}


def gptq_mixed_quantize_legacy(state_dict, hessians, h):
    result = {}
    meta = {}
    quant_gate = bool(getattr(h, "gated_attn_quant_gate", False))
    lqer_on = bool(getattr(h, "lqer_enabled", False))
    lqer_cands = {}
    for (name, tensor) in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if (
            quant_gate
            and t.is_floating_point()
            and t.ndim == 2
            and name.endswith(".attn_gate_w")
            and 32 <= t.numel() <= 8192
        ):
            gq, gs = _quantize_gate_int8_row(t)
            result[name + ".gq"] = gq
            result[name + ".gs"] = gs
            meta[name] = "gate_int8_row"
            continue
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        if "tok_emb" in name:
            cs = h.embed_clip_sigmas
        elif ".mlp." in name:
            cs = h.mlp_clip_sigmas
        elif ".attn." in name:
            cs = h.attn_clip_sigmas
        else:
            cs = h.matrix_clip_sigmas
        bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
        clip_range = 2 ** (bits - 1) - 1
        ret = gptq_quantize_weight(
            t, hessians[name], clip_sigmas=cs, clip_range=clip_range
        )
        q, s = ret
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
        if lqer_on:
            W_q = q.float() * s.float().view(-1, 1)
            E = t.float() - W_q
            lqer_cands[name] = (E, float(E.norm()))
    if lqer_on and lqer_cands:
        top = sorted(lqer_cands.items(), key=lambda kv: -kv[1][1])[: h.lqer_top_k]
        asym_on = bool(getattr(h, "lqer_asym_enabled", False))
        asym_g = int(getattr(h, "lqer_asym_group", 64))
        for (name, (E, _)) in top:
            U, S, Vh = torch.linalg.svd(E, full_matrices=False)
            r = min(h.lqer_rank, S.numel())
            A = (U[:, :r] * S[:r]).contiguous()
            B = Vh[:r, :].contiguous()
            if asym_on and B.numel() % asym_g == 0:
                qA, sA, qB, sB = _lqer_pack_asym(A, B, asym_g)
                result[name + ".lqA_a"] = qA
                result[name + ".lqAs_a"] = sA
                result[name + ".lqB_a"] = qB
                result[name + ".lqBs_a"] = sB
                meta[name] = meta[name] + "+lqer_asym"
            else:
                qA, sA, qB, sB = _lqer_pack(A, B, h.lqer_factor_bits)
                result[name + ".lqA"] = qA
                result[name + ".lqAs"] = sA
                result[name + ".lqB"] = qB
                result[name + ".lqBs"] = sB
                meta[name] = meta[name] + "+lqer"
    categories = collections.defaultdict(set)
    for (name, cat) in meta.items():
        short = re.sub(r"\.\d+$", "", re.sub(r"blocks\.\d+", "blocks", name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return result, meta


def dequantize_mixed(result, meta, template_sd):
    out = {}
    for (name, orig) in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        if info == "gate_int8_row":
            gq = result[name + '.gq']
            gs = result[name + '.gs']
            out[name] = (gq.float() * gs.float().view(-1, 1)).to(orig_dtype)
            continue
        if isinstance(info, dict) and info.get('format') == 'group_gptq_v1':
            q, s       = result[name + '.q'], result[name + '.scale']
            group_cols = int(info['group_cols'])
            rows, cols = q.shape
            recon      = torch.empty((rows, cols), dtype=torch.float32)
            for gi, c1 in enumerate(range(0, cols, group_cols)):
                c2 = min(c1 + group_cols, cols)
                recon[:, c1:c2] = q[:, c1:c2].float() * s[:, gi].float().view(rows, 1)
            out[name] = recon.to(orig_dtype)
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            W = q.float() * s.float().view(q.shape[0], *[1] * (q.ndim - 1))
        else:
            W = q.float() * float(s.item())
        if "lqer_asym" in info:
            qA_t = result[name + ".lqA_a"]
            sA_t = result[name + ".lqAs_a"]
            qB_t = result[name + ".lqB_a"]
            sB_t = result[name + ".lqBs_a"]
            qA = qA_t.float() * float(sA_t)
            g_sz = qB_t.numel() // sB_t.numel()
            qB = (qB_t.reshape(-1, g_sz).float() * sB_t.float().view(-1, 1)).reshape(
                qB_t.shape
            )
            W = W + qA @ qB
        elif "lqer" in info:
            qA = result[name + ".lqA"].float() * result[name + ".lqAs"].float().view(-1, 1)
            qB = result[name + ".lqB"].float() * result[name + ".lqBs"].float().view(-1, 1)
            W = W + qA @ qB
        out[name] = W.to(orig_dtype)
    return out


# ── compression ───────────────────────────────────────────────────────────────

_BSHF_MAGIC = b'BSHF'

def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride:
        return data
    src  = np.frombuffer(data, dtype=np.uint8)
    n    = len(src)
    out  = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off:dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride  = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n       = len(payload)
    out     = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()

def _compress(data, compressor):
    data = _byte_shuffle(data)
    if compressor == 'lzma':
        return lzma.compress(data, preset=6)
    if compressor == 'brotli':
        import brotli
        return brotli.compress(data, quality=11)
    raise ValueError(f'Unknown compressor: {compressor!r}')

def _decompress(data, compressor):
    if compressor == 'lzma':
        raw = lzma.decompress(data)
    elif compressor == 'brotli':
        import brotli
        raw = brotli.decompress(data)
    else:
        raise ValueError(f'Unknown compressor: {compressor!r}')
    return _byte_unshuffle(raw)


# ── serialize / deserialize ───────────────────────────────────────────────────

def serialize(h, base_model, code):
    code_bytes = len(code.encode('utf-8'))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        log(f'Serialized model: {os.path.getsize(h.model_path)} bytes')
        log(f'Code size source: {code_bytes} bytes')
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    device = torch.device('cuda', h.local_rank)
    log('GPTQ:collecting Hessians from calibration data...')
    t0           = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians     = collect_hessians(base_model, calib_loader, h, device,
                                    n_calibration_batches=h.gptq_calibration_batches)
    log(f'GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t0:.1f}s')
    if h.export_allocator == 'entropy':
        quant_obj, quant_stats = gptq_entropy_allocator_quantize(sd_cpu, hessians, h, code)
        quant_blob      = quant_stats['blob']
        quant_file_bytes = quant_stats['compressed_bytes']
        bytes_total     = quant_stats['total_bytes']
        code_bytes      = quant_stats['code_bytes']
        quant_raw_bytes = quant_stats['raw_bytes']
    else:
        quant_result, quant_meta = gptq_mixed_quantize_legacy(sd_cpu, hessians, h)
        quant_buf  = io.BytesIO()
        torch.save({'w': quant_result, 'm': quant_meta}, quant_buf)
        quant_raw  = quant_buf.getvalue()
        quant_blob = _compress(quant_raw, h.compressor)
        quant_file_bytes = len(quant_blob)
        bytes_total     = quant_file_bytes + code_bytes
        quant_raw_bytes = len(quant_raw)
    if h.is_main_process:
        with open(h.quantized_model_path, 'wb') as f:
            f.write(quant_blob)
        log(f'Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes  raw_torch:{quant_raw_bytes}')
        log(f'Code size selected: {code_bytes} bytes')
        log(f'Total submission size quantized+{h.compressor}: {bytes_total} bytes')
    return bytes_total, quant_file_bytes


def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    with open(h.quantized_model_path, 'rb') as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location='cpu')
    deq_state = dequantize_mixed(quant_state['w'], quant_state['m'], sd_cpu)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model


# ── eval helpers ─────────────────────────────────────────────────────────────

def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb  = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def _allreduce_scalars(*tensors):
    if dist.is_available() and dist.is_initialized():
        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)


def eval_val(h, device, val_data, model):
    seq_len             = h.eval_seq_len
    local_batch_tokens  = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError('VAL_BATCH_TOKENS too small for distributed eval')
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs       = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank       // h.world_size
    seq_end   = total_seqs * (h.rank + 1) // h.world_size
    loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    tok_count  = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end   = batch_seq_end   * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            n = float(y.numel())
            loss_sum  += batch_loss.to(torch.float64) * n
            tok_count += n
            prev_ids = x.reshape(-1)
            tgt_ids  = y.reshape(-1)
            tb = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (val_data.has_leading_space_lut[tgt_ids]
                   & ~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            byte_count += tb.to(torch.float64).sum()
    _allreduce_scalars(loss_sum, tok_count, byte_count)
    model.train()
    return _loss_bpb(loss_sum, tok_count, byte_count)


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    logits_fn    = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    seq_len      = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride)
                     if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank       // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    tok_count  = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz      = len(batch_ws)
            x_batch  = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch  = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens    = []
            for i, ws in enumerate(batch_ws):
                we   = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  y_batch.reshape(-1), reduction='none').reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s    = 0 if ws == 0 else context_size
                scored_nll  = nll[i, s:wlen].to(torch.float64)
                loss_sum   += scored_nll.sum()
                tok_count  += float(wlen - s)
                tgt  = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb   = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb  += (val_data.has_leading_space_lut[tgt]
                        & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    _allreduce_scalars(loss_sum, tok_count, byte_count)
    base_model.train()
    return _loss_bpb(loss_sum, tok_count, byte_count)


def _score_chunk_windows(base_model, logits_fn, val_data, my_windows,
                         seq_len, context_size, total_tokens,
                         loss_sum, tok_count, byte_count, device, batch_seqs):
    """Score windows in no_grad; accumulate loss/token/byte counts in-place."""
    with torch.no_grad():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz      = len(batch_ws)
            x_batch  = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch  = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens    = []
            for i, ws in enumerate(batch_ws):
                we   = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                tok = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = tok[:-1]
                y_batch[i, :wlen] = tok[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  y_batch.reshape(-1), reduction='none').reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s    = 0 if ws == 0 else context_size
                loss_sum  += nll[i, s:wlen].to(torch.float64).sum()
                tok_count += float(wlen - s)
                tgt  = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb   = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb  += (val_data.has_leading_space_lut[tgt]
                        & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()


# ── PR #1727 batched-document TTT helpers ─────────────────────────────────────

def _find_docs(all_tokens):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = (
            int(bos_positions[i + 1])
            if i + 1 < len(bos_positions)
            else all_tokens.numel()
        )
        if i + 1 < len(bos_positions):
            end += 1
        assert end - start >= 2
        docs.append((start, end - start))
    return docs


def _build_ttt_global_batches(doc_entries, h, ascending=False):
    batch_size = h.ttt_batch_size
    global_doc_entries = sorted(doc_entries, key=lambda x: x[1][1])
    global_batches = [
        global_doc_entries[i : i + batch_size]
        for i in range(0, len(global_doc_entries), batch_size)
    ]
    indexed = list(enumerate(global_batches))
    if not ascending:
        indexed.sort(key=lambda ib: -max(dl for _, (_, dl) in ib[1]))
    return indexed


def _init_batch_counter(path):
    with open(path, "wb") as f:
        f.write((0).to_bytes(4, "little"))


def _claim_next_batch(counter_path, queue_len):
    try:
        with open(counter_path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            idx = int.from_bytes(f.read(4), "little")
            f.seek(0)
            f.write((idx + 1).to_bytes(4, "little"))
            f.flush()
    except FileNotFoundError:
        return queue_len
    return idx


def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_start = ci * chunk_size
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def _accumulate_bpb(
    ptl,
    x,
    y,
    chunk_offsets,
    chunk_lens,
    pos_idx,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    loss_sum,
    byte_sum,
    token_count,
):
    pos = pos_idx[: x.size(1)].unsqueeze(0)
    mask = (
        (chunk_lens.unsqueeze(1) > 0)
        & (pos >= chunk_offsets.unsqueeze(1))
        & (pos < (chunk_offsets + chunk_lens).unsqueeze(1))
    )
    mask_f64 = mask.to(torch.float64)
    tok_bytes = base_bytes_lut[y].to(torch.float64)
    tok_bytes += (has_leading_space_lut[y] & ~is_boundary_token_lut[x]).to(
        torch.float64
    )
    loss_sum += (ptl.to(torch.float64) * mask_f64).sum()
    byte_sum += (tok_bytes * mask_f64).sum()
    token_count += chunk_lens.to(torch.float64).sum()


def _split_doc_entries_for_phased(doc_entries, prefix_docs):
    prefix_docs = max(0, min(len(doc_entries), int(prefix_docs)))
    return doc_entries[:prefix_docs], doc_entries[prefix_docs:]


def _add_to_counter(path, delta):
    try:
        with open(path, "r+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            cur = int.from_bytes(f.read(8), "little", signed=True)
            cur += int(delta)
            f.seek(0)
            f.write(int(cur).to_bytes(8, "little", signed=True))
            f.flush()
            return cur
    except FileNotFoundError:
        return int(delta)


def _init_int64_counter(path):
    with open(path, "wb") as f:
        f.write((0).to_bytes(8, "little", signed=True))


def _select_ttt_doc_entries(docs, h):
    doc_entries = list(enumerate(docs))
    if h.val_doc_fraction < 1.0:
        sample_n = max(1, int(round(len(docs) * h.val_doc_fraction)))
        sampled_indices = sorted(
            random.Random(h.seed).sample(range(len(docs)), sample_n)
        )
        return [(i, docs[i]) for i in sampled_indices]
    return doc_entries


def _split_flat_windows_on_boundaries(x_flat, y_flat, bos_id, max_seq_len):
    starts = [0]
    starts.extend(
        int(i) for i in (x_flat == bos_id).nonzero(as_tuple=True)[0].tolist() if int(i) > 0
    )
    starts = sorted(set(starts))
    windows = []
    total = int(x_flat.numel())
    for si, start in enumerate(starts):
        end = starts[si + 1] if si + 1 < len(starts) else total
        if end <= start:
            continue
        seg_x = x_flat[start:end]
        seg_y = y_flat[start:end]
        for off in range(0, int(seg_x.numel()), max_seq_len):
            x_win = seg_x[off : off + max_seq_len]
            y_win = seg_y[off : off + max_seq_len]
            if x_win.numel() > 0:
                windows.append((x_win, y_win))
    return windows


def _masked_doc_batch_loss(base_model, x_flat, y_flat, bos_id, seq_len):
    windows = _split_flat_windows_on_boundaries(x_flat, y_flat, bos_id, seq_len)
    if not windows:
        raise RuntimeError("Document-boundary-respecting TTT produced no token windows")
    max_len = max(int(x_win.numel()) for x_win, _ in windows)
    bsz = len(windows)
    x_batch = torch.full((bsz, max_len), bos_id, device=x_flat.device, dtype=torch.int64)
    y_batch = torch.full((bsz, max_len), bos_id, device=y_flat.device, dtype=torch.int64)
    mask = torch.zeros((bsz, max_len), device=x_flat.device, dtype=torch.bool)
    for wi, (x_win, y_win) in enumerate(windows):
        wlen = int(x_win.numel())
        x_batch[wi, :wlen] = x_win
        y_batch[wi, :wlen] = y_win
        mask[wi, :wlen] = True
    logits = base_model.forward_logits(x_batch)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        y_batch.reshape(-1),
        reduction="none",
    ).reshape(bsz, max_len)
    mask_f = mask.to(nll.dtype)
    return (nll * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def train_val_ttt_global_sgd_distributed(h, device, val_data, base_model, val_tokens, batch_seqs=None):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    base_model.eval()
    seq_len = h.eval_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = h.global_ttt_chunk_tokens
    batch_seqs = h.global_ttt_batch_seqs if batch_seqs is None else batch_seqs
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    ttt_params = [p for p in base_model.parameters()]
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(
        ttt_params, lr=h.global_ttt_lr, momentum=h.global_ttt_momentum
    )
    t_start = time.perf_counter()
    for ci in range(num_chunks):
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        is_last_chunk = ci == num_chunks - 1
        if is_last_chunk or h.global_ttt_epochs <= 0:
            continue
        base_model.train()
        chunk_seqs = (chunk_end - chunk_start) // seq_len
        if chunk_seqs <= 0:
            continue
        warmup_chunks = max(0, min(h.global_ttt_warmup_chunks, num_chunks - 1))
        if warmup_chunks > 0 and ci < warmup_chunks:
            warmup_denom = max(warmup_chunks - 1, 1)
            warmup_t = ci / warmup_denom
            lr_now = (
                h.global_ttt_warmup_start_lr
                + (h.global_ttt_lr - h.global_ttt_warmup_start_lr) * warmup_t
            )
        else:
            decay_steps = max(num_chunks - 1 - warmup_chunks, 1)
            decay_ci = max(ci - warmup_chunks, 0)
            lr_now = h.global_ttt_lr * 0.5 * (
                1.0 + math.cos(math.pi * decay_ci / decay_steps)
            )
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        my_seq_s = chunk_seqs * h.rank // h.world_size
        my_seq_e = chunk_seqs * (h.rank + 1) // h.world_size
        my_chunk_seqs = my_seq_e - my_seq_s
        for _ in range(h.global_ttt_epochs):
            for bs in range(0, my_chunk_seqs, batch_seqs):
                be = min(bs + batch_seqs, my_chunk_seqs)
                actual_bs = my_seq_s + bs
                start_tok = chunk_start + actual_bs * seq_len
                end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                if end_tok > val_tokens.numel():
                    continue
                local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                x_flat = local[:-1]
                y_flat = local[1:]
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if h.global_ttt_respect_doc_boundaries:
                            loss = _masked_doc_batch_loss(
                                base_model, x_flat, y_flat, BOS_ID, seq_len
                            )
                        else:
                            x = x_flat.reshape(-1, seq_len)
                            y = y_flat.reshape(-1, seq_len)
                            loss = base_model(x, y)
                loss.backward()
                if dist.is_available() and dist.is_initialized():
                    for p in ttt_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.mul_(1.0 / h.world_size)
                if h.global_ttt_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ttt_params, h.global_ttt_grad_clip)
                optimizer.step()
        base_model.eval()
        if h.rank == 0:
            elapsed = time.perf_counter() - t_start
            log(
                f"tttg: c{ci+1}/{num_chunks} lr:{lr_now:.6f} t:{elapsed:.1f}s"
            )
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()


def eval_val_ttt_phased(h, base_model, device, val_data, forward_ttt_train):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    all_tokens = val_data.val_tokens
    all_tokens_idx = all_tokens.to(torch.int32)
    docs = _find_docs(all_tokens)
    doc_entries = _select_ttt_doc_entries(docs, h)
    prefix_doc_limit = max(0, min(len(doc_entries), int(h.phased_ttt_prefix_docs)))
    num_phases = max(1, int(h.phased_ttt_num_phases))
    phase_boundaries = []
    for pi in range(num_phases):
        boundary = prefix_doc_limit * (pi + 1) // num_phases
        phase_boundaries.append(boundary)
    current_phase = 0
    current_phase_boundary = phase_boundaries[0]
    log(
        "ttt_phased:"
        f" total_docs:{len(doc_entries)} prefix_docs:{prefix_doc_limit} "
        f"suffix_docs:{len(doc_entries) - prefix_doc_limit}"
        f" num_phases:{num_phases} boundaries:{phase_boundaries}"
    )
    chunk_size, eval_seq_len = h.ttt_chunk_size, h.ttt_eval_seq_len
    eval_batch_set = None
    if h.ttt_eval_batches:
        eval_batch_set = set(int(x) for x in h.ttt_eval_batches.split(",") if x.strip())
    use_ascending = eval_batch_set is not None
    global_batches_sorted = _build_ttt_global_batches(
        doc_entries, h, ascending=use_ascending
    )
    queue_len = len(global_batches_sorted)
    counter_path = f"/tmp/ttt_counter_{h.run_id}"
    prefix_counter_path = f"/tmp/ttt_prefix_counter_{h.run_id}"
    pause_flag_path = f"/tmp/ttt_pause_flag_{h.run_id}"
    if h.rank == 0:
        _init_batch_counter(counter_path)
        _init_int64_counter(prefix_counter_path)
        try:
            os.remove(pause_flag_path)
        except FileNotFoundError:
            pass
    if dist.is_available() and dist.is_initialized():
        path_list = [counter_path, prefix_counter_path, pause_flag_path]
        dist.broadcast_object_list(path_list, src=0)
        counter_path, prefix_counter_path, pause_flag_path = path_list
        dist.barrier()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    t_start = time.perf_counter()
    reusable_lora = BatchedTTTLoRA(
        h.ttt_batch_size, base_model, h.ttt_lora_rank,
        k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
    ).to(device)

    def _build_opt(lora):
        if h.ttt_optimizer == "sgd":
            return torch.optim.SGD(
                lora.parameters(), lr=h.ttt_lora_lr,
                momentum=h.ttt_beta1, weight_decay=h.ttt_weight_decay,
            )
        return torch.optim.AdamW(
            lora.parameters(), lr=h.ttt_lora_lr,
            betas=(h.ttt_beta1, h.ttt_beta2),
            eps=1e-10, weight_decay=h.ttt_weight_decay, fused=True,
        )

    reusable_opt = _build_opt(reusable_lora)
    local_scored_docs = []
    global_ttt_done = prefix_doc_limit == 0
    try:
        while True:
            queue_idx = _claim_next_batch(counter_path, queue_len)
            if queue_idx >= queue_len:
                break
            orig_batch_idx, batch_entries = global_batches_sorted[queue_idx]
            batch = [doc for _, doc in batch_entries]
            bsz = len(batch)
            prev_loss = loss_sum.item()
            prev_bytes = byte_sum.item()
            prev_tokens = token_count.item()
            if bsz == reusable_lora.bsz:
                reusable_lora.reset()
                for s in reusable_opt.state.values():
                    for k, v in s.items():
                        if isinstance(v, torch.Tensor):
                            v.zero_()
                        elif k == "step":
                            s[k] = 0
                cur_lora = reusable_lora
                cur_opt = reusable_opt
            else:
                cur_lora = BatchedTTTLoRA(
                    bsz, base_model, h.ttt_lora_rank,
                    k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
                ).to(device)
                cur_opt = _build_opt(cur_lora)
            pred_lens = [doc_len - 1 for _, doc_len in batch]
            num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
            max_nc = max(num_chunks)
            num_chunks_t = torch.tensor(num_chunks, dtype=torch.int64, device=device)
            for ci in range(max_nc):
                active = [ci < nc for nc in num_chunks]
                needs_train = any(ci < nc - 1 for nc in num_chunks)
                tok_starts = torch.zeros(bsz, dtype=torch.int64)
                tok_wls = torch.zeros(bsz, dtype=torch.int64)
                chunk_offsets_cpu = torch.zeros(bsz, dtype=torch.int64)
                chunk_lens_cpu = torch.zeros(bsz, dtype=torch.int64)
                for b in range(bsz):
                    if not active[b]:
                        continue
                    doc_start, doc_len = batch[b]
                    win_start, win_len, chunk_offset, chunk_len = _compute_chunk_window(
                        ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len
                    )
                    tok_starts[b] = doc_start + win_start
                    tok_wls[b] = win_len
                    chunk_offsets_cpu[b] = chunk_offset
                    chunk_lens_cpu[b] = chunk_len
                _, context_size, chunk_offset, _ = _compute_chunk_window(
                    ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len
                )
                col_idx = torch.arange(context_size + 1)
                idx = tok_starts.unsqueeze(1) + col_idx.unsqueeze(0)
                idx.clamp_(max=all_tokens.numel() - 1)
                gathered_gpu = all_tokens_idx[idx].to(
                    device=device, dtype=torch.int64, non_blocking=True
                )
                valid = (col_idx[:context_size].unsqueeze(0) < tok_wls.unsqueeze(1)).to(
                    device, non_blocking=True
                )
                chunk_offsets = chunk_offsets_cpu.to(device, non_blocking=True)
                chunk_lens = chunk_lens_cpu.to(device, non_blocking=True)
                x = torch.where(valid, gathered_gpu[:, :context_size], 0)
                y = torch.where(valid, gathered_gpu[:, 1 : context_size + 1], 0)
                ctx_pos = torch.arange(context_size, device=device, dtype=torch.int64)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
                with torch.no_grad():
                    _accumulate_bpb(
                        per_tok_loss,
                        x,
                        y,
                        chunk_offsets,
                        chunk_lens,
                        ctx_pos,
                        val_data.base_bytes_lut,
                        val_data.has_leading_space_lut,
                        val_data.is_boundary_token_lut,
                        loss_sum,
                        byte_sum,
                        token_count,
                    )
                if needs_train:
                    activate_chunk_mask = (num_chunks_t - 1 > ci).float()
                    for gi in range(h.ttt_grad_steps):
                        if gi > 0:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                per_tok_loss = forward_ttt_train(x, y, lora=cur_lora)
                        per_doc = per_tok_loss[
                            :, chunk_offset : chunk_offset + chunk_size
                        ].mean(dim=-1)
                        cur_opt.zero_grad(set_to_none=True)
                        (per_doc * activate_chunk_mask).sum().backward()
                        cur_opt.step()
                else:
                    del per_tok_loss
            batch_num = orig_batch_idx + 1
            doc_lens = [dl for _, dl in batch]
            should_report = batch_num in eval_batch_set if eval_batch_set is not None else True
            if should_report:
                cur_tokens = token_count.item()
                cur_loss_val = loss_sum.item()
                cur_bytes_val = byte_sum.item()
                dt = cur_tokens - prev_tokens
                db = cur_bytes_val - prev_bytes
                if dt > 0 and db > 0:
                    b_loss = (cur_loss_val - prev_loss) / dt
                    b_bpb = b_loss / math.log(2.0) * (dt / db)
                else:
                    b_loss = b_bpb = 0.0
                r_loss = cur_loss_val / max(cur_tokens, 1)
                r_bpb = r_loss / math.log(2.0) * (cur_tokens / max(cur_bytes_val, 1))
                elapsed = time.perf_counter() - t_start
                log(
                    f"ttp: b{batch_num}/{queue_len} bl:{b_loss:.4f} bb:{b_bpb:.4f} "
                    f"rl:{r_loss:.4f} rb:{r_bpb:.4f} dl:{min(doc_lens)}-{max(doc_lens)} "
                    f"gd:{int(global_ttt_done)}"
                )
            if not global_ttt_done:
                local_scored_docs.extend(
                    (orig_batch_idx, pos, doc_start, doc_len)
                    for pos, (doc_start, doc_len) in enumerate(batch)
                )
                prefix_done = _add_to_counter(prefix_counter_path, len(batch_entries))
                if prefix_done >= current_phase_boundary:
                    try:
                        with open(pause_flag_path, "x"):
                            pass
                    except FileExistsError:
                        pass
                should_pause = os.path.exists(pause_flag_path)
                if should_pause:
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    gathered_scored_docs = [None] * h.world_size
                    if dist.is_available() and dist.is_initialized():
                        dist.all_gather_object(gathered_scored_docs, local_scored_docs)
                    else:
                        gathered_scored_docs = [local_scored_docs]
                    scored_docs_for_global = []
                    for rank_docs in gathered_scored_docs:
                        if rank_docs:
                            scored_docs_for_global.extend(rank_docs)
                    scored_docs_for_global.sort(key=lambda x: (x[0], x[1]))
                    scored_docs_for_global = scored_docs_for_global[:current_phase_boundary]
                    scored_token_chunks = [
                        val_data.val_tokens[doc_start : doc_start + doc_len]
                        for _, _, doc_start, doc_len in scored_docs_for_global
                    ]
                    if scored_token_chunks:
                        global_ttt_tokens = torch.cat(scored_token_chunks)
                    else:
                        global_ttt_tokens = val_data.val_tokens[:0]
                    if h.rank == 0:
                        prefix_done = 0
                        try:
                            with open(prefix_counter_path, "rb") as f:
                                prefix_done = int.from_bytes(
                                    f.read(8), "little", signed=True
                                )
                        except FileNotFoundError:
                            pass
                        log(
                            f"ttpp: phase:{current_phase + 1}/{num_phases} pd:{prefix_done} "
                            f"gd:{len(scored_docs_for_global)} "
                            f"t:{time.perf_counter() - t_start:.1f}s"
                        )
                    train_val_ttt_global_sgd_distributed(
                        h, device, val_data, base_model, global_ttt_tokens
                    )
                    for p in base_model.parameters():
                        p.requires_grad_(False)
                    reusable_lora = BatchedTTTLoRA(
                        h.ttt_batch_size, base_model, h.ttt_lora_rank,
                        k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
                    ).to(device)
                    reusable_opt = _build_opt(reusable_lora)
                    current_phase += 1
                    if current_phase >= num_phases:
                        global_ttt_done = True
                    else:
                        current_phase_boundary = phase_boundaries[current_phase]
                        if h.rank == 0:
                            try:
                                os.remove(pause_flag_path)
                            except FileNotFoundError:
                                pass
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    if h.rank == 0:
                        log(f"ttpr: phase:{current_phase}/{num_phases} t:{time.perf_counter() - t_start:.1f}s")
            del cur_lora, cur_opt
    finally:
        pass
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_sum)


def eval_val_lact_ttt(h, device, val_data, base_model, batch_seqs=None):
    """LaCT fast-weight TTT: adapter-only or hybrid adapter + base-model TTT.

    The fast-weight adapter lives only in GPU memory during eval; it is NOT
    included in the submission artifact, so the 16 MB budget is unaffected.
    Score-first ordering is preserved: each chunk is fully scored before the
    adapter (and optionally the base model) is updated on that chunk's tokens.
    """
    rank        = h.rank
    world_size  = h.world_size
    seq_len     = h.eval_seq_len
    stride      = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk    = h.lact_chunk_tokens
    context_size = seq_len - stride
    batch_seqs   = batch_seqs or h.lact_batch_seqs
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]
    num_chunks   = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen         = min(ws + seq_len, total_tokens) - ws
        s            = 0 if ws == 0 else context_size
        ci           = min((ws + s) // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log(f'lact_ttt:start chunks={num_chunks} fast_weight={h.lact_fast_weight} '
        f'state_dim={h.lact_state_dim} update={h.lact_update} '
        f'base_ttt={h.lact_base_ttt} lact_lr={h.lact_lr} epochs={h.lact_epochs}')

    adapter = build_lact_adapter(h, base_model, device)
    adapter.train()
    base_requires = [(p, p.requires_grad) for p in base_model.parameters()]
    for p in base_model.parameters():
        p.requires_grad_(h.lact_base_ttt)
    for p in adapter.parameters():
        p.requires_grad_(True)

    loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    tok_count  = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    adapter_states: dict = {}
    adapter_sgd   = (torch.optim.SGD(adapter.parameters(), lr=h.lact_lr, momentum=h.lact_momentum)
                     if h.lact_update == 'sgd' else None)
    base_ttt_params = [p for p in base_model.parameters() if p.requires_grad and p.ndim >= 2]
    base_optimizer  = (torch.optim.SGD(base_ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
                       if h.lact_base_ttt else None)

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end   = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank       // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]

        # ── SCORE FIRST ──────────────────────────────────────────────────
        base_model.eval()
        adapter.eval()
        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz      = len(batch_ws)
                x_batch  = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch  = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens    = []
                for i, ws in enumerate(batch_ws):
                    we   = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    tok = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = tok[:-1]
                    y_batch[i, :wlen] = tok[1:]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch, lact_adapter=adapter)
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                      y_batch.reshape(-1), reduction='none').reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s    = 0 if ws == 0 else context_size
                    loss_sum  += nll[i, s:wlen].to(torch.float64).sum()
                    tok_count += float(wlen - s)
                    tgt  = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb   = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb  += (val_data.has_leading_space_lut[tgt]
                            & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # ── UPDATE AFTER SCORING ─────────────────────────────────────────
        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and h.lact_epochs > 0:
            base_model.train()
            adapter.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lact_lr = h.lact_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                cos_base_lr = h.ttt_lr  * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                if adapter_sgd is not None:
                    for pg in adapter_sgd.param_groups:
                        pg['lr'] = cos_lact_lr
                if base_optimizer is not None:
                    for pg in base_optimizer.param_groups:
                        pg['lr'] = cos_base_lr
                my_seq_s      = chunk_seqs * rank       // world_size
                my_seq_e      = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(h.lact_epochs):
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be    = min(bs + batch_seqs, my_chunk_seqs)
                        s_tok = chunk_start + (my_seq_s + bs) * seq_len
                        e_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if e_tok > val_data.val_tokens.numel():
                            continue
                        local = val_data.val_tokens[s_tok:e_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        if adapter_sgd is not None:
                            adapter_sgd.zero_grad(set_to_none=True)
                        if base_optimizer is not None:
                            base_optimizer.zero_grad(set_to_none=True)
                        for p in adapter.parameters():
                            p.grad = None
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits = base_model.forward_logits(x, lact_adapter=adapter)
                            loss   = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                                     y.reshape(-1), reduction='mean')
                        loss.backward()
                        if world_size > 1:
                            all_update = list(adapter.parameters())
                            if h.lact_base_ttt:
                                all_update += base_ttt_params
                            for p in all_update:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(adapter.parameters(), h.lact_grad_clip)
                        if h.lact_update == 'muon':
                            lact_muon_step(adapter, h, adapter_states, cos_lact_lr)
                        elif adapter_sgd is not None:
                            adapter_sgd.step()
                            if h.lact_normalize:
                                adapter.normalize_()
                        if base_optimizer is not None:
                            torch.nn.utils.clip_grad_norm_(base_ttt_params, 1.0)
                            base_optimizer.step()

        if rank == 0 and (ci % 50 == 0 or ci == num_chunks - 1):
            running_bpb = (loss_sum.item() / math.log(2.0)) / max(byte_count.item(), 1.0)
            log(f'  lact_chunk [{ci + 1}/{num_chunks}] bpb={running_bpb:.6f}')

    _allreduce_scalars(loss_sum, tok_count, byte_count)
    for p, req in base_requires:
        p.requires_grad_(req)
    for p in base_model.parameters():
        p.grad = None
    base_model.eval()
    return _loss_bpb(loss_sum, tok_count, byte_count)


def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)
    log(f'{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms')
    return val_loss, val_bpb


# ── training loop ─────────────────────────────────────────────────────────────

def train_model(h, device, val_data):
    base_model     = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model
    log(f'model_params:{sum(p.numel() for p in base_model.parameters())}')
    optimizers   = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    max_wallclock_ms = 1000.0 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1000.0
        log(f'gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms')

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

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = (micro_step == h.grad_accum_steps - 1)
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group['momentum'] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    if h.warmup_steps > 0:
        initial_model_state    = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                log(f'warmup_step: {warmup_step + 1}/{h.warmup_steps}')
        if h.num_loops > 0:
            base_model.looping_active = True
            log(f'loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}')
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    log(f'loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}')
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)

    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay  = h.ema_decay
    training_time_ms = 0.0
    stop_after_step  = None
    torch.cuda.synchronize()
    t0   = time.perf_counter()
    step = 0
    while True:
        last_step      = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(h, device, val_data, model)
            log(f'{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}')
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(f'stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{h.iterations}')
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        frac       = training_frac(step, elapsed_ms)
        scale      = lr_mul(frac)
        if h.num_loops > 0 and not base_model.looping_active and frac >= h.enable_looping_at:
            base_model.looping_active = True
            log(f'layer_loop:enabled step:{step} frac:{frac:.3f} '
                f'encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}')
        train_loss = step_fn(step, scale)
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None):
            tok_per_sec = step * h.train_batch_tokens / (approx_ms / 1000.0)
            log(f'{step}/{h.iterations} train_loss:{train_loss.item():.4f} '
                f'train_time:{approx_ms / 60000:.1f}m tok/s:{tok_per_sec:.0f}')
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            cap_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log(f'peak memory allocated:{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB '
        f'reserved:{torch.cuda.max_memory_reserved() // 1024 // 1024} MiB')
    log('ema:applying EMA weights')
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model


def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    if h.ttt_lora_enabled:
        raise RuntimeError(
            "Standalone TTT_LORA_ENABLED eval is not supported in this record. "
            "Use TTT_ENABLED=1 with PHASED_TTT_ENABLED=1 for the PR #1727 phased TTT path."
        )
    if h.ttt_enabled and not h.phased_ttt_enabled:
        raise RuntimeError(
            "Standalone single-phase TTT has been removed from this record. "
            "Use TTT_ENABLED=1 with PHASED_TTT_ENABLED=1, or disable TTT entirely."
        )
    if h.phased_ttt_enabled and not h.ttt_enabled:
        raise RuntimeError(
            "PHASED_TTT_ENABLED=1 requires TTT_ENABLED=1 so the phased TTT path actually runs."
        )
    val_data = ValidationData(h, device)
    log(f"train_shards:{len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}")
    log(f'val_tokens:{val_data.val_tokens.numel() - 1}')

    base_model, compiled_model = train_model(h, device, val_data)
    torch._dynamo.reset()
    timed_eval('pre-quantization post-ema', eval_val, h, device, val_data, compiled_model)
    serialize(h, base_model, Path(__file__).read_text(encoding='utf-8'))
    if h.distributed:
        dist.barrier()

    # ── fixed-predictor evals ────────────────────────────────────────────
    eval_model    = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval('quantized', eval_val, h, device, val_data, compiled_eval)
    if h.sliding_window_enabled:
        timed_eval('quantized_sliding_window', eval_val_sliding, h, device, val_data, eval_model)
    del eval_model, compiled_eval
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    # ── phased TTT (PR #1727 path) ────────────────────────────────────────
    if h.ttt_enabled:
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        for p in ttt_model.parameters():
            p.requires_grad_(False)

        def _fwd_ttt_inner(input_ids, target_ids, lora):
            return ttt_model.forward_ttt(input_ids, target_ids, lora=lora)

        _fwd_ttt_compiled_inner = None

        def _fwd_ttt(input_ids, target_ids, lora):
            nonlocal _fwd_ttt_compiled_inner
            if _fwd_ttt_compiled_inner is None:
                _fwd_ttt_compiled_inner = torch.compile(_fwd_ttt_inner, dynamic=True)
            return _fwd_ttt_compiled_inner(input_ids, target_ids, lora=lora)

        fwd_ttt_compiled = _fwd_ttt
        log("ttt_lora:warming up compile (random tokens, no val data)")
        global BOS_ID
        if BOS_ID is None:
            BOS_ID = 1
        t_warmup = time.perf_counter()
        warmup_bszes = [h.ttt_batch_size]
        for bsz in warmup_bszes:
            wl = BatchedTTTLoRA(
                bsz, ttt_model, h.ttt_lora_rank,
                k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
            ).to(device)
            wo = torch.optim.AdamW(
                wl.parameters(),
                lr=h.ttt_lora_lr,
                betas=(h.ttt_beta1, h.ttt_beta2),
                eps=1e-10,
                weight_decay=h.ttt_weight_decay,
                fused=True,
            )
            for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
                xw = torch.randint(0, h.vocab_size, (bsz, ctx_len), device=device, dtype=torch.int64)
                yw = torch.randint(0, h.vocab_size, (bsz, ctx_len), device=device, dtype=torch.int64)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = fwd_ttt_compiled(xw, yw, lora=wl)
                ptl[:, : min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
                wo.step()
                wo.zero_grad(set_to_none=True)
            del wl, wo
        torch.cuda.empty_cache()
        compile_elapsed = time.perf_counter() - t_warmup
        log(f"ttt_lora:compile warmup done ({compile_elapsed:.1f}s)")
        log("\nbeginning TTT eval timer")
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb = eval_val_ttt_phased(
            h, ttt_model, device, val_data, forward_ttt_train=fwd_ttt_compiled
        )
        torch.cuda.synchronize()
        ttt_eval_elapsed = time.perf_counter() - t_ttt
        log(
            "quantized_ttt_phased "
            f"val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f} "
            f"eval_time:{1e3*ttt_eval_elapsed:.0f}ms"
        )
        log(f"total_eval_time:{ttt_eval_elapsed:.1f}s")
        del ttt_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    # ── LaCT fast-weight TTT (optional additional metric) ────────────────
    if h.lact_ttt_enabled and h.sliding_window_enabled:
        lact_model = deserialize(h, device)
        if h.num_loops > 0:
            lact_model.looping_active = True
        timed_eval('quantized_lact_ttt', eval_val_lact_ttt,
                   h, device, val_data, lact_model)
        del lact_model


def main():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if world_size <= 0:
        raise ValueError(f'WORLD_SIZE must be positive, got {world_size}')
    if 8 % world_size != 0:
        raise ValueError(f'WORLD_SIZE={world_size} must divide 8')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True
    torch.set_float32_matmul_precision('high')
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                     enable_math_sdp, enable_mem_efficient_sdp)
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs('logs', exist_ok=True)
        log('=' * 100, console=False)
        log('Hyperparameters:', console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith('_'):
                log(f'  {k}: {v}', console=True)
        log('=' * 100, console=False)
        log(f'Running Python {sys.version}', console=False)
        log(f'Running PyTorch {torch.__version__}', console=False)
        log(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
        log('=' * 100, console=False)
    train_and_eval(h, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
