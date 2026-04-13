import base64, collections, copy, fcntl, glob, io, json, lzma, math, os
from pathlib import Path
import random, re, subprocess, sys, time, uuid, numpy as np, sentencepiece as spm, torch, torch.distributed as dist, torch.nn.functional as F
from torch import nn
from flash_attn_interface import (
    flash_attn_func as flash_attn_3_func,
    flash_attn_varlen_func,
)
from concurrent.futures import ThreadPoolExecutor
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


class Hyperparameters:
    data_dir = os.environ.get("DATA_DIR", "./data/")
    seed = int(os.environ.get("SEED", 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.75))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 6e2))
    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 524288))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    sliding_window_enabled = bool(int(os.environ.get("SLIDING_WINDOW_ENABLED", "0")))
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
    rope_yarn = bool(int(os.environ.get("ROPE_YARN", "0")))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0))
    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))
    parallel_start_layer = int(os.environ.get("PARALLEL_START_LAYER", 8))
    parallel_final_lane = os.environ.get("PARALLEL_FINAL_LANE", "mean")
    min_lr = float(os.environ.get("MIN_LR", 0.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.026))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.97))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
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
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 96))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.0001))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 48))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 2048))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 64))
    ttt_grad_steps = int(os.environ.get("TTT_GRAD_STEPS", 1))
    ttt_weight_decay = float(os.environ.get("TTT_WEIGHT_DECAY", 0.5))
    ttt_beta1 = float(os.environ.get("TTT_BETA1", 0))
    ttt_beta2 = float(os.environ.get("TTT_BETA2", 0.999))
    ttt_k_lora = bool(int(os.environ.get("TTT_K_LORA", "1")))
    ttt_mlp_lora = bool(int(os.environ.get("TTT_MLP_LORA", "1")))
    ttt_o_lora = bool(int(os.environ.get("TTT_O_LORA", "1")))
    ttt_optimizer = os.environ.get("TTT_OPTIMIZER", "adam")
    ttt_eval_batches = os.environ.get("TTT_EVAL_BATCHES", "")
    ttt_output_dir = os.environ.get("TTT_OUTPUT_DIR", "")
    val_doc_fraction = float(os.environ.get("VAL_DOC_FRACTION", 1.0))
    compressor = os.environ.get("COMPRESSOR", "brotli")
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 64))
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 13.0))
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    embed_bits = int(os.environ.get("EMBED_BITS", 7))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 15.0))
    mlp_clip_sigmas = float(os.environ.get("MLP_CLIP_SIGMAS", 12.0))
    attn_clip_sigmas = float(os.environ.get("ATTN_CLIP_SIGMAS", 13.0))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    datasets_dir = os.path.join(data_dir, "datasets", f"fineweb10B_sp{vocab_size}")
    train_files = os.path.join(datasets_dir, "fineweb_train_*.bin")
    val_files = os.path.join(datasets_dir, "fineweb_val_*.bin")
    tokenizer_path = os.path.join(
        data_dir, "tokenizers", f"fineweb_{vocab_size}_bpe.model"
    )
    artifact_dir = os.environ.get("ARTIFACT_DIR", "")
    eval_only_path = os.environ.get("EVAL_ONLY_PATH", "")
    logfile = (
        os.path.join(artifact_dir, f"{run_id}.txt")
        if artifact_dir
        else f"logs/{run_id}.txt"
    )
    model_path = (
        os.path.join(artifact_dir, "final_model.pt")
        if artifact_dir
        else "final_model.pt"
    )
    quantized_model_path = (
        os.path.join(artifact_dir, "final_model.int6.ptz")
        if artifact_dir
        else "final_model.int6.ptz"
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
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
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
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
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
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
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
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


BOS_ID = None


def get_next_multiple_of_n(v, n):
    return ((v + n - 1) // n) * n


def _build_cu_seqlens(bos_pos, total_len, device, max_doc_len=0, bucket_size=64):
    if not bos_pos or bos_pos[0] != 0:
        bos_pos = [0] + bos_pos
    seg_starts = []
    starts_with_end = bos_pos + [total_len]
    for i in range(len(starts_with_end) - 1):
        start = starts_with_end[i]
        end = starts_with_end[i + 1]
        if max_doc_len > 0:
            pos = start
            while pos < end:
                seg_starts.append(pos)
                pos += max_doc_len
        else:
            seg_starts.append(start)
    boundaries = seg_starts + [total_len]
    padded_len = get_next_multiple_of_n(len(boundaries), bucket_size)
    cu = torch.full((padded_len,), total_len, dtype=torch.int32, device=device)
    cu[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
    seg_ends = seg_starts[1:] + [total_len]
    max_seqlen = max(end - start for start, end in zip(seg_starts, seg_ends))
    return cu, max_seqlen

class DocumentPackingLoader:
    _shard_pool = ThreadPoolExecutor(1)

    def __init__(self, h, device, cu_bucket_size=64):
        self.rank = h.rank
        self.world_size = h.world_size
        self.device = device
        self.cu_bucket_size = cu_bucket_size
        self.max_seq_len = h.train_seq_len
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files
        self.file_iter = iter(self.files)
        self._init_shard(load_data_shard(next(self.file_iter)))
        self._next_shard = self._submit_next_shard()
        self._batch_pool = ThreadPoolExecutor(1)
        self._next_batch = None

    def _init_shard(self, tokens):
        global BOS_ID
        self.tokens = tokens
        self.shard_size = tokens.numel()
        if BOS_ID is None:
            BOS_ID = 1
        self.bos_idx = (
            (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        )
        if self.bos_idx.size == 0:
            self.bos_idx = np.array([0], dtype=np.int64)
        self.cursor = int(self.bos_idx[0])

    def _submit_next_shard(self):
        try:
            path = next(self.file_iter)
            return self._shard_pool.submit(load_data_shard, path)
        except StopIteration:
            return None

    def _advance_shard(self):
        if self._next_shard is None:
            self.file_iter = iter(self.files)
            self._next_shard = self._shard_pool.submit(
                load_data_shard, next(self.file_iter)
            )
        self._init_shard(self._next_shard.result())
        self._next_shard = self._submit_next_shard()

    def _local_doc_starts(self, local_start, total_len):
        lo = np.searchsorted(self.bos_idx, local_start, side="left")
        hi = np.searchsorted(self.bos_idx, local_start + total_len, side="left")
        return (self.bos_idx[lo:hi] - local_start).tolist()

    def _prepare_batch(self, num_tokens_local, max_seq_len):
        per_rank_span = num_tokens_local + 1
        global_span = per_rank_span * self.world_size
        while self.cursor + global_span > self.shard_size:
            self._advance_shard()
        local_start = self.cursor + self.rank * per_rank_span
        buf = self.tokens[local_start : local_start + per_rank_span]
        inputs = buf[:-1].to(dtype=torch.int64).pin_memory()
        targets = buf[1:].to(dtype=torch.int64).pin_memory()
        starts = self._local_doc_starts(local_start, inputs.numel())
        cu_seqlens, max_seqlen = _build_cu_seqlens(
            starts, inputs.numel(), inputs.device, max_seq_len, self.cu_bucket_size
        )
        cu_seqlens = cu_seqlens.pin_memory()
        self.cursor += global_span
        return inputs, targets, cu_seqlens, max_seqlen

    def next_batch(self, global_tokens, grad_accum_steps):
        num_tokens_local = global_tokens // (self.world_size * grad_accum_steps)
        if self._next_batch is not None:
            inputs, targets, cu_seqlens, max_seqlen = self._next_batch.result()
        else:
            inputs, targets, cu_seqlens, max_seqlen = self._prepare_batch(
                num_tokens_local, self.max_seq_len
            )
        self._next_batch = self._batch_pool.submit(
            self._prepare_batch, num_tokens_local, self.max_seq_len
        )
        return (
            inputs[None].to(self.device, non_blocking=True),
            targets[None].to(self.device, non_blocking=True),
            cu_seqlens.to(self.device, non_blocking=True),
            max_seqlen,
        )


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
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
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
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
                np.array(mm[start_ind : start_ind + self.seq_len + 1], dtype=np.int64)
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


@triton.jit
def linear_leaky_relu_square_kernel(
    a_desc,
    b_desc,
    c_desc,
    aux_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FORWARD: tl.constexpr,
):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
        tile_id_c += NUM_SMS
        offs_am_c = offs_am
        offs_bn_c = offs_bn
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c0 = acc0.to(dtype)
        c1 = acc1.to(dtype)
        if not FORWARD:
            pre0 = aux_desc.load([offs_am_c, offs_bn_c])
            pre1 = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c0 = c0 * tl.where(pre0 > 0, 2.0 * pre0, 0.5 * pre0)
            c1 = c1 * tl.where(pre1 > 0, 2.0 * pre1, 0.5 * pre1)
        c_desc.store([offs_am_c, offs_bn_c], c0)
        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        if FORWARD:
            aux0 = tl.where(c0 > 0, c0, 0.5 * c0)
            aux1 = tl.where(c1 > 0, c1, 0.5 * c1)
            aux_desc.store([offs_am_c, offs_bn_c], aux0 * aux0)
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], aux1 * aux1)


def linear_leaky_relu_square(a, b, aux=None):
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    forward = aux is None
    if aux is None:
        aux = torch.empty((M, N), device=a.device, dtype=a.dtype)
    num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 64
    num_stages = 4 if forward else 3
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    grid = lambda _meta: (
        min(num_sms, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),
    )
    linear_leaky_relu_square_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        aux_desc,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NUM_SMS=num_sms,
        FORWARD=forward,
        num_stages=num_stages,
        num_warps=8,
    )
    if forward:
        return c, aux
    return c


class FusedLinearLeakyReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):
        x_flat = x.reshape(-1, x.shape[-1])
        pre, post = linear_leaky_relu_square(x_flat, w1)
        out = F.linear(post, w2)
        ctx.save_for_backward(x, w1, w2, pre, post)
        return out.view(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, w2, pre, post = ctx.saved_tensors
        x_flat = x.reshape(-1, x.shape[-1])
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        dw2 = grad_output_flat.T @ post
        dpre = linear_leaky_relu_square(grad_output_flat, w2.T.contiguous(), aux=pre)
        dw1 = dpre.T @ x_flat
        dx = dpre @ w1
        return dx.view_as(x), dw1, dw2


FusedLeakyReLUSquareMLP = FusedLinearLeakyReLUSquareFunction.apply


class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0, yarn=True):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.yarn = yarn
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached < seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if self.yarn and seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (
                    torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd
                )
            else:
                inv_freq = self.inv_freq.float().to(device)
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached[:, :seq_len].to(dtype=dtype), self._sin_cached[:, :seq_len].to(dtype=dtype)


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
        self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len, yarn=True
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
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len, yarn=yarn)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x, q_w, k_w, v_w, out_w, cu_seqlens=None, max_seqlen=0):
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if cu_seqlens is not None:
            y = flash_attn_varlen_func(
                q[0],
                k[0],
                v[0],
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1),
            )[None]
        else:
            y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        self._last_proj_input = y.detach() if getattr(self, "_calib", False) else None
        return F.linear(y, out_w.to(x.dtype))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.use_fused = True

    def forward(self, x, up_w, down_w):
        if self.training and self.use_fused:
            return FusedLeakyReLUSquareMLP(x, up_w.to(x.dtype), down_w.to(x.dtype))
        hidden = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5).square()
        self._last_down_input = hidden.detach() if getattr(self, "_calib", False) else None
        return F.linear(hidden, down_w.to(x.dtype))


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
        yarn=True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len, yarn=yarn
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=None, max_seqlen=0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor,
            q_w, k_w, v_w, out_w,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[
            None, None, :
        ] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out

class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
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
        self.num_layers = h.num_layers
        head_dim = h.model_dim // h.num_heads
        kv_dim = h.num_kv_heads * head_dim
        hidden_dim = int(h.mlp_mult * h.model_dim)
        self.qo_bank = nn.Parameter(torch.empty(2 * h.num_layers, h.model_dim, h.model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * h.num_layers, kv_dim, h.model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(h.num_layers, hidden_dim, h.model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(h.num_layers, h.model_dim, hidden_dim))
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    h.mlp_mult,
                    h.rope_base,
                    h.qk_gain_init,
                    h.train_seq_len,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                    yarn=h.rope_yarn,
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
                    yarn=h.rope_yarn,
                )
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
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        self.num_skip_weights = min(
            len(self.encoder_indices), len(self.decoder_indices)
        )
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32)
        )
        self.skip_gates = (
            nn.Parameter(
                torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)
            )
            if h.skip_gates_enabled
            else None
        )
        self.parallel_start_layer = h.parallel_start_layer
        self.parallel_final_lane = h.parallel_final_lane.lower()
        self.parallel_post_lambdas = nn.Parameter(
            torch.ones(h.num_layers, 2, 2, dtype=torch.float32)
        )
        self.parallel_resid_lambdas = nn.Parameter(
            torch.full((h.num_layers, 2), 1.1, dtype=torch.float32)
        )
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[n + i])
            self.qo_bank.data[n + i].mul_(proj_scale)
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.mlp_down_bank.data[i].mul_(proj_scale)
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

    def _bank_weights(self, i):
        n = self.num_layers
        return (
            self.qo_bank[i],
            self.kv_bank[i],
            self.kv_bank[n + i],
            self.qo_bank[n + i],
            self.mlp_up_bank[i],
            self.mlp_down_bank[i],
        )

    def _parallel_block(
        self, block_idx, lane0, lane1, x0,
        q_w, k_w, v_w, out_w, up_w, down_w,
        cu_seqlens=None, max_seqlen=0,
    ):
        block = self.blocks[block_idx]
        mix = block.resid_mix.to(dtype=lane0.dtype)
        attn_read = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
        attn_out = block.attn(
            block.attn_norm(attn_read) * block.ln_scale_factor,
            q_w, k_w, v_w, out_w,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
        attn_out = block.attn_scale.to(dtype=attn_out.dtype)[None, None, :] * attn_out
        mlp_read = lane1
        mlp_out = block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * block.mlp(
            block.mlp_norm(mlp_read) * block.ln_scale_factor, up_w, down_w
        )
        attn_resid = self.parallel_resid_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        attn_post = self.parallel_post_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        mlp_resid = self.parallel_resid_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        mlp_post = self.parallel_post_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
        lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
        return lane0, lane1

    def _final_parallel_hidden(self, lane0, lane1):
        if self.parallel_final_lane == "mlp":
            return lane1
        if self.parallel_final_lane == "attn":
            return lane0
        return 0.5 * (lane0 + lane1)

    def forward_logits(self, input_ids, cu_seqlens=None, max_seqlen=0):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        enc_iter = (
            self.encoder_indices
            if self.looping_active
            else range(self.num_encoder_layers)
        )
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else range(
                self.num_encoder_layers,
                self.num_encoder_layers + self.num_decoder_layers,
            )
        )
        for i in enc_iter:
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            x = self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                lane0, lane1 = self._parallel_block(
                    i, lane0, lane1, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                )
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                x = self.blocks[i](x, x0, q_w, k_w, v_w, out_w, up_w, down_w, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids, cu_seqlens=None, max_seqlen=0):
        logits = self.forward_logits(
            input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )

    def forward_ttt(self, input_ids, target_ids, lora):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        enc_iter = (
            self.encoder_indices
            if self.looping_active
            else list(range(self.num_encoder_layers))
        )
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else list(
                range(
                    self.num_encoder_layers,
                    self.num_encoder_layers + self.num_decoder_layers,
                )
            )
        )
        slot = 0
        for i in enc_iter:
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            x = self._block_with_lora(self.blocks[i], x, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w)
            slot += 1
            skips.append(x)
        psl = self.parallel_start_layer
        lane0 = None
        lane1 = None
        for skip_idx, i in enumerate(dec_iter):
            q_w, k_w, v_w, out_w, up_w, down_w = self._bank_weights(i)
            if i >= psl and psl > 0:
                if lane0 is None:
                    lane0 = x
                    lane1 = x
                if skip_idx < self.num_skip_weights and skips:
                    skip = skips.pop()
                    w = self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None, None, :]
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None, None, :]
                        lane0 = torch.lerp(w * skip, lane0, g)
                    else:
                        lane0 = lane0 + w * skip
                lane0, lane1 = self._parallel_block_with_lora(
                    i, lane0, lane1, x0, lora, slot,
                    q_w, k_w, v_w, out_w, up_w, down_w,
                )
            else:
                if skip_idx < self.num_skip_weights and skips:
                    scaled_skip = (
                        self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                        * skips.pop()
                    )
                    if self.skip_gates is not None:
                        g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(scaled_skip, x, g)
                    else:
                        x = x + scaled_skip
                x = self._block_with_lora(self.blocks[i], x, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w)
            slot += 1
        if lane0 is not None:
            x = self._final_parallel_hidden(lane0, lane1)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = logits + lora.lm_head_lora(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        bsz, sl, V = logits.shape
        return F.cross_entropy(
            logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
        ).reshape(bsz, sl)

    def _block_with_lora(self, block, x, x0, lora, slot, q_w, k_w, v_w, out_w, up_w, down_w):
        mix = block.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = block.attn_norm(x_in) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = n.shape
        q = (F.linear(n, q_w.to(n.dtype)) + lora.q_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_heads, attn.head_dim
        )
        k = F.linear(n, k_w.to(n.dtype))
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        v = (F.linear(n, v_w.to(n.dtype)) + lora.v_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_kv_heads, attn.head_dim
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, n.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
        k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        attn_out = F.linear(y, out_w.to(n.dtype))
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](n)
        x_out = x_in + block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        mlp_n = block.mlp_norm(x_out) * block.ln_scale_factor
        mlp_out = block.mlp(mlp_n, up_w, down_w)
        if lora.mlp_loras is not None:
            mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
        x_out = x_out + block.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out
        return x_out

    def _parallel_block_with_lora(
        self, block_idx, lane0, lane1, x0, lora, slot,
        q_w, k_w, v_w, out_w, up_w, down_w,
    ):
        block = self.blocks[block_idx]
        mix = block.resid_mix.to(dtype=lane0.dtype)
        attn_read = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
        n = block.attn_norm(attn_read) * block.ln_scale_factor
        attn = block.attn
        bsz, seqlen, dim = n.shape
        q = (F.linear(n, q_w.to(n.dtype)) + lora.q_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_heads, attn.head_dim
        )
        k = F.linear(n, k_w.to(n.dtype))
        if lora.k_loras is not None:
            k = k + lora.k_loras[slot](n)
        k = k.reshape(bsz, seqlen, attn.num_kv_heads, attn.head_dim)
        v = (F.linear(n, v_w.to(n.dtype)) + lora.v_loras[slot](n)).reshape(
            bsz, seqlen, attn.num_kv_heads, attn.head_dim
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = attn.rotary(seqlen, n.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, attn.rope_dims)
        k = apply_rotary_emb(k, cos, sin, attn.rope_dims)
        q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if attn.use_xsa:
            y = attn._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        attn_out = F.linear(y, out_w.to(n.dtype))
        if lora.o_loras is not None:
            attn_out = attn_out + lora.o_loras[slot](n)
        attn_out = block.attn_scale.to(dtype=attn_out.dtype)[None, None, :] * attn_out
        mlp_read = lane1
        mlp_n = block.mlp_norm(mlp_read) * block.ln_scale_factor
        mlp_out = block.mlp(mlp_n, up_w, down_w)
        if lora.mlp_loras is not None:
            mlp_out = mlp_out + lora.mlp_loras[slot](mlp_n)
        mlp_out = block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * mlp_out
        attn_resid = self.parallel_resid_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        attn_post = self.parallel_post_lambdas[block_idx, 0].to(dtype=lane0.dtype)
        mlp_resid = self.parallel_resid_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        mlp_post = self.parallel_post_lambdas[block_idx, 1].to(dtype=lane0.dtype)
        lane0 = attn_resid * lane0 + attn_post[0] * attn_out + mlp_post[0] * mlp_out
        lane1 = mlp_resid * lane1 + attn_post[1] * attn_out + mlp_post[1] * mlp_out
        return lane0, lane1


class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz, in_features, out_features, rank):
        super().__init__()
        self._bound = 1.0 / math.sqrt(in_features)
        self.A = nn.Parameter(
            torch.empty(bsz, rank, in_features).uniform_(-self._bound, self._bound)
        )
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))

    def reset(self):
        with torch.no_grad():
            self.A.uniform_(-self._bound, self._bound)
            self.B.zero_()

    def forward(self, x):
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)


class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz, model, rank, k_lora=True, mlp_lora=True, o_lora=True):
        super().__init__()
        self.bsz = bsz
        dim = model.qo_bank.shape[-1]
        vocab = model.tok_emb.num_embeddings
        if getattr(model, "looping_active", False):
            num_slots = len(model.encoder_indices) + len(model.decoder_indices)
        else:
            num_slots = len(model.blocks)
        kv_dim = model.blocks[0].attn.num_kv_heads * (
            dim // model.blocks[0].attn.num_heads
        )
        embed_dim = model.tok_emb.embedding_dim
        self.lm_head_lora = BatchedLinearLoRA(bsz, embed_dim, vocab, rank)
        self.q_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
        )
        self.v_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]
        )
        self.k_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, kv_dim, rank) for _ in range(num_slots)]
            )
            if k_lora
            else None
        )
        self.mlp_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
            )
            if mlp_lora
            else None
        )
        self.o_loras = (
            nn.ModuleList(
                [BatchedLinearLoRA(bsz, dim, dim, rank) for _ in range(num_slots)]
            )
            if o_lora
            else None
        )

    def reset(self):
        with torch.no_grad():
            self.lm_head_lora.reset()
            for loras in [self.q_loras, self.v_loras, self.k_loras,
                          self.mlp_loras, self.o_loras]:
                if loras is not None:
                    for lora in loras:
                        lora.reset()


def classify_param(name):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or ".proj." in name and ".mlp." not in name:
        return "attn"
    return "other"


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-07):
    a, b, c = 3.4445, -4.775, 2.0315
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


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
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    "p": p,
                    "B": B,
                    "padded_grad": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard_mom": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "full_update": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "scale": max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m["p"].numel())
        self._built = True

    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m["p"]
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m["padded_grad"]
            pg[: m["B"]].copy_(p.grad.bfloat16())
            if pg.shape[0] > m["B"]:
                pg[m["B"] :].zero_()
            fut = dist.reduce_scatter_tensor(
                m["shard"], pg, op=dist.ReduceOp.AVG, async_op=True
            )
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self._built:
            self._build()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            row_normalize = group.get("row_normalize", False)
            prev_ag_handle = None
            prev_m = None
            sharded = self._distributed and hasattr(self, "_rs_futures")
            for idx, m in enumerate(self._bank_meta):
                p = m["p"]
                if p.grad is None:
                    continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m["p"]
                    upd = prev_m["full_update"][: prev_m["B"]]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
                if sharded and self._rs_futures[idx] is not None:
                    self._rs_futures[idx].wait()
                    g = m["shard"]
                    buf = m["shard_mom"]
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf
                if row_normalize:
                    rn = update.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                    update = update / rn.to(update.dtype)
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m["full_update"], update, async_op=True
                    )
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m["scale"])
            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m["p"]
                upd = prev_m["full_update"][: prev_m["B"]]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m["scale"])
            if hasattr(self, "_rs_futures"):
                del self._rs_futures
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,parallel_post_lambdas,parallel_resid_lambdas",
    ).split(",")
    if pattern
)


PACKED_REPLICATED_GRAD_MAX_NUMEL = 1 << 15


class Optimizers:
    def __init__(self, h, base_model):
        matrix_params = [
            base_model.qo_bank,
            base_model.kv_bank,
            base_model.mlp_up_bank,
            base_model.mlp_down_bank,
        ]
        block_named_params = list(base_model.blocks.named_parameters())
        scalar_params = [
            p
            for (name, p) in block_named_params
            if p.ndim < 2
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        if base_model.parallel_post_lambdas is not None:
            scalar_params.append(base_model.parallel_post_lambdas)
        if base_model.parallel_resid_lambdas is not None:
            scalar_params.append(base_model.parallel_resid_lambdas)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [
            {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}
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
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
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
        self.replicated_params = list(tok_params[0]["params"])
        self.replicated_params.extend(scalar_params)
        if base_model.lm_head is not None:
            self.replicated_params.append(base_model.lm_head.weight)
        self.replicated_large_params = []
        self.replicated_packed_params = []
        for p in self.replicated_params:
            if p.numel() <= PACKED_REPLICATED_GRAD_MAX_NUMEL:
                self.replicated_packed_params.append(p)
            else:
                self.replicated_large_params.append(p)

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def _all_reduce_packed_grads(self):
        grads_by_key = collections.defaultdict(list)
        for p in self.replicated_packed_params:
            if p.grad is not None:
                grads_by_key[(p.grad.device, p.grad.dtype)].append(p.grad)
        for grads in grads_by_key.values():
            flat = torch.empty(
                sum(g.numel() for g in grads),
                device=grads[0].device,
                dtype=grads[0].dtype,
            )
            offset = 0
            for g in grads:
                n = g.numel()
                flat[offset : offset + n].copy_(g.contiguous().view(-1))
                offset += n
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            offset = 0
            for g in grads:
                n = g.numel()
                g.copy_(flat[offset : offset + n].view_as(g))
                offset += n

    def step(self, distributed=False):
        self.optimizer_muon.launch_reduce_scatters()
        if distributed:
            reduce_handles = [
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
                for p in self.replicated_large_params
                if p.grad is not None
            ]
            self._all_reduce_packed_grads()
            for handle in reduce_handles:
                handle.wait()
        self.optimizer_tok.step()
        self.optimizer_scalar.step()
        if self.optimizer_head is not None:
            self.optimizer_head.step()
        self.optimizer_muon.step()
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
    if hasattr(model, "qo_bank"):
        model.qo_bank.data = model.qo_bank.data.float()
        model.kv_bank.data = model.kv_bank.data.float()
        model.mlp_up_bank.data = model.mlp_up_bank.data.float()
        model.mlp_down_bank.data = model.mlp_down_bank.data.float()


def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []
    for i, block in enumerate(model.blocks):
        block.attn._calib = True
        block.mlp._calib = True
        block.mlp.use_fused = False

    def make_attn_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            for suffix in ["c_q", "c_k", "c_v"]:
                name = f"blocks.{layer_idx}.attn.{suffix}.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(x.T, x)
            y = module._last_proj_input
            if y is not None:
                y = y.float()
                if y.ndim == 3:
                    y = y.reshape(-1, y.shape[-1])
                name = f"blocks.{layer_idx}.attn.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        y.shape[1], y.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(y.T, y)
        return hook_fn

    def make_mlp_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            name = f"blocks.{layer_idx}.mlp.fc.weight"
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
            hessians[name].addmm_(x.T, x)
            h_act = module._last_down_input
            if h_act is not None:
                h_act = h_act.float()
                if h_act.ndim == 3:
                    h_act = h_act.reshape(-1, h_act.shape[-1])
                name = f"blocks.{layer_idx}.mlp.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        h_act.shape[1], h_act.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(h_act.T, h_act)
        return hook_fn

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(block.mlp.register_forward_hook(make_mlp_hook(i)))
    if model.tie_embeddings:
        hook_module = (
            model.head_proj if model.head_proj is not None else model.final_norm
        )

        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(x.T, x)
            return hook_fn

        hooks.append(
            hook_module.register_forward_hook(make_output_hook("tok_emb.weight"))
        )
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    for i, block in enumerate(model.blocks):
        block.attn._calib = False
        block.mlp._calib = False
        block.mlp.use_fused = True
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
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s


def gptq_mixed_quantize(state_dict, hessians, h):
    result = {}
    meta = {}
    for (name, tensor) in state_dict.items():
        t = tensor.detach().cpu().contiguous()
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
        q, s = gptq_quantize_weight(
            t, hessians[name], clip_sigmas=cs, clip_range=2 ** (bits - 1) - 1
        )
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
    categories = collections.defaultdict(set)
    for (name, cat) in meta.items():
        short = re.sub("\\.\\d+$", "", re.sub("blocks\\.\\d+", "blocks", name))
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


def _unbank_state_dict(state_dict, num_layers):
    sd = {}
    n = num_layers
    for k, v in state_dict.items():
        t = v.detach().cpu()
        if k == "qo_bank":
            for i in range(n):
                sd[f"blocks.{i}.attn.c_q.weight"] = t[i]
                sd[f"blocks.{i}.attn.proj.weight"] = t[n + i]
        elif k == "kv_bank":
            for i in range(n):
                sd[f"blocks.{i}.attn.c_k.weight"] = t[i]
                sd[f"blocks.{i}.attn.c_v.weight"] = t[n + i]
        elif k == "mlp_up_bank":
            for i in range(n):
                sd[f"blocks.{i}.mlp.fc.weight"] = t[i]
        elif k == "mlp_down_bank":
            for i in range(n):
                sd[f"blocks.{i}.mlp.proj.weight"] = t[i]
        else:
            sd[k] = t
    return sd


def _rebank_state_dict(flat_sd, num_layers, model_dim, kv_dim, hidden_dim):
    sd = {}
    n = num_layers
    sd["qo_bank"] = torch.zeros(2 * n, model_dim, model_dim)
    sd["kv_bank"] = torch.zeros(2 * n, kv_dim, model_dim)
    sd["mlp_up_bank"] = torch.zeros(n, hidden_dim, model_dim)
    sd["mlp_down_bank"] = torch.zeros(n, model_dim, hidden_dim)
    for i in range(n):
        sd["qo_bank"][i] = flat_sd[f"blocks.{i}.attn.c_q.weight"]
        sd["qo_bank"][n + i] = flat_sd[f"blocks.{i}.attn.proj.weight"]
        sd["kv_bank"][i] = flat_sd[f"blocks.{i}.attn.c_k.weight"]
        sd["kv_bank"][n + i] = flat_sd[f"blocks.{i}.attn.c_v.weight"]
        sd["mlp_up_bank"][i] = flat_sd[f"blocks.{i}.mlp.fc.weight"]
        sd["mlp_down_bank"][i] = flat_sd[f"blocks.{i}.mlp.proj.weight"]
    for k, v in flat_sd.items():
        if not (
            k.startswith("blocks.")
            and any(
                p in k
                for p in [
                    ".attn.c_q.", ".attn.c_k.", ".attn.c_v.",
                    ".attn.proj.", ".mlp.fc.", ".mlp.proj.",
                ]
            )
        ):
            sd[k] = v
    return sd


def _compressed_code_size(code):
    code_raw = code.encode("utf-8")
    minified = subprocess.run(
        ["pyminify", "--no-rename-locals", "--no-hoist-literals", "--remove-literal-statements", "-"],
        input=code_raw, capture_output=True, check=True,
    ).stdout
    compressed = lzma.compress(minified)
    encoded = base64.b85encode(compressed)
    wrapper = b'import lzma as L,base64 as B\nexec(L.decompress(B.b85decode("' + encoded + b'")))\n'
    return len(code_raw), len(wrapper)


def serialize(h, base_model, code):
    code_bytes_uncompressed, code_bytes = _compressed_code_size(code)
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size (uncompressed): {code_bytes_uncompressed} bytes")
        log(f"Code size (compressed): {code_bytes} bytes")
    sd_cpu = _unbank_state_dict(base_model.state_dict(), h.num_layers)
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
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s")
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
        log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes")
        log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
    return bytes_total, quant_file_bytes


def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    flat_template = _unbank_state_dict(eval_model.state_dict(), h.num_layers)
    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location="cpu"
    )
    deq_flat = dequantize_mixed(quant_state["w"], quant_state["m"], flat_template)
    head_dim = h.model_dim // h.num_heads
    kv_dim = h.num_kv_heads * head_dim
    hidden_dim = int(h.mlp_mult * h.model_dim)
    deq_state = _rebank_state_dict(deq_flat, h.num_layers, h.model_dim, kv_dim, hidden_dim)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model


def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def eval_val(h, device, val_data, model, forward_logits_fn=None):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size

    # TODO: Don't truncate this.
    seq_end = seq_start + ((seq_end - seq_start) // local_batch_seqs) * local_batch_seqs

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    run_forward_logits = (
        (model.module.forward_logits if hasattr(model, "module") else model.forward_logits)
        if forward_logits_fn is None
        else forward_logits_fn
    )
    model.eval()
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1]
            y = local[1:]
            bos_pos = (x == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu_seqlens, max_seqlen = _build_cu_seqlens(
                bos_pos, x.numel(), x.device, h.eval_seq_len, 64
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = run_forward_logits(
                    x[None], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
                ).detach()
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="none",
            )
            val_loss_sum += per_token_loss.to(torch.float64).sum()
            val_token_count += float(y.numel())
            prev_ids = x
            tgt_ids = y
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
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


def eval_val_sliding(h, device, val_data, base_model, forward_logits_fn=None, batch_seqs=32):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    base_model.eval()
    run_forward_logits = base_model.forward_logits if forward_logits_fn is None else forward_logits_fn
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = (total_windows * h.rank) // h.world_size
    my_e = (total_windows * (h.rank + 1)) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    total_batches = (len(my_windows) + batch_seqs - 1) // batch_seqs
    is_master = h.rank == 0
    cu_bucket = 64
    t_sw_start = time.perf_counter()
    with torch.no_grad():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_idx = bi // batch_seqs
            if is_master and (batch_idx % 50 == 0 or batch_idx == total_batches - 1):
                elapsed = time.perf_counter() - t_sw_start
                rl = float(loss_sum.item() / token_count.item()) if token_count.item() > 0 else 0.0
                rb = float((rl / math.log(2.0)) * token_count.item() / byte_count.item()) if byte_count.item() > 0 else 0.0
                log(f"sliding_progress: batch {batch_idx+1}/{total_batches} "
                    f"tokens:{int(token_count.item())} running_loss:{rl:.4f} running_bpb:{rb:.4f} "
                    f"elapsed:{elapsed:.1f}s")
            batch_ws = my_windows[bi:bi + batch_seqs]
            x_parts = []
            y_parts = []
            cu_starts = []
            score_ranges = []
            offset = 0
            for ws in batch_ws:
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                chunk_cpu = val_data.val_tokens[ws:end + 1]
                bos_pos = (chunk_cpu[:-1] == BOS_ID).nonzero(as_tuple=True)[0].tolist()
                if not bos_pos or bos_pos[0] != 0:
                    bos_pos = [0] + bos_pos
                cu_starts.extend(offset + pos for pos in bos_pos)
                chunk = chunk_cpu.to(dtype=torch.int64, device=device)
                x_parts.append(chunk[:-1])
                y_parts.append(chunk[1:])
                score_ranges.append((offset, wlen, ws))
                offset += wlen
            x_cat = torch.cat(x_parts, dim=0)[None]
            y_cat = torch.cat(y_parts, dim=0)
            boundaries = cu_starts + [offset]
            padded_len = get_next_multiple_of_n(len(boundaries), cu_bucket)
            cu_seqlens = torch.full((padded_len,), offset, dtype=torch.int32, device=device)
            cu_seqlens[:len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = run_forward_logits(x_cat, cu_seqlens=cu_seqlens, max_seqlen=seq_len)
            flat_nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_cat,
                reduction="none",
            )
            flat_x = x_cat.reshape(-1)
            for off, wlen, ws in score_ranges:
                s = 0 if ws == 0 else context_size
                lo = off + s
                hi = off + wlen
                scored_nll = flat_nll[lo:hi].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(hi - lo)
                tgt = y_cat[lo:hi]
                prev = flat_x[lo:hi]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


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

def eval_val_ttt_lora(h, base_model, device, val_data, forward_ttt_train):
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    all_tokens = val_data.val_tokens
    all_tokens_idx = all_tokens.to(torch.int32)
    docs = _find_docs(all_tokens)
    doc_entries = list(enumerate(docs))
    if h.val_doc_fraction < 1.0:
        sample_n = max(1, int(round(len(docs) * h.val_doc_fraction)))
        sampled_indices = sorted(
            random.Random(h.seed).sample(range(len(docs)), sample_n)
        )
        doc_entries = [(i, docs[i]) for i in sampled_indices]
    log(
        f"ttt_lora:docs:{len(doc_entries)} rank:{h.ttt_lora_rank} lr:{h.ttt_lora_lr} chunk:{h.ttt_chunk_size}"
    )
    if os.environ.get("TTT_DEBUG_BYPASS") and h.rank == 0:
        test_doc = doc_entries[0][1]
        ds, dl = test_doc
        log(f"DEBUG: test doc start={ds} len={dl}")
        toks = all_tokens_idx[ds : ds + dl].to(device=device, dtype=torch.int64)
        x_d = toks[:-1].unsqueeze(0)
        y_d = toks[1:].unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_d = base_model.forward_logits(x_d)
        ptl_d = F.cross_entropy(
            logits_d.float().reshape(-1, logits_d.size(-1)),
            y_d.reshape(-1), reduction="none",
        )
        direct_loss = ptl_d.mean().item()
        direct_bpb = direct_loss / math.log(2.0)
        log(f"DEBUG: direct forward_logits loss={direct_loss:.6f} bpb={direct_bpb:.6f} ntokens={y_d.numel()}")
        toks_first5 = toks[:5].tolist()
        ptl_first5 = ptl_d[:5].tolist()
        log(f"DEBUG: first 5 tokens={toks_first5} ptl={[f'{v:.4f}' for v in ptl_first5]}")
    chunk_size, eval_seq_len = h.ttt_chunk_size, h.ttt_eval_seq_len
    eval_batch_set = None
    if h.ttt_eval_batches:
        eval_batch_set = set(int(x) for x in h.ttt_eval_batches.split(",") if x.strip())
    use_ascending = eval_batch_set is not None
    global_batches_sorted = _build_ttt_global_batches(doc_entries, h, ascending=use_ascending)
    queue_len = len(global_batches_sorted)
    counter_path = f"/tmp/ttt_counter_{h.run_id}"
    if h.rank == 0:
        _init_batch_counter(counter_path)
    if dist.is_available() and dist.is_initialized():
        path_list = [counter_path]
        dist.broadcast_object_list(path_list, src=0)
        counter_path = path_list[0]
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
    progress_f = None
    if h.ttt_output_dir and h.rank == 0:
        os.makedirs(h.ttt_output_dir, exist_ok=True)
        progress_f = open(os.path.join(h.ttt_output_dir, "progress.jsonl"), "w")
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
        should_report = False
        if eval_batch_set is not None:
            should_report = batch_num in eval_batch_set
        else:
            # should_report = local_batch_count % 10 == 0
            should_report = True
        if should_report:
            cur_tokens = token_count.item()
            cur_loss_val = loss_sum.item()
            cur_bytes_val = byte_sum.item()
            dt = cur_tokens - prev_tokens
            if dt > 0:
                b_loss = (cur_loss_val - prev_loss) / dt
                b_bpb = b_loss / math.log(2.0) * (dt / (cur_bytes_val - prev_bytes))
            else:
                b_loss = b_bpb = 0.0
            r_loss = cur_loss_val / max(cur_tokens, 1)
            r_bpb = r_loss / math.log(2.0) * (cur_tokens / max(cur_bytes_val, 1))
            elapsed = time.perf_counter() - t_start
            log(
                f"ttt_progress: batch {batch_num}/{queue_len} batch_loss:{b_loss:.4f} "
                f"batch_bpb:{b_bpb:.4f} running_loss:{r_loss:.4f} running_bpb:{r_bpb:.4f} "
                f"doc_len:{min(doc_lens)}-{max(doc_lens)}"
            )
            if progress_f is not None:
                progress_f.write(
                    json.dumps({
                        "batch": batch_num, "total_batches": queue_len,
                        "batch_loss": round(b_loss, 8), "batch_bpb": round(b_bpb, 8),
                        "running_loss": round(r_loss, 8), "running_bpb": round(r_bpb, 8),
                        "doc_len_min": min(doc_lens), "doc_len_max": max(doc_lens),
                        "chunk_size": chunk_size,
                        "elapsed_s": round(elapsed, 3),
                        "batch_t_s": round(elapsed, 3),
                    }) + "\n"
                )
                progress_f.flush()
        del cur_lora, cur_opt
    finally:
        if progress_f is not None:
            progress_f.close()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.train()
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_sum.item())
    return val_loss, val_bpb


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


def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    compiled_forward_logits = torch.compile(
        base_model.forward_logits, dynamic=False, fullgraph=True
    )
    model = compiled_model
    log(f"model_params:{sum(p.numel()for p in base_model.parameters())}")
    optimizers = Optimizers(h, base_model)
    train_loader = DocumentPackingLoader(h, device)
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
            x, y, cu_seqlens, _max_seqlen = train_loader.next_batch(
                h.train_batch_tokens, h.grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, cu_seqlens=cu_seqlens, max_seqlen=h.train_seq_len)
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
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step(distributed=h.distributed)
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
        num_tokens_local = h.train_batch_tokens // h.world_size
        for blk in base_model.blocks:
            blk.attn.rotary(num_tokens_local, device, torch.bfloat16)
        cu_bucket_size = train_loader.cu_bucket_size
        warmup_cu_buckets = tuple(cu_bucket_size * i for i in range(1, 5))
        warmup_cu_iters = 3
        x, y, cu_seqlens, _ = train_loader.next_batch(
            h.train_batch_tokens, h.grad_accum_steps
        )
        log(f"warmup_cu_buckets:{','.join(str(b) for b in warmup_cu_buckets)} iters_each:{warmup_cu_iters}")
        def _run_cu_bucket_warmup():
            for bucket_len in warmup_cu_buckets:
                boundaries = list(range(0, x.size(1), max(h.train_seq_len, 1)))
                if boundaries[-1] != x.size(1):
                    boundaries.append(x.size(1))
                cu = torch.full((bucket_len,), x.size(1), dtype=torch.int32, device=device)
                cu[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32, device=device)
                for _ in range(warmup_cu_iters):
                    optimizers.zero_grad_all()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        wloss = model(x, y, cu_seqlens=cu, max_seqlen=h.train_seq_len)
                    (wloss / h.grad_accum_steps).backward()
            optimizers.zero_grad_all()
        _run_cu_bucket_warmup()
        if h.num_loops > 0:
            base_model.looping_active = True
            _run_cu_bucket_warmup()
            base_model.looping_active = False
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
        for (opt, state) in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        train_loader = DocumentPackingLoader(h, device)
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
            last_step or h.val_loss_every > 0 and step % h.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                h, device, val_data, model, compiled_forward_logits
            )
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
        if (
            h.num_loops > 0
            and not base_model.looping_active
            and frac >= h.enable_looping_at
        ):
            base_model.looping_active = True
            log(
                f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
        train_loss = step_fn(step, scale)
        with torch.no_grad():
            for (name, t) in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(
                    t.detach().float(), alpha=1.0 - ema_decay
                )
        step += 1
        approx_training_time_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (
            step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None
        )
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1e3)
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}"
            )
        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
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
        name: t.to(dtype=current_state[name].dtype) for (name, t) in ema_state.items()
    }
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model, compiled_forward_logits


def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    if h.artifact_dir and h.is_main_process:
        os.makedirs(h.artifact_dir, exist_ok=True)
    val_data = ValidationData(h, device)
    if h.eval_only_path:
        log(f"eval_only:loading checkpoint from {h.eval_only_path}")
        base_model = GPT(h).to(device).bfloat16()
        restore_fp32_params(base_model)
        base_model.load_state_dict(torch.load(h.eval_only_path, map_location=device))
        if h.num_loops > 0:
            base_model.looping_active = True
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
        compiled_forward_logits = torch.compile(
            base_model.forward_logits, dynamic=False, fullgraph=True
        )
    else:
        log(
            f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}"
        )
        log(f"val_tokens: {val_data.val_tokens.numel()-1}")
        base_model, compiled_model, compiled_forward_logits = train_model(
            h, device, val_data
        )
    _skip_training = bool(h.eval_only_path)
    torch._dynamo.reset()
    timed_eval(
        "diagnostic pre-quantization post-ema",
        eval_val,
        h,
        device,
        val_data,
        compiled_model,
        compiled_forward_logits,
    )
    if not _skip_training:
        serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
    else:
        log("eval_only: skipping serialize (already have quantized model)")
        if not os.path.exists(h.quantized_model_path):
            log("eval_only: no quantized model found, running serialize anyway")
            serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
    if h.distributed:
        dist.barrier()
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    compiled_forward_logits = torch.compile(
        eval_model.forward_logits, dynamic=False, fullgraph=True
    )
    timed_eval(
        "diagnostic quantized",
        eval_val,
        h,
        device,
        val_data,
        compiled_model,
        compiled_forward_logits,
    )
    if h.sliding_window_enabled:
        timed_eval(
            "diagnostic quantized_sliding_window",
            eval_val_sliding,
            h,
            device,
            val_data,
            eval_model,
            forward_logits_fn=compiled_forward_logits,
        )
    if h.ttt_enabled:
        del eval_model, compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        if h.num_loops > 0:
            ttt_model.looping_active = True
        for p in ttt_model.parameters():
            p.requires_grad_(False)

        if h.rope_yarn:
            _yarn_seqlen = h.train_batch_tokens // h.grad_accum_steps
            for block in ttt_model.blocks:
                block.attn.rotary(_yarn_seqlen, device, torch.bfloat16)
        else:
            for block in ttt_model.blocks:
                block.attn.rotary._cos_cached = None
                block.attn.rotary._sin_cached = None
                block.attn.rotary._seq_len_cached = 0
                block.attn.rotary(h.ttt_eval_seq_len, device, torch.bfloat16)

        def _fwd_ttt_inner(input_ids, target_ids, lora):
            return ttt_model.forward_ttt(input_ids, target_ids, lora=lora)

        _fwd_ttt_compiled_inner = None

        def _fwd_ttt(input_ids, target_ids, lora):
            nonlocal _fwd_ttt_compiled_inner
            if _fwd_ttt_compiled_inner is None:
                _fwd_ttt_compiled_inner = torch.compile(_fwd_ttt_inner, dynamic=True)
            return _fwd_ttt_compiled_inner(input_ids, target_ids, lora=lora)

        _ttt_debug_bypass = bool(os.environ.get("TTT_DEBUG_BYPASS"))
        if _ttt_debug_bypass:
            def _fwd_ttt_bypass(input_ids, target_ids, lora):
                logits = ttt_model.forward_logits(input_ids)
                dummy = lora.q_loras[0].B.sum() * 0
                logits = logits + dummy
                bsz, sl, V = logits.shape
                return F.cross_entropy(
                    logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none"
                ).reshape(bsz, sl)
            fwd_ttt_compiled = _fwd_ttt_bypass
            log("ttt_lora:DEBUG BYPASS active - using forward_logits directly (no compile warmup)")
        else:
            fwd_ttt_compiled = _fwd_ttt
            log(f"ttt_lora:warming up compile (random tokens, no val data)")
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
        ttt_val_loss, ttt_val_bpb = eval_val_ttt_lora(
            h, ttt_model, device, val_data, forward_ttt_train=fwd_ttt_compiled
        )
        torch.cuda.synchronize()
        ttt_eval_elapsed = time.perf_counter() - t_ttt
        log(
            f"quantized_ttt_lora val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f} eval_time:{1e3*ttt_eval_elapsed:.0f}ms"
        )
        log(f"total_eval_time:{ttt_eval_elapsed:.1f}s")
        del ttt_model


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
    torch._dynamo.config.cache_size_limit = 16
    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs(h.artifact_dir if h.artifact_dir else "logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:", console=True)
        for (k, v) in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log("=" * 100, console=False)
        log("Source code:", console=False)
        log("=" * 100, console=False)
        with open(__file__, "r", encoding="utf-8") as _src:
            log(_src.read(), console=False)
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
