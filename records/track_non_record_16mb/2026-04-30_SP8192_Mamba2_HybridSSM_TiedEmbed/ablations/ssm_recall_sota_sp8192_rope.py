import copy
from contextlib import nullcontext
import glob
import math
import os
import pickle
import random
import subprocess
import sys
import time
import threading
import queue
import zlib
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from mamba_ssm.modules.mamba2 import Mamba2

# Strictly require fused CUDA kernels from mamba_ssm.
try:
    import selective_scan_cuda  # type: ignore  # noqa: F401
    import causal_conv1d_cuda  # type: ignore  # noqa: F401
except Exception as e:
    raise RuntimeError(
        "mamba_ssm CUDA kernels are unavailable. Install matching CUDA build of "
        "`mamba-ssm` and `causal-conv1d` for this PyTorch/CUDA environment."
    ) from e

# Memory allocator config — set before any CUDA allocation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DATA_PATH = "./data/datasets/fineweb10B_sp8192"
TOK_PATH = "./data/tokenizers/fineweb_8192_bpe.model"
VOCAB_SIZE = 8192
SEQ_LEN = int(os.environ.get("SEQ_LEN", "2048"))
TRAIN_SHARDS = int(os.environ.get("TRAIN_SHARDS", "80"))
DOWNLOAD_DATA = os.environ.get("DOWNLOAD_DATA", "0") == "1"
USE_ASYNC_LOADER = os.environ.get("USE_ASYNC_LOADER", "1") == "1"
# sp8192 lives in a fork — set MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf
os.environ.setdefault("MATCHED_FINEWEB_REPO_ID", "kevclark/parameter-golf")

# SSM architecture
D_MODEL = 512
D_STATE = int(os.environ.get("D_STATE", "64"))
N_UNIQUE_BLOCKS = int(os.environ.get("N_UNIQUE_BLOCKS", "10"))
N_UNROLLS = int(os.environ.get("N_UNROLLS", "1"))
CONV_KERNEL = 4
FFN_MULT = int(os.environ.get("FFN_MULT", "3"))
FFN_EVERY = int(os.environ.get("FFN_EVERY", "1"))  # Run FFN every Nth layer/unroll
FFN_FREQ_MODE = os.environ.get("FFN_FREQ_MODE", "layer").lower()  # "layer" or "unroll"
MEMORY_DIM = 256
LOGIT_SOFTCAP = 30.0
EMBED_INIT_STD = 0.005
USE_MEMORY = os.environ.get("USE_MEMORY", "0") == "1"
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "1") == "1"

# Factored embedding: tok_emb at EMBED_DIM, project to D_MODEL.
# Saves params at vocab=8192. SOTA uses similar factoring (kev's 8192 setup).
EMBED_DIM = int(os.environ.get("EMBED_DIM", "512"))  # best current: factored 512

# Untied output head + neural copy/pointer head.
# Goal: improve token-output expressivity and exact/contextual token reuse.
UNTIE_LM_HEAD = os.environ.get("UNTIE_LM_HEAD", "1") == "1"
LM_HEAD_BIAS = os.environ.get("LM_HEAD_BIAS", "1") == "1"
COPY_HEAD_ENABLED = os.environ.get("COPY_HEAD_ENABLED", "0") == "1"
COPY_DIM = int(os.environ.get("COPY_DIM", "64"))
COPY_GATE_BIAS_INIT = float(os.environ.get("COPY_GATE_BIAS_INIT", "-2.0"))
COPY_SCALE_INIT = float(os.environ.get("COPY_SCALE_INIT", "1.0"))
COPY_USE_QK_NORM = os.environ.get("COPY_USE_QK_NORM", "1") == "1"

# Hybrid attention config
ATTN_LAYER_IDXS = [int(x) for x in os.environ.get("ATTN_LAYER_IDXS", "6").split(",") if x.strip()]
ATTN_N_HEADS = int(os.environ.get("ATTN_N_HEADS", "8"))
QK_GAIN_INIT = float(os.environ.get("QK_GAIN_INIT", "5.25"))  # best current: learnable per-head query scaling

# Partial RoPE in the attention checkpoint.
# Useful for 8192-context tests: lets a subset of each attention head encode relative position
# while leaving the remaining dimensions content-only.
ROPE_ENABLED = os.environ.get("ROPE_ENABLED", "0") == "1"
ROPE_DIM = int(os.environ.get("ROPE_DIM", "16"))
ROPE_BASE = float(os.environ.get("ROPE_BASE", "10000.0"))

# Activation choice
ACTIVATION = os.environ.get("ACTIVATION", "swiglu").lower()  # "swiglu" or "leaky_relu2"

# Depth recurrence — loop layers [LOOP_START..LOOP_END] inclusive twice during decoder phase
LOOP_START = int(os.environ.get("LOOP_START", "-1"))  # -1 = disabled
LOOP_END = int(os.environ.get("LOOP_END", "-1"))      # inclusive
LOOP_ACTIVATE_FRAC = float(os.environ.get("LOOP_ACTIVATE_FRAC", "0.35"))

# Training
ITERATIONS = int(os.environ.get("ITERATIONS", "20000"))
VAL_EVERY = int(os.environ.get("VAL_EVERY", "0"))  # 0=final only (each val costs ~100 steps)
LOG_EVERY = int(os.environ.get("LOG_EVERY", "500"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "0.3"))
STREAM_CHUNKS = int(os.environ.get("STREAM_CHUNKS", "1"))
BPTT_CHUNKS = int(os.environ.get("BPTT_CHUNKS", "1"))
SEED = int(os.environ.get("SEED", "7"))
MAX_WALLCLOCK_SECONDS = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "600"))

# LR schedule: short warmup, then long warmdown (matching competitive GPT config)
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "20"))
LR_MIN_SCALE = float(os.environ.get("LR_MIN_SCALE", "0.0"))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine_late")
LR_WARMDOWN_START_FRAC = float(os.environ.get("LR_WARMDOWN_START_FRAC", "0.28"))  # SOTA: 0.72 of training is warmdown

# Batch size warmup
BATCH_WARMUP_STEPS = int(os.environ.get("BATCH_WARMUP_STEPS", "0"))
BATCH_WARMUP_FRAC = float(os.environ.get("BATCH_WARMUP_FRAC", "0.25"))

# BigramHash embedding
BIGRAM_ENABLED = os.environ.get("BIGRAM_ENABLED", "1") == "1"
BIGRAM_BUCKETS = int(os.environ.get("BIGRAM_BUCKETS", "10240"))
BIGRAM_DIM = int(os.environ.get("BIGRAM_DIM", "128"))

# Sliding window eval — stride < SEQ_LEN gives each token more context
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", "64"))  # 64 matches SOTA

# Optimizer — higher LRs for 42M model, matching SOTA WD
MATRIX_LR = float(os.environ.get("MATRIX_LR", "0.022"))  # SOTA: 0.022
SCALAR_LR = float(os.environ.get("SCALAR_LR", "0.02"))   # SOTA: 0.02
EMBED_LR = float(os.environ.get("EMBED_LR", "0.03"))     # SOTA: tied_embed_lr=0.03
BETA1, BETA2 = 0.9, 0.95
ADAM_EPS = 1e-8
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.095"))  # SOTA: muon_wd=0.095
ENABLE_S4D_INIT = os.environ.get("ENABLE_S4D_INIT", "1") == "1"

MUON_MOMENTUM = float(os.environ.get("MUON_MOMENTUM", "0.99"))
MUON_MOMENTUM_WARMUP_START = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", "0.85"))
MUON_MOMENTUM_WARMUP_STEPS = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", "500"))
MUON_BACKEND_STEPS = int(os.environ.get("MUON_BACKEND_STEPS", "5"))
MUON_NESTEROV = os.environ.get("MUON_NESTEROV", "1") == "1"
MUON_ONLY_2D = os.environ.get("MUON_ONLY_2D", "1") == "1"

if D_STATE > 256:
    raise RuntimeError(
        f"D_STATE={D_STATE} is unsupported by the fused mamba_ssm kernel in this build. "
        "Please set D_STATE <= 256 (e.g., D_STATE=256)."
    )

CTRL_PATTERNS = (
    "ssm_scale",
    "mlp_scale",
    "attn_scale",
    "resid_mix",
    "mem_to_model",
    "mem_in_gate",
    "memory_proj",
    "memory_gate",
    "q_gain",
)

# -----------------------------------------------------------------------------
# Distributed setup (strict 6xGPU)
# -----------------------------------------------------------------------------
assert torch.cuda.is_available(), "CUDA is required."
DISTRIBUTED = "RANK" in os.environ and "WORLD_SIZE" in os.environ
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "8"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

if not DISTRIBUTED:
    raise RuntimeError(
        "This script requires torchrun distributed launch. "
        "Example: torchrun --standalone --nproc_per_node=8 ssm_recall_tied64.py"
    )
if WORLD_SIZE != 8:
    raise RuntimeError(f"Expected WORLD_SIZE=8, got WORLD_SIZE={WORLD_SIZE}")

torch.cuda.set_device(LOCAL_RANK)
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", LOCAL_RANK))
DEVICE = torch.device("cuda", LOCAL_RANK)
MASTER_PROCESS = RANK == 0
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def log0(msg: str):
    if MASTER_PROCESS:
        print(msg, flush=True)


def ensure_repo_root() -> Path:
    repo_url = os.environ.get("REPO_URL", "https://github.com/openai/parameter-golf.git")
    clone_dir = Path(os.environ.get("REPO_CLONE_DIR", "/workspace/parameter-golf-repo")).resolve()
    candidates = [Path.cwd().resolve(), Path(__file__).resolve().parent, clone_dir]
    for c in candidates:
        if (c / "data" / "cached_challenge_fineweb.py").exists():
            return c
    if MASTER_PROCESS:
        log0(f"Repo not found locally, cloning from {repo_url} -> {clone_dir}")
        if not clone_dir.exists():
            subprocess.run(["git", "clone", repo_url, str(clone_dir)], check=True)
        elif not (clone_dir / ".git").exists():
            raise RuntimeError(f"REPO_CLONE_DIR exists but is not a git repo: {clone_dir}")
    dist.barrier()
    if not (clone_dir / "data" / "cached_challenge_fineweb.py").exists():
        raise RuntimeError(
            "Could not locate data/cached_challenge_fineweb.py. "
            "Set REPO_CLONE_DIR to a valid parameter-golf checkout."
        )
    return clone_dir


PROJECT_ROOT = ensure_repo_root()
os.chdir(PROJECT_ROOT)
log0(f"Project root: {PROJECT_ROOT}")


def gather_gpu_inventory():
    props = torch.cuda.get_device_properties(DEVICE)
    local = {
        "rank": RANK,
        "local_rank": LOCAL_RANK,
        "name": props.name,
        "total_mem_gb": round(props.total_memory / (1024**3), 2),
    }
    all_inv = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(all_inv, local)
    if MASTER_PROCESS:
        print("GPU inventory:")
        for row in all_inv:
            print(row)
        bad = [row for row in all_inv if "H100" not in row["name"]]
        if bad:
            raise RuntimeError(f"Non-H100 devices detected: {bad}")


gather_gpu_inventory()

# ---------------------------------------------------------------------------
# Batch config
# ---------------------------------------------------------------------------
TOK_PER_RANK_TARGET = int(os.environ.get("TOK_PER_RANK", "65536"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
TRAIN_BATCH_TOK = TOK_PER_RANK_TARGET * WORLD_SIZE
LOCAL_BATCH_TOK = TOK_PER_RANK_TARGET

seed_offset = SEED + RANK
random.seed(seed_offset)
np.random.seed(seed_offset)
torch.manual_seed(seed_offset)
torch.cuda.manual_seed_all(seed_offset)

EFF_LAYERS = N_UNIQUE_BLOCKS * N_UNROLLS
log0(f"AMP dtype: {AMP_DTYPE}")
log0(f"Config: {N_UNIQUE_BLOCKS} unique x {N_UNROLLS} unrolls = {EFF_LAYERS} effective layers")
log0(f"d_model={D_MODEL}, d_state={D_STATE}, ffn_mult={FFN_MULT}, memory_dim={MEMORY_DIM}")
log0(f"ffn_every={FFN_EVERY}, ffn_freq_mode={FFN_FREQ_MODE}")
log0(f"train_batch_tok(global)={TRAIN_BATCH_TOK}, local_batch_tok={LOCAL_BATCH_TOK}, grad_accum={GRAD_ACCUM}")
log0(f"effective_batch_tok/step={TRAIN_BATCH_TOK}, micro_batch_tok/rank={LOCAL_BATCH_TOK // max(GRAD_ACCUM, 1)}")
log0(f"warmup_steps={WARMUP_STEPS}, iterations={ITERATIONS}, max_wallclock_seconds={MAX_WALLCLOCK_SECONDS}")
log0(
    f"lr_schedule={LR_SCHEDULE}, lr_min_scale={LR_MIN_SCALE}, "
    f"enable_s4d_init={ENABLE_S4D_INIT}"
)
log0(f"stream_chunks={STREAM_CHUNKS}, bptt_chunks={BPTT_CHUNKS}")
log0(f"use_async_loader={USE_ASYNC_LOADER}")
log0(f"use_memory={USE_MEMORY}, use_torch_compile={USE_TORCH_COMPILE}")
log0(f"vocab={VOCAB_SIZE}, factored_embed_dim={EMBED_DIM} ({'enabled' if EMBED_DIM > 0 else 'disabled'})")
log0(
    f"output_copy: untie_lm_head={UNTIE_LM_HEAD}, lm_head_bias={LM_HEAD_BIAS}, "
    f"copy_enabled={COPY_HEAD_ENABLED}, copy_dim={COPY_DIM}, "
    f"copy_gate_bias_init={COPY_GATE_BIAS_INIT}, copy_scale_init={COPY_SCALE_INIT}"
)
log0(f"hybrid_attn: layer_idxs={ATTN_LAYER_IDXS}, n_heads={ATTN_N_HEADS}, qk_gain_init={QK_GAIN_INIT}")
log0(f"rope: enabled={ROPE_ENABLED}, dim={ROPE_DIM}, base={ROPE_BASE}")
log0(f"activation: {ACTIVATION}")
log0(f"depth_recur: loop=[{LOOP_START}..{LOOP_END}] activate@{LOOP_ACTIVATE_FRAC}")
log0(f"bigram: enabled={BIGRAM_ENABLED}, buckets={BIGRAM_BUCKETS}, dim={BIGRAM_DIM}")
log0(f"eval_stride={EVAL_STRIDE}")
log0(f"optimizer: matrix_lr={MATRIX_LR}, scalar_lr={SCALAR_LR}, embed_lr={EMBED_LR}, wd={WEIGHT_DECAY}, grad_clip={GRAD_CLIP}")
log0(f"batch_warmup: steps={BATCH_WARMUP_STEPS}, start_frac={BATCH_WARMUP_FRAC}")
log0(
    f"hybrid_optimizer: muon_momentum={MUON_MOMENTUM}, "
    f"muon_momentum_warmup={MUON_MOMENTUM_WARMUP_START}->{MUON_MOMENTUM} over {MUON_MOMENTUM_WARMUP_STEPS} steps, "
    f"muon_backend_steps={MUON_BACKEND_STEPS}, muon_nesterov={MUON_NESTEROV}"
)

if GRAD_ACCUM < 1:
    raise ValueError(f"GRAD_ACCUM must be >= 1, got {GRAD_ACCUM}")
if STREAM_CHUNKS < 1:
    raise ValueError(f"STREAM_CHUNKS must be >= 1, got {STREAM_CHUNKS}")
if BPTT_CHUNKS < 1:
    raise ValueError(f"BPTT_CHUNKS must be >= 1, got {BPTT_CHUNKS}")
if SEQ_LEN % STREAM_CHUNKS != 0:
    raise ValueError(f"SEQ_LEN ({SEQ_LEN}) must be divisible by STREAM_CHUNKS ({STREAM_CHUNKS})")


# -----------------------------------------------------------------------------
# Data — threaded async loader with pinned memory
# -----------------------------------------------------------------------------
def maybe_download_data():
    if not DOWNLOAD_DATA:
        dist.barrier()
        return
    if MASTER_PROCESS:
        subprocess.run(
            [sys.executable, "data/cached_challenge_fineweb.py",
             "--variant", "sp8192", "--train-shards", str(TRAIN_SHARDS)],
            check=True,
        )
        log0("Data download complete.")
    dist.barrier()


def load_data_shard(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    assert header.size == 256 and int(header[0]) == 20240520 and int(header[1]) == 1
    n = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    return np.fromfile(path, dtype="<u2", count=n, offset=offset).astype(np.int32, copy=False)


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        assert self.files, f"No files for {pattern}"
        self.idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def reset(self):
        """Reset to beginning of first shard — replays all previously seen tokens."""
        self.idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.idx = (self.idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.idx])
        self.pos = 0

    def take(self, n, return_boundaries=False):
        parts, left = [], n
        boundaries = np.zeros(n, dtype=np.bool_) if return_boundaries else None
        out_pos = 0
        while left > 0:
            if self.pos >= self.tokens.size:
                self._advance()
                if return_boundaries and out_pos < n:
                    boundaries[out_pos] = True
            k = min(left, self.tokens.size - self.pos)
            parts.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
            out_pos += k
        chunk = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if return_boundaries:
            return chunk, boundaries
        return chunk


def build_chunk_reset_mask(boundaries, seq_len, stream_chunks):
    """
    Build per-sequence reset flags for each streamed chunk.
    boundaries marks token positions in local stream where a new shard begins.
    """
    if stream_chunks <= 1:
        return None
    if seq_len % stream_chunks != 0:
        raise ValueError(f"SEQ_LEN ({seq_len}) must be divisible by STREAM_CHUNKS ({stream_chunks})")
    n_tokens = boundaries.size - 1  # x/y consume span-1 tokens
    n_seqs = n_tokens // seq_len
    chunk_len = seq_len // stream_chunks
    reset = np.zeros((n_seqs, stream_chunks), dtype=np.bool_)
    for s in range(n_seqs):
        base = s * seq_len
        for c in range(1, stream_chunks):
            prev_end = base + (c * chunk_len)
            prev_start = base + ((c - 1) * chunk_len)
            if np.any(boundaries[prev_start + 1 : prev_end + 1]):
                reset[s, c] = True
    return reset


class AsyncDistributedTokenLoader:
    """
    Background-threaded data loader:
    - Worker thread reads from disk + slices numpy on CPU
    - Pins tensors to page-locked memory
    - Transfers to GPU on a dedicated CUDA stream
    - Main thread picks up ready batches with zero wait
    """
    def __init__(self, pattern, rank, world_size, global_tokens, seq_len,
                 grad_accum_steps, device, prefetch_depth=3):
        self.stream_obj = TokenStream(pattern)
        self.rank = rank
        self.world_size = world_size
        self.global_tokens = global_tokens
        self.seq_len = seq_len
        self.grad_accum_steps = grad_accum_steps
        self.device = device
        self.cuda_stream = torch.cuda.Stream(device=device)

        # Queue of ready (x_gpu, y_gpu) batches
        self.ready_queue = queue.Queue(maxsize=prefetch_depth)
        self.stop_event = threading.Event()

        # Start background worker
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _produce_one_batch(self):
        """Read from disk, slice, pin, transfer to GPU."""
        local_tokens = self.global_tokens // (self.world_size * self.grad_accum_steps)
        local_tokens = (local_tokens // self.seq_len) * self.seq_len
        span = local_tokens + 1
        chunk, boundaries = self.stream_obj.take(span * self.world_size, return_boundaries=True)
        local = chunk[self.rank * span : (self.rank + 1) * span]
        local_boundaries = boundaries[self.rank * span : (self.rank + 1) * span]
        reset_np = build_chunk_reset_mask(local_boundaries, self.seq_len, STREAM_CHUNKS)

        # Create pinned tensors on CPU
        x_cpu = torch.from_numpy(local[:-1].reshape(-1, self.seq_len)).long().pin_memory()
        y_cpu = torch.from_numpy(local[1:].reshape(-1, self.seq_len)).long().pin_memory()

        # Async transfer on dedicated stream
        with torch.cuda.stream(self.cuda_stream):
            x_gpu = x_cpu.to(self.device, non_blocking=True)
            y_gpu = y_cpu.to(self.device, non_blocking=True)
            reset_gpu = (
                torch.from_numpy(reset_np).to(self.device, non_blocking=True)
                if reset_np is not None else None
            )

        # Record event so consumer knows when transfer is done
        event = self.cuda_stream.record_event()
        return x_gpu, y_gpu, reset_gpu, event

    def _worker_loop(self):
        """Continuously prefetch batches in background thread."""
        while not self.stop_event.is_set():
            try:
                batch = self._produce_one_batch()
                self.ready_queue.put(batch, timeout=1.0)
            except queue.Full:
                continue
            except Exception:
                if self.stop_event.is_set():
                    break
                raise

    def next_batch(self):
        """Get the next ready batch. Waits for GPU transfer to complete."""
        x_gpu, y_gpu, reset_gpu, event = self.ready_queue.get()
        event.wait()  # ensure H2D transfer finished
        return x_gpu, y_gpu, reset_gpu

    def reset_stream(self):
        """Drain prefetch queue and reset token stream to replay from start."""
        while not self.ready_queue.empty():
            try:
                self.ready_queue.get_nowait()
            except queue.Empty:
                break
        self.stream_obj.reset()

    def shutdown(self):
        self.stop_event.set()
        self.worker.join(timeout=5)


class DistributedTokenLoader:
    """
    Synchronous loader: deterministic and simpler collective behavior.
    """
    def __init__(self, pattern, rank, world_size, global_tokens, seq_len, grad_accum_steps, device):
        self.stream_obj = TokenStream(pattern)
        self.rank = rank
        self.world_size = world_size
        self.global_tokens = global_tokens
        self.seq_len = seq_len
        self.grad_accum_steps = grad_accum_steps
        self.device = device

    def next_batch(self):
        local_tokens = self.global_tokens // (self.world_size * self.grad_accum_steps)
        local_tokens = (local_tokens // self.seq_len) * self.seq_len
        span = local_tokens + 1
        chunk, boundaries = self.stream_obj.take(span * self.world_size, return_boundaries=True)
        local = chunk[self.rank * span : (self.rank + 1) * span]
        local_boundaries = boundaries[self.rank * span : (self.rank + 1) * span]
        reset_np = build_chunk_reset_mask(local_boundaries, self.seq_len, STREAM_CHUNKS)
        x = torch.from_numpy(local[:-1].reshape(-1, self.seq_len)).long().to(self.device, non_blocking=True)
        y = torch.from_numpy(local[1:].reshape(-1, self.seq_len)).long().to(self.device, non_blocking=True)
        reset = torch.from_numpy(reset_np).to(self.device, non_blocking=True) if reset_np is not None else None
        return x, y, reset

    def reset_stream(self):
        """Reset token stream to replay from start."""
        self.stream_obj.reset()

    def shutdown(self):
        return


def load_val_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    assert files, f"No val files for {pattern}"
    tokens = np.concatenate([load_data_shard(f) for f in files])
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def build_sp_luts(sp, vocab_size):
    sz = max(int(sp.vocab_size()), vocab_size)
    base_bytes = np.zeros(sz, dtype=np.int16)
    has_space = np.zeros(sz, dtype=np.bool_)
    is_bound = np.ones(sz, dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_bound[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return base_bytes, has_space, is_bound


# -----------------------------------------------------------------------------
# Model — Hybrid Mamba2 + Sliding Window Attention
# -----------------------------------------------------------------------------
def rms_norm(x, eps=1e-6):
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), eps=eps)
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return rms_norm(x, self.eps) * self.weight


class SwiGLU_FFN(nn.Module):
    """SwiGLU: gate * swish(gate_proj) * up_proj. LeakyReLU² variant if ACTIVATION='leaky_relu2'."""
    def __init__(self, d_model, ffn_mult=3):
        super().__init__()
        hidden = d_model * ffn_mult
        if ACTIVATION == "leaky_relu2":
            # SOTA-style: single up matrix, leaky_relu(0.5)² activation
            self.gate_up = nn.Linear(d_model, hidden, bias=False)
            self.down = nn.Linear(hidden, d_model, bias=False)
            self.is_swiglu = False
        else:
            # SwiGLU: fused gate + up
            self.gate_up = nn.Linear(d_model, hidden * 2, bias=False)
            self.down = nn.Linear(hidden, d_model, bias=False)
            self.is_swiglu = True
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        if self.is_swiglu:
            gu = self.gate_up(x)
            gate, up = gu.chunk(2, dim=-1)
            return self.down(F.silu(gate) * up)
        else:
            h = F.leaky_relu(self.gate_up(x), negative_slope=0.5)
            return self.down(h * h)  # LeakyReLU(0.5)²


class SelectiveSSMBlock(nn.Module):
    def __init__(self, d_model, d_state, memory_dim, conv_kernel=4, ffn_mult=3):
        super().__init__()
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=conv_kernel,
            expand=2,
        )
        self.ssm_norm = RMSNorm(d_model)
        self.ssm_scale = nn.Parameter(torch.ones(d_model))
        self.ffn = SwiGLU_FFN(d_model, ffn_mult)
        self.ffn_norm = RMSNorm(d_model)
        self.mlp_scale = nn.Parameter(torch.ones(d_model))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(d_model), torch.zeros(d_model)]))

        # Memory params (only used if USE_MEMORY)
        if USE_MEMORY:
            self.mem_to_model = nn.Linear(memory_dim, d_model, bias=False)
            self.mem_in_gate = nn.Parameter(torch.tensor(0.0))
            self.memory_proj = nn.Linear(d_model, memory_dim, bias=False)
            self.memory_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, x0, mem, run_ffn=True):
        mix = self.resid_mix
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if USE_MEMORY and mem is not None:
            mem_ctx = self.mem_to_model(mem)[:, None, :]
            mem_mix = torch.sigmoid(self.mem_in_gate)
            x_in = x + mem_mix * mem_ctx
        else:
            x_in = x
        y = self.mamba(self.ssm_norm(x_in))
        x = x + self.ssm_scale[None, None, :] * y
        if run_ffn:
            x = x + self.mlp_scale[None, None, :] * self.ffn(self.ffn_norm(x))
        if USE_MEMORY and mem is not None:
            new_mem = torch.tanh(self.memory_proj(x[:, -1, :]))
            g = torch.sigmoid(self.memory_gate)
            mem = g * mem + (1.0 - g) * new_mem
        return x, mem



def apply_partial_rope(q: torch.Tensor, k: torch.Tensor, rotary_dim: int, base: float):
    """
    Apply partial RoPE to q/k tensors shaped (B, H, T, D).
    Only the first rotary_dim dimensions are rotated; remaining dims are content-only.
    """
    if not ROPE_ENABLED or rotary_dim <= 0:
        return q, k
    head_dim = q.shape[-1]
    rd = min(int(rotary_dim), head_dim)
    rd = rd - (rd % 2)
    if rd <= 0:
        return q, k

    device = q.device
    T = q.shape[-2]
    inv_freq = 1.0 / (base ** (torch.arange(0, rd, 2, device=device, dtype=torch.float32) / rd))
    pos = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (T, rd/2)
    cos = freqs.cos()[None, None, :, :]  # (1,1,T,rd/2)
    sin = freqs.sin()[None, None, :, :]

    def rotate(x):
        x_rope = x[..., :rd].float()
        x_pass = x[..., rd:]
        x_even = x_rope[..., 0::2]
        x_odd = x_rope[..., 1::2]
        x_rot = torch.stack((x_even * cos - x_odd * sin,
                             x_even * sin + x_odd * cos), dim=-1).flatten(-2)
        return torch.cat((x_rot.to(dtype=x.dtype), x_pass), dim=-1)

    return rotate(q), rotate(k)



class CausalAttentionBlock(nn.Module):
    """Causal attention with learnable per-head QK gain (SOTA: QK_GAIN_INIT=5.25)."""
    def __init__(self, d_model, n_heads=8, ffn_mult=3):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.attn_norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        # Learnable per-head query gain (SOTA technique)
        self.q_gain = nn.Parameter(torch.full((n_heads,), QK_GAIN_INIT))

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU_FFN(d_model, ffn_mult)
        self.attn_scale = nn.Parameter(torch.ones(d_model))
        self.mlp_scale = nn.Parameter(torch.ones(d_model))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(d_model), torch.zeros(d_model)]))

    def forward(self, x, x0, mem, run_ffn=True):
        mix = self.resid_mix
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        B, T, D = x.shape
        normed = self.attn_norm(x)
        qkv = self.qkv(normed).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)

        # QK RMSNorm + optional partial RoPE + learnable per-head query gain.
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        q, k = apply_partial_rope(q, k, ROPE_DIM, ROPE_BASE)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        x = x + self.attn_scale[None, None, :] * self.out_proj(attn_out)

        if run_ffn:
            x = x + self.mlp_scale[None, None, :] * self.ffn(self.ffn_norm(x))
        return x, mem  # pass mem through unchanged


class BigramHash(nn.Module):
    """Hash consecutive token pairs into a learned embedding table."""
    def __init__(self, vocab_size, n_buckets, bigram_dim, d_model):
        super().__init__()
        self.n_buckets = n_buckets
        self.embed = nn.Embedding(n_buckets, bigram_dim)
        self.proj = nn.Linear(bigram_dim, d_model, bias=False)
        nn.init.normal_(self.embed.weight, std=0.01)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, ids):
        # ids: (B, T) token ids
        # Compute bigram hashes: hash(ids[t-1], ids[t]) for each position
        # For position 0, use a zero bigram (no previous token)
        prev = F.pad(ids[:, :-1], (1, 0), value=0)  # (B, T) — shifted right, pad with 0
        bigram_hash = (prev * 1009 + ids) % self.n_buckets  # simple hash
        bigram_emb = self.embed(bigram_hash)  # (B, T, bigram_dim)
        return self.proj(bigram_emb)  # (B, T, d_model)


class SSM_LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.softcap = LOGIT_SOFTCAP
        self.n_unrolls = N_UNROLLS
        self.memory_dim = MEMORY_DIM

        # Factored embedding for vocab=8192: tok_emb at EMBED_DIM, project to D_MODEL.
        # Without factoring, vocab*D_MODEL=8192*512=4.2M params just for embeddings.
        # With EMBED_DIM=256: tok_emb=2.1M + projections=131K+131K = ~2.4M (saves ~1.8M).
        self.use_factored = EMBED_DIM > 0
        if self.use_factored:
            self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
            self.embed_proj = nn.Linear(EMBED_DIM, D_MODEL, bias=False)
            self.embed_proj_rev = nn.Linear(D_MODEL, EMBED_DIM, bias=False)
            self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=LM_HEAD_BIAS)
        else:
            self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
            self.embed_proj = None
            self.embed_proj_rev = None
            self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=LM_HEAD_BIAS)
        with torch.no_grad():
            self.tok_emb.weight.normal_(mean=0.0, std=EMBED_INIT_STD)
            if UNTIE_LM_HEAD:
                # Start from the tied solution for stability, then let output
                # classifier specialize separately from input embeddings.
                if self.lm_head.weight.shape == self.tok_emb.weight.shape:
                    self.lm_head.weight.copy_(self.tok_emb.weight)
                else:
                    self.lm_head.weight.normal_(mean=0.0, std=EMBED_INIT_STD)
                if self.lm_head.bias is not None:
                    self.lm_head.bias.zero_()
            else:
                # Keep tied weights; optional bias remains separate.
                self.lm_head.weight = self.tok_emb.weight
                if self.lm_head.bias is not None:
                    self.lm_head.bias.zero_()

        # Neural copy/pointer head.
        # It computes learned Q/K over hidden states, scatters attention mass
        # onto source token IDs in the causal context, and adds a small gated
        # positive bonus to those vocabulary logits.
        if COPY_HEAD_ENABLED:
            self.copy_q_norm = RMSNorm(D_MODEL)
            self.copy_k_norm = RMSNorm(D_MODEL)
            self.copy_q = nn.Linear(D_MODEL, COPY_DIM, bias=False)
            self.copy_k = nn.Linear(D_MODEL, COPY_DIM, bias=False)
            self.copy_gate = nn.Linear(D_MODEL, 1, bias=True)
            self.copy_scale = nn.Parameter(torch.tensor(float(COPY_SCALE_INIT)))
            with torch.no_grad():
                self.copy_gate.weight.zero_()
                self.copy_gate.bias.fill_(COPY_GATE_BIAS_INIT)
        else:
            self.copy_q_norm = None
            self.copy_k_norm = None
            self.copy_q = None
            self.copy_k = None
            self.copy_gate = None
            self.copy_scale = None

        # BigramHash embedding
        self.bigram = BigramHash(VOCAB_SIZE, BIGRAM_BUCKETS, BIGRAM_DIM, D_MODEL) if BIGRAM_ENABLED else None

        # Build layers: SSM blocks + attention layers at specified positions
        eff = N_UNIQUE_BLOCKS * N_UNROLLS
        attn_set = set(ATTN_LAYER_IDXS)
        layers = []
        for i in range(N_UNIQUE_BLOCKS):
            if i in attn_set:
                layers.append(CausalAttentionBlock(
                    D_MODEL, ATTN_N_HEADS, FFN_MULT))
            else:
                layers.append(SelectiveSSMBlock(
                    D_MODEL, D_STATE, MEMORY_DIM, CONV_KERNEL, FFN_MULT))
        self.blocks = nn.ModuleList(layers)

        self.n_enc = eff // 2
        self.n_skip = min(self.n_enc, eff - self.n_enc)
        self.skip_weights = nn.Parameter(torch.ones(self.n_skip, D_MODEL))
        self.final_norm = RMSNorm(D_MODEL)

        # Depth recurrence (SOTA): re-run a window of layers to gain virtual depth
        # for free. Built-in indices: encoder visits each layer once; decoder loops
        # the [LOOP_START..LOOP_END] window if enabled.
        self.loop_enabled_external = False  # toggled at runtime by training loop
        if LOOP_START >= 0 and LOOP_END >= LOOP_START:
            # Build encoder/decoder layer index sequences with loop inserted in encoder.
            # Mirroring SOTA: encoder=[0..LE, LS..LE-1], decoder=[LE, LS..N-1]
            enc_seq = list(range(0, LOOP_END + 1)) + list(range(LOOP_START, LOOP_END))
            dec_seq = [LOOP_END] + list(range(LOOP_START, N_UNIQUE_BLOCKS))
            self.loop_enc_seq = enc_seq
            self.loop_dec_seq = dec_seq
        else:
            self.loop_enc_seq = None
            self.loop_dec_seq = None

    def output_logits_from_hidden(self, hidden):
        if self.use_factored:
            flat_h = hidden.reshape(-1, D_MODEL)
            h_proj = F.linear(flat_h, self.embed_proj_rev.weight.to(flat_h.dtype))
            logits = F.linear(
                h_proj,
                self.lm_head.weight.to(h_proj.dtype),
                self.lm_head.bias.to(h_proj.dtype) if self.lm_head.bias is not None else None,
            )
        else:
            flat_h = hidden.reshape(-1, self.lm_head.weight.shape[1])
            logits = F.linear(
                flat_h,
                self.lm_head.weight.to(flat_h.dtype),
                self.lm_head.bias.to(flat_h.dtype) if self.lm_head.bias is not None else None,
            )
        return self.softcap * torch.tanh(logits / self.softcap)

    def add_copy_logits(self, logits, query_hidden, source_hidden, source_ids, query_start=None):
        """
        query_hidden: (B, U, D), positions being scored.
        source_hidden: (B, T, D), full visible context hidden states.
        source_ids: (B, T), token IDs corresponding to source_hidden.
        query_start: index in source sequence of query_hidden[:, 0].
        """
        if not COPY_HEAD_ENABLED:
            return logits

        B, U, _ = query_hidden.shape
        T = source_hidden.shape[1]
        if query_start is None:
            query_start = T - U

        q = self.copy_q(self.copy_q_norm(query_hidden))
        k = self.copy_k(self.copy_k_norm(source_hidden))
        if COPY_USE_QK_NORM:
            q = F.rms_norm(q, (COPY_DIM,))
            k = F.rms_norm(k, (COPY_DIM,))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(max(COPY_DIM, 1))

        q_pos = torch.arange(query_start, query_start + U, device=query_hidden.device)
        k_pos = torch.arange(T, device=query_hidden.device)
        future_mask = k_pos[None, :] > q_pos[:, None]
        scores = scores.masked_fill(future_mask[None, :, :], float("-inf"))

        attn = torch.softmax(scores.float(), dim=-1).to(query_hidden.dtype)  # (B, U, T)

        copy_probs = torch.zeros(B, U, VOCAB_SIZE, device=logits.device, dtype=query_hidden.dtype)
        copy_index = source_ids[:, None, :].expand(B, U, T)
        copy_probs.scatter_add_(dim=2, index=copy_index, src=attn)

        gate = torch.sigmoid(self.copy_gate(query_hidden)).reshape(B * U, 1).to(logits.dtype)
        scale = F.softplus(self.copy_scale).to(dtype=logits.dtype, device=logits.device)
        copy_bonus = copy_probs.reshape(B * U, VOCAB_SIZE).to(logits.dtype)
        return logits + gate * scale * copy_bonus

    def forward(self, ids, state=None, return_state=False):
        bsz = ids.shape[0]
        x = self.tok_emb(ids)
        if self.use_factored:
            x = self.embed_proj(x)
        if self.bigram is not None:
            x = x + self.bigram(ids)
        x0 = x
        mem = None
        if USE_MEMORY:
            mem = None if state is None else state.get("mem", None)
            if mem is None:
                mem = torch.zeros(bsz, self.memory_dim, device=ids.device, dtype=x.dtype)
            else:
                mem = mem.to(device=ids.device, dtype=x.dtype)

        # Pick layer sequence: looped or default
        use_loop = self.loop_enabled_external and self.loop_enc_seq is not None
        if use_loop:
            enc_indices = self.loop_enc_seq
            dec_indices = self.loop_dec_seq
        else:
            half = N_UNIQUE_BLOCKS // 2
            enc_indices = list(range(0, half))
            dec_indices = list(range(half, N_UNIQUE_BLOCKS))

        n_enc_eff = len(enc_indices)
        n_dec_eff = len(dec_indices)
        n_skip_eff = min(n_enc_eff, n_dec_eff, self.n_skip)

        skips = []
        # Encoder
        for layer_pos, block_idx in enumerate(enc_indices):
            block = self.blocks[block_idx]
            if FFN_FREQ_MODE == "unroll":
                run_ffn = True
            else:
                run_ffn = (layer_pos % max(FFN_EVERY, 1)) == 0
            x, mem = block(x, x0, mem, run_ffn=run_ffn)
            skips.append(x)

        # Decoder with skip connections
        for layer_pos, block_idx in enumerate(dec_indices):
            block = self.blocks[block_idx]
            run_ffn = (layer_pos % max(FFN_EVERY, 1)) == 0
            if layer_pos < n_skip_eff and skips:
                x = x + self.skip_weights[layer_pos][None, None, :] * skips.pop()
            x, mem = block(x, x0, mem, run_ffn=run_ffn)

        out = self.final_norm(x)
        if return_state:
            return out, {"mem": mem} if USE_MEMORY else None
        return out


def cast_params_to_dtype(model, dtype):
    """
    Pre-cast small non-matrix parameters (scales, gates, mix vectors) to compute
    dtype so we don't launch hundreds of aten::copy_ kernels every forward pass.
    Only casts 1D params and small 2D params that are control/scale params.
    Leaves embedding, linear weights, and Mamba internals in their native dtype.
    """
    cast_count = 0
    for name, p in model.named_parameters():
        if "mamba" in name:
            continue
        # Cast 1D params (scales, gates, norms) and the 2D resid_mix (2, d_model)
        is_ctrl = any(c in name for c in CTRL_PATTERNS) or "skip_weights" in name
        is_norm = "final_norm" in name or (p.ndim == 1 and "weight" in name)
        if (p.ndim <= 1 or (p.ndim == 2 and is_ctrl)) and p.dtype != dtype:
            p.data = p.data.to(dtype)
            cast_count += 1
    return cast_count


def apply_s4d_init_to_mamba(model: nn.Module):
    """
    Safely apply S4D-style diagonal initialization to Mamba modules that expose
    A_log and D parameters. This is done pre-DDP and keeps fused kernel paths.
    """
    initialized = 0
    skipped = 0
    for mod_name, module in model.named_modules():
        if not (hasattr(module, "A_log") and hasattr(module, "D")):
            continue

        A_log = getattr(module, "A_log")
        D = getattr(module, "D")
        if not (torch.is_tensor(A_log) and torch.is_tensor(D)):
            skipped += 1
            continue
        if A_log.numel() == 0:
            skipped += 1
            continue

        d_state = A_log.shape[-1]
        if d_state <= 0:
            skipped += 1
            continue

        # S4D-Real: A = - (1/2, 3/2, 5/2, ...), parameterized as log(-A).
        a_init = torch.arange(1, 2 * d_state, 2, device=A_log.device, dtype=torch.float32) / 2.0
        while a_init.ndim < A_log.ndim:
            a_init = a_init.unsqueeze(0)
        a_init = a_init.expand_as(A_log).to(dtype=A_log.dtype)

        try:
            with torch.no_grad():
                A_log.copy_(torch.log(a_init))
                D.fill_(1.0)
            initialized += 1
        except Exception as e:
            skipped += 1
            log0(f"[s4d_init] skip module '{mod_name}': {e}")

    return initialized, skipped


def lm_loss(model_or_ddp, ids, targets, state=None, return_state=False):
    if return_state:
        hidden, next_state = model_or_ddp(ids, state=state, return_state=True)
    else:
        hidden = model_or_ddp(ids, state=state, return_state=False)
        next_state = None
    core = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp
    y = targets.reshape(-1)
    logits = core.output_logits_from_hidden(hidden)
    logits = core.add_copy_logits(logits, hidden, hidden, ids, query_start=0)
    loss = F.cross_entropy(logits.float(), y, reduction="mean")
    if return_state:
        return loss, next_state
    return loss


def detach_state(state):
    if state is None:
        return None
    out = {}
    for k, v in state.items():
        out[k] = v.detach() if torch.is_tensor(v) else v
    return out


def apply_state_reset_mask(state, reset_mask):
    """
    Zero state rows where a new shard/document boundary is crossed.
    """
    if state is None or reset_mask is None:
        return state
    if not torch.any(reset_mask):
        return state
    out = dict(state)
    mem = out.get("mem", None)
    if mem is None:
        return out
    out["mem"] = mem.masked_fill(reset_mask[:, None], 0.0)
    return out


# -----------------------------------------------------------------------------
# Train / eval helpers
# -----------------------------------------------------------------------------
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Orthogonalize a 2D update matrix via a fast Newton-Schulz iteration.
    Muon-style update preconditioner for matrix parameters.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

# Compile for fused CUDA kernels — significant speedup on the NS5 iterations
zeropower_via_newtonschulz5 = torch.compile(_zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for matrix-shaped parameters.
    Uses compiled NS5 for speed. DDP synchronizes gradients during backward.
    """
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay, base_lr=lr),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                upd = g.add(buf, alpha=momentum) if nesterov else buf
                upd_2d = upd.reshape(upd.shape[0], -1) if upd.ndim > 2 else upd
                if upd_2d.ndim >= 2:
                    upd_orth = zeropower_via_newtonschulz5(upd_2d, steps=backend_steps)
                    upd_orth = upd_orth * (max(1, upd_orth.size(0) / upd_orth.size(1)) ** 0.5)
                    upd = upd_orth.reshape(upd.shape)
                p.add_(upd.to(dtype=p.dtype), alpha=-lr)
        return loss


def build_optimizers(model):
    mat_params, scalar_params, embed_params = [], [], []
    muon_eligible_2d = 0
    muon_eligible_nd = 0
    _seen_data_ptrs = set()  # handle tied params
    for name, p in model.named_parameters():
        dp = p.data_ptr()
        if dp in _seen_data_ptrs:
            continue  # skip tied duplicate (lm_head.weight = tok_emb.weight)
        _seen_data_ptrs.add(dp)
        if name in ("tok_emb.weight", "lm_head.weight"):
            embed_params.append(p)
        else:
            is_control = any(c in name for c in CTRL_PATTERNS)
            if p.ndim == 2 and not is_control:
                muon_eligible_2d += 1
            elif p.ndim > 2 and not is_control:
                muon_eligible_nd += 1

            if ((p.ndim == 2) if MUON_ONLY_2D else (p.ndim >= 2)) and not is_control:
                mat_params.append(p)
            else:
                scalar_params.append(p)

    optimizer_muon = Muon(
        mat_params,
        lr=MATRIX_LR,
        momentum=MUON_MOMENTUM,
        backend_steps=MUON_BACKEND_STEPS,
        nesterov=MUON_NESTEROV,
        weight_decay=WEIGHT_DECAY,
    ) if len(mat_params) > 0 else None

    optimizer_adamw = torch.optim.AdamW(
        [
            {"params": scalar_params, "lr": SCALAR_LR, "weight_decay": WEIGHT_DECAY, "base_lr": SCALAR_LR},
            {"params": embed_params, "lr": EMBED_LR, "weight_decay": WEIGHT_DECAY, "base_lr": EMBED_LR},
        ],
        betas=(BETA1, BETA2),
        eps=ADAM_EPS,
        fused=True,
    )
    log0(
        f"Optimizer split: matrix={len(mat_params)} (Muon), "
        f"scalar/control={len(scalar_params)} (AdamW), embed={len(embed_params)} (AdamW)"
    )
    log0(
        f"Muon config: muon_only_2d={int(MUON_ONLY_2D)} "
        f"eligible_2d={muon_eligible_2d} eligible_nd={muon_eligible_nd}"
    )
    return optimizer_adamw, optimizer_muon, mat_params


def lr_schedule(step, elapsed_sec):
    """
    LR schedule with two modes:
    - cosine_full: warmup then cosine decay from peak to LR_MIN_SCALE over full training.
    - cosine_late: warmup, hold at peak, then cosine decay in final phase (original behavior).
    Both use wallclock progress as the time axis.
    """
    if step < WARMUP_STEPS:
        return step / max(WARMUP_STEPS, 1)
    if MAX_WALLCLOCK_SECONDS <= 0:
        return 1.0

    prog = max(0.0, min(1.0, elapsed_sec / MAX_WALLCLOCK_SECONDS))

    if LR_SCHEDULE == "step":
        # Manual step schedule: 1.0 → 0.8 → 0.5 → 0.3
        if prog < 0.35:
            return 1.0
        elif prog < 0.65:
            return 0.8
        elif prog < 0.85:
            return 0.5
        else:
            return 0.3
    elif LR_SCHEDULE == "cosine_full":
        t = prog
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return LR_MIN_SCALE + (1.0 - LR_MIN_SCALE) * cosine
    else:
        # Original: hold at peak, then cosine warmdown
        if prog < LR_WARMDOWN_START_FRAC:
            return 1.0
        denom = max(1.0 - LR_WARMDOWN_START_FRAC, 1e-8)
        t = (prog - LR_WARMDOWN_START_FRAC) / denom
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return LR_MIN_SCALE + (1.0 - LR_MIN_SCALE) * cosine


@torch.no_grad()
def eval_val(model_or_ddp, val_tokens, bb, hs, ib):
    # Unwrap DDP — eval doesn't need gradient sync and DDP forward hooks
    # can cause NCCL hangs if ranks have mismatched batch counts.
    core = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp
    core.eval()
    seq_len = SEQ_LEN
    vbs = 131072
    batch_seqs = max(vbs // seq_len, 1)
    total_seqs = (val_tokens.size - 1) // seq_len
    seq_start = (total_seqs * RANK) // WORLD_SIZE
    seq_end = (total_seqs * (RANK + 1)) // WORLD_SIZE
    loss_sum = 0.0
    tok_sum = 0.0
    byt_sum = 0.0
    for s in range(seq_start, seq_end, batch_seqs):
        e = min(s + batch_seqs, seq_end)
        chunk = val_tokens[s * seq_len : (e * seq_len) + 1]
        xn = chunk[:-1].reshape(-1, seq_len)
        yn = chunk[1:].reshape(-1, seq_len)
        x = torch.from_numpy(xn).long().to(DEVICE, non_blocking=True)
        y = torch.from_numpy(yn).long().to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
            loss = lm_loss(core, x, y)
        cnt = float(y.numel())
        loss_sum += float(loss.detach().float().item()) * cnt
        p, t = xn.reshape(-1), yn.reshape(-1)
        b = bb[t].astype(np.int16, copy=True)
        b += (hs[t] & ~ib[p]).astype(np.int16)
        tok_sum += cnt
        byt_sum += float(b.astype(np.float64).sum())
    stats = torch.tensor([loss_sum, tok_sum, byt_sum], device=DEVICE, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, tok_sum, byt_sum = float(stats[0]), float(stats[1]), float(stats[2])
    val_loss = loss_sum / max(tok_sum, 1.0)
    bpt = val_loss / math.log(2.0)
    core.train()
    return float(val_loss), float(bpt * (tok_sum / max(byt_sum, 1.0)))


@torch.no_grad()
def eval_val_sliding(model_or_ddp, val_tokens, bb, hs, ib, stride=None):
    """
    Sliding window evaluation: each token gets near-full context.
    Process windows of SEQ_LEN tokens with small stride, only score the last
    `stride` tokens of each window. Much more accurate BPB than non-overlapping eval.
    """
    if stride is None:
        stride = EVAL_STRIDE
    # Unwrap DDP — eval doesn't need gradient sync and DDP forward hooks
    # can cause NCCL hangs if ranks have mismatched batch counts.
    core = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp
    core.eval()
    seq_len = SEQ_LEN

    total_tokens = val_tokens.size - 1  # available for input-target pairs
    # Number of windows: each window is seq_len tokens, stride forward by `stride`
    n_windows = max(0, (total_tokens - seq_len) // stride + 1)

    # Distribute windows across ranks
    win_per_rank = n_windows // WORLD_SIZE
    win_start = win_per_rank * RANK
    win_end = win_start + win_per_rank

    # Batch multiple windows together for efficiency
    # Each window is (1, seq_len), we batch up to `batch_windows` at a time
    batch_windows = max(1, 131072 // seq_len)  # ~32 at seq_len=4096

    loss_sum = 0.0
    tok_sum = 0.0
    byt_sum = 0.0
    t0_eval = time.perf_counter()
    total_batches = (win_per_rank + batch_windows - 1) // batch_windows
    log_every_batches = max(1, total_batches // 10)

    batch_idx = 0
    for wb in range(win_start, win_end, batch_windows):
        we = min(wb + batch_windows, win_end)
        bsz = we - wb

        # Build batch of overlapping windows
        x_list = []
        y_list = []
        for w in range(wb, we):
            pos = w * stride
            x_list.append(val_tokens[pos : pos + seq_len])
            y_list.append(val_tokens[pos + 1 : pos + seq_len + 1])

        xn = np.stack(x_list)  # (bsz, seq_len)
        yn = np.stack(y_list)  # (bsz, seq_len)

        x = torch.from_numpy(xn).long().to(DEVICE, non_blocking=True)
        y = torch.from_numpy(yn).long().to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
            hidden = core(x, state=None, return_state=False)

        # Only score the last `stride` positions of each window
        hidden_tail = hidden[:, -stride:, :]  # (bsz, stride, d_model)
        y_tail = y[:, -stride:]  # (bsz, stride)

        flat_y = y_tail.reshape(-1)
        logits = core.output_logits_from_hidden(hidden_tail)
        logits = core.add_copy_logits(logits, hidden_tail, hidden, x, query_start=seq_len - stride)
        loss = F.cross_entropy(logits.float(), flat_y, reduction="sum")

        cnt = float(flat_y.numel())
        loss_sum += float(loss.detach().float().item())

        # BPB: compute bytes only for scored tokens
        p_tail = xn[:, -stride:].reshape(-1)  # input tokens at scored positions
        t_tail = yn[:, -stride:].reshape(-1)  # target tokens at scored positions
        b = bb[t_tail].astype(np.int16, copy=True)
        b += (hs[t_tail] & ~ib[p_tail]).astype(np.int16)
        tok_sum += cnt
        byt_sum += float(b.astype(np.float64).sum())

        batch_idx += 1
        if batch_idx % log_every_batches == 0:
            pct = 100.0 * batch_idx / total_batches
            elapsed_e = time.perf_counter() - t0_eval
            eta = elapsed_e / batch_idx * (total_batches - batch_idx)
            log0(f"  sliding_eval: {pct:.0f}% ({batch_idx}/{total_batches} batches, eta:{eta:.0f}s)")

    elapsed = time.perf_counter() - t0_eval
    log0(f"  sliding_eval: {win_per_rank} windows/rank, stride={stride}, "
         f"{tok_sum:.0f} scored tokens, {elapsed:.1f}s")

    stats = torch.tensor([loss_sum, tok_sum, byt_sum], device=DEVICE, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, tok_sum, byt_sum = float(stats[0]), float(stats[1]), float(stats[2])
    val_loss = loss_sum / max(tok_sum, 1.0)
    bpt = val_loss / math.log(2.0)
    core.train()
    return float(val_loss), float(bpt * (tok_sum / max(byt_sum, 1.0)))




def batch_warmup_seqs(step, full_batch_seqs):
    """
    Return the number of sequences to use from the batch at this step.
    Ramps linearly from BATCH_WARMUP_FRAC * full_batch_seqs to full_batch_seqs
    over BATCH_WARMUP_STEPS steps.
    """
    if BATCH_WARMUP_STEPS <= 0 or step >= BATCH_WARMUP_STEPS:
        return full_batch_seqs
    frac = BATCH_WARMUP_FRAC + (1.0 - BATCH_WARMUP_FRAC) * (step / BATCH_WARMUP_STEPS)
    n = max(1, int(frac * full_batch_seqs))
    return min(n, full_batch_seqs)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    maybe_download_data()
    sp = spm.SentencePieceProcessor(model_file=TOK_PATH)
    assert int(sp.vocab_size()) == VOCAB_SIZE

    val_tokens_full = load_val_tokens(f"{DATA_PATH}/fineweb_val_*.bin", SEQ_LEN)
    if int(os.environ.get("SMOKE_VAL_TOK", "0")) > 0:
        cap = ((int(os.environ["SMOKE_VAL_TOK"]) // SEQ_LEN) * SEQ_LEN) + 1
        val_tokens = val_tokens_full[:cap]
    else:
        val_tokens = val_tokens_full
    bb, hs, ib = build_sp_luts(sp, VOCAB_SIZE)
    log0(f"Val tokens: {val_tokens.size - 1:,} (full: {val_tokens_full.size - 1:,})")

    # -----------------------------------------------------------------------
    # Data loader.
    # -----------------------------------------------------------------------
    def _make_loader(seq_len_value, prefetch_depth=3):
        if USE_ASYNC_LOADER:
            return AsyncDistributedTokenLoader(
                pattern=f"{DATA_PATH}/fineweb_train_*.bin",
                rank=RANK,
                world_size=WORLD_SIZE,
                global_tokens=TRAIN_BATCH_TOK,
                seq_len=seq_len_value,
                grad_accum_steps=GRAD_ACCUM,
                device=DEVICE,
                prefetch_depth=prefetch_depth,
            )
        return DistributedTokenLoader(
            pattern=f"{DATA_PATH}/fineweb_train_*.bin",
            rank=RANK,
            world_size=WORLD_SIZE,
            global_tokens=TRAIN_BATCH_TOK,
            seq_len=seq_len_value,
            grad_accum_steps=GRAD_ACCUM,
            device=DEVICE,
        )

    loader = _make_loader(SEQ_LEN, prefetch_depth=3)

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    base_model = SSM_LM().to(DEVICE)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"Model parameters: {n_params:,}")

    if ENABLE_S4D_INIT:
        n_init, n_skip = apply_s4d_init_to_mamba(base_model)
        log0(f"S4D init applied to {n_init} module(s), skipped {n_skip}")

    # Pre-cast small params to bf16 to eliminate per-step .to(dtype) copies
    n_cast = cast_params_to_dtype(base_model, AMP_DTYPE)
    log0(f"Pre-cast {n_cast} parameters to {AMP_DTYPE}")

    model_for_ddp = base_model
    if USE_TORCH_COMPILE:
        try:
            # fullgraph=False required: mamba_ssm custom CUDA ops (causal_conv1d)
            # use non-contiguous out= tensors which break dynamo's fullgraph mode.
            model_for_ddp = torch.compile(base_model, dynamic=False)
            log0("torch_compile: enabled")
        except Exception as e:
            log0(f"torch_compile: failed ({e}); falling back to eager")

    model = DDP(model_for_ddp, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
                broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True)

    optimizer_adamw, optimizer_muon, matrix_params = build_optimizers(base_model)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))

    losses = []
    t0 = time.perf_counter()
    step = 0
    stop_after = None
    last_log_t = t0
    last_log_step = 0
    train_tok_processed = 0.0
    last_log_tok_processed = 0.0
    # Pre-compute constants outside the loop
    _ga_inv = 1.0 / float(GRAD_ACCUM)
    _sc_inv = 1.0 / float(STREAM_CHUNKS)
    _chunk_len = SEQ_LEN // STREAM_CHUNKS  # final/eval chunk length; train chunk len can change with curriculum
    # Accumulator on GPU — avoids .item() CUDA sync every micro-batch
    _loss_accum = torch.zeros((), device=DEVICE, dtype=torch.float32)
    # Pre-allocated stop signal — reused every check to avoid tensor allocation
    _stop_flag = torch.zeros(1, device=DEVICE, dtype=torch.float32)
    _STOP_CHECK_EVERY = int(os.environ.get("STOP_CHECK_EVERY", "100"))  # fewer syncs; may overshoot slightly
    _REPLAY_FRAC = float(os.environ.get("REPLAY_FRAC", "0.0"))  # race default: disabled
    _replay_triggered = False
    log0(f"Training for up to {ITERATIONS} steps (cap={MAX_WALLCLOCK_SECONDS}s)")
    log0(f"Replay: will reset data stream at {_REPLAY_FRAC*100:.0f}% wall time")

    while step < ITERATIONS:
        # Wallclock stop: broadcast from rank 0 every N steps so all ranks
        # exit on the exact same step. DDP requires all ranks to participate
        # in every forward/backward, so even ±1 step desync causes a hang.
        if step % _STOP_CHECK_EVERY == 0:
            elapsed = time.perf_counter() - t0
            _stop_flag.fill_(
                1.0 if (MAX_WALLCLOCK_SECONDS > 0 and elapsed >= MAX_WALLCLOCK_SECONDS) else 0.0
            )
            dist.broadcast(_stop_flag, src=0)
            if _stop_flag.item() > 0.5:
                stop_after = step
                break

        if step % _STOP_CHECK_EVERY == 0:
            elapsed = time.perf_counter() - t0
        # Replay trigger: reset data stream to replay previously seen tokens
        if not _replay_triggered and _REPLAY_FRAC > 0 and MAX_WALLCLOCK_SECONDS > 0:
            frac = elapsed / MAX_WALLCLOCK_SECONDS
            if frac >= _REPLAY_FRAC:
                loader.reset_stream()
                _replay_triggered = True
                log0(f"REPLAY: data stream reset at step {step} ({frac*100:.0f}% wall time)")
        # Depth recurrence activation: enable looping after warmup phase
        if (LOOP_START >= 0 and not base_model.loop_enabled_external
                and MAX_WALLCLOCK_SECONDS > 0):
            frac = elapsed / MAX_WALLCLOCK_SECONDS
            if frac >= LOOP_ACTIVATE_FRAC:
                base_model.loop_enabled_external = True
                log0(f"LOOP: depth recurrence activated at step {step} "
                     f"({frac*100:.0f}% wall) enc={base_model.loop_enc_seq} "
                     f"dec={base_model.loop_dec_seq}")
        mul = lr_schedule(step, elapsed)
        for group in optimizer_adamw.param_groups:
            group["lr"] = group["base_lr"] * mul
        if optimizer_muon is not None:
            for group in optimizer_muon.param_groups:
                group["lr"] = group["base_lr"] * mul
            # Muon momentum warmup: ramp from MUON_MOMENTUM_WARMUP_START to MUON_MOMENTUM
            if MUON_MOMENTUM_WARMUP_STEPS > 0:
                frac = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1.0)
                cur_momentum = (1 - frac) * MUON_MOMENTUM_WARMUP_START + frac * MUON_MOMENTUM
                for group in optimizer_muon.param_groups:
                    group["momentum"] = cur_momentum

        optimizer_adamw.zero_grad(set_to_none=True)
        if optimizer_muon is not None:
            optimizer_muon.zero_grad(set_to_none=True)

        # --- Gradient accumulation inner loop ---
        _loss_accum.zero_()
        accum_tok = 0.0

        for ga_step in range(GRAD_ACCUM):
            x, y, reset_chunks = loader.next_batch()

            # Batch warmup: use fewer sequences early for faster updates
            full_seqs = x.shape[0]
            use_seqs = batch_warmup_seqs(step, full_seqs)
            if use_seqs < full_seqs:
                x = x[:use_seqs]
                y = y[:use_seqs]
                if reset_chunks is not None:
                    reset_chunks = reset_chunks[:use_seqs]

            current_seq_len = x.shape[1]
            _cur_chunk_len = current_seq_len // STREAM_CHUNKS
            active_stream_chunks = STREAM_CHUNKS
            # Disable DDP gradient sync on all but the last micro-batch
            no_sync = (ga_step < GRAD_ACCUM - 1) and hasattr(model, 'no_sync')
            ctx = model.no_sync() if no_sync else nullcontext()
            with ctx:
                if active_stream_chunks == 1:
                    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
                        xs = x[:, :_cur_chunk_len]
                        ys = y[:, :_cur_chunk_len]
                        micro_loss = lm_loss(model, xs, ys)
                else:
                    state = None
                    micro_loss = torch.zeros((), device=DEVICE, dtype=torch.float32)
                    for c in range(active_stream_chunks):
                        xs = x[:, c * _cur_chunk_len : (c + 1) * _cur_chunk_len]
                        ys = y[:, c * _cur_chunk_len : (c + 1) * _cur_chunk_len]
                        if c > 0 and reset_chunks is not None:
                            state = apply_state_reset_mask(state, reset_chunks[:, c])
                        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
                            loss_c, state = lm_loss(model, xs, ys, state=state, return_state=True)
                        micro_loss = micro_loss + (loss_c / float(active_stream_chunks))
                        if ((c + 1) % BPTT_CHUNKS) == 0:
                            state = detach_state(state)

                # Scale loss by 1/GRAD_ACCUM so gradients average correctly
                scaled_loss = micro_loss * _ga_inv
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            # Accumulate on GPU — no .item() CUDA sync
            _loss_accum += micro_loss.detach()
            micro_tok_ratio = (use_seqs / full_seqs) * (active_stream_chunks * _sc_inv)
            accum_tok += TRAIN_BATCH_TOK * micro_tok_ratio * _ga_inv

        # --- End accumulation loop, now step ---
        train_tok_processed += accum_tok

        if scaler.is_enabled():
            scaler.unscale_(optimizer_adamw)
            inv_scale = 1.0 / float(scaler.get_scale())
            for p in matrix_params:
                if p.grad is not None:
                    p.grad.mul_(inv_scale)

        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(base_model.parameters(), GRAD_CLIP)

        if scaler.is_enabled():
            scaler.step(optimizer_adamw)
            if optimizer_muon is not None:
                optimizer_muon.step()
            scaler.update()
        else:
            optimizer_adamw.step()
            if optimizer_muon is not None:
                optimizer_muon.step()

        step += 1

        # Logging — only .item() on log steps (forces one CUDA sync per log)
        if step <= 5 or step % LOG_EVERY == 0:
            lv = float((_loss_accum * _ga_inv).item())
            now = time.perf_counter()
            dt = max(now - last_log_t, 1e-9)
            dtok = max(train_tok_processed - last_log_tok_processed, 0.0)
            tok_s_inst = dtok / dt
            tok_s_avg = train_tok_processed / max((now - t0), 1e-9)
            log0(
                f"step:{step}/{ITERATIONS} loss:{lv:.4f} lr:{mul:.3f} "
                f"tok/s_inst:{tok_s_inst:.0f} tok/s_avg:{tok_s_avg:.0f}"
            )
            last_log_t = now
            last_log_step = step
            last_log_tok_processed = train_tok_processed
            losses.append(lv)
        else:
            losses.append(None)  # placeholder — no sync on non-log steps

        if VAL_EVERY > 0 and step % VAL_EVERY == 0:
            vl, vb = eval_val(model, val_tokens, bb, hs, ib)
            log0(f"  -> val_loss:{vl:.4f} val_bpb:{vb:.4f}")

    train_elapsed = time.perf_counter() - t0
    log0(f"Training complete: {step} steps in {train_elapsed:.1f}s")
    if stop_after is not None:
        log0(f"stopping_early: wallclock_cap at step {stop_after}/{ITERATIONS}")

    # Sync all ranks before eval to prevent NCCL desync
    dist.barrier()
    log0("All ranks synced. Starting evaluation...")

    # Eval with live weights (standard, fast)
    vl, vb = eval_val(model, val_tokens, bb, hs, ib)
    log0(f"Final val_loss:{vl:.4f} val_bpb:{vb:.4f}")

    # Sliding window eval
    if EVAL_STRIDE < SEQ_LEN:
        log0(f"Running sliding window eval (stride={EVAL_STRIDE})...")
        sw_vl, sw_vb = eval_val_sliding(model, val_tokens, bb, hs, ib, stride=EVAL_STRIDE)
        log0(f"Sliding val_loss:{sw_vl:.4f} val_bpb:{sw_vb:.4f}")
        log0(f"Sliding vs standard: bpb={vb - sw_vb:+.4f}")

    # -----------------------------------------------------------------------
    # Int8 quantization + compression + roundtrip validation
    # -----------------------------------------------------------------------
    log0("Quantizing model to int8...")
    state_dict = base_model.state_dict()
    quantized = {}
    scales = {}
    passthrough = {}
    for name, t in state_dict.items():
        t = t.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            # Small tensors / non-float: keep as fp16
            if t.is_floating_point() and t.dtype in (torch.float32, torch.bfloat16):
                passthrough[name] = t.to(torch.float16).contiguous()
            else:
                passthrough[name] = t
            continue
        # Per-row int8 quantization for 2D, per-tensor for others
        t32 = t.float()
        if t32.ndim == 2:
            clip_abs = torch.quantile(t32.abs(), 0.9999984, dim=1)
            scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
            scales[name] = scale.to(torch.float16).contiguous()
        else:
            clip_abs = float(torch.quantile(t32.abs().flatten(), 0.9999984).item()) if t32.numel() else 0.0
            scale = clip_abs / 127.0 if clip_abs > 0 else 1.0
            q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8)
            scales[name] = torch.tensor(scale, dtype=torch.float32)
        quantized[name] = q.contiguous()

    quant_obj = {"quantized": quantized, "scales": scales, "passthrough": passthrough}
    import io
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()

    if _COMPRESSOR == "zstd":
        cctx = zstandard.ZstdCompressor(level=22)
        quant_blob = cctx.compress(quant_raw)
    else:
        quant_blob = zlib.compress(quant_raw, level=9)

    compressed_bytes = len(quant_blob)
    log0(f"Compressed model ({_COMPRESSOR}): {compressed_bytes:,} bytes ({compressed_bytes / 1024 / 1024:.2f} MB)")

    # Roundtrip: decompress, dequantize, reload, and re-evaluate
    if _COMPRESSOR == "zstd":
        dctx = zstandard.ZstdDecompressor()
        quant_raw_rt = dctx.decompress(quant_blob)
    else:
        quant_raw_rt = zlib.decompress(quant_blob)
    quant_obj_rt = torch.load(io.BytesIO(quant_raw_rt), map_location="cpu")

    # Dequantize
    dequant_state = {}
    for name, q in quant_obj_rt["quantized"].items():
        s = quant_obj_rt["scales"][name]
        if s.ndim > 0:
            dequant_state[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(torch.bfloat16)
        else:
            dequant_state[name] = (q.float() * float(s.item())).to(torch.bfloat16)
    for name, t in quant_obj_rt["passthrough"].items():
        dequant_state[name] = t

    base_model.load_state_dict(dequant_state, strict=True)
    q_vl, q_vb = eval_val(model, val_tokens, bb, hs, ib)
    log0(f"Post-quant val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f}")
    if EVAL_STRIDE < SEQ_LEN:
        q_sw_vl, q_sw_vb = eval_val_sliding(model, val_tokens, bb, hs, ib, stride=EVAL_STRIDE)
        log0(f"Post-quant sliding val_loss:{q_sw_vl:.4f} val_bpb:{q_sw_vb:.4f}")

    # Save compressed artifact
    if MASTER_PROCESS:
        artifact_path = "final_model.int8.ptz"
        with open(artifact_path, "wb") as f:
            f.write(quant_blob)
        log0(f"Saved artifact: {artifact_path} ({compressed_bytes:,} bytes)")

    # Shutdown async loader
    try:
        loader.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()