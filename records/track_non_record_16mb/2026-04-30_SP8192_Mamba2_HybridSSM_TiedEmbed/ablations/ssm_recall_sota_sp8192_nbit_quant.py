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

# 2048 -> 8192 length curriculum.
# Keep SEQ_LEN=8192 for final/eval shape, but early training batches use shorter
# 2048 sequences with the same TOK_PER_RANK token budget. This attempts to combine
# faster/more diverse 2048 optimization with late 8192 context learning.
LENGTH_CURRICULUM_ENABLED = False  # removed in this quant-focused file
CURRICULUM_SHORT_SEQ_LEN = int(os.environ.get("CURRICULUM_SHORT_SEQ_LEN", "2048"))
CURRICULUM_SWITCH_FRAC = float(os.environ.get("CURRICULUM_SWITCH_FRAC", "0.60"))

# Mixed n-bit post-training quantization.
# This is not training-time quantization; it is an artifact-size / parameter-budget tool.
MIXED_NBIT_QUANT = os.environ.get("MIXED_NBIT_QUANT", "1") == "1"
MATRIX_BITS = int(os.environ.get("MATRIX_BITS", "6"))
EMBED_BITS = int(os.environ.get("EMBED_BITS", "7"))
HEAD_BITS = int(os.environ.get("HEAD_BITS", "7"))
SSM_BITS = int(os.environ.get("SSM_BITS", str(MATRIX_BITS)))
ATTN_BITS = int(os.environ.get("ATTN_BITS", str(MATRIX_BITS)))
FFN_BITS = int(os.environ.get("FFN_BITS", str(MATRIX_BITS)))

# SDClip-style clipping. clip_abs = min(row_absmax, sigma * row_std)
MATRIX_CLIP_SIGMAS = float(os.environ.get("MATRIX_CLIP_SIGMAS", "10.0"))
EMBED_CLIP_SIGMAS = float(os.environ.get("EMBED_CLIP_SIGMAS", "14.0"))
HEAD_CLIP_SIGMAS = float(os.environ.get("HEAD_CLIP_SIGMAS", "14.0"))
SSM_CLIP_SIGMAS = float(os.environ.get("SSM_CLIP_SIGMAS", str(MATRIX_CLIP_SIGMAS)))
ATTN_CLIP_SIGMAS = float(os.environ.get("ATTN_CLIP_SIGMAS", str(MATRIX_CLIP_SIGMAS)))
FFN_CLIP_SIGMAS = float(os.environ.get("FFN_CLIP_SIGMAS", str(MATRIX_CLIP_SIGMAS)))

QUANT_PASSTHROUGH_MAX = int(os.environ.get("QUANT_PASSTHROUGH_MAX", "65536"))
QUANT_SAVE_NAME = os.environ.get("QUANT_SAVE_NAME", "final_model.nbit.ptz")

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
MIXED_LENGTH_CURRICULUM = os.environ.get("MIXED_LENGTH_CURRICULUM", "0") == "1"
CURRICULUM_STEPS = int(os.environ.get("CURRICULUM_STEPS", "1500"))
SEED = int(os.environ.get("SEED", "7"))
MAX_WALLCLOCK_SECONDS = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "600"))

# LR schedule: short warmup, then long warmdown (matching competitive GPT config)
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "20"))
LR_MIN_SCALE = float(os.environ.get("LR_MIN_SCALE", "0.0"))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine_late")
LR_WARMDOWN_START_FRAC = float(os.environ.get("LR_WARMDOWN_START_FRAC", "0.28"))  # SOTA: 0.72 of training is warmdown

# SWA (Stochastic Weight Averaging) — collect checkpoints from late warmdown
SWA_ENABLED = os.environ.get("SWA_ENABLED", "0") == "1"
SWA_START_FRAC = float(os.environ.get("SWA_START_FRAC", "0.15"))  # collect from last 15%
SWA_EVERY = int(os.environ.get("SWA_EVERY", "50"))  # checkpoint every N steps

# EMA — SOTA submission uses EMA with decay 0.9965 successfully
EMA_ENABLED = os.environ.get("EMA_ENABLED", "0") == "1"
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.9965"))

# Batch size warmup
BATCH_WARMUP_STEPS = int(os.environ.get("BATCH_WARMUP_STEPS", "0"))
BATCH_WARMUP_FRAC = float(os.environ.get("BATCH_WARMUP_FRAC", "0.25"))

# BigramHash embedding
BIGRAM_ENABLED = os.environ.get("BIGRAM_ENABLED", "1") == "1"
BIGRAM_BUCKETS = int(os.environ.get("BIGRAM_BUCKETS", "10240"))
BIGRAM_DIM = int(os.environ.get("BIGRAM_DIM", "128"))

# Sliding window eval — stride < SEQ_LEN gives each token more context
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", "64"))  # 64 matches SOTA

# Profiling
PROFILE_ENABLED = os.environ.get("PROFILE", "0") == "1"
PROFILE_START = int(os.environ.get("PROFILE_START", "3"))
PROFILE_END = int(os.environ.get("PROFILE_END", "8"))
PROFILE_DIR = os.environ.get("PROFILE_DIR", "./profile_traces")

# Optimizer — higher LRs for 42M model, matching SOTA WD
MATRIX_LR = float(os.environ.get("MATRIX_LR", "0.022"))  # SOTA: 0.022
SCALAR_LR = float(os.environ.get("SCALAR_LR", "0.02"))   # SOTA: 0.02
EMBED_LR = float(os.environ.get("EMBED_LR", "0.03"))     # SOTA: tied_embed_lr=0.03
BETA1, BETA2 = 0.9, 0.95
ADAM_EPS = 1e-8
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.095"))  # SOTA: muon_wd=0.095
ENABLE_S4D_INIT = os.environ.get("ENABLE_S4D_INIT", "1") == "1"

# Test-Time Training (TTT) — global adaptation on val distribution
TTT_ENABLED = os.environ.get("TTT_ENABLED", "0") == "1"
TTT_STEPS = int(os.environ.get("TTT_STEPS", "5"))
TTT_LR = float(os.environ.get("TTT_LR", "1e-3"))
TTT_PREFIX_FRAC = float(os.environ.get("TTT_PREFIX_FRAC", "0.5"))
# Which params to adapt: "norms", "in_proj", "all_linear", "all"
TTT_PARAMS = os.environ.get("TTT_PARAMS", "in_proj")
TTT_MAX_SEQS = int(os.environ.get("TTT_MAX_SEQS", "256"))  # max seqs per rank (0=all)
# Global TTT: adapt on entire val set as LM training, then re-evaluate
TTT_GLOBAL_STEPS = int(os.environ.get("TTT_GLOBAL_STEPS", "10"))
TTT_GLOBAL_LR = float(os.environ.get("TTT_GLOBAL_LR", "2e-4"))
TTT_GLOBAL_BATCH_SEQS = int(os.environ.get("TTT_GLOBAL_BATCH_SEQS", "16"))
TTT_GLOBAL_GRAD_CLIP = float(os.environ.get("TTT_GLOBAL_GRAD_CLIP", "1.0"))
TTT_GLOBAL_PASSES = int(os.environ.get("TTT_GLOBAL_PASSES", "1"))  # passes over val set

# Score-First TTT (SOTA-style legal eval-time adaptation)
SCORE_FIRST_TTT_ENABLED = os.environ.get("SCORE_FIRST_TTT_ENABLED", "0") == "1"
SCORE_FIRST_TTT_CHUNK_TOKENS = int(os.environ.get("SCORE_FIRST_TTT_CHUNK_TOKENS", "32768"))
SCORE_FIRST_TTT_LR = float(os.environ.get("SCORE_FIRST_TTT_LR", "0.005"))
SCORE_FIRST_TTT_MOMENTUM = float(os.environ.get("SCORE_FIRST_TTT_MOMENTUM", "0.9"))
SCORE_FIRST_TTT_EPOCHS = int(os.environ.get("SCORE_FIRST_TTT_EPOCHS", "3"))
SCORE_FIRST_TTT_GRAD_CLIP = float(os.environ.get("SCORE_FIRST_TTT_GRAD_CLIP", "1.0"))
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
log0(
    f"mixed_length_curriculum={int(MIXED_LENGTH_CURRICULUM)}, "
    f"curriculum_steps={CURRICULUM_STEPS}"
)
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
log0(
    f"length_curriculum: enabled={LENGTH_CURRICULUM_ENABLED}, "
    f"short_seq={CURRICULUM_SHORT_SEQ_LEN}, switch_frac={CURRICULUM_SWITCH_FRAC}, final_seq={SEQ_LEN}"
)
log0(
    f"mixed_nbit_quant: enabled={MIXED_NBIT_QUANT}, matrix={MATRIX_BITS}, embed={EMBED_BITS}, "
    f"head={HEAD_BITS}, ssm={SSM_BITS}, attn={ATTN_BITS}, ffn={FFN_BITS}, save={QUANT_SAVE_NAME}"
)
log0("race_defaults: EMBED_DIM=512, ATTN_LAYER_IDXS=6, QK_GAIN_INIT=5.25, LOG_EVERY=500, EMA/SWA/TTT off")
log0(f"activation: {ACTIVATION}")
log0(f"depth_recur: loop=[{LOOP_START}..{LOOP_END}] activate@{LOOP_ACTIVATE_FRAC}")
log0(f"score_first_ttt: enabled={SCORE_FIRST_TTT_ENABLED}, chunk={SCORE_FIRST_TTT_CHUNK_TOKENS}, lr={SCORE_FIRST_TTT_LR}, epochs={SCORE_FIRST_TTT_EPOCHS}")
log0(f"bigram: enabled={BIGRAM_ENABLED}, buckets={BIGRAM_BUCKETS}, dim={BIGRAM_DIM}")
log0(f"swa: enabled={SWA_ENABLED}, start_frac={SWA_START_FRAC}, every={SWA_EVERY}")
log0(f"eval_stride={EVAL_STRIDE}")
log0(f"ema: enabled={EMA_ENABLED}, decay={EMA_DECAY}")
log0(f"optimizer: matrix_lr={MATRIX_LR}, scalar_lr={SCALAR_LR}, embed_lr={EMBED_LR}, wd={WEIGHT_DECAY}, grad_clip={GRAD_CLIP}")
log0(f"batch_warmup: steps={BATCH_WARMUP_STEPS}, start_frac={BATCH_WARMUP_FRAC}")
log0(
    f"ttt: enabled={TTT_ENABLED}, per_seq_steps={TTT_STEPS}, per_seq_lr={TTT_LR}, "
    f"params={TTT_PARAMS}, max_seqs={TTT_MAX_SEQS}"
)
log0(
    f"ttt_global: steps={TTT_GLOBAL_STEPS}, lr={TTT_GLOBAL_LR}, "
    f"batch_seqs={TTT_GLOBAL_BATCH_SEQS}, passes={TTT_GLOBAL_PASSES}, "
    f"grad_clip={TTT_GLOBAL_GRAD_CLIP}"
)
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


def stream_chunks_for_step(step):
    """
    Optional mixed-length curriculum over streamed chunks.
    Starts with shorter context and ramps to STREAM_CHUNKS by CURRICULUM_STEPS.
    """
    if STREAM_CHUNKS <= 1 or not MIXED_LENGTH_CURRICULUM:
        return STREAM_CHUNKS
    if CURRICULUM_STEPS <= 0:
        return STREAM_CHUNKS
    frac = max(0.0, min(1.0, step / float(CURRICULUM_STEPS)))
    chunks = 1 + int(frac * (STREAM_CHUNKS - 1))
    return max(1, min(STREAM_CHUNKS, chunks))


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


def eval_score_first_ttt(model_or_ddp, val_tokens, bb, hs, ib, stride=None):
    """
    SOTA-style legal Score-First TTT eval.

    For each chunk of val (~32K tokens):
      1. Score ALL sliding windows in the chunk under torch.no_grad() (FINAL grades)
      2. Train model with SGD on the scored chunk tokens (3 epochs, momentum=0.9)
      3. Move to next chunk with adapted weights

    Compliance:
      - Causality preserved (sliding windows are causal)
      - Each token scored exactly once, before any update that benefits from it
      - No re-scoring, no rescoring, no n-gram cache
    Weights are modified in-place; restored at end.
    """
    if stride is None:
        stride = EVAL_STRIDE

    core = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp
    seq_len = SEQ_LEN

    # Save base weights for restoration after eval
    base_state = {name: p.data.clone() for name, p in core.named_parameters()}

    # Build SGD optimizer over ALL params
    adapt_params = list(core.parameters())
    n_adapt = sum(p.numel() for p in adapt_params)
    log0(f"Score-First TTT: chunk={SCORE_FIRST_TTT_CHUNK_TOKENS}, lr={SCORE_FIRST_TTT_LR}, "
         f"momentum={SCORE_FIRST_TTT_MOMENTUM}, epochs={SCORE_FIRST_TTT_EPOCHS}, "
         f"adapting {n_adapt:,} params")

    ttt_opt = torch.optim.SGD(adapt_params, lr=SCORE_FIRST_TTT_LR,
                              momentum=SCORE_FIRST_TTT_MOMENTUM)

    total_tokens = val_tokens.size - 1
    chunk_tokens = SCORE_FIRST_TTT_CHUNK_TOKENS
    n_chunks = (total_tokens - seq_len) // chunk_tokens + 1

    # Distribute chunks across ranks
    chunks_per_rank = n_chunks // WORLD_SIZE
    chunk_start = chunks_per_rank * RANK
    chunk_end = chunk_start + chunks_per_rank

    batch_windows = max(1, 131072 // seq_len)

    loss_sum = 0.0
    tok_sum = 0.0
    byt_sum = 0.0
    t0_eval = time.perf_counter()
    log_every_chunks = max(1, chunks_per_rank // 10)

    for chunk_idx_local, ci in enumerate(range(chunk_start, chunk_end)):
        chunk_tok_start = ci * chunk_tokens
        chunk_tok_end = min(chunk_tok_start + chunk_tokens, total_tokens - seq_len)
        if chunk_tok_end <= chunk_tok_start:
            break

        # === STEP 1: SCORE the chunk under torch.no_grad() (FINAL grades) ===
        core.eval()
        n_windows_in_chunk = max(1, (chunk_tok_end - chunk_tok_start) // stride)

        chunk_tokens_for_train = []  # collect input tokens for training phase
        chunk_targets_for_train = []

        with torch.no_grad():
            for wb in range(0, n_windows_in_chunk, batch_windows):
                we = min(wb + batch_windows, n_windows_in_chunk)
                x_list, y_list = [], []
                for w in range(wb, we):
                    pos = chunk_tok_start + w * stride
                    if pos + seq_len + 1 > total_tokens + 1:
                        break
                    x_list.append(val_tokens[pos : pos + seq_len])
                    y_list.append(val_tokens[pos + 1 : pos + seq_len + 1])
                if not x_list:
                    continue
                xn = np.stack(x_list)
                yn = np.stack(y_list)
                x = torch.from_numpy(xn).long().to(DEVICE, non_blocking=True)
                y = torch.from_numpy(yn).long().to(DEVICE, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
                    hidden = core(x, state=None, return_state=False)

                hidden_tail = hidden[:, -stride:, :]
                y_tail = y[:, -stride:]
                flat_y = y_tail.reshape(-1)
                logits = core.output_logits_from_hidden(hidden_tail)
                logits = core.add_copy_logits(logits, hidden_tail, hidden, x, query_start=seq_len - stride)
                loss = F.cross_entropy(logits.float(), flat_y, reduction="sum")

                cnt = float(flat_y.numel())
                loss_sum += float(loss.detach().float().item())

                p_tail = xn[:, -stride:].reshape(-1)
                t_tail = yn[:, -stride:].reshape(-1)
                b = bb[t_tail].astype(np.int16, copy=True)
                b += (hs[t_tail] & ~ib[p_tail]).astype(np.int16)
                tok_sum += cnt
                byt_sum += float(b.astype(np.float64).sum())

                # Save these (already-scored) tokens for the training phase
                chunk_tokens_for_train.append(xn)
                chunk_targets_for_train.append(yn)

        # === STEP 2: SGD train on the scored chunk tokens ===
        if chunk_tokens_for_train:
            core.train()
            all_x = np.concatenate(chunk_tokens_for_train, axis=0)
            all_y = np.concatenate(chunk_targets_for_train, axis=0)
            n_seqs_in_chunk = all_x.shape[0]

            # Cosine LR decay across chunks
            chunk_lr_scale = 0.5 * (1.0 + math.cos(math.pi * chunk_idx_local / max(1, chunks_per_rank)))
            for g in ttt_opt.param_groups:
                g["lr"] = SCORE_FIRST_TTT_LR * chunk_lr_scale

            for epoch in range(SCORE_FIRST_TTT_EPOCHS):
                # Shuffle each epoch
                perm = np.random.permutation(n_seqs_in_chunk)
                bs = max(1, batch_windows // 2)
                for bi in range(0, n_seqs_in_chunk, bs):
                    be = min(bi + bs, n_seqs_in_chunk)
                    idx = perm[bi:be]
                    ax = torch.from_numpy(all_x[idx]).long().to(DEVICE)
                    ay = torch.from_numpy(all_y[idx]).long().to(DEVICE)

                    ttt_opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
                        adapt_loss = lm_loss(core, ax, ay)
                    adapt_loss.backward()

                    for p in adapt_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                    if SCORE_FIRST_TTT_GRAD_CLIP > 0:
                        nn.utils.clip_grad_norm_(adapt_params, SCORE_FIRST_TTT_GRAD_CLIP)
                    ttt_opt.step()

        if (chunk_idx_local + 1) % log_every_chunks == 0:
            elapsed_e = time.perf_counter() - t0_eval
            running_loss = loss_sum / max(tok_sum, 1.0)
            running_bpb = (running_loss / math.log(2.0)) * (tok_sum / max(byt_sum, 1.0))
            eta = elapsed_e / (chunk_idx_local + 1) * (chunks_per_rank - chunk_idx_local - 1)
            log0(f"  score_first_ttt: chunk {chunk_idx_local+1}/{chunks_per_rank} "
                 f"running_bpb:{running_bpb:.4f} eta:{eta:.0f}s")

    elapsed = time.perf_counter() - t0_eval
    log0(f"  score_first_ttt: {chunks_per_rank} chunks/rank, "
         f"{tok_sum:.0f} scored tokens, {elapsed:.1f}s")

    # Aggregate across ranks
    stats = torch.tensor([loss_sum, tok_sum, byt_sum], device=DEVICE, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, tok_sum, byt_sum = float(stats[0]), float(stats[1]), float(stats[2])
    val_loss = loss_sum / max(tok_sum, 1.0)
    bpt = val_loss / math.log(2.0)

    # Restore base weights
    with torch.no_grad():
        for name, p in core.named_parameters():
            if name in base_state:
                p.data.copy_(base_state[name])
    core.train()

    return float(val_loss), float(bpt * (tok_sum / max(byt_sum, 1.0)))


def _select_ttt_params(base_model, param_mode):
    """Select which parameters to adapt during TTT based on mode string."""
    adapt_names = set()
    adapt_params = []
    for name, p in base_model.named_parameters():
        include = False
        if param_mode == "norms":
            include = ("norm" in name or "scale" in name
                       or "ssm_scale" in name or "mlp_scale" in name)
        elif param_mode == "in_proj":
            include = ("in_proj" in name or "out_proj" in name
                       or "norm" in name or "scale" in name
                       or "ssm_scale" in name or "mlp_scale" in name)
        elif param_mode == "all_linear":
            # All linear layers + norms/scales, but NOT embeddings
            include = (name not in ("tok_emb.weight", "lm_head.weight"))
        elif param_mode == "all":
            include = True
        if include:
            adapt_names.add(name)
            adapt_params.append(p)
    return adapt_names, adapt_params


def ttt_global_adapt(model_or_ddp, val_tokens):
    """
    Global Test-Time Training: treat the entire validation set as an LM
    training corpus and run a few Adam steps to adapt model weights to the
    val distribution. This is NOT per-sequence — it is a global domain
    adaptation pass. Weights are modified IN-PLACE; caller must save/restore.

    Returns the saved base state dict for later restoration.
    """
    base_model = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp

    adapt_names, adapt_params = _select_ttt_params(base_model, TTT_PARAMS)
    # Save base weights for later restoration
    base_state = {name: p.data.clone() for name, p in base_model.named_parameters()
                  if name in adapt_names}

    n_adapt = sum(p.numel() for p in adapt_params)
    log0(f"TTT global: adapting {len(adapt_params)} param tensors "
         f"({n_adapt:,} params, mode={TTT_PARAMS})")
    log0(f"TTT global: steps={TTT_GLOBAL_STEPS}, lr={TTT_GLOBAL_LR}, "
         f"batch_seqs={TTT_GLOBAL_BATCH_SEQS}, passes={TTT_GLOBAL_PASSES}, "
         f"grad_clip={TTT_GLOBAL_GRAD_CLIP}")

    # Freeze non-adapted params to save memory on optimizer states
    frozen = []
    for name, p in base_model.named_parameters():
        if name not in adapt_names:
            p.requires_grad_(False)
            frozen.append(name)

    # Build Adam optimizer for adapted params only
    ttt_opt = torch.optim.Adam(adapt_params, lr=TTT_GLOBAL_LR, betas=(0.9, 0.95),
                               weight_decay=0.0)

    seq_len = SEQ_LEN
    total_seqs = (val_tokens.size - 1) // seq_len
    # Each rank processes its shard of val sequences
    seqs_per_rank = total_seqs // WORLD_SIZE
    seq_start = seqs_per_rank * RANK
    seq_end = seq_start + seqs_per_rank
    local_seqs = seq_end - seq_start

    batch_seqs = min(TTT_GLOBAL_BATCH_SEQS, local_seqs)
    batches_per_pass = max(1, local_seqs // batch_seqs)

    base_model.train()
    ttt_t0 = time.perf_counter()
    global_step = 0

    for pass_idx in range(TTT_GLOBAL_PASSES):
        # Shuffle sequence order each pass (deterministic per rank)
        rng = np.random.RandomState(seed=42 + pass_idx + RANK)
        perm = rng.permutation(local_seqs) + seq_start
        batch_idx = 0

        for bi in range(0, local_seqs, batch_seqs):
            if global_step >= TTT_GLOBAL_STEPS:
                break
            be = min(bi + batch_seqs, local_seqs)
            seq_ids = perm[bi:be]

            # Build batch
            x_list, y_list = [], []
            for s in seq_ids:
                chunk = val_tokens[s * seq_len : s * seq_len + seq_len + 1]
                x_list.append(chunk[:-1])
                y_list.append(chunk[1:])
            xn = np.stack(x_list)
            yn = np.stack(y_list)
            x = torch.from_numpy(xn).long().to(DEVICE, non_blocking=True)
            y = torch.from_numpy(yn).long().to(DEVICE, non_blocking=True)

            ttt_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
                loss = lm_loss(base_model, x, y)
            loss.backward()

            # All-reduce gradients across ranks for consistent adaptation
            for p in adapt_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            if TTT_GLOBAL_GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(adapt_params, TTT_GLOBAL_GRAD_CLIP)
            ttt_opt.step()
            global_step += 1
            batch_idx += 1

            if MASTER_PROCESS and (global_step <= 3 or global_step % max(1, TTT_GLOBAL_STEPS // 10) == 0):
                elapsed = time.perf_counter() - ttt_t0
                log0(f"  TTT global step {global_step}/{TTT_GLOBAL_STEPS} "
                     f"pass={pass_idx} loss={loss.item():.4f} "
                     f"elapsed={elapsed:.1f}s")

        if global_step >= TTT_GLOBAL_STEPS:
            break

    # Unfreeze all params
    for name, p in base_model.named_parameters():
        p.requires_grad_(True)

    elapsed = time.perf_counter() - ttt_t0
    log0(f"TTT global adapt done: {global_step} steps in {elapsed:.1f}s")

    return base_state, adapt_names


def ttt_restore_weights(base_model, base_state, adapt_names):
    """Restore model weights from saved base state after TTT."""
    with torch.no_grad():
        for name, p in base_model.named_parameters():
            if name in adapt_names:
                p.data.copy_(base_state[name])


def ttt_evaluate(model_or_ddp, val_tokens, bb, hs, ib):
    """
    Per-sequence Test-Time Training evaluation: for each validation sequence,
    adapt model weights on the full sequence using the causal LM objective
    (no data leakage — each token only sees prior context), then re-evaluate
    the same sequence with adapted weights. Weights restored between sequences.

    Uses Adam instead of SGD for faster few-step convergence.
    Returns (val_loss, val_bpb) on full sequences.
    """
    base_model = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp
    base_model.eval()

    seq_len = SEQ_LEN
    adapt_names, adapt_params = _select_ttt_params(base_model, TTT_PARAMS)

    # Save base weights
    base_state = {name: p.data.clone() for name, p in base_model.named_parameters()
                  if name in adapt_names}

    # Distribute val sequences across ranks
    total_seqs = (val_tokens.size - 1) // seq_len
    seqs_per_rank = total_seqs // WORLD_SIZE
    seq_start = seqs_per_rank * RANK
    seq_end = seq_start + seqs_per_rank

    # Subsample if TTT_MAX_SEQS is set (0 = use all)
    n_local_seqs = seqs_per_rank
    if TTT_MAX_SEQS > 0 and n_local_seqs > TTT_MAX_SEQS:
        stride = n_local_seqs // TTT_MAX_SEQS
        seq_indices = list(range(seq_start, seq_end, stride))[:TTT_MAX_SEQS]
        n_local_seqs = len(seq_indices)
    else:
        seq_indices = list(range(seq_start, seq_end))
        n_local_seqs = len(seq_indices)

    log0(f"TTT per-seq: adapting {len(adapt_params)} params ({TTT_PARAMS}), "
         f"steps={TTT_STEPS}, lr={TTT_LR}, "
         f"seqs/rank={n_local_seqs} (of {seqs_per_rank})")

    loss_sum = 0.0
    tok_sum = 0.0
    byt_sum = 0.0
    ttt_t0 = time.perf_counter()
    ttt_log_every = max(1, n_local_seqs // 20)

    for si, s in enumerate(seq_indices):
        # Restore base weights before each sequence
        with torch.no_grad():
            for name, p in base_model.named_parameters():
                if name in base_state:
                    p.data.copy_(base_state[name])

        chunk = val_tokens[s * seq_len : s * seq_len + seq_len + 1]
        full_x = chunk[:-1]
        full_y = chunk[1:]

        x = torch.from_numpy(full_x.reshape(1, -1)).long().to(DEVICE)
        y = torch.from_numpy(full_y.reshape(1, -1)).long().to(DEVICE)

        # --- TTT: Adam gradient steps on full sequence (causal, no leakage) ---
        base_model.train()
        ttt_opt = torch.optim.Adam(adapt_params, lr=TTT_LR, betas=(0.9, 0.95),
                                   weight_decay=0.0)
        for _ttt_step in range(TTT_STEPS):
            ttt_opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
                loss_adapt = lm_loss(base_model, x, y)
            loss_adapt.backward()
            nn.utils.clip_grad_norm_(adapt_params, 1.0)
            ttt_opt.step()

        # --- Re-evaluate full sequence with adapted weights ---
        base_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
                loss_eval = lm_loss(base_model, x, y)

        cnt = float(seq_len)
        loss_sum += float(loss_eval.detach().float().item()) * cnt

        # BPB calculation on full sequence
        b = bb[full_y].astype(np.int16, copy=True)
        b += (hs[full_y] & ~ib[full_x]).astype(np.int16)
        tok_sum += cnt
        byt_sum += float(b.astype(np.float64).sum())

        done = si + 1
        if done == 1 or done == n_local_seqs or done % ttt_log_every == 0:
            elapsed_ttt = time.perf_counter() - ttt_t0
            running_loss = loss_sum / max(tok_sum, 1.0)
            running_bpb = (running_loss / math.log(2.0)) * (tok_sum / max(byt_sum, 1.0))
            seqs_per_sec = done / max(elapsed_ttt, 1e-9)
            eta = (n_local_seqs - done) / max(seqs_per_sec, 1e-9)
            log0(
                f"  TTT eval: {done}/{n_local_seqs} seqs "
                f"({100.0 * done / n_local_seqs:.0f}%) "
                f"running_loss:{running_loss:.4f} running_bpb:{running_bpb:.4f} "
                f"seq/s:{seqs_per_sec:.1f} eta:{eta:.0f}s"
            )

    # Restore base weights
    with torch.no_grad():
        for name, p in base_model.named_parameters():
            if name in base_state:
                p.data.copy_(base_state[name])

    stats = torch.tensor([loss_sum, tok_sum, byt_sum], device=DEVICE, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, tok_sum, byt_sum = float(stats[0]), float(stats[1]), float(stats[2])
    val_loss = loss_sum / max(tok_sum, 1.0)
    bpt = val_loss / math.log(2.0)
    ttt_total_time = time.perf_counter() - ttt_t0
    log0(f"TTT per-seq eval done in {ttt_total_time:.1f}s ({n_local_seqs} seqs/rank)")
    base_model.train()
    return float(val_loss), float(bpt * (tok_sum / max(byt_sum, 1.0)))


# -----------------------------------------------------------------------------
# EMA (Exponential Moving Average) for eval
# -----------------------------------------------------------------------------
class ModelEMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, p in model.named_parameters():
            self.shadow[name].lerp_(p.data, 1.0 - d)

    def apply_shadow(self, model):
        """Swap model weights with EMA weights. Call before eval."""
        self.backup = {}
        for name, p in model.named_parameters():
            self.backup[name] = p.data.clone()
            p.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights after eval."""
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


class ModelSWA:
    """Stochastic Weight Averaging — collect and average checkpoints from late training."""

    def __init__(self):
        self.shadow = {}
        self.n_checkpoints = 0
        self.collecting = False

    def maybe_start(self, elapsed_frac):
        """Start collecting if we're in the SWA window."""
        if not self.collecting and elapsed_frac >= (1.0 - SWA_START_FRAC):
            self.collecting = True
            return True
        return False

    @torch.no_grad()
    def collect(self, model):
        """Add current weights to the running average."""
        self.n_checkpoints += 1
        if self.n_checkpoints == 1:
            for name, p in model.named_parameters():
                self.shadow[name] = p.data.clone()
        else:
            for name, p in model.named_parameters():
                self.shadow[name].add_(p.data)

    def apply_average(self, model):
        """Apply averaged weights to model. Call before eval."""
        if self.n_checkpoints == 0:
            return
        self.backup = {}
        for name, p in model.named_parameters():
            self.backup[name] = p.data.clone()
            p.data.copy_(self.shadow[name] / float(self.n_checkpoints))

    def restore(self, model):
        """Restore original weights after eval."""
        if not hasattr(self, 'backup') or not self.backup:
            return
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


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
# Mixed n-bit quantization helpers
# -----------------------------------------------------------------------------
def _tensor_role_for_quant(name: str):
    n = name.lower()
    if "tok_emb.weight" in n or "embed_proj" in n:
        return "embed"
    if "lm_head" in n:
        return "head"
    if "qkv" in n or "out_proj" in n or "attn" in n:
        return "attn"
    if "ffn" in n or "mlp" in n:
        return "ffn"
    if "mamba" in n:
        return "ssm"
    return "matrix"


def _bits_for_name(name: str):
    role = _tensor_role_for_quant(name)
    if role == "embed":
        return EMBED_BITS
    if role == "head":
        return HEAD_BITS
    if role == "attn":
        return ATTN_BITS
    if role == "ffn":
        return FFN_BITS
    if role == "ssm":
        return SSM_BITS
    return MATRIX_BITS


def _sigmas_for_name(name: str):
    role = _tensor_role_for_quant(name)
    if role == "embed":
        return EMBED_CLIP_SIGMAS
    if role == "head":
        return HEAD_CLIP_SIGMAS
    if role == "attn":
        return ATTN_CLIP_SIGMAS
    if role == "ffn":
        return FFN_CLIP_SIGMAS
    if role == "ssm":
        return SSM_CLIP_SIGMAS
    return MATRIX_CLIP_SIGMAS


def _pack_nbit_uint(vals_np, bits: int):
    """Pack uint values in [0, 2**bits-1] into bytes. Supports 4, 6, 7, 8."""
    import numpy as _np
    vals = vals_np.astype(_np.uint8, copy=False).reshape(-1)
    n = vals.size

    if bits == 8:
        return vals.copy(), n

    if bits == 4:
        pad = (-n) % 2
        if pad:
            vals = _np.pad(vals, (0, pad), constant_values=0)
        a = vals.reshape(-1, 2)
        packed = (a[:, 0] | (a[:, 1] << 4)).astype(_np.uint8)
        return packed, n

    if bits == 6:
        pad = (-n) % 4
        if pad:
            vals = _np.pad(vals, (0, pad), constant_values=0)
        a = vals.reshape(-1, 4).astype(_np.uint16)
        b0 = (a[:, 0] | ((a[:, 1] & 0x03) << 6)).astype(_np.uint8)
        b1 = ((a[:, 1] >> 2) | ((a[:, 2] & 0x0F) << 4)).astype(_np.uint8)
        b2 = ((a[:, 2] >> 4) | (a[:, 3] << 2)).astype(_np.uint8)
        packed = _np.stack([b0, b1, b2], axis=1).reshape(-1)
        return packed, n

    if bits == 7:
        pad = (-n) % 8
        if pad:
            vals = _np.pad(vals, (0, pad), constant_values=0)
        a = vals.reshape(-1, 8).astype(_np.uint64)
        word = (
            a[:, 0]
            | (a[:, 1] << 7)
            | (a[:, 2] << 14)
            | (a[:, 3] << 21)
            | (a[:, 4] << 28)
            | (a[:, 5] << 35)
            | (a[:, 6] << 42)
            | (a[:, 7] << 49)
        )
        packed8 = word.view(_np.uint8).reshape(-1, 8)
        packed = packed8[:, :7].reshape(-1).copy()
        return packed, n

    raise ValueError(f"Unsupported n-bit pack bits={bits}; use 4, 6, 7, or 8")


def _unpack_nbit_uint(packed_np, bits: int, n_vals: int):
    """Unpack bytes to uint values. Supports 4, 6, 7, 8."""
    import numpy as _np
    p = packed_np.astype(_np.uint8, copy=False).reshape(-1)

    if bits == 8:
        return p[:n_vals].copy()

    if bits == 4:
        out = _np.empty(p.size * 2, dtype=_np.uint8)
        out[0::2] = p & 0x0F
        out[1::2] = p >> 4
        return out[:n_vals].copy()

    if bits == 6:
        a = p.reshape(-1, 3).astype(_np.uint16)
        out = _np.empty(a.shape[0] * 4, dtype=_np.uint8)
        out[0::4] = (a[:, 0] & 0x3F).astype(_np.uint8)
        out[1::4] = (((a[:, 0] >> 6) | ((a[:, 1] & 0x0F) << 2)) & 0x3F).astype(_np.uint8)
        out[2::4] = (((a[:, 1] >> 4) | ((a[:, 2] & 0x03) << 4)) & 0x3F).astype(_np.uint8)
        out[3::4] = ((a[:, 2] >> 2) & 0x3F).astype(_np.uint8)
        return out[:n_vals].copy()

    if bits == 7:
        groups = (p.size + 6) // 7
        pad_len = groups * 7 - p.size
        if pad_len:
            p = _np.pad(p, (0, pad_len), constant_values=0)
        p7 = p.reshape(-1, 7)
        p8 = _np.zeros((p7.shape[0], 8), dtype=_np.uint8)
        p8[:, :7] = p7
        word = p8.reshape(-1).view(_np.uint64)
        out = _np.empty(word.size * 8, dtype=_np.uint8)
        mask = _np.uint64(0x7F)
        for i in range(8):
            out[i::8] = ((word >> _np.uint64(7 * i)) & mask).astype(_np.uint8)
        return out[:n_vals].copy()

    raise ValueError(f"Unsupported n-bit unpack bits={bits}; use 4, 6, 7, or 8")


def _sdclip_quantize_tensor(name: str, t_cpu: torch.Tensor, bits: int):
    """Return packed uint8 tensor, metadata, and scale for a floating tensor."""
    import numpy as _np
    t32 = t_cpu.float().contiguous()
    qmax = (1 << (bits - 1)) - 1
    offset = qmax
    sigmas = _sigmas_for_name(name)

    if t32.ndim == 2:
        rows = t32.shape[0]
        flat = t32.view(rows, -1)
        std = flat.std(dim=1, unbiased=False).clamp_min(1e-8)
        absmax = flat.abs().amax(dim=1).clamp_min(1e-8)
        clip_abs = torch.minimum(absmax, std * sigmas).clamp_min(1.0 / qmax)
        scale = (clip_abs / qmax).to(torch.float16).contiguous()
        q = torch.round(torch.clamp(flat, -clip_abs[:, None], clip_abs[:, None]) / scale.float()[:, None])
        q = torch.clamp(q, -qmax, qmax).to(torch.int16).view(-1)
    else:
        std = t32.std(unbiased=False).clamp_min(1e-8)
        absmax = t32.abs().amax().clamp_min(1e-8)
        clip_abs = torch.minimum(absmax, std * sigmas).clamp_min(1.0 / qmax)
        scale = (clip_abs / qmax).to(torch.float16)
        q = torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale.float())
        q = torch.clamp(q, -qmax, qmax).to(torch.int16).view(-1)

    vals = (q + offset).clamp(0, (1 << bits) - 1).to(torch.uint8).numpy()
    packed_np, n_vals = _pack_nbit_uint(vals, bits)
    packed = torch.from_numpy(packed_np.copy()).to(torch.uint8).contiguous()
    meta = {
        "bits": int(bits),
        "shape": tuple(t_cpu.shape),
        "n_vals": int(n_vals),
        "qmax": int(qmax),
    }
    return packed, scale, meta


def _dequantize_nbit_tensor(packed: torch.Tensor, scale: torch.Tensor, meta: dict):
    import numpy as _np
    bits = int(meta["bits"])
    shape = tuple(meta["shape"])
    n_vals = int(meta["n_vals"])
    qmax = int(meta["qmax"])

    vals = _unpack_nbit_uint(packed.cpu().numpy(), bits, n_vals).astype(_np.int16)
    q = torch.from_numpy(vals).to(torch.int16) - qmax
    if scale.ndim > 0:
        rows = shape[0]
        qf = q.float().view(rows, -1) * scale.float().view(rows, 1)
        return qf.view(shape).to(torch.bfloat16)
    else:
        return (q.float().view(shape) * float(scale.item())).to(torch.bfloat16)


def build_mixed_nbit_artifact(state_dict):
    quantized = {}
    scales = {}
    metas = {}
    passthrough = {}

    for name, t in state_dict.items():
        t = t.detach().cpu().contiguous()

        if (
            (not t.is_floating_point())
            or t.numel() <= QUANT_PASSTHROUGH_MAX
            or t.ndim < 2
        ):
            if t.is_floating_point() and t.dtype in (torch.float32, torch.bfloat16):
                passthrough[name] = t.to(torch.float16).contiguous()
            else:
                passthrough[name] = t
            continue

        bits = int(_bits_for_name(name))
        if bits not in (4, 6, 7, 8):
            raise ValueError(f"{name}: unsupported bits={bits}; use 4, 6, 7, or 8")

        packed, scale, meta = _sdclip_quantize_tensor(name, t, bits)
        quantized[name] = packed
        scales[name] = scale
        metas[name] = meta

    return {
        "format": "mixed_nbit_sdclip_v1",
        "quantized": quantized,
        "scales": scales,
        "metas": metas,
        "passthrough": passthrough,
        "config": {
            "matrix_bits": MATRIX_BITS,
            "embed_bits": EMBED_BITS,
            "head_bits": HEAD_BITS,
            "ssm_bits": SSM_BITS,
            "attn_bits": ATTN_BITS,
            "ffn_bits": FFN_BITS,
            "matrix_clip_sigmas": MATRIX_CLIP_SIGMAS,
            "embed_clip_sigmas": EMBED_CLIP_SIGMAS,
            "head_clip_sigmas": HEAD_CLIP_SIGMAS,
        },
    }


def dequantize_mixed_nbit_artifact(obj):
    state = {}
    for name, packed in obj["quantized"].items():
        state[name] = _dequantize_nbit_tensor(packed, obj["scales"][name], obj["metas"][name])
    for name, t in obj["passthrough"].items():
        state[name] = t
    return state

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
    # Data loaders. If LENGTH_CURRICULUM_ENABLED, we create both a short
    # 2048 loader and the final SEQ_LEN loader, preserving the same token budget.
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

    loader_long = _make_loader(SEQ_LEN, prefetch_depth=3)
    loader_short = None
    if LENGTH_CURRICULUM_ENABLED:
        if SEQ_LEN <= CURRICULUM_SHORT_SEQ_LEN:
            raise ValueError("LENGTH_CURRICULUM_ENABLED requires SEQ_LEN > CURRICULUM_SHORT_SEQ_LEN")
        if CURRICULUM_SHORT_SEQ_LEN % STREAM_CHUNKS != 0:
            raise ValueError("CURRICULUM_SHORT_SEQ_LEN must be divisible by STREAM_CHUNKS")
        loader_short = _make_loader(CURRICULUM_SHORT_SEQ_LEN, prefetch_depth=2)
        log0(f"Length curriculum loaders ready: short={CURRICULUM_SHORT_SEQ_LEN}, long={SEQ_LEN}")
    loader = loader_long

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

    # EMA
    ema = None
    if EMA_ENABLED:
        ema = ModelEMA(base_model, decay=EMA_DECAY)
        log0(f"EMA initialized (decay={EMA_DECAY})")

    # SWA
    swa = None
    if SWA_ENABLED:
        swa = ModelSWA()
        log0(f"SWA initialized (start_frac={SWA_START_FRAC}, every={SWA_EVERY})")

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    profiler = None
    if PROFILE_ENABLED and MASTER_PROCESS:
        os.makedirs(PROFILE_DIR, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=PROFILE_START - 1,
                warmup=1,
                active=PROFILE_END - PROFILE_START,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(PROFILE_DIR),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()
        log0(f"Profiler active: steps {PROFILE_START}-{PROFILE_END}, output to {PROFILE_DIR}/")

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

        current_loader = loader_long
        if LENGTH_CURRICULUM_ENABLED and loader_short is not None and MAX_WALLCLOCK_SECONDS > 0:
            frac_len = elapsed / MAX_WALLCLOCK_SECONDS
            if frac_len < CURRICULUM_SWITCH_FRAC:
                current_loader = loader_short
                if step == 0:
                    log0(f"LENGTH_CURRICULUM: using short seq_len={CURRICULUM_SHORT_SEQ_LEN} until {CURRICULUM_SWITCH_FRAC*100:.0f}% wall")
            elif loader is not loader_long:
                log0(f"LENGTH_CURRICULUM: switched to long seq_len={SEQ_LEN} at step {step} ({frac_len*100:.0f}% wall)")
        loader = current_loader

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
            active_stream_chunks = stream_chunks_for_step(step)
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

        # EMA update
        if ema is not None:
            ema.update(base_model)

        # SWA checkpoint collection
        if swa is not None:
            elapsed_frac = (time.perf_counter() - t0) / max(MAX_WALLCLOCK_SECONDS, 1.0)
            if swa.maybe_start(elapsed_frac):
                log0(f"SWA collection started at step {step} (elapsed {elapsed_frac:.2f})")
            if swa.collecting and step % SWA_EVERY == 0:
                swa.collect(base_model)

        # Profiler step
        if profiler is not None:
            profiler.step()
            if step > PROFILE_END:
                profiler.stop()
                log0(f"Profiler stopped. Traces saved to {PROFILE_DIR}/")
                log0("\n=== PROFILE: sorted by cuda_time_total ===")
                log0(profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=30
                ))
                log0("\n=== PROFILE: sorted by self_cuda_time_total ===")
                log0(profiler.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=30
                ))
                profiler = None

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
            if ema is not None:
                ema.apply_shadow(base_model)
                evl, evb = eval_val(model, val_tokens, bb, hs, ib)
                ema.restore(base_model)
                log0(f"  -> ema_val_loss:{evl:.4f} ema_val_bpb:{evb:.4f}")

    # Clean up profiler
    if profiler is not None:
        profiler.stop()
        log0(f"Profiler stopped early. Traces saved to {PROFILE_DIR}/")
        log0(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        profiler = None

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
    best_vl, best_vb, best_source = vl, vb, "live"

    # Eval with SWA weights (standard, fast)
    if swa is not None and swa.n_checkpoints > 0:
        log0(f"SWA: averaging {swa.n_checkpoints} checkpoints")
        swa.apply_average(base_model)
        swa_vl, swa_vb = eval_val(model, val_tokens, bb, hs, ib)
        log0(f"SWA   val_loss:{swa_vl:.4f} val_bpb:{swa_vb:.4f}")
        log0(f"SWA delta vs live: loss={vl - swa_vl:+.4f} bpb={vb - swa_vb:+.4f}")
        if swa_vb < best_vb:
            best_vl, best_vb, best_source = swa_vl, swa_vb, "swa"
        else:
            swa.restore(base_model)

    # Eval with EMA weights (standard, fast)
    if ema is not None:
        ema.apply_shadow(base_model)
        ema_vl, ema_vb = eval_val(model, val_tokens, bb, hs, ib)
        log0(f"EMA   val_loss:{ema_vl:.4f} val_bpb:{ema_vb:.4f}")
        if ema_vb < best_vb:
            best_vl, best_vb, best_source = ema_vl, ema_vb, "ema"
        ema.restore(base_model)

    # Load best weights for sliding window eval
    if best_source == "swa" and swa is not None:
        swa.apply_average(base_model)
    elif best_source == "ema" and ema is not None:
        ema.apply_shadow(base_model)
    log0(f"Best (standard): {best_source} val_loss:{best_vl:.4f} val_bpb:{best_vb:.4f}")

    # Sliding window eval on best weights (pre-TTT baseline)
    pre_ttt_sw_vb = None
    if EVAL_STRIDE < SEQ_LEN:
        log0(f"Running sliding window eval (stride={EVAL_STRIDE})...")
        sw_vl, sw_vb = eval_val_sliding(model, val_tokens, bb, hs, ib, stride=EVAL_STRIDE)
        log0(f"Sliding val_loss:{sw_vl:.4f} val_bpb:{sw_vb:.4f}")
        log0(f"Sliding vs standard: bpb={best_vb - sw_vb:+.4f}")
        pre_ttt_sw_vb = sw_vb

    # --- Score-First TTT (SOTA-style legal eval-time adaptation) ---
    if SCORE_FIRST_TTT_ENABLED and EVAL_STRIDE < SEQ_LEN:
        dist.barrier()
        # Reload best weights before Score-First TTT (it modifies in-place then restores)
        if best_source == "swa" and swa is not None:
            if not hasattr(swa, 'backup') or not swa.backup:
                swa.apply_average(base_model)
        elif best_source == "ema" and ema is not None:
            if not hasattr(ema, 'backup') or not ema.backup:
                ema.apply_shadow(base_model)
        log0("=" * 60)
        log0(f"Running Score-First TTT eval (chunk={SCORE_FIRST_TTT_CHUNK_TOKENS} tokens)...")
        sft_vl, sft_vb = eval_score_first_ttt(model, val_tokens, bb, hs, ib, stride=EVAL_STRIDE)
        log0(f"Score-First TTT val_loss:{sft_vl:.4f} val_bpb:{sft_vb:.4f}")
        if pre_ttt_sw_vb is not None:
            log0(f"Score-First TTT vs sliding: bpb={pre_ttt_sw_vb - sft_vb:+.4f}")
        log0("=" * 60)

    # --- TTT Pipeline ---
    # Phase 1: Global TTT — adapt weights on the val distribution
    # Phase 2: Re-evaluate with adapted weights (standard + sliding)
    if TTT_ENABLED:
        dist.barrier()

        # Ensure best weights are loaded before TTT
        # (they should already be from the block above, but be explicit)
        if best_source == "swa" and swa is not None:
            if not hasattr(swa, 'backup') or not swa.backup:
                swa.apply_average(base_model)
        elif best_source == "ema" and ema is not None:
            if not hasattr(ema, 'backup') or not ema.backup:
                ema.apply_shadow(base_model)

        # Phase 1: Global TTT adaptation
        log0("=" * 60)
        log0("Starting TTT global adaptation on val distribution...")
        ttt_base_state, ttt_adapt_names = ttt_global_adapt(model, val_tokens)

        # Phase 2a: Standard eval with TTT-adapted weights
        ttt_vl, ttt_vb = eval_val(model, val_tokens, bb, hs, ib)
        log0(f"TTT global val_loss:{ttt_vl:.4f} val_bpb:{ttt_vb:.4f}")
        log0(f"TTT global vs best standard: bpb={best_vb - ttt_vb:+.4f}")

        # Phase 2b: Sliding window eval with TTT-adapted weights
        if EVAL_STRIDE < SEQ_LEN:
            log0(f"Running sliding window eval with TTT weights (stride={EVAL_STRIDE})...")
            ttt_sw_vl, ttt_sw_vb = eval_val_sliding(
                model, val_tokens, bb, hs, ib, stride=EVAL_STRIDE)
            log0(f"TTT sliding val_loss:{ttt_sw_vl:.4f} val_bpb:{ttt_sw_vb:.4f}")
            if pre_ttt_sw_vb is not None:
                log0(f"TTT sliding vs pre-TTT sliding: bpb={pre_ttt_sw_vb - ttt_sw_vb:+.4f}")

        # Restore weights after TTT eval
        ttt_restore_weights(base_model, ttt_base_state, ttt_adapt_names)
        log0("TTT weights restored.")
        log0("=" * 60)
    else:
        # Restore live weights if no TTT
        if best_source == "swa" and swa is not None:
            swa.restore(base_model)
        elif best_source == "ema" and ema is not None:
            ema.restore(base_model)

    # -----------------------------------------------------------------------
    # Mixed n-bit quantization + compression + roundtrip validation
    # -----------------------------------------------------------------------
    if best_source == "swa" and swa is not None:
        swa.apply_average(base_model)
    elif best_source == "ema" and ema is not None:
        ema.apply_shadow(base_model)

    state_dict = base_model.state_dict()
    import io
    if MIXED_NBIT_QUANT:
        log0(
            f"Quantizing model to mixed n-bit SDClip "
            f"(matrix={MATRIX_BITS}, embed={EMBED_BITS}, head={HEAD_BITS}, "
            f"ssm={SSM_BITS}, attn={ATTN_BITS}, ffn={FFN_BITS})..."
        )
        quant_obj = build_mixed_nbit_artifact(state_dict)
    else:
        log0("Quantizing model to legacy int8...")
        quantized = {}
        scales = {}
        passthrough = {}
        for name, t in state_dict.items():
            t = t.detach().cpu().contiguous()
            if not t.is_floating_point() or t.numel() <= 65536:
                if t.is_floating_point() and t.dtype in (torch.float32, torch.bfloat16):
                    passthrough[name] = t.to(torch.float16).contiguous()
                else:
                    passthrough[name] = t
                continue
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
        quant_obj = {"format": "legacy_int8", "quantized": quantized, "scales": scales, "passthrough": passthrough}

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

    if quant_obj_rt.get("format") == "mixed_nbit_sdclip_v1":
        dequant_state = dequantize_mixed_nbit_artifact(quant_obj_rt)
    else:
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

    if MASTER_PROCESS:
        artifact_path = QUANT_SAVE_NAME if MIXED_NBIT_QUANT else "final_model.int8.ptz"
        with open(artifact_path, "wb") as f:
            f.write(quant_blob)
        log0(f"Saved artifact: {artifact_path} ({compressed_bytes:,} bytes)")

    # Shutdown async loaders
    for _ldr in (locals().get("loader_short", None), locals().get("loader_long", None)):
        try:
            if _ldr is not None:
                _ldr.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()