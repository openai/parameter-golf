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
DOWNLOAD_DATA = os.environ.get("DOWNLOAD_DATA", "1") == "1"
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
EMBED_DIM = int(os.environ.get("EMBED_DIM", "256"))  # 0 = disabled (full D_MODEL)

# Hybrid attention config
ATTN_LAYER_IDXS = [int(x) for x in os.environ.get("ATTN_LAYER_IDXS", "5").split(",") if x.strip()]
ATTN_N_HEADS = int(os.environ.get("ATTN_N_HEADS", "8"))
QK_GAIN_INIT = float(os.environ.get("QK_GAIN_INIT", "2.25"))  # learnable per-head query scaling

# Eval-only score-first LoRA TTT config.
# This script loads a checkpoint/artifact and adapts only LoRA adapters on
# already-scored validation chunks.
TTT_LORA_ENABLED = os.environ.get("TTT_LORA_ENABLED", "1") == "1"
TTT_LORA_RANK = int(os.environ.get("TTT_LORA_RANK", "4"))
TTT_LORA_ALPHA = float(os.environ.get("TTT_LORA_ALPHA", "8.0"))
TTT_LORA_STD = float(os.environ.get("TTT_LORA_STD", "0.01"))
TTT_LORA_QKV = os.environ.get("TTT_LORA_QKV", "1") == "1"
TTT_LORA_OUT = os.environ.get("TTT_LORA_OUT", "1") == "1"

# Score-first adapter TTT runtime settings.
ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH", "/workspace/parameter-golf/final_model.int8.ptz")
RUN_BASELINE = os.environ.get("RUN_BASELINE", "1") == "1"
TTT_CHUNK_TOKENS = int(os.environ.get("TTT_CHUNK_TOKENS", "262144"))
TTT_TRAIN_WINDOWS = int(os.environ.get("TTT_TRAIN_WINDOWS", "4"))
TTT_EPOCHS = int(os.environ.get("TTT_EPOCHS", "1"))
TTT_ADAPTER_LR = float(os.environ.get("TTT_ADAPTER_LR", "5e-4"))
TTT_ADAPTER_WD = float(os.environ.get("TTT_ADAPTER_WD", "0.0"))
TTT_LORA_DECAY = float(os.environ.get("TTT_LORA_DECAY", "0.995"))
TTT_GRAD_CLIP = float(os.environ.get("TTT_GRAD_CLIP", "1.0"))
TTT_OPT = os.environ.get("TTT_OPT", "adamw").lower()  # "adamw" or "sgd"
TTT_MAX_CHUNKS = int(os.environ.get("TTT_MAX_CHUNKS", "0"))  # 0 = all

# Score-first logit-bias TTT.
# This is a tiny 8192-dim adapter: logits = base_logits + vocab_bias.
# It learns local token/document frequency from already-scored chunks.
TTT_BIAS_ENABLED = os.environ.get("TTT_BIAS_ENABLED", "1") == "1"
TTT_BIAS_LR = float(os.environ.get("TTT_BIAS_LR", "0.05"))
TTT_BIAS_WD = float(os.environ.get("TTT_BIAS_WD", "0.0"))
TTT_BIAS_DECAY = float(os.environ.get("TTT_BIAS_DECAY", "0.995"))
TTT_BIAS_GRAD_CLIP = float(os.environ.get("TTT_BIAS_GRAD_CLIP", "5.0"))
TTT_BIAS_CENTER = os.environ.get("TTT_BIAS_CENTER", "1") == "1"
TTT_BIAS_OPT = os.environ.get("TTT_BIAS_OPT", "adamw").lower()  # "adamw" or "sgd"

# Dynamic / continuous-thought TTT.
# Always do TTT_MIN_STEPS after scoring a chunk. If the scored chunk is harder
# than the moving average, spend up to TTT_MAX_STEPS, stopping early when
# adaptation stops improving. This stays score-first because all updates happen
# after the chunk has already been scored.
TTT_DYNAMIC_ENABLED = os.environ.get("TTT_DYNAMIC_ENABLED", "0") == "1"
TTT_MIN_STEPS = int(os.environ.get("TTT_MIN_STEPS", "1"))
TTT_MAX_STEPS = int(os.environ.get("TTT_MAX_STEPS", "4"))
TTT_SCORE_EMA_BETA = float(os.environ.get("TTT_SCORE_EMA_BETA", "0.95"))
TTT_DYNAMIC_SCORE_MARGIN = float(os.environ.get("TTT_DYNAMIC_SCORE_MARGIN", "0.02"))
TTT_DYNAMIC_ADAPT_MARGIN = float(os.environ.get("TTT_DYNAMIC_ADAPT_MARGIN", "0.02"))
TTT_MIN_IMPROVEMENT = float(os.environ.get("TTT_MIN_IMPROVEMENT", "0.001"))
TTT_EXTRA_CHUNK_FRAC_CAP = float(os.environ.get("TTT_EXTRA_CHUNK_FRAC_CAP", "0.35"))
TTT_DYNAMIC_LOG = os.environ.get("TTT_DYNAMIC_LOG", "1") == "1"

# Force depth recurrence at eval-time for checkpoints trained/evaluated with recurrence on.
# Needed because training script toggles loop_enabled_external during training,
# but a fresh eval-only model starts with it disabled.
FORCE_LOOP_ENABLED = os.environ.get("FORCE_LOOP_ENABLED", "0") == "1"

# Activation choice
ACTIVATION = os.environ.get("ACTIVATION", "swiglu").lower()  # "swiglu" or "leaky_relu2"

# Depth recurrence — loop layers [LOOP_START..LOOP_END] inclusive twice during decoder phase
LOOP_START = int(os.environ.get("LOOP_START", "-1"))  # -1 = disabled
LOOP_END = int(os.environ.get("LOOP_END", "-1"))      # inclusive
LOOP_ACTIVATE_FRAC = float(os.environ.get("LOOP_ACTIVATE_FRAC", "0.35"))

# Training
ITERATIONS = int(os.environ.get("ITERATIONS", "20000"))
VAL_EVERY = int(os.environ.get("VAL_EVERY", "0"))  # 0=final only (each val costs ~100 steps)
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "0.3"))
STREAM_CHUNKS = int(os.environ.get("STREAM_CHUNKS", "1"))
BPTT_CHUNKS = int(os.environ.get("BPTT_CHUNKS", "1"))
MIXED_LENGTH_CURRICULUM = os.environ.get("MIXED_LENGTH_CURRICULUM", "0") == "1"
CURRICULUM_STEPS = int(os.environ.get("CURRICULUM_STEPS", "1500"))
SEED = int(os.environ.get("SEED", "7"))
MAX_WALLCLOCK_SECONDS = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "700"))

# LR schedule: short warmup, then long warmdown (matching competitive GPT config)
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "20"))
LR_MIN_SCALE = float(os.environ.get("LR_MIN_SCALE", "0.0"))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine_late")
LR_WARMDOWN_START_FRAC = float(os.environ.get("LR_WARMDOWN_START_FRAC", "0.28"))  # SOTA: 0.72 of training is warmdown

# SWA (Stochastic Weight Averaging) — collect checkpoints from late warmdown
SWA_ENABLED = os.environ.get("SWA_ENABLED", "1") == "1"
SWA_START_FRAC = float(os.environ.get("SWA_START_FRAC", "0.15"))  # collect from last 15%
SWA_EVERY = int(os.environ.get("SWA_EVERY", "50"))  # checkpoint every N steps

# EMA — SOTA submission uses EMA with decay 0.9965 successfully
EMA_ENABLED = os.environ.get("EMA_ENABLED", "1") == "1"
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
SCORE_FIRST_TTT_ENABLED = os.environ.get("SCORE_FIRST_TTT_ENABLED", "1") == "1"
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
log0(f"hybrid_attn: layer_idxs={ATTN_LAYER_IDXS}, n_heads={ATTN_N_HEADS}, qk_gain_init={QK_GAIN_INIT}")
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

        # Eval-time LoRA adapters. Zero-B init means loaded checkpoint behavior
        # is exactly preserved before TTT adaptation.
        self.ttt_lora_rank = TTT_LORA_RANK if TTT_LORA_ENABLED else 0
        self.ttt_lora_scale = TTT_LORA_ALPHA / max(TTT_LORA_RANK, 1)
        if self.ttt_lora_rank > 0 and TTT_LORA_QKV:
            self.qkv_lora_A = nn.Parameter(torch.empty(self.ttt_lora_rank, d_model))
            self.qkv_lora_B = nn.Parameter(torch.zeros(3 * d_model, self.ttt_lora_rank))
            nn.init.normal_(self.qkv_lora_A, std=TTT_LORA_STD)
        else:
            self.qkv_lora_A = None
            self.qkv_lora_B = None

        if self.ttt_lora_rank > 0 and TTT_LORA_OUT:
            self.out_lora_A = nn.Parameter(torch.empty(self.ttt_lora_rank, d_model))
            self.out_lora_B = nn.Parameter(torch.zeros(d_model, self.ttt_lora_rank))
            nn.init.normal_(self.out_lora_A, std=TTT_LORA_STD)
        else:
            self.out_lora_A = None
            self.out_lora_B = None

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
        qkv_raw = self.qkv(normed)
        if self.qkv_lora_A is not None:
            qkv_delta = F.linear(F.linear(normed, self.qkv_lora_A), self.qkv_lora_B)
            qkv_raw = qkv_raw + self.ttt_lora_scale * qkv_delta
        qkv = qkv_raw.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)

        # QK RMSNorm + learnable per-head query gain (SOTA)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(attn_out)
        if self.out_lora_A is not None:
            out_delta = F.linear(F.linear(attn_out, self.out_lora_A), self.out_lora_B)
            out = out + self.ttt_lora_scale * out_delta
        x = x + self.attn_scale[None, None, :] * out

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
            self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        else:
            self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
            self.embed_proj = None
            self.embed_proj_rev = None
            self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        with torch.no_grad():
            self.tok_emb.weight.normal_(mean=0.0, std=EMBED_INIT_STD)
        # Tied embeddings
        self.lm_head.weight = self.tok_emb.weight

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
    if core.use_factored:
        flat_h = hidden.reshape(-1, D_MODEL)
        h_proj = F.linear(flat_h, core.embed_proj_rev.weight.to(flat_h.dtype))
        logits = F.linear(h_proj, core.lm_head.weight.to(h_proj.dtype))
    else:
        flat_h = hidden.reshape(-1, core.lm_head.weight.shape[1])
        logits = F.linear(flat_h, core.lm_head.weight.to(flat_h.dtype))
    logits = core.softcap * torch.tanh(logits / core.softcap)
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
        if core.use_factored:
            flat_h = hidden_tail.reshape(-1, D_MODEL)
            h_proj = F.linear(flat_h, core.embed_proj_rev.weight.to(flat_h.dtype))
            logits = F.linear(h_proj, core.lm_head.weight.to(h_proj.dtype))
        else:
            flat_h = hidden_tail.reshape(-1, core.lm_head.weight.shape[1])
            logits = F.linear(flat_h, core.lm_head.weight.to(flat_h.dtype))
        logits = core.softcap * torch.tanh(logits / core.softcap)
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
                if core.use_factored:
                    flat_h = hidden_tail.reshape(-1, D_MODEL)
                    h_proj = F.linear(flat_h, core.embed_proj_rev.weight.to(flat_h.dtype))
                    logits = F.linear(h_proj, core.lm_head.weight.to(h_proj.dtype))
                else:
                    flat_h = hidden_tail.reshape(-1, core.lm_head.weight.shape[1])
                    logits = F.linear(flat_h, core.lm_head.weight.to(flat_h.dtype))
                logits = core.softcap * torch.tanh(logits / core.softcap)
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
# Eval-only checkpoint loading + score-first LoRA TTT
# -----------------------------------------------------------------------------
def _try_torch_load_bytes(raw: bytes):
    import io
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu")
    except Exception:
        return None


def load_checkpoint_state(path: str):
    """
    Loads either:
      - raw torch state_dict / {"state_dict": ...}
      - compressed int8 artifact from this script:
          {"quantized": ..., "scales": ..., "passthrough": ...}
    """
    import io
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ARTIFACT_PATH not found: {path}")

    raw = p.read_bytes()
    obj = _try_torch_load_bytes(raw)

    if obj is None:
        # Try zstd, then zlib.
        if _COMPRESSOR == "zstd":
            try:
                dctx = zstandard.ZstdDecompressor()
                obj = torch.load(io.BytesIO(dctx.decompress(raw)), map_location="cpu")
            except Exception:
                obj = None
        if obj is None:
            try:
                obj = torch.load(io.BytesIO(zlib.decompress(raw)), map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"Could not load checkpoint/artifact {path}: {e}") from e

    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]

    if isinstance(obj, dict) and "quantized" in obj and "scales" in obj and "passthrough" in obj:
        state = {}
        for name, q in obj["quantized"].items():
            s = obj["scales"][name]
            if s.ndim > 0:
                state[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(torch.bfloat16)
            else:
                state[name] = (q.float() * float(s.item())).to(torch.bfloat16)
        for name, t in obj["passthrough"].items():
            state[name] = t
        return state

    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint object type: {type(obj)}")
    return obj


def get_lora_params(model: nn.Module):
    params = []
    names = []
    for name, p in model.named_parameters():
        is_lora = (
            "qkv_lora_A" in name or "qkv_lora_B" in name or
            "out_lora_A" in name or "out_lora_B" in name
        )
        p.requires_grad_(is_lora)
        if is_lora:
            params.append(p)
            names.append(name)
    return names, params


@torch.no_grad()
def reset_lora_adapters(model: nn.Module):
    for name, p in model.named_parameters():
        if "lora_B" in name:
            p.zero_()


@torch.no_grad()
def decay_lora_adapters(model: nn.Module, decay: float):
    if decay >= 1.0:
        return
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.mul_(decay)


@torch.no_grad()
def decay_vocab_bias(vocab_bias, decay: float):
    if vocab_bias is None or decay >= 1.0:
        return
    vocab_bias.mul_(decay)
    if TTT_BIAS_CENTER:
        vocab_bias.sub_(vocab_bias.mean())


def logits_from_hidden(core, hidden, vocab_bias=None):
    if core.use_factored:
        flat_h = hidden.reshape(-1, D_MODEL)
        h_proj = F.linear(flat_h, core.embed_proj_rev.weight.to(flat_h.dtype))
        logits = F.linear(h_proj, core.lm_head.weight.to(h_proj.dtype))
    else:
        flat_h = hidden.reshape(-1, core.lm_head.weight.shape[1])
        logits = F.linear(flat_h, core.lm_head.weight.to(flat_h.dtype))
    logits = core.softcap * torch.tanh(logits / core.softcap)
    if vocab_bias is not None:
        logits = logits + vocab_bias.to(dtype=logits.dtype, device=logits.device)[None, :]
    return logits


def score_window_batch(core, xn, yn, bb, hs, ib, stride, vocab_bias=None):
    x = torch.from_numpy(xn).long().to(DEVICE, non_blocking=True)
    y = torch.from_numpy(yn).long().to(DEVICE, non_blocking=True)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
            hidden = core(x, state=None, return_state=False)
            hidden_tail = hidden[:, -stride:, :]
            flat_y = y[:, -stride:].reshape(-1)
            logits = logits_from_hidden(core, hidden_tail, vocab_bias=vocab_bias)
            loss = F.cross_entropy(logits.float(), flat_y, reduction="sum")

    cnt = float(flat_y.numel())
    p_tail = xn[:, -stride:].reshape(-1)
    t_tail = yn[:, -stride:].reshape(-1)
    b = bb[t_tail].astype(np.int16, copy=True)
    b += (hs[t_tail] & ~ib[p_tail]).astype(np.int16)
    byt = float(b.astype(np.float64).sum())
    return float(loss.detach().float().item()), cnt, byt


def adapt_window_batch(core, xn, yn, stride, vocab_bias=None, bias_only=False):
    x = torch.from_numpy(xn).long().to(DEVICE, non_blocking=True)
    y = torch.from_numpy(yn).long().to(DEVICE, non_blocking=True)
    if bias_only:
        # Freeze base compute entirely; only gradient is through vocab_bias.
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
                hidden = core(x, state=None, return_state=False)
                hidden_tail = hidden[:, -stride:, :]
                base_logits = logits_from_hidden(core, hidden_tail, vocab_bias=None).detach()
        flat_y = y[:, -stride:].reshape(-1)
        logits = base_logits.float() + vocab_bias.float()[None, :]
        loss = F.cross_entropy(logits, flat_y, reduction="mean")
    else:
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
            hidden = core(x, state=None, return_state=False)
            hidden_tail = hidden[:, -stride:, :]
            flat_y = y[:, -stride:].reshape(-1)
            logits = logits_from_hidden(core, hidden_tail, vocab_bias=vocab_bias)
            loss = F.cross_entropy(logits.float(), flat_y, reduction="mean")
    return loss


def make_windows(val_tokens, window_ids, seq_len, stride):
    x_list, y_list = [], []
    max_tok = val_tokens.size
    for w in window_ids:
        pos = int(w) * stride
        if pos + seq_len + 1 <= max_tok:
            x_list.append(val_tokens[pos : pos + seq_len])
            y_list.append(val_tokens[pos + 1 : pos + seq_len + 1])
    if not x_list:
        return None, None
    return np.stack(x_list), np.stack(y_list)


def eval_score_first_lora_ttt(core, val_tokens, bb, hs, ib, stride=None):
    """
    Legal score-first adapter TTT with optional dynamic compute.

    For each validation chunk:
      1. Score chunk with current adapters/bias. These are final scores.
      2. Adapt only on already-scored windows from that chunk.
      3. If dynamic is enabled, do extra adaptation steps only for hard chunks,
         stopping early when improvement is too small.
    """
    if stride is None:
        stride = EVAL_STRIDE

    core.train()  # adapters update; dropout absent
    reset_lora_adapters(core)
    lora_names, lora_params = get_lora_params(core)
    n_lora = sum(p.numel() for p in lora_params)
    log0(
        f"LoRA TTT: params={len(lora_params)} ({n_lora:,}), rank={TTT_LORA_RANK}, "
        f"alpha={TTT_LORA_ALPHA}, chunk_tokens={TTT_CHUNK_TOKENS}, "
        f"train_windows/rank={TTT_TRAIN_WINDOWS}, static_epochs={TTT_EPOCHS}, "
        f"lr={TTT_ADAPTER_LR}, decay={TTT_LORA_DECAY}, opt={TTT_OPT}"
    )
    if TTT_DYNAMIC_ENABLED:
        log0(
            f"Dynamic TTT: min_steps={TTT_MIN_STEPS}, max_steps={TTT_MAX_STEPS}, "
            f"score_margin={TTT_DYNAMIC_SCORE_MARGIN}, adapt_margin={TTT_DYNAMIC_ADAPT_MARGIN}, "
            f"min_improvement={TTT_MIN_IMPROVEMENT}, extra_cap={TTT_EXTRA_CHUNK_FRAC_CAP}"
        )
    if MASTER_PROCESS and len(lora_names) > 0:
        log0("  lora preview: " + ", ".join(lora_names[:8]) + ("..." if len(lora_names) > 8 else ""))

    # Optional vocab logit-bias adapter. This is often the cheapest useful
    # TTT state: 8192 params that learn local token frequency after each scored chunk.
    vocab_bias = None
    bias_opt = None
    if TTT_BIAS_ENABLED:
        vocab_bias = nn.Parameter(torch.zeros(VOCAB_SIZE, device=DEVICE, dtype=torch.float32))
        if TTT_BIAS_OPT == "sgd":
            bias_opt = torch.optim.SGD([vocab_bias], lr=TTT_BIAS_LR, momentum=0.0, weight_decay=TTT_BIAS_WD)
        else:
            bias_opt = torch.optim.AdamW([vocab_bias], lr=TTT_BIAS_LR, betas=(0.9, 0.95), weight_decay=TTT_BIAS_WD)
        log0(
            f"Logit-bias TTT: enabled=True, params={VOCAB_SIZE}, lr={TTT_BIAS_LR}, "
            f"decay={TTT_BIAS_DECAY}, opt={TTT_BIAS_OPT}, center={TTT_BIAS_CENTER}"
        )

    if len(lora_params) == 0 and not TTT_BIAS_ENABLED:
        raise RuntimeError("No TTT params found. Enable LoRA and/or TTT_BIAS_ENABLED.")

    opt = None
    if len(lora_params) > 0:
        if TTT_OPT == "sgd":
            opt = torch.optim.SGD(lora_params, lr=TTT_ADAPTER_LR, momentum=0.0, weight_decay=TTT_ADAPTER_WD)
        else:
            opt = torch.optim.AdamW(lora_params, lr=TTT_ADAPTER_LR, betas=(0.9, 0.95), weight_decay=TTT_ADAPTER_WD)

    seq_len = SEQ_LEN
    total_tokens = val_tokens.size - 1
    total_windows = max(0, (total_tokens - seq_len) // stride + 1)
    windows_per_chunk = max(1, TTT_CHUNK_TOKENS // stride)
    n_chunks = (total_windows + windows_per_chunk - 1) // windows_per_chunk
    if TTT_MAX_CHUNKS > 0:
        n_chunks = min(n_chunks, TTT_MAX_CHUNKS)

    batch_windows = max(1, 131072 // seq_len)
    loss_sum = 0.0
    tok_sum = 0.0
    byt_sum = 0.0
    t0_eval = time.perf_counter()
    log_every = max(1, n_chunks // 10)

    score_loss_ema = None
    extra_chunks_used = 0
    total_adapt_steps = 0

    for ci in range(n_chunks):
        w0 = ci * windows_per_chunk
        w1 = min((ci + 1) * windows_per_chunk, total_windows)
        n_local = max(0, w1 - w0)

        # Score first: split this same global chunk across ranks.
        local_start = w0 + (n_local * RANK) // WORLD_SIZE
        local_end = w0 + (n_local * (RANK + 1)) // WORLD_SIZE

        chunk_loss_sum = 0.0
        chunk_tok_sum = 0.0
        chunk_byt_sum = 0.0

        core.eval()
        for wb in range(local_start, local_end, batch_windows):
            we = min(wb + batch_windows, local_end)
            ids = list(range(wb, we))
            xn, yn = make_windows(val_tokens, ids, seq_len, stride)
            if xn is None:
                continue
            l, c, b = score_window_batch(core, xn, yn, bb, hs, ib, stride, vocab_bias=vocab_bias)
            loss_sum += l
            tok_sum += c
            byt_sum += b
            chunk_loss_sum += l
            chunk_tok_sum += c
            chunk_byt_sum += b

        # Aggregate chunk score to decide whether to spend extra thought.
        chunk_stats = torch.tensor([chunk_loss_sum, chunk_tok_sum, chunk_byt_sum], device=DEVICE, dtype=torch.float64)
        dist.all_reduce(chunk_stats, op=dist.ReduceOp.SUM)
        g_chunk_loss = float(chunk_stats[0])
        g_chunk_tok = float(chunk_stats[1])
        chunk_score_loss = g_chunk_loss / max(g_chunk_tok, 1.0)

        prev_ema = score_loss_ema
        if score_loss_ema is None:
            score_loss_ema = chunk_score_loss
        else:
            score_loss_ema = (
                TTT_SCORE_EMA_BETA * score_loss_ema
                + (1.0 - TTT_SCORE_EMA_BETA) * chunk_score_loss
            )

        extra_budget = max(0, int(round(TTT_EXTRA_CHUNK_FRAC_CAP * n_chunks)))
        hard_chunk = (
            TTT_DYNAMIC_ENABLED
            and prev_ema is not None
            and chunk_score_loss > (prev_ema + TTT_DYNAMIC_SCORE_MARGIN)
            and extra_chunks_used < extra_budget
        )

        if TTT_DYNAMIC_ENABLED:
            max_steps_this = TTT_MAX_STEPS if hard_chunk else TTT_MIN_STEPS
            min_steps_this = TTT_MIN_STEPS
        else:
            max_steps_this = TTT_EPOCHS
            min_steps_this = TTT_EPOCHS

        actual_steps = 0
        last_metric = None
        last_improvement = None

        # Then adapt on already-scored windows from this chunk only.
        if TTT_TRAIN_WINDOWS > 0 and max_steps_this > 0:
            core.train()
            n_train_total = max(TTT_TRAIN_WINDOWS * WORLD_SIZE, WORLD_SIZE)
            sample_ids = np.linspace(w0, max(w0, w1 - 1), num=n_train_total, dtype=np.int64)
            sample_ids = sample_ids[RANK::WORLD_SIZE][:TTT_TRAIN_WINDOWS]
            sample_ids = [int(s) for s in sample_ids if local_start <= int(s) < local_end or w0 <= int(s) < w1]
            if sample_ids:
                xn, yn = make_windows(val_tokens, sample_ids, seq_len, stride)
                if xn is not None:
                    for step_idx in range(max_steps_this):
                        metrics = []

                        # Update LoRA params, if enabled.
                        if len(lora_params) > 0 and opt is not None:
                            opt.zero_grad(set_to_none=True)
                            loss = adapt_window_batch(core, xn, yn, stride, vocab_bias=vocab_bias, bias_only=False)
                            loss.backward()
                            metrics.append(float(loss.detach().float().item()))
                            for p in lora_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                            if TTT_GRAD_CLIP > 0:
                                nn.utils.clip_grad_norm_(lora_params, TTT_GRAD_CLIP)
                            opt.step()

                        # Update vocab bias separately with frozen base logits.
                        if TTT_BIAS_ENABLED and bias_opt is not None:
                            bias_opt.zero_grad(set_to_none=True)
                            bias_loss = adapt_window_batch(core, xn, yn, stride, vocab_bias=vocab_bias, bias_only=True)
                            bias_loss.backward()
                            metrics.append(float(bias_loss.detach().float().item()))
                            if vocab_bias.grad is not None:
                                dist.all_reduce(vocab_bias.grad, op=dist.ReduceOp.AVG)
                            if TTT_BIAS_GRAD_CLIP > 0 and vocab_bias.grad is not None:
                                vocab_bias.grad.clamp_(-TTT_BIAS_GRAD_CLIP, TTT_BIAS_GRAD_CLIP)
                            bias_opt.step()
                            with torch.no_grad():
                                if TTT_BIAS_CENTER:
                                    vocab_bias.sub_(vocab_bias.mean())

                        actual_steps += 1
                        total_adapt_steps += 1

                        metric = sum(metrics) / max(len(metrics), 1)
                        if last_metric is not None:
                            last_improvement = last_metric - metric
                        last_metric = metric

                        # Dynamic early stop after the mandatory minimum.
                        if TTT_DYNAMIC_ENABLED and actual_steps >= min_steps_this:
                            target = (prev_ema if prev_ema is not None else chunk_score_loss) + TTT_DYNAMIC_ADAPT_MARGIN
                            if not hard_chunk:
                                break
                            if metric <= target:
                                break
                            if last_improvement is not None and last_improvement < TTT_MIN_IMPROVEMENT:
                                break

                    if actual_steps > TTT_MIN_STEPS:
                        extra_chunks_used += 1

                    decay_lora_adapters(core, TTT_LORA_DECAY)
                    decay_vocab_bias(vocab_bias, TTT_BIAS_DECAY)

        if (ci + 1) % log_every == 0 or ci == 0 or ci == n_chunks - 1:
            elapsed = time.perf_counter() - t0_eval
            stats_now = torch.tensor([loss_sum, tok_sum, byt_sum], device=DEVICE, dtype=torch.float64)
            dist.all_reduce(stats_now, op=dist.ReduceOp.SUM)
            rs_loss, rs_tok, rs_byt = float(stats_now[0]), float(stats_now[1]), float(stats_now[2])
            run_loss = rs_loss / max(rs_tok, 1.0)
            run_bpb = (run_loss / math.log(2.0)) * (rs_tok / max(rs_byt, 1.0))
            eta = elapsed / max(ci + 1, 1) * (n_chunks - ci - 1)
            dyn = ""
            if TTT_DYNAMIC_ENABLED and TTT_DYNAMIC_LOG:
                dyn = (
                    f" chunk_loss:{chunk_score_loss:.4f} ema:{score_loss_ema:.4f} "
                    f"hard:{int(hard_chunk)} steps:{actual_steps} extra:{extra_chunks_used}"
                )
            log0(f"  lora_ttt: chunk {ci+1}/{n_chunks} running_bpb:{run_bpb:.4f}{dyn} eta:{eta:.0f}s")

    stats = torch.tensor([loss_sum, tok_sum, byt_sum], device=DEVICE, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, tok_sum, byt_sum = float(stats[0]), float(stats[1]), float(stats[2])
    val_loss = loss_sum / max(tok_sum, 1.0)
    bpt = val_loss / math.log(2.0)
    elapsed = time.perf_counter() - t0_eval
    log0(
        f"LoRA/Bias TTT done: chunks={n_chunks}, scored_tokens={tok_sum:.0f}, "
        f"adapt_steps={total_adapt_steps}, extra_chunks={extra_chunks_used}, elapsed={elapsed:.1f}s"
    )
    return float(val_loss), float(bpt * (tok_sum / max(byt_sum, 1.0)))


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

    base_model = SSM_LM().to(DEVICE)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"Model parameters: {n_params:,}")

    if ENABLE_S4D_INIT:
        # Safe even though weights will be overwritten for base params.
        n_init, n_skip = apply_s4d_init_to_mamba(base_model)
        log0(f"S4D init applied to {n_init} module(s), skipped {n_skip}")

    n_cast = cast_params_to_dtype(base_model, AMP_DTYPE)
    log0(f"Pre-cast {n_cast} parameters to {AMP_DTYPE}")

    log0(f"Loading checkpoint/artifact: {ARTIFACT_PATH}")
    state = load_checkpoint_state(ARTIFACT_PATH)
    missing, unexpected = base_model.load_state_dict(state, strict=False)
    missing_non_lora = [k for k in missing if "lora_" not in k]
    unexpected_non_lora = [k for k in unexpected if "lora_" not in k]
    if missing_non_lora:
        log0(f"WARNING: non-LoRA missing keys: {missing_non_lora[:12]}{'...' if len(missing_non_lora) > 12 else ''}")
    if unexpected_non_lora:
        log0(f"WARNING: unexpected keys: {unexpected_non_lora[:12]}{'...' if len(unexpected_non_lora) > 12 else ''}")
    log0(f"Missing LoRA keys initialized fresh: {len([k for k in missing if 'lora_' in k])}")

    if FORCE_LOOP_ENABLED:
        if base_model.loop_enc_seq is None or base_model.loop_dec_seq is None:
            raise RuntimeError(
                "FORCE_LOOP_ENABLED=1 but recurrence sequences are not configured. "
                "Pass LOOP_START and LOOP_END, e.g. LOOP_START=5 LOOP_END=5."
            )
        base_model.loop_enabled_external = True
        log0(
            f"FORCE_LOOP_ENABLED=1: using recurrence "
            f"enc={base_model.loop_enc_seq} dec={base_model.loop_dec_seq}"
        )
    else:
        log0("FORCE_LOOP_ENABLED=0: recurrence disabled for eval")

    # Freeze base and enable LoRA params only.
    lora_names, lora_params = get_lora_params(base_model)
    log0(f"Trainable LoRA params: {len(lora_params)} tensors, {sum(p.numel() for p in lora_params):,} params")

    model_for_eval = base_model
    if USE_TORCH_COMPILE:
        try:
            model_for_eval = torch.compile(base_model, dynamic=False)
            log0("torch_compile: enabled")
        except Exception as e:
            model_for_eval = base_model
            log0(f"torch_compile: failed ({e}); falling back to eager")
    else:
        log0("torch_compile: disabled")

    # Optional baseline before TTT.
    if RUN_BASELINE:
        vl, vb = eval_val(model_for_eval, val_tokens, bb, hs, ib)
        log0(f"Loaded baseline standard val_loss:{vl:.4f} val_bpb:{vb:.4f}")
        if EVAL_STRIDE < SEQ_LEN:
            sw_vl, sw_vb = eval_val_sliding(model_for_eval, val_tokens, bb, hs, ib, stride=EVAL_STRIDE)
            log0(f"Loaded baseline sliding val_loss:{sw_vl:.4f} val_bpb:{sw_vb:.4f}")

    dist.barrier()
    log0("=" * 60)
    log0("Running score-first attention-LoRA TTT...")
    ttt_vl, ttt_vb = eval_score_first_lora_ttt(model_for_eval, val_tokens, bb, hs, ib, stride=EVAL_STRIDE)
    log0(f"Score-first LoRA TTT val_loss:{ttt_vl:.4f} val_bpb:{ttt_vb:.4f}")
    log0("=" * 60)


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()