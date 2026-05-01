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
try:
    from mamba_ssm.utils.generation import InferenceParams
except Exception:
    InferenceParams = None  # state-carryover eval will be unavailable
# Direct kernel access for proper chunked state passing — InferenceParams
# routes through step() one token at a time and silently no-ops state
# carryover on the chunked path. mamba_chunk_scan_combined natively supports
# initial_states / return_final_states.
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    _HAS_CHUNK_SCAN = True
except Exception:
    mamba_chunk_scan_combined = None
    _HAS_CHUNK_SCAN = False
try:
    from einops import rearrange
except Exception:
    rearrange = None
# Brotli for compressed artifact (preferred over zstd).
try:
    import brotli  # type: ignore
    _HAS_BROTLI = True
except ImportError:
    _HAS_BROTLI = False

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

# torch._dynamo specializes on each block instance reference inside the
# forward loop (10 distinct block objects → 10 recompiles for the dispatch
# wrapper). Default cache_size_limit is 8, so we hit the warning and dynamo
# falls back to eager for that frame. Bump it well above n_blocks.
try:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 256
except Exception:
    pass

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DATA_PATH = "./data/datasets/fineweb10B_sp8192"
TOK_PATH = "./data/tokenizers/fineweb_8192_bpe.model"
VOCAB_SIZE = 8192
SEQ_LEN = int(os.environ.get("SEQ_LEN", "8192"))
TRAIN_SHARDS = int(os.environ.get("TRAIN_SHARDS", "80"))
DOWNLOAD_DATA = os.environ.get("DOWNLOAD_DATA", "0") == "1"
USE_ASYNC_LOADER = os.environ.get("USE_ASYNC_LOADER", "1") == "1"
# sp8192 lives in a fork — set MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf
os.environ.setdefault("MATCHED_FINEWEB_REPO_ID", "kevclark/parameter-golf")

# SSM architecture
D_MODEL = 512
D_STATE = int(os.environ.get("D_STATE", "64"))  # tested 128 — added ~17% throughput cost without recovering it in BPB at 10min training budget. 64 wins empirically here.
N_UNIQUE_BLOCKS = int(os.environ.get("N_UNIQUE_BLOCKS", "10"))
N_UNROLLS = int(os.environ.get("N_UNROLLS", "1"))
CONV_KERNEL = 4
# Mamba2-specific knobs (passed to mamba_ssm.modules.mamba2.Mamba2)
HEADDIM = int(os.environ.get("HEADDIM", "64"))   # tested 32 — slower kernel path at this size, didn't pay back in BPB
DT_MIN = float(os.environ.get("DT_MIN", "0.0005"))  # sharper than Mamba2 default 0.001
DT_MAX = float(os.environ.get("DT_MAX", "0.05"))    # sharper than Mamba2 default 0.1
FFN_MULT = int(os.environ.get("FFN_MULT", "2"))  # was 3 — FFN dominates the param budget; cutting to 2 saves ~8M params (~6MB compressed) at modest BPB cost
FFN_EVERY = int(os.environ.get("FFN_EVERY", "1"))  # Run FFN every Nth layer/unroll
FFN_FREQ_MODE = os.environ.get("FFN_FREQ_MODE", "layer").lower()  # "layer" or "unroll"
MEMORY_DIM = 256
LOGIT_SOFTCAP = 30.0
EMBED_INIT_STD = 0.005
USE_MEMORY = os.environ.get("USE_MEMORY", "0") == "1"
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "1") == "1"

# Document boundary handling for SSM training. SSMs accumulate state across
# the entire sequence, so when multiple short docs are packed into a single
# 8192-token training row, doc B's state is contaminated by doc A. Mamba2's
# chunk-scan kernel accepts a `seq_idx` (B, T) int32 that zeros state
# transitions at chunk boundaries where seq_idx changes. Reset granularity
# is the chunk_size of the scan (256 by default), so contamination is
# bounded to <=256 tokens after each boundary instead of unbounded.
SEQ_IDX_ENABLED = os.environ.get("SEQ_IDX_ENABLED", "1") == "1"
DOC_BOUNDARY_TOKEN = int(os.environ.get("DOC_BOUNDARY_TOKEN", "1"))  # this tokenizer has bos=1 (<s>), pad=0; <s> appears every ~811 tokens in val data → real doc separator

# Factored embedding: tok_emb at EMBED_DIM, project to D_MODEL.
# Saves params at vocab=8192. SOTA uses similar factoring (kev's 8192 setup).
EMBED_DIM = int(os.environ.get("EMBED_DIM", "256"))  # genuine factoring: tok_emb is 8192x256, embed_proj 256->512. Saves ~2.1M raw on tok_emb, costs ~131K on embed_proj. Net ~1.5MB compressed savings vs EMBED_DIM=512 (which is no-op factoring).

# Untied output head + neural copy/pointer head.
# Goal: improve token-output expressivity and exact/contextual token reuse.
UNTIE_LM_HEAD = os.environ.get("UNTIE_LM_HEAD", "0") == "1"  # default tied: saves 4.2M params at vocab=8192. Untied costs ~3MB compressed for ~0.005 BPB; not worth it under 16MB cap.
LM_HEAD_BIAS = os.environ.get("LM_HEAD_BIAS", "1") == "1"
COPY_HEAD_ENABLED = os.environ.get("COPY_HEAD_ENABLED", "0") == "1"
COPY_DIM = int(os.environ.get("COPY_DIM", "64"))
COPY_GATE_BIAS_INIT = float(os.environ.get("COPY_GATE_BIAS_INIT", "-2.0"))
COPY_SCALE_INIT = float(os.environ.get("COPY_SCALE_INIT", "1.0"))
COPY_USE_QK_NORM = os.environ.get("COPY_USE_QK_NORM", "1") == "1"

# Hybrid attention config
# This file's new default is mode D: two hard RoPE attention teachers at 2,5,
# followed by cheap context bridges into later SSM layers.
ATTN_LAYER_IDXS = [int(x) for x in os.environ.get("ATTN_LAYER_IDXS", "2,5").split(",") if x.strip()]
ATTN_N_HEADS = int(os.environ.get("ATTN_N_HEADS", "8"))
QK_GAIN_INIT = float(os.environ.get("QK_GAIN_INIT", "5.25"))

# Attention/SSM fusion experiments.
# FUSION_MODE:
#   NONE: original hard SSM/attention block replacement
#   A: attention-corrected SSM block. FUSION_LAYERS stay SSM blocks and get a gated attention residual after Mamba.
#   B: attention-before-Mamba injection. FUSION_LAYERS stay SSM blocks and get a small attention residual before Mamba's input projection.
#   C: SSM/current-hidden-conditioned Q/K gain inside regular attention blocks.
#   D: RoPE-Bridge-C: existing RoPE attention layers teach context; later SSM layers carry it cheaply.
FUSION_MODE = os.environ.get("FUSION_MODE", "D").strip().upper()
if FUSION_MODE == "":
    FUSION_MODE = "D"
if FUSION_MODE not in ("NONE", "A", "B", "C", "D"):
    raise ValueError(f"FUSION_MODE must be one of NONE,A,B,C,D; got {FUSION_MODE!r}")

def _parse_int_list_env(name: str, default: str):
    raw = os.environ.get(name, default)
    return [int(x) for x in raw.split(",") if x.strip()]

def _parse_layer_value_map(raw: str, cast=float):
    out = {}
    if not raw:
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        k, v = item.split(":", 1)
        out[int(k.strip())] = cast(v.strip())
    return out

def _layer_value(layer_idx, mapping, default):
    if layer_idx is None:
        return default
    return mapping.get(int(layer_idx), default)

_default_fusion_layers = "2,5" if FUSION_MODE == "A" else ("2" if FUSION_MODE == "B" else "")
FUSION_LAYERS = _parse_int_list_env("FUSION_LAYERS", _default_fusion_layers)
FUSION_ATTN_HEADS = int(os.environ.get("FUSION_ATTN_HEADS", "4"))
FUSION_ATTENTION_GATE_INIT = float(os.environ.get("FUSION_ATTENTION_GATE_INIT", "-2.5"))
FUSION_PRE_GATE_INIT = float(os.environ.get("FUSION_PRE_GATE_INIT", "0.0"))
FUSION_PRE_SCALE = float(os.environ.get("FUSION_PRE_SCALE", "0.1"))
FUSION_QK_DELTA_SCALE = float(os.environ.get("FUSION_QK_DELTA_SCALE", "0.1"))

# RoPE-Bridge mode D: no extra full attention modules. Existing attention
# blocks produce RoPE-aware teacher context; later SSM blocks receive it through
# a tiny low-rank bridge before Mamba's input projection.
CONTEXT_BRIDGE_ENABLED = os.environ.get("CONTEXT_BRIDGE_ENABLED", "1" if FUSION_MODE == "D" else "0") == "1"
CONTEXT_BRIDGE_LAYERS = _parse_int_list_env("CONTEXT_BRIDGE_LAYERS", "3,4,6,7" if FUSION_MODE == "D" else "")
CONTEXT_BRIDGE_DIM = int(os.environ.get("CONTEXT_BRIDGE_DIM", "64"))
CONTEXT_BRIDGE_GATE_INIT = float(os.environ.get("CONTEXT_BRIDGE_GATE_INIT", "-2.0"))
CONTEXT_BRIDGE_CTX_DECAY = float(os.environ.get("CONTEXT_BRIDGE_CTX_DECAY", "0.70"))
SKIP_INIT = float(os.environ.get("SKIP_INIT", "0.5" if FUSION_MODE == "D" else "1.0"))

# Partial RoPE in the attention checkpoint.
# Useful for 8192-context tests: lets a subset of each attention head encode relative position
# while leaving the remaining dimensions content-only.
ROPE_ENABLED = os.environ.get("ROPE_ENABLED", "1") == "1"
ROPE_DIM = int(os.environ.get("ROPE_DIM", "16"))
ROPE_BASE = float(os.environ.get("ROPE_BASE", "10000.0"))
ROPE_LAYER_DIM_MAP = _parse_layer_value_map(os.environ.get("ROPE_LAYER_DIMS", "2:16,5:32" if FUSION_MODE == "D" else ""), cast=int)
ROPE_LAYER_BASE_MAP = _parse_layer_value_map(os.environ.get("ROPE_LAYER_BASES", "2:10000,5:50000" if FUSION_MODE == "D" else ""), cast=float)
Q_GAIN_SPLIT = os.environ.get("Q_GAIN_SPLIT", "1" if FUSION_MODE == "D" else "0") == "1"
Q_GAIN_ROPE_INIT = float(os.environ.get("Q_GAIN_ROPE_INIT", str(QK_GAIN_INIT)))
Q_GAIN_CONTENT_INIT = float(os.environ.get("Q_GAIN_CONTENT_INIT", str(QK_GAIN_INIT)))

# 2048 -> 8192 length curriculum.
# Keep SEQ_LEN=8192 for final/eval shape, but early training batches use shorter
# 2048 sequences with the same TOK_PER_RANK token budget. This attempts to combine
# faster/more diverse 2048 optimization with late 8192 context learning.
LENGTH_CURRICULUM_ENABLED = os.environ.get("LENGTH_CURRICULUM_ENABLED", "0") == "1"
CURRICULUM_SHORT_SEQ_LEN = int(os.environ.get("CURRICULUM_SHORT_SEQ_LEN", "2048"))
CURRICULUM_SWITCH_FRAC = float(os.environ.get("CURRICULUM_SWITCH_FRAC", "0.60"))


# Activation choice
ACTIVATION = os.environ.get("ACTIVATION", "swiglu").lower()  # conservative default: your known-good SwiGLU; try leaky_relu2 only as an ablation

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
LR_MIN_SCALE = float(os.environ.get("LR_MIN_SCALE", "0.0"))  # conservative default: restore baseline schedule; try 0.10 only as an ablation
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "cosine_late")
LR_WARMDOWN_START_FRAC = float(os.environ.get("LR_WARMDOWN_START_FRAC", "0.20"))  # 80% warmdown (was 0.28 = 72%); EMA/SWA do not work for this SSM, so the schedule itself has to land precisely

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
BIGRAM_ENABLED = os.environ.get("BIGRAM_ENABLED", "0") == "1"  # off by default: saves ~1MB compressed (1.4M params), BPB cost ~0.002-0.005. Bigram hash gave marginal lift; not worth the bytes given the 16MB cap.
BIGRAM_BUCKETS = int(os.environ.get("BIGRAM_BUCKETS", "10240"))
BIGRAM_DIM = int(os.environ.get("BIGRAM_DIM", "128"))

# Sliding window eval — stride < SEQ_LEN gives each token more context
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", "64"))  # 64 matches SOTA

# State carryover at eval — SSM-native: thread Mamba2 hidden state across
# non-overlapping seq_len blocks of the val set so every scored token has the
# entire validation prefix in its recurrent state (transformers can't do this).
# Implementation: mamba_ssm InferenceParams cache, seqlen_offset held at 0 so
# the chunked scan path is used with carried initial_states. Attention layers
# (which can't carryover) see only their own block, matching training context.
CARRYOVER_EVAL_ENABLED = os.environ.get("CARRYOVER_EVAL_ENABLED", "0") == "1"  # disabled by default — model not trained for stateful init, has been net-negative across 12+ runs. Re-enable with stateful-overlap to test.
CARRYOVER_BLOCK_LEN = int(os.environ.get("CARRYOVER_BLOCK_LEN", str(SEQ_LEN)))
CARRYOVER_OVERLAP = int(os.environ.get("CARRYOVER_OVERLAP", "1024"))  # stateful-overlap: each block = overlap + score_region. Mamba state carries; attention sees overlap+score_region; only score_region scored. PR #1644 reports this matches sliding within 0.3 mBPB.
# Warmup tokens at the start of the val stream where Mamba state is still
# "cold" (consumed too few tokens to be informative). These are processed to
# advance state but their losses are not counted. Default to one full block
# so the cold-start block is always excluded. Set to 0 to score everything.
CARRYOVER_WARMUP_TOKENS = int(os.environ.get("CARRYOVER_WARMUP_TOKENS", "0"))
# Direct chunk-scan path (Task 3): bypasses Mamba2.forward and calls the
# triton kernel mamba_chunk_scan_combined with explicit initial_states /
# return_final_states. This is the only path that actually carries SSM state
# across non-overlapping val blocks at chunked-scan throughput. Set to "0"
# to fall back to the (broken) inference_params version for diagnosis.
CARRYOVER_DIRECT_KERNEL = os.environ.get("CARRYOVER_DIRECT_KERNEL", "1") == "1"

# Compression / quantization config — int6 GPTQ-lite + Brotli-11 stack.
# Goal: <16,000,000-byte artifact. Naive int8+zstd lands ~35MB; int6 packed
# + Brotli-11 + tied LM head should land ~13-15MB.
QUANT_MODE = os.environ.get("QUANT_MODE", "int6_packed").lower()  # "int6_packed", "int8" (legacy)
QUANT_K_MATRIX = float(os.environ.get("QUANT_K_MATRIX", "12.85"))  # SDClip k for matrix tensors (int6); SOTA value from PR #1394
QUANT_K_EMBED = float(os.environ.get("QUANT_K_EMBED", "20.0"))    # SDClip k for embedding-like tensors (int8); higher k = less aggressive clipping
QUANT_PASSTHROUGH_NUMEL = int(os.environ.get("QUANT_PASSTHROUGH_NUMEL", "0"))  # quantize all 2D float tensors by default; keeps 1D/3D fp16 and helps stay under 16MB
QUANT_PROTECT_DYNAMICS = os.environ.get("QUANT_PROTECT_DYNAMICS", "1") == "1"  # PR #1890 / Q-Mamba: promote dt rows of mamba.in_proj.weight to INT8 to protect SSM recurrence from quant noise. ~16 extra bytes/row, big post-quant BPB recovery.
QUANT_OPTCLIP_EMBED = os.environ.get("QUANT_OPTCLIP_EMBED", "1") == "1"  # GPTQ-lite for embeddings: per-row optimal clip search instead of single global k. Targets rare-token rows whose distribution differs from common rows. ~0.005-0.010 BPB recovery on post-quant.

# Profiling
PROFILE_ENABLED = os.environ.get("PROFILE", "0") == "1"
PROFILE_START = int(os.environ.get("PROFILE_START", "3"))
PROFILE_END = int(os.environ.get("PROFILE_END", "8"))
PROFILE_DIR = os.environ.get("PROFILE_DIR", "./profile_traces")

# Optimizer — higher LRs for 42M model, matching SOTA WD
MATRIX_LR = float(os.environ.get("MATRIX_LR", "0.022"))  # SOTA: 0.022
SCALAR_LR = float(os.environ.get("SCALAR_LR", "0.02"))   # SOTA: 0.02
EMBED_LR = float(os.environ.get("EMBED_LR", "0.03"))     # SOTA: tied_embed_lr=0.03
BETA1 = float(os.environ.get("BETA1", "0.9"))
BETA2 = float(os.environ.get("BETA2", "0.95"))  # restore baseline optimizer default; use BETA2=0.99 only as an ablation
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
SCORE_FIRST_TTT_ENABLED = os.environ.get("SCORE_FIRST_TTT_ENABLED", "0") == "1"  # default OFF for architecture sweeps
SCORE_FIRST_TTT_CHUNK_TOKENS = int(os.environ.get("SCORE_FIRST_TTT_CHUNK_TOKENS", "32768"))
SCORE_FIRST_TTT_LR = float(os.environ.get("SCORE_FIRST_TTT_LR", "0.003"))
SCORE_FIRST_TTT_MOMENTUM = float(os.environ.get("SCORE_FIRST_TTT_MOMENTUM", "0.9"))
SCORE_FIRST_TTT_EPOCHS = int(os.environ.get("SCORE_FIRST_TTT_EPOCHS", "1"))
SCORE_FIRST_TTT_GRAD_CLIP = float(os.environ.get("SCORE_FIRST_TTT_GRAD_CLIP", "1.0"))
# LoRA-scoped TTT params (legal score-first TTT per reviewer / PR #1797)
SCORE_FIRST_TTT_USE_LORA = os.environ.get("SCORE_FIRST_TTT_USE_LORA", "1") == "1"  # adapt only LoRA params instead of all model params. Required for legal score-first TTT (otherwise the adaptation is too unstable and was the root cause of the previous all-params getting-stuck bug).
SCORE_FIRST_TTT_LORA_RANK = int(os.environ.get("SCORE_FIRST_TTT_LORA_RANK", "32"))  # conservative default for eval-time cost; try 64 only after smoke testing
SCORE_FIRST_TTT_LORA_ALPHA = float(os.environ.get("SCORE_FIRST_TTT_LORA_ALPHA", "64.0"))  # alpha/rank scaling
SCORE_FIRST_TTT_OPT = os.environ.get("SCORE_FIRST_TTT_OPT", "adamw").lower()  # "adamw" or "sgd"
SCORE_FIRST_TTT_BETA2 = float(os.environ.get("SCORE_FIRST_TTT_BETA2", "0.99"))  # AdamW beta2 (PR #1797 uses 0.99)
SCORE_FIRST_TTT_WD = float(os.environ.get("SCORE_FIRST_TTT_WD", "0.1"))  # AdamW weight decay
SCORE_FIRST_TTT_NO_ALLREDUCE = os.environ.get("SCORE_FIRST_TTT_NO_ALLREDUCE", "1") == "1"  # CRITICAL: ranks process disjoint val slices, so all-reducing TTT grads leaks future tokens between ranks. Default to rank-local TTT.
SCORE_FIRST_TTT_TARGETS = tuple(x.strip() for x in os.environ.get("SCORE_FIRST_TTT_TARGETS", "qkv,out_proj,gate_up,down").split(",") if x.strip())
SCORE_FIRST_TTT_WRAP_MAMBA = os.environ.get("SCORE_FIRST_TTT_WRAP_MAMBA", "0") == "1"  # default false: mamba_ssm.forward reads .weight directly and naive LoRA wrappers crash/deopt
RUN_PREQUANT_SCORE_FIRST_TTT = os.environ.get("RUN_PREQUANT_SCORE_FIRST_TTT", "0") == "1"  # default OFF for architecture sweeps
RUN_POSTQUANT = os.environ.get("RUN_POSTQUANT", "0") == "1"  # eval-only sweep default: skip quant/post-quant/artifact
MUON_MOMENTUM = float(os.environ.get("MUON_MOMENTUM", "0.99"))
MUON_MOMENTUM_WARMUP_START = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", "0.85"))
MUON_MOMENTUM_WARMUP_STEPS = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", "500"))
MUON_BACKEND_STEPS = int(os.environ.get("MUON_BACKEND_STEPS", "5"))
MUON_NESTEROV = os.environ.get("MUON_NESTEROV", "1") == "1"
MUON_ONLY_2D = os.environ.get("MUON_ONLY_2D", "1") == "1"

# TaskMuon: component-aware Muon for this SSM/attention hybrid.
# Set TASKMUON_ENABLED=0 to fall back to the original one-bucket Muon.
TASKMUON_ENABLED = os.environ.get("TASKMUON_ENABLED", "1") == "1"
TASKMUON_ROW_BALANCE = os.environ.get("TASKMUON_ROW_BALANCE", "1" if FUSION_MODE == "D" else "0") == "1"
TASKMUON_TRUST_CLIP = os.environ.get("TASKMUON_TRUST_CLIP", "1" if FUSION_MODE == "D" else "0") == "1"
TASKMUON_TRUST_MULT = float(os.environ.get("TASKMUON_TRUST_MULT", "8.0"))
TASKMUON_TRUST_FLOOR = float(os.environ.get("TASKMUON_TRUST_FLOOR", "1e-3"))
TASKMUON_QKV_LR_MULT = float(os.environ.get("TASKMUON_QKV_LR_MULT", "1.00"))
TASKMUON_QK_ROPE_LR_MULT = float(os.environ.get("TASKMUON_QK_ROPE_LR_MULT", "0.55"))
TASKMUON_QK_CONTENT_LR_MULT = float(os.environ.get("TASKMUON_QK_CONTENT_LR_MULT", "1.00"))
TASKMUON_V_LR_MULT = float(os.environ.get("TASKMUON_V_LR_MULT", "0.90"))
TASKMUON_ATTN_OUT_LR_MULT = float(os.environ.get("TASKMUON_ATTN_OUT_LR_MULT", "0.75"))
TASKMUON_MAMBA_ZX_LR_MULT = float(os.environ.get("TASKMUON_MAMBA_ZX_LR_MULT", "0.75"))
TASKMUON_MAMBA_BC_LR_MULT = float(os.environ.get("TASKMUON_MAMBA_BC_LR_MULT", "0.45"))
TASKMUON_MAMBA_DT_LR_MULT = float(os.environ.get("TASKMUON_MAMBA_DT_LR_MULT", "0.25"))
TASKMUON_MAMBA_OUT_LR_MULT = float(os.environ.get("TASKMUON_MAMBA_OUT_LR_MULT", "0.65"))
TASKMUON_FFN_LR_MULT = float(os.environ.get("TASKMUON_FFN_LR_MULT", "1.00"))
TASKMUON_FFN_GATE_LR_MULT = float(os.environ.get("TASKMUON_FFN_GATE_LR_MULT", "1.00"))
TASKMUON_FFN_UP_LR_MULT = float(os.environ.get("TASKMUON_FFN_UP_LR_MULT", "1.00"))
TASKMUON_FFN_DOWN_LR_MULT = float(os.environ.get("TASKMUON_FFN_DOWN_LR_MULT", "0.75"))
TASKMUON_BRIDGE_LR_MULT = float(os.environ.get("TASKMUON_BRIDGE_LR_MULT", "0.75"))
TASKMUON_GENERIC_LR_MULT = float(os.environ.get("TASKMUON_GENERIC_LR_MULT", "1.00"))
TASKMUON_QKV_SHAPE_CAP = float(os.environ.get("TASKMUON_QKV_SHAPE_CAP", "1.75"))
TASKMUON_QK_ROPE_SHAPE_CAP = float(os.environ.get("TASKMUON_QK_ROPE_SHAPE_CAP", "1.00"))
TASKMUON_MAMBA_IN_SHAPE_CAP = float(os.environ.get("TASKMUON_MAMBA_IN_SHAPE_CAP", "1.25"))
TASKMUON_MAMBA_BC_SHAPE_CAP = float(os.environ.get("TASKMUON_MAMBA_BC_SHAPE_CAP", "1.00"))
TASKMUON_MAMBA_OUT_SHAPE_CAP = float(os.environ.get("TASKMUON_MAMBA_OUT_SHAPE_CAP", "1.00"))
TASKMUON_FFN_SHAPE_CAP = float(os.environ.get("TASKMUON_FFN_SHAPE_CAP", "0.0"))  # 0 = no cap
TASKMUON_ATTN_WD = float(os.environ.get("TASKMUON_ATTN_WD", str(WEIGHT_DECAY)))
TASKMUON_QK_ROPE_WD = float(os.environ.get("TASKMUON_QK_ROPE_WD", "0.0"))
TASKMUON_QK_CONTENT_WD = float(os.environ.get("TASKMUON_QK_CONTENT_WD", str(WEIGHT_DECAY)))
TASKMUON_V_WD = float(os.environ.get("TASKMUON_V_WD", str(WEIGHT_DECAY)))
TASKMUON_MAMBA_ZX_WD = float(os.environ.get("TASKMUON_MAMBA_ZX_WD", "0.06"))
TASKMUON_MAMBA_BC_WD = float(os.environ.get("TASKMUON_MAMBA_BC_WD", "0.02"))
TASKMUON_MAMBA_DT_WD = float(os.environ.get("TASKMUON_MAMBA_DT_WD", "0.0"))
TASKMUON_MAMBA_OUT_WD = float(os.environ.get("TASKMUON_MAMBA_OUT_WD", "0.04"))
TASKMUON_FFN_WD = float(os.environ.get("TASKMUON_FFN_WD", "0.12"))
TASKMUON_FFN_DOWN_WD = float(os.environ.get("TASKMUON_FFN_DOWN_WD", "0.08"))
TASKMUON_BRIDGE_WD = float(os.environ.get("TASKMUON_BRIDGE_WD", "0.05"))
TASKMUON_GENERIC_WD = float(os.environ.get("TASKMUON_GENERIC_WD", str(WEIGHT_DECAY)))

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
    "fusion_gate",
    "fusion_pre_gate",
    "q_gain_delta",
    "q_gain_rope",
    "q_gain_content",
    "ctx_gate",
)

# -----------------------------------------------------------------------------
# Optional single-file ABC launcher
# -----------------------------------------------------------------------------
def _maybe_launch_abc_sweep():
    """When run as `python this_file.py`, launch A -> B -> C under torchrun.

    Child torchrun workers have RANK/WORLD_SIZE set, so they skip this launcher
    and execute the normal distributed training path below.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return
    if os.environ.get("RUN_ABC_SWEEP", "1") != "1":
        return

    modes = [m.strip().upper() for m in os.environ.get("ABC_MODES", "D").split(",") if m.strip()]
    bad = [m for m in modes if m not in ("A", "B", "C", "D")]
    if bad:
        raise ValueError(f"ABC_MODES may only contain A,B,C,D; got {bad}")

    script = Path(__file__).resolve()
    nproc = int(os.environ.get("ABC_NPROC", os.environ.get("NPROC_PER_NODE", "8")))
    log_dir = Path(os.environ.get("ABC_LOG_DIR", "./fusion_abc_runs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Conservative defaults that can still be overridden by the caller's env.
    common = {
        "ABC_SWEEP_CHILD": "1",
        "SCORE_FIRST_TTT_ENABLED": "0",       # architecture sweep: TTT off by default
        "RUN_PREQUANT_SCORE_FIRST_TTT": "0",
        "RUN_POSTQUANT": "0",                 # skip quant/post-quant/artifact during architecture sweeps
        "SCORE_FIRST_TTT_WRAP_MAMBA": "0",
        "SCORE_FIRST_TTT_TARGETS": "qkv,out_proj,gate_up,down",
        "QUANT_PASSTHROUGH_NUMEL": "0",
        "STOP_CHECK_EVERY": "10",
        "TASKMUON_ENABLED": "1",            # use component-aware optimizer for this sweep
        "ARTIFACT_DIR": str(log_dir),
    }

    # Per-mode architecture defaults. These are only defaults: any value already
    # set in the shell wins, except FUSION_MODE/RUN_TAG which are mode-specific.
    per_mode = {
        # A: SSM blocks at layers 2,5 with attention correction after Mamba.
        "A": {
            "FUSION_MODE": "A",
            "FUSION_LAYERS": "2,5",
            "ATTN_LAYER_IDXS": "",
            "N_UNIQUE_BLOCKS": "8",
            "FUSION_ATTN_HEADS": "4",
            "FUSION_ATTENTION_GATE_INIT": "-2.5",
        },
        # B: one early attention injection before Mamba's input projection.
        "B": {
            "FUSION_MODE": "B",
            "FUSION_LAYERS": "2",
            "ATTN_LAYER_IDXS": "",
            "N_UNIQUE_BLOCKS": "8",
            "FUSION_ATTN_HEADS": "4",
            "FUSION_PRE_SCALE": "0.1",
            "FUSION_PRE_GATE_INIT": "0.0",
        },
        # C: regular hard attention layers, but Q/K gain is conditioned on current hidden state.
        "C": {
            "FUSION_MODE": "C",
            "ATTN_LAYER_IDXS": "2,5",
            "N_UNIQUE_BLOCKS": "8",
            "FUSION_QK_DELTA_SCALE": "0.1",
        },
        # D: RoPE attention at 2,5 produces teacher context; SSM layers 3,4,6,7 carry it via low-rank bridges.
        "D": {
            "FUSION_MODE": "D",
            "ATTN_LAYER_IDXS": "2,5",
            "N_UNIQUE_BLOCKS": "8",
            "CONTEXT_BRIDGE_ENABLED": "1",
            "CONTEXT_BRIDGE_LAYERS": "3,4,6,7",
            "CONTEXT_BRIDGE_DIM": "64",
            "CONTEXT_BRIDGE_GATE_INIT": "-2.0",
            "CONTEXT_BRIDGE_CTX_DECAY": "0.70",
            "ROPE_LAYER_DIMS": "2:16,5:32",
            "ROPE_LAYER_BASES": "2:10000,5:50000",
            "Q_GAIN_SPLIT": "1",
            "SKIP_INIT": "0.5",
            "TASKMUON_ENABLED": "1",
            "TASKMUON_ROW_BALANCE": "1",
            "TASKMUON_TRUST_CLIP": "1",
        },
    }

    print(f"ABCD/RoPE-Bridge sweep launcher: modes={modes}, nproc={nproc}, log_dir={log_dir}", flush=True)
    for mode in modes:
        env = os.environ.copy()
        respect_common_env = os.environ.get("ABC_RESPECT_COMMON_ENV", "0") == "1"
        for k, v in common.items():
            if respect_common_env:
                env.setdefault(k, v)
            else:
                env[k] = v
        respect_mode_env = os.environ.get("ABC_RESPECT_MODE_ENV", "0") == "1"
        for k, v in per_mode[mode].items():
            # By default, mode-defining architecture values are forced so A/B/C
            # really mean A/B/C even if the shell still has old ablation envs set.
            # Set ABC_RESPECT_MODE_ENV=1 if you intentionally want shell values to win.
            if respect_mode_env and k != "FUSION_MODE":
                env.setdefault(k, v)
            else:
                env[k] = v
        env["RUN_TAG"] = env.get("RUN_TAG", f"fusion_{mode.lower()}")
        if os.environ.get("ABC_APPEND_MODE_TO_TAG", "1") == "1":
            base_tag = os.environ.get("RUN_TAG", "fusion")
            env["RUN_TAG"] = f"{base_tag}_{mode.lower()}"
        log_path = log_dir / f"{env['RUN_TAG']}.log"
        cmd = ["torchrun", "--standalone", "--nproc_per_node", str(nproc), str(script)]
        print("=" * 80, flush=True)
        print(f"Launching fusion {mode}: {' '.join(cmd)}", flush=True)
        print(f"  RUN_TAG={env['RUN_TAG']} FUSION_MODE={env['FUSION_MODE']} "
              f"ATTN_LAYER_IDXS={env.get('ATTN_LAYER_IDXS','')} "
              f"FUSION_LAYERS={env.get('FUSION_LAYERS','')} "
              f"N_UNIQUE_BLOCKS={env.get('N_UNIQUE_BLOCKS','')}", flush=True)
        print(f"  log: {log_path}", flush=True)
        with open(log_path, "w", buffering=1) as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
            rc = proc.wait()
        if rc != 0:
            print(f"Fusion mode {mode} failed with exit code {rc}. Stopping sweep.", flush=True)
            sys.exit(rc)
    print("ABCD/RoPE-Bridge sweep complete.", flush=True)
    sys.exit(0)


# -----------------------------------------------------------------------------
# Distributed setup (strict 8xGPU by default)
# -----------------------------------------------------------------------------
_maybe_launch_abc_sweep()

assert torch.cuda.is_available(), "CUDA is required."
DISTRIBUTED = "RANK" in os.environ and "WORLD_SIZE" in os.environ
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "8"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

if not DISTRIBUTED:
    raise RuntimeError(
        "This script requires torchrun distributed launch, or run it normally as "
        "`python ssm_fusion_abc_sweep.py` to launch A->B->C automatically."
    )
EXPECTED_WORLD_SIZE = int(os.environ.get("EXPECTED_WORLD_SIZE", "8"))
if WORLD_SIZE != EXPECTED_WORLD_SIZE:
    raise RuntimeError(f"Expected WORLD_SIZE={EXPECTED_WORLD_SIZE}, got WORLD_SIZE={WORLD_SIZE}")

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
log0(f"mamba2: headdim={HEADDIM}, nheads={(2*D_MODEL)//HEADDIM}, dt_min={DT_MIN}, dt_max={DT_MAX}")
log0(f"seq_idx: enabled={SEQ_IDX_ENABLED}, doc_boundary_token={DOC_BOUNDARY_TOKEN} (-1 to disable)")
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
log0(f"fusion: mode={FUSION_MODE}, layers={FUSION_LAYERS}, fusion_heads={FUSION_ATTN_HEADS}, gate_init={FUSION_ATTENTION_GATE_INIT}, pre_scale={FUSION_PRE_SCALE}, qk_delta_scale={FUSION_QK_DELTA_SCALE}")
log0(f"rope_bridge: enabled={CONTEXT_BRIDGE_ENABLED}, bridge_layers={CONTEXT_BRIDGE_LAYERS}, bridge_dim={CONTEXT_BRIDGE_DIM}, gate_init={CONTEXT_BRIDGE_GATE_INIT}, ctx_decay={CONTEXT_BRIDGE_CTX_DECAY}, skip_init={SKIP_INIT}")
log0(f"rope: enabled={ROPE_ENABLED}, dim={ROPE_DIM}, base={ROPE_BASE}, layer_dims={ROPE_LAYER_DIM_MAP}, layer_bases={ROPE_LAYER_BASE_MAP}, q_gain_split={Q_GAIN_SPLIT}")
log0(
    f"length_curriculum: enabled={LENGTH_CURRICULUM_ENABLED}, "
    f"short_seq={CURRICULUM_SHORT_SEQ_LEN}, switch_frac={CURRICULUM_SWITCH_FRAC}, final_seq={SEQ_LEN}"
)
log0("current_defaults: RoPE-Bridge mode D, ATTN_LAYER_IDXS=2,5, context bridges 3,4,6,7, TaskMuon/RopeTaskMuon on, TTT/postquant off")
log0(f"activation: {ACTIVATION}")
log0(f"depth_recur: loop=[{LOOP_START}..{LOOP_END}] activate@{LOOP_ACTIVATE_FRAC}")
log0(f"score_first_ttt: enabled={SCORE_FIRST_TTT_ENABLED}, prequant={RUN_PREQUANT_SCORE_FIRST_TTT}, chunk={SCORE_FIRST_TTT_CHUNK_TOKENS}, lr={SCORE_FIRST_TTT_LR}, epochs={SCORE_FIRST_TTT_EPOCHS}, targets={SCORE_FIRST_TTT_TARGETS}")
log0(f"postquant: run_postquant={RUN_POSTQUANT} (0 = skip quant/post-quant/artifact for sweeps)")
log0(f"bigram: enabled={BIGRAM_ENABLED}, buckets={BIGRAM_BUCKETS}, dim={BIGRAM_DIM}")
log0(f"swa: enabled={SWA_ENABLED}, start_frac={SWA_START_FRAC}, every={SWA_EVERY}")
log0(f"eval_stride={EVAL_STRIDE}")
log0(f"carryover_eval: enabled={CARRYOVER_EVAL_ENABLED}, block_len={CARRYOVER_BLOCK_LEN}, warmup_tokens={CARRYOVER_WARMUP_TOKENS}, direct_kernel={CARRYOVER_DIRECT_KERNEL}")
log0(f"quant: mode={QUANT_MODE}, k_matrix={QUANT_K_MATRIX}, k_embed={QUANT_K_EMBED}, passthrough_numel<={QUANT_PASSTHROUGH_NUMEL}, brotli={_HAS_BROTLI}")
log0(f"lr_schedule: {LR_SCHEDULE}, warmdown_start={LR_WARMDOWN_START_FRAC} ({(1.0-LR_WARMDOWN_START_FRAC)*100:.0f}% warmdown)")
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
    f"hybrid_optimizer: taskmuon={TASKMUON_ENABLED}, muon_momentum={MUON_MOMENTUM}, "
    f"muon_momentum_warmup={MUON_MOMENTUM_WARMUP_START}->{MUON_MOMENTUM} over {MUON_MOMENTUM_WARMUP_STEPS} steps, "
    f"muon_backend_steps={MUON_BACKEND_STEPS}, muon_nesterov={MUON_NESTEROV}, "
    f"taskmuon_row_balance={TASKMUON_ROW_BALANCE}, taskmuon_trust_clip={TASKMUON_TRUST_CLIP}"
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



def _apply_fusion_rope(q: torch.Tensor, k: torch.Tensor, rotary_dim: int, base: float):
    """RoPE helper for the small fusion attention modules, defined early so
    SelectiveSSMBlock can use attention before the regular CausalAttentionBlock
    class is declared.
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
    freqs = torch.outer(pos, inv_freq)
    cos = freqs.cos()[None, None, :, :]
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


class CausalFusionAttention(nn.Module):
    """Small causal attention core used to fuse attention into SSM blocks.

    It intentionally has no FFN and returns only a residual correction. The
    out_proj is zero-initialized, so variants A/B begin as the base SSM model
    and learn the fusion path only if useful.
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by fusion n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.q_gain = nn.Parameter(torch.full((n_heads,), QK_GAIN_INIT))

    def forward(self, x):
        B, T, D = x.shape
        normed = self.attn_norm(x)
        qkv = self.qkv(normed).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        q, k = _apply_fusion_rope(q, k, ROPE_DIM, ROPE_BASE)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn_out)

class ContextBridge(nn.Module):
    """Cheap RoPE-context bridge from attention teacher output into later SSM inputs.

    This deliberately does not add another attention module. It takes the
    RoPE-aware attention context produced by existing attention layers and
    injects a low-rank, gated residual before Mamba's in_proj so the SSM can
    store/carry that contextual signal.
    """
    def __init__(self, d_model, rank=64, gate_init=-2.0):
        super().__init__()
        rank = max(1, int(rank))
        self.ctx_norm = RMSNorm(d_model)
        self.ctx_down = nn.Linear(d_model, rank, bias=False)
        self.ctx_up = nn.Linear(rank, d_model, bias=False)
        self.ctx_gate = nn.Parameter(torch.full((d_model,), float(gate_init)))
        nn.init.normal_(self.ctx_down.weight, std=0.01)
        nn.init.zeros_(self.ctx_up.weight)

    def forward(self, x, ctx):
        if ctx is None:
            return x
        # ctx has the same (B,T,D) shape as x. Cast only at the end to keep the
        # low-rank bridge numerically calm under autocast.
        h = self.ctx_down(self.ctx_norm(ctx))
        h = F.silu(h)
        delta = self.ctx_up(h)
        gate = torch.sigmoid(self.ctx_gate).to(dtype=x.dtype, device=x.device)
        return x + gate[None, None, :] * delta.to(dtype=x.dtype)


class SelectiveSSMBlock(nn.Module):
    def __init__(self, d_model, d_state, memory_dim, conv_kernel=4, ffn_mult=3, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=conv_kernel,
            expand=2,
            headdim=HEADDIM,
            dt_min=DT_MIN,
            dt_max=DT_MAX,
            layer_idx=layer_idx,  # required for inference_params cache (state-carryover eval)
        )
        self.ssm_norm = RMSNorm(d_model)
        self.ssm_scale = nn.Parameter(torch.ones(d_model))
        self.ffn = SwiGLU_FFN(d_model, ffn_mult)
        self.ffn_norm = RMSNorm(d_model)
        self.mlp_scale = nn.Parameter(torch.ones(d_model))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(d_model), torch.zeros(d_model)]))

        # Fusion variants A/B keep this as an SSM block but add a small attention path.
        self.fusion_mode = FUSION_MODE
        self.fusion_enabled = (layer_idx in set(FUSION_LAYERS)) and (FUSION_MODE in ("A", "B"))
        if self.fusion_enabled:
            self.fusion_attn = CausalFusionAttention(d_model, FUSION_ATTN_HEADS)
            if FUSION_MODE == "A":
                self.fusion_gate = nn.Parameter(torch.full((d_model,), FUSION_ATTENTION_GATE_INIT))
                self.fusion_pre_gate = None
            else:
                self.fusion_gate = None
                self.fusion_pre_gate = nn.Parameter(torch.tensor(float(FUSION_PRE_GATE_INIT)))
        else:
            self.fusion_attn = None
            self.fusion_gate = None
            self.fusion_pre_gate = None

        self.context_bridge = (
            ContextBridge(d_model, CONTEXT_BRIDGE_DIM, CONTEXT_BRIDGE_GATE_INIT)
            if CONTEXT_BRIDGE_ENABLED and (layer_idx in set(CONTEXT_BRIDGE_LAYERS))
            else None
        )

        # Memory params (only used if USE_MEMORY)
        if USE_MEMORY:
            self.mem_to_model = nn.Linear(memory_dim, d_model, bias=False)
            self.mem_in_gate = nn.Parameter(torch.tensor(0.0))
            self.memory_proj = nn.Linear(d_model, memory_dim, bias=False)
            self.memory_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, x0, mem, run_ffn=True, inference_params=None, seq_idx=None, ctx=None):
        mix = self.resid_mix
        x_mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if USE_MEMORY and mem is not None:
            mem_ctx = self.mem_to_model(mem)[:, None, :]
            mem_mix = torch.sigmoid(self.mem_in_gate)
            x_in = x_mixed + mem_mix * mem_ctx
        else:
            x_in = x_mixed

        # Mode D: inject RoPE-aware teacher context from previous attention
        # blocks before Mamba's in_proj, so the recurrent stream can store it.
        if self.context_bridge is not None and ctx is not None:
            x_in = self.context_bridge(x_in, ctx)

        # Variant B: let a tiny causal attention path steer Mamba's input
        # projection, so attention can influence z/x/B/C/dt rather than only
        # adding a post-hoc residual.
        if self.fusion_enabled and self.fusion_mode == "B":
            a = self.fusion_attn(x_in)
            gate = torch.sigmoid(self.fusion_pre_gate).to(dtype=x_in.dtype, device=x_in.device)
            x_in = x_in + (FUSION_PRE_SCALE * gate) * a

        # When inference_params is given the Mamba2 chunked scan reads the
        # carried initial_states from the cache and writes back the final
        # state. We hold seqlen_offset at 0 so we never enter the step path.
        # seq_idx (B, T) int32 zeroes state transitions at chunk-aligned
        # document boundaries.
        kwargs = {}
        if inference_params is not None:
            kwargs["inference_params"] = inference_params
        if seq_idx is not None:
            kwargs["seq_idx"] = seq_idx
        y = self.mamba(self.ssm_norm(x_in), **kwargs)
        x = x_mixed + self.ssm_scale[None, None, :] * y

        # Variant A: attention-corrected SSM. The block remains recurrent, but
        # a gated attention residual fixes local/exact-token mistakes.
        if self.fusion_enabled and self.fusion_mode == "A":
            a = self.fusion_attn(x)
            gate = torch.sigmoid(self.fusion_gate).to(dtype=x.dtype, device=x.device)
            x = x + gate[None, None, :] * a

        if run_ffn:
            x = x + self.mlp_scale[None, None, :] * self.ffn(self.ffn_norm(x))
        if USE_MEMORY and mem is not None:
            new_mem = torch.tanh(self.memory_proj(x[:, -1, :]))
            g = torch.sigmoid(self.memory_gate)
            mem = g * mem + (1.0 - g) * new_mem
        return x, mem

    def _mamba_forward_with_state(self, u, initial_ssm_state, prev_conv_input, seq_idx=None):
        """
        Manual Mamba2 forward with explicit SSM-state carryover via the
        triton kernel. Bypasses Mamba2.forward (whose inference_params path
        goes through step() one token at a time, which silently no-ops the
        chunked-scan state).

        u: (B, L, d_model)
        initial_ssm_state: (B, nheads, headdim, d_state) or None
        prev_conv_input: (B, d_conv-1, conv_channels) or None.
            Last d_conv-1 inputs to conv1d from previous block; None on
            first block. With d_conv=4 this is 3 timesteps.
        seq_idx: (B, L) int32 or None — document boundaries within block.
            State transitions are zeroed at chunk-aligned positions where
            seq_idx changes.

        Returns: (y, final_ssm_state, last_conv_input)
        """
        m = self.mamba
        B, L, _ = u.shape

        # in_proj split — handle optional d_mlp path defensively
        zxbcdt = m.in_proj(u)  # (B, L, total)
        d_inner = getattr(m, "d_inner", getattr(m, "d_ssm", None))
        nheads = m.nheads
        ngroups = getattr(m, "ngroups", 1)
        d_state = m.d_state
        d_conv = m.d_conv
        headdim = m.headdim
        chunk_size = m.chunk_size
        conv_channels = d_inner + 2 * ngroups * d_state
        rmsnorm = getattr(m, "rmsnorm", False)

        full = zxbcdt.shape[-1]
        expected = 2 * d_inner + 2 * ngroups * d_state + nheads
        d_mlp = (full - expected) // 2

        if d_mlp > 0:
            z0, x0_mlp, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, d_inner, conv_channels, nheads],
                dim=-1,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt,
                [d_inner, conv_channels, nheads],
                dim=-1,
            )
            z0 = x0_mlp = None

        # Conv1d with state-passing. Mamba2's conv1d is depthwise (groups=C)
        # with kernel d_conv. To carry state across blocks we prepend the
        # previous block's last (d_conv-1) inputs and run with padding=0,
        # producing exactly L outputs.
        xBC_t = xBC.transpose(1, 2).contiguous()  # (B, C, L)
        if prev_conv_input is not None and d_conv > 1:
            prev_t = prev_conv_input.transpose(1, 2).contiguous()  # (B, C, d_conv-1)
            xBC_padded = torch.cat([prev_t, xBC_t], dim=-1)  # (B, C, L + d_conv - 1)
            xBC_conv = F.conv1d(
                xBC_padded,
                m.conv1d.weight,
                m.conv1d.bias,
                stride=1, padding=0, dilation=1,
                groups=conv_channels,
            )  # (B, C, L)
        else:
            # First block of stream — let the module's own conv handle causal
            # padding, then trim trailing positions.
            xBC_conv = m.conv1d(xBC_t)[..., :L]

        new_conv_input = xBC[:, -(d_conv - 1):, :].contiguous() if d_conv > 1 else None

        xBC_conv = F.silu(xBC_conv).transpose(1, 2)  # (B, L, C)

        x, Bm, Cm = torch.split(
            xBC_conv,
            [d_inner, ngroups * d_state, ngroups * d_state],
            dim=-1,
        )

        x_r = rearrange(x, "b l (h p) -> b l h p", p=headdim)
        Bm_r = rearrange(Bm, "b l (g n) -> b l g n", g=ngroups)
        Cm_r = rearrange(Cm, "b l (g n) -> b l g n", g=ngroups)

        A = -torch.exp(m.A_log.float())  # (nheads,)

        z_for_scan = (
            rearrange(z, "b l (h p) -> b l h p", p=headdim) if not rmsnorm else None
        )

        # Direct kernel call — this is the only path that actually carries
        # SSM state across calls at chunked-scan throughput.
        y, final_ssm_state = mamba_chunk_scan_combined(
            x_r, dt, A, Bm_r, Cm_r,
            chunk_size=chunk_size,
            D=m.D,
            z=z_for_scan,
            dt_bias=m.dt_bias,
            dt_softplus=True,
            initial_states=initial_ssm_state,
            seq_idx=seq_idx,
            return_final_states=True,
        )

        y = rearrange(y, "b l h p -> b l (h p)")

        if rmsnorm:
            y = m.norm(y, z)

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0_mlp, y], dim=-1)

        out = m.out_proj(y)
        return out, final_ssm_state, new_conv_input

    def forward_with_state(self, x, x0, mem, run_ffn, initial_ssm_state, prev_conv_input, seq_idx=None):
        """
        Mirror of forward() that threads SSM and conv state across blocks.
        Returns (x_out, mem, final_ssm_state, last_conv_input).
        """
        mix = self.resid_mix
        x_mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if USE_MEMORY and mem is not None:
            mem_ctx = self.mem_to_model(mem)[:, None, :]
            mem_mix = torch.sigmoid(self.mem_in_gate)
            x_in = x_mixed + mem_mix * mem_ctx
        else:
            x_in = x_mixed

        if self.fusion_enabled and self.fusion_mode == "B":
            a = self.fusion_attn(x_in)
            gate = torch.sigmoid(self.fusion_pre_gate).to(dtype=x_in.dtype, device=x_in.device)
            x_in = x_in + (FUSION_PRE_SCALE * gate) * a

        y, final_ssm_state, last_conv_input = self._mamba_forward_with_state(
            self.ssm_norm(x_in), initial_ssm_state, prev_conv_input, seq_idx=seq_idx
        )

        x_out = x_mixed + self.ssm_scale[None, None, :] * y
        if self.fusion_enabled and self.fusion_mode == "A":
            a = self.fusion_attn(x_out)
            gate = torch.sigmoid(self.fusion_gate).to(dtype=x_out.dtype, device=x_out.device)
            x_out = x_out + gate[None, None, :] * a

        if run_ffn:
            x_out = x_out + self.mlp_scale[None, None, :] * self.ffn(self.ffn_norm(x_out))

        if USE_MEMORY and mem is not None:
            new_mem = torch.tanh(self.memory_proj(x_out[:, -1, :]))
            g = torch.sigmoid(self.memory_gate)
            mem = g * mem + (1.0 - g) * new_mem

        return x_out, mem, final_ssm_state, last_conv_input



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
    """Causal RoPE attention teacher with optional split RoPE/content Q gain."""
    def __init__(self, d_model, n_heads=8, ffn_mult=3, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_dim = int(_layer_value(layer_idx, ROPE_LAYER_DIM_MAP, ROPE_DIM))
        self.rope_base = float(_layer_value(layer_idx, ROPE_LAYER_BASE_MAP, ROPE_BASE))
        self.rope_dim = max(0, min(self.rope_dim, self.head_dim))
        self.rope_dim -= self.rope_dim % 2

        self.attn_norm = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)

        # Q gain: split RoPE positional subspace from content subspace in mode D.
        # Do not leave an unused q_gain Parameter when split is enabled: DDP
        # find_unused_parameters=False would otherwise break.
        if Q_GAIN_SPLIT:
            self.register_buffer("q_gain", torch.empty(0), persistent=False)
            self.q_gain_rope = nn.Parameter(torch.full((n_heads,), Q_GAIN_ROPE_INIT))
            self.q_gain_content = nn.Parameter(torch.full((n_heads,), Q_GAIN_CONTENT_INIT))
        else:
            self.q_gain = nn.Parameter(torch.full((n_heads,), QK_GAIN_INIT))
            self.q_gain_rope = None
            self.q_gain_content = None

        if FUSION_MODE == "C":
            # Variant C: current-hidden-conditioned Q/K gain.
            self.q_gain_delta = nn.Linear(d_model, n_heads, bias=False)
            nn.init.zeros_(self.q_gain_delta.weight)
        else:
            self.q_gain_delta = None

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU_FFN(d_model, ffn_mult)
        self.attn_scale = nn.Parameter(torch.ones(d_model))
        self.mlp_scale = nn.Parameter(torch.ones(d_model))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(d_model), torch.zeros(d_model)]))

    def _apply_q_gain(self, q, normed):
        if Q_GAIN_SPLIT and self.q_gain_content is not None:
            rd = min(self.rope_dim, q.shape[-1])
            rope_gain = self.q_gain_rope.to(q.dtype)[None, :, None, None]
            content_gain = self.q_gain_content.to(q.dtype)[None, :, None, None]
            if self.q_gain_delta is not None:
                delta = self.q_gain_delta(normed).permute(0, 2, 1).unsqueeze(-1)
                delta = FUSION_QK_DELTA_SCALE * torch.tanh(delta).to(q.dtype)
                rope_gain = rope_gain + delta
                content_gain = content_gain + delta
            if rd > 0:
                q_rope = q[..., :rd] * rope_gain
                q_content = q[..., rd:] * content_gain
                return torch.cat((q_rope, q_content), dim=-1)
            return q * content_gain

        gain = self.q_gain.to(q.dtype)[None, :, None, None]
        if self.q_gain_delta is not None:
            delta = self.q_gain_delta(normed).permute(0, 2, 1).unsqueeze(-1)
            gain = gain + FUSION_QK_DELTA_SCALE * torch.tanh(delta).to(q.dtype)
        return q * gain

    def forward(self, x, x0, mem, run_ffn=True, return_context=False):
        mix = self.resid_mix
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        B, T, D = x.shape
        normed = self.attn_norm(x)
        qkv = self.qkv(normed).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)

        # QK RMSNorm + layer-specific partial RoPE + split learnable Q gain.
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        q, k = apply_partial_rope(q, k, self.rope_dim, self.rope_base)
        q = self._apply_q_gain(q, normed)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_ctx = attn_out.transpose(1, 2).reshape(B, T, D)
        projected = self.out_proj(attn_ctx)
        x = x + self.attn_scale[None, None, :] * projected

        if run_ffn:
            x = x + self.mlp_scale[None, None, :] * self.ffn(self.ffn_norm(x))
        if return_context:
            # Return unprojected attention context. It carries RoPE-relative
            # structure without relying on out_proj, which is zero-init.
            return x, mem, attn_ctx
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
                blk = CausalAttentionBlock(D_MODEL, ATTN_N_HEADS, FFN_MULT, layer_idx=i)
                blk._is_ssm = False
            else:
                blk = SelectiveSSMBlock(
                    D_MODEL, D_STATE, MEMORY_DIM, CONV_KERNEL, FFN_MULT, layer_idx=i)
                blk._is_ssm = True
            layers.append(blk)
        self.blocks = nn.ModuleList(layers)

        self.n_enc = eff // 2
        self.n_skip = min(self.n_enc, eff - self.n_enc)
        self.skip_weights = nn.Parameter(torch.full((self.n_skip, D_MODEL), float(SKIP_INIT)))
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

        # Depth-scaled init for mamba_ssm's internal Mamba2.out_proj.weight.
        # mamba_ssm's standalone Mamba2 class uses kaiming_uniform with no
        # depth correction; depth scaling (1/sqrt(2*n_layer)) is only applied
        # by the official MambaConfig _init_weights hook, which we don't use.
        # Without this, `out_proj` is the only residual-output projection
        # initialized "loud" at start of training (FFN.down and attention's
        # out_proj are both zero-initialized in our code). Confirmed via
        # verify_init_and_fp32.py: mamba.out_proj.weight std=0.018 vs the
        # depth-scaled target of 0.004 — about 4.5x too large.
        n_layer_for_init = N_UNIQUE_BLOCKS * N_UNROLLS
        depth_init_scale = 1.0 / math.sqrt(2 * n_layer_for_init)
        with torch.no_grad():
            n_scaled = 0
            for pname, p in self.named_parameters():
                if pname.endswith("mamba.out_proj.weight"):
                    p.mul_(depth_init_scale)
                    n_scaled += 1
            log0(f"depth-scaled init: scaled {n_scaled} mamba.out_proj.weight tensors by {depth_init_scale:.4f} (1/sqrt(2*{n_layer_for_init}))")

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

    def forward(self, ids, state=None, return_state=False, inference_params=None, seq_idx=None):
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

        # Auto-compute seq_idx from the input ids if not provided. This is
        # what tells Mamba2's chunk-scan kernel to zero state transitions at
        # document boundaries (chunk-aligned). Disabled when DOC_BOUNDARY_TOKEN
        # is negative or SEQ_IDX_ENABLED is off.
        if seq_idx is None and SEQ_IDX_ENABLED and DOC_BOUNDARY_TOKEN >= 0:
            with torch.no_grad():
                is_boundary = (ids == DOC_BOUNDARY_TOKEN).int()
                seq_idx = torch.cumsum(is_boundary, dim=-1).to(torch.int32).contiguous()

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

        # State carryover: only pass inference_params to SSM blocks (attention
        # blocks have no recurrent state to carry; they see only the current
        # input window). When loop is active and inference_params is set, we
        # would update the same block's state multiple times in one pass,
        # which double-counts: explicitly forbid that combination.
        if inference_params is not None and use_loop:
            raise RuntimeError(
                "inference_params is incompatible with depth recurrence "
                "(would update Mamba state multiple times per forward). "
                "Disable LOOP_START/LOOP_END for state-carryover eval."
            )

        skips = []
        rope_ctx = None  # Mode D: rolling RoPE-attention teacher context
        # Encoder. Inline dispatch (no closure) so torch.compile compiles
        # each block once with stable arg shapes rather than re-specializing
        # a wrapper per block id.
        for layer_pos, block_idx in enumerate(enc_indices):
            block = self.blocks[block_idx]
            if FFN_FREQ_MODE == "unroll":
                run_ffn = True
            else:
                run_ffn = (layer_pos % max(FFN_EVERY, 1)) == 0
            if block._is_ssm:
                x, mem = block(x, x0, mem, run_ffn=run_ffn,
                               inference_params=inference_params, seq_idx=seq_idx,
                               ctx=rope_ctx)
            else:
                if CONTEXT_BRIDGE_ENABLED:
                    x, mem, attn_ctx = block(x, x0, mem, run_ffn=run_ffn, return_context=True)
                    rope_ctx = attn_ctx if rope_ctx is None else (CONTEXT_BRIDGE_CTX_DECAY * rope_ctx + (1.0 - CONTEXT_BRIDGE_CTX_DECAY) * attn_ctx)
                else:
                    x, mem = block(x, x0, mem, run_ffn=run_ffn)
            skips.append(x)

        # Decoder with skip connections
        for layer_pos, block_idx in enumerate(dec_indices):
            block = self.blocks[block_idx]
            run_ffn = (layer_pos % max(FFN_EVERY, 1)) == 0
            if layer_pos < n_skip_eff and skips:
                x = x + self.skip_weights[layer_pos][None, None, :] * skips.pop()
            if block._is_ssm:
                x, mem = block(x, x0, mem, run_ffn=run_ffn,
                               inference_params=inference_params, seq_idx=seq_idx,
                               ctx=rope_ctx)
            else:
                if CONTEXT_BRIDGE_ENABLED:
                    x, mem, attn_ctx = block(x, x0, mem, run_ffn=run_ffn, return_context=True)
                    rope_ctx = attn_ctx if rope_ctx is None else (CONTEXT_BRIDGE_CTX_DECAY * rope_ctx + (1.0 - CONTEXT_BRIDGE_CTX_DECAY) * attn_ctx)
                else:
                    x, mem = block(x, x0, mem, run_ffn=run_ffn)

        out = self.final_norm(x)
        if return_state:
            return out, {"mem": mem} if USE_MEMORY else None
        return out

    def forward_with_state(self, ids, layer_states, seq_idx_base=0):
        """
        Stateful forward for cross-block carryover at eval time. Threads
        Mamba2 SSM state and conv1d state across calls via direct kernel
        access (mamba_chunk_scan_combined). Attention blocks see only the
        current block (no recurrent state to carry).

        ids: (B, L)
        layer_states: dict {block_idx: (ssm_state, conv_input)} or None.
            On first call pass None; the returned dict is fed back in for
            the next block.
        seq_idx_base: int — counter offset to keep doc ids monotonic across
            blocks. The new last seq_idx value is returned so the caller
            can pass it as seq_idx_base for the next block.

        Returns: (hidden_out, new_layer_states, next_seq_idx_base)
        """
        if not _HAS_CHUNK_SCAN or rearrange is None:
            raise RuntimeError(
                "mamba_chunk_scan_combined or einops not importable — "
                "direct-kernel state carryover is unavailable."
            )

        bsz = ids.shape[0]
        x = self.tok_emb(ids)
        if self.use_factored:
            x = self.embed_proj(x)
        if self.bigram is not None:
            x = x + self.bigram(ids)
        x0 = x
        mem = None  # USE_MEMORY interaction with carryover not implemented

        # Build seq_idx for this block. Monotonic across blocks via
        # seq_idx_base — when state is being carried, doc ids must keep
        # incrementing so a doc boundary at block boundary correctly
        # zeroes the carried state's contribution.
        #
        # NOTE: empirically this offset-by-seq_idx_base behavior is what
        # works. We tried "reset seq_idx to 0 at every block" thinking the
        # kernel's internal seq_idx=0 start was an assertion about carried
        # state — that interpretation produced a -0.04 BPB regression. The
        # original behavior, which silently zeroes the carried state on
        # block boundaries via seq_idx mismatch, is actually less harmful
        # than letting stale state from millions of tokens ago flow into
        # the next block's first chunk. The model never trained for non-
        # zero initial_states, so OOD carryover is worse than fresh start.
        if SEQ_IDX_ENABLED and DOC_BOUNDARY_TOKEN >= 0:
            with torch.no_grad():
                is_boundary = (ids == DOC_BOUNDARY_TOKEN).int()
                local_seq_idx = torch.cumsum(is_boundary, dim=-1).to(torch.int32)
                seq_idx = (local_seq_idx + int(seq_idx_base)).contiguous()
                next_seq_idx_base = int(seq_idx[:, -1].max().item())
        else:
            seq_idx = None
            next_seq_idx_base = int(seq_idx_base)

        # Carryover is incompatible with depth recurrence (would update the
        # same block's state multiple times per forward).
        use_loop = self.loop_enabled_external and self.loop_enc_seq is not None
        if use_loop:
            raise RuntimeError(
                "forward_with_state is incompatible with depth recurrence."
            )
        half = N_UNIQUE_BLOCKS // 2
        enc_indices = list(range(0, half))
        dec_indices = list(range(half, N_UNIQUE_BLOCKS))

        n_skip_eff = min(len(enc_indices), len(dec_indices), self.n_skip)

        if layer_states is None:
            layer_states = {}
        new_layer_states = {}

        skips = []
        for layer_pos, block_idx in enumerate(enc_indices):
            block = self.blocks[block_idx]
            if FFN_FREQ_MODE == "unroll":
                run_ffn = True
            else:
                run_ffn = (layer_pos % max(FFN_EVERY, 1)) == 0
            if block._is_ssm:
                init_ssm, init_conv = layer_states.get(block_idx, (None, None))
                x, _, fin_ssm, fin_conv = block.forward_with_state(
                    x, x0, mem, run_ffn, init_ssm, init_conv, seq_idx=seq_idx
                )
                new_layer_states[block_idx] = (fin_ssm, fin_conv)
            else:
                x, _ = block(x, x0, mem, run_ffn=run_ffn)
            skips.append(x)

        for layer_pos, block_idx in enumerate(dec_indices):
            block = self.blocks[block_idx]
            run_ffn = (layer_pos % max(FFN_EVERY, 1)) == 0
            if layer_pos < n_skip_eff and skips:
                x = x + self.skip_weights[layer_pos][None, None, :] * skips.pop()
            if block._is_ssm:
                init_ssm, init_conv = layer_states.get(block_idx, (None, None))
                x, _, fin_ssm, fin_conv = block.forward_with_state(
                    x, x0, mem, run_ffn, init_ssm, init_conv, seq_idx=seq_idx
                )
                new_layer_states[block_idx] = (fin_ssm, fin_conv)
            else:
                x, _ = block(x, x0, mem, run_ffn=run_ffn)

        out = self.final_norm(x)
        return out, new_layer_states, next_seq_idx_base


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
    Original one-bucket Muon optimizer for matrix-shaped parameters.
    Kept for ablations via TASKMUON_ENABLED=0.
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


def _taskmuon_row_balance(G: torch.Tensor, eps: float = 1e-8, max_gain: float = 10.0) -> torch.Tensor:
    """MuonEq-lite row balancing before the NS iteration."""
    Gf = G.float()
    global_rms = Gf.pow(2).mean().sqrt().clamp_min(eps)
    row_rms = Gf.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    gain = (global_rms / row_rms).clamp(max=max_gain)
    return (Gf * gain).to(dtype=G.dtype)


def _taskmuon_shape_scale(rows: int, cols: int, cap: float) -> float:
    scale = math.sqrt(max(1.0, float(rows) / max(float(cols), 1.0)))
    if cap and cap > 0:
        scale = min(scale, float(cap))
    return float(scale)


class TaskMuon(torch.optim.Optimizer):
    """
    Component-aware Muon for the Parameter Golf SSM/attention hybrid.

    What changes versus generic Muon:
      - attention qkv is optimized as q/k/v row blocks, not one fused matrix
      - Mamba in_proj is split into feature rows (z/x), dynamics rows (B/C), and dt rows
      - dt rows use AdamW-style updates instead of NS/Muon
      - Mamba projections use lower LR/WD and capped rectangular scaling
      - optional MuonEq-lite row balancing before Newton-Schulz
    """
    def __init__(self, param_groups, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True):
        defaults = dict(lr=lr, base_lr=lr, momentum=momentum,
                        backend_steps=backend_steps, nesterov=nesterov,
                        weight_decay=WEIGHT_DECAY)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def _muon_update_2d(self, upd_2d: torch.Tensor, *, backend_steps: int,
                        row_balance: bool, shape_cap: float,
                        trust_clip: bool) -> torch.Tensor:
        raw = upd_2d
        work = _taskmuon_row_balance(raw) if row_balance else raw
        upd_orth = zeropower_via_newtonschulz5(work, steps=backend_steps)
        upd_orth = upd_orth * _taskmuon_shape_scale(upd_orth.size(0), upd_orth.size(1), shape_cap)
        if trust_clip:
            raw_rms = raw.float().pow(2).mean().sqrt().clamp_min(1e-12)
            orth_rms = upd_orth.float().pow(2).mean().sqrt().clamp_min(1e-12)
            max_rms = torch.clamp(raw_rms * TASKMUON_TRUST_MULT, min=TASKMUON_TRUST_FLOOR)
            if bool((orth_rms > max_rms).item()):
                upd_orth = upd_orth * (max_rms / orth_rms).to(upd_orth.dtype)
        return upd_orth

    @torch.no_grad()
    def _step_adamw_rows(self, p: torch.Tensor, g: torch.Tensor, state: dict,
                         sl: slice, *, lr: float, wd: float, seg_key: str):
        if wd > 0:
            p[sl].mul_(1.0 - lr * wd)
        g_seg = g[sl].float()
        m_key = f"adam_m_{seg_key}"
        v_key = f"adam_v_{seg_key}"
        step_key = f"adam_step_{seg_key}"
        if m_key not in state:
            state[m_key] = torch.zeros_like(g_seg)
            state[v_key] = torch.zeros_like(g_seg)
            state[step_key] = 0
        exp_avg = state[m_key]
        exp_avg_sq = state[v_key]
        state[step_key] += 1
        step_i = int(state[step_key])
        exp_avg.mul_(BETA1).add_(g_seg, alpha=1.0 - BETA1)
        exp_avg_sq.mul_(BETA2).addcmul_(g_seg, g_seg, value=1.0 - BETA2)
        bc1 = 1.0 - (BETA1 ** step_i)
        bc2 = 1.0 - (BETA2 ** step_i)
        denom = (exp_avg_sq.sqrt() / math.sqrt(max(bc2, 1e-12))).add_(ADAM_EPS)
        update = (exp_avg / denom).to(dtype=p.dtype)
        p[sl].add_(update, alpha=-(lr / max(bc1, 1e-12)))

    @torch.no_grad()
    def _step_muon_rows(self, p: torch.Tensor, g: torch.Tensor, state: dict,
                        sl: slice, *, lr: float, wd: float, momentum: float,
                        backend_steps: int, nesterov: bool, shape_cap: float,
                        row_balance: bool, trust_clip: bool, seg_key: str):
        if wd > 0:
            p[sl].mul_(1.0 - lr * wd)
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(g)
        buf = state["momentum_buffer"]
        g_seg = g[sl]
        buf_seg = buf[sl]
        buf_seg.mul_(momentum).add_(g_seg)
        upd = g_seg.add(buf_seg, alpha=momentum) if nesterov else buf_seg
        orig_shape = upd.shape
        upd_2d = upd.reshape(upd.shape[0], -1) if upd.ndim > 2 else upd
        if upd_2d.ndim >= 2:
            upd_orth = self._muon_update_2d(
                upd_2d,
                backend_steps=backend_steps,
                row_balance=row_balance,
                shape_cap=shape_cap,
                trust_clip=trust_clip,
            )
            upd = upd_orth.reshape(orig_shape)
        p[sl].add_(upd.to(dtype=p.dtype), alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr_base = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            default_wd = group.get("weight_decay", WEIGHT_DECAY)
            default_lr_mult = group.get("lr_mult", 1.0)
            default_shape_cap = group.get("shape_cap", 0.0)
            default_row_balance = group.get("row_balance", TASKMUON_ROW_BALANCE)
            default_trust_clip = group.get("trust_clip", TASKMUON_TRUST_CLIP)
            segments = group.get("segments", None)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not segments:
                    sl = slice(0, p.shape[0])
                    self._step_muon_rows(
                        p, g, state, sl,
                        lr=lr_base * default_lr_mult,
                        wd=default_wd,
                        momentum=momentum,
                        backend_steps=backend_steps,
                        nesterov=nesterov,
                        shape_cap=default_shape_cap,
                        row_balance=default_row_balance,
                        trust_clip=default_trust_clip,
                        seg_key="full",
                    )
                    continue

                for si, seg in enumerate(segments):
                    st, en = int(seg["start"]), int(seg["end"])
                    if en <= st:
                        continue
                    sl = slice(st, en)
                    lr = lr_base * float(seg.get("lr_mult", default_lr_mult))
                    wd = float(seg.get("weight_decay", default_wd))
                    seg_key = f"{si}_{seg.get('name', 'seg')}"
                    if seg.get("optimizer", "muon") == "adamw":
                        self._step_adamw_rows(p, g, state, sl, lr=lr, wd=wd, seg_key=seg_key)
                    else:
                        self._step_muon_rows(
                            p, g, state, sl,
                            lr=lr,
                            wd=wd,
                            momentum=momentum,
                            backend_steps=int(seg.get("backend_steps", backend_steps)),
                            nesterov=nesterov,
                            shape_cap=float(seg.get("shape_cap", default_shape_cap)),
                            row_balance=bool(seg.get("row_balance", default_row_balance)),
                            trust_clip=bool(seg.get("trust_clip", default_trust_clip)),
                            seg_key=seg_key,
                        )
        return loss


def _seg(name, start, end, lr_mult, weight_decay, shape_cap, row_balance=True,
         optimizer="muon", trust_clip=None, backend_steps=None):
    out = dict(
        name=name,
        start=int(start),
        end=int(end),
        lr_mult=float(lr_mult),
        weight_decay=float(weight_decay),
        shape_cap=float(shape_cap),
        row_balance=bool(row_balance),
        optimizer=optimizer,
    )
    if trust_clip is not None:
        out["trust_clip"] = bool(trust_clip)
    if backend_steps is not None:
        out["backend_steps"] = int(backend_steps)
    return out


def _qkv_rope_segments(name: str, p: torch.nn.Parameter, modules: dict):
    """Split fused attention qkv into Q/K RoPE rows, Q/K content rows, and V.

    Rows are laid out q(0:D), k(D:2D), v(2D:3D). Within q/k, each head is a
    contiguous block of head_dim rows, and the first rope_dim rows per head are
    the RoPE positional subspace.
    """
    parent_name = name.rsplit(".qkv.weight", 1)[0]
    m = modules.get(parent_name, None)
    if m is None:
        return None
    try:
        rows = int(p.shape[0])
        if rows % 3 != 0:
            return None
        d = rows // 3
        n_heads = int(getattr(m, "n_heads"))
        head_dim = int(getattr(m, "head_dim"))
        rope_dim = int(getattr(m, "rope_dim", ROPE_DIM))
        rope_dim = max(0, min(rope_dim, head_dim))
        rope_dim -= rope_dim % 2
        if d != n_heads * head_dim:
            return None
        segs = []
        for block_name, block_off in (("q", 0), ("k", d)):
            for h in range(n_heads):
                h0 = block_off + h * head_dim
                if rope_dim > 0:
                    segs.append(_seg(
                        f"{block_name}{h}_rope", h0, h0 + rope_dim,
                        TASKMUON_QK_ROPE_LR_MULT, TASKMUON_QK_ROPE_WD,
                        TASKMUON_QK_ROPE_SHAPE_CAP,
                        row_balance=False,
                        trust_clip=True,
                    ))
                if rope_dim < head_dim:
                    segs.append(_seg(
                        f"{block_name}{h}_content", h0 + rope_dim, h0 + head_dim,
                        TASKMUON_QK_CONTENT_LR_MULT, TASKMUON_QK_CONTENT_WD,
                        TASKMUON_QKV_SHAPE_CAP,
                        row_balance=TASKMUON_ROW_BALANCE,
                        trust_clip=TASKMUON_TRUST_CLIP,
                    ))
        segs.append(_seg(
            "v", 2 * d, 3 * d,
            TASKMUON_V_LR_MULT, TASKMUON_V_WD,
            TASKMUON_QKV_SHAPE_CAP,
            row_balance=TASKMUON_ROW_BALANCE,
        ))
        return segs
    except Exception:
        return None


def _ffn_gate_up_segments(name: str, p: torch.nn.Parameter, modules: dict):
    """Split SwiGLU gate_up into gate and up halves for Muon geometry."""
    parent_name = name.rsplit(".gate_up.weight", 1)[0]
    m = modules.get(parent_name, None)
    if m is None:
        return None
    rows = int(p.shape[0])
    try:
        if bool(getattr(m, "is_swiglu", False)) and rows % 2 == 0:
            half = rows // 2
            return [
                _seg("ffn_gate", 0, half, TASKMUON_FFN_GATE_LR_MULT, TASKMUON_FFN_WD,
                     TASKMUON_FFN_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE),
                _seg("ffn_up", half, rows, TASKMUON_FFN_UP_LR_MULT, TASKMUON_FFN_WD,
                     TASKMUON_FFN_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE),
            ]
        return [
            _seg("ffn_up", 0, rows, TASKMUON_FFN_UP_LR_MULT, TASKMUON_FFN_WD,
                 TASKMUON_FFN_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE)
        ]
    except Exception:
        return None


def _mamba_in_proj_segments(name: str, p: torch.nn.Parameter, modules: dict):
    parent_name = name.rsplit(".in_proj.weight", 1)[0]
    m = modules.get(parent_name, None)
    if m is None:
        return None
    try:
        rows = int(p.shape[0])
        d_inner = int(getattr(m, "d_inner", getattr(m, "d_ssm", 0)))
        nheads = int(getattr(m, "nheads"))
        ngroups = int(getattr(m, "ngroups", 1))
        d_state = int(getattr(m, "d_state"))
        conv_channels = d_inner + 2 * ngroups * d_state
        expected = 2 * d_inner + 2 * ngroups * d_state + nheads
        d_mlp = (rows - expected) // 2
        if d_inner <= 0 or d_mlp < 0 or (rows - expected) % 2 != 0:
            return None
        segs = []
        offset = 0
        if d_mlp > 0:
            # Optional Mamba2 MLP branch rows. Treat as feature rows, not dynamics.
            segs.append(_seg("mamba_mlp", offset, offset + 2 * d_mlp,
                             TASKMUON_MAMBA_ZX_LR_MULT, TASKMUON_MAMBA_ZX_WD,
                             TASKMUON_MAMBA_IN_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE))
            offset += 2 * d_mlp
        # z rows
        segs.append(_seg("mamba_z", offset, offset + d_inner,
                         TASKMUON_MAMBA_ZX_LR_MULT, TASKMUON_MAMBA_ZX_WD,
                         TASKMUON_MAMBA_IN_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE))
        offset += d_inner
        # x rows inside xBC
        segs.append(_seg("mamba_x", offset, offset + d_inner,
                         TASKMUON_MAMBA_ZX_LR_MULT, TASKMUON_MAMBA_ZX_WD,
                         TASKMUON_MAMBA_IN_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE))
        offset += d_inner
        bc = ngroups * d_state
        segs.append(_seg("mamba_B", offset, offset + bc,
                         TASKMUON_MAMBA_BC_LR_MULT, TASKMUON_MAMBA_BC_WD,
                         TASKMUON_MAMBA_BC_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE,
                         trust_clip=TASKMUON_TRUST_CLIP))
        offset += bc
        segs.append(_seg("mamba_C", offset, offset + bc,
                         TASKMUON_MAMBA_BC_LR_MULT, TASKMUON_MAMBA_BC_WD,
                         TASKMUON_MAMBA_BC_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE,
                         trust_clip=TASKMUON_TRUST_CLIP))
        offset += bc
        segs.append(_seg("mamba_dt", offset, offset + nheads,
                         TASKMUON_MAMBA_DT_LR_MULT, TASKMUON_MAMBA_DT_WD,
                         1.0, row_balance=False, optimizer="adamw"))
        offset += nheads
        if offset != rows:
            return None
        return segs
    except Exception:
        return None


def _taskmuon_group_for_param(name: str, p: torch.nn.Parameter, modules: dict):
    """Return a per-parameter TaskMuon group dict, or None for non-Muon params."""
    if ((p.ndim == 2) if MUON_ONLY_2D else (p.ndim >= 2)) is False:
        return None
    if any(c in name for c in CTRL_PATTERNS):
        return None

    base = {
        "params": [p],
        "name": name,
        "lr": MATRIX_LR,
        "base_lr": MATRIX_LR,
        "momentum": MUON_MOMENTUM,
        "backend_steps": MUON_BACKEND_STEPS,
        "nesterov": MUON_NESTEROV,
    }

    if name.endswith(".qkv.weight") and p.shape[0] % 3 == 0:
        rope_segs = _qkv_rope_segments(name, p, modules) if Q_GAIN_SPLIT else None
        if rope_segs is not None:
            base.update(
                role="attn_qkv_rope_split",
                weight_decay=TASKMUON_ATTN_WD,
                lr_mult=TASKMUON_QKV_LR_MULT,
                shape_cap=TASKMUON_QKV_SHAPE_CAP,
                row_balance=TASKMUON_ROW_BALANCE,
                segments=rope_segs,
            )
        else:
            n = int(p.shape[0]) // 3
            base.update(
                role="attn_qkv_split",
                weight_decay=TASKMUON_ATTN_WD,
                lr_mult=TASKMUON_QKV_LR_MULT,
                shape_cap=TASKMUON_QKV_SHAPE_CAP,
                row_balance=TASKMUON_ROW_BALANCE,
                segments=[
                    _seg("q", 0, n, TASKMUON_QKV_LR_MULT, TASKMUON_ATTN_WD, TASKMUON_QKV_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE),
                    _seg("k", n, 2 * n, TASKMUON_QKV_LR_MULT, TASKMUON_ATTN_WD, TASKMUON_QKV_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE),
                    _seg("v", 2 * n, 3 * n, TASKMUON_QKV_LR_MULT, TASKMUON_ATTN_WD, TASKMUON_QKV_SHAPE_CAP, row_balance=TASKMUON_ROW_BALANCE),
                ],
            )
        return base

    if name.endswith(".mamba.in_proj.weight"):
        segs = _mamba_in_proj_segments(name, p, modules)
        if segs is not None:
            base.update(
                role="mamba_in_proj_split",
                weight_decay=TASKMUON_MAMBA_ZX_WD,
                lr_mult=TASKMUON_MAMBA_ZX_LR_MULT,
                shape_cap=TASKMUON_MAMBA_IN_SHAPE_CAP,
                row_balance=TASKMUON_ROW_BALANCE,
                segments=segs,
            )
            return base

    if name.endswith(".mamba.out_proj.weight"):
        base.update(
            role="mamba_out_proj",
            weight_decay=TASKMUON_MAMBA_OUT_WD,
            lr_mult=TASKMUON_MAMBA_OUT_LR_MULT,
            shape_cap=TASKMUON_MAMBA_OUT_SHAPE_CAP,
            row_balance=False,
        )
        return base

    if name.endswith(".out_proj.weight"):
        base.update(
            role="attn_out_proj",
            weight_decay=TASKMUON_ATTN_WD,
            lr_mult=TASKMUON_ATTN_OUT_LR_MULT,
            shape_cap=1.0,
            row_balance=False,
        )
        return base

    if name.endswith(".gate_up.weight"):
        segs = _ffn_gate_up_segments(name, p, modules)
        base.update(
            role="ffn_gate_up_split" if segs is not None else "ffn",
            weight_decay=TASKMUON_FFN_WD,
            lr_mult=TASKMUON_FFN_LR_MULT,
            shape_cap=TASKMUON_FFN_SHAPE_CAP,
            row_balance=TASKMUON_ROW_BALANCE,
            segments=segs,
        )
        return base

    if name.endswith(".down.weight"):
        base.update(
            role="ffn_down",
            weight_decay=TASKMUON_FFN_DOWN_WD,
            lr_mult=TASKMUON_FFN_DOWN_LR_MULT,
            shape_cap=TASKMUON_FFN_SHAPE_CAP,
            row_balance=TASKMUON_ROW_BALANCE,
        )
        return base

    if ".context_bridge." in name and name.endswith(".weight"):
        base.update(
            role="context_bridge",
            weight_decay=TASKMUON_BRIDGE_WD,
            lr_mult=TASKMUON_BRIDGE_LR_MULT,
            shape_cap=1.0,
            row_balance=False,
        )
        return base

    # Keep all other eligible matrices under TaskMuon with generic settings so
    # embeddings projections, fusion projections, etc. do not silently move to AdamW.
    base.update(
        role="generic",
        weight_decay=TASKMUON_GENERIC_WD,
        lr_mult=TASKMUON_GENERIC_LR_MULT,
        shape_cap=0.0,
        row_balance=False,
    )
    return base


def build_optimizers(model):
    scalar_wd_params, no_wd_params, embed_params = [], [], []
    mat_params = []
    task_groups = []
    role_counts = {}
    muon_eligible_2d = 0
    muon_eligible_nd = 0
    modules = dict(model.named_modules())
    _seen_data_ptrs = set()  # handle tied params

    for name, p in model.named_parameters():
        dp = p.data_ptr()
        if dp in _seen_data_ptrs:
            continue  # skip tied duplicate (lm_head.weight = tok_emb.weight)
        _seen_data_ptrs.add(dp)
        if name in ("tok_emb.weight", "lm_head.weight"):
            embed_params.append(p)
            continue

        no_wd = (
            getattr(p, "_no_weight_decay", False)
            or name.endswith(".bias")
            or p.ndim < 2
        )
        if no_wd:
            no_wd_params.append(p)
            continue

        is_control = any(c in name for c in CTRL_PATTERNS)
        if p.ndim == 2 and not is_control:
            muon_eligible_2d += 1
        elif p.ndim > 2 and not is_control:
            muon_eligible_nd += 1

        if ((p.ndim == 2) if MUON_ONLY_2D else (p.ndim >= 2)) and not is_control:
            if TASKMUON_ENABLED:
                g = _taskmuon_group_for_param(name, p, modules)
                if g is not None:
                    task_groups.append(g)
                    mat_params.append(p)
                    role_counts[g.get("role", "unknown")] = role_counts.get(g.get("role", "unknown"), 0) + 1
                else:
                    scalar_wd_params.append(p)
            else:
                mat_params.append(p)
        else:
            scalar_wd_params.append(p)

    if TASKMUON_ENABLED:
        optimizer_muon = TaskMuon(
            task_groups,
            lr=MATRIX_LR,
            momentum=MUON_MOMENTUM,
            backend_steps=MUON_BACKEND_STEPS,
            nesterov=MUON_NESTEROV,
        ) if len(task_groups) > 0 else None
    else:
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
            {"params": scalar_wd_params, "lr": SCALAR_LR, "weight_decay": WEIGHT_DECAY, "base_lr": SCALAR_LR},
            {"params": no_wd_params, "lr": SCALAR_LR, "weight_decay": 0.0, "base_lr": SCALAR_LR},
            {"params": embed_params, "lr": EMBED_LR, "weight_decay": WEIGHT_DECAY, "base_lr": EMBED_LR},
        ],
        betas=(BETA1, BETA2),
        eps=ADAM_EPS,
        fused=True,
    )
    opt_name = "TaskMuon" if TASKMUON_ENABLED else "Muon"
    log0(
        f"Optimizer split: matrix={len(mat_params)} ({opt_name}), "
        f"scalar_wd={len(scalar_wd_params)} (AdamW wd={WEIGHT_DECAY}), "
        f"no_wd={len(no_wd_params)} (AdamW wd=0), "
        f"embed={len(embed_params)} (AdamW)"
    )
    log0(
        f"Muon config: taskmuon={int(TASKMUON_ENABLED)} muon_only_2d={int(MUON_ONLY_2D)} "
        f"eligible_2d={muon_eligible_2d} eligible_nd={muon_eligible_nd}"
    )
    if TASKMUON_ENABLED:
        role_summary = ", ".join(f"{k}={v}" for k, v in sorted(role_counts.items()))
        log0(f"TaskMuon roles: {role_summary}")
        log0(
            "TaskMuon knobs: "
            f"qkv_lr={TASKMUON_QKV_LR_MULT}, attn_out_lr={TASKMUON_ATTN_OUT_LR_MULT}, "
            f"mamba_zx_lr={TASKMUON_MAMBA_ZX_LR_MULT}, mamba_bc_lr={TASKMUON_MAMBA_BC_LR_MULT}, "
            f"mamba_dt_lr={TASKMUON_MAMBA_DT_LR_MULT}, mamba_out_lr={TASKMUON_MAMBA_OUT_LR_MULT}, "
            f"qk_rope_lr={TASKMUON_QK_ROPE_LR_MULT}, qk_content_lr={TASKMUON_QK_CONTENT_LR_MULT}, "
            f"v_lr={TASKMUON_V_LR_MULT}, ffn_gate/up/down_lr="
            f"{TASKMUON_FFN_GATE_LR_MULT}/{TASKMUON_FFN_UP_LR_MULT}/{TASKMUON_FFN_DOWN_LR_MULT}, "
            f"bridge_lr={TASKMUON_BRIDGE_LR_MULT}"
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




def _rank_bounds(n_items: int):
    """Exact contiguous sharding with no dropped remainder items."""
    start = (int(n_items) * RANK) // WORLD_SIZE
    end = (int(n_items) * (RANK + 1)) // WORLD_SIZE
    return start, end


def _sliding_score_records(total_tokens: int, seq_len: int, stride: int,
                           score_start: int = 0, score_end=None):
    """Yield full-length causal windows that score each target position once.

    A record is (window_start, tail_start, tail_len). The input window is
    val_tokens[window_start : window_start + seq_len], and only
    [tail_start : tail_start + tail_len] positions in that window are scored.

    The first prefix is scored from a normal full-length window starting at 0,
    so Mamba/attention kernels still see the regular SEQ_LEN shape.
    """
    total_tokens = int(total_tokens)
    seq_len = int(seq_len)
    stride = int(stride)
    if score_end is None:
        score_end = total_tokens
    score_start = max(0, int(score_start))
    score_end = min(total_tokens, int(score_end))
    if score_end <= score_start:
        return

    cur = score_start
    if cur < min(score_end, seq_len):
        prefix_end = min(score_end, seq_len)
        yield 0, cur, prefix_end - cur
        cur = prefix_end

    while cur < score_end:
        se = min(cur + stride, score_end)
        window_start = max(0, se - seq_len)
        if window_start + seq_len > total_tokens:
            window_start = max(0, total_tokens - seq_len)
        tail_start = cur - window_start
        tail_len = se - cur
        if tail_len > 0:
            yield window_start, tail_start, tail_len
        cur = se


def _iter_sliding_score_batches(val_tokens, record_indices, seq_len: int, batch_windows: int):
    """Batch sliding-score records by compatible tail slice."""
    pending_x, pending_y = [], []
    pending_key = None

    def flush():
        nonlocal pending_x, pending_y, pending_key
        if not pending_x:
            return None
        xn = np.stack(pending_x)
        yn = np.stack(pending_y)
        tail_start, tail_len = pending_key
        pending_x, pending_y, pending_key = [], [], None
        return xn, yn, tail_start, tail_len

    for window_start, tail_start, tail_len in record_indices:
        key = (int(tail_start), int(tail_len))
        if pending_key is not None and (key != pending_key or len(pending_x) >= batch_windows):
            item = flush()
            if item is not None:
                yield item
        pending_key = key
        pending_x.append(val_tokens[window_start : window_start + seq_len])
        pending_y.append(val_tokens[window_start + 1 : window_start + seq_len + 1])

    item = flush()
    if item is not None:
        yield item


def lm_loss_tail(core, ids, targets, tail_start: int, tail_len: int):
    """LM loss restricted to already-scored tail tokens for legal TTT."""
    hidden = core(ids, state=None, return_state=False)
    tail_start = int(tail_start)
    tail_len = int(tail_len)
    hidden_tail = hidden[:, tail_start : tail_start + tail_len, :]
    y_tail = targets[:, tail_start : tail_start + tail_len]
    logits = core.output_logits_from_hidden(hidden_tail)
    logits = core.add_copy_logits(logits, hidden_tail, hidden, ids, query_start=tail_start)
    return F.cross_entropy(logits.float(), y_tail.reshape(-1), reduction="mean")

@torch.no_grad()
def eval_val_sliding(model_or_ddp, val_tokens, bb, hs, ib, stride=None):
    """
    Exact sliding-window evaluation.

    Scores every target token exactly once. The first prefix is scored from a
    full SEQ_LEN window starting at 0; subsequent tokens are scored in `stride`
    tails from windows ending at the scored segment. Remainder records are
    sharded exactly across ranks.
    """
    if stride is None:
        stride = EVAL_STRIDE
    core = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp
    core.eval()
    seq_len = SEQ_LEN
    total_tokens = val_tokens.size - 1

    records = list(_sliding_score_records(total_tokens, seq_len, stride))
    rec_start, rec_end = _rank_bounds(len(records))
    local_records = records[rec_start:rec_end]

    batch_windows = max(1, 131072 // seq_len)
    loss_sum = 0.0
    tok_sum = 0.0
    byt_sum = 0.0
    t0_eval = time.perf_counter()
    total_batches = max(1, (len(local_records) + batch_windows - 1) // batch_windows)
    log_every_batches = max(1, total_batches // 10)

    batch_idx = 0
    for xn, yn, tail_start, tail_len in _iter_sliding_score_batches(
            val_tokens, local_records, seq_len, batch_windows):
        x = torch.from_numpy(xn).long().to(DEVICE, non_blocking=True)
        y = torch.from_numpy(yn).long().to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
            hidden = core(x, state=None, return_state=False)
            hidden_tail = hidden[:, tail_start : tail_start + tail_len, :]
            y_tail = y[:, tail_start : tail_start + tail_len]
            logits = core.output_logits_from_hidden(hidden_tail)
            logits = core.add_copy_logits(logits, hidden_tail, hidden, x, query_start=tail_start)
            loss = F.cross_entropy(logits.float(), y_tail.reshape(-1), reduction="sum")

        cnt = float(y_tail.numel())
        loss_sum += float(loss.detach().float().item())

        p_tail = xn[:, tail_start : tail_start + tail_len].reshape(-1)
        t_tail = yn[:, tail_start : tail_start + tail_len].reshape(-1)
        b = bb[t_tail].astype(np.int16, copy=True)
        b += (hs[t_tail] & ~ib[p_tail]).astype(np.int16)
        tok_sum += cnt
        byt_sum += float(b.astype(np.float64).sum())

        batch_idx += 1
        if batch_idx % log_every_batches == 0:
            pct = 100.0 * batch_idx / total_batches
            elapsed_e = time.perf_counter() - t0_eval
            eta = elapsed_e / batch_idx * max(0, total_batches - batch_idx)
            log0(f"  sliding_eval: {pct:.0f}% ({batch_idx}/{total_batches} batches, eta:{eta:.0f}s)")

    elapsed = time.perf_counter() - t0_eval
    log0(f"  sliding_eval: {len(local_records)} records/rank, stride={stride}, "
         f"{tok_sum:.0f} scored tokens, {elapsed:.1f}s")

    stats = torch.tensor([loss_sum, tok_sum, byt_sum], device=DEVICE, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, tok_sum, byt_sum = float(stats[0]), float(stats[1]), float(stats[2])
    val_loss = loss_sum / max(tok_sum, 1.0)
    bpt = val_loss / math.log(2.0)
    core.train()
    return float(val_loss), float(bpt * (tok_sum / max(byt_sum, 1.0)))


@torch.no_grad()
def eval_val_carryover(model_or_ddp, val_tokens, bb, hs, ib,
                       block_len=None, warmup_tokens=None):
    """
    SSM-native state-carryover eval.

    Process the val set as non-overlapping `block_len`-token blocks. The
    Mamba2 SSM hidden state and conv1d state for every layer are threaded
    across blocks via the direct kernel call (mamba_chunk_scan_combined
    with initial_states / return_final_states), so a token at position p
    sees ALL p preceding tokens via the recurrence (not just the SEQ_LEN
    window an attention model would have). Attention layers (which can't
    carry state) see only the current block, matching training context.

    Each rank gets a contiguous slice of the val stream so state passing
    stays causal within rank. The first `warmup_tokens` of each rank's
    slice are processed to warm the recurrent state but their losses are
    not scored (otherwise positions with near-empty state would be unfairly
    counted).

    If CARRYOVER_DIRECT_KERNEL is False, falls back to the inference_params
    path (broken in current mamba_ssm builds — kept for diagnosis only).

    Returns (val_loss, val_bpb).
    """
    if block_len is None:
        block_len = CARRYOVER_BLOCK_LEN
    if warmup_tokens is None:
        warmup_tokens = CARRYOVER_WARMUP_TOKENS
    warmup_blocks = warmup_tokens // block_len  # round down to block boundary

    use_direct = CARRYOVER_DIRECT_KERNEL and _HAS_CHUNK_SCAN and rearrange is not None
    if not use_direct:
        if not _HAS_CHUNK_SCAN:
            log0("[carryover_eval] mamba_chunk_scan_combined unavailable")
        if rearrange is None:
            log0("[carryover_eval] einops not importable")
        if InferenceParams is None:
            raise RuntimeError("no carryover path available (no chunk_scan, no InferenceParams)")
        log0("[carryover_eval] falling back to InferenceParams path (likely broken)")

    core = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp
    core.eval()

    if getattr(core, "loop_enabled_external", False):
        log0("[carryover_eval] depth recurrence active — temporarily disabled for this eval")
        prev_loop = True
        core.loop_enabled_external = False
    else:
        prev_loop = False

    total_tokens = val_tokens.size - 1  # x/y consume span-1 tokens

    # Stateful-overlap config (PR #1644 / reviewer guidance):
    #   - block has length `block_len` total tokens
    #   - first `overlap` tokens are "context" — Mamba state has carried
    #     them via initial_states; attention re-sees them inside the block;
    #     they are NOT scored (already counted as score_region in prior block)
    #   - remaining `score_region = block_len - overlap` tokens are scored
    #   - block stride = score_region (not block_len), so consecutive blocks
    #     overlap by exactly `overlap` tokens
    #
    # When overlap=0, this collapses to non-overlap (legacy) behavior.
    overlap = max(0, min(int(CARRYOVER_OVERLAP), block_len - 1))
    score_region = block_len - overlap

    # Exact block sharding. Global block index g starts at g * score_region.
    # Only full block_len windows are used here to keep the direct Mamba kernel
    # shape stable. No block index is dropped across ranks.
    if score_region <= 0 or total_tokens < block_len:
        n_blocks_total = 0
    else:
        n_blocks_total = max(0, (total_tokens - block_len) // score_region + 1)
    block_start_idx, block_end_idx = _rank_bounds(n_blocks_total)
    n_blocks = block_end_idx - block_start_idx
    rank_start = block_start_idx * score_region
    n_scored_blocks = max(0, n_blocks - warmup_blocks)

    # State containers
    if use_direct:
        layer_states = None  # filled on first block by forward_with_state
        seq_idx_base = 0     # monotonic doc-id counter across blocks
    else:
        inference_params = InferenceParams(max_seqlen=block_len, max_batch_size=1)

    loss_sum = 0.0
    tok_sum = 0.0
    byt_sum = 0.0
    t0_eval = time.perf_counter()
    log_every_blocks = max(1, n_blocks // 20) if n_blocks > 0 else 1

    path_label = "direct_kernel" if use_direct else "inference_params"
    log0(
        f"  carryover_eval: path={path_label}, block_len={block_len}, "
        f"overlap={overlap}, score_region={score_region}, "
        f"n_blocks/rank={n_blocks}, global_blocks={n_blocks_total}, warmup_blocks={warmup_blocks}"
    )

    for b_idx in range(n_blocks):
        global_b_idx = block_start_idx + b_idx
        # Block starts at global_b_idx * score_region (so consecutive blocks
        # overlap by `overlap` tokens). We need block_len + 1 tokens total:
        # block_len input tokens plus the next-token target.
        pos = rank_start + b_idx * score_region
        chunk = val_tokens[pos : pos + block_len + 1]
        if chunk.size < block_len + 1:
            break
        xn = chunk[:-1].reshape(1, block_len)
        yn = chunk[1:].reshape(1, block_len)
        x = torch.from_numpy(xn).long().to(DEVICE, non_blocking=True)
        y = torch.from_numpy(yn).long().to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
            if use_direct:
                hidden, layer_states, seq_idx_base = core.forward_with_state(
                    x, layer_states, seq_idx_base=seq_idx_base
                )
            else:
                hidden = core(x, inference_params=inference_params)
            logits = core.output_logits_from_hidden(hidden)
            logits = core.add_copy_logits(logits, hidden, hidden, x, query_start=0)

            # Score only new positions. The first GLOBAL block must score from
            # position 0 so official cold-start tokens are counted; every later
            # block skips the overlap because those tokens were already scored.
            score_start = overlap if (overlap > 0 and global_b_idx > 0) else 0
            score_logits = logits[:, score_start:, :]
            score_y = y[:, score_start:]
            loss = F.cross_entropy(
                score_logits.float().reshape(-1, score_logits.shape[-1]),
                score_y.reshape(-1),
                reduction="sum",
            )

        if b_idx >= warmup_blocks:
            cnt = float(score_y.numel())
            loss_sum += float(loss.detach().float().item())
            # Bytes accounting also restricted to the positions actually scored.
            p_flat = xn[:, score_start:].reshape(-1)
            t_flat = yn[:, score_start:].reshape(-1)
            b_arr = bb[t_flat].astype(np.int16, copy=True)
            b_arr += (hs[t_flat] & ~ib[p_flat]).astype(np.int16)
            tok_sum += cnt
            byt_sum += float(b_arr.astype(np.float64).sum())

        if (b_idx + 1) % log_every_blocks == 0:
            pct = 100.0 * (b_idx + 1) / n_blocks
            elapsed_e = time.perf_counter() - t0_eval
            eta = elapsed_e / (b_idx + 1) * (n_blocks - (b_idx + 1))
            log0(f"  carryover_eval: {pct:.0f}% ({b_idx + 1}/{n_blocks} blocks, eta:{eta:.0f}s)")

    elapsed = time.perf_counter() - t0_eval
    log0(
        f"  carryover_eval: rank={RANK} scored={n_scored_blocks} blocks "
        f"(warmup={warmup_blocks}), block_len={block_len}, path={path_label}, {elapsed:.1f}s"
    )

    stats = torch.tensor([loss_sum, tok_sum, byt_sum], device=DEVICE, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, tok_sum, byt_sum = float(stats[0]), float(stats[1]), float(stats[2])
    val_loss = loss_sum / max(tok_sum, 1.0)
    bpt = val_loss / math.log(2.0)

    if prev_loop:
        core.loop_enabled_external = True
    core.train()

    return float(val_loss), float(bpt * (tok_sum / max(byt_sum, 1.0)))


# -----------------------------------------------------------------------------
# LoRA infrastructure for legal score-first TTT
# -----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """
    LoRA adapter wrapping an existing nn.Linear without changing its semantics
    at init. Output is `linear(x) + alpha/r * (x @ A^T @ B^T)`. At init B=0
    so the wrapped output exactly matches the base linear.

    The base linear's weight is held *frozen* via a registered buffer ref;
    only the LoRA A, B matrices have requires_grad=True (per LORA_FROZEN_BASE
    convention).
    """
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_linear)}")
        self.base = base_linear  # not registered as submodule param-wise — we freeze it externally
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        # A: (rank, in_features) — initialized small random
        # B: (out_features, rank) — initialized zero so initial output unchanged
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero

    @property
    def weight(self):
        # Expose an effective weight so modules that read `.weight` directly
        # do not crash if accidentally wrapped. Gradients still flow to LoRA.
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A).to(self.base.weight.dtype)

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        # Use the effective weight path. This is slightly more expensive than
        # two matmuls but is robust for modules that expect .weight/.bias.
        return F.linear(x, self.weight, self.bias)

    def reset_lora(self):
        """Zero out B to restore base-only behavior. A is left as-is."""
        with torch.no_grad():
            self.lora_B.zero_()


def install_lora_adapters(core: nn.Module, rank: int, alpha: float,
                           target_substrings=("qkv", "out_proj", "gate_up", "down", "in_proj")):
    """
    Walk the model and replace selected nn.Linear modules with LoRALinear
    wrappers. Returns (lora_params, n_replaced). Base weights are frozen
    (requires_grad=False) — only LoRA A,B have grads.

    target_substrings: any module name *segment* matching one of these gets
      LoRA wrapped. Defaults match attention qkv/out_proj, FFN gate_up/down,
      Mamba internals are skipped by default; enable SCORE_FIRST_TTT_WRAP_MAMBA=1
      only for a smoke-tested ablation.
    """
    # Snapshot the modules to replace (don't mutate during iteration)
    to_replace = []  # list of (parent_module, attr_name, base_linear)
    for module_name, module in core.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            # Match if any target substring appears as a segment (last component)
            # AND the linear is not the lm_head/tok_emb/embed projections.
            if "lm_head" in full_name or "tok_emb" in full_name or "embed_proj" in full_name:
                continue
            # Critical crash fix: mamba_ssm.Mamba2.forward reads in_proj/out_proj.weight
            # directly in its fused path. The LoRALinear properties above make accidental
            # wrapping survivable, but by default we keep Mamba internals unwrapped because
            # it is slower and was the source of the previous run's AttributeError.
            if (not SCORE_FIRST_TTT_WRAP_MAMBA) and (".mamba." in full_name or full_name.startswith("mamba.")):
                continue
            if any(t == child_name for t in target_substrings):
                to_replace.append((module, child_name, child, full_name))

    n_replaced = 0
    lora_params = []
    for parent, child_name, base_linear, full_name in to_replace:
        wrapper = LoRALinear(base_linear, rank=rank, alpha=alpha)
        # Move adapter to the same device, but keep LoRA A/B in fp32 for
        # AdamW stability. The forward path casts the effective delta to the
        # base weight dtype.
        wrapper = wrapper.to(device=base_linear.weight.device)
        setattr(parent, child_name, wrapper)
        n_replaced += 1
        lora_params.append(wrapper.lora_A)
        lora_params.append(wrapper.lora_B)

    # Freeze all non-LoRA params
    for p in core.parameters():
        p.requires_grad_(False)
    # Re-enable grads on LoRA params
    for p in lora_params:
        p.requires_grad_(True)

    return lora_params, n_replaced


def uninstall_lora_adapters(core: nn.Module):
    """
    Restore base nn.Linear modules in place of LoRALinear wrappers and
    re-enable grads on all params. Called after TTT eval to leave the
    model in its pre-eval state for any subsequent operations (compression,
    further evals, etc.).
    """
    to_restore = []
    for module_name, module in core.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                to_restore.append((module, child_name, child))
    for parent, child_name, wrapper in to_restore:
        setattr(parent, child_name, wrapper.base)
    for p in core.parameters():
        p.requires_grad_(True)
    return len(to_restore)


def reset_all_lora(core: nn.Module):
    """Zero out all lora_B in the model (resets to base-only behavior).
    Used between eval phases or to clear adapter state."""
    n = 0
    for m in core.modules():
        if isinstance(m, LoRALinear):
            m.reset_lora()
            n += 1
    return n


# -----------------------------------------------------------------------------


def eval_score_first_ttt(model_or_ddp, val_tokens, bb, hs, ib, stride=None):
    """
    Legal score-first TTT with LoRA/adapters.

    For each rank-local validation chunk:
      1. Score each target token exactly once under no_grad.
      2. Train adapters only on those already-scored tail tokens.
      3. Carry adapter weights forward to the next chunk.

    This fixes the previous overweighting bug where TTT trained on the full
    overlapping windows even though only the scored tail tokens should adapt.
    """
    if stride is None:
        stride = EVAL_STRIDE

    core = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp
    seq_len = SEQ_LEN

    if SCORE_FIRST_TTT_USE_LORA:
        lora_params, n_replaced = install_lora_adapters(
            core,
            rank=SCORE_FIRST_TTT_LORA_RANK,
            alpha=SCORE_FIRST_TTT_LORA_ALPHA,
            target_substrings=SCORE_FIRST_TTT_TARGETS,
        )
        adapt_params = lora_params
        n_adapt = sum(p.numel() for p in adapt_params)
        log0(
            f"Score-First TTT (LoRA): rank={SCORE_FIRST_TTT_LORA_RANK} "
            f"alpha={SCORE_FIRST_TTT_LORA_ALPHA} replaced={n_replaced} linears, "
            f"adapting {n_adapt:,} LoRA params, targets={SCORE_FIRST_TTT_TARGETS}, "
            f"opt={SCORE_FIRST_TTT_OPT}, lr={SCORE_FIRST_TTT_LR}, "
            f"wd={SCORE_FIRST_TTT_WD}, all_reduce={'NO' if SCORE_FIRST_TTT_NO_ALLREDUCE else 'YES'}"
        )
        base_state = None
    else:
        adapt_params = list(core.parameters())
        n_adapt = sum(p.numel() for p in adapt_params)
        log0(
            f"Score-First TTT (FULL): chunk={SCORE_FIRST_TTT_CHUNK_TOKENS}, "
            f"lr={SCORE_FIRST_TTT_LR}, momentum={SCORE_FIRST_TTT_MOMENTUM}, "
            f"epochs={SCORE_FIRST_TTT_EPOCHS}, adapting {n_adapt:,} params"
        )
        base_state = {name: p.data.clone() for name, p in core.named_parameters()}

    if SCORE_FIRST_TTT_OPT == "adamw":
        ttt_opt = torch.optim.AdamW(
            adapt_params,
            lr=SCORE_FIRST_TTT_LR,
            betas=(0.9, SCORE_FIRST_TTT_BETA2),
            weight_decay=SCORE_FIRST_TTT_WD,
        )
    else:
        ttt_opt = torch.optim.SGD(adapt_params, lr=SCORE_FIRST_TTT_LR,
                                  momentum=SCORE_FIRST_TTT_MOMENTUM)

    total_tokens = val_tokens.size - 1
    chunk_tokens = SCORE_FIRST_TTT_CHUNK_TOKENS
    n_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens
    chunk_start_idx, chunk_end_idx = _rank_bounds(n_chunks)
    local_chunks = chunk_end_idx - chunk_start_idx
    batch_windows = max(1, 131072 // seq_len)
    train_batch_windows = max(1, batch_windows // 2)

    loss_sum = 0.0
    tok_sum = 0.0
    byt_sum = 0.0
    t0_eval = time.perf_counter()
    log_every_chunks = max(1, local_chunks // 10)

    try:
        for chunk_idx_local, ci in enumerate(range(chunk_start_idx, chunk_end_idx)):
            score_start = ci * chunk_tokens
            score_end = min(score_start + chunk_tokens, total_tokens)
            if score_end <= score_start:
                continue

            records = list(_sliding_score_records(total_tokens, seq_len, stride, score_start, score_end))
            train_batches = []

            core.eval()
            with torch.no_grad():
                for xn, yn, tail_start, tail_len in _iter_sliding_score_batches(
                        val_tokens, records, seq_len, batch_windows):
                    x = torch.from_numpy(xn).long().to(DEVICE, non_blocking=True)
                    y = torch.from_numpy(yn).long().to(DEVICE, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
                        hidden = core(x, state=None, return_state=False)
                        hidden_tail = hidden[:, tail_start : tail_start + tail_len, :]
                        y_tail = y[:, tail_start : tail_start + tail_len]
                        logits = core.output_logits_from_hidden(hidden_tail)
                        logits = core.add_copy_logits(logits, hidden_tail, hidden, x, query_start=tail_start)
                        loss = F.cross_entropy(logits.float(), y_tail.reshape(-1), reduction="sum")

                    cnt = float(y_tail.numel())
                    loss_sum += float(loss.detach().float().item())
                    p_tail = xn[:, tail_start : tail_start + tail_len].reshape(-1)
                    t_tail = yn[:, tail_start : tail_start + tail_len].reshape(-1)
                    b = bb[t_tail].astype(np.int16, copy=True)
                    b += (hs[t_tail] & ~ib[p_tail]).astype(np.int16)
                    tok_sum += cnt
                    byt_sum += float(b.astype(np.float64).sum())
                    train_batches.append((xn, yn, int(tail_start), int(tail_len)))

            if train_batches:
                core.train()
                chunk_lr_scale = 0.5 * (1.0 + math.cos(math.pi * chunk_idx_local / max(1, local_chunks)))
                for g in ttt_opt.param_groups:
                    g["lr"] = SCORE_FIRST_TTT_LR * chunk_lr_scale

                for _epoch in range(SCORE_FIRST_TTT_EPOCHS):
                    order = np.random.permutation(len(train_batches))
                    for batch_i in order:
                        bx, by, tail_start, tail_len = train_batches[int(batch_i)]
                        for bi in range(0, bx.shape[0], train_batch_windows):
                            be = min(bi + train_batch_windows, bx.shape[0])
                            ax = torch.from_numpy(bx[bi:be]).long().to(DEVICE, non_blocking=True)
                            ay = torch.from_numpy(by[bi:be]).long().to(DEVICE, non_blocking=True)

                            ttt_opt.zero_grad(set_to_none=True)
                            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
                                adapt_loss = lm_loss_tail(core, ax, ay, tail_start, tail_len)
                            adapt_loss.backward()

                            if not SCORE_FIRST_TTT_NO_ALLREDUCE:
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
                eta = elapsed_e / (chunk_idx_local + 1) * max(0, local_chunks - chunk_idx_local - 1)
                log0(f"  score_first_ttt: chunk {chunk_idx_local+1}/{local_chunks} "
                     f"running_bpb:{running_bpb:.4f} eta:{eta:.0f}s")

        elapsed = time.perf_counter() - t0_eval
        log0(f"  score_first_ttt: {local_chunks} chunks/rank, "
             f"{tok_sum:.0f} scored tokens, {elapsed:.1f}s")

        stats = torch.tensor([loss_sum, tok_sum, byt_sum], device=DEVICE, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum, tok_sum, byt_sum = float(stats[0]), float(stats[1]), float(stats[2])
        val_loss = loss_sum / max(tok_sum, 1.0)
        bpt = val_loss / math.log(2.0)
        return float(val_loss), float(bpt * (tok_sum / max(byt_sum, 1.0)))

    finally:
        if SCORE_FIRST_TTT_USE_LORA:
            n_uninstalled = uninstall_lora_adapters(core)
            log0(f"  score_first_ttt: uninstalled {n_uninstalled} LoRA adapters, base weights restored")
        else:
            with torch.no_grad():
                for name, p in core.named_parameters():
                    if name in base_state:
                        p.data.copy_(base_state[name])
        core.train()


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
    # Each rank processes an exact contiguous shard of val sequences.
    seq_start, seq_end = _rank_bounds(total_seqs)
    local_seqs = seq_end - seq_start

    batch_seqs = max(1, min(TTT_GLOBAL_BATCH_SEQS, max(local_seqs, 1)))
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

    # Distribute val sequences across ranks with exact remainder handling.
    total_seqs = (val_tokens.size - 1) // seq_len
    seq_start, seq_end = _rank_bounds(total_seqs)

    # Subsample if TTT_MAX_SEQS is set (0 = use all)
    n_local_seqs = seq_end - seq_start
    local_total_seqs = n_local_seqs
    if TTT_MAX_SEQS > 0 and n_local_seqs > TTT_MAX_SEQS:
        stride = n_local_seqs // TTT_MAX_SEQS
        seq_indices = list(range(seq_start, seq_end, stride))[:TTT_MAX_SEQS]
        n_local_seqs = len(seq_indices)
    else:
        seq_indices = list(range(seq_start, seq_end))
        n_local_seqs = len(seq_indices)

    log0(f"TTT per-seq: adapting {len(adapt_params)} params ({TTT_PARAMS}), "
         f"steps={TTT_STEPS}, lr={TTT_LR}, "
         f"seqs/rank={n_local_seqs} (of {local_total_seqs})")

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
# Compression / quantization (int6 packed + Brotli-11)
# -----------------------------------------------------------------------------
def _quantize_row_sdclip(t: torch.Tensor, k: float, half_levels: int):
    """Per-row symmetric quantization with std-based clipping.

    t: (rows, cols) float
    k: clipping coefficient (clip = k * std(row))
    half_levels: max abs value (e.g., 31 for int6, 127 for int8)
    Returns: (q int8 tensor, scale fp16 tensor)
    """
    t32 = t.float()
    row_std = t32.std(dim=1).clamp_min(1e-8)
    clip = (k * row_std).clamp_min(1e-8)  # (rows,)
    scale = (clip / half_levels).clamp_min(1e-8)
    clipped = torch.clamp(t32, -clip[:, None], clip[:, None])
    q = torch.round(clipped / scale[:, None]).clamp(-half_levels, half_levels).to(torch.int8)
    return q.contiguous(), scale.to(torch.float16).contiguous()


def _quantize_row_optclip(t: torch.Tensor, half_levels: int,
                           k_grid=(2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0)):
    """Per-row optimal-clip quantization (lite-GPTQ for embeddings).

    For each row, sweep k_grid and pick the k that minimizes L2 reconstruction
    error. The dominant quant-error driver for embedding tensors is rare-token
    rows whose magnitude distribution differs from common-token rows; a single
    global clip hurts them disproportionately. This captures most of GPTQ's
    benefit for embeddings without needing calibration data.

    NOT full GPTQ (no Hessian, no per-column error compensation). For
    embeddings specifically the dominant error term is per-row magnitude
    calibration, which this captures.
    """
    t32 = t.float()
    rows, cols = t32.shape
    row_std = t32.std(dim=1).clamp_min(1e-8)
    row_abs_max = t32.abs().max(dim=1).values.clamp_min(1e-8)

    best_err = torch.full((rows,), float("inf"), device=t32.device)
    best_scale = torch.zeros((rows,), device=t32.device)
    best_q = torch.zeros_like(t32, dtype=torch.int8)

    for k in k_grid:
        clip = (k * row_std).clamp_min(1e-8)
        # Cap clip at row_abs_max so we don't waste levels on impossible values
        clip = torch.minimum(clip, row_abs_max)
        scale = (clip / half_levels).clamp_min(1e-8)
        clipped = torch.clamp(t32, -clip[:, None], clip[:, None])
        q = torch.round(clipped / scale[:, None]).clamp(-half_levels, half_levels)
        recon = q * scale[:, None]
        err = ((recon - t32) ** 2).sum(dim=1)
        improved = err < best_err
        best_err = torch.where(improved, err, best_err)
        best_scale = torch.where(improved, scale, best_scale)
        improved_mask = improved[:, None].expand_as(t32)
        best_q = torch.where(improved_mask, q.to(torch.int8), best_q)

    return best_q.contiguous(), best_scale.to(torch.float16).contiguous()


def _pack_int6(q_int8: torch.Tensor):
    """Pack int8-stored int6 values (range [-31, 31]) into uint8.
    4 values × 6 bits = 24 bits = 3 bytes."""
    flat = q_int8.flatten().to(torch.int32) + 32  # shift to [0, 63]
    n = flat.numel()
    pad = (-n) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int32, device=flat.device)])
    g = flat.view(-1, 4)
    a, b, c, d = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    byte0 = ((a << 2) | (b >> 4)) & 0xFF
    byte1 = (((b & 0xF) << 4) | (c >> 2)) & 0xFF
    byte2 = (((c & 0x3) << 6) | d) & 0xFF
    packed = torch.stack([byte0, byte1, byte2], dim=1).flatten().to(torch.uint8)
    return packed.contiguous(), n


def _unpack_int6(packed_u8: torch.Tensor, n_orig: int, shape):
    """Inverse of _pack_int6."""
    arr = packed_u8.to(torch.int32)
    g = arr.view(-1, 3)
    byte0, byte1, byte2 = g[:, 0], g[:, 1], g[:, 2]
    a = byte0 >> 2
    b = ((byte0 & 0x3) << 4) | (byte1 >> 4)
    c = ((byte1 & 0xF) << 2) | (byte2 >> 6)
    d = byte2 & 0x3F
    unpacked = torch.stack([a, b, c, d], dim=1).flatten()[:n_orig]
    signed = unpacked.to(torch.int32) - 32
    return signed.view(shape).to(torch.int8)


def _is_embedding_tensor(name: str) -> bool:
    """Embedding-like tensors get int8 (more sensitive to quantization);
    everything else gets int6."""
    n = name
    return (
        "tok_emb" in n
        or "lm_head" in n
        or "embed_proj" in n
        or ("bigram" in n and "embed" in n)
    )


def _compress_best(raw: bytes):
    """Try Brotli-11 (best); fall back to zstd-22; final fallback zlib-9."""
    if _HAS_BROTLI:
        try:
            return brotli.compress(raw, quality=11), "brotli11"
        except Exception:
            pass
    if "zstandard" in sys.modules:
        try:
            cctx = sys.modules["zstandard"].ZstdCompressor(level=22)
            return cctx.compress(raw), "zstd22"
        except Exception:
            pass
    return zlib.compress(raw, 9), "zlib9"


def _decompress(blob: bytes, scheme: str) -> bytes:
    if scheme == "brotli11":
        return brotli.decompress(blob)
    if scheme == "zstd22":
        return sys.modules["zstandard"].ZstdDecompressor().decompress(blob)
    return zlib.decompress(blob)


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
    _STOP_CHECK_EVERY = int(os.environ.get("STOP_CHECK_EVERY", "10"))  # keep training inside 600s; 100-step checks overshot by ~8s
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

    # --- State-carryover eval (SSM-native) ---
    # Threads Mamba2 hidden state across non-overlapping val blocks via the
    # direct mamba_chunk_scan_combined kernel (with initial_states /
    # return_final_states). Falls back to InferenceParams path for diagnosis
    # if direct kernel unavailable. Compare against the sliding-window
    # number above to see whether long-prefix recurrent state actually helps.
    pre_ttt_co_vb = None
    have_carry_path = (CARRYOVER_DIRECT_KERNEL and _HAS_CHUNK_SCAN and rearrange is not None) \
                       or (InferenceParams is not None)
    if CARRYOVER_EVAL_ENABLED and have_carry_path:
        dist.barrier()
        log0("=" * 60)
        log0(f"Running state-carryover eval (block_len={CARRYOVER_BLOCK_LEN}, warmup={CARRYOVER_WARMUP_TOKENS} tokens)...")
        try:
            co_vl, co_vb = eval_val_carryover(
                model, val_tokens, bb, hs, ib,
                block_len=CARRYOVER_BLOCK_LEN,
                warmup_tokens=CARRYOVER_WARMUP_TOKENS,
            )
            log0(f"Carryover val_loss:{co_vl:.4f} val_bpb:{co_vb:.4f}")
            log0(f"Carryover vs standard:    bpb={best_vb - co_vb:+.4f}")
            if pre_ttt_sw_vb is not None:
                log0(f"Carryover vs sliding:     bpb={pre_ttt_sw_vb - co_vb:+.4f}")
            pre_ttt_co_vb = co_vb
        except Exception as e:
            log0(f"[carryover_eval] FAILED: {type(e).__name__}: {e}")
            import traceback
            log0(traceback.format_exc())
            log0("[carryover_eval] continuing — sliding-window number above is unaffected")
        log0("=" * 60)
    elif CARRYOVER_EVAL_ENABLED:
        log0("[carryover_eval] skipped: no carryover path available (need either mamba_chunk_scan_combined or InferenceParams)")

    # --- Score-First TTT (SOTA-style legal eval-time adaptation) ---
    if RUN_PREQUANT_SCORE_FIRST_TTT and SCORE_FIRST_TTT_ENABLED and EVAL_STRIDE < SEQ_LEN:
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
    # Quantization + compression + roundtrip validation
    # int6 GPTQ-lite (per-row SDClip) for matrices, int8 for embeddings,
    # fp16 passthrough for small/sensitive. Brotli-11 with zstd fallback.
    # -----------------------------------------------------------------------
    # Load best weights for quantization
    if best_source == "swa" and swa is not None:
        swa.apply_average(base_model)
    elif best_source == "ema" and ema is not None:
        ema.apply_shadow(base_model)

    if not RUN_POSTQUANT:
        log0("Skipping quantization/post-quant eval/artifact because RUN_POSTQUANT=0")
        for _ldr in (locals().get("loader_short", None), locals().get("loader_long", None)):
            try:
                if _ldr is not None:
                    _ldr.shutdown()
            except Exception:
                pass
        return

    log0(f"Quantizing model (mode={QUANT_MODE}, k_mat={QUANT_K_MATRIX}, k_emb={QUANT_K_EMBED})...")
    state_dict = base_model.state_dict()

    # Dynamics-protected quantization (PR #1890 / Q-Mamba ICLR 2025).
    # Mamba2.in_proj output is split into [z, x, B, C, dt]; the last `nheads`
    # rows of in_proj.weight produce dt (the SSM time-step generator).
    # Q-Mamba shows that uniform 6-bit PTQ destabilizes the SSM recurrence
    # because errors in dt (and consequently A_bar = exp(dt*A)) compound over
    # many tokens. Promoting just these dt rows to INT8 costs ~16 extra bytes
    # per packed row (about 0.01 MB total at our scale) and recovers most of
    # the post-quant BPB regression.
    #
    # We derive the row range from the live module's .nheads attribute rather
    # than hardcoding indices, so this works across configs with different
    # ngroups, d_state, headdim, etc. The map is keyed by the parameter name
    # we'd see in state_dict (e.g. "blocks.0.mamba.in_proj.weight").
    dt_row_map = {}  # name -> (dt_start, dt_end) in row indexing
    if QUANT_PROTECT_DYNAMICS:
        for mod_name, mod in base_model.named_modules():
            # mamba_ssm Mamba2 modules expose .nheads, .in_proj
            if hasattr(mod, "in_proj") and hasattr(mod, "nheads"):
                w = getattr(mod.in_proj, "weight", None)
                if w is None or w.ndim != 2:
                    continue
                total_rows = w.shape[0]
                nheads = int(mod.nheads)
                if nheads <= 0 or nheads >= total_rows:
                    continue
                weight_name = f"{mod_name}.in_proj.weight"
                dt_row_map[weight_name] = (total_rows - nheads, total_rows)
        log0(
            f"  dynamics-protected rows: {len(dt_row_map)} in_proj tensors flagged "
            f"({sum(end-start for start,end in dt_row_map.values())} total dt rows -> int8)"
        )

    # Stats counters
    n_int6 = n_int8 = n_passthrough = n_alias = 0
    bytes_int6 = bytes_int8 = bytes_passthrough = 0

    quant_entries = {}    # name -> {"mode", ...} or {"mode": "alias", "target": canonical_name}
    # Tied-weight dedup keyed by the ORIGINAL tensor's data_ptr/storage. We
    # must NOT key on the CPU copy's data_ptr — CPU buffers get freed and
    # reused across iterations, causing false-positive aliases between
    # unrelated tensors. Original state_dict tensors share storage with
    # live model parameters, so their data_ptrs are stable for the duration
    # of this loop.
    seen_ptrs = {}

    for name, t_orig in state_dict.items():
        # Check aliasing on the original tensor BEFORE making any copies.
        # Combine data_ptr with shape to be doubly safe — two tensors that
        # share storage AND have the same shape are genuine aliases (e.g.
        # tied lm_head.weight ↔ tok_emb.weight). Different shapes at the
        # same ptr would indicate a view, not a tied weight.
        ptr_key = (t_orig.data_ptr(), tuple(t_orig.shape))
        if ptr_key in seen_ptrs:
            quant_entries[name] = {"mode": "alias", "target": seen_ptrs[ptr_key]}
            n_alias += 1
            continue
        seen_ptrs[ptr_key] = name

        t = t_orig.detach().cpu().contiguous()

        # Non-float or small/sensitive: keep as fp16 (or original int)
        if not t.is_floating_point():
            quant_entries[name] = {"mode": "raw", "tensor": t}
            n_passthrough += 1
            bytes_passthrough += t.numel() * t.element_size()
            continue

        if t.numel() <= QUANT_PASSTHROUGH_NUMEL or t.ndim != 2:
            t16 = t.to(torch.float16).contiguous()
            quant_entries[name] = {"mode": "fp16", "tensor": t16}
            n_passthrough += 1
            bytes_passthrough += t16.numel() * 2
            continue

        # 2D float, large: quantize
        is_embed = _is_embedding_tensor(name)
        if QUANT_MODE == "int8" or is_embed:
            # int8 with SDClip — embedding-like tensors get optimal-clip
            # search (GPTQ-lite) when enabled, otherwise the single global k.
            if is_embed and QUANT_OPTCLIP_EMBED:
                q, scale = _quantize_row_optclip(t, half_levels=127)
            else:
                q, scale = _quantize_row_sdclip(t, k=QUANT_K_EMBED if is_embed else QUANT_K_MATRIX, half_levels=127)
            # Store as raw int8 bytes (no need to pack — one byte per value)
            quant_entries[name] = {
                "mode": "int8",
                "packed": q.numpy().tobytes(),
                "scale": scale,
                "shape": tuple(t.shape),
                "numel": int(t.numel()),
            }
            n_int8 += 1
            bytes_int8 += q.numel()
        elif name in dt_row_map:
            # Dynamics-protected int6+int8 split: low rows int6, dt rows int8.
            dt_start, dt_end = dt_row_map[name]
            t_low = t[:dt_start].contiguous()        # z, x, B, C rows -> int6
            t_dt = t[dt_start:dt_end].contiguous()   # dt rows -> int8

            q_low, scale_low = _quantize_row_sdclip(t_low, k=QUANT_K_MATRIX, half_levels=31)
            packed_low, n_orig_low = _pack_int6(q_low)

            q_dt, scale_dt = _quantize_row_sdclip(t_dt, k=QUANT_K_MATRIX, half_levels=127)

            quant_entries[name] = {
                "mode": "mixed_int6_int8",
                "packed_low": packed_low.numpy().tobytes(),
                "scale_low": scale_low,
                "shape_low": tuple(t_low.shape),
                "numel_low": int(n_orig_low),
                "packed_dt": q_dt.numpy().tobytes(),
                "scale_dt": scale_dt,
                "shape_dt": tuple(t_dt.shape),
                "shape_full": tuple(t.shape),
            }
            n_int6 += 1  # count under int6 since the bulk is int6
            bytes_int6 += packed_low.numel()
            bytes_int8 += q_dt.numel()
        else:
            # int6 packed (4 vals per 3 bytes) with SDClip
            q, scale = _quantize_row_sdclip(t, k=QUANT_K_MATRIX, half_levels=31)
            packed, n_orig = _pack_int6(q)
            quant_entries[name] = {
                "mode": "int6",
                "packed": packed.numpy().tobytes(),
                "scale": scale,
                "shape": tuple(t.shape),
                "numel": int(n_orig),
            }
            n_int6 += 1
            bytes_int6 += packed.numel()

    log0(
        f"  quant tensors: int6={n_int6} ({bytes_int6/1e6:.2f}MB) "
        f"int8={n_int8} ({bytes_int8/1e6:.2f}MB) "
        f"fp16/raw={n_passthrough} ({bytes_passthrough/1e6:.2f}MB) "
        f"alias={n_alias}"
    )
    raw_total = bytes_int6 + bytes_int8 + bytes_passthrough
    log0(f"  raw quantized total before compression: {raw_total/1e6:.2f}MB")

    # Serialize and compress
    import io
    quant_buf = io.BytesIO()
    torch.save(quant_entries, quant_buf)
    quant_raw = quant_buf.getvalue()

    quant_blob, scheme = _compress_best(quant_raw)
    compressed_bytes = len(quant_blob)
    log0(
        f"Compressed model ({scheme}): {compressed_bytes:,} bytes "
        f"({compressed_bytes / 1024 / 1024:.2f} MB) — "
        f"{'UNDER' if compressed_bytes <= 16_000_000 else 'OVER'} 16MB cap"
    )

    # Roundtrip: decompress, dequantize, reload, and re-evaluate
    quant_raw_rt = _decompress(quant_blob, scheme)
    quant_entries_rt = torch.load(io.BytesIO(quant_raw_rt), map_location="cpu")

    dequant_state = {}
    # First pass: dequantize non-alias entries
    for name, entry in quant_entries_rt.items():
        mode = entry["mode"]
        if mode == "alias":
            continue  # second pass
        if mode == "raw":
            dequant_state[name] = entry["tensor"]
        elif mode == "fp16":
            dequant_state[name] = entry["tensor"].to(torch.bfloat16)
        elif mode == "int8":
            shape = entry["shape"]
            packed = np.frombuffer(entry["packed"], dtype=np.int8)
            q = torch.from_numpy(packed.copy()).view(shape)
            scale = entry["scale"].float()
            dq = q.float() * scale.view(shape[0], 1)
            dequant_state[name] = dq.to(torch.bfloat16)
        elif mode == "int6":
            shape = entry["shape"]
            n_orig = entry["numel"]
            packed_u8 = torch.from_numpy(np.frombuffer(entry["packed"], dtype=np.uint8).copy())
            q = _unpack_int6(packed_u8, n_orig, shape)
            scale = entry["scale"].float()
            dq = q.float() * scale.view(shape[0], 1)
            dequant_state[name] = dq.to(torch.bfloat16)
        elif mode == "mixed_int6_int8":
            # Dynamics-protected: low rows from int6, dt rows from int8.
            shape_low = entry["shape_low"]
            shape_dt = entry["shape_dt"]
            shape_full = entry["shape_full"]
            n_orig_low = entry["numel_low"]

            # Decode int6 part
            packed_u8 = torch.from_numpy(np.frombuffer(entry["packed_low"], dtype=np.uint8).copy())
            q_low = _unpack_int6(packed_u8, n_orig_low, shape_low)
            scale_low = entry["scale_low"].float()
            dq_low = q_low.float() * scale_low.view(shape_low[0], 1)

            # Decode int8 part
            packed_dt = np.frombuffer(entry["packed_dt"], dtype=np.int8)
            q_dt = torch.from_numpy(packed_dt.copy()).view(shape_dt)
            scale_dt = entry["scale_dt"].float()
            dq_dt = q_dt.float() * scale_dt.view(shape_dt[0], 1)

            # Stitch back to full shape
            dq = torch.cat([dq_low, dq_dt], dim=0)
            assert dq.shape == shape_full, f"mixed dequant shape mismatch: got {dq.shape}, expected {shape_full}"
            dequant_state[name] = dq.to(torch.bfloat16)
        else:
            raise RuntimeError(f"unknown quant mode: {mode}")
    # Second pass: resolve aliases
    for name, entry in quant_entries_rt.items():
        if entry["mode"] == "alias":
            dequant_state[name] = dequant_state[entry["target"]]

    base_model.load_state_dict(dequant_state, strict=True)
    q_vl, q_vb = eval_val(model, val_tokens, bb, hs, ib)
    log0(f"Post-quant val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f}")
    q_sw_vb = None
    if EVAL_STRIDE < SEQ_LEN:
        q_sw_vl, q_sw_vb = eval_val_sliding(model, val_tokens, bb, hs, ib, stride=EVAL_STRIDE)
        log0(f"Post-quant sliding val_loss:{q_sw_vl:.4f} val_bpb:{q_sw_vb:.4f}")

    # Final leaderboard-relevant diagnostic: dequantized artifact path +
    # legal score-first LoRA TTT. Reload dequant weights first so any earlier
    # eval state is clean; eval_score_first_ttt restores base weights after it.
    if SCORE_FIRST_TTT_ENABLED and EVAL_STRIDE < SEQ_LEN:
        dist.barrier()
        base_model.load_state_dict(dequant_state, strict=True)
        log0("=" * 60)
        log0(f"Running POST-QUANT Score-First TTT eval (chunk={SCORE_FIRST_TTT_CHUNK_TOKENS} tokens)...")
        q_sft_vl, q_sft_vb = eval_score_first_ttt(model, val_tokens, bb, hs, ib, stride=EVAL_STRIDE)
        log0(f"Post-quant Score-First TTT val_loss:{q_sft_vl:.4f} val_bpb:{q_sft_vb:.4f}")
        if q_sw_vb is not None:
            log0(f"Post-quant Score-First TTT vs sliding: bpb={q_sw_vb - q_sft_vb:+.4f}")
        log0("=" * 60)

    # Save compressed artifact
    if MASTER_PROCESS:
        ext = scheme  # brotli11 / zstd22 / zlib9
        run_tag = os.environ.get("RUN_TAG", "").strip()
        artifact_dir = os.environ.get("ARTIFACT_DIR", ".").strip() or "."
        os.makedirs(artifact_dir, exist_ok=True)
        prefix = f"final_model.{run_tag}." if run_tag else "final_model."
        artifact_path = os.path.join(artifact_dir, f"{prefix}{QUANT_MODE}.{ext}.bin")
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