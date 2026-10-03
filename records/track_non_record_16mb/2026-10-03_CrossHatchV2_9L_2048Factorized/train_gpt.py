#!/usr/bin/env python3
"""
Parameter Golf — Cross-Hatch V2

Non-record 16 MB submission trainer.

Cross-Hatch architecture:
    vocab      = 1024
    layers     = 9
    channels   = 128
    lanes      = 16
    model_dim  = 128 * 16 = 2048
    FFN        = 1408

This file intentionally preserves the successful Cross-Hatch model and
optimizer configuration while adapting the surrounding execution to the
current Parameter Golf distributed-training / validation conventions.

Target execution:
    torchrun --standalone --nproc_per_node=8 train_gpt.py

Default 8-GPU batch:
    8 GPUs
    64 sequences/GPU
    sequence length 1024
    grad accumulation 1
    global batch = 524,288 tokens

For an H100 memory fallback:
    MICRO_BATCH_SIZE=32
    GRAD_ACCUM=2

The training artifact is exported as FP16 without INT8/INT6/GPTQ.
"""

from __future__ import annotations

import glob
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP


# =============================================================================
# DISTRIBUTED / RUNTIME HELPERS
# =============================================================================

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

DIST_INITIALIZED = False


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def log0(*args, **kwargs) -> None:
    if RANK == 0:
        print(*args, **kwargs, flush=True)


def barrier() -> None:
    if is_distributed():
        dist.barrier()


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
#
# Environment variables are intentionally used so the same source can be:
#   1. run locally for smoke tests,
#   2. run on 8xH100 without source edits,
#   3. benchmark microbatch sizes safely.
#
# Core Cross-Hatch values are NOT meant to be tuned by the submission wrapper.
# =============================================================================

class Hyperparameters:
    # Data / tokenizer
    data_path = os.environ.get(
        "DATA_PATH",
        "./data/datasets/fineweb10B_sp1024",
    )
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH",
        "./data/tokenizers/fineweb_1024_bpe.model",
    )

    # Development may restrict the number of training shards.  For an official
    # execution, leave TRAIN_SHARDS unset so all available train shards are used.
    train_shards = int(os.environ.get("TRAIN_SHARDS", "0"))

    # Validation always uses all validation shards.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", "524288"))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", "1000"))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", "200"))

    # Training budget / schedule.
    iterations = int(os.environ.get("ITERATIONS", "20000"))
    max_wallclock_seconds = float(
        os.environ.get("MAX_WALLCLOCK_SECONDS", "600.0")
    )
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "20"))
    warmdown_start = float(os.environ.get("WARMDOWN_START", "0.72"))

    # Official target global batch.
    train_batch_tokens = int(
        os.environ.get("TRAIN_BATCH_TOKENS", "524288")
    )
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))

    # H100 default: 64 sequences/GPU, grad_accum=1.
    # If 64 does not fit on the target H100, use MICRO_BATCH_SIZE=32,
    # GRAD_ACCUM=2.
    micro_batch_size = int(
        os.environ.get("MICRO_BATCH_SIZE", "64")
    )
    requested_grad_accum = os.environ.get("GRAD_ACCUM", "")

    # Cross-Hatch architecture.
    vocab_size = int(os.environ.get("VOCAB_SIZE", "1024"))
    num_layers = int(os.environ.get("NUM_LAYERS", "9"))
    channel_dim = int(os.environ.get("CHANNEL_DIM", "128"))
    num_lanes = int(os.environ.get("NUM_LANES", "16"))
    model_dim = int(os.environ.get("MODEL_DIM", "2048"))
    ffn_dim = int(os.environ.get("FFN_DIM", "1408"))

    # Cross-Hatch attention / normalization.
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", "1.5"))
    rope_base = float(os.environ.get("ROPE_BASE", "10000.0"))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", "30.0"))

    attn_vector_mix = bool(
        int(os.environ.get("ATTN_VECTOR_MIX", "1"))
    )
    ffn_pre_mix = bool(
        int(os.environ.get("FFN_PRE_MIX", "0"))
    )
    ffn_post_mix = bool(
        int(os.environ.get("FFN_POST_MIX", "1"))
    )
    per_lane_scale = bool(
        int(os.environ.get("PER_LANE_SCALE", "1"))
    )

    # Successful optimizer settings.
    muon_lr = float(os.environ.get("MUON_LR", "0.025"))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", "0.05"))
    scalar_lr = float(os.environ.get("SCALAR_LR", "0.04"))

    muon_momentum = float(os.environ.get("MUON_MOMENTUM", "0.99"))
    muon_momentum_start = float(
        os.environ.get("MUON_MOMENTUM_START", "0.92")
    )
    muon_momentum_warmup = int(
        os.environ.get("MUON_MOMENTUM_WARMUP", "1500")
    )
    beta1 = float(os.environ.get("BETA1", "0.9"))
    beta2 = float(os.environ.get("BETA2", "0.95"))
    eps = float(os.environ.get("ADAM_EPS", "1e-8"))
    grad_clip = float(os.environ.get("GRAD_CLIP", "0.0"))

    # EMA is preserved from the successful Cross-Hatch experiment.
    ema_decay = float(os.environ.get("EMA_DECAY", "0.9965"))
    ema_start_frac = float(
        os.environ.get("EMA_START_FRAC", "0.72")
    )

    # Initialization.
    tied_embed_init_std = float(
        os.environ.get("TIED_EMBED_INIT_STD", "0.005")
    )

    # Compilation.
    use_compile = bool(int(os.environ.get("USE_COMPILE", "1")))
    compile_mode = os.environ.get("COMPILE_MODE", "max-autotune")

    # Reproducibility.
    seed = int(os.environ.get("SEED", "1337"))

    # Export.
    artifact_name = os.environ.get(
        "FP16_ARTIFACT_NAME",
        "final_model.fp16.pt",
    )


def resolve_grad_accum(args: Hyperparameters) -> int:
    per_micro_global = (
        WORLD_SIZE
        * args.micro_batch_size
        * args.train_seq_len
    )

    if args.requested_grad_accum:
        grad_accum = int(args.requested_grad_accum)
        if grad_accum <= 0:
            raise ValueError("GRAD_ACCUM must be positive.")
    else:
        if args.train_batch_tokens % per_micro_global != 0:
            raise ValueError(
                "The requested MICRO_BATCH_SIZE does not divide the global "
                "batch exactly. Choose an exact configuration. "
                f"global={args.train_batch_tokens}, "
                f"per_micro_global={per_micro_global}, "
                f"world={WORLD_SIZE}, "
                f"micro={args.micro_batch_size}, "
                f"seq={args.train_seq_len}"
            )
        grad_accum = args.train_batch_tokens // per_micro_global

    effective = per_micro_global * grad_accum
    if effective != args.train_batch_tokens:
        raise ValueError(
            "Effective global batch does not equal TRAIN_BATCH_TOKENS: "
            f"{effective} != {args.train_batch_tokens}"
        )

    return grad_accum


# =============================================================================
# PATH RESOLUTION
# =============================================================================

def find_repo_root(script_dir: Path) -> Path:
    root = script_dir.resolve()
    for _ in range(12):
        if (root / "data" / "datasets").exists():
            return root
        if root.parent == root:
            break
        root = root.parent

    # If the submission directory is nested under records/, the parent
    # repository root is normally still discoverable from here.
    root = script_dir.resolve()
    for _ in range(12):
        if (root / ".git").exists():
            return root
        if root.parent == root:
            break
        root = root.parent

    return script_dir.resolve()


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(SCRIPT_DIR)
ARGS = Hyperparameters()

DATA_DIR = Path(
    os.environ.get(
        "DATA_PATH",
        str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"),
    )
)
TOKENIZER_PATH = Path(
    os.environ.get(
        "TOKENIZER_PATH",
        str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"),
    )
)

TRAIN_PATTERN = str(DATA_DIR / "fineweb_train_*.bin")
VAL_PATTERN = str(DATA_DIR / "fineweb_val_*.bin")


# =============================================================================
# GPU / AMP SETUP
# =============================================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for Parameter Golf training.")

device = torch.device("cuda", LOCAL_RANK)
torch.cuda.set_device(device)

GPU_NAME = torch.cuda.get_device_name(LOCAL_RANK)
GPU_CC = torch.cuda.get_device_capability(LOCAL_RANK)

USE_BF16 = GPU_CC[0] >= 8
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
USE_GRAD_SCALER = not USE_BF16


# =============================================================================
# SEED
# =============================================================================

random.seed(ARGS.seed + RANK)
np.random.seed(ARGS.seed + RANK)
torch.manual_seed(ARGS.seed + RANK)
torch.cuda.manual_seed_all(ARGS.seed + RANK)


# =============================================================================
# DATA FORMAT
# =============================================================================

HEADER_BYTES = 256 * 4
MAGIC = 20240520
VERSION = 1


def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(
        file,
        dtype="<i4",
        count=256,
    )
    if (
        header.size != 256
        or int(header[0]) != MAGIC
        or int(header[1]) != VERSION
    ):
        raise ValueError(f"Unexpected shard header: {file}")

    num_tokens = int(header[2])
    expected_size = HEADER_BYTES + num_tokens * 2
    actual_size = file.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: "
            f"expected={expected_size}, actual={actual_size}"
        )

    tokens_np = np.fromfile(
        file,
        dtype="<u2",
        count=num_tokens,
        offset=HEADER_BYTES,
    )
    if tokens_np.size != num_tokens:
        raise ValueError(
            f"Short read for shard: {file}"
        )

    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    """Sequential FineWeb token stream, wrapping across available shards."""

    def __init__(self, pattern: str, max_shards: int = 0):
        files = sorted(glob.glob(pattern))
        if max_shards > 0:
            files = files[:max_shards]
        if not files:
            raise FileNotFoundError(
                f"No dataset shards found for pattern: {pattern}"
            )

        self.files = [Path(p) for p in files]
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (
            self.file_idx + 1
        ) % len(self.files)
        self.tokens = load_data_shard(
            self.files[self.file_idx]
        )
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n

        while remaining > 0:
            available = self.tokens.numel() - self.pos

            if available <= 0:
                self._advance_file()
                continue

            k = min(remaining, available)
            chunks.append(
                self.tokens[self.pos : self.pos + k]
            )
            self.pos += k
            remaining -= k

        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks)


class DistributedTokenLoader:
    """
    Each rank reads the same global stream position and takes a disjoint span.

    This mirrors the current repository's distributed token-stream convention:
    one optimizer step corresponds to TRAIN_BATCH_TOKENS global tokens.
    """

    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        max_shards: int = 0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.stream = TokenStream(
            pattern,
            max_shards=max_shards,
        )

    def next_batch(
        self,
        global_tokens: int,
        seq_len: int,
        grad_accum_steps: int,
    ) -> Tuple[Tensor, Tensor]:
        local_tokens = (
            global_tokens
            // (self.world_size * grad_accum_steps)
        )

        if local_tokens % seq_len != 0:
            raise ValueError(
                "Per-rank microbatch is not an integer number of sequences: "
                f"local_tokens={local_tokens}, seq_len={seq_len}"
            )

        per_rank_span = local_tokens + 1

        chunk = self.stream.take(
            per_rank_span * self.world_size
        )

        start = self.rank * per_rank_span
        local = chunk[
            start : start + per_rank_span
        ].to(dtype=torch.int64)

        x = local[:-1].reshape(
            -1,
            seq_len,
        )
        y = local[1:].reshape(
            -1,
            seq_len,
        )

        return (
            x.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
        )


# =============================================================================
# TOKENIZER / OFFICIAL BPB ACCOUNTING
# =============================================================================

sp = spm.SentencePieceProcessor(
    model_file=str(TOKENIZER_PATH)
)

if int(sp.vocab_size()) != ARGS.vocab_size:
    raise ValueError(
        f"VOCAB_SIZE={ARGS.vocab_size} does not match tokenizer "
        f"vocab_size={int(sp.vocab_size())}"
    )


def build_sentencepiece_luts(
    sp_model: spm.SentencePieceProcessor,
    vocab_size: int,
    dev: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp_model.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)

    base_bytes_np = np.zeros(
        (table_size,),
        dtype=np.int16,
    )
    has_leading_space_np = np.zeros(
        (table_size,),
        dtype=np.bool_,
    )
    is_boundary_token_np = np.ones(
        (table_size,),
        dtype=np.bool_,
    )

    for token_id in range(sp_vocab_size):
        if (
            sp_model.is_control(token_id)
            or sp_model.is_unknown(token_id)
            or sp_model.is_unused(token_id)
        ):
            continue

        is_boundary_token_np[token_id] = False

        if sp_model.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue

        piece = sp_model.id_to_piece(token_id)

        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]

        base_bytes_np[token_id] = len(
            piece.encode("utf-8")
        )

    return (
        torch.tensor(
            base_bytes_np,
            dtype=torch.int16,
            device=dev,
        ),
        torch.tensor(
            has_leading_space_np,
            dtype=torch.bool,
            device=dev,
        ),
        torch.tensor(
            is_boundary_token_np,
            dtype=torch.bool,
            device=dev,
        ),
    )


base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
    build_sentencepiece_luts(
        sp,
        ARGS.vocab_size,
        device,
    )
)


def load_validation_tokens(
    pattern: str,
    seq_len: int,
) -> Tensor:
    files = [
        Path(p)
        for p in sorted(glob.glob(pattern))
    ]
    if not files:
        raise FileNotFoundError(
            f"No validation files found for pattern: {pattern}"
        )

    tokens = torch.cat(
        [
            load_data_shard(file).to(
                dtype=torch.int64
            )
            for file in files
        ]
    ).contiguous()

    usable = (
        (tokens.numel() - 1)
        // seq_len
    ) * seq_len

    if usable <= 0:
        raise ValueError(
            "Validation split is too short."
        )

    return tokens[: usable + 1]


val_tokens = load_validation_tokens(
    VAL_PATTERN,
    ARGS.train_seq_len,
)


# =============================================================================
# MODEL
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(
                dim,
                dtype=torch.float32,
            )
        )
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(
            x,
            (x.size(-1),),
            self.weight.to(dtype=x.dtype),
            self.eps,
        )


class SharedLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        zero_init: bool = False,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                in_dim,
                out_dim,
                dtype=torch.float32,
            )
        )

        if zero_init:
            nn.init.zeros_(self.weight)
        else:
            nn.init.orthogonal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape

        x = x.reshape(
            -1,
            shape[-1],
        )

        y = x @ self.weight.to(
            dtype=x.dtype
        )

        return y.reshape(
            *shape[:-1],
            self.weight.size(1),
        )


class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        base: float,
    ):
        super().__init__()

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        pos = torch.arange(
            seq_len,
            dtype=torch.float32,
        )

        freq = torch.outer(
            pos,
            inv_freq,
        )

        self.register_buffer(
            "cos_cache",
            freq.cos()[None, None, :, :],
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            freq.sin()[None, None, :, :],
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        t = x.size(-2)

        cos = self.cos_cache[
            :, :, :t, :
        ].to(
            x.device,
            x.dtype,
        )
        sin = self.sin_cache[
            :, :, :t, :
        ].to(
            x.device,
            x.dtype,
        )

        xe = x[..., 0::2]
        xo = x[..., 1::2]

        ye = xe * cos - xo * sin
        yo = xe * sin + xo * cos

        return torch.stack(
            [ye, yo],
            dim=-1,
        ).flatten(-2)


class FactorizedAttention(nn.Module):
    def __init__(self):
        super().__init__()

        C = ARGS.channel_dim
        V = ARGS.num_lanes

        self.q = SharedLinear(C, C)
        self.k = SharedLinear(C, C)
        self.v = SharedLinear(C, C)
        self.o = SharedLinear(
            C,
            C,
            zero_init=True,
        )

        self.q_norm = RMSNorm(C)
        self.k_norm = RMSNorm(C)

        self.q_gain = nn.Parameter(
            torch.full(
                (V,),
                ARGS.qk_gain_init,
                dtype=torch.float32,
            )
        )

        if ARGS.attn_vector_mix:
            self.vector_mix = nn.Parameter(
                torch.eye(
                    V,
                    dtype=torch.float32,
                )
                + 0.01
                * torch.randn(
                    V,
                    V,
                )
            )
        else:
            self.vector_mix = None

        self.rope = RoPE(
            C,
            ARGS.train_seq_len,
            ARGS.rope_base,
        )

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, C, V = x.shape

        xf = x.reshape(
            bsz * seqlen * V,
            C,
        )

        q = (
            xf
            @ self.q.weight.to(
                dtype=x.dtype
            )
        ).reshape(
            bsz,
            seqlen,
            V,
            C,
        )

        k = (
            xf
            @ self.k.weight.to(
                dtype=x.dtype
            )
        ).reshape(
            bsz,
            seqlen,
            V,
            C,
        )

        v = (
            xf
            @ self.v.weight.to(
                dtype=x.dtype
            )
        ).reshape(
            bsz,
            seqlen,
            V,
            C,
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.permute(
            0,
            2,
            1,
            3,
        )
        k = k.permute(
            0,
            2,
            1,
            3,
        )
        v = v.permute(
            0,
            2,
            1,
            3,
        )

        q = self.rope(q)
        k = self.rope(k)

        q = (
            q
            * self.q_gain.to(
                q.dtype
            )[None, :, None, None]
        )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        y = y.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        if self.vector_mix is not None:
            y = torch.matmul(
                y,
                self.vector_mix.to(
                    y.dtype
                ),
            )

        y = y.permute(
            0,
            1,
            3,
            2,
        ).contiguous()

        yf = y.reshape(
            bsz * seqlen * V,
            C,
        )

        out = (
            yf
            @ self.o.weight.to(
                y.dtype
            )
        ).reshape(
            bsz,
            seqlen,
            V,
            C,
        )

        return out.permute(
            0,
            1,
            3,
            2,
        ).contiguous()


class CrossHatchSwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

        C = ARGS.channel_dim
        V = ARGS.num_lanes
        FFD = ARGS.ffn_dim

        self.gate = SharedLinear(C, FFD)
        self.up = SharedLinear(C, FFD)
        self.down = SharedLinear(
            FFD,
            C,
            zero_init=True,
        )

        if ARGS.ffn_pre_mix:
            self.pre_mix = nn.Parameter(
                torch.eye(
                    V,
                    dtype=torch.float32,
                )
                + 0.01
                * torch.randn(
                    V,
                    V,
                )
            )
        else:
            self.pre_mix = None

        if ARGS.ffn_post_mix:
            self.post_mix = nn.Parameter(
                torch.eye(
                    V,
                    dtype=torch.float32,
                )
                + 0.01
                * torch.randn(
                    V,
                    V,
                )
            )
        else:
            self.post_mix = None

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, C, V = x.shape

        if self.pre_mix is not None:
            x = torch.matmul(
                x,
                self.pre_mix.to(
                    x.dtype
                ),
            )

        xf = x.reshape(
            bsz * seqlen * V,
            C,
        )

        gate = (
            xf
            @ self.gate.weight.to(
                x.dtype
            )
        )
        up = (
            xf
            @ self.up.weight.to(
                x.dtype
            )
        )

        hidden = F.silu(gate) * up

        out = (
            hidden
            @ self.down.weight.to(
                x.dtype
            )
        ).reshape(
            bsz,
            seqlen,
            V,
            C,
        ).permute(
            0,
            1,
            3,
            2,
        ).contiguous()

        if self.post_mix is not None:
            out = torch.matmul(
                out,
                self.post_mix.to(
                    out.dtype
                ),
            )

        return out


class CrossHatchBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn_norm = RMSNorm(
            ARGS.model_dim
        )
        self.ffn_norm = RMSNorm(
            ARGS.model_dim
        )

        self.attn = FactorizedAttention()
        self.ffn = CrossHatchSwiGLU()

        self.attn_alpha = nn.Parameter(
            torch.ones(
                1,
                dtype=torch.float32,
            )
        )
        self.ffn_alpha = nn.Parameter(
            torch.ones(
                1,
                dtype=torch.float32,
            )
        )

        if ARGS.per_lane_scale:
            self.lane_scale = nn.Parameter(
                torch.ones(
                    ARGS.num_lanes,
                    dtype=torch.float32,
                )
            )
        else:
            self.lane_scale = None

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, C, V = x.shape

        z = self.attn_norm(
            x.reshape(
                bsz,
                seqlen,
                ARGS.model_dim,
            )
        ).reshape_as(x)

        if self.lane_scale is not None:
            z = (
                z
                * self.lane_scale.to(
                    z.dtype
                )[None, None, None, :]
            )

        x = (
            x
            + self.attn_alpha.to(
                z.dtype
            )
            * self.attn(z)
        )

        z = self.ffn_norm(
            x.reshape(
                bsz,
                seqlen,
                ARGS.model_dim,
            )
        ).reshape_as(x)

        if self.lane_scale is not None:
            z = (
                z
                * self.lane_scale.to(
                    z.dtype
                )[None, None, None, :]
            )

        x = (
            x
            + self.ffn_alpha.to(
                z.dtype
            )
            * self.ffn(z)
        )

        return x


class CrossHatchGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok_emb = nn.Embedding(
            ARGS.vocab_size,
            ARGS.model_dim,
        )

        self.input_norm = RMSNorm(
            ARGS.model_dim
        )

        self.blocks = nn.ModuleList(
            [
                CrossHatchBlock()
                for _ in range(ARGS.num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            ARGS.model_dim
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(
            self.tok_emb.weight,
            mean=0.0,
            std=ARGS.tied_embed_init_std,
        )

        for module in self.modules():
            if isinstance(module, SharedLinear):
                if not torch.all(
                    module.weight == 0
                ):
                    nn.init.orthogonal_(
                        module.weight
                    )
            elif isinstance(module, RMSNorm):
                nn.init.ones_(
                    module.weight
                )

        for block in self.blocks:
            nn.init.zeros_(
                block.attn.o.weight
            )
            nn.init.zeros_(
                block.ffn.down.weight
            )

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
    ) -> Tensor:
        bsz, seqlen = input_ids.shape

        x = self.tok_emb(
            input_ids
        ).to(
            AMP_DTYPE
        )

        x = self.input_norm(x)

        x = x.reshape(
            bsz,
            seqlen,
            ARGS.channel_dim,
            ARGS.num_lanes,
        )

        for block in self.blocks:
            x = block(x)

        h = self.final_norm(
            x.reshape(
                bsz,
                seqlen,
                ARGS.model_dim,
            )
        )

        logits = F.linear(
            h.float(),
            self.tok_emb.weight.float(),
        )

        logits = (
            ARGS.logit_softcap
            * torch.tanh(
                logits / ARGS.logit_softcap
            )
        )

        if targets is None:
            return logits

        return F.cross_entropy(
            logits.reshape(
                -1,
                ARGS.vocab_size,
            ),
            targets.reshape(-1),
        )


# =============================================================================
# PARAMETER / ARTIFACT AUDIT
# =============================================================================

def parameter_count(module: nn.Module) -> int:
    return sum(
        int(p.numel())
        for p in module.parameters()
    )


def fp16_parameter_bytes(module: nn.Module) -> int:
    return parameter_count(module) * 2


# =============================================================================
# MUON
# =============================================================================

def zeropower_ns5(
    G: Tensor,
    steps: int = 10,
    eps: float = 1e-7,
) -> Tensor:
    X = G.float()

    transposed = X.size(0) > X.size(1)

    if transposed:
        X = X.T

    X = X / (
        X.norm() + eps
    )

    a = 3.4445
    b = -4.7750
    c = 2.0315

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X


class Muon(torch.optim.Optimizer):
    """
    Minimal Muon implementation adapted to distributed parameter ownership.

    DDP gives each rank the same globally-averaged gradient.
    Each rank orthogonalizes a disjoint subset of matrix parameters, then
    all-reduces the assembled update so every rank applies the same full update.

    This preserves the optimizer logic while reducing redundant Muon state.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        momentum: float,
        momentum_start: float,
        warmup_steps: int,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            momentum_start=momentum_start,
            warmup_steps=warmup_steps,
        )
        super().__init__(
            params,
            defaults,
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = group["params"]

            group["step"] = (
                group.get("step", 0) + 1
            )
            step_num = group["step"]
            warmup = group["warmup_steps"]

            if warmup > 0:
                alpha = min(
                    1.0,
                    step_num / warmup,
                )
                momentum = (
                    group["momentum_start"]
                    + alpha
                    * (
                        group["momentum"]
                        - group["momentum_start"]
                    )
                )
            else:
                momentum = group["momentum"]

            lr = group["lr"]

            distributed = is_distributed()
            world_size = (
                dist.get_world_size()
                if distributed
                else 1
            )
            rank = (
                dist.get_rank()
                if distributed
                else 0
            )

            updates_flat = None

            if distributed and params:
                total_params = sum(
                    int(p.numel())
                    for p in params
                )
                updates_flat = torch.zeros(
                    total_params,
                    device=params[0].device,
                    dtype=torch.bfloat16,
                )

            cursor = 0

            for idx, p in enumerate(params):
                owned = (
                    (not distributed)
                    or (idx % world_size == rank)
                )

                if owned and p.grad is not None:
                    if p.ndim != 2:
                        raise RuntimeError(
                            "Muon received a non-2D parameter."
                        )

                    g = p.grad.detach().float()

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = (
                            torch.zeros_like(g)
                        )

                    buf = state["momentum_buffer"]
                    buf.lerp_(
                        g,
                        1.0 - momentum,
                    )

                    update = zeropower_ns5(buf)

                    update *= max(
                        1.0,
                        p.shape[0] / p.shape[1],
                    ) ** 0.5

                    if distributed:
                        assert updates_flat is not None
                        updates_flat[
                            cursor : cursor + p.numel()
                        ] = update.reshape(-1).to(
                            dtype=torch.bfloat16
                        )
                    else:
                        p.add_(
                            update.to(p.dtype),
                            alpha=-lr,
                        )

                cursor += p.numel()

            if distributed:
                assert updates_flat is not None
                dist.all_reduce(
                    updates_flat,
                    op=dist.ReduceOp.SUM,
                )

                cursor = 0
                for p in params:
                    update = updates_flat[
                        cursor : cursor + p.numel()
                    ].view_as(p).to(
                        dtype=p.dtype
                    )
                    p.add_(
                        update,
                        alpha=-lr,
                    )
                    cursor += p.numel()


# =============================================================================
# DDP SETUP / MODEL / OPTIMIZERS
# =============================================================================

def setup_distributed() -> None:
    global DIST_INITIALIZED

    distributed_requested = (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    )

    if distributed_requested:
        dist.init_process_group(
            backend="nccl",
            device_id=device,
        )
        DIST_INITIALIZED = True
        barrier()

    if WORLD_SIZE != 8 and os.environ.get(
        "ALLOW_NON8_WORLD_SIZE", "0"
    ) != "1":
        raise RuntimeError(
            "This submission trainer targets the official 8-GPU environment. "
            f"Detected WORLD_SIZE={WORLD_SIZE}. "
            "For local smoke tests only, set ALLOW_NON8_WORLD_SIZE=1."
        )


def build_model_and_optimizers(
    grad_accum_steps: int,
):
    base_model = CrossHatchGPT().to(device)

    # Compile Newton-Schulz before the optimizer starts using it.
    global zeropower_ns5

    if ARGS.use_compile and GPU_CC[0] >= 8:
        try:
            zeropower_ns5 = torch.compile(
                zeropower_ns5,
                dynamic=False,
            )
            log0("Muon Newton-Schulz compile: ENABLED")
        except Exception as exc:
            log0(
                "Muon Newton-Schulz compile: FAILED",
                repr(exc),
            )

    # Compile the base model before DDP, following the current official
    # repository's distributed execution pattern.
    compiled_model: nn.Module = base_model

    if ARGS.use_compile and GPU_CC[0] >= 8:
        try:
            compiled_model = torch.compile(
                base_model,
                mode=ARGS.compile_mode,
                dynamic=False,
                fullgraph=True,
            )
            log0("torch.compile: ENABLED")
        except Exception as exc:
            log0(
                "torch.compile: FAILED; using eager mode.",
                repr(exc),
            )
            compiled_model = base_model

    if is_distributed():
        model = DDP(
            compiled_model,
            device_ids=[LOCAL_RANK],
            broadcast_buffers=False,
        )
    else:
        model = compiled_model

    # Optimizers must reference the underlying base_model parameters.
    matrix_params = []
    embed_params = [base_model.tok_emb.weight]
    scalar_params = []

    for name, p in base_model.named_parameters():
        if name == "tok_emb.weight":
            continue

        if (
            p.ndim == 2
            and p.numel() > 4096
        ):
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    muon = Muon(
        matrix_params,
        lr=ARGS.muon_lr,
        momentum=ARGS.muon_momentum,
        momentum_start=ARGS.muon_momentum_start,
        warmup_steps=ARGS.muon_momentum_warmup,
    )

    try:
        optimizer_embed = torch.optim.Adam(
            [
                {
                    "params": embed_params,
                    "lr": ARGS.tied_embed_lr,
                    "base_lr": ARGS.tied_embed_lr,
                }
            ],
            betas=(ARGS.beta1, ARGS.beta2),
            eps=ARGS.eps,
            fused=True,
        )

        optimizer_scalar = torch.optim.Adam(
            [
                {
                    "params": scalar_params,
                    "lr": ARGS.scalar_lr,
                    "base_lr": ARGS.scalar_lr,
                }
            ],
            betas=(ARGS.beta1, ARGS.beta2),
            eps=ARGS.eps,
            fused=True,
        )

        log0("Adam fused: ENABLED")
    except (TypeError, RuntimeError):
        optimizer_embed = torch.optim.Adam(
            [
                {
                    "params": embed_params,
                    "lr": ARGS.tied_embed_lr,
                    "base_lr": ARGS.tied_embed_lr,
                }
            ],
            betas=(ARGS.beta1, ARGS.beta2),
            eps=ARGS.eps,
        )

        optimizer_scalar = torch.optim.Adam(
            [
                {
                    "params": scalar_params,
                    "lr": ARGS.scalar_lr,
                    "base_lr": ARGS.scalar_lr,
                }
            ],
            betas=(ARGS.beta1, ARGS.beta2),
            eps=ARGS.eps,
        )

        log0("Adam fused: DISABLED; using standard Adam")

    for group in muon.param_groups:
        group["base_lr"] = ARGS.muon_lr

    optimizers = [
        muon,
        optimizer_embed,
        optimizer_scalar,
    ]

    return (
        base_model,
        model,
        optimizers,
        matrix_params,
        embed_params,
        scalar_params,
    )


# =============================================================================
# LR SCHEDULE
# =============================================================================

def lr_multiplier(
    elapsed_seconds: float,
    step: int,
) -> float:
    if step < ARGS.warmup_steps:
        return max(
            1e-3,
            step / max(
                1,
                ARGS.warmup_steps,
            ),
        )

    frac = (
        elapsed_seconds
        / max(
            ARGS.max_wallclock_seconds,
            1e-9,
        )
    )

    if frac <= ARGS.warmdown_start:
        return 1.0

    t = (
        frac - ARGS.warmdown_start
    ) / max(
        1.0 - ARGS.warmdown_start,
        1e-9,
    )

    t = min(
        1.0,
        max(0.0, t),
    )

    return 0.5 * (
        1.0
        + math.cos(
            math.pi * t
        )
    )


def set_lrs(
    optimizers,
    elapsed_seconds: float,
    step: int,
) -> float:
    m = lr_multiplier(
        elapsed_seconds,
        step,
    )

    muon, optimizer_embed, optimizer_scalar = optimizers

    muon.param_groups[0]["lr"] = (
        ARGS.muon_lr * m
    )
    optimizer_embed.param_groups[0]["lr"] = (
        ARGS.tied_embed_lr * m
    )
    optimizer_scalar.param_groups[0]["lr"] = (
        ARGS.scalar_lr * m
    )

    return m


# =============================================================================
# EMA
# =============================================================================

class EMA:
    def __init__(
        self,
        module: nn.Module,
        decay: float,
    ):
        self.module = module
        self.decay = decay
        self.shadow: Dict[str, Tensor] = {}
        self.initialized = False

    @torch.no_grad()
    def update(self) -> None:
        if not self.initialized:
            for name, p in self.module.named_parameters():
                self.shadow[name] = (
                    p.detach().float().clone()
                )
            self.initialized = True
            return

        for name, p in self.module.named_parameters():
            self.shadow[name].lerp_(
                p.detach().float(),
                1.0 - self.decay,
            )

    @torch.no_grad()
    def backup_and_copy(self) -> Dict[str, Tensor]:
        backup = {}

        for name, p in self.module.named_parameters():
            backup[name] = (
                p.detach().float().clone()
            )
            p.copy_(
                self.shadow[name].to(
                    dtype=p.dtype
                )
            )

        return backup

    @torch.no_grad()
    def restore(
        self,
        backup: Dict[str, Tensor],
    ) -> None:
        for name, p in self.module.named_parameters():
            p.copy_(
                backup[name].to(
                    dtype=p.dtype
                )
            )


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def eval_val(
    model: nn.Module,
    grad_accum_steps: int,
) -> Tuple[float, float]:
    """
    Current Parameter Golf-style full validation.

    Returns:
        val_loss: natural-log token cross entropy
        val_bpb: bits per byte using SentencePiece byte accounting
    """
    model.eval()

    local_batch_tokens = (
        ARGS.val_batch_size
        // (
            WORLD_SIZE
            * grad_accum_steps
        )
    )

    if local_batch_tokens < ARGS.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank."
        )

    local_batch_seqs = (
        local_batch_tokens
        // ARGS.train_seq_len
    )

    total_seqs = (
        (val_tokens.numel() - 1)
        // ARGS.train_seq_len
    )

    seq_start = (
        total_seqs * RANK
    ) // WORLD_SIZE
    seq_end = (
        total_seqs * (RANK + 1)
    ) // WORLD_SIZE

    val_loss_sum = torch.zeros(
        (),
        device=device,
        dtype=torch.float64,
    )
    val_token_count = torch.zeros(
        (),
        device=device,
        dtype=torch.float64,
    )
    val_byte_count = torch.zeros(
        (),
        device=device,
        dtype=torch.float64,
    )

    for batch_seq_start in range(
        seq_start,
        seq_end,
        local_batch_seqs,
    ):
        batch_seq_end = min(
            batch_seq_start + local_batch_seqs,
            seq_end,
        )

        raw_start = (
            batch_seq_start
            * ARGS.train_seq_len
        )
        raw_end = (
            batch_seq_end
            * ARGS.train_seq_len
            + 1
        )

        local = val_tokens[
            raw_start:raw_end
        ].to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        )

        x = local[:-1].reshape(
            -1,
            ARGS.train_seq_len,
        )
        y = local[1:].reshape(
            -1,
            ARGS.train_seq_len,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
        ):
            batch_loss = model(
                x,
                y,
            ).detach()

        batch_token_count = float(
            y.numel()
        )

        val_loss_sum += (
            batch_loss.to(
                torch.float64
            )
            * batch_token_count
        )
        val_token_count += batch_token_count

        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)

        token_bytes = (
            base_bytes_lut[tgt_ids]
            .to(dtype=torch.int16)
        )

        token_bytes += (
            (
                has_leading_space_lut[tgt_ids]
                & ~is_boundary_token_lut[prev_ids]
            )
            .to(dtype=torch.int16)
        )

        val_byte_count += (
            token_bytes
            .to(torch.float64)
            .sum()
        )

    if is_distributed():
        dist.all_reduce(
            val_loss_sum,
            op=dist.ReduceOp.SUM,
        )
        dist.all_reduce(
            val_token_count,
            op=dist.ReduceOp.SUM,
        )
        dist.all_reduce(
            val_byte_count,
            op=dist.ReduceOp.SUM,
        )

    val_loss = (
        val_loss_sum
        / val_token_count
    )

    bits_per_token = (
        val_loss.item()
        / math.log(2.0)
    )

    tokens_per_byte = (
        val_token_count.item()
        / val_byte_count.item()
    )

    model.train()

    return (
        float(val_loss.item()),
        float(bits_per_token * tokens_per_byte),
    )


# =============================================================================
# ARTIFACT EXPORT
# =============================================================================

def export_fp16_artifact(
    base_model: nn.Module,
) -> Tuple[int, int, int, Path]:
    """
    Export exactly the FP16 payload requested for this experiment.

    The training model keeps its parameters in FP32 so that the optimizer
    retains the established behavior. The exported artifact explicitly stores
    floating tensors as FP16.

    No INT8 / INT6 / GPTQ is used here.
    """
    artifact_path = SCRIPT_DIR / ARGS.artifact_name

    state = base_model.state_dict()

    fp16_state = {}
    for name, tensor in state.items():
        if tensor.is_floating_point():
            fp16_state[name] = (
                tensor.detach()
                .cpu()
                .to(dtype=torch.float16)
                .contiguous()
            )
        else:
            fp16_state[name] = (
                tensor.detach()
                .cpu()
                .contiguous()
            )

    torch.save(
        fp16_state,
        artifact_path,
    )

    model_bytes = artifact_path.stat().st_size

    code_bytes = len(
        Path(__file__).read_bytes()
    )

    total_bytes = (
        model_bytes
        + code_bytes
    )

    return (
        model_bytes,
        code_bytes,
        total_bytes,
        artifact_path,
    )


# =============================================================================
# ROUNDTRIP
# =============================================================================

def roundtrip_fp16_artifact(
    base_model: nn.Module,
    model: nn.Module,
    artifact_path: Path,
    grad_accum_steps: int,
) -> Tuple[float, float]:
    barrier()

    state = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )

    reconstructed = CrossHatchGPT().to(device)

    reconstructed.load_state_dict(
        state,
        strict=True,
    )

    del state
    torch.cuda.empty_cache()

    rt_vloss, rt_bpb = eval_val(
        reconstructed,
        grad_accum_steps,
    )

    # Restore the original base_model after validation.
    del reconstructed
    torch.cuda.empty_cache()

    return rt_vloss, rt_bpb


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    setup_distributed()

    grad_accum_steps = resolve_grad_accum(
        ARGS
    )

    log0("============================================================")
    log0("PARAMETER GOLF — CROSS-HATCH V2")
    log0("============================================================")
    log0(f"PyTorch: {torch.__version__}")
    log0(f"CUDA: {torch.version.cuda}")
    log0(f"GPU: {GPU_NAME}")
    log0(f"GPU compute capability: {GPU_CC}")
    log0(f"World size: {WORLD_SIZE}")
    log0(f"Rank: {RANK}")
    log0(f"Micro batch / GPU: {ARGS.micro_batch_size}")
    log0(f"Grad accumulation: {grad_accum_steps}")
    log0(
        "Global batch tokens: "
        f"{ARGS.train_batch_tokens}"
    )
    log0(f"Sequence length: {ARGS.train_seq_len}")
    log0(f"AMP dtype: {AMP_DTYPE}")
    log0(f"GradScaler: {USE_GRAD_SCALER}")

    log0("")
    log0("=== MODEL ===")
    log0(
        f"{ARGS.num_layers} layers × "
        f"{ARGS.channel_dim} × "
        f"{ARGS.num_lanes}"
    )
    log0(
        f"Model dim: {ARGS.model_dim}"
    )
    log0(
        f"FFN: {ARGS.channel_dim} → "
        f"{ARGS.ffn_dim} → "
        f"{ARGS.channel_dim}"
    )
    log0(
        f"Vocab: {ARGS.vocab_size}"
    )
    log0(
        f"Embedding init std: "
        f"{ARGS.tied_embed_init_std}"
    )
    log0(
        f"QK gain: {ARGS.qk_gain_init}"
    )
    log0(
        f"Logit softcap: {ARGS.logit_softcap}"
    )

    log0("")
    log0("=== DATA ===")
    log0(f"Data path: {DATA_DIR}")
    log0(f"Tokenizer: {TOKENIZER_PATH}")
    log0(
        f"Train pattern: {TRAIN_PATTERN}"
    )
    log0(
        f"Validation pattern: {VAL_PATTERN}"
    )

    train_loader = DistributedTokenLoader(
        TRAIN_PATTERN,
        RANK,
        WORLD_SIZE,
        max_shards=ARGS.train_shards,
    )

    (
        base_model,
        model,
        optimizers,
        matrix_params,
        embed_params,
        scalar_params,
    ) = build_model_and_optimizers(
        grad_accum_steps
    )

    total_params = parameter_count(
        base_model
    )
    raw_fp16_bytes = fp16_parameter_bytes(
        base_model
    )

    log0("")
    log0("=== PARAMETER AUDIT ===")
    log0(
        f"Parameters: {total_params:,}"
    )
    log0(
        f"Raw FP16 parameter bytes: "
        f"{raw_fp16_bytes:,}"
    )
    log0(
        f"Muon tensors: "
        f"{len(matrix_params)}"
    )
    log0(
        f"Embedding tensors: "
        f"{len(embed_params)}"
    )
    log0(
        f"Scalar/control tensors: "
        f"{len(scalar_params)}"
    )

    # -------------------------------------------------------------------------
    # Minimal fresh-init smoke test.
    # -------------------------------------------------------------------------
    model.train()
    smoke_x = torch.randint(
        0,
        ARGS.vocab_size,
        (1, 64),
        device=device,
    )
    smoke_y = torch.randint(
        0,
        ARGS.vocab_size,
        (1, 64),
        device=device,
    )

    for opt in optimizers:
        opt.zero_grad(
            set_to_none=True
        )

    if is_distributed():
        model.require_backward_grad_sync = True

    with torch.autocast(
        device_type="cuda",
        dtype=AMP_DTYPE,
    ):
        smoke_loss = model(
            smoke_x,
            smoke_y,
        )

    if not torch.isfinite(smoke_loss):
        raise RuntimeError(
            "Initialization smoke-test loss is non-finite."
        )

    # The smoke test is not an optimizer update.
    smoke_loss.backward()

    bad_grads = []
    for name, p in base_model.named_parameters():
        if p.grad is not None and not torch.isfinite(
            p.grad
        ).all():
            bad_grads.append(name)

    if bad_grads:
        raise RuntimeError(
            "Non-finite gradients during smoke test: "
            + ", ".join(bad_grads[:20])
        )

    for opt in optimizers:
        opt.zero_grad(
            set_to_none=True
        )

    log0(
        f"Initialization smoke loss: "
        f"{smoke_loss.detach().float().item():.6f}"
    )
    log0("Initialization: OK")

    # -------------------------------------------------------------------------
    # GradScaler.
    # -------------------------------------------------------------------------
    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=USE_GRAD_SCALER,
        )
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=USE_GRAD_SCALER
        )

    # -------------------------------------------------------------------------
    # Training.
    #
    # The measured timer excludes:
    #   - process-group startup
    #   - model construction
    #   - compilation
    #   - dataset/tokenizer setup
    #   - final validation
    #
    # This follows the repository's training-time accounting convention.
    # -------------------------------------------------------------------------
    log0("")
    log0("=== TRAINING ===")
    log0(
        f"Wallclock budget: "
        f"{ARGS.max_wallclock_seconds:.3f}s"
    )

    barrier()

    torch.cuda.synchronize()
    train_start = time.perf_counter()

    optimizer_step = 0
    tokens_seen_global = 0
    stop_after_step = None

    ema = EMA(
        base_model,
        ARGS.ema_decay,
    )

    model.train()

    while True:
        elapsed = (
            time.perf_counter()
            - train_start
        )

        if elapsed >= ARGS.max_wallclock_seconds:
            break

        if (
            stop_after_step is not None
            and optimizer_step >= stop_after_step
        ):
            break

        # Stop before beginning a partial accumulation window.
        if (
            time.perf_counter()
            - train_start
        ) >= ARGS.max_wallclock_seconds:
            break

        for opt in optimizers:
            opt.zero_grad(
                set_to_none=True
            )

        set_lrs(
            optimizers,
            elapsed,
            optimizer_step,
        )

        step_loss = 0.0
        completed_micro = 0

        for micro_step in range(
            grad_accum_steps
        ):
            current_elapsed = (
                time.perf_counter()
                - train_start
            )

            if current_elapsed >= (
                ARGS.max_wallclock_seconds
            ):
                break

            if is_distributed():
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1
                )

            x, y = train_loader.next_batch(
                ARGS.train_batch_tokens,
                ARGS.train_seq_len,
                grad_accum_steps,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=AMP_DTYPE,
            ):
                loss = model(
                    x,
                    y,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at optimizer step "
                    f"{optimizer_step}"
                )

            loss_for_backward = (
                loss
                / grad_accum_steps
            )

            if scaler.is_enabled():
                scaler.scale(
                    loss_for_backward
                ).backward()
            else:
                loss_for_backward.backward()

            step_loss += (
                loss.detach()
                .float()
                .item()
            )

            tokens_seen_global += (
                int(y.numel())
                * WORLD_SIZE
            )
            completed_micro += 1

        if is_distributed():
            model.require_backward_grad_sync = True

        # Never apply a partial optimizer step.
        if completed_micro != grad_accum_steps:
            for opt in optimizers:
                opt.zero_grad(
                    set_to_none=True
                )
            break

        if scaler.is_enabled():
            for opt in optimizers:
                scaler.unscale_(opt)

        if ARGS.grad_clip > 0.0:
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(),
                    ARGS.grad_clip,
                )
            )
        else:
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    base_model.parameters(),
                    float("inf"),
                )
            )

        if scaler.is_enabled():
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            for opt in optimizers:
                opt.step()

        optimizer_step += 1

        elapsed = (
            time.perf_counter()
            - train_start
        )

        if (
            elapsed / ARGS.max_wallclock_seconds
        ) >= ARGS.ema_start_frac:
            ema.update()

        tok_s = (
            tokens_seen_global
            / max(
                elapsed,
                1e-9,
            )
        )

        if (
            optimizer_step <= 10
            or optimizer_step % ARGS.train_log_every == 0
        ):
            log0(
                f"step={optimizer_step:5d} "
                f"time={elapsed:8.2f}s "
                f"loss={step_loss / grad_accum_steps:.5f} "
                f"tok/s={tok_s:,.0f} "
                f"MuonLR={optimizers[0].param_groups[0]['lr']:.6g} "
                f"EmbLR={optimizers[1].param_groups[0]['lr']:.6g} "
                f"ScalarLR={optimizers[2].param_groups[0]['lr']:.6g} "
                f"grad={grad_norm.detach().float().item():.4f}"
            )

        reached_cap = (
            elapsed
            >= ARGS.max_wallclock_seconds
        )

        if is_distributed():
            reached_tensor = torch.tensor(
                int(reached_cap),
                device=device,
            )
            dist.all_reduce(
                reached_tensor,
                op=dist.ReduceOp.MAX,
            )
            reached_cap = bool(
                reached_tensor.item()
            )

        if reached_cap:
            stop_after_step = optimizer_step
            break

        # Optional repository-style periodic full validation.  It is not
        # included in the measured training timer.
        if (
            ARGS.val_loss_every > 0
            and optimizer_step % ARGS.val_loss_every == 0
        ):
            torch.cuda.synchronize()
            pause_start = time.perf_counter()

            vloss, vbpb = eval_val(
                model,
                grad_accum_steps,
            )

            torch.cuda.synchronize()
            pause_seconds = (
                time.perf_counter()
                - pause_start
            )

            log0(
                f"validation@step={optimizer_step}: "
                f"val_loss={vloss:.6f} "
                f"val_bpb={vbpb:.6f} "
                f"time={pause_seconds:.2f}s"
            )

            torch.cuda.synchronize()

            # Resume timing from the same measured training clock.
            train_start += pause_seconds

    torch.cuda.synchronize()
    measured_training_seconds = (
        time.perf_counter()
        - train_start
    )

    # -------------------------------------------------------------------------
    # Final health check.
    # -------------------------------------------------------------------------
    log0("")
    log0("=== HEALTH CHECK ===")

    max_abs = 0.0
    bad_params = []

    for name, p in base_model.named_parameters():
        pf = p.detach().float()

        if not torch.isfinite(pf).all():
            bad_params.append(name)
        else:
            max_abs = max(
                max_abs,
                float(pf.abs().max().item()),
            )

    if bad_params:
        raise RuntimeError(
            "Non-finite model parameters: "
            + ", ".join(bad_params[:20])
        )

    log0("Parameters: finite")
    log0(
        f"Maximum |parameter|: {max_abs:.6f}"
    )

    log0(
        f"Training time: "
        f"{measured_training_seconds:.3f}s"
    )
    log0(
        f"Optimizer steps: "
        f"{optimizer_step}"
    )
    log0(
        f"Tokens seen: "
        f"{tokens_seen_global:,}"
    )
    log0(
        "Training tok/s: "
        f"{tokens_seen_global / max(measured_training_seconds, 1e-9):,.0f}"
    )

    # -------------------------------------------------------------------------
    # Official full validation.
    # -------------------------------------------------------------------------
    log0("")
    log0("=== OFFICIAL VALIDATION ===")

    torch.cuda.synchronize()
    val_start = time.perf_counter()

    raw_vloss, raw_bpb = eval_val(
        model,
        grad_accum_steps,
    )

    torch.cuda.synchronize()
    val_seconds = (
        time.perf_counter()
        - val_start
    )

    log0(
        f"pre_quant_val_loss: "
        f"{raw_vloss:.8f}"
    )
    log0(
        f"pre_quant_val_bpb: "
        f"{raw_bpb:.8f}"
    )
    log0(
        f"validation_time: "
        f"{val_seconds:.3f}s"
    )

    # Evaluate EMA only if it has actually been initialized.
    ema_vloss = None
    ema_bpb = None

    if ema.initialized:
        log0("")
        log0("=== EMA VALIDATION ===")

        backup = ema.backup_and_copy()

        try:
            ema_vloss, ema_bpb = eval_val(
                model,
                grad_accum_steps,
            )
        finally:
            ema.restore(backup)

        log0(
            f"ema_val_loss: "
            f"{ema_vloss:.8f}"
        )
        log0(
            f"ema_val_bpb: "
            f"{ema_bpb:.8f}"
        )

    # -------------------------------------------------------------------------
    # Artifact export.
    #
    # Rank 0 writes the file; all ranks synchronize before roundtrip loading.
    # -------------------------------------------------------------------------
    artifact_path = (
        SCRIPT_DIR
        / ARGS.artifact_name
    )

    model_bytes = 0
    code_bytes = len(
        Path(__file__).read_bytes()
    )
    total_artifact_bytes = 0

    if RANK == 0:
        log0("")
        log0("=== FP16 ARTIFACT EXPORT ===")

        (
            model_bytes,
            code_bytes,
            total_artifact_bytes,
            artifact_path,
        ) = export_fp16_artifact(
            base_model
        )

        log0(
            f"Model FP16 file bytes: "
            f"{model_bytes:,}"
        )
        log0(
            f"Code bytes: "
            f"{code_bytes:,}"
        )
        log0(
            f"TOTAL ARTIFACT BYTES: "
            f"{total_artifact_bytes:,}"
        )
        log0(
            "ARTIFACT LIMIT: "
            "16,000,000"
        )
        log0(
            "HEADROOM: "
            f"{16_000_000 - total_artifact_bytes:,}"
        )

        if total_artifact_bytes > 16_000_000:
            raise RuntimeError(
                "FP16 artifact exceeds the 16,000,000-byte limit. "
                "This script intentionally does NOT switch to INT8."
            )

    barrier()

    # -------------------------------------------------------------------------
    # Roundtrip from the actual serialized FP16 file.
    # -------------------------------------------------------------------------
    log0("")
    log0("=== FP16 ROUNDTRIP VALIDATION ===")

    state = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )

    roundtrip_model = CrossHatchGPT().to(device)
    roundtrip_model.load_state_dict(
        state,
        strict=True,
    )

    del state
    torch.cuda.empty_cache()

    rt_vloss, rt_bpb = eval_val(
        roundtrip_model,
        grad_accum_steps,
    )

    log0(
        f"roundtrip_val_loss: "
        f"{rt_vloss:.8f}"
    )
    log0(
        f"roundtrip_val_bpb: "
        f"{rt_bpb:.8f}"
    )
    log0(
        f"roundtrip_delta_val_loss: "
        f"{rt_vloss - raw_vloss:+.8f}"
    )
    log0(
        f"roundtrip_delta_val_bpb: "
        f"{rt_bpb - raw_bpb:+.8f}"
    )

    del roundtrip_model
    torch.cuda.empty_cache()

    # Synchronize before final reporting.
    barrier()

    if is_distributed():
        dist.destroy_process_group()

    log0("")
    log0("=== FINAL SUMMARY ===")
    log0("Status: NON-RECORD / EXPERIMENTAL")
    log0(
        f"Official full val_bpb: {raw_bpb:.8f}"
    )
    log0(
        f"FP16 roundtrip val_bpb: {rt_bpb:.8f}"
    )
    log0(
        f"Training seconds: {measured_training_seconds:.3f}"
    )
    log0(
        f"Optimizer steps: {optimizer_step}"
    )
    log0(
        f"Global tokens: {tokens_seen_global:,}"
    )

    if RANK == 0:
        # Final explicit artifact audit, measured from the actual files.
        actual_code_bytes = len(
            Path(__file__).read_bytes()
        )
        actual_model_bytes = (
            artifact_path.stat().st_size
        )
        actual_total = (
            actual_code_bytes
            + actual_model_bytes
        )

        log0(
            f"Final code bytes: "
            f"{actual_code_bytes:,}"
        )
        log0(
            f"Final model bytes: "
            f"{actual_model_bytes:,}"
        )
        log0(
            f"Final total artifact bytes: "
            f"{actual_total:,}"
        )
        log0(
            "Artifact size PASS: "
            f"{actual_total <= 16_000_000}"
        )


if __name__ == "__main__":
    main()
