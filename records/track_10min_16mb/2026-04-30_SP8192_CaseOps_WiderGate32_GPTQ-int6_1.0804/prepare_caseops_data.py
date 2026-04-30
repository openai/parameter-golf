"""Prepare CaseOps-tokenized FineWeb shards + per-token byte sidecar.

CaseOps (``lossless_caps_caseops_v1``) is a bijective, character-level text
transform that introduces four operator tokens in place of explicit
capitalization: TITLE, ALLCAPS, CAPNEXT, ESC. The transform is fully
reversible — no information is lost relative to the untransformed UTF-8
text, so BPB stays computable on TRUE byte counts.

Forward pipeline:
  1. Read the canonical FineWeb-10B doc stream (``docs_selected.jsonl``
     produced by ``data/download_hf_docs_and_tokenize.py`` in the root repo).
  2. Apply ``encode_lossless_caps_v2`` (the caseops_v1 alias) to each doc.
  3. Tokenize with the shipped SP model
     ``tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model``
     (reserves TITLE/ALLCAPS/CAPNEXT/ESC + sentinel as user_defined_symbols).
  4. Write uint16 train/val shards (``fineweb_{train,val}_XXXXXX.bin``).
  5. For the VAL stream only, emit per-token byte sidecar shards
     (``fineweb_val_bytes_XXXXXX.bin``, uint16 parallel arrays) that record
     each token's ORIGINAL pre-transform UTF-8 byte count. BPB is computed
     from these canonical bytes so the score is on the untransformed text
     (not the transformed representation).

Output layout — matches what ``train_gpt.py`` expects under
``DATA_DIR=./data`` with ``CASEOPS_ENABLED=1``:

    data/datasets/fineweb10B_sp8192_caseops/datasets/
      tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
      datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
        fineweb_train_000000.bin
        fineweb_train_000001.bin
        ...
        fineweb_val_000000.bin
        fineweb_val_bytes_000000.bin

Usage:

    python3 prepare_caseops_data.py \\
        --docs ./fineweb10B_raw/docs_selected.jsonl \\
        --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \\
        --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model

Requirements: sentencepiece, numpy. CPU-only. Runs once; reused across seeds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np
import sentencepiece as spm

# Local import — lossless_caps.py ships next to this script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lossless_caps import (  # noqa: E402
    LOSSLESS_CAPS_CASEOPS_V1,
    encode_lossless_caps_v2,
    surface_piece_original_byte_counts,
)


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_TOKENS = 10_000_000  # tokens per shard — matches the main pipeline
BOS_ID = 1  # SP model's <s> control token; train_gpt.py:_find_docs requires BOS per doc


def _write_shard(out_path: pathlib.Path, arr: np.ndarray) -> None:
    """Write a uint16 shard in the standard header-prefixed format."""
    assert arr.dtype == np.uint16
    header = np.zeros(256, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(arr.size)
    with out_path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(arr.tobytes())


def _iter_docs(docs_path: pathlib.Path):
    """Yield doc strings from a jsonl file (one json object per line)."""
    with docs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Support both {"text": ...} and raw strings.
            yield obj["text"] if isinstance(obj, dict) else obj


def _token_original_byte_counts(
    sp: spm.SentencePieceProcessor,
    original_text: str,
    transformed_text: str,
) -> np.ndarray:
    """Per-token canonical (pre-transform) UTF-8 byte counts.

    Delegates to ``surface_piece_original_byte_counts`` in ``lossless_caps.py``
    — the canonical exporter used by the PR #1729 / HF-hosted CaseOps dataset.
    Operator pieces (U+E001..U+E004) contribute 0 original bytes; letter pieces
    contribute their pre-transform UTF-8 byte count.
    """
    proto = sp.encode_as_immutable_proto(transformed_text)
    byte_counts = surface_piece_original_byte_counts(
        (piece.surface for piece in proto.pieces),
        text_transform_name=LOSSLESS_CAPS_CASEOPS_V1,
    )
    return np.asarray(list(byte_counts), dtype=np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True, type=pathlib.Path, help="Path to docs_selected.jsonl")
    ap.add_argument("--out",  required=True, type=pathlib.Path, help="Output datasets dir")
    ap.add_argument("--sp",   required=True, type=pathlib.Path, help="Path to CaseOps SP model")
    ap.add_argument("--val-docs", type=int, default=10_000, help="Validation docs count")
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.sp))
    print(f"loaded sp: vocab={sp.vocab_size()}", flush=True)

    train_out = args.out / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
    train_out.mkdir(parents=True, exist_ok=True)

    val_buf_tokens: list[int] = []
    val_buf_bytes: list[int] = []
    train_buf: list[int] = []
    val_written = 0
    train_written = 0
    n_docs = 0

    for text in _iter_docs(args.docs):
        transformed = encode_lossless_caps_v2(text)
        token_ids = [BOS_ID] + sp.encode(transformed, out_type=int)
        if n_docs < args.val_docs:
            # Validation doc — also compute byte sidecar
            byte_counts = _token_original_byte_counts(sp, text, transformed)
            val_buf_tokens.extend(token_ids)
            val_buf_bytes.append(0)  # BOS contributes 0 original bytes
            val_buf_bytes.extend(int(b) for b in byte_counts)
            if len(val_buf_tokens) >= SHARD_TOKENS:
                _write_shard(train_out / f"fineweb_val_{val_written:06d}.bin",
                             np.array(val_buf_tokens[:SHARD_TOKENS], dtype=np.uint16))
                _write_shard(train_out / f"fineweb_val_bytes_{val_written:06d}.bin",
                             np.array(val_buf_bytes[:SHARD_TOKENS], dtype=np.uint16))
                val_buf_tokens = val_buf_tokens[SHARD_TOKENS:]
                val_buf_bytes = val_buf_bytes[SHARD_TOKENS:]
                val_written += 1
        else:
            train_buf.extend(token_ids)
            if len(train_buf) >= SHARD_TOKENS:
                _write_shard(train_out / f"fineweb_train_{train_written:06d}.bin",
                             np.array(train_buf[:SHARD_TOKENS], dtype=np.uint16))
                train_buf = train_buf[SHARD_TOKENS:]
                train_written += 1
        n_docs += 1
        if n_docs % 10_000 == 0:
            print(f"  processed {n_docs} docs  train_shards={train_written}  val_shards={val_written}", flush=True)

    # Flush tail buffers into final (possibly short) shards.
    if val_buf_tokens:
        _write_shard(train_out / f"fineweb_val_{val_written:06d}.bin",
                     np.array(val_buf_tokens, dtype=np.uint16))
        _write_shard(train_out / f"fineweb_val_bytes_{val_written:06d}.bin",
                     np.array(val_buf_bytes, dtype=np.uint16))
    if train_buf:
        _write_shard(train_out / f"fineweb_train_{train_written:06d}.bin",
                     np.array(train_buf, dtype=np.uint16))

    print(f"done. docs={n_docs} train_shards={train_written + (1 if train_buf else 0)} val_shards={val_written + (1 if val_buf_tokens else 0)}")


if __name__ == "__main__":
    main()
# Python deps. Install with: pip install -r requirements.txt
torch==2.9.1+cu128
sentencepiece
brotli
huggingface_hub
numpy
python-minifier

# FlashAttention 3 must be installed separately (not on PyPI):
# pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# System dep (apt): lrzip (used by per-group compressor)
# apt-get install -y lrzip
{
  "author": "Benjamin Hadad",
  "github_id": "codemath3000",
  "name": "11L XSA + LQER + SparseAttnGate + SmearGate (BOS-fixed) + PolarNS Muon + 9-hparam stack",
  "blurb": "11L 512d 8H/4KV transformer with U-Net skips, parallel decoder, partial RoPE, Polar-Express Newton-Schulz Muon (5 steps), LQER asymmetric int4 rank-4 quant correction, sparse attention head-output gate (gate_window=12), SmearGate position-mixing (with cross-document leak fix on BOS positions), fused LeakyReLU-square MLP, fused softcapped CE Triton kernel, GPTQ int6 + int7 embed + per-row int8 attn-gate quantization, per-group lrzip+brotli compression, phased TTT eval (3 phases, prefix=2500 docs). Plus 9 greedy-validated hyperparameter overrides on top of the published baseline. 3-seed mean: 1.06107587 BPB, beating the current official leaderboard (1.0810 BPB) by 0.01992 BPB / 0.04359 nats.",
  "date": "2026-04-27",
  "track": "10min_16mb",
  "val_loss": 2.32202732,
  "val_bpb": 1.06107587,
  "val_loss_std": 0.00198,
  "val_bpb_std": 0.00090,
  "seeds": [42, 0, 1234],
  "seed_results": {
    "42": {
      "val_loss": 2.31944212,
      "val_bpb": 1.05989454,
      "artifact_bytes": 15897259,
      "steps": 4945,
      "step_avg_ms": 121.3,
      "eval_time_s": 508.8
    },
    "0": {
      "val_loss": 2.32239991,
      "val_bpb": 1.06124613,
      "artifact_bytes": 15900947,
      "steps": 4932,
      "step_avg_ms": 121.7,
      "eval_time_s": 455.1
    },
    "1234": {
      "val_loss": 2.32423994,
      "val_bpb": 1.06208695,
      "artifact_bytes": 15907550,
      "steps": 4917,
      "step_avg_ms": 122.0,
      "eval_time_s": 470.0
    }
  },
  "comparison_baseline_bpb": 1.0810,
  "delta_vs_leaderboard_bpb": -0.01992,
  "delta_vs_leaderboard_nats": -0.04359,
  "artifact_bytes_mean": 15901919,
  "artifact_bytes_max": 15907550,
  "bytes_total": 15907550,
  "train_steps_mean": 4931.33,
  "step_avg_ms_mean": 121.7,
  "hardware": "8xH100 80GB SXM",
  "pytorch_version": "2.9.1+cu128",
  "cuda_version": "12.8",
  "flash_attn_version": "FA3 (cu128_torch291 wheel)",
  "technique_summary": "11L XSA + LQER int4-rank4 + SparseAttnGate + BOS-fixed SmearGate + Polar-Express Muon + per-group lrzip compression + 9-hparam greedy stack (clips_tighter, BETA2=0.99, TTT_BETA2=0.99, TTT_WEIGHT_DECAY=0.5, TTT_LORA_RANK=80, SPARSE_ATTN_GATE_SCALE=0.5, PHASED_TTT_PREFIX_DOCS=2500, WARMDOWN_FRAC=0.85)"
}
import base64, collections, copy, fcntl, glob, io, lzma, math, os
from pathlib import Path
import random, re, subprocess, sys, time, uuid, numpy as np, sentencepiece as spm, torch, torch.distributed as dist, torch.nn.functional as F
from torch import Tensor, nn
from flash_attn_interface import (
    flash_attn_func as flash_attn_3_func,
    flash_attn_varlen_func,
)
from concurrent.futures import ThreadPoolExecutor
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# ===== Fused softcapped cross-entropy (Triton) — training-only path =====
# Replaces the eager
#     logits_softcap = softcap * tanh(logits / softcap)
#     F.cross_entropy(logits_softcap.float(), targets, reduction="mean")
# sequence with a single fused kernel that reads logits_proj once, applies
# softcap in-register, and computes (LSE, loss) in one streaming pass. The
# backward kernel mirrors the forward so there's no stored softcapped logits.
# Numerically identical to the eager path up to fp32 accumulation differences.
_FUSED_CE_LIBRARY = "pgsubmission1draft7fusedce"
_FUSED_CE_BLOCK_SIZE = 1024
_FUSED_CE_NUM_WARPS = 4


@triton.jit
def _softcapped_ce_fwd_kernel(
    logits_ptr, losses_ptr, lse_ptr, targets_ptr,
    stride_logits_n, stride_logits_v,
    n_rows, n_cols, softcap,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    max_val = -float("inf")
    sum_exp = 0.0
    A = 2.0 * softcap
    inv_C = 2.0 / softcap
    for off in range(0, n_cols, block_size):
        cols = off + tl.arange(0, block_size)
        mask = cols < n_cols
        val = tl.load(
            logits_row_ptr + cols * stride_logits_v,
            mask=mask, other=-float("inf"),
        ).to(tl.float32)
        z = A * tl.sigmoid(val * inv_C)
        z = tl.where(mask, z, -float("inf"))
        curr_max = tl.max(z, axis=0)
        new_max = tl.maximum(max_val, curr_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
        max_val = new_max
    lse = max_val + tl.log(sum_exp)
    tl.store(lse_ptr + row_idx, lse)
    target = tl.load(targets_ptr + row_idx).to(tl.int32)
    target_val = tl.load(logits_row_ptr + target * stride_logits_v).to(tl.float32)
    target_z = A * tl.sigmoid(target_val * inv_C)
    tl.store(losses_ptr + row_idx, lse - target_z)


@triton.jit
def _softcapped_ce_bwd_kernel(
    grad_logits_ptr, grad_losses_ptr, lse_ptr, logits_ptr, targets_ptr,
    stride_logits_n, stride_logits_v,
    stride_grad_n, stride_grad_v,
    n_rows, n_cols, softcap,
    block_size: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    grad_row_ptr = grad_logits_ptr + row_idx * stride_grad_n
    lse = tl.load(lse_ptr + row_idx)
    grad_loss = tl.load(grad_losses_ptr + row_idx).to(tl.float32)
    target = tl.load(targets_ptr + row_idx).to(tl.int32)
    A = 2.0 * softcap
    inv_C = 2.0 / softcap
    dz_dx_scale = A * inv_C
    for off in range(0, n_cols, block_size):
        cols = off + tl.arange(0, block_size)
        mask = cols < n_cols
        val = tl.load(
            logits_row_ptr + cols * stride_logits_v,
            mask=mask, other=0.0,
        ).to(tl.float32)
        sigmoid_u = tl.sigmoid(val * inv_C)
        z = A * sigmoid_u
        probs = tl.exp(z - lse)
        grad_z = grad_loss * (probs - tl.where(cols == target, 1.0, 0.0))
        grad_x = grad_z * (dz_dx_scale * sigmoid_u * (1.0 - sigmoid_u))
        tl.store(grad_row_ptr + cols * stride_grad_v, grad_x, mask=mask)


def _validate_softcapped_ce_inputs(
    logits: Tensor, targets: Tensor, softcap: float,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits.ndim=2, got {logits.ndim}")
    if targets.ndim != 1:
        raise ValueError(f"Expected targets.ndim=1, got {targets.ndim}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Expected matching rows, got logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )
    if not logits.is_cuda or not targets.is_cuda:
        raise ValueError("softcapped_cross_entropy requires CUDA tensors")
    if softcap <= 0.0:
        raise ValueError(f"softcap must be positive, got {softcap}")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported logits dtype: {logits.dtype}")
    logits = logits.contiguous()
    targets = targets.contiguous()
    if targets.dtype != torch.int64:
        targets = targets.to(dtype=torch.int64)
    return logits, targets


@torch.library.custom_op(f"{_FUSED_CE_LIBRARY}::softcapped_ce", mutates_args=())
def softcapped_ce_op(logits: Tensor, targets: Tensor, softcap: float) -> tuple[Tensor, Tensor]:
    logits, targets = _validate_softcapped_ce_inputs(logits, targets, float(softcap))
    n_rows, n_cols = logits.shape
    losses = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    lse = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    _softcapped_ce_fwd_kernel[(n_rows,)](
        logits, losses, lse, targets,
        logits.stride(0), logits.stride(1),
        n_rows, n_cols, float(softcap),
        block_size=_FUSED_CE_BLOCK_SIZE, num_warps=_FUSED_CE_NUM_WARPS,
    )
    return losses, lse


@softcapped_ce_op.register_fake
def _(logits: Tensor, targets: Tensor, softcap: float):
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError("softcapped_ce fake impl expects 2D logits and 1D targets")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Expected matching rows, got logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
        )
    n_rows = logits.shape[0]
    return (
        logits.new_empty((n_rows,), dtype=torch.float32),
        logits.new_empty((n_rows,), dtype=torch.float32),
    )


@torch.library.custom_op(f"{_FUSED_CE_LIBRARY}::softcapped_ce_backward", mutates_args=())
def softcapped_ce_backward_op(
    logits: Tensor, targets: Tensor, lse: Tensor, grad_losses: Tensor, softcap: float,
) -> Tensor:
    logits, targets = _validate_softcapped_ce_inputs(logits, targets, float(softcap))
    lse = lse.contiguous()
    grad_losses = grad_losses.contiguous().to(dtype=torch.float32)
    if lse.ndim != 1 or grad_losses.ndim != 1:
        raise ValueError("Expected 1D lse and grad_losses")
    if lse.shape[0] != logits.shape[0] or grad_losses.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Expected row-aligned lse/grad_losses, got logits={tuple(logits.shape)} "
            f"lse={tuple(lse.shape)} grad_losses={tuple(grad_losses.shape)}"
        )
    grad_logits = torch.empty_like(logits)
    n_rows, n_cols = logits.shape
    _softcapped_ce_bwd_kernel[(n_rows,)](
        grad_logits, grad_losses, lse, logits, targets,
        logits.stride(0), logits.stride(1),
        grad_logits.stride(0), grad_logits.stride(1),
        n_rows, n_cols, float(softcap),
        block_size=_FUSED_CE_BLOCK_SIZE, num_warps=_FUSED_CE_NUM_WARPS,
    )
    return grad_logits


@softcapped_ce_backward_op.register_fake
def _(logits: Tensor, targets: Tensor, lse: Tensor, grad_losses: Tensor, softcap: float):
    if logits.ndim != 2 or targets.ndim != 1 or lse.ndim != 1 or grad_losses.ndim != 1:
        raise ValueError("softcapped_ce_backward fake impl expects 2D logits and 1D row tensors")
    if (
        logits.shape[0] != targets.shape[0]
        or logits.shape[0] != lse.shape[0]
        or logits.shape[0] != grad_losses.shape[0]
    ):
        raise ValueError("softcapped_ce_backward fake impl expects row-aligned tensors")
    return logits.new_empty(logits.shape)


def _softcapped_ce_setup_context(
    ctx: torch.autograd.function.FunctionCtx, inputs, output,
) -> None:
    logits, targets, softcap = inputs
    _losses, lse = output
    ctx.save_for_backward(logits, targets, lse)
    ctx.softcap = float(softcap)


def _softcapped_ce_backward(
    ctx: torch.autograd.function.FunctionCtx, grad_losses: Tensor, grad_lse: "Tensor | None",
):
    del grad_lse
    logits, targets, lse = ctx.saved_tensors
    grad_logits = torch.ops.pgsubmission1draft7fusedce.softcapped_ce_backward(
        logits, targets, lse, grad_losses, ctx.softcap
    )
    return grad_logits, None, None


softcapped_ce_op.register_autograd(
    _softcapped_ce_backward, setup_context=_softcapped_ce_setup_context,
)


def softcapped_cross_entropy(
    logits: Tensor, targets: Tensor, softcap: float, reduction: str = "mean",
) -> Tensor:
    losses, _lse = torch.ops.pgsubmission1draft7fusedce.softcapped_ce(
        logits, targets, float(softcap)
    )
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"Unsupported reduction={reduction!r}")


class Hyperparameters:
    data_dir = os.environ.get("DATA_DIR", "./data/")
    seed = int(os.environ.get("SEED", 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    iterations = int(os.environ.get("ITERATIONS", 20000))
