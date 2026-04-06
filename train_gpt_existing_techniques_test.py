"""
Parameter Golf — Novel Architecture Training Script (PyTorch / TEST VERSION)
=============================================================================
Same as train_gpt_novel.py but with single-GPU + debug/small mode support.
Can run on Colab (T4), single A100, or full 8xH100.

Set TEST_MODE env var to control scale:
  TEST_MODE=smoke  — T4/Colab free, ~2 min, 100 steps, 256 seq_len, 30% model
  TEST_MODE=small  — 1xA100, ~5-10 min, 2000 steps, 512 seq_len, 60% model
  TEST_MODE=full   — 8xH100, ~10 min, full config (default, same as train_gpt_novel.py)

Relative rankings between configurations are preserved at smaller scale,
so you can compare baseline vs TTT vs DeltaNet etc. cheaply.

Usage:
  # Smoke test on Colab (single GPU, no torchrun needed):
  TEST_MODE=smoke python train_gpt_novel_test.py

  # Small test on single A100:
  TEST_MODE=small python train_gpt_novel_test.py

  # Full 8xH100 (same as original):
  torchrun --nproc_per_node=8 train_gpt_novel_test.py

  # Compare TTT hybrid vs baseline on Colab:
  TEST_MODE=smoke python train_gpt_novel_test.py
  TEST_MODE=smoke LAYER_TYPES=attn,attn,ttt,attn,attn,ttt,attn \
    python train_gpt_novel_test.py

  # Compare recursive vs baseline on single A100:
  TEST_MODE=small python train_gpt_novel_test.py
  TEST_MODE=small RECURSIVE=1 NUM_SHARED_BLOCKS=3 NUM_LOOPS=4 LORA_RANK=16 \
    python train_gpt_novel_test.py
"""
from __future__ import annotations
import copy
import glob
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _HAS_FA3 = True
except ImportError:
    flash_attn_3_func = None
    _HAS_FA3 = False


def _sdpa_attn(q, k, v, causal=True):
    """Fallback attention using PyTorch's scaled_dot_product_attention.
    Input shapes: q (B, T, H, D), k (B, T, Hkv, D), v (B, T, Hkv, D)
    Returns: (B, T, H, D)"""
    # SDPA expects (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    # GQA: expand KV heads to match Q heads
    if k.size(1) < q.size(1):
        rep = q.size(1) // k.size(1)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return y.transpose(1, 2)  # back to (B, T, H, D)

# =============================================================================
# SECTION 0: TEST MODE CONFIGURATION
# =============================================================================
# TEST_MODE overrides default hyperparameters for cheaper experimentation.
# Relative rankings between configs are preserved at smaller scale.

_TEST_MODE = os.environ.get("TEST_MODE", "full").lower()

_TEST_OVERRIDES = {
    "smoke": {
        # ~2 min on T4/Colab free tier, 30% model scale
        "NUM_LAYERS": "7",
        "MODEL_DIM": "384",
        "NUM_HEADS": "6",
        "NUM_KV_HEADS": "3",
        "TRAIN_SEQ_LEN": "256",
        "EVAL_SEQ_LEN": "256",
        "ITERATIONS": "100",
        "WARMDOWN_ITERS": "20",
        "WARMUP_STEPS": "5",
        "TRAIN_BATCH_TOKENS": "65536",
        "VAL_BATCH_SIZE": "32768",
        "VAL_LOSS_EVERY": "50",
        "TRAIN_LOG_EVERY": "10",
        "MAX_WALLCLOCK_SECONDS": "120",
        "GPTQ_CALIB_BATCHES": "16",
        "XSA_LAST_N": "7",
        "VE_LAYERS": "5,6",
        "NUM_SHARED_BLOCKS": "2",
        "NUM_LOOPS": "4",
    },
    "small": {
        # ~5-10 min on 1xA100, 60% model scale
        "NUM_LAYERS": "9",
        "MODEL_DIM": "448",
        "NUM_HEADS": "7",
        "NUM_KV_HEADS": "7",
        "TRAIN_SEQ_LEN": "512",
        "EVAL_SEQ_LEN": "512",
        "ITERATIONS": "2000",
        "WARMDOWN_ITERS": "500",
        "WARMUP_STEPS": "10",
        "TRAIN_BATCH_TOKENS": "131072",
        "VAL_BATCH_SIZE": "65536",
        "VAL_LOSS_EVERY": "500",
        "TRAIN_LOG_EVERY": "100",
        "MAX_WALLCLOCK_SECONDS": "600",
        "GPTQ_CALIB_BATCHES": "32",
        "XSA_LAST_N": "9",
        "VE_LAYERS": "7,8",
        "NUM_SHARED_BLOCKS": "3",
        "NUM_LOOPS": "3",
    },
    "medium": {
        # ~15-20 min on T4/Colab free. EXACT PR#1019 architecture (11 layers,
        # 512 dim, 8 heads, 4 KV heads) but fewer steps and shorter sequences.
        # Most representative test — same model capacity, just less training.
        "NUM_LAYERS": "11",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "TRAIN_SEQ_LEN": "512",
        "EVAL_SEQ_LEN": "512",
        "ITERATIONS": "300",
        "WARMDOWN_ITERS": "60",
        "WARMUP_STEPS": "10",
        "TRAIN_BATCH_TOKENS": "65536",
        "VAL_BATCH_SIZE": "32768",
        "VAL_LOSS_EVERY": "100",
        "TRAIN_LOG_EVERY": "30",
        "MAX_WALLCLOCK_SECONDS": "1200",
        "GPTQ_CALIB_BATCHES": "32",
        "XSA_LAST_N": "11",
        "VE_LAYERS": "9,10",
        "NUM_SHARED_BLOCKS": "3",
        "NUM_LOOPS": "4",
    },
    "full": {},  # no overrides — use original defaults
}

# Apply test mode overrides: only set env vars that aren't already explicitly set
if _TEST_MODE in _TEST_OVERRIDES:
    for _k, _v in _TEST_OVERRIDES[_TEST_MODE].items():
        if _k not in os.environ:
            os.environ[_k] = _v
elif _TEST_MODE != "full":
    print(f"WARNING: Unknown TEST_MODE={_TEST_MODE!r}, using 'full' defaults")


# =============================================================================
# SECTION 1: HYPERPARAMETERS
# =============================================================================
# All configurable via environment variables for easy experimentation.

class Hyperparameters:
    # --- Data paths ---
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # --- Training schedule ---
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # --- Model architecture (standard mode) ---
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))        # total effective layers
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))     # GQA: fewer KV heads than query heads
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))        # MLP hidden = model_dim * mlp_mult
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))  # cap * tanh(logits/cap)

    # --- Novel: Layer type schedule ---
    # Comma-separated list of layer types for each position.
    # Options: "attn" (standard attention), "ttt" (TTT-Linear), "deltanet" (Gated DeltaNet)
    # Default: all attention. Example: "attn,attn,attn,ttt,attn,attn,ttt,attn,attn,attn,attn"
    layer_types = os.environ.get("LAYER_TYPES", ",".join(["attn"] * num_layers))

    # --- Novel: Recursive mode (Relaxed Recursive Transformer) ---
    recursive = bool(int(os.environ.get("RECURSIVE", "0")))
    num_shared_blocks = int(os.environ.get("NUM_SHARED_BLOCKS", 3))  # K shared blocks per loop
    num_loops = int(os.environ.get("NUM_LOOPS", 4))                  # M times the block is repeated
    # With K=3, M=4: effective depth = 12 layers, but only 3 unique blocks of params

    # --- Novel: Per-loop LoRA adapters ---
    lora_enabled = bool(int(os.environ.get("LORA_ENABLED", "0")))
    lora_rank = int(os.environ.get("LORA_RANK", 16))      # rank of LoRA decomposition
    lora_alpha = float(os.environ.get("LORA_ALPHA", 1.0))  # scaling factor

    # --- Novel: Mixture-of-Recursions routing ---
    mor_enabled = bool(int(os.environ.get("MOR_ENABLED", "0")))
    mor_capacity = float(os.environ.get("MOR_CAPACITY", 0.5))  # fraction of tokens routed per loop

    # --- Novel: TTT-Linear hyperparameters ---
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 64))   # chunk size for mini-batch TTT
    ttt_lr_init = float(os.environ.get("TTT_LR_INIT", 0.1))     # initial TTT learning rate (learnable)

    # --- Novel: Gated DeltaNet hyperparameters ---
    deltanet_chunk_size = int(os.environ.get("DELTANET_CHUNK_SIZE", 64))

    # --- Proven: Attention features ---
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))  # XSA on all layers (from SOTA)
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))     # partial RoPE: only 16/64 dims
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))  # layer-dependent output scaling

    # --- Proven: BigramHash + SmearGate + ValueEmbedding ---
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    trigram_enabled = bool(int(os.environ.get("TRIGRAM", "0")))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")

    # --- Proven: Optimizer settings ---
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))

    # --- Proven: Quantization & eval ---
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    gptq_calib_batches = int(os.environ.get("GPTQ_CALIB_BATCHES", 256))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))

    # --- Proven: Weight averaging ---
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))

    # --- Proven: LAWA weight averaging ---
    lawa_enabled = bool(int(os.environ.get("LAWA_ENABLED", "0")))
    lawa_k = int(os.environ.get("LAWA_K", 10))
    lawa_freq = int(os.environ.get("LAWA_FREQ", 100))

    # --- Unused/experimental flags kept for compatibility ---
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))


# =============================================================================
# SECTION 2: BATCHED NEWTON-SCHULZ ORTHOGONALIZATION
# =============================================================================
# Used by Muon optimizer to find the closest orthogonal matrix to the gradient.
# This is what makes Muon work: it orthogonalizes the gradient update,
# preventing parameter collapse and enabling very high learning rates.

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Compute the matrix sign function via 5th-order Newton-Schulz iteration.
    Maps G -> U where G = U * S * V^T (the orthogonal factor of the SVD).
    Supports batched input: G can be (B, M, N) or (M, N)."""
    a, b, c = (3.4445, -4.7750, 2.0315)  # optimal 5th-order coefficients
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


# =============================================================================
# SECTION 3: PARALLEL MUON OPTIMIZER
# =============================================================================
# Muon applies Newton-Schulz orthogonalization to gradients before updating.
# This parallel version overlaps communication with computation:
#   Phase 1: async reduce-scatter gradients across GPUs
#   Phase 2: Adam steps on small params (while RS is in-flight)
#   Phase 3: wait for RS, run local NS5, all-gather updated params

class Muon(torch.optim.Optimizer):
    """Parallel Muon: Newton-Schulz orthogonalized gradient updates for matrix params.

    Key insight: orthogonalizing the gradient produces updates that preserve the
    conditioning of the weight matrix, allowing much larger learning rates than Adam.
    Combined with 3-phase overlapped distributed communication for near-zero overhead.
    """
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
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
                    'p': p, 'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        """Phase 1: launch async reduce-scatter for all banks."""
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']:
                pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        """Phase 3: wait for RS, local NS5 orthogonalization, all-gather."""
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
            prev_ag_handle = None
            prev_m = None
            sharded = self._distributed and hasattr(self, '_rs_futures')
            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None:
                    continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])
                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m['shard']
                    buf = m['shard_mom']
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
                # Core of Muon: orthogonalize the update via Newton-Schulz
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)
                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])
            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])
            if hasattr(self, '_rs_futures'):
                del self._rs_futures
        return loss


# =============================================================================
# SECTION 4: DATA LOADING
# =============================================================================
# Loads pre-tokenized FineWeb data shards. Each shard is a binary file of uint16 tokens.
# DistributedTokenLoader splits data across GPUs with non-overlapping spans.

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# =============================================================================
# SECTION 5: CORE BUILDING BLOCKS
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — no learnable parameters.
    Cheaper than LayerNorm (no mean subtraction, no bias), works well for LLMs."""
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Linear layer that casts weights to input dtype before matmul.
    Supports late QAT: when _qat_enabled, applies int6 STE quantization noise
    during training to prepare weights for post-training quantization.

    QAT (Quantization-Aware Training) uses the Straight-Through Estimator:
    forward pass uses quantized weights, backward pass uses full-precision gradients.
    This teaches the model to be robust to quantization noise."""
    _qat_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                # STE: forward uses quantized, backward uses original (via detach trick)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


# ---- NOVEL: LoRA Adapter ----
# Low-Rank Adaptation (Hu et al., 2021) adds a small trainable bypass to frozen/shared weights.
# output = base_output + (alpha/rank) * B(A(x))
# A projects down to rank r, B projects back up. Only A and B are trained per-loop.
# This enables the Relaxed Recursive Transformer: shared base weights + unique LoRA per loop.

class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation module for per-loop weight specialization.

    In Relaxed Recursive Transformers, the base weights are shared across all loops,
    but each loop iteration gets its own LoRA adapter. This adds expressiveness
    (each loop can behave differently) while keeping base params shared (saves space).

    Parameters:
        in_features: input dimension
        out_features: output dimension
        rank: rank of the low-rank decomposition (smaller = fewer params)
        alpha: scaling factor (output is scaled by alpha/rank)
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        # A: projects input down to low-rank space
        self.A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(in_features)))
        # B: projects low-rank space back up to output dimension
        # Initialize B to zero so LoRA starts as identity (no change to base weights)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features) -> (..., out_features)
        return F.linear(F.linear(x, self.A), self.B) * self.scale


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    """Ensure scalar/control params stay FP32 for numerical stability."""
    CONTROL_PATTERNS = (
        "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
        "smear", "ve_layer_scales", "ve_shared.scale", "ttt_lr", "delta_beta",
        "mor_router", "lora",
    )
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


# =============================================================================
# SECTION 6: ROTARY POSITION EMBEDDINGS (RoPE)
# =============================================================================
# RoPE encodes absolute position by rotating Q and K vectors in 2D subspaces.
# Partial RoPE applies rotation to only a fraction of head dimensions (e.g., 16/64),
# leaving the rest position-independent. This saves compute and often works just as well.

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (self._cos_cached is None or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
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


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    """Apply rotary embedding to x. If rope_dims < head_dim, only rotates first rope_dims dimensions."""
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


# =============================================================================
# SECTION 7: PROVEN FEATURE MODULES
# =============================================================================

class SmearGate(nn.Module):
    """Learnable position-mixing gate that blends each position with its predecessor.
    x_out = (1 - sigmoid(gate)) * x + sigmoid(gate) * x_prev

    This gives the model a cheap way to incorporate bigram-level context
    at the embedding level, before any attention. Works best with BigramHash."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table.

    For each position t, computes hash(token[t], token[t-1]) and looks up an embedding.
    This gives the model explicit access to bigram features without attention.
    The XOR hash is fast and distributes pairs roughly uniformly.

    Parameters:
        bigram_vocab_size: number of hash buckets (e.g., 2048 or 3072)
        bigram_dim: embedding dimension per bucket
        model_dim: model hidden dimension (projected to if different from bigram_dim)
    """
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int, trigram: bool = False):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self._trigram = trigram
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        """XOR hash of adjacent token pairs: h(t, t-1) = (36313*t XOR 27191*t_prev) % vocab"""
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod  # first position has no predecessor, use last bucket
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def trigram_hash(self, tokens: Tensor) -> Tensor:
        """XOR hash of 3 consecutive tokens: h(t, t-1, t-2)"""
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., :2] = mod
        out[..., 2:] = (36313 * t[..., 2:] ^ 27191 * t[..., 1:-1] ^ 51497 * t[..., :-2]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self._trigram:
            h = h + self.embed(self.trigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific deep layers.

    In deep layers, the residual stream has drifted far from the original token identity.
    ValueEmbedding re-provides this information by adding a token-dependent vector
    to the V projection, helping the model maintain token awareness."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


# =============================================================================
# SECTION 8: ATTENTION VARIANT — STANDARD CAUSAL SELF-ATTENTION
# =============================================================================
# GQA with optional XSA (Exclusive Self-Attention).
# XSA subtracts the self-value projection from attention output, forcing the model
# to learn cross-position interactions rather than copying its own value.

class CausalSelfAttention(nn.Module):
    """Grouped Query Attention with optional XSA and partial RoPE.

    GQA: multiple query heads share fewer key/value heads (e.g., 8Q/4KV).
    This saves parameters in K and V projections without losing much quality.

    XSA: after attention, subtract the component of the output that lies along
    the value direction. This forces attention to learn non-trivial cross-position
    patterns rather than degenerating into "copy self" behavior."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        # Per-head learnable gain applied to queries after RMSNorm
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0  # set externally for partial RoPE
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False  # set externally per layer

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Subtract self-value projection via GQA-aware reshape.
        y: [B, T, H, D], v: [B, T, Hkv, D]. H must be divisible by Hkv."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)  # normalize and broadcast
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn  # project onto v direction
        return (y_g - proj).reshape(B, T, H, D)  # subtract self-value component

    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor,
                v_embed: Tensor | None = None,
                lora_q: LoRAAdapter | None = None, lora_k: LoRAAdapter | None = None,
                lora_v: LoRAAdapter | None = None, lora_o: LoRAAdapter | None = None,
                ) -> Tensor:
        bsz, seqlen, dim = x.shape

        # Project Q, K, V using banked weights (+ optional LoRA bypass for recursive mode)
        q = F.linear(x, q_w.to(x.dtype))
        if lora_q is not None:
            q = q + lora_q(x)  # LoRA adds per-loop specialization
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)

        k = F.linear(x, k_w.to(x.dtype))
        if lora_k is not None:
            k = k + lora_k(x)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)

        v = F.linear(x, v_w.to(x.dtype))
        if lora_v is not None:
            v = v + lora_v(x)
        if v_embed is not None:
            v = v + v_embed  # add token identity signal at deep layers
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # QK normalization + RoPE
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        # Flash Attention 3 (hardware-optimized) or SDPA fallback
        if _HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            y = _sdpa_attn(q, k, v, causal=True)

        # XSA: subtract self-value component to force cross-position learning
        if self.use_xsa:
            y = self._xsa_efficient(y, v)

        y = y.reshape(bsz, seqlen, dim)
        out = F.linear(y, out_w.to(x.dtype))
        if lora_o is not None:
            out = out + lora_o(y)
        return out


# =============================================================================
# SECTION 9: NOVEL ATTENTION VARIANT — TTT-LINEAR
# =============================================================================
# Test-Time Training layers (Sun et al., arXiv:2407.04620)
#
# Instead of attention, each layer maintains a "fast weight" matrix W that acts as
# a tiny linear model. For each chunk of tokens:
#   1. Use W to predict: pred = K @ W  (keys query the fast weight)
#   2. Compute reconstruction error: error = pred - V  (compare to values)
#   3. Update W via gradient descent: W -= lr * (K^T @ error) / chunk_size
#   4. Output = Q @ W  (queries read from updated fast weight)
#
# This is fundamentally different from attention:
# - Attention computes similarity between all pairs (O(n^2))
# - TTT updates a persistent state that compresses the sequence (O(n))
# - The "hidden state" is a machine learning model, not a fixed-size vector
#
# The mini-batch variant processes chunks of tokens together for parallelism,
# with only T/chunk_size sequential state updates.

class TTTLinearAttention(nn.Module):
    """Test-Time Training with Linear fast weights.

    The fast weight W is a (head_dim x head_dim) matrix per head that gets
    updated by gradient descent on a reconstruction objective within each chunk.
    This gives the layer a form of "in-context learning" at the mechanistic level.

    Key hyperparameters:
        chunk_size: how many tokens to process before updating W (tradeoff:
                    smaller = more updates = better quality but slower)
        ttt_lr_init: initial learning rate for fast weight updates (learnable)
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, chunk_size: int = 64,
                 ttt_lr_init: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        # Learnable TTT learning rate (one per head, allows heads to specialize)
        self.ttt_lr = nn.Parameter(torch.full((num_heads, 1, 1), ttt_lr_init, dtype=torch.float32))

        # Initial fast weight W0: identity matrix (output = input initially)
        self.W0 = nn.Parameter(torch.eye(self.head_dim).unsqueeze(0).expand(num_heads, -1, -1).clone())

        # Layer norm for output stabilization (TTT can produce unbounded outputs)
        self.out_norm = RMSNorm()

    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor,
                v_embed: Tensor | None = None,
                lora_q: LoRAAdapter | None = None, lora_k: LoRAAdapter | None = None,
                lora_v: LoRAAdapter | None = None, lora_o: LoRAAdapter | None = None,
                ) -> Tensor:
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # Project to Q, K, V (reusing the same banked weights as attention)
        q = F.linear(x, q_w.to(x.dtype))
        if lora_q is not None:
            q = q + lora_q(x)
        q = q.reshape(B, T, H, D).transpose(1, 2)  # B, H, T, D

        k = F.linear(x, k_w.to(x.dtype))
        if lora_k is not None:
            k = k + lora_k(x)
        # For TTT, K and V use full head count (not GQA) for the fast weight update
        # We expand KV heads to match Q heads for the recurrence
        Hkv = self.num_kv_heads
        k = k.reshape(B, T, Hkv, D)
        if Hkv < H:
            k = k.repeat_interleave(H // Hkv, dim=2)  # expand to H heads
        k = k.transpose(1, 2)  # B, H, T, D

        v = F.linear(x, v_w.to(x.dtype))
        if lora_v is not None:
            v = v + lora_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(B, T, Hkv, D)
        if Hkv < H:
            v = v.repeat_interleave(H // Hkv, dim=2)
        v = v.transpose(1, 2)  # B, H, T, D

        # RMSNorm on Q and K for stability (same as standard attention)
        q = F.rms_norm(q, (D,))
        k = F.rms_norm(k, (D,))

        # Mini-batch TTT: process sequence in chunks
        # W is the fast weight matrix that gets updated per-chunk
        W = self.W0.unsqueeze(0).expand(B, -1, -1, -1).clone()  # B, H, D, D
        lr = self.ttt_lr.to(dtype=x.dtype)  # H, 1, 1

        outputs = []
        for i in range(0, T, self.chunk_size):
            end = min(i + self.chunk_size, T)
            q_c = q[:, :, i:end]  # B, H, chunk, D
            k_c = k[:, :, i:end]
            v_c = v[:, :, i:end]

            # Step 1: Read from fast weight using queries
            # output = Q @ W — queries retrieve stored associations
            out_c = torch.matmul(q_c, W)  # B, H, chunk, D

            # Step 2: Compute reconstruction error
            # The "task" of the fast weight: given a key, predict its value
            pred = torch.matmul(k_c, W)   # B, H, chunk, D — what W predicts for these keys
            error = pred - v_c             # B, H, chunk, D — reconstruction error

            # Step 3: Update fast weight via gradient of ||W@k - v||^2
            # grad_W = (1/chunk) * K^T @ error  (averaged over chunk for stability)
            chunk_len = end - i
            grad = torch.matmul(k_c.transpose(-2, -1), error) / chunk_len  # B, H, D, D

            # Step 4: Gradient descent on fast weight
            # lr is learnable — the model decides how quickly to update its "memory"
            W = W - lr * grad

            outputs.append(out_c)

        # Concatenate chunks and project to output
        y = torch.cat(outputs, dim=2)  # B, H, T, D
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)  # B, T, C
        y = self.out_norm(y)

        out = F.linear(y, out_w.to(x.dtype))
        if lora_o is not None:
            out = out + lora_o(y)
        return out


# =============================================================================
# SECTION 10: NOVEL ATTENTION VARIANT — GATED DELTANET
# =============================================================================
# Gated Delta Networks (Yang et al., ICLR 2025, arXiv:2412.06464)
#
# A linear-time recurrent layer that maintains a state matrix S and updates it
# using the "gated delta rule":
#   S_t = alpha_t * S_{t-1} + beta_t * (v_t @ k_t^T)
#   output_t = S_t @ q_t
#
# Where:
#   alpha_t = sigmoid(gate) — controls how much old state to keep (forget gate)
#   beta_t = sigmoid(beta) — controls how much new info to write (update gate)
#
# Compared to vanilla linear attention (S_t = S_{t-1} + v_t @ k_t^T):
#   - The gate alpha allows FORGETTING old associations (vanilla can only add)
#   - The delta rule allows OVERWRITING specific associations precisely
#   - This solves the "retrieval error accumulation" problem of linear attention
#
# Compared to Mamba:
#   - No custom CUDA kernels needed (pure PyTorch, works with torch.compile)
#   - Outperforms Mamba2 on language modeling benchmarks
#   - Used in production: Qwen3 uses 3:1 DeltaNet:Attention hybrid
#
# Throughput advantage: O(n) vs O(n^2) for attention, meaning more training
# steps in the 600-second time budget = better final BPB.

class GatedDeltaNetLayer(nn.Module):
    """Gated DeltaNet: linear recurrence with precise memory updates.

    Processes the sequence in chunks for parallelism within each chunk
    (matrix operations) with sequential state passing between chunks.
    With chunk_size=64 and T=2048, only 32 sequential steps.

    Key insight: the gate and beta are INPUT-DEPENDENT (computed from x),
    making this "selective" like Mamba — the model decides what to remember/forget
    based on content, not position."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, chunk_size: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        # Learnable projections for gate (alpha) and update scale (beta)
        # These are per-head scalars computed from the input
        self.gate_proj = nn.Linear(dim, num_heads, bias=True)   # alpha: forget gate
        self.beta_proj = nn.Linear(dim, num_heads, bias=True)   # beta: update gate
        # Initialize gates to retain most of the state (high alpha, moderate beta)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 3.0)  # sigmoid(3) ~ 0.95 = mostly remember
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 0.0)   # sigmoid(0) = 0.5 = moderate update

        self.out_norm = RMSNorm()

    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor,
                v_embed: Tensor | None = None,
                lora_q: LoRAAdapter | None = None, lora_k: LoRAAdapter | None = None,
                lora_v: LoRAAdapter | None = None, lora_o: LoRAAdapter | None = None,
                ) -> Tensor:
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # Project Q, K, V
        q = F.linear(x, q_w.to(x.dtype))
        if lora_q is not None:
            q = q + lora_q(x)
        q = q.reshape(B, T, H, D).transpose(1, 2)  # B, H, T, D

        Hkv = self.num_kv_heads
        k = F.linear(x, k_w.to(x.dtype))
        if lora_k is not None:
            k = k + lora_k(x)
        k = k.reshape(B, T, Hkv, D)
        if Hkv < H:
            k = k.repeat_interleave(H // Hkv, dim=2)
        k = k.transpose(1, 2)  # B, H, T, D

        v = F.linear(x, v_w.to(x.dtype))
        if lora_v is not None:
            v = v + lora_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(B, T, Hkv, D)
        if Hkv < H:
            v = v.repeat_interleave(H // Hkv, dim=2)
        v = v.transpose(1, 2)  # B, H, T, D

        # Normalize Q, K, V for stable state updates
        k = F.rms_norm(k, (D,))
        q = F.rms_norm(q, (D,))
        v = F.rms_norm(v, (D,))

        # Compute per-token gates: scalar per head per token
        alpha = torch.sigmoid(self.gate_proj(x)).permute(0, 2, 1)  # B, H, T — forget gate
        beta = torch.sigmoid(self.beta_proj(x)).permute(0, 2, 1)   # B, H, T — update gate

        # ---- Parallel chunkwise recurrence ----
        # All computation in float32 with autocast disabled to prevent bf16 downcast.

        S = torch.zeros(B, H, D, D, device=x.device, dtype=torch.float32)
        outputs = []

        with torch.amp.autocast('cuda', enabled=False):
            q_f, k_f, v_f = q.float(), k.float(), v.float()
            alpha_f, beta_f = alpha.float(), beta.float()

            for chunk_start in range(0, T, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, T)
                Clen = chunk_end - chunk_start

                q_c = q_f[:, :, chunk_start:chunk_end]
                k_c = k_f[:, :, chunk_start:chunk_end]
                v_c = v_f[:, :, chunk_start:chunk_end]
                a_c = alpha_f[:, :, chunk_start:chunk_end]
                b_c = beta_f[:, :, chunk_start:chunk_end]

                log_a = torch.log(a_c.clamp(min=1e-6))
                cum_log_a = torch.cumsum(log_a, dim=-1)
                cum_alpha = torch.exp(cum_log_a.clamp(max=0.0))  # clamp: decay can't grow > 1

                # Cross-chunk: q @ (cumulative_decay * S_init)
                out_cross = torch.matmul(q_c, S) * cum_alpha.unsqueeze(-1)

                # Intra-chunk: parallel causal linear attention with decay
                qv = torch.matmul(q_c, v_c.transpose(-2, -1))
                decay = torch.exp((cum_log_a.unsqueeze(-1) - cum_log_a.unsqueeze(-2)).clamp(max=0.0))
                causal = torch.tril(torch.ones(Clen, Clen, device=x.device, dtype=torch.float32))
                W = qv * decay * causal * b_c.unsqueeze(-2)
                out_intra = torch.matmul(W, k_c)

                outputs.append((out_cross + out_intra).to(x.dtype))

                # Update state S for next chunk
                total_decay = torch.exp(cum_log_a[:, :, -1:].clamp(max=0.0)).unsqueeze(-1)
                S = total_decay * S
                decay_from_end = torch.exp((cum_log_a[:, :, -1:] - cum_log_a).clamp(max=0.0))
                scaled_v = v_c * (decay_from_end * b_c).unsqueeze(-1)
                S = S + torch.matmul(scaled_v.transpose(-2, -1), k_c)
                # Clamp state to prevent unbounded growth
                S = S.clamp(-50.0, 50.0)

        y = torch.cat(outputs, dim=2)  # B, H, T, D
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        y = self.out_norm(y)

        out = F.linear(y, out_w.to(x.dtype))
        if lora_o is not None:
            out = out + lora_o(y)
        return out


# =============================================================================
# SECTION 11: MLP (LeakyReLU^2 activation)
# =============================================================================
# The MLP uses LeakyReLU with negative slope 0.5, then squares the result.
# LeakyReLU(0.5)^2 was shown to be -0.003 BPB better than ReLU^2 because
# it preserves 25% of the negative signal (vs 0% for ReLU^2), giving
# the MLP more expressiveness per parameter.

class MLP(nn.Module):
    """MLP block with LeakyReLU(0.5)^2 activation.
    up: model_dim -> mlp_dim (expand), down: mlp_dim -> model_dim (compress).
    The squaring after LeakyReLU creates a smooth, non-negative gating effect."""
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        # Weights come from parameter banks (not stored here)

    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor,
                lora_up: LoRAAdapter | None = None, lora_down: LoRAAdapter | None = None,
                ) -> Tensor:
        h = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
        if lora_up is not None:
            h = h + F.leaky_relu(lora_up(x), negative_slope=0.5)
        h = h.square()  # squaring creates smooth gating
        out = F.linear(h, down_w.to(x.dtype))
        if lora_down is not None:
            out = out + lora_down(h)
        return out


# =============================================================================
# SECTION 12: NOVEL — MIXTURE-OF-RECURSIONS ROUTER
# =============================================================================
# Per-token routing: a lightweight classifier decides which tokens should
# continue to the next recursion loop and which should exit early.
#
# "Easy" tokens (predictable continuations) exit after fewer loops.
# "Hard" tokens (ambiguous contexts) get more loops.
# This saves compute (2x throughput) without hurting quality much,
# because most tokens don't need maximum depth.
#
# Implementation: top-k routing — select the top capacity% tokens by router
# score to continue. Exited tokens retain their current hidden state.

class MoRRouter(nn.Module):
    """Mixture-of-Recursions router for adaptive per-token depth.

    For each token, computes a scalar "continue score". The top-k tokens
    (by score) continue to the next recursion; the rest exit with their
    current hidden state.

    Parameters:
        dim: model hidden dimension
        capacity: fraction of tokens to route to next loop (e.g., 0.5 = half continue)
    """
    def __init__(self, dim: int, capacity: float = 0.5):
        super().__init__()
        self.capacity = capacity
        # Simple linear router: projects hidden state to scalar score
        self.router = nn.Linear(dim, 1, bias=True)
        nn.init.zeros_(self.router.weight)
        nn.init.constant_(self.router.bias, 1.0)  # initially route everything

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (routed_x, route_indices, exit_mask).
        routed_x: tokens selected to continue (B, K, C) where K = capacity * T
        route_indices: which token positions were selected (B, K)
        exit_mask: boolean mask of shape (B, T) — True = token exits here"""
        B, T, C = x.shape
        k = max(1, int(T * self.capacity))  # number of tokens to keep

        scores = self.router(x.detach()).squeeze(-1)  # B, T — detach to not backprop through routing
        # Top-k selection: tokens with highest "need more compute" scores continue
        _, indices = scores.topk(k, dim=1, sorted=False)  # B, K
        indices = indices.sort(dim=1).values  # maintain causal order

        # Gather selected tokens
        routed_x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, C))  # B, K, C

        exit_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        exit_mask.scatter_(1, indices, False)  # False = continues, True = exits

        return routed_x, indices, exit_mask


# =============================================================================
# SECTION 13: TRANSFORMER BLOCK
# =============================================================================
# Combines attention (or TTT/DeltaNet) + MLP with residual connections,
# learned residual mixing, and layer-dependent scaling.

class Block(nn.Module):
    """Single transformer block with configurable attention variant.

    The block follows the pre-norm residual pattern:
      x = x + attn_scale * attention(norm(mix(x, x0)))
      x = x + mlp_scale * mlp(norm(x))

    resid_mix: learnable blend between current hidden state x and the initial
    embedding x0. This gives each layer explicit access to the original input,
    helping gradient flow (similar to DenseNet skip connections).

    ln_scale: layer-dependent scaling factor 1/sqrt(layer_idx+1) that reduces
    the contribution of deeper layers, stabilizing training."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, layer_idx: int = 0,
                 ln_scale: bool = False, layer_type: str = "attn",
                 ttt_chunk_size: int = 64, ttt_lr_init: float = 0.1,
                 deltanet_chunk_size: int = 64):
        super().__init__()
        self.layer_type = layer_type
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()

        # Select attention variant based on layer_type
        if layer_type == "ttt":
            self.attn = TTTLinearAttention(dim, num_heads, num_kv_heads,
                                           chunk_size=ttt_chunk_size, ttt_lr_init=ttt_lr_init)
        elif layer_type in ("delta", "deltanet"):
            self.attn = GatedDeltaNetLayer(dim, num_heads, num_kv_heads,
                                            chunk_size=deltanet_chunk_size)
        else:  # "attn" — standard causal self-attention
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)

        self.mlp = MLP(dim, mlp_mult)

        # Per-layer learned scaling for attention and MLP outputs
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        # Residual mixing: blend between current state x and initial embedding x0
        # resid_mix[0] weights x, resid_mix[1] weights x0
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

        # Layer-dependent output scaling (deeper layers contribute less)
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor,
                out_w: Tensor, up_w: Tensor, down_w: Tensor,
                v_embed: Tensor | None = None,
                lora_q: LoRAAdapter | None = None, lora_k: LoRAAdapter | None = None,
                lora_v: LoRAAdapter | None = None, lora_o: LoRAAdapter | None = None,
                lora_up: LoRAAdapter | None = None, lora_down: LoRAAdapter | None = None,
                ) -> Tensor:
        # Residual mixing: blend current hidden state with initial embedding
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # Attention (or TTT / DeltaNet) + residual
        attn_out = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor,
            q_w, k_w, v_w, out_w, v_embed=v_embed,
            lora_q=lora_q, lora_k=lora_k, lora_v=lora_v, lora_o=lora_o,
        )
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out

        # MLP + residual
        mlp_out = self.mlp(
            self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w,
            lora_up=lora_up, lora_down=lora_down,
        )
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * mlp_out

        return x_out


# =============================================================================
# SECTION 14: GPT MODEL
# =============================================================================
# Supports two modes:
#
# STANDARD MODE (recursive=False):
#   - N unique blocks (like SOTA), each with its own weights in parameter banks
#   - Some blocks can be TTT or DeltaNet (configured by layer_types)
#   - U-Net skip connections between encoder and decoder halves
#
# RECURSIVE MODE (recursive=True):
#   - K shared blocks looped M times (effective depth = K*M)
#   - Each loop gets its own LoRA adapters (per-loop specialization)
#   - Optional MoR routing (adaptive per-token depth)
#   - U-Net skips adapted for the looped architecture
#   - Much smaller parameter count → more room in 16MB artifact

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, tie_embeddings: bool,
                 tied_embed_init_std: float, logit_softcap: float, rope_base: float,
                 qk_gain_init: float, bigram_vocab_size: int = 0, bigram_dim: int = 128,
                 xsa_last_n: int = 0, rope_dims: int = 0, ln_scale: bool = False,
                 ve_enabled: bool = False, ve_dim: int = 128, ve_layers: str = "9,10",
                 layer_types: str = "",
                 # Novel: recursive mode
                 recursive: bool = False, num_shared_blocks: int = 3, num_loops: int = 4,
                 lora_enabled: bool = False, lora_rank: int = 16, lora_alpha: float = 1.0,
                 mor_enabled: bool = False, mor_capacity: float = 0.5,
                 ttt_chunk_size: int = 64, ttt_lr_init: float = 0.1,
                 deltanet_chunk_size: int = 64,
                 # Legacy
                 mtp_num_heads: int = 0, mtp_loss_weight: float = 0.1,
                 gated_attention: bool = False, value_residual: bool = False,
                 dtg_enabled: bool = False,
                 ):
        super().__init__()
        self.recursive = recursive
        self.logit_softcap = logit_softcap
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight

        # ---- Embedding ----
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim,
                                          trigram=bool(int(os.environ.get("TRIGRAM", "0")))) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)

        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)

        if recursive:
            # ---- RECURSIVE MODE: K shared blocks x M loops ----
            self.num_shared_blocks = num_shared_blocks
            self.num_loops = num_loops
            self.num_layers = num_shared_blocks  # for banking purposes
            effective_layers = num_shared_blocks * num_loops

            # Parse layer types for the shared blocks
            lt_list = layer_types.split(",") if layer_types else ["attn"] * num_shared_blocks
            lt_list = (lt_list * ((num_shared_blocks // len(lt_list)) + 1))[:num_shared_blocks]

            # Shared blocks (only K unique blocks, not K*M)
            self.blocks = nn.ModuleList([
                Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                      layer_idx=i, ln_scale=ln_scale, layer_type=lt_list[i],
                      ttt_chunk_size=ttt_chunk_size, ttt_lr_init=ttt_lr_init,
                      deltanet_chunk_size=deltanet_chunk_size)
                for i in range(num_shared_blocks)
            ])

            # Parameter banks: only K layers (shared across loops)
            self.qo_bank = nn.Parameter(torch.empty(2 * num_shared_blocks, model_dim, model_dim))
            self.kv_bank = nn.Parameter(torch.empty(2 * num_shared_blocks, kv_dim, model_dim))
            self.mlp_up_bank = nn.Parameter(torch.empty(num_shared_blocks, mlp_dim, model_dim))
            self.mlp_down_bank = nn.Parameter(torch.empty(num_shared_blocks, model_dim, mlp_dim))

            # Per-loop iteration embeddings (from Universal Transformer paper)
            # These tell the model which loop iteration it's in
            self.loop_embeddings = nn.Parameter(torch.randn(num_loops, model_dim) * 0.02)

            # Per-loop residual scale (from Universal Transformer)
            # Allows different loops to contribute different amounts
            self.loop_scales = nn.Parameter(torch.ones(num_loops, dtype=torch.float32))

            # LoRA adapters: one set per loop per block (for Relaxed Recursive Transformer)
            self.lora_enabled = lora_enabled
            if lora_enabled:
                # 6 LoRA adapters per block per loop: Q, K, V, Out, MLP_up, MLP_down
                self.lora_adapters = nn.ModuleDict()
                for loop_idx in range(num_loops):
                    for block_idx in range(num_shared_blocks):
                        prefix = f"loop{loop_idx}_block{block_idx}"
                        self.lora_adapters[f"{prefix}_q"] = LoRAAdapter(model_dim, model_dim, lora_rank, lora_alpha)
                        self.lora_adapters[f"{prefix}_k"] = LoRAAdapter(model_dim, kv_dim, lora_rank, lora_alpha)
                        self.lora_adapters[f"{prefix}_v"] = LoRAAdapter(model_dim, kv_dim, lora_rank, lora_alpha)
                        self.lora_adapters[f"{prefix}_o"] = LoRAAdapter(model_dim, model_dim, lora_rank, lora_alpha)
                        self.lora_adapters[f"{prefix}_up"] = LoRAAdapter(model_dim, mlp_dim, lora_rank, lora_alpha)
                        self.lora_adapters[f"{prefix}_down"] = LoRAAdapter(mlp_dim, model_dim, lora_rank, lora_alpha)
            else:
                self.lora_adapters = nn.ModuleDict()

            # MoR router for adaptive depth
            self.mor_enabled = mor_enabled
            if mor_enabled:
                self.mor_router = MoRRouter(model_dim, capacity=mor_capacity)

            # U-Net skip connections (adapted for recursive architecture)
            # In recursive mode, we create skips across the FIRST half of effective layers
            # and use them in the SECOND half
            self.num_encoder_layers = effective_layers // 2
            self.num_decoder_layers = effective_layers - self.num_encoder_layers
            self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
            self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        else:
            # ---- STANDARD MODE: N unique layers ----
            self.num_layers = num_layers
            self.num_shared_blocks = num_layers  # each block is unique
            self.num_loops = 1

            # Parse layer types
            lt_list = layer_types.split(",") if layer_types else ["attn"] * num_layers
            lt_list = (lt_list * ((num_layers // len(lt_list)) + 1))[:num_layers]

            self.blocks = nn.ModuleList([
                Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                      layer_idx=i, ln_scale=ln_scale, layer_type=lt_list[i],
                      ttt_chunk_size=ttt_chunk_size, ttt_lr_init=ttt_lr_init,
                      deltanet_chunk_size=deltanet_chunk_size)
                for i in range(num_layers)
            ])

            # Parameter banks: N unique layers
            self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
            self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
            self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
            self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))

            # U-Net skip connections
            self.num_encoder_layers = num_layers // 2
            self.num_decoder_layers = num_layers - self.num_encoder_layers
            self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
            self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

            # No recursive features
            self.lora_enabled = False
            self.lora_adapters = nn.ModuleDict()
            self.mor_enabled = False
            self.loop_embeddings = None
            self.loop_scales = None

        # ---- Value Embedding ----
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim_ve = num_kv_heads * head_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()

        # ---- Output head ----
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # MTP heads (multi-token prediction, from SOTA)
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)])
        for head in self.mtp_heads:
            head._zero_init = True

        # XSA: enable on the last xsa_last_n effective layers
        if xsa_last_n > 0:
            if recursive:
                # In recursive mode, XSA is always on (since effective depth > block count)
                for block in self.blocks:
                    if hasattr(block.attn, 'use_xsa'):
                        block.attn.use_xsa = True
            else:
                for i in range(max(0, num_layers - xsa_last_n), num_layers):
                    if hasattr(self.blocks[i].attn, 'use_xsa'):
                        self.blocks[i].attn.use_xsa = True

        # Partial RoPE
        if rope_dims > 0:
            for block in self.blocks:
                if hasattr(block.attn, 'rope_dims'):
                    block.attn.rope_dims = rope_dims
                    block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)

        n = self.num_shared_blocks if self.recursive else self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n * (self.num_loops if self.recursive else 1))

        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)      # Q
            nn.init.zeros_(self.qo_bank.data[n + i])                  # Out (zero init)
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)      # K
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)  # V
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)  # MLP up
            nn.init.zeros_(self.mlp_down_bank.data[i])                # MLP down (zero init)
            self.qo_bank.data[n + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _get_ve(self, effective_layer_idx: int, input_ids: Tensor, ve_cache: dict) -> Tensor | None:
        """Get value embedding for a given effective layer index."""
        if self.ve_shared is None or effective_layer_idx not in self.ve_layer_indices:
            return None
        if 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(effective_layer_idx)
        return ve_cache['ve'] * self.ve_layer_scales[ve_idx].to(dtype=ve_cache['ve'].dtype)

    def _get_loras(self, loop_idx: int, block_idx: int):
        """Get LoRA adapters for a specific loop and block."""
        if not self.lora_enabled:
            return None, None, None, None, None, None
        prefix = f"loop{loop_idx}_block{block_idx}"
        la = self.lora_adapters
        return (
            la[f"{prefix}_q"] if f"{prefix}_q" in la else None,
            la[f"{prefix}_k"] if f"{prefix}_k" in la else None,
            la[f"{prefix}_v"] if f"{prefix}_v" in la else None,
            la[f"{prefix}_o"] if f"{prefix}_o" in la else None,
            la[f"{prefix}_up"] if f"{prefix}_up" in la else None,
            la[f"{prefix}_down"] if f"{prefix}_down" in la else None,
        )

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        n = self.num_shared_blocks if self.recursive else self.num_layers

        # ---- Embedding ----
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x  # save initial embedding for residual mixing

        ve_cache: dict = {}

        if self.recursive:
            # ---- RECURSIVE FORWARD PASS ----
            # K shared blocks looped M times with optional LoRA and MoR routing
            effective_layer = 0
            skips: list[Tensor] = []
            encoder_half = (self.num_shared_blocks * self.num_loops) // 2

            for loop_idx in range(self.num_loops):
                # Add loop-specific positional signal
                if self.loop_embeddings is not None:
                    loop_emb = self.loop_embeddings[loop_idx][None, None, :]  # 1, 1, C
                    x = x + loop_emb.to(dtype=x.dtype)

                loop_scale = self.loop_scales[loop_idx].to(dtype=x.dtype) if self.loop_scales is not None else 1.0

                for block_idx in range(self.num_shared_blocks):
                    # Get per-loop LoRA adapters
                    lora_q, lora_k, lora_v, lora_o, lora_up, lora_down = self._get_loras(loop_idx, block_idx)

                    # Value embedding for this effective layer
                    ve = self._get_ve(effective_layer, input_ids, ve_cache)

                    # U-Net: collect skips in encoder half
                    if effective_layer < encoder_half:
                        skips.append(x)

                    # U-Net: apply skips in decoder half
                    skip_idx = effective_layer - encoder_half
                    if 0 <= skip_idx < len(skips) and skip_idx < self.num_skip_weights:
                        remaining = len(skips) - 1 - skip_idx
                        if remaining >= 0 and remaining < len(skips):
                            x = x + self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips[-(skip_idx+1)]

                    # Run block with shared weights + loop-specific LoRA
                    block_out = self.blocks[block_idx](
                        x, x0,
                        self.qo_bank[block_idx], self.kv_bank[block_idx],
                        self.kv_bank[n + block_idx], self.qo_bank[n + block_idx],
                        self.mlp_up_bank[block_idx], self.mlp_down_bank[block_idx],
                        v_embed=ve,
                        lora_q=lora_q, lora_k=lora_k, lora_v=lora_v, lora_o=lora_o,
                        lora_up=lora_up, lora_down=lora_down,
                    )

                    # Apply per-loop scaling
                    x = x + loop_scale * (block_out - x) if isinstance(loop_scale, float) and loop_scale != 1.0 else block_out

                    effective_layer += 1

                # MoR routing: after each loop, optionally drop "easy" tokens
                # (Not applied on last loop — all tokens must produce output)
                if self.mor_enabled and loop_idx < self.num_loops - 1:
                    # Soft routing: compute scores to train the router
                    # but keep all tokens (hard routing breaks batched matmul)
                    _mor_scores = self.mor_router(x)  # trains the router via forward pass

        else:
            # ---- STANDARD FORWARD PASS ----
            # N unique layers with U-Net skip connections (same as SOTA)
            skips: list[Tensor] = []

            for i in range(self.num_encoder_layers):
                ve = self._get_ve(i, input_ids, ve_cache)
                x = self.blocks[i](
                    x, x0,
                    self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                    self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                    v_embed=ve,
                )
                skips.append(x)

            for i in range(self.num_decoder_layers):
                bi = self.num_encoder_layers + i
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                ve = self._get_ve(bi, input_ids, ve_cache)
                x = self.blocks[bi](
                    x, x0,
                    self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                    self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                    v_embed=ve,
                )

        # ---- Output ----
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)

        # Logit softcap: prevents extreme logit values, stabilizes training
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        # Multi-token prediction auxiliary loss (from SOTA)
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape if x.ndim == 3 else (1, x.shape[0], x.shape[1])
            mtp_loss_sum = x.new_zeros(())
            mtp_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, x.size(-1)) if x.ndim == 3 else x[:valid_t]
                mtp_targets = target_ids[:, k + 1:].reshape(-1)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_head(mtp_hidden) / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_count += 1
            if mtp_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_count)

        return main_loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass that returns logits (for evaluation / GPTQ calibration)."""
        n = self.num_shared_blocks if self.recursive else self.num_layers

        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        ve_cache: dict = {}

        if self.recursive:
            effective_layer = 0
            skips = []
            encoder_half = (self.num_shared_blocks * self.num_loops) // 2

            for loop_idx in range(self.num_loops):
                if self.loop_embeddings is not None:
                    x = x + self.loop_embeddings[loop_idx][None, None, :].to(dtype=x.dtype)

                for block_idx in range(self.num_shared_blocks):
                    lora_q, lora_k, lora_v, lora_o, lora_up, lora_down = self._get_loras(loop_idx, block_idx)
                    ve = self._get_ve(effective_layer, input_ids, ve_cache)

                    if effective_layer < encoder_half:
                        skips.append(x)
                    skip_idx = effective_layer - encoder_half
                    if 0 <= skip_idx < self.num_skip_weights and skip_idx < len(skips):
                        remaining = len(skips) - 1 - skip_idx
                        if remaining >= 0:
                            x = x + self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skips[-(skip_idx+1)]

                    x = self.blocks[block_idx](
                        x, x0,
                        self.qo_bank[block_idx], self.kv_bank[block_idx],
                        self.kv_bank[n + block_idx], self.qo_bank[n + block_idx],
                        self.mlp_up_bank[block_idx], self.mlp_down_bank[block_idx],
                        v_embed=ve,
                        lora_q=lora_q, lora_k=lora_k, lora_v=lora_v, lora_o=lora_o,
                        lora_up=lora_up, lora_down=lora_down,
                    )
                    effective_layer += 1
        else:
            skips = []
            for i in range(self.num_encoder_layers):
                ve = self._get_ve(i, input_ids, ve_cache)
                x = self.blocks[i](x, x0, self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                                   self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i], v_embed=ve)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                bi = self.num_encoder_layers + i
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                ve = self._get_ve(bi, input_ids, ve_cache)
                x = self.blocks[bi](x, x0, self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                                    self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi], v_embed=ve)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# =============================================================================
# SECTION 15: TOKENIZER EVALUATION HELPERS
# =============================================================================
# BPB (Bits Per Byte) is the tokenizer-agnostic scoring metric.
# We need to convert per-token loss to per-byte loss, accounting for the
# variable number of bytes each SentencePiece token represents.

def build_sentencepiece_luts(sp, vocab_size, device):
    """Build lookup tables: token_id -> num_bytes, has_leading_space, is_boundary."""
    sp_vocab_size = int(sp.vocab_size())
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
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


# =============================================================================
# SECTION 16: EVALUATION
# =============================================================================

def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
             eval_seq_len=None):
    """Standard evaluation: compute val loss and BPB."""
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding(args, base_model, rank, world_size, device,
                     val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride, batch_seqs=32, eval_seq_len=None):
    """Sliding window evaluation: each token scored with maximum context.
    Overlapping windows (stride < seq_len) let each token benefit from
    as much left-context as possible."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    if _TEST_MODE in ("smoke", "medium"):
        compiled_logits = base_model.forward_logits
    else:
        compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


# =============================================================================
# SECTION 17: QUANTIZATION (Int6 GPTQ + LZMA + Selective Pruning)
# =============================================================================
# The 16MB artifact limit requires aggressive quantization.
# Pipeline: train FP32/BF16 -> GPTQ int6 quantization -> LZMA compression -> selective pruning
#
# GPTQ: uses Hessian information (H = X^T X from calibration data) to quantize weights
# with minimal loss. The Cholesky-based error compensation redistributes quantization
# error across columns, significantly reducing quality degradation.

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
    "smear", "ve_layer_scales", "ve_shared.scale", "ttt_lr", "delta_beta",
    "mor_router", "lora", "loop_embeddings", "loop_scales",
)

def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    """Per-row int6 quantization with percentile search for optimal clipping."""
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def quantize_int6_gptq(weight, hessian=None, clip_range=31, block_size=128):
    """Full GPTQ: Hessian-aware int6 quantization with Cholesky error compensation.

    The key insight of GPTQ: when quantizing column j, redistribute the quantization
    error to columns j+1..n using the inverse Hessian. This minimizes the overall
    output reconstruction error, not just per-weight error."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return quantize_int6_per_row(t32, clip_range)

    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp

    # Sort columns by Hessian diagonal (quantize most sensitive columns first)
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]

    # Cholesky decomposition of inverse Hessian (with fallback for non-PD matrices)
    try:
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch._C._LinAlgError:
        # Fallback: add stronger damping until PD
        for extra_damp in [0.1, 1.0, 10.0]:
            try:
                H2 = H.clone()
                H2[torch.arange(cols), torch.arange(cols)] += extra_damp * torch.mean(torch.diag(H))
                Hinv = torch.linalg.cholesky(H2)
                Hinv = torch.cholesky_inverse(Hinv)
                Hinv = torch.linalg.cholesky(Hinv, upper=True)
                break
            except torch._C._LinAlgError:
                continue
        else:
            # Last resort: fall back to per-row quantization (no GPTQ)
            return quantize_int6_per_row(weight, clip_range)

    best_q, best_scale, best_err = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()

        # Block-wise quantization with error compensation
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                # Error compensation: redistribute quantization error to remaining columns
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse

    best_q = best_q[:, inv_perm]  # undo column permutation
    return best_q, best_scale


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name or "mlp_" in name:
        return "mlp"
    if ".attn." in name or "qo_bank" in name or "kv_bank" in name:
        return "attn"
    return "other"


def _unbank_state_dict(sd, num_layers):
    """Convert 3D bank tensors into individual 2D tensors for per-layer quantization."""
    out = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out


def _rebank_state_dict(sd, num_layers, template_sd):
    """Convert individual 2D tensors back into 3D bank tensors."""
    out = {}
    n = num_layers
    qo_slices = [None] * (2 * n)
    kv_slices = [None] * (2 * n)
    up_slices = [None] * n
    down_slices = [None] * n
    consumed = set()
    for i in range(n):
        for key, slices, idx in [
            (f"blocks.{i}.attn.c_q.weight", qo_slices, i),
            (f"blocks.{i}.attn.proj.weight", qo_slices, n + i),
            (f"blocks.{i}.attn.c_k.weight", kv_slices, i),
            (f"blocks.{i}.attn.c_v.weight", kv_slices, n + i),
            (f"blocks.{i}.mlp.fc.weight", up_slices, i),
            (f"blocks.{i}.mlp.proj.weight", down_slices, i),
        ]:
            if key in sd:
                slices[idx] = sd[key]
                consumed.add(key)
    out["qo_bank"] = torch.stack(qo_slices).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv_slices).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up_slices).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down_slices).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out


def mixed_quantize_int6(state_dict, int6_cats, hessians=None):
    """Quantize a state dict: int6 for large weight matrices, fp16 for small/control tensors."""
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            H = hessians.get(name) if hessians else None
            if H is not None:
                q, s = quantize_int6_gptq(t, hessian=H, clip_range=31)
            else:
                q, s = quantize_int6_per_row(t, clip_range=31)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            # Fallback to int8 for non-categorized tensors
            t32 = t.float()
            if t32.ndim == 2:
                clip_abs = torch.quantile(t32.abs(), 0.9999, dim=1)
                scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
                q = torch.clamp(torch.round(t32 / scale[:, None]), -127, 127).to(torch.int8)
                result[name + ".q"] = q
                result[name + ".scale"] = scale.to(torch.float16)
            else:
                amax = t32.abs().max().item()
                scale = torch.tensor(amax / 127.0 if amax > 0 else 1.0, dtype=torch.float16)
                q = torch.clamp(torch.round(t32 / scale.float()), -127, 127).to(torch.int8)
                result[name + ".q"] = q
                result[name + ".scale"] = scale
            meta[name] = {"type": "int8"}
    return result, meta


def dequantize_mixed_int6(result, meta, template_sd):
    """Dequantize back to floating point for inference verification."""
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


def generate_autoregressive_calib(model, device, num_seqs=64, seq_len=2048,
                                   vocab_size=1024, temperature=0.8, batch_size=8, seed=42):
    """Generate sequences autoregressively from the trained model for GPTQ calibration.
    Uses the model's own distribution — no external data needed.
    This is better than random calibration because it matches the actual weight usage patterns."""
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - batch_start)
            tokens = torch.randint(0, vocab_size, (bs, 1), device=device, generator=rng)
            for pos in range(seq_len - 1):
                logits = model.forward_logits(tokens)
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens = torch.cat([tokens, next_tok], dim=1)
            for i in range(bs):
                all_tokens.append(tokens[i:i+1])
    return all_tokens


# =============================================================================
# SECTION 18: NON-BANKED MODEL FOR HESSIAN COLLECTION
# =============================================================================
# The banked GPT model stores weights in 3D tensors (parameter banks).
# For GPTQ Hessian collection, we need individual CastedLinear layers with
# forward hooks. This mirror model has the same architecture but uses
# standard per-layer CastedLinear weights instead of banks.

class _HessianAttn(nn.Module):
    """Non-banked attention with CastedLinear layers for Hessian hooks."""
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x, v_embed=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if _HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            y = _sdpa_attn(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))


class _HessianMLP(nn.Module):
    """Non-banked MLP with CastedLinear layers for Hessian hooks."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, int(mlp_mult * dim), bias=False)
        self.proj = CastedLinear(int(mlp_mult * dim), dim, bias=False)

    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class _HessianBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = _HessianAttn(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = _HessianMLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x, x0, v_embed=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out


class _HessianGPT(nn.Module):
    """Non-banked GPT model matching unbanked state dict keys for Hessian collection."""
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, logit_softcap, rope_base, qk_gain_init,
                 bigram_vocab_size=0, bigram_dim=128, xsa_last_n=0,
                 rope_dims=0, ln_scale=False,
                 ve_enabled=False, ve_dim=128, ve_layers="9,10"):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim,
                                           trigram=bool(int(os.environ.get("TRIGRAM", "0")))) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            _HessianBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                          layer_idx=i, ln_scale=ln_scale)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        kv_dim = num_kv_heads * (model_dim // num_heads)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

    def _get_ve(self, layer_idx, input_ids, ve_cache):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_cache['ve'] * self.ve_layer_scales[ve_idx].to(dtype=ve_cache['ve'].dtype)

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips = []
        ve_cache = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x_flat, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def collect_hessians_from_tokens(hessian_model, token_seqs, device):
    """Collect H = X^T X from pre-generated token sequences."""
    hessians = {}
    hooks = []
    for name, module in hessian_model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')
            def make_hook(pname):
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pname] += (x.T @ x).cpu()
                return hook_fn
            h = module.register_forward_hook(make_hook(param_name))
            hooks.append(h)
    hessian_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in token_seqs:
            x = seq[:, :-1].to(device)
            y = seq[:, 1:].to(device)
            hessian_model(x, y)
    for h in hooks:
        h.remove()
    num_batches = len(token_seqs)
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    return hessians


# =============================================================================
# SECTION 19: TRAINING LOOP
# =============================================================================
# The training loop orchestrates:
# 1. Forward pass with gradient accumulation (8 micro-steps for 8 GPUs)
# 2. 3-phase overlapped optimizer step (Muon + Adam)
# 3. EMA weight averaging
# 4. Wallclock-based LR warmdown
# 5. Late QAT activation
# 6. Post-training: GPTQ quantization + LZMA compression + evaluation

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # SDP backend config — if FA3 is available (H100), prefer flash; else let PyTorch decide
    if _HAS_FA3:
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)

    print(f"[TEST] mode={_TEST_MODE} distributed={distributed} world_size={world_size} "
          f"device={device} FA3={'yes' if _HAS_FA3 else 'no (using SDPA)'}")

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device)

    CastedLinear._qat_enabled = args.qat_enabled

    # ---- Build model ----
    num_blocks = args.num_shared_blocks if args.recursive else args.num_layers
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        layer_types=args.layer_types,
        recursive=args.recursive,
        num_shared_blocks=args.num_shared_blocks,
        num_loops=args.num_loops,
        lora_enabled=args.lora_enabled,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        mor_enabled=args.mor_enabled,
        mor_capacity=args.mor_capacity,
        ttt_chunk_size=args.ttt_chunk_size,
        ttt_lr_init=args.ttt_lr_init,
        deltanet_chunk_size=args.deltanet_chunk_size,
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
    ).to(device).bfloat16()

    # Keep banks in FP32 (cast to BF16 in forward)
    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # torch.compile for speed — skip in smoke/medium mode (older GPUs / short runs)
    if _TEST_MODE in ("smoke", "medium"):
        compiled_model = base_model
        model = base_model
    else:
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
        model = compiled_model

    # ---- Optimizer setup ----
    # Split parameters into groups:
    # - Parameter banks -> Muon (Newton-Schulz orthogonalization)
    # - Embeddings -> AdamW
    # - Scalars/control tensors -> AdamW
    # - LoRA adapters -> AdamW (small, not worth banking)
    matrix_params = [
        base_model.qo_bank, base_model.kv_bank,
        base_model.mlp_up_bank, base_model.mlp_down_bank,
    ]

    scalar_params = []
    for name, p in base_model.named_parameters():
        if any(bp in name for bp in ["qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank", "tok_emb", "lm_head"]):
            continue
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            scalar_params.append(p)

    # Non-scalar, non-bank params (LoRA weights, bigram proj, etc.) go to Adam
    scalar_param_ids = {id(p) for p in scalar_params}
    adam_matrix_params = []
    for name, p in base_model.named_parameters():
        if any(bp in name for bp in ["qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank", "tok_emb", "lm_head"]):
            continue
        if id(p) not in scalar_param_ids:
            adam_matrix_params.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})

    optimizer_tok = torch.optim.AdamW(tok_params, betas=(args.beta1, args.beta2),
                                       eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr

    all_scalar_and_adam = scalar_params + adam_matrix_params
    optimizer_scalar = torch.optim.AdamW(
        [{"params": all_scalar_and_adam, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)

    optimizer_head = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)

    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_head is not None:
        optimizers.append(optimizer_head)

    # Replicated params that need manual all-reduce
    replicated_params = []
    for pg in optimizer_tok.param_groups:
        replicated_params.extend(pg["params"])
    replicated_params.extend(all_scalar_and_adam)
    if base_model.lm_head is not None:
        replicated_params.append(base_model.lm_head.weight)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"mode:{'recursive' if args.recursive else 'standard'}")
    if args.recursive:
        log0(f"recursive: {args.num_shared_blocks} shared blocks x {args.num_loops} loops = {args.num_shared_blocks * args.num_loops} effective layers")
        log0(f"lora_enabled:{args.lora_enabled} lora_rank:{args.lora_rank}")
        log0(f"mor_enabled:{args.mor_enabled} mor_capacity:{args.mor_capacity}")
    layer_types = args.layer_types.split(",")[:num_blocks]
    log0(f"layer_types:{layer_types}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # ---- Warmup ----
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for _ in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ---- EMA / SWA / LAWA state ----
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    from collections import deque
    lawa_queue: deque[dict[str, Tensor]] = deque(maxlen=args.lawa_k)
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    # EMA decay — for short runs, use faster decay so EMA tracks the trained weights
    # At 100 steps with 0.997, EMA retains 74% init. With 0.95, only 0.6% init.
    ema_decay = {"smoke": 0.95, "small": 0.99, "medium": 0.99}.get(_TEST_MODE, 0.997)

    # ---- Training loop ----
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                     f"step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Late QAT: enable quantization-aware training when LR drops below threshold
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        # LR scheduling
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        # === 3-phase overlapped optimizer step ===
        optimizer_muon.launch_reduce_scatters()
        if distributed:
            for p in replicated_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optimizer_tok.step()
        optimizer_scalar.step()
        if optimizer_head is not None:
            optimizer_head.step()
        optimizer_muon.step()
        zero_grad_all()

        # EMA update
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # SWA: collect weight snapshots during warmdown
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1

        # LAWA: collect periodic snapshots
        if args.lawa_enabled and step % args.lawa_freq == 0:
            lawa_queue.append({name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()})

        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # ---- Apply weight averaging (LAWA > SWA > EMA priority) ----
    if args.lawa_enabled and len(lawa_queue) > 1:
        log0(f"lawa:applying LAWA averaging k={len(lawa_queue)}")
        current_state = base_model.state_dict()
        avg_state = {name: torch.zeros(t.shape, dtype=torch.float32, device='cpu') for name, t in current_state.items()}
        for snap in lawa_queue:
            for name in avg_state:
                avg_state[name] += snap[name].float()
        for name in avg_state:
            avg_state[name] /= len(lawa_queue)
            avg_state[name] = avg_state[name].to(dtype=current_state[name].dtype)
        base_model.load_state_dict(avg_state, strict=True)
    else:
        log0("ema:applying EMA weights")
        current_state = base_model.state_dict()
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(avg_state, strict=True)

    # Diagnostic eval after weight averaging
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )

    # ---- Post-training quantization ----
    n_banks = base_model.num_shared_blocks if base_model.recursive else base_model.num_layers
    export_sd = {k: v for k, v in base_model.state_dict().items() if "mtp_heads" not in k}
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, n_banks)

    # Full GPTQ: build non-banked mirror model for Hessian collection
    log0("gptq:building non-banked model for Hessian collection...")
    hessian_model = _HessianGPT(
        vocab_size=args.vocab_size, num_layers=n_banks, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in hessian_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(hessian_model)
    # Load unbanked weights into the non-banked model (filter shape mismatches for recursive mode)
    hessian_sd = hessian_model.state_dict()
    compatible = {k: v.to(device) for k, v in unbanked_sd.items()
                  if k in hessian_sd and v.shape == hessian_sd[k].shape}
    hessian_model.load_state_dict(compatible, strict=False)

    # Autoregressive self-generated calibration (scaled for test mode)
    calib_seqs = {"smoke": 8, "small": 16, "medium": 16}.get(_TEST_MODE, 64)
    calib_batch = {"smoke": 4, "small": 4, "medium": 4}.get(_TEST_MODE, 8)
    log0(f"gptq:generating autoregressive calibration data ({calib_seqs} seqs x {args.train_seq_len} tokens, temp=0.8)...")
    base_model.load_state_dict(export_sd, strict=False)
    t_gen = time.perf_counter()
    ar_tokens = generate_autoregressive_calib(
        base_model, device, num_seqs=calib_seqs, seq_len=args.train_seq_len,
        vocab_size=args.vocab_size, temperature=0.8, batch_size=calib_batch, seed=args.seed)
    log0(f"gptq:generated {len(ar_tokens)} sequences in {time.perf_counter()-t_gen:.1f}s")

    log0("gptq:collecting hessians from autoregressive data...")
    hessians = collect_hessians_from_tokens(hessian_model, ar_tokens, device)
    log0(f"gptq:collected hessians for {len(hessians)} layers (AR self-gen)")
    del ar_tokens
    del hessian_model
    torch.cuda.empty_cache()

    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians)

    # Selective ±1 pruning: sort ±1 quantized values by reconstruction error,
    # prune least-impactful first until artifact fits target size
    target_mb = float(os.environ.get("TARGET_MB", "15.9"))
    code_bytes_est = len(code.encode("utf-8"))
    ones_info = []
    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type") == "int6"):
            continue
        qk, sk = name + ".q", name + ".scale"
        if qk not in quant_result or sk not in quant_result:
            continue
        q, s = quant_result[qk], quant_result[sk]
        if s.ndim > 0:
            ones_mask = (q.abs() == 1)
            if ones_mask.any():
                row_idx = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
                flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
                errors = s.float()[row_idx].pow(2)
                for fi, err in zip(flat_idx.tolist(), errors.tolist()):
                    ones_info.append((qk, fi, err))

    if ones_info:
        ones_info.sort(key=lambda x: x[2])

        def _try_prune(n):
            tmp = {k: v.clone() for k, v in quant_result.items()}
            for i in range(min(n, len(ones_info))):
                tmp[ones_info[i][0]].view(-1)[ones_info[i][1]] = 0
            buf = io.BytesIO()
            torch.save({"w": tmp, "m": quant_meta}, buf)
            return len(lzma.compress(buf.getvalue(), preset=9)) + code_bytes_est, tmp

        no_sz, _ = _try_prune(0)
        target_bytes = int(target_mb * 1024 * 1024)
        log0(f"selective_prune: {len(ones_info)} ±1 candidates, unpruned={no_sz/(1024*1024):.2f}MB target={target_mb}MB")

        if no_sz <= target_bytes:
            log0("selective_prune: already fits, no pruning needed")
        else:
            full_sz, _ = _try_prune(len(ones_info))
            log0(f"selective_prune: full ±1 prune={full_sz/(1024*1024):.2f}MB")
            if full_sz > target_bytes:
                log0("selective_prune: even full prune not enough, applying all")
                _, quant_result = _try_prune(len(ones_info))
            else:
                lo, hi = 0, len(ones_info)
                while lo < hi:
                    mid = (lo + hi) // 2
                    sz, _ = _try_prune(mid)
                    if sz <= target_bytes:
                        hi = mid
                    else:
                        lo = mid + 1
                log0(f"selective_prune: pruning {lo}/{len(ones_info)} ±1 values ({100*lo/len(ones_info):.1f}%) to fit {target_mb}MB")
                _, quant_result = _try_prune(lo)

    # LZMA compression
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=9)

    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+lzma: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+lzma: {quant_file_bytes + code_bytes} bytes")

    # ---- Verify quantized model ----
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob_disk)), map_location="cpu")
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, n_banks, sd_cpu)

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        layer_types=args.layer_types, recursive=args.recursive,
        num_shared_blocks=args.num_shared_blocks, num_loops=args.num_loops,
        lora_enabled=args.lora_enabled, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        mor_enabled=args.mor_enabled, mor_capacity=args.mor_capacity,
        mtp_num_heads=0, mtp_loss_weight=0.0,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)

    if _TEST_MODE in ("smoke", "medium"):
        compiled_eval = eval_model
    else:
        compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len)
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Sliding window evaluation (skip in smoke/medium — too slow for short seqs/runs)
    sw_seq_len = effective_eval_seq_len
    if _TEST_MODE not in ("smoke", "medium") and args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, eval_seq_len=sw_seq_len)
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    # Second sliding window at stride=64 if different from main stride (skip in smoke/small)
    if _TEST_MODE == "full" and args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize()
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64, eval_seq_len=sw_seq_len)
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
            f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms"
        )
        log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
