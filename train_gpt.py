"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
import concurrent.futures
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run (not actually reflected within Hyperparameters):
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 40))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 2048))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 512))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 8))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 1.75))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_max_base = float(os.environ.get("ROPE_MAX_BASE", 8192.0))
    rope_min_base = float(os.environ.get("ROPE_MIN_BASE", 512.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.025))
    head_lr = float(os.environ.get("HEAD_LR", 0.025))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.015))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.125))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.0275))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.0175))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 256))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    # 1. Eliminate CPU-GPU synchronization (Graph Break)
    # isfinite().all() creates a scalar tensor on the device.
    # Multiplying by this mask eliminates NaNs/Infs without a host-sync.
    valid_mask = torch.isfinite(G).all().to(G.dtype)
    G = G * valid_mask 

    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    
    # 2. Multiply by reciprocal (faster than tensor division)
    X.mul_(1.0 / (X.norm() + eps))

    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        
        # 3. Fuse pointwise operations into the GEMM kernel via addmm
        # Equivalent to: B = b * A + c * (A @ A)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        
        # Equivalent to: X = a * X + 1.0 * (B @ X)
        X = torch.addmm(X, B, X, beta=a, alpha=1.0)

    return X.T if transposed else X

@torch.no_grad()
def get_hadamard_matrix(n, device):
    """Generates a deterministic, orthonormal Hadamard matrix."""
    p2 = 2**math.ceil(math.log2(n))
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < p2:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H[:n, :n] / math.sqrt(p2)

@torch.no_grad()
def apply_zero_init(model, std=0.02):
    """
    Unified ZerO Init: 
    - Embeddings: Hadamard
    - Linear Layers: Last in any sub-module is 0 (Exit), others are Hadamard (Internal).
    """
    for m in model.modules():
        # 1. Handle Embeddings (Identity-like initialization)
        if isinstance(m, nn.Embedding):
            d_out, d_in = m.weight.shape
            H = get_hadamard_matrix(max(d_out, d_in), m.weight.device)
            m.weight.copy_(H[:d_out, :d_in]*std)

        # 2. Handle Linear Layers by inspecting the module's direct children
        # We look for direct Linear children to identify "Branches"
        linears = [sub for sub in m.children() if isinstance(sub, nn.Linear)]
        if linears:
            for i, l in enumerate(linears):
                d_out, d_in = l.weight.shape
                
                # The 'Exit' layer is the last Linear in this specific module
                if i == len(linears) - 1:
                    nn.init.normal_(l.weight, std=1e-5)
                # 'Internal' layers get Hadamard symmetry breaking
                else:
                    H = get_hadamard_matrix(max(d_out, d_in), l.weight.device)
                    l.weight.copy_(H[:d_out, :d_in])
                
                if l.bias is not None:
                    nn.init.zeros_(l.bias)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95, backend_steps: int = 5, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov)
        super().__init__(params, defaults)
        self._is_initialized = False

    def _init_state(self):
        """Pre-allocates buffers for each group individually."""
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            # Check if THIS specific group needs initialization
            if "updates_flat" in group or not group["params"]:
                continue

            params = group["params"]
            total_params = sum(p.numel() for p in params)
            
            # Pre-allocate the communication buffer for this group
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            
            group["updates_flat"] = updates_flat
            group["param_views"] = []
            group["rank_params"] = []
            group["rank_param_views"] = []

            curr = 0
            for i, p in enumerate(params):
                numel = p.numel()
                view = updates_flat[curr : curr + numel].view_as(p)
                group["param_views"].append(view)

                if i % world_size == rank:
                    group["rank_params"].append(p)
                    group["rank_param_views"].append(view)

                curr += numel

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self._init_state()

        for group in self.param_groups:
            if not group["params"]:
                continue

            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            updates_flat = group["updates_flat"]

            updates_flat.zero_()

            # 1. Local Computation
            for p, view in zip(group["rank_params"], group["rank_param_views"]):
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                
                buf = state["momentum_buffer"]
                
                # Momentum update: m = m * momentum + g
                buf.mul_(momentum).add_(g)
                
                # Nesterov logic: use (g + momentum * m) or just m
                u = g.add(buf, alpha=momentum) if nesterov else buf

                # Apply Newton-Schulz (on 2D flattened version if ndim > 2)
                original_shape = u.shape
                if u.ndim > 2:
                    u = u.view(u.size(0), -1)
                
                g_ns = zeropower_via_newtonschulz5(u, steps=backend_steps)
                
                # Scaling factor (standard for Muon)
                g_ns.mul_(max(1, g_ns.size(0) / g_ns.size(1))**0.5)
                
                view.copy_(g_ns.view(original_shape))

            # 2. Distributed Sync
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # 3. Parameter Update (Kernel Fusion)
            # Use foreach_add with a list of tensors to avoid multiple kernel launches
            params = group["params"]
            views = group["param_views"]
            if updates_flat.dtype != params[0].dtype:
                views = [v.to(dtype=p.dtype, non_blocking=True) for v, p in zip(views, params)]
            torch._foreach_add_(params, views, alpha=-lr)

        return loss

# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
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
        if piece.startswith("▁"):
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
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
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

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Initial load and pre-queue the NEXT file
        self.tokens = load_data_shard(self.files[0])
        self.future_tokens = self._queue_next_file()
        self.pos = 0

    def _queue_next_file(self):
        next_idx = (self.file_idx + 1) % len(self.files)
        return self.executor.submit(load_data_shard, self.files[next_idx])

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        # result() will block ONLY if the disk is slower than your training 
        # (unlikely with NVMe), otherwise it returns instantly.
        self.tokens = self.future_tokens.result()
        self.pos = 0
        # Start loading the one AFTER this
        self.future_tokens = self._queue_next_file()

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
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
        self.next_x, self.next_y = None, None
        self.transfer_stream = torch.cuda.Stream(device)

    def preload(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        with torch.cuda.stream(self.transfer_stream):
            local_tokens = global_tokens // (self.world_size * grad_accum_steps)
            per_rank_span = local_tokens + 1
            chunk = self.stream.take(per_rank_span * self.world_size)
            start = self.rank * per_rank_span
            
            # OPTIMIZATION: Move to device as int32 (4 bytes) instead of int64 (8 bytes)
            local = chunk[start : start + per_rank_span].pin_memory().to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )
            # Embedding layer handles int32 perfectly.
            self.next_x = local[:-1].reshape(-1, seq_len)
            self.next_y = local[1:].reshape(-1, seq_len).to(torch.long) # CrossEntropy needs long

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if self.next_x is None:
            self.preload(global_tokens, seq_len, grad_accum_steps)
            
        # Ensure the default stream waits for the background transfer to finish
        torch.cuda.current_stream().wait_stream(self.transfer_stream)
        x, y = self.next_x, self.next_y
        
        # Immediately queue up the next batch in the background
        self.preload(global_tokens, seq_len, grad_accum_steps)
        return x, y

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, p: float = 0.5, base: float = 10000.0):
        super().__init__()
        self.rotary_dim = (int(dim * p) // 8) * 8
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        # Register as float32; shapes: (1, 1, max_seq_len, rotary_dim // 2)
        self.register_buffer("cos", freqs.cos().view(1, 1, max_seq_len, -1), persistent=False)
        self.register_buffer("sin", freqs.sin().view(1, 1, max_seq_len, -1), persistent=False)

    def forward(self, x: Tensor):
        t = x.size(2)
        return self.cos[:, :, :t, :].to(x.dtype), self.sin[:, :, :t, :].to(x.dtype)

def rotate_half(x: Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d = cos.shape[-1] * 2
    x_rop = x[..., :d]
    x_pass = x[..., d:]
    x_rop = x_rop.view(*x_rop.shape[:-1], -1, 2)
    x0 = x_rop[..., 0]
    x1 = x_rop[..., 1]
    res = torch.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1)
    return torch.cat([res.flatten(-2), x_pass], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, seq_len=1024, use_rope=True, rope_proportion=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.c_qk = nn.Linear(dim, dim + self.kv_dim, bias=False)
        self.c_v = nn.Linear(dim, self.kv_dim, bias=False)
        self.v_mix = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, max_seq_len=seq_len, p=rope_proportion, base=rope_base)
        self.use_rope = use_rope

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        qk = self.c_qk(x)
        q, k = qk.split([dim, self.kv_dim], dim=-1)
        v_input = torch.lerp(emb, x, self.v_mix.to(x.dtype))
        v = self.c_v(v_input)

        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        if self.use_rope:
            cos, sin = self.rotary(q)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.view(1, -1, 1, 1).to(q.dtype)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True, 
            enable_gqa=(self.num_kv_heads != self.num_heads)
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        self.hidden_dim = int(mlp_mult * 2/3 * dim // 32) * 32
        self.c_fc = nn.Linear(dim, 2 * self.hidden_dim, bias=False)
        self.c_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        fused_x = self.c_fc(x)
        gate, value = fused_x.chunk(2, dim=-1)
        return self.c_proj(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        qk_gain_init: float,
        seq_len: int=1024,
        use_rope: bool=True,
        resid_scale: float=0.125,
        rope_proportion: float=0.5,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, seq_len, use_rope, rope_proportion)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32).mul(0.05))
        self.resid_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32).mul(resid_scale))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32).mul(0.05))

    # Inside Block.forward
    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        dtype = x.dtype
        normed_x = self.attn_norm(x)
        attn_out = self.attn(normed_x, emb)
        y = x + self.attn_scale.to(dtype) * attn_out
        normed_y = self.mlp_norm(y)
        mlp_out = self.mlp(normed_y)
        return self.resid_scale.to(dtype) * x + y + self.mlp_scale.to(dtype) * mlp_out

def get_linear_progression_kv_heads(layer_idx, total_layers, num_heads):
    # Progresses from 2 heads at layer 0 to num_heads at the final layer
    min_kv = 2
    max_kv = num_heads
    
    # Simple linear interpolation from layer 0 to total_layers
    fraction = layer_idx / (total_layers - 1)
    raw_kv = min_kv + (max_kv - min_kv) * fraction
    
    valid_kvs = [i for i in range(1, num_heads + 1) if num_heads % i == 0]
    kv_heads = min(valid_kvs, key=lambda x: abs(x - raw_kv))
    
    return int(max(min_kv, kv_heads))

def get_rope_p_smooth(i: int, num_layers: int, p_min=0.375, p_max=0.875) -> float:
    if num_layers <= 1:
        return p_min
    progress = i / (num_layers - 1)
    scale = math.sin(progress * math.pi)
    return p_min + (p_max - p_min) * scale

def get_rope_base_progression(layer_idx: int, total_layers: int, min_base: float, max_base: float) -> float:
    if total_layers <= 1:
        return max_base
    fraction = layer_idx / (total_layers - 1)
    return min_base * ((max_base / min_base) ** fraction)

def get_linear_progression_mlp_mult(layer_idx: int, total_layers: int, base_mult: int) -> float:
    # If base_mult is 2, this progresses from 1.0 (Layer 0) to 3.0 (Final Layer)
    min_mult = float(base_mult) * 0.5
    max_mult = float(base_mult) * 1.5
    fraction = layer_idx / max(1, total_layers - 1)
    return min_mult + (max_mult - min_mult) * fraction

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_max_base: float,
        rope_min_base: float,
        qk_gain_init: float,
        seq_len: int=1024,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_layers = num_layers
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList([
            Block(
                model_dim,
                num_heads,
                get_linear_progression_kv_heads(i, num_layers, num_kv_heads),
                get_linear_progression_mlp_mult(i, num_layers, mlp_mult),
                get_rope_base_progression(i, num_layers, rope_min_base, rope_max_base),
                qk_gain_init,
                seq_len=seq_len,
                use_rope=True,
                resid_scale=1/math.sqrt(2 * num_layers),
                rope_proportion=get_rope_p_smooth(i, num_layers)
            ) for i in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.logit_softcap = logit_softcap
        self.lm_head = None if tie_embeddings else nn.Linear(model_dim, vocab_size, bias=False)
        self.skip_scales = nn.Parameter(torch.zeros(self.num_encoder_layers, model_dim, dtype=torch.float32))
        
        apply_zero_init(self, std=self.tied_embed_init_std)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        emb = self.tok_emb(input_ids)
        x = F.rms_norm(emb, (emb.size(-1),))

        skips = []
        for i in range(self.num_encoder_layers):
            skips.append(x)
            x = self.blocks[i](x, emb)
        for j in range(self.num_decoder_layers):
            block_idx = j + self.num_encoder_layers
            skip_idx = self.num_encoder_layers - 1 - j
            if skip_idx >= 0:
                skip_x = F.rms_norm(skips[skip_idx], (skips[skip_idx].size(-1),), eps=self.final_norm.eps)
                x = x + (self.skip_scales[skip_idx] * skip_x)
            
            x = self.blocks[block_idx](x, emb)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
            
        logits = self.logit_softcap * torch.tanh(logits.float() / self.logit_softcap)
        return F.cross_entropy(logits, targets)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    # 1. Force the PyTorch dispatcher to use Tensor Cores for all matmuls
    torch.set_float32_matmul_precision('high')
    import torch._inductor.config as inductor_config
    inductor_config.fx_graph_cache = True               # Caches compiled kernels to disk (saves 5+ minutes on restart)
    inductor_config.triton.unique_kernel_names = True   # Prevents Triton kernel namespace collisions in DDP
    inductor_config.freezing = True                     # Aggressive constant-folding for inference/eval
    inductor_config.shape_padding = True
    inductor_config.epilogue_fusion = True

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    @torch.compiler.disable
    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
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

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

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
        rope_max_base=args.rope_max_base,
        rope_min_base=args.rope_min_base,
        qk_gain_init=args.qk_gain_init,
        seq_len=args.train_seq_len
    ).to(device).bfloat16()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    gpt_scalars = [p for name, p in base_model.named_parameters() if "skip_scales" in name or "skip_gain" in name]
    scalar_params.extend(gpt_scalars)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        warmup_mul = min(step / args.warmup_steps, 1.0) if args.warmup_steps > 0 else 1.0

        # --- Warmdown Logic ---
        if args.warmdown_iters <= 0:
            return warmup_mul
            
        if max_wallclock_ms is None:
            # Iteration-based warmdown
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            warmdown_mul = max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if step >= warmdown_start else 1.0
        else:
            # Time-based warmdown
            step_ms = elapsed_ms / max(step, 1)
            warmdown_ms = args.warmdown_iters * step_ms
            remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
            warmdown_mul = remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

        # The effective multiplier is the intersection of warmup and warmdown
        return min(warmup_mul, warmdown_mul)

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
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
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
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

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)  # ← Use len() instead of getsize()
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()

def main_params():
    args = Hyperparameters()
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
        rope_max_base=args.rope_max_base,
        rope_min_base=args.rope_min_base,
        qk_gain_init=args.qk_gain_init,
        seq_len=args.train_seq_len
    )
    print(sum([p.numel() for p in base_model.parameters()]))

if __name__ == "__main__":
    if not torch.cuda.is_available():
        main_params()
    else:
        main()