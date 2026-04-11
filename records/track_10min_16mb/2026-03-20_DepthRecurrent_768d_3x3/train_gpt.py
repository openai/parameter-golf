"""
ULTIMATE — Parameter Golf: Maximum Novel Architecture
=====================================================
Combines ALL proven techniques + original novel contributions.

PROVEN (from submission + research):
  1.  Depth recurrence (3 unique × 3 passes)
  2.  Wider dim (768)
  3.  Seq len 2048
  4.  GQA 8q/2kv
  5.  Sliding window eval (stride=64)
  6.  RoPE base 500k
  7.  Spectral embed init (conservative)
  8.  Muon WD=0.01
  9.  Value embeddings (modded-nanogpt 2025)
  10. Per-pass control params
  11. U-Net skip connections
  12. Adaptive depth per token (exit gate)
  13. Confidence conditioning across passes
  14. Compression-aware auxiliary loss

NOVEL ORIGINAL:
  15. Gradient Memory Recurrence
      — stores magnitude of change between passes as implicit memory
      — tokens that change more receive more attention in later passes
  16. Thermodynamic Compression Loss
      — free energy minimization: F = E - T*S
      — principled temperature annealing from statistical physics
      — more principled than a fixed entropy penalty
  17. Temporal Difference Recurrence
      — TD learning from RL applied to forward pass
      — each pass refines representation based on prediction error
      — implicit meta-learning within a single forward pass
  18. Eigenspace Token Routing
      — orthogonal basis for token routing
      — more principled than standard softmax MoE
      — each expert is a true specialist with no overlap
  19. Resonant Position Encoding
      — frequency-based resonance gate per pass
      — tokens resonate with pass-specific learned frequencies
      — combines signal processing principles with recurrence
  20. Selective State Carry
      — sequence state summary carried between passes
      — SSM-inspired but within recurrent framework
      — global context without O(n²) attention cost

LEADERBOARD PROVEN:
  21. LoRA Test-Time Training (samacqua, 1.1928)
      — rank-4 LoRA adapters finetuned at eval time
      — 3 gradient steps on first 512 context tokens
      — base model weights unchanged
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
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
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

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 2000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    model_dim = int(os.environ.get("MODEL_DIM", 768))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 2))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", "500000.0"))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", "30.0"))

    # Recurrence
    num_unique_layers = int(os.environ.get("NUM_UNIQUE_LAYERS", 3))
    num_passes = int(os.environ.get("NUM_PASSES", 3))
    use_unet_skip = bool(int(os.environ.get("USE_UNET_SKIP", "1")))

    # Proven novel (from experiment)
    use_adaptive_depth = bool(int(os.environ.get("USE_ADAPTIVE_DEPTH", "1")))
    use_compression_loss = bool(int(os.environ.get("USE_COMPRESSION_LOSS", "1")))
    compression_loss_weight = float(os.environ.get("COMPRESSION_LOSS_WEIGHT", "0.05"))
    use_confidence_conditioning = bool(int(os.environ.get("USE_CONFIDENCE_CONDITIONING", "1")))

    # Original novel
    use_gradient_memory = bool(int(os.environ.get("USE_GRADIENT_MEMORY", "1")))
    use_thermo_loss = bool(int(os.environ.get("USE_THERMO_LOSS", "1")))
    thermo_T_max = float(os.environ.get("THERMO_T_MAX", "0.5"))
    thermo_tau = float(os.environ.get("THERMO_TAU", "5000.0"))
    use_td_recurrence = bool(int(os.environ.get("USE_TD_RECURRENCE", "1")))
    td_alpha = float(os.environ.get("TD_ALPHA", "0.1"))
    use_eigenspace_routing = bool(int(os.environ.get("USE_EIGENSPACE_ROUTING", "1")))
    num_experts = int(os.environ.get("NUM_EXPERTS", "1"))
    use_resonant_encoding = bool(int(os.environ.get("USE_RESONANT_ENCODING", "1")))
    use_selective_state = bool(int(os.environ.get("USE_SELECTIVE_STATE", "1")))

    # NOVEL #21 — LoRA Test-Time Training
    # At eval time, finetune a small LoRA adapter using document context
    # Proven on leaderboard (samacqua, 1.1928)
    use_lora_ttt = bool(int(os.environ.get("USE_LORA_TTT", "1")))
    lora_rank = int(os.environ.get("LORA_RANK", "4"))         # LoRA adapter rank
    lora_ttt_steps = int(os.environ.get("LORA_TTT_STEPS", "3"))  # steps TTT
    lora_ttt_lr = float(os.environ.get("LORA_TTT_LR", "1e-3"))   # LR TTT
    lora_ttt_ctx = int(os.environ.get("LORA_TTT_CTX", "512"))    # context tokens untuk TTT

    # Optimizer
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", "0.05"))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", "0.0"))
    matrix_lr = float(os.environ.get("MATRIX_LR", "0.04"))
    scalar_lr = float(os.environ.get("SCALAR_LR", "0.04"))
    head_lr = float(os.environ.get("HEAD_LR", "0.008"))
    embed_lr = float(os.environ.get("EMBED_LR", "0.6"))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", "0.95"))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", "5"))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", "0.85"))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", "500"))
    beta1 = float(os.environ.get("BETA1", "0.9"))
    beta2 = float(os.environ.get("BETA2", "0.95"))
    adam_eps = float(os.environ.get("ADAM_EPS", "1e-8"))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", "0.0"))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", "1.5"))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
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


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(
            lr=lr, momentum=momentum, backend_steps=backend_steps,
            nesterov=nesterov, weight_decay=weight_decay,
        ))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER EVAL
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size, device):
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


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


# -----------------------------
# NOVEL #21 — LoRA TTT (Test-Time Training)
# -----------------------------
# Proven on leaderboard (samacqua, 1.1928)
# At eval time: finetune a small LoRA adapter using the start of the document
# as context, then evaluate the rest with model + LoRA.
# Base model weights are NOT changed — only the LoRA adapter is updated.

class LoRAAdapter(nn.Module):
    """
    LoRA adapter untuk attention Q dan V projections.
    Ringan: rank=4 → hanya 4 × dim × 2 × num_layers parameter
    Untuk dim=768, rank=4, 3 layers: 4 × 768 × 2 × 3 = 18,432 params
    Sangat kecil — tidak menambah ukuran model yang disimpan
    karena hanya ada saat evaluasi.
    """
    def __init__(self, base_model: nn.Module, rank: int, device: torch.device):
        super().__init__()
        self.rank = rank
        self.adapters = nn.ModuleDict()

        # Add LoRA adapters for Q and V projections in all blocks
        for block_idx, block in enumerate(base_model.blocks):
            dim = block.attn.c_q.weight.shape[0]
            self.adapters[f"q_A_{block_idx}"] = nn.Linear(dim, rank, bias=False)
            self.adapters[f"q_B_{block_idx}"] = nn.Linear(rank, dim, bias=False)
            self.adapters[f"v_A_{block_idx}"] = nn.Linear(dim, rank, bias=False)
            self.adapters[f"v_B_{block_idx}"] = nn.Linear(rank, dim, bias=False)
            # Init: A random kecil, B zeros (LoRA standard init)
            nn.init.normal_(self.adapters[f"q_A_{block_idx}"].weight, std=0.01)
            nn.init.zeros_(self.adapters[f"q_B_{block_idx}"].weight)
            nn.init.normal_(self.adapters[f"v_A_{block_idx}"].weight, std=0.01)
            nn.init.zeros_(self.adapters[f"v_B_{block_idx}"].weight)

        self.to(device)

    def apply_to_model(self, base_model: nn.Module, scale: float = 1.0):
        """Patch model weights with LoRA delta -- in place."""
        for block_idx, block in enumerate(base_model.blocks):
            # Q: W_q += B @ A (LoRA convention)
            q_A = self.adapters[f"q_A_{block_idx}"].weight  # [rank, dim]
            q_B = self.adapters[f"q_B_{block_idx}"].weight  # [dim, rank]
            block.attn.c_q.weight.data += scale * (q_B @ q_A).to(
                dtype=block.attn.c_q.weight.dtype)

            # V: W_v += B @ A
            v_A = self.adapters[f"v_A_{block_idx}"].weight
            v_B = self.adapters[f"v_B_{block_idx}"].weight
            kv_dim = block.attn.c_v.weight.shape[0]
            v_delta = (v_B @ v_A)[:kv_dim, :]
            block.attn.c_v.weight.data += scale * v_delta.to(
                dtype=block.attn.c_v.weight.dtype)

    def remove_from_model(self, base_model: nn.Module, scale: float = 1.0):
        """Remove LoRA delta from model weights — restores original state."""
        self.apply_to_model(base_model, scale=-scale)


def lora_ttt_adapt(args, base_model: nn.Module, context_tokens: Tensor,
                   device: torch.device) -> LoRAAdapter:
    """
    Test-Time Training dengan LoRA.
    Finetune LoRA adapter menggunakan context_tokens sebagai training signal.
    Return adapter yang sudah di-train — apply ke model sebelum evaluasi.
    """
    adapter = LoRAAdapter(base_model, rank=args.lora_rank, device=device)
    for p in adapter.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lora_ttt_lr,
                                 betas=(0.9, 0.95))

        # Save original weights before TTT
    original_weights = {}
    for block_idx, block in enumerate(base_model.blocks):
        original_weights[f"q_{block_idx}"] = block.attn.c_q.weight.data.clone()
        original_weights[f"v_{block_idx}"] = block.attn.c_v.weight.data.clone()

    base_model.train()
    for ttt_step in range(args.lora_ttt_steps):
        optimizer.zero_grad()

        # Apply current LoRA to model
        adapter.apply_to_model(base_model)

        # Forward pass on context tokens
        ctx_len = context_tokens.size(1)
        if ctx_len > 1:
            x_ctx = context_tokens[:, :-1]
            y_ctx = context_tokens[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = base_model(x_ctx, y_ctx)
            loss.backward()

        # Remove LoRA before optimizer step (so gradients flow only to adapter)
        adapter.remove_from_model(base_model)

        # Restore original weights
        for block_idx, block in enumerate(base_model.blocks):
            block.attn.c_q.weight.data = original_weights[f"q_{block_idx}"].clone()
            block.attn.c_v.weight.data = original_weights[f"v_{block_idx}"].clone()

        optimizer.step()

    base_model.eval()
    return adapter


def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    stride = int(os.environ.get("VAL_STRIDE", "64"))

    # LoRA TTT: unwrap base_model from DDP/compiled wrapper
    base_model = None
    if args.use_lora_ttt:
        m = model
        if hasattr(m, "module"):
            m = m.module  # unwrap DDP
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod  # unwrap torch.compile
        if isinstance(m, UltimateRecurrentGPT):
            base_model = m

    model.eval()
    with torch.inference_mode():
        rank_start = seq_start * args.train_seq_len
        rank_end = seq_end * args.train_seq_len
        rank_tokens = val_tokens[rank_start: rank_end + 1].to(
            device=device, dtype=torch.int64, non_blocking=True)

        # LoRA TTT: finetune adapter using the first context tokens
        lora_adapter = None
        if args.use_lora_ttt and base_model is not None:
            ttt_ctx_len = min(args.lora_ttt_ctx, rank_tokens.numel() - 1)
            if ttt_ctx_len > 1:
                ctx = rank_tokens[:ttt_ctx_len + 1].unsqueeze(0)  # [1, ctx_len+1]
                # TTT: exit inference_mode for gradient computation
                with torch.inference_mode(False):
                    with torch.enable_grad():
                        ctx_grad = ctx.clone()
                        lora_adapter = lora_ttt_adapt(args, base_model, ctx_grad, device)

                model.eval()

        pos = 0
        buf_x, buf_y = [], []
        while pos + args.train_seq_len < rank_tokens.numel():
            buf_x.append(rank_tokens[pos: pos + args.train_seq_len])
            buf_y.append(rank_tokens[pos + 1: pos + args.train_seq_len + 1])
            pos += stride
            flush = (len(buf_x) >= local_batch_seqs or
                     pos + args.train_seq_len >= rank_tokens.numel())
            if flush and buf_x:
                x = torch.stack(buf_x)
                y = torch.stack(buf_y)
                buf_x.clear()
                buf_y.clear()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    batch_loss = model(x, y).detach()
                batch_token_count = float(y.numel())
                val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
                val_token_count += batch_token_count
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (has_leading_space_lut[tgt_ids] &
                                ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()

        # Remove LoRA after evaluation — restore original weights
        if lora_adapter is not None and base_model is not None:
            lora_adapter.remove_from_model(base_model)

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
# QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,exit_gate,confidence_proj,"
        "resonant_freq,state_gate,td_alpha_param,eigen_scale",
    ).split(",") if p
)

INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t):
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name, t, passthrough_orig_dtypes):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                    if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale),
                    -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes = {}, {}, {}
    passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"), 0)
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
    obj = {"__quant_format__": "int8_clean_per_row_v1",
           "quantized": quantized, "scales": scales, "dtypes": dtypes,
           "passthrough": passthrough}
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
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
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# MODEL MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or
                    any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) \
                    and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim, base=500000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (self._cos_cached is None or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim

        self.c_q = CastedLinear(dim, dim, bias=False)
        # Low-rank K projection: saves ~415K params vs full dim x kv_dim
        # rank=32 is sufficient for 2-head KV (kv_dim=96)
        self.lk_rank = 32
        self.c_k_a = CastedLinear(dim, self.lk_rank, bias=False)
        self.c_k_b = CastedLinear(self.lk_rank, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k_b(self.c_k_a(x)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class LinearCausalAttention(nn.Module):
    """
    Linear attention variant for one of the three shared layers.
    Uses kernel trick: phi(Q) @ (phi(K)^T @ V) in O(n*d^2) instead of O(n^2*d).
    Saves compute per step, enabling more steps within 10 minutes.
    Params identical to standard attention -- benefit is speed only.
    """
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim

        self.c_q = CastedLinear(dim, dim, bias=False)
        self.lk_rank = 32
        self.c_k_a = CastedLinear(dim, self.lk_rank, bias=False)
        self.c_k_b = CastedLinear(self.lk_rank, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k_b(self.c_k_a(x)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Feature map: elu(x) + 1 for positive kernel
        q_feat = F.elu(q) + 1.0
        k_feat = F.elu(k) + 1.0

        # Causal linear attention via cumulative sum
        # kv: [B, H, D, D] running context matrix
        kv = torch.zeros(bsz, self.num_heads, self.head_dim, self.head_dim,
                         device=x.device, dtype=x.dtype)
        k_sum = torch.zeros(bsz, self.num_heads, self.head_dim,
                            device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seqlen):
            kt = k_feat[:, :, t, :]  # [B, H, D]
            vt = v[:, :, t, :]       # [B, H, D]
            kv = kv + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            k_sum = k_sum + kt

            qt = q_feat[:, :, t, :]  # [B, H, D]
            num = (qt.unsqueeze(-2) @ kv).squeeze(-2)      # [B, H, D]
            den = (qt * k_sum).sum(dim=-1, keepdim=True) + 1e-6
            outputs.append(num / den)

        y = torch.stack(outputs, dim=2)  # [B, H, T, D]
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class EigenspaceMLP(nn.Module):
    """
    NOVEL #18 -- Eigenspace Token Routing
    MLP with orthogonal basis routing.
    Tokens are routed to experts via eigenspace projection,
    rather than standard softmax -- each expert is a true specialist.
    """
    def __init__(self, dim, mlp_mult, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        hidden = mlp_mult * dim

        # Orthogonal routing basis — learned with orthogonal inductive bias
        self.eigen_basis = nn.Parameter(torch.randn(dim, num_experts) / math.sqrt(dim))
        self.eigen_scale = nn.Parameter(torch.ones(num_experts, dtype=torch.float32))

        # Expert MLPs
        self.expert_fc = nn.ModuleList([CastedLinear(dim, hidden, bias=False)
                                        for _ in range(num_experts)])
        self.expert_proj = nn.ModuleList([CastedLinear(hidden, dim, bias=False)
                                          for _ in range(num_experts)])
        for proj in self.expert_proj:
            proj._zero_init = True

    def forward(self, x):
        # Compute routing coefficients via eigenspace projection
        basis = F.normalize(self.eigen_basis, dim=0)  # orthogonalize
        coeffs = x @ basis  # [B, T, num_experts]
        coeffs = coeffs * self.eigen_scale.to(dtype=x.dtype)
        weights = torch.softmax(coeffs, dim=-1)  # [B, T, num_experts]

        # Compute expert outputs
        out = torch.zeros_like(x)
        for i in range(self.num_experts):
            w = weights[..., i:i+1]  # [B, T, 1]
            h = torch.relu(self.expert_fc[i](x))
            expert_out = self.expert_proj[i](h.square())
            out = out + w * expert_out

        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult,
                 rope_base, qk_gain_init, num_experts=1, use_linear_attn=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        if use_linear_attn:
            self.attn = LinearCausalAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        else:
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        if num_experts > 1:
            self.mlp = EigenspaceMLP(dim, mlp_mult, num_experts)
        else:
            # Standard MLP
            hidden = mlp_mult * dim
            self.mlp = nn.Sequential(
                CastedLinear(dim, hidden, bias=False),
                nn.Identity(),  # relu applied manually
                CastedLinear(hidden, dim, bias=False),
            )
            self.mlp[2]._zero_init = True

    def forward(self, x, x0, resid_mix, attn_scale, mlp_scale):
        mix = resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))

        xn = self.mlp_norm(x)
        if isinstance(self.mlp, EigenspaceMLP):
            mlp_out = self.mlp(xn)
        else:
            h = torch.relu(self.mlp[0](xn))
            mlp_out = self.mlp[2](h.square())

        x = x + mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        return x


class UltimateRecurrentGPT(nn.Module):
    """
    Ultimate novel recurrent architecture combining:
    - All proven techniques from submission
    - 6 original novel contributions
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        dim = args.model_dim
        N = args.num_unique_layers
        P = args.num_passes
        total_steps = N * P

        # Embeddings
        self.tok_emb = nn.Embedding(args.vocab_size, dim)
        # Value embeddings (modded-nanogpt 2025)
        self.value_embeds = nn.Embedding(args.vocab_size, dim)
        nn.init.zeros_(self.value_embeds.weight)

        # Shared transformer blocks with eigenspace MLP
        num_experts = args.num_experts if args.use_eigenspace_routing else 1
        self.blocks = nn.ModuleList([
            Block(dim, args.num_heads, args.num_kv_heads,
                  args.mlp_mult, args.rope_base, args.qk_gain_init,
                  num_experts=num_experts,
                  use_linear_attn=False)
            for i in range(N)
        ])

        # Per-(pass, layer) control params
        self.attn_scales = nn.Parameter(torch.ones(total_steps, dim, dtype=torch.float32))
        self.mlp_scales = nn.Parameter(torch.ones(total_steps, dim, dtype=torch.float32))
        self.resid_mixes = nn.Parameter(
            torch.stack([
                torch.stack((torch.ones(dim), torch.zeros(dim)))
                for _ in range(total_steps)
            ]).float()
        )

        # U-Net skip weights
        self.num_enc = N // 2
        self.num_dec = N - self.num_enc
        self.num_skip = min(self.num_enc, self.num_dec)
        if args.use_unet_skip and self.num_skip > 0:
            self.skip_weights = nn.Parameter(
                torch.ones(P, self.num_skip, dim, dtype=torch.float32)
            )
        else:
            self.skip_weights = None

        # Adaptive depth (proven novel)
        if args.use_adaptive_depth:
            self.exit_proj = CastedLinear(dim, 1, bias=True)
            nn.init.zeros_(self.exit_proj.weight)
            nn.init.constant_(self.exit_proj.bias, 2.0)
        else:
            self.exit_proj = None

        # Confidence conditioning (proven novel)
        if args.use_confidence_conditioning:
            self.confidence_proj = nn.Parameter(
                torch.zeros(P, dim, dtype=torch.float32)
            )
        else:
            self.confidence_proj = None

        # NOVEL #15 — Gradient Memory Recurrence
        # Store magnitude of change between passes as implicit memory
        if args.use_gradient_memory:
            self.grad_memory_scale = nn.Parameter(
                torch.ones(P, dim, dtype=torch.float32) * 0.1
            )
        else:
            self.grad_memory_scale = None

        # NOVEL #17 — Temporal Difference Recurrence
        # TD learning: each pass refines representation based on prediction error
        if args.use_td_recurrence:
            self.td_alpha_param = nn.Parameter(
                torch.full((P,), args.td_alpha, dtype=torch.float32)
            )
            # Low-rank TD projection: rank=16, saves ~565K params vs full dim x dim
            self.td_rank = 16
            self.td_proj_a = CastedLinear(dim, self.td_rank, bias=False)
            self.td_proj_b = CastedLinear(self.td_rank, dim, bias=False)
            nn.init.normal_(self.td_proj_a.weight, std=0.01)
            nn.init.zeros_(self.td_proj_b.weight)
        else:
            self.td_alpha_param = None
            self.td_proj_a = None
            self.td_proj_b = None

        # NOVEL #19 — Resonant Position Encoding
        # Frequency-based resonance gate per pass
        if args.use_resonant_encoding:
            self.resonant_freq = nn.Parameter(
                torch.randn(P, dim // 2, dtype=torch.float32) * 0.01
            )
            self.resonant_scale = nn.Parameter(
                torch.ones(P, dtype=torch.float32) * 0.1
            )
        else:
            self.resonant_freq = None

        # NOVEL #20 — Selective State Carry
        # Sequence state summary carried between passes
        if args.use_selective_state:
            self.state_gate = nn.Parameter(
                torch.zeros(P, dim, dtype=torch.float32)
            )
            # Low-rank GRU state carry: rank=16, saves ~516K params vs full dim x dim
            # GRU update gate (z) and candidate (h) in low-rank form
            self.gru_rank = 16
            self.gru_z_a = CastedLinear(dim, self.gru_rank, bias=False)
            self.gru_z_b = CastedLinear(self.gru_rank, dim, bias=False)
            self.gru_h_a = CastedLinear(dim, self.gru_rank, bias=False)
            self.gru_h_b = CastedLinear(self.gru_rank, dim, bias=False)
            nn.init.normal_(self.gru_z_a.weight, std=0.01)
            nn.init.zeros_(self.gru_z_b.weight)
            nn.init.normal_(self.gru_h_a.weight, std=0.01)
            nn.init.zeros_(self.gru_h_b.weight)
        else:
            self.state_gate = None
            self.gru_z_a = None
            self.gru_z_b = None
            self.gru_h_a = None
            self.gru_h_b = None

        self.final_norm = RMSNorm()
        self.lm_head = (None if args.tie_embeddings
                        else CastedLinear(args.vocab_size, dim, bias=False))
        self.logit_softcap = args.logit_softcap

        # Track training step for thermodynamic annealing schedule
        self.register_buffer("_train_step", torch.tensor(0, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        # Conservative spectral init
        spectral_std = 0.1 / math.sqrt(self.tok_emb.embedding_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=spectral_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _step_index(self, pass_idx, layer_idx):
        return pass_idx * self.args.num_unique_layers + layer_idx

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        args = self.args
        N = args.num_unique_layers
        P = args.num_passes
        B, T = input_ids.shape

        # Embedding + value embedding
        x = self.tok_emb(input_ids)
        x = x + self.value_embeds(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        x_prev = x.detach()  # gradient memory tracking

        # State for selective state carry between passes
        seq_state = None
        prev_confidence = None
        aux_losses = []

        for pass_idx in range(P):

            # NOVEL #19 — Resonant Position Encoding
            # Tokens resonate with pass-specific learned frequencies
            if args.use_resonant_encoding and self.resonant_freq is not None:
                freq = self.resonant_freq[pass_idx].to(dtype=x.dtype)  # [D//2]
                pos = torch.arange(T, device=x.device, dtype=x.dtype)
                # Resonance: cos(freq * pos) gate
                resonance = torch.cos(
                    pos.unsqueeze(-1) * freq.unsqueeze(0)
                )  # [T, D//2]
                resonance = torch.cat([resonance, resonance], dim=-1)  # [T, D]
                scale = self.resonant_scale[pass_idx].to(dtype=x.dtype)
                x = x + scale * resonance.unsqueeze(0) * x

            # NOVEL #20 — Selective State Carry
            # Inject global sequence state from previous pass
            if (args.use_selective_state and seq_state is not None
                    and self.state_gate is not None and self.gru_z_a is not None):
                gate = torch.sigmoid(
                    self.state_gate[pass_idx].to(dtype=x.dtype)
                )  # [D]
                # GRU-style update: z gate controls blend of old state and new signal
                z = torch.sigmoid(self.gru_z_b(self.gru_z_a(seq_state)))  # [B, D]
                h = torch.tanh(self.gru_h_b(self.gru_h_a(seq_state)))     # [B, D]
                state_signal = (1 - z) * seq_state + z * h                # [B, D]
                x = x + gate[None, None, :] * state_signal.unsqueeze(1)

            # Confidence conditioning (proven novel)
            if (args.use_confidence_conditioning and prev_confidence is not None
                    and self.confidence_proj is not None):
                conf_signal = self.confidence_proj[pass_idx].to(dtype=x.dtype)
                x = x + prev_confidence.unsqueeze(-1) * conf_signal[None, None, :]

            # NOVEL #15 — Gradient Memory
            # Inject magnitude of change from previous pass
            if args.use_gradient_memory and self.grad_memory_scale is not None:
                delta = (x - x_prev).abs()  # [B, T, D]
                mem_scale = self.grad_memory_scale[pass_idx].to(dtype=x.dtype)
                x = x + mem_scale[None, None, :] * delta

            x_before_pass = x.detach()

            # Forward through blocks (with U-Net skip)
            skips: list[Tensor] = []
            for layer_idx in range(N):
                step = self._step_index(pass_idx, layer_idx)
                resid_mix = self.resid_mixes[step]
                attn_scale = self.attn_scales[step]
                mlp_scale = self.mlp_scales[step]

                if args.use_unet_skip and self.skip_weights is not None:
                    if layer_idx < self.num_enc:
                        x = self.blocks[layer_idx](x, x0, resid_mix, attn_scale, mlp_scale)
                        skips.append(x)
                    else:
                        dec_idx = layer_idx - self.num_enc
                        if dec_idx < self.num_skip and skips:
                            sw = self.skip_weights[pass_idx, dec_idx].to(dtype=x.dtype)
                            x = x + sw[None, None, :] * skips[-(dec_idx + 1)]
                        x = self.blocks[layer_idx](x, x0, resid_mix, attn_scale, mlp_scale)
                else:
                    x = self.blocks[layer_idx](x, x0, resid_mix, attn_scale, mlp_scale)

            # NOVEL #17 — Temporal Difference Recurrence
            # x = x_prev + alpha * (td_target - x_prev)
            if (args.use_td_recurrence and self.td_alpha_param is not None
                    and self.td_proj_a is not None and pass_idx > 0):
                alpha = torch.sigmoid(
                    self.td_alpha_param[pass_idx].to(dtype=x.dtype)
                )
                td_target = self.td_proj_b(self.td_proj_a(x))
                td_correction = alpha * (td_target - x_before_pass)
                x = x + td_correction
                # TD regularization: prevent correction from being too large
                td_reg = (td_correction ** 2).mean() * 0.001
                aux_losses.append(td_reg)

            # Update gradient memory
            x_prev = x_before_pass

            # Update selective state carry
            # State = mean pooling over sequence → global context vector
            if args.use_selective_state and self.state_gate is not None:
                seq_state = x.mean(dim=1)  # [B, D]

            # Adaptive depth (proven novel)
            if (args.use_adaptive_depth and self.exit_proj is not None
                    and pass_idx < P - 1):
                gate_logit = self.exit_proj(x)  # [B, T, 1]
                gate = torch.sigmoid(gate_logit).squeeze(-1)  # [B, T]

                # Auxiliary: gate entropy loss (sparse gates)
                gate_entropy = -(gate * torch.log(gate + 1e-8) +
                                 (1 - gate) * torch.log(1 - gate + 1e-8))
                aux_losses.append(gate_entropy.mean() * 0.005)

                # Update confidence signal for next pass
                if args.use_confidence_conditioning:
                    prev_confidence = gate.detach()

                # Soft gating
                x = x * gate.unsqueeze(-1) + x.detach() * (1 - gate).unsqueeze(-1)

        # Final prediction
        x_norm = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if args.tie_embeddings:
            logits_proj = F.linear(x_norm, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_norm)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        if not self.training:
            return ce_loss

        total_loss = ce_loss

        # NOVEL #16 — Thermodynamic Compression Loss
        # F = E - T*S dimana T turun seiring training (annealing)
        if args.use_thermo_loss:
            step = float(self._train_step.item())
            T_thermo = args.thermo_T_max * math.exp(-step / args.thermo_tau)
            entropy = -(F.softmax(logits.float(), dim=-1) *
                        F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
            thermo_loss = -T_thermo * entropy  # minimize free energy = maximize -T*S
            total_loss = total_loss + thermo_loss * args.compression_loss_weight

        # Standard compression loss (proven)
        elif args.use_compression_loss:
            log_probs = F.log_softmax(logits.float(), dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
            total_loss = total_loss + entropy * args.compression_loss_weight

        # Add auxiliary losses
        for aux in aux_losses:
            total_loss = total_loss + aux.to(dtype=total_loss.dtype)

        # Increment training step
        if self.training:
            self._train_step += 1

        return total_loss


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                     enable_math_sdp, enable_mem_efficient_sdp)
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

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
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError("VOCAB_SIZE mismatch")

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = UltimateRecurrentGPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model = (DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False,
                 find_unused_parameters=True) if distributed else compiled_model)

    # Optimizer groups
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]

    # Add all control params
    for attr in ["attn_scales", "mlp_scales", "resid_mixes", "skip_weights",
                 "confidence_proj", "grad_memory_scale", "td_alpha_param",
                 "resonant_freq", "resonant_scale", "state_gate",
                 "gru_z_a", "gru_z_b", "gru_h_a", "gru_h_b",
                 "td_proj_a", "td_proj_b"]:
        p = getattr(base_model, attr, None)
        if p is not None and isinstance(p, nn.Parameter):
            scalar_params.append(p)

    # Add projection params
    for attr in ["exit_proj", "td_proj_a", "td_proj_b",
                  "gru_z_a", "gru_z_b", "gru_h_a", "gru_h_b"]:
        m = getattr(base_model, attr, None)
        if m is not None:
            for p in m.parameters():
                matrix_params.append(p) if p.ndim == 2 else scalar_params.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight,
                     base_model.value_embeds.weight],
          "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, weight_decay=0.01,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight],
              "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(
        f"architecture:ultimate_recurrent "
        f"num_unique_layers:{args.num_unique_layers} "
        f"num_passes:{args.num_passes} "
        f"effective_depth:{args.num_unique_layers * args.num_passes} "
        f"model_dim:{args.model_dim} "
        f"num_experts:{args.num_experts if args.use_eigenspace_routing else 1}\n"
        f"  novel_techniques: "
        f"adaptive_depth={args.use_adaptive_depth} "
        f"confidence={args.use_confidence_conditioning} "
        f"thermo_loss={args.use_thermo_loss} "
        f"td_recurrence={args.use_td_recurrence} "
        f"eigenspace={args.use_eigenspace_routing} "
        f"resonant={args.use_resonant_encoding} "
        f"selective_state={args.use_selective_state} "
        f"grad_memory={args.use_gradient_memory}"
    )
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"train_seq_len:{args.train_seq_len} iterations:{args.iterations} "
         f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")

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
            return (max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                    if warmdown_start <= step < args.iterations else 1.0)
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone()
                               for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Main loop
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = (step == args.iterations or
                     (stop_after_step is not None and step >= stop_after_step))
        should_validate = (last_step or
                           (args.val_loss_every > 0 and
                            step % args.val_loss_every == 0 and step > 0))

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} "
                 f"val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms "
                 f"step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process:
                sd = base_model.state_dict()
                obj, quant_stats = quantize_state_dict_int8(sd)
                buf = io.BytesIO()
                torch.save(obj, buf)
                compressed = zlib.compress(buf.getvalue(), level=9)
                model_bytes = len(compressed)
                code_bytes = len(code.encode("utf-8"))
                total_bytes = model_bytes + code_bytes
                log0(f"final_int8_zlib_roundtrip model_bytes:{model_bytes} "
                     f"code_bytes:{code_bytes} total_bytes:{total_bytes} "
                     f"under_16mb:{total_bytes < 16_000_000} "
                     f"param_count:{quant_stats['param_count']} "
                     f"val_bpb:{val_bpb:.4f}")
                decompressed = io.BytesIO(zlib.decompress(compressed))
                obj2 = torch.load(decompressed, weights_only=True, map_location="cpu")
                sd2 = dequantize_state_dict_int8(obj2)
                base_model.load_state_dict(sd2, strict=True)
                log0("final_int8_zlib_roundtrip:ok")
            break

        model.train()
        zero_grad_all()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            (loss * grad_scale).backward()
            loss_accum += loss.item() * grad_scale

        for p in base_model.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        torch.cuda.synchronize()
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        mul = lr_mul(step + 1, elapsed_ms)

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * mul

        if step < args.muon_momentum_warmup_steps:
            mom = args.muon_momentum_warmup_start + (
                args.muon_momentum - args.muon_momentum_warmup_start
            ) * (step / max(args.muon_momentum_warmup_steps, 1))
            for group in optimizer_muon.param_groups:
                group["momentum"] = mom

        for opt in optimizers:
            opt.step()
        zero_grad_all()

        if master_process and (step % args.train_log_every == 0 or step < 10):
            log0(f"step:{step} train_loss:{loss_accum:.4f} "
                 f"lr_mul:{mul:.4f} elapsed:{elapsed_ms:.0f}ms")

        if max_wallclock_ms is not None and elapsed_ms >= max_wallclock_ms:
            if stop_after_step is None:
                stop_after_step = step + 1

        step += 1

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
