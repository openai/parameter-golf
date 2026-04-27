"""Mamba-3 Hybrid for Parameter Golf.

Sequential hybrid: Mamba-3 SSD blocks + a small number of standalone attention
layers (~2 attention + 5 SSM is our best-measured config at SP8192 7L). Env-var
driven; see `Hyperparameters` for all knobs.

Feature surface in this file is the banked submission only. Many ideas were
tested and *removed* because they underperformed or required structural
changes unsuitable for an SSM at 25M / 10min / 16MB. Full list with verdicts
and the three structural findings (SSM LZMA penalty, Muon-SSM discordance,
SP4096 → SP8192 non-transfer) are documented in
`docs/ssm_structural_findings_writeup.md`. Git history holds the removed code
on commits prior to the cleanup.
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
import lzma
import zlib
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
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    # Defaults auto-select based on VOCAB_SIZE (e.g. 4096 → fineweb10B_sp4096).
    _vs = os.environ.get("VOCAB_SIZE", "1024")
    data_path = os.environ.get("DATA_PATH", f"./data/datasets/fineweb10B_sp{_vs}")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", f"./data/tokenizers/fineweb_{_vs}_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmdown_shape = os.environ.get("WARMDOWN_SHAPE", "cosine")  # "linear" or "cosine"
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 1_048_576))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    sweep_mode = bool(int(os.environ.get("SWEEP_MODE", "0")))  # skip post-training (quant, serialize, TTT)

    # Evaluation.
    eval_stride = int(os.environ.get("EVAL_STRIDE", 16))  # sliding window stride (0 = disabled)
    eval_overlap = int(os.environ.get("EVAL_OVERLAP", 0))  # stateful-overlap eval (0 = disabled, e.g. 1024)
    eval_state_reset = int(os.environ.get("EVAL_STATE_RESET", 0))  # reset SSM state every N windows during eval (0 = never)
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))  # batch size for sliding eval
    # Test-Time Training (TTT): chunk-based score-first adaptation on val data.
    # Score each chunk (32×4096) under no_grad, then SGD adapt on the same chunk.
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.010))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_optimizer = os.environ.get("TTT_OPTIMIZER", "sgd")  # "sgd" or "adamw"
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 1))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))

    # Quantization.
    quant_bits = int(os.environ.get("QUANT_BITS", 6))
    quant_bits_embed = int(os.environ.get("QUANT_BITS_EMBED", 0))  # 0 = same as quant_bits
    # Mixed-precision SSM: dd_A and dd_dt rows of mamba3.in_proj.weight at higher bits.
    # A/dt errors compound through the recurrence; only ~32 rows per SSM block, cheap to protect.
    quant_bits_ssm_dynamics = int(os.environ.get("QUANT_BITS_SSM_DYNAMICS", 8))
    gptq_lite = bool(int(os.environ.get("GPTQ_LITE", "1")))  # search optimal clip percentile per tensor
    use_lzma = bool(int(os.environ.get("USE_LZMA", "1")))
    eval_temp = float(os.environ.get("EVAL_TEMP", "1.0"))
    use_gptq = bool(int(os.environ.get("USE_GPTQ", "0")))
    gptq_num_seqs = int(os.environ.get("GPTQ_NUM_SEQS", "32"))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", "0.0"))
    target_mb = float(os.environ.get("TARGET_MB", "15.25"))  # target compressed size in MiB
    muon_eq_r = bool(int(os.environ.get("MUON_EQ_R", "0")))  # row-normalize before Newton-Schulz

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 8))

    # OrthoInit + SWA.
    use_ortho_init = bool(int(os.environ.get("USE_ORTHO_INIT", "1")))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.4))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    mlp_mult = float(os.environ.get("MLP_MULT", 3))
    mamba3_d_state = int(os.environ.get("MAMBA3_D_STATE", 64))
    mamba3_expand = float(os.environ.get("MAMBA3_EXPAND", 2))
    mamba3_headdim = int(os.environ.get("MAMBA3_HEADDIM", 64))
    mamba3_chunk_size = int(os.environ.get("MAMBA3_CHUNK_SIZE", 64))
    mamba3_ngroups = int(os.environ.get("MAMBA3_NGROUPS", 1))
    mamba3_rope_fraction = float(os.environ.get("MAMBA3_ROPE_FRACTION", 0.5))
    # Attention layers (evenly spaced among SSD layers).
    num_attn_layers = int(os.environ.get("NUM_ATTN_LAYERS", 1))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.02))
    embed_wd = float(os.environ.get("EMBED_WD", 0.0))  # embedding weight decay (SOTA: 0.085)
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

# MUON OPTIMIZER

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7, eq_r: bool = False) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if eq_r:
        row_norms = X.norm(dim=-1, keepdim=True).clamp(min=eps)
        X = X / row_norms
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0, eq_r: bool = False):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay, eq_r=eq_r),
        )

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
            weight_decay = group.get("weight_decay", 0.0)
            eq_r = group.get("eq_r", False)

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
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps, eq_r=eq_r)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                if weight_decay > 0:
                    p.data.mul_(1.0 - lr * weight_decay)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# TOKENIZER + EVALUATION

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
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters, model: nn.Module, rank: int, world_size: int,
    device: torch.device, grad_accum_steps: int, val_tokens: Tensor,
    base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
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


def eval_val_sliding(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Sliding window evaluation: score each token with maximal left-context."""
    seq_len = args.train_seq_len
    stride = args.eval_stride
    batch_size = args.eval_batch_seqs
    total_tokens = val_tokens.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens - seq_len + 1, stride)]
    if window_starts[-1] + seq_len < total_tokens:
        window_starts.append(total_tokens - seq_len)

    my_starts = window_starts[rank::world_size]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(my_starts), batch_size):
            batch_ws = my_starts[batch_start:batch_start + batch_size]
            bsz = len(batch_ws)

            x_list, y_list = [], []
            for ws in batch_ws:
                chunk = val_tokens[ws:ws + seq_len + 1].to(dtype=torch.int64)
                x_list.append(chunk[:-1])
                y_list.append(chunk[1:])
            x_batch = torch.stack(x_list).to(device=device, non_blocking=True)
            y_batch = torch.stack(y_list).to(device=device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(x_batch)
            scaled_logits = logits.float()
            if args.eval_temp != 1.0:
                scaled_logits = scaled_logits / args.eval_temp
            nll = F.cross_entropy(
                scaled_logits.reshape(-1, scaled_logits.size(-1)),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = min(seq_len, total_tokens - ws)
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)

                prev_ids = x_batch[i, s:wlen]
                tgt_ids = y_batch[i, s:wlen]
                tbytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                tbytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                byte_count += tbytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return float(val_loss), float(bits_per_token * tokens_per_byte)


def eval_val_stateful_overlap(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    initial_states: list | None = None,
) -> tuple[float, float]:
    """Stateful eval with overlapping windows: SSM state carries forward,
    attention gets `overlap` tokens of prior context per window."""
    seq_len = args.train_seq_len
    overlap = args.eval_overlap
    score_len = seq_len - overlap
    total_tokens = val_tokens.numel() - 1

    segment_tokens = (total_tokens // world_size // score_len) * score_len
    seg_start = rank * segment_tokens
    num_windows = segment_tokens // score_len

    log0 = (lambda msg: print(msg)) if rank == 0 else (lambda msg: None)
    state_reset = args.eval_state_reset
    log0(f"stateful_overlap_eval: {num_windows} windows, overlap={overlap}, "
         f"score_len={score_len}, segment={segment_tokens} tokens, state_reset={state_reset}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    layer_states = initial_states

    with torch.inference_mode():
        # Pre-warm SSM state for non-first segments so only rank 0 pays cold-start
        if layer_states is None and seg_start > 0:
            warmup_len = min(seq_len, seg_start)
            warmup_chunk = val_tokens[seg_start - warmup_len : seg_start + 1].to(dtype=torch.int64)
            wx = warmup_chunk[:-1].unsqueeze(0).to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                _, layer_states = base_model.forward_logits_stateful(wx, None)
            layer_states = [tuple(s.detach() for s in st) for st in layer_states]
        log0(f"stateful_overlap_eval: pre-warmed={seg_start > 0} warmup_len={min(seq_len, seg_start) if seg_start > 0 else 0} initial_states={'provided' if initial_states is not None else 'None'}")

        for wi in range(num_windows):
            if state_reset > 0 and wi % state_reset == 0:
                layer_states = None

            score_start = seg_start + wi * score_len
            tok_start = max(score_start - overlap, seg_start)
            tok_end = score_start + score_len
            actual_overlap = score_start - tok_start

            chunk = val_tokens[tok_start:tok_end + 1].to(dtype=torch.int64)
            x = chunk[:-1].unsqueeze(0).to(device=device, non_blocking=True)
            y = chunk[1:].unsqueeze(0).to(device=device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits, layer_states = base_model.forward_logits_stateful(x, layer_states)

            layer_states = [tuple(s.detach() for s in st) for st in layer_states]

            scored_logits = logits[:, actual_overlap:, :].float()
            scored_y = y[:, actual_overlap:]

            if args.eval_temp != 1.0:
                scored_logits = scored_logits / args.eval_temp

            nll = F.cross_entropy(
                scored_logits.reshape(-1, scored_logits.size(-1)),
                scored_y.reshape(-1),
                reduction="none",
            )

            loss_sum += nll.to(torch.float64).sum()
            token_count += score_len

            prev_ids = x.squeeze(0)[actual_overlap:]
            tgt_ids = y.squeeze(0)[actual_overlap:]
            tbytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tbytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            byte_count += tbytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return float(val_loss), float(bits_per_token * tokens_per_byte)

# QUANTIZATION

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "m3_scale,mlp_scale,attn_scale,resid_mix,skip_weight,skip_weights,dt_bias,B_bias,C_bias,.D,A_log,q_gain",
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


def compute_ssm_dynamics_row_mask(name: str, rows: int, cfg: dict) -> Tensor | None:
    """For mamba3.in_proj.weight (dense) OR mamba3.in_proj.1.weight (factored up-projection),
    return a boolean mask of shape (rows,) marking dd_A and dd_dt rows (the recurrence-sensitive
    subset). Returns None for other tensors.
    in_proj output layout: [z | xv | B | C | dd_dt | dd_A | trap | angles]. Factored up-proj
    keeps the same row layout since the output dim matches."""
    is_dense_in_proj = "mamba3.in_proj.weight" in name
    is_factored_up = "mamba3.in_proj.1.weight" in name
    if not (is_dense_in_proj or is_factored_up):
        return None
    d_inner = int(cfg["model_dim"] * cfg["mamba3_expand"])
    d_state = int(cfg["mamba3_d_state"])
    ngroups = int(cfg["mamba3_ngroups"])
    mimo_rank = 1  # SISO (is_mimo=False in Mamba3Layer)
    nheads = d_inner // int(cfg["mamba3_headdim"])
    off = 2 * d_inner + 2 * d_state * ngroups * mimo_rank
    dt_start, dt_end = off, off + nheads
    A_start, A_end = dt_end, dt_end + nheads
    if A_end > rows:
        return None  # shape mismatch — bail silently
    mask = torch.zeros(rows, dtype=torch.bool)
    mask[dt_start:A_end] = True  # dt + A span
    return mask

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def _quantize_with_clip(t32: Tensor, clip_abs: Tensor | float, qmax: Tensor | int) -> tuple[Tensor, Tensor, Tensor]:
    # qmax may be a per-row tensor (mixed precision). When per-row, we quantize with
    # per-row step size but still store into an int8 tensor — the higher-bit rows use
    # the wider range of int8, the lower-bit rows stay within their narrower clip.
    if t32.ndim == 2 and isinstance(clip_abs, Tensor):
        qmax_col = qmax[:, None] if isinstance(qmax, Tensor) else qmax
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / (qmax.float() if isinstance(qmax, Tensor) else float(qmax)))
        # Per-row scale floor must match the per-row bit width. 1/qmax = smallest scale
        # such that the minimum nonzero quantized magnitude (±1 * scale) ≈ 1/qmax range.
        # Using a uniform 1/127 floor (INT8) was a bug: it let INT6 rows use scales ~4× too
        # small, inflating quantized magnitudes and defeating selective ±1 pruning.
        if isinstance(qmax, Tensor):
            scale = torch.maximum(scale, 1.0 / qmax.float())
        else:
            scale = scale.clamp_min(1.0 / float(qmax))
        q_unclamped = torch.round(clipped / scale[:, None])
        if isinstance(qmax_col, Tensor):
            qmax_f = qmax_col.float()
            q = torch.maximum(torch.minimum(q_unclamped, qmax_f), -qmax_f).to(torch.int8)
        else:
            q = torch.clamp(q_unclamped, -qmax, qmax).to(torch.int8)
        recon = q.float() * scale[:, None]
        return q.contiguous(), scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(), recon
    clip_abs_f = float(clip_abs) if isinstance(clip_abs, Tensor) else clip_abs
    qmax_i = int(qmax) if not isinstance(qmax, Tensor) else int(qmax.item())
    scale_f = clip_abs_f / qmax_i if clip_abs_f > 0 else 1.0
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs_f, clip_abs_f) / scale_f), -qmax_i, qmax_i).to(torch.int8)
    recon = q.float() * scale_f
    return q.contiguous(), torch.tensor(scale_f, dtype=torch.float32), recon

def quantize_float_tensor(t: Tensor, bits: int | Tensor = 8, search_clip: bool = False) -> tuple[Tensor, Tensor]:
    # bits may be a per-row tensor of shape (rows,) for mixed-precision 2D weights.
    if isinstance(bits, Tensor):
        qmax = ((1 << (bits.long() - 1)) - 1).to(torch.int64)
    else:
        qmax = (1 << (int(bits) - 1)) - 1
    t32 = t.float()

    if not search_clip:
        if t32.ndim == 2:
            clip_abs = (
                torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                if t32.numel()
                else torch.empty((t32.shape[0],), dtype=torch.float32)
            )
            q, scale, _ = _quantize_with_clip(t32, clip_abs, qmax)
            return q, scale
        clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
        q, scale, _ = _quantize_with_clip(t32, clip_abs, qmax)
        return q, scale

    candidates = [0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999999, 1.0]
    best_q, best_scale, best_mse = None, None, float("inf")

    for pct in candidates:
        if t32.ndim == 2:
            if pct >= 1.0:
                clip_abs = t32.abs().amax(dim=1)
            else:
                clip_abs = torch.quantile(t32.abs(), pct, dim=1)
            q, scale, recon = _quantize_with_clip(t32, clip_abs, qmax)
        else:
            if pct >= 1.0:
                clip_abs = float(t32.abs().max().item())
            else:
                clip_abs = float(torch.quantile(t32.abs().flatten(), pct).item()) if t32.numel() else 0.0
            q, scale, recon = _quantize_with_clip(t32, clip_abs, qmax)
        mse = (t32 - recon).pow(2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_q, best_scale = q, scale

    return best_q, best_scale

def generate_autoregressive_calib(
    model: nn.Module, device: torch.device, num_seqs: int = 64, seq_len: int = 2048,
    vocab_size: int = 1024, temperature: float = 0.8, batch_size: int = 8, seed: int = 42,
) -> list[Tensor]:
    """Generate sequences autoregressively from the model for GPTQ calibration.
    No external data accessed — fully self-contained and legal."""
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - batch_start)
            tokens = torch.randint(0, vocab_size, (bs, 1), device=device, generator=rng)
            for _ in range(seq_len - 1):
                logits = model.forward_logits(tokens)
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens = torch.cat([tokens, next_tok], dim=1)
            for i in range(bs):
                all_tokens.append(tokens[i:i+1])
    return all_tokens


def collect_hessians_from_tokens(
    model: nn.Module, token_seqs: list[Tensor], device: torch.device,
    gptq_embed: bool = False,
) -> dict[str, Tensor]:
    """Collect H = X^T X from pre-generated token sequences via forward hooks on CastedLinear layers."""
    hessians: dict[str, Tensor] = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device=device)
            def make_hook(pname: str):
                def hook_fn(mod: nn.Module, inp: tuple, out: Tensor) -> None:
                    x = inp[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pname].addmm_(x.T, x)
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(param_name)))

    if gptq_embed and model.tie_embeddings:
        def make_output_hook(pname: str):
            def hook_fn(mod: nn.Module, inp: tuple, out: Tensor) -> None:
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if pname not in hessians:
                    hessians[pname] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[pname].addmm_(x.T, x)
            return hook_fn
        hooks.append(model.final_norm.register_forward_hook(make_output_hook("tok_emb.weight")))

    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in token_seqs:
            x = seq[:, :-1].to(device)
            y = seq[:, 1:].to(device)
            model(x, y)
    for h in hooks:
        h.remove()
    num_batches = len(token_seqs)
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0], device=device)
        hessians[name] = H.cpu()
    return hessians


def collect_hessians_from_train_data(
    model: nn.Module, train_loader, device: torch.device,
    train_batch_tokens: int, seq_len: int, grad_accum_steps: int,
    n_batches: int = 64, gptq_embed: bool = True,
) -> dict[str, Tensor]:
    """Collect GPTQ Hessians from training data (like SOTA). Much faster than AR self-gen.
    Also collects Hessian for tied embeddings via output hook on final_norm."""
    hessians: dict[str, Tensor] = {}
    hooks = []

    def make_hook(pname: str):
        def hook_fn(mod: nn.Module, inp: tuple, out: Tensor) -> None:
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if pname not in hessians:
                hessians[pname] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[pname].addmm_(x.T, x)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            hooks.append(module.register_forward_hook(make_hook(name + ".weight")))

    # Embedding Hessian: hook the output of final_norm (input to logit projection = tok_emb.weight^T)
    if gptq_embed and model.tie_embeddings:
        def make_output_hook(pname: str):
            def hook_fn(mod: nn.Module, inp: tuple, out: Tensor) -> None:
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if pname not in hessians:
                    hessians[pname] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[pname].addmm_(x.T, x)
            return hook_fn
        hooks.append(model.final_norm.register_forward_hook(make_output_hook("tok_emb.weight")))

    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(n_batches):
            x, _ = train_loader.next_batch(train_batch_tokens, seq_len, grad_accum_steps)
            model.forward_logits(x)

    for h in hooks:
        h.remove()
    for name in hessians:
        H = hessians[name]
        H /= n_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0], device=device)
        hessians[name] = H.cpu()
    return hessians


def quantize_int6_gptq(
    weight: Tensor, hessian: Tensor | None = None, clip_range: int | Tensor = 31, block_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """Full GPTQ: Hessian-aware int6 quantization with Cholesky error compensation and column reordering.
    Falls back to percentile search if hessian is None.
    clip_range may be a per-row tensor of shape (rows,) for mixed-precision quantization."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _quantize_int6_percentile(t32, clip_range)
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp
    # Column reordering: quantize most-activated (sensitive) columns first
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    # Compute upper Cholesky of H_inv for the error propagation sweep
    try:
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        print(f"gptq:cholesky_fallback shape={tuple(weight.shape)} dead={int(dead.sum())}/{cols}", flush=True)
        return _quantize_int6_percentile(weight, clip_range=clip_range)

    # Per-row clip_range support: when a Tensor, broadcast as (rows,) for 1D ops
    # and (rows, 1) for 2D ops. Scale clamp_min is per-row when clip_range is per-row.
    cr_is_tensor = isinstance(clip_range, Tensor)
    if cr_is_tensor:
        cr_1d = clip_range.float()  # (rows,)
        cr_min = (1.0 / cr_1d).to(torch.float32)
    else:
        cr_1d = float(clip_range)
        cr_min = 1.0 / float(clip_range)

    def _gptq_sweep(s: Tensor) -> tuple[Tensor, float]:
        """Run GPTQ block-wise quantization with given per-row scale s."""
        sf = s.float()
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()
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
                if cr_is_tensor:
                    q = torch.maximum(torch.minimum(torch.round(w / sf), cr_1d), -cr_1d).to(torch.int8)
                else:
                    q = torch.clamp(torch.round(w / sf), -cr_1d, cr_1d).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        return Q, mse

    def _scale_from_row_clip(row_clip: Tensor) -> Tensor:
        # s = max(row_clip / clip_range, 1 / clip_range). Works scalar or per-row.
        raw = row_clip / cr_1d if cr_is_tensor else row_clip / cr_1d
        if cr_is_tensor:
            return torch.maximum(raw, cr_min).to(torch.float16)
        return raw.clamp_min(cr_min).to(torch.float16)

    # Percentile search: pick the row-wise clip percentile that minimizes MSE.
    best_q, best_scale, best_err = None, None, float("inf")
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
        s = _scale_from_row_clip(row_clip)
        Q, mse = _gptq_sweep(s)
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, inv_perm]
    return best_q, best_scale


def _quantize_int6_percentile(t32: Tensor, clip_range: int | Tensor = 31) -> tuple[Tensor, Tensor]:
    """Fallback percentile-search quantization (no Hessian).
    clip_range may be a per-row tensor of shape (rows,) for mixed precision."""
    cr_is_tensor = isinstance(clip_range, Tensor)
    if t32.ndim == 2:
        cr_col = clip_range[:, None].float() if cr_is_tensor else float(clip_range)
        cr_1d = clip_range.float() if cr_is_tensor else float(clip_range)
        cr_min = (1.0 / cr_1d) if cr_is_tensor else 1.0 / cr_1d
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
            raw = row_clip / cr_1d
            s = (torch.maximum(raw, cr_min) if cr_is_tensor else raw.clamp_min(cr_min)).to(torch.float16)
            if cr_is_tensor:
                q = torch.maximum(torch.minimum(torch.round(t32 / s.float()[:, None]), cr_col), -cr_col).to(torch.int8)
            else:
                q = torch.clamp(torch.round(t32 / s.float()[:, None]), -cr_col, cr_col).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    # 1D fallback — scalar clip_range only
    cr_scalar = int(clip_range) if not cr_is_tensor else int(clip_range.max().item())
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / cr_scalar if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -cr_scalar, cr_scalar).to(torch.int8)
    return q, scale


def quantize_state_dict_int8(
    state_dict: dict[str, Tensor], quant_bits: int = 8,
    quant_bits_embed: int = 0, search_clip: bool = False,
    quant_bits_ssm_dynamics: int = 0, ssm_cfg: dict | None = None,
):
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

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        bits: int | Tensor = quant_bits
        if quant_bits_embed > 0 and "tok_emb" in name:
            bits = quant_bits_embed
        # Mixed-precision in_proj: promote dd_A + dd_dt rows to higher bits.
        if quant_bits_ssm_dynamics > 0 and ssm_cfg is not None and t.ndim == 2:
            dyn_mask = compute_ssm_dynamics_row_mask(name, t.shape[0], ssm_cfg)
            if dyn_mask is not None:
                base = int(bits) if not isinstance(bits, Tensor) else int(quant_bits)
                per_row = torch.full((t.shape[0],), base, dtype=torch.int64)
                per_row[dyn_mask] = quant_bits_ssm_dynamics
                bits = per_row
        q, s = quantize_float_tensor(t, bits=bits, search_clip=search_clip)
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
    passthrough_data = obj["passthrough"]
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            deq = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            deq = (q.float() * scale).to(dtype=dtype).contiguous()
        out[name] = deq
    for name, t in passthrough_data.items():
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


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class FakeQuantizeSTE(torch.autograd.Function):
    """Simulated quantization with Straight-Through Estimator for QAT."""
    @staticmethod
    def forward(ctx, w: Tensor, bits: int) -> Tensor:
        qmax = (1 << (bits - 1)) - 1
        if w.ndim == 2:
            scale = w.detach().abs().amax(dim=1, keepdim=True) / qmax
            scale = scale.clamp_min(1.0 / qmax)
            return (torch.clamp(torch.round(w / scale), -qmax, qmax) * scale).to(w.dtype)
        scale = w.detach().abs().amax() / qmax
        scale = scale.clamp_min(1.0 / qmax)
        return (torch.clamp(torch.round(w / scale), -qmax, qmax) * scale).to(w.dtype)

    @staticmethod
    def backward(ctx, grad: Tensor) -> tuple[Tensor, None]:
        return grad, None


class CastedLinear(nn.Linear):
    _qat_bits: int = 0

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if self._qat_bits > 0 and self.weight.numel() > 65536:
            w = FakeQuantizeSTE.apply(w, self._qat_bits)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


@torch._dynamo.disable
def _mamba3_ssd_kernel(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z,
                       chunk_size, Input_States=None, return_final_states=False):
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    return mamba3_siso_combined(
        Q=Q, K=K, V=V, ADT=ADT, DT=DT, Trap=Trap,
        Q_bias=Q_bias, K_bias=K_bias, Angles=Angles, D=D, Z=Z,
        chunk_size=chunk_size, Input_States=Input_States,
        return_final_states=return_final_states,
    )


class Mamba3Layer(nn.Module):
    """Pure Mamba-3 SISO layer. Wraps upstream Mamba3 with CastedLinear projections
    so QAT fake-quant and fp32 master weights apply to the outlier-heavy projections."""
    def __init__(self, dim: int, d_state: int = 64, expand: float = 2,
                 headdim: int = 64, chunk_size: int = 64,
                 ngroups: int = 1, rope_fraction: float = 0.5):
        super().__init__()
        from mamba_ssm.modules.mamba3 import Mamba3
        self.mamba3 = Mamba3(
            d_model=dim, d_state=d_state, expand=expand,
            headdim=headdim, is_mimo=False, chunk_size=chunk_size,
            ngroups=ngroups, rope_fraction=rope_fraction,
            is_outproj_norm=False,
        )
        for attr in ("in_proj", "out_proj"):
            src = getattr(self.mamba3, attr)
            dst = CastedLinear(src.in_features, src.out_features, bias=src.bias is not None)
            dst.weight = src.weight
            if src.bias is not None:
                dst.bias = src.bias
            setattr(self.mamba3, attr, dst)

    def _pre_ssd(self, x):
        """Pre-SSD ops: in_proj, split, reshape, compute ADT/DT, norms."""
        from einops import rearrange
        m = self.mamba3
        zxBCdtAtrap = m.in_proj(x)
        z, xv, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [m.d_inner, m.d_inner,
             m.d_state * m.num_bc_heads * m.mimo_rank,
             m.d_state * m.num_bc_heads * m.mimo_rank,
             m.nheads, m.nheads, m.nheads, m.num_rope_angles],
            dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", p=m.headdim)
        xv = rearrange(xv, "b l (h p) -> b l h p", p=m.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=m.mimo_rank, g=m.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")
        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-m.A_floor)
        DT = F.softplus(dd_dt + m.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")
        angles = angles.unsqueeze(-2).expand(-1, -1, m.nheads, -1)
        B = m.B_norm(B).squeeze(2)
        C = m.C_norm(C).squeeze(2)
        return z, xv, B, C, ADT, DT, trap, angles

    def _post_ssd(self, y):
        from einops import rearrange
        m = self.mamba3
        y = rearrange(y, "b l h p -> b l (h p)")
        return m.out_proj(y)

    def forward(self, x: Tensor) -> Tensor:
        m = self.mamba3
        z, xv, B, C, ADT, DT, trap, angles = self._pre_ssd(x)
        y = _mamba3_ssd_kernel(
            Q=C, K=B, V=xv,
            ADT=ADT, DT=DT, Trap=trap,
            Q_bias=m.C_bias.squeeze(1), K_bias=m.B_bias.squeeze(1),
            Angles=angles, D=m.D,
            Z=z,
            chunk_size=m.chunk_size,
        )
        return self._post_ssd(y)

    def forward_stateful(self, x: Tensor, input_states=None):
        """Forward with SSM state carry. Returns (output, final_states)."""
        m = self.mamba3
        z, xv, B, C, ADT, DT, trap, angles = self._pre_ssd(x)
        result = _mamba3_ssd_kernel(
            Q=C, K=B, V=xv,
            ADT=ADT, DT=DT, Trap=trap,
            Q_bias=m.C_bias.squeeze(1), K_bias=m.B_bias.squeeze(1),
            Angles=angles, D=m.D,
            Z=z,
            chunk_size=m.chunk_size,
            Input_States=input_states,
            return_final_states=True,
        )
        y, last_angle, last_state, last_k, last_v, *_rest = result
        final_states = (last_angle, last_state, last_k, last_v)
        return self._post_ssd(y), final_states


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class AttentionLayer(nn.Module):
    """Causal self-attention with GQA and RoPE. QK-norm + per-head q_gain."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim

        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v * v.sigmoid()
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self, dim: int, mlp_mult: int,
        mamba3_d_state: int = 64, mamba3_expand: float = 2,
        mamba3_headdim: int = 64, mamba3_chunk_size: int = 64,
        mamba3_ngroups: int = 1, mamba3_rope_fraction: float = 0.5,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.m3_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.mamba3 = Mamba3Layer(
            dim, d_state=mamba3_d_state, expand=mamba3_expand,
            headdim=mamba3_headdim, chunk_size=mamba3_chunk_size,
            ngroups=mamba3_ngroups, rope_fraction=mamba3_rope_fraction,
        )
        self.mlp = MLP(dim, mlp_mult)
        self.m3_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        m3_out = self.mamba3(self.m3_norm(x))
        x = x + self.m3_scale.to(dtype=x.dtype)[None, None, :] * m3_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

    def forward_stateful(self, x: Tensor, x0: Tensor, ssm_states=None):
        """Forward with SSM state carry. Returns (output, final_states)."""
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        m3_out, fstate = self.mamba3.forward_stateful(self.m3_norm(x), input_states=ssm_states)
        x = x + self.m3_scale.to(dtype=x.dtype)[None, None, :] * m3_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x, fstate


class AttnBlock(nn.Module):
    """Block with standalone attention (no SSM)."""
    def __init__(
        self, dim: int, mlp_mult: int,
        num_heads: int = 8, num_kv_heads: int = 4,
        rope_base: float = 10000.0, qk_gain_init: float = 1.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = AttentionLayer(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, num_layers: int, model_dim: int,
        mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float,
        logit_softcap: float,
        use_ortho_init: bool = False,
        mamba3_d_state: int = 64, mamba3_expand: float = 2,
        mamba3_headdim: int = 64, mamba3_chunk_size: int = 64,
        mamba3_ngroups: int = 1, mamba3_rope_fraction: float = 0.5,
        num_attn_layers: int = 1, num_heads: int = 8, num_kv_heads: int = 4,
        rope_base: float = 10000.0, qk_gain_init: float = 1.0,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.use_ortho_init = use_ortho_init
        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(1, self.num_skip_weights, model_dim, dtype=torch.float32)
        )
        self.encoder_indices = list(range(self.num_encoder_layers))
        self.decoder_indices = list(range(self.num_encoder_layers, num_layers))

        # Evenly-spaced attention layer indices among SSM layers.
        attn_indices = set()
        if num_attn_layers > 0:
            for i in range(1, num_attn_layers + 1):
                attn_indices.add(round(i * num_layers / (num_attn_layers + 1)))
        self.attn_indices = sorted(attn_indices)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i in attn_indices:
                self.blocks.append(AttnBlock(
                    model_dim, mlp_mult,
                    num_heads=num_heads, num_kv_heads=num_kv_heads,
                    rope_base=rope_base, qk_gain_init=qk_gain_init,
                    layer_idx=i,
                ))
            else:
                self.blocks.append(Block(
                    model_dim, mlp_mult,
                    mamba3_d_state=mamba3_d_state, mamba3_expand=mamba3_expand,
                    mamba3_headdim=mamba3_headdim, mamba3_chunk_size=mamba3_chunk_size,
                    mamba3_ngroups=mamba3_ngroups, mamba3_rope_fraction=mamba3_rope_fraction,
                    layer_idx=i,
                ))

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif self.use_ortho_init and module.weight.ndim == 2 and min(module.weight.shape) >= 16:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj" in name and name.split(".")[-1] in ("proj", "proj_D"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def _compute_logits_and_loss(self, x: Tensor, target_ids: Tensor) -> Tensor:
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    _embed_qat_bits: int = 0

    def _embed(self, input_ids: Tensor) -> Tensor:
        if self._embed_qat_bits > 0:
            w = FakeQuantizeSTE.apply(self.tok_emb.weight.to(torch.bfloat16), self._embed_qat_bits)
            x = F.embedding(input_ids, w)
        else:
            x = self.tok_emb(input_ids)
        return F.rms_norm(x, (x.size(-1),))

    def forward(self, input_ids: Tensor, target_ids: Tensor,
                layer_states: list | None = None, stateful: bool = False):
        x = self._embed(input_ids)
        if stateful:
            x, new_states = self._run_blocks_stateful(x, x, layer_states)
            loss = self._compute_logits_and_loss(x, target_ids)
            return loss, new_states
        x = self._run_blocks(x, x)
        return self._compute_logits_and_loss(x, target_ids)

    def _run_blocks(self, x: Tensor, x0: Tensor) -> Tensor:
        enc_idx = list(range(self.num_encoder_layers))
        dec_idx = list(range(self.num_encoder_layers, len(self.blocks)))
        skips: list[Tensor] = []
        for bi in enc_idx:
            x = self.blocks[bi](x, x0)
            skips.append(x)
        for i, bi in enumerate(dec_idx):
            if i < len(skips) and i < self.skip_weights.shape[1]:
                x = x + self.skip_weights[0, i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            elif skips:
                skips.pop()
            x = self.blocks[bi](x, x0)
        return x

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        x = self._embed(input_ids)
        x = self._run_blocks(x, x)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        w = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
        logits = self.logit_softcap * torch.tanh(F.linear(x, w) / self.logit_softcap)
        return logits.reshape(bsz, seqlen, -1)

    def _run_blocks_stateful(self, x: Tensor, x0: Tensor,
                             layer_states: list | None = None):
        """Like _run_blocks but carries SSM state per Mamba-3 layer. Returns (output, new_layer_states)."""
        new_states: list = []
        si = 0
        enc_idx = list(range(self.num_encoder_layers))
        dec_idx = list(range(self.num_encoder_layers, len(self.blocks)))
        skips: list[Tensor] = []
        for bi in enc_idx:
            block = self.blocks[bi]
            if isinstance(block, Block):
                prev = layer_states[si] if layer_states is not None else None
                x, fstate = block.forward_stateful(x, x0, ssm_states=prev)
                new_states.append(fstate)
                si += 1
            else:
                x = block(x, x0)
            skips.append(x)
        for i, bi in enumerate(dec_idx):
            if i < len(skips) and i < self.skip_weights.shape[1]:
                x = x + self.skip_weights[0, i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            elif skips:
                skips.pop()
            block = self.blocks[bi]
            if isinstance(block, Block):
                prev = layer_states[si] if layer_states is not None else None
                x, fstate = block.forward_stateful(x, x0, ssm_states=prev)
                new_states.append(fstate)
                si += 1
            else:
                x = block(x, x0)
        return x, new_states

    def forward_logits_stateful(self, input_ids: Tensor, layer_states: list | None = None):
        """Forward returning (logits, new_layer_states) for stateful eval."""
        bsz, seqlen = input_ids.shape
        x = self._embed(input_ids)
        x, new_states = self._run_blocks_stateful(x, x, layer_states)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        w = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
        logits = self.logit_softcap * torch.tanh(F.linear(x, w) / self.logit_softcap)
        return logits.reshape(bsz, seqlen, -1), new_states


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
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 8 // world_size))
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
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

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

    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        use_ortho_init=args.use_ortho_init,
        mamba3_d_state=args.mamba3_d_state, mamba3_expand=args.mamba3_expand,
        mamba3_headdim=args.mamba3_headdim, mamba3_chunk_size=args.mamba3_chunk_size,
        mamba3_ngroups=args.mamba3_ngroups, mamba3_rope_fraction=args.mamba3_rope_fraction,
        num_attn_layers=args.num_attn_layers, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) \
                      if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if hasattr(base_model, 'skip_weights') and base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    optimizer_tok = torch.optim.Adam(
        tok_params, weight_decay=args.embed_wd,
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, weight_decay=args.weight_decay,
        eq_r=args.muon_eq_r,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"mode:mamba3_hybrid num_attn_layers:{args.num_attn_layers} attn_indices:{base_model.attn_indices}")
    log0(f"ssd: d_state:{args.mamba3_d_state} expand:{args.mamba3_expand} headdim:{args.mamba3_headdim} ngroups:{args.mamba3_ngroups} rope_frac:{args.mamba3_rope_fraction}")
    log0(f"attn: num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} rope_base:{args.rope_base}")
    log0(f"num_layers:{args.num_layers} mlp_mult:{args.mlp_mult}")
    if args.muon_eq_r:
        log0(f"muon_eq_r:enabled")
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

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            frac = max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        else:
            step_ms = elapsed_ms / max(step, 1)
            warmdown_ms = args.warmdown_iters * step_ms
            remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
            frac = remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
        if args.warmdown_shape == "cosine" and frac < 1.0:
            return 0.5 * (1.0 + math.cos(math.pi * (1.0 - frac)))
        return frac

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

    training_time_ms = 0.0
    stop_after_step: int | None = None
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0

    ema_state: dict[str, Tensor] | None = None
    if args.ema_enabled:
        ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
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
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
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

        # Late QAT: trigger when lr_mul drops below threshold (SOTA approach)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold:
            if any(m._qat_bits == 0 for m in base_model.modules() if isinstance(m, CastedLinear)):
                qat_bits = args.quant_bits
                embed_qat_bits = args.quant_bits_embed if args.quant_bits_embed > 0 else args.quant_bits
                log0(f"late_qat:enabled bits={qat_bits} embed_bits={embed_qat_bits} at step {step} scale={scale:.4f}")
                for m in base_model.modules():
                    if isinstance(m, CastedLinear):
                        m._qat_bits = qat_bits
                base_model._embed_qat_bits = embed_qat_bits

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

        if ema_state is not None:
            d = args.ema_decay
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)

        if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone().float() for name, t in base_model.state_dict().items()}
                swa_count = 1
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu().float()
                swa_count += 1

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            tok_per_sec = step * args.train_batch_tokens / (approx_training_time_ms / 1000.0)
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"tok/s:{tok_per_sec:.0f}"
            )

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

    if args.sweep_mode:
        log0("sweep_mode:exiting after training loop (skipping quantization/serialization)")
        if distributed:
            dist.destroy_process_group()
        return

    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m._qat_bits = 0
    base_model._embed_qat_bits = 0

    if ema_state is not None:
        log0("ema:applying EMA weights")
        current_state = base_model.state_dict()
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
        del ema_state
        base_model.load_state_dict(avg_state, strict=True)
    elif args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"swa:applying averaged {swa_count} checkpoints")
        current_state = base_model.state_dict()
        avg_state = {
            name: (tensor / swa_count).to(dtype=current_state[name].dtype)
            for name, tensor in swa_state.items()
        }
        del swa_state
        base_model.load_state_dict(avg_state, strict=True)

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    # Tear down DDP before GPTQ and eval: GPTQ hooks + TTT backward are incompatible
    # with live NCCL context (CUDA illegal memory access in hook_fn on DDP teardown).
    # TTT also doesn't parallelize across GPUs. Non-master ranks exit here.
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
        distributed = False
    torch.cuda.synchronize()  # drain async NCCL event destruction before GPTQ touches memory
    torch.cuda.empty_cache()
    if not master_process:
        return

    ssm_cfg = {
        "model_dim": args.model_dim,
        "mamba3_expand": args.mamba3_expand,
        "mamba3_d_state": args.mamba3_d_state,
        "mamba3_ngroups": args.mamba3_ngroups,
        "mamba3_headdim": args.mamba3_headdim,
    }

    if args.use_gptq:
        t_gptq = time.perf_counter()
        # AR GPTQ: torch.compile + triton.set_allocator(ContextVar.set) in _Mamba3Function
        # corrupts the CUDA driver state, making train-data GPTQ crash (even in subprocesses
        # on different GPUs). AR generation uses the compiled model (kernels cached), then
        # Hessian collection with hooks also uses cached kernels — no autotuning needed.
        # Cost: +5.5 mBPB vs train-data GPTQ (9.8 vs 4.3 gap), +220s eval time.
        base_model.eval()
        log0("gptq:generating autoregressive calibration data...")
        ar_tokens = generate_autoregressive_calib(
            base_model, device, num_seqs=args.gptq_num_seqs,
            seq_len=args.train_seq_len, vocab_size=args.vocab_size,
        )
        log0(f"gptq:generated {len(ar_tokens)} seqs in {time.perf_counter()-t_gptq:.1f}s")
        log0("gptq:collecting hessians...")
        hessians = collect_hessians_from_tokens(base_model, ar_tokens, device, gptq_embed=args.tie_embeddings)
        del ar_tokens
        log0(f"gptq:collected hessians for {len(hessians)} layers in {time.perf_counter()-t_gptq:.1f}s")
        torch.cuda.empty_cache()
        # Apply GPTQ to each quantizable layer in the state dict.
        sd = base_model.state_dict()
        gptq_sd = {}
        for name, t in sd.items():
            t_cpu = t.detach().cpu()
            H = hessians.get(name)
            if H is not None and t_cpu.is_floating_point() and t_cpu.ndim == 2 and t_cpu.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
                is_embed = "tok_emb" in name
                bits = args.quant_bits_embed if (is_embed and args.quant_bits_embed > 0) else args.quant_bits
                cr: int | Tensor = (1 << (bits - 1)) - 1
                # Per-row clip range for SSM dynamics rows: larger range = higher precision.
                if args.quant_bits_ssm_dynamics > 0 and not is_embed:
                    dyn_mask = compute_ssm_dynamics_row_mask(name, t_cpu.shape[0], ssm_cfg)
                    if dyn_mask is not None:
                        dyn_cr = (1 << (args.quant_bits_ssm_dynamics - 1)) - 1
                        cr_t = torch.full((t_cpu.shape[0],), float(cr), dtype=torch.float32)
                        cr_t[dyn_mask] = float(dyn_cr)
                        cr = cr_t
                q, s = quantize_int6_gptq(t_cpu, hessian=H, clip_range=cr)
                gptq_sd[name] = (q.float() * s.float()[:, None]).to(t_cpu.dtype)
            else:
                gptq_sd[name] = t_cpu
        base_model.load_state_dict(gptq_sd, strict=True)
        del hessians, gptq_sd
        torch.cuda.empty_cache()
        log0(f"gptq:quantization complete in {time.perf_counter()-t_gptq:.1f}s total")

    quant_obj, quant_stats = quantize_state_dict_int8(
        base_model.state_dict(), quant_bits=args.quant_bits,
        quant_bits_embed=args.quant_bits_embed,
        search_clip=args.gptq_lite,
        quant_bits_ssm_dynamics=args.quant_bits_ssm_dynamics, ssm_cfg=ssm_cfg,
    )
    # Selective ±1 pruning: zero out least-impactful ±1 quantized values to fit target size
    if args.use_lzma and args.target_mb > 0 and master_process:
        target_bytes = int(args.target_mb * 1024 * 1024)
        code_bytes_est = len(code.encode("utf-8"))
        ones_info = []
        for name, q in quant_obj["quantized"].items():
            s = quant_obj["scales"].get(name)
            if s is None or s.ndim == 0:
                continue
            ones_mask = (q.abs() == 1)
            if ones_mask.any():
                row_idx = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
                flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
                errors = s.float()[row_idx].pow(2)
                for fi, err in zip(flat_idx.tolist(), errors.tolist()):
                    ones_info.append((name, fi, err))
        if ones_info:
            ones_info.sort(key=lambda x: x[2])
            def _compress_for_prune(raw):
                return lzma.compress(raw, preset=9)
            def _try_prune(n):
                tmp_q = {k: v.clone() for k, v in quant_obj["quantized"].items()}
                for i in range(min(n, len(ones_info))):
                    tmp_q[ones_info[i][0]].view(-1)[ones_info[i][1]] = 0
                tmp_obj = {**quant_obj, "quantized": tmp_q}
                buf = io.BytesIO()
                torch.save(tmp_obj, buf)
                return len(_compress_for_prune(buf.getvalue())) + code_bytes_est, tmp_q
            no_sz, _ = _try_prune(0)
            log0(f"selective_prune: {len(ones_info)} ±1 candidates, unpruned={no_sz/(1024*1024):.2f}MB target={args.target_mb}MB")
            if no_sz <= target_bytes:
                log0("selective_prune: already fits, no pruning needed")
            else:
                full_sz, _ = _try_prune(len(ones_info))
                log0(f"selective_prune: full ±1 prune={full_sz/(1024*1024):.2f}MB")
                if full_sz > target_bytes:
                    log0("selective_prune: even full prune not enough, applying all")
                    _, quant_obj["quantized"] = _try_prune(len(ones_info))
                else:
                    # Linear interpolation instead of binary search (LZMA-9 is too slow for 22 iterations)
                    frac = (no_sz - target_bytes) / max(no_sz - full_sz, 1)
                    est = int(frac * len(ones_info) * 1.1) + 1  # 10% margin
                    est = min(est, len(ones_info))
                    log0(f"selective_prune: pruning {est}/{len(ones_info)} ±1 values ({100*est/len(ones_info):.1f}%) to fit {args.target_mb}MB")
                    _, quant_obj["quantized"] = _try_prune(est)

    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    if args.use_lzma:
        quant_blob = lzma.compress(quant_raw, preset=9)
        compress_fmt = "lzma-9"
    else:
        quant_blob = zlib.compress(quant_raw, level=9)
        compress_fmt = "zlib-9"
    quant_raw_bytes = len(quant_raw)
    quant_filename = f"final_model.int{args.quant_bits}.ptz"
    if master_process:
        with open(quant_filename, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(quant_filename)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int{args.quant_bits}+{compress_fmt}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int{args.quant_bits}+{compress_fmt}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open(quant_filename, "rb") as f:
        quant_blob_disk = f.read()
    if args.use_lzma:
        quant_decompressed = lzma.decompress(quant_blob_disk)
    else:
        quant_decompressed = zlib.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(quant_decompressed), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()

        num_frozen = min(args.ttt_freeze_blocks, len(base_model.blocks))
        for i in range(num_frozen):
            for p in base_model.blocks[i].parameters():
                p.requires_grad_(False)
        ttt_params = [p for p in base_model.parameters() if p.requires_grad]
        if args.ttt_optimizer == "adamw":
            ttt_opt = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, weight_decay=0.01)
        else:
            ttt_opt = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
        ttt_seq_len = args.train_seq_len

        total_tokens_val = val_tokens.numel() - 1
        total_seqs = total_tokens_val // ttt_seq_len
        total_chunks = (total_seqs + args.eval_batch_seqs - 1) // args.eval_batch_seqs

        ttt_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        ttt_token_count = torch.zeros((), device=device, dtype=torch.float64)
        ttt_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        ttt_step = 0

        log0(f"ttt:starting optimizer={args.ttt_optimizer} lr={args.ttt_lr} freeze_blocks={num_frozen} epochs={args.ttt_epochs} chunks={total_chunks}")

        for seq_idx in range(0, total_seqs, args.eval_batch_seqs):
            batch_end = min(seq_idx + args.eval_batch_seqs, total_seqs)
            bsz = batch_end - seq_idx
            raw_start = seq_idx * ttt_seq_len
            raw_end = batch_end * ttt_seq_len + 1
            chunk = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = chunk[:-1].reshape(bsz, ttt_seq_len)
            y = chunk[1:].reshape(bsz, ttt_seq_len)

            base_model.eval()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none",
                )
                ttt_loss_sum += nll.to(torch.float64).sum()
                n_tokens = float(y.numel())
                ttt_token_count += n_tokens
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                tbytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                tbytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                ttt_byte_count += tbytes.to(torch.float64).sum()

            base_model.train()
            for _epoch in range(args.ttt_epochs):
                ttt_opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = base_model(x, y)
                loss.backward()
                ttt_opt.step()
                ttt_step += 1

        for p in base_model.parameters():
            p.requires_grad_(True)
        base_model.eval()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(ttt_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ttt_token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(ttt_byte_count, op=dist.ReduceOp.SUM)

        q_val_loss = (ttt_loss_sum / ttt_token_count).item()
        bits_per_token = q_val_loss / math.log(2.0)
        tokens_per_byte = ttt_token_count.item() / ttt_byte_count.item()
        q_val_bpb = float(bits_per_token * tokens_per_byte)
        eval_mode = "online_ttt"
        log0(f"ttt:completed steps:{ttt_step} time:{time.perf_counter() - t_ttt:.1f}s")

    else:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()

    if not args.ttt_enabled:
        if args.eval_overlap > 0:
            q_val_loss, q_val_bpb = eval_val_stateful_overlap(
                args, base_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            eval_mode = "stateful_overlap"
        elif args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
            q_val_loss, q_val_bpb = eval_val_sliding(
                args, base_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            eval_mode = "sliding"
        else:
            q_val_loss, q_val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            eval_mode = "standard"
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_mode:{eval_mode} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

if __name__ == "__main__":
    main()
