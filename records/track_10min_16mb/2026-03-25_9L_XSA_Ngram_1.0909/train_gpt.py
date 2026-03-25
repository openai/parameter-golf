"""Parameter Golf: competitive submission for openai/parameter-golf."""
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
import lzma
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
# HYPERPARAMETERS
# Default Frontier run:
# - 11 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 3x MLP expansion
# - vocab size 1024, sequence length 2048, tied embeddings
# - 786,432 train tokens per step for 20,000 iterations with a ~10 minute cap
class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    eval_only = os.environ.get("EVAL_ONLY", "")  # path to saved model.pt, skips training
    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.04))
# MUON OPTIMIZER 
# 
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
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
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
#
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
    # Sliding window evaluation with stride=64.
    # Each window of seq_len tokens scores only the last `stride` tokens,
    # giving each scored token ~(seq_len - stride) tokens of context.
    # This is a FREE improvement over non-overlapping chunks.
    stride = 64
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1  # -1 for the target shift
    batch_size = max(1, args.val_batch_size // (world_size * seq_len))
    # Compute window positions for this rank
    num_windows = max(1, (total_tokens - seq_len) // stride + 1)
    win_start = (num_windows * rank) // world_size
    win_end = (num_windows * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        win_idx = win_start
        while win_idx < win_end:
            cur_batch = min(batch_size, win_end - win_idx)
            x_list, y_masked_list, y_orig_list, x_orig_list = [], [], [], []
            actual_batch = 0
            for bi in range(cur_batch):
                wi = win_idx + bi
                offset = wi * stride
                end = offset + seq_len + 1
                if end > val_tokens.numel():
                    break
                chunk = val_tokens[offset:end].long()
                xi = chunk[:-1]
                yi = chunk[1:]
                yi_masked = yi.clone()
                sc = 0 if wi == 0 else seq_len - stride
                yi_masked[:sc] = -100
                x_list.append(xi)
                y_masked_list.append(yi_masked)
                x_orig_list.append(xi[sc:])
                y_orig_list.append(yi[sc:])
                actual_batch += 1
            if actual_batch == 0:
                break
            x = torch.stack(x_list).to(device=device, non_blocking=True)
            y_masked = torch.stack(y_masked_list).to(device=device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y_masked).detach()
            # For first window, scored = seq_len; others = stride
            first_in_batch = (win_idx == 0)
            scored_per_window = [seq_len if (win_idx + bi == 0) else stride for bi in range(actual_batch)]
            total_scored = sum(scored_per_window)
            val_loss_sum += batch_loss.to(torch.float64) * total_scored
            val_token_count += total_scored
            # BPB: accumulate byte counts for scored tokens
            for bi in range(actual_batch):
                prev_ids = x_orig_list[bi].to(device=device)
                tgt_ids = y_orig_list[bi].to(device=device)
                tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
                val_byte_count += tb.to(torch.float64).sum()
            win_idx += actual_batch
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
NGRAM_ORDER = int(os.environ.get("NGRAM_ORDER", 7))  # max order for backoff
NGRAM_MIN_ORDER = int(os.environ.get("NGRAM_MIN_ORDER", 2))  # backoff down to this
NGRAM_ALPHA = float(os.environ.get("NGRAM_ALPHA", 0.40))  # base alpha (PR #727 uses 0.40)
NGRAM_ENTROPY_ADAPTIVE = bool(int(os.environ.get("NGRAM_ENTROPY_ADAPTIVE", "1")))
NGRAM_ENT_BASE = float(os.environ.get("NGRAM_ENT_BASE", 0.05))
NGRAM_ENT_RANGE = float(os.environ.get("NGRAM_ENT_RANGE", 0.55))
# PAQ-style online adaptive mixing (challenger innovation)
NGRAM_PAQ_ADAPTIVE = bool(int(os.environ.get("NGRAM_PAQ_ADAPTIVE", "0")))  # off by default
NGRAM_PAQ_LR = float(os.environ.get("NGRAM_PAQ_LR", 0.02))  # weight update rate
NGRAM_PAQ_CONTEXTS = int(os.environ.get("NGRAM_PAQ_CONTEXTS", 256))  # context-dependent weight tables
NGRAM_MIN_COUNT = int(os.environ.get("NGRAM_MIN_COUNT", 2))
NGRAM_BUCKETS = int(os.environ.get("NGRAM_BUCKETS", 4_194_304))
NGRAM_PRIMES = np.array([np.uint64(36313), np.uint64(27191), np.uint64(51647),
                          np.uint64(81929), np.uint64(131071), np.uint64(65537),
                          np.uint64(104729)], dtype=np.uint64)
def eval_val_ngram(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Score-first sliding eval with multi-order backoff + entropy-adaptive alpha.
    Orders 2-7 with highest-order-first backoff. Entropy-adaptive alpha trusts
    n-gram more when model is uncertain. Legal: no target-aware gating."""
    stride = 128
    seq_len = args.train_seq_len
    max_order = NGRAM_ORDER
    min_order = NGRAM_MIN_ORDER
    base_alpha = NGRAM_ALPHA
    min_count = NGRAM_MIN_COUNT
    buckets = NGRAM_BUCKETS
    mask = np.uint64(buckets - 1)
    total_tokens = val_tokens.numel() - 1
    batch_seqs = 32
    # Separate hash tables per order (critical — shared tables cause cross-order collisions)
    n_orders = max_order - min_order + 1
    ctx_tables = [np.zeros(buckets, dtype=np.uint32) for _ in range(n_orders)]
    full_tables = [np.zeros(buckets, dtype=np.uint32) for _ in range(n_orders)]
    # PAQ: initialize per-context mixing weights (logit space, 0 = alpha=0.5)
    if NGRAM_PAQ_ADAPTIVE:
        eval_val_ngram._paq_weights = np.zeros(NGRAM_PAQ_CONTEXTS, dtype=np.float64)
    all_ws = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    my_s = (len(all_ws) * rank) // world_size
    my_e = (len(all_ws) * (rank + 1)) // world_size
    windows = all_ws[my_s:my_e]
    val_np = val_tokens.numpy()
    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0
    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(windows), batch_seqs):
            batch_ws = windows[bi:bi + batch_seqs]
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
                logits = base_model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                seg_len = wlen - s
                if seg_len <= 0:
                    continue
                seg_nll = nll[i, s:wlen].to(torch.float64).cpu().numpy()
                seg_p = np.exp(-seg_nll)
                # Compute per-token alpha: entropy-adaptive or PAQ online learning
                if NGRAM_PAQ_ADAPTIVE and hasattr(eval_val_ngram, '_paq_weights'):
                    # PAQ: use learned weights indexed by context hash
                    paq_w = eval_val_ngram._paq_weights
                    x_seg = val_np[ws + s:ws + wlen].astype(np.uint64)
                    prev_seg = val_np[ws + s - 1:ws + wlen - 1].astype(np.uint64) if ws + s > 0 else np.zeros(wlen - s, dtype=np.uint64)
                    ctx_idx = ((x_seg * np.uint64(31) + prev_seg) % np.uint64(NGRAM_PAQ_CONTEXTS)).astype(np.int64)
                    # Sigmoid to convert logit weights to alpha
                    token_alpha = 1.0 / (1.0 + np.exp(-paq_w[ctx_idx]))
                elif NGRAM_ENTROPY_ADAPTIVE:
                    log_probs = F.log_softmax(logits[i, s:wlen].float(), dim=-1)
                    probs = log_probs.exp()
                    entropy = -(probs * log_probs).sum(dim=-1).cpu().numpy()  # per-token entropy
                    token_alpha = NGRAM_ENT_BASE + NGRAM_ENT_RANGE / (1.0 + np.exp(-2.0 * (entropy - 4.0)))
                else:
                    token_alpha = np.full(wlen - s, base_alpha)
                # Multi-order backoff: try highest order first, cascade down on miss
                global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
                mixed_mask = np.zeros(len(global_j), dtype=bool)
                all_cache_updates = []
                for order in range(max_order, min_order - 1, -1):
                    oi = order - min_order  # index into per-order tables
                    valid = (global_j >= (order - 1)) & ~mixed_mask
                    if not valid.any():
                        continue
                    v_idx = np.nonzero(valid)[0]
                    jv = global_j[v_idx]
                    ctx_width = order - 1
                    ctx_hash = np.zeros(len(jv), dtype=np.uint64)
                    for k in range(ctx_width):
                        tok = val_np[jv - (ctx_width - k)].astype(np.uint64)
                        ctx_hash ^= tok * NGRAM_PRIMES[k % len(NGRAM_PRIMES)]
                    ctx_key = (ctx_hash & mask).astype(np.int64)
                    tgt_np = val_np[jv].astype(np.uint64)
                    full_key = ((ctx_hash ^ (tgt_np * NGRAM_PRIMES[ctx_width % len(NGRAM_PRIMES)])) & mask).astype(np.int64)
                    ctx_counts = ctx_tables[oi][ctx_key].astype(np.float64)
                    full_counts = full_tables[oi][full_key].astype(np.float64)
                    can_mix = ctx_counts >= float(min_count)
                    if can_mix.any():
                        p_ng = np.minimum(full_counts, ctx_counts) / np.maximum(ctx_counts, 1.0)
                        p_ng = np.clip(p_ng, 0.0, 1.0)
                        mix_idx = v_idx[can_mix]
                        a = token_alpha[mix_idx]
                        mixed = (1.0 - a) * seg_p[mix_idx] + a * p_ng[can_mix]
                        seg_p[mix_idx] = mixed
                        mixed_mask[mix_idx] = True
                    all_cache_updates.append((oi, ctx_key, full_key))
                seg_nll = -np.log(np.clip(seg_p, 1e-12, 1.0))
                # PAQ: update mixing weights based on prediction quality
                if NGRAM_PAQ_ADAPTIVE and hasattr(eval_val_ngram, '_paq_weights'):
                    # For tokens that were mixed: update weight toward better component
                    if mixed_mask.any():
                        mixed_idx = np.nonzero(mixed_mask)[0]
                        p_mixed = seg_p[mixed_idx]
                        # Gradient: positive = n-gram helped, negative = n-gram hurt
                        error = 1.0 - np.clip(p_mixed, 1e-7, 1.0)
                        x_seg = val_np[ws + s:ws + wlen].astype(np.uint64)
                        prev_seg = val_np[ws + s - 1:ws + wlen - 1].astype(np.uint64) if ws + s > 0 else np.zeros(wlen - s, dtype=np.uint64)
                        ctx_idx = ((x_seg * np.uint64(31) + prev_seg) % np.uint64(NGRAM_PAQ_CONTEXTS)).astype(np.int64)
                        np.add.at(eval_val_ngram._paq_weights, ctx_idx[mixed_idx], NGRAM_PAQ_LR * error)
                # Score-first: update per-order caches AFTER scoring
                for oi, ctx_key, full_key in all_cache_updates:
                    np.add.at(ctx_tables[oi], ctx_key, 1)
                    np.add.at(full_tables[oi], full_key, 1)
                loss_sum += float(seg_nll.sum())
                token_count += float(seg_len)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += float(tb.sum().item())
    if dist.is_available() and dist.is_initialized():
        all_r = torch.tensor([loss_sum, token_count, byte_count], device=device, dtype=torch.float64)
        dist.all_reduce(all_r, op=dist.ReduceOp.SUM)
        loss_sum, token_count, byte_count = all_r[0].item(), all_r[1].item(), all_r[2].item()
    val_loss = loss_sum / max(token_count, 1.0)
    bpt = val_loss / math.log(2.0)
    tpb = token_count / max(byte_count, 1.0)
    return float(val_loss), float(bpt * tpb)
# POST-TRAINING QUANTIZATION
#
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
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
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536  # Only small control tensors stay fp16; large tensors MUST be int8/int6
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
QUANT_BITS = int(os.environ.get("QUANT_BITS", 6))  # int6: 64 unique values → highly compressible by zstd
QUANT_MAX = (1 << (QUANT_BITS - 1)) - 1  # 31 for int6
GPTQ_ENABLED = bool(int(os.environ.get("GPTQ_ENABLED", "1")))
GPTQ_BLOCK_SIZE = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))
GPTQ_DAMP = float(os.environ.get("GPTQ_DAMP", 0.01))
GPTQ_N_SAMPLES = int(os.environ.get("GPTQ_N_SAMPLES", 256))
def _find_best_row_scales(W: Tensor, clip_range: int) -> Tensor:
    """Find best per-row scale by searching over percentiles."""
    percentiles = [0.999, 0.9995, 0.9999, 0.99999, 1.0]
    best_scales = torch.zeros(W.shape[0], device=W.device)
    best_mse = torch.full((W.shape[0],), float("inf"), device=W.device)
    for p in percentiles:
        clip_abs = torch.quantile(W.abs(), p, dim=1) if W.numel() else torch.zeros(W.shape[0], device=W.device)
        scale = (clip_abs / clip_range).clamp_min(1e-8)
        q = torch.clamp(torch.round(W / scale[:, None]), -clip_range - 1, clip_range)
        mse = ((W - q * scale[:, None]) ** 2).mean(dim=1)
        improved = mse < best_mse
        best_mse[improved] = mse[improved]
        best_scales[improved] = scale[improved]
    return best_scales
def gptq_quantize_weight(W: Tensor, H: Tensor, clip_range: int = QUANT_MAX,
                         block_size: int = GPTQ_BLOCK_SIZE, percdamp: float = GPTQ_DAMP) -> tuple[Tensor, Tensor]:
    """GPTQ: Hessian-aware quantization with error redistribution."""
    W = W.float().clone()
    rows, cols = W.shape
    row_scale = _find_best_row_scales(W, clip_range)
    H = H.float().to(W.device).clone()
    # Handle dead columns
    dead = H.diag() == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    # Dampen
    damp = percdamp * H.diag().mean()
    H.diagonal().add_(damp)
    # Column reorder by ascending Hessian diagonal
    perm = torch.argsort(H.diag())
    invperm = torch.argsort(perm)
    W = W[:, perm]
    H = H[perm][:, perm]
    # Cholesky of H^{-1}
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch._C._LinAlgError:
        Hinv = torch.diag(1.0 / H.diag().clamp_min(1e-6))
    Q = torch.zeros(rows, cols, dtype=torch.int8, device=W.device)
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros_like(W_block)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            h_jj = Hinv_block[j, j].clamp_min(1e-8)
            q_col = torch.clamp(torch.round(w_col / row_scale), -clip_range - 1, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col * row_scale) / h_jj
            Err[:, j] = err
            if j + 1 < i2 - i1:
                W_block[:, j + 1 :] -= err.unsqueeze(1) * Hinv_block[j, j + 1 :].unsqueeze(0)
        if i2 < cols:
            W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    Q = Q[:, invperm]
    return Q.contiguous(), row_scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
def gptq_calibrate(model: nn.Module, train_pattern: str, device: torch.device,
                   seq_len: int, n_samples: int = GPTQ_N_SAMPLES) -> dict[str, Tensor]:
    """Collect Hessian (H = X^T X) for each Linear layer via forward hooks."""
    hessians: dict[str, Tensor] = {}
    n_seen: dict[str, int] = {}
    hooks = []
    def make_hook(name: str):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                n_seen[name] = 0
            hessians[name].addmm_(x.t(), x)
            n_seen[name] += x.shape[0]
        return hook_fn
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, CastedLinear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    stream = TokenStream(train_pattern)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        for _ in range(n_samples):
            tokens = stream.take(seq_len + 1).to(device=device, dtype=torch.int64)
            model(tokens[:-1].unsqueeze(0), tokens[1:].unsqueeze(0))
    for h in hooks:
        h.remove()
    for name in hessians:
        hessians[name] /= max(n_seen[name], 1)
    return hessians
def pack_int6_numpy(q_int8: Tensor) -> Tensor:
    """Pack int6 values stored as int8 into 6-bit packed bytes using numpy. 4 values → 3 bytes."""
    arr = q_int8.numpy().astype(np.int8)
    flat = arr.reshape(-1)
    # Convert to unsigned [0, 63]
    u = (flat.astype(np.int16) + 32).astype(np.uint8)
    # Pad to multiple of 4
    pad = (4 - len(u) % 4) % 4
    if pad:
        u = np.concatenate([u, np.zeros(pad, dtype=np.uint8)])
    u = u.reshape(-1, 4)
    # Pack: 4 x 6-bit → 3 bytes
    b0 = (u[:, 0] << 2) | (u[:, 1] >> 4)
    b1 = ((u[:, 1] & 0x0F) << 4) | (u[:, 2] >> 2)
    b2 = ((u[:, 2] & 0x03) << 6) | u[:, 3]
    packed = np.stack([b0, b1, b2], axis=1).reshape(-1)
    return torch.from_numpy(packed.astype(np.uint8))
def unpack_int6_numpy(packed: Tensor, numel: int) -> Tensor:
    """Unpack 6-bit packed bytes back to int8 values using numpy."""
    arr = packed.numpy().astype(np.uint8)
    arr = arr.reshape(-1, 3)
    b0, b1, b2 = arr[:, 0], arr[:, 1], arr[:, 2]
    u0 = (b0 >> 2) & 0x3F
    u1 = ((b0 & 0x03) << 4) | (b1 >> 4)
    u2 = ((b1 & 0x0F) << 2) | (b2 >> 6)
    u3 = b2 & 0x3F
    u = np.stack([u0, u1, u2, u3], axis=1).reshape(-1)[:numel]
    result = (u.astype(np.int16) - 32).astype(np.int8)
    return torch.from_numpy(result)
def quantize_float_tensor(t: Tensor, hessian: Tensor | None = None) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        qmax = float(QUANT_MAX)
        if hessian is not None and GPTQ_ENABLED:
            q, scale = gptq_quantize_weight(t32, hessian, clip_range=QUANT_MAX)
        else:
            # Fallback: naive per-row quantization
            clip_abs = (
                torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                if t32.numel()
                else torch.empty((t32.shape[0],), dtype=torch.float32)
            )
            clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
            scale = (clip_abs / qmax).clamp_min(1.0 / qmax)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -qmax - 1, qmax).to(torch.int8)
            scale = scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE)
        # Store as int8 (LZMA compresses the restricted [-32,31] range well)
        # Pruning happens globally AFTER all tensors are quantized (see quantize_state_dict_int8)
        return q.contiguous(), scale.contiguous()
    # Vectors / scalars use int8
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale
def quantize_state_dict_int8(state_dict: dict[str, Tensor], hessians: dict[str, Tensor] | None = None):
    # Quantization with optional GPTQ (Hessian-aware error redistribution).
    # - per-row int6/int8 for 2D float tensors (GPTQ if hessians provided)
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
        # Hessian keys use module names (e.g. "blocks.0.attn.c_q"),
        # state dict keys have ".weight" suffix
        h_key = name.removesuffix(".weight").removesuffix(".bias") if hessians else None
        h = hessians.get(h_key) if h_key and hessians else None
        q, s = quantize_float_tensor(t, hessian=h)
        if t.ndim == 2:
            method = "GPTQ" if h is not None else "naive"
            print(f"  quant {name}: {list(t.shape)} {method}")
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
    # Targeted ±1 magnitude pruning: zero the least-important ±1 quantized values.
    # ±1 values have minimal reconstruction impact (scale * 1 = scale).
    # Zeroed values compress to nearly nothing under LZMA.
    prune_pct = float(os.environ.get("PRUNE_PCT", "25.0"))  # percentage of ALL 2D params to prune
    if prune_pct > 0:
        # Collect importance scores for ALL quantized values (not just ±1)
        # Importance = |q_val| * scale (actual fp32 magnitude of each weight)
        all_importances = []
        for name, q in quantized.items():
            s = scales[name]
            if s.ndim > 0:
                imp = q.abs().float() * s.float().unsqueeze(1).expand_as(q)
            else:
                imp = q.abs().float() * float(s.item())
            all_importances.append(imp.view(-1))
        all_imp = torch.cat(all_importances)
        nonzero_mask = all_imp > 0
        n_nonzero = nonzero_mask.sum().item()
        n_prune = int(n_nonzero * prune_pct / 100.0)
        if n_prune > 0:
            threshold = torch.kthvalue(all_imp[nonzero_mask], min(n_prune, n_nonzero)).values.item()
            total_pruned = 0
            for name, q in quantized.items():
                s = scales[name]
                if s.ndim > 0:
                    imp = q.abs().float() * s.float().unsqueeze(1).expand_as(q)
                else:
                    imp = q.abs().float() * float(s.item())
                prune_mask = (imp > 0) & (imp <= threshold)
                q[prune_mask] = 0
                total_pruned += prune_mask.sum().item()
            print(f"  pruning: zeroed {total_pruned}/{n_nonzero} weights ({100*total_pruned/max(n_nonzero,1):.1f}%)")
    return obj, stats
def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        if meta.get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
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
# DATA LOADING 
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
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
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
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
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
# TRANSFORMER MODULES
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
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
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        # Partial RoPE: only apply to first 25% of head dims (pairs)
        self.rope_dims = self.head_dim // 4  # number of pairs to rotate
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.rope_dims * 2, base=rope_base)
    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # Partial RoPE: only rotate first rope_dims*2 dimensions
        rd2 = self.rope_dims * 2
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q_rope = apply_rotary_emb(q[..., :rd2], cos, sin)
        k_rope = apply_rotary_emb(k[..., :rd2], cos, sin)
        q = torch.cat([q_rope, q[..., rd2:]], dim=-1)
        k = torch.cat([k_rope, k[..., rd2:]], dim=-1)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        # XSA: remove self-value component via orthogonal projection
        # Expand v to match num_heads for GQA before normalization
        v_for_xsa = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1) if self.num_kv_heads != self.num_heads else v
        v_normed = F.normalize(v_for_xsa, dim=-1)
        y = y - (y * v_normed).sum(dim=-1, keepdim=True) * v_normed
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)
class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())
class BigramHash(nn.Module):
    def __init__(self, num_buckets: int = 2048, dim: int = 128, model_dim: int = 512):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, dim)
        self.proj = CastedLinear(dim, model_dim, bias=False)
    def forward(self, input_ids: Tensor) -> Tensor:
        prev_ids = F.pad(input_ids[:, :-1], (1, 0))
        bigram_hash = (input_ids * 2654435761 + prev_ids * 40503) % self.num_buckets
        return self.proj(self.embed(bigram_hash))
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)
    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x) * self.ln_scale)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * self.ln_scale)
        return x
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram_hash = BigramHash(4096, 128, model_dim)
        self.smear_gate = nn.Parameter(torch.zeros(model_dim))
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.orthogonal_(module.weight)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        # BigramHash: add bigram features before SmearGate
        x = x + self.bigram_hash(input_ids)
        # SmearGate: blend current token with previous token
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        gate = torch.sigmoid(self.smear_gate)
        x = gate * x + (1.0 - gate) * x_prev
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
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
    @torch.no_grad()
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits without computing loss. Used for eval-time n-gram mixing."""
        x = self.tok_emb(input_ids)
        x = x + self.bigram_hash(input_ids)
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        gate = torch.sigmoid(self.smear_gate)
        x = gate * x + (1.0 - gate) * x_prev
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
# TRAINING
def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
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
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
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
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    # Add BigramHash and SmearGate params to scalar (Adam) optimizer
    scalar_params.append(base_model.smear_gate)
    for p in base_model.bigram_hash.parameters():
        scalar_params.append(p)
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
        weight_decay=args.weight_decay,
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
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
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
    training_time_ms = 0.0
    # EVAL_ONLY: skip training, load saved model, jump to serialization+eval
    if args.eval_only:
        log0(f"EVAL_ONLY: loading model from {args.eval_only}")
        sd = torch.load(args.eval_only, map_location="cpu")
        base_model.load_state_dict(sd)
        restore_low_dim_params_to_fp32(base_model)
        step = 0
    if not args.eval_only:
        stop_after_step: int | None = None
        # EMA: exponential moving average (GPU-only, no CPU transfers!)
        ema_decay = 0.997
        ema_state: dict[str, Tensor] | None = None
        ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "0")))  # disabled by default until proven helpful
        ema_start_pct = 0.1
        # SWA: accumulate weight averages during warmdown
        swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "0")))  # disabled until proven helpful
        swa_start_pct = 0.4
        swa_every = 50
        swa_state: dict[str, Tensor] | None = None
        swa_count = 0
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
            # EMA: update every step (GPU-only)
            ema_active = ema_enabled and max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms * ema_start_pct
            if ema_active:
                sd = base_model.state_dict()
                if ema_state is None:
                    ema_state = {k: v.clone() for k, v in sd.items()}
                else:
                    for k in ema_state:
                        ema_state[k].mul_(ema_decay).add_(sd[k], alpha=1.0 - ema_decay)
            # SWA: accumulate model weights on GPU (wallclock-based start)
            swa_active = swa_enabled and max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms * swa_start_pct
            if swa_active and step % swa_every == 0:
                sd = base_model.state_dict()
                if swa_state is None:
                    swa_state = {k: v.clone() for k, v in sd.items()}
                else:
                    for k in swa_state:
                        swa_state[k].add_(sd[k])
                swa_count += 1
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
        # Apply EMA + SWA blend (frontier uses 50/50 blend)
        if ema_state is not None and swa_state is not None and swa_count > 0:
            log0(f"EMA+SWA blend: EMA(decay={ema_decay}) + SWA({swa_count} checkpoints)")
            for k in swa_state:
                swa_state[k].div_(swa_count)
            # 50/50 blend of EMA and SWA
            for k in ema_state:
                ema_state[k].mul_(0.5).add_(swa_state[k], alpha=0.5)
            base_model.load_state_dict(ema_state, strict=True)
            del swa_state
            del ema_state
        elif ema_state is not None:
            log0(f"EMA: applying EMA weights (decay={ema_decay})")
            base_model.load_state_dict(ema_state, strict=True)
            del ema_state
        elif swa_state is not None and swa_count > 0:
            log0(f"SWA: applying averaged weights from {swa_count} checkpoints")
            for k in swa_state:
                swa_state[k].div_(swa_count)
            base_model.load_state_dict(swa_state, strict=True)
            del swa_state
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
    # GPTQ calibration: collect Hessians for Hessian-aware quantization
    hessians = None
    if GPTQ_ENABLED and master_process:
        torch.cuda.synchronize()
        t_gptq = time.perf_counter()
        log0(f"GPTQ calibration: {GPTQ_N_SAMPLES} samples...")
        hessians = gptq_calibrate(base_model, args.train_files, device, args.train_seq_len, GPTQ_N_SAMPLES)
        log0(f"GPTQ calibration done: {len(hessians)} layers, {1000*(time.perf_counter()-t_gptq):.0f}ms")
    # Only master rank does quantization + serialization (CPU-bound, no benefit from parallelism).
    # Non-master ranks skip to the barrier and load from the file master writes.
    if master_process:
        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict(), hessians=hessians)
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf, pickle_protocol=5)
        quant_raw = quant_buf.getvalue()
        import zstandard as zstd_mod
        quant_blob = zstd_mod.ZstdCompressor(level=22).compress(quant_raw)
        quant_raw_bytes = len(quant_raw)
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int6+zstd22: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int6+zstd22: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    import zstandard as zstd_mod
    quant_state = torch.load(io.BytesIO(zstd_mod.ZstdDecompressor().decompress(quant_blob_disk)), map_location="cpu")
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
    # N-gram enhanced eval (disabled by default — needs debugging)
    ngram_enabled = bool(int(os.environ.get("NGRAM_ENABLED", "0")))
    if ngram_enabled:
        torch.cuda.synchronize()
        t_ngram = time.perf_counter()
        log0("Starting n-gram enhanced eval...")
        ng_val_loss, ng_val_bpb = eval_val_ngram(
            args, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        log0(
            f"final_ngram val_loss:{ng_val_loss:.4f} val_bpb:{ng_val_bpb:.4f} "
            f"ngram_gain:{q_val_bpb - ng_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_ngram):.0f}ms"
        )
        log0(f"final_ngram_exact val_loss:{ng_val_loss:.8f} val_bpb:{ng_val_bpb:.8f}")
    else:
        ng_val_bpb = q_val_bpb
        ng_val_loss = q_val_loss
    final_bpb = min(q_val_bpb, ng_val_bpb)
    final_loss = ng_val_loss if ng_val_bpb < q_val_bpb else q_val_loss
    # Write machine-readable results JSON (robust against log streaming failures)
    if master_process:
        import json as _json
        results_obj = {
            "val_bpb": round(final_bpb, 8),
            "val_loss": round(final_loss, 8),
            "val_bpb_no_ngram": round(q_val_bpb, 8),
            "val_bpb_ngram": round(ng_val_bpb, 8),
            "bytes_model": quant_file_bytes,
            "bytes_code": code_bytes,
            "bytes_total": quant_file_bytes + code_bytes,
            "peak_memory_mib": torch.cuda.max_memory_allocated() // 1024 // 1024,
            "model_params": n_params,
            "training_steps": step,
            "training_time_ms": training_time_ms,
            "step_avg_ms": training_time_ms / max(step, 1),
            "seed": args.seed,
            "config": {
                "num_layers": args.num_layers, "model_dim": args.model_dim,
                "mlp_mult": args.mlp_mult, "num_heads": args.num_heads,
                "quant_bits": QUANT_BITS, "gptq_enabled": GPTQ_ENABLED,
            },
        }
        with open("results.json", "w") as rf:
            _json.dump(results_obj, rf, indent=2)
        log0(f"results_json: {_json.dumps(results_obj)}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
