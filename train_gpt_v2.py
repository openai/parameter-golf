"""
Hybrid Table + Neural Architecture for Parameter Golf v2.0

Core idea: Precomputed bigram log-probability table (from training data counts)
provides a strong baseline. A smaller neural network learns only the RESIDUAL
correction on top of table predictions. Tables are 19x more BPB-per-byte
efficient than neural weights.

Artifact layout:
  - bigram_table: (1024, 1024) float16 — precomputed log P(next|prev)
  - neural weights: INT6 quantized state dict
  - metadata: vocab_size, model config
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
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# HYPERPARAMETERS
# -----------------------------------------------------------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "/workspace/data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "/workspace/data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape — smaller than v1.5 to leave room for bigram table
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 10))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Bigram table — built from training data in Phase 1
    bigram_smoothing = float(os.environ.get("BIGRAM_SMOOTHING", 1.0))  # Laplace smoothing
    bigram_scale_mode = os.environ.get("BIGRAM_SCALE", "fixed")  # "fixed" | "learnable" | "0"

    # v1.5 features
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 0))  # 0=off, N=apply XSA to last N layers
    smear_gate = bool(int(os.environ.get("SMEAR_GATE", "0")))
    bigram_hash_vocab = int(os.environ.get("BIGRAM_HASH_VOCAB", 0))  # 0=off, e.g. 2048 or 3072
    bigram_hash_dim = int(os.environ.get("BIGRAM_HASH_DIM", 128))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 0))  # 0=off, e.g. 64 for sliding window

    # GPTQ quantization
    use_gptq = bool(int(os.environ.get("USE_GPTQ", "1")))
    gptq_calib_seqs = int(os.environ.get("GPTQ_CALIB_SEQS", 64))
    gptq_calib_temp = float(os.environ.get("GPTQ_CALIB_TEMP", 0.8))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))

# -----------------------------------------------------------------------------
# MUON OPTIMIZER
# -----------------------------------------------------------------------------

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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
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
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# -----------------------------------------------------------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------------------------------------------------------

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
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
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

# -----------------------------------------------------------------------------
def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
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
    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits = model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
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
    model.train()
    return val_loss, bits_per_token * tokens_per_byte


# -----------------------------------------------------------------------------
# INT6 QUANTIZATION (percentile search + GPTQ)
# -----------------------------------------------------------------------------

CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    """Percentile-search INT6 quantization (fallback when no Hessian)."""
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


def quantize_int6_gptq(weight: Tensor, hessian: Tensor | None = None,
                        clip_range: int = 31, block_size: int = 128) -> tuple[Tensor, Tensor]:
    """Full GPTQ: Hessian-aware INT6 quantization with Cholesky error compensation.
    Falls back to percentile search if hessian is None or weight is not 2D."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return quantize_int6_per_row(t32, clip_range)
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    best_q = None; best_scale = None; best_err = float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
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
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
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
    best_q = best_q[:, inv_perm]
    return best_q, best_scale


def generate_autoregressive_calib(model, device, num_seqs=64, seq_len=2048,
                                   vocab_size=1024, temperature=0.8, batch_size=8, seed=42):
    """Generate sequences autoregressively from the model for GPTQ calibration.
    No external data accessed — fully self-contained."""
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                 enabled=(device.type == "cuda")):
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


def collect_hessians_from_tokens(model, token_seqs, device):
    """Collect H = X^T X from pre-generated token sequences via forward hooks on CastedLinear."""
    hessians = {}
    hooks = []
    for name, module in model.named_modules():
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
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                 enabled=(device.type == "cuda")):
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
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    return hessians


def quantize_state_dict_int6(state_dict: dict[str, Tensor],
                              hessians: dict[str, Tensor] | None = None) -> tuple[dict, dict]:
    """Quantize neural weights to INT6. Uses GPTQ when hessians provided.
    Returns (result_dict, meta_dict)."""
    result: dict[str, Tensor] = {}
    meta: dict[str, dict] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        is_control = any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)
        # Small tensors and control params: keep as float16
        if t.numel() <= 65536 or is_control or not t.is_floating_point():
            if t.is_floating_point() and t.dtype in (torch.float32, torch.bfloat16):
                result[name] = t.to(torch.float16)
            else:
                result[name] = t
            meta[name] = {"type": "passthrough", "orig_dtype": str(t.dtype).removeprefix("torch.")}
            continue
        # Large float tensors: INT6 (with GPTQ if hessian available)
        cat = _classify_param(name)
        H = hessians.get(name) if hessians and cat in ("mlp", "attn") else None
        if H is not None:
            q, s = quantize_int6_gptq(t, hessian=H)
        else:
            q, s = quantize_int6_per_row(t)
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = {"type": "int6", "orig_dtype": str(t.dtype).removeprefix("torch.")}
    return result, meta


def dequantize_state_dict_int6(result: dict, meta: dict) -> dict[str, Tensor]:
    """Dequantize INT6 back to original dtypes."""
    out: dict[str, Tensor] = {}
    for name, info in meta.items():
        dtype = getattr(torch, info["orig_dtype"])
        if info["type"] == "passthrough":
            out[name] = result[name].to(dtype=dtype)
        elif info["type"] == "int6":
            q = result[name + ".q"]
            s = result[name + ".scale"]
            if s.ndim > 0:
                out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype)
            else:
                out[name] = (q.float() * s.float().item()).to(dtype=dtype)
    return out

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# PHASE 1: BIGRAM TABLE BUILDING
# -----------------------------------------------------------------------------

def build_bigram_table(
    data_dir: str,
    vocab_size: int = 1024,
    smoothing: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Count all bigram occurrences across training shards, return log-prob table.

    Returns: (vocab_size, vocab_size) float16 tensor where entry [i, j] = log P(j | i).
    """
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "fineweb_train_*.bin")))
    if not shard_paths:
        raise FileNotFoundError(f"No training shards found in {data_dir}")

    # Accumulate counts on CPU (int64 to avoid overflow)
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.int64)
    total_bigrams = 0
    for i, shard_path in enumerate(shard_paths):
        tokens = load_data_shard(Path(shard_path))
        n = tokens.numel()
        # Vectorized counting via scatter_add
        prev_tokens = tokens[:-1].long()
        next_tokens = tokens[1:].long()
        flat_idx = prev_tokens * vocab_size + next_tokens
        counts.view(-1).scatter_add_(0, flat_idx, torch.ones(n - 1, dtype=torch.int64))
        total_bigrams += n - 1
        if (i + 1) % 10 == 0 or i + 1 == len(shard_paths):
            print(f"  bigram_table: shard {i+1}/{len(shard_paths)}, {total_bigrams:,} bigrams counted")

    # Convert to log-probabilities with Laplace smoothing
    counts_f = counts.double() + smoothing
    log_probs = counts_f.log() - counts_f.sum(dim=1, keepdim=True).log()

    table = log_probs.to(torch.float16).to(device)
    print(f"  bigram_table: done. {total_bigrams:,} total bigrams from {len(shard_paths)} shards")
    print(f"  bigram_table: shape={tuple(table.shape)}, min={table.min().item():.4f}, max={table.max().item():.4f}")
    return table

# -----------------------------------------------------------------------------
# TRANSFORMER MODULES
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


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


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
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
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.use_xsa = False  # set by HybridGPT.__init__

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """XSA: subtract self-value projection. y: [B,T,H,D], v: [B,T,Hkv,D]."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
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
        # y is [B, H, T, D] from SDPA → transpose to [B, T, H, D] for XSA
        y = y.transpose(1, 2).contiguous()
        if self.use_xsa:
            # v is still [B, Hkv, T, D] → need [B, T, Hkv, D]
            v_for_xsa = v.transpose(1, 2).contiguous()
            y = self._xsa_efficient(y, v_for_xsa)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())


class SmearGate(nn.Module):
    """Mixes each token with its predecessor via a learnable sigmoid gate."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Input-level bigram features via hash embedding. Adds context about token pairs."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
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

# -----------------------------------------------------------------------------
# HYBRID MODEL: BIGRAM TABLE + NEURAL RESIDUAL
# -----------------------------------------------------------------------------

class HybridGPT(nn.Module):
    """GPT that combines precomputed bigram log-probs with neural residual logits.

    At inference: logits = bigram_table[prev_token] + neural_network(sequence)
    The neural network learns to CORRECT the bigram prediction, not duplicate it.
    """

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
        bigram_table: Tensor | None = None,
        bigram_scale_mode: str = "fixed",
        xsa_last_n: int = 0,
        smear_gate: bool = False,
        bigram_hash_vocab: int = 0,
        bigram_hash_dim: int = 128,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size
        self.bigram_scale_mode = bigram_scale_mode
        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # SmearGate: mix adjacent tokens after embedding
        self.smear = SmearGate(model_dim) if smear_gate else None

        # BigramHash: input-level n-gram features
        self.bigram_hash = BigramHashEmbedding(bigram_hash_vocab, bigram_hash_dim, model_dim) if bigram_hash_vocab > 0 else None

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
                for _ in range(num_layers)
            ]
        )

        # XSA: apply to last N layers
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # Bigram table: registered as buffer (not a parameter, no gradients)
        if bigram_table is not None:
            self.register_buffer("bigram_table", bigram_table)
        else:
            self.register_buffer("bigram_table", torch.zeros(vocab_size, vocab_size, dtype=torch.float16))

        # Bigram scale: "fixed"=1.0 constant, "learnable"=nn.Parameter, "0"=disabled
        if bigram_scale_mode == "learnable":
            self.bigram_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        else:
            self.register_buffer("bigram_scale", torch.tensor(
                0.0 if bigram_scale_mode == "0" else 1.0, dtype=torch.float32
            ))

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _encode(self, input_ids: Tensor) -> Tensor:
        """Shared encoder path for forward and forward_logits."""
        x = self.tok_emb(input_ids)
        if self.bigram_hash is not None:
            x = x + self.bigram_hash(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def _compute_logits(self, x: Tensor, input_ids: Tensor) -> Tensor:
        """Compute combined logits from encoded representation."""
        if self.tie_embeddings:
            neural_logits = F.linear(x, self.tok_emb.weight)
        else:
            neural_logits = self.lm_head(x)
        bigram_logits = self.bigram_table[input_ids.reshape(-1)].to(dtype=neural_logits.dtype)
        combined = self.bigram_scale * bigram_logits + neural_logits
        return self.logit_softcap * torch.tanh(combined / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._encode(input_ids).reshape(-1, self.tok_emb.weight.size(1))
        logits = self._compute_logits(x, input_ids)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return combined logits without computing loss (for generation/calibration)."""
        x = self._encode(input_ids)
        x_flat = x.reshape(-1, x.size(-1))
        logits = self._compute_logits(x_flat, input_ids)
        return logits.reshape(x.shape[0], x.shape[1], -1)

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    # ----- DISTRIBUTED + CUDA SETUP -----
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

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
    if device.type == "cuda":
        log0(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
            console=False,
        )
    log0("=" * 100, console=False)

    # ----- SEED -----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ----- TOKENIZER + VALIDATION SETUP -----
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:tokens:{val_tokens.numel() - 1}")

    # =========================================================================
    # PHASE 1: BUILD BIGRAM TABLE (30-60 seconds on GPU, ~2 min on CPU)
    # =========================================================================
    log0("=" * 60)
    log0("PHASE 1: Building bigram table from training data...")
    t_table = time.perf_counter()
    bigram_table = build_bigram_table(
        str(dataset_dir), args.vocab_size, args.bigram_smoothing, device
    )
    table_time = time.perf_counter() - t_table
    log0(f"PHASE 1 complete: {table_time:.1f}s")

    # Quick sanity: bigram-only BPB (table is already log-probs, so CE against
    # uniform would be high, but CE against actual distribution = bigram entropy)
    log0("=" * 60)

    # =========================================================================
    # PHASE 2: TRAIN NEURAL RESIDUAL MODEL
    # =========================================================================
    log0("PHASE 2: Training neural residual model...")
    log0(f"  Architecture: {args.num_layers}L × {args.model_dim}d × MLP {args.mlp_mult}x")

    base_model = HybridGPT(
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
        bigram_table=bigram_table,
        bigram_scale_mode=args.bigram_scale_mode,
        xsa_last_n=args.xsa_last_n,
        smear_gate=args.smear_gate,
        bigram_hash_vocab=args.bigram_hash_vocab,
        bigram_hash_dim=args.bigram_hash_dim,
    ).to(device)

    if device.type == "cuda":
        base_model = base_model.bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    if device.type == "cuda":
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    else:
        compiled_model = base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer setup (same split as base)
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if args.bigram_scale_mode == "learnable":
        scalar_params.append(base_model.bigram_scale)
    # SmearGate and BigramHash params go into scalar optimizer
    if base_model.smear is not None:
        scalar_params.append(base_model.smear.gate)
    if base_model.bigram_hash is not None:
        for p in base_model.bigram_hash.parameters():
            if p.ndim < 2:
                scalar_params.append(p)
            else:
                matrix_params.append(p)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=(device.type == "cuda"),
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=(device.type == "cuda"),
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=(device.type == "cuda"),
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} (neural only, bigram table is buffer)")
    log0(f"bigram_table_bytes:{bigram_table.numel() * 2}")
    log0(f"bigram_scale_mode:{args.bigram_scale_mode} init_value:{base_model.bigram_scale.item():.4f}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")

    # ----- DATA LOADER -----
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

    # ----- WARMUP (torch.compile priming) -----
    if args.warmup_steps > 0 and device.type == "cuda":
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()
                               if name != "bigram_table"}
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
        base_model.load_state_dict(
            {**initial_model_state, "bigram_table": base_model.bigram_table},
            strict=True,
        )
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ----- MAIN TRAINING LOOP -----
    training_time_ms = 0.0
    stop_after_step: int | None = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            if device.type == "cuda":
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
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
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
            scale_str = f" bigram_scale:{base_model.bigram_scale.item():.4f}" if args.bigram_scale_mode == "learnable" else ""
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms{scale_str}"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if device.type == "cuda":
        log0(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
        )

    # =========================================================================
    # PHASE 3: QUANTIZE + COMPRESS + SAVE ARTIFACT
    # =========================================================================
    log0("=" * 60)
    log0("PHASE 3: Quantizing neural weights + packaging artifact...")

    if master_process:
        # Separate bigram table + scale from neural weights
        full_sd = base_model.state_dict()
        bigram_table_save = full_sd.pop("bigram_table").cpu().half()
        bigram_scale_save = full_sd.pop("bigram_scale").cpu().float()
        neural_sd = {k: v.detach().cpu() for k, v in full_sd.items()}

        # Collect Hessians for GPTQ (if enabled)
        hessians = None
        if args.use_gptq:
            calib_batch = 8 if device.type == "cuda" else 2
            calib_seqs = args.gptq_calib_seqs
            log0(f"gptq: generating AR calibration data ({calib_seqs} seqs x {args.train_seq_len} tokens, temp={args.gptq_calib_temp})...")
            t_gen = time.perf_counter()
            ar_tokens = generate_autoregressive_calib(
                base_model, device, num_seqs=calib_seqs,
                seq_len=args.train_seq_len, vocab_size=args.vocab_size,
                temperature=args.gptq_calib_temp, batch_size=calib_batch, seed=args.seed,
            )
            log0(f"gptq: generated {len(ar_tokens)} sequences in {time.perf_counter()-t_gen:.1f}s")
            log0("gptq: collecting hessians...")
            hessians = collect_hessians_from_tokens(base_model, ar_tokens, device)
            log0(f"gptq: collected hessians for {len(hessians)} layers")
            del ar_tokens
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Quantize neural weights to INT6 (with GPTQ if hessians available)
        quant_result, quant_meta = quantize_state_dict_int6(neural_sd, hessians=hessians)

        # Package artifact
        artifact = {
            "bigram_table": bigram_table_save,
            "bigram_scale": bigram_scale_save,
            "neural_weights": quant_result,
            "neural_meta": quant_meta,
            "config": {
                "vocab_size": args.vocab_size,
                "num_layers": args.num_layers,
                "model_dim": args.model_dim,
                "num_heads": args.num_heads,
                "num_kv_heads": args.num_kv_heads,
                "mlp_mult": args.mlp_mult,
                "tie_embeddings": args.tie_embeddings,
                "tied_embed_init_std": args.tied_embed_init_std,
                "logit_softcap": args.logit_softcap,
                "rope_base": args.rope_base,
                "qk_gain_init": args.qk_gain_init,
                "bigram_scale_mode": args.bigram_scale_mode,
                "xsa_last_n": args.xsa_last_n,
                "smear_gate": args.smear_gate,
                "bigram_hash_vocab": args.bigram_hash_vocab,
                "bigram_hash_dim": args.bigram_hash_dim,
            },
        }
        buf = io.BytesIO()
        torch.save(artifact, buf)
        raw_bytes = buf.getvalue()
        compressed = lzma.compress(raw_bytes, preset=9)

        artifact_path = "final_model.ptz"
        with open(artifact_path, "wb") as f:
            f.write(compressed)
        artifact_bytes = len(compressed)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Artifact: {artifact_bytes} bytes (raw: {len(raw_bytes)}, ratio: {len(raw_bytes)/artifact_bytes:.2f}x)")
        log0(f"Code: {code_bytes} bytes")
        log0(f"Total submission: {artifact_bytes + code_bytes} bytes ({(artifact_bytes + code_bytes) / (1024*1024):.2f} MB)")
        log0(f"Budget remaining: {16*1024*1024 - artifact_bytes - code_bytes} bytes")

    if distributed:
        dist.barrier()

    # ----- ROUNDTRIP VALIDATION -----
    log0("Validating artifact roundtrip...")
    with open("final_model.ptz", "rb") as f:
        blob = f.read()
    loaded = torch.load(io.BytesIO(lzma.decompress(blob)), map_location="cpu")
    loaded_bigram = loaded["bigram_table"].to(device)
    loaded_scale = loaded["bigram_scale"].to(device)
    loaded_neural_sd = dequantize_state_dict_int6(loaded["neural_weights"], loaded["neural_meta"])

    eval_model = HybridGPT(
        bigram_table=loaded_bigram, **loaded["config"]
    ).to(device)
    if device.type == "cuda":
        eval_model = eval_model.bfloat16()
    for module in eval_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(
        {**loaded_neural_sd, "bigram_table": loaded_bigram, "bigram_scale": loaded_scale},
        strict=True,
    )
    log0(f"loaded bigram_scale:{eval_model.bigram_scale.item():.4f}")

    if device.type == "cuda":
        eval_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
        torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, eval_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    log0(
        f"final_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Sliding window eval (if enabled)
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=args.train_seq_len,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        log0(
            f"final_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
