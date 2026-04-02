"""
Annealed-Muon 1.58-bit model utilizing Grouped Parameter Tying
Compliant with the 10-minute / 16MB Parameter Golf Constraints.
Featuring cosine-annealed quantization, Muon+, XSA, U-Net skips, and surprise LR boost.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
try:
    import zstandard
except ImportError:
    zstandard = None
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    def __init__(self):
        self.data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
        self.train_files = os.path.join(self.data_path, "fineweb_train_*.bin")
        self.val_files = os.path.join(self.data_path, "fineweb_val_*.bin")
        self.tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
        self.run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
        self.seed = int(os.environ.get("SEED", 1337))

        self.val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
        self.val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 100))
        self.train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 20))
        self.muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 100))

        self.iterations = int(os.environ.get("ITERATIONS", 3000))
        self.warmup_steps = int(os.environ.get("WARMUP_STEPS", 5))
        self.train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 262_144))
        self.train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
        self.max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
        self.qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

        self.vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
        self.num_layers = int(os.environ.get("NUM_LAYERS", 10))
        self.num_unique_blocks = int(os.environ.get("NUM_UNIQUE_BLOCKS", 10))
        self.model_dim = int(os.environ.get("MODEL_DIM", 768))
        self.num_heads = int(os.environ.get("NUM_HEADS", 8))
        self.mlp_mult = int(os.environ.get("MLP_MULT", 4))

        self.num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 8))
        self.tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
        self.rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
        self.logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

        self.embed_lr = float(os.environ.get("EMBED_LR", 0.05))
        self.head_lr = float(os.environ.get("HEAD_LR", 0.008))
        self.tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
        self.tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

        self.matrix_lr = float(os.environ.get("MATRIX_LR", 0.8))
        self.scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))

        self.muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.90))
        self.muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
        self.muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))

        self.beta1 = float(os.environ.get("BETA1", 0.9))
        self.beta2 = float(os.environ.get("BETA2", 0.95))
        self.adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
        self.grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
        self.latent_clip_scale = float(os.environ.get("LATENT_CLIP_SCALE", 3.0))

        self.eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
        self.ppm_enabled = bool(int(os.environ.get("PPM_ENABLED", "1")))
        self.ppm_alpha = float(os.environ.get("PPM_ALPHA", 0.95))

        self.bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 0))
        self.bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))


# -----------------------------
# MUON+ OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    X /= X.norm() + eps
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, backend_steps, nesterov = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            updates_flat = torch.zeros(sum(int(p.numel()) for p in params), device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                p.add_(updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype), alpha=-lr)
                curr += p.numel()
        return loss

# -----------------------------
# TOKENIZER & DATA
# -----------------------------

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    table_size = max(int(sp.vocab_size()), vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = torch.cat([torch.from_numpy(np.fromfile(file, dtype="<u2", count=int(np.fromfile(file, dtype="<i4", count=256)[2]), offset=256 * 4).astype(np.uint16, copy=False)) for file in files]).contiguous()
    return tokens[: ((tokens.numel() - 1) // seq_len) * seq_len + 1]

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.rank, self.world_size, self.device = rank, world_size, device
        self.file_idx, self.pos = 0, 0
        self.tokens = torch.from_numpy(np.fromfile(self.files[0], dtype="<u2", count=int(np.fromfile(self.files[0], dtype="<i4", count=256)[2]), offset=256 * 4).astype(np.uint16, copy=False))

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        per_rank_span = (global_tokens // (self.world_size * grad_accum_steps)) + 1
        n = per_rank_span * self.world_size
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens = torch.from_numpy(np.fromfile(self.files[self.file_idx], dtype="<u2", count=int(np.fromfile(self.files[self.file_idx], dtype="<i4", count=256)[2]), offset=256 * 4).astype(np.uint16, copy=False))
                self.pos = 0
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        local = (chunks[0] if len(chunks) == 1 else torch.cat(chunks))[self.rank * per_rank_span : (self.rank + 1) * per_rank_span].to(dtype=torch.int64)
        return local[:-1].reshape(-1, seq_len).to(self.device, non_blocking=True), local[1:].reshape(-1, seq_len).to(self.device, non_blocking=True)

# -----------------------------
# EVALUATION
# -----------------------------

def eval_val(args: Hyperparameters, model: nn.Module, rank: int, world_size: int, device: torch.device,
             grad_accum_steps: int, val_tokens: Tensor, base_bytes_lut: Tensor,
             has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
             step_fraction_val: float = 1.0, stride: int = 0) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    use_sliding = 0 < stride < seq_len
    batch_seqs = max(1, args.val_batch_size // (seq_len * world_size))

    sf_t = torch.tensor([step_fraction_val], device=device, dtype=torch.float32)

    if use_sliding:
        n_windows = (total_tokens - seq_len) // stride + 1
        win_start = (n_windows * rank) // world_size
        win_end = (n_windows * (rank + 1)) // world_size
    else:
        total_seqs = total_tokens // seq_len
        win_start = (total_seqs * rank) // world_size
        win_end = (total_seqs * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    with torch.no_grad():
        for wb in range(win_start, win_end, batch_seqs):
            we = min(wb + batch_seqs, win_end)
            if use_sliding:
                x = torch.stack([val_tokens[w * stride : w * stride + seq_len] for w in range(wb, we)])
                y = torch.stack([val_tokens[w * stride + 1 : w * stride + seq_len + 1] for w in range(wb, we)])
            else:
                raw_start = wb * seq_len
                raw_end = we * seq_len + 1
                local = val_tokens[raw_start:raw_end]
                x = local[:-1].reshape(-1, seq_len)
                y = local[1:].reshape(-1, seq_len)
            x = x.to(device=device, dtype=torch.int64, non_blocking=True)
            y = y.to(device=device, dtype=torch.int64, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                if use_sliding:
                    per_token = model(x, y, step_fraction=sf_t, per_token=True)
                    scored_loss = per_token[:, -stride:]
                else:
                    batch_loss = model(x, y, step_fraction=sf_t)

            if use_sliding:
                val_loss_sum += scored_loss.to(torch.float64).sum()
                val_token_count += scored_loss.numel()
                scored_x, scored_y = x[:, -stride:], y[:, -stride:]
            else:
                val_loss_sum += batch_loss.to(torch.float64) * float(y.numel())
                val_token_count += float(y.numel())
                scored_x, scored_y = x, y

            tgt_ids = scored_y.reshape(-1)
            prev_ids = scored_x.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in (val_loss_sum, val_token_count, val_byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    return float(val_loss.item()), float(val_loss.item() / math.log(2.0) * (val_token_count.item() / val_byte_count.item()))


def build_ppm_predictions(tokens: Tensor, vocab_size: int) -> np.ndarray:
    tok = tokens.numpy().astype(np.int64) if isinstance(tokens, Tensor) else tokens.astype(np.int64)
    n = len(tok) - 1
    base_prob = np.float32(1.0 / vocab_size)
    prev_tok, next_tok = tok[:n], tok[1:n + 1]
    bigram_counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(bigram_counts, (prev_tok, next_tok), 1)
    row_totals = bigram_counts.sum(axis=1)
    row_distinct = (bigram_counts > 0).sum(axis=1).astype(np.float64)
    denom = np.maximum(row_totals + row_distinct, 1.0)
    smoothed = bigram_counts / denom[:, None] + (row_distinct / denom)[:, None] * base_prob
    probs = smoothed[prev_tok, next_tok].astype(np.float32)
    probs[row_totals[prev_tok] < 2] = base_prob
    return probs


def eval_val_hybrid(args: Hyperparameters, model: nn.Module, rank: int, world_size: int,
                    device: torch.device, grad_accum_steps: int, val_tokens: Tensor,
                    base_bytes_lut: Tensor, has_leading_space_lut: Tensor,
                    is_boundary_token_lut: Tensor, ppm_probs: np.ndarray,
                    alpha: float = 0.95,
                    step_fraction_val: float = 0.0) -> tuple[float, float]:
    seq_len, stride = args.train_seq_len, args.eval_stride
    total_tokens = val_tokens.numel() - 1
    n_windows = (total_tokens - seq_len) // stride + 1
    batch_seqs = max(1, args.val_batch_size // (seq_len * grad_accum_steps))
    win_start = (n_windows * rank) // world_size
    win_end = (n_windows * (rank + 1)) // world_size

    sf_t = torch.tensor([step_fraction_val], device=device, dtype=torch.float32)

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ppm_probs_t = torch.from_numpy(ppm_probs).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        for wb in range(win_start, win_end, batch_seqs):
            we = min(wb + batch_seqs, win_end)
            x = torch.stack([val_tokens[w * stride : w * stride + seq_len] for w in range(wb, we)])
            y = torch.stack([val_tokens[w * stride + 1 : w * stride + seq_len + 1] for w in range(wb, we)])
            x = x.to(device=device, dtype=torch.int64, non_blocking=True)
            y = y.to(device=device, dtype=torch.int64, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x, y, step_fraction=sf_t, return_logits=True)
            scored_logits = logits[:, -stride:, :].float()
            scored_targets = y[:, -stride:]
            p_neural = F.softmax(scored_logits, dim=-1)
            p_neural_correct = p_neural.gather(-1, scored_targets.unsqueeze(-1)).squeeze(-1)
            bsz = we - wb
            global_positions = torch.zeros(bsz, stride, dtype=torch.long, device=device)
            for b_idx, w in enumerate(range(wb, we)):
                start_pos = w * stride + (seq_len - stride)
                global_positions[b_idx] = torch.arange(start_pos, start_pos + stride, device=device)
            p_ppm = ppm_probs_t[global_positions.clamp(max=len(ppm_probs) - 1)]
            p_blend = alpha * p_neural_correct + (1.0 - alpha) * p_ppm
            per_token_loss = -torch.log(p_blend.clamp(min=1e-30))
            val_loss_sum += per_token_loss.to(torch.float64).sum()
            val_token_count += per_token_loss.numel()
            scored_x, scored_y = x[:, -stride:], y[:, -stride:]
            tgt_ids, prev_ids = scored_y.reshape(-1), scored_x.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in (val_loss_sum, val_token_count, val_byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    return float(val_loss.item()), float(val_loss.item() / math.log(2.0) * (val_token_count.item() / val_byte_count.item()))

# -----------------------------
# TERNARY PACKING / UNPACKING
# -----------------------------

def pack_base3_uint8(w_ternary: Tensor) -> tuple[Tensor, int, int, int]:
    out_features, in_features = w_ternary.shape
    d = (w_ternary.clamp(-1.0, 1.0).round() + 1.0).to(torch.uint8).flatten()
    pad_len = (5 - (d.numel() % 5)) % 5
    if pad_len != 0: d = F.pad(d, (0, pad_len), value=1)
    return (d.view(-1, 5).to(torch.int32) * torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=d.device)).sum(dim=1).to(torch.uint8), out_features, in_features, pad_len

def unpack_base3_uint8(v_packed: Tensor, out_features: int, in_features: int, pad_len: int) -> Tensor:
    d = ((v_packed.to(torch.int32).unsqueeze(1) // torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=v_packed.device)) % 3).flatten()
    if pad_len > 0: d = d[:-pad_len]
    return (d.to(torch.bfloat16) - 1.0).view(out_features, in_features)


# -----------------------------
# TRANSFORMER ARCHITECTURE
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class AnnealedBitLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight_latent = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight_latent, a=math.sqrt(5))
        if bias: self.bias = nn.Parameter(torch.zeros(out_features))
        else: self.register_parameter('bias', None)
        self._zero_init = False

    def forward(self, x: Tensor, step_fraction: Tensor) -> Tensor:
        w = self.weight_latent
        w_det = w.detach()
        w_abs = w_det.abs()

        scale = w_abs.mean(dim=1, keepdim=True).clamp(min=1e-8)
        w_quant = torch.sign(w_det) * (w_abs > 0.7 * scale).float() * scale

        W = w + (step_fraction * (w_quant - w)).detach()

        return F.linear(x, W.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1.0 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
            self.proj._zero_init = True
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

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return self.cos_cached[:, :, :seq_len, :].to(dtype), self.sin_cached[:, :, :seq_len, :].to(dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, dim // num_heads
        self.c_q = AnnealedBitLinear(dim, dim, bias=False)
        self.c_k = AnnealedBitLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.c_v = AnnealedBitLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.proj = AnnealedBitLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def _xsa(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(3)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, step_fraction: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x, step_fraction).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x, step_fraction).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x, step_fraction).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, q.dtype)
        q = apply_rotary_emb(q, cos, sin) * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        k = apply_rotary_emb(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2)
        y = self._xsa(y, v.transpose(1, 2))
        return self.proj(y.contiguous().reshape(bsz, seqlen, dim), step_fraction)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        self.fc = AnnealedBitLinear(dim, mlp_mult * dim, bias=False)
        self.proj = AnnealedBitLinear(mlp_mult * dim, dim, bias=False)

    def forward(self, x: Tensor, step_fraction: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x, step_fraction), 0.5).square(), step_fraction)

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float, layer_idx: int):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.register_buffer("ln_scale_factor_t", torch.tensor(1.0 / math.sqrt(layer_idx + 1)))

    def forward(self, x: Tensor, x0: Tensor, step_fraction: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        sf = self.ln_scale_factor_t.to(dtype=x.dtype)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x) * sf, step_fraction)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * sf, step_fraction)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_unique_blocks: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 bigram_vocab_size: int = 0, bigram_dim: int = 128):
        super().__init__()
        self.tie_embeddings, self.logit_softcap = tie_embeddings, logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear_gate = SmearGate(model_dim)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        self.blocks = nn.ModuleList([Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx=i) for i in range(num_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True

        if self.tie_embeddings: nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, CastedLinear, AnnealedBitLinear)):
                w = getattr(module, 'weight_latent', getattr(module, 'weight', None))
                if w is not None:
                    if getattr(module, "_zero_init", False):
                        nn.init.zeros_(w)
                    elif isinstance(module, AnnealedBitLinear):
                        pass
                    else:
                        nn.init.orthogonal_(w)
                        if ".proj" in name:
                            with torch.no_grad(): w.mul_(1.0 / math.sqrt(2 * num_layers))

        for i in range(num_layers):
            leader_index = i % num_unique_blocks
            if i == leader_index: continue
            block, leader = self.blocks[i], self.blocks[leader_index]
            block.attn.c_q.weight_latent = leader.attn.c_q.weight_latent
            block.attn.c_k.weight_latent = leader.attn.c_k.weight_latent
            block.attn.c_v.weight_latent = leader.attn.c_v.weight_latent
            block.attn.proj.weight_latent = leader.attn.proj.weight_latent
            block.mlp.fc.weight_latent = leader.mlp.fc.weight_latent
            block.mlp.proj.weight_latent = leader.mlp.proj.weight_latent
            block.attn.q_gain = leader.attn.q_gain

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None,
                step_fraction: Tensor | None = None,
                per_token: bool = False, return_logits: bool = False) -> Tensor:
        if step_fraction is None: step_fraction = torch.tensor([1.0], device=input_ids.device, dtype=torch.float32)

        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.weight.size(-1),))
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = self.smear_gate(x)
        x0 = x.clone()

        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, step_fraction)
            skips.append(x)

        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0, step_fraction)

        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        logits_proj = F.linear(x_flat, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

        if return_logits:
            return logits.view(input_ids.shape[0], input_ids.shape[1], -1)

        targets = target_ids.reshape(-1)
        if per_token:
            return F.cross_entropy(logits.float(), targets, reduction="none").view(input_ids.shape)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# DIAGNOSTIC SNIFFER
# -----------------------------
@torch.no_grad()
def measure_quantization_health(model: nn.Module) -> tuple[float, float]:
    total_params, stuck_at_zero, active_edges = 0, 0, 0
    processed_ids = set()
    for module in model.modules():
        if isinstance(module, AnnealedBitLinear):
            w = module.weight_latent
            wid = id(w)
            if wid not in processed_ids:
                processed_ids.add(wid)
                scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
                threshold = 0.7 * scale
                stuck_at_zero += (w.abs() < threshold).sum().item()
                active_edges += (w.abs() >= threshold).sum().item()
                total_params += w.numel()
    zero_pct = (stuck_at_zero / max(total_params, 1)) * 100.0
    edge_pct = (active_edges / max(total_params, 1)) * 100.0
    return zero_pct, edge_pct


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5
    args = Hyperparameters()
    # zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if 8 % world_size != 0: raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device)
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = True, True
    torch._dynamo.config.optimize_ddp = False
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = f"logs/{args.run_id}.txt" if master_process else None
    if master_process: os.makedirs("logs", exist_ok=True)
    def log0(msg: str):
        if master_process:
            print(msg)
            if logfile:
                with open(logfile, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = GPT(args.vocab_size, args.num_layers, args.num_unique_blocks, args.model_dim, args.num_heads,
                     args.num_kv_heads, args.mlp_mult, args.tie_embeddings, args.tied_embed_init_std,
                     args.logit_softcap, args.rope_base, args.qk_gain_init,
                     bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, (CastedLinear, AnnealedBitLinear)): module.float()

    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if (param.ndim < 2 or "scale" in name or "norm" in name or "gate" in name or "resid_mix" in name or "skip_weights" in name) and param.dtype != torch.float32:
                param.data = param.data.float()

    raw_model = base_model
    base_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    matrix_params, scalar_params = [], []
    for name, p in base_model.named_parameters():
        if "tok_emb" in name or "lm_head" in name: continue
        if p.ndim == 2 and "weight_latent" in name: matrix_params.append(p)
        else: scalar_params.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizers = [
        torch.optim.AdamW([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True),
        Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps),
        torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    ]
    for group in optimizers[1].param_groups: group["base_lr"] = args.matrix_lr

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    # Layer-wise clip multipliers
    middle_counts: dict[int, int] = {}
    total_counts: dict[int, int] = {}
    for layer_idx in range(args.num_layers):
        bi = layer_idx % args.num_unique_blocks
        total_counts[bi] = total_counts.get(bi, 0) + 1
        if 4 <= layer_idx <= 9:
            middle_counts[bi] = middle_counts.get(bi, 0) + 1
    clip_multipliers: dict[int, float] = {}
    for bi in range(args.num_unique_blocks):
        mid_frac = middle_counts.get(bi, 0) / total_counts.get(bi, 1)
        clip_multipliers[bi] = 1.0 - 0.4 * mid_frac

    # --- COMPILER WARMUP (not counted against training budget) ---
    avg_step_ms = 600.0
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        sf_zero = torch.tensor([0.0], device=device, dtype=torch.float32)

        t_compile_start = time.perf_counter()
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        for micro_step in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                (model(x, y, step_fraction=sf_zero) * (1.0 / grad_accum_steps)).backward()
        for opt in optimizers: opt.step()
        torch.cuda.synchronize()
        compile_ms = 1000.0 * (time.perf_counter() - t_compile_start)
        log0(f"Compilation (primer step): {compile_ms:.0f}ms")

        t_warmup_start = time.perf_counter()
        for _ in range(args.warmup_steps):
            for opt in optimizers: opt.zero_grad(set_to_none=True)
            for micro_step in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    (model(x, y, step_fraction=sf_zero) * (1.0 / grad_accum_steps)).backward()
            for opt in optimizers: opt.step()
        torch.cuda.synchronize()
        warmup_time_ms = 1000.0 * (time.perf_counter() - t_warmup_start)
        avg_step_ms = warmup_time_ms / max(1, args.warmup_steps)
        log0(f"Warmup ({args.warmup_steps} steps): {warmup_time_ms:.0f}ms, avg {avg_step_ms:.0f}ms/step")

        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states):
            opt.load_state_dict(state)
        for opt in optimizers: opt.zero_grad(set_to_none=True)
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        del initial_model_state, initial_optimizer_states

    if distributed:
        sync_metrics = torch.tensor([avg_step_ms], dtype=torch.float32, device=device)
        dist.broadcast(sync_metrics, src=0)
        avg_step_ms = sync_metrics.item()

    safety_buffer_ms = (avg_step_ms * 2.5) + 5000.0
    projected_main_steps = max(1, int((max_wallclock_ms - safety_buffer_ms) / avg_step_ms))

    args.muon_momentum_warmup_steps = max(2, int(projected_main_steps * 0.10))
    args.val_loss_every = 0
    args.train_log_every = max(1, projected_main_steps // 20)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    log0("\n" + "=" * 50)
    log0("HARDWARE AUTO-CALIBRATION COMPLETE")
    log0(f"   - GPU:               {gpu_name} x{world_size}")
    log0(f"   - Hardware Speed:    {avg_step_ms:.0f} ms/step")
    log0(f"   - Projected Steps:   {projected_main_steps} steps remaining")
    log0(f"   - Muon Warmup Phase: {args.muon_momentum_warmup_steps} steps")
    log0(f"   - Validation:        End-of-training only")
    log0(f"   - Sniffer Log Freq:  Every {args.train_log_every} steps")
    log0(f"   - I/O Safety Buffer: {safety_buffer_ms / 1000.0:.1f} seconds")
    log0(f"   - Model:             {args.model_dim}d / {args.num_layers}L / {args.num_unique_blocks}UB / MLP{args.mlp_mult}x")
    log0(f"   - NS Steps:          {args.muon_backend_steps}")
    log0(f"   - BigramHash:        {args.bigram_vocab_size} buckets, dim={args.bigram_dim}")
    log0(f"   - LR Schedule:       Hold-Cosine (hold=0.70, min_lr=0.01)")
    log0(f"   - Step Fraction:     phi-exponent (t^1.618)")
    log0(f"   - matrix_lr:         {args.matrix_lr}")
    log0("=" * 50 + "\n")

    loop_max_ms = max_wallclock_ms - safety_buffer_ms
    training_time_ms = 0.0
    t0 = time.perf_counter()

    step = 0
    stop_after_step = None

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)

            val_loss, val_bpb = eval_val(args, raw_model, rank, world_size, device, grad_accum_steps,
                                         val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step: break

        # --- HOLD-COSINE LR SCHEDULE ---
        loop_elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        fraction = min(loop_elapsed_ms / max(loop_max_ms, 1.0), 1.0)
        _phi = (1.0 + math.sqrt(5.0)) / 2.0
        step_frac_val = fraction ** _phi
        sf_t = torch.tensor([step_frac_val], device=device, dtype=torch.float32)

        for opt in optimizers: opt.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, step_fraction=sf_t)
            train_loss += loss.detach()
            (loss * (1.0 / grad_accum_steps)).backward()
        train_loss /= grad_accum_steps

        hold = 0.70
        min_lr_scale = 0.01
        if fraction < hold:
            scale = 1.0
        else:
            cosine_progress = (fraction - hold) / (1.0 - hold)
            scale = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        # ---

        frac_muon = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for group in optimizers[1].param_groups: group["momentum"] = (1 - frac_muon) * args.muon_momentum_warmup_start + frac_muon * args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        for opt in optimizers: opt.step()

        # Layer-wise latent clipping
        with torch.no_grad():
            for bi in range(args.num_unique_blocks):
                bound = 1.0 + args.latent_clip_scale * clip_multipliers[bi] * (1.0 - step_frac_val)
                block = base_model.blocks[bi]
                for module in block.modules():
                    if isinstance(module, AnnealedBitLinear):
                        module.weight_latent.clamp_(-bound, bound)

        step += 1

        # --- KILL SWITCH (based on training time only, excludes compile warmup) ---
        train_elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        reached_cap = max_wallclock_ms is not None and train_elapsed_ms >= (max_wallclock_ms - safety_buffer_ms)
        if distributed and max_wallclock_ms is not None:
            cap_tensor = torch.tensor([1 if reached_cap else 0], dtype=torch.int32, device=device)
            dist.all_reduce(cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = cap_tensor.item() > 0

        if stop_after_step is None and reached_cap:
            log0(f"stopping_early: cap reached at step {step} (Train time: {train_elapsed_ms / 1000.0:.1f}s)")
            stop_after_step = step

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            zero_pct, edge_pct = measure_quantization_health(base_model)
            log0(f"step:{step} train_loss:{train_loss.item():.4f} "
                 f"step_avg:{loop_elapsed_ms / max(step, 1):.0f}ms "
                 f"frac:{fraction:.2f} lr_scale:{scale:.3f} "
                 f"[Zero: {zero_pct:.1f}% | Edges: {edge_pct:.1f}%]")

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------

    if master_process:
        packed_state_dict: dict[str, Tensor] = {}
        packed_ids: set[int] = set()
        for name, module in base_model.named_modules():
            if isinstance(module, AnnealedBitLinear):
                wid = id(module.weight_latent)
                if wid not in packed_ids:
                    packed_ids.add(wid)
                    w = module.weight_latent.detach().float()
                    row_scale = w.abs().mean(dim=1).clamp(min=1e-8)
                    threshold = 0.7 * row_scale
                    w_ternary = torch.zeros_like(w, dtype=torch.int8)
                    w_ternary[w > threshold[:, None]] = 1
                    w_ternary[w < -threshold[:, None]] = -1
                    v_packed, out_f, in_f, pad_len = pack_base3_uint8(w_ternary.float())
                    packed_state_dict[f"{name}.v_packed"] = v_packed.cpu()
                    packed_state_dict[f"{name}.shape_info"] = torch.tensor([out_f, in_f, pad_len], dtype=torch.int32).cpu()
                    packed_state_dict[f"{name}.row_scale"] = row_scale.to(torch.float16).cpu()

        for sname, param in base_model.state_dict().items():
            if "weight_latent" not in sname:
                p = param.detach().cpu()
                if p.is_floating_point() and p.numel() > 1:
                    packed_state_dict[sname] = p.to(torch.float8_e4m3fn)
                else:
                    packed_state_dict[sname] = p

        quant_buf = io.BytesIO()
        torch.save(packed_state_dict, quant_buf)
        quant_raw = quant_buf.getvalue()
        if zstandard is not None:
            compressed = zstandard.ZstdCompressor(level=22).compress(quant_raw)
            cname = "zstd-22"
        else:
            compressed = zlib.compress(quant_raw, level=9)
            cname = "zlib-9"
        with open("final_model.ptz", "wb") as f:
            f.write(compressed)
        log0(f"Serialized model {cname}: {os.path.getsize('final_model.ptz')} bytes")

    if distributed: dist.barrier()

    with open("final_model.ptz", "rb") as f:
        blob = f.read()
    if zstandard is not None:
        raw = zstandard.ZstdDecompressor().decompress(blob)
    else:
        raw = zlib.decompress(blob)
    packed_state_dict = torch.load(io.BytesIO(raw), map_location="cpu")
    dequant_state_dict: dict[str, Tensor] = {}
    for sname, param in packed_state_dict.items():
        if sname.endswith(".v_packed"):
            prefix = sname[:-len(".v_packed")]
            out_f, in_f, pad_len = packed_state_dict[f"{prefix}.shape_info"].tolist()
            row_scale = packed_state_dict[f"{prefix}.row_scale"].float()
            w_ternary = unpack_base3_uint8(param, out_f, in_f, pad_len)
            dequant_state_dict[f"{prefix}.weight_latent"] = (w_ternary.float() * row_scale[:, None]).to(torch.bfloat16)
        elif not sname.endswith(".shape_info") and not sname.endswith(".row_scale"):
            if param.dtype == torch.float8_e4m3fn:
                dequant_state_dict[sname] = param.to(torch.bfloat16)
            else:
                dequant_state_dict[sname] = param

    base_model.load_state_dict(dequant_state_dict, strict=False)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, raw_model, rank, world_size, device, grad_accum_steps,
                                     val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                                     step_fraction_val=0.0)
    torch.cuda.synchronize()
    log0(f"final_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")

    # Sliding window + PPM evals disabled for speed; enable when model is competitive
    # if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
    #     ...sliding window + PPM hybrid eval...

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
