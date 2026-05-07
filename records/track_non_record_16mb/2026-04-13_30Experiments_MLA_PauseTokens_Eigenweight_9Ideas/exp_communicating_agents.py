"""
Experiment: Communicating Agents (Idea 5)

Two tiny neural networks (encoder + decoder) communicate through a bottleneck
channel to do language modeling.

Core insight: The information bottleneck (MSG_DIM) forces the encoder to learn
efficient compression of context. This is a creative/non-record submission --
the BPB won't be competitive but the approach is interesting.

Architecture:
  Agent A (Encoder): Small causal transformer reads the full context, mean-pools
                     hidden states to produce a message vector of MSG_DIM dims.
  Agent B (Decoder): Small MLP takes [message, local_token_embeddings] and
                     predicts the next token.

The model predicts ONE token at a time. For training efficiency, we compute loss
at every STRIDE-th position (not every position).

Env vars for architecture:
  MSG_DIM          - bottleneck message dimension (default 64)
  LOCAL_CTX        - tokens the decoder sees directly (default 8)
  ENCODER_LAYERS   - transformer layers in the encoder (default 3)
  ENCODER_DIM      - hidden dimension of the encoder (default 128)
  STRIDE           - compute loss every N positions (default 64)
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
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 200))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 0))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))

    # Communicating agents config
    msg_dim = int(os.environ.get("MSG_DIM", 64))
    local_ctx = int(os.environ.get("LOCAL_CTX", 8))
    encoder_layers = int(os.environ.get("ENCODER_LAYERS", 3))
    encoder_dim = int(os.environ.get("ENCODER_DIM", 128))
    encoder_heads = int(os.environ.get("ENCODER_HEADS", 4))
    decoder_hidden = int(os.environ.get("DECODER_HIDDEN", 256))
    stride = int(os.environ.get("STRIDE", 64))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))

    # Optimizer
    learning_rate = float(os.environ.get("LEARNING_RATE", 3e-4))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.01))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))


# -----------------------------
# BPB EVAL HELPERS
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
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for {pattern}")
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
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)


def quantize_float_tensor(t):
    t32 = t.float()
    clip_q = 99.99984 / 100.0
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), clip_q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"), 0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(t.numel()) * int(t.element_size())
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += int(t.numel()) * int(t.element_size())
            continue
        if t.numel() <= 65536:
            if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
                kept = t.float().contiguous()
            else:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(dtype=torch.float16).contiguous()
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.numel()) * int(kept.element_size())
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += int(q.numel()) * int(q.element_size()) + int(s.numel()) * int(s.element_size())
    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough,
    }
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
# ENCODER: Small causal transformer
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim=None, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
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


class EncoderAttention(nn.Module):
    """Lightweight causal multi-head attention for the encoder."""
    def __init__(self, dim, num_heads, rope_base):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        qkv = self.c_qkv(x).reshape(bsz, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)  # (bsz, heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class EncoderMLP(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        hidden = dim * mult
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, rope_base):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = EncoderAttention(dim, num_heads, rope_base)
        self.norm2 = RMSNorm(dim)
        self.mlp = EncoderMLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    """
    Agent A: reads full context tokens[0:t], produces a message vector.
    Uses a small causal transformer then mean-pools to get msg_dim output.
    """
    def __init__(self, vocab_size, enc_dim, num_layers, num_heads, msg_dim, rope_base):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, enc_dim)
        self.blocks = nn.ModuleList([
            EncoderBlock(enc_dim, num_heads, rope_base) for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(enc_dim)
        self.msg_proj = nn.Linear(enc_dim, msg_dim, bias=False)

    def forward(self, token_ids):
        """
        token_ids: (batch, seq_len) -- full context up to position t
        Returns: (batch, seq_len, msg_dim) -- a message vector per position
                 (using causal cumulative mean pool)
        """
        x = self.tok_emb(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # Causal cumulative mean pool: message at position t is mean of x[0:t+1]
        # This gives us a message for every position in one forward pass.
        cumsum = torch.cumsum(x, dim=1)  # (batch, seq_len, enc_dim)
        positions = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
        pooled = cumsum / positions  # (batch, seq_len, enc_dim)
        return self.msg_proj(pooled)  # (batch, seq_len, msg_dim)


# -----------------------------
# DECODER: MLP that reads message + local tokens
# -----------------------------

class Decoder(nn.Module):
    """
    Agent B: takes the message vector + embeddings of the last local_ctx tokens,
    predicts next token logits.
    """
    def __init__(self, vocab_size, msg_dim, local_ctx, enc_dim, hidden_dim, logit_softcap):
        super().__init__()
        self.local_ctx = local_ctx
        self.logit_softcap = logit_softcap
        # Local token embeddings (separate from encoder to keep things clean)
        self.local_emb = nn.Embedding(vocab_size, enc_dim)
        # Input: msg_dim + local_ctx * enc_dim
        input_dim = msg_dim + local_ctx * enc_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size, bias=False),
        )

    def forward(self, message, local_token_ids):
        """
        message: (batch, msg_dim)
        local_token_ids: (batch, local_ctx) -- last local_ctx tokens before target
        Returns: logits (batch, vocab_size)
        """
        local_embs = self.local_emb(local_token_ids)  # (batch, local_ctx, enc_dim)
        local_flat = local_embs.reshape(local_embs.size(0), -1)  # (batch, local_ctx * enc_dim)
        inp = torch.cat([message, local_flat], dim=-1)  # (batch, msg_dim + local_ctx * enc_dim)
        logits = self.net(inp)
        if self.logit_softcap > 0:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits


# -----------------------------
# COMBINED MODEL
# -----------------------------

class CommunicatingAgents(nn.Module):
    """
    Two-agent language model:
    - Encoder processes full context causally, produces per-position messages
    - Decoder reads message + local tokens, predicts next token
    - Loss computed at strided positions for efficiency
    """
    def __init__(self, vocab_size, enc_dim, enc_layers, enc_heads, msg_dim,
                 local_ctx, dec_hidden, logit_softcap, rope_base):
        super().__init__()
        self.local_ctx = local_ctx
        self.encoder = Encoder(vocab_size, enc_dim, enc_layers, enc_heads, msg_dim, rope_base)
        self.decoder = Decoder(vocab_size, msg_dim, local_ctx, enc_dim, dec_hidden, logit_softcap)

    def forward(self, input_ids, target_ids, stride=1):
        """
        input_ids: (batch, seq_len) -- tokens[0:T]
        target_ids: (batch, seq_len) -- tokens[1:T+1]
        stride: compute loss every stride positions
        Returns: scalar loss
        """
        batch_size, seq_len = input_ids.shape

        # Encoder: get message vectors for all positions
        messages = self.encoder(input_ids)  # (batch, seq_len, msg_dim)

        # Select strided positions for loss computation
        # positions are indices into the seq_len dimension
        # At position t, encoder has seen input_ids[0:t+1], target is target_ids[t]
        positions = list(range(self.local_ctx, seq_len, stride))
        if not positions:
            positions = [seq_len - 1]
        num_pos = len(positions)
        pos_tensor = torch.tensor(positions, device=input_ids.device, dtype=torch.long)

        # Gather messages at selected positions
        msg_at_pos = messages[:, pos_tensor, :]  # (batch, num_pos, msg_dim)
        msg_flat = msg_at_pos.reshape(batch_size * num_pos, -1)  # (batch*num_pos, msg_dim)

        # Gather local context tokens for each position
        # At position t, local tokens are input_ids[t-local_ctx+1 : t+1]
        # (the last local_ctx tokens up to and including position t)
        local_indices = []
        for t in positions:
            start = t - self.local_ctx + 1
            local_indices.append(list(range(max(start, 0), t + 1)))
        # Pad with 0 for positions near the start where we don't have enough context
        local_ids = torch.zeros(batch_size, num_pos, self.local_ctx,
                                device=input_ids.device, dtype=input_ids.dtype)
        for i, t in enumerate(positions):
            start = t - self.local_ctx + 1
            if start >= 0:
                local_ids[:, i, :] = input_ids[:, start:t + 1]
            else:
                # Pad the beginning with zeros
                pad_len = -start
                local_ids[:, i, pad_len:] = input_ids[:, 0:t + 1]

        local_ids_flat = local_ids.reshape(batch_size * num_pos, self.local_ctx)

        # Decoder forward
        logits = self.decoder(msg_flat, local_ids_flat)  # (batch*num_pos, vocab_size)

        # Targets at selected positions
        targets = target_ids[:, pos_tensor].reshape(-1)  # (batch*num_pos,)

        loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        return loss


# -----------------------------
# VALIDATION
# -----------------------------

def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """
    Evaluate on validation set. For the communicating agents model, we need to
    compute loss at ALL positions (not strided) for fair BPB comparison.
    We use stride=1 but process in chunks to avoid OOM.
    """
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = max(local_batch_tokens // args.train_seq_len, 1)
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
                # Use stride=1 for accurate validation
                batch_loss = model(x, y, stride=1).detach()

            val_loss_sum += batch_loss.to(torch.float64) * float(y.numel())
            val_token_count += float(y.numel())

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
# MAIN
# -----------------------------

def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(8 // world_size, 1)
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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

    log0(f"=== COMMUNICATING AGENTS EXPERIMENT ===")
    log0(code, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer and validation data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Build model
    base_model = CommunicatingAgents(
        vocab_size=args.vocab_size,
        enc_dim=args.encoder_dim,
        enc_layers=args.encoder_layers,
        enc_heads=args.encoder_heads,
        msg_dim=args.msg_dim,
        local_ctx=args.local_ctx,
        dec_hidden=args.decoder_hidden,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
    ).to(device).bfloat16()

    # Count and log parameters
    enc_params = sum(p.numel() for p in base_model.encoder.parameters())
    dec_params = sum(p.numel() for p in base_model.decoder.parameters())
    total_params = sum(p.numel() for p in base_model.parameters())
    log0(f"encoder_params:{enc_params} decoder_params:{dec_params} total_params:{total_params}")
    log0(f"msg_dim:{args.msg_dim} local_ctx:{args.local_ctx} "
         f"encoder_layers:{args.encoder_layers} encoder_dim:{args.encoder_dim} "
         f"decoder_hidden:{args.decoder_hidden} stride:{args.stride}")

    model = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    # Optimizer: simple Adam for everything
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_schedule(step, elapsed_ms):
        """Cosine warmdown schedule."""
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            return remaining_ms / max(warmdown_ms, 1e-9)
        return 1.0

    # Training loop
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
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                     f"step:{step}/{args.iterations}")
            break

        # LR schedule
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_schedule(step, elapsed_ms)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale

        # Forward/backward with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, stride=args.stride)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Gradient clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        optimizer.step()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms "
                 f"step_avg:{approx_training_time_ms / step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # Quantize and save
    if master_process:
        code_bytes = len(code.encode("utf-8"))
        log0(f"Code size: {code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)

    if master_process:
        with open("final_model_agents.int8.ptz", "wb") as f:
            f.write(quant_blob)
        total_size = os.path.getsize("final_model_agents.int8.ptz") + len(code.encode("utf-8"))
        log0(f"Total submission size int8+zlib: {total_size} bytes")
        log0(f"quant_stats: {quant_stats}")

    # Roundtrip: load quantized model and re-evaluate
    with open("final_model_agents.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
