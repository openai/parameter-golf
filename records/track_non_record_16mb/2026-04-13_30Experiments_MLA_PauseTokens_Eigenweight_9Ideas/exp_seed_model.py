"""
Experiment: Seed Model (Idea 0)
Train a tiny seed vector phi in R^d, expand to full transformer weights via:
    theta = theta_0 + P * phi
where theta_0 and P are regenerated from a fixed integer PRNG seed.

The ONLY trained parameters are:
  - phi (seed vector, size SEED_DIM)
  - token embedding
  - per-layer scalar controls (attn_scale, mlp_scale, q_gain)

P is generated chunk-by-chunk to avoid materializing the full D x d matrix.
Columns of P are normalized to unit norm to prevent gradient vanishing.
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


# --- HYPERPARAMETERS ---
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
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Seed model specific
    seed_dim = int(os.environ.get("SEED_DIM", 2048))
    prng_seed = int(os.environ.get("PRNG_SEED", 42))  # Fixed seed for P and theta_0
    seed_lr = float(os.environ.get("SEED_LR", 0.01))
    chunk_rows = int(os.environ.get("CHUNK_ROWS", 10000))  # Rows of P per chunk

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))


# --- TOKENIZER-AGNOSTIC BPB EVAL (unchanged from baseline) ---
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
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
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
        self.rank = rank
        self.world_size = world_size
        self.device = device
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


# --- QUANTIZATION (unchanged from baseline) ---
CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
                                  "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights")
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0

def tensor_nbytes(t):
    return int(t.numel()) * int(t.element_size())

def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes = {}
    qmeta = {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
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
            if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
                kept = t.float().contiguous()
            else:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
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
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
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


# --- TRANSFORMER HELPERS (RoPE, RMSNorm) ---

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


# --- WEIGHT LAYOUT: define which tensors are expanded from seed ---
# Each entry: (name_template, shape) where name_template uses {i} for layer index.
# These are the linear weight matrices that get expanded from the seed vector.

def build_weight_layout(num_layers, model_dim, num_heads, num_kv_heads, mlp_mult):
    """Return list of (key, shape) for all weight tensors expanded from seed."""
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    hidden_dim = mlp_mult * model_dim
    layout = []
    for i in range(num_layers):
        layout.append((f"blocks.{i}.attn.c_q.weight", (model_dim, model_dim)))
        layout.append((f"blocks.{i}.attn.c_k.weight", (kv_dim, model_dim)))
        layout.append((f"blocks.{i}.attn.c_v.weight", (kv_dim, model_dim)))
        layout.append((f"blocks.{i}.attn.proj.weight", (model_dim, model_dim)))
        layout.append((f"blocks.{i}.mlp.fc.weight", (hidden_dim, model_dim)))
        layout.append((f"blocks.{i}.mlp.proj.weight", (model_dim, hidden_dim)))
    return layout


def compute_full_model_dim(layout):
    """Total number of scalar weights across all expanded tensors."""
    return sum(s[0] * s[1] for _, s in layout)


# --- CHUNKED SEED EXPANSION ---

def expand_seed_chunked(phi, layout, prng_seed, chunk_rows, device):
    """
    Expand seed vector phi (d,) into a flat weight vector (D,) via:
        theta = theta_0 + P @ phi
    where theta_0 and P are generated chunk-by-chunk from prng_seed.

    P columns are normalized to unit norm per chunk to prevent gradient vanishing.
    Returns a flat fp32 tensor of size D.
    """
    d = phi.shape[0]
    D = sum(s[0] * s[1] for _, s in layout)
    theta = torch.zeros(D, device=device, dtype=torch.float32)

    offset = 0
    chunk_idx = 0
    remaining = D
    while remaining > 0:
        rows = min(chunk_rows, remaining)
        # Deterministic PRNG for this chunk
        gen = torch.Generator(device="cpu")
        gen.manual_seed(prng_seed * 1000000 + chunk_idx)

        # Generate theta_0 chunk (small init scale)
        theta_0_chunk = torch.randn(rows, generator=gen, device="cpu", dtype=torch.float32) * 0.02
        theta_0_chunk = theta_0_chunk.to(device)

        # Generate P chunk (rows x d) and normalize columns
        P_chunk = torch.randn(rows, d, generator=gen, device="cpu", dtype=torch.float32)
        P_chunk = P_chunk.to(device)
        # Normalize columns to unit norm for this chunk
        col_norms = P_chunk.norm(dim=0, keepdim=True).clamp(min=1e-8)
        P_chunk = P_chunk / col_norms

        # Expand: theta_chunk = theta_0_chunk + P_chunk @ phi
        theta[offset : offset + rows] = theta_0_chunk + P_chunk @ phi

        offset += rows
        chunk_idx += 1
        remaining -= rows

    return theta


def slice_weights_from_flat(theta_flat, layout):
    """Slice a flat theta vector into a dict of named weight tensors."""
    weights = {}
    offset = 0
    for name, shape in layout:
        numel = shape[0] * shape[1]
        weights[name] = theta_flat[offset : offset + numel].view(shape)
        offset += numel
    return weights


# --- STATELESS TRANSFORMER FORWARD PASS ---
# The transformer forward pass takes a weight dict as argument rather than
# storing weights as nn.Parameters. This lets us expand from seed each step.

def stateless_forward(
    input_ids,          # (B, T)
    target_ids,         # (B, T)
    weight_dict,        # dict of expanded weight tensors (bf16)
    tok_emb_weight,     # (V, D) token embedding
    per_layer_params,   # dict of per-layer scalar params
    rotary,             # Rotary module
    num_layers,
    num_heads,
    num_kv_heads,
    model_dim,
    mlp_mult,
    logit_softcap,
    num_encoder_layers,
    num_decoder_layers,
    num_skip_weights,
):
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim

    # Embedding
    x = F.embedding(input_ids, tok_emb_weight)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    bsz, seqlen, dim = x.shape

    # RoPE
    cos, sin = rotary(seqlen, x.device, x.dtype)

    skips = []
    for i in range(num_layers):
        # Skip connection (U-net style)
        if i >= num_encoder_layers and skips:
            skip_idx = i - num_encoder_layers
            sw = per_layer_params[f"skip_weights"][skip_idx].to(dtype=x.dtype)
            x = x + sw[None, None, :] * skips.pop()

        # Residual mix
        mix = per_layer_params[f"blocks.{i}.resid_mix"].to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # --- Attention ---
        h = F.rms_norm(x, (x.size(-1),))

        # Q, K, V projections
        w_q = weight_dict[f"blocks.{i}.attn.c_q.weight"]
        w_k = weight_dict[f"blocks.{i}.attn.c_k.weight"]
        w_v = weight_dict[f"blocks.{i}.attn.c_v.weight"]
        w_proj = weight_dict[f"blocks.{i}.attn.proj.weight"]

        q = F.linear(h, w_q).reshape(bsz, seqlen, num_heads, head_dim).transpose(1, 2)
        k = F.linear(h, w_k).reshape(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2)
        v = F.linear(h, w_v).reshape(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2)

        # QK norm + RoPE
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Q gain
        q_gain = per_layer_params[f"blocks.{i}.attn.q_gain"].to(dtype=q.dtype)
        q = q * q_gain[None, :, None, None]

        # Scaled dot product attention
        attn_scale = per_layer_params[f"blocks.{i}.attn_scale"].to(dtype=x.dtype)
        y_attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(num_kv_heads != num_heads),
        )
        y_attn = y_attn.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        y_attn = F.linear(y_attn, w_proj)
        x = x + attn_scale[None, None, :] * y_attn

        # --- MLP (relu^2) ---
        h2 = F.rms_norm(x, (x.size(-1),))
        w_fc = weight_dict[f"blocks.{i}.mlp.fc.weight"]
        w_mlp_proj = weight_dict[f"blocks.{i}.mlp.proj.weight"]
        mlp_scale = per_layer_params[f"blocks.{i}.mlp_scale"].to(dtype=x.dtype)

        h2 = torch.relu(F.linear(h2, w_fc))
        h2 = F.linear(h2.square(), w_mlp_proj)
        x = x + mlp_scale[None, None, :] * h2

        # Store skip for encoder layers
        if i < num_encoder_layers:
            skips.append(x)

    # Final norm + logits
    x = F.rms_norm(x, (x.size(-1),)).reshape(-1, dim)
    targets = target_ids.reshape(-1)

    # Tied embeddings
    logits_proj = F.linear(x, tok_emb_weight)
    logits = logit_softcap * torch.tanh(logits_proj / logit_softcap)
    return F.cross_entropy(logits.float(), targets, reduction="mean")


# --- SEED MODEL WRAPPER ---

class SeedModel(nn.Module):
    """
    Wraps the seed vector and per-layer controls. The forward pass:
    1. Expands phi -> flat weight vector (fp32)
    2. Slices into weight tensors
    3. Casts to bf16 for the transformer forward pass
    4. Runs stateless transformer forward
    """
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)

        # Build weight layout
        self.layout = build_weight_layout(
            args.num_layers, args.model_dim, args.num_heads,
            args.num_kv_heads, args.mlp_mult,
        )
        self.full_model_dim = compute_full_model_dim(self.layout)

        # === TRAINED PARAMETERS ===

        # 1. Seed vector phi
        self.phi = nn.Parameter(torch.randn(args.seed_dim) * 0.01)

        # 2. Token embedding
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        if args.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=args.tied_embed_init_std)

        # 3. Per-layer scalar controls
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, args.model_dim, dtype=torch.float32)
        )
        # Per-block scalars
        for i in range(args.num_layers):
            self.register_parameter(
                f"attn_scale_{i}",
                nn.Parameter(torch.ones(args.model_dim, dtype=torch.float32)),
            )
            self.register_parameter(
                f"mlp_scale_{i}",
                nn.Parameter(torch.ones(args.model_dim, dtype=torch.float32)),
            )
            self.register_parameter(
                f"q_gain_{i}",
                nn.Parameter(torch.full((args.num_heads,), args.qk_gain_init, dtype=torch.float32)),
            )
            self.register_parameter(
                f"resid_mix_{i}",
                nn.Parameter(torch.stack((torch.ones(args.model_dim), torch.zeros(args.model_dim))).float()),
            )

        # Rotary embedding (not trained, just cached)
        head_dim = args.model_dim // args.num_heads
        self.rotary = Rotary(head_dim, base=args.rope_base)

    def _build_per_layer_params(self):
        """Collect per-layer scalar parameters into a dict for stateless forward."""
        params = {}
        params["skip_weights"] = self.skip_weights
        for i in range(self.args.num_layers):
            params[f"blocks.{i}.attn_scale"] = getattr(self, f"attn_scale_{i}")
            params[f"blocks.{i}.mlp_scale"] = getattr(self, f"mlp_scale_{i}")
            params[f"blocks.{i}.attn.q_gain"] = getattr(self, f"q_gain_{i}")
            params[f"blocks.{i}.resid_mix"] = getattr(self, f"resid_mix_{i}")
        return params

    def forward(self, input_ids, target_ids):
        # 1. Expand seed -> flat weight vector (fp32)
        theta_flat = expand_seed_chunked(
            self.phi, self.layout, self.args.prng_seed,
            self.args.chunk_rows, self.phi.device,
        )

        # 2. Slice into named weight tensors
        weight_dict = slice_weights_from_flat(theta_flat, self.layout)

        # 3. Cast weights to bf16 for the forward pass
        weight_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in weight_dict.items()}

        # 4. Collect per-layer params
        per_layer_params = self._build_per_layer_params()

        # 5. Run stateless transformer forward
        return stateless_forward(
            input_ids=input_ids,
            target_ids=target_ids,
            weight_dict=weight_dict_bf16,
            tok_emb_weight=self.tok_emb.weight,
            per_layer_params=per_layer_params,
            rotary=self.rotary,
            num_layers=self.args.num_layers,
            num_heads=self.args.num_heads,
            num_kv_heads=self.args.num_kv_heads,
            model_dim=self.args.model_dim,
            mlp_mult=self.args.mlp_mult,
            logit_softcap=self.args.logit_softcap,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            num_skip_weights=self.num_skip_weights,
        )

    def expanded_state_dict(self):
        """
        Return a full state dict with expanded weights for quantization/serialization.
        This produces the same format as the baseline GPT model's state_dict.
        """
        theta_flat = expand_seed_chunked(
            self.phi, self.layout, self.args.prng_seed,
            self.args.chunk_rows, self.phi.device,
        )
        weight_dict = slice_weights_from_flat(theta_flat, self.layout)

        sd = {}
        # Token embedding
        sd["tok_emb.weight"] = self.tok_emb.weight.data
        # Skip weights
        sd["skip_weights"] = self.skip_weights.data
        # Expanded linear weights
        for name, tensor in weight_dict.items():
            sd[name] = tensor.detach()
        # Per-layer scalars
        for i in range(self.args.num_layers):
            sd[f"blocks.{i}.attn_scale"] = getattr(self, f"attn_scale_{i}").data
            sd[f"blocks.{i}.mlp_scale"] = getattr(self, f"mlp_scale_{i}").data
            sd[f"blocks.{i}.attn.q_gain"] = getattr(self, f"q_gain_{i}").data
            sd[f"blocks.{i}.resid_mix"] = getattr(self, f"resid_mix_{i}").data
        return sd


# --- VALIDATION ---

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
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


# --- MAIN TRAINING ---

def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

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

    def log0(msg, console=True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(f"=== SEED MODEL EXPERIMENT === seed_dim={args.seed_dim} prng_seed={args.prng_seed}")
    log0(code, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")

    # Build model
    base_model = SeedModel(args).to(device)

    # Count parameters
    layout = build_weight_layout(args.num_layers, args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult)
    full_dim = compute_full_model_dim(layout)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} seed_dim:{args.seed_dim} full_model_dim:{full_dim} expansion_ratio:{full_dim / args.seed_dim:.1f}x")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")

    # No torch.compile for seed model -- the chunked expansion is not compile-friendly
    model = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    # Optimizer: Adam for everything (seed uses aggressive LR)
    seed_params = [base_model.phi]
    embed_params = [base_model.tok_emb.weight]
    scalar_params = []
    for name, p in base_model.named_parameters():
        if name in ("phi", "tok_emb.weight"):
            continue
        if "rotary" in name or "inv_freq" in name:
            continue
        scalar_params.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_seed = torch.optim.Adam(
        [{"params": seed_params, "lr": args.seed_lr, "base_lr": args.seed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
    )
    optimizer_tok = torch.optim.Adam(
        [{"params": embed_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_seed, optimizer_tok, optimizer_scalar]

    log0(f"seed_lr:{args.seed_lr} embed_lr:{token_lr} scalar_lr:{args.scalar_lr}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
         f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
         f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")
    log0(f"seed:{args.seed}")

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
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup (prime CUDA kernels, then restore state)
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Main training loop
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
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                         val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
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
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

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
        should_log_train = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0)
        if should_log_train:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # --- SERIALIZATION + ROUNDTRIP VALIDATION ---
    # Export the expanded weights (not the seed) for quantization compatibility
    if master_process:
        expanded_sd = base_model.expanded_state_dict()
        code_bytes = len(code.encode("utf-8"))
        log0(f"Code size: {code_bytes} bytes")

        quant_obj, quant_stats = quantize_state_dict_int8(expanded_sd)
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
        with open("final_model_seed.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model_seed.int8.ptz")
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    # Roundtrip validation: load quantized weights back into the expanded state dict
    # and run validation to measure quantization degradation
    if distributed:
        dist.barrier()

    if master_process:
        with open("final_model_seed.int8.ptz", "rb") as f:
            quant_blob_disk = f.read()
        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
        roundtrip_sd = dequantize_state_dict_int8(quant_state)

        # Build a standalone model for roundtrip eval using the expanded weights
        # We create a fresh SeedModel but override its forward to use fixed weights
        class ExpandedModel(nn.Module):
            """Minimal wrapper for roundtrip validation with expanded weights."""
            def __init__(self, sd, args_):
                super().__init__()
                self.sd = {k: v.to(device) for k, v in sd.items()}
                self.args_ = args_
                self.num_encoder_layers = args_.num_layers // 2
                self.num_decoder_layers = args_.num_layers - self.num_encoder_layers
                self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
                head_dim = args_.model_dim // args_.num_heads
                self.rotary = Rotary(head_dim, base=args_.rope_base).to(device)

            def forward(self, input_ids, target_ids):
                sd = self.sd
                # Separate weight dict and per-layer params
                weight_dict = {}
                per_layer_params = {}
                per_layer_params["skip_weights"] = sd["skip_weights"]
                for k, v in sd.items():
                    if any(k.startswith(f"blocks.{i}.attn.c_") or k.startswith(f"blocks.{i}.attn.proj.") or
                           k.startswith(f"blocks.{i}.mlp.") for i in range(self.args_.num_layers)):
                        weight_dict[k] = v.to(torch.bfloat16)
                    elif k.startswith("blocks.") and "attn_scale" in k:
                        per_layer_params[k] = v
                    elif k.startswith("blocks.") and "mlp_scale" in k:
                        per_layer_params[k] = v
                    elif k.startswith("blocks.") and "q_gain" in k:
                        per_layer_params[k] = v
                    elif k.startswith("blocks.") and "resid_mix" in k:
                        per_layer_params[k] = v

                return stateless_forward(
                    input_ids=input_ids,
                    target_ids=target_ids,
                    weight_dict=weight_dict,
                    tok_emb_weight=sd["tok_emb.weight"].to(torch.bfloat16),
                    per_layer_params=per_layer_params,
                    rotary=self.rotary,
                    num_layers=self.args_.num_layers,
                    num_heads=self.args_.num_heads,
                    num_kv_heads=self.args_.num_kv_heads,
                    model_dim=self.args_.model_dim,
                    mlp_mult=self.args_.mlp_mult,
                    logit_softcap=self.args_.logit_softcap,
                    num_encoder_layers=self.num_encoder_layers,
                    num_decoder_layers=self.num_decoder_layers,
                    num_skip_weights=self.num_skip_weights,
                )

        roundtrip_model = ExpandedModel(roundtrip_sd, args).to(device)
        torch.cuda.synchronize()
        t_qeval = time.perf_counter()
        q_val_loss, q_val_bpb = eval_val(
            args, roundtrip_model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
             f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
