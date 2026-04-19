"""
Nairi-Optimized Model - Beat the Baseline
=========================================
Key improvements:
1. Baseline architecture (9L, 512d, 1024 vocab)
2. Gated skip connections (learnable gates)
3. Early stopping at best val_loss
4. Proper quantization for 16MB limit
"""

from __future__ import annotations

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
    data_path = os.environ.get("DATA_PATH", "./data")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH",
        "./records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/fineweb_8192_bpe.model",
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 50))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 20))

    iterations = int(os.environ.get("ITERATIONS", 500))
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

    embed_lr = float(os.environ.get("EMBED_LR", 0.01))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.01))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.01))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.01))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.1))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------
def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
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
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov
            ),
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
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )

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


# -----------------------------
# TOKENIZER SETUP
# -----------------------------
def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
):
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
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.int64))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


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
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------
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


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
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
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = (
            self.c_q(x)
            .reshape(bsz, seqlen, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.c_k(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.c_v(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x)
        )
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
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        input_ids = input_ids.clamp(0, self.tok_emb.num_embeddings - 1)
        target_ids = target_ids.clamp(0, self.tok_emb.num_embeddings - 1)
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)

        for i in range(self.num_decoder_layers):
            if skips:
                x = (
                    x
                    + self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# VALIDATION
# -----------------------------
def eval_val(
    model,
    rank: int,
    world_size: int,
    device: torch.device,
    args: Hyperparameters,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
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
            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
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
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
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
# QUANTIZATION
# -----------------------------
def quantize_state_dict_int8(state_dict):
    result = {}
    for k, v in state_dict.items():
        if v.dtype == torch.float32 or v.dtype == torch.bfloat16:
            if v.ndim == 2:
                scale = v.abs().max() / 127.0
                result[k] = {
                    "data": (v / scale).to(torch.int8),
                    "scale": scale.to(torch.float32),
                }
            else:
                result[k] = v.float()
        else:
            result[k] = v
    return result


def save_model_compressed(model, filepath, code_path=None):
    state = model.state_dict()
    quantized = quantize_state_dict_int8(state)

    buffer = io.BytesIO()
    torch.save(quantized, buffer)
    model_bytes = buffer.getvalue()

    compressed = zlib.compress(model_bytes, level=9)

    with open(filepath, "wb") as f:
        f.write(compressed)

    return len(compressed)


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
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100)
    log0(f"Running Python {sys.version}")
    log0(f"Running PyTorch {torch.__version__}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_path)
    if sp.vocab_size() != args.vocab_size:
        log0(
            f"WARNING: VOCAB_SIZE={args.vocab_size} != tokenizer vocab_size={sp.vocab_size()}"
        )

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )

    log0(
        f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}"
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:tokens:{val_tokens.numel() - 1}")

    model = (
        GPT(
            args.vocab_size,
            args.num_layers,
            args.model_dim,
            args.num_heads,
            args.num_kv_heads,
            args.mlp_mult,
            args.tie_embeddings,
            args.tied_embed_init_std,
            args.logit_softcap,
            args.rope_base,
            args.qk_gain_init,
        )
        .to(device)
        .bfloat16()
    )

    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.float()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim < 2 and param.dtype != torch.float32:
                param.data = param.data.float()

    model = (
        DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else model
    )

    block_params = (
        list(model.module.blocks.parameters())
        if distributed
        else list(model.blocks.parameters())
    )
    matrix_params = [p for p in block_params if p.ndim == 2]
    scalar_params = [p for p in block_params if p.ndim < 2]
    if hasattr(model.module if distributed else model, "skip_weights"):
        scalar_params.extend(
            [model.module.skip_weights] if distributed else [model.skip_weights]
        )

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [
            {
                "params": [
                    model.module.tok_emb.weight if distributed else model.tok_emb.weight
                ],
                "lr": token_lr,
                "weight_decay": args.weight_decay,
            }
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    # Use AdamW instead of Muon for better seed stability
    optimizer_matrix = torch.optim.AdamW(
        [
            {
                "params": matrix_params,
                "lr": args.matrix_lr,
                "weight_decay": args.weight_decay,
            }
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_scalar = torch.optim.AdamW(
        [
            {
                "params": scalar_params,
                "lr": args.scalar_lr,
                "weight_decay": args.weight_decay,
            }
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers = [optimizer_tok, optimizer_matrix, optimizer_scalar]

    n_params = sum(p.numel() for p in model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} iterations:{args.iterations}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )
    step = 0
    start_time = time.perf_counter()
    zero_grad_all()

    # Early stopping tracking
    best_val_loss = float("inf")
    best_step = 0
    patience = 50

    # Warmup
    if master_process:
        for warmup_step in range(1, args.warmup_steps + 1):
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            for chunk_idx in range(grad_accum_steps):
                xi = x[chunk_idx::grad_accum_steps]
                yi = y[chunk_idx::grad_accum_steps]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(xi, yi)
                (loss / grad_accum_steps).backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for opt in optimizers
                        for group in opt.param_groups
                        for p in group["params"]
                    ],
                    args.grad_clip_norm,
                )
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            log0(f"warmup_step:{warmup_step}/{args.warmup_steps}")

    if distributed:
        dist.barrier()

    while True:
        if max_wallclock_ms is not None:
            elapsed_ms = 1000.0 * (time.perf_counter() - start_time)
            if elapsed_ms >= max_wallclock_ms:
                log0(
                    f"stopping_early: wallclock_cap train_time:{elapsed_ms:.0f}ms step:{step}"
                )
                break

        if step >= args.iterations:
            break

        x, y = train_loader.next_batch(
            args.train_batch_tokens, args.train_seq_len, grad_accum_steps
        )
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        for chunk_idx in range(grad_accum_steps):
            xi = x[chunk_idx::grad_accum_steps]
            yi = y[chunk_idx::grad_accum_steps]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(xi, yi)
            (loss / grad_accum_steps).backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for opt in optimizers
                    for group in opt.param_groups
                    for p in group["params"]
                ],
                args.grad_clip_norm,
            )

        for opt in optimizers:
            opt.step()

        step += 1

        if step % args.train_log_every == 0 and master_process:
            train_time = 1000.0 * (time.perf_counter() - start_time)
            log0(
                f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} train_time:{train_time:.0f}ms"
            )

        if step % args.val_loss_every == 0 and step > 0 and master_process:
            val_loss, val_bpb = eval_val(
                model,
                rank,
                world_size,
                device,
                args,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}"
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                log0(f"*** NEW BEST val_loss:{val_loss:.4f} at step {step} ***")

    if distributed:
        dist.barrier()

    if master_process:
        log0(f"Training completed: {step} steps")
        log0(f"Best val_loss: {best_val_loss:.4f} at step {best_step}")

        # Save model
        model_to_save = model.module if distributed else model
        compressed_bytes = save_model_compressed(
            model_to_save, f"nairi_model_compressed.ptz"
        )
        log0(
            f"Saved compressed model: {compressed_bytes} bytes ({compressed_bytes / 1024 / 1024:.2f} MB)"
        )


if __name__ == "__main__":
    main()
