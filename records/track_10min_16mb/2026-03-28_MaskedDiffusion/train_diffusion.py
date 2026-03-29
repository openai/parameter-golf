"""
train_diffusion.py — Discrete Masked Diffusion Language Model

Key insight: discrete masking (like BERT) + causal attention = fully compatible
with the standard autoregressive eval harness and TTT LoRA pipeline.

Training:  randomly mask tokens at varying rates, predict originals (CE on masked positions)
Inference: standard GPT forward pass — NO masking needed, logits are autoregressive P(next token)
TTT LoRA:  identical to GPT — fine-tune LoRA on doc, eval with causal next-token prediction

Adaptive denoising (for generation, not eval):
  Start: all tokens = [MASK]
  Each step: model predicts all masked positions, unmask top-confidence ones
  Stop: when no [MASK] tokens remain (naturally adaptive, no extra head needed)

The "nearest token" problem from continuous diffusion is gone:
  softmax over vocab always gives a valid token distribution.

Usage:
  MASK_RATE=0.5 SEED=1337 torchrun --standalone --nproc_per_node=8 train_diffusion.py
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
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


# ============================================================
# HYPERPARAMETERS
# ============================================================

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
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model architecture (same as SOTA baseline)
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))

    # Optimizer (same as SOTA)
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))

    # Discrete diffusion masking
    # MASK_RATE_MIN/MAX: uniform range for mask rate during training.
    # Each batch element gets its own rate ∈ [min, max].
    # 0.0 = no masking (pure GPT), 1.0 = all masked.
    mask_rate_min = float(os.environ.get("MASK_RATE_MIN", "0.15"))
    mask_rate_max = float(os.environ.get("MASK_RATE_MAX", "0.85"))

    # USE_MASK_LOSS_ONLY: if 1, CE loss only on masked tokens (diffusion objective).
    # If 0, CE loss on ALL tokens like GPT (standard next-token loss).
    # Default 0: adds a second "free" loss signal on all positions.
    use_mask_loss_only = bool(int(os.environ.get("USE_MASK_LOSS_ONLY", "0")))

    # MASK_WEIGHT: weight for masked-position loss vs unmasked-position loss.
    # Only used when use_mask_loss_only=0.
    mask_weight = float(os.environ.get("MASK_WEIGHT", "2.0"))

    # USE_BIDIRECTIONAL_TRAIN: if 1, use full (non-causal) attention during training.
    # Each [MASK] token attends to ALL other tokens including future ones.
    # Better training signal for masked prediction at the cost of train/eval mismatch.
    use_bidirectional_train = bool(int(os.environ.get("USE_BIDIRECTIONAL_TRAIN", "0")))

    # USE_MASKED_EVAL: if 1, evaluate using pseudo-log-likelihood (multi-mask bidirectional).
    # For each sequence, runs EVAL_MASK_PASSES forward passes with random masks,
    # predicts masked tokens with bidirectional attention, averages CE over masked positions.
    # Better matches how the model was trained (bidirectional + masking).
    # If 0, uses standard next-token causal eval (identical to GPT).
    use_masked_eval = bool(int(os.environ.get("USE_MASKED_EVAL", "0")))

    # EVAL_MASK_PASSES: number of random masked versions per sequence in masked eval.
    # More passes = better estimate of true BPB, but slower.
    eval_mask_passes = int(os.environ.get("EVAL_MASK_PASSES", "8"))

    # EVAL_MASK_RATE: fraction of tokens to mask per pass in masked eval.
    # 0.5 = each token is masked ~50% of passes → good coverage with few passes.
    eval_mask_rate = float(os.environ.get("EVAL_MASK_RATE", "0.5"))

    # TTT LoRA hyperparameters (test-time training at eval)
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", "8"))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", "0.01"))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", "256"))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", "1024"))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", "64"))
    use_ttt_eval = bool(int(os.environ.get("USE_TTT_EVAL", "0")))

    # MIX_GPT_PROB: probability per training step of running as pure causal GPT
    # (no masking, causal attention, standard next-token loss) vs diffusion mode
    # (bidirectional attention, masked tokens, masked CE loss).
    # 0.5 = 50% GPT steps + 50% diffusion steps.
    # 0.0 = pure diffusion (original behavior).
    # Reduces train/eval mismatch: model learns both causal and bilateral prediction.
    mix_gpt_prob = float(os.environ.get("MIX_GPT_PROB", "0.0"))


# ============================================================
# MUON OPTIMIZER (same as train_gpt.py)
# ============================================================

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
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, momentum, nesterov, backend_steps = (
                group["lr"], group["momentum"], group["nesterov"], group["backend_steps"],
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf
                if g.ndim == 2:
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)


# ============================================================
# MASKING SCHEDULE
# ============================================================

def sample_mask(tokens: Tensor, mask_id: int, rate_min: float, rate_max: float) -> tuple[Tensor, Tensor]:
    """
    Randomly mask tokens at a per-element rate sampled from [rate_min, rate_max].

    Returns:
        x_masked: token IDs with some replaced by mask_id
        mask:     boolean tensor, True where tokens were masked
    """
    B, T = tokens.shape
    # Per-element mask rate
    rates = torch.empty(B, device=tokens.device).uniform_(rate_min, rate_max)
    # Per-token mask decision: mask[i,j] = True with probability rates[i]
    mask = torch.rand(B, T, device=tokens.device) < rates[:, None]
    x_masked = tokens.clone()
    x_masked[mask] = mask_id
    return x_masked, mask


# ============================================================
# QUANTIZATION (same as train_gpt.py)
# ============================================================

INT8_CLIP_Q = 99.99984 / 100.0
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = t32.clamp(-clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = clipped.div(scale[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
        return q, scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs_val = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs_val / 127.0 if clip_abs_val > 0 else 1.0, dtype=torch.float32)
    q = t32.clamp(-clip_abs_val, clip_abs_val).div(scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]) -> tuple[dict, dict]:
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += t.numel() * t.element_size()
            continue
        is_control = any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or is_control:
            if t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                passthrough[name] = t.to(INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
            else:
                passthrough[name] = t
            stats["int8_payload_bytes"] += passthrough[name].numel() * passthrough[name].element_size()
            continue
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += q.numel() + s.numel() * s.element_size()
    obj: dict = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized,
                 "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.to(torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# ============================================================
# DATA LOADING (same as train_gpt.py)
# ============================================================

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No data files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            take = min(avail, remaining)
            chunks.append(self.tokens[self.pos: self.pos + take])
            self.pos += take
            remaining -= take
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.stream = TokenStream(pattern)
        self.device = device
        self.world_size = world_size

    def next_batch(self, total_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        tokens_per_step = total_tokens // self.world_size // grad_accum_steps
        raw = self.stream.take(tokens_per_step + 1)
        x = raw[:-1].view(-1, seq_len).long()
        y = raw[1:].view(-1, seq_len).long()
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def load_validation_tokens(val_files: str, seq_len: int) -> Tensor:
    chunks = [load_data_shard(Path(p)) for p in sorted(glob.glob(val_files))]
    if not chunks:
        raise FileNotFoundError(f"No val files: {val_files}")
    tokens = torch.cat(chunks)
    n = (tokens.numel() - 1) // seq_len * seq_len
    return tokens[: n + 1]


# ============================================================
# MODEL (same transformer backbone as train_gpt.py)
# ============================================================

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
        self.register_buffer("inv_freq", (1.0 / base ** (torch.arange(0, dim, 2).float() / dim)).float())

    def forward(self, seqlen: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, is_causal: bool = True, q_lora=None, v_lora=None) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x) + (q_lora(x) if q_lora is not None else 0)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x) + (v_lora(x) if v_lora is not None else 0)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary(q, cos, sin) * self.q_gain.to(q.dtype)[None, :, None, None]
        k = apply_rotary(k, cos, sin)
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        self.fc = CastedLinear(dim, mlp_mult * dim, bias=False)
        self.proj = CastedLinear(mlp_mult * dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.attn = Attention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp_norm = RMSNorm()
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x: Tensor, is_causal: bool = True, q_lora=None, v_lora=None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), is_causal=is_causal, q_lora=q_lora, v_lora=v_lora)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MaskedDiffusionLM(nn.Module):
    """
    Discrete Masked Diffusion Language Model.

    Identical backbone to train_gpt.py but trained with masked token prediction.
    The MASK token (id = vocab_size) is an extra embedding that acts as a learnable
    "unknown token" signal. At eval time, no masking is needed — forward pass is
    identical to a standard causal LM.

    vocab_size + 1 embeddings: [0..vocab_size-1] = real tokens, [vocab_size] = MASK
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        logit_softcap: float,
        qk_gain_init: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size           # MASK token = one beyond real vocab
        self.logit_softcap = logit_softcap
        self.tie_embeddings = tie_embeddings

        # vocab_size + 1 to include the MASK token embedding
        self.tok_emb = nn.Embedding(vocab_size + 1, model_dim)
        self.norm = RMSNorm()
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()

        if tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = CastedLinear(model_dim, vocab_size, bias=False)
            self.lm_head._zero_init = True

        self._init_weights(tied_embed_init_std)

    def _init_weights(self, tied_embed_init_std: float):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, CastedLinear):
                if getattr(m, "_zero_init", False):
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: Tensor, is_causal: bool = True, lora=None) -> Tensor:
        """
        Forward pass. input_ids may contain MASK tokens (mask_id) during training.

        is_causal=True:  causal attention — standard next-token prediction (eval mode)
        is_causal=False: bidirectional attention — each token sees all others (train/masked-eval mode)
        lora:            optional BatchedTTTLoRADiffusion for test-time training

        Returns logits (B, T, vocab_size) — only over real vocab, not MASK.
        """
        x = self.tok_emb(input_ids)
        x = self.norm(x)
        for i, block in enumerate(self.blocks):
            q_lora = lora.q_loras[i] if lora is not None else None
            v_lora = lora.v_loras[i] if lora is not None else None
            x = block(x, is_causal=is_causal, q_lora=q_lora, v_lora=v_lora)
        x = self.final_norm(x)

        if self.tie_embeddings:
            logits = x @ self.tok_emb.weight[:self.vocab_size].to(x.dtype).T
        else:
            logits = self.lm_head(x)

        if lora is not None:
            logits = logits + lora.lm_head_lora(x)

        if self.logit_softcap > 0:
            logits = torch.tanh(logits / self.logit_softcap) * self.logit_softcap
        return logits

    def loss(self, input_ids: Tensor, targets: Tensor, mask: Tensor,
             use_mask_loss_only: bool, mask_weight: float, is_causal: bool = True) -> Tensor:
        """
        Compute training loss.

        input_ids: (B, T) tokens, some replaced with mask_id
        targets:   (B, T) original tokens (always real, no MASK)
        mask:      (B, T) bool, True where tokens were masked
        use_mask_loss_only: if True, loss only on masked positions
        mask_weight: extra weight on masked positions when use_mask_loss_only=False
        is_causal: if False, use bidirectional attention (better for masked prediction)
        """
        logits = self.forward(input_ids, is_causal=is_causal)  # (B, T, V)
        V = self.vocab_size

        if use_mask_loss_only:
            # Pure diffusion objective: CE only on masked positions
            if mask.any():
                return F.cross_entropy(logits[mask].float(), targets[mask])
            return logits.sum() * 0.0  # empty batch edge case

        # Hybrid: CE on all positions + extra weight on masked positions
        # This gives both a strong diffusion signal AND standard GPT signal
        ce_all = F.cross_entropy(logits.view(-1, V).float(), targets.view(-1), reduction="none")
        ce_all = ce_all.view_as(targets)
        weights = torch.ones_like(ce_all)
        weights[mask] = mask_weight
        return (ce_all * weights).mean()


# ============================================================
# TEST-TIME TRAINING (LoRA)
# ============================================================

BOS_ID = 1

class BatchedLinearLoRA(nn.Module):
    """LoRA delta for one linear layer, independent weights per batch element."""
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()


class BatchedTTTLoRADiffusion(nn.Module):
    """LoRA adapters for Q, V projections and LM head — one set per batch element."""
    def __init__(self, bsz: int, model: "MaskedDiffusionLM", rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.vocab_size
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, block.attn.c_q.weight.shape[0], rank)
             for block in model.blocks]
        )
        self.v_loras = nn.ModuleList(
            [BatchedLinearLoRA(bsz, dim, block.attn.c_v.weight.shape[0], rank)
             for block in model.blocks]
        )

    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()


def _reset_ttt_optimizer(opt: torch.optim.Optimizer) -> None:
    for group in opt.param_groups:
        for p in group["params"]:
            s = opt.state.get(p)
            if not s:
                continue
            s["exp_avg"].zero_()
            s["exp_avg_sq"].zero_()
            s["step"].fill_(0)


def _find_docs(all_tokens: Tensor) -> list[tuple[int, int]]:
    """Return (start_offset, length) for each document identified by BOS boundaries."""
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else all_tokens.numel()
        if i + 1 < len(bos_positions):
            end += 1  # include next BOS
        if end - start >= 2:
            docs.append((start, end - start))
    return docs


def _compute_chunk_window(ci: int, pred_len: int, num_chunks: int, chunk_size: int, eval_seq_len: int):
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def eval_val_ttt_lora(
    args: "Hyperparameters",
    base_model: "MaskedDiffusionLM",
    rank: int,
    world_size: int,
    device: torch.device,
    base_bytes_lut: Tensor,
) -> tuple[float, float]:
    """
    TTT LoRA eval for the diffusion model.

    If args.use_masked_eval=False: causal next-token prediction (like GPT).
    If args.use_masked_eval=True:  bidirectional masked scoring (true diffusion).
      - Score each chunk via pseudo-log-likelihood (eval_mask_passes masked versions)
      - Train LoRA on masked prediction with bidirectional attention
      - Coherent with how the model was trained
    """
    files = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _find_docs(all_tokens)

    rank_docs = docs[(len(docs) * rank) // world_size: (len(docs) * (rank + 1)) // world_size]
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len
    batch_size = args.ttt_batch_size
    lora_rank = args.ttt_lora_rank

    rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    lora = BatchedTTTLoRADiffusion(batch_size, base_model, lora_rank).to(device)
    opt = torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr,
                           betas=(args.beta1, args.beta2), eps=1e-10)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    for bi in range(0, len(rank_docs), batch_size):
        batch = rank_docs[bi: bi + batch_size]
        bsz = len(batch)

        if bsz == batch_size:
            cur_lora, cur_opt = lora, opt
            cur_lora.reset()
            _reset_ttt_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRADiffusion(bsz, base_model, lora_rank).to(device)
            cur_opt = torch.optim.Adam(cur_lora.parameters(), lr=args.ttt_lora_lr,
                                       betas=(args.beta1, args.beta2), eps=1e-10)

        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)

        for ci in range(max_nc):
            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc - 1 for nc in num_chunks)

            # Build context window for this chunk
            win_lens, doc_info = [], []
            for b in range(bsz):
                if not active[b]:
                    doc_info.append((0, 0))
                    win_lens.append(0)
                    continue
                _, wl, co, cl = _compute_chunk_window(
                    ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                doc_info.append((co, cl))
                win_lens.append(wl)

            max_win = max(win_lens) if win_lens else 0
            if max_win == 0:
                continue

            x = torch.zeros(bsz, max_win, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, max_win, dtype=torch.int64, device=device)
            for b in range(bsz):
                if not active[b]:
                    continue
                ds, _ = batch[b]
                ws, wl, _, _ = _compute_chunk_window(
                    ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                toks = all_tokens[ds + ws: ds + ws + wl + 1].to(dtype=torch.int64, device=device)
                x[b, :wl] = toks[:-1]
                y[b, :wl] = toks[1:]

            if not args.use_masked_eval:
                # ── Causal mode (standard next-token) ──────────────────────────
                if needs_train:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = base_model(x, is_causal=True, lora=cur_lora)
                else:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = base_model(x, is_causal=True, lora=cur_lora)

                with torch.no_grad():
                    ce = F.cross_entropy(
                        logits.float().view(-1, args.vocab_size), y.view(-1), reduction="none",
                    ).view(bsz, max_win)
                    for b in range(bsz):
                        if not active[b]:
                            continue
                        co, cl = doc_info[b]
                        loss_sum += ce[b, co: co + cl].to(torch.float64).sum()
                        byte_sum += base_bytes_lut[y[b, co: co + cl].cpu()].to(torch.float64).sum().to(device)
                        token_count += cl

                if needs_train:
                    chunk_losses = []
                    for b in range(bsz):
                        co, cl = doc_info[b]
                        if cl > 0 and ci < num_chunks[b] - 1:
                            chunk_losses.append(F.cross_entropy(logits[b, co: co + cl].float(), y[b, co: co + cl]))
                    if chunk_losses:
                        cur_opt.zero_grad()
                        torch.stack(chunk_losses).sum().backward()
                        cur_opt.step()

            else:
                # ── Masked bidirectional mode (true diffusion) ─────────────────
                # Score: pseudo-log-likelihood over eval_mask_passes masked versions
                with torch.no_grad():
                    ce_accum = torch.zeros(bsz, max_win, device=device, dtype=torch.float64)
                    cnt_accum = torch.zeros(bsz, max_win, device=device, dtype=torch.float64)
                    for _ in range(args.eval_mask_passes):
                        pmask = torch.rand(bsz, max_win, device=device) < args.eval_mask_rate
                        x_m = x.clone()
                        x_m[pmask] = base_model.mask_id
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            lg = base_model(x_m, is_causal=False, lora=cur_lora)
                        ce_p = F.cross_entropy(
                            lg.float().view(-1, args.vocab_size), x.view(-1), reduction="none",
                        ).view(bsz, max_win)
                        ce_accum += torch.where(pmask, ce_p.double(), torch.zeros_like(ce_p, dtype=torch.float64))
                        cnt_accum += pmask.double()
                    safe_cnt = cnt_accum.clamp(min=1)
                    ce_mean = (ce_accum / safe_cnt).float()  # (bsz, max_win)

                for b in range(bsz):
                    if not active[b]:
                        continue
                    co, cl = doc_info[b]
                    loss_sum += ce_mean[b, co: co + cl].to(torch.float64).sum()
                    byte_sum += base_bytes_lut[x[b, co: co + cl].cpu()].to(torch.float64).sum().to(device)
                    token_count += cl

                # Train LoRA on masked prediction (bidirectional)
                # Use the full batch to match LoRA batch size
                if needs_train:
                    active_mask = torch.zeros(bsz, max_win, dtype=torch.bool, device=device)
                    for b in range(bsz):
                        co, cl = doc_info[b]
                        if cl > 0 and ci < num_chunks[b] - 1:
                            active_mask[b, co: co + cl] = True
                    if active_mask.any():
                        pmask = torch.rand(bsz, max_win, device=device) < args.mask_rate_max
                        x_m = x.clone()
                        x_m[pmask] = base_model.mask_id
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            lg = base_model(x_m, is_causal=False, lora=cur_lora)
                        combined = pmask & active_mask
                        if combined.any():
                            loss = F.cross_entropy(lg[combined].float(), x[combined])
                            cur_opt.zero_grad()
                            loss.backward()
                            cur_opt.step()

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.train()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    val_loss = float(loss_sum.item() / max(token_count.item(), 1))
    val_bpb = float((loss_sum.item() / math.log(2.0)) / max(byte_sum.item(), 1))
    return val_loss, val_bpb


def compute_masked_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    use_mask_loss_only: bool,
    mask_weight: float,
    vocab_size: int,
) -> Tensor:
    """
    Standalone loss function for use with compiled model in training loop.
    Identical logic to MaskedDiffusionLM.loss() but takes pre-computed logits.
    """
    V = vocab_size
    if use_mask_loss_only:
        if mask.any():
            return F.cross_entropy(logits[mask].float(), targets[mask])
        return logits.sum() * 0.0
    ce_all = F.cross_entropy(logits.view(-1, V).float(), targets.view(-1), reduction="none")
    ce_all = ce_all.view_as(targets)
    weights = torch.ones_like(ce_all)
    weights[mask] = mask_weight
    return (ce_all * weights).mean()


# ============================================================
# EVALUATION
# ============================================================

def eval_val(
    model: MaskedDiffusionLM,
    val_tokens: Tensor,
    args: Hyperparameters,
    base_bytes_lut: Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    grad_accum_steps: int,
) -> tuple[float, float]:
    """Standard next-token BPB eval with causal attention. Identical to GPT eval."""
    model.eval()
    seq_len = args.train_seq_len
    max_seq = args.val_batch_size // seq_len
    n_seq = min((val_tokens.numel() - 1) // seq_len, max_seq)

    val_loss_sum = torch.zeros((), device=device)
    val_token_count = torch.zeros((), device=device, dtype=torch.long)
    val_byte_count = torch.zeros((), device=device, dtype=torch.long)

    # Distribute across ranks
    seq_per_rank = (n_seq + world_size - 1) // world_size
    start = rank * seq_per_rank
    end = min(start + seq_per_rank, n_seq)

    with torch.no_grad():
        for i in range(start, end):
            x = val_tokens[i * seq_len: (i + 1) * seq_len].long().unsqueeze(0).to(device)
            y = val_tokens[i * seq_len + 1: (i + 1) * seq_len + 1].long().unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(logits.view(-1, args.vocab_size).float(), y.view(-1), reduction="sum")
            val_loss_sum += loss
            val_token_count += y.numel()
            val_byte_count += base_bytes_lut[y.view(-1).cpu()].sum().to(device)

    if world_size > 1:
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = (val_loss_sum / val_token_count.float()).item()
    bpb = (val_loss / math.log(2.0)) * (val_token_count.item() / max(val_byte_count.item(), 1))
    model.train()
    return float(val_loss), float(bpb)


def eval_val_masked(
    model: MaskedDiffusionLM,
    val_tokens: Tensor,
    args: Hyperparameters,
    base_bytes_lut: Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    grad_accum_steps: int,
) -> tuple[float, float]:
    """
    Pseudo-log-likelihood BPB eval using multi-mask bidirectional attention.

    For each validation sequence, runs args.eval_mask_passes forward passes.
    Each pass randomly masks ~eval_mask_rate of tokens and predicts them
    using bidirectional attention (full context, left AND right).

    This matches how the model was trained when use_bidirectional_train=True,
    and gives each token access to its full bilateral context for prediction.

    BPB is computed over the masked positions only — an unbiased estimator
    of the true per-token CE under bilateral context.
    """
    model.eval()
    seq_len = args.train_seq_len
    max_seq = args.val_batch_size // seq_len
    n_seq = min((val_tokens.numel() - 1) // seq_len, max_seq)

    val_loss_sum = torch.zeros((), device=device)
    val_token_count = torch.zeros((), device=device, dtype=torch.long)
    val_byte_count = torch.zeros((), device=device, dtype=torch.long)

    seq_per_rank = (n_seq + world_size - 1) // world_size
    start = rank * seq_per_rank
    end = min(start + seq_per_rank, n_seq)

    with torch.no_grad():
        for i in range(start, end):
            x = val_tokens[i * seq_len: (i + 1) * seq_len].long().to(device)  # (T,)

            # Batch eval_mask_passes masked versions together: (passes, T)
            x_batch = x.unsqueeze(0).expand(args.eval_mask_passes, -1).contiguous()
            mask = torch.rand(args.eval_mask_passes, seq_len, device=device) < args.eval_mask_rate
            x_masked = x_batch.clone()
            x_masked[mask] = model.mask_id

            if not mask.any():
                continue

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Bidirectional: each [MASK] sees all tokens including future ones
                logits = model(x_masked, is_causal=False)  # (passes, T, V)

            # CE only on masked positions (bilateral prediction)
            ce = F.cross_entropy(
                logits[mask].float(),
                x_batch[mask],
                reduction="sum",
            )
            val_loss_sum += ce
            val_token_count += mask.sum()
            val_byte_count += base_bytes_lut[x_batch[mask].cpu()].sum().to(device)

    if world_size > 1:
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = (val_loss_sum / val_token_count.float()).item()
    bpb = (val_loss / math.log(2.0)) * (val_token_count.item() / max(val_byte_count.item(), 1))
    model.train()
    return float(val_loss), float(bpb)


# ============================================================
# MAIN
# ============================================================

def main():
    distributed = int(os.environ.get("RANK", -1)) != -1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    master_process = rank == 0
    device = torch.device("cuda", local_rank)

    def log0(msg: str):
        if master_process:
            print(msg, flush=True)

    args = Hyperparameters()
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Tokenizer for BPB
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    base_bytes_lut = torch.zeros(args.vocab_size, dtype=torch.int32)
    for i in range(args.vocab_size):
        base_bytes_lut[i] = len(sp.IdToPiece(i).replace("\u2581", " ").encode("utf-8"))
    base_bytes_lut = base_bytes_lut.to(device)

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len).to(device)
    log0(f"val_tokens:{val_tokens.numel()} val_sequences:{(val_tokens.numel()-1)//args.train_seq_len}")

    total_batch = args.train_batch_tokens
    seqs_per_step = total_batch // world_size // args.train_seq_len
    grad_accum_steps = max(1, seqs_per_step // 32)
    micro_seqs = seqs_per_step // grad_accum_steps
    log0(f"grad_accum:{grad_accum_steps} micro_seqs:{micro_seqs}")

    base_model = MaskedDiffusionLM(
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        rope_base=args.rope_base,
        logit_softcap=args.logit_softcap,
        qk_gain_init=args.qk_gain_init,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
    ).to(device).bfloat16()

    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
        if isinstance(m, Rotary):
            m.inv_freq.data = m.inv_freq.data.float()
    # q_gain and norms stay fp32
    for name, p in base_model.named_parameters():
        if p.ndim < 2 and p.dtype != torch.float32:
            p.data = p.data.float()

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else compiled_model
    )

    # Optimizer — same split as train_gpt.py
    block_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for _, p in block_params if p.ndim == 2]
    other_params = [p for n, p in base_model.named_parameters()
                    if not any(p is bp for _, bp in block_params) and p.ndim < 2]

    if args.tie_embeddings:
        token_lr = args.tied_embed_lr
        embed_params = [base_model.tok_emb.weight]
        extra_params: list[nn.Parameter] = []
    else:
        token_lr = args.embed_lr
        embed_params = [base_model.tok_emb.weight]
        extra_params = list(base_model.lm_head.parameters()) if base_model.lm_head else []

    optimizer_muon = Muon(
        [{"params": matrix_params, "base_lr": args.matrix_lr}],
        lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps,
    )
    optimizer_adam = torch.optim.Adam(
        [
            {"params": embed_params, "base_lr": token_lr, "lr": token_lr},
            {"params": other_params + extra_params, "base_lr": args.scalar_lr, "lr": args.scalar_lr},
        ],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = group["lr"]
    optimizers = [optimizer_muon, optimizer_adam]

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                if ws <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    log0(f"MaskedDiffusionLM mask_rate:[{args.mask_rate_min},{args.mask_rate_max}] "
         f"mask_loss_only:{args.use_mask_loss_only} mask_weight:{args.mask_weight} "
         f"bidirectional_train:{args.use_bidirectional_train} "
         f"masked_eval:{args.use_masked_eval} eval_mask_passes:{args.eval_mask_passes} "
         f"eval_mask_rate:{args.eval_mask_rate}")
    log0(f"vocab_size:{args.vocab_size}+1(MASK) num_layers:{args.num_layers} "
         f"model_dim:{args.model_dim} tie_embeddings:{args.tie_embeddings}")

    # Warmup
    if args.warmup_steps > 0:
        init_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opts = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for _ in range(args.warmup_steps):
            zero_grad_all()
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            x_masked, mask = sample_mask(x, base_model.mask_id, args.mask_rate_min, args.mask_rate_max)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = base_model.loss(
                    x_masked, x, mask, args.use_mask_loss_only, args.mask_weight,
                    is_causal=not args.use_bidirectional_train,
                )
            loss.backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(init_state, strict=True)
        for opt, state in zip(optimizers, init_opts):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Training loop
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            if args.use_masked_eval:
                val_loss, val_bpb = eval_val_masked(
                    base_model, val_tokens, args, base_bytes_lut, device, rank, world_size, grad_accum_steps,
                )
            else:
                val_loss, val_bpb = eval_val(
                    base_model, val_tokens, args, base_bytes_lut, device, rank, world_size, grad_accum_steps,
                )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()

        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            # Apply random masking to input tokens
            if args.mix_gpt_prob > 0 and torch.rand(1).item() < args.mix_gpt_prob:
                # GPT mode: causal next-token prediction (no masking, no mismatch)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x, is_causal=True)
                loss = F.cross_entropy(
                    logits.view(-1, args.vocab_size).float(), y.view(-1),
                ) / grad_accum_steps
            else:
                # Diffusion mode: bidirectional masked prediction
                x_masked, mask = sample_mask(x, base_model.mask_id, args.mask_rate_min, args.mask_rate_max)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x_masked, is_causal=not args.use_bidirectional_train)
                loss = compute_masked_loss(
                    logits, x, mask, args.use_mask_loss_only, args.mask_weight, args.vocab_size,
                ) / grad_accum_steps
            loss.backward()
            train_loss += loss.detach()

        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_mom
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            cap_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    # Export int8+zlib
    if master_process:
        torch.save(base_model.state_dict(), "final_diffusion.pt")
    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    if master_process:
        with open("final_diffusion.int8.ptz", "wb") as f:
            f.write(blob)
        code_bytes = len(open(__file__).read().encode("utf-8"))
        log0(f"artifact:{len(blob)} bytes ({len(blob)/1e6:.1f}MB) code:{code_bytes} "
             f"total:{len(blob)+code_bytes} bytes")

    if distributed:
        dist.barrier()

    # Roundtrip validation
    quant_state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    if args.use_masked_eval:
        val_loss_q, val_bpb_q = eval_val_masked(
            base_model, val_tokens, args, base_bytes_lut, device, rank, world_size, grad_accum_steps,
        )
    else:
        val_loss_q, val_bpb_q = eval_val(
            base_model, val_tokens, args, base_bytes_lut, device, rank, world_size, grad_accum_steps,
        )
    log0(f"final_int8_roundtrip val_loss:{val_loss_q:.4f} val_bpb:{val_bpb_q:.4f}")

    if args.use_ttt_eval:
        val_loss_ttt, val_bpb_ttt = eval_val_ttt_lora(
            args, base_model, rank, world_size, device, base_bytes_lut,
        )
        log0(f"final_ttt_lora val_loss:{val_loss_ttt:.4f} val_bpb:{val_bpb_ttt:.4f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
