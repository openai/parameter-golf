"""Parameter Golf research trainer: structured initialization + compressible updates.

A single-file successor to the supplied train_gpt.py, not a leaderboard-tuned
configuration or a claim of rule-compliant performance. The 600-second budget
counts measured training steps, not compilation, warmup, validation, or export.
Verify timing accounting against any particular competition rule set.

Research questions:
  * Does deterministic ZerO initialization make W_final - W_init compressible?
  * Does Muon help optimization but harm low-rank update structure?
  * Are the U-shaped skips / x0 mixing useful at a fixed training budget?

The saved .ptz artifact is self-contained when used with THIS script. It stores
no training dataset, tokenizer, or validation information. It reconstructs its
reference initialization using a stored seed and the model-building algorithm.

Examples (from the openai/parameter-golf checkout):
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  torchrun --standalone --nproc_per_node=8 train_gpt_rewrite.py

  # Ablation against conventional initialization and Adam (no Muon):
  INIT_MODE=standard OPTIMIZER=adam CODEC=weight_int8 \
  torchrun --standalone --nproc_per_node=8 train_gpt_rewrite.py

  # Inspect whether the trained displacement really has low effective rank:
  ANALYZE_SPECTRA=1 CODEC=delta_hybrid python train_gpt_rewrite.py

  # CPU-only functional checks (does not require a FineWeb dataset):
  python train_gpt_rewrite.py --smoke-test
"""

from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import sys
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------


def env_bool(name: str, default: bool) -> bool:
    return bool(int(os.environ.get(name, str(int(default)))))


@dataclass(frozen=True)
class ModelSpec:
    vocab_size: int = 1024
    num_layers: int = 10
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 2
    tie_embeddings: bool = True
    rope_base: float = 10000.0
    qk_gain_init: float = 1.5
    tied_embed_init_std: float = 0.005
    logit_softcap: float = 30.0
    init_mode: str = "zero"  # zero (ZerO) or standard
    use_long_skips: bool = True
    use_resid_mix: bool = True

    def validate(self) -> None:
        if self.init_mode not in {"zero", "standard"}:
            raise ValueError("INIT_MODE must be zero or standard")
        if self.model_dim % self.num_heads:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("NUM_HEADS must be divisible by NUM_KV_HEADS")
        if (self.model_dim // self.num_heads) % 2:
            raise ValueError("Attention head dimension must be even for RoPE")
        if self.num_layers < 1 or self.vocab_size < 2 or self.mlp_mult < 1:
            raise ValueError("Invalid model shape")
        if self.logit_softcap <= 0:
            raise ValueError("LOGIT_SOFTCAP must be positive")


@dataclass(frozen=True)
class Config:
    data_path: str = field(default_factory=lambda: os.getenv("DATA_PATH", "./data/datasets/fineweb10B_sp1024"))
    tokenizer_path: str = field(default_factory=lambda: os.getenv("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"))
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "."))
    run_id: str = field(default_factory=lambda: os.getenv("RUN_ID", uuid.uuid4().hex[:12]))
    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "1337")))
    vocab_size: int = field(default_factory=lambda: int(os.getenv("VOCAB_SIZE", "1024")))
    num_layers: int = field(default_factory=lambda: int(os.getenv("NUM_LAYERS", "10")))
    model_dim: int = field(default_factory=lambda: int(os.getenv("MODEL_DIM", "512")))
    num_heads: int = field(default_factory=lambda: int(os.getenv("NUM_HEADS", "8")))
    num_kv_heads: int = field(default_factory=lambda: int(os.getenv("NUM_KV_HEADS", "4")))
    mlp_mult: int = field(default_factory=lambda: int(os.getenv("MLP_MULT", "2")))
    tie_embeddings: bool = field(default_factory=lambda: env_bool("TIE_EMBEDDINGS", True))
    rope_base: float = field(default_factory=lambda: float(os.getenv("ROPE_BASE", "10000")))
    qk_gain_init: float = field(default_factory=lambda: float(os.getenv("QK_GAIN_INIT", "1.5")))
    tied_embed_init_std: float = field(default_factory=lambda: float(os.getenv("TIED_EMBED_INIT_STD", "0.005")))
    logit_softcap: float = field(default_factory=lambda: float(os.getenv("LOGIT_SOFTCAP", "30")))
    init_mode: str = field(default_factory=lambda: os.getenv("INIT_MODE", "zero"))
    use_long_skips: bool = field(default_factory=lambda: env_bool("USE_LONG_SKIPS", True))
    use_resid_mix: bool = field(default_factory=lambda: env_bool("USE_RESID_MIX", True))

    # 524288 tokens/step across all GPUs; at 8 GPUs there is 1 microstep/rank.
    train_batch_tokens: int = field(default_factory=lambda: int(os.getenv("TRAIN_BATCH_TOKENS", "524288")))
    train_seq_len: int = field(default_factory=lambda: int(os.getenv("TRAIN_SEQ_LEN", "1024")))
    grad_accum_global: int = field(default_factory=lambda: int(os.getenv("GRAD_ACCUM_GLOBAL", "8")))
    max_steps: int = field(default_factory=lambda: int(os.getenv("ITERATIONS", "20000")))
    train_seconds: float = field(default_factory=lambda: float(os.getenv("MAX_WALLCLOCK_SECONDS", "600")))
    warmdown_seconds: float = field(default_factory=lambda: float(os.getenv("WARMDOWN_SECONDS", "60")))
    warmup_steps: int = field(default_factory=lambda: int(os.getenv("WARMUP_STEPS", "5")))
    val_every: int = field(default_factory=lambda: int(os.getenv("VAL_LOSS_EVERY", "0")))
    val_batch_seqs_per_rank: int = field(default_factory=lambda: int(os.getenv("VAL_BATCH_SEQS_PER_RANK", "8")))
    train_log_every: int = field(default_factory=lambda: int(os.getenv("TRAIN_LOG_EVERY", "100")))

    optimizer: str = field(default_factory=lambda: os.getenv("OPTIMIZER", "muon"))  # muon or adam
    tied_embed_lr: float = field(default_factory=lambda: float(os.getenv("TIED_EMBED_LR", "0.05")))
    embed_lr: float = field(default_factory=lambda: float(os.getenv("EMBED_LR", "0.6")))
    head_lr: float = field(default_factory=lambda: float(os.getenv("HEAD_LR", "0.008")))
    matrix_lr: float = field(default_factory=lambda: float(os.getenv("MATRIX_LR", "0.04")))
    scalar_lr: float = field(default_factory=lambda: float(os.getenv("SCALAR_LR", "0.04")))
    beta1: float = field(default_factory=lambda: float(os.getenv("BETA1", "0.9")))
    beta2: float = field(default_factory=lambda: float(os.getenv("BETA2", "0.95")))
    adam_eps: float = field(default_factory=lambda: float(os.getenv("ADAM_EPS", "1e-8")))
    muon_momentum: float = field(default_factory=lambda: float(os.getenv("MUON_MOMENTUM", "0.95")))
    muon_momentum_start: float = field(default_factory=lambda: float(os.getenv("MUON_MOMENTUM_START", "0.85")))
    muon_momentum_ramp_steps: int = field(default_factory=lambda: int(os.getenv("MUON_MOMENTUM_RAMP_STEPS", "500")))
    muon_ns_steps: int = field(default_factory=lambda: int(os.getenv("MUON_NS_STEPS", "5")))
    grad_clip_norm: float = field(default_factory=lambda: float(os.getenv("GRAD_CLIP_NORM", "0")))
    compile_model: bool = field(default_factory=lambda: env_bool("COMPILE", True))

    # Export is outside the training-step timer. Always measure post-load BPB.
    codec: str = field(default_factory=lambda: os.getenv("CODEC", "delta_int8"))
    svd_ranks: str = field(default_factory=lambda: os.getenv("SVD_RANKS", "8,16,32,64,128"))
    svd_max_rel_error: float = field(default_factory=lambda: float(os.getenv("SVD_MAX_REL_ERROR", "0.03")))
    analyze_spectra: bool = field(default_factory=lambda: env_bool("ANALYZE_SPECTRA", False))
    save_raw: bool = field(default_factory=lambda: env_bool("SAVE_RAW", False))
    size_cap: int = field(default_factory=lambda: int(os.getenv("SIZE_CAP_BYTES", "16000000")))
    enforce_size_cap: bool = field(default_factory=lambda: env_bool("ENFORCE_SIZE_CAP", False))

    def model_spec(self) -> ModelSpec:
        keys = ModelSpec.__dataclass_fields__.keys()
        return ModelSpec(**{key: getattr(self, key) for key in keys})

    def validate(self) -> None:
        self.model_spec().validate()
        if self.optimizer not in {"muon", "adam"}:
            raise ValueError("OPTIMIZER must be muon or adam")
        if self.codec not in {"weight_int8", "delta_int8", "delta_hybrid"}:
            raise ValueError("CODEC must be weight_int8, delta_int8 or delta_hybrid")
        if self.train_seconds < 0 or self.warmup_steps < 0 or self.max_steps < 1:
            raise ValueError("Invalid training length")
        if self.train_seq_len < 1 or self.train_batch_tokens < 1:
            raise ValueError("Invalid batch / sequence length")
        if self.grad_accum_global < 1:
            raise ValueError("GRAD_ACCUM_GLOBAL must be positive")
        if self.val_batch_seqs_per_rank < 1:
            raise ValueError("VAL_BATCH_SEQS_PER_RANK must be positive")
        if self.svd_max_rel_error < 0:
            raise ValueError("SVD_MAX_REL_ERROR cannot be negative")


# -----------------------------------------------------------------------------
# MODEL / DETERMINISTIC INITIALIZATION
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    """FP32 master weight and bf16 GEMMs when activations are bf16."""

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", freqs, persistent=False)
        self.cached_key: tuple[int, str] | None = None
        self.cached_cos: Tensor | None = None
        self.cached_sin: Tensor | None = None

    def forward(self, seq_len: int, x: Tensor) -> tuple[Tensor, Tensor]:
        key = (seq_len, str(x.device))
        if self.cached_key != key:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            theta = torch.outer(positions, self.inv_freq.to(x.device))
            self.cached_cos = theta.cos()[None, None]
            self.cached_sin = theta.sin()[None, None]
            self.cached_key = key
        assert self.cached_cos is not None and self.cached_sin is not None
        return self.cached_cos.to(dtype=x.dtype), self.cached_sin.to(dtype=x.dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((left * cos + right * sin, right * cos - left * sin), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        d = spec.model_dim
        self.heads = spec.num_heads
        self.kv_heads = spec.num_kv_heads
        self.head_dim = d // spec.num_heads
        kv_dim = self.kv_heads * self.head_dim
        self.c_q = CastedLinear(d, d, bias=False)
        self.c_k = CastedLinear(d, kv_dim, bias=False)
        self.c_v = CastedLinear(d, kv_dim, bias=False)
        self.proj = CastedLinear(d, d, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((self.heads,), spec.qk_gain_init))
        self.rotary = Rotary(self.head_dim, spec.rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, length, dim = x.shape
        q = self.c_q(x).view(bsz, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(bsz, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(bsz, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        cos, sin = self.rotary(length, q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.heads != self.kv_heads
        )
        return self.proj(y.transpose(1, 2).contiguous().view(bsz, length, dim))


class MLP(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.fc = CastedLinear(spec.model_dim, spec.mlp_mult * spec.model_dim, bias=False)
        self.proj = CastedLinear(spec.mlp_mult * spec.model_dim, spec.model_dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        d = spec.model_dim
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(spec)
        self.mlp = MLP(spec)
        self.attn_scale = nn.Parameter(torch.ones(d))
        self.mlp_scale = nn.Parameter(torch.ones(d))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(d), torch.zeros(d)))) if spec.use_resid_mix else None

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        if self.resid_mix is not None:
            mix = self.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None] * x + mix[1][None, None] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None] * self.attn(self.attn_norm(x))
        return x + self.mlp_scale.to(dtype=x.dtype)[None, None] * self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.tok_emb = nn.Embedding(spec.vocab_size, spec.model_dim)
        nn.init.normal_(self.tok_emb.weight, std=spec.tied_embed_init_std)
        self.blocks = nn.ModuleList([Block(spec) for _ in range(spec.num_layers)])
        self.encoder_layers = spec.num_layers // 2
        self.decoder_layers = spec.num_layers - self.encoder_layers
        n_skips = min(self.encoder_layers, self.decoder_layers) if spec.use_long_skips else 0
        self.skip_weights = nn.Parameter(torch.ones(n_skips, spec.model_dim)) if n_skips else None
        self.final_norm = RMSNorm()
        self.lm_head = None if spec.tie_embeddings else CastedLinear(spec.model_dim, spec.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> Tensor:
        x = F.rms_norm(self.tok_emb(input_ids), (self.spec.model_dim,))
        x0 = x
        if self.skip_weights is None:
            for block in self.blocks:
                x = block(x, x0)
        else:
            saved: list[Tensor] = []
            for i in range(self.encoder_layers):
                x = self.blocks[i](x, x0)
                saved.append(x)
            for j in range(self.decoder_layers):
                if saved:
                    x = x + self.skip_weights[j].to(dtype=x.dtype)[None, None] * saved.pop()
                x = self.blocks[self.encoder_layers + j](x, x0)
        x = self.final_norm(x)
        if self.lm_head is None:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        cap = self.spec.logit_softcap
        logits = cap * torch.tanh(logits.float() / cap)
        if targets is None:
            return logits
        return F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1))


def hadamard_square(n: int) -> Tensor:
    """Construct a Sylvester Hadamard matrix of power-of-two order (CPU)."""
    if n < 1 or (n & (n - 1)):
        raise ValueError("Hadamard order must be a power of two")
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


@torch.no_grad()
def initialize_linears(model: GPT, mode: str) -> None:
    # Always zero terminal branch projections for identity residual initialization.
    # ZerO initialization additionally sets other linears to partial-I/Hadamard.
    cache: dict[int, Tensor] = {}
    for module in model.modules():
        if not isinstance(module, CastedLinear):
            continue
        if getattr(module, "_zero_init", False):
            module.weight.zero_()
        elif mode == "zero":
            m, n = module.weight.shape
            if m <= n:
                module.weight.copy_(torch.eye(m, n, dtype=torch.float32))
            else:
                order = 1 << (m - 1).bit_length()
                if order not in cache:
                    cache[order] = hadamard_square(order)
                scale = 2.0 ** (-(math.log2(order) - 1) / 2.0)
                module.weight.copy_(scale * cache[order][:m, :n])


def build_model(spec: ModelSpec, seed: int, device: torch.device, *, mixed_precision: bool = True) -> GPT:
    """Canonical initialization. MUST stay unchanged for delta checkpoint decoding.

    Construct and initialize on CPU under an isolated seed; CUDA DDP ranks and
    CPU checkpoint decoding therefore reconstruct exactly the same reference.
    """
    spec.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GPT(spec)
        initialize_linears(model, spec.init_mode)
    if mixed_precision:
        model = model.to(device=device, dtype=torch.bfloat16)
        # Keep matrix optimizer states and small control variables in fp32.
        for module in model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        for name, p in model.named_parameters():
            if p.ndim < 2 or "skip_weights" in name or "resid_mix" in name:
                p.data = p.data.float()
    else:
        model = model.to(device)
    return model


# -----------------------------------------------------------------------------
# OPTIMIZERS
# -----------------------------------------------------------------------------


@torch.no_grad()
def newton_schulz5(g: Tensor, steps: int) -> Tensor:
    """Muon polar-factor approximation (modded-nanogpt coefficients)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = (g.float() / g.float().norm().clamp_min(1e-7)).to(torch.bfloat16)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    for _ in range(steps):
        xx = x @ x.T
        x = a * x + (b * xx + c * (xx @ xx)) @ x
    return x.T if transpose else x


class Muon(torch.optim.Optimizer):
    """DDP-sharded orthogonalization, with identical synchronized updates.

    DDP first synchronizes all grads. Rank k orthogonalizes matrices with index
    k % world_size, then all ranks all_reduce the packed update buffer.
    """

    def __init__(self, params: list[nn.Parameter], lr: float, momentum: float, ns_steps: int):
        super().__init__(params, dict(lr=lr, base_lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            total = sum(p.numel() for p in params)
            packed = torch.zeros(total, dtype=torch.bfloat16, device=params[0].device)
            pos = 0
            for i, p in enumerate(params):
                size = p.numel()
                if i % world_size == rank and p.grad is not None:
                    state = self.state[p]
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(p.grad)
                    buf = state["momentum"]
                    buf.mul_(group["momentum"]).add_(p.grad)
                    update = p.grad + group["momentum"] * buf  # Nesterov
                    update = newton_schulz5(update, group["ns_steps"])
                    update = update * max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                    packed[pos:pos + size] = update.reshape(-1)
                pos += size
            if distributed:
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            pos = 0
            for p in params:
                size = p.numel()
                p.add_(packed[pos:pos + size].view_as(p).to(p.dtype), alpha=-group["lr"])
                pos += size


def make_optimizers(model: GPT, cfg: Config) -> tuple[list[torch.optim.Optimizer], Muon | None]:
    named_blocks = list(model.blocks.named_parameters())
    matrices = [p for name, p in named_blocks if p.ndim == 2 and "resid_mix" not in name]
    controls = [p for name, p in named_blocks if p.ndim != 2 or "resid_mix" in name]
    if model.skip_weights is not None:
        controls.append(model.skip_weights)

    def adam(params: list[nn.Parameter], lr: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            [{"params": params, "lr": lr, "base_lr": lr}],
            betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps,
            fused=next(model.parameters()).device.type == "cuda",
        )

    emb_lr = cfg.tied_embed_lr if model.lm_head is None else cfg.embed_lr
    opts = [adam([model.tok_emb.weight], emb_lr)]
    if model.lm_head is not None:
        opts.append(adam([model.lm_head.weight], cfg.head_lr))
    muon: Muon | None = None
    if cfg.optimizer == "muon":
        muon = Muon(matrices, cfg.matrix_lr, cfg.muon_momentum_start, cfg.muon_ns_steps)
        opts.append(muon)
    else:
        opts.append(adam(matrices, cfg.matrix_lr))
    if controls:
        opts.append(adam(controls, cfg.scalar_lr))
    return opts, muon


# -----------------------------------------------------------------------------
# FINEWEB SHARDS, DISTRIBUTED LOADING, VALIDATION
# -----------------------------------------------------------------------------


def load_shard(path: Path) -> Tensor:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or header[0] != 20240520 or header[1] != 1:
        raise ValueError(f"Invalid FineWeb shard header: {path}")
    length = int(header[2])
    expected = 1024 + length * 2
    if path.stat().st_size != expected:
        raise ValueError(f"Shard length mismatch: {path}")
    values = np.fromfile(path, dtype="<u2", count=length, offset=1024)
    return torch.from_numpy(values.astype(np.int32))


class TokenStream:
    def __init__(self, glob_pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(glob_pattern))]
        if not self.files:
            raise FileNotFoundError(f"No shards match {glob_pattern}")
        self.file_index = 0
        self.tokens = load_shard(self.files[0])
        self.pos = 0

    def take(self, length: int) -> Tensor:
        parts = []
        remaining = length
        while remaining:
            available = len(self.tokens) - self.pos
            if available == 0:
                self.file_index = (self.file_index + 1) % len(self.files)
                self.tokens = load_shard(self.files[self.file_index])
                self.pos = 0
                continue
            take = min(available, remaining)
            parts.append(self.tokens[self.pos:self.pos + take])
            self.pos += take
            remaining -= take
        return parts[0] if len(parts) == 1 else torch.cat(parts)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.stream = TokenStream(pattern)
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum: int) -> tuple[Tensor, Tensor]:
        per_rank = global_tokens // (self.world_size * grad_accum)
        width = per_rank + 1
        all_ranks = self.stream.take(self.world_size * width)
        start = self.rank * width
        tokens = all_ranks[start:start + width]
        x = tokens[:-1].reshape(-1, seq_len).to(self.device, dtype=torch.int64, non_blocking=True)
        y = tokens[1:].reshape(-1, seq_len).to(self.device, dtype=torch.int64, non_blocking=True)
        return x, y


def load_validation(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No validation shards match {pattern}")
    values = torch.cat([load_shard(f) for f in files])
    usable = (len(values) - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError("Validation data shorter than one sequence")
    return values[:usable + 1]


def byte_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    """Matches the original script's SentencePiece space-boundary convention."""
    n = max(vocab_size, int(sp.vocab_size()))
    lengths = np.zeros(n, dtype=np.int16)
    leading = np.zeros(n, dtype=np.bool_)
    boundary = np.ones(n, dtype=np.bool_)
    for i in range(sp.vocab_size()):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i):
            continue
        boundary[i] = False
        if sp.is_byte(i):
            lengths[i] = 1
            continue
        piece = sp.id_to_piece(i)
        if piece.startswith("▁"):
            leading[i] = True
            piece = piece[1:]
        lengths[i] = len(piece.encode("utf-8"))
    return (torch.tensor(lengths, device=device),
            torch.tensor(leading, device=device),
            torch.tensor(boundary, device=device))


@torch.no_grad()
def evaluate(
    model: nn.Module, tokens: Tensor, luts: tuple[Tensor, Tensor, Tensor],
    cfg: Config, rank: int, world_size: int, device: torch.device,
) -> tuple[float, float]:
    # Score the full fixed validation split. Never train on it.
    n_seq = (len(tokens) - 1) // cfg.train_seq_len
    lo = n_seq * rank // world_size
    hi = n_seq * (rank + 1) // world_size
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    byte_sum = torch.zeros((), dtype=torch.float64, device=device)
    tok_sum = torch.zeros((), dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    base_bytes, leading, boundary = luts
    for start in range(lo, hi, cfg.val_batch_seqs_per_rank):
        end = min(start + cfg.val_batch_seqs_per_rank, hi)
        piece = tokens[start * cfg.train_seq_len:end * cfg.train_seq_len + 1]
        x = piece[:-1].reshape(-1, cfg.train_seq_len).to(device, dtype=torch.long)
        y = piece[1:].reshape(-1, cfg.train_seq_len).to(device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(x, y)
        loss_sum += loss.double() * y.numel()
        tok_sum += y.numel()
        x_flat, y_flat = x.reshape(-1), y.reshape(-1)
        count = base_bytes[y_flat].to(torch.int32) + (leading[y_flat] & ~boundary[x_flat]).to(torch.int32)
        byte_sum += count.sum(dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        for item in (loss_sum, byte_sum, tok_sum):
            dist.all_reduce(item, op=dist.ReduceOp.SUM)
    model.train(was_training)
    loss = float((loss_sum / tok_sum).item())
    bpb = float((loss_sum / byte_sum / math.log(2)).item())
    return loss, bpb


# -----------------------------------------------------------------------------
# WEIGHT / DISPLACEMENT CODECS (NO DATA-DEPENDENT CALIBRATION)
# -----------------------------------------------------------------------------


@torch.no_grad()
def quantize_matrix(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Full-range symmetric per-row INT8; no artificial 1/127 scale floor."""
    mat = matrix.float().cpu().contiguous()
    high = mat.abs().amax(dim=1)
    scale = torch.where(high > 0, high / 127.0, torch.ones_like(high))
    q = torch.clamp(torch.round(mat / scale[:, None]), -127, 127).to(torch.int8)
    return q, scale.float()


@torch.no_grad()
def dequantize_matrix(q: Tensor, scale: Tensor) -> Tensor:
    return q.float() * scale.float()[:, None]


def candidate_ranks(text: str) -> list[int]:
    ranks = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
    if any(r < 1 for r in ranks):
        raise ValueError("SVD_RANKS must be positive")
    return ranks


@torch.no_grad()
def encode_matrix(mat: Tensor, codec: str, ranks: list[int], tolerance: float) -> dict:
    q, scale = quantize_matrix(mat)
    item = {"kind": "int8", "q": q, "scale": scale}
    if codec != "delta_hybrid" or min(mat.shape) < 2:
        return item

    # SVD factorization is performed on the learned DISPLACEMENT. Only store it
    # if reconstruction is within the declared tolerance and cheaper in bytes.
    m, n = mat.shape
    a = mat.float().cpu()
    norm = float(torch.linalg.vector_norm(a))
    if norm == 0.0:
        return item
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    for rank in ranks:
        if rank >= min(m, n):
            continue
        # Conservative rate check before performing matrix reconstruction.
        factor_bytes = 2 * rank * (m + n)
        dense_bytes = m * n + 4 * m
        if factor_bytes >= dense_bytes:
            continue
        left = (u[:, :rank] * s[:rank]).to(torch.float16).contiguous()
        right = vh[:rank].to(torch.float16).contiguous()
        approx = left.float() @ right.float()
        rel_error = float(torch.linalg.vector_norm(a - approx)) / norm
        if rel_error <= tolerance:
            return {"kind": "svd_f16", "left": left, "right": right, "rel_error": rel_error}
    return item


@torch.no_grad()
def encode_checkpoint(
    trained: dict[str, Tensor], spec: ModelSpec, seed: int,
    codec: str, rank_text: str = "8,16,32,64,128", tolerance: float = 0.03,
) -> tuple[bytes, dict[str, int]]:
    if codec not in {"weight_int8", "delta_int8", "delta_hybrid"}:
        raise ValueError(f"Unknown codec {codec}")
    use_delta = codec != "weight_int8"
    reference = build_model(spec, seed, torch.device("cpu")).state_dict() if use_delta else {}
    tensors: dict[str, dict] = {}
    count_kinds: dict[str, int] = {}
    ranks = candidate_ranks(rank_text)
    for name, actual in trained.items():
        w = actual.detach().cpu().contiguous()
        # High-sensitivity control tensors stay unquantized. A small footprint
        # also makes preserving scalars/vectors inexpensive.
        if not w.is_floating_point() or w.ndim != 2 or w.numel() <= 65536 or \
                "skip_weights" in name or "resid_mix" in name:
            item = {"kind": "raw", "data": w}
        else:
            mat = w.float() - reference[name].float() if use_delta else w.float()
            item = encode_matrix(mat, codec, ranks, tolerance)
        tensors[name] = item
        kind = item["kind"]
        count_kinds[kind] = count_kinds.get(kind, 0) + 1
    payload = {
        "format": "golf_delta_v1",
        "model_spec": asdict(spec),
        "init_seed": seed,
        "codec": codec,
        "tensors": tensors,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return zlib.compress(buffer.getvalue(), 9), count_kinds


@torch.no_grad()
def decode_checkpoint(blob: bytes, device: torch.device = torch.device("cpu")) -> GPT:
    payload = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu", weights_only=True)
    if payload["format"] != "golf_delta_v1":
        raise ValueError("Unknown encoded model format")
    spec = ModelSpec(**payload["model_spec"])
    init_seed = int(payload["init_seed"])
    use_delta = payload["codec"] != "weight_int8"
    model = build_model(spec, init_seed, torch.device("cpu"))
    original = model.state_dict()
    rebuilt = {}
    for name, expected in original.items():
        item = payload["tensors"][name]
        if item["kind"] == "raw":
            result = item["data"]
        elif item["kind"] == "int8":
            approx = dequantize_matrix(item["q"], item["scale"])
            result = (expected.float() + approx) if use_delta else approx
        elif item["kind"] == "svd_f16":
            if not use_delta:
                raise ValueError("SVD codec is only defined for deltas")
            result = expected.float() + item["left"].float() @ item["right"].float()
        else:
            raise ValueError(f"Invalid encoding for {name}")
        if result.shape != expected.shape:
            raise ValueError(f"Shape mismatch in tensor {name}")
        rebuilt[name] = result.to(dtype=expected.dtype)
    model.load_state_dict(rebuilt, strict=True)
    return model.to(device)


@torch.no_grad()
def report_displacement_spectra(
    trained: dict[str, Tensor], spec: ModelSpec, seed: int, log,
) -> None:
    initial = build_model(spec, seed, torch.device("cpu")).state_dict()
    for name, w in trained.items():
        if w.ndim != 2 or w.numel() <= 65536 or "skip_weights" in name:
            continue
        delta = w.float().cpu() - initial[name].float()
        sv = torch.linalg.svdvals(delta)
        cumulative = torch.cumsum(sv.square(), 0)
        if cumulative[-1] == 0:
            log(f"spectrum:{name} rank95:0 rank99:0")
            continue
        fraction = cumulative / cumulative[-1]
        r95 = int(torch.searchsorted(fraction, 0.95).item() + 1)
        r99 = int(torch.searchsorted(fraction, 0.99).item() + 1)
        log(f"spectrum:{name} rank95:{r95} rank99:{r99} full_rank:{min(delta.shape)}")


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_cuda() -> tuple[torch.device, int, int, bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; use --smoke-test for CPU verification")
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    return device, rank, world_size, world_size > 1


def learning_rate_factor(elapsed: float, cfg: Config) -> float:
    if cfg.train_seconds <= 0 or cfg.warmdown_seconds <= 0:
        return 1.0
    remaining = max(0.0, cfg.train_seconds - elapsed)
    return min(1.0, remaining / cfg.warmdown_seconds)


def train_step(
    model: nn.Module, loader: DistributedTokenLoader, optimizers: list[torch.optim.Optimizer],
    cfg: Config, grad_accum: int, distributed: bool, clip_model: GPT,
) -> Tensor:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=next(clip_model.parameters()).device)
    for micro in range(grad_accum):
        if distributed:
            model.require_backward_grad_sync = (micro == grad_accum - 1)
        x, y = loader.next_batch(cfg.train_batch_tokens, cfg.train_seq_len, grad_accum)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        (loss / grad_accum).backward()
        total_loss += loss.detach() / grad_accum
    if cfg.grad_clip_norm > 0:
        nn.utils.clip_grad_norm_(clip_model.parameters(), cfg.grad_clip_norm)
    for optimizer in optimizers:
        optimizer.step()
    return total_loss


def train() -> None:
    cfg = Config()
    cfg.validate()
    device, rank, world_size, distributed = setup_cuda()
    master = rank == 0
    set_seed(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"train_{cfg.run_id}.log"

    def log(message: str) -> None:
        if master:
            print(message, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                print(message, file=handle)

    if cfg.grad_accum_global % world_size:
        raise ValueError("GRAD_ACCUM_GLOBAL must be divisible by WORLD_SIZE")
    grad_accum = cfg.grad_accum_global // world_size
    denom = world_size * grad_accum * cfg.train_seq_len
    if cfg.train_batch_tokens % denom:
        raise ValueError(f"TRAIN_BATCH_TOKENS must be divisible by {denom}")

    spec = cfg.model_spec()
    base = build_model(spec, cfg.seed, device)
    opts, muon = make_optimizers(base, cfg)
    if cfg.compile_model:
        compiled = torch.compile(base, dynamic=False, fullgraph=True)
    else:
        compiled = base
    model: nn.Module = DDP(compiled, device_ids=[device.index], broadcast_buffers=False) if distributed else compiled
    model.train()
    train_pattern = os.path.join(cfg.data_path, "fineweb_train_*.bin")
    val_pattern = os.path.join(cfg.data_path, "fineweb_val_*.bin")
    loader = DistributedTokenLoader(train_pattern, rank, world_size, device)
    log(f"config:{json.dumps(asdict(cfg), sort_keys=True)}")
    log(f"model_params:{sum(p.numel() for p in base.parameters())} world_size:{world_size} accum:{grad_accum}")
    log(f"dataset:{cfg.data_path} train_shards:{len(loader.stream.files)}")
    log("timing:training steps only; compilation/warmup/validation/export excluded")

    # Compilation and warmup are allowed as a separate phase in this research
    # harness; save and restore all trained state before starting the timed run.
    if cfg.warmup_steps:
        start_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        opt_states = [copy.deepcopy(opt.state_dict()) for opt in opts]
        for i in range(cfg.warmup_steps):
            _ = train_step(model, loader, opts, cfg, grad_accum, distributed, base)
            if (i + 1) % max(1, cfg.warmup_steps // 5) == 0:
                log(f"warmup:{i + 1}/{cfg.warmup_steps}")
        base.load_state_dict(start_state, strict=True)
        for opt, state in zip(opts, opt_states, strict=True):
            opt.load_state_dict(state)
            opt.zero_grad(set_to_none=True)
        # Warmup consumed training tokens; replay the identical stream from zero.
        loader = DistributedTokenLoader(train_pattern, rank, world_size, device)
        if distributed:
            model.require_backward_grad_sync = True

    # Read-only validation may be expensive; it is deliberately not required
    # until the final checkpoint unless VAL_LOSS_EVERY > 0.
    val_tokens = None
    luts = None

    def validate_now(tag: str) -> tuple[float, float]:
        nonlocal val_tokens, luts
        if val_tokens is None:
            val_tokens = load_validation(val_pattern, cfg.train_seq_len)
            sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
            if int(sp.vocab_size()) != spec.vocab_size:
                raise ValueError("Tokenizer vocabulary size does not match model")
            luts = byte_luts(sp, spec.vocab_size, device)
        assert luts is not None
        loss, bpb = evaluate(model, val_tokens, luts, cfg, rank, world_size, device)
        log(f"{tag} val_loss:{loss:.6f} val_bpb:{bpb:.6f}")
        return loss, bpb

    elapsed_train = 0.0
    step = 0
    while step < cfg.max_steps and (cfg.train_seconds == 0 or elapsed_train < cfg.train_seconds):
        factor = learning_rate_factor(elapsed_train, cfg)
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * factor
        if muon is not None:
            fraction = min(1.0, step / max(cfg.muon_momentum_ramp_steps, 1))
            for group in muon.param_groups:
                group["momentum"] = cfg.muon_momentum_start + fraction * (cfg.muon_momentum - cfg.muon_momentum_start)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        loss = train_step(model, loader, opts, cfg, grad_accum, distributed, base)
        torch.cuda.synchronize(device)
        elapsed_train += time.perf_counter() - t0
        step += 1
        if step <= 10 or (cfg.train_log_every > 0 and step % cfg.train_log_every == 0):
            log(f"step:{step} train_loss:{loss.item():.5f} train_time:{elapsed_train:.2f}s lr_factor:{factor:.4f}")
        if cfg.val_every > 0 and step % cfg.val_every == 0:
            validate_now(f"step:{step}")
        # In distributed training, stop only after all ranks agree.
        reached = cfg.train_seconds > 0 and elapsed_train >= cfg.train_seconds
        if distributed:
            reached_tensor = torch.tensor(int(reached), device=device)
            dist.all_reduce(reached_tensor, op=dist.ReduceOp.MAX)
            reached = bool(reached_tensor.item())
        if reached:
            break

    log(f"training_complete:step:{step} train_time:{elapsed_train:.3f}s")
    pre_loss, pre_bpb = validate_now("pre_export")
    state = {name: value.detach().cpu().clone() for name, value in base.state_dict().items()}
    size_ok = True
    if master:
        if cfg.save_raw:
            torch.save(state, output / "final_model_raw.pt")
        if cfg.analyze_spectra:
            report_displacement_spectra(state, spec, cfg.seed, log)
        blob, kinds = encode_checkpoint(state, spec, cfg.seed, cfg.codec,
                                        cfg.svd_ranks, cfg.svd_max_rel_error)
        artifact_path = output / "final_model.ptz"
        artifact_path.write_bytes(blob)
        code_bytes = Path(__file__).stat().st_size
        artifact_bytes = len(blob)
        total_bytes = artifact_bytes + code_bytes
        log(f"codec:{cfg.codec} tensors:{kinds} model_bytes:{artifact_bytes} code_bytes:{code_bytes} total_bytes:{total_bytes}")
        size_ok = total_bytes <= cfg.size_cap
        log(f"size_cap:{cfg.size_cap} within_cap:{int(size_ok)}")
    if distributed:
        flag = torch.tensor(int(size_ok), dtype=torch.int32, device=device)
        dist.broadcast(flag, src=0)
        size_ok = bool(flag.item())
        dist.barrier()
    if cfg.enforce_size_cap and not size_ok:
        raise RuntimeError(f"Artifact + code exceeds {cfg.size_cap} bytes")
    # Every rank reloads to ensure reported BPB is for the submitted artifact.
    restored = decode_checkpoint((output / "final_model.ptz").read_bytes(), device)
    # Evaluate decoded model directly, avoiding reliance on compiled model caches.
    assert val_tokens is not None and luts is not None
    q_loss, q_bpb = evaluate(restored, val_tokens, luts, cfg, rank, world_size, device)
    log(f"post_export val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")
    log(f"export_impact delta_val_loss:{q_loss - pre_loss:+.8f} delta_val_bpb:{q_bpb - pre_bpb:+.8f}")
    if distributed:
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# CPU SMOKE TESTS: INITIALIZATION, FORWARD/BACKWARD, CODEC ROUNDTRIP
# -----------------------------------------------------------------------------


def smoke_test() -> None:
    spec = ModelSpec(vocab_size=64, num_layers=2, model_dim=32, num_heads=4,
                     num_kv_heads=2, mlp_mult=2, init_mode="zero")
    model = build_model(spec, 123, torch.device("cpu"), mixed_precision=False)
    second = build_model(spec, 123, torch.device("cpu"), mixed_precision=False)
    for (name, a), (_, b) in zip(model.state_dict().items(), second.state_dict().items(), strict=True):
        assert torch.equal(a, b), f"Initializer is not deterministic: {name}"
    x = torch.randint(0, spec.vocab_size, (2, 8))
    y = torch.randint(0, spec.vocab_size, (2, 8))
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    after = copy.deepcopy(model.state_dict())
    assert any(not torch.equal(initial[k], after[k]) for k in after)
    for codec in ("weight_int8", "delta_int8", "delta_hybrid"):
        # Decoder uses bf16 for canonical export. For smoke testing, convert
        # our fp32-trained state to the same canonical dtypes before encoding.
        canonical = build_model(spec, 123, torch.device("cpu"))
        canonical.load_state_dict(after)
        packed, kinds = encode_checkpoint(canonical.state_dict(), spec, 123, codec, "1,2,4", 0.2)
        decoded = decode_checkpoint(packed)
        for name, t in decoded.state_dict().items():
            assert torch.isfinite(t.float()).all(), (codec, name)
        test_loss = decoded(x, y)
        assert torch.isfinite(test_loss), codec
        print(f"smoke {codec}: {len(packed)} bytes, {kinds}, loss={test_loss.item():.4f}")
    # Exercise the actual quantization and SVD branches (tiny models consist
    # mostly of passthrough tensors under the 65,536-element threshold).
    torch.manual_seed(7)
    u = torch.randn(160, 3)
    v = torch.randn(3, 160)
    low_rank = u @ v
    compressed = encode_matrix(low_rank, "delta_hybrid", [2, 3, 4, 8], 0.02)
    assert compressed["kind"] == "svd_f16", compressed["kind"]
    reconstructed = compressed["left"].float() @ compressed["right"].float()
    relative_error = float(torch.linalg.vector_norm(reconstructed - low_rank) / torch.linalg.vector_norm(low_rank))
    assert relative_error < 0.02
    full = encode_matrix(low_rank, "delta_int8", [3], 0.02)
    assert full["kind"] == "int8"
    assert float(torch.linalg.vector_norm(dequantize_matrix(full["q"], full["scale"]) - low_rank)) > 0
    # Quantization must preserve small scales, not floor them at 1/127.
    tiny = torch.tensor([[0.0001, -0.0002], [0.0, 0.0]], dtype=torch.float32)
    q, scales = quantize_matrix(tiny)
    assert scales[0] < 0.001 and torch.max((dequantize_matrix(q, scales) - tiny).abs()) < 0.00001
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke-test"]:
        smoke_test()
    elif sys.argv[1:]:
        raise SystemExit("Only --smoke-test is supported; configure runs with environment variables")
    else:
        train()
