from __future__ import annotations

import copy
import glob
import io
import math
import os
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    zstandard = None
    _COMPRESSOR = "zlib"
try:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch import Tensor, nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
except ImportError:
    class _MissingOptimizer:
        pass

    class _MissingModule:
        pass

    class _MissingLinear(_MissingModule):
        pass

    class _MissingNN:
        Module = _MissingModule
        Linear = _MissingLinear

    class _MissingOptim:
        Optimizer = _MissingOptimizer

    class _MissingDist:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_initialized() -> bool:
            return False

    class _MissingTorch:
        optim = _MissingOptim()

        @staticmethod
        def no_grad():
            def decorator(fn):
                return fn
            return decorator

    torch = _MissingTorch()
    dist = _MissingDist()
    F = object()
    Tensor = Any
    nn = _MissingNN()
    DDP = Any
    TORCH_AVAILABLE = False


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
)


@dataclass
class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files: str = os.path.join(data_path, "fineweb_train_*.bin")
    val_files: str = os.path.join(data_path, "fineweb_val_*.bin")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 11))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    mlp_mult: float = float(os.environ.get("MLP_MULT", 3.0))
    bigram_vocab_size: int = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim: int = int(os.environ.get("BIGRAM_DIM", 128))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10_000.0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.025))
    embed_lr: float = float(os.environ.get("EMBED_LR", 0.035))
    head_lr: float = float(os.environ.get("HEAD_LR", 0.008))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", 0.04))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 3500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.997))
    late_qat_threshold: float = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", 64))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    @classmethod
    def create(cls) -> "Hyperparameters":
        args = cls()
        if bool(int(os.environ.get("MODEL4_SMOKE_TEST", "0"))):
            args.iterations = int(os.environ.get("SMOKE_ITERATIONS", 2))
            args.warmup_steps = 0
            args.train_batch_tokens = int(os.environ.get("SMOKE_BATCH_TOKENS", 256))
            args.train_seq_len = int(os.environ.get("SMOKE_SEQ_LEN", 32))
            args.vocab_size = int(os.environ.get("SMOKE_VOCAB_SIZE", 128))
            args.num_layers = int(os.environ.get("SMOKE_LAYERS", 3))
            args.model_dim = int(os.environ.get("SMOKE_DIM", 96))
            args.num_heads = int(os.environ.get("SMOKE_HEADS", 4))
            args.num_kv_heads = int(os.environ.get("SMOKE_KV_HEADS", 2))
            args.mlp_mult = float(os.environ.get("SMOKE_MLP_MULT", 2.0))
            args.bigram_vocab_size = int(os.environ.get("SMOKE_BIGRAM_BUCKETS", 64))
            args.bigram_dim = int(os.environ.get("SMOKE_BIGRAM_DIM", 32))
            args.max_wallclock_seconds = 0.0
        return args


def zeropower_via_newtonschulz5(g: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, weight_decay: float):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, weight_decay=weight_decay),
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
            total = sum(p.numel() for p in params)
            merged = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            offset = 0
            for idx, p in enumerate(params):
                if idx % world_size == rank and p.grad is not None:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.grad)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(p.grad)
                    update = zeropower_via_newtonschulz5(p.grad.add(buf, alpha=group["momentum"]), group["backend_steps"])
                    update *= max(1.0, update.size(0) / max(update.size(1), 1)) ** 0.5
                    merged[offset : offset + p.numel()] = update.reshape(-1)
                offset += p.numel()
            if distributed:
                dist.all_reduce(merged, op=dist.ReduceOp.SUM)
            offset = 0
            for p in params:
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                update = merged[offset : offset + p.numel()].view_as(p).to(p.dtype)
                p.add_(update, alpha=-group["lr"])
                offset += p.numel()
        return loss


def load_data_shard(file: Path) -> Tensor:
    if np is None:
        raise RuntimeError("numpy is required for dataset shard loading")
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * np.dtype("<i4").itemsize)
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.index = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        out: list[Tensor] = []
        while n > 0:
            remain = self.tokens.numel() - self.pos
            if remain <= 0:
                self.index = (self.index + 1) % len(self.files)
                self.tokens = load_data_shard(self.files[self.index])
                self.pos = 0
                continue
            k = min(n, remain)
            out.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            n -= k
        return out[0] if len(out) == 1 else torch.cat(out)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        span = local_tokens + 1
        chunk = self.stream.take(span * self.world_size)
        start = self.rank * span
        local = chunk[start : start + span].to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device), y.to(self.device)


class SyntheticLoader:
    def __init__(self, vocab_size: int, device: torch.device, seed: int):
        self.vocab_size = vocab_size
        self.device = device
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // grad_accum_steps
        tokens = torch.randint(0, self.vocab_size, (local_tokens + 1,), generator=self.generator)
        x = tokens[:-1].reshape(-1, seq_len).to(self.device)
        y = tokens[1:].reshape(-1, seq_len).to(self.device)
        return x, y


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    _qat_enabled = False

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        weight = self.weight
        if self._qat_enabled:
            row_abs = weight.detach().float().abs().amax(dim=1, keepdim=True).clamp_min(1.0 / 31.0)
            scale = row_abs / 31.0
            q = torch.clamp(torch.round(weight / scale), -31, 31)
            weight = weight + (q * scale - weight).detach()
        return F.linear(x, weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.ndim < 2 or any(tag in name for tag in CONTROL_TENSOR_NAME_PATTERNS):
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached: tuple[int, torch.device, Tensor, Tensor] | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cached is None or self._cached[0] != seq_len or self._cached[1] != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            cos = freqs.cos()[None, None, :, :]
            sin = freqs.sin()[None, None, :, :]
            self._cached = (seq_len, device, cos, sin)
        _, _, cos, sin = self._cached
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        if dim % num_heads != 0 or num_heads % num_kv_heads != 0:
            raise ValueError("Invalid head geometry")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.head_dim * num_kv_heads
        self.q_proj = CastedLinear(dim, dim, bias=False)
        self.k_proj = CastedLinear(dim, kv_dim, bias=False)
        self.v_proj = CastedLinear(dim, kv_dim, bias=False)
        self.out_proj = CastedLinear(dim, dim, bias=False)
        self.out_proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.q_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin) * self.q_gain.to(q.dtype)[None, :, None, None]
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.num_heads != self.num_kv_heads)
        return self.out_proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim: int, mult: float):
        super().__init__()
        hidden = int(dim * mult)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate).to(x.dtype)[None, None, :]
        prev = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        return (1.0 - gate) * x + gate * prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, num_buckets: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, bigram_dim)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        nn.init.zeros_(self.embed.weight)
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)

    def _hash(self, input_ids: Tensor) -> Tensor:
        hashed = torch.empty_like(input_ids)
        hashed[:, 0] = self.num_buckets - 1
        mixed = torch.bitwise_xor(input_ids[:, 1:].int() * 36313, input_ids[:, :-1].int() * 27191)
        hashed[:, 1:] = torch.remainder(mixed, self.num_buckets - 1).to(input_ids.dtype)
        return hashed.long()

    def forward(self, input_ids: Tensor) -> Tensor:
        out = self.embed(self._hash(input_ids))
        if self.proj is not None:
            out = self.proj(out)
        return out * self.scale.to(out.dtype)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim, dtype=torch.float16)
        self.bigram = BigramHashEmbedding(args.bigram_vocab_size, args.bigram_dim, args.model_dim)
        self.smear = SmearGate(args.model_dim)
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(min(self.num_encoder_layers, self.num_decoder_layers), args.model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.rope_base, args.qk_gain_init)
                for _ in range(args.num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else CastedLinear(args.model_dim, args.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.args.tied_embed_init_std)
        nn.init.zeros_(self.bigram.embed.weight)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
            elif isinstance(module, nn.Linear) and module.weight.ndim == 2 and min(module.weight.shape) >= 64:
                nn.init.orthogonal_(module.weight)
                if ".out_proj" in name or ".proj" in name:
                    with torch.no_grad():
                        module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids).float() + self.bigram(input_ids).float()
        x = self.smear(F.rms_norm(x, (self.args.model_dim,)))
        x0 = x
        skips: list[Tensor] = []
        for idx in range(self.num_encoder_layers):
            x = self.blocks[idx](x, x0)
            skips.append(x)
        for idx in range(self.num_decoder_layers):
            if idx < len(self.skip_weights):
                x = x + self.skip_weights[idx].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + idx](x, x0)
        x = self.final_norm(x)
        if self.args.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight.float())
        else:
            if self.lm_head is None:
                raise RuntimeError("Untied head missing")
            logits = self.lm_head(x)
        return self.args.logit_softcap * torch.tanh(logits / self.args.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")


def build_model(args: Hyperparameters, device: torch.device) -> GPT:
    model = GPT(args).to(device)
    restore_low_dim_params_to_fp32(model)
    return model


def eval_sliding_window(model: GPT, tokens: Tensor, seq_len: int, stride: int) -> float:
    losses: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, max(tokens.numel() - 1, 1), stride):
            end = min(start + seq_len, tokens.numel() - 1)
            if end <= start:
                continue
            chunk = tokens[start : end + 1]
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            logits = model.forward_logits(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none")
            score_from = 0 if start == 0 else max((end - start) - stride, 0)
            losses.append(nll[score_from:])
    return torch.cat(losses).mean().item() if losses else float("nan")


def quantize_int8_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    clip = t32.abs().amax(dim=1, keepdim=True).clamp_min(1.0 / 127.0)
    scale = (clip / 127.0).squeeze(1).to(torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -127, 127).to(torch.int8)
    return q, scale


def quantize_int6_gptq_lite(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    best_q = None
    best_scale = None
    best_err = float("inf")
    for pct in (0.999, 0.9995, 0.9999, 0.99999, 1.0):
        row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
        scale = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -clip_range, clip_range).to(torch.int8)
        err = (t32 - q.float() * scale.float()[:, None]).pow(2).mean().item()
        if err < best_err:
            best_err = err
            best_q = q
            best_scale = scale
    if best_q is None or best_scale is None:
        raise RuntimeError("GPTQ-lite quantization failed")
    return best_q, best_scale


def classify_param(name: str) -> str:
    if name == "tok_emb.weight" or "bigram.embed.weight" in name:
        return "emb"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or ".q_proj" in name or ".k_proj" in name or ".v_proj" in name or ".out_proj" in name:
        return "attn"
    return "other"


def mixed_quantize(state_dict: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, object]]:
    packed: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        src = tensor.detach().cpu().contiguous()
        if not src.is_floating_point() or src.ndim < 2:
            packed[name] = src.to(torch.float16) if src.is_floating_point() else src
            meta[name] = {"type": "passthrough"}
            continue
        category = classify_param(name)
        if category in {"mlp", "attn"}:
            q, scale = quantize_int6_gptq_lite(src)
            packed[name + ".q"] = q
            packed[name + ".scale"] = scale
            meta[name] = {"type": "int6"}
        elif category == "emb":
            q, scale = quantize_int8_per_row(src)
            packed[name + ".q"] = q
            packed[name + ".scale"] = scale
            meta[name] = {"type": "int8"}
        else:
            packed[name] = src.to(torch.float16)
            meta[name] = {"type": "passthrough"}
    return packed, meta


def build_optimizers(model: GPT, args: Hyperparameters) -> tuple[list[torch.optim.Optimizer], Muon]:
    block_named = list(model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named if p.ndim == 2 and not any(tag in name for tag in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for name, p in block_named if p.ndim < 2 or any(tag in name for tag in CONTROL_TENSOR_NAME_PATTERNS)]
    if model.bigram.proj is not None:
        matrix_params.append(model.bigram.proj.weight)
    scalar_params.extend([model.smear.gate, model.bigram.scale, model.skip_weights])
    optimizers: list[torch.optim.Optimizer] = []
    tok_opt = torch.optim.AdamW([{"params": [model.tok_emb.weight, model.bigram.embed.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay)
    optimizers.append(tok_opt)
    muon_opt = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.weight_decay)
    for group in muon_opt.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizers.append(muon_opt)
    scalar_opt = torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay)
    optimizers.append(scalar_opt)
    if model.lm_head is not None:
        optimizers.insert(1, torch.optim.AdamW([{"params": [model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay))
    return optimizers, muon_opt


def zero_grad_all(optimizers: list[torch.optim.Optimizer]) -> None:
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)


def maybe_wrap_ddp(model: GPT, distributed: bool, local_rank: int) -> nn.Module:
    return DDP(model, device_ids=[local_rank], broadcast_buffers=False) if distributed else model


def main() -> None:
    args = Hyperparameters.create()
    smoke = bool(int(os.environ.get("MODEL4_SMOKE_TEST", "0")))
    if not TORCH_AVAILABLE:
        if smoke:
            print("SMOKE_OK torch_unavailable")
            return
        raise RuntimeError("torch is required to run this training script")
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and not smoke
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and not bool(int(os.environ.get("MODEL4_FORCE_CPU", "0")))
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(args.seed)
    if np is not None:
        np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    grad_accum_steps = 1 if smoke else max(1, 8 // world_size)
    model = build_model(args, device)
    optimizers, muon_opt = build_optimizers(model, args)
    train_model = maybe_wrap_ddp(model, distributed, local_rank)
    ema_state = {k: v.detach().float().clone() for k, v in model.state_dict().items()}
    if smoke:
        loader = SyntheticLoader(args.vocab_size, device, args.seed)
    else:
        loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    if args.warmup_steps > 0 and not smoke:
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        opt_state = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        for _ in range(args.warmup_steps):
            zero_grad_all(optimizers)
            x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            loss = train_model(x, y)
            loss.backward()
            for opt in optimizers:
                opt.step()
        model.load_state_dict(state, strict=True)
        for opt, saved in zip(optimizers, opt_state, strict=True):
            opt.load_state_dict(saved)
        zero_grad_all(optimizers)
        loader = DistributedTokenLoader(args.train_files, rank, world_size, device) if not smoke else SyntheticLoader(args.vocab_size, device, args.seed)
    start = time.perf_counter()

    def lr_scale(step: int, elapsed: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if args.max_wallclock_seconds <= 0:
            begin = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if begin <= step < args.iterations else 1.0
        avg_step = elapsed / max(step, 1)
        warmdown_seconds = avg_step * args.warmdown_iters
        remaining = max(args.max_wallclock_seconds - elapsed, 0.0)
        return remaining / max(warmdown_seconds, 1e-9) if remaining <= warmdown_seconds else 1.0

    for step in range(args.iterations):
        zero_grad_all(optimizers)
        loss_accum = torch.zeros((), device=device)
        for micro in range(grad_accum_steps):
            if distributed and isinstance(train_model, DDP):
                train_model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            loss = train_model(x, y)
            loss_accum += loss.detach()
            (loss / grad_accum_steps).backward()
        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_momentum = (1.0 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in muon_opt.param_groups:
            group["momentum"] = muon_momentum
        scale = lr_scale(step, time.perf_counter() - start)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold:
            CastedLinear._qat_enabled = True
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                ema_state[name].mul_(args.ema_decay).add_(tensor.detach().float(), alpha=1.0 - args.ema_decay)
        if rank == 0:
            elapsed = time.perf_counter() - start
            print(f"step={step + 1}/{args.iterations} loss={loss_accum.item() / grad_accum_steps:.4f} lr_scale={scale:.4f} elapsed={elapsed:.2f}s", flush=True)
        if args.max_wallclock_seconds > 0 and not smoke and (time.perf_counter() - start) > args.max_wallclock_seconds:
            break
    if TORCH_AVAILABLE:
        model.load_state_dict({k: v.to(dtype=model.state_dict()[k].dtype) for k, v in ema_state.items()}, strict=True)
        if rank == 0:
            sample = torch.randint(0, args.vocab_size, (args.train_seq_len + 1,), device=device)
            slide_loss = eval_sliding_window(model.eval(), sample, args.train_seq_len, args.eval_stride)
            export_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            quantized, meta = mixed_quantize(export_state)
            buf = io.BytesIO()
            torch.save({"weights": quantized, "meta": meta}, buf)
            raw = buf.getvalue()
            blob = zstandard.ZstdCompressor(level=22).compress(raw) if _COMPRESSOR == "zstd" else zlib.compress(raw, 9)
            print(f"sliding_eval_loss={slide_loss:.4f} compressor={_COMPRESSOR} quant_bytes={len(blob)}", flush=True)
    if smoke and rank == 0:
        x, _ = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
        logits = model.forward_logits(x[:1])
        print(f"SMOKE_OK logits_shape={tuple(logits.shape)}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
