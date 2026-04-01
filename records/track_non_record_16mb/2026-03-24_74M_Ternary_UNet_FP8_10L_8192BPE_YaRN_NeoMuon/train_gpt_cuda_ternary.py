"Ternary training script for OpenAI's Parameter Golf Challenge. Ciprian-Florin Ifrim - 24 March 2026"

import copy
import glob
import io
import math
import os
import random
import sys
import time
import lzma
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn_interface import flash_attn_func

# ---------------------------------------------------------------------------
# Hyperparameters (all configurable via environment variables)
# ---------------------------------------------------------------------------
def _e(k, d, t=str):
    v = os.environ.get(k, str(d))
    if t == bool: return bool(int(v))
    return t(v)

class Hyperparameters:
    data_path = _e("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = _e("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", f"run_{int(time.time())}")
    seed = _e("SEED", 1337, int)
    compile_mode = _e("COMPILE_MODE", "default")
    val_batch_size = _e("VAL_BATCH_SIZE", 524288, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 500, int)
    train_log_every = _e("TRAIN_LOG_EVERY", 10, int)
    iterations = _e("ITERATIONS", 2000, int)
    warmdown_fraction = _e("WARMDOWN_FRACTION", 0.2, float)
    warmup_steps = _e("WARMUP_STEPS", 20, int)
    train_batch_tokens = _e("TRAIN_BATCH_TOKENS", 524288, int)
    train_seq_len = _e("TRAIN_SEQ_LEN", 1024, int)
    max_wallclock_seconds = _e("MAX_WALLCLOCK_SECONDS", 0.0, float)
    vocab_size = _e("VOCAB_SIZE", 1024, int)
    num_layers = _e("NUM_LAYERS", 16, int)
    num_kv_heads = _e("NUM_KV_HEADS", 4, int)
    model_dim = _e("MODEL_DIM", 512, int)
    num_heads = _e("NUM_HEADS", 8, int)
    mlp_mult = _e("MLP_MULT", 2, int)
    tie_embeddings = _e("TIE_EMBEDDINGS", 1, int)
    rope_base = _e("ROPE_BASE", 10000.0, float)
    rope_type = _e("ROPE_TYPE", "rope")
    yarn_max_len = _e("YARN_MAX_LEN", 4096, int)
    logit_softcap = _e("LOGIT_SOFTCAP", 30.0, float)
    softcap_type = _e("SOFTCAP_TYPE", "poly")
    tied_embed_init_std = _e("TIED_EMBED_INIT_STD", 0.005, float)
    qk_gain_init = _e("QK_GAIN_INIT", 1.5, float)
    activation_type = _e("ACTIVATION", "swiglu")
    embed_dim = _e("EMBED_DIM", 0, int)
    bigram_hash = _e("BIGRAM_HASH", 0, bool)
    mtp_heads_count = _e("MTP_HEADS", 0, int)
    training_depth_recurrence = _e("TRAINING_DEPTH_RECURRENCE", 1, int)
    eval_depth_recurrence = _e("EVAL_DEPTH_RECURRENCE", 1, int)
    attn_proj_type = _e("ATTN_PROJ_TYPE", "standard")
    logit_head_type = _e("LOGIT_HEAD_TYPE", "standard")
    tversky_num_features = _e("TVERSKY_NUM_FEATURES", 16, int)
    tversky_feature_pools = _e("TVERSKY_FEATURE_POOLS", 0, int)
    tversky_membership = _e("TVERSKY_MEMBERSHIP", "sigmoid")
    diff_attn = _e("DIFF_ATTN", 0, bool)
    refiner = _e("REFINER", 0, bool)
    refiner_kernel = _e("REFINER_KERNEL", 3, int)
    mlp_groups = _e("MLP_GROUPS", 0, int)
    embed_lr = _e("EMBED_LR", 0.6, float)
    head_lr = _e("HEAD_LR", 0.008, float)
    adam_lr = _e("ADAM_LR", 1e-3, float)
    adam_wd = _e("ADAM_WD", 0.05, float)
    untie_at_fraction = _e("UNTIE_AT_FRACTION", 0.0, float)
    tied_embed_lr = _e("TIED_EMBED_LR", 0.05, float)
    corr_weight_lr = _e("CORR_WEIGHT_LR", 0.05, float)
    smear = _e("SMEAR", 0, bool)
    seq_len_start = _e("SEQ_LEN_START", 0, int)
    seq_schedule_fraction = _e("SEQ_SCHEDULE_FRACTION", 0.33, float)
    batch_tokens_start = _e("BATCH_TOKENS_START", 0, int)
    batch_schedule_fraction = _e("BATCH_SCHEDULE_FRACTION", 0.33, float)
    churn_log_every = _e("CHURN_LOG_EVERY", 500, int)
    matrix_lr = _e("MATRIX_LR", 0.04, float)
    scalar_lr = _e("SCALAR_LR", 0.04, float)
    muon_momentum = _e("MUON_MOMENTUM", 0.95, float)
    muon_backend_steps = _e("MUON_BACKEND_STEPS", 5, int)
    muon_wd = _e("MUON_WD", 0.0, float)
    matrix_optimizer = _e("MATRIX_OPTIMIZER", "muon")
    muon_momentum_warmup_start = _e("MUON_MOMENTUM_WARMUP_START", 0.85, float)
    muon_momentum_warmup_steps = _e("MUON_MOMENTUM_WARMUP_STEPS", 500, int)
    beta1 = _e("BETA1", 0.9, float)
    beta2 = _e("BETA2", 0.95, float)
    adam_eps = _e("ADAM_EPS", 1e-8, float)
    grad_clip_norm = _e("GRAD_CLIP_NORM", 0.0, float)
    bitnet_group_size = _e("BITNET_GROUP_SIZE", 64, int)
    sliding_eval = _e("SLIDING_EVAL", 0, bool)
    sliding_eval_stride = _e("SLIDING_EVAL_STRIDE", 64, int)
    sliding_batch_size = _e("SLIDING_BATCH_SIZE", 64, int)
    temp_scaling = _e("TEMP_SCALING", 0, bool)
    _fp_raw = os.environ.get("FP_STORAGE", "0")
    fp_storage = True if _fp_raw == "FP8" else ("fp4" if _fp_raw == "FP4" else False)

CTP = ("attn_scale","attn_scales","mlp_scale","mlp_scales","resid_mix","resid_mixes","q_gain","diff_lambda","skip_weight","skip_weights","vocab_bias","refiner.gate")

# ---------------------------------------------------------------------------
# Ternary packing — base-3 encoding (5 trits/byte)
# ---------------------------------------------------------------------------
def pack_ternary(q: Tensor):
    f = (q.reshape(-1).to(torch.int8) + 1).numpy()
    n = len(f)
    p = (5 - n % 5) % 5
    if p: f = np.concatenate([f, np.zeros(p, dtype=np.int8)])
    g = f.reshape(-1, 5).astype(np.uint8)
    return (g[:,0] + g[:,1]*3 + g[:,2]*9 + g[:,3]*27 + g[:,4]*81).tobytes(), n

def unpack_ternary(data: bytes, n: int) -> Tensor:
    v = np.frombuffer(data, dtype=np.uint8).astype(np.int16)
    t = np.zeros((len(v), 5), dtype=np.int8)
    for i in range(5): t[:,i] = v % 3; v //= 3
    return torch.from_numpy(t.reshape(-1)[:n].astype(np.int8) - 1)

def pack_ternary_bitmask(q: Tensor):
    f = q.reshape(-1).to(torch.int8).numpy(); n = len(f)
    nz = (f != 0)
    return np.packbits(nz).tobytes() + np.packbits(f[nz] > 0).tobytes(), n

def unpack_ternary_bitmask(data: bytes, n: int) -> Tensor:
    ms = (n + 7) // 8
    nz = np.unpackbits(np.frombuffer(data[:ms], dtype=np.uint8))[:n].astype(bool)
    s = np.unpackbits(np.frombuffer(data[ms:], dtype=np.uint8))[:int(nz.sum())].astype(bool)
    w = np.zeros(n, dtype=np.int8); w[nz] = np.where(s, 1, -1)
    return torch.from_numpy(w)

# ---------------------------------------------------------------------------
# FP4 quantization (per-row absmax, 2 values packed per byte)
# ---------------------------------------------------------------------------
def quantize_to_int4(t: Tensor) -> tuple[Tensor, Tensor, list]:
    t32 = t.float()
    orig_shape = t32.shape
    if t32.ndim < 2:
        t32 = t32.unsqueeze(0)
    absmax = t32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 7.0
    q = torch.clamp(torch.round(t32 / scale), -7, 7).to(torch.int8)
    flat = q.reshape(-1)
    if flat.numel() % 2 != 0:
        flat = F.pad(flat, (0, 1))
    low = (flat[0::2] + 8).to(torch.uint8)
    high = (flat[1::2] + 8).to(torch.uint8)
    return low | (high << 4), scale.half().squeeze(-1), list(orig_shape)

def dequantize_from_int4(packed: Tensor, scale: Tensor, shape: list) -> Tensor:
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    flat = torch.zeros(packed.numel() * 2, dtype=torch.int8)
    flat[0::2] = low
    flat[1::2] = high
    numel = 1
    for s in shape:
        numel *= s
    flat = flat[:numel].float()
    if len(shape) <= 1:
        return (flat * scale.float().squeeze()).reshape(shape)
    return (flat.reshape(-1, shape[-1]) * scale.float().unsqueeze(-1)).reshape(shape)

# ---------------------------------------------------------------------------
# State dict serialization (ternary + fp16/fp8/fp4)
# ---------------------------------------------------------------------------
def q_sd(state_dict: dict, group_size: int = 64, fp_storage=False, ternary_method="standard", ternary_override_names: set | None = None) -> tuple[dict, dict]:
    "Ternary for large 2D weight matrices, fp16/fp8/fp4 for everything else."
    quantized = {}
    stats = {"ternary_params": 0, "ternary_bytes": 0, "fp_params": 0, "fp_bytes": 0}
    for name, tensor in state_dict.items():
        if "mtp_heads" in name:
            continue
        t = tensor.detach().cpu().float().contiguous()
        t_orig_shape = list(t.shape)
        if t.ndim == 3:
            t = t.reshape(t.shape[0], -1)
        is_ternary_candidate = (
            t.ndim == 2 and t.numel() > 65_536
            and "tok_emb" not in name and "lm_head" not in name and "embed_proj" not in name and "bigram_emb" not in name and "lm_head_correction" not in name and "lm_head_U" not in name and "lm_head_V" not in name
            and "prototypes" not in name and "tversky" not in name
        ) or (ternary_override_names is not None and name in ternary_override_names)
        if is_ternary_candidate:
            pad = (group_size - t.shape[1] % group_size) % group_size
            t_padded = F.pad(t, (0, pad)) if pad > 0 else t
            t_grouped = t_padded.reshape(-1, group_size)
            scale = t_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
            q = (t_grouped / scale).round().clamp(-1, 1).to(torch.int8)

            if ternary_method == "standard":
                packed_bytes, n_trits = pack_ternary(q)
                entry_type = "ternary"
            else:
                packed_bytes, n_trits = pack_ternary_bitmask(q)
                entry_type = "ternary_bitmask"

            quantized[name] = {
                "type": entry_type, "packed": packed_bytes,
                "scale": scale.half().squeeze(-1),
                "shape": list(t.shape), "padded_cols": t_padded.shape[1],
                "group_size": group_size, "n_trits": n_trits,
                "orig_shape": t_orig_shape,
            }
            stats["ternary_params"] += t.numel()
            stats["ternary_bytes"] += len(packed_bytes) + scale.numel() * 2
        elif fp_storage == "fp4" and t.ndim == 2:
            packed, scale, orig_shape = quantize_to_int4(t)
            quantized[name] = {"type": "fp4", "packed": packed, "scale": scale, "shape": orig_shape}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += packed.numel() + scale.numel() * 2
        elif fp_storage and t.ndim == 2:
            quantized[name] = {"type": "fp8", "data": t.to(torch.float8_e4m3fn)}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel()
        else:
            quantized[name] = {"type": "fp16", "data": t.half()}
            stats["fp_params"] += t.numel()
            stats["fp_bytes"] += t.numel() * 2
    return quantized, stats

def deq_sd(quantized: dict, target_dtype=torch.bfloat16):
    "Reconstruct full-precision state dict from quantized representation."
    out = {}
    for name, entry in quantized.items():
        if entry["type"] in ("ternary", "ternary_bitmask"):
            if entry["type"] == "ternary":
                q = unpack_ternary(entry["packed"], entry["n_trits"])
            else:
                q = unpack_ternary_bitmask(entry["packed"], entry["n_trits"])

            q = q.float().reshape(-1, entry["group_size"])
            scale = entry["scale"].float().unsqueeze(-1)
            q_absmean = q.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            t = (q * (scale / q_absmean)).reshape(-1, entry["padded_cols"])
            shape = entry["shape"]
            result = t[:shape[0], :shape[1]].to(target_dtype)
            orig = entry.get("orig_shape")
            out[name] = result.reshape(orig).contiguous() if orig and orig != shape else result.contiguous()
        elif entry["type"] == "fp8":
            out[name] = entry["data"].to(torch.float32).to(target_dtype).contiguous()
        elif entry["type"] == "fp4":
            out[name] = dequantize_from_int4(entry["packed"], entry["scale"], entry["shape"]).to(target_dtype).contiguous()
        else:
            out[name] = entry["data"].to(target_dtype).contiguous()
    return out

# ---------------------------------------------------------------------------
# Ternary diagnostics (logged during training)
# ---------------------------------------------------------------------------
def tern_stats(model: nn.Module, group_size: int = 64):
    total = zeros = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and ("weight" in name or "prototypes" in name) and p.shape[0] > 1:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1)
                zeros += int((q == 0).sum().item())
                total += int(q.numel())
    return {"zero_frac": zeros / max(total, 1), "total_weights": total}

_prev_committed: dict = {}

def churn_fn(model: nn.Module, group_size: int = 64):
    global _prev_committed
    total = flipped = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and ("weight" in name or "prototypes" in name) and p.shape[0] > 1:
                w = p.detach().float().reshape(-1, group_size)
                scale = w.abs().mean(-1, keepdim=True).clamp(min=1e-8).half().float()
                q = (w / scale).round().clamp(-1, 1).cpu().numpy()
                if name in _prev_committed:
                    flipped += int(np.sum(q != _prev_committed[name]))
                    total += q.size
                _prev_committed[name] = q
    return flipped / max(total, 1)

# ---------------------------------------------------------------------------
# Muon optimizer (Newton-Schulz orthogonalized momentum)
# ---------------------------------------------------------------------------
def ns_orth(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, wd: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, wd=wd))

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
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
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
                    g = F.rms_norm(g.float(), (g.size(-1),)).bfloat16()
                    g = ns_orth(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("wd", 0.0)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def ld_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = ld_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = ld_shard(self.files[self.file_idx])
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
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start:start + per_rank_span].pin_memory().to(self.device, non_blocking=True).to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x, y

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

def apply_qat_ste(w: Tensor, fp_storage: str | bool) -> Tensor:
    """Applies Straight-Through Estimator (STE) for FP4 or FP8 simulated quantization."""
    if not fp_storage:
        return w
    if fp_storage == "fp4":
        absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 7.0
        q = torch.clamp(torch.round(w / scale), -7.0, 7.0)
        w_sim = q * scale
        return (w_sim - w).detach() + w
    elif fp_storage is True or fp_storage == "fp8":
        w_sim = w.to(torch.float8_e4m3fn).to(w.dtype)
        return (w_sim - w).detach() + w
    return w

class QATLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, fp_storage: str | bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.fp_storage = fp_storage

    def forward(self, x: Tensor) -> Tensor:
        w_qat = apply_qat_ste(self.weight, self.fp_storage)
        return F.linear(x, w_qat.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class QATEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, fp_storage: str | bool = False):
        super().__init__(num_embeddings, embedding_dim)
        self.fp_storage = fp_storage

    def forward(self, input: Tensor) -> Tensor:
        w_qat = apply_qat_ste(self.weight, self.fp_storage)
        return F.embedding(input, w_qat, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)

class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, group_size=64):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.bfloat16()
        g = self.group_size
        w_g = w.reshape(-1, g)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_g / scale).round().clamp(-1, 1)
        w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
        return F.linear(x, w_ternary,
                        self.bias.to(x.dtype) if self.bias is not None else None)


class NormedTernaryLinear(TernaryLinear):
    "Ternary linear with RMSNorm on input — for output projections receiving un-normalized activations."
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.rms_norm(x, (x.size(-1),)))

class GroupedTernaryLinear(nn.Module):
    "Grouped linear with ternary STE. Weight stored as 2D [groups*group_out, group_in] for ternary quantization compatibility."
    def __init__(self, in_features, out_features, groups=4, group_size=64, normed=False):
        super().__init__()
        assert in_features % groups == 0 and out_features % groups == 0
        self.groups = groups
        self.group_in = in_features // groups
        self.group_out = out_features // groups
        self.group_size = group_size
        self.normed = normed
        self.weight = nn.Parameter(torch.randn(groups * self.group_out, self.group_in) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        if self.normed:
            x = F.rms_norm(x, (x.size(-1),))
        w = self.weight.bfloat16()
        g = self.group_size
        w_g = w.reshape(-1, g)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_g / scale).round().clamp(-1, 1)
        w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
        w_grouped = w_ternary.reshape(self.groups, self.group_out, self.group_in)
        bsz = x.shape[:-1]
        x_g = x.reshape(*bsz, self.groups, self.group_in)
        out = torch.einsum('...gi,goi->...go', x_g, w_grouped)
        return out.reshape(*bsz, self.groups * self.group_out)

class TverskyProjection(nn.Module):
    "Tversky similarity: S = θ·f(A∩B) - α·f(A\\B) - β·f(B\\A). Three modes."
    def __init__(self, in_features: int, out_features: int, num_features: int = 16,
                 group_size: int = 64, use_shared_features: bool = False,
                 membership: str = "sigmoid"):
        super().__init__()
        self.group_size = group_size
        self.num_features = num_features
        self.membership_type = membership
        self.no_features_mode = (num_features == 0)

        if not self.no_features_mode and not use_shared_features:
            self.features = nn.Parameter(torch.empty(num_features, in_features).uniform_(-0.02, 0.02))
        else:
            self.register_parameter('features', None)

        self.prototypes = nn.Parameter(torch.empty(out_features, in_features).uniform_(-0.02, 0.02))
        self.theta = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def _ternary_ste(self, w: Tensor) -> Tensor:
        w_bf16 = w.bfloat16()
        g = self.group_size
        w_grouped = w_bf16.reshape(-1, g)
        scale = w_grouped.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_grouped / scale).round().clamp(-1, 1)
        w_ternary = w_bf16 + ((q * scale).reshape(w_bf16.shape) - w_bf16).detach()
        return w_ternary.reshape(w.shape)

    def _membership(self, t: Tensor) -> Tensor:
        if self.membership_type == "poly":
            return torch.clamp(t * 5.0 / 4.0 + 0.5, 0.0, 1.0)
        elif self.membership_type == "tanh":
            return (torch.tanh(t * 5.0) + 1.0) * 0.5
        else:
            return torch.sigmoid(t * 5.0)

    def forward(self, x: Tensor, shared_features: Tensor | None = None) -> Tensor:
        proto = self._ternary_ste(self.prototypes)

        if self.no_features_mode:
            # NoFeatures: prototypes are their own feature universe
            x_f = x @ proto.t()                          # [B, S, out]
            p_norm = F.normalize(proto, dim=-1)
            p_f = p_norm @ p_norm.t()                    # [out, out]
        else:
            feat = (shared_features if shared_features is not None else self.features).float()
            x_f = x @ feat.t()                           # [B, S, nf]
            p_f = proto @ feat.t()                       # [out, nf]

        x_s = self._membership(x_f)
        p_s = self._membership(p_f)
        x_a = x_f * x_s
        p_a = p_f * p_s

        t, a, b = self.theta.abs(), self.alpha.abs(), self.beta.abs()
        return t * (x_a @ p_a.t()) - a * (x_a @ (1 - p_s).t()) - b * ((1 - x_s) @ p_a.t())

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CTP)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, no_cache: bool = False,
                 rope_type: str = "rope", yarn_max_len: int = 4096, train_seq_len: int = 1024):
        super().__init__()
        self.no_cache = no_cache
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if rope_type == "yarn":
            scale = train_seq_len / yarn_max_len
            freq_idx = torch.arange(0, dim, 2, dtype=torch.float32)
            ramp = torch.clamp((freq_idx / dim - 0.25) / 0.75, 0.0, 1.0)
            inv_freq = inv_freq / (ramp * (1.0 / scale - 1.0) + 1.0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len, device, dtype):
        if self.no_cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            return freqs.cos()[None, :, None, :].to(dtype=dtype), freqs.sin()[None, :, None, :].to(dtype=dtype)
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 group_size=64, attn_proj_type="standard", tversky_num_features=16,
                 tversky_feature_pools=0, no_cache=False, rope_type="rope",
                 yarn_max_len=4096, train_seq_len=1024, tversky_membership="sigmoid",
                 diff_attn=False):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.diff_attn = diff_attn
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.c_qkv = TernaryLinear(dim, self.q_size + 2 * self.kv_size, bias=False, group_size=group_size)
        self.proj = NormedTernaryLinear(dim, dim, bias=False, group_size=group_size) if attn_proj_type != "tversky" else None
        if self.proj is not None:
            self.proj._zero_init = True
        self.tversky_proj = TverskyProjection(
            dim, dim, num_features=tversky_num_features, group_size=group_size,
            use_shared_features=(tversky_feature_pools > 0),
            membership=tversky_membership,
        ) if attn_proj_type == "tversky" else None
        self.shared_features = None
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        if diff_attn:
            self.diff_lambda = nn.Parameter(torch.full((num_heads,), 0.5, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, no_cache=no_cache,
                             rope_type=rope_type, yarn_max_len=yarn_max_len,
                             train_seq_len=train_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        qkv_out = self.c_qkv(x)
        q_out, k_out, v_out = qkv_out.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q_out.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if self.diff_attn:
            half = self.head_dim // 2
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k[..., :half], k[..., half:]
            v1, v2 = v[..., :half], v[..., half:]
            y1 = flash_attn_func(q1.contiguous(), k1.contiguous(), v1.contiguous(), causal=True)
            y2 = flash_attn_func(q2.contiguous(), k2.contiguous(), v2.contiguous(), causal=True)
            lam = self.diff_lambda.to(dtype=y1.dtype)[None, None, :, None]
            y = torch.cat([y1 - lam * y2, y1 + lam * y2], dim=-1)
        else:
            y = flash_attn_func(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                causal=True
            )
        y = y.reshape(bsz, seqlen, dim)
        return self.tversky_proj(y, self.shared_features) if self.tversky_proj is not None else self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, group_size=64, activation="swiglu", mlp_groups=0):
        super().__init__()
        hidden = mlp_mult * dim
        self.activation = activation
        if mlp_groups > 0:
            if activation == "swiglu":
                self.gate_up = GroupedTernaryLinear(dim, hidden * 2, groups=mlp_groups, group_size=group_size)
            else:
                self.fc = GroupedTernaryLinear(dim, hidden, groups=mlp_groups, group_size=group_size)
            self.proj = GroupedTernaryLinear(hidden, dim, groups=mlp_groups, group_size=group_size, normed=True)
        else:
            if activation == "swiglu":
                self.gate_up = TernaryLinear(dim, hidden * 2, bias=False, group_size=group_size)
            else:
                self.fc = TernaryLinear(dim, hidden, bias=False, group_size=group_size)
            self.proj = NormedTernaryLinear(hidden, dim, bias=False, group_size=group_size)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "swiglu":
            gu = self.gate_up(x)
            gate, up = gu.chunk(2, dim=-1)
            return self.proj(F.silu(gate) * up)
        elif self.activation == "relu":
            return self.proj(torch.relu(self.fc(x)))
        elif self.activation == "leaky_relu":
            return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.01))
        else:  # relu2
            return self.proj(torch.relu(self.fc(x)).square())

class SmearModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        cumsum = x.cumsum(dim=1)
        counts = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        smeared = cumsum / counts
        gate = torch.tanh(self.gate.to(dtype=x.dtype))
        return x + gate * (smeared - x)


class CausalConvRefiner(nn.Module):
    "Causal Conv1d that refines hidden states using local n-gram context."
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=0, bias=False)
        self.gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        h = x.permute(0, 2, 1)  # [B, D, S]
        h = F.pad(h, (self.kernel_size - 1, 0))  # causal pad
        h = self.conv(h)
        h = h.permute(0, 2, 1)  # [B, S, D]
        return x + torch.tanh(self.gate.to(dtype=x.dtype)) * F.rms_norm(h, (h.size(-1),))

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, group_size: int=64,
                 activation: str="swiglu", attn_proj_type: str="standard",
                 tversky_num_features: int=16, tversky_feature_pools: int=0, no_cache: bool=False,
                 smear: bool=False, rope_type: str="rope", yarn_max_len: int=4096,
                 train_seq_len: int=1024, tversky_membership: str="sigmoid",
                 diff_attn: bool=False, mlp_groups: int=0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        group_size, attn_proj_type, tversky_num_features,
                                        tversky_feature_pools, no_cache, rope_type, yarn_max_len,
                                        train_seq_len, tversky_membership, diff_attn)
        self.mlp = MLP(dim, mlp_mult, group_size, activation, mlp_groups)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.smear = SmearModule(dim) if smear else None

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0
        n = self.attn_norm(x)
        x = x + self.attn_scale.to(dtype=x.dtype) * self.attn(n)
        x = x + self.mlp_scale.to(dtype=x.dtype) * self.mlp(self.mlp_norm(x))
        if self.smear is not None:
            x = self.smear(x)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init,
                 group_size: int = 64, activation: str = "swiglu", mtp_heads_count: int = 0,
                 embed_dim: int = 0, attn_proj_type: str = "standard", logit_head_type: str = "standard",
                 tversky_num_features: int = 16, tversky_feature_pools: int = 0,
                 training_depth_recurrence: int=1, fp_storage=False, bigram_hash: bool=False,
                 softcap_type: str="poly", no_cache: bool=False,
                 smear: bool=False, rope_type: str="rope", yarn_max_len: int=4096,
                 train_seq_len: int=1024, tversky_membership: str="sigmoid",
                 diff_attn=False, mlp_groups=0, refiner=False, refiner_kernel=3):
        super().__init__()
        self.training_depth_recurrence = training_depth_recurrence
        self.fp_storage = fp_storage
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.softcap_type = softcap_type
        self.embed_dim = embed_dim if embed_dim > 0 else model_dim
        self.tok_emb = QATEmbedding(vocab_size, self.embed_dim, fp_storage=fp_storage)
        self.bigram_emb = QATEmbedding(vocab_size, self.embed_dim, fp_storage=fp_storage) if bigram_hash else None
        if self.bigram_emb is not None:
            nn.init.zeros_(self.bigram_emb.weight)
        self.lm_head_correction = nn.Parameter(
            torch.zeros(vocab_size, self.embed_dim)) if tie_embeddings == 2 else None
        self.embed_proj = QATLinear(self.embed_dim, model_dim, bias=False, fp_storage=fp_storage) if self.embed_dim != model_dim else None
        self.embed_proj_rev = QATLinear(model_dim, self.embed_dim, bias=False, fp_storage=fp_storage) if (
            self.embed_dim != model_dim and logit_head_type != "tversky") else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Shared Tversky feature pools (if enabled and num_features > 0)
        if attn_proj_type == "tversky" and tversky_feature_pools > 0 and tversky_num_features > 0:
            self.tversky_feature_pools_list = nn.ParameterList([
                nn.Parameter(torch.empty(tversky_num_features, model_dim).uniform_(-0.02, 0.02))
                for _ in range(tversky_feature_pools)
            ])
        else:
            self.tversky_feature_pools_list = None

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  group_size, activation, attn_proj_type, tversky_num_features, tversky_feature_pools,
                  no_cache, smear, rope_type, yarn_max_len, train_seq_len, tversky_membership,
                  diff_attn, mlp_groups)
            for _ in range(num_layers)
        ])

        # Inject shared feature pool references into attention layers
        if self.tversky_feature_pools_list is not None:
            for i, block in enumerate(self.blocks):
                pool_idx = (i * tversky_feature_pools) // num_layers
                block.attn.shared_features = self.tversky_feature_pools_list[pool_idx]

        self.final_norm = RMSNorm()
        self.refiner = CausalConvRefiner(model_dim, kernel_size=refiner_kernel) if refiner else None
        self.mtp_heads = nn.ModuleList([
            nn.Linear(model_dim, vocab_size, bias=False) for _ in range(mtp_heads_count)
        ])
        for h in self.mtp_heads:
            nn.init.zeros_(h.weight)
        self.logit_head_type = logit_head_type
        if logit_head_type == "tversky" and tversky_num_features == 0 and vocab_size > 1024:
            raise ValueError(
                f"Tversky logit head with no-features mode creates O(V^2) = {vocab_size}x{vocab_size} "
                f"matrix per forward pass. Use tversky_num_features > 0 or a smaller vocab."
            )
        self.tversky_head = TverskyProjection(
            model_dim, vocab_size, num_features=tversky_num_features,
            membership=tversky_membership,
        ) if logit_head_type == "tversky" else None
        self.lm_head = QATLinear(model_dim, vocab_size, bias=False, fp_storage=fp_storage)
        self.lm_head._zero_init = True
        if self.lm_head is not None and (tie_embeddings or logit_head_type == "tversky"):
            self.lm_head.weight.requires_grad_(False)

        self.vocab_bias = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))
        self._init_weights(tied_embed_init_std)

    def _init_weights(self, tied_embed_init_std: float) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, TernaryLinear) and not getattr(module, "_zero_init", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _compute_logits(self, x: Tensor) -> Tensor:
        if self.tversky_head is not None:
            logits_raw = self.tversky_head(x)
        elif self.tie_embeddings:
            if self.embed_proj_rev is not None:
                proj = self.embed_proj_rev(x)
            else:
                proj = x
            weight = self.tok_emb.weight
            if self.lm_head_correction is not None:
                weight = weight + self.lm_head_correction
            logits_raw = F.linear(proj, weight.to(x.dtype))
        else:
            logits_raw = self.lm_head(x)
        return logits_raw + self.vocab_bias.to(x.dtype)

    def _softcap(self, logits: Tensor) -> Tensor:
        s = self.logit_softcap
        if self.softcap_type == "tanh":
            return s * torch.tanh(logits / s)
        x_sc = torch.clamp(logits / s, -2.0, 2.0)
        x2 = x_sc * x_sc
        return s * torch.clamp(x_sc * (1.0 - x2 / 3.0 + x2 * x2 / 15.0), -1.0, 1.0)

    def forward(self, input_ids: Tensor, target_ids: Tensor, reduction: str = "mean", temperature: float = 1.0) -> Tensor:
        x = self.tok_emb(input_ids).float()
        if self.bigram_emb is not None:
            prev = F.pad(input_ids[:, :-1], (1, 0), value=0)
            x = x + self.bigram_emb(prev).float()
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # U-Net style encoder/decoder with skip connections
        skips = []
        for i in range(self.num_encoder_layers):
            for _ in range(max(1, self.training_depth_recurrence)):
                x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype) * skips.pop()
            for _ in range(max(1, self.training_depth_recurrence)):
                x = self.blocks[bi](x, x0)

        x_normed = self.final_norm(x)
        if self.refiner is not None:
            x_normed = self.refiner(x_normed)

        # Standard training/eval path
        x_flat = x_normed.reshape(-1, x_normed.size(-1))
        targets = target_ids.reshape(-1)
        logits = self._softcap(self._compute_logits(x_flat))

        if reduction == "none":
            return F.cross_entropy(logits.float(), targets, reduction="none").reshape(input_ids.shape)

        # Fused CE + Z-loss: single logsumexp computation
        logits_f = logits.float()
        lse = torch.logsumexp(logits_f, dim=-1)
        target_logits = logits_f.gather(1, targets.unsqueeze(1)).squeeze(1)
        main_loss = (lse - target_logits).mean() + 1e-4 * (lse ** 2).mean()

        # Multi-token prediction auxiliary loss (training only)
        if self.training and len(self.mtp_heads) > 0:
            mtp_loss = torch.zeros((), device=main_loss.device)
            for k, head in enumerate(self.mtp_heads):
                shift = k + 2
                if target_ids.shape[1] > shift:
                    mtp_tgt = target_ids[:, shift:].reshape(-1)
                    mtp_in = x_normed[:, :target_ids.shape[1] - shift, :].reshape(-1, x_normed.shape[-1])
                    mtp_loss = mtp_loss + F.cross_entropy(head(mtp_in).float(), mtp_tgt, reduction="mean")
            main_loss = main_loss + 0.1 * mtp_loss / len(self.mtp_heads)
        return main_loss

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def build_luts(sp, vocab_size: int, device: torch.device):
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

def ld_val(pattern, seq_len, max_tok=int(os.environ.get("VAL_MAX_TOKENS", 500000))):
    files = sorted(glob.glob(pattern))
    assert files, f"No files: {pattern}"
    tok = torch.cat([ld_shard(Path(p)) for p in files]).contiguous()
    if max_tok > 0: tok = tok[:max_tok + 1]
    u = ((tok.numel() - 1) // seq_len) * seq_len
    return tok[:u + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, temperature: float = 1.0):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = max(1, local_batch_tokens // args.train_seq_len)
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, seq_end)
            raw_start = batch_start * args.train_seq_len
            raw_end = batch_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y, temperature=temperature).detach()
            n = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * n
            token_count += n
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            tok_bytes = base_bytes_lut[tgt_ids].to(torch.int16)
            tok_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            byte_count += tok_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = loss_sum / token_count
    bpb = (val_loss.item() / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return float(val_loss.item()), float(bpb)

def eval_val_sliding(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride: int = 64, temperature: float = 1.0):
    seq_len = args.train_seq_len
    batch_size = args.sliding_batch_size
    total_tokens = val_tokens.numel() - 1
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    all_starts = list(range(0, total_tokens - seq_len, stride))
    my_starts = all_starts[rank::world_size]

    model.eval()
    with torch.inference_mode():
        for i in range(0, len(my_starts), batch_size):
            batch_starts = my_starts[i:i + batch_size]

            starts_t = torch.tensor(batch_starts, dtype=torch.int64)
            offsets = torch.arange(seq_len + 1, dtype=torch.int64)
            indices = starts_t.unsqueeze(1) + offsets.unsqueeze(0)

            local_batch = val_tokens[indices].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local_batch[:, :-1]
            y = local_batch[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_token_loss = model(x, y, reduction="none", temperature=temperature).detach()

            for b, start in enumerate(batch_starts):
                score_from = 0 if start == 0 else seq_len - stride
                scored = per_token_loss[b, score_from:]
                sx, sy = x[b, score_from:], y[b, score_from:]

                loss_sum += scored.to(torch.float64).sum()
                token_count += scored.numel()

                tok_bytes = base_bytes_lut[sy].to(torch.int16)
                tok_bytes += (has_leading_space_lut[sy] & ~is_boundary_token_lut[sx]).to(torch.int16)
                byte_count += tok_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bpb = (val_loss.item() / math.log(2.0)) * (token_count.item() / byte_count.item())
    model.train()
    return float(val_loss.item()), float(bpb)

# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def find_temp(args, base_model, rank, world_size, device, grad_accum_steps,
                              calibration_tokens, base_bytes_lut, has_leading_space_lut,
                              is_boundary_token_lut):
    best_t, best_loss = 1.0, float("inf")
    for t in [0.90, 0.95, 1.00, 1.05, 1.10]:
        loss, _ = eval_val(args, base_model, rank, world_size, device, grad_accum_steps,
                           calibration_tokens, base_bytes_lut, has_leading_space_lut,
                           is_boundary_token_lut, temperature=t)
        if loss < best_loss:
            best_loss = loss
            best_t = t
    return best_t

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main() -> None:
    args = Hyperparameters()
    code = Path(__file__).read_text(encoding="utf-8")

    if args.matrix_optimizer != "adamw":
        global ns_orth
        ns_orth = torch.compile(ns_orth)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, 8 // world_size)
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

    os.makedirs("logs/cuda/", exist_ok=True)
    logfile = f"logs/cuda/{args.run_id}.txt" if master_process else None
    if master_process:
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)

    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = ld_val(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_luts(
        sp, args.vocab_size, device)

    # --- Model ---
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        group_size=args.bitnet_group_size, activation=args.activation_type, mtp_heads_count=args.mtp_heads_count,
        embed_dim=args.embed_dim, attn_proj_type=args.attn_proj_type, logit_head_type=args.logit_head_type,
        tversky_num_features=args.tversky_num_features, tversky_feature_pools=args.tversky_feature_pools,
        training_depth_recurrence=args.training_depth_recurrence, fp_storage=args.fp_storage,
        bigram_hash=args.bigram_hash, softcap_type=args.softcap_type, no_cache=(args.compile_mode == "reduce-overhead"),
        smear=args.smear, rope_type=args.rope_type, yarn_max_len=args.yarn_max_len, train_seq_len=args.train_seq_len,
        tversky_membership=args.tversky_membership, diff_attn=args.diff_attn,
        refiner=args.refiner, refiner_kernel=args.refiner_kernel, mlp_groups=args.mlp_groups,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, nn.Linear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if base_model.lm_head is not None and (args.tie_embeddings or args.logit_head_type == "tversky"):
        base_model.lm_head.weight.requires_grad_(False)

    torch._dynamo.config.optimize_ddp = False

    compiled_model = torch.compile(base_model, mode=args.compile_mode if args.compile_mode != "default" else None)
    use_find_unused = args.untie_at_fraction > 0 or args.mtp_heads_count > 0 or not args.tie_embeddings
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False,
                find_unused_parameters=use_find_unused,
                static_graph=not use_find_unused,
                gradient_as_bucket_view=True) if distributed else compiled_model

    # --- Optimizers ---
    _excl = {"tok_emb.weight", "lm_head.weight", "lm_head_correction"}
    all_other_params = [(n, p) for n, p in base_model.named_parameters()
                        if not any(eh in n for eh in _excl)]
    matrix_params = [p for n, p in all_other_params
                     if p.ndim == 2 and not any(pat in n for pat in CTP)]
    scalar_params = [p for n, p in all_other_params
                     if p.ndim < 2 or any(pat in n for pat in CTP)]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    if args.matrix_optimizer == "adamw":
        opt_muon = torch.optim.AdamW(
            [{"params": matrix_params, "lr": args.adam_lr, "base_lr": args.adam_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    else:
        opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                        backend_steps=args.muon_backend_steps, wd=args.muon_wd)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_head = torch.optim.Adam(
        [{"params": [base_model.lm_head.weight], "lr": 0.0, "base_lr": 0.0}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)

    optimizers = [opt for opt in [opt_tok, opt_muon, opt_scalar, opt_head] if opt is not None]

    if base_model.lm_head_correction is not None:
        opt_corr = torch.optim.Adam(
            [{"params": [base_model.lm_head_correction],
              "lr": args.corr_weight_lr, "base_lr": args.corr_weight_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.append(opt_corr)

    # --- Log all hyperparameters ---
    log0("--- Hyperparameters ---", console=False)
    log0(" ".join(f"{a}={getattr(args,a)}" for a in sorted(dir(args)) if not a.startswith("_") and a not in ("train_files","val_files") and not callable(getattr(args,a))), console=False)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"params:{n_params} L:{args.num_layers} d:{args.model_dim} h:{args.num_heads} kv:{args.num_kv_heads} ws:{world_size} ga:{grad_accum_steps} s:{args.seed}")

    # --- Data loader & helpers ---
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float):
        if args.warmdown_fraction <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = int(args.iterations * (1.0 - args.warmdown_fraction))
            return max((args.iterations - step) / max(args.iterations * args.warmdown_fraction, 1), 0.0) if step >= warmdown_start else 1.0
        warmdown_ms = max_wallclock_ms * args.warmdown_fraction
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    _seq_switched = False
    _batch_switched = False
    active_seq_len = args.seq_len_start if args.seq_len_start > 0 else args.train_seq_len
    active_batch_tokens = args.batch_tokens_start if args.batch_tokens_start > 0 else args.train_batch_tokens

    # --- Compiler warmup ---
    if args.warmup_steps > 0:
        _ms = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        _os = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for mi in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = mi == grad_accum_steps - 1
                x, y = train_loader.next_batch(active_batch_tokens, active_seq_len, grad_accum_steps)
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(x, y)
                (loss * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            log0(f"warmup:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(_ms, strict=True)
        for o, s in zip(optimizers, _os): o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # --- Main training loop ---
    training_time_ms = 0.0
    stop_after_step: int | None = None
    _untied = False
    train_loss = torch.zeros((), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                         val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            tstats = tern_stats(base_model, group_size=args.bitnet_group_size)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms zero_frac:{tstats['zero_frac']:.3f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Sequence length schedule
        if args.seq_len_start > 0 and not _seq_switched:
            if max_wallclock_ms is not None:
                should_switch_seq = elapsed_ms >= args.seq_schedule_fraction * max_wallclock_ms
            else:
                should_switch_seq = step >= int(args.iterations * args.seq_schedule_fraction)
            if should_switch_seq:
                active_seq_len = args.train_seq_len
                _seq_switched = True
                torch._dynamo.reset()
                train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
                log0(f"step:{step} seq_len_switch:{args.seq_len_start}->{active_seq_len}")

        # Batch size schedule
        if args.batch_tokens_start > 0 and not _batch_switched:
            if max_wallclock_ms is not None:
                should_switch_batch = elapsed_ms >= args.batch_schedule_fraction * max_wallclock_ms
            else:
                should_switch_batch = step >= int(args.iterations * args.batch_schedule_fraction)
            if should_switch_batch:
                active_batch_tokens = args.train_batch_tokens
                _batch_switched = True
                log0(f"step:{step} batch_switch:{args.batch_tokens_start}->{active_batch_tokens}")

        zero_grad_all()
        train_loss.zero_()

        for micro in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(active_batch_tokens, active_seq_len, grad_accum_steps)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss.add_(loss.detach())
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Untie lm_head at configured fraction of training
        if args.untie_at_fraction > 0:
            if max_wallclock_ms is not None:
                should_untie = not _untied and elapsed_ms >= args.untie_at_fraction * max_wallclock_ms
            else:
                should_untie = not _untied and step >= int(args.iterations * args.untie_at_fraction)
            if should_untie and base_model.tie_embeddings:
                with torch.no_grad():
                    base_weight = base_model.tok_emb.weight.float()
                    if base_model.lm_head_correction is not None:
                        base_weight = base_weight + base_model.lm_head_correction.float()
                    if base_model.embed_proj_rev is not None:
                        full_weight = base_weight @ base_model.embed_proj_rev.weight.float()
                    else:
                        full_weight = base_weight
                    base_model.lm_head.weight.copy_(full_weight)
                base_model.tie_embeddings = False
                base_model.lm_head.weight.requires_grad_(True)
                for g in opt_head.param_groups:
                    g["lr"] = g["base_lr"] = args.head_lr
                _untied = True
                torch._dynamo.reset()
                log0(f"step:{step} untied lm_head (head_lr={args.head_lr})")

        # Muon momentum warmup
        if args.matrix_optimizer != "adam":
            frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
            for g in opt_muon.param_groups:
                g["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum

        # LR scheduling
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
            opt.step()
        zero_grad_all()
        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        
        if args.train_log_every > 0 and step % args.train_log_every == 0:
            log0(f"step:{step}/{args.iterations} loss:{train_loss.item():.4f} t:{approx_ms:.0f}ms avg:{approx_ms/step:.1f}ms")
        if args.churn_log_every > 0 and step % args.churn_log_every == 0:
            log0(f"step:{step} churn:{churn_fn(base_model, args.bitnet_group_size):.4f} zero:{tern_stats(base_model, args.bitnet_group_size)['zero_frac']:.3f}")

        # Wallclock cap sync
        if stop_after_step is None and max_wallclock_ms is not None and step % 10 == 0:
            reached_cap = approx_ms >= max_wallclock_ms
            if distributed:
                cap_t = torch.tensor(int(reached_cap), device=device)
                dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
                reached_cap = bool(cap_t.item())
            if reached_cap:
                stop_after_step = step

    # --- Serialization ---
    if master_process:
        sd = base_model.state_dict()
        if base_model.tie_embeddings or args.logit_head_type == "tversky":
            sd.pop("lm_head.weight", None)

        # Compute ternary overrides for no-features Tversky prototypes
        ternary_overrides = set()
        for n, m in base_model.named_modules():
            if isinstance(m, TverskyProjection) and m.no_features_mode:
                ternary_overrides.add(n + ".prototypes")
        ternary_overrides = ternary_overrides or None

        # Two methods: Standard Base-3 vs Bitmask Mapping
        methods = {}
        for method in ("standard", "bitmask"):
            q_obj, stats = q_sd(sd, group_size=args.bitnet_group_size, fp_storage=args.fp_storage, ternary_method=method, ternary_override_names=ternary_overrides)
            buf = io.BytesIO()
            torch.save(q_obj, buf)
            methods[method] = {"blob": lzma.compress(buf.getvalue(), preset=9), "stats": stats}
        best = min(methods, key=lambda m: len(methods[m]["blob"]))
        final_blob, q_stats = methods[best]["blob"], methods[best]["stats"]
        with open("final_model.ternary.ptz", "wb") as f:
            f.write(final_blob)

        artifact_bytes = len(final_blob)
        code_bytes = len(code.encode("utf-8"))

        total = artifact_bytes + code_bytes
        log0(f"artifact:{artifact_bytes/1e6:.2f}MB ternary:{q_stats['ternary_params']}({q_stats['ternary_bytes']}B) fp:{q_stats['fp_params']}({q_stats['fp_bytes']}B) code:{code_bytes}")
        log0(f"budget:{total}/{16000000} ({total/1e6:.2f}/{16.00:.2f}MB) {'FITS' if total <= 16000000 else 'OVER'}")

        if args.eval_depth_recurrence > 0:
            base_model.training_depth_recurrence = args.eval_depth_recurrence
            log0(f"eval_depth_recurrence:{args.eval_depth_recurrence}")

    # --- All ranks load roundtrip weights and evaluate ---
    if distributed:
        dist.barrier()

    with open("final_model.ternary.ptz", "rb") as f:
        loaded = torch.load(io.BytesIO(lzma.decompress(f.read())), map_location="cpu", weights_only=False)
    base_model.load_state_dict(deq_sd(loaded), strict=False)
    torch._dynamo.reset()

    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                     val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"final_ternary_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")

    opt_temp = 1.0
    if args.temp_scaling:
        torch.cuda.synchronize()
        t_temp = time.perf_counter()
        calibration_tokens = train_loader.stream.take(65536).to(device)
        opt_temp = find_temp(args, base_model, rank, world_size, device, grad_accum_steps,
                                            calibration_tokens, base_bytes_lut, has_leading_space_lut,
                                            is_boundary_token_lut)
        torch.cuda.synchronize()
        temp_time_ms = 1000.0 * (time.perf_counter() - t_temp)
        log0(f"temp_scaling optimal_T:{opt_temp:.2f} eval_time:{temp_time_ms:.0f}ms")

    if args.sliding_eval:
        torch.cuda.synchronize()
        t_sliding = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(args, base_model, rank, world_size, device, grad_accum_steps,
                                           val_tokens, base_bytes_lut, has_leading_space_lut,
                                           is_boundary_token_lut, stride=args.sliding_eval_stride,
                                           temperature=opt_temp)
        torch.cuda.synchronize()
        sliding_time_ms = 1000.0 * (time.perf_counter() - t_sliding)
        log0(f"final_sliding val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} "
             f"(stride={args.sliding_eval_stride}, T={opt_temp:.2f}) eval_time:{sliding_time_ms:.0f}ms")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()