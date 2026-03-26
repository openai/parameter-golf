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
    data_path = _e("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = _e("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", f"run_{int(time.time())}")
    seed = _e("SEED", 1337, int)
    compile_mode = _e("COMPILE_MODE", "default")
    val_batch_size = _e("VAL_BATCH_SIZE", 524288, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 0, int)
    train_log_every = _e("TRAIN_LOG_EVERY", 1000, int)
    iterations = _e("ITERATIONS", 10000, int)
    warmdown_fraction = _e("WARMDOWN_FRACTION", 0.5, float)
    warmup_steps = _e("WARMUP_STEPS", 5, int)
    train_batch_tokens = _e("TRAIN_BATCH_TOKENS", 786432, int)
    train_seq_len = _e("TRAIN_SEQ_LEN", 2048, int)
    max_wallclock_seconds = _e("MAX_WALLCLOCK_SECONDS", 599.0, float)
    vocab_size = _e("VOCAB_SIZE", 8192, int)
    num_layers = _e("NUM_LAYERS", 12, int)
    num_kv_heads = _e("NUM_KV_HEADS", 4, int)
    model_dim = _e("MODEL_DIM", 768, int)
    num_heads = _e("NUM_HEADS", 8, int)
    mlp_mult = _e("MLP_MULT", 4, int)
    tie_embeddings = _e("TIE_EMBEDDINGS", 1, int)
    rope_base = _e("ROPE_BASE", 5000.0, float)
    rope_type = _e("ROPE_TYPE", "yarn")
    yarn_max_len = _e("YARN_MAX_LEN", 4096, int)
    logit_softcap = _e("LOGIT_SOFTCAP", 30.0, float)
    softcap_type = _e("SOFTCAP_TYPE", "poly")
    tied_embed_init_std = _e("TIED_EMBED_INIT_STD", 0.005, float)
    qk_gain_init = _e("QK_GAIN_INIT", 2.25, float)
    activation_type = _e("ACTIVATION", "lrelu2")
    leaky_relu_slope = _e("LEAKY_RELU_SLOPE", 0.5, float)
    embed_dim = _e("EMBED_DIM", 254, int)
    training_depth_recurrence = _e("TRAINING_DEPTH_RECURRENCE", 0, int)
    eval_depth_recurrence = _e("EVAL_DEPTH_RECURRENCE", 0, int)
    feedback_enabled = _e("FEEDBACK_ENABLED", 1, bool)
    feedback_dim = _e("FEEDBACK_DIM", 64, int)
    feedback_sketch_tokens = _e("FEEDBACK_SKETCH_TOKENS", 4, int)
    feedback_replay = _e("FEEDBACK_REPLAY", "decoder")
    feedback_target = _e("FEEDBACK_TARGET", "decoder")
    # Iterative refinement: how many backward correction passes (1=single feedback, 2+=iterative)
    feedback_passes = _e("FEEDBACK_PASSES", 1, int)
    eval_feedback_passes = _e("EVAL_FEEDBACK_PASSES", 0, int)  # 0=same as training
    _feedback_fp_raw = os.environ.get("FEEDBACK_FP_STORAGE", "FP8")
    feedback_fp_storage = True if _feedback_fp_raw == "FP8" else ("fp4" if _feedback_fp_raw == "FP4" else False)
    embed_lr = _e("EMBED_LR", 0.6, float)
    head_lr = _e("HEAD_LR", 0.02, float)
    adam_lr = _e("ADAM_LR", 0.05, float)
    adam_wd = _e("ADAM_WD", 0.04, float)
    untie_at_fraction = _e("UNTIE_AT_FRACTION", 0.0, float)
    tied_embed_lr = _e("TIED_EMBED_LR", 0.035, float)
    seq_len_start = _e("SEQ_LEN_START", 0, int)
    seq_schedule_fraction = _e("SEQ_SCHEDULE_FRACTION", 0.0, float)
    batch_tokens_start = _e("BATCH_TOKENS_START", 0, int)
    batch_schedule_fraction = _e("BATCH_SCHEDULE_FRACTION", 0.33, float)
    churn_log_every = _e("CHURN_LOG_EVERY", 0, int)
    matrix_lr = _e("MATRIX_LR", 0.025, float)
    scalar_lr = _e("SCALAR_LR", 0.025, float)
    muon_momentum = _e("MUON_MOMENTUM", 0.95, float)
    muon_backend_steps = _e("MUON_BACKEND_STEPS", 5, int)
    muon_wd = _e("MUON_WD", 0.04, float)
    matrix_optimizer = _e("MATRIX_OPTIMIZER", "muon")
    muon_momentum_warmup_start = _e("MUON_MOMENTUM_WARMUP_START", 0.85, float)
    muon_momentum_warmup_steps = _e("MUON_MOMENTUM_WARMUP_STEPS", 1500, int)
    beta1 = _e("BETA1", 0.9, float)
    beta2 = _e("BETA2", 0.95, float)
    adam_eps = _e("ADAM_EPS", 1e-8, float)
    grad_clip_norm = _e("GRAD_CLIP_NORM", 0.3, float)
    bitnet_group_size = _e("BITNET_GROUP_SIZE", 128, int)
    sliding_eval = _e("SLIDING_EVAL", 1, bool)
    sliding_eval_stride = _e("SLIDING_EVAL_STRIDE", 64, int)
    sliding_batch_size = _e("SLIDING_BATCH_SIZE", 256, int)
    temp_scaling = _e("TEMP_SCALING", 1, bool)
    # Shared block recurrence: use N unique blocks and tile them across num_layers positions
    shared_blocks = _e("SHARED_BLOCKS", 0, int)  # 0 = off (all unique), 2-3 = shared block count
    # Capsule bank
    capsule_enabled = _e("CAPSULE_ENABLED", 0, bool)
    capsule_num = _e("CAPSULE_NUM", 16, int)
    capsule_dim = _e("CAPSULE_DIM", 64, int)
    # N-gram eval cache
    ngram_cache_enabled = _e("NGRAM_CACHE_ENABLED", 0, bool)
    ngram_max_order = _e("NGRAM_MAX_ORDER", 5, int)
    ngram_alpha_base = _e("NGRAM_ALPHA_BASE", 0.05, float)
    ngram_alpha_scale = _e("NGRAM_ALPHA_SCALE", 0.55, float)
    ngram_entropy_center = _e("NGRAM_ENTROPY_CENTER", 4.0, float)
    # BigramHash embeddings
    bigram_hash_enabled = _e("BIGRAM_HASH_ENABLED", 1, bool)
    bigram_hash_buckets = _e("BIGRAM_HASH_BUCKETS", 4096, int)
    bigram_hash_dim = _e("BIGRAM_HASH_DIM", 128, int)
    # Value Residual Learning (deep layers)
    vrl_enabled = _e("VRL_ENABLED", 1, bool)
    vrl_start_layer = _e("VRL_START_LAYER", 10, int)
    # LN Scale damping
    ln_scale_damping = _e("LN_SCALE_DAMPING", 1, bool)
    # Partial RoPE
    partial_rope_dims = _e("PARTIAL_ROPE_DIMS", 16, int)  # 0=full, 16=only 16 dims rotated
    # XSA (Exclusive Self-Attention): subtract self-value from attention output
    xsa_start_layer = _e("XSA_START_LAYER", 8, int)  # -1=off, 0+=enable from this layer
    # EMA
    ema_enabled = _e("EMA_ENABLED", 1, bool)
    ema_decay = _e("EMA_DECAY", 0.997, float)
    ema_start_fraction = _e("EMA_START_FRACTION", 0.5, float)
    # GPTQ-lite post-training clip search
    gptq_lite_enabled = _e("GPTQ_LITE_ENABLED", 1, bool)
    gptq_lite_percentiles = _e("GPTQ_LITE_PERCENTILES", 5, int)
    # TTT
    ttt_enabled = _e("TTT_ENABLED", 0, bool)
    ttt_lr = _e("TTT_LR", 0.002, float)
    ttt_epochs = _e("TTT_EPOCHS", 1, int)
    ttt_chunk_tokens = _e("TTT_CHUNK_TOKENS", 32768, int)
    ttt_scope = _e("TTT_SCOPE", "feedback")
    ttt_momentum = _e("TTT_MOMENTUM", 0.9, float)
    ttt_batch_seqs = _e("TTT_BATCH_SEQS", 32, int)
    ttt_grad_clip = _e("TTT_GRAD_CLIP", 1.0, float)
    _fp_raw = os.environ.get("FP_STORAGE", "FP8")
    fp_storage = True if _fp_raw == "FP8" else ("fp4" if _fp_raw == "FP4" else False)

CTP = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales", "resid_mix", "resid_mixes",
    "q_gain", "skip_weight", "skip_weights", "vocab_bias", "add_gate", "mul_gate",
    "recurrent_gate", "vrl_alpha",
)

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
        t = tensor.detach().cpu().float().contiguous()
        t_orig_shape = list(t.shape)
        if t.ndim == 3:
            t = t.reshape(t.shape[0], -1)
        is_ternary_candidate = (
            t.ndim == 2 and t.numel() > 65_536
            and "tok_emb" not in name and "lm_head" not in name and "embed_proj" not in name
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

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CTP)) and param.dtype != torch.float32:
                param.data = param.data.float()


class BigramHash(nn.Module):
    """Deterministic bigram hashing for cheap local context injection.
    Maps consecutive token pairs via XOR hash into a fixed-size embedding table,
    projects to model dim, and adds to residual. Zero inference cost beyond a lookup + linear."""
    def __init__(self, num_buckets: int, hash_dim: int, model_dim: int, fp_storage: str | bool):
        super().__init__()
        self.num_buckets = num_buckets
        self.table = QATEmbedding(num_buckets, hash_dim, fp_storage=fp_storage)
        self.proj = QATLinear(hash_dim, model_dim, bias=False, fp_storage=fp_storage)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: (B, T) — compute bigram hash for positions 1..T-1
        prev = input_ids[:, :-1]
        curr = input_ids[:, 1:]
        # Deterministic hash: prev * 92821 + curr mod num_buckets
        indices = ((prev.long() * 92821 + curr.long()) % self.num_buckets).clamp(0, self.num_buckets - 1)
        # Pad position 0 with zero hash (no previous token)
        indices = F.pad(indices, (1, 0), value=0)
        emb = self.table(indices)
        return self.proj(emb)


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
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        group_size: int = 64,
        no_cache: bool = False,
        rope_type: str = "rope",
        yarn_max_len: int = 4096,
        train_seq_len: int = 1024,
        partial_rope_dims: int = 0,
        vrl_enabled: bool = False,
        xsa: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Partial RoPE: only rotate first N dims per head, rest attend without position
        self.rope_dims = partial_rope_dims if partial_rope_dims > 0 else self.head_dim
        self.vrl_enabled = vrl_enabled
        self.xsa = xsa
        self.c_qkv = TernaryLinear(dim, self.q_size + 2 * self.kv_size, bias=False, group_size=group_size)
        self.proj = NormedTernaryLinear(dim, dim, bias=False, group_size=group_size)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(
            self.rope_dims,
            base=rope_base,
            no_cache=no_cache,
            rope_type=rope_type,
            yarn_max_len=yarn_max_len,
            train_seq_len=train_seq_len,
        )
        # VRL: learnable mixing weight for value residual
        if vrl_enabled:
            self.vrl_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: Tensor, v0: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        qkv_out = self.c_qkv(x)
        q_out, k_out, v_out = qkv_out.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q_out.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = k_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # Partial RoPE: only rotate first rope_dims dims
        if self.rope_dims < self.head_dim:
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q_rot, q_pass = q[..., :self.rope_dims], q[..., self.rope_dims:]
            k_rot, k_pass = k[..., :self.rope_dims], k[..., self.rope_dims:]
            q = torch.cat((apply_rotary_emb(q_rot, cos, sin), q_pass), dim=-1)
            k = torch.cat((apply_rotary_emb(k_rot, cos, sin), k_pass), dim=-1)
        else:
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        # VRL: blend current values with first-layer values
        if self.vrl_enabled and v0 is not None:
            alpha = torch.sigmoid(self.vrl_alpha).to(dtype=v.dtype)
            v = alpha * v + (1 - alpha) * v0
        y = flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), causal=True)
        # XSA: subtract self-value to force reliance on context, not self-attending
        if self.xsa:
            # Expand KV heads to match query heads for subtraction
            kv_rep = self.num_heads // self.num_kv_heads
            v_expanded = v.repeat_interleave(kv_rep, dim=2) if kv_rep > 1 else v
            y = y - v_expanded
        return self.proj(y.reshape(bsz, seqlen, dim)), v


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_mult: int,
        group_size: int = 64,
        activation: str = "relu2",
        leaky_relu_slope: float = 0.5,
    ):
        super().__init__()
        hidden = mlp_mult * dim
        self.activation = activation
        self.leaky_relu_slope = leaky_relu_slope
        if activation == "swiglu":
            self.gate_up = TernaryLinear(dim, hidden * 2, bias=False, group_size=group_size)
        else:
            self.fc = TernaryLinear(dim, hidden, bias=False, group_size=group_size)
        self.proj = NormedTernaryLinear(hidden, dim, bias=False, group_size=group_size)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "swiglu":
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            hidden = F.silu(gate) * up
        elif self.activation == "relu":
            hidden = torch.relu(self.fc(x))
        elif self.activation == "lrelu2":
            hidden = F.leaky_relu(self.fc(x), negative_slope=self.leaky_relu_slope).square()
        else:
            hidden = torch.relu(self.fc(x)).square()
        return self.proj(hidden)


class FeedbackPooler(nn.Module):
    def __init__(self, model_dim: int, feedback_dim: int, num_tokens: int, fp_storage: str | bool):
        super().__init__()
        self.num_tokens = max(1, num_tokens)
        self.proj = QATLinear(model_dim, feedback_dim, bias=False, fp_storage=fp_storage)

    def forward(self, x: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_tokens).transpose(1, 2)
        return self.proj(F.rms_norm(pooled, (pooled.size(-1),)))


class FeedbackAdapter(nn.Module):
    """Hadamard-gated backward semantic correction.

    The key idea of Ternary Reasoner: later layers produce a compressed semantic
    sketch. This adapter projects that sketch and applies it to earlier/lower
    representations via a Hadamard (element-wise) gated residual path.

    Two channels:
      - additive:       x += gate_a * proj_a(sketch)          (content injection)
      - multiplicative:  x *= 1 + gate_m * tanh(proj_m(sketch))  (feature modulation)

    Both gates are zero-initialized → adapter is identity at init → safe to train.
    The Hadamard product between the modulation signal and x is what gives the
    architecture its name: it is a structured element-wise correction, not a
    low-rank bias shift.
    """
    def __init__(self, model_dim: int, feedback_dim: int, fp_storage: str | bool):
        super().__init__()
        self.read = QATLinear(feedback_dim, model_dim * 2, bias=False, fp_storage=fp_storage)
        self.add_gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
        self.mul_gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))

    def forward(self, x: Tensor, sketch: Tensor | None) -> Tensor:
        if sketch is None:
            return x
        # sketch: (B, S, feedback_dim) — compressed semantic summary from later layers
        context = sketch.mean(dim=1)  # (B, feedback_dim)
        add_term, mul_term = self.read(context).unsqueeze(1).chunk(2, dim=-1)
        # Hadamard-gated additive correction
        gate_a = torch.tanh(self.add_gate).to(dtype=x.dtype)
        # Hadamard-gated multiplicative correction (modulates features, not replaces)
        gate_m = torch.tanh(self.mul_gate).to(dtype=x.dtype)
        return x * (1.0 + gate_m * torch.tanh(mul_term)) + gate_a * add_term


class CapsuleBank(nn.Module):
    """Structured semantic state carriers for the Ternary Reasoner.

    Capsules are NOT just an attention gimmick. They serve a specific architectural
    role: they compress the token stream into a small set of structured semantic
    slots that persist across feedback iterations. This makes the backward
    correction loop informed by global structure, not just local activations.

    Design:
      1. Read: project tokens into capsule space, soft-assign to prototypes
      2. Pool: aggregate token info into N_caps compact slots
      3. Update: optionally integrate with previous capsule state (recurrent)
      4. Write: broadcast capsule-level corrections back to token positions

    The capsule state is returned separately so it can be carried across
    feedback passes (iterative refinement).
    """
    def __init__(self, model_dim: int, capsule_num: int, capsule_dim: int, fp_storage: str | bool):
        super().__init__()
        self.capsule_num = capsule_num
        self.capsule_dim = capsule_dim
        # Learned prototype slots — define the semantic "topics" capsules attend to
        self.prototypes = nn.Parameter(torch.randn(capsule_num, capsule_dim) * 0.02)
        # Read: token → capsule space
        self.read_proj = QATLinear(model_dim, capsule_dim, bias=False, fp_storage=fp_storage)
        # Write: capsule → token space
        self.write_proj = QATLinear(capsule_dim, model_dim, bias=False, fp_storage=fp_storage)
        self.write_proj._zero_init = True
        # Recurrent gate: blend new capsule state with previous iteration's state
        self.recurrent_gate = nn.Parameter(torch.zeros(capsule_dim, dtype=torch.float32))
        # Output gate
        self.gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))

    def forward(self, x: Tensor, prev_capsules: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Returns (corrected_x, capsule_state) so state persists across iterations."""
        bsz, seqlen, dim = x.shape
        # Project tokens to capsule space
        x_proj = self.read_proj(F.rms_norm(x, (dim,)))
        # Soft-assignment: each token attends to each capsule prototype
        scores = torch.einsum("btd,nd->btn", x_proj, self.prototypes.to(x_proj.dtype))
        attn = torch.softmax(scores / (self.capsule_dim ** 0.5), dim=1)  # (B, T, N)
        # Pool tokens into capsule slots
        capsules = torch.einsum("btn,btd->bnd", attn, x_proj)  # (B, N, C_dim)
        # Recurrent update: blend with previous capsule state if available
        if prev_capsules is not None:
            rg = torch.sigmoid(self.recurrent_gate).to(dtype=capsules.dtype)
            capsules = rg * capsules + (1 - rg) * prev_capsules
        # Broadcast capsule corrections back to token positions
        readout = torch.einsum("btn,bnd->btd", attn, capsules)
        correction = self.write_proj(readout)
        g = torch.tanh(self.gate).to(dtype=x.dtype)
        return x + g * correction, capsules


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        group_size: int = 64,
        activation: str = "relu2",
        leaky_relu_slope: float = 0.5,
        no_cache: bool = False,
        rope_type: str = "rope",
        yarn_max_len: int = 4096,
        train_seq_len: int = 1024,
        partial_rope_dims: int = 0,
        vrl_enabled: bool = False,
        ln_scale_factor: float = 1.0,
        xsa: bool = False,
    ):
        super().__init__()
        self.ln_scale_factor = ln_scale_factor
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            group_size=group_size,
            no_cache=no_cache,
            rope_type=rope_type,
            yarn_max_len=yarn_max_len,
            train_seq_len=train_seq_len,
            partial_rope_dims=partial_rope_dims,
            vrl_enabled=vrl_enabled,
            xsa=xsa,
        )
        self.mlp = MLP(
            dim,
            mlp_mult,
            group_size=group_size,
            activation=activation,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0
        attn_out, v_out = self.attn(self.attn_norm(x) * self.ln_scale_factor, v0=v0)
        x = x + self.attn_scale.to(dtype=x.dtype) * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype) * self.mlp(self.mlp_norm(x) * self.ln_scale_factor)
        return x, v_out


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
        group_size: int = 64,
        activation: str = "relu2",
        leaky_relu_slope: float = 0.5,
        embed_dim: int = 0,
        training_depth_recurrence: int = 0,
        fp_storage: str | bool = False,
        softcap_type: str = "poly",
        no_cache: bool = False,
        rope_type: str = "rope",
        yarn_max_len: int = 4096,
        train_seq_len: int = 1024,
        feedback_enabled: bool = True,
        feedback_dim: int = 64,
        feedback_sketch_tokens: int = 4,
        feedback_replay: str = "decoder",
        feedback_target: str = "decoder",
        feedback_fp_storage: str | bool = True,
        feedback_passes: int = 1,
        shared_blocks: int = 0,
        capsule_enabled: bool = False,
        capsule_num: int = 16,
        capsule_dim: int = 64,
        partial_rope_dims: int = 0,
        vrl_enabled: bool = False,
        vrl_start_layer: int = 8,
        ln_scale_damping: bool = False,
        bigram_hash_enabled: bool = False,
        bigram_hash_buckets: int = 4096,
        bigram_hash_dim: int = 128,
        xsa_start_layer: int = -1,
    ):
        super().__init__()
        self.training_depth_recurrence = training_depth_recurrence
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.softcap_type = softcap_type
        self.embed_dim = embed_dim if embed_dim > 0 else model_dim
        self.feedback_enabled = feedback_enabled
        self.feedback_replay = feedback_replay.lower()
        self.feedback_target = feedback_target.lower()
        self.shared_blocks = shared_blocks
        self.capsule_enabled = capsule_enabled
        self.vrl_enabled = vrl_enabled
        self.vrl_start_layer = vrl_start_layer
        self.feedback_passes = feedback_passes
        if self.feedback_replay not in {"decoder", "none", "off"}:
            raise ValueError(f"Unsupported FEEDBACK_REPLAY={feedback_replay}")
        if self.feedback_target not in {"decoder"}:
            raise ValueError(f"Unsupported FEEDBACK_TARGET={feedback_target}")
        self.tok_emb = QATEmbedding(vocab_size, self.embed_dim, fp_storage=fp_storage)
        self.embed_proj = QATLinear(self.embed_dim, model_dim, bias=False, fp_storage=fp_storage) if self.embed_dim != model_dim else None
        self.embed_proj_rev = QATLinear(model_dim, self.embed_dim, bias=False, fp_storage=fp_storage) if self.embed_dim != model_dim else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Block construction: shared_blocks>0 means create N unique blocks and tile them
        # across num_layers effective positions. Per-layer scales/resid_mix are still unique.
        def _block_kwargs_for(layer_idx):
            layer_vrl = vrl_enabled and layer_idx >= vrl_start_layer
            ln_sf = 1.0 / (layer_idx + 1) ** 0.5 if ln_scale_damping else 1.0
            layer_xsa = xsa_start_layer >= 0 and layer_idx >= xsa_start_layer
            return dict(
                dim=model_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
                mlp_mult=mlp_mult, rope_base=rope_base, qk_gain_init=qk_gain_init,
                group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope,
                no_cache=no_cache, rope_type=rope_type, yarn_max_len=yarn_max_len,
                train_seq_len=train_seq_len, partial_rope_dims=partial_rope_dims,
                vrl_enabled=layer_vrl, ln_scale_factor=ln_sf, xsa=layer_xsa,
            )
        if shared_blocks > 0:
            # Shared weight blocks — only N unique blocks, tiled across depth
            # Use layer 0 kwargs for shared blocks (VRL/LN damping applied at forward time)
            base_kwargs = _block_kwargs_for(0)
            base_kwargs["vrl_enabled"] = False  # VRL handled at GPT level for shared mode
            base_kwargs["ln_scale_factor"] = 1.0
            self.shared_block_bank = nn.ModuleList([Block(**base_kwargs) for _ in range(shared_blocks)])
            # Per-position lightweight adapters: unique scale/mix per effective layer
            self.per_layer_attn_scales = nn.ParameterList([
                nn.Parameter(torch.ones(model_dim, dtype=torch.float32)) for _ in range(num_layers)])
            self.per_layer_mlp_scales = nn.ParameterList([
                nn.Parameter(torch.ones(model_dim, dtype=torch.float32)) for _ in range(num_layers)])
            self.per_layer_resid_mixes = nn.ParameterList([
                nn.Parameter(torch.stack((torch.ones(model_dim), torch.zeros(model_dim))).float())
                for _ in range(num_layers)])
            # Map effective layer index -> shared block index (round-robin)
            self._block_map = [i % shared_blocks for i in range(num_layers)]
            self.blocks = None  # signal to forward that we use shared mode
        else:
            self.shared_block_bank = None
            self.per_layer_attn_scales = None
            self.per_layer_mlp_scales = None
            self.per_layer_resid_mixes = None
            self._block_map = None
            self.blocks = nn.ModuleList([Block(**_block_kwargs_for(i)) for i in range(num_layers)])

        self.final_norm = RMSNorm()

        # BigramHash — cheap local context injection
        self.bigram_hash = None
        if bigram_hash_enabled:
            self.bigram_hash = BigramHash(bigram_hash_buckets, bigram_hash_dim, model_dim, fp_storage=fp_storage)

        # Capsule bank — lightweight semantic routing between encoder and decoder
        self.capsule_bank = None
        if capsule_enabled:
            self.capsule_bank = CapsuleBank(model_dim, capsule_num, capsule_dim, fp_storage=feedback_fp_storage)

        self.feedback_pooler = None
        self.feedback_adapters = None
        if self.feedback_enabled and self.feedback_replay == "decoder":
            self.feedback_pooler = FeedbackPooler(
                model_dim,
                feedback_dim,
                feedback_sketch_tokens,
                fp_storage=feedback_fp_storage,
            )
            self.feedback_adapters = nn.ModuleList(
                [
                    FeedbackAdapter(model_dim, feedback_dim, fp_storage=feedback_fp_storage)
                    for _ in range(self.num_decoder_layers)
                ]
            )
        self.lm_head = QATLinear(model_dim, vocab_size, bias=False, fp_storage=fp_storage)
        self.lm_head._zero_init = True
        if self.tie_embeddings:
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

    def _apply_embedding(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        x = self.tok_emb(input_ids).float()
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        # Add bigram hash if enabled
        if self.bigram_hash is not None:
            x = x + self.bigram_hash(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        return x, x

    def _run_block(self, layer_idx: int, x: Tensor, x0: Tensor, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Run one effective layer. In shared mode, uses shared weights + per-layer scales."""
        if self.blocks is not None:
            # Standard unique-block mode — Block.forward returns (x, v_out)
            return self.blocks[layer_idx](x, x0, v0=v0)
        # Shared block mode: use shared weights but per-layer scale/mix params
        block = self.shared_block_bank[self._block_map[layer_idx]]
        mix = self.per_layer_resid_mixes[layer_idx].to(dtype=x.dtype)
        x = mix[0] * x + mix[1] * x0
        # Use shared block's attention and MLP but override with per-layer scales
        attn_out, v_out = block.attn(block.attn_norm(x), v0=v0)
        x = x + self.per_layer_attn_scales[layer_idx].to(dtype=x.dtype) * attn_out
        x = x + self.per_layer_mlp_scales[layer_idx].to(dtype=x.dtype) * block.mlp(block.mlp_norm(x))
        return x, v_out

    def _decoder_pass(self, x: Tensor, x0: Tensor, skips: list[Tensor], sketch: Tensor | None, v0: Tensor | None = None) -> Tensor:
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if i < self.num_skip_weights:
                x = x + self.skip_weights[i].to(dtype=x.dtype) * skips[-(i + 1)]
            for _ in range(max(1, self.training_depth_recurrence)):
                x, _ = self._run_block(bi, x, x0, v0=v0)
            if self.feedback_adapters is not None and sketch is not None:
                x = self.feedback_adapters[i](x, sketch)
        return x

    def _compute_hidden(self, input_ids: Tensor) -> Tensor:
        """Core Ternary Reasoner forward: encode → [correct]^N → decode.

        The iterative correction loop is the defining architectural feature:
          Pass 0: blind decoder pass (no feedback, no capsule history)
          Pass 1..N: decoder receives backward semantic sketch from previous pass,
                     capsule state accumulates across iterations,
                     Hadamard-gated adapters apply structured corrections.

        This is not a standard transformer forward. It is hierarchical
        iterative refinement with backward semantic flow.
        """
        x, x0 = self._apply_embedding(input_ids)
        skips: list[Tensor] = []
        v0 = None  # VRL: first-layer values, captured once

        # --- Encoder pass (runs once) ---
        for i in range(self.num_encoder_layers):
            for _ in range(max(1, self.training_depth_recurrence)):
                x, v_out = self._run_block(i, x, x0, v0=v0)
                if v0 is None and v_out is not None:
                    v0 = v_out.detach()
            skips.append(x)

        # Capsule state initialization (persists across correction iterations)
        capsule_state = None
        if self.capsule_bank is not None:
            x, capsule_state = self.capsule_bank(x, prev_capsules=None)

        encoded = x
        num_passes = self.feedback_passes

        # --- Iterative correction loop ---
        # Pass 0: blind (no sketch), establishes initial decoder output
        # Pass 1..N: backward semantic flow — sketch from previous output
        #            corrects the next decoder pass via Hadamard-gated adapters
        sketch = None
        for correction_pass in range(num_passes + 1):  # 0=blind, 1..N=corrected
            if correction_pass > 0 and self.feedback_enabled and self.feedback_pooler is not None:
                # Compress previous decoder output into semantic sketch
                sketch = self.feedback_pooler(self.final_norm(x))
            else:
                sketch = None

            # Update capsule state with current representation (iterative accumulation)
            if self.capsule_bank is not None and correction_pass > 0:
                encoded, capsule_state = self.capsule_bank(encoded, prev_capsules=capsule_state)

            # Decoder pass with backward correction
            x = self._decoder_pass(encoded, x0, skips, sketch=sketch, v0=v0)

            # Early exit if feedback is disabled — no point iterating
            if not self.feedback_enabled or self.feedback_pooler is None:
                break

        return self.final_norm(x)

    def _compute_logits(self, x: Tensor) -> Tensor:
        if self.tie_embeddings:
            proj = self.embed_proj_rev(x) if self.embed_proj_rev is not None else x
            logits_raw = F.linear(proj, self.tok_emb.weight.to(x.dtype))
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

    def forward_logits(self, input_ids: Tensor, temperature: float = 1.0) -> Tensor:
        hidden = self._compute_hidden(input_ids)
        logits = self._softcap(self._compute_logits(hidden.reshape(-1, hidden.size(-1))))
        if temperature != 1.0:
            logits = logits / temperature
        return logits.reshape(input_ids.size(0), input_ids.size(1), -1)

    def forward(self, input_ids: Tensor, target_ids: Tensor, reduction: str = "mean", temperature: float = 1.0) -> Tensor:
        logits = self.forward_logits(input_ids, temperature=temperature).reshape(-1, self.vocab_bias.numel())
        targets = target_ids.reshape(-1)
        if reduction == "none":
            return F.cross_entropy(logits.float(), targets, reduction="none").reshape(input_ids.shape)
        logits_f = logits.float()
        lse = torch.logsumexp(logits_f, dim=-1)
        target_logits = logits_f.gather(1, targets.unsqueeze(1)).squeeze(1)
        ce_loss = (lse - target_logits).mean()
        # Z-loss regularization only during training; excluded during eval to avoid inflating BPB
        if self.training:
            ce_loss = ce_loss + 1e-4 * (lse ** 2).mean()
        return ce_loss

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

def eval_val_sliding(args, base_model, rank, world_size, device, grad_accum_steps, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride: int = 64, temperature: float = 1.0):
    del grad_accum_steps
    seq_len = args.train_seq_len
    batch_size = args.sliding_batch_size
    total_tokens = val_tokens.numel() - 1
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    all_starts = list(range(0, total_tokens, stride))
    my_starts = [s for idx, s in enumerate(all_starts) if idx % world_size == rank and min(s + seq_len, total_tokens) - s >= 1]

    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for i in range(0, len(my_starts), batch_size):
            batch_starts = my_starts[i:i + batch_size]
            bsz = len(batch_starts)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for j, start in enumerate(batch_starts):
                end = min(start + seq_len, total_tokens)
                wlen = end - start
                wlens.append(wlen)
                chunk = val_tokens[start:end + 1].to(dtype=torch.int64, device=device)
                x_batch[j, :wlen] = chunk[:-1]
                y_batch[j, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch, temperature=temperature)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for j, start in enumerate(batch_starts):
                wlen = wlens[j]
                score_from = 0 if start == 0 else max(wlen - stride, 0)
                scored = nll[j, score_from:wlen]
                sx = x_batch[j, score_from:wlen]
                sy = y_batch[j, score_from:wlen]
                loss_sum += scored.to(torch.float64).sum()
                token_count += float(wlen - score_from)
                tok_bytes = base_bytes_lut[sy].to(torch.int16)
                tok_bytes += (has_leading_space_lut[sy] & ~is_boundary_token_lut[sx]).to(torch.int16)
                byte_count += tok_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bpb = (val_loss.item() / math.log(2.0)) * (token_count.item() / byte_count.item())
    base_model.train()
    return float(val_loss.item()), float(bpb)


def collect_ttt_params(base_model: nn.Module, scope: str) -> tuple[dict[str, bool], list[Tensor]]:
    if scope != "feedback":
        raise ValueError(f"Unsupported TTT_SCOPE={scope}")
    original: dict[str, bool] = {}
    params: list[Tensor] = []
    for name, p in base_model.named_parameters():
        original[name] = p.requires_grad
        allow = (name.startswith("feedback_pooler.") or name.startswith("feedback_adapters.")
                or name == "skip_weights" or name.startswith("capsule_bank."))
        if name.startswith("blocks."):
            parts = name.split(".")
            if len(parts) >= 3:
                block_idx = int(parts[1])
                leaf = parts[2]
                if block_idx >= base_model.num_encoder_layers and leaf in {"attn_scale", "mlp_scale"}:
                    allow = True
        # Per-layer scales in shared block mode — allow decoder-half scales for TTT
        if name.startswith("per_layer_attn_scales.") or name.startswith("per_layer_mlp_scales."):
            idx = int(name.split(".")[1])
            if idx >= base_model.num_encoder_layers:
                allow = True
        p.requires_grad_(allow)
        if allow:
            params.append(p)
    return original, params


def restore_requires_grad(base_model: nn.Module, original: dict[str, bool]) -> None:
    for name, p in base_model.named_parameters():
        p.requires_grad_(original.get(name, True))


def eval_val_sliding_ttt(
    args,
    base_model,
    rank,
    world_size,
    device,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    stride: int,
    batch_seqs: int = 32,
    temperature: float = 1.0,
    log0=print,
):
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        score_from = 0 if ws == 0 else max(wlen - stride, 0)
        chunk_idx = min((ws + score_from) // ttt_chunk, num_chunks - 1)
        chunk_windows[chunk_idx].append(ws)

    log0(
        f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} stride={stride} "
        f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} scope={args.ttt_scope}"
    )
    original_grad, ttt_params = collect_ttt_params(base_model, args.ttt_scope)
    log0(f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)}")
    if not ttt_params:
        raise RuntimeError("TTT enabled but no parameters matched TTT_SCOPE")
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    try:
        for ci in range(num_chunks):
            windows = chunk_windows[ci]
            if not windows:
                continue
            chunk_start = ci * ttt_chunk
            chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
            my_s = (len(windows) * rank) // world_size
            my_e = (len(windows) * (rank + 1)) // world_size
            my_windows = windows[my_s:my_e]

            base_model.eval()
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
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = base_model.forward_logits(x_batch, temperature=temperature)
                    nll = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        y_batch.reshape(-1),
                        reduction="none",
                    ).reshape(bsz, seq_len)
                    for i, ws in enumerate(batch_ws):
                        wlen = wlens[i]
                        score_from = 0 if ws == 0 else max(wlen - stride, 0)
                        scored = nll[i, score_from:wlen].to(torch.float64)
                        loss_sum += scored.sum()
                        token_count += float(wlen - score_from)
                        sx = x_batch[i, score_from:wlen]
                        sy = y_batch[i, score_from:wlen]
                        tok_bytes = base_bytes_lut[sy].to(torch.int16)
                        tok_bytes += (has_leading_space_lut[sy] & ~is_boundary_token_lut[sx]).to(torch.int16)
                        byte_count += tok_bytes.to(torch.float64).sum()

            if ci == num_chunks - 1 or args.ttt_epochs <= 0:
                continue

            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs <= 0:
                continue
            cosine_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr
            my_seq_s = (chunk_seqs * rank) // world_size
            my_seq_e = (chunk_seqs * (rank + 1)) // world_size
            my_chunk_seqs = my_seq_e - my_seq_s
            for _ in range(args.ttt_epochs):
                for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                    be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                    actual_bs = my_seq_s + bs
                    start_tok = chunk_start + actual_bs * seq_len
                    end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                    if end_tok > val_tokens.numel():
                        continue
                    local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                    x = local[:-1].reshape(-1, seq_len)
                    y = local[1:].reshape(-1, seq_len)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = base_model(x, y, temperature=temperature)
                    loss.backward()
                    if world_size > 1:
                        for p in ttt_params:
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                    optimizer.step()

            if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
                elapsed = time.perf_counter() - t0
                running_loss = loss_sum.item() / max(token_count.item(), 1)
                running_bpb = running_loss / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
                log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={running_bpb:.6f} time={elapsed:.1f}s")

        if dist.is_available() and dist.is_initialized():
            for t in (loss_sum, token_count, byte_count):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
        val_loss = (loss_sum / token_count).item()
        val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
        log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} elapsed={time.perf_counter() - t0:.1f}s")
        return val_loss, val_bpb
    finally:
        restore_requires_grad(base_model, original_grad)
        base_model.eval()

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
# N-gram evaluation cache with entropy-adaptive mixing
# ---------------------------------------------------------------------------
class NgramCache:
    """Dynamic n-gram cache built from already-scored tokens. Zero artifact cost.
    Uses entropy-adaptive alpha to blend n-gram empirical probabilities with neural logits."""
    def __init__(self, max_order: int = 5, alpha_base: float = 0.05,
                 alpha_scale: float = 0.55, entropy_center: float = 4.0):
        self.max_order = max_order
        self.alpha_base = alpha_base
        self.alpha_scale = alpha_scale
        self.entropy_center = entropy_center
        # counts[order] maps tuple(context) -> {next_token: count}
        self.counts: list[dict] = [{} for _ in range(max_order + 1)]
        self.total_counts: list[dict] = [{} for _ in range(max_order + 1)]

    def update(self, tokens: list[int]) -> None:
        """Add scored tokens to the cache."""
        for order in range(2, self.max_order + 1):
            for i in range(len(tokens) - order + 1):
                ctx = tuple(tokens[i:i + order - 1])
                nxt = tokens[i + order - 1]
                if ctx not in self.counts[order]:
                    self.counts[order][ctx] = {}
                    self.total_counts[order][ctx] = 0
                self.counts[order][ctx][nxt] = self.counts[order][ctx].get(nxt, 0) + 1
                self.total_counts[order][ctx] += 1

    def predict(self, context: list[int], vocab_size: int) -> Tensor | None:
        """Return n-gram log-probability distribution via backoff, or None if no match."""
        for order in range(self.max_order, 1, -1):
            if len(context) < order - 1:
                continue
            ctx = tuple(context[-(order - 1):])
            if ctx in self.counts[order]:
                total = self.total_counts[order][ctx]
                probs = torch.zeros(vocab_size)
                for tok, count in self.counts[order][ctx].items():
                    if tok < vocab_size:
                        probs[tok] = count / total
                if probs.sum() > 0:
                    # Laplace smoothing
                    probs = (probs + 1e-8) / (probs.sum() + 1e-8 * vocab_size)
                    return probs.log()
        return None

    def entropy_alpha(self, neural_logprobs: Tensor) -> float:
        """Compute entropy-adaptive mixing weight."""
        probs = neural_logprobs.exp()
        H = -(probs * neural_logprobs).sum().item()
        # Sigmoid schedule: higher entropy → more trust in cache
        return self.alpha_base + self.alpha_scale * (1.0 / (1.0 + math.exp(-2.0 * (H - self.entropy_center))))


# ---------------------------------------------------------------------------
# GPTQ-lite: per-row clip percentile search for ternary quantization
# ---------------------------------------------------------------------------
def gptq_lite_clip_search(state_dict: dict, group_size: int, num_percentiles: int = 5) -> dict:
    """For each ternary-candidate weight matrix, search over clip percentiles
    to minimize per-row MSE between original and ternary-quantized weights.
    Returns a modified state_dict with clipped weights."""
    percentiles = [0.995 + 0.001 * i for i in range(num_percentiles)]
    improved = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float()
        if t.ndim != 2 or t.numel() <= 65536 or "tok_emb" in name or "lm_head" in name or "embed_proj" in name:
            improved[name] = tensor
            continue
        best_t = t.clone()
        best_mse = float("inf")
        # Pad columns to be divisible by group_size (same as q_sd does)
        pad = (group_size - t.shape[1] % group_size) % group_size
        for pct in percentiles:
            clip_val = torch.quantile(t.abs().flatten(), pct)
            t_clipped = t.clamp(-clip_val, clip_val)
            t_padded = F.pad(t_clipped, (0, pad)) if pad > 0 else t_clipped
            t_g = t_padded.reshape(-1, group_size)
            scale = t_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            q = (t_g / scale).round().clamp(-1, 1)
            recon = (q * scale).reshape(t_padded.shape)[:t.shape[0], :t.shape[1]]
            mse = (t_clipped - recon).pow(2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_t = t_clipped
        improved[name] = best_t.to(tensor.dtype).to(tensor.device)
    return improved


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------
class EMAHelper:
    """Exponential Moving Average for model weights. Maintains shadow copy.
    Applied to latent FP32 weights; ternary quantization happens at export."""
    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = decay
        self.shadow: dict[str, Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module) -> dict[str, Tensor]:
        """Replace model weights with EMA shadow. Returns original weights for restore."""
        original = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                original[name] = p.data.clone()
                p.data.copy_(self.shadow[name])
        return original

    @staticmethod
    def restore(model: nn.Module, original: dict[str, Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in original:
                p.data.copy_(original[name])


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
        group_size=args.bitnet_group_size, activation=args.activation_type, leaky_relu_slope=args.leaky_relu_slope,
        embed_dim=args.embed_dim, training_depth_recurrence=args.training_depth_recurrence, fp_storage=args.fp_storage,
        softcap_type=args.softcap_type, no_cache=(args.compile_mode == "reduce-overhead"),
        rope_type=args.rope_type, yarn_max_len=args.yarn_max_len, train_seq_len=args.train_seq_len,
        feedback_enabled=args.feedback_enabled, feedback_dim=args.feedback_dim,
        feedback_sketch_tokens=args.feedback_sketch_tokens, feedback_replay=args.feedback_replay,
        feedback_target=args.feedback_target, feedback_fp_storage=args.feedback_fp_storage,
        feedback_passes=args.feedback_passes,
        shared_blocks=args.shared_blocks, capsule_enabled=args.capsule_enabled,
        capsule_num=args.capsule_num, capsule_dim=args.capsule_dim,
        partial_rope_dims=args.partial_rope_dims, vrl_enabled=args.vrl_enabled,
        vrl_start_layer=args.vrl_start_layer, ln_scale_damping=args.ln_scale_damping,
        bigram_hash_enabled=args.bigram_hash_enabled,
        bigram_hash_buckets=args.bigram_hash_buckets, bigram_hash_dim=args.bigram_hash_dim,
        xsa_start_layer=args.xsa_start_layer,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, nn.Linear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if base_model.lm_head is not None and args.tie_embeddings:
        base_model.lm_head.weight.requires_grad_(False)

    torch._dynamo.config.optimize_ddp = False

    compiled_model = torch.compile(base_model, mode=args.compile_mode if args.compile_mode != "default" else None)
    # shared_blocks>0 creates unused params inside shared_block_bank (attn_scale, mlp_scale, resid_mix)
    # since we override them with per_layer_* params. Must enable find_unused_parameters.
    use_find_unused = args.untie_at_fraction > 0 or not args.tie_embeddings or args.shared_blocks > 0
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False,
                find_unused_parameters=use_find_unused,
                static_graph=not use_find_unused,
                gradient_as_bucket_view=True) if distributed else compiled_model

    # --- Optimizers ---
    _excl = {"tok_emb.weight", "lm_head.weight"}
    all_other_params = [(n, p) for n, p in base_model.named_parameters()
                        if not any(eh in n for eh in _excl)]
    matrix_params: list[Tensor] = []
    scalar_params: list[Tensor] = []
    for name, param in all_other_params:
        if param.ndim == 2 and not any(pat in name for pat in CTP) and "feedback" not in name and "capsule" not in name:
            matrix_params.append(param)
        else:
            scalar_params.append(param)

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

    # --- EMA ---
    ema = None  # initialized after warmup

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

    # --- EMA init (after warmup so shadow starts from clean weights) ---
    if args.ema_enabled:
        ema = EMAHelper(base_model, decay=args.ema_decay)
        log0(f"ema:enabled decay={args.ema_decay} start_fraction={args.ema_start_fraction}")

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

        # Gradient clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        # Untie lm_head at configured fraction of training
        if args.untie_at_fraction > 0:
            if max_wallclock_ms is not None:
                should_untie = not _untied and elapsed_ms >= args.untie_at_fraction * max_wallclock_ms
            else:
                should_untie = not _untied and step >= int(args.iterations * args.untie_at_fraction)
            if should_untie and base_model.tie_embeddings:
                with torch.no_grad():
                    base_weight = base_model.tok_emb.weight.float()
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

        # EMA update (only after start_fraction of training)
        if ema is not None:
            ema_progress = elapsed_ms / max_wallclock_ms if max_wallclock_ms else step / args.iterations
            if ema_progress >= args.ema_start_fraction:
                ema.update(base_model)

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

    # --- Apply EMA shadow weights before serialization ---
    _ema_original = None
    if ema is not None:
        _ema_original = ema.apply_shadow(base_model)
        log0("ema:applied shadow weights for serialization")

    # --- Serialization ---
    if master_process:
        sd = base_model.state_dict()

        # GPTQ-lite: per-row clip percentile search before quantization
        if args.gptq_lite_enabled:
            log0(f"gptq_lite:searching {args.gptq_lite_percentiles} percentiles...")
            sd = gptq_lite_clip_search(sd, group_size=args.bitnet_group_size,
                                       num_percentiles=args.gptq_lite_percentiles)
            log0("gptq_lite:done")

        if base_model.tie_embeddings:
            sd.pop("lm_head.weight", None)

        # Two methods: Standard Base-3 vs Bitmask Mapping
        methods = {}
        for method in ("standard", "bitmask"):
            q_obj, stats = q_sd(sd, group_size=args.bitnet_group_size, fp_storage=args.fp_storage, ternary_method=method)
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
        if args.eval_feedback_passes > 0:
            base_model.feedback_passes = args.eval_feedback_passes
            log0(f"eval_feedback_passes:{args.eval_feedback_passes}")

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

    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args, base_model, rank, world_size, device, val_tokens, base_bytes_lut,
            has_leading_space_lut, is_boundary_token_lut, stride=args.sliding_eval_stride,
            batch_seqs=args.ttt_batch_seqs, temperature=opt_temp, log0=log0,
        )
        torch.cuda.synchronize()
        ttt_time_ms = 1000.0 * (time.perf_counter() - t_ttt)
        log0(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} eval_time:{ttt_time_ms:.0f}ms")
        log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    # --- N-gram cache evaluation (single-rank, sequential) ---
    if args.ngram_cache_enabled and master_process:
        torch.cuda.synchronize()
        t_ngram = time.perf_counter()
        ngram_cache = NgramCache(
            max_order=args.ngram_max_order,
            alpha_base=args.ngram_alpha_base,
            alpha_scale=args.ngram_alpha_scale,
            entropy_center=args.ngram_entropy_center,
        )
        seq_len = args.train_seq_len
        total_tokens_ng = val_tokens.numel() - 1
        ngram_loss_sum = 0.0
        ngram_byte_sum = 0.0
        ngram_tok_count = 0
        scored_tokens: list[int] = []
        base_model.eval()
        # Process validation sequentially, building cache as we go
        with torch.inference_mode():
            for pos in range(0, total_tokens_ng, seq_len):
                end = min(pos + seq_len, total_tokens_ng)
                wlen = end - pos
                chunk = val_tokens[pos:end + 1].to(dtype=torch.int64, device=device)
                x_ng = chunk[:-1].unsqueeze(0)
                y_ng = chunk[1:].unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits_ng = base_model.forward_logits(x_ng, temperature=opt_temp)
                log_probs_ng = F.log_softmax(logits_ng.squeeze(0).float(), dim=-1)
                for t_idx in range(wlen):
                    target_tok = y_ng[0, t_idx].item()
                    neural_lp = log_probs_ng[t_idx]
                    # Try n-gram cache prediction
                    ngram_lp = ngram_cache.predict(scored_tokens[-(args.ngram_max_order - 1):], args.vocab_size)
                    if ngram_lp is not None:
                        ngram_lp = ngram_lp.to(device)
                        alpha = ngram_cache.entropy_alpha(neural_lp)
                        # Mix in log space: log((1-a)*exp(neural) + a*exp(ngram))
                        mixed = torch.logaddexp(
                            neural_lp + math.log(1 - alpha + 1e-10),
                            ngram_lp + math.log(alpha + 1e-10),
                        )
                        token_nll = -mixed[target_tok].item()
                    else:
                        token_nll = -neural_lp[target_tok].item()
                    ngram_loss_sum += token_nll
                    # BPB accounting — use actual prev token (from input sequence)
                    tok_b = base_bytes_lut[target_tok].item()
                    prev_tok = x_ng[0, t_idx].item()
                    if has_leading_space_lut[target_tok].item() and not is_boundary_token_lut[prev_tok].item():
                        tok_b += 1
                    ngram_byte_sum += tok_b
                    ngram_tok_count += 1
                    scored_tokens.append(target_tok)
                # Update cache with scored tokens
                ngram_cache.update(chunk[1:].tolist())
        ngram_val_loss = ngram_loss_sum / max(ngram_tok_count, 1)
        ngram_bpb = (ngram_val_loss / math.log(2.0)) * (ngram_tok_count / max(ngram_byte_sum, 1))
        ngram_time_ms = 1000.0 * (time.perf_counter() - t_ngram)
        log0(f"ngram_cache val_loss:{ngram_val_loss:.4f} val_bpb:{ngram_bpb:.4f} "
             f"(order={args.ngram_max_order}) eval_time:{ngram_time_ms:.0f}ms")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
