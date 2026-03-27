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
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
_HAS_FA3 = False
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _HAS_FA3 = True
except ImportError:
    try:
        from flash_attn import flash_attn_func as flash_attn_3_func
        _HAS_FA3 = True
    except ImportError:
        pass
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 8))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))  # tighter: collect more recent checkpoints
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 4096))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))  # XSA on all layers (PR #825)
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.5))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
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
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss
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
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
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
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())
def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t
def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale
def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
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
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
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
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats
def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out
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
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 15.0).clamp_min(1.0 / 15.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -15, 15) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()
class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
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
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
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
        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False  # set by GPT.__init__ for deep layers only
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Efficient XSA: subtract self-value projection via GQA-aware reshape (no repeat_interleave).
        y: [B, T, H, D], v: [B, T, Hkv, D]. H must be divisible by Hkv."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
        vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D] — broadcast ready
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if _HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            # fallback to pytorch SDPA (q,k,v need to be [bsz, heads, seq, dim])
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads))
            y = y.transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
class BigramHashEmbedding(nn.Module):
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
class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers.
    Each table maps vocab tokens to a low-dim embedding, projected to model_dim."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor) -> Tensor:
        # leaky_relu(0.5)^2 preserves negative gradient flow vs relu^2
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
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
        layer_idx: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None
    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out
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
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.1,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
    ):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)  # kv_dim for value projection
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                    ln_scale=ln_scale,
                    dtg=dtg,
                )
                for i in range(num_layers)
            ]
        )
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()  # keep empty for compat
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
        )
        for head in self.mtp_heads:
            head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        """Get value embedding for a specific layer using shared table + per-layer scale."""
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_targets = target_ids[:, k + 1 :].reshape(-1)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
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
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
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
                logits = compiled_logits(x_batch)
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
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte
class LongPhraseCache:
    """variable-length suffix matcher for verbatim repetition (PR #880).
    probes at lengths [48,36,28,20,16] using rolling hashes."""
    PROBE_LENGTHS = [48, 36, 28, 20, 16]  # full probes, stride=64 saves eval time
    PRIMES = [np.uint64(p) for p in [
        36313, 27191, 51647, 81929, 131071, 174763, 233017, 299993, 350377,
        412391, 479909, 541267, 613651, 700897, 786433, 850001, 921587,
        982451, 1048573, 1114111, 1179641, 1245169, 1310719, 1376257,
        1441793, 1507321, 1572869, 1638391, 1703933, 1769473, 1835009,
        1900543, 1966079, 2031617, 2097143, 2162689, 2228223, 2293759,
        2359291, 2424833, 2490367, 2555903, 2621431, 2686979, 2752511,
        2818049, 2883577, 2949121,
    ]]  # 48 primes for longest probe
    BUCKETS = 4194304
    MASK = np.uint64(BUCKETS - 1)

    def __init__(self):
        self.ctx_tables = {L: np.zeros(self.BUCKETS, dtype=np.uint32) for L in self.PROBE_LENGTHS}
        self.full_tables = {L: np.zeros(self.BUCKETS, dtype=np.uint32) for L in self.PROBE_LENGTHS}

    def _rolling_hash(self, val_np: np.ndarray, positions: np.ndarray, length: int) -> np.ndarray:
        h = np.zeros(len(positions), dtype=np.uint64)
        for k in range(length):
            toks = val_np[(positions - length + k).astype(np.int64)].astype(np.uint64)
            h ^= toks * self.PRIMES[k]
        return h

    def build_full(self, val_np: np.ndarray, log_fn=None):
        """build phrase cache from all tokens."""
        n = len(val_np) - 1
        for L in self.PROBE_LENGTHS:
            if n <= L:
                continue
            positions = np.arange(L, n, dtype=np.int64)
            ctx_hash = self._rolling_hash(val_np, positions, L)
            ctx_key = (ctx_hash & self.MASK).astype(np.int64)
            targets = val_np[positions + 1].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * self.PRIMES[L % len(self.PRIMES)])) & self.MASK).astype(np.int64)
            np.add.at(self.ctx_tables[L], ctx_key, 1)
            np.add.at(self.full_tables[L], full_key, 1)
            if log_fn:
                log_fn(f"phrase_cache: length {L} done")

    def update(self, val_np: np.ndarray, start: int, end: int):
        """incremental score-first update for a window segment."""
        for L in self.PROBE_LENGTHS:
            first_valid = max(L, start)
            n_pos = end - first_valid
            if n_pos <= 0:
                continue
            positions = np.arange(first_valid, end, dtype=np.int64)
            ctx_hash = self._rolling_hash(val_np, positions, L)
            ctx_key = (ctx_hash & self.MASK).astype(np.int64)
            targets = val_np[(positions + 1).astype(np.int64)].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * self.PRIMES[L % len(self.PRIMES)])) & self.MASK).astype(np.int64)
            np.add.at(self.ctx_tables[L], ctx_key, 1)
            np.add.at(self.full_tables[L], full_key, 1)

    def lookup(self, val_np: np.ndarray, positions: np.ndarray, min_count: int = 2
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """lookup phrase matches. returns (p_phrase, has_match, match_length, ctx_counts, full_counts)."""
        n_pos = len(positions)
        p_phrase = np.zeros(n_pos, dtype=np.float64)
        has_match = np.zeros(n_pos, dtype=np.bool_)
        match_length = np.zeros(n_pos, dtype=np.int32)
        ctx_counts = np.zeros(n_pos, dtype=np.float64)
        full_counts = np.zeros(n_pos, dtype=np.float64)
        for L in self.PROBE_LENGTHS:  # longest first
            valid = (positions >= L) & ~has_match
            if not valid.any():
                continue
            pos_valid = positions[valid]
            ctx_hash = self._rolling_hash(val_np, pos_valid, L)
            ctx_key = (ctx_hash & self.MASK).astype(np.int64)
            targets = val_np[(pos_valid + 1).astype(np.int64)].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * self.PRIMES[L % len(self.PRIMES)])) & self.MASK).astype(np.int64)
            ctx_c = self.ctx_tables[L][ctx_key]
            full_c = np.minimum(self.full_tables[L][full_key], ctx_c)
            eligible = (ctx_c >= min_count) & (full_c > 0)
            if eligible.any():
                valid_idx = np.where(valid)[0][eligible]
                p_phrase[valid_idx] = full_c[eligible].astype(np.float64) / ctx_c[eligible].astype(np.float64)
                has_match[valid_idx] = True
                match_length[valid_idx] = L
                ctx_counts[valid_idx] = ctx_c[eligible].astype(np.float64)
                full_counts[valid_idx] = full_c[eligible].astype(np.float64)
        return p_phrase, has_match, match_length, ctx_counts, full_counts


class NgramCache:
    """n-gram cache matching PR #753/#769/#779: two flat uint32 arrays per order
    (ctx_counts, full_counts). hash context and full n-gram (context+target) separately."""
    PRIMES = [np.uint64(p) for p in [36313, 27191, 51647, 81929, 131071, 174763, 233017, 299993, 350377, 412391, 479909, 541267, 613651, 700897, 786433]]

    def __init__(self, max_order: int = 7, min_order: int = 2, num_buckets: int = 4194304,
                 min_count: int = 2, **kwargs):
        self.max_order = max_order
        self.min_order = min_order
        self.num_buckets = num_buckets
        self.min_count = min_count
        self.mask = np.uint64(num_buckets - 1)
        self.num_orders = max_order - min_order + 1
        # ~32MB per order (4M * 4 bytes * 2 arrays) = ~192MB for 6 orders
        self.ctx_counts = [np.zeros(num_buckets, dtype=np.uint32) for _ in range(self.num_orders)]
        self.full_counts = [np.zeros(num_buckets, dtype=np.uint32) for _ in range(self.num_orders)]

    def build_full(self, val_np: np.ndarray, log_fn=None):
        """build complete cache from all tokens at once (for two-pass rescoring)."""
        n = len(val_np) - 1
        mask = self.mask
        primes = self.PRIMES
        for oi in range(self.num_orders):
            order = self.min_order + oi
            cw = order - 1
            if n <= cw:
                continue
            valid_start = cw
            n_pos = n - valid_start
            # context hash
            ctx_hash = np.zeros(n_pos, dtype=np.uint64)
            for k in range(cw):
                t = val_np[valid_start - cw + k:valid_start - cw + k + n_pos].astype(np.uint64)
                ctx_hash ^= t * np.uint64(primes[k])
            ctx_key = (ctx_hash & mask).astype(np.int64)
            # full hash
            targets = val_np[valid_start + 1:valid_start + 1 + n_pos].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * np.uint64(primes[cw]))) & mask).astype(np.int64)
            # bincount-based bulk add
            np.add.at(self.ctx_counts[oi], ctx_key, 1)
            np.add.at(self.full_counts[oi], full_key, 1)
            if log_fn:
                log_fn(f"ngram_build: order {order} done, {n_pos} positions")

    def lookup(self, val_np: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """score positions [start, end). returns (p_ngram, has_match, matched_order, ctx_counts, full_counts)."""
        seg_len = end - start
        p_ngram = np.zeros(seg_len, dtype=np.float64)
        has_match = np.zeros(seg_len, dtype=np.bool_)
        matched_order = np.zeros(seg_len, dtype=np.int32)
        ctx_counts_out = np.zeros(seg_len, dtype=np.float64)
        full_counts_out = np.zeros(seg_len, dtype=np.float64)
        mask = self.mask
        primes = self.PRIMES
        # backoff: highest order first
        for oi in range(self.num_orders - 1, -1, -1):
            order = self.min_order + oi
            cw = order - 1
            first_valid = max(cw, start) - start
            n_pos = seg_len - first_valid
            if n_pos <= 0:
                continue
            abs_s = start + first_valid
            ctx_hash = np.zeros(n_pos, dtype=np.uint64)
            for k in range(cw):
                t = val_np[abs_s - cw + k:abs_s - cw + k + n_pos].astype(np.uint64)
                ctx_hash ^= t * np.uint64(primes[k])
            ctx_key = (ctx_hash & mask).astype(np.int64)
            targets = val_np[abs_s + 1:abs_s + 1 + n_pos].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * np.uint64(primes[cw]))) & mask).astype(np.int64)
            ctx_c = self.ctx_counts[oi][ctx_key]
            full_c = self.full_counts[oi][full_key]
            valid = (ctx_c >= self.min_count) & (full_c > 0) & ~has_match[first_valid:first_valid + n_pos]
            if valid.any():
                idx = np.nonzero(valid)[0]
                capped_full = np.minimum(full_c[idx], ctx_c[idx]).astype(np.float64)
                p_ngram[first_valid + idx] = capped_full / ctx_c[idx].astype(np.float64)
                has_match[first_valid + idx] = True
                matched_order[first_valid + idx] = order
                ctx_counts_out[first_valid + idx] = ctx_c[idx].astype(np.float64)
                full_counts_out[first_valid + idx] = capped_full
        return p_ngram, has_match, matched_order, ctx_counts_out, full_counts_out

    def lookup_hierarchical(self, val_np: np.ndarray, start: int, end: int, concentration: float, base_p: np.ndarray) -> np.ndarray:
        """hierarchical Dirichlet mixing (CTW-style, PR #900 / Teh 2006).
        for each position, iterate from lowest to highest order. each order's posterior
        becomes the next order's prior: p = (c * p_prev + full_c) / (c + ctx_c).
        returns the final blended probability array."""
        seg_len = end - start
        blended = base_p.copy()
        mask = self.mask
        primes = self.PRIMES
        # iterate lowest to highest order — each posterior becomes next prior
        for oi in range(self.num_orders):
            order = self.min_order + oi
            cw = order - 1
            first_valid = max(cw, start) - start
            n_pos = seg_len - first_valid
            if n_pos <= 0:
                continue
            abs_s = start + first_valid
            ctx_hash = np.zeros(n_pos, dtype=np.uint64)
            for k in range(cw):
                t = val_np[abs_s - cw + k:abs_s - cw + k + n_pos].astype(np.uint64)
                ctx_hash ^= t * np.uint64(primes[k])
            ctx_key = (ctx_hash & mask).astype(np.int64)
            targets = val_np[abs_s + 1:abs_s + 1 + n_pos].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * np.uint64(primes[cw]))) & mask).astype(np.int64)
            ctx_c = self.ctx_counts[oi][ctx_key]
            full_c = np.minimum(self.full_counts[oi][full_key], ctx_c)
            valid = (ctx_c >= self.min_count) & (full_c > 0)
            if valid.any():
                idx = np.nonzero(valid)[0]
                fc = full_c[idx].astype(np.float64)
                cc = ctx_c[idx].astype(np.float64)
                prev_p = blended[first_valid + idx]
                blended[first_valid + idx] = (concentration * prev_p + fc) / (concentration + cc)
        return blended

    def update(self, val_np: np.ndarray, start: int, end: int) -> None:
        """update cache with tokens from [start, end)."""
        seg_len = end - start
        mask = self.mask
        primes = self.PRIMES
        for oi in range(self.num_orders):
            order = self.min_order + oi
            cw = order - 1
            first_valid = max(cw, start) - start
            n_pos = seg_len - first_valid
            if n_pos <= 0:
                continue
            abs_s = start + first_valid
            ctx_hash = np.zeros(n_pos, dtype=np.uint64)
            for k in range(cw):
                t = val_np[abs_s - cw + k:abs_s - cw + k + n_pos].astype(np.uint64)
                ctx_hash ^= t * np.uint64(primes[k])
            ctx_key = (ctx_hash & mask).astype(np.int64)
            targets = val_np[abs_s + 1:abs_s + 1 + n_pos].astype(np.uint64)
            full_key = ((ctx_hash ^ (targets * np.uint64(primes[cw]))) & mask).astype(np.int64)
            np.add.at(self.ctx_counts[oi], ctx_key, 1)
            np.add.at(self.full_counts[oi], full_key, 1)


def eval_val_ngram(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int,
    stride: int,
    batch_seqs: int = 32,
    ngram_order: int = 7,
    ngram_min_order: int = 2,
    ngram_buckets: int = 4194304,
    ngram_min_count: int = 2,
    fixed_alpha: float = 0.2,
    ent_base: float = 0.05,
    ent_range: float = 0.55,
    ent_scale: float = 2.0,
    ent_thresh: float = 4.0,
    dirichlet_concentration: float = 0.0,
    log_fn=None,
) -> tuple[float, float]:
    """sliding window eval with n-gram cache, matching PR #753/#769/#779.
    score-first: for each window, compute neural logits, lookup cache, mix, then update.
    if dirichlet_concentration > 0, uses Dirichlet-Multinomial posterior predictive mixing
    (PR #900 / CTW / Teh 2006) instead of linear interpolation."""
    total_tokens = val_tokens.numel() - 1
    seq_len = eval_seq_len
    vocab_size = args.vocab_size
    val_np = val_tokens[:total_tokens + 1].numpy()
    adaptive = ent_range > 0

    # distribute windows across ranks
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    model.eval()
    compiled_logits = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    cache = NgramCache(max_order=ngram_order, min_order=ngram_min_order,
                       num_buckets=ngram_buckets, min_count=ngram_min_count)

    # phrase cache (single-pass score-first, same as n-gram)
    phrase_cache = LongPhraseCache()

    # prefill: pre-warm both caches with all tokens before this rank's first window
    if my_windows:
        prefill_end = my_windows[0]
        if prefill_end > 0:
            chunk_sz = 65536
            for pf_start in range(0, prefill_end, chunk_sz):
                pf_end = min(pf_start + chunk_sz, prefill_end)
                cache.update(val_np, pf_start, pf_end)
                phrase_cache.update(val_np, pf_start, pf_end)
            if log_fn:
                log_fn(f"prefill: warmed caches with {prefill_end} tokens for rank {rank}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    loss_sum_neural = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    ngram_hits = 0
    ngram_total = 0
    base_bytes_cpu = base_bytes_lut.cpu()
    has_space_cpu = has_leading_space_lut.cpu()
    is_boundary_cpu = is_boundary_token_lut.cpu()

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
                logits = compiled_logits(x_batch)
            logits_f = logits.float()
            probs_all = torch.softmax(logits_f, dim=-1)
            log_probs_all = torch.log_softmax(logits_f, dim=-1)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                seg_len = wlen - s
                abs_start = ws + s
                abs_end = ws + wlen

                # neural prob of target
                seg_targets = y_batch[i, s:wlen]
                model_p = probs_all[i, s:wlen].gather(1, seg_targets.unsqueeze(1)).squeeze(1).cpu().numpy().astype(np.float64)
                seg_nll_neural = F.cross_entropy(logits_f[i, s:wlen], seg_targets, reduction='none').cpu().numpy().astype(np.float64)

                # n-gram: score-first (lookup THEN update)
                if dirichlet_concentration > 0:
                    # flat Dirichlet mixing (best of exp71-73 sweep)
                    p_ngram, has_match, matched_order, ng_ctx_c, ng_full_c = cache.lookup(val_np, abs_start, abs_end)
                    blended_p = model_p.copy()
                    if has_match.any():
                        m = has_match
                        conc = dirichlet_concentration
                        blended_p[m] = (conc * model_p[m] + ng_full_c[m]) / (conc + ng_ctx_c[m])
                else:
                    p_ngram, has_match, matched_order, _, _ = cache.lookup(val_np, abs_start, abs_end)
                    # legacy linear interpolation with per-order entropy thresholds
                    blended_p = model_p.copy()
                    if has_match.any():
                        m = has_match
                        ent_centers = {7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5, 8: 2.8, 9: 2.6}
                        if adaptive:
                            seg_ent = (-(probs_all[i, s:wlen] * log_probs_all[i, s:wlen]).sum(dim=-1)).cpu().numpy()
                            alpha = np.full(seg_len, fixed_alpha, dtype=np.float64)
                            for pos_idx in range(seg_len):
                                if has_match[pos_idx]:
                                    order = int(matched_order[pos_idx])
                                    center = ent_centers.get(order, ent_thresh)
                                    sig = 1.0 / (1.0 + np.exp(-ent_scale * (seg_ent[pos_idx] - center)))
                                    alpha[pos_idx] = ent_base + ent_range * sig
                        else:
                            alpha = np.full(seg_len, fixed_alpha, dtype=np.float64)
                        blended_p[m] = (1.0 - alpha[m]) * model_p[m] + alpha[m] * p_ngram[m]
                cache.update(val_np, abs_start, abs_end)

                # phrase cache: lookup THEN update (score-first)
                positions = np.arange(abs_start, abs_end, dtype=np.int64)
                p_phrase, phrase_match, phrase_len, phr_ctx_c, phr_full_c = phrase_cache.lookup(val_np, positions, min_count=2)
                phrase_cache.update(val_np, abs_start, abs_end)
                if phrase_match.any():
                    pm = phrase_match
                    if dirichlet_concentration > 0:
                        # phrase Dirichlet with lower concentration (phrases are more specific)
                        phr_conc = dirichlet_concentration * 0.2
                        blended_p[pm] = (phr_conc * blended_p[pm] + phr_full_c[pm]) / (phr_conc + phr_ctx_c[pm])
                    else:
                        pa = 0.3 + (0.95 - 0.3) * (phrase_len[phrase_match].astype(np.float64) - 16.0) / 32.0
                        pa = np.clip(pa, 0.0, 0.95)
                        blended_p[pm] = (1.0 - pa) * blended_p[pm] + pa * p_phrase[pm]

                blended_p = np.maximum(blended_p, 1e-30)
                seg_nll = -np.log(blended_p)

                loss_sum += float(seg_nll.sum())
                loss_sum_neural += float(seg_nll_neural.sum())
                token_count += float(seg_len)
                ngram_hits += int(has_match.sum())
                ngram_total += seg_len

                # bytes
                tgt_ids = seg_targets.cpu()
                prev_ids = x_batch[i, s:wlen].cpu()
                tb = base_bytes_cpu[tgt_ids].to(torch.float64)
                tb += (has_space_cpu[tgt_ids] & ~is_boundary_cpu[prev_ids]).to(torch.float64)
                byte_count += float(tb.sum())

    if dist.is_available() and dist.is_initialized():
        for t in [loss_sum, loss_sum_neural, token_count, byte_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_loss_neural = (loss_sum_neural / token_count).item()
    bpb = (val_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
    bpb_neural = (val_loss_neural / math.log(2.0)) * (token_count.item() / byte_count.item())
    hit_rate = ngram_hits / max(ngram_total, 1) * 100
    if log_fn:
        log_fn(f"neural_only_sw val_loss:{val_loss_neural:.4f} val_bpb:{bpb_neural:.4f}")
        log_fn(f"ngram_hit_rate:{hit_rate:.1f}% ({ngram_hits}/{ngram_total})")
        if dirichlet_concentration > 0:
            log_fn(f"mixing:hierarchical_dirichlet concentration={dirichlet_concentration:.2f} phrase_probes={LongPhraseCache.PROBE_LENGTHS}")
        else:
            log_fn(f"mixing:linear_interp adaptive={adaptive}")
    model.train()
    return val_loss, bpb


def eval_ngram_two_pass(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int,
    stride: int,
    batch_seqs: int = 32,
    ngram_order: int = 9,
    ngram_min_order: int = 2,
    ngram_buckets: int = 16777216,
    ngram_min_count: int = 2,
    ent_base: float = 0.05,
    ent_range: float = 0.55,
    ent_scale: float = 2.0,
    ent_thresh: float = 4.0,
    log_fn=None,
) -> tuple[float, float]:
    """two-pass n-gram eval (PR #870 BROADSIDE approach).
    pass 1: store model_p + entropy per scored position.
    build full cache from all val tokens.
    pass 2: rescore all positions with full cache."""
    total_tokens = val_tokens.numel() - 1
    seq_len = eval_seq_len
    val_np = val_tokens[:total_tokens + 1].numpy()
    ent_centers = {15: 1.8, 14: 1.9, 13: 2.0, 12: 2.1, 11: 2.2, 10: 2.4,
                   9: 2.6, 8: 2.8, 7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5}

    # distribute windows
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    model.eval()
    compiled_logits = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    base_bytes_cpu = base_bytes_lut.cpu()
    has_space_cpu = has_leading_space_lut.cpu()
    is_boundary_cpu = is_boundary_token_lut.cpu()

    # pass 1: store model_p, entropy, bytes per scored position
    stored_positions = []
    stored_model_p = []
    stored_entropy = []
    stored_bytes = []

    if log_fn:
        log_fn(f"two_pass: pass 1 — storing model predictions for {len(my_windows)} windows")

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
                logits = compiled_logits(x_batch)
            logits_f = logits.float()
            probs_all = torch.softmax(logits_f, dim=-1)
            log_probs_all = torch.log_softmax(logits_f, dim=-1)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                seg_targets = y_batch[i, s:wlen]
                model_p = probs_all[i, s:wlen].gather(1, seg_targets.unsqueeze(1)).squeeze(1).cpu().numpy().astype(np.float64)
                seg_ent = (-(probs_all[i, s:wlen] * log_probs_all[i, s:wlen]).sum(dim=-1)).cpu().numpy().astype(np.float64)
                # positions (global target token indices)
                positions = np.arange(ws + s, ws + wlen, dtype=np.int64)
                # bytes
                tgt_ids = seg_targets.cpu()
                prev_ids = x_batch[i, s:wlen].cpu()
                tb = base_bytes_cpu[tgt_ids].to(torch.float64)
                tb += (has_space_cpu[tgt_ids] & ~is_boundary_cpu[prev_ids]).to(torch.float64)

                stored_positions.append(positions)
                stored_model_p.append(model_p)
                stored_entropy.append(seg_ent)
                stored_bytes.append(tb.numpy())

    # concatenate all stored data
    all_positions = np.concatenate(stored_positions)
    all_model_p = np.concatenate(stored_model_p)
    all_entropy = np.concatenate(stored_entropy)
    all_bytes = np.concatenate(stored_bytes)

    if log_fn:
        neural_loss = -np.log(np.maximum(all_model_p, 1e-30)).mean()
        neural_bpb = (neural_loss / math.log(2.0)) * (len(all_model_p) / all_bytes.sum())
        log_fn(f"two_pass: pass 1 done, {len(all_model_p)} positions, neural_bpb={neural_bpb:.4f}")

    # build full cache from ALL val tokens
    if log_fn:
        log_fn(f"two_pass: building full cache ({total_tokens} tokens, {ngram_order}-gram, {ngram_buckets} buckets)")
    cache = NgramCache(max_order=ngram_order, min_order=ngram_min_order,
                       num_buckets=ngram_buckets, min_count=ngram_min_count)
    cache.build_full(val_np, log_fn=log_fn)

    # pass 2: rescore all stored positions using full cache
    if log_fn:
        log_fn(f"two_pass: pass 2 — rescoring {len(all_positions)} positions with full cache")

    # lookup n-gram probs for all stored positions (vectorized per order)
    n_pos = len(all_positions)
    p_ngram = np.zeros(n_pos, dtype=np.float64)
    has_match = np.zeros(n_pos, dtype=np.bool_)
    matched_order = np.zeros(n_pos, dtype=np.int32)
    mask = cache.mask
    primes = cache.PRIMES

    for oi in range(cache.num_orders - 1, -1, -1):
        order = cache.min_order + oi
        cw = order - 1
        # positions with enough context
        valid = (all_positions >= cw) & ~has_match
        if not valid.any():
            continue
        pos_valid = all_positions[valid]
        # context hash
        ctx_hash = np.zeros(len(pos_valid), dtype=np.uint64)
        for k in range(cw):
            t = val_np[(pos_valid - cw + k).astype(np.int64)].astype(np.uint64)
            ctx_hash ^= t * np.uint64(primes[k])
        ctx_key = (ctx_hash & mask).astype(np.int64)
        # full hash
        targets = val_np[(pos_valid + 1).astype(np.int64)].astype(np.uint64)
        full_key = ((ctx_hash ^ (targets * np.uint64(primes[cw]))) & mask).astype(np.int64)
        # lookup
        ctx_c = cache.ctx_counts[oi][ctx_key]
        full_c = np.minimum(cache.full_counts[oi][full_key], ctx_c)
        eligible = (ctx_c >= ngram_min_count) & (full_c > 0)
        if eligible.any():
            valid_idx = np.where(valid)[0][eligible]
            p_ngram[valid_idx] = full_c[eligible].astype(np.float64) / ctx_c[eligible].astype(np.float64)
            has_match[valid_idx] = True
            matched_order[valid_idx] = order

    # per-order multipliers: boost higher orders, suppress low orders (PR #870/#782)
    order_mults = {2: 0.3, 3: 0.3, 4: 0.7, 5: 1.0, 6: 1.5, 7: 2.0, 8: 2.0, 9: 2.0,
                   10: 2.0, 11: 2.0, 12: 2.0, 13: 2.0, 14: 2.0, 15: 2.0}

    # compute per-position alpha with per-order entropy thresholds + multipliers
    alpha = np.full(n_pos, 0.05, dtype=np.float64)
    matched_idx = np.where(has_match)[0]
    if len(matched_idx) > 0:
        orders = matched_order[matched_idx]
        entropies = all_entropy[matched_idx]
        # vectorized: compute centers and multipliers
        centers = np.array([ent_centers.get(int(o), ent_thresh) for o in orders])
        mults = np.array([order_mults.get(int(o), 1.0) for o in orders])
        sig = 1.0 / (1.0 + np.exp(-ent_scale * (entropies - centers)))
        raw_alpha = (ent_base + ent_range * sig) * mults
        alpha[matched_idx] = np.clip(raw_alpha, 0.0, 0.95)

    # blend n-gram
    blended_p = all_model_p.copy()
    m = has_match
    if m.any():
        blended_p[m] = (1.0 - alpha[m]) * all_model_p[m] + alpha[m] * p_ngram[m]

    # phrase cache: second layer of blending for long verbatim repetitions
    if log_fn:
        log_fn(f"two_pass: building phrase cache...")
    phrase_cache = LongPhraseCache()
    phrase_cache.build_full(val_np, log_fn=log_fn)
    p_phrase, phrase_match, phrase_len, _, _ = phrase_cache.lookup(val_np, all_positions, min_count=2)
    if phrase_match.any():
        # alpha based on match length: longer = higher trust (up to 0.99 for 48-token match)
        base_alpha = 0.3
        phrase_alpha = base_alpha + (0.99 - base_alpha) * (phrase_len[phrase_match].astype(np.float64) - 16.0) / 32.0
        phrase_alpha = np.clip(phrase_alpha, 0.0, 0.99)
        pm = phrase_match
        blended_p[pm] = (1.0 - phrase_alpha) * blended_p[pm] + phrase_alpha * p_phrase[pm]
        if log_fn:
            log_fn(f"phrase_cache: {phrase_match.sum()} matches, mean_len={phrase_len[phrase_match].mean():.1f}")

    blended_p = np.maximum(blended_p, 1e-30)
    blended_nll = -np.log(blended_p)

    # aggregate
    loss_sum_t = torch.tensor(float(blended_nll.sum()), device=device, dtype=torch.float64)
    token_count_t = torch.tensor(float(n_pos), device=device, dtype=torch.float64)
    byte_count_t = torch.tensor(float(all_bytes.sum()), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count_t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum_t / token_count_t).item()
    bpb = (val_loss / math.log(2.0)) * (token_count_t.item() / byte_count_t.item())
    hit_rate = has_match.sum() / max(n_pos, 1) * 100
    if log_fn:
        log_fn(f"two_pass: hit_rate={hit_rate:.1f}%, val_loss={val_loss:.4f}, val_bpb={bpb:.4f}")
    model.train()
    return val_loss, bpb


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_int6_per_row(t: Tensor, clip_range: int = 15) -> tuple[Tensor, Tensor]:
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
def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):
    num_layers_total = max(
        (int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
        default=0,
    ) + 1
    late_k_layers = set(range(num_layers_total - 2, num_layers_total))
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta
def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out
def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
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
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    CastedLinear._qat_enabled = args.qat_enabled
    base_model = GPT(
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
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.mtp_num_heads > 0:
        matrix_params.extend([p for p in base_model.mtp_heads.parameters() if p.ndim == 2])
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            matrix_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params = sum(p.numel() for p in base_model.mtp_heads.parameters())
    log0(f"model_params:{n_params}")
    log0(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
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
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
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
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = 0.997
    training_time_ms = 0.0
    stop_after_step: int | None = None
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
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
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
        # EMA update
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    # Apply EMA weights (better than SWA alone per PR#401)
    log0("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    # skip diagnostic eval to save eval-time budget
    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)
    if excluded_mtp > 0:
        log0(f"export_excluding_mtp_params:{excluded_mtp}")
    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw) if _COMPRESSOR == "zstd" else zlib.compress(quant_raw, 9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{_COMPRESSOR}: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+{_COMPRESSOR}: {quant_file_bytes + code_bytes} bytes")
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob_disk) if _COMPRESSOR == "zstd" else zlib.decompress(quant_blob_disk)),
        map_location="cpu",
    )
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,  # must match training model
        rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    # eval_model is used directly by n-gram eval (which compiles internally)

    # TTT: preeval (bulk train then score) or legal (score-first, chunk by chunk)
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 0))
    ttt_lr = float(os.environ.get("TTT_LR", 0.0005))
    ttt_mode = os.environ.get("TTT_MODE", "preeval")  # "preeval" or "legal"
    if ttt_epochs > 0 and ttt_mode == "preeval":
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        log0(f"ttt: starting {ttt_epochs} epochs, lr={ttt_lr}, cosine+perlayer")
        # per-layer LR groups: 3x for MLP output projections, 0.5x for MLP input
        proj_params, fc_params, other_params = [], [], []
        for name, p in eval_model.named_parameters():
            p.requires_grad_(True)
            if "mlp.proj" in name:
                proj_params.append(p)
            elif "mlp.fc" in name:
                fc_params.append(p)
            else:
                other_params.append(p)
        ttt_opt = torch.optim.AdamW([
            {"params": proj_params, "lr": ttt_lr * 3.0},
            {"params": fc_params, "lr": ttt_lr * 0.5},
            {"params": other_params, "lr": ttt_lr},
        ], weight_decay=0.0)
        total_val = val_tokens.numel() - 1
        ttt_batch = 32
        rank_tokens = total_val // world_size
        rank_start = rank * rank_tokens
        rank_end = rank_start + rank_tokens
        steps_per_epoch = max(1, (rank_end - rank_start - args.train_seq_len) // (ttt_batch * args.train_seq_len))
        total_steps = ttt_epochs * steps_per_epoch
        global_step = 0
        eval_model.train()
        for ep in range(ttt_epochs):
            ep_loss, ep_steps = 0.0, 0
            for bs in range(rank_start, rank_end - args.train_seq_len, ttt_batch * args.train_seq_len):
                be = min(bs + ttt_batch * args.train_seq_len + 1, rank_end + 1)
                local = val_tokens[bs:be].to(device=device, dtype=torch.int64)
                n = (local.numel() - 1) // args.train_seq_len
                if n == 0:
                    continue
                x = local[:n * args.train_seq_len].reshape(n, args.train_seq_len)
                y = local[1:n * args.train_seq_len + 1].reshape(n, args.train_seq_len)
                # cosine LR schedule
                progress = global_step / max(total_steps, 1)
                cos_mul = 0.5 * (1.0 + math.cos(math.pi * progress))
                for g in ttt_opt.param_groups:
                    g["lr"] = g.get("initial_lr", g["lr"]) * cos_mul
                if global_step == 0:
                    for g in ttt_opt.param_groups:
                        g["initial_lr"] = g["lr"]
                ttt_opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = eval_model(x, y)
                loss.backward()
                # sync gradients across ranks
                if distributed:
                    for p in eval_model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(eval_model.parameters(), 1.0)
                ttt_opt.step()
                ep_loss += loss.item()
                ep_steps += 1
                global_step += 1
            if master_process and (ep + 1) % 5 == 0:
                log0(f"ttt_epoch:{ep + 1}/{ttt_epochs} avg_loss:{ep_loss / max(ep_steps, 1):.4f}")
        del ttt_opt
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log0(f"ttt: completed in {1000.0 * (time.perf_counter() - t_ttt):.0f}ms")

    # legal score-first TTT: score chunk, then train on scored tokens
    if ttt_epochs > 0 and ttt_mode == "legal":
        torch.cuda.synchronize(); t_ttt = time.perf_counter()
        sl = effective_eval_seq_len; st = args.eval_stride if args.eval_stride > 0 else sl; scl = min(st, sl)
        for p in eval_model.parameters(): p.requires_grad_(False)
        nb = len(eval_model.blocks) if hasattr(eval_model, 'blocks') else 0
        tp = []
        for nm, p in eval_model.named_parameters():
            bi = next((i for i in range(nb) if f"blocks.{i}." in nm), -1)
            if bi >= nb - 2 or any(k in nm for k in ("norm","scale","q_gain","lm_head","tok_emb","smear","bigram")):
                p.requires_grad_(True); tp.append(p)
        to = torch.optim.AdamW(tp, lr=ttt_lr * 0.2, weight_decay=0.0)
        log0(f"legal_ttt: {len(tp)} params, {ttt_epochs}ep/chunk")
        tot = val_tokens.numel() - 1; cs = 65536
        ns, nc, nb2 = torch.zeros((),dtype=torch.float64,device=device), torch.zeros((),dtype=torch.float64,device=device), torch.zeros((),dtype=torch.float64,device=device)
        for c0 in range(0, tot - sl + 1, cs):
            eval_model.eval()
            with torch.inference_mode():
                for ws in range(c0, min(c0+cs, tot-sl+1), st*world_size):
                    s = ws + rank*st
                    if s+sl > tot: continue
                    x = val_tokens[s:s+sl].to(device=device,dtype=torch.int64).unsqueeze(0)
                    y = val_tokens[s+1:s+sl+1].to(device=device,dtype=torch.int64).unsqueeze(0)
                    with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
                        lo = eval_model.forward_logits(x) if hasattr(eval_model,'forward_logits') else None
                    if lo is not None:
                        sf = sl-scl; lt = lo[:,sf:,:].reshape(-1,lo.size(-1)).float(); tt = y[:,sf:].reshape(-1)
                        ns += F.cross_entropy(lt,tt,reduction="sum").to(torch.float64); nc += scl
                        pr,tg = x[:,sf:].reshape(-1), tt
                        tb = base_bytes_lut[tg].to(torch.int16) + (has_leading_space_lut[tg]&~is_boundary_token_lut[pr]).to(torch.int16)
                        nb2 += tb.to(torch.float64).sum()
            eval_model.train()
            ct = val_tokens[c0:min(c0+cs+sl,tot+1)].to(device=device,dtype=torch.int64)
            nq = (ct.numel()-1)//sl
            if nq > 0:
                for _ in range(ttt_epochs):
                    xc,yc = ct[:nq*sl].reshape(nq,sl), ct[1:nq*sl+1].reshape(nq,sl)
                    for bi in range(0,nq,4):
                        xb,yb = xc[bi:bi+4], yc[bi:bi+4]
                        if xb.shape[0]==0: continue
                        to.zero_grad()
                        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True): l=eval_model(xb,yb)
                        l.backward(); to.step()
        if distributed:
            for t in (ns,nc,nb2): dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if nc.item()>0:
            ll=ns.item()/nc.item(); bb=float(ll/math.log(2.0)*nc.item()/nb2.item())
            log0(f"legal_ttt val_loss:{ll:.4f} val_bpb:{bb:.4f} time:{1000*(time.perf_counter()-t_ttt):.0f}ms")
            log0(f"legal_ttt_exact val_loss:{ll:.8f} val_bpb:{bb:.8f}")
        del to; torch.cuda.empty_cache()

    # n-gram cache eval (includes sliding window — replaces standalone sw eval)
    ngram_enabled = bool(int(os.environ.get("NGRAM_ENABLED", "1")))
    sw_seq_len = effective_eval_seq_len
    if ngram_enabled:
        ngram_order = int(os.environ.get("NGRAM_ORDER", "13"))
        ngram_min_order = int(os.environ.get("NGRAM_MIN_ORDER", "2"))
        ngram_buckets = int(os.environ.get("NGRAM_BUCKETS", "4194304"))
        ngram_min_count = int(os.environ.get("NGRAM_MIN_COUNT", "2"))
        ngram_alpha = float(os.environ.get("NGRAM_ALPHA", "0.2"))
        ngram_ent_base = float(os.environ.get("NGRAM_ENT_BASE", "0.05"))
        ngram_ent_range = float(os.environ.get("NGRAM_ENT_RANGE", "0.90"))
        ngram_ent_scale = float(os.environ.get("NGRAM_ENT_SCALE", "2.0"))
        ngram_ent_thresh = float(os.environ.get("NGRAM_ENT_THRESH", "4.0"))
        dirichlet_conc = float(os.environ.get("DIRICHLET_CONCENTRATION", "1.0"))
        torch.cuda.synchronize()
        t_ngram = time.perf_counter()
        ngram_two_pass = bool(int(os.environ.get("NGRAM_TWO_PASS", "0")))  # default single-pass for legality
        log0(f"ngram_eval: order={ngram_order} min_order={ngram_min_order} buckets={ngram_buckets} two_pass={ngram_two_pass} dirichlet={dirichlet_conc}")
        if ngram_two_pass:
            ng_val_loss, ng_val_bpb = eval_ngram_two_pass(
                args, eval_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                eval_seq_len=sw_seq_len if args.eval_stride > 0 else effective_eval_seq_len,
                stride=args.eval_stride if args.eval_stride > 0 else effective_eval_seq_len,
                ngram_order=ngram_order, ngram_min_order=ngram_min_order,
                ngram_buckets=16777216,
                ngram_min_count=ngram_min_count,
                ent_base=ngram_ent_base, ent_range=ngram_ent_range,
                ent_scale=ngram_ent_scale, ent_thresh=ngram_ent_thresh,
                log_fn=log0,
            )
        else:
            ng_val_loss, ng_val_bpb = eval_val_ngram(
                args, eval_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                eval_seq_len=sw_seq_len if args.eval_stride > 0 else effective_eval_seq_len,
                stride=args.eval_stride if args.eval_stride > 0 else effective_eval_seq_len,
                ngram_order=ngram_order, ngram_min_order=ngram_min_order,
                ngram_buckets=ngram_buckets, ngram_min_count=ngram_min_count,
                fixed_alpha=ngram_alpha,
                ent_base=ngram_ent_base, ent_range=ngram_ent_range,
                dirichlet_concentration=dirichlet_conc,
                ent_scale=ngram_ent_scale, ent_thresh=ngram_ent_thresh,
                log_fn=log0,
            )
        torch.cuda.synchronize()
        log0(f"ngram_eval val_loss:{ng_val_loss:.4f} val_bpb:{ng_val_bpb:.4f} eval_time:{1000.0*(time.perf_counter()-t_ngram):.0f}ms")
        log0(f"ngram_eval_exact val_loss:{ng_val_loss:.8f} val_bpb:{ng_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{ng_val_loss:.8f} val_bpb:{ng_val_bpb:.8f}")
    else:
        if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
            torch.cuda.synchronize()
            t_slide = time.perf_counter()
            sw_val_loss, sw_val_bpb = eval_val_sliding(
                args, eval_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=args.eval_stride, eval_seq_len=sw_seq_len,
            )
            torch.cuda.synchronize()
            log0(f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} stride:{args.eval_stride} eval_time:{1000.0*(time.perf_counter()-t_slide):.0f}ms")
            log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
