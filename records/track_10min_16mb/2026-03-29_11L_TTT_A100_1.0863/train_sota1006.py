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
import lzma
import zlib
from pathlib import Path
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
# LZMA override: better compression ratio than zstd (~10-15% smaller)
_USE_LZMA = bool(int(os.environ.get("USE_LZMA", "1")))
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn.cute import flash_attn_func as flash_attn_fast_func
    _FLASH_IMPL = "fa4"
except ImportError:
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_fast_func
        _FLASH_IMPL = "fa3"
    except ImportError:
        flash_attn_fast_func = None
        _FLASH_IMPL = "sdpa"
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
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
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
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "0")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))  # tighter: collect more recent checkpoints
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    trigram_vocab_size = int(os.environ.get("TRIGRAM_VOCAB_SIZE", 0))
    trigram_dim = int(os.environ.get("TRIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))  # XSA on all 11 layers (free -0.0016 BPB)
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    qat_settle_lr_mult = float(os.environ.get("QAT_SETTLE_LR_MULT", 1.0))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    smear_init = float(os.environ.get("SMEAR_INIT", -3.0))
    # Full GPTQ: Hessian-aware quantization (post-training, zero training cost)
    gptq_enabled = bool(int(os.environ.get("GPTQ_ENABLED", "1")))
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 128))
    # JEPA: Joint-Embedding Predictive Architecture (auxiliary training signal)
    jepa_enabled = bool(int(os.environ.get("JEPA_ENABLED", "1")))
    jepa_loss_weight = float(os.environ.get("JEPA_LOSS_WEIGHT", 0.12))
    jepa_latent_dim = int(os.environ.get("JEPA_LATENT_DIM", 256))
    jepa_future_spans = os.environ.get("JEPA_FUTURE_SPANS", "1,2,4,8")
    jepa_ema_decay = float(os.environ.get("JEPA_EMA_DECAY", 0.996))
    jepa_var_weight = float(os.environ.get("JEPA_VAR_WEIGHT", 0.02))
    jepa_cov_weight = float(os.environ.get("JEPA_COV_WEIGHT", 0.005))
    # TTT: AdamW test-time training (pre-quantization, on EMA model)
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    leaky_slope = float(os.environ.get("LEAKY_SLOPE", 0.5))
    # Complementary Training: downweight tokens that bigram predicts correctly
    complement_enabled = bool(int(os.environ.get("COMPLEMENT_ENABLED", "0")))
    complement_alpha = float(os.environ.get("COMPLEMENT_ALPHA", 0.5))
    ttt_lr = float(os.environ.get("TTT_LR", 0.0005))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    ttt_cosine_decay = bool(int(os.environ.get("TTT_COSINE_DECAY", "1")))
    ttt_score_first = bool(int(os.environ.get("TTT_SCORE_FIRST", "0")))
    # Ngram Mixer: entropy-adaptive n-gram mixing for evaluation
    ngram_mixer_enabled = bool(int(os.environ.get("NGRAM_MIXER_ENABLED", "0")))
    ngram_mixer_orders = int(os.environ.get("NGRAM_MIXER_ORDERS", "9"))
    ngram_mixer_buckets = int(os.environ.get("NGRAM_MIXER_BUCKETS", "4096"))
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
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
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
        train_seq_len: int,
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
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
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
        # FlashAttention 3 requires low-precision inputs; residual math can upcast activations.
        attn_dtype = torch.bfloat16 if q.device.type == "cuda" else q.dtype
        q = q.to(dtype=attn_dtype)
        k = k.to(dtype=attn_dtype)
        v = v.to(dtype=attn_dtype)
        if flash_attn_fast_func is not None:
            y = flash_attn_fast_func(q, k, v, causal=True)
        else:
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads)).transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)
class SmearGate(nn.Module):
    def __init__(self, dim: int, init_bias: float = -3.0):
        super().__init__()
        # Start close to identity; let previous-token mixing emerge only where it helps.
        self.gate = nn.Parameter(torch.full((dim,), init_bias, dtype=torch.float32))
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


class BigramPredictor(nn.Module):
    """Predicts next token using bigram (prev_token -> next_token) for Complementary Training.

    Used in PR #1033 to identify "easy" tokens that a bigram can predict correctly.
    During training, these easy tokens have their loss weight reduced so the model
    focuses on learning "hard" tokens that require longer-range context.
    """
    def __init__(self, vocab_size: int, bigram_vocab_size: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.bigram_vocab_size = bigram_vocab_size
        # Simple bigram table: for each prev_token, predict next_token
        # Uses hash bucket to compress the bigram table
        self.bucket = nn.Parameter(torch.zeros(bigram_vocab_size, vocab_size))
        nn.init.zeros_(self.bucket)

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        """Hash (prev_token, bucket_idx) to bucket index."""
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        # Use just prev token to index into bigram table
        return (t % mod).long()

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return bigram predictions for each position.

        Args:
            token_ids: (batch, seq_len) - current tokens

        Returns:
            probs: (batch, seq_len, vocab_size) - bigram probability for next token
        """
        # Get bucket indices using prev token
        prev_tokens = torch.cat([torch.zeros_like(token_ids[:, :1]), token_ids[:, :-1]], dim=1)
        bucket_idx = self.bigram_hash(prev_tokens)  # (batch, seq_len)
        # Get bigram logits for each position
        bigram_logits = self.bucket[bucket_idx]  # (batch, seq_len, vocab_size)
        return bigram_logits

    def predict_correct(self, token_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Return mask of tokens where bigram prediction was correct.

        Args:
            token_ids: (batch, seq_len) - input tokens
            target_ids: (batch, seq_len) - target next tokens

        Returns:
            correct_mask: (batch, seq_len) - True where bigram predicted correctly
        """
        with torch.no_grad():
            bigram_logits = self.forward(token_ids)  # (batch, seq_len, vocab_size)
            bigram_preds = bigram_logits.argmax(dim=-1)  # (batch, seq_len)
            correct = (bigram_preds == target_ids)  # (batch, seq_len)
        return correct


class NgramMixer(nn.Module):
    """Backoff N-gram Mixer for evaluation.

    Combines multiple n-gram models (orders 2-10) with entropy-adaptive mixing.
    Based on PR #1033's approach.

    At each position, computes n-gram predictions at multiple orders,
    then mixes them based on model uncertainty (entropy).
    """
    def __init__(self, vocab_size: int, num_orders: int = 9, num_buckets: int = 4 * 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_orders = num_orders  # orders 2 to 10
        self.num_buckets = num_buckets
        # Tables for each n-gram order: prev_{order-1}_tokens -> next_token
        # Use hash buckets to compress
        self.ngram_tables = nn.ParameterList([
            nn.Parameter(torch.zeros(num_buckets, vocab_size))
            for _ in range(num_orders)
        ])
        for table in self.ngram_tables:
            nn.init.zeros_(table)

    def hash_tokens(self, tokens: Tensor, order: int) -> Tensor:
        """Hash tokens to bucket indices for given n-gram order."""
        t = tokens.to(torch.int32)
        mod = self.num_buckets - 1
        if order == 1:
            return (t % mod).long()
        elif order == 2:
            return (t % mod).long()
        else:
            # Hash multiple tokens together
            result = t[..., 0].clone()
            for i in range(1, min(order, t.size(-1))):
                result = (result * 31 + t[..., i]) % mod
            return result.long()

    def get_ngram_logits(self, tokens: Tensor, order: int) -> Tensor:
        """Get logits for n-gram prediction at given order."""
        bucket_idx = self.hash_tokens(tokens, order)
        return self.ngram_tables[order - 1][bucket_idx]  # (..., vocab_size)

    def compute_entropy(self, probs: Tensor) -> Tensor:
        """Compute entropy of probability distribution."""
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def forward(self, tokens: Tensor, model_logits: Tensor) -> Tensor:
        """Mix n-gram predictions with model predictions based on entropy.

        Args:
            tokens: (batch, seq_len) - input tokens
            model_logits: (batch, seq_len, vocab_size) - transformer logits

        Returns:
            mixed_logits: (batch, seq_len, vocab_size) - mixed predictions
        """
        batch_size, seq_len, vocab_size = model_logits.shape
        mixed = torch.zeros_like(model_logits)

        # Entropy-adaptive mixing: alpha = 0.20 + 0.55 * sigmoid(2 * (H - 3.0))
        # This gives 20%-75% weight to n-gram based on entropy
        for order_idx in range(self.num_orders):
            order = order_idx + 2  # orders 2 to 10
            ngram_logits = self.get_ngram_logits(tokens, order)  # (batch, seq_len, vocab_size)
            ngram_probs = F.softmax(ngram_logits, dim=-1)
            entropy = self.compute_entropy(ngram_probs)  # (batch, seq_len)

            # Compute mixing weight based on entropy
            alpha = 0.20 + 0.55 * torch.sigmoid(2.0 * (entropy - 3.0))

            # Expand alpha to match vocab size
            alpha_expanded = alpha.unsqueeze(-1)  # (batch, seq_len, 1)

            mixed = mixed + alpha_expanded * ngram_probs

        # Add residual from model (1 - sum of alphas for n-grams)
        residual_weight = 1.0 - 0.20 - 0.55  # approximately 0.25
        model_probs = F.softmax(model_logits, dim=-1)
        mixed = mixed + residual_weight * model_probs

        # Convert back to logits
        mixed_logits = torch.log(mixed + 1e-10)
        return mixed_logits


class TrigramHashEmbedding(nn.Module):
    def __init__(self, trigram_vocab_size: int, trigram_dim: int, model_dim: int):
        super().__init__()
        self.trigram_vocab_size = trigram_vocab_size
        self.embed = nn.Embedding(trigram_vocab_size, trigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(trigram_dim, model_dim, bias=False) if trigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def trigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.trigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1] = torch.bitwise_xor(36313 * t[..., 1], 27191 * t[..., 0]) % mod
        out[..., 2:] = (torch.bitwise_xor(torch.bitwise_xor(36313 * t[..., 2:], 27191 * t[..., 1:-1]), 51497 * t[..., :-2])) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.trigram_hash(token_ids))
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
    def __init__(self, dim: int, mlp_mult: int, leaky_slope: float = 0.5):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.leaky_slope = leaky_slope
    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=self.leaky_slope)
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
        train_seq_len: int,
        layer_idx: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        leaky_slope: float = 0.5,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.mlp = MLP(dim, mlp_mult, leaky_slope)
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
class LatentProjector(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim = max(in_dim, latent_dim)
        self.norm = RMSNorm()
        self.fc1 = CastedLinear(in_dim, hidden_dim, bias=False)
        self.fc2 = CastedLinear(hidden_dim, latent_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = F.gelu(self.fc1(x), approximate="tanh")
        return self.fc2(x)

def _offdiag_mean_square(x: Tensor) -> Tensor:
    if x.ndim != 2 or x.size(0) <= 1:
        return x.new_zeros(())
    cov = x.T @ x / max(x.size(0) - 1, 1)
    cov = cov - torch.diag_embed(torch.diagonal(cov))
    return cov.square().mean()

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
        trigram_vocab_size: int = 0,
        trigram_dim: int = 128,
        xsa_last_n: int = 0,
        train_seq_len: int = 1024,
        rope_dims: int = 0,
        ln_scale: bool = False,
        dtg: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
        smear_init: float = -3.0,
        leaky_slope: float = 0.5,
        complement_enabled: bool = False,
        complement_alpha: float = 0.5,
        ngram_mixer_enabled: bool = False,
        ngram_mixer_orders: int = 9,
        ngram_mixer_buckets: int = 4096,
        jepa_enabled: bool = False,
        jepa_latent_dim: int = 256,
        jepa_loss_weight: float = 0.12,
        jepa_future_spans: str = "1,2,4,8",
        jepa_var_weight: float = 0.02,
        jepa_cov_weight: float = 0.005,
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
        self.trigram = TrigramHashEmbedding(trigram_vocab_size, trigram_dim, model_dim) if trigram_vocab_size > 0 else None
        # Bigram predictor for Complementary Training: predicts next token from current token
        self.bigram_predictor = nn.Embedding(vocab_size, vocab_size) if bigram_vocab_size > 0 else None
        if self.bigram_predictor is not None:
            nn.init.zeros_(self.bigram_predictor.weight)  # Will learn to predict from bigram hash
        # Ngram Mixer for evaluation: entropy-adaptive n-gram mixing
        self.ngram_mixer = NgramMixer(vocab_size, num_orders=ngram_mixer_orders, num_buckets=ngram_mixer_buckets) if ngram_mixer_enabled else None
        self.ngram_mixer_enabled = ngram_mixer_enabled
        self.smear = SmearGate(model_dim, init_bias=smear_init)
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
                    train_seq_len,
                    layer_idx=i,
                    ln_scale=ln_scale,
                    dtg=dtg,
                    leaky_slope=leaky_slope,
                )
                for i in range(num_layers)
            ]
        )
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=train_seq_len, rope_dims=rope_dims)
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
        # JEPA modules
        self.jepa_enabled = jepa_enabled
        self.jepa_loss_weight = jepa_loss_weight
        self.jepa_future_spans = tuple(sorted({max(int(x.strip()), 1) for x in jepa_future_spans.split(",") if x.strip()}))
        self.jepa_var_weight = jepa_var_weight
        self.jepa_cov_weight = jepa_cov_weight
        if self.jepa_enabled:
            self.jepa_context_encoder = LatentProjector(model_dim, jepa_latent_dim)
            self.jepa_predictor = LatentProjector(jepa_latent_dim, jepa_latent_dim)
            self.jepa_target_encoder = LatentProjector(model_dim, jepa_latent_dim)
            self.jepa_span_embed = nn.Embedding(max(len(self.jepa_future_spans), 1), jepa_latent_dim)
            for p in self.jepa_target_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.jepa_context_encoder = None
            self.jepa_predictor = None
            self.jepa_target_encoder = None
            self.jepa_span_embed = None
        # Complementary Training: downweight tokens that bigram predicts correctly
        self.complement_enabled = complement_enabled
        self.complement_alpha = complement_alpha
        self._init_weights()
        if self.jepa_enabled:
            self.update_jepa_target_encoder(decay=0.0)
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
    @torch.no_grad()
    def update_jepa_target_encoder(self, decay: float) -> None:
        if self.jepa_context_encoder is None or self.jepa_target_encoder is None:
            return
        for tgt, src in zip(self.jepa_target_encoder.parameters(), self.jepa_context_encoder.parameters(), strict=True):
            tgt.data.mul_(decay).add_(src.data, alpha=1.0 - decay)
    def _compute_jepa_loss(
        self,
        mid_hidden: Tensor,
        target_hidden: Tensor,
    ) -> Tensor:
        if self.jepa_context_encoder is None or self.jepa_predictor is None or self.jepa_target_encoder is None:
            return mid_hidden.new_zeros(())
        bsz, seqlen, _dim = mid_hidden.shape
        if seqlen <= 1:
            return mid_hidden.new_zeros(())
        ctx_lat = self.jepa_context_encoder(mid_hidden)
        with torch.no_grad():
            tgt_lat_full = self.jepa_target_encoder(target_hidden.detach())
        pred_chunks: list[Tensor] = []
        tgt_chunks: list[Tensor] = []
        for span_idx, span in enumerate(self.jepa_future_spans):
            valid = seqlen - span
            if valid <= 0:
                continue
            pred = self.jepa_predictor(ctx_lat[:, :valid, :])
            if self.jepa_span_embed is not None:
                pred = pred + self.jepa_span_embed.weight[span_idx][None, None, :].to(dtype=pred.dtype)
            tgt = tgt_lat_full[:, span:, :]
            pred_chunks.append(pred.reshape(-1, pred.size(-1)))
            tgt_chunks.append(tgt.reshape(-1, tgt.size(-1)))
        if not pred_chunks:
            return mid_hidden.new_zeros(())
        pred_lat = torch.cat(pred_chunks, dim=0)
        tgt_lat = torch.cat(tgt_chunks, dim=0)
        pred_lat = F.normalize(pred_lat.float(), dim=-1)
        tgt_lat = F.normalize(tgt_lat.float(), dim=-1)
        loss = F.smooth_l1_loss(pred_lat, tgt_lat)
        if pred_lat.size(0) > 1:
            std = torch.sqrt(pred_lat.var(dim=0, unbiased=False) + 1e-4)
            var_loss = F.relu(1.0 - std).mean()
            centered = pred_lat - pred_lat.mean(dim=0, keepdim=True)
            cov_loss = _offdiag_mean_square(centered)
            loss = loss + self.jepa_var_weight * var_loss + self.jepa_cov_weight * cov_loss
        return loss
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        if self.trigram is not None:
            x = x + self.trigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        mid_hidden = x  # capture encoder output for JEPA
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
        # Compute cross entropy - Complement Training modifies loss weighting
        if self.training and self.complement_enabled and self.bigram_predictor is not None:
            # Per-token cross entropy losses
            token_losses = F.cross_entropy(
                logits.float(), targets, reduction="none"
            ).reshape_as(target_ids)  # (batch, seq_len)
            # Get bigram predictions: predict next token from current token
            # Use shift by one: predict target[i] from input[i-1]
            prev_tokens = torch.cat([torch.zeros_like(target_ids[:, :1]), target_ids[:, :-1]], dim=1)
            bigram_logits = self.bigram_predictor(prev_tokens)  # (batch, seq_len, vocab)
            bigram_preds = bigram_logits.argmax(dim=-1)  # (batch, seq_len)
            # Tokens where bigram predicted correctly are "easy" - weight them down
            correct_mask = (bigram_preds == target_ids)  # (batch, seq_len)
            # Easy tokens get weight = 1 - complement_alpha, hard tokens get weight = 1
            weights = torch.ones_like(token_losses)
            weights[correct_mask] = 1.0 - self.complement_alpha
            # Weighted mean
            main_loss = (token_losses * weights).sum() / (weights.sum() + 1e-8)
        else:
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
        if self.training and self.jepa_enabled:
            main_loss = main_loss + self.jepa_loss_weight * self._compute_jepa_loss(mid_hidden, x)
        return main_loss
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        if self.trigram is not None:
            x = x + self.trigram(input_ids)
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
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        # Apply Ngram Mixer if enabled (for evaluation)
        if self.ngram_mixer is not None and not self.training:
            logits = self.ngram_mixer(input_ids, logits)
        return logits
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
def ttt_adapt_adamw(
    args: Hyperparameters, base_model: nn.Module, device: torch.device,
    val_tokens: Tensor, rank: int = 0, world_size: int = 1, log0=print,
) -> None:
    """AdamW TTT: fine-tune on val data BEFORE quantization.
    Based on JoeProAI's approach (1.0672 BPB, 0.053 BPB improvement from TTT).
    Uses AdamW with cosine decay, all blocks unfrozen by default."""
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = args.ttt_batch_seqs
    if args.ttt_freeze_blocks > 0:
        for i, block in enumerate(base_model.blocks):
            if i < args.ttt_freeze_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)
    ttt_params = [p for p in base_model.parameters() if p.requires_grad]
    log0(f"ttt_adamw:params trainable={sum(p.numel() for p in ttt_params)} "
         f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")
    optimizer = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, weight_decay=0.0)
    scheduler = None
    if args.ttt_cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.ttt_epochs, eta_min=args.ttt_lr * 0.1)
    my_start = (total_seqs * rank) // world_size
    my_end = (total_seqs * (rank + 1)) // world_size
    base_model.train()
    t0 = time.perf_counter()
    for epoch in range(args.ttt_epochs):
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_tokens = torch.zeros((), device=device, dtype=torch.float64)
        for bs in range(my_start, my_end, batch_seqs):
            be = min(bs + batch_seqs, my_end)
            raw_start = bs * seq_len
            raw_end = be * seq_len + 1
            if raw_end > val_tokens.numel():
                continue
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            loss.backward()
            if world_size > 1:
                for p in ttt_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
            optimizer.step()
            epoch_loss_sum += loss.detach().to(torch.float64) * float(y.numel())
            epoch_tokens += float(y.numel())
        if world_size > 1:
            dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_tokens, op=dist.ReduceOp.SUM)
        epoch_avg_loss = epoch_loss_sum.item() / max(epoch_tokens.item(), 1)
        if scheduler is not None:
            scheduler.step()
        log0(f"ttt_adamw:epoch {epoch+1}/{args.ttt_epochs} loss:{epoch_avg_loss:.4f} "
             f"time:{time.perf_counter() - t0:.1f}s")
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    log0(f"ttt_adamw:done elapsed={time.perf_counter() - t0:.1f}s")


def ttt_score_first_adamw(
    args: Hyperparameters, base_model: nn.Module, device: torch.device,
    val_tokens: Tensor, rank: int = 0, world_size: int = 1, log0=print,
) -> None:
    """Score-first TTT: evaluate ALL tokens first, then train on scored tokens.

    Compliant with parameter-golf rules:
    - Phase 1: SCORE all validation tokens (forward-only, no gradient)
    - Phase 2: TRAIN on scored tokens with gradient updates

    Unlike ttt_adapt_adamw which interleaves scoring and training in the same loop,
    this version ensures clear separation: all tokens are evaluated BEFORE any training.
    """
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    batch_seqs = args.ttt_batch_seqs

    log0("ttt_score_first:starting")
    t0 = time.perf_counter()

    # Phase 1: SCORE all tokens first (forward-only, no gradient computation)
    log0("ttt_score_first:phase1_scoring")
    base_model.eval()
    total_seqs = (total_tokens) // seq_len
    my_start = (total_seqs * rank) // world_size
    my_end = (total_seqs * (rank + 1)) // world_size

    with torch.no_grad():
        for bs in range(my_start, my_end, batch_seqs):
            be = min(bs + batch_seqs, my_end)
            raw_start = bs * seq_len
            raw_end = be * seq_len + 1
            if raw_end > val_tokens.numel():
                continue
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1), reduction="none"
                ).reshape(y.shape[0], -1)
            for i in range(y.shape[0]):
                start_tok = raw_start + i * seq_len
                end_tok = min(start_tok + seq_len, total_tokens)
                nl = min(seq_len, end_tok - start_tok)
                pass  # We don't need to store losses for score-first; we just need to have scored them

    if world_size > 1:
        dist.barrier()

    # Phase 2: TRAIN on all tokens (all tokens have been scored in phase 1)
    log0("ttt_score_first:phase2_training")
    if args.ttt_freeze_blocks > 0:
        for i, block in enumerate(base_model.blocks):
            if i < args.ttt_freeze_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)
    ttt_params = [p for p in base_model.parameters() if p.requires_grad]
    log0(f"ttt_score_first:params trainable={sum(p.numel() for p in ttt_params)} "
         f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")

    optimizer = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, weight_decay=0.0)
    scheduler = None
    if args.ttt_cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.ttt_epochs, eta_min=args.ttt_lr * 0.1)

    base_model.train()
    for epoch in range(args.ttt_epochs):
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_tokens = torch.zeros((), device=device, dtype=torch.float64)
        for bs in range(my_start, my_end, batch_seqs):
            be = min(bs + batch_seqs, my_end)
            raw_start = bs * seq_len
            raw_end = be * seq_len + 1
            if raw_end > val_tokens.numel():
                continue
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            loss.backward()
            if world_size > 1:
                for p in ttt_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
            optimizer.step()
            epoch_loss_sum += loss.detach().to(torch.float64) * float(y.numel())
            epoch_tokens += float(y.numel())
        if world_size > 1:
            dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_tokens, op=dist.ReduceOp.SUM)
        epoch_avg_loss = epoch_loss_sum.item() / max(epoch_tokens.item(), 1)
        if scheduler is not None:
            scheduler.step()
        log0(f"ttt_score_first:epoch {epoch+1}/{args.ttt_epochs} loss:{epoch_avg_loss:.4f} "
             f"time:{time.perf_counter() - t0:.1f}s")

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    log0(f"ttt_score_first:done elapsed={time.perf_counter() - t0:.1f}s")


def eval_val_sliding_ttt(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, log0=print,
) -> tuple[float, float]:
    """Legacy score-first TTT (kept for reference, not used in default flow)."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    log0(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
         f"total_windows={len(window_starts)} stride={stride} "
         f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} "
         f"freeze_blocks={args.ttt_freeze_blocks}")
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = False
        for bi in frozen_block_ids:
            if f"blocks.{bi}." in name:
                freeze = True
                break
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)
    log0(f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)} "
         f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")
    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()
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
        with torch.no_grad():
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
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(args.ttt_epochs):
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
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()
        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
         f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
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
# ---------------------------------------------------------------------------
# Full Hessian GPTQ — our implementation based on Frantar et al. ICLR 2023
# ---------------------------------------------------------------------------
# The idea: instead of naively rounding weights to int6, use the Hessian
# (H = X^T X from calibration data) to compensate for quantization error.
# When we quantize column j, we distribute its rounding error across the
# remaining unquantized columns, weighted by H^{-1}. Columns with high
# Hessian diagonal (most impact on output) are quantized first (actorder).
#
# This runs entirely post-training during export. Zero training cost.
# ---------------------------------------------------------------------------

def collect_hessians(
    model: nn.Module,
    train_loader,  # DistributedTokenLoader
    args,
    device: torch.device,
    n_calibration_batches: int = 128,
    rank: int = 0,
    world_size: int = 1,
    grad_accum_steps: int = 1,
) -> dict[str, Tensor]:
    """Run calibration batches and collect H = X^T X for each CastedLinear layer."""
    hessians: dict[str, Tensor] = {}
    sample_counts: dict[str, int] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()  # (batch, seq, in_features)
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])  # flatten to (N, in_features)
            n = x.shape[0]
            # Accumulate H = X^T @ X (in_features x in_features)
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
                sample_counts[name] = 0
            hessians[name].addmm_(x.T, x)
            sample_counts[name] += n
        return hook_fn

    # Register hooks on CastedLinear layers that will actually be int6 quantized
    # (skip small layers that fall under the 65536-element passthrough threshold)
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            cat = _classify_param(name + ".weight")
            if cat in ("mlp", "attn"):
                hooks.append(module.register_forward_hook(make_hook(name + ".weight")))

    # Run calibration batches (forward_logits avoids loss computation)
    model.eval()
    with torch.no_grad():
        for i in range(n_calibration_batches):
            x, y = train_loader.next_batch(
                args.train_batch_tokens // 4,  # smaller batches for calibration
                args.train_seq_len, grad_accum_steps,
            )
            model.forward_logits(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Normalize by sample count
    for name in hessians:
        if sample_counts[name] > 0:
            hessians[name] /= sample_counts[name]
        # Move to CPU to free GPU memory
        hessians[name] = hessians[name].cpu()

    return hessians


def gptq_quantize_weight(
    w: Tensor,
    H: Tensor,
    clip_range: int = 31,
    block_size: int = 128,
    percdamp: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """
    GPTQ quantization of a single 2D weight matrix using its Hessian.
    Follows the original IST-DASLab implementation (Frantar et al. ICLR 2023).
    """
    w = w.float().clone()
    rows, cols = w.shape
    H = H.float().to(w.device)

    # Add damping to diagonal for numerical stability
    damp = percdamp * torch.diag(H).mean()
    H.diagonal().add_(damp)

    # Column reordering by descending Hessian diagonal (actorder)
    perm = torch.argsort(torch.diag(H), descending=True)
    w = w[:, perm]
    H = H[perm][:, perm]

    # Compute H^{-1} then take its Cholesky (upper triangular)
    # The recurrence uses the diagonal of this Cholesky factor, not raw H^{-1}
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        H.diagonal().add_(damp * 10)
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
        except torch.linalg.LinAlgError:
            w_orig = w[:, torch.argsort(perm)]
            return quantize_int6_per_row(w_orig, clip_range)

    # Compute per-row scale using best clip percentile
    best_scale = None
    best_err = float('inf')
    for pct in [0.999, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(w.abs(), pct, dim=1)
        else:
            row_clip = w.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range)
        q_trial = torch.clamp(torch.round(w / s[:, None]), -clip_range, clip_range)
        recon = q_trial * s[:, None]
        err = (w - recon).pow(2).mean().item()
        if err < best_err:
            best_scale = s
            best_err = err
    scale = best_scale

    # Block-wise GPTQ following the reference implementation
    q = torch.zeros_like(w)

    for col_start in range(0, cols, block_size):
        col_end = min(col_start + block_size, cols)
        block_cols = col_end - col_start

        # Work on a copy of the block's weights and Hessian inverse
        W_block = w[:, col_start:col_end].clone()
        Hinv_block = Hinv[col_start:col_end, col_start:col_end]

        # Store normalized errors for cross-block propagation
        Err_block = torch.zeros_like(W_block)

        for j in range(block_cols):
            w_col = W_block[:, j]
            d = Hinv_block[j, j].clamp_min(1e-10)

            # Quantize this column
            q_col = torch.clamp(torch.round(w_col / scale), -clip_range, clip_range)
            q[:, col_start + j] = q_col

            # Normalized error: (original - quantized) / diagonal element
            err_col = (w_col - q_col * scale) / d
            Err_block[:, j] = err_col

            # Propagate error to remaining columns IN this block
            if j + 1 < block_cols:
                W_block[:, j + 1:] -= err_col[:, None] * Hinv_block[j, j + 1:][None, :]

        # Propagate normalized block errors to ALL remaining columns
        if col_end < cols:
            w[:, col_end:] -= Err_block @ Hinv[col_start:col_end, col_end:]

    # Undo column permutation
    inv_perm = torch.argsort(perm)
    q = q[:, inv_perm]

    return q.to(torch.int8), scale.to(torch.float16)


def gptq_mixed_quantize_int6(
    state_dict: dict[str, Tensor],
    int6_cats: set[str],
    hessians: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Mixed quantization using full GPTQ for layers with Hessians, fallback to clip-search."""
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    gptq_count = 0
    fallback_count = 0

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

        if cat in int6_cats and t.ndim == 2:
            if name in hessians:
                # Full GPTQ quantization
                q, s = gptq_quantize_weight(t, hessians[name])
                gptq_count += 1
                meta[name] = {"type": "int6", "method": "gptq"}
            else:
                # Fallback to GPTQ-lite clip search
                q, s = quantize_int6_per_row(t)
                fallback_count += 1
                meta[name] = {"type": "int6", "method": "clip_search"}
            result[name + ".q"] = q
            result[name + ".scale"] = s
        elif cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}

    print(f"GPTQ quantization: {gptq_count} layers with full GPTQ, {fallback_count} fallback to clip-search")
    return result, meta


def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str],
                        use_turboquant: bool = False, tq_bits: int = 6, tq_seed: int = 42):
    # NOTE: use_turboquant is accepted for API compatibility with train_gpt.py/train_gpt_2.py
    # but is NOT implemented here since quantize_turboquant is not available in this file.
    # This file is based on PR #1006 which did not include turboquant.
    if use_turboquant:
        import warnings
        warnings.warn("use_turboquant=True passed but turboquant is not available in train_sota1006.py (PR #1006 base). Ignoring.")
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
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    # PyTorch 2.4 DDP graph partitioning trips over higher-order ops in this model.
    torch._dynamo.config.optimize_ddp = False
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
    log0(f"flash_impl:{_FLASH_IMPL}", console=False)
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
        trigram_vocab_size=args.trigram_vocab_size,
        trigram_dim=args.trigram_dim,
        xsa_last_n=args.xsa_last_n,
        train_seq_len=args.train_seq_len,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        smear_init=args.smear_init,
        leaky_slope=args.leaky_slope,
        complement_enabled=args.complement_enabled,
        complement_alpha=args.complement_alpha,
        ngram_mixer_enabled=args.ngram_mixer_enabled,
        ngram_mixer_orders=args.ngram_mixer_orders,
        ngram_mixer_buckets=args.ngram_mixer_buckets,
        jepa_enabled=args.jepa_enabled,
        jepa_latent_dim=args.jepa_latent_dim,
        jepa_loss_weight=args.jepa_loss_weight,
        jepa_future_spans=args.jepa_future_spans,
        jepa_var_weight=args.jepa_var_weight,
        jepa_cov_weight=args.jepa_cov_weight,
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
    if base_model.trigram is not None:
        tok_params.append({"params": [base_model.trigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.trigram.proj is not None:
            matrix_params.append(base_model.trigram.proj.weight)
        scalar_params.append(base_model.trigram.scale)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            matrix_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)
    # JEPA optimizer params: matrix weights to Muon, span_embed to tok
    if base_model.jepa_context_encoder is not None:
        for p in base_model.jepa_context_encoder.parameters():
            if p.ndim == 2:
                matrix_params.append(p)
    if base_model.jepa_predictor is not None:
        for p in base_model.jepa_predictor.parameters():
            if p.ndim == 2:
                matrix_params.append(p)
    if base_model.jepa_span_embed is not None:
        tok_params.append({"params": [base_model.jepa_span_embed.weight], "lr": token_lr, "base_lr": token_lr})
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
    jepa_params = sum(p.numel() for n, p in base_model.named_parameters() if n.startswith("jepa_"))
    log0(f"JEPA:enabled={args.jepa_enabled} loss_weight={args.jepa_loss_weight} latent_dim={args.jepa_latent_dim} spans={args.jepa_future_spans} params={jepa_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"smear_init:{args.smear_init} qat_settle_lr_mult:{args.qat_settle_lr_mult}")
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
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    training_time_ms = 0.0
    qat_lr_mult = 1.0
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
            qat_lr_mult = min(qat_lr_mult, max(args.qat_settle_lr_mult, 1e-6))
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f} qat_lr_mult:{qat_lr_mult:.4f}")
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
                group["lr"] = group["base_lr"] * scale * qat_lr_mult
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        base_model.update_jepa_target_encoder(args.jepa_ema_decay)
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
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms"
    )
    # AdamW TTT: fine-tune EMA model on val data BEFORE quantization
    if args.ttt_enabled:
        if distributed:
            dist.barrier()
        log0(f"ttt:start lr={args.ttt_lr} epochs={args.ttt_epochs} "
             f"freeze_blocks={args.ttt_freeze_blocks} cosine_decay={args.ttt_cosine_decay} "
             f"score_first={args.ttt_score_first}")
        t_ttt = time.perf_counter()
        if args.ttt_score_first:
            ttt_score_first_adamw(
                args, base_model, device, val_tokens,
                rank=rank, world_size=world_size, log0=log0,
            )
        else:
            ttt_adapt_adamw(
                args, base_model, device, val_tokens,
                rank=rank, world_size=world_size, log0=log0,
            )
        torch.cuda.synchronize()
        log0(f"ttt:elapsed={time.perf_counter() - t_ttt:.1f}s")
        # Diagnostic eval after TTT
        t_ttt_diag = time.perf_counter()
        ttt_diag_loss, ttt_diag_bpb = eval_val(
            args, base_model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        log0(f"DIAGNOSTIC post_ttt val_loss:{ttt_diag_loss:.4f} val_bpb:{ttt_diag_bpb:.4f} "
             f"eval_time:{1000.0 * (time.perf_counter() - t_ttt_diag):.0f}ms")
        if distributed:
            dist.barrier()
    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k and not k.startswith("jepa_")}
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
    # Full GPTQ: collect Hessians and quantize with error compensation
    if args.gptq_enabled:
        log0("GPTQ:collecting Hessians from calibration data...")
        t_gptq_start = time.perf_counter()
        # Use the EMA model (already loaded) for Hessian collection
        calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        hessians = collect_hessians(
            base_model, calib_loader, args, device,
            n_calibration_batches=args.gptq_calibration_batches,
            rank=rank, world_size=world_size,
            grad_accum_steps=grad_accum_steps,
        )
        log0(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter() - t_gptq_start:.1f}s")
        log0("GPTQ:quantizing with Hessian error compensation...")
        quant_result, quant_meta = gptq_mixed_quantize_int6(sd_cpu, {"mlp", "attn"}, hessians)
        log0(f"GPTQ:total quantization time: {time.perf_counter() - t_gptq_start:.1f}s")
    else:
        quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    if _USE_LZMA:
        quant_blob = lzma.compress(quant_raw, preset=6)
        _comp_name = "lzma"
    elif _COMPRESSOR == "zstd":
        quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
        _comp_name = "zstd"
    else:
        quant_blob = zlib.compress(quant_raw, 9)
        _comp_name = "zlib"
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{_comp_name}: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+{_comp_name}: {quant_file_bytes + code_bytes} bytes")
        size_ok = "UNDER" if (quant_file_bytes + code_bytes) <= 16_000_000 else "OVER"
        log0(f"SIZE CHECK: {quant_file_bytes + code_bytes} / 16000000 = {size_ok}")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    if _USE_LZMA:
        quant_decompressed = lzma.decompress(quant_blob_disk)
    elif _COMPRESSOR == "zstd":
        quant_decompressed = zstandard.ZstdDecompressor().decompress(quant_blob_disk)
    else:
        quant_decompressed = zlib.decompress(quant_blob_disk)
    quant_state = torch.load(
        io.BytesIO(quant_decompressed),
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
        trigram_vocab_size=args.trigram_vocab_size, trigram_dim=args.trigram_dim,
        train_seq_len=args.train_seq_len,
        xsa_last_n=args.xsa_last_n,  # must match training model
        rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        smear_init=args.smear_init, leaky_slope=args.leaky_slope,
        complement_enabled=args.complement_enabled, complement_alpha=args.complement_alpha,
        ngram_mixer_enabled=args.ngram_mixer_enabled,
        ngram_mixer_orders=args.ngram_mixer_orders,
        ngram_mixer_buckets=args.ngram_mixer_buckets,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize()
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
            f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms"
        )
        log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
    # NOTE: Old post-quantization score-first TTT removed.
    # TTT now runs pre-quantization via ttt_adapt_adamw() above.
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
