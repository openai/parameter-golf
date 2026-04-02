from __future__ import annotations
import copy
import glob
import io
import lzma
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
import os as _os
_use_compile = _os.environ.get("USE_COMPILE", "1") == "1"
_USE_FUSED_MLP = _os.environ.get("FUSED_MLP", "0") == "1"
if _USE_FUSED_MLP:
    import sys as _sys; _sys.path.insert(0, "/home/charlie")
    from fused_leaky_mlp import FusedLinearLeakyReLUSquareFunction
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as _F_sdpa
try:
    from flash_attn_interface import flash_attn_func as _real_fa3
    _USE_FA3 = True
except ImportError:
    _USE_FA3 = False
def flash_attn_3_func(q, k, v, causal=True):
    if _USE_FA3:
        return _real_fa3(q, k, v, causal=causal)
    return _F_sdpa.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, enable_gqa=True
    ).transpose(1, 2)
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
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    lawa_enabled = bool(int(os.environ.get("LAWA_ENABLED", "0")))
    lawa_k = int(os.environ.get("LAWA_K", 10))
    lawa_freq = int(os.environ.get("LAWA_FREQ", 100))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2816))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))  # Changed from 4 to 11 (all layers)
    # SLOT (per-batch delta optimization at last hidden layer)
    slot_enabled = bool(int(os.environ.get("SLOT_ENABLED", "0")))
    slot_lr = float(os.environ.get("SLOT_LR", 0.003))
    slot_steps = int(os.environ.get("SLOT_STEPS", 5))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "0")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.0))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    ttt_muon = bool(int(os.environ.get("TTT_MUON", "1")))        # use Muon-style NS update for TTT
    ttt_ns_steps = int(os.environ.get("TTT_NS_STEPS", "5"))      # Newton-Schulz steps for TTT Muon
    use_gptq = bool(int(os.environ.get("USE_GPTQ", "0")))
    gptq_calib_samples = int(os.environ.get("GPTQ_CALIB_SAMPLES", "64"))
    gptq_reserve_ms = float(os.environ.get("GPTQ_RESERVE_MS", "10000"))
    quant_clip_range = int(os.environ.get("QUANT_CLIP_RANGE", 31))
    # --- WARP-Output word type hyperparameters ---
    num_word_types = int(os.environ.get("NUM_WORD_TYPES", 64))
    type_head_lr = float(os.environ.get("TYPE_HEAD_LR", 0.04))

# --- Batched Newton-Schulz orthogonalization ---

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Batched Newton-Schulz orthogonalization. G: (B,M,N) or (M,N)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X

# --- Parallel Muon optimizer ---

class Muon(torch.optim.Optimizer):
    """Parallel Muon: post-backward reduce-scatter -> local NS5 -> all-gather.

    No DDP for bank params. After backward, this optimizer:
    1. Launches async reduce-scatter for all banks (biggest first)
    2. Returns control so Adam can step on small params while RS is in-flight
    3. Waits for each RS, runs local NS5 on the shard, launches async all-gather
    4. Each all-gather overlaps with next bank's NS5
    """
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size

        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    'p': p,
                    'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        # Sort by size descending -- launch biggest reduce-scatters first
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        """Phase 1: launch async reduce-scatter for all banks. Call right after backward."""
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']:
                pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        """Phase 3: wait for RS, local NS5, all-gather. Call AFTER Adam steps."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self._built:
            self._build()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)

            prev_ag_handle = None
            prev_m = None

            sharded = self._distributed and hasattr(self, '_rs_futures')

            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None:
                    continue

                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m['shard']
                    buf = m['shard_mom']
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                update = zeropower_via_newtonschulz5(update, steps=backend_steps)

                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])

            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

            if hasattr(self, '_rs_futures'):
                del self._rs_futures

        return loss

# --- Tokenizer evaluation helpers ---

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
        if piece.startswith("\u2581"):
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

# --- Quantization helpers ---

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda,warp_attn.layer_scales",
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

# --- Data loading ---

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
_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}

def _read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n

def _get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm

class DistributedTokenLoader:
    """Coprime-stride + multi-shard loader (PR #726 style). No daemon thread / prefetch."""
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self._num_tokens = np.array([_read_num_tokens(f) for f in self.files], dtype=np.int64)
        seed = 0
        for f in self.files:
            for b in str(f).encode():
                seed = ((seed ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._cfg: tuple[int, int, int, int] | None = None
        self._eligible_shards: np.ndarray | None = None
        self._base_block_counts: np.ndarray | None = None
        n = len(self.files)
        self._cursor_phase = np.zeros(n, dtype=np.int64)
        self._cursor_block_count = np.zeros(n, dtype=np.int64)
        self._cursor_next = np.zeros(n, dtype=np.int64)
        self._cursor_start = np.zeros(n, dtype=np.int64)
        self._cursor_stride = np.ones(n, dtype=np.int64)
        self._cursor_init = np.zeros(n, dtype=np.bool_)
        self._batches_built = 0
    def _pick_coprime_stride(self, n: int) -> int:
        if n <= 1:
            return 1
        while True:
            s = int(self._rng.integers(1, n))
            if math.gcd(s, n) == 1:
                return s
    def _reset_cursor(self, si: int, seq_len: int) -> None:
        nt = int(self._num_tokens[si])
        max_phase = min(seq_len - 1, max(0, nt - seq_len - 1))
        phase = int(self._rng.integers(max_phase + 1)) if max_phase > 0 else 0
        bc = (nt - 1 - phase) // seq_len
        self._cursor_phase[si] = phase
        self._cursor_block_count[si] = bc
        self._cursor_next[si] = 0
        self._cursor_start[si] = int(self._rng.integers(bc)) if bc > 1 else 0
        self._cursor_stride[si] = self._pick_coprime_stride(bc)
        self._cursor_init[si] = True
    def _ensure_cursor(self, si: int, seq_len: int) -> None:
        if not self._cursor_init[si] or self._cursor_next[si] >= self._cursor_block_count[si]:
            self._reset_cursor(si, seq_len)
    def _take_from_shard(self, si: int, seq_len: int, count: int, out: list) -> None:
        rem = count
        while rem > 0:
            self._ensure_cursor(si, seq_len)
            bc = int(self._cursor_block_count[si])
            ni = int(self._cursor_next[si])
            take = min(rem, bc - ni)
            phase = int(self._cursor_phase[si])
            start = int(self._cursor_start[si])
            stride = int(self._cursor_stride[si])
            for j in range(take):
                bi = (start + (ni + j) * stride) % bc
                out.append((si, phase + bi * seq_len))
            self._cursor_next[si] = ni + take
            rem -= take
    def _init_pipeline(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> None:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        num_seqs = local_tokens // seq_len
        global_num_seqs = num_seqs * self.world_size
        self._cfg = (local_tokens, seq_len, num_seqs, global_num_seqs)
        bbc = (self._num_tokens - 1) // seq_len
        eligible = bbc > 0
        self._eligible_shards = np.nonzero(eligible)[0].astype(np.int64)
        self._base_block_counts = bbc[self._eligible_shards].astype(np.int64)
    def _sample_global_windows(self) -> list:
        assert self._cfg is not None and self._eligible_shards is not None
        _, seq_len, _, gns = self._cfg
        ec = int(self._eligible_shards.size)
        progress = min(self._batches_built / 1800.0, 1.0)
        remaining = np.empty(ec, dtype=np.float64)
        for i, si in enumerate(self._eligible_shards.tolist()):
            if self._cursor_init[si]:
                r = int(self._cursor_block_count[si]) - int(self._cursor_next[si])
                remaining[i] = float(max(r, 1))
            else:
                remaining[i] = float(self._base_block_counts[i])
        alpha = 0.90 - 0.40 * progress
        weights = np.power(remaining, alpha)
        ws = float(weights.sum())
        if not np.isfinite(ws) or ws <= 0.0:
            weights = np.ones(ec, dtype=np.float64)
            ws = float(weights.sum())
        probs = weights / ws
        low = min(max(8, self.world_size), ec, gns)
        high = min(max(32, self.world_size * 8), ec, gns)
        mix = max(1, min(int(round(low + progress * (high - low))), ec, gns))
        cp = self._rng.choice(ec, size=mix, replace=False, p=probs)
        cs = self._eligible_shards[cp]
        cpr = probs[cp].copy()
        cpr /= cpr.sum()
        counts = np.ones(mix, dtype=np.int64)
        extra = gns - mix
        if extra > 0:
            counts += self._rng.multinomial(extra, cpr).astype(np.int64)
        perm = self._rng.permutation(mix)
        cs, counts = cs[perm], counts[perm]
        buckets = []
        for si, cnt in zip(cs.tolist(), counts.tolist()):
            b = []
            self._take_from_shard(int(si), seq_len, int(cnt), b)
            if b:
                if len(b) > 1:
                    bp = self._rng.permutation(len(b))
                    b = [b[int(k)] for k in bp.tolist()]
                buckets.append(b)
        windows = []
        active = [i for i, bk in enumerate(buckets) if bk]
        while active:
            order = self._rng.permutation(len(active))
            new_active = []
            for oi in order.tolist():
                bi = active[oi]
                if buckets[bi]:
                    windows.append(buckets[bi].pop())
                if buckets[bi]:
                    new_active.append(bi)
            active = new_active
        return windows
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        if self._cfg is None:
            self._init_pipeline(global_tokens, seq_len, grad_accum_steps)
        _, _, num_seqs, gns = self._cfg
        gw = self._sample_global_windows()
        local_w = gw[self.rank::self.world_size]
        x = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        y = torch.empty((num_seqs, seq_len), dtype=torch.int64)
        for slot, (si, pos) in enumerate(local_w):
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[pos:pos + seq_len + 1], dtype=np.int64))
            x[slot] = window[:-1]
            y[slot] = window[1:]
        self._batches_built += 1
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# --- Transformer modules ---

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
        # No CastedLinear -- weights come from banks
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0  # set by GPT.__init__ for partial RoPE
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False  # set by GPT.__init__ for deep layers only
        self.warp_attn = None  # set by GPT.__init__ if WARP-Attn enabled
        self.layer_idx = 0     # set by GPT.__init__
    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Efficient XSA: subtract self-value projection via GQA-aware reshape (no repeat_interleave).
        y: [B, T, H, D], v: [B, T, Hkv, D]. H must be divisible by Hkv."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)        # [B, T, Hkv, group, D]
        vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D] -- broadcast ready
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, v_embed: Tensor | None = None, position_in_word: Tensor | None = None) -> Tensor:
        if getattr(self, '_save_gptq', False):
            self._gptq_qkv_in = x.detach()
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # WARP-Attn: add word position bias to BOTH Q and K BEFORE RoPE
        if self.warp_attn is not None and position_in_word is not None:
            q = q + self.warp_attn.get_q_bias(position_in_word, self.layer_idx).unsqueeze(2)
            k = k + self.warp_attn.get_k_bias(position_in_word, self.layer_idx).unsqueeze(2)
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        if getattr(self, '_save_gptq', False):
            self._gptq_o_in = y.detach()
        return F.linear(y, out_w.to(x.dtype))

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
        # No CastedLinear -- weights come from banks
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        if getattr(self, '_save_gptq', False):
            self._gptq_up_in = x.detach()
        if _USE_FUSED_MLP:
            result = FusedLinearLeakyReLUSquareFunction.apply(x, up_w.to(x.dtype), down_w.to(x.dtype).T.contiguous())
            if getattr(self, '_save_gptq', False):
                with torch.no_grad():
                    act = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
                    self._gptq_down_in = act.square().detach()
            return result
        x2 = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
        x2 = x2.square()
        if getattr(self, '_save_gptq', False):
            self._gptq_down_in = x2.detach()
        return F.linear(x2, down_w.to(x.dtype))
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
    def forward(self, x: Tensor, x0: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, up_w: Tensor, down_w: Tensor, v_embed: Tensor | None = None, position_in_word: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, q_w, k_w, v_w, out_w, v_embed=v_embed, position_in_word=position_in_word)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out

# --- WARP: Word boundary detection, attention bias, and output logit bias ---

def compute_word_boundary_maps(input_ids: Tensor, has_leading_space_lut: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Detect word boundaries from SentencePiece tokens. All ops are torch.compile friendly."""
    B, T = input_ids.shape
    device = input_ids.device
    is_word_start = has_leading_space_lut[input_ids]
    is_word_start[:, 0] = True
    word_ids = is_word_start.long().cumsum(dim=1) - 1
    positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    word_last_pos = torch.zeros(B, T, dtype=torch.long, device=device)
    word_last_pos.scatter_(1, word_ids, positions)
    reversed_positions = (T - 1) - positions
    word_first_pos_rev = torch.zeros(B, T, dtype=torch.long, device=device)
    word_first_pos_rev.scatter_(1, word_ids, reversed_positions)
    word_first_pos = (T - 1) - word_first_pos_rev
    prev_word_ids = (word_ids - 1).clamp(min=0)
    last_tok_of_prev_word = word_last_pos.gather(1, prev_word_ids)
    last_tok_of_prev_word[word_ids == 0] = 0
    first_tok_of_curr_word = word_first_pos.gather(1, word_ids)
    return word_ids, last_tok_of_prev_word, first_tok_of_curr_word

class WordLengthEmbed(nn.Module):
    def __init__(self, model_dim, max_len=13):
        super().__init__()
        self.embed = nn.Embedding(max_len, model_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, word_length):
        return self.embed(word_length) * self.scale

class WARPAttn(nn.Module):
    """Word position embeddings added to BOTH Q and K in attention."""
    def __init__(self, head_dim: int, num_layers: int, max_word_pos: int = 8):
        super().__init__()
        self.q_word_pos_embed = nn.Embedding(max_word_pos, head_dim)
        self.k_word_pos_embed = nn.Embedding(max_word_pos, head_dim)
        nn.init.normal_(self.q_word_pos_embed.weight, std=0.01)
        nn.init.normal_(self.k_word_pos_embed.weight, std=0.01)
        # Per-layer learned scale (num_layers layers)
        self.layer_scales = nn.Parameter(torch.full((num_layers,), 0.15))

    def get_q_bias(self, position_in_word: Tensor, layer_idx: int) -> Tensor:
        scale = self.layer_scales[layer_idx]
        return self.q_word_pos_embed(position_in_word) * scale

    def get_k_bias(self, position_in_word: Tensor, layer_idx: int) -> Tensor:
        scale = self.layer_scales[layer_idx]
        return self.k_word_pos_embed(position_in_word) * scale

class WARPOutput(nn.Module):
    """Classify word type from final hidden states, produce direct logit bias."""
    def __init__(self, model_dim: int, num_types: int, vocab_size: int):
        super().__init__()
        self.classifier_fc1 = CastedLinear(model_dim, 192, bias=False)
        self.classifier_fc2 = CastedLinear(192, num_types, bias=False)
        # DIRECT logit bias per type -- initialized to zeros, gets gradient from CE loss
        self.type_vocab_bias = nn.Parameter(torch.zeros(num_types, vocab_size))

    def forward(self, x_flat: Tensor) -> tuple[Tensor, Tensor]:
        """
        x_flat: [B*T, D] -- final normalized hidden states (after final_norm, reshaped)
        Returns:
            vocab_bias: [B*T, vocab_size] -- per-token logit bias
            type_logits: [B*T, num_types] -- raw type logits (for logging)
        """
        h = F.rms_norm(x_flat.float(), (x_flat.size(-1),))
        h = F.leaky_relu(self.classifier_fc1(h.to(x_flat.dtype)), negative_slope=0.5)
        type_logits = self.classifier_fc2(h).float()
        type_probs = F.softmax(type_logits, dim=-1)
        # Direct logit bias: type_probs @ type_vocab_bias
        vocab_bias = type_probs @ self.type_vocab_bias.float()  # [B*T, vocab_size]
        return vocab_bias, type_logits

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
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
        # WARP-Output params
        num_word_types: int = 64,
        has_leading_space_lut: Tensor | None = None,
        # unused legacy args kept for call-site compat
        dtg: bool = False,
        gated_attention: bool = False,
        value_residual: bool = False,
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
        # Parameter banks: contiguous 3D tensors for batched optimizer
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)
        self.num_layers = num_layers
        self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))
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
        kv_dim_ve = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
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
        # --- WARP-Attn: shared across all layers ---
        head_dim_warp = model_dim // num_heads
        self.warp_attn = WARPAttn(head_dim_warp, num_layers, max_word_pos=8)
        for i, block in enumerate(self.blocks):
            block.attn.warp_attn = self.warp_attn
            block.attn.layer_idx = i
        # --- WARP-Output module ---
        self.warp_output = WARPOutput(model_dim, num_word_types, vocab_size)
        # --- Word-Length Embedding ---
        self.word_length_embed = WordLengthEmbed(model_dim, max_len=13)
        # Register has_leading_space_lut as buffer for word boundary detection
        if has_leading_space_lut is not None:
            self.register_buffer("has_leading_space_lut", has_leading_space_lut, persistent=False)
        else:
            self.register_buffer("has_leading_space_lut", torch.zeros(vocab_size, dtype=torch.bool), persistent=False)
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        # Init banks: orthogonal, with proj layers scaled down and out/down zero-init
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)        # Q
            nn.init.zeros_(self.qo_bank.data[n + i])                    # Out (zero init)
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)        # K
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)    # V
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)    # MLP up
            nn.init.zeros_(self.mlp_down_bank.data[i])                  # MLP down (zero init)
            # Scale proj layers (out_proj and mlp_down are "proj" layers)
            self.qo_bank.data[n + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)
        # Init remaining nn.Linear modules (bigram proj, mtp heads, lm_head, WARP classifier)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
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
        n = self.num_layers
        B, T = input_ids.shape
        device = input_ids.device
        # Compute word boundaries and position_in_word for WARP-Attn
        _word_ids, _last_tok_prev, first_tok_curr = compute_word_boundary_maps(
            input_ids, self.has_leading_space_lut)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        position_in_word = (positions - first_tok_curr).clamp(0, 7)
        # Compute word_length for WordLengthEmbed
        word_token_counts = torch.zeros(B, T, dtype=torch.long, device=device)
        ones = torch.ones(B, T, dtype=torch.long, device=device)
        word_token_counts.scatter_add_(1, _word_ids, ones)
        word_length = word_token_counts.gather(1, _word_ids).clamp(1, 12)  # cap at 12
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = x + self.word_length_embed(word_length)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                v_embed=ve, position_in_word=position_in_word)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, position_in_word=position_in_word)
        # Output with WARP-Output logit bias
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        # WARP-Output: direct logit bias from word type classification
        vocab_bias, _type_logits = self.warp_output(x_flat)
        logits_proj = logits_proj + vocab_bias
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        return loss
    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        """Return final hidden states before lm_head."""
        n = self.num_layers
        B, T = input_ids.shape
        device = input_ids.device
        # Compute position_in_word for WARP-Attn
        _word_ids, _last_tok_prev, first_tok_curr = compute_word_boundary_maps(
            input_ids, self.has_leading_space_lut)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        position_in_word = (positions - first_tok_curr).clamp(0, 7)
        # Compute word_length for WordLengthEmbed
        word_token_counts = torch.zeros(B, T, dtype=torch.long, device=device)
        ones = torch.ones(B, T, dtype=torch.long, device=device)
        word_token_counts.scatter_add_(1, _word_ids, ones)
        word_length = word_token_counts.gather(1, _word_ids).clamp(1, 12)
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = x + self.word_length_embed(word_length)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                v_embed=ve, position_in_word=position_in_word)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, position_in_word=position_in_word)
        return self.final_norm(x)

    def compute_logits(self, hidden_states: Tensor) -> Tensor:
        """Compute logits from hidden states, including WARP-Output bias."""
        x_flat = hidden_states.reshape(-1, hidden_states.size(-1))
        if self.tie_embeddings:
            lp = F.linear(x_flat, self.tok_emb.weight)
        else:
            lp = self.lm_head(x_flat)
        # WARP-Output: add logit bias for eval path too
        vocab_bias, _type_logits = self.warp_output(x_flat)
        lp = lp + vocab_bias
        logits = self.logit_softcap * torch.tanh(lp / self.logit_softcap)
        return logits.reshape(*hidden_states.shape[:-1], -1)

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        return self.compute_logits(self.forward_hidden(input_ids))

# --- Sliding window evaluation ---

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
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True) if _use_compile else base_model.forward_logits
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


def eval_val_sliding_ttt(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, log0=print,
) -> tuple[float, float]:
    """Legal score-first TTT (PR #461 recipe): score each chunk with sliding windows,
    then train on it. Every token scored BEFORE any update that could use it."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    # Pre-compute all window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]

    # Assign each window to a chunk based on the first token it scores
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

    # Freeze first N blocks
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

    # Muon TTT: no external optimizer; we apply Newton-Schulz orthogonalized updates
    use_muon_ttt = args.ttt_muon
    if not use_muon_ttt:
        optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    else:
        optimizer = None  # manual update below
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        # --- Phase 1: SCORE this chunk's windows (inference_mode) ---
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

        # --- Phase 2: TRAIN on this chunk (already scored = legal) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                if not use_muon_ttt:
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
                        if not use_muon_ttt:
                            optimizer.zero_grad(set_to_none=True)
                        else:
                            for p in ttt_params:
                                p.grad = None
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        if not use_muon_ttt:
                            optimizer.step()
                        else:
                            with torch.no_grad():
                                for p in ttt_params:
                                    if p.grad is None:
                                        continue
                                    g = p.grad.detach().float()
                                    if g.ndim >= 2:
                                        g = zeropower_via_newtonschulz5(g, steps=args.ttt_ns_steps)
                                    p.data.add_(g.to(p.dtype), alpha=-cos_lr)

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


# --- GPTQ-lite int6 quantization ---

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


def _unbank_state_dict(sd: dict[str, Tensor], num_layers: int) -> dict[str, Tensor]:
    """Convert 3D bank tensors into individual 2D tensors with standard names."""
    out: dict[str, Tensor] = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out

def _rebank_state_dict(sd: dict[str, Tensor], num_layers: int, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert individual 2D tensors back into 3D bank tensors."""
    out: dict[str, Tensor] = {}
    n = num_layers
    # Reconstruct banks from individual weight keys
    qo_slices = [None] * (2 * n)
    kv_slices = [None] * (2 * n)
    up_slices = [None] * n
    down_slices = [None] * n
    consumed = set()
    for i in range(n):
        qk = f"blocks.{i}.attn.c_q.weight"
        if qk in sd:
            qo_slices[i] = sd[qk]
            consumed.add(qk)
        ok = f"blocks.{i}.attn.proj.weight"
        if ok in sd:
            qo_slices[n + i] = sd[ok]
            consumed.add(ok)
        kk = f"blocks.{i}.attn.c_k.weight"
        if kk in sd:
            kv_slices[i] = sd[kk]
            consumed.add(kk)
        vk = f"blocks.{i}.attn.c_v.weight"
        if vk in sd:
            kv_slices[n + i] = sd[vk]
            consumed.add(vk)
        fk = f"blocks.{i}.mlp.fc.weight"
        if fk in sd:
            up_slices[i] = sd[fk]
            consumed.add(fk)
        dk = f"blocks.{i}.mlp.proj.weight"
        if dk in sd:
            down_slices[i] = sd[dk]
            consumed.add(dk)
    out["qo_bank"] = torch.stack(qo_slices).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv_slices).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up_slices).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down_slices).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out

# --- Full Hessian GPTQ int6 quantization (PR #1060) ---

def gptq_quantize_weight(W: Tensor, H: Tensor, clip_range: int = 31,
                          block_size: int = 128, percdamp: float = 0.01) -> tuple[Tensor, Tensor]:
    """GPTQ with Cholesky error compensation and actorder (Frantar et al., ICLR 2023)."""
    W_orig = W.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = percdamp * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        return quantize_int6_per_row(W_orig, clip_range)
    best_q, best_scale, best_err = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(W_orig.abs(), pct, dim=1)
        else:
            row_clip = W_orig.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros(rows, cols, dtype=torch.int8)
        W_work = W_perm.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            W_block = W_work[:, i1:i2].clone()
            Hinv_block = Hinv[i1:i2, i1:i2]
            Err = torch.zeros(rows, i2 - i1)
            for j in range(i2 - i1):
                w_col = W_block[:, j]
                d = Hinv_block[j, j]
                q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
                Q[:, i1 + j] = q_col.to(torch.int8)
                err = (w_col - q_col.float() * sf) / d
                Err[:, j] = err
                W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
            if i2 < cols:
                W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W_perm - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, invperm]
    return best_q, best_scale

def _init_hessians(nl: int, dim: int, mlp_dim: int, device: torch.device) -> dict[str, Tensor]:
    h: dict[str, Tensor] = {}
    for i in range(nl):
        for k in ['c_q', 'c_k', 'c_v']:
            h[f'blocks.{i}.attn.{k}.weight'] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        h[f'blocks.{i}.attn.proj.weight'] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        h[f'blocks.{i}.mlp.fc.weight'] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
        h[f'blocks.{i}.mlp.proj.weight'] = torch.zeros(mlp_dim, mlp_dim, dtype=torch.float32, device=device)
    return h

def _accum_hessians(hessians: dict[str, Tensor], blocks: nn.ModuleList, dim: int, mlp_dim: int) -> None:
    for i, block in enumerate(blocks):
        qkv_in = block.attn._gptq_qkv_in.float().reshape(-1, dim)
        h_qkv = qkv_in.t() @ qkv_in
        hessians[f'blocks.{i}.attn.c_q.weight'] += h_qkv
        hessians[f'blocks.{i}.attn.c_k.weight'] += h_qkv
        hessians[f'blocks.{i}.attn.c_v.weight'] += h_qkv
        o_in = block.attn._gptq_o_in.float().reshape(-1, dim)
        hessians[f'blocks.{i}.attn.proj.weight'] += o_in.t() @ o_in
        up_in = block.mlp._gptq_up_in.float().reshape(-1, dim)
        hessians[f'blocks.{i}.mlp.fc.weight'] += up_in.t() @ up_in
        down_in = block.mlp._gptq_down_in.float().reshape(-1, mlp_dim)
        hessians[f'blocks.{i}.mlp.proj.weight'] += down_in.t() @ down_in

def _finalize_hessians(hessians: dict[str, Tensor], num_batches: int) -> None:
    for name in hessians:
        hessians[name] = hessians[name].cpu() / num_batches
        damp = 0.01 * torch.diag(hessians[name]).mean().clamp_min(1e-6)
        hessians[name] += damp * torch.eye(hessians[name].shape[0])

def gptq_collect_hessians(base_model: nn.Module, train_loader, device: torch.device,
                           num_batches: int, batch_tokens: int, seq_len: int,
                           grad_accum_steps: int) -> dict[str, Tensor]:
    """Collect Hessians H = X^T X from training data."""
    nl = base_model.num_layers
    dim = base_model.tok_emb.weight.shape[1]
    mlp_dim = base_model.mlp_up_bank.shape[1]
    hessians = _init_hessians(nl, dim, mlp_dim, device)
    for block in base_model.blocks:
        block.attn._save_gptq = True
        block.mlp._save_gptq = True
    base_model.eval()
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for _ in range(num_batches):
            x, y = train_loader.next_batch(batch_tokens, seq_len, grad_accum_steps)
            base_model(x, y)
            _accum_hessians(hessians, base_model.blocks, dim, mlp_dim)
    for block in base_model.blocks:
        block.attn._save_gptq = False
        block.mlp._save_gptq = False
    _finalize_hessians(hessians, num_batches)
    base_model.train()
    return hessians

def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str], clip_range: int = 31,
                        hessians: dict[str, Tensor] | None = None):
    num_layers_total = max(
        (int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
        default=0,
    ) + 1
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    gptq_count, naive_count = 0, 0
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
            H = hessians.get(name) if hessians else None
            if H is not None and t.ndim == 2:
                q, s = gptq_quantize_weight(t, H.cpu(), clip_range=clip_range)
                gptq_count += 1
            else:
                q, s = quantize_int6_per_row(t, clip_range=clip_range)
                naive_count += 1
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    if hessians:
        print(f"gptq_quantize: {gptq_count} GPTQ layers, {naive_count} naive layers", flush=True)
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
        if isinstance(info, dict) and info.get("type") == "int6_turbo":
            seed = int(result[name + ".seed"].item())
            out[name] = dequantize_int6_turbo(q, s, seed, orig_dtype)
        elif s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

# --- Training ---

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    # zeropower_via_newtonschulz5 runs eagerly with bmm -- do NOT compile
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
    enable_mem_efficient_sdp(True)
    enable_math_sdp(True)
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
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
        # WARP-Output params
        num_word_types=args.num_word_types,
        has_leading_space_lut=has_leading_space_lut,
    ).to(device).bfloat16()
    # Banks stay FP32 (like CastedLinear weights), cast to BF16 in forward
    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    # No DDP -- Parallel Muon handles bank grad communication via reduce-scatter,
    # and non-bank grads are manually all-reduced before Adam steps.
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if _use_compile else base_model
    model = compiled_model

    # Optimizer split:
    # - 4 parameter banks -> Muon (batched Newton-Schulz)
    # - token embedding -> Adam
    # - scalars/control tensors -> Adam
    # - bigram proj, mtp heads, VE proj -> Adam (small matrix params not worth banking)
    # - WARP-Output type head -> separate Adam with type_head_lr
    matrix_params = [
        base_model.qo_bank, base_model.kv_bank,
        base_model.mlp_up_bank, base_model.mlp_down_bank,
    ]
    block_named_params = list(base_model.blocks.named_parameters())
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
    # WARP-Attn: add Q/K word pos embeddings to token optimizer, layer_scales to scalar optimizer
    if base_model.warp_attn is not None:
        tok_params.append({"params": [base_model.warp_attn.q_word_pos_embed.weight], "lr": token_lr, "base_lr": token_lr})
        tok_params.append({"params": [base_model.warp_attn.k_word_pos_embed.weight], "lr": token_lr, "base_lr": token_lr})
        scalar_params.append(base_model.warp_attn.layer_scales)
    # Word-Length Embedding: embed weight -> tok optimizer, scale -> scalar optimizer
    tok_params.append({"params": [base_model.word_length_embed.embed.weight], "lr": token_lr, "base_lr": token_lr})
    scalar_params.append(base_model.word_length_embed.scale)
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            scalar_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            scalar_params.append(base_model.ve_shared.proj.weight)
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
    # WARP-Output type head optimizer (separate LR)
    warp_output_params = list(base_model.warp_output.parameters())
    optimizer_type = torch.optim.AdamW(
        [{"params": warp_output_params, "lr": args.type_head_lr, "base_lr": args.type_head_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    # Non-bank params that need manual all-reduce (replicated across GPUs)
    replicated_params = list(optimizer_tok.param_groups[0]["params"])
    for pg in optimizer_tok.param_groups[1:]:
        replicated_params.extend(pg["params"])
    replicated_params.extend(scalar_params)
    replicated_params.extend(warp_output_params)

    optimizer_head = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        replicated_params.append(base_model.lm_head.weight)
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_type]
    if optimizer_head is not None:
        optimizers.append(optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params = sum(p.numel() for p in base_model.mtp_heads.parameters())
    warp_output_nparams = sum(p.numel() for p in base_model.warp_output.parameters())
    warp_attn_nparams = sum(p.numel() for p in base_model.warp_attn.parameters()) if base_model.warp_attn is not None else 0
    log0(f"model_params:{n_params}")
    log0(f"warp_output_params:{warp_output_nparams} num_word_types:{args.num_word_types} type_head_lr:{args.type_head_lr}")
    log0(f"WARP-Attn:enabled head_dim:{args.model_dim // args.num_heads} "
         f"warp_attn_params:{warp_attn_nparams} (Q+K embeds + per-layer scales, shared across all layers)")
    log0(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=True math=True")
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
    if args.use_gptq and max_wallclock_ms is not None:
        max_wallclock_ms -= args.gptq_reserve_ms
        log0(f"gptq:reserving {args.gptq_reserve_ms:.0f}ms from training budget, effective={max_wallclock_ms:.0f}ms")
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
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            # All-reduce all grads for warmup (simple, not optimized)
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    from collections import deque
    lawa_queue: deque[dict[str, Tensor]] = deque(maxlen=args.lawa_k)
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = 0.0  # EMA disabled for short runs
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
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
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
        # === 3-phase overlapped optimizer step ===
        # Phase 1: Launch async reduce-scatter for banks (biggest first)
        optimizer_muon.launch_reduce_scatters()
        # Phase 2: All-reduce non-bank grads + step Adam (while bank RS is in-flight)
        if distributed:
            for p in replicated_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optimizer_tok.step()
        optimizer_scalar.step()
        optimizer_type.step()
        if optimizer_head is not None:
            optimizer_head.step()
        # Phase 3: Wait for RS, local NS5, all-gather (banks processed last)
        optimizer_muon.step()
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
    # Apply weight averaging
    log0("ema:applying EMA weights")
    if True:
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
    # Unbank 3D tensors into individual 2D tensors for quantization
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    # GPTQ calibration: collect Hessians from training data
    gptq_hessians = None
    if args.use_gptq:
        t_gptq = time.perf_counter()
        log0(f"gptq:calibrating with {args.gptq_calib_samples} batches (training data)...")
        calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        gptq_hessians = gptq_collect_hessians(
            base_model, calib_loader, device, num_batches=args.gptq_calib_samples,
            batch_tokens=args.train_batch_tokens, seq_len=args.train_seq_len,
            grad_accum_steps=grad_accum_steps)
        del calib_loader
        gptq_elapsed = time.perf_counter() - t_gptq
        log0(f"gptq:calibrated {len(gptq_hessians)} layers in {gptq_elapsed:.1f}s")
        torch.cuda.empty_cache()
    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, clip_range=args.quant_clip_range, hessians=gptq_hessians)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+lzma: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+lzma: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(lzma.decompress(quant_blob_disk)),
        map_location="cpu",
    )
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    # Re-bank the dequantized tensors
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        gated_attention=args.gated_attention, value_residual=args.value_residual,
        num_word_types=args.num_word_types,
        has_leading_space_lut=has_leading_space_lut,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True) if _use_compile else eval_model
    skip_pre_ttt = bool(int(os.environ.get("SKIP_PRE_TTT_EVALS", "0")))
    torch.cuda.synchronize()
    if not skip_pre_ttt:
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
    if not skip_pre_ttt:
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
    # Legal score-first TTT (PR #461 recipe)
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, log0=log0,
        )
        torch.cuda.synchronize()
        log0(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
             f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")
        log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
    # --- SLOT: per-batch delta optimization at last hidden layer ---
    if args.slot_enabled:
        log0(f"slot:starting lr={args.slot_lr} steps={args.slot_steps}")
        slot_stride = args.eval_stride if args.eval_stride > 0 else 64
        seq_s = effective_eval_seq_len
        total_tok = val_tokens.numel() - 1
        ws_list = [ws for ws in range(0, total_tok, slot_stride) if min(ws + seq_s, total_tok) - ws >= 1]
        my_s = (len(ws_list) * rank) // world_size
        my_e = (len(ws_list) * (rank + 1)) // world_size
        my_ws = ws_list[my_s:my_e]
        sl_loss = torch.zeros((), device=device, dtype=torch.float64)
        sl_tc = torch.zeros((), device=device, dtype=torch.float64)
        sl_bc = torch.zeros((), device=device, dtype=torch.float64)
        t_slot = time.perf_counter()
        for bi in range(0, len(my_ws), 32):
            bws = my_ws[bi:bi+32]; bsz = len(bws)
            xb = torch.zeros(bsz, seq_s, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, seq_s, dtype=torch.int64, device=device)
            wls = []
            for i, ws in enumerate(bws):
                end = min(ws + seq_s, total_tok); wl = end - ws; wls.append(wl)
                ct = val_tokens[ws:end+1].to(dtype=torch.int64, device=device)
                xb[i,:wl] = ct[:-1]; yb[i,:wl] = ct[1:]
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                H = eval_model.forward_hidden(xb)
            H = H.detach().float()
            delta = torch.zeros(1, 1, H.shape[-1], device=device, dtype=H.dtype, requires_grad=True)
            sopt = torch.optim.AdamW([delta], lr=args.slot_lr, weight_decay=1e-8, eps=1e-5)
            for _ in range(args.slot_steps):
                sopt.zero_grad()
                lg = eval_model.compute_logits((H + delta).to(torch.bfloat16)).float()
                loss_s = F.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1), reduction="mean")
                loss_s.backward()
                sopt.step()
            with torch.no_grad():
                lg = eval_model.compute_logits((H + delta.detach()).to(torch.bfloat16)).float()
            nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1), reduction="none").reshape(bsz, seq_s)
            for i, ws in enumerate(bws):
                wl = wls[i]; s = 0 if ws == 0 else max(wl - slot_stride, 0)
                sl_loss += nll[i, s:wl].to(torch.float64).sum(); sl_tc += float(wl - s)
                tgt, prev = yb[i, s:wl], xb[i, s:wl]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                sl_bc += tb.sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sl_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(sl_tc, op=dist.ReduceOp.SUM)
            dist.all_reduce(sl_bc, op=dist.ReduceOp.SUM)
        sv_loss = (sl_loss / sl_tc).item()
        sv_bpb = sv_loss / math.log(2.0) * (sl_tc.item() / sl_bc.item())
        log0(f"final_slot val_loss:{sv_loss:.4f} val_bpb:{sv_bpb:.4f} time:{1000*(time.perf_counter()-t_slot):.0f}ms")
        log0(f"final_slot_exact val_loss:{sv_loss:.8f} val_bpb:{sv_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
