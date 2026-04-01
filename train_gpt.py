"""

Key improvements over the baseline:
  - Better hyperparameters (larger batch, seq len, more layers, tuned LRs)
  - SmearGate: temporal context smoothing at the embedding level
  - Improved BigramHashEmbedding: XOR hash + learnable scale + compact dim
  - ValueEmbedding: reinjects token identity into attention values at deep layers
  - Partial RoPE + length-extrapolation in Rotary
  - XSA (Cross-Self Attention removal) on last N layers
  - LN scale: damps deeper layers without extra params
  - DTG gate: optional dynamic token gating (disabled by default)
  - MTP heads: optional multi-token prediction auxiliary loss (disabled by default)
  - Orthogonal weight initialization for large matrices
  - AdamW (adds weight decay to Adam for embeddings/scalars)
  - SWA (Stochastic Weight Averaging) during warmdown
  - Inline EMA (decay=0.997) applied as the final model state
  - Late QAT: triggered by LR scale threshold, not step count
  - Mixed int6/int8 quantization (attn/MLP → int6, rest → int8)
  - zstandard compression with fallback to zlib
  - eval_val_sliding: sliding window evaluation with stride for better BPB
  - Distributed wallclock cap synchronisation across ranks
  - Memory usage reported at end of training

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

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _USE_FLASH3 = True
except ImportError:
    _USE_FLASH3 = False

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────
class Hyperparameters:
    data_path        = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files      = os.path.join(data_path, "fineweb_train_*.bin")
    val_files        = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path   = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id           = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed             = int(os.environ.get("SEED", 1337))

    # Validation
    val_batch_size   = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every   = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every  = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    eval_stride      = int(os.environ.get("EVAL_STRIDE", 64))

    # Training length
    iterations            = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters        = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps          = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens    = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len         = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len          = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape
    vocab_size       = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers       = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads     = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim        = int(os.environ.get("MODEL_DIM", 512))
    num_heads        = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult         = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings   = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base        = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims        = int(os.environ.get("ROPE_DIMS", 16))     # partial RoPE dims (0=full)
    logit_softcap    = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init     = float(os.environ.get("QK_GAIN_INIT", 1.5))
    ln_scale         = bool(int(os.environ.get("LN_SCALE", "1")))   # depth-damped LN scale
    xsa_last_n       = int(os.environ.get("XSA_LAST_N", 4))         # XSA on last N layers
    dtg_enabled      = bool(int(os.environ.get("DTG_ENABLED", "0"))) # dynamic token gate

    # MTP (multi-token prediction auxiliary loss)
    mtp_num_heads    = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight  = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))

    # BigramHash embedding
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim        = int(os.environ.get("BIGRAM_DIM", 128))

    # Value embedding
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim     = int(os.environ.get("VE_DIM", 128))
    ve_layers  = os.environ.get("VE_LAYERS", "9,10")

    # Optimizer
    embed_lr                  = float(os.environ.get("EMBED_LR", 0.6))
    head_lr                   = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr             = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std       = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr                 = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr                 = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum             = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps        = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start= float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps= int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_wd                   = float(os.environ.get("MUON_WD", 0.04))
    beta1                     = float(os.environ.get("BETA1", 0.9))
    beta2                     = float(os.environ.get("BETA2", 0.95))
    adam_eps                  = float(os.environ.get("ADAM_EPS", 1e-8))
    adam_wd                   = float(os.environ.get("ADAM_WD", 0.04))
    grad_clip_norm            = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # SWA (Stochastic Weight Averaging during warmdown)
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every   = int(os.environ.get("SWA_EVERY", 50))

    # QAT — triggered by LR-scale threshold (late_qat_threshold > 0) OR enabled from step 0
    qat_enabled         = bool(int(os.environ.get("QAT_ENABLED", "0")))
    late_qat_threshold  = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))


# ─────────────────────────────────────────────
# MUON OPTIMIZER
# ─────────────────────────────────────────────
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if G.size(0) > G.size(1) else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                     nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank       = dist.get_rank()       if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            total_params   = sum(int(p.numel()) for p in params)
            updates_flat   = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    if wd != 0.0:
                        g = g + wd * p.data.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ─────────────────────────────────────────────
# TOKENIZER-AGNOSTIC EVALUATION
# ─────────────────────────────────────────────
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,"
        "skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if p
)
INT8_KEEP_FLOAT_MAX_NUMEL   = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE    = torch.float16
INT8_CLIP_Q                 = 99.99984 / 100.0


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np         = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np  = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np  = np.ones((table_size,),  dtype=np.bool_)
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
        torch.tensor(base_bytes_np,        dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool,  device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool,  device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs   = max(1, local_batch_tokens // seq_len)
    total_seqs  = (val_tokens.numel() - 1) // seq_len
    seq_start   = (total_seqs * rank) // world_size
    seq_end     = (total_seqs * (rank + 1)) // world_size

    val_loss_sum  = torch.zeros((), device=device, dtype=torch.float64)
    val_tok_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count= torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, seq_end)
            raw_start, raw_end = batch_start * seq_len, batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1: ].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            btoken = float(y.numel())
            val_loss_sum  += batch_loss.to(torch.float64) * btoken
            val_tok_count += btoken
            tgt_ids = y.reshape(-1);  prev_ids = x.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,   op=dist.ReduceOp.SUM)
        dist.all_reduce(val_tok_count,  op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_tok_count
    bpt  = val_loss.item() / math.log(2.0)
    tpb  = val_tok_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)


def eval_val_sliding(
        args, base_model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride: int, batch_seqs: int = 32, eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding-window evaluation: each token scored with maximum context."""
    seq_len = eval_seq_len or args.train_seq_len
    total   = val_tokens.numel() - 1
    windows = [ws for ws in range(0, total, stride) if min(ws + seq_len, total) - ws >= 1]
    my_s = (len(windows) * rank) // world_size
    my_e = (len(windows) * (rank + 1)) // world_size
    my_windows = windows[my_s:my_e]

    loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    tok_count  = torch.zeros((), device=device, dtype=torch.float64)
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
                end  = min(ws + seq_len, total)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                loss_sum  += nll[i, s:wlen].to(torch.float64).sum()
                tok_count += float(wlen - s)
                tgt  = y_batch[i, s:wlen];  prev = x_batch[i, s:wlen]
                tb   = base_bytes_lut[tgt].to(torch.float64)
                tb  += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum,   op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count,  op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / tok_count).item()
    base_model.train()
    return val_loss, val_loss / math.log(2.0) * (tok_count.item() / byte_count.item())


# ─────────────────────────────────────────────
# QUANTIZATION
# ─────────────────────────────────────────────
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:  return "embed"
    if ".mlp." in name:                           return "mlp"
    if ".attn." in name:                          return "attn"
    return "other"


def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.999, 0.9995, 0.9999, 0.99999, 1.0]:
            rc = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
            s  = (rc / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q  = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax  = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q     = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def quantize_float_tensor_int8(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2 and t32.numel():
        clip = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
        clipped = torch.clamp(t32, -clip[:, None], clip[:, None])
        scale   = (clip / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip  = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip / 127.0 if clip > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip, clip) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict) -> Tensor:
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def mixed_quantize(state_dict: dict[str, Tensor], int6_cats: set[str]):
    """Mixed precision: int6 for attn/MLP weights, int8 for the rest."""
    result: dict[str, Tensor] = {}
    meta:   dict[str, object] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes) if t.is_floating_point() else t
            result[name] = kept
            meta[name]   = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name]   = "passthrough_ctrl"
            continue
        if cat in int6_cats:
            q, s = quantize_int6_per_row(t)
            meta[name] = {"type": "int6", "orig_dtype": str(t.dtype).removeprefix("torch.")}
        else:
            q, s = quantize_float_tensor_int8(t)
            meta[name] = {"type": "int8", "orig_dtype": str(t.dtype).removeprefix("torch.")}
        result[name + ".q"] = q
        result[name + ".s"] = s
    if passthrough_orig_dtypes:
        meta["__passthrough_orig_dtypes__"] = passthrough_orig_dtypes
    return result, meta


def dequantize_mixed(result: dict[str, Tensor], meta: dict, template_sd: dict[str, Tensor]):
    out: dict[str, Tensor] = {}
    pod = meta.get("__passthrough_orig_dtypes__", {})
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl"):
            t = result[name]
            od = pod.get(name)
            if od:
                t = t.to(dtype=getattr(torch, od))
            elif t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".s"]
        odtype = getattr(torch, info["orig_dtype"])
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(odtype)
        else:
            out[name] = (q.float() * float(s.item())).to(odtype)
    return out


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens    = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found: {pattern}")
        self.file_idx = 0
        self.tokens   = load_data_shard(self.files[0])
        self.pos      = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens   = load_data_shard(self.files[self.file_idx])
        self.pos      = 0

    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos      += k
            remaining     -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device):
        self.rank = rank; self.world_size = world_size; self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens   = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span  = local_tokens + 1
        chunk          = self.stream.take(per_rank_span * self.world_size)
        start          = self.rank * per_rank_span
        local          = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1: ].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ─────────────────────────────────────────────
# TRANSFORMER MODULES
# ─────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__(); self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32    = self.weight.float()
                rmax   = w32.abs().amax(dim=1)
                scale  = (rmax / 31.0).clamp_min(1.0 / 31.0)
                w_q    = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) \
                    and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    """Supports partial RoPE (rope_dims < head_dim) and length extrapolation."""
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 2048, rope_dims: int = 0):
        super().__init__()
        self.dim           = dim
        self.base          = base
        self.train_seq_len = train_seq_len
        self.rope_dims     = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device, dtype) -> tuple[Tensor, Tensor]:
        if (self._cos_cached is None or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale    = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached     = freqs.cos()[None, :, None, :]   # [1, T, 1, rd//2]
            self._sin_cached     = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        return torch.cat(
            (torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1), x_pass), dim=-1
        )
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len=2048):
        super().__init__()
        if dim % num_heads != 0 or num_heads % num_kv_heads != 0:
            raise ValueError("dim must be divisible by num_heads, num_heads by num_kv_heads")
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        kv_dim            = num_kv_heads * self.head_dim
        self.c_q   = CastedLinear(dim, dim,    bias=False)
        self.c_k   = CastedLinear(dim, kv_dim, bias=False)
        self.c_v   = CastedLinear(dim, kv_dim, bias=False)
        self.proj  = CastedLinear(dim, dim,    bias=False)
        self.proj._zero_init = True
        self.q_gain  = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0   # set by GPT for partial RoPE
        self.rotary    = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa   = False   # set by GPT for XSA layers

    def _xsa(self, y: Tensor, v: Tensor) -> Tensor:
        """Subtract self-value projection (GQA-aware, no repeat_interleave)."""
        B, T, H, D = y.shape
        Hkv  = v.size(-2)
        g    = H // Hkv
        y_g  = y.reshape(B, T, Hkv, g, D)
        vn   = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads,    self.head_dim)
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
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]
        if _USE_FLASH3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               enable_gqa=(self.num_kv_heads != self.num_heads))
        if self.use_xsa:
            y = self._xsa(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden    = int(mlp_mult * dim)
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class SmearGate(nn.Module):
    """Temporal context smoothing: blends current token emb with the previous one."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g     = torch.sigmoid(self.gate.to(x.dtype))[None, None, :]
        x_prev= torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Compact bigram features via XOR hashing + learnable scale."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj  = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def _hash(self, tokens: Tensor) -> Tensor:
        t   = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0]  = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self._hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(h.dtype)


class ValueEmbedding(nn.Module):
    """Reinjects token identity into attention values at specified deep layers."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj  = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(h.dtype)


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 layer_idx: int = 0, ln_scale: bool = False, dtg: bool = False,
                 train_seq_len: int = 2048):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.mlp  = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        # Depth-damped LN scale: 1/sqrt(layer_idx+1) — zero new params
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        # Optional dynamic token gate
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None

    def forward(self, x: Tensor, x0: Tensor, v_embed: Tensor | None = None) -> Tensor:
        mix  = self.resid_mix.to(x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(x_out.dtype)[None, None, :] \
                * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        if self.dtg_gate is not None:
            gate  = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, rope_dims=0, ln_scale=False, xsa_last_n=0,
                 dtg=False, mtp_num_heads=0, mtp_loss_weight=0.1,
                 bigram_vocab_size=2048, bigram_dim=128,
                 ve_enabled=False, ve_dim=128, ve_layers="9,10",
                 train_seq_len=2048):
        super().__init__()
        if logit_softcap <= 0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings     = tie_embeddings
        self.tied_embed_init_std= tied_embed_init_std
        self.logit_softcap      = logit_softcap
        self.mtp_num_heads      = mtp_num_heads
        self.mtp_loss_weight    = mtp_loss_weight
        self._kv_dim            = num_kv_heads * (model_dim // num_heads)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram  = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) \
                       if bigram_vocab_size > 0 else None
        self.smear   = SmearGate(model_dim)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights       = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, ln_scale=ln_scale, dtg=dtg, train_seq_len=train_seq_len)
            for i in range(num_layers)
        ])

        # Partial RoPE
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base,
                                           train_seq_len=train_seq_len, rope_dims=rope_dims)

        # XSA on last N layers
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True

        # Value embeddings
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, self._kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices]
            )
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # MTP auxiliary heads
        self.mtp_heads = nn.ModuleList([
            CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)
        ])
        for h in self.mtp_heads:
            h._zero_init = True

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = len(self.blocks)
        proj_scale = 1.0 / math.sqrt(2 * n)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and min(module.weight.shape) >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(proj_scale)

    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict) -> Tensor | None:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if "ve" not in ve_cache:
            ve_cache["ve"] = self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_cache["ve"] * self.ve_layer_scales[ve_idx].to(ve_cache["ve"].dtype)

    def _run_blocks(self, x: Tensor, input_ids: Tensor) -> Tensor:
        x0 = x
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, v_embed=self._get_ve(i, input_ids, ve_cache))
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[bi](x, x0, v_embed=self._get_ve(bi, input_ids, ve_cache))
        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x = self._run_blocks(x, input_ids)
        x = self.final_norm(x)

        x_flat  = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x_flat, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        loss   = F.cross_entropy(logits.float(), targets)

        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0:
            _, seqlen, dim = x.shape
            mtp_loss, cnt = x.new_zeros(()), 0
            for k, head in enumerate(self.mtp_heads):
                vt = seqlen - (k + 1)
                if vt <= 0:
                    continue
                ml  = head(x[:, :vt, :].reshape(-1, dim))
                ml  = self.logit_softcap * torch.tanh(ml / self.logit_softcap)
                mtp_loss += F.cross_entropy(ml.float(), target_ids[:, k + 1:].reshape(-1))
                cnt += 1
            if cnt > 0:
                loss = loss + self.mtp_loss_weight * (mtp_loss / cnt)
        return loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x = self._run_blocks(x, input_ids)
        x = self.final_norm(x)
        lp = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank        = int(os.environ.get("RANK",       "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must be positive and divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale       = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}")

    dataset_dir        = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len        = max(args.train_seq_len, effective_eval_seq)
    val_tokens         = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    log0(f"val_bpb:enabled tokenizer=sentencepiece path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:tokens:{val_tokens.numel() - 1}")

    CastedLinear._qat_enabled = args.qat_enabled

    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        rope_dims=args.rope_dims, ln_scale=args.ln_scale, xsa_last_n=args.xsa_last_n,
        dtg=args.dtg_enabled, mtp_num_heads=args.mtp_num_heads, mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        train_seq_len=args.train_seq_len,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = (DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
                        if distributed else compiled_model)

    # ── Optimizer setup ──
    block_named = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params  = [p for n, p in block_named
                     if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.mtp_num_heads > 0:
        matrix_params.extend(p for p in base_model.mtp_heads.parameters() if p.ndim == 2)

    token_lr  = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params= [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
        scalar_params.append(base_model.bigram.scale)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            matrix_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        scalar_params.extend(base_model.ve_layer_scales)

    # AdamW for embeddings and scalars (proper weight decay)
    optimizer_tok = torch.optim.AdamW(
        tok_params, betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_wd, fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for g in optimizer_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} | world_size:{world_size} | grad_accum:{grad_accum_steps}")
    log0(f"num_layers:{args.num_layers} model_dim:{args.model_dim} mlp_mult:{args.mlp_mult} "
         f"xsa_last_n:{args.xsa_last_n} rope_dims:{args.rope_dims} ln_scale:{args.ln_scale}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA active layers: {xsa_layers}")

    # Inline EMA state dict (decay=0.997)
    ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
    ema_decay  = 0.997

    # SWA state
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            wds = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                   if wds <= step < args.iterations else 1.0
        step_ms     = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining   = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining / max(warmdown_ms, 1e-9) if remaining <= warmdown_ms else 1.0

    # ── Warmup (compile graph, then reset) ──
    if args.warmup_steps > 0:
        init_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_states  = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for micro in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wloss = model(x, y)
                (wloss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if ws == 0 or (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup_step:{ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_model_state, strict=True)
        for opt, state in zip(optimizers, init_opt_states):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        ema_state    = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}

    # ── Main training loop ──
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0   = time.perf_counter()
    step = 0
    model.train()

    while True:
        last_step     = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale      = lr_mul(step, elapsed_ms)

        # Late QAT: enable when LR scale drops below threshold
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")

        # Muon momentum warmup
        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in optimizer_muon.param_groups:
            g["momentum"] = muon_mom

        # LR schedule
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # Inline EMA update
        with torch.no_grad():
            for n, t in base_model.state_dict().items():
                ema_state[n].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)

        # SWA: collect checkpoints during warmdown
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for n, t in base_model.state_dict().items():
                    swa_state[n] += t.detach().cpu()
                swa_count += 1

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        should_log = args.train_log_every > 0 and (
            step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None
        )
        if should_log:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")

        # Distributed wallclock cap
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak_memory_allocated:{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
         f"reserved:{torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    # ── Apply weight averaging ──
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"swa:applying SWA averaging count={swa_count}")
        current_state = base_model.state_dict()
        avg = {n: (swa_state[n] / swa_count).to(dtype=current_state[n].dtype) for n in swa_state}
        base_model.load_state_dict(avg, strict=True)
    else:
        log0("ema:applying EMA weights (decay=0.997)")
        current_state = base_model.state_dict()
        avg = {n: t.to(dtype=current_state[n].dtype) for n, t in ema_state.items()}
        base_model.load_state_dict(avg, strict=True)

    # Diagnostic eval after weight averaging
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_loss, diag_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"DIAGNOSTIC post_avg val_loss:{diag_loss:.4f} val_bpb:{diag_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms")

    # ── Export & quantize ──
    full_sd  = base_model.state_dict()
    export_sd = {k: v for k, v in full_sd.items() if "mtp_heads" not in k}
    excl     = sum(int(t.numel()) for k, t in full_sd.items() if "mtp_heads" in k)
    if excl > 0:
        log0(f"export_excluding_mtp_params:{excl}")

    sd_cpu        = {k: v.detach().cpu() for k, v in export_sd.items()}
    quant_result, quant_meta = mixed_quantize(sd_cpu, {"mlp", "attn"})
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    raw = quant_buf.getvalue()
    if _COMPRESSOR == "zstd":
        compressed = zstandard.ZstdCompressor(level=22).compress(raw)
    else:
        compressed = zlib.compress(raw, level=9)

    model_bytes = len(compressed)
    code_bytes  = len(code.encode("utf-8"))
    log0(f"model_bytes:{model_bytes} code_bytes:{code_bytes} total:{model_bytes + code_bytes} compressor:{_COMPRESSOR}")

    artifact_path = f"logs/{args.run_id}_model.ptz"
    if master_process:
        with open(artifact_path, "wb") as f:
            f.write(compressed)
        log0(f"artifact_saved:{artifact_path}")

    # Round-trip validation
    if distributed:
        dist.barrier()
    with open(artifact_path, "rb") as f:
        blob = f.read()
    raw_rt = zstandard.ZstdDecompressor().decompress(blob) if _COMPRESSOR == "zstd" else zlib.decompress(blob)
    quant_state = torch.load(io.BytesIO(raw_rt), map_location="cpu")
    deq_sd = dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu)

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        rope_dims=args.rope_dims, ln_scale=args.ln_scale, xsa_last_n=args.xsa_last_n,
        dtg=args.dtg_enabled, mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        train_seq_len=args.train_seq_len,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_sd, strict=True)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)

    torch.cuda.synchronize()
    t_rt = time.perf_counter()
    rt_loss, rt_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq,
    )
    torch.cuda.synchronize()
    log0(f"final_roundtrip val_loss:{rt_loss:.4f} val_bpb:{rt_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_rt):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{rt_loss:.8f} val_bpb:{rt_bpb:.8f}")

    # Sliding window evaluation
    if args.eval_stride > 0 and args.eval_stride < effective_eval_seq:
        torch.cuda.synchronize()
        t_sw = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, eval_seq_len=effective_eval_seq,
        )
        torch.cuda.synchronize()
        log0(f"final_sliding_window val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} "
             f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_sw):.0f}ms")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_loss:.8f} val_bpb:{sw_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()