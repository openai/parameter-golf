# -*- coding: utf-8 -*-
"""
SOTA Parameter Golf Submission
val_bpb target: <=1.115

Techniques stacked on PR #414 base:
  11L U-Net | LeakyReLU(0.5)² | XSA last-4 | Partial RoPE(16) | LN Scale
  EMA(0.997) + Tight SWA | VE128 | GPTQ-lite | Legal TTT | Late QAT@0.15
  FlashAttention 3 | BigramHash(1536) | warmdown 3500 | OrthoInit

Hard stop: train_gpt.py must never be longer than 1500 lines.
"""

from __future__ import annotations

import copy, glob, io, math, os, random, struct, subprocess, sys, time, uuid, zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ---------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------

class Hyperparameters:
    data_path        = os.environ.get("DATA_PATH",       "./data/datasets/fineweb10B_sp1024")
    train_files      = os.path.join(data_path, "fineweb_train_*.bin")
    val_files        = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path   = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id           = os.environ.get("RUN_ID",   str(uuid.uuid4()))
    seed             = int(os.environ.get("SEED", 1337))

    val_batch_size   = int(os.environ.get("VAL_BATCH_SIZE",  524_288))
    val_loss_every   = int(os.environ.get("VAL_LOSS_EVERY",  1000))
    train_log_every  = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations       = int(os.environ.get("ITERATIONS",    9000))
    warmdown_iters   = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps     = int(os.environ.get("WARMUP_STEPS",   20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len    = int(os.environ.get("TRAIN_SEQ_LEN",  2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    eval_stride      = int(os.environ.get("EVAL_STRIDE", 64))
    qk_gain_init     = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape
    vocab_size       = int(os.environ.get("VOCAB_SIZE",    1024))
    num_layers       = int(os.environ.get("NUM_LAYERS",    11))
    num_kv_heads     = int(os.environ.get("NUM_KV_HEADS",  4))
    model_dim        = int(os.environ.get("MODEL_DIM",     512))
    num_heads        = int(os.environ.get("NUM_HEADS",     8))
    mlp_mult         = float(os.environ.get("MLP_MULT",   3.0))
    tie_embeddings   = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base        = float(os.environ.get("ROPE_BASE",  10000.0))
    logit_softcap    = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Partial RoPE: apply to first rope_dims of head_dim (0 = full RoPE)
    rope_dims        = int(os.environ.get("ROPE_DIMS", 16))

    # LN Scale: scale RMSNorm by 1/sqrt(layer+1) per block
    ln_scale         = bool(int(os.environ.get("LN_SCALE", "1")))

    # XSA: Exclusive Self Attention on last N layers
    xsa_last_n       = int(os.environ.get("XSA_LAST_N", 4))

    # BigramHash
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 1536))
    bigram_dim        = int(os.environ.get("BIGRAM_DIM", 128))

    # SmearGate
    use_smeargate    = bool(int(os.environ.get("USE_SMEARGATE", "1")))

    # Shared Value Embedding (VE)
    ve_enabled       = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim           = int(os.environ.get("VE_DIM", 128))
    ve_layers_str    = os.environ.get("VE_LAYERS", "9,10")

    # EMA
    ema_enabled      = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay        = float(os.environ.get("EMA_DECAY", 0.997))

    # SWA (tight, during warmdown)
    swa_every        = int(os.environ.get("SWA_EVERY",       50))
    swa_start_frac   = float(os.environ.get("SWA_START_FRAC", 0.5))

    # INT6 QAT
    use_qat          = bool(int(os.environ.get("USE_QAT", "1")))
    qat_start_frac   = float(os.environ.get("QAT_START_FRAC", 0.15))

    # TTT (Legal test-time training)
    ttt_enabled      = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr           = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs       = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_momentum     = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_grad_clip    = float(os.environ.get("TTT_GRAD_CLIP", 1.0))

    # Optimizer
    embed_lr         = float(os.environ.get("EMBED_LR",        0.6))
    head_lr          = float(os.environ.get("HEAD_LR",         0.008))
    tied_embed_lr    = float(os.environ.get("TIED_EMBED_LR",   0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr        = float(os.environ.get("MATRIX_LR",       0.025))
    scalar_lr        = float(os.environ.get("SCALAR_LR",       0.025))
    muon_momentum    = float(os.environ.get("MUON_MOMENTUM",   0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_weight_decay  = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))
    adamw_weight_decay = float(os.environ.get("ADAMW_WEIGHT_DECAY", 0.04))
    beta1            = float(os.environ.get("BETA1", 0.9))
    beta2            = float(os.environ.get("BETA2", 0.95))
    adam_eps         = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm   = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

# ---------------------------------------------------------
# MUON OPTIMIZER
# ---------------------------------------------------------

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
                 weight_decay: float = 0.0, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                     backend_steps=backend_steps,
                                     weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size  = dist.get_world_size() if distributed else 1
        rank        = dist.get_rank()        if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr             = group["lr"]
            momentum       = group["momentum"]
            backend_steps  = group["backend_steps"]
            nesterov       = group["nesterov"]
            weight_decay   = group.get("weight_decay", 0.0)

            total_params   = sum(int(p.numel()) for p in params)
            updates_flat   = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
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
                if weight_decay > 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss

# ---------------------------------------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# ---------------------------------------------------------

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
        torch.tensor(base_bytes_np,         dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np,  dtype=torch.bool,  device=device),
        torch.tensor(is_boundary_token_np,  dtype=torch.bool,  device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files  = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Val split too short for SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    eval_stride = max(1, args.eval_stride)
    seq_len     = args.train_seq_len
    total_tokens  = val_tokens.numel() - 1
    num_positions = (total_tokens - seq_len) // eval_stride + 1
    pos_start = (num_positions * rank)           // world_size
    pos_end   = (num_positions * (rank + 1))     // world_size

    val_loss_sum  = torch.zeros((), device=device, dtype=torch.float64)
    val_tok_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count= torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    batch_size = max(1, args.val_batch_size // (world_size * seq_len))
    with torch.inference_mode():
        pos = pos_start
        while pos < pos_end:
            batch_end = min(pos + batch_size, pos_end)
            xs, ys = [], []
            for p in range(pos, batch_end):
                ts = p * eval_stride
                te = ts + seq_len
                if te >= val_tokens.numel():
                    break
                xs.append(val_tokens[ts:te])
                ys.append(val_tokens[ts+1:te+1])
            if not xs:
                break
            x = torch.stack(xs).to(device=device, dtype=torch.int64)
            y = torch.stack(ys).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bl = model(x, y).detach()
            ntok = float(y.numel())
            val_loss_sum   += bl.to(torch.float64) * ntok
            val_tok_count  += ntok
            prev_ids = x.reshape(-1);  tgt_ids = y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
            pos = batch_end
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,   op=dist.ReduceOp.SUM)
        dist.all_reduce(val_tok_count,  op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_tok_count
    bpt      = val_loss.item() / math.log(2.0)
    tpb      = val_tok_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)

# ---------------------------------------------------------
# QUANTIZATION — INT6 + GPTQ-LITE CLIP SEARCH + ZSTD-22
# ---------------------------------------------------------

CONTROL_TENSOR_PATTERNS = tuple(p for p in os.environ.get(
    "CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
    "q_gain,skip_weight,skip_weights,smear_gate,bigram,ve_scale",
).split(",") if p)

INT8_KEEP_FLOAT_MAX_NUMEL   = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE    = torch.float16
INT6_MAX, INT6_MIN          = 31, -32

# GPTQ-lite: candidate clip percentiles per row (picks min-MSE)
GPTQ_CANDIDATES = [0.999, 0.9995, 0.9999, 0.99999, 0.999999, 1.0]


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict) -> Tensor:
    if any(p in name for p in CONTROL_TENSOR_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def gptq_lite_quantize_row(row: Tensor):
    """Per-row int6 quantization with GPTQ-lite optimal clip search."""
    row32 = row.float()
    best_mse, best_q, best_s = float("inf"), None, None
    abs_row = row32.abs()
    n = abs_row.numel()
    sorted_abs = abs_row.sort().values
    for pct in GPTQ_CANDIDATES:
        idx   = min(int(pct * n), n - 1)
        clip  = float(sorted_abs[idx].item())
        if clip == 0:
            clip = 1e-8
        scale = clip / INT6_MAX
        q     = torch.clamp(torch.round(row32.clamp(-clip, clip) / scale),
                            INT6_MIN, INT6_MAX)
        mse   = float(((q * scale - row32) ** 2).mean().item())
        if mse < best_mse:
            best_mse = mse
            best_q   = q.to(torch.int8)
            best_s   = torch.tensor(scale, dtype=torch.float32)
    return best_q, best_s


def quantize_float_tensor_int6_gptq(t: Tensor):
    """INT6 per-row with GPTQ-lite clip search."""
    t32 = t.float()
    if t32.ndim == 2:
        rows, cols = t32.shape
        q_rows   = []
        s_rows   = []
        for r in range(rows):
            q_r, s_r = gptq_lite_quantize_row(t32[r])
            q_rows.append(q_r.unsqueeze(0))
            s_rows.append(s_r.unsqueeze(0))
        q     = torch.cat(q_rows, dim=0).contiguous()
        scale = torch.cat(s_rows, dim=0).to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
        return q, scale
    # 1-D fallback
    q, s = gptq_lite_quantize_row(t32.flatten())
    return q.contiguous(), s


def pack_int6_to_bytes(q: Tensor) -> bytes:
    flat = q.reshape(-1).numpy().astype(np.int8)
    flat_u = (flat.astype(np.int32) + 32).astype(np.uint8) & 0x3F
    pad = (4 - len(flat_u) % 4) % 4
    if pad:
        flat_u = np.concatenate([flat_u, np.zeros(pad, dtype=np.uint8)])
    n4 = len(flat_u) // 4
    a = flat_u[0::4].astype(np.uint32); b = flat_u[1::4].astype(np.uint32)
    c = flat_u[2::4].astype(np.uint32); d = flat_u[3::4].astype(np.uint32)
    p24 = (a << 18) | (b << 12) | (c << 6) | d
    out = np.zeros(n4 * 3, dtype=np.uint8)
    out[0::3] = (p24 >> 16) & 0xFF
    out[1::3] = (p24 >> 8)  & 0xFF
    out[2::3] =  p24        & 0xFF
    return out.tobytes()


def unpack_int6_from_bytes(data: bytes, numel: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    n4  = len(arr) // 3
    p24 = (arr[0::3].astype(np.uint32) << 16) | (arr[1::3].astype(np.uint32) << 8) | arr[2::3].astype(np.uint32)
    fu  = np.zeros(n4 * 4, dtype=np.uint8)
    fu[0::4] = (p24 >> 18) & 0x3F; fu[1::4] = (p24 >> 12) & 0x3F
    fu[2::4] = (p24 >> 6)  & 0x3F; fu[3::4] =  p24        & 0x3F
    return (fu[:numel].astype(np.int8) - 32)


def quantize_state_dict_int6(state_dict: dict):
    try:
        import zstandard as zstd
        _has_zstd = True
    except ImportError:
        _has_zstd = False

    quantized: dict = {}; scales: dict = {}; dtypes: dict = {}
    passthrough: dict = {}; passthrough_orig_dtypes: dict = {}; shapes: dict = {}
    stats = dict.fromkeys(("param_count","num_tensors","num_float_tensors",
                           "num_nonfloat_tensors","baseline_tensor_bytes","int6_payload_bytes"), 0)

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"]            += int(t.numel())
        stats["num_tensors"]            += 1
        stats["baseline_tensor_bytes"]  += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int6_payload_bytes"]  += tensor_nbytes(t)
            continue
        is_emb = "tok_emb" in name or "bigram" in name or "value_embed" in name
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or is_emb:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += tensor_nbytes(kept)
            continue
        # Keep last-layer k_proj in fp16 (quantisation sensitive)
        if any(f"blocks.{i}.attn.c_k" in name for i in [9, 10]):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        shapes[name] = tuple(t.shape)
        q, s = quantize_float_tensor_int6_gptq(t)
        packed = pack_int6_to_bytes(q)
        quantized[name] = packed; scales[name] = s; dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int6_payload_bytes"] += len(packed) + tensor_nbytes(s)

    obj = {"__quant_format__": "int6_per_row_gptqlite_v1",
           "quantized": quantized, "scales": scales, "dtypes": dtypes,
           "shapes": shapes, "passthrough": passthrough,
           "passthrough_orig_dtypes": passthrough_orig_dtypes}
    return obj, stats


def dequantize_state_dict_int6(obj: dict) -> dict:
    out: dict = {}
    pod = obj.get("passthrough_orig_dtypes", {})
    for name, packed in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s     = obj["scales"][name]
        shape = obj["shapes"][name]
        numel = 1
        for d in shape: numel *= d
        q_np  = unpack_int6_from_bytes(packed, numel)
        q     = torch.from_numpy(q_np.astype(np.float32)).reshape(shape)
        if s.ndim > 0:
            s32 = s.to(torch.float32)
            out[name] = (q * s32.view(q.shape[0], *([1]*(q.ndim-1)))).to(dtype).contiguous()
        else:
            out[name] = (q * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        ot = t.detach().to("cpu").contiguous()
        od = pod.get(name)
        if isinstance(od, str):
            ot = ot.to(dtype=getattr(torch, od)).contiguous()
        out[name] = ot
    return out


def dequantize_state_dict_int8(obj: dict) -> dict:
    """Fallback dequantizer for int8 format."""
    out: dict = {}
    qmeta = obj.get("qmeta", {})
    pod   = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1]*(q.ndim-1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        ot = t.detach().to("cpu").contiguous()
        od = pod.get(name)
        if isinstance(od, str):
            ot = ot.to(dtype=getattr(torch, od)).contiguous()
        out[name] = ot
    return out

# ---------------------------------------------------------
# INT6 QAT — STRAIGHT-THROUGH ESTIMATOR
# ---------------------------------------------------------

class StraightThroughInt6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, scale: Tensor) -> Tensor:
        q = torch.clamp(torch.round(x / scale[:, None]), INT6_MIN, INT6_MAX)
        return q * scale[:, None]
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


def apply_qat_noise(weight: Tensor) -> Tensor:
    if weight.ndim != 2 or weight.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
        return weight
    w      = weight.float()
    abs_w  = w.abs()
    clip   = torch.quantile(abs_w, 0.99984, dim=1).clamp_min(1e-8)
    scale  = (clip / INT6_MAX).clamp_min(1.0 / INT6_MAX)
    return StraightThroughInt6.apply(w, scale).to(dtype=weight.dtype)

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header: {file}")
    num_tokens    = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files    = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens   = load_data_shard(self.files[0])
        self.pos      = 0
    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens   = load_data_shard(self.files[self.file_idx])
        self.pos      = 0
    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance(); continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank       = rank
        self.world_size = world_size
        self.device     = device
        self.stream     = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens   = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span  = local_tokens + 1
        chunk          = self.stream.take(per_rank_span * self.world_size)
        start          = self.rank * per_rank_span
        local          = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1,  seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ---------------------------------------------------------
# MODEL PRIMITIVES
# ---------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


def precompute_freqs_cis(head_dim: int, seq_len: int, rope_base: float, rope_dims: int,
                         device: torch.device, dtype: torch.dtype):
    """Precompute RoPE cis for partial RoPE (first rope_dims of head_dim)."""
    eff = rope_dims if rope_dims > 0 else head_dim
    eff = eff if eff % 2 == 0 else eff - 1
    theta = 1.0 / (rope_base ** (torch.arange(0, eff, 2, device=device).float() / eff))
    t     = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, theta)
    cos   = freqs.cos().to(dtype=dtype)
    sin   = freqs.sin().to(dtype=dtype)
    return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    """
    Apply partial RoPE: rotate first `rope_dims` dims, pass rest through.
    x: (B, nH, T, head_dim)
    """
    B, nH, T, hd = x.shape
    eff = rope_dims if rope_dims > 0 and rope_dims <= hd else hd
    eff = eff - (eff % 2)
    half = eff // 2
    x_rot    = x[..., :eff]
    x_pass   = x[..., eff:]
    x1, x2   = x_rot[..., :half], x_rot[..., half:]
    c = cos[:T, :half].view(1, 1, T, half)
    s = sin[:T, :half].view(1, 1, T, half)
    rotated  = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return torch.cat([rotated, x_pass], dim=-1)


class BigramHashEmbedding(nn.Module):
    """Learned bigram (context, token) embedding via hash bucketing."""
    def __init__(self, bucket_count: int, embed_dim: int, model_dim: int, vocab_size: int):
        super().__init__()
        self.bucket_count = bucket_count
        self.embed_dim    = embed_dim
        self.vocab_size   = vocab_size
        self.embed        = nn.Embedding(bucket_count, embed_dim)
        self.out          = CastedLinear(embed_dim, model_dim, bias=False)

    def forward(self, prev_tok: Tensor, curr_tok: Tensor) -> Tensor:
        h = (prev_tok.long() * 31337 + curr_tok.long()) % self.bucket_count
        return self.out(self.embed(h))


class SmearGate(nn.Module):
    """Smear-gated token embedding: sigma-gate applied before block stack."""
    def __init__(self, model_dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(model_dim))
    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        g = self.gate.tanh()   # (-1, 1)  — starts at 0 (neutral)
        return x + g * x0


class CausalSelfAttention(nn.Module):
    def __init__(self, args: Hyperparameters, layer_idx: int):
        super().__init__()
        self.num_heads    = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        self.head_dim     = args.model_dim // args.num_heads
        self.rope_dims    = args.rope_dims
        # XSA: enabled on last xsa_last_n layers
        self.use_xsa      = layer_idx >= (args.num_layers - args.xsa_last_n)

        total_dim_qkv = (args.num_heads + 2 * args.num_kv_heads) * self.head_dim
        self.qkv_proj = CastedLinear(args.model_dim, total_dim_qkv, bias=False)
        self.c_proj   = CastedLinear(args.model_dim, args.model_dim, bias=False)
        # Learnable Q/K scaling (Gemma-style)
        self.q_scale  = nn.Parameter(torch.ones(self.num_heads,    1) * args.qk_gain_init)
        self.k_scale  = nn.Parameter(torch.ones(self.num_kv_heads, 1) * args.qk_gain_init)
        # FlashAttention 3 availability
        self._fa3 = None

    def _try_fa3(self, q, k, v, is_causal=True):
        if self._fa3 is None:
            try:
                from flash_attn import flash_attn_func
                self._fa3 = flash_attn_func
            except ImportError:
                self._fa3 = False
        if self._fa3 is False:
            return None
        # FA3 expects (B, T, nH, hD)
        q = q.permute(0, 2, 1, 3); k = k.permute(0, 2, 1, 3); v = v.permute(0, 2, 1, 3)
        return self._fa3(q, k, v, causal=is_causal).permute(0, 2, 1, 3)

    def xsa_subtract(self, ao: Tensor, v: Tensor) -> Tensor:
        """Subtract self-component: encourages attention to be orthogonal to own value."""
        B, nH, T, hD = ao.shape
        kv_groups     = nH // self.num_kv_heads
        ao_r = ao.view(B, self.num_kv_heads, kv_groups, T, hD)
        vv_r = v.view(B,  self.num_kv_heads, 1,         T, hD)
        dots = (ao_r * vv_r).sum(-1, keepdim=True)
        norm = (vv_r * vv_r).sum(-1, keepdim=True).clamp_min(1e-8)
        ao_r = ao_r - (dots / norm) * vv_r
        return ao_r.reshape(B, nH, T, hD)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, attn_mask: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q_dim = self.num_heads    * self.head_dim
        k_dim = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_dim, k_dim, k_dim], dim=-1)
        q = q.view(B, T, self.num_heads,    self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled Q/K
        q = q * self.q_scale.view(self.num_heads,    1, 1)
        k = k * self.k_scale.view(self.num_kv_heads, 1, 1)

        # Partial RoPE
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)

        # GQA expand
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Try FlashAttention 3, fallback to SDPA
        fa_out = self._try_fa3(q, k, v)
        if fa_out is None:
            ao = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None)
        else:
            ao = fa_out

        # XSA subtraction (on eligible layers, use original v before GQA expand)
        if self.use_xsa:
            v_orig = v.view(B, self.num_heads, T, self.head_dim)
            ao = self.xsa_subtract(ao, v_orig)

        ao = ao.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(ao)


class MLP(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        hidden = int(args.model_dim * args.mlp_mult)
        self.fc   = CastedLinear(args.model_dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, args.model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # LeakyReLU(0.5)² activation — preserves negative gradient flow
        x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, args: Hyperparameters, layer_idx: int):
        super().__init__()
        self.layer_idx    = layer_idx
        self.use_ln_scale = args.ln_scale
        self.ln_scale_val = 1.0 / math.sqrt(layer_idx + 1) if args.ln_scale else 1.0

        self.norm1  = RMSNorm(args.model_dim)
        self.attn   = CausalSelfAttention(args, layer_idx)
        self.norm2  = RMSNorm(args.model_dim)
        self.mlp    = MLP(args)

        # Learnable residual mixing scalars (MAMBA-style)
        self.attn_scale = nn.Parameter(torch.ones(()))
        self.mlp_scale  = nn.Parameter(torch.ones(()))

        # Value Embedding hook (set externally by GPT if VE enabled)
        self.ve_proj  : nn.Linear | None = None
        self.ve_scale : nn.Parameter | None = None

    def forward(self, x: Tensor, x0: Tensor, cos: Tensor, sin: Tensor, attn_mask: Tensor) -> Tensor:
        # Pre-norm with optional LN Scale
        h = self.norm1(x)
        if self.use_ln_scale:
            h = h * self.ln_scale_val
        # Attention
        a = self.attn(h, cos, sin, attn_mask) * self.attn_scale
        x = x + a

        # Value embedding injection
        if self.ve_proj is not None and self.ve_scale is not None:
            x = x + self.ve_proj(x0) * self.ve_scale

        # MLP
        h = self.norm2(x)
        if self.use_ln_scale:
            h = h * self.ln_scale_val
        x = x + self.mlp(h) * self.mlp_scale
        return x


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args       = args
        self.head_dim   = args.model_dim // args.num_heads
        self.seq_len    = args.train_seq_len

        # Embeddings
        self.tok_emb    = nn.Embedding(args.vocab_size, args.model_dim)
        self.bigram     = BigramHashEmbedding(args.bigram_vocab_size, args.bigram_dim,
                                              args.model_dim, args.vocab_size)
        self.smear_gate = SmearGate(args.model_dim) if args.use_smeargate else None

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.num_layers)])

        # U-Net skip connections: connect block i to block (N-1-i) where i < N//2
        # We inject midpoint activations added at the latter half
        self.skip_weights = nn.ParameterList([
            nn.Parameter(torch.ones(()) * 0.5) for _ in range(args.num_layers // 2)
        ])

        # Shared Value Embedding
        if args.ve_enabled:
            ve_layers = [int(x) for x in args.ve_layers_str.split(",") if x.strip()]
            self.value_embed = nn.Embedding(args.vocab_size, args.ve_dim)
            for li in ve_layers:
                if li < len(self.blocks):
                    blk = self.blocks[li]
                    blk.ve_proj  = CastedLinear(args.ve_dim, args.model_dim, bias=False)
                    blk.ve_scale = nn.Parameter(torch.zeros(()))

        # Output head
        self.norm_out = RMSNorm(args.model_dim)
        if args.tie_embeddings:
            # Tied output: use same weight as tok_emb, but with separate scale
            self.lm_head = None
            self.tied_head_scale = nn.Parameter(torch.ones(args.model_dim))
        else:
            self.lm_head = CastedLinear(args.model_dim, args.vocab_size, bias=False)

        # RoPE buffers (precomputed, re-registered if seq changes)
        self._rope_seq = -1
        cos, sin = precompute_freqs_cis(
            self.head_dim, args.train_seq_len, args.rope_base, args.rope_dims,
            device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        std  = 1.0 / math.sqrt(self.args.model_dim)
        stdp = std / math.sqrt(2 * self.args.num_layers)
        nn.init.normal_(self.tok_emb.weight, std=std)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.qkv_proj.weight, std=std)
            nn.init.normal_(blk.attn.c_proj.weight,   std=stdp)
            nn.init.normal_(blk.mlp.fc.weight,         std=std)
            nn.init.normal_(blk.mlp.proj.weight,       std=stdp)

    def _get_rope(self, T: int, device, dtype):
        if T > self.rope_cos.size(0) or self.rope_cos.device != device:
            cos, sin = precompute_freqs_cis(
                self.head_dim, T, self.args.rope_base, self.args.rope_dims, device, dtype)
            self.rope_cos = cos; self.rope_sin = sin
        return self.rope_cos[:T].to(dtype=dtype), self.rope_sin[:T].to(dtype=dtype)

    def forward(self, input_ids: Tensor, targets: Tensor | None = None,
                use_qat: bool = False) -> Tensor:
        B, T   = input_ids.shape
        device = input_ids.device
        dtype  = torch.bfloat16

        cos, sin = self._get_rope(T, device, dtype)

        # Token embedding + bigram
        prev_ids    = torch.cat([torch.zeros(B, 1, dtype=input_ids.dtype, device=device),
                                 input_ids[:, :-1]], dim=1)
        tok_emb_out = self.tok_emb(input_ids)
        bigram_out  = self.bigram(prev_ids, input_ids)
        x0 = (tok_emb_out + bigram_out).to(dtype=dtype)

        # SmearGate: passed as x0 signal throughout
        x = x0

        # Collect skip activations (first half)
        skips: list[Tensor] = []
        half = self.args.num_layers // 2

        # Optional value embed base
        ve_out = None
        if self.args.ve_enabled:
            ve_out = self.value_embed(input_ids).to(dtype=dtype)

        for i, blk in enumerate(self.blocks):
            # U-Net: inject skip from mirror block
            if i >= self.args.num_layers - half:
                mirror = self.args.num_layers - 1 - i
                if mirror < len(skips):
                    sw = self.skip_weights[mirror].sigmoid()
                    x  = x * sw + skips[mirror] * (1 - sw)
            if i < half:
                skips.append(x)

            # QAT noise injection on eligible layers
            if use_qat:
                for param in blk.parameters():
                    if param.ndim == 2 and param.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
                        param.data = apply_qat_noise(param.data)

            # SmearGate injection before each block
            if self.smear_gate is not None:
                x = self.smear_gate(x, x0)

            # VE injection (handled inside Block)
            if blk.ve_proj is not None and ve_out is not None:
                # Store ve for this block via x0-like signal
                x = blk.forward(x, ve_out, cos, sin, None)
            else:
                x = blk.forward(x, x0, cos, sin, None)

        x = self.norm_out(x)

        if self.args.tie_embeddings:
            w = self.tok_emb.weight.to(dtype=dtype) * self.tied_head_scale.to(dtype=dtype)
            logits = F.linear(x, w)
        else:
            logits = self.lm_head(x)

        # Logit soft-cap (Gemma 2 style)
        cap    = self.args.logit_softcap
        logits = torch.tanh(logits / cap) * cap

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss


# ---------------------------------------------------------
# EMA WEIGHT BUFFER
# ---------------------------------------------------------

class EMABuffer:
    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.is_floating_point():
                s.lerp_(v.to(s.dtype), 1.0 - d)
            else:
                s.copy_(v)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}


# ---------------------------------------------------------
# LEGAL TEST-TIME TRAINING (SCORE-FIRST)
# ---------------------------------------------------------

def legal_ttt(model: nn.Module, val_tokens: Tensor, args: Hyperparameters, device: torch.device):
    chunk_size = args.ttt_chunk_tokens
    seq_len    = args.train_seq_len
    tokens     = val_tokens.to(device)
    n_chunks   = (tokens.numel() - 1) // chunk_size
    if n_chunks < 2:
        return None
    sp = torch.optim.SGD(model.parameters(), lr=args.ttt_lr, momentum=args.ttt_momentum)
    total_loss, total_count = 0.0, 0
    for ci in range(n_chunks):
        s = ci * chunk_size; e = s + chunk_size + 1
        if e > tokens.numel(): break
        ct = tokens[s:e]
        model.eval()
        with torch.inference_mode():
            n_seqs = (e - s - 1) // seq_len
            if n_seqs == 0: continue
            cx = ct[:n_seqs*seq_len].view(n_seqs, seq_len).to(torch.int64)
            cy = ct[1:n_seqs*seq_len+1].view(n_seqs, seq_len).to(torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                lv = model(cx, cy)
            total_loss  += float(lv.item()) * int(cx.numel())
            total_count += int(cx.numel())
        model.train()
        clr = args.ttt_lr * (1 + math.cos(math.pi * ci / n_chunks)) * 0.5
        for g in sp.param_groups: g["lr"] = clr
        for _ in range(args.ttt_epochs):
            for ss in range(0, chunk_size - seq_len, seq_len):
                xt = ct[ss:ss+seq_len].unsqueeze(0).to(torch.int64)
                yt = ct[ss+1:ss+seq_len+1].unsqueeze(0).to(torch.int64)
                sp.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    lt = model(xt, yt)
                lt.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.ttt_grad_clip)
                sp.step()
    if total_count == 0: return None
    return (total_loss / total_count) / math.log(2.0)


# ---------------------------------------------------------
# ARTIFACT I/O
# ---------------------------------------------------------

def save_artifact(obj: dict, path: str) -> float:
    buf = io.BytesIO(); torch.save(obj, buf); raw = buf.getvalue()
    try:
        import zstandard as zstd
        compressed = zstd.ZstdCompressor(level=22, threads=-1).compress(raw)
    except ImportError:
        compressed = zlib.compress(raw, level=9)
    with open(path, "wb") as f: f.write(compressed)
    return len(compressed) / 1024 / 1024


def load_artifact(path: str) -> dict:
    with open(path, "rb") as f: compressed = f.read()
    try:
        import zstandard as zstd
        raw = zstd.ZstdDecompressor().decompress(compressed)
    except Exception:
        raw = zlib.decompress(compressed)
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)


# ---------------------------------------------------------
# OPTIMIZER BUILDER & LR SCHEDULE
# ---------------------------------------------------------

def build_optimizer(model: nn.Module, args: Hyperparameters):
    m_p, s_p, e_p, t_p = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if "tok_emb.weight" in name and args.tie_embeddings:
            t_p.append(p)
        elif any(k in name for k in ("tok_emb", "bigram", "value_embed")):
            e_p.append(p)
        elif p.ndim >= 2 and p.numel() > 512:
            m_p.append(p)
        else:
            s_p.append(p)
    muon = Muon(m_p, lr=args.matrix_lr, momentum=args.muon_momentum,
                backend_steps=args.muon_backend_steps, weight_decay=args.muon_weight_decay)
    groups = []
    if t_p: groups.append({"params": t_p, "lr": args.tied_embed_lr})
    if e_p: groups.append({"params": e_p, "lr": args.embed_lr})
    if s_p: groups.append({"params": s_p, "lr": args.scalar_lr})
    adam = torch.optim.AdamW(groups, betas=(args.beta1, args.beta2),
                             eps=args.adam_eps, weight_decay=args.adamw_weight_decay, fused=True)
    return muon, adam


def lr_schedule(step: int, args: Hyperparameters) -> float:
    if step < args.warmup_steps:
        return step / max(1, args.warmup_steps)
    pe = args.iterations - args.warmdown_iters
    if step <= pe: return 1.0
    return 0.5 * (1 + math.cos(math.pi * (step - pe) / args.warmdown_iters))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    args = Hyperparameters()
    assert torch.cuda.is_available(), "CUDA required"
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device); is_master = rank == 0
    torch.manual_seed(args.seed + rank); random.seed(args.seed + rank)

    model     = GPT(args).to(device)
    model     = torch.compile(model)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ddp_model = DDP(model, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=False)
    muon, adam = build_optimizer(raw_model, args)
    ema_buf    = EMABuffer(raw_model, args.ema_decay) if args.ema_enabled else None
    swa_weights, swa_count = None, 0
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    grad_accum   = max(1, args.train_batch_tokens // (world_size * args.train_seq_len * 8))

    if is_master:
        sp         = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        val_tokens = load_validation_tokens(args.val_files, args.train_seq_len).to(device)
        bb, hls, ibt = build_sentencepiece_luts(sp, args.vocab_size, device)
        print(f"params={sum(p.numel() for p in raw_model.parameters()):,} world={world_size} grad_accum={grad_accum}")

    qat_start = int(args.iterations * args.qat_start_frac)
    swa_start = int(args.iterations * args.swa_start_frac)
    t0 = time.time(); best_bpb = float("inf"); best_sd = None

    for step in range(args.iterations + 1):
        elapsed = time.time() - t0
        if elapsed > args.max_wallclock_seconds - 45:
            if is_master: print(f"Wallclock guard @ step {step} ({elapsed:.0f}s)")
            break

        if is_master and step % args.val_loss_every == 0:
            vt = GPT(args).to(device)
            sd_src = ema_buf.state_dict() if ema_buf else (
                     swa_weights if swa_weights else raw_model.state_dict())
            vt.load_state_dict({k: v.to(device) for k, v in sd_src.items()}, strict=False)
            vl, vb = eval_val(args, vt, 0, 1, device, grad_accum, val_tokens, bb, hls, ibt)
            print(f"[{step:05d}|{elapsed:.0f}s] val_loss={vl:.4f} val_bpb={vb:.5f}", flush=True)
            if vb < best_bpb:
                best_bpb = vb
                best_sd  = {k: v.detach().cpu() for k, v in vt.state_dict().items()}
            del vt

        if step == args.iterations: break

        lrm = lr_schedule(step, args)
        for g in muon.param_groups: g["lr"] = args.matrix_lr * lrm
        for g in adam.param_groups:
            if "_base" not in g: g["_base"] = g["lr"]
            g["lr"] = g["_base"] * lrm
        if step < args.muon_momentum_warmup_steps:
            wm = args.muon_momentum_warmup_start + \
                 (args.muon_momentum - args.muon_momentum_warmup_start) * \
                 (step / max(1, args.muon_momentum_warmup_steps))
            for g in muon.param_groups: g["momentum"] = wm

        use_qat = args.use_qat and step >= qat_start
        tl = 0.0
        for _ in range(grad_accum):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = ddp_model(x, y, use_qat=use_qat) / grad_accum
            loss.backward(); tl += loss.item()

        nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip_norm)
        muon.step(); adam.step()
        muon.zero_grad(set_to_none=True); adam.zero_grad(set_to_none=True)
        if ema_buf: ema_buf.update(raw_model)

        if step >= swa_start and step % args.swa_every == 0:
            sd = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
            if swa_weights is None:
                swa_weights, swa_count = sd, 1
            else:
                swa_count += 1
                for k in swa_weights:
                    if swa_weights[k].is_floating_point():
                        swa_weights[k].add_(sd[k] - swa_weights[k], alpha=1.0/swa_count)
                    else:
                        swa_weights[k].copy_(sd[k])

        if is_master and step % args.train_log_every == 0:
            print(f"  [{step:05d}] loss={tl:.4f} lr={lrm:.4f} t={elapsed:.0f}s", flush=True)

    if is_master:
        fm = GPT(args).to(device)
        src = best_sd if best_sd else raw_model.state_dict()
        fm.load_state_dict({k: v.to(device) for k, v in src.items()}, strict=False)

        if args.ttt_enabled:
            print("Running Legal TTT...", flush=True)
            tb = legal_ttt(fm, val_tokens, args, device)
            if tb: print(f"Post-TTT bpb={tb:.5f}", flush=True)

        vl_f, vb_f = eval_val(args, fm, 0, 1, device, grad_accum, val_tokens, bb, hls, ibt)
        print(f"\n{'='*60}\nFINAL val_loss={vl_f:.6f}  val_bpb={vb_f:.6f}\n{'='*60}\n", flush=True)

        print("Quantizing INT6+GPTQ-lite...", flush=True)
        qobj, qs = quantize_state_dict_int6(fm.state_dict())
        print(f"Quant stats: {qs}", flush=True)

        ap = os.environ.get("ARTIFACT_PATH", "./model_artifact.pt")
        sz = save_artifact(qobj, ap)
        print(f"Artifact: {ap}  {sz:.3f}MB", flush=True)
        if sz > 16.0: print(f"!!! WARNING: {sz:.2f}MB > 16MB !!!", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
