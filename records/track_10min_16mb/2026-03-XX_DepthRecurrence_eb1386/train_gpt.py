# 8x2 recurrence, int6, swiglu 3x, smeargate, bigramhash, swa, qat

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
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
    import zlib

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
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    recurrence = int(os.environ.get("RECURRENCE", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.3))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 4096))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.5))
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.7))

# muon with wd

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
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      weight_decay=weight_decay, nesterov=nesterov))

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
            wd = group["weight_decay"]
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
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# bpb eval

def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
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
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Val too short for seq_len={seq_len}")
    return tokens[: usable + 1]

def eval_val_sliding(args, model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    # sliding window eval
    sl = args.eval_seq_len
    stride = args.eval_stride
    batch_seqs = args.eval_batch_seqs
    total_tokens = val_tokens.numel()
    all_starts = list(range(0, total_tokens - sl, stride))
    if not all_starts or all_starts[-1] + sl < total_tokens:
        all_starts.append(max(total_tokens - sl, 0))
    my_starts = all_starts[rank::world_size]
    val_nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_starts), batch_seqs):
            batch_starts = my_starts[bi:bi + batch_seqs]
            bs = len(batch_starts)
            x_list, y_list = [], []
            for ws in batch_starts:
                we = min(ws + sl, total_tokens - 1)
                chunk = val_tokens[ws:we + 1].to(dtype=torch.int64)
                x_list.append(chunk[:-1])
                y_list.append(chunk[1:])
            max_len = max(x.size(0) for x in x_list)
            x_batch = torch.zeros(bs, max_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bs, max_len, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(zip(x_list, y_list)):
                x_batch[i, :x.size(0)] = x.to(device)
                y_batch[i, :y.size(0)] = y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.forward_logits(x_batch) if hasattr(model, 'forward_logits') else model.module.forward_logits(x_batch)
            nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y_batch.reshape(-1), reduction="none").reshape(bs, max_len)
            for i, ws in enumerate(batch_starts):
                wlen = min(sl, total_tokens - 1 - ws)
                score_from = max(wlen - stride, 0) if ws > 0 else 0
                scored_nll = nll[i, score_from:wlen]
                scored_y = y_batch[i, score_from:wlen]
                prev_start = max(score_from - 1, 0)
                scored_prev = x_batch[i, prev_start:prev_start + scored_nll.numel()]
                val_nll_sum += scored_nll.to(torch.float64).sum()
                val_token_count += scored_nll.numel()
                tgt_ids = scored_y
                tb = base_bytes_lut[tgt_ids].to(torch.int16)
                if scored_prev.numel() == tgt_ids.numel():
                    tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[scored_prev]).to(torch.int16)
                val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_nll_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)

# int6 quantization

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get("CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,smear_gate,bigram_scale"
    ).split(",") if p
)
INT6_CLIP = 31
INT6_KEEP_FLOAT_MAX_NUMEL = 65_536

def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / INT6_CLIP).clamp_min(1e-12).to(torch.float16)
        scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -INT6_CLIP - 1, INT6_CLIP).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / INT6_CLIP, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -INT6_CLIP - 1, INT6_CLIP).to(torch.int8)
    return q, scale

def quantize_state_dict_int6(state_dict, embed_name="tok_emb.weight"):
    result, meta, stats = {}, {}, {"param_count": 0, "payload_bytes": 0}
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        if not t.is_floating_point():
            result[name] = t
            meta[name] = "passthrough"
            stats["payload_bytes"] += t.numel() * t.element_size()
            continue
        if name == embed_name:
            result[name] = t.to(torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            stats["payload_bytes"] += t.numel() * 2
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float().contiguous()
            meta[name] = "passthrough_ctrl"
            stats["payload_bytes"] += t.numel() * 4
            continue
        if t.numel() <= INT6_KEEP_FLOAT_MAX_NUMEL:
            result[name] = t.to(torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            stats["payload_bytes"] += t.numel() * 2
            continue
        q, s = quantize_int6_per_row(t)
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = "int6"
        stats["payload_bytes"] += q.numel() + s.numel() * 2
    return result, meta, stats

def dequantize_state_dict_int6(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta[name]
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            out[name] = t.to(orig.dtype).contiguous() if t.dtype != orig.dtype else t.contiguous()
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig.dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(orig.dtype).contiguous()
    return out

def compress_bytes(data: bytes) -> bytes:
    if _COMPRESSOR == "zstd":
        return zstandard.ZstdCompressor(level=22).compress(data)
    return zlib.compress(data, level=9)

def decompress_bytes(data: bytes) -> bytes:
    if _COMPRESSOR == "zstd":
        return zstandard.ZstdDecompressor().decompress(data)
    return zlib.decompress(data)

# data loading

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard: {file}")
    num_tokens = int(header[2])
    if file.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read: {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
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
        self.rank, self.world_size, self.device = rank, world_size, device
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

# transformer

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
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
                rm = w32.abs().amax(dim=1)
                s = (rm / INT6_CLIP).clamp_min(1e-12)
                wq = (torch.clamp(torch.round(w32 / s[:, None]), -INT6_CLIP - 1, INT6_CLIP) * s[:, None]).to(x.dtype)
            w = w + (wq - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim * 2 / 3)
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.gate(x)) * self.fc(x))

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bvs = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bvs - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 recurrence, tie_embeddings, tied_embed_init_std, logit_softcap, rope_base,
                 qk_gain_init, bigram_vocab_size, bigram_dim):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.recurrence = recurrence
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = SmearGate(model_dim)
        self.bigram_hash = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init) for _ in range(num_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        # ortho init
        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "_zero_init", False):
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.orthogonal_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        return F.cross_entropy(self.forward_logits(input_ids).float(), target_ids.reshape(-1), reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = self.smear_gate(x)
        x = x + self.bigram_hash(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        for _ in range(self.recurrence):
            for block in self.blocks:
                x = block(x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        if self.tie_embeddings:
            lp = F.linear(x, self.tok_emb.weight)
        else:
            lp = self.lm_head(x)
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)

def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assert world_size > 0 and 8 % world_size == 0
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    assert torch.cuda.is_available()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)
    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)
    log0(code, console=False); log0("=" * 80, console=False)
    log0(f"Python {sys.version} PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    assert int(sp.vocab_size()) == args.vocab_size
    dataset_dir = Path(args.data_path).resolve()
    val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_tokens:{val_tokens.numel()-1} compressor:{_COMPRESSOR}")
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        recurrence=args.recurrence, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    ).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    block_np = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_np if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_np if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.Adam([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.muon_weight_decay)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    all_scalar = scalar_params + list(base_model.smear_gate.parameters())
    opt_scalar = torch.optim.Adam([{"params": all_scalar, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_bigram = torch.optim.Adam([{"params": list(base_model.bigram_hash.parameters()), "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar, opt_bigram]
    if base_model.lm_head is not None:
        opt_head = torch.optim.Adam([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, opt_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"params:{n_params} layers:{args.num_layers} rec:{args.recurrence} eff:{args.num_layers*args.recurrence} dim:{args.model_dim}")
    log0(f"mlp:{args.mlp_mult}x lr:{args.matrix_lr} wd:{args.muon_weight_decay} clip:{args.grad_clip_norm}")
    log0(f"batch:{args.train_batch_tokens} seq:{args.train_seq_len} eval_seq:{args.eval_seq_len} stride:{args.eval_stride}")
    log0(f"swa:{args.swa_enabled}/{args.swa_every} qat:{args.qat_start_frac} seed:{args.seed}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)
    max_wc_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wc_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        sms = elapsed_ms / max(step, 1)
        wms = args.warmdown_iters * sms
        rms = max(max_wc_ms - elapsed_ms, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0
    if args.warmup_steps > 0:
        init_sd = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opts = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): (model(x, y) * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (ws+1) % 10 == 0: log0(f"warmup:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opts, strict=True): o.load_state_dict(s)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state, swa_count = None, 0
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val_sliding(args, model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} time:{training_time_ms:.0f}ms avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early step:{step}/{args.iterations} time:{training_time_ms:.0f}ms")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        frac = elapsed_ms / max_wc_ms if max_wc_ms else step / max(args.iterations, 1)
        CastedLinear._qat_enabled = frac >= args.qat_start_frac
        if args.swa_enabled and scale < args.swa_start_frac and step > 0 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {n: t.detach().cpu().clone().float() for n, t in base_model.state_dict().items()}
                swa_count = 1
            else:
                for n, t in base_model.state_dict().items(): swa_state[n] += t.detach().cpu().float()
                swa_count += 1
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = micro == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        mfrac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for g in opt_muon.param_groups: g["momentum"] = (1-mfrac)*args.muon_momentum_warmup_start + mfrac*args.muon_momentum
        for o in optimizers:
            for g in o.param_groups: g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers: o.step()
        zero_grad_all()
        step += 1
        atm = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log0(f"step:{step}/{args.iterations} loss:{train_loss.item():.4f} time:{atm:.0f}ms avg:{atm/step:.2f}ms")
        reached = max_wc_ms is not None and atm >= max_wc_ms
        if distributed and max_wc_ms is not None:
            rt = torch.tensor(int(reached), device=device)
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
            reached = bool(rt.item())
        if stop_after_step is None and reached: stop_after_step = step
    log0(f"peak_mem:{torch.cuda.max_memory_allocated()//1024//1024}MiB")
    if args.swa_enabled and swa_state and swa_count > 1:
        log0(f"swa_avg:{swa_count} checkpoints")
        avg = {n: (t / swa_count).to(base_model.state_dict()[n].dtype) for n, t in swa_state.items()}
        base_model.load_state_dict(avg, strict=True)
        del swa_state, avg
    CastedLinear._qat_enabled = False
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        cb = len(code.encode("utf-8"))
        log0(f"raw_model:{os.path.getsize('final_model.pt')} code:{cb}")
    template_sd = base_model.state_dict()
    qr, qm, qs = quantize_state_dict_int6(template_sd)
    buf = io.BytesIO(); torch.save({"result": qr, "meta": qm}, buf)
    blob = compress_bytes(buf.getvalue())
    if master_process:
        with open("final_model.int6.ptz", "wb") as f: f.write(blob)
        qfb = os.path.getsize("final_model.int6.ptz")
        cb = len(code.encode("utf-8"))
        log0(f"int6+{_COMPRESSOR}:{qfb} code:{cb} total:{qfb+cb} params:{qs['param_count']}")
    if distributed: dist.barrier()
    with open("final_model.int6.ptz", "rb") as f: disk = f.read()
    loaded = torch.load(io.BytesIO(decompress_bytes(disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int6(loaded["result"], loaded["meta"], template_sd), strict=True)
    torch.cuda.synchronize(); tq = time.perf_counter()
    ql, qb = eval_val_sliding(args, model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int6_{_COMPRESSOR}_roundtrip val_loss:{ql:.4f} val_bpb:{qb:.4f} eval_time:{1000*(time.perf_counter()-tq):.0f}ms")
    log0(f"final_int6_{_COMPRESSOR}_roundtrip_exact val_loss:{ql:.8f} val_bpb:{qb:.8f}")
    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
