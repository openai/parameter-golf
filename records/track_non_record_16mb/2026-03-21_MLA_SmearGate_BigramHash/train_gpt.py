"""
train_gpt_mla.py
================
All techniques stacked. Run with:
  torchrun --nproc_per_node=8 train_gpt_mla.py          (H100 submission)
  USE_COMPILE=0 MAX_VAL_TOKENS=65536 python train_gpt_mla.py  (laptop test)

Requires: pip install zstandard

Techniques vs baseline (1.2244 bpb):
  1. MLA — Multi-Head Latent Attention (kv_down + k_up/v_up)
     Saves ~131k params/layer vs GQA → funds extra layers.
  2. 13 layers (was 9)
  3. SmearGate MLP — relu^2 gated MLP, mlp_mult=3 (was relu^2 ungated, mult=2)
  4. BigramHash(10240, dim=128) — proven -0.002 bpb
  5. OrthoInit — orthogonal init on all linear weights
  6. int5 MLP / int6 attn / fp16 embed + zstd-22 — better compression
  7. SWA (start_frac=0.4, every=50) — proven -0.0006 bpb
  8. Muon WD=0.04, momentum=0.99, warmdown=3000, grad_clip=0.3
  9. Sliding-window eval (stride=64) — every token gets ≥960 context tokens
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import pickle
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

# zstd compression — install with: pip install zstandard
try:
    import zstandard as _zstd

    _ZSTD_LEVEL = int(os.environ.get("ZSTD_LEVEL", "22"))

    def compress_bytes(b: bytes) -> bytes:
        return _zstd.ZstdCompressor(level=_ZSTD_LEVEL).compress(b)

    def decompress_bytes(b: bytes) -> bytes:
        return _zstd.ZstdDecompressor().decompress(b)

    COMPRESSOR = f"zstd-{_ZSTD_LEVEL}"
except ImportError:
    import zlib

    def compress_bytes(b: bytes) -> bytes:
        return zlib.compress(b, level=9)

    def decompress_bytes(b: bytes) -> bytes:
        return zlib.decompress(b)

    COMPRESSOR = "zlib-9 (install zstandard for zstd-22!)"

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    max_val_tokens = int(os.environ.get("MAX_VAL_TOKENS", 0))  # 0 = full

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 13))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    kv_rank = int(os.environ.get("KV_RANK", 128))
    bigram_buckets = int(os.environ.get("BIGRAM_BUCKETS", 10240))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.4))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    sliding_window_stride = int(os.environ.get("SLIDING_WINDOW_STRIDE", 64))


# ─────────────────────────────────────────────────────────────────────────────
# MUON OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return (X.T if G.size(0) > G.size(1) else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, wd=0.0):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                wd=wd,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        is_dist = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        rank = dist.get_rank() if is_dist else 0
        for group in self.param_groups:
            params, lr = group["params"], group["lr"]
            momentum, ns, nesterov = (
                group["momentum"],
                group["backend_steps"],
                group["nesterov"],
            )
            wd = group["wd"]
            if not params:
                continue
            total = sum(p.numel() for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(g)
                    buf = st["buf"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if is_dist:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = flat[curr : curr + p.numel()].view_as(p).to(p.dtype)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# MIXED QUANTIZATION + ZSTD SERIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
# int5 [-16,15]  → MLP weights     (most compressible under zstd)
# int6 [-32,31]  → Attention weights
# fp16           → tok_emb.weight  (used twice, int error compounds)
# fp32           → control scalars + small tensors

CONTROL_PATTERNS = tuple(
    p
    for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,bigram.proj",
    ).split(",")
    if p
)

MLP_PATTERNS = ("mlp.gate", "mlp.up", "mlp.proj")
ATTN_PATTERNS = ("attn.c_q", "attn.kv_down", "attn.k_up", "attn.v_up", "attn.proj")
FP16_PATTERNS = ("tok_emb.weight",)
MAX_NUMEL_FLOAT_PASSTHROUGH = 65_536


def _quant_bounds(name: str) -> tuple[int, int]:
    if any(p in name for p in MLP_PATTERNS):
        return -16, 15  # int5
    if any(p in name for p in ATTN_PATTERNS):
        return -32, 31  # int6
    return -127, 127  # int8


def _quantize_tensor(t: Tensor, lo: int, hi: int) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    hi_f = float(hi)
    if t32.ndim == 2:
        clip = torch.quantile(t32.abs(), 0.9999984, dim=1).clamp(min=1e-8)
        scale = (clip / hi_f).clamp(min=1.0 / hi_f)
        q = (
            torch.clamp(
                torch.round(
                    torch.clamp(t32, -clip[:, None], clip[:, None]) / scale[:, None]
                ),
                lo,
                hi,
            )
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(torch.float16).contiguous()
    clip = float(torch.quantile(t32.abs().flatten(), 0.9999984).item())
    scale = torch.tensor(max(clip / hi_f, 1.0 / hi_f), dtype=torch.float32)
    q = (
        torch.clamp(torch.round(torch.clamp(t32, -clip, clip) / scale), lo, hi)
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


def quantize_state_dict(sd: dict[str, Tensor]) -> tuple[bytes, dict]:
    buf = io.BytesIO()
    meta: dict[str, dict] = {}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        if any(p in name for p in FP16_PATTERNS):
            data = t.to(torch.float16).numpy().tobytes()
            meta[name] = {
                "kind": "fp16",
                "shape": list(t.shape),
                "offset": buf.tell(),
                "n": len(data),
            }
            buf.write(data)
        elif (
            not t.is_floating_point()
            or t.numel() <= MAX_NUMEL_FLOAT_PASSTHROUGH
            or any(p in name for p in CONTROL_PATTERNS)
        ):
            data = t.to(torch.float32).numpy().tobytes()
            meta[name] = {
                "kind": "fp32",
                "shape": list(t.shape),
                "dtype": str(t.dtype).removeprefix("torch."),
                "offset": buf.tell(),
                "n": len(data),
            }
            buf.write(data)
        else:
            lo, hi = _quant_bounds(name)
            q, scale = _quantize_tensor(t, lo, hi)
            qb, sb = q.numpy().tobytes(), scale.numpy().tobytes()
            meta[name] = {
                "kind": "quant",
                "shape": list(t.shape),
                "dtype": str(t.dtype).removeprefix("torch."),
                "lo": lo,
                "hi": hi,
                "per_row": (t.ndim == 2),
                "scale_shape": list(scale.shape),
                "q_off": buf.tell(),
                "q_n": len(qb),
                "s_off": buf.tell() + len(qb),
                "s_n": len(sb),
            }
            buf.write(qb)
            buf.write(sb)
    return compress_bytes(buf.getvalue()), meta


def dequantize_state_dict(compressed: bytes, meta: dict) -> dict[str, Tensor]:
    raw = decompress_bytes(compressed)
    out: dict[str, Tensor] = {}
    for name, m in meta.items():
        kind = m["kind"]
        if kind == "fp16":
            arr = np.frombuffer(
                raw,
                dtype=np.float16,
                count=int(np.prod(m["shape"])),
                offset=m["offset"],
            )
            out[name] = (
                torch.from_numpy(arr.copy()).reshape(m["shape"]).to(torch.bfloat16)
            )
        elif kind == "fp32":
            arr = np.frombuffer(
                raw,
                dtype=np.float32,
                count=int(np.prod(m["shape"])),
                offset=m["offset"],
            )
            t = torch.from_numpy(arr.copy()).reshape(m["shape"])
            dt_str = m.get("dtype", "float32")
            out[name] = t.to(getattr(torch, dt_str)) if dt_str != "float32" else t
        else:
            q = (
                torch.from_numpy(
                    np.frombuffer(
                        raw,
                        dtype=np.int8,
                        count=int(np.prod(m["shape"])),
                        offset=m["q_off"],
                    ).copy()
                )
                .reshape(m["shape"])
                .float()
            )
            sc = np.frombuffer(
                raw,
                dtype=np.float16,
                count=int(np.prod(m["scale_shape"])),
                offset=m["s_off"],
            ).copy()
            scale = torch.from_numpy(sc).reshape(m["scale_shape"]).float()
            if m["per_row"]:
                dq = q * scale.view(-1, *([1] * (q.ndim - 1)))
            else:
                dq = q * float(scale.item())
            out[name] = dq.to(getattr(torch, m["dtype"])).contiguous()
    return out


def save_artifact(model: nn.Module, path: str) -> int:
    compressed, meta = quantize_state_dict(model.state_dict())
    payload = pickle.dumps({"c": compressed, "m": meta})
    Path(path).write_bytes(payload)
    return len(payload)


def load_artifact(path: str, model: nn.Module) -> None:
    obj = pickle.loads(Path(path).read_bytes())
    model.load_state_dict(dequantize_state_dict(obj["c"], obj["m"]), strict=True)


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER-AGNOSTIC BPB EVAL
# ─────────────────────────────────────────────────────────────────────────────


def build_sentencepiece_luts(sp, vocab_size: int, device):
    sz = max(int(sp.vocab_size()), vocab_size)
    bb = np.zeros(sz, dtype=np.int16)
    hs = np.zeros(sz, dtype=np.bool_)
    ib = np.ones(sz, dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        ib[tid] = False
        if sp.is_byte(tid):
            bb[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            hs[tid] = True
            piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(bb, dtype=torch.int16, device=device),
        torch.tensor(hs, dtype=torch.bool, device=device),
        torch.tensor(ib, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([_load_shard(Path(f)) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def _byte_count(x: Tensor, y: Tensor, bb_lut, hs_lut, ib_lut) -> Tensor:
    tgt, prev = y.reshape(-1), x.reshape(-1)
    tb = bb_lut[tgt].to(torch.int16)
    tb += (hs_lut[tgt] & ~ib_lut[prev]).to(torch.int16)
    return tb.to(torch.float64).sum()


def _reduce(ls, tc, bc, is_dist):
    if is_dist:
        for t in (ls, tc, bc):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (ls / tc).item()
    return vl, (vl / math.log(2)) * (tc.item() / bc.item())


def eval_val(args, model, rank, world_size, device, gas, val_tokens, bb, hs, ib):
    lbt = args.val_batch_size // (world_size * gas)
    lbs = lbt // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    if args.max_val_tokens > 0:
        total_seqs = max(min(total_seqs, args.max_val_tokens // args.train_seq_len), 1)
    ss, se = (total_seqs * rank) // world_size, (total_seqs * (rank + 1)) // world_size
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(ss, se, lbs):
            bse = min(bss + lbs, se)
            local = val_tokens[
                bss * args.train_seq_len : bse * args.train_seq_len + 1
            ].to(device=device, dtype=torch.int64)
            x, y = (
                local[:-1].reshape(-1, args.train_seq_len),
                local[1:].reshape(-1, args.train_seq_len),
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                bl = model(x, y).detach()
            n = float(y.numel())
            ls += bl.to(torch.float64) * n
            tc += n
            bc += _byte_count(x, y, bb, hs, ib)
    model.train()
    return _reduce(ls, tc, bc, dist.is_available() and dist.is_initialized())


@torch.no_grad()
def eval_val_sliding_window(
    args, base_model, rank, world_size, device, val_tokens, bb, hs, ib
):
    stride, seq_len = args.sliding_window_stride, args.train_seq_len
    base_model.eval()
    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    total = val_tokens.numel() - 1
    if args.max_val_tokens > 0:
        total = min(total, args.max_val_tokens)
    starts = list(range(0, total - seq_len + 1, stride))
    my_starts = starts[rank::world_size]
    WB = 16
    with torch.inference_mode():
        for bi in range(0, len(my_starts), WB):
            batch = my_starts[bi : bi + WB]
            xs = torch.stack([val_tokens[p : p + seq_len] for p in batch]).to(
                device, dtype=torch.int64
            )
            ys = torch.stack([val_tokens[p + 1 : p + seq_len + 1] for p in batch]).to(
                device, dtype=torch.int64
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(xs)
            ptl = F.cross_entropy(
                logits.float().reshape(-1, args.vocab_size),
                ys.reshape(-1),
                reduction="none",
            ).reshape(len(batch), seq_len)
            for i, pos in enumerate(batch):
                cs = 0 if pos == 0 else seq_len - stride
                ls += ptl[i, cs:].double().sum()
                tc += ptl[i, cs:].numel()
                bc += _byte_count(xs[i : i + 1, cs:], ys[i : i + 1, cs:], bb, hs, ib)
    base_model.train()
    return _reduce(ls, tc, bc, dist.is_available() and dist.is_initialized())


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────


def _load_shard(file: Path) -> Tensor:
    hb = 256 * 4
    hdr = np.fromfile(file, dtype="<i4", count=256)
    if hdr.size != 256 or int(hdr[0]) != 20240520 or int(hdr[1]) != 1:
        raise ValueError(f"Bad shard: {file}")
    n = int(hdr[2])
    if file.stat().st_size != hb + n * 2:
        raise ValueError(f"Size mismatch: {file}")
    return torch.from_numpy(
        np.fromfile(file, dtype="<u2", count=n, offset=hb).astype(np.uint16, copy=False)
    )


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.fi, self.pos = 0, 0
        self.tokens = _load_shard(self.files[0])

    def _adv(self):
        self.fi = (self.fi + 1) % len(self.files)
        self.tokens = _load_shard(self.files[self.fi])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks, rem = [], n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._adv()
                continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, gas):
        lt = global_tokens // (self.world_size * gas)
        span = lt + 1
        chunk = self.stream.take(span * self.world_size)
        local = chunk[self.rank * span : self.rank * span + span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(
            x,
            self.weight.to(x.dtype),
            self.bias.to(x.dtype) if self.bias is not None else None,
        )


def _ortho(w: Tensor, scale: float = 1.0):
    nn.init.orthogonal_(w, gain=scale)


def restore_fp32(module: nn.Module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (
                p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS)
            ) and p.dtype != torch.float32:
                p.data = p.data.float()


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)),
            persistent=False,
        )
        self._sl, self._cos, self._sin = 0, None, None

    def forward(self, sl, device, dtype):
        if self._cos is None or self._sl != sl or self._cos.device != device:
            t = torch.arange(sl, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos, self._sin = f.cos()[None, None], f.sin()[None, None]
            self._sl = sl
        return self._cos.to(dtype), self._sin.to(dtype)


def _rope(x, cos, sin):
    h = x.size(-1) // 2
    return torch.cat(
        (x[..., :h] * cos + x[..., h:] * sin, x[..., :h] * (-sin) + x[..., h:] * cos),
        dim=-1,
    )


# BigramHash
class BigramHashEmbedding(nn.Module):
    """Hash (prev_tok, cur_tok) pairs into bigram_buckets slots → project to model_dim."""

    PRIME = 1_000_003

    def __init__(self, num_buckets, bigram_dim, model_dim):
        super().__init__()
        self.nb = num_buckets
        self.table = nn.Embedding(num_buckets, bigram_dim)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        nn.init.normal_(self.table.weight, std=0.02)
        nn.init.zeros_(self.proj.weight)

    def forward(self, ids: Tensor) -> Tensor:
        B, T = ids.shape
        prev = torch.cat([ids.new_zeros(B, 1), ids[:, :-1]], dim=1)
        idx = (prev * self.PRIME + ids) % self.nb
        return self.proj(self.table(idx))


# MLA Attention
class MLAAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, kv_rank, rope_base, qk_gain_init):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.nh, self.nkv, self.hd = num_heads, num_kv_heads, dim // num_heads
        kv_dim = num_kv_heads * self.hd
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.kv_down = CastedLinear(dim, kv_rank, bias=False)
        self.k_up = CastedLinear(kv_rank, kv_dim, bias=False)
        self.v_up = CastedLinear(kv_rank, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.hd, base=rope_base)
        s = dim**-0.5
        for w in (self.c_q, self.kv_down, self.k_up, self.v_up):
            _ortho(w.weight, s)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q = self.c_q(x).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        lat = self.kv_down(x)
        k = self.k_up(lat).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_up(lat).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(
            y.transpose(1, 2).contiguous().reshape(B, T, self.nh * self.hd)
        )


# SmearGate MLP
class SmearGateMLP(nn.Module):
    """Gated MLP: proj(relu(gate(x))^2 * up(x)).  mlp_mult=3 → hidden=1536 @ d=512."""

    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = dim * mlp_mult
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.up = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        s = dim**-0.5
        _ortho(self.gate.weight, s)
        _ortho(self.up.weight, s)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.gate(x)).square() * self.up(x))


# Block
class Block(nn.Module):
    def __init__(
        self, dim, num_heads, num_kv_heads, kv_rank, mlp_mult, rope_base, qk_gain_init
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = MLAAttention(
            dim, num_heads, num_kv_heads, kv_rank, rope_base, qk_gain_init
        )
        self.mlp = SmearGateMLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack([torch.ones(dim), torch.zeros(dim)]).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None] * x + mix[1][None, None] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None] * self.mlp(self.mlp_norm(x))
        return x


# GPT
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        model_dim,
        num_heads,
        num_kv_heads,
        kv_rank,
        mlp_mult,
        tie_embeddings,
        tied_embed_init_std,
        logit_softcap,
        rope_base,
        qk_gain_init,
        bigram_buckets,
        bigram_dim,
    ):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_buckets, bigram_dim, model_dim)
        self.num_enc = num_layers // 2
        self.num_dec = num_layers - self.num_enc
        self.num_skips = min(self.num_enc, self.num_dec)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skips, model_dim, dtype=torch.float32)
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    kv_rank,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        nn.init.normal_(self.tok_emb.weight, std=tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def _backbone(self, ids: Tensor) -> Tensor:
        x = self.tok_emb(ids) + self.bigram(ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0, skips = x, []
        for i in range(self.num_enc):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_dec):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None] * skips.pop()
            x = self.blocks[self.num_enc + i](x, x0)
        return self.final_norm(x)

    def forward_logits(self, ids: Tensor) -> Tensor:
        x = self._backbone(ids)
        lp = (
            F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        )
        return self.logit_softcap * torch.tanh(lp / self.logit_softcap)

    def forward(self, ids: Tensor, targets: Tensor) -> Tensor:
        logits = self.forward_logits(ids)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────


def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    use_compile = bool(int(os.environ.get("USE_COMPILE", "1")))
    if use_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    gas = 8 // world_size
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    flash = sys.platform != "win32"
    enable_cudnn_sdp(False)
    enable_flash_sdp(flash)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(not flash)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log(msg, console=True):
        if not master:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log(code, console=False)
    log(
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout,
        console=False,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}"
        )

    ddir = Path(args.data_path).resolve()
    nshards = len(list(ddir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb, hs, ib = build_sentencepiece_luts(sp, args.vocab_size, device)

    log(f"val_bpb:enabled  tokenizer:{args.tokenizer_path}  compressor:{COMPRESSOR}")
    log(f"train_shards:{nshards}  val_tokens:{val_tokens.numel() - 1}")

    base_model = (
        GPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            kv_rank=args.kv_rank,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            bigram_buckets=args.bigram_buckets,
            bigram_dim=args.bigram_dim,
        )
        .to(device)
        .bfloat16()
    )
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_fp32(base_model)

    compiled = (
        torch.compile(base_model, dynamic=False, fullgraph=True)
        if use_compile
        else base_model
    )
    if not use_compile:
        log("[info] torch.compile disabled (USE_COMPILE=0)")
    model = (
        DDP(compiled, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled
    )

    # Optimizer groups
    block_np = list(base_model.blocks.named_parameters())
    matrix_p = [
        p
        for n, p in block_np
        if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)
    ]
    scalar_p = [
        p
        for n, p in block_np
        if p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_p.append(base_model.skip_weights)

    tok_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    bigram_params = list(base_model.bigram.parameters())

    opt_tok = torch.optim.AdamW(
        [
            {
                "params": [base_model.tok_emb.weight] + bigram_params,
                "lr": tok_lr,
                "base_lr": tok_lr,
            }
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    opt_muon = Muon(
        matrix_p,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        wd=args.muon_wd,
    )
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.AdamW(
        [{"params": scalar_p, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers = [opt_tok, opt_muon, opt_scalar]

    nparams = sum(p.numel() for p in base_model.parameters())
    log(
        f"params:{nparams}  layers:{args.num_layers}  dim:{args.model_dim}  "
        f"mlp_mult:{args.mlp_mult}  kv_rank:{args.kv_rank}  "
        f"bigram:{args.bigram_buckets}x{args.bigram_dim}"
    )
    log(
        f"muon_wd:{args.muon_wd}  momentum:{args.muon_momentum}  "
        f"warmdown:{args.warmdown_iters}  swa_frac:{args.swa_start_frac}  "
        f"flash:{flash}  compile:{use_compile}"
    )

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    mwms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if mwms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if ws <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem = max(mwms - elapsed_ms, 0.0)
        return rem / max(wd_ms, 1e-9) if rem <= wd_ms else 1.0

    def in_swa(step, elapsed_ms):
        m = lr_mul(step, elapsed_ms)
        return 0.0 < m < (1.0 - args.swa_start_frac + 1e-6)

    # Warmup
    if args.warmup_steps > 0:
        init_sd = {
            n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()
        }
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws_i in range(args.warmup_steps):
            zero_grad()
            for micro in range(gas):
                if distributed:
                    model.require_backward_grad_sync = micro == gas - 1
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, gas
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    loss = model(x, y)
                (loss / gas).backward()
            for o in optimizers:
                o.step()
            zero_grad()
            if (
                args.warmup_steps <= 20
                or (ws_i + 1) % 10 == 0
                or ws_i + 1 == args.warmup_steps
            ):
                log(f"warmup:{ws_i + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, st in zip(optimizers, init_opt, strict=True):
            o.load_state_dict(st)
        zero_grad()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device
        )

    # Main loop
    train_ms = 0.0
    stop_after: int | None = None
    swa_snaps: list[dict] = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last = step == args.iterations or (
            stop_after is not None and step >= stop_after
        )
        elapsed_ms = train_ms + 1000.0 * (time.perf_counter() - t0)

        if not last and args.val_loss_every > 0 and step % args.val_loss_every == 0:
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(
                args, model, rank, world_size, device, gas, val_tokens, bb, hs, ib
            )
            log(
                f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                f"swa:{len(swa_snaps)} train_time:{train_ms:.0f}ms step_avg:{train_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last:
            if stop_after is not None and step < args.iterations:
                log(f"stopping_early step:{step}")
            break

        scale = lr_mul(step, elapsed_ms)
        zero_grad()
        tl = torch.zeros((), device=device)
        for micro in range(gas):
            if distributed:
                model.require_backward_grad_sync = micro == gas - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, gas
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            tl += loss.detach()
            (loss / gas).backward()
        tl /= gas

        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        cur_mom = (
            1 - frac
        ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups:
            g["momentum"] = cur_mom
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers:
            o.step()
        zero_grad()

        step += 1
        approx = train_ms + 1000.0 * (time.perf_counter() - t0)

        # SWA snapshot collection
        if master and in_swa(step, approx) and step % args.swa_every == 0:
            swa_snaps.append(
                {
                    k: v.detach().cpu().clone()
                    for k, v in base_model.state_dict().items()
                }
            )

        if args.train_log_every > 0 and (
            step <= 10 or step % args.train_log_every == 0 or stop_after is not None
        ):
            log(
                f"step:{step}/{args.iterations} train_loss:{tl.item():.4f} "
                f"train_time:{approx:.0f}ms step_avg:{approx / step:.2f}ms"
            )

        reached = mwms is not None and approx >= mwms
        if distributed and mwms is not None:
            rc = torch.tensor(int(reached), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached = bool(rc.item())
        if stop_after is None and reached:
            stop_after = step

    log(f"peak_mem:{torch.cuda.max_memory_allocated() // 1024 // 1024}MiB")

    # Apply SWA
    if master and swa_snaps:
        log(f"\n[SWA] averaging {len(swa_snaps)} snapshots...")
        avg = {}
        for key in swa_snaps[0]:
            avg[key] = (
                torch.stack([s[key].float() for s in swa_snaps])
                .mean(0)
                .to(swa_snaps[0][key].dtype)
            )
        base_model.load_state_dict(avg, strict=True)
        log("[SWA] done.")
    elif master:
        log("[SWA] no snapshots collected.")

    # Final sliding-window eval (pre-quant)
    log("\n[final sliding_window eval] pre-quantization...")
    torch.cuda.synchronize()
    tsw = time.perf_counter()
    swl, swb = eval_val_sliding_window(
        args, base_model, rank, world_size, device, val_tokens, bb, hs, ib
    )
    torch.cuda.synchronize()
    log(
        f"pre_quant val_loss:{swl:.4f} val_bpb:{swb:.4f} eval_time:{1000.0 * (time.perf_counter() - tsw):.0f}ms"
    )

    # Serialize
    if master:
        nb = save_artifact(base_model, "final_model.ptz")
        cb = len(code.encode("utf-8"))
        total = nb + cb
        log(f"\nartifact:{nb}B  code:{cb}B  total:{total}B  ({total / 1e6:.3f}MB)")
        log("OK: within 16MB" if total <= 16_000_000 else "WARNING: EXCEEDS 16MB")

    # Roundtrip
    if distributed:
        dist.barrier()
    load_artifact("final_model.ptz", base_model)
    torch.cuda.synchronize()
    trt = time.perf_counter()
    rtl, rtb = eval_val_sliding_window(
        args, base_model, rank, world_size, device, val_tokens, bb, hs, ib
    )
    torch.cuda.synchronize()
    log(
        f"roundtrip val_loss:{rtl:.4f} val_bpb:{rtb:.4f} eval_time:{1000.0 * (time.perf_counter() - trt):.0f}ms"
    )
    log(f"roundtrip_exact val_loss:{rtl:.8f} val_bpb:{rtb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
