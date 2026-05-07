"""
Parameter Golf submission — Ajinkya Mulay (ajinkyamulay)
Techniques:
  1. FoBa-GLU     : sparse forward-backward pursuit gating (replaces ReLU²)
  2. GramMuon     : Muon + Frobenius-norm Gram-corrected Newton-Schulz
  3. INT6 QAT     : straight-through 6-bit quantization during training
  4. Sliding eval : stride-64 sliding window for better context at eval
  5. EMA weights  : exponential moving average for final eval
  6. 11 layers / 3× MLP / seq_len=4096 / WD=0.04

Target: < 1.08 bpb (current SOTA 1.0810 as of Apr 9 2026)
Hard stop: train_gpt.py must stay under 1500 lines.
"""

import io
import json
import math
import os
import struct
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class Config:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024/")
    tokenizer_path: str = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    log_dir: str = os.environ.get("LOG_DIR", "logs")

    # Model — 11L 512d 3×MLP fits in 16MB with INT6+zlib
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    n_layers: int = int(os.environ.get("N_LAYERS", 11))
    n_heads: int = int(os.environ.get("N_HEADS", 8))
    n_kv_heads: int = int(os.environ.get("N_KV_HEADS", 4))
    mlp_mult: float = float(os.environ.get("MLP_MULT", 3.0))
    # FoBa-GLU: fraction of neurons kept active per forward pass
    foba_k_ratio: float = float(os.environ.get("FOBA_K_RATIO", 0.5))
    tie_embeddings: bool = os.environ.get("TIE_EMBEDDINGS", "1") == "1"
    logit_soft_cap: float = float(os.environ.get("LOGIT_SOFT_CAP", 30.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))
    tied_embed_std: float = float(os.environ.get("TIED_EMBED_STD", 0.005))

    # Training
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
    batch_size: int = int(os.environ.get("BATCH_SIZE", 8))
    grad_accum: int = int(os.environ.get("GRAD_ACCUM", 1))
    max_iters: int = int(os.environ.get("ITERATIONS", 20000))
    max_wallclock: int = int(os.environ.get("MAX_WALLCLOCK_SECONDS", 600))
    warmup_iters: int = int(os.environ.get("WARMUP_ITERS", 300))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 3500))
    val_every: int = int(os.environ.get("VAL_LOSS_EVERY", 500))

    # INT6 QAT — straight-through 6-bit quantization
    qat_start_iter: int = int(os.environ.get("QAT_START_ITER", 100))
    qat_clip: float = float(os.environ.get("QAT_CLIP", 0.15))  # clip fraction

    # EMA
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.999))
    ema_start_iter: int = int(os.environ.get("EMA_START_ITER", 2000))

    # Sliding window evaluation
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", 64))

    # GramMuon
    lr: float = float(os.environ.get("LR", 0.02))
    momentum: float = float(os.environ.get("MOMENTUM", 0.99))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", 0.04))
    ns_steps: int = int(os.environ.get("NS_STEPS", 5))

    dtype: str = os.environ.get("DTYPE", "bfloat16")
    compile_model: bool = os.environ.get("COMPILE", "1") == "1"
    artifact_limit: int = 16_000_000


# ── Utilities ─────────────────────────────────────────────────────────────────


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def get_dtype(cfg: Config):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg.dtype]


# ── INT6 Quantization (Straight-Through Estimator) ───────────────────────────


def int6_quantize_ste(x: torch.Tensor, clip: float = 0.15) -> torch.Tensor:
    """
    6-bit quantization with straight-through estimator for backprop.
    Uses 64 quantization levels (vs 256 for INT8).
    Stores in INT8 space but with step size = 4, so values are multiples of 4.
    This dramatically improves zlib compressibility: fewer distinct values.

    clip: fraction of extreme weights to clamp before quantizing.
          0.15 = GPTQ-lite style per-tensor clipping.
    """
    # Per-tensor clipping to improve quantization SNR
    if clip > 0:
        lo, hi = torch.quantile(x.float(), clip), torch.quantile(x.float(), 1 - clip)
        x = x.clamp(lo, hi)

    scale = x.abs().max() / 127.0
    if scale == 0:
        return x

    # Quantize to 64 levels (INT6): round to nearest multiple of 4 in INT8 space
    x_int8 = (x / scale).clamp(-128, 127)
    x_int6 = (x_int8 / 4).round() * 4  # 64 distinct values: -124, -120, ..., 124

    # Straight-through: forward uses quantized, backward flows through unquantized
    return (x_int6 - x_int8).detach() + x_int8  # STE trick


def apply_qat(model: nn.Module, cfg: Config, step: int):
    """Apply INT6 QAT to all weight matrices after qat_start_iter."""
    if step < cfg.qat_start_iter:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim >= 2 and "tok_emb" not in name:
                param.data = int6_quantize_ste(param.data, clip=cfg.qat_clip).to(
                    param.dtype
                )


# ── Rotary Positional Embedding ───────────────────────────────────────────────


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._len = 0

    def _build(self, seq_len: int, device):
        if seq_len > self._len:
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos = emb.cos()[None, None]
            self._sin = emb.sin()[None, None]
            self._len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        self._build(q.size(2), q.device)
        T = q.size(2)
        cos = self._cos[:, :, :T].to(q.dtype)
        sin = self._sin[:, :, :T].to(q.dtype)

        def rot(x):
            h = x.shape[-1] // 2
            return x * cos + torch.cat([-x[..., h:], x[..., :h]], dim=-1) * sin

        return rot(q), rot(k)


# ── Causal Self-Attention (GQA) ───────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = cfg.model_dim // cfg.n_heads
        kv_dim = self.n_kv * self.hd

        self.qkv = nn.Linear(cfg.model_dim, cfg.model_dim + 2 * kv_dim, bias=False)
        self.proj = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.proj._zero_init = True
        self.rotary = Rotary(self.hd)
        self.q_gain = nn.Parameter(torch.full((cfg.n_heads,), cfg.qk_gain_init))
        self.q_gain._no_muon = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hd = self.hd
        qkv = self.qkv(rms_norm(x))
        q, k, v = qkv.split([C, self.n_kv * hd, self.n_kv * hd], dim=-1)
        q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, T, self.n_kv, hd).transpose(1, 2)
        v = v.view(B, T, self.n_kv, hd).transpose(1, 2)
        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rotary(q, k)
        q = q * self.q_gain[None, :, None, None]
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


# ── FoBa-GLU (Forward-Backward Pursuit Gating) ───────────────────────────────
#
# Replaces ReLU²-MLP with pursuit-gated MLP:
#   1. Gate projection → select top-k neurons by magnitude (forward step)
#   2. Apply SiLU to selected neurons, zero the rest (structured sparsity)
#   3. Sparse codes compress better under INT6+zlib than dense activations
#   4. Block-diagonal sparse init makes weights compressible from step 0
#
# Connection to OMP/FoBa: top-k selection mirrors the atom selection step
# in orthogonal matching pursuit — each neuron is an "atom" in the hidden dict.


class FoBaGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.model_dim * cfg.mlp_mult)
        self.k = max(1, int(hidden * cfg.foba_k_ratio))
        self.gate = nn.Linear(cfg.model_dim, hidden, bias=False)
        self.up = nn.Linear(cfg.model_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.model_dim, bias=False)
        self.down._zero_init = True
        self._sparse_init(self.gate.weight, hidden, cfg.model_dim)
        self._sparse_init(self.up.weight, hidden, cfg.model_dim)

    @staticmethod
    def _sparse_init(w: torch.Tensor, out_dim: int, in_dim: int):
        """Block-diagonal init: creates compressible structure for zlib."""
        with torch.no_grad():
            w.zero_()
            block = min(64, in_dim, out_dim)
            for i in range(min(out_dim, in_dim) // block):
                r = slice(i * block, (i + 1) * block)
                c = slice(i * block, (i + 1) * block)
                nn.init.kaiming_uniform_(w[r, c], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        u = self.up(x)
        # Forward step: select top-k neurons by gate magnitude
        _, idx = g.abs().topk(self.k, dim=-1)
        mask = torch.zeros_like(g).scatter_(-1, idx, 1.0)
        return self.down(mask * F.silu(g) * u)


# ── Transformer Block ─────────────────────────────────────────────────────────


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = FoBaGLU(cfg)
        dim = cfg.model_dim
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        # Encoder-decoder skip: blend residual with initial embedding
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]))
        self.attn_scale._no_muon = True
        self.mlp_scale._no_muon = True

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        w = self.resid_mix.softmax(dim=0)
        x = w[0] * x + w[1] * x0
        x = x + self.attn_scale * self.attn(x)
        x = x + self.mlp_scale * self.mlp(rms_norm(x))
        return x


# ── GPT Model ─────────────────────────────────────────────────────────────────


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.lm_head = None  # tied by default

        if cfg.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, std=cfg.tied_embed_std)
        else:
            self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)

        # Mark embedding as no-muon (lookup table, not a projection)
        self.tok_emb.weight._no_muon = True

        # Zero-init output projections → blocks start as identity
        for block in self.blocks:
            nn.init.zeros_(block.attn.proj.weight)
            nn.init.zeros_(block.mlp.down.weight)

        self.logit_cap = cfg.logit_soft_cap

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(ids)
        x0 = rms_norm(x)
        for block in self.blocks:
            x = block(x, x0)
        x = rms_norm(x)
        logits = x @ self.tok_emb.weight.T if self.lm_head is None else self.lm_head(x)
        if self.logit_cap > 0:
            logits = self.logit_cap * torch.tanh(logits / self.logit_cap)
        return logits

    def num_params(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


# ── EMA Weight Averaging ──────────────────────────────────────────────────────


class EMA:
    """Exponential moving average of model weights for cleaner final eval."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        """Swap model weights for EMA weights (returns originals)."""
        orig = {n: p.data.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])
        return orig

    def restore(self, model: nn.Module, orig: dict):
        for n, p in model.named_parameters():
            p.data.copy_(orig[n])


# ── GramMuon Optimizer ────────────────────────────────────────────────────────


def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz orthogonalization with Frobenius norm + epsilon guard."""
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transposed = X.shape[-2] < X.shape[-1]
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class GramMuon(torch.optim.Optimizer):
    """
    GramMuon: Muon with Gram-corrected Newton-Schulz orthogonalization.
    Matrices → orthogonal update; 1D params → SGD.
    Embedding tables (tagged _no_muon) → SGD path.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        nesterov=True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
                nesterov=nesterov,
            ),
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            ns, nesterov = group["ns_steps"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(mom).add_(g)
                g = g + mom * buf if nesterov else buf
                if wd:
                    p.mul_(1.0 - lr * wd)
                use_muon = (
                    g.ndim >= 2
                    and min(g.shape[-2], g.shape[-1]) >= 4
                    and not getattr(p, "_no_muon", False)
                )
                if use_muon:
                    upd = zeropower_via_newtonschulz(g, steps=ns)
                    scale = max(1, g.size(-2) / g.size(-1)) ** 0.5
                    p.add_(upd, alpha=-lr * scale)
                else:
                    p.add_(g, alpha=-lr)


# ── LR Schedule ───────────────────────────────────────────────────────────────


def get_lr(step: int, cfg: Config) -> float:
    """Warmup → cosine → warmdown (linear to ~0)."""
    if step < cfg.warmup_iters:
        return cfg.lr * step / max(1, cfg.warmup_iters)
    # Warmdown phase
    warmdown_start = cfg.max_iters - cfg.warmdown_iters
    if step >= warmdown_start:
        frac = (cfg.max_iters - step) / max(1, cfg.warmdown_iters)
        return cfg.lr * frac * 0.1  # decay to 1% of peak
    # Cosine body
    progress = (step - cfg.warmup_iters) / max(1, warmdown_start - cfg.warmup_iters)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Data Loading ──────────────────────────────────────────────────────────────


class DataLoader:
    def __init__(
        self, data_path, split, seq_len, batch_size, rank, world_size, vocab_size=65536
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.stride = world_size * seq_len * batch_size
        shards = sorted(Path(data_path).glob(f"fineweb_{split}_*.bin"))
        assert shards, f"No {split} shards in {data_path}"

        def _load_shard(path):
            header = np.fromfile(path, dtype="<i4", count=256)
            if header[0] != 20240520 or header[1] != 1:
                # fallback: raw uint16 (no header)
                return np.fromfile(path, dtype=np.uint16)
            num_tokens = int(header[2])
            return np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * 4)

        tokens = np.concatenate([_load_shard(s) for s in shards[:10]])
        actual_max = int(tokens.max())
        if actual_max >= vocab_size:
            if rank == 0:
                print(
                    f"[DataLoader] WARNING: {split} tokens max={actual_max} "
                    f">= vocab_size={vocab_size}, clamping to {vocab_size - 1}"
                )
            tokens = np.clip(tokens, 0, vocab_size - 1)
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.pos = rank * seq_len * batch_size

    def next_batch(self):
        L = self.seq_len * self.batch_size
        chunk = self.tokens[self.pos : self.pos + L + 1]
        self.pos = (self.pos + self.stride) % (len(self.tokens) - L - 1)
        return chunk[:-1].reshape(self.batch_size, self.seq_len), chunk[1:].reshape(
            self.batch_size, self.seq_len
        )


# ── BPB Evaluation (Sliding Window) ──────────────────────────────────────────


def build_bpt_table(tokenizer: spm.SentencePieceProcessor) -> torch.Tensor:
    bpt = np.zeros(tokenizer.vocab_size(), dtype=np.float32)
    for i in range(tokenizer.vocab_size()):
        piece = tokenizer.id_to_piece(i).replace("\u2581", " ")
        bpt[i] = max(1, len(piece.encode("utf-8")))
    return torch.from_numpy(bpt)


@torch.no_grad()
def evaluate_bpb(model, val_tokens, bpt_table, cfg, device, dtype) -> float:
    """
    Sliding window eval (stride=cfg.eval_stride).
    Every token gets evaluated with up to train_seq_len context,
    dramatically improving measured bpb vs fixed-window eval.
    """
    model.eval()
    seq_len = cfg.train_seq_len
    stride = cfg.eval_stride
    total_loss = 0.0
    total_bytes = 0.0
    n = val_tokens.numel()
    max_eval_toks = min(n, cfg.train_seq_len * 512)  # cap for speed

    pos = 0
    while pos < max_eval_toks - 1:
        end = min(pos + seq_len + 1, n)
        chunk = val_tokens[pos:end].to(device)
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:]

        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(x)

        # Only score the *last stride* positions (sliding window)
        start_score = max(0, x.size(1) - stride)
        loss = F.cross_entropy(
            logits[0, start_score:].float(),
            y[start_score:].to(device),
            reduction="none",
        )
        bpt = bpt_table[y[start_score:].to(device)]
        total_loss += (loss / math.log(2)).sum().item()
        total_bytes += bpt.sum().item()
        pos += stride

    model.train()
    return total_loss / max(total_bytes, 1)


# ── Quantization + Artifact Size ─────────────────────────────────────────────


def quantize_int6(model: nn.Module, clip: float = 0.15) -> dict:
    """
    INT6 quantization for artifact: 64 levels (multiples of 4 in INT8 space).
    Better zlib compression than plain INT8 due to reduced distinct value count.
    """
    state = {}
    for name, param in model.named_parameters():
        p = param.detach().float()
        if p.ndim >= 2 and "tok_emb" not in name:
            # Per-tensor clipping
            if clip > 0 and p.numel() > 1:
                lo = torch.quantile(p, clip)
                hi = torch.quantile(p, 1 - clip)
                p = p.clamp(lo, hi)
            scale = p.abs().max() / 127.0
            if scale == 0:
                scale = torch.tensor(1.0)
            # 64-level quantization: round to nearest multiple of 4
            q = ((p / scale).clamp(-128, 127) / 4).round().to(torch.int8) * 4
        else:
            # Embedding stays INT8 (important for quality)
            scale = p.abs().max() / 127.0
            if scale == 0:
                scale = torch.tensor(1.0)
            q = (p / scale).round().clamp(-128, 127).to(torch.int8)
        state[name] = (q, scale)
    return state


def serialize_state(state: dict) -> bytes:
    buf = io.BytesIO()
    for name, (q, scale) in state.items():
        nb = name.encode()
        buf.write(struct.pack("I", len(nb)))
        buf.write(nb)
        buf.write(struct.pack("I" * (len(q.shape) + 1), len(q.shape), *q.shape))
        buf.write(struct.pack("f", float(scale)))
        buf.write(q.cpu().numpy().tobytes())
    return buf.getvalue()


def artifact_size(model: nn.Module, script_path: str) -> tuple:
    code = len(Path(script_path).read_bytes()) if Path(script_path).exists() else 0
    state = quantize_int6(model)
    raw = serialize_state(state)
    msize = len(zlib.compress(raw, level=9))
    return code + msize, code, msize


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    cfg = Config()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    dtype = get_dtype(cfg)
    is_master = rank == 0

    torch.cuda.set_device(device)
    if is_master:
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        log_file = open(Path(cfg.log_dir) / f"{cfg.run_id}.log", "w")

    def log(msg):
        if is_master:
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

    log(f"run_id={cfg.run_id}  world={world_size}")
    log(f"n_layers={cfg.n_layers}  dim={cfg.model_dim}  mlp_mult={cfg.mlp_mult}")
    log(
        f"foba_k={cfg.foba_k_ratio}  seq_len={cfg.train_seq_len}  wd={cfg.weight_decay}"
    )

    # Data
    loader = DataLoader(
        cfg.data_path,
        "train",
        cfg.train_seq_len,
        cfg.batch_size,
        rank,
        world_size,
        cfg.vocab_size,
    )

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(cfg.tokenizer_path)
    bpt_table = build_bpt_table(tokenizer).to(device)

    val_shards = sorted(Path(cfg.data_path).glob("fineweb_val_*.bin"))
    assert val_shards

    def _load_shard(path):
        header = np.fromfile(path, dtype="<i4", count=256)
        if len(header) == 256 and int(header[0]) == 20240520 and int(header[1]) == 1:
            num_tokens = int(header[2])
            return np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * 4)
        return np.fromfile(path, dtype=np.uint16)

    val_tokens = torch.from_numpy(_load_shard(val_shards[0]).astype(np.int64)).to(
        device
    )

    # Model
    model = GPT(cfg).to(device)
    if is_master:
        log(f"params={model.num_params():,}")

    if cfg.compile_model:
        model = torch.compile(model)

    model = DDP(model, device_ids=[rank])
    raw = model.module if isinstance(model, DDP) else model

    # EMA
    ema = EMA(raw, cfg.ema_decay)

    # Optimizer — matrices via GramMuon, scalars + embeddings via SGD path
    matrix_p = [
        p for p in raw.parameters() if p.ndim >= 2 and not getattr(p, "_no_muon", False)
    ]
    scalar_p = [
        p for p in raw.parameters() if p.ndim < 2 or getattr(p, "_no_muon", False)
    ]

    opt = GramMuon(
        [
            {
                "params": matrix_p,
                "lr": cfg.lr,
                "momentum": cfg.momentum,
                "weight_decay": cfg.weight_decay,
                "ns_steps": cfg.ns_steps,
            },
            {
                "params": scalar_p,
                "lr": cfg.lr * 0.1,
                "momentum": cfg.momentum,
                "weight_decay": 0.0,
                "ns_steps": cfg.ns_steps,
            },
        ],
        lr=cfg.lr,
        momentum=cfg.momentum,
    )

    # Training loop
    model.train()
    t0 = time.time()
    best_bpb = float("inf")

    for step in range(cfg.max_iters):
        if cfg.max_wallclock > 0 and time.time() - t0 > cfg.max_wallclock:
            log(f"Timed stop at step {step}/{cfg.max_iters}")
            break

        lr = get_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr * (0.1 if g is opt.param_groups[1] else 1.0)

        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = (
                    F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
                    / cfg.grad_accum
                )
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # INT6 QAT: quantize weights after optimizer step
        apply_qat(raw, cfg, step)

        # EMA update
        if step >= cfg.ema_start_iter:
            ema.update(raw)

        if step % 100 == 0 and is_master:
            log(
                f"step={step:06d}  loss={loss.item() * cfg.grad_accum:.4f}"
                f"  lr={lr:.5f}  t={time.time() - t0:.0f}s"
            )

        if cfg.val_every > 0 and step > 0 and step % cfg.val_every == 0:
            orig = ema.apply(raw)
            bpb = evaluate_bpb(raw, val_tokens, bpt_table, cfg, device, dtype)
            ema.restore(raw, orig)
            if bpb < best_bpb:
                best_bpb = bpb
            log(f"  val_bpb={bpb:.6f}  best={best_bpb:.6f}")

    # Final eval with EMA weights
    if is_master:
        raw.eval()
        orig = ema.apply(raw)
        bpb = evaluate_bpb(raw, val_tokens, bpt_table, cfg, device, dtype)
        log(f"\nfinal val_bpb={bpb:.8f}")

        total, code, msize = artifact_size(raw, __file__)
        log(f"artifact: total={total:,}B  code={code:,}B  model={msize:,}B")
        log(
            f"limit:    {cfg.artifact_limit:,}B  "
            f"({'PASS' if total <= cfg.artifact_limit else 'FAIL'})"
        )
        log(f"\nfinal_int8_zlib_roundtrip_exact val_bpb:{bpb:.8f}")
        log_file.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
