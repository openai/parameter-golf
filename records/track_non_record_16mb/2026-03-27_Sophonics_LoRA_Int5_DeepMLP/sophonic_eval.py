"""
sophonic_eval.py — Run Sophonic Quantization ablation on a trained model.

This script is SEPARATE from training. It:
1. Loads a model saved by train_gpt_mlx.py (or train_gpt.py)
2. Runs the standard int8 quantization eval
3. Runs the Sophonic ablation (int5 base + low-rank residuals, multiple routing strategies)
4. Prints the comparison table

Runs on CPU or MPS — no CUDA required.

Usage:
    python3 sophonic_eval.py                          # Uses defaults
    python3 sophonic_eval.py --model final_model.pt   # Specify model path
    SOPHONIC_RANK=8 python3 sophonic_eval.py          # Override rank
"""

from __future__ import annotations

import glob
import io
import math
import os
import random
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class EvalConfig:
    model_path: str = os.environ.get("MODEL_PATH", "")
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    val_files: str = os.path.join(
        os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"),
        "fineweb_val_*.bin",
    )
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", "65536"))

    # Model shape (must match training)
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", "1024"))
    num_layers: int = int(os.environ.get("NUM_LAYERS", "9"))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", "4"))
    model_dim: int = int(os.environ.get("MODEL_DIM", "512"))
    num_heads: int = int(os.environ.get("NUM_HEADS", "8"))
    mlp_mult: int = int(os.environ.get("MLP_MULT", "2"))
    rope_base: float = float(os.environ.get("ROPE_BASE", "10000.0"))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", "30.0"))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", "1.5"))


@dataclass
class SophonicConfig:
    base_bits: int = int(os.environ.get("SOPHONIC_BASE_BITS", "5"))
    high_bits: int = int(os.environ.get("SOPHONIC_HIGH_BITS", "8"))
    rank: int = int(os.environ.get("SOPHONIC_RANK", "4"))
    k: int = int(os.environ.get("SOPHONIC_K", "4"))


# ─────────────────────────────────────────────
# MODEL (same architecture as train_gpt.py / train_gpt_mlx.py)
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: tuple[int, Tensor, Tensor] | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._cache is None or self._cache[0] != seq_len:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len, freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :])
        return self._cache[1].to(dtype=dtype), self._cache[2].to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        # GQA: expand KV heads to match Q heads
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(bsz, self.num_heads, seqlen, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, rep, -1, -1).reshape(bsz, self.num_heads, seqlen, self.head_dim)

        # Manual causal attention (no CUDA-only flash attention needed)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        self.fc = CastedLinear(dim, mlp_mult * dim, bias=False)
        self.proj = CastedLinear(mlp_mult * dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float):
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
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: EvalConfig):
        super().__init__()
        self.logit_softcap = cfg.logit_softcap
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.num_encoder_layers = cfg.num_layers // 2
        self.num_decoder_layers = cfg.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, cfg.model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(cfg.model_dim, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_mult,
                  cfg.rope_base, cfg.qk_gain_init)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = RMSNorm()

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits = self.logit_softcap * torch.tanh(F.linear(x, self.tok_emb.weight) / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


# ─────────────────────────────────────────────
# BPB EVALUATION
# ─────────────────────────────────────────────

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size())
    sz = max(sp_vs, vocab_size)
    base_bytes = np.zeros(sz, dtype=np.int16)
    has_space = np.zeros(sz, dtype=np.bool_)
    is_boundary = np.ones(sz, dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary, dtype=torch.bool, device=device),
    )


@torch.inference_mode()
def eval_bpb(model, val_tokens, seq_len, batch_size, base_bytes_lut, has_space_lut, is_boundary_lut, device):
    batch_seqs = max(batch_size // seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0

    model.eval()
    max_seqs = int(os.environ.get("SOPHONIC_MAX_SEQS", "0"))
    eval_seqs = min(total_seqs, max_seqs) if max_seqs > 0 else total_seqs
    for start in range(0, eval_seqs, batch_seqs):
        end = min(start + batch_seqs, total_seqs)
        raw_s = start * seq_len
        raw_e = end * seq_len + 1
        local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        loss = model(x, y).item()
        n = float(y.numel())
        loss_sum += loss * n
        token_count += n
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        tb = base_bytes_lut[tgt_ids].to(torch.int16)
        tb += (has_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).to(torch.int16)
        byte_count += tb.to(torch.float32).sum().item()

        if (start // batch_seqs) % 20 == 0:
            done_pct = 100 * end / eval_seqs
            print(f"    eval: {done_pct:.0f}% ({end}/{eval_seqs} seqs)", end="\r")

    val_loss = loss_sum / token_count
    bpb = (val_loss / math.log(2.0)) * (token_count / byte_count)
    print(f"    eval: 100% ({eval_seqs}/{eval_seqs} seqs)      ")
    return val_loss, bpb


# ─────────────────────────────────────────────
# QUANTIZATION UTILITIES
# ─────────────────────────────────────────────

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")
SMALL_TENSOR_MAX = 65_536


def quant_per_row(t: Tensor, bits: int, clip_q: float = 0.9999984) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    qmax = (1 << (bits - 1)) - 1
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), clip_q, dim=1) if t32.numel() else torch.empty(t32.shape[0])
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / qmax).clamp_min(1.0 / qmax)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax).to(torch.int8)
        return q, scale.to(torch.float16)
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / qmax if clip_abs > 0 else 1.0)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequant(q: Tensor, scale: Tensor, dtype=torch.float32) -> Tensor:
    if scale.ndim > 0:
        return (q.float() * scale.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype)
    return (q.float() * float(scale.item())).to(dtype)


def int8_roundtrip(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Standard int8 quantize → dequantize roundtrip."""
    out = {}
    for name, t in state_dict.items():
        t = t.detach().cpu()
        if not t.is_floating_point() or t.numel() <= SMALL_TENSOR_MAX:
            out[name] = t
            continue
        q, s = quant_per_row(t, 8)
        out[name] = dequant(q, s, t.dtype)
    return out


# ─────────────────────────────────────────────
# SOPHONIC CORE
# ─────────────────────────────────────────────

def compute_residual_lr(weight: Tensor, base_bits: int, high_bits: int, rank: int):
    """Compute rank-R SVD of Q_high(W) - Q_low(W)."""
    q_lo, s_lo = quant_per_row(weight, base_bits)
    q_hi, s_hi = quant_per_row(weight, high_bits)
    residual = (dequant(q_hi, s_hi) - dequant(q_lo, s_lo)).float()
    if residual.ndim != 2 or min(residual.shape) < rank:
        return None, None, {"skipped": True}
    U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
    U_r = (U[:, :rank] * S[:rank].unsqueeze(0)).half()
    V_r = Vh[:rank, :].T.half()
    total = (residual ** 2).sum().item()
    error = ((residual - U_r.float() @ V_r.float().T) ** 2).sum().item()
    return U_r, V_r, {"energy_pct": 100.0 * (1.0 - error / max(total, 1e-12))}


def sophonic_quantize(state_dict: dict[str, Tensor], cfg: SophonicConfig):
    """Returns (base_sd, residuals, stats) — base at cfg.base_bits, residuals as low-rank."""
    base_sd = {}
    residuals = {}
    energies = []
    base_bytes = 0
    res_bytes = 0

    for name, t in state_dict.items():
        t = t.detach().cpu()
        if not t.is_floating_point() or t.numel() <= SMALL_TENSOR_MAX:
            base_sd[name] = t
            continue
        # Base quantize → dequantize
        q, s = quant_per_row(t, cfg.base_bits)
        base_sd[name] = dequant(q, s, t.dtype)
        base_bytes += q.numel() + s.numel() * 2

        # Low-rank residual
        U_r, V_r, info = compute_residual_lr(t, cfg.base_bits, cfg.high_bits, cfg.rank)
        if U_r is not None:
            residuals[name] = (U_r, V_r)
            res_bytes += (U_r.numel() + V_r.numel()) * 2
            energies.append(info["energy_pct"])

    stats = {
        "base_bytes": base_bytes,
        "residual_bytes": res_bytes,
        "n_residuals": len(residuals),
        "mean_energy_pct": sum(energies) / max(len(energies), 1),
    }
    return base_sd, residuals, stats


def apply_corrections(base_sd: dict, residuals: dict, active: set[str] | None) -> dict[str, Tensor]:
    """Return state dict with corrections applied to active layers."""
    out = {}
    for name, t in base_sd.items():
        if active and name in active and name in residuals:
            U_r, V_r = residuals[name]
            out[name] = (t.float() + U_r.float() @ V_r.float().T).to(t.dtype)
        else:
            out[name] = t
    return out


# ─────────────────────────────────────────────
# LAYER SELECTION STRATEGIES
# ─────────────────────────────────────────────

def _layer_map(residuals: dict, num_layers: int) -> dict[str, int]:
    m = {}
    for name in residuals:
        for i in range(num_layers):
            if f"blocks.{i}." in name:
                m[name] = i
                break
    return m


def select_static(residuals, k, num_layers):
    lm = _layer_map(residuals, num_layers)
    deepest = sorted(set(lm.values()), reverse=True)[:k]
    return {n for n, i in lm.items() if i in deepest}


def select_by_norm(residuals, k, num_layers):
    lm = _layer_map(residuals, num_layers)
    layer_norms = {}
    layer_names = {}
    for name, (U, V) in residuals.items():
        i = lm.get(name)
        if i is not None:
            layer_norms[i] = layer_norms.get(i, 0.0) + (U.float() @ V.float().T).norm().item()
            layer_names.setdefault(i, []).append(name)
    top = sorted(layer_norms, key=layer_norms.get, reverse=True)[:k]
    out = set()
    for l in top:
        out.update(layer_names.get(l, []))
    return out


def select_random(residuals, k, num_layers):
    lm = _layer_map(residuals, num_layers)
    all_l = sorted(set(lm.values()))
    chosen = set(random.sample(all_l, min(k, len(all_l))))
    return {n for n, i in lm.items() if i in chosen}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def find_model_file() -> str:
    """Find the most recent saved model."""
    candidates = [
        os.environ.get("MODEL_PATH", ""),
        "final_model.pt",
        # MLX saves as <run_id>_mlx_model.npz — we can't load that directly.
        # Look for any .pt file in logs/
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c

    # Search logs/ for .pt files
    pts = sorted(glob.glob("logs/*_model.pt"), key=os.path.getmtime, reverse=True)
    if pts:
        return pts[0]

    # Search for any .pt
    pts = sorted(glob.glob("*.pt"), key=os.path.getmtime, reverse=True)
    if pts:
        return pts[0]

    raise FileNotFoundError(
        "No model file found. Set MODEL_PATH or ensure final_model.pt exists.\n"
        "If you trained with MLX, you need to save as .pt — see instructions below.\n"
        "After MLX training, add this to save a PyTorch-compatible checkpoint:\n"
        "  import torch, numpy as np\n"
        "  data = dict(np.load('your_model.npz'))\n"
        "  sd = {k: torch.from_numpy(v) for k, v in data.items()}\n"
        "  torch.save(sd, 'final_model.pt')\n"
    )


def main():
    cfg = EvalConfig()
    scfg = SophonicConfig()

    # ── Device selection ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Find and load model ──
    model_path = find_model_file()
    print(f"Loading model: {model_path}")

    model = GPT(cfg).to(device).float()
    raw_sd = torch.load(model_path, map_location="cpu", weights_only=True)

    # Handle both full state dicts and bare weight dicts
    if any(k.startswith("module.") for k in raw_sd):
        raw_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(raw_sd, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ── Load validation data ──
    print(f"Loading validation tokens: {cfg.val_files}")
    val_tokens = load_validation_tokens(cfg.val_files, cfg.seq_len)
    print(f"Validation tokens: {val_tokens.numel() - 1:,}")

    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    base_bytes_lut, has_space_lut, is_boundary_lut = build_sentencepiece_luts(sp, cfg.vocab_size, device)

    def _eval(label: str, sd: dict[str, Tensor]) -> float:
        model.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=False)
        t0 = time.time()
        val_loss, bpb = eval_bpb(model, val_tokens, cfg.seq_len, cfg.val_batch_size,
                                  base_bytes_lut, has_space_lut, is_boundary_lut, device)
        dt = time.time() - t0
        print(f"  {label:<42} val_bpb={bpb:.4f}  val_loss={val_loss:.4f}  ({dt:.0f}s)")
        return bpb

    # ── Original (no quantization) ──
    print(f"\n{'='*60}")
    print("EVALUATION SUITE")
    print(f"{'='*60}")

    bpb_orig = _eval("Original (fp32, no quantization)", raw_sd)

    # ── Uniform int8 ──
    int8_sd = int8_roundtrip(raw_sd)
    bpb_int8 = _eval("Uniform int8 (competition baseline)", int8_sd)

    # ── Sophonic ──
    print(f"\n--- Sophonic: int{scfg.base_bits} base + rank-{scfg.rank} ---")
    base_sd, residuals, stats = sophonic_quantize(raw_sd, scfg)
    print(f"  Base: {stats['base_bytes']/1e6:.2f} MB | Residuals: {stats['residual_bytes']/1e6:.2f} MB "
          f"({stats['n_residuals']} matrices)")
    print(f"  Mean energy captured: {stats['mean_energy_pct']:.1f}%")

    bpb_base = _eval(f"int{scfg.base_bits} base only", base_sd)
    bpb_all = _eval(f"int{scfg.base_bits} + ALL rank-{scfg.rank} corrections",
                     apply_corrections(base_sd, residuals, set(residuals.keys())))
    bpb_static = _eval(f"int{scfg.base_bits} + static top-{scfg.k} deepest",
                        apply_corrections(base_sd, residuals, select_static(residuals, scfg.k, cfg.num_layers)))
    bpb_norm = _eval(f"int{scfg.base_bits} + residual-norm top-{scfg.k}",
                      apply_corrections(base_sd, residuals, select_by_norm(residuals, scfg.k, cfg.num_layers)))
    bpb_rand = _eval(f"int{scfg.base_bits} + random {scfg.k} layers",
                      apply_corrections(base_sd, residuals, select_random(residuals, scfg.k, cfg.num_layers)))

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<42} {'BPB':>8}  {'Δ vs int8':>10}")
    print(f"  {'-'*42} {'-'*8}  {'-'*10}")
    rows = [
        ("Original fp32", bpb_orig),
        ("Uniform int8", bpb_int8),
        (f"int{scfg.base_bits} base only", bpb_base),
        (f"int{scfg.base_bits} + ALL rank-{scfg.rank}", bpb_all),
        (f"int{scfg.base_bits} + static top-{scfg.k}", bpb_static),
        (f"int{scfg.base_bits} + norm top-{scfg.k}", bpb_norm),
        (f"int{scfg.base_bits} + random {scfg.k}", bpb_rand),
    ]
    for label, bpb in rows:
        delta = f"{bpb - bpb_int8:+.4f}" if label != "Uniform int8" else "—"
        print(f"  {label:<42} {bpb:>8.4f}  {delta:>10}")

    damage = bpb_base - bpb_int8
    recovery = bpb_base - bpb_all if bpb_all < bpb_base else 0.0
    print(f"\n  int{scfg.base_bits} damage vs int8:  {damage:+.4f} BPB")
    print(f"  Rank-{scfg.rank} recovery:      {recovery:+.4f} BPB ({100*recovery/max(damage,1e-8):.0f}% of damage)")
    print(f"  Net vs int8:          {bpb_all - bpb_int8:+.4f} BPB")

    if bpb_all < bpb_int8:
        print("\n  ✅ POSITIVE: Sophonic corrections beat uniform int8!")
    elif recovery > 0:
        print("\n  ⚠️  PARTIAL: Corrections recover some damage. Try higher rank or int6 base.")
    else:
        print("\n  ❌ NEGATIVE: Corrections don't help at this rank.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
