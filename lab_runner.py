from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import train_gpt as baseline


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _format_seconds(seconds: float) -> str:
    if seconds <= 0:
        return "00:00:00"
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _boolify(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_kv_overrides(items: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set '{item}', expected key=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if raw.lower() in {"true", "false"}:
            out[key] = _boolify(raw)
            continue
        try:
            if "." in raw:
                out[key] = float(raw)
            else:
                out[key] = int(raw)
            continue
        except ValueError:
            out[key] = raw
    return out


def _config_id(cfg: Dict[str, Any]) -> str:
    packed = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return str(abs(hash(packed)))[:10]


class RMSNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),))


class HashEmbedding(nn.Module):
    """Hash embedding variants for ablation: gated, dual_concat, single, off."""

    def __init__(
        self,
        vocab_size: int,
        bigram_vocab_size: int,
        bigram_dim: int,
        model_dim: int,
        variant: str,
    ):
        super().__init__()
        self.variant = variant
        self.bigram_vocab_size = bigram_vocab_size
        self.embed1 = nn.Embedding(bigram_vocab_size, bigram_dim)
        self.embed2 = nn.Embedding(bigram_vocab_size, bigram_dim)
        if variant == "single":
            self.proj = nn.Linear(bigram_dim, model_dim, bias=False)
        elif variant == "off":
            self.proj = None
        else:
            self.proj = nn.Linear(2 * bigram_dim, model_dim, bias=False)
        self.gate = nn.Parameter(torch.ones(1, 1, 1) * 0.5)

    def _h1(self, prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        return (prev * 1024 + cur) % self.bigram_vocab_size

    def _h2(self, prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        return (prev + 31 * cur) % self.bigram_vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "off":
            return torch.zeros(x.size(0), x.size(1), self.embed1.embedding_dim * 0 + 1, device=x.device)

        prev = F.pad(x[:, :-1], (1, 0), value=0)
        h1 = self._h1(prev, x)
        e1 = self.embed1(h1)

        if self.variant == "single":
            return self.proj(e1)

        h2 = self._h2(prev, x)
        e2 = self.embed2(h2)

        if self.variant == "gated":
            g = torch.sigmoid(self.gate)
            e1 = g * e1
            e2 = (1 - g) * e2

        return self.proj(torch.cat([e1, e2], dim=-1))


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        layer_idx: int,
        rope_enabled: bool,
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.repeat_factor = num_heads // num_kv_heads
        self.layer_idx = layer_idx
        self.rope_enabled = rope_enabled

        self.c_attn = nn.Linear(dim, (num_heads + 2 * num_kv_heads) * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.ones(num_heads))
        self.attn_temp = nn.Parameter(torch.ones(num_heads))

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, t, d = q.shape
        half = d // 2
        if half == 0:
            return q, k
        freqs = torch.arange(half, device=q.device, dtype=q.dtype) / max(half, 1)
        angles = torch.einsum("t,d->td", torch.arange(t, device=q.device, dtype=q.dtype), freqs)
        sin = angles.sin().view(1, 1, t, half)
        cos = angles.cos().view(1, 1, t, half)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    x[..., :half] * cos - x[..., half:] * sin,
                    x[..., :half] * sin + x[..., half:] * cos,
                ],
                dim=-1,
            )

        return rotate(q), rotate(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=2,
        )
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.repeat_factor > 1:
            k = k.repeat_interleave(self.repeat_factor, dim=1)
            v = v.repeat_interleave(self.repeat_factor, dim=1)

        if self.rope_enabled:
            q, k = self._apply_rope(q, k)

        q = q * self.q_gain.view(1, -1, 1, 1) * self.attn_temp.view(1, -1, 1, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(self.head_dim))
        return self.proj(y.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        layer_idx: int,
        rope_enabled: bool,
        parallel_residual: bool,
    ):
        super().__init__()
        self.parallel_residual = parallel_residual
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            layer_idx=layer_idx,
            rope_enabled=rope_enabled,
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_mult * dim, bias=False),
            nn.GELU(),
            nn.Linear(mlp_mult * dim, dim, bias=False),
        )
        scale = 1.0 / math.sqrt(2 * (layer_idx + 1))
        self.attn_scale = nn.Parameter(torch.full((dim,), scale))
        self.mlp_scale = nn.Parameter(torch.full((dim,), scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parallel_residual:
            a = self.attn_scale * self.attn(self.attn_norm(x))
            m = self.mlp_scale * self.mlp(self.mlp_norm(x))
            return x + a + m
        x = x + self.attn_scale * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale * self.mlp(self.mlp_norm(x))
        return x


class DemoGPT(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.hash_embed = HashEmbedding(
            vocab_size=cfg.vocab_size,
            bigram_vocab_size=cfg.bigram_vocab_size,
            bigram_dim=cfg.bigram_dim,
            model_dim=cfg.model_dim,
            variant=cfg.hash_variant,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=cfg.model_dim,
                    num_heads=cfg.num_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    mlp_mult=cfg.mlp_mult,
                    layer_idx=i,
                    rope_enabled=cfg.use_rope,
                    parallel_residual=cfg.parallel_residual,
                )
                for i in range(cfg.num_layers)
            ]
        )
        self.final_norm = RMSNorm()

        self.use_recurrence = cfg.use_recurrence
        self.recurrence_interval = max(1, cfg.recurrence_interval)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.01)
        torch.nn.init.normal_(self.tok_emb.weight, std=0.02)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok_emb(x)
        he = self.hash_embed(x)
        if he.dim() == 3 and he.size(-1) == h.size(-1):
            h = h + he

        for i, block in enumerate(self.blocks):
            h = block(h)
            if self.use_recurrence and i > 0 and i % self.recurrence_interval == 0:
                recur_idx = max(0, i - self.recurrence_interval)
                h = self.blocks[recur_idx](h)

        logits = F.linear(self.final_norm(h), self.tok_emb.weight)
        scale = torch.rsqrt((logits ** 2).mean(dim=-1, keepdim=True) + 1e-5)
        return logits * scale * 6

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.get_logits(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


@dataclass
class TrialResult:
    rank: int
    run_id: str
    profile: str
    search_method: str
    status: str
    best_bpb: float
    best_val_loss: float
    final_train_loss: float
    params: int
    steps: int
    val_steps: int
    tokens_per_step: int
    tokens_per_sec: float
    peak_gpu_mem_gb: float
    wallclock_sec: float
    config: Dict[str, Any]
    timestamp: str


def default_config() -> Dict[str, Any]:
    hp = baseline.Hyperparameters
    return {
        "vocab_size": _safe_int(os.environ.get("VOCAB_SIZE", hp.vocab_size), hp.vocab_size),
        "num_layers": _safe_int(os.environ.get("NUM_LAYERS", hp.num_layers), hp.num_layers),
        "model_dim": _safe_int(os.environ.get("MODEL_DIM", hp.model_dim), hp.model_dim),
        "num_heads": _safe_int(os.environ.get("NUM_HEADS", hp.num_heads), hp.num_heads),
        "num_kv_heads": _safe_int(os.environ.get("NUM_KV_HEADS", hp.num_kv_heads), hp.num_kv_heads),
        "mlp_mult": _safe_int(os.environ.get("MLP_MULT", hp.mlp_mult), hp.mlp_mult),
        "bigram_vocab_size": _safe_int(os.environ.get("BIGRAM_VOCAB_SIZE", hp.bigram_vocab_size), hp.bigram_vocab_size),
        "bigram_dim": _safe_int(os.environ.get("BIGRAM_DIM", hp.bigram_dim), hp.bigram_dim),
        "train_seq_len": _safe_int(os.environ.get("TRAIN_SEQ_LEN", hp.train_seq_len), hp.train_seq_len),
        "iterations": _safe_int(os.environ.get("ITERATIONS", hp.iterations), hp.iterations),
        "warmup_steps": _safe_int(os.environ.get("WARMUP_STEPS", hp.warmup_steps), hp.warmup_steps),
        "warmdown_steps": _safe_int(os.environ.get("WARMDOWN_STEPS", hp.warmdown_steps), hp.warmdown_steps),
        "matrix_lr": _safe_float(os.environ.get("MATRIX_LR", hp.matrix_lr), hp.matrix_lr),
        "scalar_lr": _safe_float(os.environ.get("SCALAR_LR", hp.scalar_lr), hp.scalar_lr),
        "tied_embed_lr": _safe_float(os.environ.get("TIED_EMBED_LR", hp.tied_embed_lr), hp.tied_embed_lr),
        "weight_decay": _safe_float(os.environ.get("WEIGHT_DECAY", hp.weight_decay), hp.weight_decay),
        "grad_clip": _safe_float(os.environ.get("GRAD_CLIP", hp.grad_clip), hp.grad_clip),
        "schedule_mode": os.environ.get("SCHEDULE_MODE", hp.schedule_mode),
        "use_ema": _boolify(os.environ.get("USE_EMA", hp.use_ema)),
        "ema_decay": _safe_float(os.environ.get("EMA_DECAY", hp.ema_decay), hp.ema_decay),
        "use_swa": _boolify(os.environ.get("USE_SWA", hp.use_swa)),
        "swa_start": _safe_int(os.environ.get("SWA_START", hp.swa_start), hp.swa_start),
        "use_sliding_window_eval": _boolify(os.environ.get("USE_SLIDING_EVAL", hp.use_sliding_window_eval)),
        "eval_stride": _safe_int(os.environ.get("EVAL_STRIDE", hp.eval_stride), hp.eval_stride),
        "eval_steps": _safe_int(os.environ.get("EVAL_STEPS", 20), 20),
        "eval_every": _safe_int(os.environ.get("VAL_CHECK_FREQ", 20), 20),
        "batch_size": _safe_int(os.environ.get("BATCH_SIZE", 1), 1),
        "tokens_per_step": _safe_int(os.environ.get("TRAIN_BATCH_TOKENS", 0), 0),
        "use_rope": _boolify(os.environ.get("USE_ROPE", "true")),
        "parallel_residual": _boolify(os.environ.get("PARALLEL_RESIDUAL", "false")),
        "use_recurrence": _boolify(os.environ.get("USE_DEPTH_RECURRENCE", hp.use_depth_recurrence)),
        "recurrence_interval": _safe_int(os.environ.get("RECURRENCE_INTERVAL", hp.recurrence_interval), hp.recurrence_interval),
        "hash_variant": os.environ.get("HASH_VARIANT", "gated"),
        "seed": _safe_int(os.environ.get("SEED", 1337), 1337),
        "train_pattern": os.environ.get("TRAIN_PATTERN", "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"),
        "val_pattern": os.environ.get("VAL_PATTERN", "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"),
    }


def apply_profile(cfg: Dict[str, Any], profile: str) -> Dict[str, Any]:
    out = dict(cfg)
    if profile == "quick":
        out["iterations"] = 20
        out["eval_every"] = 5
        out["eval_steps"] = 4
    elif profile == "medium":
        out["iterations"] = 200
        out["eval_every"] = 20
        out["eval_steps"] = 8
    elif profile == "full":
        out["iterations"] = max(out["iterations"], 2000)
        out["eval_every"] = max(50, out["eval_every"])
        out["eval_steps"] = max(20, out["eval_steps"])
    return out


def build_cfg_namespace(cfg: Dict[str, Any]) -> SimpleNamespace:
    ns = SimpleNamespace(**cfg)
    if ns.tokens_per_step and ns.tokens_per_step > 0:
        ns.batch_size = max(1, ns.tokens_per_step // max(1, ns.train_seq_len))
    ns.batch_size = max(1, int(ns.batch_size))
    ns.hash_variant = str(ns.hash_variant).strip().lower()
    if ns.hash_variant not in {"gated", "dual_concat", "single", "off"}:
        ns.hash_variant = "gated"
    if ns.schedule_mode not in {"linear", "cosine"}:
        ns.schedule_mode = "cosine"
    return ns


def get_lr_mult(step: int, cfg: SimpleNamespace) -> float:
    return baseline.get_lr_schedule(
        step=step,
        total_steps=cfg.iterations,
        warmup_steps=cfg.warmup_steps,
        warmdown_steps=cfg.warmdown_steps,
        mode=cfg.schedule_mode,
    )


def eval_model(model: nn.Module, loader: baseline.DataLoader, cfg: SimpleNamespace) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        if cfg.use_sliding_window_eval:
            loader.eval_offset = 0
            for _ in range(cfg.eval_steps):
                x, y, _ = loader.get_eval_batch_sliding(stride=cfg.eval_stride)
                loss = model(x, y)
                losses.append(loss.item())
        else:
            for _ in range(cfg.eval_steps):
                x, y = loader.get_batch(batch_size=cfg.batch_size)
                loss = model(x, y)
                losses.append(loss.item())
    model.train()
    avg_loss = float(sum(losses) / max(1, len(losses)))
    bpb = float((avg_loss / math.log(2.0)) * 0.4412)
    return avg_loss, bpb


def _make_optimizer(model: nn.Module, cfg: SimpleNamespace, device: torch.device) -> torch.optim.Optimizer:
    tied_weight = model.tok_emb.weight
    scalar_params = [p for _, p in model.named_parameters() if p.ndim < 2 and p is not tied_weight]
    matrix_params = [p for _, p in model.named_parameters() if p.ndim == 2 and p is not tied_weight]

    param_groups = [
        {"params": [tied_weight], "lr": cfg.tied_embed_lr, "weight_decay": cfg.weight_decay * 0.1},
        {"params": scalar_params, "lr": cfg.scalar_lr, "weight_decay": 0.0},
        {"params": matrix_params, "lr": cfg.matrix_lr, "weight_decay": cfg.weight_decay},
    ]

    if device.type == "cuda":
        try:
            return torch.optim.Adam(param_groups, fused=True)
        except TypeError:
            return torch.optim.Adam(param_groups)
    return torch.optim.Adam(param_groups)


def _headers() -> str:
    return (
        f"{'step':>6} {'train_loss':>10} {'val_loss':>9} {'BPB':>7} {'best_BPB':>9} "
        f"{'params(M)':>9} {'tok/s':>10} {'gpu_GB':>8} {'lr_m':>8} {'lr_s':>8} {'lr_t':>8} "
        f"{'g_norm':>8} {'eta':>9} {'cfg':>10}"
    )


def run_trial(config: Dict[str, Any], profile: str, search_method: str) -> TrialResult:
    cfg = build_cfg_namespace(config)
    cfg_id = _config_id(config)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DemoGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = _make_optimizer(model, cfg, device)

    train_loader = baseline.DataLoader(cfg.train_pattern, cfg.train_seq_len, device)
    val_loader = baseline.DataLoader(cfg.val_pattern, cfg.train_seq_len, device)

    ema_state = baseline.EMAState(model, decay=cfg.ema_decay) if cfg.use_ema else None
    swa_acc = baseline.SWAAccumulator() if cfg.use_swa else None

    best_bpb = float("inf")
    best_val_loss = float("inf")
    final_train_loss = float("inf")
    last_tok_s = 0.0
    peak_mem_gb = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    tokens_seen = 0

    print("\n" + "=" * 120)
    print(f"Run {cfg_id} | profile={profile} | search={search_method} | device={device.type} | params={n_params:,}")
    print(f"Config: {json.dumps(config, sort_keys=True)}")
    print(_headers())

    for step in range(1, cfg.iterations + 1):
        lr_mult = get_lr_mult(step, cfg)
        optimizer.param_groups[0]["lr"] = cfg.tied_embed_lr * lr_mult
        optimizer.param_groups[1]["lr"] = cfg.scalar_lr * lr_mult
        optimizer.param_groups[2]["lr"] = cfg.matrix_lr * lr_mult

        x, y = train_loader.get_batch(batch_size=cfg.batch_size)
        loss = model(x, y)
        final_train_loss = float(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if cfg.grad_clip > 0:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item())
        else:
            grad_norm = 0.0
            total_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.detach().data.float().norm(2).item()
                    total_sq += g * g
            grad_norm = math.sqrt(total_sq)

        optimizer.step()
        if ema_state is not None:
            ema_state.update(model)
        if swa_acc is not None and step >= cfg.swa_start:
            swa_acc.add(model.state_dict())

        tokens_seen += int(x.numel())

        if step % cfg.eval_every == 0 or step == cfg.iterations:
            val_loss, bpb = eval_model(model, val_loader, cfg)
            if bpb < best_bpb:
                best_bpb = bpb
                best_val_loss = val_loss

            elapsed = max(time.time() - start, 1e-6)
            last_tok_s = tokens_seen / elapsed
            eta = (cfg.iterations - step) * (elapsed / step)

            if device.type == "cuda":
                mem_gb = torch.cuda.memory_allocated(device) / 1e9
                peak_mem_gb = max(peak_mem_gb, torch.cuda.max_memory_allocated(device) / 1e9)
            else:
                mem_gb = 0.0

            print(
                f"{step:6d} {final_train_loss:10.4f} {val_loss:9.4f} {bpb:7.4f} {best_bpb:9.4f} "
                f"{n_params/1e6:9.3f} {last_tok_s:10.0f} {mem_gb:8.2f} "
                f"{optimizer.param_groups[2]['lr']:8.5f} {optimizer.param_groups[1]['lr']:8.5f} {optimizer.param_groups[0]['lr']:8.5f} "
                f"{grad_norm:8.3f} {_format_seconds(eta):>9} {cfg_id:>10}"
            )

    if ema_state is not None:
        snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
        ema_state.apply(model)
        ema_val_loss, ema_bpb = eval_model(model, val_loader, cfg)
        if ema_bpb < best_bpb:
            best_bpb = ema_bpb
            best_val_loss = ema_val_loss
        for name, p in model.named_parameters():
            p.data.copy_(snapshot[name].data)

    if swa_acc is not None and swa_acc.get_count() > 0:
        snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
        model.load_state_dict(swa_acc.get_state_dict(), strict=False)
        swa_val_loss, swa_bpb = eval_model(model, val_loader, cfg)
        if swa_bpb < best_bpb:
            best_bpb = swa_bpb
            best_val_loss = swa_val_loss
        for name, p in model.named_parameters():
            p.data.copy_(snapshot[name].data)

    wallclock = time.time() - start

    return TrialResult(
        rank=0,
        run_id=cfg_id,
        profile=profile,
        search_method=search_method,
        status="ok",
        best_bpb=float(best_bpb),
        best_val_loss=float(best_val_loss),
        final_train_loss=float(final_train_loss),
        params=int(n_params),
        steps=int(cfg.iterations),
        val_steps=int(cfg.eval_steps),
        tokens_per_step=int(cfg.batch_size * cfg.train_seq_len),
        tokens_per_sec=float(last_tok_s),
        peak_gpu_mem_gb=float(peak_mem_gb),
        wallclock_sec=float(wallclock),
        config=dict(config),
        timestamp=_now_ts(),
    )


def _leaderboard(results: List[TrialResult]) -> List[TrialResult]:
    ranked = sorted(results, key=lambda r: (r.best_bpb, r.params, -r.tokens_per_sec))
    for idx, row in enumerate(ranked, start=1):
        row.rank = idx
    return ranked


def print_leaderboard(results: List[TrialResult], top_k: int = 20) -> None:
    ranked = _leaderboard(results)
    print("\n" + "=" * 120)
    print("Leaderboard (lower BPB is better)")
    print(f"{'Rank':>4} {'BPB':>8} {'Params(M)':>10} {'tok/s':>10} {'Runtime':>10} {'RunID':>12} Config")
    for row in ranked[:top_k]:
        cfg_short = {
            "layers": row.config.get("num_layers"),
            "dim": row.config.get("model_dim"),
            "heads": row.config.get("num_heads"),
            "kv": row.config.get("num_kv_heads"),
            "mlp": row.config.get("mlp_mult"),
            "seq": row.config.get("train_seq_len"),
            "m_lr": row.config.get("matrix_lr"),
            "warm": row.config.get("warmup_steps"),
            "wd": row.config.get("weight_decay"),
            "rope": row.config.get("use_rope"),
            "parallel": row.config.get("parallel_residual"),
            "recur": row.config.get("use_recurrence"),
            "hash": row.config.get("hash_variant"),
            "ema": row.config.get("use_ema"),
            "swa": row.config.get("use_swa"),
        }
        print(
            f"{row.rank:4d} {row.best_bpb:8.4f} {row.params/1e6:10.3f} {row.tokens_per_sec:10.0f} "
            f"{_format_seconds(row.wallclock_sec):>10} {row.run_id:>12} {json.dumps(cfg_short, separators=(',', ':'))}"
        )


def save_results(results: List[TrialResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = _leaderboard(results)

    json_path = out_dir / "leaderboard.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in ranked], f, indent=2)

    csv_path = out_dir / "leaderboard.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "run_id",
                "best_bpb",
                "best_val_loss",
                "final_train_loss",
                "params",
                "steps",
                "tokens_per_step",
                "tokens_per_sec",
                "peak_gpu_mem_gb",
                "wallclock_sec",
                "profile",
                "search_method",
                "timestamp",
                "config_json",
            ]
        )
        for r in ranked:
            writer.writerow(
                [
                    r.rank,
                    r.run_id,
                    r.best_bpb,
                    r.best_val_loss,
                    r.final_train_loss,
                    r.params,
                    r.steps,
                    r.tokens_per_step,
                    r.tokens_per_sec,
                    r.peak_gpu_mem_gb,
                    r.wallclock_sec,
                    r.profile,
                    r.search_method,
                    r.timestamp,
                    json.dumps(r.config, separators=(",", ":")),
                ]
            )

    jsonl_path = out_dir / "history.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        for r in ranked:
            f.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")


def _toggle_effect(results: List[TrialResult], key: str) -> Tuple[float, int, int]:
    on = [r.best_bpb for r in results if _boolify(r.config.get(key))]
    off = [r.best_bpb for r in results if not _boolify(r.config.get(key))]
    if not on or not off:
        return 0.0, len(on), len(off)
    return statistics.mean(off) - statistics.mean(on), len(on), len(off)


def recommendations(results: List[TrialResult]) -> List[str]:
    if not results:
        return ["No runs completed."]

    ranked = _leaderboard(results)
    best = ranked[0]
    recs: List[str] = []

    recs.append(f"Best current config is run {best.run_id} with BPB={best.best_bpb:.4f}.")

    if best.config.get("warmup_steps", 0) < max(200, best.steps // 20):
        recs.append("Increase warmup_steps; current warmup may be too short for stability.")

    if best.config.get("matrix_lr", 0.0) > 0.05:
        recs.append("Reduce matrix_lr slightly (for example by 10-20%) to reduce oscillation risk.")

    if best.config.get("grad_clip", 1.0) >= 1.5:
        recs.append("Lower grad_clip toward 1.0 if you see unstable or noisy validation BPB.")

    for toggle_key, label in [
        ("parallel_residual", "Parallel residual"),
        ("use_recurrence", "Recurrence"),
        ("use_rope", "RoPE"),
        ("use_ema", "EMA"),
        ("use_swa", "SWA"),
    ]:
        delta, on_n, off_n = _toggle_effect(results, toggle_key)
        if on_n > 0 and off_n > 0:
            if delta > 0.002:
                recs.append(f"{label} improved BPB by about {delta:.4f} on average.")
            elif delta < -0.002:
                recs.append(f"{label} hurt BPB by about {-delta:.4f} on average.")

    by_hash: Dict[str, List[float]] = {}
    for r in results:
        by_hash.setdefault(str(r.config.get("hash_variant", "gated")), []).append(r.best_bpb)
    if len(by_hash) > 1:
        avg_hash = {k: statistics.mean(v) for k, v in by_hash.items()}
        best_hash = min(avg_hash, key=avg_hash.get)
        recs.append(f"Best hash embedding variant so far: {best_hash}.")

    recs.append("Prioritize BPB first; use params and tokens/sec as tie-breakers for Top-5 iteration speed.")
    return recs


def _search_space() -> Dict[str, List[Any]]:
    return {
        "num_layers": [10, 11, 12],
        "model_dim": [448, 512, 576],
        "num_heads": [8],
        "num_kv_heads": [2, 4],
        "mlp_mult": [2, 3],
        "train_seq_len": [512, 1024],
        "matrix_lr": [0.035, 0.04, 0.045],
        "scalar_lr": [0.015, 0.02],
        "tied_embed_lr": [0.02, 0.03],
        "warmup_steps": [500, 1000, 1500],
        "weight_decay": [0.0005, 0.001, 0.002],
        "grad_clip": [0.8, 1.0, 1.2],
        "use_rope": [True, False],
        "parallel_residual": [False, True],
        "use_recurrence": [False, True],
        "recurrence_interval": [2, 3, 4],
        "hash_variant": ["gated", "dual_concat", "single"],
        "use_ema": [True, False],
        "use_swa": [False, True],
        "eval_stride": [256, 512],
    }


def grid_candidates(base: Dict[str, Any], space: Dict[str, List[Any]], limit: int) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    out: List[Dict[str, Any]] = []

    def rec(i: int, cur: Dict[str, Any]) -> None:
        if len(out) >= limit:
            return
        if i == len(keys):
            merged = dict(base)
            merged.update(cur)
            out.append(merged)
            return
        key = keys[i]
        for value in space[key]:
            cur[key] = value
            rec(i + 1, cur)
            if len(out) >= limit:
                return

    rec(0, {})
    return out


def random_candidates(base: Dict[str, Any], space: Dict[str, List[Any]], trials: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for _ in range(trials):
        c = dict(base)
        for key, values in space.items():
            c[key] = rng.choice(values)
        out.append(c)
    return out


def greedy_candidates(base: Dict[str, Any], space: Dict[str, List[Any]], trials: int) -> List[Dict[str, Any]]:
    """Coordinate greedy search skeleton using a planned candidate list."""
    out = [dict(base)]
    for key, values in space.items():
        for value in values:
            cand = dict(base)
            cand[key] = value
            out.append(cand)
            if len(out) >= trials:
                return out
    return out[:trials]


def export_integration(best: TrialResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    best_json = out_dir / "best_config_values.json"
    with best_json.open("w", encoding="utf-8") as f:
        json.dump(best.config, f, indent=2)

    ps1 = out_dir / "apply_best_config.ps1"
    env_keys = {
        "num_layers": "NUM_LAYERS",
        "model_dim": "MODEL_DIM",
        "num_heads": "NUM_HEADS",
        "num_kv_heads": "NUM_KV_HEADS",
        "mlp_mult": "MLP_MULT",
        "train_seq_len": "TRAIN_SEQ_LEN",
        "matrix_lr": "MATRIX_LR",
        "scalar_lr": "SCALAR_LR",
        "tied_embed_lr": "TIED_EMBED_LR",
        "warmup_steps": "WARMUP_STEPS",
        "warmdown_steps": "WARMDOWN_STEPS",
        "weight_decay": "WEIGHT_DECAY",
        "grad_clip": "GRAD_CLIP",
        "use_rope": "USE_ROPE",
        "parallel_residual": "PARALLEL_RESIDUAL",
        "use_recurrence": "USE_DEPTH_RECURRENCE",
        "recurrence_interval": "RECURRENCE_INTERVAL",
        "hash_variant": "HASH_VARIANT",
        "batch_size": "BATCH_SIZE",
        "tokens_per_step": "TRAIN_BATCH_TOKENS",
        "eval_stride": "EVAL_STRIDE",
        "use_ema": "USE_EMA",
        "use_swa": "USE_SWA",
    }
    with ps1.open("w", encoding="utf-8") as f:
        f.write("# Best config exported by lab_runner.py\n")
        for key, env_name in env_keys.items():
            if key in best.config:
                f.write(f"$env:{env_name}=\"{best.config[key]}\"\n")

    patch_text = out_dir / "train_gpt_final_apply.txt"
    with patch_text.open("w", encoding="utf-8") as f:
        f.write("Apply these exact values to Hyperparameters in train_gpt_final.py:\n\n")
        for key in sorted(best.config.keys()):
            f.write(f"{key} = {repr(best.config[key])}\n")
        f.write(
            "\nIf train_gpt_final.py uses environment variables, run apply_best_config.ps1 before launching training.\n"
        )


def load_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if p.suffix.lower() in {".json"}:
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError("Only JSON config files are supported currently.")


def run_manual(base_cfg: Dict[str, Any], trials: int, profile: str) -> List[TrialResult]:
    results: List[TrialResult] = []
    for i in range(trials):
        cfg = dict(base_cfg)
        cfg["seed"] = int(base_cfg.get("seed", 1337)) + i
        results.append(run_trial(cfg, profile=profile, search_method="manual"))
    return results


def run_search(base_cfg: Dict[str, Any], method: str, trials: int, profile: str, seed: int) -> List[TrialResult]:
    space = _search_space()

    if method == "grid":
        cands = grid_candidates(base_cfg, space, trials)
    elif method == "random":
        cands = random_candidates(base_cfg, space, trials, seed)
    elif method == "greedy":
        cands = greedy_candidates(base_cfg, space, trials)
    else:
        raise ValueError(f"Unknown search method: {method}")

    results: List[TrialResult] = []
    if method != "greedy":
        for idx, cfg in enumerate(cands):
            cfg["seed"] = seed + idx
            results.append(run_trial(cfg, profile=profile, search_method=method))
        return results

    best_cfg = dict(cands[0])
    best_result = run_trial(best_cfg, profile=profile, search_method=method)
    results.append(best_result)

    for cfg in cands[1:]:
        merged = dict(best_cfg)
        for k, v in cfg.items():
            if k in space and k in cfg:
                merged[k] = v
        merged["seed"] = seed + len(results)
        r = run_trial(merged, profile=profile, search_method=method)
        results.append(r)
        if r.best_bpb < best_result.best_bpb:
            best_result = r
            best_cfg = dict(r.config)

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment control center for parameter-golf model optimization.")
    p.add_argument("--mode", choices=["manual", "search"], default="manual")
    p.add_argument("--profile", choices=["quick", "medium", "full"], default="quick")
    p.add_argument("--search-method", choices=["grid", "random", "greedy"], default="random")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--config", type=str, default="", help="Path to JSON config file.")
    p.add_argument("--set", action="append", default=[], help="Override config using key=value. Repeatable.")
    p.add_argument("--out-dir", type=str, default="./experiment_lab")
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = default_config()
    cfg = apply_profile(cfg, args.profile)

    if args.config:
        cfg.update(load_config_file(args.config))

    if args.set:
        cfg.update(_parse_kv_overrides(args.set))

    cfg = apply_profile(cfg, args.profile)
    out_dir = Path(args.out_dir)

    print("\nParameter Golf Lab Runner")
    print(f"mode={args.mode} profile={args.profile} trials={args.trials} search={args.search_method}")
    print(f"base_config={json.dumps(cfg, sort_keys=True)}")

    if args.mode == "manual":
        results = run_manual(cfg, trials=args.trials, profile=args.profile)
    else:
        results = run_search(
            base_cfg=cfg,
            method=args.search_method,
            trials=args.trials,
            profile=args.profile,
            seed=args.seed,
        )

    print_leaderboard(results, top_k=args.top_k)
    save_results(results, out_dir=out_dir)

    ranked = _leaderboard(results)
    best = ranked[0]
    export_integration(best, out_dir=out_dir)

    print("\nRecommendations")
    for rec in recommendations(results):
        print(f"- {rec}")

    print("\nArtifacts")
    print(f"- {out_dir / 'leaderboard.csv'}")
    print(f"- {out_dir / 'leaderboard.json'}")
    print(f"- {out_dir / 'history.jsonl'}")
    print(f"- {out_dir / 'best_config_values.json'}")
    print(f"- {out_dir / 'apply_best_config.ps1'}")
    print(f"- {out_dir / 'train_gpt_final_apply.txt'}")

    print("\nTop-5 focus metrics")
    print("1. BPB (primary objective): lower is better.")
    print("2. Stability trend: best_BPB improves without spikes/divergence.")
    print("3. Params under budget with quality gains.")
    print("4. tokens/sec for iteration speed and search throughput.")


if __name__ == "__main__":
    main()
