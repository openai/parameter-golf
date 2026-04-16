#!/usr/bin/env python3
"""Direction-5 Training Script — GDN Hybrid, wallclock-limited.

Trains the GDN-Hybrid backbone (Model D: GDN×5 → SWA → GDN×5 → SWA_shared)
within the competition's 10-minute training budget on 8×H100 SXM.

Key differences from PR #1370 train_gdn_7k.py:
  - TRAIN_SEQ_LEN=2048 (longer context forces better GDN recurrence)
  - MAX_WALLCLOCK_SECONDS=590 (10-min budget minus 10s safety margin)
  - ITERATIONS=9999 (wallclock is the real limit)
  - WARMDOWN_ITERS=3000 (30% of expected ~9000 steps)
  - MuonEq-R: row-normalize before Newton-Schulz for better equivariance
  - ARCH_MODE=D (Model D GDN Hybrid)
  - No TTT in post-training eval (use eval_rls.py separately)

Environment variables:
    ARCH_MODE:            Model config key (default: D)
    TRAIN_SEQ_LEN:        Training context length (default: 2048)
    MAX_WALLCLOCK_SECONDS: Hard stop time (default: 590)
    ITERATIONS:           Max steps (default: 9999, wallclock-limited)
    WARMDOWN_ITERS:       Steps in LR warmdown (default: 3000)
    QK_GAIN_INIT:         SWA Q-gain init override (default: use config value)
    SEED:                 Random seed (default: 42)
    DATA_PATH:            Dataset directory
    CKPT_DIR:             Checkpoint output directory (default: checkpoints)
"""
from __future__ import annotations

import glob
import io
import json
import lzma
import math
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn.functional as F
import brotli
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Safety guard: if dynamo is ever invoked on code paths containing GDN layers
# (e.g. FLA internal usage), each unique `layer_idx` integer attribute would be
# treated as a static guard and trigger a separate recompilation. The default
# limit=8 would cause layers 8-9 to permanently fall back to eager mode.
# We no longer call torch.compile on the eval forward (see evaluate_sliding_window),
# so this guard is mainly defensive. 64 > 10 GDN layers, so it's always safe.
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 64

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs_gdn_mega import get_config

HybridGDN = None
CastedLinear = None


def _ensure_architecture_imported():
    global HybridGDN, CastedLinear
    if HybridGDN is None or CastedLinear is None:
        from architectures_gdn_mega import HybridGDN as _HybridGDN, CastedLinear as _CastedLinear
        HybridGDN = _HybridGDN
        CastedLinear = _CastedLinear
    return HybridGDN, CastedLinear


# ─── Hyperparameters ──────────────────────────────────────────────────────────

class Hyperparameters:
    score_mode = bool(int(os.environ.get("SCORE_MODE", "0")))
    dry_run_quant_selector_ckpt = os.environ.get("DRY_RUN_QUANT_SELECTOR_CKPT", "").strip()
    arch_mode = os.environ.get("ARCH_MODE", "D")
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))

    # Training length — wallclock-limited
    iterations = int(os.environ.get("ITERATIONS", 9999))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))   # Direction-5: 2048
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 590.0))  # 9m50s

    # Validation
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))
    save_every = int(os.environ.get("SAVE_EVERY", 1000))

    # Optimizer
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.97))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))

    # Eval
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    megakernel_enabled = bool(int(os.environ.get("MEGAKERNEL_ENABLED", "1")))
    debug_tapin = bool(int(os.environ.get("DEBUG_TAPIN", "1")))
    tapin_enabled = False

    # Diagnostics for low-cost 1-GPU probe runs
    diagnostics_enabled = bool(int(os.environ.get("DIAGNOSTICS_ENABLED", "1")))
    diagnostics_dir = os.environ.get("DIAGNOSTICS_DIR", "diagnostics")
    diagnostics_timing_steps = int(os.environ.get("DIAGNOSTICS_TIMING_STEPS", 3))
    diagnostics_eval_seq_len = int(os.environ.get("DIAGNOSTICS_EVAL_SEQ_LEN", 512))
    diagnostics_dump_tokens = int(os.environ.get("DIAGNOSTICS_DUMP_TOKENS", 128))
    diagnostics_top_k = int(os.environ.get("DIAGNOSTICS_TOP_K", 8))
    save_sweep_artifacts = bool(int(os.environ.get("SAVE_SWEEP_ARTIFACTS", "1")))

    # Checkpoint
    ckpt_dir = os.environ.get("CKPT_DIR", "checkpoints")

    # Compile
    compile_enabled = bool(int(os.environ.get("COMPILE_ENABLED", "1")))

    # Resume
    resume_ckpt = os.environ.get("RESUME_CKPT", "")

    # EMA / SWA
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))

    # Late QAT
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))

    # Chained job support
    auto_save_seconds = float(os.environ.get("AUTO_SAVE_SECONDS", "0"))
    total_iterations = int(os.environ.get("TOTAL_ITERATIONS", "0"))

    # Direction-5: QK gain override (set in config; this overrides config value if set)
    qk_gain_init_override = os.environ.get("QK_GAIN_INIT", "")
    bigram_enabled_override = os.environ.get("BIGRAM_ENABLED", "").strip()


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype=np.uint32, count=256)
    assert header[0] == 20240520, f"Bad magic: {header[0]}"
    assert header[1] in (1, 7), f"Bad version: {header[1]}"
    ntok = int(header[2])
    return torch.from_numpy(np.fromfile(file, dtype=np.uint16, offset=256 * 4)[:ntok].astype(np.int64))


class TokenStream:
    """Reads shards sequentially, supports coprime ordering via SHARD_ORDER_FILE."""
    def __init__(self, pattern: str):
        shard_order_file = os.environ.get("SHARD_ORDER_FILE", "")
        if shard_order_file and os.path.exists(shard_order_file):
            with open(shard_order_file) as f:
                self.files = [Path(line.strip()) for line in f if line.strip()]
        else:
            self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        assert self.files, f"No files matching {pattern}"
        self.idx = 0
        self.buf = load_data_shard(self.files[self.idx])
        self.pos = 0

    def _advance_file(self) -> None:
        self.idx = (self.idx + 1) % len(self.files)
        self.buf = load_data_shard(self.files[self.idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        parts = []
        remaining = n
        while remaining > 0:
            avail = self.buf.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            take_n = min(avail, remaining)
            parts.append(self.buf[self.pos:self.pos + take_n])
            self.pos += take_n
            remaining -= take_n
        return torch.cat(parts)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.stream = TokenStream(pattern)
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        tokens_per_rank = global_tokens // self.world_size
        seqs_per_rank = tokens_per_rank // seq_len
        total_seqs = seqs_per_rank * self.world_size
        total_needed = total_seqs * seq_len + 1
        all_tokens = self.stream.take(total_needed)
        start = self.rank * seqs_per_rank * seq_len
        chunk = all_tokens[start:start + seqs_per_rank * seq_len + 1]
        x = chunk[:-1].reshape(seqs_per_rank, seq_len)
        y = chunk[1:].reshape(seqs_per_rank, seq_len)
        return x.to(self.device), y.to(self.device)


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    parts = [load_data_shard(Path(f)) for f in files]
    combined = torch.cat(parts)
    return combined[:((combined.numel() - 1) // seq_len) * seq_len + 1]


def build_sentencepiece_luts(sp, vocab_size, device):
    base_bytes = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    has_space = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        raw = piece.encode("utf-8")
        base_bytes[i] = len(raw)
        if piece.startswith("\u2581"):
            has_space[i] = True
            base_bytes[i] = len(piece[1:].encode("utf-8")) + 1
        if sp.is_control(i) or sp.is_unknown(i):
            is_boundary[i] = True
    return base_bytes, has_space, is_boundary


def _tensor_summary(t: Tensor) -> dict[str, object]:
    x = t.detach().float()
    return {
        "shape": list(x.shape),
        "numel": int(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()) if x.numel() > 1 else 0.0,
        "rms": float(x.square().mean().sqrt().item()),
        "norm": float(x.norm().item()),
        "max_abs": float(x.abs().max().item()),
    }


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _block_name(prefix: str, idx: int, block_type: str, suffix: str = "") -> str:
    base = f"{prefix}.layer_{idx}_{block_type}"
    return f"{base}.{suffix}" if suffix else base


def _collect_parameter_summaries(model: nn.Module) -> dict[str, dict[str, object]]:
    return {name: _tensor_summary(param) for name, param in model.named_parameters()}


def _collect_model_probe(
    model: nn.Module,
    sample_x: Tensor,
    sample_y: Tensor,
    device: torch.device,
    top_k: int,
    dump_tokens: int,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    base_model = model.module if hasattr(model, "module") else model
    stats: dict[str, object] = {}
    tensor_dump: dict[str, Tensor] = {}
    pre_stats: dict[str, dict[str, object]] = {}
    post_stats: dict[str, dict[str, object]] = {}
    module_timing: dict[str, float] = {}
    handles = []
    cuda_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    cpu_starts: dict[str, float] = {}

    def _pick_tensor(obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, (tuple, list)) and obj:
            for item in obj:
                if torch.is_tensor(item):
                    return item
        return None

    def _pre_hook(name):
        def hook(_module, inputs):
            x = _pick_tensor(inputs)
            if x is not None:
                pre_stats[name] = _tensor_summary(x)
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                cuda_events[name] = (start, end)
            else:
                cpu_starts[name] = time.perf_counter()
        return hook

    def _post_hook(name):
        def hook(_module, _inputs, output):
            y = _pick_tensor(output)
            if y is not None:
                post_stats[name] = _tensor_summary(y)
            if device.type == "cuda":
                start, end = cuda_events[name]
                end.record()
            else:
                module_timing[name] = (time.perf_counter() - cpu_starts[name]) * 1000.0
        return hook

    for idx, (block, block_type) in enumerate(zip(base_model.blocks, base_model._block_types)):
        block_name = _block_name("block", idx, block_type)
        handles.append(block.register_forward_pre_hook(_pre_hook(block_name)))
        handles.append(block.register_forward_hook(_post_hook(block_name)))
        if block_type in ("swa", "swa_shared"):
            handles.append(block.attn.register_forward_pre_hook(_pre_hook(_block_name("attn", idx, block_type))))
            handles.append(block.attn.register_forward_hook(_post_hook(_block_name("attn", idx, block_type))))
        else:
            handles.append(block.recurrent.register_forward_pre_hook(_pre_hook(_block_name("recurrent", idx, block_type))))
            handles.append(block.recurrent.register_forward_hook(_post_hook(_block_name("recurrent", idx, block_type))))
        handles.append(block.mlp.register_forward_pre_hook(_pre_hook(_block_name("mlp", idx, block_type))))
        handles.append(block.mlp.register_forward_hook(_post_hook(_block_name("mlp", idx, block_type))))

    base_model.eval()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden, logits = base_model.forward_hidden(sample_x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        for name, (start, end) in cuda_events.items():
            module_timing[name] = float(start.elapsed_time(end))
    for handle in handles:
        handle.remove()
    base_model.train()

    hidden = hidden.detach()
    logits = logits.detach()
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=-1)
    preds = logits.argmax(dim=-1)
    top_vals, top_idx = logits.topk(min(top_k, logits.size(-1)), dim=-1)

    stats["sample_input"] = _tensor_summary(sample_x)
    stats["sample_target"] = _tensor_summary(sample_y)
    stats["final_hidden"] = _tensor_summary(hidden)
    stats["logits"] = _tensor_summary(logits)
    stats["entropy"] = _tensor_summary(ent)
    stats["top1_accuracy"] = float((preds == sample_y).float().mean().item())
    stats["modules"] = {
        name: {
            "input": pre_stats.get(name),
            "output": post_stats.get(name),
            "elapsed_ms": float(module_timing.get(name, 0.0)),
        }
        for name in sorted(set(pre_stats) | set(post_stats) | set(module_timing))
    }

    keep = min(dump_tokens, sample_x.shape[1])
    tensor_dump["sample_x"] = sample_x[:, :keep].detach().cpu()
    tensor_dump["sample_y"] = sample_y[:, :keep].detach().cpu()
    tensor_dump["hidden"] = hidden[:, :keep].detach().cpu().to(torch.float16)
    tensor_dump["entropy"] = ent[:, :keep].detach().cpu().to(torch.float16)
    tensor_dump["preds"] = preds[:, :keep].detach().cpu()
    tensor_dump["top_logits_values"] = top_vals[:, :keep].detach().cpu().to(torch.float16)
    tensor_dump["top_logits_indices"] = top_idx[:, :keep].detach().cpu()
    return stats, tensor_dump


def _summarize_hessians(hessians: dict[str, Tensor]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name, H in hessians.items():
        d = torch.diag(H.float())
        out[name] = {
            "shape": list(H.shape),
            "diag_mean": float(d.mean().item()),
            "diag_max": float(d.max().item()),
            "trace": float(d.sum().item()),
            "fro_norm": float(H.float().norm().item()),
        }
    return out


def _quant_summary(name: str, t: Tensor, q: Tensor, s: Tensor, quant_type: str) -> dict[str, object]:
    if s.ndim > 0:
        recon = q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
    else:
        recon = q.float() * float(s.item())
    err = (t.float() - recon.float())
    denom = t.float().pow(2).mean().item() + 1e-12
    return {
        "type": quant_type,
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "mse": float(err.pow(2).mean().item()),
        "rmse": float(err.pow(2).mean().sqrt().item()),
        "rel_mse": float(err.pow(2).mean().item() / denom),
        "max_abs_err": float(err.abs().max().item()),
        "quant_bytes_est": int(q.numel() + 2 * s.numel()),
    }


def _write_diagnostics_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)


ARTIFACT_CAP_BYTES = 16_000_000


def compress_artifact(quant_result: dict[str, Tensor], quant_meta: dict[str, object]) -> bytes:
    """Serialize quantized model with brotli compression and size-descending tensor order.

    Sorting tensors largest-first improves brotli's LZ back-references across
    large homogeneous int8 blocks. Brotli q11 beats zstd-22 by ~692KB on this model.
    """
    sorted_result = dict(
        sorted(quant_result.items(), key=lambda kv: kv[1].numel(), reverse=True)
    )
    buf = io.BytesIO()
    torch.save({"w": sorted_result, "m": quant_meta}, buf)
    return brotli.compress(buf.getvalue(), quality=11)


def _annotate_quant_stats(
    quant_stats: dict[str, dict[str, object]],
    block_types: list[str],
) -> dict[str, dict[str, object]]:
    block_re = re.compile(r"blocks\.(\d+)\.")
    out: dict[str, dict[str, object]] = {}
    for name, stats in quant_stats.items():
        entry = dict(stats)
        m = block_re.search(name)
        if name.startswith("tok_emb") or name.startswith("bigram"):
            entry["category"] = "embedding"
        elif m:
            idx = int(m.group(1))
            entry["block_index"] = idx
            entry["category"] = block_types[idx] if idx < len(block_types) else "unknown_block"
        else:
            entry["category"] = "global"
        out[name] = entry
    return out


def _make_tapin_state(h: Hyperparameters):
    if not h.tapin_enabled:
        return None
    return {"_stats": {"fires": 0, "cross_fires": 0, "within_fires": 0}}


def _maybe_apply_tapin_logits(
    h: Hyperparameters,
    logits: Tensor,
    x_batch: Tensor,
    ws_list: list[int],
    wlens: list[int],
    val_tokens_np,
    bos_token_id: int,
    tapin_state,
):
    if not h.tapin_enabled or tapin_state is None:
        return logits
    try:
        from tapin_v6 import apply_tapin_v4_v6
        return apply_tapin_v4_v6(
            logits, x_batch, ws_list, wlens,
            val_tokens_np, tapin_state, h,
            bos_token_id=bos_token_id,
        )
    except Exception as e:
        if getattr(h, "is_main_process", False):
            print(f"tapin:disabled (runtime error: {e})", flush=True)
        h.tapin_enabled = False
        return logits


def generate_coprime_shard_order(shard_files: list, seed: int = 42) -> list:
    n = len(shard_files)
    if n <= 1:
        return shard_files
    target = max(1, int(n / 1.618))
    stride = target
    while math.gcd(stride, n) != 1:
        stride += 1
    rng = random.Random(seed)
    start = rng.randint(0, n - 1)
    order = []
    pos = start
    for _ in range(n):
        order.append(shard_files[pos])
        pos = (pos + stride) % n
    return order


# ─── Muon Optimizer (MuonEq-R) ───────────────────────────────────────────────

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz 5th-order iteration with MuonEq-R row normalization.

    MuonEq-R: row-normalize each gradient row before the Frobenius normalization.
    This makes the update equivariant to row-wise rescaling (~0.001 BPB gain
    observed in transformer competition experiments).
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # MuonEq-R: row-normalize before NS
    if X.ndim == 2:
        row_norms = X.norm(dim=1, keepdim=True).clamp_min(eps)
        X = X / row_norms
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                       nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g + momentum * buf
                else:
                    g = buf
                if g.ndim == 2 and min(g.shape) >= 2:
                    g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(g, alpha=-lr)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def eval_val_sliding(
    h: Hyperparameters,
    model: nn.Module,
    val_tokens: Tensor,
    val_tokens_np,
    bos_token_id: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
    seq_len: int = 2048,
    stride: int = 64,
    batch_seqs: int = 128,
    xsa_eval: bool = False,
    logit_temperature: float = 1.0,
) -> tuple[float, float]:
    """Standard sliding window evaluation."""
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

    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    if xsa_eval and hasattr(base_model, 'set_xsa'):
        base_model.set_xsa(True)
    tapin_state = _make_tapin_state(h)

    # Do NOT torch.compile here. FLA's GatedDeltaNet has integer `layer_idx`
    # attributes; dynamo treats each as a unique static guard and recompiles once
    # per layer (10 layers = 10 compilations). On a warm Triton cache this is
    # ~3s total. On a cold cache (fresh pod) it is ~107s — eating 18% of the
    # 590s budget and causing ~314 fewer training steps. FLA's Triton kernels
    # are already hand-optimized; there is nothing for dynamo to gain here.
    compiled_logits = base_model.forward_logits

    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            logits = _maybe_apply_tapin_logits(
                h, logits, x_batch, batch_ws, wlens, val_tokens_np, bos_token_id, tapin_state,
            )
            logits = logits.float()
            if logit_temperature != 1.0:
                logits = logits / logit_temperature

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
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

    if xsa_eval and hasattr(base_model, 'set_xsa'):
        base_model.set_xsa(False)

    if h.tapin_enabled and tapin_state is not None and getattr(h, "debug_tapin", False) and rank == 0:
        stats = tapin_state.get("_stats", {})
        print(
            f"tapin: fires={int(stats.get('fires', 0))} "
            f"within={int(stats.get('within_fires', 0))} "
            f"cross={int(stats.get('cross_fires', 0))}",
            flush=True,
        )

    model.train()
    return val_loss, bits_per_token * tokens_per_byte


# ─── Quantization ────────────────────────────────────────────────────────────

CONTROL_PATTERNS = (
    "resid_mix", "q_gain", "smear", "skip_weight", "attn_scale", "mlp_scale",
)


def generate_autoregressive_calib(model, device, num_seqs=64, seq_len=2048,
                                   vocab_size=1024, temperature=0.8, batch_size=8, seed=42):
    # RoPE bug workaround: apply_rotary_emb uses x.shape[-2] (= num_heads=8) to slice cos.
    # When T < num_heads, cos[:num_heads] clips to [T, D//2] which fails to broadcast with
    # the head dimension. Fix: start with init_len >= num_heads+1 tokens to skip T in [2,8).
    init_len = 16
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - batch_start)
            tokens = torch.randint(0, vocab_size, (bs, init_len), device=device, generator=rng)
            for pos in range(seq_len - init_len):
                logits = model.forward_logits(tokens)
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens = torch.cat([tokens, next_tok], dim=1)
            for i in range(bs):
                all_tokens.append(tokens[i:i+1])
    return all_tokens


def collect_hessians_from_tokens(hessian_model, token_seqs, device):
    _, casted_linear_cls = _ensure_architecture_imported()
    hessians = {}
    hooks = []
    for name, module in hessian_model.named_modules():
        if isinstance(module, casted_linear_cls):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')
            def make_hook(pname):
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pname] += (x.T @ x).cpu()
                return hook_fn
            h = module.register_forward_hook(make_hook(param_name))
            hooks.append(h)
    hessian_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in token_seqs:
            x = seq[:, :-1].to(device)
            y = seq[:, 1:].to(device)
            hessian_model(x, y)
    for h in hooks:
        h.remove()
    num_batches = len(token_seqs)
    for name in hessians:
        hessians[name] /= num_batches
    return hessians


def quantize_int6_gptq(weight, hessian=None, clip_range=31, block_size=128):
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        raise ValueError("quantize_int6_gptq requires a 2D weight matrix and a Hessian")
    rows, cols = t32.shape
    H = hessian.float().clone()
    H = 0.5 * (H + H.T)
    diag_idx = torch.arange(cols, device=H.device)
    diag = torch.diag(H)
    dead = diag <= 0
    H[dead, dead] = 1
    diag = torch.diag(H)
    mean_diag = torch.mean(diag).clamp_min(1e-8)
    H[diag_idx, diag_idx] += 0.01 * mean_diag
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = None
    for attempt in range(4):
        try:
            Hinv = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(Hinv)
            Hinv = torch.linalg.cholesky(Hinv, upper=True)
            break
        except torch.linalg.LinAlgError:
            extra_damp = 0.1 * (10 ** attempt) * torch.mean(torch.diag(H)).clamp_min(1e-8)
            H[diag_idx, diag_idx] += extra_damp
    if Hinv is None:
        eigvals, eigvecs = torch.linalg.eigh(H)
        floor = torch.mean(torch.diag(H)).clamp_min(1e-8) * 1e-6
        eigvals = eigvals.clamp_min(floor)
        H = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T
        H = 0.5 * (H + H.T)
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    best_q = None; best_scale = None; best_err = float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, inv_perm]
    return best_q, best_scale


def quantize_int_gptq(weight: Tensor, bits: int, hessian: Tensor | None = None, block_size: int = 128):
    clip_range = (1 << (bits - 1)) - 1
    return quantize_int6_gptq(weight, hessian=hessian, clip_range=clip_range, block_size=block_size)


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


def quantize_int_per_row(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    clip_range = (1 << (bits - 1)) - 1
    return quantize_int6_per_row(t, clip_range=clip_range)


def quantize_int8_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    clip_q = 0.9999984
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), clip_q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0).to(torch.float16)
        q = torch.clamp(torch.round(clipped / scale.float()[:, None]), -127, 127).to(torch.int8)
        return q, scale
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale.float()), -127, 127).to(torch.int8)
    return q, scale


def _compile_bit_rules(bit_rules: list[tuple[str, int]] | None) -> list[tuple[re.Pattern[str], int]]:
    compiled: list[tuple[re.Pattern[str], int]] = []
    if not bit_rules:
        return compiled
    for pattern, bits in bit_rules:
        compiled.append((re.compile(pattern), int(bits)))
    return compiled


def _resolve_matrix_bits(name: str, default_bits: int, compiled_bit_rules: list[tuple[re.Pattern[str], int]]) -> tuple[int, bool]:
    bits = default_bits
    matched = False
    for pattern, override_bits in compiled_bit_rules:
        if pattern.search(name):
            bits = override_bits
            matched = True
    return bits, matched


def mixed_quantize(
    state_dict: dict[str, Tensor],
    hessians: dict[str, Tensor] | None = None,
    return_stats: bool = False,
    bit_rules: list[tuple[str, int]] | None = None,
    default_matrix_bits: int = 6,
    require_hessians: bool = False,
) -> tuple[dict[str, Tensor], dict[str, object]] | tuple[dict[str, Tensor], dict[str, object], dict[str, dict[str, object]]]:
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    stats: dict[str, dict[str, object]] = {}
    compiled_bit_rules = _compile_bit_rules(bit_rules)
    alias_sources: dict[tuple[int, int, tuple[int, ...], tuple[int, ...], torch.dtype], str] = {}
    for name, tensor in state_dict.items():
        alias_key = None
        if torch.is_tensor(tensor):
            try:
                alias_key = (
                    int(tensor.untyped_storage().data_ptr()),
                    int(tensor.storage_offset()),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.dtype,
                )
            except RuntimeError:
                alias_key = None
        if alias_key is not None and alias_key in alias_sources:
            meta[name] = {"type": "alias", "source": alias_sources[alias_key]}
            continue
        t = tensor.detach().cpu().contiguous()
        if any(p in name for p in CONTROL_PATTERNS):
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            if alias_key is not None:
                alias_sources[alias_key] = name
            continue
        if not t.is_floating_point():
            result[name] = t
            meta[name] = "passthrough"
            if alias_key is not None:
                alias_sources[alias_key] = name
            continue
        if t.ndim == 2:
            matrix_bits, has_override = _resolve_matrix_bits(name, default_matrix_bits, compiled_bit_rules)
            if t.numel() <= 65536 and not has_override:
                result[name] = t.to(torch.float16)
                meta[name] = "passthrough"
                if alias_key is not None:
                    alias_sources[alias_key] = name
                continue
            H = hessians.get(name) if hessians else None
            if require_hessians and H is None and hessians is not None:
                # Only warn — non-CastedLinear matrices (embeddings, recurrent, shared SWA aliases)
                # legitimately lack Hessians and should quantize per-row
                pass
            if H is not None:
                q, s = quantize_int_gptq(t, bits=matrix_bits, hessian=H)
            elif matrix_bits == 8:
                q, s = quantize_int8_per_row(t)
            else:
                q, s = quantize_int_per_row(t, bits=matrix_bits)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": f"int{matrix_bits}"}
            if alias_key is not None:
                alias_sources[alias_key] = name
            if return_stats:
                stats[name] = _quant_summary(name, t, q, s, f"int{matrix_bits}")
        else:
            q, s = quantize_int8_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
            if alias_key is not None:
                alias_sources[alias_key] = name
            if return_stats:
                stats[name] = _quant_summary(name, t, q, s, "int8")
    if return_stats:
        return result, meta, stats
    return result, meta


def dequantize_mixed(result: dict[str, Tensor], meta: dict[str, object],
                     template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    def resolve(name: str):
        if name in out:
            return out[name]
        info = meta.get(name)
        if info is None:
            return None
        if isinstance(info, dict) and info.get("type") == "alias":
            source = str(info["source"])
            t = resolve(source)
            if t is None:
                return None
            out[name] = t
            return t
        orig = template_sd[name]
        orig_dtype = orig.dtype
        if info == "passthrough":
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            return t
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
        return out[name]
    for name in template_sd:
        resolve(name)
    return out


def _code_paths_for_artifact(script_path: Path, tapin_enabled: bool) -> list[Path]:
    code_paths = [
        script_path.resolve(),
        script_path.resolve().with_name("architectures_gdn_mega.py"),
        script_path.resolve().with_name("configs_gdn_mega.py"),
    ]
    if tapin_enabled:
        code_paths.append(script_path.resolve().with_name("tapin_v6.py"))
    return [path for path in code_paths if path.exists()]


def _estimate_compressed_code_bytes(code_paths: list[Path]) -> int:
    total = 0
    for path in code_paths:
        total += len(lzma.compress(path.read_bytes(), preset=9 | lzma.PRESET_EXTREME))
    return total


def _build_quant_variant_specs(spec: str) -> list[dict[str, object]]:
    return [{"label": "baseline_int6", "bit_rules": [], "priority": 0, "default_matrix_bits": 6}]


def _eval_sliding_variant(
    h: Hyperparameters,
    model: nn.Module,
    val_tokens: Tensor,
    val_tokens_np: np.ndarray,
    bos_token_id: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
    *,
    seq_len: int,
    stride: int,
    xsa_eval: bool,
    tapin_enabled: bool,
    logit_temperature: float = 1.0,
) -> tuple[float, float]:
    old_tapin = h.tapin_enabled
    h.tapin_enabled = tapin_enabled
    try:
        return eval_val_sliding(
            h, model, val_tokens, val_tokens_np, bos_token_id,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            rank, world_size, device,
            seq_len=seq_len, stride=stride, xsa_eval=xsa_eval,
            logit_temperature=logit_temperature,
        )
    finally:
        h.tapin_enabled = old_tapin


def _unwrap_state_dict_for_quant_dry_run(payload) -> dict[str, Tensor]:
    if isinstance(payload, dict):
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        if all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise ValueError("DRY_RUN_QUANT_SELECTOR_CKPT must point to a raw state_dict or a checkpoint with model_state_dict")


def _prepare_state_dict_for_export(state_dict: dict[str, Tensor], config: dict[str, object]) -> dict[str, Tensor]:
    if bool(config.get("bigram_enabled", True)):
        return state_dict
    return {k: v for k, v in state_dict.items() if not k.startswith("bigram.")}


def _load_model_state_dict_compat(model: nn.Module, state_dict: dict[str, Tensor], log_fn=None) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        result = model.load_state_dict(state_dict, strict=False)
        bad_missing = [k for k in result.missing_keys if not k.startswith("bigram.")]
        bad_unexpected = [k for k in result.unexpected_keys if not k.startswith("bigram.")]
        if bad_missing or bad_unexpected:
            raise
        if log_fn is not None and (result.missing_keys or result.unexpected_keys):
            log_fn(
                "  Ignored bigram state mismatch "
                f"(missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)})"
            )


def _run_quant_selector_dry_run(args: Hyperparameters, config: dict[str, object]) -> None:
    ckpt_path = Path(args.dry_run_quant_selector_ckpt).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"DRY_RUN_QUANT_SELECTOR_CKPT not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_cpu = _prepare_state_dict_for_export(_unwrap_state_dict_for_quant_dry_run(payload), config)
    code_paths = _code_paths_for_artifact(Path(__file__), args.tapin_enabled)
    compressed_code_bytes = _estimate_compressed_code_bytes(code_paths)
    raw_code_bytes = sum(path.stat().st_size for path in code_paths)
    variant = _build_quant_variant_specs("baseline_int6")[0]

    print(f"=== Quant selector dry run: {config['arch_name']} ===", flush=True)
    print(f"checkpoint: {ckpt_path}", flush=True)
    print(f"tapin_enabled: {int(args.tapin_enabled)}", flush=True)
    print(f"raw_code_bytes: {raw_code_bytes:,}", flush=True)
    print(f"compressed_code_bytes: {compressed_code_bytes:,}", flush=True)
    print(f"artifact_cap_bytes: {ARTIFACT_CAP_BYTES:,}", flush=True)

    label = str(variant["label"])
    quant_result, quant_meta = mixed_quantize(
        sd_cpu,
        hessians=None,
        bit_rules=variant.get("bit_rules", []),
        default_matrix_bits=int(variant.get("default_matrix_bits", 6)),
    )
    quant_blob = compress_artifact(quant_result, quant_meta)
    artifact_bytes = len(quant_blob)
    estimated_total_bytes = artifact_bytes + compressed_code_bytes
    print(
        f"{label}: model={artifact_bytes:,} total={estimated_total_bytes:,} "
        f"fits={estimated_total_bytes <= ARTIFACT_CAP_BYTES}",
        flush=True,
    )
    print("selected: baseline_int6", flush=True)


# ─── Checkpoint Saving ───────────────────────────────────────────────────────

def save_checkpoint(model, step, val_bpb, ckpt_dir, arch_name, seed):
    base = model.module if hasattr(model, 'module') else model
    ckpt = {
        "step": step, "val_bpb": val_bpb,
        "arch_name": arch_name, "seed": seed,
        "model_state_dict": base.state_dict(),
    }
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{arch_name}_step{step}_seed{seed}.pt")
    torch.save(ckpt, path)
    return path


def save_full_checkpoint(model, step, val_bpb, ckpt_dir, arch_name, seed,
                         muon_opt, adam_opt, ema_state, swa_state, swa_count,
                         qat_enabled, rng_states=None, stream_state=None):
    base = model.module if hasattr(model, 'module') else model
    ckpt = {
        "step": step, "val_bpb": val_bpb,
        "arch_name": arch_name, "seed": seed,
        "model_state_dict": {k: v.cpu() for k, v in base.state_dict().items()},
        "muon_opt_state": muon_opt.state_dict(),
        "adam_opt_state": adam_opt.state_dict(),
        "ema_state": {k: v.cpu() for k, v in ema_state.items()},
        "swa_state": {k: v.cpu() for k, v in swa_state.items()} if swa_state is not None else None,
        "swa_count": swa_count,
        "qat_enabled": qat_enabled,
    }
    if rng_states is not None:
        ckpt["rng_states"] = rng_states
    if stream_state is not None:
        ckpt["stream_state"] = stream_state
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"full_ckpt_step{step}_seed{seed}.pt")
    torch.save(ckpt, path)
    return path


def _find_latest_full_ckpt(ckpt_dir):
    import re
    pattern = os.path.join(ckpt_dir, "full_ckpt_step*_seed*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    step_re = re.compile(r"full_ckpt_step(\d+)_seed")
    best_step, best_path = -1, None
    for f in files:
        m = step_re.search(os.path.basename(f))
        if m:
            s = int(m.group(1))
            if s > best_step:
                best_step, best_path = s, f
    return best_path


# ─── Main Training Loop ─────────────────────────────────────────────────────

def main():
    global zeropower_via_newtonschulz5
    args = Hyperparameters()
    config = get_config(args.arch_mode)
    score_mode = args.score_mode

    if score_mode:
        args.tapin_enabled = False
        args.debug_tapin = False
        args.diagnostics_enabled = False
        args.save_sweep_artifacts = False
        args.train_log_every = max(args.train_log_every, 500)

    if args.megakernel_enabled and args.late_qat_threshold > 0:
        args.late_qat_threshold = 0.0

    if args.dry_run_quant_selector_ckpt:
        _run_quant_selector_dry_run(args, config)
        return

    hybrid_gdn_cls, casted_linear_cls = _ensure_architecture_imported()

    # Direction-5: optional QK_GAIN_INIT override from env
    if args.qk_gain_init_override:
        config["qk_gain_init"] = float(args.qk_gain_init_override)
    if args.bigram_enabled_override:
        config["bigram_enabled"] = bool(int(args.bigram_enabled_override))

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, 8 // world_size)
    master_process = rank == 0
    args.is_main_process = master_process

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.compile_enabled:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt" if master_process else None

    def log0(msg: str, console: bool = True):
        if not master_process:
            return
        if console:
            print(msg, flush=True)
        if logfile:
            with open(logfile, "a") as f:
                print(msg, file=f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log0(f"=== Direction-5: GDN Hybrid Training ===")
    if score_mode:
        log0("Score mode: enabled")
    log0(f"Arch: {config['arch_name']} (ARCH_MODE={args.arch_mode})")
    log0(f"Seed: {args.seed}, Max steps: {args.iterations}, Warmdown: {args.warmdown_iters}")
    if args.total_iterations > 0:
        log0(f"Schedule total iterations: {args.total_iterations}")
    log0(f"Train seq_len: {args.train_seq_len}, Wallclock budget: {args.max_wallclock_seconds}s")
    log0(f"QK_GAIN_INIT: {config.get('qk_gain_init', 1.5)}")
    log0(f"World size: {world_size}, Grad accum: {grad_accum_steps}")
    log0(f"EMA decay: {args.ema_decay}, SWA: {args.swa_enabled} (every {args.swa_every})")
    log0(f"Late QAT threshold: {args.late_qat_threshold}")
    log0(f"MuonEq-R: enabled")
    log0(f"Megakernel MLP: {int(args.megakernel_enabled)}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    assert int(sp.vocab_size()) == args.vocab_size

    val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)
    val_tokens_np = np.ascontiguousarray(val_tokens.numpy(), dtype=np.int32)
    bos_token_id = int(sp.bos_id()) if int(sp.bos_id()) >= 0 else 1
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"Validation tokens: {val_tokens.numel()-1:,}")
    diagnostics_dir = Path(args.diagnostics_dir)
    diagnostics_path = diagnostics_dir / f"{config['arch_name']}_seed{args.seed}_summary.json"
    diagnostics_tensor_path = diagnostics_dir / f"{config['arch_name']}_seed{args.seed}_tensors.pt"
    diagnostics_payload: dict[str, object] = {
        "arch_name": config["arch_name"],
        "seed": args.seed,
        "env": {
            "megakernel_enabled": int(args.megakernel_enabled),
            "tapin_enabled": int(args.tapin_enabled),
            "compile_enabled": int(args.compile_enabled),
            "gptq_enabled": int(bool(int(os.environ.get("GPTQ_ENABLED", "0")))),
            "world_size": world_size,
            "train_seq_len": args.train_seq_len,
            "eval_seq_len": args.eval_seq_len,
        },
        "config": config,
        "step_timings_ms": [],
    }
    diag_seq_len = min(args.diagnostics_eval_seq_len, args.eval_seq_len, val_tokens.numel() - 1)
    sample_chunk = val_tokens[: diag_seq_len + 1].to(dtype=torch.int64, device=device)
    diagnostics_x = sample_chunk[:-1].unsqueeze(0)
    diagnostics_y = sample_chunk[1:].unsqueeze(0)

    _t0 = time.time()
    model = hybrid_gdn_cls(config, args.vocab_size)
    model = model.to(device).bfloat16()
    log0(f"Model built in {time.time()-_t0:.1f}s")

    for module in model.modules():
        if isinstance(module, casted_linear_cls):
            module.float()
    for name, p in model.named_parameters():
        if p.ndim <= 1:
            p.data = p.data.float()

    param_counts = model.count_params()
    log0(f"Parameters: {param_counts}")
    log0(f"Total params: {param_counts['total']:,}")
    if args.diagnostics_enabled and master_process:
        diagnostics_payload["param_counts"] = param_counts
        diagnostics_payload["parameter_summaries"] = _collect_parameter_summaries(model)
        diagnostics_payload["weight_diagnostics"] = model.get_diagnostics()

    start_step = 0
    resume_state = None
    resume_ckpt_path = args.resume_ckpt
    if resume_ckpt_path == "auto":
        resume_ckpt_path = _find_latest_full_ckpt(args.ckpt_dir) or ""
        if resume_ckpt_path:
            log0(f"Auto-detected resume checkpoint: {resume_ckpt_path}")
        else:
            log0("Auto-resume: no full checkpoint found, starting fresh")
    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        log0(f"Resuming from checkpoint: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
        base_sd = ckpt["model_state_dict"]
        _load_model_state_dict_compat(model, {k: v.to(device) for k, v in base_sd.items()}, log0)
        start_step = ckpt.get("step", 0)
        log0(f"Resumed model at step {start_step}, val_bpb={ckpt.get('val_bpb', 'N/A')}")
        if "muon_opt_state" in ckpt:
            resume_state = ckpt
            log0("  Full checkpoint detected — will restore optimizers, EMA, SWA, RNG")
        else:
            log0("  Lightweight checkpoint — model only")
            del ckpt

    base_model = model
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    matrix_params = []
    scalar_params = []
    embed_params = []
    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        if "tok_emb" in name:
            embed_params.append(p)
        elif p.ndim == 2 and min(p.shape) >= 2:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    log0(f"Matrix params: {sum(p.numel() for p in matrix_params):,}")
    log0(f"Scalar params: {sum(p.numel() for p in scalar_params):,}")
    log0(f"Embed params: {sum(p.numel() for p in embed_params):,}")

    muon_opt = Muon(
        matrix_params, lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    adam_opt = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr},
         {"params": embed_params, "lr": args.tied_embed_lr}],
        betas=(args.beta1, args.beta2),
        weight_decay=args.adam_wd,
        fused=True,
    )

    if resume_state is not None:
        muon_opt.load_state_dict(resume_state["muon_opt_state"])
        adam_opt.load_state_dict(resume_state["adam_opt_state"])
        log0("  Restored optimizer states (Muon + Adam)")

    shard_order_file = os.environ.get("SHARD_ORDER_FILE", "")
    if not shard_order_file:
        shard_files = sorted(glob.glob(args.train_files))
        if shard_files:
            ordered = generate_coprime_shard_order(shard_files, seed=args.seed)
            # Use rank-specific path to avoid concurrent write race across 8 processes
            shard_order_path = f"/tmp/shard_order_{args.run_id}_rank{rank}.txt"
            with open(shard_order_path, "w") as f:
                for sf in ordered:
                    f.write(str(sf) + "\n")
            os.environ["SHARD_ORDER_FILE"] = shard_order_path
            log0(f"Generated coprime shard order: stride across {len(shard_files)} shards")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    schedule_total = args.total_iterations if args.total_iterations > 0 else args.iterations

    def lr_schedule(step: int) -> float:
        warmdown_start = max(args.warmup_steps, schedule_total - args.warmdown_iters)
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        elif step >= warmdown_start:
            progress = min(max((step - warmdown_start) / max(1, args.warmdown_iters), 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0

    if resume_state is not None:
        saved_ema = resume_state.get("ema_state")
        if saved_ema is not None:
            ema_state = {k: v.to(device).float() for k, v in saved_ema.items()}
            log0("  Restored EMA state")
        saved_swa = resume_state.get("swa_state")
        if saved_swa is not None:
            swa_state = {k: v.cpu() for k, v in saved_swa.items()}
            swa_count = resume_state.get("swa_count", 0)
            log0(f"  Restored SWA state (count={swa_count})")
        else:
            swa_count = resume_state.get("swa_count", 0)
        if resume_state.get("qat_enabled", False):
            if args.late_qat_threshold > 0:
                casted_linear_cls._qat_enabled = True
                log0("  Restored QAT enabled state")
            else:
                log0("  Ignored saved QAT enabled state (late QAT disabled for this run)")
        saved_rng = resume_state.get("rng_states")
        if saved_rng is not None:
            torch.set_rng_state(saved_rng["torch_cpu"])
            torch.cuda.set_rng_state(saved_rng["torch_cuda"])
            np.random.set_state(saved_rng["numpy"])
            random.setstate(saved_rng["python"])
            log0("  Restored RNG states")
        saved_stream = resume_state.get("stream_state")
        if saved_stream is not None:
            s_idx, s_pos = saved_stream
            stream = train_loader.stream
            while stream.idx != s_idx:
                stream._advance_file()
            stream.pos = s_pos
            log0(f"  Restored stream state (shard={s_idx}, pos={s_pos})")
        else:
            if start_step > 0:
                log0(f"  Fast-forwarding data loader by {start_step} steps...")
                for _ in range(start_step):
                    train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                log0(f"  Data loader advanced to step {start_step}")
        del resume_state
        log0("  Full checkpoint restore complete")

    # ─── Training Loop ───────────────────────────────────────────────────
    stale_marker = os.path.join(args.ckpt_dir, f"CHAIN_RESUME_FROM_seed{args.seed}")
    if os.path.exists(stale_marker):
        os.remove(stale_marker)

    log0(f"\n{'='*80}")
    log0(f"Starting training: max {args.iterations} steps (from step {start_step})")
    log0(f"Wallclock budget: {args.max_wallclock_seconds}s")
    log0(f"{'='*80}\n")

    t0 = time.time()
    running_loss = 0.0
    loss_count = 0
    stop_after_step = None
    step = start_step

    for step in range(start_step + 1, args.iterations + 1):
        if stop_after_step is not None and step > stop_after_step:
            log0(f"Stopping early at step {step} (wallclock limit)")
            break

        lr_mul = lr_schedule(step)

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        current_muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in muon_opt.param_groups:
            group["lr"] = args.matrix_lr * lr_mul
            group["momentum"] = current_muon_momentum
        for i, pg in enumerate(adam_opt.param_groups):
            if i == 0:
                pg["lr"] = args.scalar_lr * lr_mul
            else:
                pg["lr"] = args.tied_embed_lr * lr_mul

        warmdown_start = max(args.warmup_steps, schedule_total - args.warmdown_iters)
        if (args.late_qat_threshold > 0 and step >= warmdown_start
                and lr_mul < args.late_qat_threshold and not casted_linear_cls._qat_enabled):
            casted_linear_cls._qat_enabled = True
            log0(f"Late QAT enabled at step {step} (lr_mul={lr_mul:.4f})")

        model.train()
        total_loss = 0.0
        timing_enabled = args.diagnostics_enabled and step <= args.diagnostics_timing_steps
        if timing_enabled and device.type == "cuda":
            torch.cuda.synchronize(device)
        step_t0 = time.perf_counter()
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
        if timing_enabled and device.type == "cuda":
            torch.cuda.synchronize(device)
        fetch_t1 = time.perf_counter()
        micro_batch = x.shape[0] // grad_accum_steps
        for micro_step in range(grad_accum_steps):
            x_micro = x[micro_step * micro_batch:(micro_step + 1) * micro_batch]
            y_micro = y[micro_step * micro_batch:(micro_step + 1) * micro_batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x_micro, y_micro)
                loss = loss / grad_accum_steps
            loss.backward()
            total_loss += loss.item()
        if timing_enabled and device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_t1 = time.perf_counter()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        muon_opt.step()
        adam_opt.step()
        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)
        if timing_enabled and device.type == "cuda":
            torch.cuda.synchronize(device)
        opt_t1 = time.perf_counter()
        if timing_enabled and master_process:
            diagnostics_payload["step_timings_ms"].append({
                "step": step,
                "fetch_ms": (fetch_t1 - step_t0) * 1000.0,
                "forward_backward_ms": (backward_t1 - fetch_t1) * 1000.0,
                "optimizer_ms": (opt_t1 - backward_t1) * 1000.0,
                "total_ms": (opt_t1 - step_t0) * 1000.0,
            })

        with torch.no_grad():
            sd = base_model.state_dict()
            ema_vals = [ema_state[k] for k in sd]
            live_vals = [sd[k].detach().float() for k in sd]
            torch._foreach_mul_(ema_vals, args.ema_decay)
            torch._foreach_add_(ema_vals, live_vals, alpha=1.0 - args.ema_decay)

        if args.swa_enabled and lr_mul < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"SWA started at step {step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1

        running_loss += total_loss
        loss_count += 1

        if step % args.train_log_every == 0 or step <= 10:
            avg_loss = running_loss / max(loss_count, 1)
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            log0(f"step {step:5d}/{args.iterations} | loss {avg_loss:.4f} | lr_mul {lr_mul:.4f} | "
                 f"mom {current_muon_momentum:.3f} | {steps_per_sec:.2f} steps/s | {elapsed:.0f}s")
            running_loss = 0.0
            loss_count = 0

        should_validate = (args.val_loss_every > 0 and step % args.val_loss_every == 0) or step == args.iterations
        if should_validate:
            val_loss, val_bpb = eval_val_sliding(
                args, model, val_tokens, val_tokens_np, bos_token_id,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                rank, world_size, device,
                seq_len=args.eval_seq_len, stride=args.eval_stride,
            )
            log0(f"step {step:5d} | val_loss {val_loss:.4f} | val_bpb {val_bpb:.4f}")

            if master_process and args.save_every > 0 and (step % args.save_every == 0 or step == args.iterations):
                ckpt_path = save_checkpoint(
                    model, step, val_bpb, args.ckpt_dir, config["arch_name"], args.seed,
                )
                log0(f"  Saved: {ckpt_path}")

        if args.max_wallclock_seconds > 0:
            elapsed = time.time() - t0
            if elapsed > args.max_wallclock_seconds and stop_after_step is None:
                stop_after_step = step
                log0(f"Wallclock limit reached ({elapsed:.0f}s), will stop after this step")

        if args.auto_save_seconds > 0:
            elapsed = time.time() - t0
            if elapsed > args.auto_save_seconds:
                log0(f"Auto-save triggered at step {step} ({elapsed:.0f}s elapsed)")
                if master_process:
                    rng_states = {
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state(),
                        "numpy": np.random.get_state(),
                        "python": random.getstate(),
                    }
                    stream = train_loader.stream
                    stream_state = (stream.idx, stream.pos)
                    ckpt_path = save_full_checkpoint(
                        model, step, 0.0, args.ckpt_dir, config["arch_name"], args.seed,
                        muon_opt, adam_opt, ema_state, swa_state, swa_count,
                        casted_linear_cls._qat_enabled,
                        rng_states=rng_states, stream_state=stream_state,
                    )
                    marker_path = os.path.join(args.ckpt_dir, f"CHAIN_RESUME_FROM_seed{args.seed}")
                    with open(marker_path, "w") as f:
                        f.write(ckpt_path + "\n")
                    log0(f"  Full checkpoint saved: {ckpt_path}")
                break

    # ─── Check if exited due to auto-save ────────────────────────────────
    chain_marker = os.path.join(args.ckpt_dir, f"CHAIN_RESUME_FROM_seed{args.seed}")
    if os.path.exists(chain_marker):
        log0("\nExiting for chained job resume (skipping post-training)")
        if distributed:
            dist.destroy_process_group()
        return

    if master_process and step >= schedule_total:
        complete_marker = os.path.join(args.ckpt_dir, f"TRAINING_COMPLETE_seed{args.seed}")
        with open(complete_marker, "w") as f:
            f.write(f"step={step}\n")

    # ─── Post-Training: Apply EMA ────────────────────────────────────────
    elapsed_total = time.time() - t0
    log0(f"\nTraining complete in {elapsed_total:.0f}s ({step} steps)")
    log0(f"Peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    log0(f"Steps/sec: {step / elapsed_total:.2f}")

    log0("\n=== Applying EMA weights ===")
    avg_state = {name: t.to(dtype=base_model.state_dict()[name].dtype) for name, t in ema_state.items()}
    if swa_state is not None and swa_count > 0:
        log0(f"SWA: averaging {swa_count} checkpoints with EMA")
        swa_avg = {k: (v / swa_count).to(device) for k, v in swa_state.items()}
        for name in avg_state:
            if name in swa_avg:
                dtype = avg_state[name].dtype
                avg_state[name] = (0.5 * avg_state[name].float() + 0.5 * swa_avg[name].float()).to(dtype)

    _load_model_state_dict_compat(base_model, avg_state, log0)

    if args.diagnostics_enabled and master_process:
        diagnostics_payload["final_parameter_summaries"] = _collect_parameter_summaries(base_model)
        diagnostics_payload["final_weight_diagnostics"] = base_model.get_diagnostics()

    val_loss_ema, val_bpb_ema = _eval_sliding_variant(
        args, model, val_tokens, val_tokens_np, bos_token_id,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        rank, world_size, device,
        seq_len=args.eval_seq_len, stride=args.eval_stride,
        xsa_eval=False, tapin_enabled=args.tapin_enabled,
    )
    log0(f"EMA BPB: {val_bpb_ema:.6f}")
    val_loss_ema_no_tapin, val_bpb_ema_no_tapin = None, None

    if master_process:
        torch.save(base_model.state_dict(), os.path.join(args.ckpt_dir, f"final_model_{config['arch_name']}_seed{args.seed}.pt"))
        log0("Saved raw EMA model")
    if args.diagnostics_enabled and master_process:
        probe_stats, tensor_dump = _collect_model_probe(
            base_model, diagnostics_x, diagnostics_y, device,
            args.diagnostics_top_k, args.diagnostics_dump_tokens,
        )
        diagnostics_payload["ema_probe"] = probe_stats
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_dump, diagnostics_tensor_path)
        log0(f"Saved diagnostics tensor dump: {diagnostics_tensor_path}")

    # ─── GPTQ Calibration (optional) ─────────────────────────────────────
    gptq_enabled = bool(int(os.environ.get("GPTQ_ENABLED", "0")))
    hessians = None
    if gptq_enabled:
        log0("\n=== GPTQ: generating autoregressive calibration data ===")
        calib_seqs = generate_autoregressive_calib(
            base_model, device, num_seqs=64, seq_len=args.train_seq_len,
            vocab_size=args.vocab_size, temperature=0.8, batch_size=8, seed=args.seed,
        )
        log0(f"GPTQ: generated {len(calib_seqs)} sequences, collecting hessians...")
        hessians = collect_hessians_from_tokens(base_model, calib_seqs, device)
        log0(f"GPTQ: collected hessians for {len(hessians)} layers")
        if args.diagnostics_enabled and master_process:
            diagnostics_payload["hessian_summary"] = _summarize_hessians(hessians)

    # ─── Quantization + Artifact Creation ────────────────────────────────
    log0("\n=== Quantizing to int6 + zstd-22 ===")
    sd_cpu = _prepare_state_dict_for_export(base_model.state_dict(), config)
    code_paths = _code_paths_for_artifact(Path(__file__), args.tapin_enabled)
    raw_code_bytes = sum(path.stat().st_size for path in code_paths)
    compressed_code_bytes = _estimate_compressed_code_bytes(code_paths)
    quant_variant_specs = _build_quant_variant_specs("baseline_int6")
    quant_sweep_results: list[dict[str, object]] = []
    quantized_baseline_probe_stats = None
    quantized_baseline_tensor_dump = None
    val_loss_q = None
    val_bpb_q = None
    artifact_path = None
    block_types = list(base_model._block_types)

    variant = quant_variant_specs[0]
    label = str(variant["label"])
    bit_rules = variant.get("bit_rules", [])
    log0(f"\n=== Preparing Quant Variant ({label}) ===")
    if args.diagnostics_enabled and master_process:
        quant_result, quant_meta, quant_stats = mixed_quantize(
            sd_cpu, hessians=hessians, return_stats=True, bit_rules=bit_rules,
            require_hessians=gptq_enabled,
            default_matrix_bits=int(variant.get("default_matrix_bits", 6)),
        )
        diagnostics_payload["quantization_summary"] = _annotate_quant_stats(
            quant_stats, block_types
        )
    else:
        quant_result, quant_meta = mixed_quantize(
            sd_cpu, hessians=hessians, bit_rules=bit_rules,
            require_hessians=gptq_enabled,
            default_matrix_bits=int(variant.get("default_matrix_bits", 6)),
        )

    quant_blob = compress_artifact(quant_result, quant_meta)
    legacy_artifact_path = os.path.join(
        args.ckpt_dir,
        f"final_model_{config['arch_name']}_seed{args.seed}.int6.ptz",
    )
    artifact_bytes = len(quant_blob)
    estimated_total_bytes = artifact_bytes + compressed_code_bytes
    fits = estimated_total_bytes <= ARTIFACT_CAP_BYTES
    log0(f"Artifact cap: {ARTIFACT_CAP_BYTES:,} bytes")
    log0(f"Raw code bytes: {raw_code_bytes:,}")
    log0(f"Compressed code estimate: {compressed_code_bytes:,}")
    log0(
        f"{label}: model={artifact_bytes:,} bytes, compressed_code_est={compressed_code_bytes:,}, "
        f"estimated_total={estimated_total_bytes:,}, fits={fits}"
    )

    if master_process:
        with open(legacy_artifact_path, "wb") as f:
            f.write(quant_blob)
        artifact_path = legacy_artifact_path
    if distributed:
        dist.barrier()

    deq_state = dequantize_mixed(quant_result, quant_meta, sd_cpu)
    eval_model = hybrid_gdn_cls(config, args.vocab_size).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, casted_linear_cls):
            m.float()
    for name, p in eval_model.named_parameters():
        if p.ndim <= 1:
            p.data = p.data.float()
    _load_model_state_dict_compat(eval_model, deq_state, log0)

    val_loss_q, val_bpb_q = _eval_sliding_variant(
        args, eval_model, val_tokens, val_tokens_np, bos_token_id,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        rank, world_size, device,
        seq_len=args.eval_seq_len, stride=args.eval_stride,
        xsa_eval=False, tapin_enabled=args.tapin_enabled,
    )
    log0(f"{label}: Quantized BPB: {val_bpb_q:.6f}")

    if args.diagnostics_enabled and master_process:
        q_probe_stats, q_tensor_dump = _collect_model_probe(
            eval_model, diagnostics_x, diagnostics_y, device,
            args.diagnostics_top_k, args.diagnostics_dump_tokens,
        )
        quantized_baseline_probe_stats = q_probe_stats
        quantized_baseline_tensor_dump = q_tensor_dump

    quant_sweep_results.append({
        "label": label,
        "artifact_bytes": int(artifact_bytes),
        "raw_code_bytes": int(raw_code_bytes),
        "compressed_code_bytes": int(compressed_code_bytes),
        "estimated_total_bytes": int(estimated_total_bytes),
        "bit_rules": bit_rules,
        "bpb": float(val_bpb_q),
        "val_loss": float(val_loss_q),
    })

    if args.diagnostics_enabled and master_process and quantized_baseline_probe_stats is not None:
        diagnostics_payload["quantized_probe"] = quantized_baseline_probe_stats
        diagnostics_payload["quant_sweep_results"] = quant_sweep_results
        diagnostics_payload["results"] = {
            "ema_bpb": float(val_bpb_ema),
            "quantized_bpb": float(val_bpb_q),
            "quantization_delta_bpb": float(val_bpb_q - val_bpb_ema),
            "elapsed_total_s": float(elapsed_total),
            "steps": int(step),
        }
        existing = {}
        if diagnostics_tensor_path.exists():
            existing = torch.load(diagnostics_tensor_path, map_location="cpu", weights_only=False)
        existing["quantized_hidden"] = quantized_baseline_tensor_dump["hidden"]
        existing["quantized_entropy"] = quantized_baseline_tensor_dump["entropy"]
        existing["quantized_preds"] = quantized_baseline_tensor_dump["preds"]
        existing["quantized_top_logits_values"] = quantized_baseline_tensor_dump["top_logits_values"]
        existing["quantized_top_logits_indices"] = quantized_baseline_tensor_dump["top_logits_indices"]
        torch.save(existing, diagnostics_tensor_path)

    log0(f"\n{'='*80}")
    log0(f"FINAL RESULTS — {config['arch_name']} seed={args.seed}")
    log0(f"  Training: {step} steps, {elapsed_total:.0f}s")
    log0(f"  EMA BPB:       {val_bpb_ema:.6f}")
    log0(f"  Quantized BPB: {val_bpb_q:.6f}")
    log0(f"  Artifact:      {artifact_path}")
    log0(f"{'='*80}")
    if args.diagnostics_enabled and master_process:
        _write_diagnostics_json(diagnostics_path, diagnostics_payload)
        log0(f"Saved diagnostics summary: {diagnostics_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
