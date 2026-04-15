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

import copy
import glob
import io
import math
import os
import random
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn.functional as F
import zstandard
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Safety guard: if dynamo is ever invoked on code paths containing GDN layers
# (e.g. FLA internal usage), each unique `layer_idx` integer attribute would be
# treated as a static guard and trigger a separate recompilation. The default
# limit=8 would cause layers 8-9 to permanently fall back to eager mode.
# We no longer call torch.compile on the eval forward (see evaluate_sliding_window),
# so this guard is mainly defensive. 64 > 10 GDN layers, so it's always safe.
torch._dynamo.config.recompile_limit = 64

sys.path.insert(0, str(Path(__file__).resolve().parent))
from architectures import HybridGDN, CastedLinear
from configs import get_config


# ─── Hyperparameters ──────────────────────────────────────────────────────────

class Hyperparameters:
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
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
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
    xsa_eval = bool(int(os.environ.get("XSA_EVAL", "0")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

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
    model: nn.Module,
    val_tokens: Tensor,
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

    if xsa_eval and hasattr(base_model, 'set_xsa'):
        base_model.set_xsa(False)

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
    hessians = {}
    hooks = []
    for name, module in hessian_model.named_modules():
        if isinstance(module, CastedLinear):
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
        return quantize_int6_per_row(t32)
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
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


def mixed_quantize(state_dict: dict[str, Tensor], hessians: dict[str, Tensor] | None = None) -> tuple[dict[str, Tensor], dict[str, object]]:
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if any(p in name for p in CONTROL_PATTERNS):
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if not t.is_floating_point():
            result[name] = t
            meta[name] = "passthrough"
            continue
        if t.numel() <= 65536:
            result[name] = t.to(torch.float16)
            meta[name] = "passthrough"
            continue
        if t.ndim == 2 and t.numel() > 65536:
            H = hessians.get(name) if hessians else None
            q, s = quantize_int6_gptq(t, hessian=H) if H is not None else quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_int8_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def dequantize_mixed(result: dict[str, Tensor], meta: dict[str, object],
                     template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info == "passthrough":
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

    # Direction-5: optional QK_GAIN_INIT override from env
    if args.qk_gain_init_override:
        config["qk_gain_init"] = float(args.qk_gain_init_override)

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = max(1, 8 // world_size)
    master_process = rank == 0

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
    log0(f"Arch: {config['arch_name']} (ARCH_MODE={args.arch_mode})")
    log0(f"Seed: {args.seed}, Max steps: {args.iterations}, Warmdown: {args.warmdown_iters}")
    log0(f"Train seq_len: {args.train_seq_len}, Wallclock budget: {args.max_wallclock_seconds}s")
    log0(f"QK_GAIN_INIT: {config.get('qk_gain_init', 1.5)}")
    log0(f"World size: {world_size}, Grad accum: {grad_accum_steps}")
    log0(f"EMA decay: {args.ema_decay}, SWA: {args.swa_enabled} (every {args.swa_every})")
    log0(f"Late QAT threshold: {args.late_qat_threshold}")
    log0(f"MuonEq-R: enabled")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    assert int(sp.vocab_size()) == args.vocab_size

    val_tokens = load_validation_tokens(args.val_files, args.eval_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"Validation tokens: {val_tokens.numel()-1:,}")

    _t0 = time.time()
    model = HybridGDN(config, args.vocab_size)
    model = model.to(device).bfloat16()
    log0(f"Model built in {time.time()-_t0:.1f}s")

    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, p in model.named_parameters():
        if p.ndim <= 1:
            p.data = p.data.float()

    param_counts = model.count_params()
    log0(f"Parameters: {param_counts}")
    log0(f"Total params: {param_counts['total']:,}")

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
        model.load_state_dict({k: v.to(device) for k, v in base_sd.items()}, strict=True)
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

    def lr_schedule(step: int) -> float:
        warmdown_start = args.iterations - args.warmdown_iters
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        elif step >= warmdown_start:
            progress = (step - warmdown_start) / args.warmdown_iters
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
            CastedLinear._qat_enabled = True
            log0("  Restored QAT enabled state")
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

        warmdown_start = args.iterations - args.warmdown_iters
        if (args.late_qat_threshold > 0 and step >= warmdown_start
                and lr_mul < args.late_qat_threshold and not CastedLinear._qat_enabled):
            CastedLinear._qat_enabled = True
            log0(f"Late QAT enabled at step {step} (lr_mul={lr_mul:.4f})")

        model.train()
        total_loss = 0.0
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
        micro_batch = x.shape[0] // grad_accum_steps
        for micro_step in range(grad_accum_steps):
            x_micro = x[micro_step * micro_batch:(micro_step + 1) * micro_batch]
            y_micro = y[micro_step * micro_batch:(micro_step + 1) * micro_batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x_micro, y_micro)
                loss = loss / grad_accum_steps
            loss.backward()
            total_loss += loss.item()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        muon_opt.step()
        adam_opt.step()
        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)

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

        if step % args.val_loss_every == 0 or step == args.iterations:
            val_loss, val_bpb = eval_val_sliding(
                model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                rank, world_size, device,
                seq_len=args.eval_seq_len, stride=args.eval_stride,
                xsa_eval=args.xsa_eval,
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
                        CastedLinear._qat_enabled,
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

    effective_total = args.total_iterations if args.total_iterations > 0 else args.iterations
    if master_process and step >= effective_total:
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
        swa_avg = {k: v / swa_count for k, v in swa_state.items()}
        for name in avg_state:
            if name in swa_avg:
                dtype = avg_state[name].dtype
                avg_state[name] = (0.5 * avg_state[name].float() + 0.5 * swa_avg[name].float()).to(dtype)

    base_model.load_state_dict(avg_state, strict=True)

    val_loss_ema, val_bpb_ema = eval_val_sliding(
        model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        rank, world_size, device,
        seq_len=args.eval_seq_len, stride=args.eval_stride,
        xsa_eval=False,
    )
    log0(f"EMA BPB (no XSA): {val_bpb_ema:.6f}")

    if master_process:
        torch.save(base_model.state_dict(), os.path.join(args.ckpt_dir, f"final_model_{config['arch_name']}_seed{args.seed}.pt"))
        log0("Saved raw EMA model")

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

    # ─── Quantization + Artifact Creation ────────────────────────────────
    log0("\n=== Quantizing to int6 + zstd-22 ===")
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    quant_result, quant_meta = mixed_quantize(sd_cpu, hessians=hessians)

    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)

    artifact_path = os.path.join(args.ckpt_dir, f"final_model_{config['arch_name']}_seed{args.seed}.int6.ptz")
    if master_process:
        with open(artifact_path, "wb") as f:
            f.write(quant_blob)
        artifact_bytes = len(quant_blob)
        log0(f"Artifact: {artifact_bytes:,} bytes ({artifact_bytes / 1024 / 1024:.2f} MB)")
        if artifact_bytes > 16 * 1024 * 1024:
            log0(f"WARNING: Artifact exceeds 16MB budget by {(artifact_bytes - 16*1024*1024) / 1024:.1f} KB")

    # ─── Roundtrip Validation ────────────────────────────────────────────
    log0("\n=== Roundtrip Validation (quantized model) ===")
    if distributed:
        dist.barrier()

    with open(artifact_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob_disk)),
        map_location="cpu",
    )
    deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu)

    eval_model = HybridGDN(config, args.vocab_size).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    for name, p in eval_model.named_parameters():
        if p.ndim <= 1:
            p.data = p.data.float()
    eval_model.load_state_dict(deq_state, strict=True)

    val_loss_q, val_bpb_q = eval_val_sliding(
        eval_model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        rank, world_size, device,
        seq_len=args.eval_seq_len, stride=args.eval_stride,
        xsa_eval=False,
    )
    log0(f"Quantized BPB (no XSA): {val_bpb_q:.6f}")
    log0(f"Quantization degradation: {val_bpb_q - val_bpb_ema:+.6f}")

    block_types = eval_model._block_types
    if any(bt in ("swa", "swa_shared") for bt in block_types):
        val_loss_qx, val_bpb_qx = eval_val_sliding(
            eval_model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            rank, world_size, device,
            seq_len=args.eval_seq_len, stride=args.eval_stride,
            xsa_eval=True,
        )
        log0(f"Quantized BPB (XSA-all): {val_bpb_qx:.6f}")

    log0(f"\n{'='*80}")
    log0(f"FINAL RESULTS — {config['arch_name']} seed={args.seed}")
    log0(f"  Training: {step} steps, {elapsed_total:.0f}s")
    log0(f"  EMA BPB:       {val_bpb_ema:.6f}")
    log0(f"  Quantized BPB: {val_bpb_q:.6f}")
    if any(bt in ("swa", "swa_shared") for bt in block_types):
        log0(f"  XSA BPB:       {val_bpb_qx:.6f}")
    log0(f"  Artifact:      {artifact_path}")
    log0(f"{'='*80}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
