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
from collections import OrderedDict
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    flash_attn_3_func = None
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    import zlib as _zlib_module
    import warnings
    _COMPRESSOR = "zlib"
    warnings.warn("zstandard not found — falling back to zlib. Artifact will be ~1.5MB larger! pip install zstandard")

if os.environ.get("TORCHDYNAMO_SUPPRESS_ERRORS", "0") == "1":
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
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
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    trigram_enabled = bool(int(os.environ.get("TRIGRAM", "0")))  # TrigramHash (off by default, risky)
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))  # XSA on ALL layers (our novel contribution)
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))  # VRL with sigmoid gates (off by default, risky)
    attn_scale_init = float(os.environ.get("ATTN_SCALE_INIT", 1.0))
    mlp_scale_init = float(os.environ.get("MLP_SCALE_INIT", 1.0))
    resid_mix_x_init = float(os.environ.get("RESID_MIX_X_INIT", 1.0))
    resid_mix_x0_init = float(os.environ.get("RESID_MIX_X0_INIT", 0.0))
    complement_alpha = float(os.environ.get("COMPLEMENT_ALPHA", "0"))
    ngram_eval_order = int(os.environ.get("NGRAM_EVAL_ORDER", 0))
    ngram_eval_min_order = int(os.environ.get("NGRAM_EVAL_MIN_ORDER", 2))
    ngram_eval_alpha = float(os.environ.get("NGRAM_EVAL_ALPHA", 0.30))
    ngram_eval_adaptive = bool(int(os.environ.get("NGRAM_EVAL_ADAPTIVE", "1")))
    ngram_eval_alpha_min = float(os.environ.get("NGRAM_EVAL_ALPHA_MIN", 0.05))
    ngram_eval_alpha_max = float(os.environ.get("NGRAM_EVAL_ALPHA_MAX", 0.60))
    ngram_eval_entropy_center = float(os.environ.get("NGRAM_EVAL_ENTROPY_CENTER", 4.0))
    ngram_eval_entropy_scale = float(os.environ.get("NGRAM_EVAL_ENTROPY_SCALE", 2.0))
    ngram_eval_min_count = int(os.environ.get("NGRAM_EVAL_MIN_COUNT", 2))
    ngram_eval_buckets = int(os.environ.get("NGRAM_EVAL_BUCKETS", 4_194_304))
    ngram_eval_max_seconds = float(os.environ.get("NGRAM_EVAL_MAX_SECONDS", 0.0))
    ngram_entropy_shift = bool(int(os.environ.get("NGRAM_ENTROPY_SHIFT", "0")))
    ngram_order_mults_str = os.environ.get("NGRAM_ORDER_MULTS", "")
    cubric_cadence = int(os.environ.get("CUBRIC_CADENCE", 0))
    skip_final_eval = bool(int(os.environ.get("SKIP_FINAL_EVAL", "0")))
    post_ema_diagnostic = bool(int(os.environ.get("POST_EMA_DIAGNOSTIC", "1")))
    compile_enabled = bool(int(os.environ.get("COMPILE_ENABLED", "1")))
    compile_mode = os.environ.get("COMPILE_MODE", "").strip()
    compile_fullgraph = bool(int(os.environ.get("COMPILE_FULLGRAPH", "1")))
    mlp_kernel_mode = os.environ.get("MLP_KERNEL_MODE", "").strip().lower()
    loader_mode = os.environ.get("LOADER_MODE", "sequential").strip().lower()
    coprime_max_loaded_shards = int(os.environ.get("COPRIME_MAX_LOADED_SHARDS", 1))
    coprime_shards_per_batch = int(os.environ.get("COPRIME_SHARDS_PER_BATCH", 4))
    coprime_shard_hold_steps = int(os.environ.get("COPRIME_SHARD_HOLD_STEPS", 64))
    slot_enabled = bool(int(os.environ.get("SLOT_ENABLED", "0")))
    slot_steps = int(os.environ.get("SLOT_STEPS", "8"))
    slot_lr = float(os.environ.get("SLOT_LR", "0.005"))
    slot_max_windows = int(os.environ.get("SLOT_MAX_WINDOWS", "512"))  # 0=all; 512=fast ablation


def maybe_compile(fn_or_module, *, enabled: bool, fullgraph: bool, mode: str = ""):
    if not enabled:
        return fn_or_module
    kwargs = dict(dynamic=False, fullgraph=fullgraph)
    if mode:
        kwargs["mode"] = mode
    return torch.compile(fn_or_module, **kwargs)


if triton is not None:
    @triton.jit
    def _leaky_relu_sq_forward_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a = tl.where(x >= 0, x, 0.5 * x)
        y = a * a
        tl.store(y_ptr + offsets, y, mask=mask)

    @triton.jit
    def _leaky_relu_sq_backward_kernel(x_ptr, grad_out_ptr, grad_in_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a = tl.where(x >= 0, x, 0.5 * x)
        slope = tl.where(x >= 0, 1.0, 0.5)
        grad_in = grad_out * (2.0 * a * slope)
        tl.store(grad_in_ptr + offsets, grad_in, mask=mask)


class TritonLeakyReluSqFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        if triton is None or not x.is_cuda:
            a = F.leaky_relu(x, negative_slope=0.5)
            ctx.save_for_backward(x)
            return a.square()
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        n_elements = x_contig.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _leaky_relu_sq_forward_kernel[grid](x_contig, y, n_elements, BLOCK_SIZE=1024)
        ctx.save_for_backward(x_contig)
        return y

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor]:
        (x,) = ctx.saved_tensors
        if triton is None or not grad_out.is_cuda:
            a = F.leaky_relu(x, negative_slope=0.5)
            slope = torch.where(x >= 0, torch.ones_like(x), torch.full_like(x, 0.5))
            return (grad_out * (2.0 * a * slope),)
        grad_out_contig = grad_out.contiguous()
        grad_in = torch.empty_like(grad_out_contig)
        n_elements = grad_out_contig.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _leaky_relu_sq_backward_kernel[grid](x, grad_out_contig, grad_in, n_elements, BLOCK_SIZE=1024)
        return (grad_in,)


def leaky_relu_sq(x: Tensor, kernel_mode: str = "") -> Tensor:
    if kernel_mode == "triton_act":
        return TritonLeakyReluSqFn.apply(x)
    a = F.leaky_relu(x, negative_slope=0.5)
    return a.square()

class TrainNgramTracker:
    """Complementary training: track bigram stats, downweight tokens n-grams can predict."""
    def __init__(self, vocab_size: int, device: torch.device, complement_alpha: float = 0.5):
        self.V = vocab_size
        self.alpha = complement_alpha
        self.bi_counts = torch.zeros(vocab_size, vocab_size, device=device, dtype=torch.float32)
        self.bi_totals = torch.zeros(vocab_size, device=device, dtype=torch.float32)
    @torch.no_grad()
    def update(self, x: Tensor, y: Tensor):
        xf = x.reshape(-1)
        yf = y.reshape(-1)
        ones = torch.ones(xf.numel(), device=xf.device, dtype=torch.float32)
        self.bi_counts.reshape(-1).scatter_add_(0, xf * self.V + yf, ones)
        self.bi_totals.scatter_add_(0, xf, ones)
    def get_weights(self, x: Tensor, y: Tensor) -> Tensor:
        xf = x.reshape(-1)
        yf = y.reshape(-1)
        total = self.bi_totals[xf]
        count = self.bi_counts.reshape(-1)[xf * self.V + yf]
        ngram_prob = count / (total + 1)
        return (1.0 - self.alpha * ngram_prob).clamp(min=0.1)

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
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda",
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
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if "f1_corr_in" in name or "f1_corr_out" in name:
        return "aux"
    if "qo_bank" in name or "kv_bank" in name:
        return "attn"
    if "mlp_up_bank" in name or "mlp_down_bank" in name:
        return "mlp"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
# GPTQ: Hessian-aware quantization with column-wise error compensation
def _find_best_row_scales(W: Tensor, clip_range: int = 31) -> Tensor:
    t32 = W.float()
    best_s = t32.abs().amax(dim=1) / clip_range
    best_s = best_s.clamp_min(1.0 / clip_range)
    best_err = torch.full((t32.shape[0],), float('inf'))
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range)
        q = torch.clamp(torch.round(t32 / s[:, None]), -clip_range, clip_range)
        recon = q * s[:, None]
        err = (t32 - recon).pow(2).mean(dim=1)
        improved = err < best_err
        best_s[improved] = s[improved]
        best_err[improved] = err[improved]
    return best_s
def gptq_quantize_weight(W: Tensor, H: Tensor, clip_range: int = 31,
                          block_size: int = 64, percdamp: float = 0.002) -> tuple[Tensor, Tensor]:
    """GPTQ: quantize weight matrix W using Hessian H = X^T X for error compensation.
    Returns (quantized_int8, scale_fp16) in int6 range [-clip_range, clip_range]."""
    W = W.float().clone()
    rows, cols = W.shape
    row_scale = _find_best_row_scales(W, clip_range)
    H = H.float().clone()
    damp = percdamp * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag())
    invperm = torch.argsort(perm)
    W = W[:, perm]
    H = H[perm][:, perm]
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
    except torch._C._LinAlgError:
        Hinv = torch.diag(1.0 / H.diag().clamp_min(1e-6))
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros_like(W_block)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            h_inv_jj = Hinv_block[j, j].clamp_min(1e-8)
            q_col = torch.clamp(torch.round(w_col / row_scale), -clip_range, clip_range)
            deq_col = q_col * row_scale
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - deq_col) / h_inv_jj
            Err[:, j] = err
            if j + 1 < i2 - i1:
                W_block[:, j + 1:] -= err.unsqueeze(1) * Hinv_block[j, j + 1:].unsqueeze(0)
        if i2 < cols:
            W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    Q = Q[:, invperm]
    return Q, row_scale.to(torch.float16)
def gptq_calibrate(model: nn.Module, train_pattern: str, device: torch.device,
                   n_samples: int = 256, seq_len: int = 2048) -> dict[str, Tensor]:
    """Collect Hessian H = X^T X for each linear layer using training data."""
    hessians: dict[str, Tensor] = {}
    n_seen: dict[str, int] = {}
    hooks = []
    def make_hook(name: str):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=torch.float32)
                n_seen[name] = 0
            hessians[name].addmm_(x.t(), x)
            n_seen[name] += x.shape[0]
        return hook_fn
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, CastedLinear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    stream = TokenStream(train_pattern)
    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            tokens = stream.take(seq_len + 1).to(device=device, dtype=torch.int64)
            x = tokens[:-1].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model.forward_logits(x)
    for h in hooks:
        h.remove()
    for name in hessians:
        hessians[name] /= max(n_seen[name], 1)
    model.train()
    return hessians
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
def mixed_quantize_int6_gptq(state_dict: dict[str, Tensor], int6_cats: set[str],
                              hessians: dict[str, Tensor]) -> tuple[dict, dict]:
    """Like mixed_quantize_int6 but uses GPTQ for int6 categories when Hessian available."""
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
        if cat in int6_cats and t.ndim == 2:
            module_name = name.rsplit(".weight", 1)[0] if name.endswith(".weight") else name
            H = hessians.get(module_name)
            if H is not None and H.shape[0] == t.shape[1]:
                q, s = gptq_quantize_weight(t, H.cpu())
                gptq_count += 1
            else:
                q, s = quantize_int6_per_row(t)
                naive_count += 1
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        elif cat in int6_cats and t.ndim >= 1:
            t_2d = t.reshape(-1, t.shape[-1]) if t.ndim > 2 else t
            q, s = quantize_int6_per_row(t_2d)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
            naive_count += 1
        else:
            t_q = t.reshape(-1, t.shape[-1]) if t.ndim > 2 else t
            q, s = quantize_float_tensor(t_q)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
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
        if s.ndim > 0:
            val = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            val = (q.float() * float(s.item())).to(orig_dtype)
        out[name] = val.reshape(orig.shape) if val.shape != orig.shape else val
    return out

# --- Data loading ---

SHARD_HEADER_DTYPE = np.dtype("<i4")
SHARD_TOKEN_DTYPE = np.dtype("<u2")
SHARD_HEADER_WORDS = 256
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_BYTES = SHARD_HEADER_WORDS * SHARD_HEADER_DTYPE.itemsize

def read_data_shard_header(file: Path) -> dict[str, int]:
    header = np.fromfile(file, dtype=SHARD_HEADER_DTYPE, count=SHARD_HEADER_WORDS)
    if header.size != SHARD_HEADER_WORDS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    return {"num_tokens": int(header[2])}

def load_data_shard(file: Path) -> Tensor:
    header = read_data_shard_header(file)
    num_tokens = header["num_tokens"]
    expected_size = SHARD_HEADER_BYTES + num_tokens * SHARD_TOKEN_DTYPE.itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype=SHARD_TOKEN_DTYPE, count=num_tokens, offset=SHARD_HEADER_BYTES)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

def choose_coprime_stride(modulus: int, salt: int) -> int:
    if modulus <= 1:
        return 1
    candidate = abs(salt) % modulus
    if candidate == 0:
        candidate = 1
    while math.gcd(candidate, modulus) != 1:
        candidate += 1
        if candidate >= modulus:
            candidate = 1
    return candidate

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
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
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
    def describe(self) -> str:
        return f"loader:sequential shards:{len(self.stream.files)}"
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

class CoprimeDistributedTokenLoader:
    """Shard-aware block sampler with deterministic coprime walks."""
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        seq_len: int,
        seed: int,
        max_loaded_shards: int,
        shards_per_batch: int,
        shard_hold_steps: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.seq_len = seq_len
        self.seed = seed
        self.token_offsets = torch.arange(seq_len + 1, dtype=torch.int64)
        self.cache: OrderedDict[Path, Tensor] = OrderedDict()
        files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.shards: list[dict[str, int | Path]] = []
        for shard_idx, file in enumerate(files):
            header = read_data_shard_header(file)
            num_blocks = (header["num_tokens"] - 1) // seq_len
            if num_blocks <= 0:
                continue
            self.shards.append(
                {
                    "file": file,
                    "num_blocks": num_blocks,
                    "offset": (seed * 131 + shard_idx * 17) % num_blocks,
                    "stride": choose_coprime_stride(num_blocks, seed * 29 + shard_idx * 7 + 1),
                }
            )
        if not self.shards:
            raise ValueError(f"No usable shards found for seq_len={seq_len}")
        self.num_shards = len(self.shards)
        self.max_loaded_shards = max(1, min(max_loaded_shards, self.num_shards))
        self.shards_per_batch = max(1, min(shards_per_batch, self.num_shards))
        self.shard_hold_steps = max(1, shard_hold_steps)
        self.batch_shard_stride = choose_coprime_stride(self.num_shards, seed * 41 + 3)
        self.batch_idx = 0
        self.shard_visits = [0 for _ in range(self.num_shards)]
    def _get_tokens(self, file: Path) -> Tensor:
        cached = self.cache.get(file)
        if cached is not None:
            self.cache.move_to_end(file)
            return cached
        # CPU advanced indexing is not implemented for uint16, so cache coprime-loader
        # shards in int32 and cast to int64 only after batch assembly.
        tokens = load_data_shard(file).to(dtype=torch.int32)
        if len(self.cache) >= self.max_loaded_shards:
            self.cache.popitem(last=False)
        self.cache[file] = tokens
        return tokens
    def _sample_sequences(self, shard_idx: int, count: int) -> Tensor:
        shard = self.shards[shard_idx]
        num_blocks = int(shard["num_blocks"])
        offset = int(shard["offset"])
        stride = int(shard["stride"])
        visits = self.shard_visits[shard_idx]
        block_ids = (
            offset
            + (visits + torch.arange(count, dtype=torch.int64)) * stride
        ) % num_blocks
        self.shard_visits[shard_idx] += count
        token_starts = block_ids * self.seq_len
        gather_idx = token_starts.unsqueeze(1) + self.token_offsets.unsqueeze(0)
        tokens = self._get_tokens(shard["file"])
        return tokens[gather_idx]
    def describe(self) -> str:
        total_blocks = sum(int(shard["num_blocks"]) for shard in self.shards)
        return (
            f"loader:coprime shards:{self.num_shards} blocks:{total_blocks} "
            f"seq_len:{self.seq_len} shards_per_batch:{self.shards_per_batch} "
            f"cache:{self.max_loaded_shards} batch_stride:{self.batch_shard_stride} "
            f"hold_steps:{self.shard_hold_steps}"
        )
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        if seq_len != self.seq_len:
            raise ValueError(f"Coprime loader was built for seq_len={self.seq_len}, got {seq_len}")
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        if local_tokens % seq_len != 0:
            raise ValueError(
                f"TRAIN_BATCH_TOKENS={global_tokens} does not divide into full local sequences "
                f"for WORLD_SIZE={self.world_size}, GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
            )
        local_seqs = local_tokens // seq_len
        active_shards = min(self.shards_per_batch, self.num_shards, local_seqs)
        if active_shards <= 0:
            raise ValueError(f"No active shards available for local_seqs={local_seqs}")
        seqs_per_shard = local_seqs // active_shards
        seq_remainder = local_seqs % active_shards
        hold_idx = self.batch_idx // self.shard_hold_steps
        shard_start = ((hold_idx * self.world_size) + self.rank) * self.batch_shard_stride
        chunks: list[Tensor] = []
        for shard_slot in range(active_shards):
            count = seqs_per_shard + (1 if shard_slot < seq_remainder else 0)
            if count <= 0:
                continue
            shard_idx = (shard_start + shard_slot * self.batch_shard_stride) % self.num_shards
            chunks.append(self._sample_sequences(shard_idx, count))
        self.batch_idx += 1
        local = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        local = local.to(dtype=torch.int64)
        x = local[:, :-1]
        y = local[:, 1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

def build_train_loader(args: Hyperparameters, rank: int, world_size: int, device: torch.device):
    if args.loader_mode == "sequential":
        return DistributedTokenLoader(args.train_files, rank, world_size, device)
    if args.loader_mode == "coprime":
        return CoprimeDistributedTokenLoader(
            args.train_files,
            rank,
            world_size,
            device,
            seq_len=args.train_seq_len,
            seed=args.seed,
            max_loaded_shards=args.coprime_max_loaded_shards,
            shards_per_batch=args.coprime_shards_per_batch,
            shard_hold_steps=args.coprime_shard_hold_steps,
        )
    raise ValueError(f"Unknown LOADER_MODE={args.loader_mode!r}")

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
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
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
        gated_attention: bool = False,
        value_residual: bool = False,
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
        # Gated attention and value residual (non-banked small params)
        self.gated_attention = gated_attention
        if gated_attention:
            self.attn_gate = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)
        self.value_residual = value_residual
        if value_residual:
            self.vrl_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))  # sigmoid gate (PR #569 style)
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
    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, v_embed: Tensor | None = None, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v if self.value_residual else None
        if self.value_residual and v0 is not None:
            alpha = torch.sigmoid(self.vrl_alpha.to(dtype=v.dtype))
            v = v + alpha * v0  # sigmoid-gated residual (PR #569 style)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if flash_attn_3_func is not None:
            q_attn, k_attn, v_attn = q, k, v
            if q_attn.dtype not in (torch.float16, torch.bfloat16):
                q_attn = q_attn.to(torch.bfloat16)
                k_attn = k_attn.to(torch.bfloat16)
                v_attn = v_attn.to(torch.bfloat16)
            y = flash_attn_3_func(q_attn, k_attn, v_attn, causal=True)
        else:
            qh = q.transpose(1, 2)
            kh = k.transpose(1, 2)
            vh = v.transpose(1, 2)
            if self.num_heads != self.num_kv_heads:
                repeat = self.num_heads // self.num_kv_heads
                kh = kh.repeat_interleave(repeat, dim=1)
                vh = vh.repeat_interleave(repeat, dim=1)
            y = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True).transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.gated_attention:
            # gate shape: (bsz, seqlen, num_heads) -> (bsz, seqlen, num_heads, 1) for B,T,H,D layout
            gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
            y = y * gate
        y = y.reshape(bsz, seqlen, dim)
        return F.linear(y, out_w.to(x.dtype)), raw_v

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int, trigram: bool = False):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self._trigram = trigram
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
    def trigram_hash(self, tokens: Tensor) -> Tensor:
        """Hash (t-2, t-1, t) trigrams into same embedding table. Zero extra params."""
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., :2] = mod
        out[..., 2:] = (36313 * t[..., 2:] ^ 27191 * t[..., 1:-1] ^ 51497 * t[..., :-2]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self._trigram:
            h = h + self.embed(self.trigram_hash(token_ids))
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
        self.kernel_mode = os.environ.get("MLP_KERNEL_MODE", "").strip().lower()
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        x = F.linear(x, up_w.to(x.dtype))
        x = leaky_relu_sq(x, kernel_mode=self.kernel_mode)
        return F.linear(x, down_w.to(x.dtype))

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
        dtg: bool = False,
        gated_attention: bool = False,
        value_residual: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        gated_attention=gated_attention, value_residual=value_residual)
        self.mlp = MLP(dim, mlp_mult)
        attn_scale_init = float(os.environ.get("ATTN_SCALE_INIT", "1.0"))
        mlp_scale_init = float(os.environ.get("MLP_SCALE_INIT", "1.0"))
        resid_mix_x_init = float(os.environ.get("RESID_MIX_X_INIT", "1.0"))
        resid_mix_x0_init = float(os.environ.get("RESID_MIX_X0_INIT", "0.0"))
        self.attn_scale = nn.Parameter(torch.full((dim,), attn_scale_init, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.full((dim,), mlp_scale_init, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack(
                (
                    torch.full((dim,), resid_mix_x_init, dtype=torch.float32),
                    torch.full((dim,), resid_mix_x0_init, dtype=torch.float32),
                )
            )
        )
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None
    def forward(self, x: Tensor, x0: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor, out_w: Tensor, up_w: Tensor, down_w: Tensor, v_embed: Tensor | None = None, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, raw_v = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, q_w, k_w, v_w, out_w, v_embed=v_embed, v0=v0)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out, raw_v

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
        dtg: bool = False,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
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
        self.value_residual = value_residual
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim, trigram=bool(int(os.environ.get("TRIGRAM", "0")))) if bigram_vocab_size > 0 else None
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
                    dtg=dtg,
                    gated_attention=gated_attention,
                    value_residual=value_residual,
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
        # Init remaining nn.Linear modules (bigram proj, mtp heads, lm_head)
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
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x, _ = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, v0=v0)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if hasattr(self, '_ngram_tracker') and self._ngram_tracker is not None and self.training:
            per_tok_loss = F.cross_entropy(logits.float(), targets, reduction="none")
            weights = self._ngram_tracker.get_weights(input_ids, target_ids)
            main_loss = (per_tok_loss * weights).mean()
        else:
            main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_targets = target_ids[:, k + 1 :].reshape(-1)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x, _ = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, v0=v0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        """Hidden states after final_norm, before logit projection. Used by SLOT."""
        n = self.num_layers
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        ve_cache: dict = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x, raw_v = self.blocks[i](x, x0,
                self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
                v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x, _ = self.blocks[bi](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[n + bi],
                self.qo_bank[n + bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, v0=v0)
        return self.final_norm(x)

    def compute_logits_from_hidden(self, hidden: Tensor, delta: Tensor | None = None) -> Tensor:
        """Logit projection from hidden states + optional additive SLOT delta."""
        x = hidden + delta if delta is not None else hidden
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

# --- N-gram bulk update and hashed n-gram sliding eval ---

def _ngram_bulk_update(val_np, start, end, ctx_tables, full_tables,
                       min_order, max_order, primes, mask):
    """Bulk update n-gram tables with a contiguous range of tokens.
    All ranks call this with the SAME token range -> identical tables everywhere."""
    t = val_np[start:end].astype(np.uint64)
    n = len(t)
    for order in range(min_order, max_order + 1):
        if n < order:
            continue
        ctx_width = order - 1
        ctx_hash = np.zeros(n - order + 1, dtype=np.uint64)
        for k in range(ctx_width):
            ctx_hash ^= t[k:n - order + 1 + k] * primes[k % len(primes)]
        ctx_key = (ctx_hash & mask).astype(np.int64)
        tgt = t[order - 1:]
        full_key = ((ctx_hash ^ (tgt * primes[ctx_width % len(primes)])) & mask).astype(np.int64)
        ctx_tables[order] += np.bincount(ctx_key, minlength=len(ctx_tables[order])).astype(np.uint32)
        full_tables[order] += np.bincount(full_key, minlength=len(full_tables[order])).astype(np.uint32)

def eval_val_sliding_hashed_ngram(
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
    order: int,
    alpha: float,
    min_count: int,
    buckets: int,
    max_seconds: float = 0.0,
    batch_seqs: int = 128,
    eval_seq_len: int | None = None,
) -> tuple[float, float, float]:
    """Score-first sliding eval with chunk-based SHARED n-gram tables + cubric.

    Key design: all ranks share identical n-gram tables via bulk chunk updates.
    Each chunk's windows are distributed across ranks for scoring, then ALL ranks
    update tables with the same contiguous token range. Every rank sees the full
    n-gram picture (not 1/world_size like per-segment updates).

    Legal: entire chunk scored before its tokens update the tables.
    """
    min_order = max(args.ngram_eval_min_order, 2)
    max_order = max(order, min_order)
    adaptive = args.ngram_eval_adaptive
    alpha_min = args.ngram_eval_alpha_min
    alpha_max = args.ngram_eval_alpha_max
    ent_center = args.ngram_eval_entropy_center
    ent_scale = args.ngram_eval_entropy_scale

    # Parse fixed per-order multipliers (PR #809 style)
    _fixed_order_mults = None
    if args.ngram_order_mults_str:
        _fixed_order_mults = np.array([float(x) for x in args.ngram_order_mults_str.split(",")], dtype=np.float64)

    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    # Build all windows and total scored tokens
    all_window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    total_scored_tokens = 0.0
    for ws in all_window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        total_scored_tokens += float(max(wlen - s, 0))

    # Group windows into chunks by scored position -- all ranks share this grouping
    chunk_tokens = int(os.environ.get("NGRAM_CHUNK_TOKENS", "1048576"))  # 1M default
    num_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in all_window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)

    val_np = val_tokens.numpy()
    ctx_tables = {n: np.zeros((buckets,), dtype=np.uint32) for n in range(min_order, max_order + 1)}
    full_tables = {n: np.zeros((buckets,), dtype=np.uint32) for n in range(min_order, max_order + 1)}
    mask = np.uint64(buckets - 1)
    primes = np.array(
        [np.uint64(36313), np.uint64(27191), np.uint64(51647), np.uint64(81929),
         np.uint64(131071), np.uint64(174763), np.uint64(233017)],
        dtype=np.uint64,
    )

    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0

    # Cubric 3D: per (order x entropy_bin x count_bin) adaptive alpha scaling
    _NUM_ENT_BINS = 3  # low / mid / high entropy
    _NUM_CNT_BINS = 3  # low / mid / high count
    _ENT_EDGES = np.array([ent_center - 1.0, ent_center + 1.0])  # [2.0, 4.0] for center=3.0
    _CNT_EDGES = np.array([5.0, 50.0])  # low=<5, mid=5-50, high=>50 context count
    _TOTAL_CELLS = _NUM_ENT_BINS * _NUM_CNT_BINS  # 9 cells per order = 54 total
    _cc = getattr(args, 'cubric_cadence', 0); _con = _cc > 0; _cfired = 0
    if _con:
        # Warm-start: proven converged values from 4+ runs (orders 2-7)
        # All 9 cells per order get the same warm-start, 3D cubric refines from there
        _WARM = {2: 0.45, 3: 0.30, 4: 0.45, 5: 1.88, 6: 2.00, 7: 2.00, 8: 2.00, 9: 2.00}
        _c_alpha_mult = {n: [_WARM.get(n, 1.0)] * _TOTAL_CELLS for n in range(min_order, max_order + 1)}
        _c_hits = {n: [0] * _TOTAL_CELLS for n in range(min_order, max_order + 1)}
        _c_beats = {n: [0] * _TOTAL_CELLS for n in range(min_order, max_order + 1)}

    base_model.eval()
    compiled_logits = maybe_compile(
        base_model.forward_logits,
        enabled=args.compile_enabled,
        fullgraph=False,
    )
    t0 = time.perf_counter()
    deadline = (t0 + max_seconds) if max_seconds > 0.0 else None
    cutoff_hit = False

    if rank == 0:
        print(f"ngram_eval:chunks={num_chunks} chunk_tokens={chunk_tokens} "
              f"windows={len(all_window_starts)} shared_tables=True", flush=True)

    with torch.inference_mode():
        for ci in range(num_chunks):
            if deadline is not None and time.perf_counter() >= deadline:
                cutoff_hit = True
                break

            windows = chunk_windows[ci]
            if not windows:
                continue

            # Distribute this chunk's windows across ranks
            my_s = (len(windows) * rank) // world_size
            my_e = (len(windows) * (rank + 1)) // world_size
            my_windows = windows[my_s:my_e]

            # --- Phase 1: SCORE this chunk's windows ---
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
                logits_f = logits.float()
                nll = F.cross_entropy(
                    logits_f.reshape(-1, logits_f.size(-1)),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)

                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    seg_len = wlen - s
                    if seg_len <= 0:
                        continue

                    seg_nll = nll[i, s:wlen].to(torch.float64).cpu().numpy()
                    seg_model_p = np.exp(-seg_nll)

                    if adaptive:
                        log_probs = F.log_softmax(logits_f[i, s:wlen], dim=-1)
                        probs_a = log_probs.exp()
                        entropy = -(probs_a * log_probs).sum(dim=-1).cpu().numpy()
                        sig = 1.0 / (1.0 + np.exp(-ent_scale * (entropy - ent_center)))
                        per_token_alpha = alpha_min + (alpha_max - alpha_min) * sig
                        # Bin entropy for 2D cubric: 0=low, 1=mid, 2=high
                        _ent_bins = np.digitize(entropy, _ENT_EDGES).astype(np.int32)
                    else:
                        per_token_alpha = np.full(seg_len, alpha)
                        _ent_bins = np.ones(seg_len, dtype=np.int32)  # all mid

                    global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
                    p_ng = np.zeros(seg_len, dtype=np.float64)
                    ng_matched = np.zeros(seg_len, dtype=np.bool_)
                    _ng_ord = np.zeros(seg_len, dtype=np.int32)
                    _ng_ctx_count = np.zeros(seg_len, dtype=np.float64)
                    tgt_np = val_np[global_j].astype(np.uint64)

                    for n in range(max_order, min_order - 1, -1):
                        ctx_width = n - 1
                        valid = (global_j >= ctx_width) & (~ng_matched)
                        if not valid.any():
                            continue
                        v_idx = np.nonzero(valid)[0]
                        jv = global_j[v_idx]
                        ctx_hash = np.zeros(len(jv), dtype=np.uint64)
                        for k in range(ctx_width):
                            tok = val_np[jv - (ctx_width - k)].astype(np.uint64)
                            ctx_hash ^= tok * primes[k % len(primes)]
                        ctx_key = (ctx_hash & mask).astype(np.int64)
                        full_key = ((ctx_hash ^ (tgt_np[v_idx] * primes[ctx_width % len(primes)])) & mask).astype(np.int64)
                        ctx_counts = ctx_tables[n][ctx_key].astype(np.float64)
                        full_counts = full_tables[n][full_key].astype(np.float64)
                        has_data = ctx_counts >= float(min_count)
                        if has_data.any():
                            p = np.minimum(full_counts, ctx_counts) / np.maximum(ctx_counts, 1.0)
                            p = np.clip(p, 0.0, 1.0)
                            hit_idx = v_idx[has_data]
                            p_ng[hit_idx] = p[has_data]
                            ng_matched[hit_idx] = True
                            _ng_ord[hit_idx] = n
                            _ng_ctx_count[hit_idx] = ctx_counts[has_data]

                    # Mix where n-gram matched (PR #809 style or cubric 3D fallback)
                    if ng_matched.any():
                        m_idx = np.nonzero(ng_matched)[0]
                        # Per-order entropy center shift (PR #809)
                        if adaptive and args.ngram_entropy_shift:
                            matched_ords = _ng_ord[m_idx].astype(np.float64)
                            shifted_centers = ent_center - 0.25 * (matched_ords - float(min_order))
                            shifted_sig = 1.0 / (1.0 + np.exp(-ent_scale * (entropy[m_idx] - shifted_centers)))
                            per_token_alpha[m_idx] = alpha_min + (alpha_max - alpha_min) * shifted_sig
                        if _fixed_order_mults is not None:
                            # PR #809 fixed order multipliers (replaces cubric)
                            a = per_token_alpha[m_idx].copy()
                            mult_indices = _ng_ord[m_idx] - min_order
                            mult_indices = np.clip(mult_indices, 0, len(_fixed_order_mults) - 1)
                            a *= _fixed_order_mults[mult_indices]
                            np.clip(a, 0.0, 0.95, out=a)
                        elif _con:
                            a = per_token_alpha[m_idx].copy()
                            m_ent_bins = _ent_bins[m_idx]
                            m_cnt_bins = np.digitize(_ng_ctx_count[m_idx], _CNT_EDGES).astype(np.int32)
                            for n in range(min_order, max_order + 1):
                                om = _ng_ord[m_idx] == n
                                if not om.any():
                                    continue
                                for eb in range(_NUM_ENT_BINS):
                                    for cb in range(_NUM_CNT_BINS):
                                        cell = eb * _NUM_CNT_BINS + cb
                                        mask_ecb = om & (m_ent_bins == eb) & (m_cnt_bins == cb)
                                        if mask_ecb.any():
                                            _c_hits[n][cell] += int(mask_ecb.sum())
                                            _c_beats[n][cell] += int((p_ng[m_idx[mask_ecb]] > seg_model_p[m_idx[mask_ecb]]).sum())
                                            a[mask_ecb] *= _c_alpha_mult[n][cell]
                            np.clip(a, 0.0, 0.95, out=a)
                        else:
                            a = per_token_alpha[m_idx]
                        seg_model_p[m_idx] = (1.0 - a) * seg_model_p[m_idx] + a * p_ng[m_idx]

                    seg_nll = -np.log(np.clip(seg_model_p, 1e-12, 1.0))
                    loss_sum += float(seg_nll.sum())
                    token_count += float(seg_len)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += float(tb.sum().item())

            # --- Phase 2: SHARED UPDATE -- all ranks update with same chunk tokens ---
            chunk_start = ci * chunk_tokens
            chunk_end = min((ci + 1) * chunk_tokens, total_tokens)
            _ngram_bulk_update(val_np, chunk_start, chunk_end + 1,
                               ctx_tables, full_tables, min_order, max_order,
                               primes, mask)

            # Cubric 2D c-step: adapt per (order x entropy_bin)
            if _con:
                # Collect all (order, ent_bin, cnt_bin) cells with enough data
                all_rates = []
                for n in range(min_order, max_order + 1):
                    for cell in range(_TOTAL_CELLS):
                        if _c_hits[n][cell] >= 8:
                            all_rates.append(_c_beats[n][cell] / _c_hits[n][cell])
                if len(all_rates) >= 4:
                    avg_rate = sum(all_rates) / len(all_rates)
                    for n in range(min_order, max_order + 1):
                        for cell in range(_TOTAL_CELLS):
                            if _c_hits[n][cell] >= 8:
                                rate = _c_beats[n][cell] / _c_hits[n][cell]
                                if rate > avg_rate + 0.05:
                                    _c_alpha_mult[n][cell] = min(_c_alpha_mult[n][cell] * 1.03, 2.0)
                                elif rate < avg_rate - 0.05:
                                    _c_alpha_mult[n][cell] = max(_c_alpha_mult[n][cell] * 0.97, 0.3)
                _cfired += 1
                if rank == 0 and _cfired % 8 == 0:
                    parts = []
                    for n in range(min_order, max_order + 1):
                        m = _c_alpha_mult[n]
                        avg_m = sum(m) / len(m)
                        parts.append(f"o{n}:avg={avg_m:.2f}")
                    print(f"cubric3d:step={_cfired} {' '.join(parts)}", flush=True)
                _c_hits = {n: [0] * _TOTAL_CELLS for n in range(min_order, max_order + 1)}
                _c_beats = {n: [0] * _TOTAL_CELLS for n in range(min_order, max_order + 1)}

            # Progress
            if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1 or ci < 3):
                elapsed = time.perf_counter() - t0
                cur_bpb = (loss_sum / max(token_count, 1.0)) / math.log(2.0) * (token_count / max(byte_count, 1.0)) if token_count > 0 else 0.0
                print(
                    f"ngram_eval:chunk [{ci+1}/{num_chunks}] bpb={cur_bpb:.6f} t={elapsed:.0f}s",
                    flush=True,
                )

    # All-reduce across ranks
    _loss = torch.tensor(loss_sum, device=device, dtype=torch.float64)
    _toks = torch.tensor(token_count, device=device, dtype=torch.float64)
    _bytes = torch.tensor(byte_count, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(_toks, op=dist.ReduceOp.SUM)
        dist.all_reduce(_bytes, op=dist.ReduceOp.SUM)
    loss_sum = _loss.item()
    token_count = _toks.item()
    byte_count = _bytes.item()

    coverage = token_count / max(total_scored_tokens, 1.0)
    if cutoff_hit:
        elapsed = time.perf_counter() - t0
        print(
            f"ngram_eval:cutoff max_seconds={max_seconds:.1f} "
            f"coverage={coverage*100:.2f}% elapsed={elapsed:.0f}s",
            flush=True,
        )

    if _con and rank == 0:
        print(f"cubric3d:final c_steps={_cfired} cells={_TOTAL_CELLS}x{max_order-min_order+1}={_TOTAL_CELLS*(max_order-min_order+1)}", flush=True)
        for n in range(min_order, max_order + 1):
            m = _c_alpha_mult[n]
            row = " ".join(f"{m[cell]:.2f}" for cell in range(_TOTAL_CELLS))
            print(f"  o{n}: [{row}]", flush=True)
    val_loss = loss_sum / max(token_count, 1.0)
    val_bpb = val_loss / math.log(2.0) * (token_count / max(byte_count, 1.0))
    base_model.train()
    return val_loss, val_bpb, coverage

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
    slot_enabled: bool = False,
    slot_steps: int = 8,
    slot_lr: float = 0.005,
    max_windows: int = 0,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context.
    If slot_enabled, optimizes a per-batch additive delta at the last hidden layer
    (SLOT: Sample-specific LM Optimization at Test-time, Hu et al. arXiv:2505.12392v2).
    Model weights are never modified; only the delta is trained. Score-first per batch.
    max_windows > 0 limits evaluation to first N windows (for fast ablations).
    """
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    if max_windows > 0:
        my_windows = my_windows[:max_windows]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    if not slot_enabled:
        compiled_logits = maybe_compile(
            base_model.forward_logits,
            enabled=args.compile_enabled,
            fullgraph=args.compile_fullgraph,
        )
    with (torch.inference_mode() if not slot_enabled else torch.no_grad()):
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
            if slot_enabled:
                # SLOT: compute frozen hidden states, then optimize per-batch delta
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    hidden = base_model.forward_hidden(x_batch)  # (bsz, seq_len, dim)
                hidden = hidden.detach()
                delta = torch.zeros(1, 1, hidden.size(-1), device=device,
                                    dtype=hidden.dtype, requires_grad=True)
                opt = torch.optim.AdamW([delta], lr=slot_lr, weight_decay=1e-8, eps=1e-5)
                with torch.enable_grad():
                    for _ in range(slot_steps):
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits_s = base_model.compute_logits_from_hidden(hidden, delta)
                        loss_s = F.cross_entropy(
                            logits_s.reshape(-1, logits_s.size(-1)).float(),
                            y_batch.reshape(-1),
                        )
                        loss_s.backward()
                        opt.step()
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.compute_logits_from_hidden(hidden, delta.detach())
            else:
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
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
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
    if args.ngram_eval_order >= 2:
        log0(f"ngram_eval:order={args.ngram_eval_order} alpha={args.ngram_eval_alpha} min_count={args.ngram_eval_min_count} buckets={args.ngram_eval_buckets}")
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
    if args.complement_alpha > 0:
        tracker = TrainNgramTracker(args.vocab_size, device, complement_alpha=args.complement_alpha)
        base_model._ngram_tracker = tracker
        log0(f"complementary_training:alpha={args.complement_alpha}")
    else:
        base_model._ngram_tracker = None
    # No DDP -- Parallel Muon handles bank grad communication via reduce-scatter,
    # and non-bank grads are manually all-reduced before Adam steps.
    compiled_model = maybe_compile(
        base_model,
        enabled=args.compile_enabled,
        fullgraph=args.compile_fullgraph,
        mode=args.compile_mode,
    )
    model = compiled_model

    # Optimizer split:
    # - 4 parameter banks -> Muon (batched Newton-Schulz)
    # - token embedding -> Adam
    # - scalars/control tensors -> Adam
    # - bigram proj, mtp heads, VE proj -> Adam (small matrix params not worth banking)
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
    # Non-bank params that need manual all-reduce (replicated across GPUs)
    replicated_params = list(optimizer_tok.param_groups[0]["params"])
    for pg in optimizer_tok.param_groups[1:]:
        replicated_params.extend(pg["params"])
    replicated_params.extend(scalar_params)

    optimizer_head = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        replicated_params.append(base_model.lm_head.weight)
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_head is not None:
        optimizers.append(optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params = sum(p.numel() for p in base_model.mtp_heads.parameters())
    log0(f"model_params:{n_params}")
    log0(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
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
    compile_mode = args.compile_mode if args.compile_mode else "default"
    log0(
        f"compile:enabled={int(args.compile_enabled)} mode:{compile_mode} "
        f"fullgraph={int(args.compile_fullgraph)}"
    )
    log0(f"mlp_kernel_mode:{args.mlp_kernel_mode or 'eager'}")
    log0(
        f"scale_init:attn={args.attn_scale_init:.4f} mlp={args.mlp_scale_init:.4f} "
        f"resid_mix=({args.resid_mix_x_init:.4f},{args.resid_mix_x0_init:.4f}) "
        f"ln_scale={int(args.ln_scale)}"
    )
    log0(f"seed:{args.seed}")
    train_loader = build_train_loader(args, rank, world_size, device)
    log0(train_loader.describe())
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    # GPTQ calibration reads training data — it must complete within the wallclock budget.
    # We stop the training loop early (by GPTQ_RESERVE_MS) so GPTQ runs before the cap.
    _skip_gptq = int(os.environ.get("SKIP_GPTQ", "0"))
    _gptq_reserve_ms = float(os.environ.get("GPTQ_RESERVE_MS", "30000")) if (max_wallclock_ms is not None and not _skip_gptq) else 0.0
    effective_max_wallclock_ms = (max_wallclock_ms - _gptq_reserve_ms) if max_wallclock_ms is not None else None
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
        train_loader = build_train_loader(args, rank, world_size, device)
        log0(f"loader_reset:{train_loader.describe()}")
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    from collections import deque
    lawa_queue: deque[dict[str, Tensor]] = deque(maxlen=args.lawa_k)
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = 0.997
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
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
            if base_model._ngram_tracker is not None:
                base_model._ngram_tracker.update(x, y)
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
        if args.lawa_enabled and step % args.lawa_freq == 0:
            lawa_queue.append({name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()})
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )
        reached_cap = effective_max_wallclock_ms is not None and approx_training_time_ms >= effective_max_wallclock_ms
        if distributed and effective_max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    # GPTQ calibration: reads training data — must complete within MAX_WALLCLOCK_SECONDS.
    # Training loop stopped GPTQ_RESERVE_MS early so this runs inside the budget.
    if _skip_gptq:
        log0("gptq:SKIPPED (SKIP_GPTQ=1) — will use naive int6")
        gptq_hessians: dict[str, Tensor] = {}
    else:
        log0("gptq:calibrating with training data...")
        t_gptq = time.perf_counter()
        gptq_hessians = gptq_calibrate(base_model, args.train_files, device, n_samples=256, seq_len=args.train_seq_len)
        log0(f"gptq:calibrated {len(gptq_hessians)} layers in {time.perf_counter()-t_gptq:.1f}s")
    # Apply weight averaging
    if args.lawa_enabled and len(lawa_queue) > 1:
        log0(f"lawa:applying LAWA averaging k={len(lawa_queue)}")
        current_state = base_model.state_dict()
        avg_state = {name: torch.zeros(t.shape, dtype=torch.float32, device='cpu') for name, t in current_state.items()}
        for snap in lawa_queue:
            for name in avg_state:
                avg_state[name] += snap[name].float()
        for name in avg_state:
            avg_state[name] /= len(lawa_queue)
            avg_state[name] = avg_state[name].to(dtype=current_state[name].dtype)
        base_model.load_state_dict(avg_state, strict=True)
    else:
        log0("ema:applying EMA weights")
        current_state = base_model.state_dict()
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(avg_state, strict=True)
    if args.post_ema_diagnostic:
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
    else:
        log0("diagnostic_eval:skipped POST_EMA_DIAGNOSTIC=0")
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
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    # GPTQ quantization using Hessians collected from training data
    quant_result, quant_meta = mixed_quantize_int6_gptq(sd_cpu, {"mlp", "attn", "aux", "embed"}, gptq_hessians)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw) if _COMPRESSOR == "zstd" else _zlib_module.compress(quant_raw, 9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        log0(f"Serialized model int6+{_COMPRESSOR}: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+{_COMPRESSOR}: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob_disk) if _COMPRESSOR == "zstd" else _zlib_module.decompress(quant_blob_disk)),
        map_location="cpu",
    )
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        gated_attention=args.gated_attention, value_residual=args.value_residual,
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
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, eval_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    del eval_model, deq_state, quant_state, sd_cpu
    torch.cuda.empty_cache()
    sw_seq_len = effective_eval_seq_len
    if args.skip_final_eval:
        log0("final_eval:skipped sliding/ngram by SKIP_FINAL_EVAL=1")
    else:
        if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
            torch.cuda.synchronize()
            t_slide = time.perf_counter()
            sw_val_loss, sw_val_bpb = eval_val_sliding(
                args, base_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=args.eval_stride,
                eval_seq_len=sw_seq_len,
                slot_enabled=args.slot_enabled,
                slot_steps=args.slot_steps,
                slot_lr=args.slot_lr,
                max_windows=args.slot_max_windows,
            )
            torch.cuda.synchronize()
            slot_tag = f"+slot{args.slot_steps}steps" if args.slot_enabled else ""
            log0(
                f"final_sliding_window{slot_tag} val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
                f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
            )
            log0(f"final_sliding_window{slot_tag}_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        if args.eval_stride != 64 and 64 < sw_seq_len:
            torch.cuda.synchronize()
            t_slide64 = time.perf_counter()
            sw64_val_loss, sw64_val_bpb = eval_val_sliding(
                args, base_model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=64,
                eval_seq_len=sw_seq_len,
            )
            torch.cuda.synchronize()
            log0(
                f"final_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
                f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms"
            )
            log0(f"final_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
        if args.ngram_eval_order >= 2:
            if distributed:
                dist.barrier()
            torch.cuda.synchronize()
            t_ng = time.perf_counter()
            ng_loss, ng_bpb, ng_coverage = eval_val_sliding_hashed_ngram(
                args,
                base_model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                stride=args.eval_stride,
                order=args.ngram_eval_order,
                alpha=args.ngram_eval_alpha,
                min_count=args.ngram_eval_min_count,
                buckets=args.ngram_eval_buckets,
                max_seconds=args.ngram_eval_max_seconds,
                eval_seq_len=sw_seq_len,
            )
            if rank == 0:
                torch.cuda.synchronize()
                ng_eval_ms = 1000.0 * (time.perf_counter() - t_ng)
                if ng_coverage >= 0.999999:
                    log0(
                        f"final_sliding_window_ngram{args.ngram_eval_order} val_loss:{ng_loss:.4f} "
                        f"val_bpb:{ng_bpb:.4f} eval_time:{ng_eval_ms:.0f}ms"
                    )
                    log0(
                        f"final_sliding_window_ngram{args.ngram_eval_order}_exact "
                        f"val_loss:{ng_loss:.8f} val_bpb:{ng_bpb:.8f}"
                    )
                else:
                    log0(
                        f"final_sliding_window_ngram{args.ngram_eval_order}_partial val_loss:{ng_loss:.4f} "
                        f"val_bpb:{ng_bpb:.4f} coverage:{ng_coverage:.4f} eval_time:{ng_eval_ms:.0f}ms"
                    )
                    log0(
                        f"final_sliding_window_ngram{args.ngram_eval_order}_partial_exact "
                        f"val_loss:{ng_loss:.8f} val_bpb:{ng_bpb:.8f} coverage:{ng_coverage:.8f}"
                    )
            if distributed:
                dist.barrier()
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()

