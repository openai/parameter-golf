from __future__ import annotations
import gc
import copy
import glob
import io
import lzma
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
    import brotli
    _COMPRESSOR = "brotli"
except ImportError:
    _COMPRESSOR = "lzma"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from flash_attn_interface import flash_attn_func as flash_attn_3_func
import queue
import threading

_THIS_DIR = Path(__file__).resolve().parent
_CUTLASS_EVT_DIR = _THIS_DIR / "cutlass_evt_fusion"
if _CUTLASS_EVT_DIR.is_dir():
    sys.path.insert(0, str(_CUTLASS_EVT_DIR))

# --- Fused Triton MLP kernel (PR #1072 approach) ---
IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None
HAS_FUSED_MLP = False
try:
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def _fused_leaky_relu_sq_kernel(a_desc, b_desc, c_desc, aux_desc,
                                     M, N, K,
                                     BLOCK_SIZE_M: tl.constexpr,
                                     BLOCK_SIZE_N: tl.constexpr,
                                     BLOCK_SIZE_K: tl.constexpr,
                                     GROUP_SIZE_M: tl.constexpr,
                                     NUM_SMS: tl.constexpr,
                                     FORWARD: tl.constexpr):
        dtype = tl.bfloat16
        start_pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        num_tiles = num_pid_m * num_pid_n
        tile_id_c = start_pid - NUM_SMS
        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K
                a = a_desc.load([offs_am, offs_k])
                b = b_desc.load([offs_bn, offs_k])
                accumulator = tl.dot(a, b.T, accumulator)
            tile_id_c += NUM_SMS
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            offs_am_c = pid_m * BLOCK_SIZE_M
            offs_bn_c = pid_n * BLOCK_SIZE_N
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            if not FORWARD:
                c0_ag = aux_desc.load([offs_am_c, offs_bn_c])
                c0 = c0 * c0_ag
                c_desc.store([offs_am_c, offs_bn_c], c0)
            if FORWARD:
                c0_ag = tl.where(c0 > 0, 2.0 * c0, 0.5 * c0)
                c_desc.store([offs_am_c, offs_bn_c], c0_ag)
                c0_post = 0.5 * c0_ag * c0
                aux_desc.store([offs_am_c, offs_bn_c], c0_post)
            c1 = acc1.to(dtype)
            if not FORWARD:
                c1_ag = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
                c1 = c1 * c1_ag
                c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
            if FORWARD:
                c1_ag = tl.where(c1 > 0, 2.0 * c1, 0.5 * c1)
                c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_ag)
                c1_post = 0.5 * c1_ag * c1
                aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)

    def _fused_leaky_relu_sq(a, b, aux=None):
        M, K = a.shape
        N, K2 = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        FORWARD = aux is None
        if FORWARD:
            aux = torch.empty((M, N), device=a.device, dtype=a.dtype)
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 256, 64
        a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
        aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
        def grid(META):
            return (min(NUM_SMS, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),)
        _fused_leaky_relu_sq_kernel[grid](
            a_desc, b_desc, c_desc, aux_desc, M, N, K,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=1, NUM_SMS=NUM_SMS, FORWARD=FORWARD,
            num_stages=4 if FORWARD else 3, num_warps=8)
        return (c, aux) if FORWARD else c

    class FusedLeakyReLUSqMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, up_w, down_w):
            x_flat = x.view(-1, x.shape[-1])
            act_grad, post = _fused_leaky_relu_sq(x_flat, up_w)
            out = F.linear(post, down_w)
            ctx.save_for_backward(x_flat, up_w, down_w, act_grad, post)
            return out.view(x.shape)
        @staticmethod
        def backward(ctx, grad_output):
            x_flat, up_w, down_w, act_grad, post = ctx.saved_tensors
            go = grad_output.view(-1, grad_output.shape[-1])
            dW2 = go.T @ post
            dpre = torch.ops.cutlass_evt.gemm_mul(go, down_w, act_grad)
            dW1 = dpre.T @ x_flat
            dx = dpre @ up_w
            return dx.view(grad_output.shape), dW1, dW2

    HAS_FUSED_MLP = True
except (ImportError, Exception):
    HAS_FUSED_MLP = False

# --- CUTLASS EVT backward fusion (required) ---
import cutlass_evt_fusion

@torch.library.register_fake("cutlass_evt::gemm_mul")
def _gemm_mul_fake(go, down_w, act_grad):
    return go.new_empty(go.size(0), down_w.size(1))

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp4608")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_4608_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.0))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 4608))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    matrix_lr_early = float(os.environ.get("MATRIX_LR_EARLY", 0.02))
    matrix_lr_late = float(os.environ.get("MATRIX_LR_LATE", 0.02))
    bank_split = int(os.environ.get("BANK_SPLIT", 5))
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
    muon_wd = float(os.environ.get("MUON_WD", 0.085))
    adam_wd = float(os.environ.get("ADAM_WD", 0.02))
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    qat_alpha_start = float(os.environ.get("QAT_ALPHA_START", 1.0))
    qat_alpha_end = float(os.environ.get("QAT_ALPHA_END", 16.0))
    qat_ramp_steps = int(os.environ.get("QAT_RAMP_STEPS", 500))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 0))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 0))
    trigram_enabled = bool(int(os.environ.get("TRIGRAM", "0")))
    # EngramLite params
    use_engramlite = bool(int(os.environ.get("ENGRAM", "0")))
    ngram_buckets = int(os.environ.get("NGRAM_BUCKETS", 8192))
    ngram_heads = int(os.environ.get("NGRAM_HEADS", 2))
    ngram_orders = int(os.environ.get("NGRAM_ORDERS", 2))
    ngram_dim_per_head = int(os.environ.get("NGRAM_DIM_PER_HEAD", 32))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))  # XSA on ALL layers (our novel contribution)
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    lr_floor = float(os.environ.get("LR_FLOOR", 0.05))  # Minimum LR multiplier
    mixed_quant = bool(int(os.environ.get("MIXED_QUANT", "1")))
    n_int6_layers = int(os.environ.get("N_INT6_LAYERS", "66"))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "0")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))  # VRL with sigmoid gates (off by default, risky)
    # GPTQ calibration
    gptq_calib_batches = int(os.environ.get("GPTQ_CALIB_BATCHES", 256))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))
    fused_ngram_eval = bool(int(os.environ.get("FUSED_NGRAM_EVAL", "1")))
    fused_ngram_device = os.environ.get("FUSED_NGRAM_DEVICE", "cuda:0")
    fused_ngram_batch_seqs = int(os.environ.get("FUSED_NGRAM_BATCH_SEQS", "64"))
    fused_ngram_base_beta = float(os.environ.get("FUSED_NGRAM_BASE_BETA", 1.0))
    fused_ngram_agree_bonus = float(os.environ.get("FUSED_NGRAM_AGREE_BONUS", 0.5))
    fused_ngram_within_threshold = float(os.environ.get("FUSED_NGRAM_WITHIN_THRESHOLD", 0.25))
    fused_ngram_within_beta = float(os.environ.get("FUSED_NGRAM_WITHIN_BETA", 0.55))
    fused_ngram_word_threshold = float(os.environ.get("FUSED_NGRAM_WORD_THRESHOLD", 0.80))
    fused_ngram_word_beta = float(os.environ.get("FUSED_NGRAM_WORD_BETA", 0.50))
    fused_ngram_open_table_bits = int(os.environ.get("FUSED_NGRAM_OPEN_TABLE_BITS", "26"))
    fused_ngram_token_threshold_scale = float(os.environ.get("FUSED_NGRAM_TOKEN_THRESHOLD_SCALE", 1.0))
    fused_ngram_order_stride = int(os.environ.get("FUSED_NGRAM_ORDER_STRIDE", "2"))

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
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gate,skip_gates,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,vr_lambda",
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
def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats
def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

# --- Data loading (memmap pipeline from PR #726) ---

_MAGIC = 20240520
_VERSION = 1
_HEADER_INTS = 256
_HEADER_DTYPE = np.dtype("<i4")
_TOKEN_DTYPE = np.dtype("<u2")
_HEADER_BYTES = _HEADER_INTS * _HEADER_DTYPE.itemsize

_HEADER_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}


def _stable_hash64(text: str) -> int:
    h = 1469598103934665603
    for b in text.encode("utf-8", errors="surrogatepass"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = _HEADER_CACHE.get(key)
    if cached is not None:
        return cached

    header = np.fromfile(file, dtype=_HEADER_DTYPE, count=_HEADER_INTS)
    if header.size != _HEADER_INTS:
        raise ValueError(f"Unexpected shard header size for {file}")
    if int(header[0]) != _MAGIC or int(header[1]) != _VERSION:
        raise ValueError(f"Unexpected shard header for {file}")

    num_tokens = int(header[2])
    expected_size = _HEADER_BYTES + num_tokens * _TOKEN_DTYPE.itemsize
    actual_size = file.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes, got {actual_size} bytes"
        )

    _HEADER_CACHE[key] = num_tokens
    return num_tokens


def _get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm

    num_tokens = _read_num_tokens(file)
    mm = np.memmap(
        file,
        mode="r",
        dtype=_TOKEN_DTYPE,
        offset=_HEADER_BYTES,
        shape=(num_tokens,),
        order="C",
    )
    _MMAP_CACHE[key] = mm
    return mm


def load_data_shard(file: Path) -> Tensor:
    return torch.from_numpy(_get_shard_memmap(file))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")

        seed = _stable_hash64(pattern)
        for file in self.files:
            seed ^= _stable_hash64(str(file))
            seed &= 0xFFFFFFFFFFFFFFFF
        self._rng = np.random.Generator(np.random.PCG64(seed))

        self._order = np.arange(len(self.files), dtype=np.int64)
        if self._order.size > 1:
            self._rng.shuffle(self._order)

        self._order_pos = 0
        self._file_idx = int(self._order[0])
        self._tokens = load_data_shard(self.files[self._file_idx])
        self._pos = 0

    def _advance_file(self) -> None:
        self._order_pos += 1
        if self._order_pos >= int(self._order.size):
            self._order_pos = 0
            if self._order.size > 1:
                self._rng.shuffle(self._order)
        self._file_idx = int(self._order[self._order_pos])
        self._tokens = load_data_shard(self.files[self._file_idx])
        self._pos = 0

    def take(self, n: int) -> Tensor:
        if n <= 0:
            return torch.empty(0, dtype=torch.uint16)

        remaining = int(n)
        chunks: list[Tensor] = []

        while remaining > 0:
            avail = int(self._tokens.numel()) - self._pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self._tokens[self._pos : self._pos + k])
            self._pos += k
            remaining -= k

        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device

        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")

        self._num_tokens = np.asarray([_read_num_tokens(f) for f in self.files], dtype=np.int64)

        seed = _stable_hash64(pattern)
        for file, n_tok in zip(self.files, self._num_tokens.tolist(), strict=True):
            seed ^= _stable_hash64(str(file))
            seed ^= (int(n_tok) * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            seed &= 0xFFFFFFFFFFFFFFFF
        self._rng = np.random.Generator(np.random.PCG64(seed))

        self._cfg: tuple[int, int, int, int] | None = None
        self._eligible_shards: np.ndarray | None = None
        self._base_block_counts: np.ndarray | None = None

        self._cursor_phase: np.ndarray | None = None
        self._cursor_block_count: np.ndarray | None = None
        self._cursor_next: np.ndarray | None = None
        self._cursor_start: np.ndarray | None = None
        self._cursor_stride: np.ndarray | None = None
        self._cursor_initialized: np.ndarray | None = None

        self._queue: queue.Queue[tuple[Tensor, Tensor]] | None = None
        self._worker: threading.Thread | None = None
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._next_gpu_batch: tuple[Tensor, Tensor] | None = None
        self._next_ready_event: torch.cuda.Event | None = None

        self._batches_built = 0
        self._merge_gap_tokens = 0

    def _pick_coprime_stride(self, n: int) -> int:
        if n <= 1:
            return 1
        while True:
            s = int(self._rng.integers(1, n))
            if math.gcd(s, n) == 1:
                return s

    def _reset_shard_cursor(self, shard_idx: int, seq_len: int) -> None:
        if (
            self._cursor_phase is None
            or self._cursor_block_count is None
            or self._cursor_next is None
            or self._cursor_start is None
            or self._cursor_stride is None
            or self._cursor_initialized is None
        ):
            raise RuntimeError("Shard cursor state is not initialized")

        n_tok = int(self._num_tokens[shard_idx])
        max_phase = min(seq_len - 1, max(0, n_tok - seq_len - 1))
        phase = int(self._rng.integers(max_phase + 1)) if max_phase > 0 else 0
        block_count = (n_tok - 1 - phase) // seq_len
        if block_count <= 0:
            raise RuntimeError(f"Ineligible shard {self.files[shard_idx]} for seq_len={seq_len}")

        self._cursor_phase[shard_idx] = phase
        self._cursor_block_count[shard_idx] = int(block_count)
        self._cursor_next[shard_idx] = 0
        self._cursor_start[shard_idx] = int(self._rng.integers(block_count)) if block_count > 1 else 0
        self._cursor_stride[shard_idx] = self._pick_coprime_stride(block_count)
        self._cursor_initialized[shard_idx] = True

    def _ensure_shard_cursor(self, shard_idx: int, seq_len: int) -> None:
        if (
            self._cursor_initialized is None
            or self._cursor_next is None
            or self._cursor_block_count is None
        ):
            raise RuntimeError("Shard cursor state is not initialized")

        if (not bool(self._cursor_initialized[shard_idx])) or (
            int(self._cursor_next[shard_idx]) >= int(self._cursor_block_count[shard_idx])
        ):
            self._reset_shard_cursor(shard_idx, seq_len)

    def _take_from_shard(
        self,
        shard_idx: int,
        seq_len: int,
        count: int,
        out: list[tuple[int, int]],
    ) -> None:
        if count <= 0:
            return
        if (
            self._cursor_phase is None
            or self._cursor_block_count is None
            or self._cursor_next is None
            or self._cursor_start is None
            or self._cursor_stride is None
        ):
            raise RuntimeError("Shard cursor state is not initialized")

        remaining = int(count)
        while remaining > 0:
            self._ensure_shard_cursor(shard_idx, seq_len)
            block_count = int(self._cursor_block_count[shard_idx])
            next_idx = int(self._cursor_next[shard_idx])
            take = min(remaining, block_count - next_idx)
            phase = int(self._cursor_phase[shard_idx])
            start = int(self._cursor_start[shard_idx])
            stride = int(self._cursor_stride[shard_idx])

            for j in range(take):
                block_idx = (start + (next_idx + j) * stride) % block_count
                pos = phase + block_idx * seq_len
                out.append((int(shard_idx), int(pos)))

            self._cursor_next[shard_idx] = next_idx + take
            remaining -= take

    def _schedule_progress(self) -> float:
        return min(self._batches_built / 1800.0, 1.0)

    def _current_mix_shards(self, eligible_count: int, global_num_seqs: int) -> int:
        progress = self._schedule_progress()
        low = min(max(8, self.world_size), eligible_count, global_num_seqs)
        high = min(max(32, self.world_size * 8), eligible_count, global_num_seqs)
        if high < low:
            high = low
        mix = int(round(low + progress * (high - low)))
        return max(1, min(mix, eligible_count, global_num_seqs))

    def _sample_global_windows(self) -> list[tuple[int, int]]:
        if self._cfg is None or self._eligible_shards is None or self._base_block_counts is None:
            raise RuntimeError("Loader pipeline not initialized")
        if (
            self._cursor_next is None
            or self._cursor_initialized is None
            or self._cursor_block_count is None
        ):
            raise RuntimeError("Shard cursor state is not initialized")

        _, seq_len, _, global_num_seqs = self._cfg
        progress = self._schedule_progress()

        remaining = np.empty_like(self._base_block_counts, dtype=np.float64)
        for i, shard_idx in enumerate(self._eligible_shards.tolist()):
            if bool(self._cursor_initialized[shard_idx]):
                rem = int(self._cursor_block_count[shard_idx]) - int(self._cursor_next[shard_idx])
                remaining[i] = float(rem if rem > 0 else int(self._base_block_counts[i]))
            else:
                remaining[i] = float(int(self._base_block_counts[i]))

        alpha = 0.90 - 0.40 * progress
        weights = np.power(np.maximum(remaining, 1.0), alpha, dtype=np.float64)
        weights_sum = float(weights.sum())
        if not np.isfinite(weights_sum) or weights_sum <= 0.0:
            weights = np.ones_like(weights, dtype=np.float64)
            weights_sum = float(weights.sum())
        probs = weights / weights_sum

        mix = self._current_mix_shards(int(self._eligible_shards.size), global_num_seqs)
        chosen_pos = self._rng.choice(int(self._eligible_shards.size), size=mix, replace=False, p=probs)
        chosen_shards = self._eligible_shards[chosen_pos]
        chosen_probs = probs[chosen_pos].astype(np.float64, copy=True)
        chosen_probs /= float(chosen_probs.sum())

        counts = np.ones(mix, dtype=np.int64)
        extra = global_num_seqs - mix
        if extra > 0:
            counts += self._rng.multinomial(extra, chosen_probs).astype(np.int64, copy=False)

        perm = self._rng.permutation(mix)
        chosen_shards = chosen_shards[perm]
        counts = counts[perm]

        buckets: list[list[tuple[int, int]]] = []
        for shard_idx, count in zip(chosen_shards.tolist(), counts.tolist(), strict=True):
            local_bucket: list[tuple[int, int]] = []
            self._take_from_shard(int(shard_idx), seq_len, int(count), local_bucket)
            if local_bucket:
                if len(local_bucket) > 1:
                    local_perm = self._rng.permutation(len(local_bucket))
                    local_bucket = [local_bucket[int(i)] for i in local_perm.tolist()]
                buckets.append(local_bucket)

        windows: list[tuple[int, int]] = []
        active = [i for i, b in enumerate(buckets) if b]
        while active:
            order = self._rng.permutation(len(active))
            new_active: list[int] = []
            for ord_idx in order.tolist():
                bi = active[ord_idx]
                bucket = buckets[bi]
                if bucket:
                    windows.append(bucket.pop())
                if bucket:
                    new_active.append(bi)
            active = new_active

        if len(windows) != global_num_seqs:
            raise RuntimeError(f"Incorrect number of sampled windows: expected {global_num_seqs}, got {len(windows)}")
        return windows

    def _copy_from_shard_group(
        self,
        shard_idx: int,
        items: list[tuple[int, int]],
        seq_len: int,
        x_cpu: Tensor,
        y_cpu: Tensor,
    ) -> None:
        shard_np = _get_shard_memmap(self.files[shard_idx])
        items.sort(key=lambda t: t[1])

        merge_gap = self._merge_gap_tokens
        run_start_idx = 0
        run_start_pos = items[0][1]
        run_end_pos = run_start_pos + seq_len + 1

        for j in range(1, len(items) + 1):
            flush = j == len(items)
            if not flush:
                next_pos = items[j][1]
                if next_pos <= run_end_pos + merge_gap:
                    candidate_end = next_pos + seq_len + 1
                    if candidate_end > run_end_pos:
                        run_end_pos = candidate_end
                    continue

            slab_np = shard_np[run_start_pos:run_end_pos]
            slab_t = torch.from_numpy(slab_np)
            for slot, pos in items[run_start_idx:j]:
                rel = pos - run_start_pos
                window_t = slab_t[rel : rel + seq_len + 1]
                if int(window_t.numel()) != seq_len + 1:
                    raise RuntimeError(
                        f"Short window read from shard {self.files[shard_idx]} at pos={pos}: "
                        f"expected {seq_len + 1}, got {int(window_t.numel())}"
                    )
                x_cpu[slot].copy_(window_t[:-1])
                y_cpu[slot].copy_(window_t[1:])

            if not flush:
                run_start_idx = j
                run_start_pos = items[j][1]
                run_end_pos = run_start_pos + seq_len + 1

    def _build_cpu_batch(self) -> tuple[Tensor, Tensor]:
        if self._cfg is None:
            raise RuntimeError("Loader pipeline not initialized")

        _, seq_len, num_seqs, global_num_seqs = self._cfg
        global_windows = self._sample_global_windows()
        if len(global_windows) != global_num_seqs:
            raise RuntimeError("Incorrect number of sampled windows")

        # Strided rank assignment gives each rank a more uniformly mixed subset
        # of the interleaved global plan than contiguous slicing.
        local_windows = global_windows[self.rank:global_num_seqs:self.world_size]
        if len(local_windows) != num_seqs:
            raise RuntimeError(
                f"Incorrect local window count: expected {num_seqs}, got {len(local_windows)}"
            )

        pin = self.device.type == "cuda"
        x_cpu = torch.empty((num_seqs, seq_len), dtype=torch.uint16, pin_memory=pin)
        y_cpu = torch.empty((num_seqs, seq_len), dtype=torch.uint16, pin_memory=pin)

        by_shard: dict[int, list[tuple[int, int]]] = {}
        for slot, (shard_idx, pos) in enumerate(local_windows):
            by_shard.setdefault(int(shard_idx), []).append((slot, int(pos)))

        for shard_idx, items in by_shard.items():
            self._copy_from_shard_group(shard_idx, items, seq_len, x_cpu, y_cpu)

        self._batches_built += 1
        return x_cpu, y_cpu

    def _worker_loop(self) -> None:
        if self._queue is None:
            return
        while True:
            self._queue.put(self._build_cpu_batch())

    def _stage_next_gpu_batch(self) -> None:
        if self._queue is None:
            raise RuntimeError("Batch queue not initialized")

        x_cpu, y_cpu = self._queue.get()

        if self.device.type != "cuda":
            self._next_gpu_batch = (
                x_cpu.to(device=self.device, dtype=torch.int64),
                y_cpu.to(device=self.device, dtype=torch.int64),
            )
            self._next_ready_event = None
            return

        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=self.device)

        with torch.cuda.stream(self._prefetch_stream):
            x_gpu = x_cpu.to(device=self.device, dtype=torch.int64, non_blocking=True)
            y_gpu = y_cpu.to(device=self.device, dtype=torch.int64, non_blocking=True)
        event = torch.cuda.Event()
        event.record(self._prefetch_stream)

        self._next_gpu_batch = (x_gpu, y_gpu)
        self._next_ready_event = event

    def _init_pipeline(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> None:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        if local_tokens <= 0:
            raise ValueError(
                f"local_tokens must be positive, got {local_tokens} from "
                f"global_tokens={global_tokens}, world_size={self.world_size}, grad_accum_steps={grad_accum_steps}"
            )
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if local_tokens % seq_len != 0:
            raise ValueError(f"local_tokens ({local_tokens}) must be divisible by seq_len ({seq_len})")

        num_seqs = local_tokens // seq_len
        global_num_seqs = num_seqs * self.world_size
        self._cfg = (local_tokens, seq_len, num_seqs, global_num_seqs)

        base_block_counts = (self._num_tokens - 1) // seq_len
        eligible_mask = base_block_counts > 0
        if not np.any(eligible_mask):
            raise ValueError(f"No shards in pattern can provide sequences of length {seq_len + 1}")

        self._eligible_shards = np.nonzero(eligible_mask)[0].astype(np.int64, copy=False)
        self._base_block_counts = base_block_counts[self._eligible_shards].astype(np.int64, copy=False)

        n_files = len(self.files)
        self._cursor_phase = np.zeros(n_files, dtype=np.int64)
        self._cursor_block_count = np.zeros(n_files, dtype=np.int64)
        self._cursor_next = np.zeros(n_files, dtype=np.int64)
        self._cursor_start = np.zeros(n_files, dtype=np.int64)
        self._cursor_stride = np.ones(n_files, dtype=np.int64)
        self._cursor_initialized = np.zeros(n_files, dtype=np.bool_)

        self._merge_gap_tokens = max(seq_len // 2, 1)

        self._queue = queue.Queue(maxsize=8)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        if self.device.type == "cuda":
            self._prefetch_stream = torch.cuda.Stream(device=self.device)

        self._stage_next_gpu_batch()

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)

        if self._cfg is None:
            self._init_pipeline(global_tokens, seq_len, grad_accum_steps)
        else:
            expected = (
                local_tokens,
                seq_len,
                local_tokens // seq_len,
                (local_tokens // seq_len) * self.world_size,
            )
            if self._cfg != expected:
                raise ValueError(
                    "DistributedTokenLoader received changing batch configuration after initialization, "
                    f"got global_tokens={global_tokens}, seq_len={seq_len}, grad_accum_steps={grad_accum_steps}"
                )

        if self._next_gpu_batch is None:
            self._stage_next_gpu_batch()

        if self.device.type == "cuda" and self._next_ready_event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._next_ready_event)

        batch = self._next_gpu_batch
        if batch is None:
            raise RuntimeError("Failed to prepare next batch")

        if self.device.type == "cuda":
            curr = torch.cuda.current_stream(self.device)
            batch[0].record_stream(curr)
            batch[1].record_stream(curr)

        self._stage_next_gpu_batch()
        return batch

# --- Transformer modules ---

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    _qat_alpha: float = 1.0
    _qat_start_step: int = 0
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            w32 = self.weight.float()
            row_max = w32.abs().amax(dim=1)
            scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
            scaled = w32 / scale[:, None]
            frac = scaled - scaled.floor()
            soft_rounded = scaled.floor() + torch.sigmoid(CastedLinear._qat_alpha * (frac - 0.5))
            w = (torch.clamp(soft_rounded, -31, 31) * scale[:, None]).to(x.dtype)
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
        y = flash_attn_3_func(q, k, v, causal=True)
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

class EngramLite(nn.Module):
    """Multi-head hash-based n-gram embedding with learned gating."""
    def __init__(self, num_buckets: int, num_heads: int, num_orders: int, dim_per_head: int, model_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.num_orders = num_orders
        self.dim_per_head = dim_per_head
        total_slots = num_orders * num_heads * num_buckets
        concat_dim = num_orders * num_heads * dim_per_head
        self.embed = nn.Embedding(total_slots, dim_per_head)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(concat_dim, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.ngram_gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
    def forward(self, input_ids: Tensor) -> Tensor:
        B = self.num_buckets
        prev_ids = F.pad(input_ids[:, :-1], (1, 0), value=0)
        bi_h0 = (prev_ids * 1009 + input_ids) % B
        bi_h1 = ((prev_ids * 2719 + 314159) ^ (input_ids * 3137)) % B
        indices = [bi_h0, bi_h1 + B]
        if self.num_orders >= 2:
            pp_ids = F.pad(prev_ids[:, :-1], (1, 0), value=0)
            tri_h0 = ((pp_ids * 36313) ^ (prev_ids * 27191) ^ (input_ids * 4903)) % B
            tri_h1 = ((pp_ids * 7919) ^ (prev_ids * 4391) ^ (input_ids * 6151)) % B
            offset = 2 * B
            indices.extend([tri_h0 + offset, tri_h1 + offset + B])
        all_idx = torch.stack(indices, dim=-1)
        all_emb = self.embed(all_idx)
        flat = all_emb.reshape(*input_ids.shape, -1)
        out = self.proj(flat)
        gate = torch.sigmoid(self.ngram_gate.to(dtype=out.dtype))[None, None, :]
        return out * gate

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
    def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        if HAS_FUSED_MLP and x.is_cuda and not IS_ROCM:
            return FusedLeakyReLUSqMLP.apply(x, up_w.to(x.dtype), down_w.to(x.dtype))
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
        return F.linear(x.square(), down_w.to(x.dtype))

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
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
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
        bigram_dim: int = 160,
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
        if bool(int(os.environ.get("ENGRAM", "0"))) and int(os.environ.get("NGRAM_BUCKETS", "0")) > 0:
            self.bigram = EngramLite(
                int(os.environ.get("NGRAM_BUCKETS", 8192)),
                int(os.environ.get("NGRAM_HEADS", 2)),
                int(os.environ.get("NGRAM_ORDERS", 2)),
                int(os.environ.get("NGRAM_DIM_PER_HEAD", 32)),
                model_dim)
        elif bigram_vocab_size > 0:
            self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim, trigram=bool(int(os.environ.get("TRIGRAM", "0"))))
        else:
            self.bigram = None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, model_dim, dtype=torch.float32))
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
                g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
                scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = torch.lerp(scaled_skip, x, g)
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
                g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
                scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = torch.lerp(scaled_skip, x, g)
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
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + seq_len - stride < total_tokens]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
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
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


def run_fused_ngram_eval(
    args: Hyperparameters,
    model_path: Path,
    stride: int,
    eval_seq_len: int,
    logger,
) -> dict[str, float]:
    eval_script = _THIS_DIR / "eval_fused.py"
    code_path = _THIS_DIR / "train_gpt.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--code", str(code_path),
        "--model", str(model_path),
        "--val-pattern", args.val_files,
        "--tokenizer", args.tokenizer_path,
        "--device", args.fused_ngram_device,
        "--stride", str(stride),
        "--seq-len", str(eval_seq_len),
        "--batch-seqs", str(args.fused_ngram_batch_seqs),
        "--base-beta", str(args.fused_ngram_base_beta),
        "--agree-bonus", str(args.fused_ngram_agree_bonus),
        "--within-threshold", str(args.fused_ngram_within_threshold),
        "--within-beta", str(args.fused_ngram_within_beta),
        "--word-threshold", str(args.fused_ngram_word_threshold),
        "--word-beta", str(args.fused_ngram_word_beta),
        "--open-table-bits", str(args.fused_ngram_open_table_bits),
        "--token-threshold-scale", str(args.fused_ngram_token_threshold_scale),
        "--order-stride", str(args.fused_ngram_order_stride),
        "--vocab-size", str(args.vocab_size),
        "--num-layers", str(args.num_layers),
        "--model-dim", str(args.model_dim),
        "--num-heads", str(args.num_heads),
        "--num-kv-heads", str(args.num_kv_heads),
        "--mlp-mult", str(args.mlp_mult),
        "--logit-softcap", str(args.logit_softcap),
        "--rope-base", str(args.rope_base),
        "--qk-gain-init", str(args.qk_gain_init),
        "--bigram-vocab-size", str(args.bigram_vocab_size),
        "--bigram-dim", str(args.bigram_dim),
        "--xsa-last-n", str(args.xsa_last_n),
        "--rope-dims", str(args.rope_dims),
        "--ve-enabled", str(int(args.ve_enabled)),
        "--ve-dim", str(args.ve_dim),
        "--ve-layers", str(args.ve_layers),
    ]
    logger(
        f"fused_ngram_eval:start device:{args.fused_ngram_device} stride:{stride} "
        f"order_stride:{args.fused_ngram_order_stride} bigram_dim:{args.bigram_dim}"
    )
    metrics: dict[str, float] = {}
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path.cwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        if line:
            logger(f"fused_ngram_eval:{line}")
        if line.startswith("Setup: "):
            metrics["setup_s"] = float(line.removeprefix("Setup: ").removesuffix("s"))
        elif line.startswith("Submission:"):
            metrics["submission_bpb"] = float(line.split("=", 1)[1].strip())
        elif line.startswith("Loop: "):
            loop_part, wall_part = [part.strip() for part in line.split("|", 1)]
            metrics["loop_s"] = float(loop_part.removeprefix("Loop: ").removesuffix("s"))
            metrics["wall_s"] = float(wall_part.removeprefix("Wall: ").removesuffix("s"))
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"fused_ngram_eval failed with exit code {ret}")
    required = {"submission_bpb", "loop_s", "wall_s"}
    missing = required.difference(metrics)
    if missing:
        raise RuntimeError(f"fused_ngram_eval missing metrics: {sorted(missing)}")
    return metrics


def generate_autoregressive_calib(model, device, num_seqs=64, seq_len=2048,
                                   vocab_size=1024, temperature=0.8, batch_size=8, seed=42):
    """Generate sequences autoregressively from the model for GPTQ calibration.
    No external data accessed — fully self-contained."""
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - batch_start)
            tokens = torch.randint(0, vocab_size, (bs, 1), device=device, generator=rng)
            for pos in range(seq_len - 1):
                logits = model.forward_logits(tokens)
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens = torch.cat([tokens, next_tok], dim=1)
            for i in range(bs):
                all_tokens.append(tokens[i:i+1])
    return all_tokens


def collect_hessians_from_tokens(hessian_model, token_seqs, device):
    """Collect H = X^T X from pre-generated token sequences."""
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
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    return hessians


# --- GPTQ-lite int6 quantization ---

def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"
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

def quantize_int6_gptq(weight, hessian=None, clip_range=31, block_size=128):
    """Full GPTQ: Hessian-aware int6 quantization with Cholesky error compensation.
    If hessian is None, falls back to percentile search."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _quantize_int6_percentile(t32, clip_range)
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

def _quantize_int6_percentile(t32, clip_range=31):
    """Fallback: percentile search (for 1D or no-Hessian cases)."""
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

def _unbank_state_dict(sd: dict[str, Tensor], num_layers: int) -> dict[str, Tensor]:
    """Convert 3D bank tensors into individual 2D tensors with standard names."""
    out: dict[str, Tensor] = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out

def _rebank_state_dict(sd: dict[str, Tensor], num_layers: int, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert individual 2D tensors back into 3D bank tensors."""
    out: dict[str, Tensor] = {}
    n = num_layers
    # Reconstruct banks from individual weight keys
    qo_slices = [None] * (2 * n)
    kv_slices = [None] * (2 * n)
    up_slices = [None] * n
    down_slices = [None] * n
    consumed = set()
    for i in range(n):
        qk = f"blocks.{i}.attn.c_q.weight"
        if qk in sd:
            qo_slices[i] = sd[qk]
            consumed.add(qk)
        ok = f"blocks.{i}.attn.proj.weight"
        if ok in sd:
            qo_slices[n + i] = sd[ok]
            consumed.add(ok)
        kk = f"blocks.{i}.attn.c_k.weight"
        if kk in sd:
            kv_slices[i] = sd[kk]
            consumed.add(kk)
        vk = f"blocks.{i}.attn.c_v.weight"
        if vk in sd:
            kv_slices[n + i] = sd[vk]
            consumed.add(vk)
        fk = f"blocks.{i}.mlp.fc.weight"
        if fk in sd:
            up_slices[i] = sd[fk]
            consumed.add(fk)
        dk = f"blocks.{i}.mlp.proj.weight"
        if dk in sd:
            down_slices[i] = sd[dk]
            consumed.add(dk)
    out["qo_bank"] = torch.stack(qo_slices).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv_slices).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up_slices).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down_slices).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out

# --- Non-banked model for Hessian collection ---
# This mirrors the unbanked state dict keys: blocks.{i}.attn.c_q/c_k/c_v/proj, blocks.{i}.mlp.fc/proj

class _HessianAttn(nn.Module):
    """Non-banked attention with CastedLinear layers for Hessian hooks."""
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False
    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)
    def forward(self, x, v_embed=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
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
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))

class _HessianMLP(nn.Module):
    """Non-banked MLP with CastedLinear layers for Hessian hooks."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, int(mlp_mult * dim), bias=False)
        self.proj = CastedLinear(int(mlp_mult * dim), dim, bias=False)
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class _HessianBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = _HessianAttn(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = _HessianMLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(self, x, x0, v_embed=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out

class _HessianGPT(nn.Module):
    """Non-banked GPT model matching unbanked state dict keys for Hessian collection."""
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, logit_softcap, rope_base, qk_gain_init,
                 bigram_vocab_size=0, bigram_dim=160, xsa_last_n=0,
                 rope_dims=0, ln_scale=False,
                 ve_enabled=False, ve_dim=128, ve_layers="9,10"):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        if bool(int(os.environ.get("ENGRAM", "0"))) and int(os.environ.get("NGRAM_BUCKETS", "0")) > 0:
            self.bigram = EngramLite(
                int(os.environ.get("NGRAM_BUCKETS", 8192)),
                int(os.environ.get("NGRAM_HEADS", 2)),
                int(os.environ.get("NGRAM_ORDERS", 2)),
                int(os.environ.get("NGRAM_DIM_PER_HEAD", 32)),
                model_dim)
        elif bigram_vocab_size > 0:
            self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim, trigram=bool(int(os.environ.get("TRIGRAM", "0"))))
        else:
            self.bigram = None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            _HessianBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                          layer_idx=i, ln_scale=ln_scale)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        kv_dim = num_kv_heads * (model_dim // num_heads)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
    def _get_ve(self, layer_idx, input_ids, ve_cache):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_cache['ve'] * self.ve_layer_scales[ve_idx].to(dtype=ve_cache['ve'].dtype)
    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        ve_cache = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
                scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = torch.lerp(scaled_skip, x, g)
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x_flat, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

def collect_hessians(hessian_model, train_loader, args, device, grad_accum_steps, num_batches=256):
    """Run calibration batches through a non-banked model, collecting H = X^T X for each CastedLinear."""
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
        for _ in range(num_batches):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            hessian_model(x, y)
    for h in hooks:
        h.remove()
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    hessian_model.train()
    return hessians

def _assign_bit_widths(hessians, quant_names, target_mb, code_bytes, default_cr=15):
    """Assign bit widths per layer based on Hessian trace sensitivity.
    Start with int5 (cr=15) for all, greedily promote most-sensitive to int6 (cr=31)
    until artifact would exceed target."""
    if not hessians:
        return {n: 31 for n in quant_names}  # fallback: all int6
    # Rank by Hessian trace (sensitivity)
    sensitivity = {}
    for name in quant_names:
        H = hessians.get(name)
        if H is not None:
            sensitivity[name] = H.diag().sum().item()
        else:
            sensitivity[name] = 0.0
    ranked = sorted(sensitivity.items(), key=lambda x: -x[1])
    # Start with all int5, promote most sensitive to int6
    clip_ranges = {name: default_cr for name in quant_names}
    # Promote top layers to int6 — each promotion adds ~0.125 bits/param ≈ param_count/8 bytes
    for name, trace in ranked:
        clip_ranges[name] = 31  # promote to int6
    return clip_ranges

def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str],
                        hessians: dict[str, Tensor] | None = None,
                        clip_ranges: dict[str, int] | None = None):
    num_layers_total = max(
        (int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
        default=0,
    ) + 1
    late_k_layers = set(range(num_layers_total - 2, num_layers_total))
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
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
        if cat in int6_cats and t.ndim >= 1:
            cr = clip_ranges.get(name, 31) if clip_ranges else 31
            bit_label = "int6" if cr >= 31 else "int5"
            H = hessians.get(name) if hessians else None
            if H is not None:
                q, s = quantize_int6_gptq(t, hessian=H, clip_range=cr)
            else:
                q, s = quantize_int6_per_row(t, clip_range=cr)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": bit_label}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
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
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

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
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    CastedLinear._qat_enabled = args.qat_enabled
    CastedLinear._qat_alpha = args.qat_alpha_start
    CastedLinear._qat_start_step = 0
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
    # No DDP -- Parallel Muon handles bank grad communication via reduce-scatter,
    # and non-bank grads are manually all-reduced before Adam steps.
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
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
        scalar_params.append(base_model.skip_gates)
    if base_model.bigram is not None:
        if hasattr(base_model.bigram, 'scale'):
            scalar_params.append(base_model.bigram.scale)
        if hasattr(base_model.bigram, 'ngram_gate'):
            scalar_params.append(base_model.bigram.ngram_gate)
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
        weight_decay=args.embed_wd,
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
    log0(f"fused_mlp:{HAS_FUSED_MLP}")
    log0(f"cutlass_evt:True")
    log0(f"compressor:{_COMPRESSOR}")
    log0(f"optimizer:standard_NS5")
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
        f"split_lr:bank_split:{args.bank_split} "
        f"matrix_lr_early:{args.matrix_lr_early} matrix_lr_late:{args.matrix_lr_late}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            raw = max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
            return max(raw, args.lr_floor)
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        raw = remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
        return max(raw, args.lr_floor)
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
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
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
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and step >= 2000:
            if not CastedLinear._qat_enabled:
                CastedLinear._qat_enabled = True
                CastedLinear._qat_start_step = step
                log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
            qat_progress = min((step - CastedLinear._qat_start_step) / max(args.qat_ramp_steps, 1), 1.0)
            CastedLinear._qat_alpha = (
                args.qat_alpha_start
                + (args.qat_alpha_end - args.qat_alpha_start) * qat_progress
            )
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
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
        if args.matrix_lr_early != args.matrix_lr or args.matrix_lr_late != args.matrix_lr:
            split = min(max(args.bank_split, 0), args.num_layers)
            early_scale = args.matrix_lr_early / args.matrix_lr
            late_scale = args.matrix_lr_late / args.matrix_lr
            with torch.no_grad():
                for bank in [base_model.qo_bank, base_model.kv_bank]:
                    if bank.grad is not None:
                        bank.grad[:split].mul_(early_scale)
                        bank.grad[split:args.num_layers].mul_(late_scale)
                        bank.grad[args.num_layers:args.num_layers + split].mul_(early_scale)
                        bank.grad[args.num_layers + split:].mul_(late_scale)
                for bank in [base_model.mlp_up_bank, base_model.mlp_down_bank]:
                    if bank.grad is not None:
                        bank.grad[:split].mul_(early_scale)
                        bank.grad[split:].mul_(late_scale)
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
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
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
    # Unbank 3D tensors into individual 2D tensors for quantization
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)
    # Full GPTQ: collect Hessians via a temporary non-banked model
    log0(f"gptq:building non-banked model for Hessian collection...")
    hessian_model = _HessianGPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in hessian_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(hessian_model)
    # Load unbanked weights into the non-banked model
    hessian_model.load_state_dict(
        {k: v.to(device) for k, v in unbanked_sd.items() if k in hessian_model.state_dict()},
        strict=False,
    )
    # Autoregressive self-generated calibration (no external data)
    log0("gptq:generating autoregressive calibration data (64 seqs x 2048 tokens, temp=0.8)...")
    base_model.load_state_dict(export_sd, strict=False)
    t_gen = time.perf_counter()
    ar_tokens = generate_autoregressive_calib(
        base_model, device, num_seqs=64, seq_len=args.train_seq_len,
        vocab_size=args.vocab_size, temperature=0.8, batch_size=8, seed=args.seed,
    )
    log0(f"gptq:generated {len(ar_tokens)} sequences in {time.perf_counter()-t_gen:.1f}s")
    log0("gptq:collecting hessians from autoregressive data...")
    hessians = collect_hessians_from_tokens(hessian_model, ar_tokens, device)
    log0(f"gptq:collected hessians for {len(hessians)} layers (AR self-gen)")
    del ar_tokens
    del hessian_model
    torch.cuda.empty_cache()
    # Hessian-based bit allocation: start int5 for all, greedily promote to int6
    quant_names = [n for n in unbanked_sd if _classify_param(n) in {"mlp", "attn"} and unbanked_sd[n].ndim >= 1 and unbanked_sd[n].numel() > 65536]
    use_mixed = args.mixed_quant
    if use_mixed:
        # Rank by Hessian trace, promote top layers to int6, rest int5
        sens = {n: hessians[n].diag().sum().item() if n in hessians else 0.0 for n in quant_names}
        ranked = sorted(sens.items(), key=lambda x: -x[1])
        # Greedy: start all int5, promote until target exceeded
        clip_ranges = {n: 15 for n in quant_names}  # int5 default
        n_int6 = min(max(args.n_int6_layers, 0), len(ranked))
        for name, _ in ranked[:n_int6]:
            clip_ranges[name] = 31
        int6_names = [n for n, cr in clip_ranges.items() if cr == 31]
        int5_names = [n for n, cr in clip_ranges.items() if cr == 15]
        log0(f"mixed_quant: {len(int6_names)} int6, {len(int5_names)} int5")
        log0(f"mixed_quant: int6 layers: {int6_names[:5]}...")
    else:
        clip_ranges = {n: 31 for n in quant_names}  # all int6
    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians, clip_ranges=clip_ranges)
    # NOVEL: Selective ±1 pruning by reconstruction error
    # Sort ±1 quantized values by their reconstruction error (scale²),
    # prune least-impactful first until artifact fits target size.
    target_mb = float(os.environ.get("TARGET_MB", "15.9"))
    code_bytes_est = len(code.encode("utf-8"))
    ones_info = []  # (tensor_key, flat_idx, error)
    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type") in ("int6", "int5")): continue
        qk, sk = name + ".q", name + ".scale"
        if qk not in quant_result or sk not in quant_result: continue
        q, s = quant_result[qk], quant_result[sk]
        if s.ndim > 0:
            ones_mask = (q.abs() == 1)
            if ones_mask.any():
                row_idx = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
                flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
                errors = s.float()[row_idx].pow(2)
                for fi, err in zip(flat_idx.tolist(), errors.tolist()):
                    ones_info.append((qk, fi, err))
    if ones_info:
        ones_info.sort(key=lambda x: x[2])
        def _try_prune(n):
            tmp = {k: v.clone() for k, v in quant_result.items()}
            for i in range(min(n, len(ones_info))):
                tmp[ones_info[i][0]].view(-1)[ones_info[i][1]] = 0
            buf = io.BytesIO(); torch.save({"w": tmp, "m": quant_meta}, buf)
            raw = buf.getvalue()
            if _COMPRESSOR == "brotli":
                return len(brotli.compress(raw, quality=11)) + code_bytes_est, tmp
            return len(lzma.compress(raw, preset=9)) + code_bytes_est, tmp
        no_sz, _ = _try_prune(0)
        target_bytes = int(target_mb * 1024 * 1024)
        log0(f"selective_prune: {len(ones_info)} ±1 candidates, unpruned={no_sz/(1024*1024):.2f}MB target={target_mb}MB")
        if no_sz <= target_bytes:
            log0("selective_prune: already fits, no pruning needed")
        else:
            full_sz, _ = _try_prune(len(ones_info))
            log0(f"selective_prune: full ±1 prune={full_sz/(1024*1024):.2f}MB")
            if full_sz > target_bytes:
                log0("selective_prune: even full prune not enough, applying all")
                _, quant_result = _try_prune(len(ones_info))
            else:
                lo, hi = 0, len(ones_info)
                while lo < hi:
                    mid = (lo + hi) // 2
                    sz, _ = _try_prune(mid)
                    if sz <= target_bytes: hi = mid
                    else: lo = mid + 1
                log0(f"selective_prune: pruning {lo}/{len(ones_info)} ±1 values ({100*lo/len(ones_info):.1f}%) to fit {target_mb}MB")
                _, quant_result = _try_prune(lo)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    if _COMPRESSOR == "brotli":
        quant_blob = brotli.compress(quant_raw, quality=11)
    else:
        quant_blob = lzma.compress(quant_raw, preset=9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{_COMPRESSOR}: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+{_COMPRESSOR}: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    # --- Sliding window eval on quantized model (proven baseline eval) ---
    log0("sliding_window_eval:loading quantized model...")
    quant_sd_for_eval = dequantize_mixed_int6(quant_result, quant_meta, unbanked_sd)
    rebanked_sd = _rebank_state_dict(quant_sd_for_eval, args.num_layers, export_sd)
    base_model.load_state_dict(rebanked_sd, strict=True)
    base_model.eval()
    sw_stride = 64 if 64 < effective_eval_seq_len else max(1, min(args.eval_stride, effective_eval_seq_len - 1))
    sw_val_loss, sw_val_bpb = eval_val_sliding(
        args, base_model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=sw_stride, batch_seqs=32, eval_seq_len=effective_eval_seq_len,
    )
    log0(f"final_int6_sliding_window val_loss:{sw_val_loss:.5f} val_bpb:{sw_val_bpb:.8f}")
    del compiled_model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    # --- N-gram tilt eval (rank 0 only, DDP already torn down) ---
    if args.fused_ngram_eval and master_process:
        fused_stride = 64 if 64 < effective_eval_seq_len else max(1, min(args.eval_stride, effective_eval_seq_len - 1))
        fused_metrics = run_fused_ngram_eval(
            args,
            Path("final_model.int6.ptz"),
            stride=fused_stride,
            eval_seq_len=effective_eval_seq_len,
            logger=log0,
        )
        log0(
            f"final_int6_fused_ngram_exact submission_val_bpb:{fused_metrics['submission_bpb']:.8f} "
            f"setup_s:{fused_metrics.get('setup_s', float('nan')):.1f} "
            f"loop_s:{fused_metrics['loop_s']:.1f} wall_s:{fused_metrics['wall_s']:.1f}"
        )
        log0(f"final_int6_fused_ngram_submission_exact val_bpb:{fused_metrics['submission_bpb']:.8f}")
if __name__ == "__main__":
    main()
