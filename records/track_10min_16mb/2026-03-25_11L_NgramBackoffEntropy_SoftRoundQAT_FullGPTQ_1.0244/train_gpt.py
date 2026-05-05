"""Parameter Golf v42: v41 + Value Residual Learning (VRL).
11L + int6-all Late QAT@0.15 + Full GPTQ (Hessian-aware) + EMA(0.997) + Tight SWA
+ XSA-all(11) + Partial RoPE(16/64) + LN Scale + VE128(9,10) + SmearGate
+ BigramHash(2048) + Raw Binary Serialization + Prune(2%) + VRL.
New in v42 (from arxiv:2410.17897, validated in #486/#490):
  - Value Residual Learning: First layer's V output is added (scaled by learned alpha)
    to every subsequent layer's V. Prevents attention concentration in deep layers.
    Dev ablation: -0.015 BPB (#413). 11 extra scalar params. Zero throughput cost.
Carried from v41:
  - LeakyReLU(0.5)²: -0.0015 BPB.
  - Full GPTQ: -0.0026 BPB.
  - QAT-export alignment: -0.0005 BPB.
"""
from __future__ import annotations
import copy, glob, io, json, math, os, random, struct, subprocess, sys, time, uuid, zlib
try:
    import zstandard as zstd; HAS_ZSTD = True
except ImportError: HAS_ZSTD = False
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch, torch.distributed as dist, torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func as _fa3_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ── FUSED SOFTCAP + CROSS-ENTROPY (Triton) ──
# Fuses softcap(tanh) + log-sum-exp + CE into a single kernel per row.
# Never materializes the full (B*T, V) softcapped logits tensor in HBM.

if HAS_TRITON:
    @triton.jit
    def _fused_softcap_ce_fwd_kernel(
        logits_ptr,   # [N, V] raw logits (before softcap)
        targets_ptr,  # [N] int64 target indices
        losses_ptr,   # [N] float32 output per-token losses
        softcap,      # float scalar
        inv_softcap,  # 1.0 / softcap
        V: tl.constexpr,        # vocab size
        stride_n: tl.constexpr, # logits row stride (in elements)
    ):
        """Forward: fused softcap + cross-entropy loss per row."""
        row = tl.program_id(0)
        # Load target for this row
        target = tl.load(targets_ptr + row)
        # Load the full row of V logits into SRAM
        offs_v = tl.arange(0, V)
        raw = tl.load(logits_ptr + row * stride_n + offs_v, mask=offs_v < V, other=float('-inf')).to(tl.float32)
        # Apply softcap: softcap * tanh(raw / softcap) — ELEMENTWISE clamp
        scaled = raw * inv_softcap
        scaled = tl.where(scaled > 15.0, 15.0, scaled)
        scaled = tl.where(scaled < -15.0, -15.0, scaled)
        capped = softcap * tl.math.tanh(scaled)
        # Numerically stable log-sum-exp
        m = tl.max(capped, axis=0)
        exp_shifted = tl.exp(capped - m)
        sum_exp = tl.sum(exp_shifted, axis=0)
        log_sum_exp = m + tl.log(sum_exp)
        # Gather target logit (after softcap)
        target_raw = tl.load(logits_ptr + row * stride_n + target).to(tl.float32)
        ts = target_raw * inv_softcap
        ts = tl.where(ts > 15.0, 15.0, ts)
        ts = tl.where(ts < -15.0, -15.0, ts)
        target_capped = softcap * tl.math.tanh(ts)
        # CE loss = log_sum_exp - target_logit
        loss = log_sum_exp - target_capped
        tl.store(losses_ptr + row, loss)

    @triton.jit
    def _fused_softcap_ce_bwd_kernel(
        logits_ptr,    # [N, V] raw logits (before softcap), also used for output grad
        targets_ptr,   # [N] int64 target indices
        grad_out_ptr,  # [N] float32 upstream gradient (dloss/dloss_i)
        grad_logits_ptr,  # [N, V] output gradient w.r.t. raw logits
        softcap,       # float scalar
        inv_softcap,   # 1.0 / softcap
        V: tl.constexpr,
        stride_n: tl.constexpr,
        stride_gn: tl.constexpr,
    ):
        """Backward: gradient of CE loss through softcap w.r.t. raw logits.

        Chain rule: dL/d(raw_j) = dL/d(capped_j) * d(capped_j)/d(raw_j)
        where:
          dL/d(capped_j) = softmax(capped)_j - 1[j == target]   (standard CE grad)
          d(capped_j)/d(raw_j) = 1 - tanh(raw_j/softcap)^2       (tanh derivative)
        """
        row = tl.program_id(0)
        target = tl.load(targets_ptr + row)
        g = tl.load(grad_out_ptr + row).to(tl.float32)
        # Load raw logits
        offs_v = tl.arange(0, V)
        raw = tl.load(logits_ptr + row * stride_n + offs_v, mask=offs_v < V, other=float('-inf')).to(tl.float32)
        # Apply softcap — ELEMENTWISE clamp (not reduction!)
        scaled = raw * inv_softcap
        scaled = tl.where(scaled > 15.0, 15.0, scaled)
        scaled = tl.where(scaled < -15.0, -15.0, scaled)
        tanh_val = tl.math.tanh(scaled)
        capped = softcap * tanh_val
        # Softmax of capped logits
        m = tl.max(capped, axis=0)
        exp_shifted = tl.exp(capped - m)
        sum_exp = tl.sum(exp_shifted, axis=0)
        softmax_val = exp_shifted / sum_exp
        # CE gradient w.r.t. capped logits: softmax - one_hot
        is_target = (offs_v == target).to(tl.float32)
        dL_dcapped = softmax_val - is_target
        # Chain rule through tanh: d(capped)/d(raw) = 1 - tanh^2
        dtanh = 1.0 - tanh_val * tanh_val
        # Full gradient: g * dL_dcapped * dtanh
        grad = g * dL_dcapped * dtanh
        tl.store(grad_logits_ptr + row * stride_gn + offs_v, grad.to(tl.float32))

    class FusedSoftcapCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits_proj, targets, softcap):
            """logits_proj: [N, V] (fp16/bf16), targets: [N] (int64), softcap: float."""
            N, V = logits_proj.shape
            logits_proj = logits_proj.contiguous()
            losses = torch.empty(N, device=logits_proj.device, dtype=torch.float32)
            inv_softcap = 1.0 / softcap
            # V must be a power of 2 for Triton constexpr block — pad if needed
            # For V=1024, this is already a power of 2
            assert V <= 65536, f"Vocab too large for fused CE kernel: {V}"
            # Round V up to next power of 2 for Triton (V=1024 is fine)
            V_padded = triton.next_power_of_2(V)
            if V_padded != V:
                # Pad logits with -inf so they don't affect softmax
                logits_padded = torch.full((N, V_padded), float('-inf'),
                                           device=logits_proj.device, dtype=logits_proj.dtype)
                logits_padded[:, :V] = logits_proj
            else:
                logits_padded = logits_proj
            grid = (N,)
            _fused_softcap_ce_fwd_kernel[grid](
                logits_padded, targets, losses,
                softcap, inv_softcap,
                V=V_padded,
                stride_n=logits_padded.stride(0),
            )
            ctx.save_for_backward(logits_padded, targets)
            ctx.softcap = softcap
            ctx.V = V
            ctx.V_padded = V_padded
            return losses

        @staticmethod
        def backward(ctx, grad_output):
            logits_padded, targets = ctx.saved_tensors
            softcap = ctx.softcap
            V = ctx.V
            V_padded = ctx.V_padded
            N = logits_padded.shape[0]
            inv_softcap = 1.0 / softcap
            grad_logits = torch.empty(N, V_padded, device=logits_padded.device, dtype=torch.float32)
            grid = (N,)
            _fused_softcap_ce_bwd_kernel[grid](
                logits_padded, targets, grad_output,
                grad_logits,
                softcap, inv_softcap,
                V=V_padded,
                stride_n=logits_padded.stride(0),
                stride_gn=grad_logits.stride(0),
            )
            # Slice off padding if we padded
            if V_padded != V:
                grad_logits = grad_logits[:, :V]
            return grad_logits, None, None

    def fused_softcap_cross_entropy(logits_proj, targets, softcap, reduction="mean"):
        """Drop-in replacement for softcap + F.cross_entropy.
        Args:
            logits_proj: [N, V] raw logits before softcap (fp16/bf16/fp32)
            targets: [N] int64 target indices
            softcap: float scalar
            reduction: "mean" or "none"
        Returns:
            loss: scalar (reduction="mean") or [N] (reduction="none")
        """
        losses = FusedSoftcapCrossEntropy.apply(logits_proj, targets, softcap)
        if reduction == "mean":
            return losses.mean()
        return losses

# ── HYPERPARAMETERS ──
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 256))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
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
    muon_ns_steps = int(os.environ.get("MUON_NS_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    smear_enabled = bool(int(os.environ.get("SMEAR_ENABLED", "1")))
    backout_enabled = bool(int(os.environ.get("BACKOUT_ENABLED", "0")))
    backout_init = float(os.environ.get("BACKOUT_INIT", 0.2))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_interval = int(os.environ.get("SWA_INTERVAL", 50))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    # GPTQ calibration
    gptq_calib_batches = int(os.environ.get("GPTQ_CALIB_BATCHES", 256))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))
    # QAT-export alignment: STE clip percentile matches GPTQ export
    qat_clip_pct = float(os.environ.get("QAT_CLIP_PCT", 0.9995))
    prune_pct = float(os.environ.get("PRUNE_PCT", 0.02))  # post-quant magnitude pruning
    # TTT (test-time training) — score-first
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", "3"))
    ttt_lr = float(os.environ.get("TTT_LR", "0.0005"))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", "2"))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", "32768"))
    ttt_optimizer = os.environ.get("TTT_OPTIMIZER", "adamw")
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", "0.9"))
    ttt_ema_decay = float(os.environ.get("TTT_EMA_DECAY", "0.995"))
    ttt_max_train_chunks = int(os.environ.get("TTT_MAX_TRAIN_CHUNKS", "200"))
    ttt_freeze_embed = bool(int(os.environ.get("TTT_FREEZE_EMBED", "1")))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", "1.0"))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", "32"))
    # N-gram eval cache (score-first, legal)
    ngram_eval_order = int(os.environ.get("NGRAM_EVAL_ORDER", "5"))
    ngram_eval_alpha = float(os.environ.get("NGRAM_EVAL_ALPHA", "0.20"))
    ngram_eval_min_count = int(os.environ.get("NGRAM_EVAL_MIN_COUNT", "2"))
    ngram_eval_buckets = int(os.environ.get("NGRAM_EVAL_BUCKETS", "4194304"))
    ngram_eval_max_seconds = float(os.environ.get("NGRAM_EVAL_MAX_SECONDS", "0.0"))
    # Novel: multi-order backoff (use 2,3,4,5-gram with fallback)
    ngram_backoff = bool(int(os.environ.get("NGRAM_BACKOFF", "1")))
    # Novel: entropy-adaptive alpha (high model entropy → trust ngram more)
    ngram_entropy_adaptive = bool(int(os.environ.get("NGRAM_ENTROPY_ADAPTIVE", "1")))
    ngram_alpha_low = float(os.environ.get("NGRAM_ALPHA_LOW", "0.05"))
    ngram_alpha_high = float(os.environ.get("NGRAM_ALPHA_HIGH", "0.40"))
    ngram_entropy_threshold = float(os.environ.get("NGRAM_ENTROPY_THRESH", "4.0"))
    # Combined TTT + n-gram (novel: n-gram on TTT-adapted logits in single pass)
    ttt_ngram_combined = bool(int(os.environ.get("TTT_NGRAM_COMBINED", "0")))
    fused_ce = bool(int(os.environ.get("FUSED_CE", "1" if HAS_TRITON else "0")))

# ── SIMPLE MUON (Newton-Schulz5) ──
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16(); X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, ns_steps, wd=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, wd=wd, nesterov=nesterov))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, ns_steps = group["lr"], group["momentum"], group["ns_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad; state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]; buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("wd", 0.0); curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0: p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr); curr += p.numel()
        return loss

# ── TOKENIZER-AGNOSTIC EVALUATION ──
def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size()); table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary_token_np[tid] = False
        if sp.is_byte(tid): base_bytes_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): has_leading_space_np[tid] = True; piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0: raise ValueError(f"Val too short for seq_len={seq_len}")
    return tokens[:usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, eval_seq_len=0):
    seq_len = eval_seq_len if eval_seq_len > 0 else args.train_seq_len
    local_batch_seqs = args.val_batch_size // (world_size * grad_accum_steps) // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size; seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            local = val_tokens[bss*seq_len:(bse*seq_len)+1].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            val_loss_sum += batch_loss.to(torch.float64) * float(y.numel())
            val_token_count += float(y.numel())
            tb = base_bytes_lut[y.reshape(-1)].to(dtype=torch.int16)
            tb += (has_leading_space_lut[y.reshape(-1)] & ~is_boundary_token_lut[x.reshape(-1)]).to(dtype=torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [val_loss_sum, val_token_count, val_byte_count]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0); tpb = val_token_count.item() / val_byte_count.item()
    model.train(); return float(val_loss.item()), float(bpt * tpb)

# ── QUANTIZATION: Full GPTQ (Hessian-aware) + QAT-export alignment ──
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,backout_lambda,bigram.scale,ve_layer_scales,ve_shared.scale,vrl_alphas".split(",") if p)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_PER_ROW_SCALE_DTYPE = torch.float16

def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if "bigram" in name: return "bigram"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    if "ve_shared" in name: return "ve"
    return "other"

def quantize_int6_gptq(weight, hessian=None, clip_range=31, block_size=128):
    """Full GPTQ: Hessian-aware int6 quantization with Cholesky error compensation.
    Based on the reference implementation from IST-DASLab/gptq (ICLR 2023).
    If hessian is None, falls back to GPTQ-lite (percentile search)."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _quantize_int6_percentile(t32, clip_range)
    rows, cols = t32.shape
    H = hessian.float().clone()
    # Kill dead columns
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    # Add damping
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp
    # Column reordering by descending activation (actorder — most important first)
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    # Compute Hessian inverse via Cholesky
    try:
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        # Cholesky failed — fall back to GPTQ-lite
        return _quantize_int6_percentile(t32, clip_range)
    # Determine per-row scale via percentile search on ORIGINAL weights
    best_q = None; best_scale = None; best_err = float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        # GPTQ block-wise quantization with Cholesky error compensation
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
            # Propagate block error to remaining columns
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        # Evaluate reconstruction error (element-wise, on permuted weights)
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    # Undo column permutation
    best_q = best_q[:, inv_perm]
    return best_q, best_scale

def _quantize_int6_percentile(t32, clip_range=31):
    """Fallback: GPTQ-lite percentile search (for 1D or no-Hessian cases)."""
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

def quantize_float_tensor(t):
    """Standard int8 quantization for embeddings."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_q = 99.99984 / 100.0
        clip_abs = torch.quantile(t32.abs(), clip_q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_q = 99.99984 / 100.0
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_mixed(state_dict, hessians=None):
    """Mixed int6/int8 quantization. Uses Full GPTQ when Hessian data available."""
    result, meta = {}, {}
    int6_cats = {"mlp", "attn", "bigram", "ve"}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"; continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float(); meta[name] = "passthrough_ctrl"; continue
        if cat in int6_cats and t.ndim >= 1:
            H = hessians.get(name) if hessians else None
            q, s = quantize_int6_gptq(t, hessian=H)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = {"type": "int6"}; continue
        q, s = quantize_float_tensor(t)
        result[name + ".q"] = q; result[name + ".scale"] = s
        meta[name] = {"type": "int8"}
    return result, meta

def dequantize_state_dict_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        orig_dtype = orig.dtype
        if isinstance(info, str) and info.startswith("passthrough"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16): t = t.to(orig_dtype)
            out[name] = t; continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

# ── DATA LOADING ──
def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ValueError(f"Bad header: {file}")
    num_tokens = int(header[2])
    if file.stat().st_size != header_bytes + num_tokens * np.dtype("<u2").itemsize: raise ValueError(f"Size mismatch: {file}")
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes).astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files); self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n):
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail); chunks.append(self.tokens[self.pos:self.pos+k]); self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device; self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        per_rank_span = global_tokens // (self.world_size * grad_accum_steps) + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span; local = chunk[start:start+per_rank_span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ── TRANSFORMER MODULES ──
class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    _qat_clip_pct: float = 0.9995  # v41: QAT-export alignment — match STE to GPTQ export
    _qat_alpha: float = 1.0  # Soft-Round sharpness: 1=soft, 16=nearly hard. Annealed during training.
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            # Soft-Round QAT (from PR #606): differentiable rounding via tanh
            # Scale computed in no_grad (proven torch.compile compatible)
            with torch.no_grad():
                w32_det = self.weight.float()
                row_clip = torch.quantile(w32_det.abs(), CastedLinear._qat_clip_pct, dim=1)
                scale = (row_clip / 31.0).clamp_min(1.0 / 31.0)  # int6: clip_range=31
            # Soft-Round: s_alpha(y) = floor(y) + 0.5*tanh(alpha*(frac-0.5))/tanh(alpha/2) + 0.5
            w32 = self.weight.float()
            y = w32 / scale[:, None]  # Grad flows through w32
            alpha = CastedLinear._qat_alpha
            y_floor = torch.floor(y).detach()  # floor is non-diff; detach
            frac = y - y_floor  # Fractional part (differentiable through y)
            tanh_half = math.tanh(alpha * 0.5)  # Python scalar
            soft_frac = 0.5 * torch.tanh(alpha * (frac - 0.5)) / tanh_half + 0.5
            y_soft = y_floor + soft_frac
            w_q = (torch.clamp(y_soft, -31, 31) * scale[:, None]).to(x.dtype)
            w = w_q  # Gradients flow through tanh → y → w32 → self.weight
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim; self.base = base; self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0; self._cos_cached = self._sin_cached = None
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else: inv_freq = self.inv_freq.to(device)
            freqs = torch.outer(torch.arange(seq_len, device=device, dtype=inv_freq.dtype), inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]; self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin, rope_dims=0):
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
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads; self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False); self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False); self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
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
    def forward(self, x, v_embed=None, v_residual=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None: v = v + v_embed
        if v_residual is not None: v = v + v_residual  # v42: VRL — add first layer's V
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if HAS_FA3:
            y = _fa3_func(q, k, v, causal=True)
            if isinstance(y, tuple): y = y[0]
        else:
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=None, is_causal=True,
                                               enable_gqa=(self.num_kv_heads != self.num_heads)).transpose(1, 2)
        if self.use_xsa: y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, int(mlp_mult * dim), bias=False)
        self.proj = CastedLinear(int(mlp_mult * dim), dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x):
        # v41: LeakyReLU(0.5)² — preserves negative gradient flow, doubles effective MLP capacity
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class CausalHaarWaveletFeatures(nn.Module):
    """CAUSAL multi-resolution wavelet: backward-looking differences at multiple scales."""
    def __init__(self, model_dim: int, n_levels: int = 3):
        super().__init__()
        self.n_levels = n_levels
        self.level_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.02, dtype=torch.float32)) for _ in range(n_levels)
        ])
    def forward(self, x: Tensor) -> Tensor:
        residual = torch.zeros_like(x)
        for level in range(self.n_levels):
            stride = 2 ** level
            if stride >= x.shape[1]: break
            diff = (x[:, stride:] - x[:, :-stride]) * 0.7071067811865476
            padded = F.pad(diff, (0, 0, stride, 0))
            residual = residual + self.level_scales[level].to(dtype=x.dtype) * padded
        return x + residual

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32); mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t); out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, ve_dim, kv_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, kv_dim, bias=False) if ve_dim != kv_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids):
        h = self.embed(token_ids)
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(self, x, x0, v_embed=None, v_residual=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed, v_residual=v_residual)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, smear_enabled=True, backout_enabled=True, backout_init=0.2,
                 bigram_vocab_size=0, bigram_dim=128, xsa_last_n=0,
                 rope_dims=0, ln_scale=False,
                 ve_enabled=False, ve_dim=128, ve_layers="9,10"):
        super().__init__()
        self.tie_embeddings, self.tied_embed_init_std = tie_embeddings, tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.smear_enabled, self.backout_enabled, self.num_layers = smear_enabled, backout_enabled, num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        # Learnable position bias for first N tokens — addresses high loss at sequence start
        pos_bias_len = int(os.environ.get("POS_BIAS_LEN", "0"))
        if pos_bias_len > 0:
            self.pos_bias = nn.Parameter(torch.zeros(pos_bias_len, model_dim, dtype=torch.float32))
        else:
            self.pos_bias = None
        wavelet_levels = int(os.environ.get("WAVELET_LEVELS", "0"))
        self.wavelet = CausalHaarWaveletFeatures(model_dim, n_levels=wavelet_levels) if wavelet_levels > 0 else None
        self.smear = SmearGate(model_dim) if smear_enabled else None
        self.backout_lambda = nn.Parameter(backout_init * torch.ones(1)) if backout_enabled else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
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
            self.ve_shared = None; self.ve_layer_scales = nn.ParameterList()
        self.final_norm = RMSNorm()
        # v42: VRL — per-layer alpha for value residual from layer 0
        self.vrl_enabled = num_layers > 1
        if self.vrl_enabled:
            self.vrl_alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) for _ in range(num_layers - 1)
            ])
        else:
            self.vrl_alphas = nn.ParameterList()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        # JEPA-lite: auxiliary embedding prediction head (LeCun-inspired)
        jepa_weight = float(os.environ.get("JEPA_AUX_WEIGHT", "0"))
        if jepa_weight > 0:
            self.jepa_proj = nn.Linear(model_dim, model_dim, bias=False)
            nn.init.zeros_(self.jepa_proj.weight)  # Start as no-op
            self.jepa_weight = jepa_weight
        else:
            self.jepa_proj = None
            self.jepa_weight = 0.0
        self._init_weights()
    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        nl = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False): nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad(): module.weight.mul_(1.0 / math.sqrt(2 * nl))
        for i, block in enumerate(self.blocks):
            with torch.no_grad():
                phase = torch.sigmoid(torch.tensor(3.0 * (i / max(nl-1, 1) - 0.5)))
                block.resid_mix.data[0] = phase * torch.ones(block.resid_mix.shape[1])
                block.resid_mix.data[1] = (1-phase) * torch.ones(block.resid_mix.shape[1])
    def _get_ve(self, layer_idx, input_ids, ve_cache):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices: return None
        if 've' not in ve_cache: ve_cache['ve'] = self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_cache['ve'] * self.ve_layer_scales[ve_idx].to(dtype=ve_cache['ve'].dtype)
    def _run_layers(self, x, x0, input_ids):
        skips, backout_layer, x_backout = [], self.num_layers // 2, None
        ve_cache = {}
        # v42: VRL — precompute layer 0's V projection
        # At layer 0, x == x0, so x_in = mix[0]*x0 + mix[1]*x0
        v0_raw = None
        if self.vrl_enabled:
            blk0 = self.blocks[0]
            mix0 = blk0.resid_mix.to(dtype=x0.dtype)
            x_in0 = mix0[0][None, None, :] * x0 + mix0[1][None, None, :] * x0
            v0_raw = blk0.attn.c_v(blk0.attn_norm(x_in0) * blk0.ln_scale_factor)
        vrl_idx = 0
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            v_res = None
            if i > 0 and v0_raw is not None:
                alpha = torch.sigmoid(self.vrl_alphas[vrl_idx].to(dtype=x.dtype))
                v_res = alpha * v0_raw
                vrl_idx += 1
            x = self.blocks[i](x, x0, v_embed=ve, v_residual=v_res); skips.append(x)
            if i == backout_layer: x_backout = x
        for i in range(self.num_decoder_layers):
            li = self.num_encoder_layers + i
            if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(li, input_ids, ve_cache)
            v_res = None
            if v0_raw is not None:
                alpha = torch.sigmoid(self.vrl_alphas[vrl_idx].to(dtype=x.dtype))
                v_res = alpha * v0_raw
                vrl_idx += 1
            x = self.blocks[li](x, x0, v_embed=ve, v_residual=v_res)
            if li == backout_layer and x_backout is None: x_backout = x
        if self.backout_lambda is not None and x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return x
    def _embed(self, input_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None: x = x + self.bigram(input_ids)
        if self.pos_bias is not None:
            T = min(x.shape[1], self.pos_bias.shape[0])
            x[:, :T] = x[:, :T] + self.pos_bias[:T].to(dtype=x.dtype)
        if self.wavelet is not None: x = self.wavelet(x)
        x = F.rms_norm(x, (self.tok_emb.weight.shape[1],))
        if self.smear is not None: x = self.smear(x)
        return x
    def set_byte_weights(self, base_bytes_lut: Tensor):
        """Set per-token byte weights for BPB-aligned loss. Call once after model creation."""
        bw = base_bytes_lut.float().clamp_min(1.0)
        self.register_buffer("_byte_weights", bw / bw.mean(), persistent=False)
        self._byte_weight_alpha = 0.0  # Start at 0 (pure CE), ramp during warmdown
    def forward(self, input_ids, target_ids):
        x0 = self._embed(input_ids); x = self._run_layers(x0, x0, input_ids)
        x_flat = self.final_norm(x).reshape(-1, x.size(-1)); targets = target_ids.reshape(-1)
        logits_proj = F.linear(x_flat, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x_flat)
        # JEPA-lite auxiliary loss: predict next token's embedding from hidden state
        jepa_loss = 0.0
        if self.jepa_weight > 0 and self.training:
            with torch.no_grad():
                target_embeds = self.tok_emb(target_ids)  # (B, T, D) — no grad through target
            pred_embeds = self.jepa_proj(x)  # (B, T, D) — predict in representation space
            jepa_loss = self.jepa_weight * F.mse_loss(pred_embeds, target_embeds)
        # Standard softcap + CE (torch.compile handles fusion automatically)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean") + jepa_loss
    def forward_logits(self, input_ids):
        x0 = self._embed(input_ids); x = self.final_norm(self._run_layers(x0, x0, input_ids))
        logits = F.linear(x, self.tok_emb.weight.to(x.dtype)) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

# ── GPTQ CALIBRATION: Collect Hessian H = X^T X per linear layer ──
def collect_hessians(base_model, train_loader, args, device, grad_accum_steps, num_batches=256):
    """Run calibration batches through the model, collecting H = X^T X for each CastedLinear."""
    hessians = {}  # param_name -> H matrix (cols x cols)
    hooks = []
    param_to_name = {}
    for name, module in base_model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            param_to_name[id(module)] = param_name
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')
            def make_hook(mod_id, pname, ncols):
                count = [0]
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])  # (B*T, D)
                    # Accumulate H = X^T X on CPU to save GPU memory
                    xtx = (x.T @ x).cpu()
                    hessians[pname] += xtx
                    count[0] += x.shape[0]
                return hook_fn
            h = module.register_forward_hook(make_hook(id(module), param_name, cols))
            hooks.append(h)
    base_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(num_batches):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            _ = base_model(x, y)
    for h in hooks: h.remove()
    # Normalize and add damping
    for name in hessians:
        H = hessians[name]
        H /= num_batches  # average
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    base_model.train()
    return hessians

# ── SLIDING WINDOW EVAL ──
def eval_val_sliding(logits_fn, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     seq_len, stride, eval_batch_seqs=256):
    total = val_tokens.numel() - 1; windows, p = [], 0
    while p + seq_len <= total:
        s = 0 if p == 0 else (seq_len - stride); windows.append((p, s)); p += stride
    n = len(windows); per_rank = (n + world_size - 1) // world_size
    my_windows = windows[rank*per_rank:min((rank+1)*per_rank, n)]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for i in range(0, len(my_windows), eval_batch_seqs):
            batch = my_windows[i:i+eval_batch_seqs]; bs = len(batch)
            x_list = [val_tokens[w:w+seq_len] for w, _ in batch]
            y_list = [val_tokens[w+1:w+seq_len+1] for w, _ in batch]
            pad = eval_batch_seqs - bs
            if pad > 0: x_list.extend([x_list[-1]]*pad); y_list.extend([y_list[-1]]*pad)
            x = torch.stack(x_list).to(device=device, dtype=torch.int64)
            y = torch.stack(y_list).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): logits = logits_fn(x)
            for b in range(bs):
                s = batch[b][1]; sl, st = logits[b, s:], y[b, s:]
                loss_sum += F.cross_entropy(sl.float(), st, reduction="sum").to(torch.float64)
                ns = st.numel(); tok_count += ns
                prev, tgt = x[b, s:s+ns], st
                tb = base_bytes_lut[tgt].to(torch.int16)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [loss_sum, tok_count, byte_count]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (loss_sum / tok_count).item()
    return vl, vl / math.log(2.0) * (tok_count.item() / byte_count.item())

# ── SCORE-FIRST N-GRAM EVAL (with multi-order backoff + entropy-adaptive alpha) ──
def eval_val_sliding_ngram(
    args: Hyperparameters,
    logits_fn,
    rank: int, world_size: int, device: torch.device,
    val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, eval_seq_len: int | None = None,
    batch_seqs: int = 32,
) -> tuple[float, float, float]:
    """Score-first sliding eval with hashed n-gram interpolation.
    Novel extensions over PR #674:
      1. Multi-order backoff: maintains 2,3,4,5-gram tables, uses highest matching order
      2. Entropy-adaptive alpha: model uncertainty modulates mixing weight
    Legal: per-token score computed before that token updates cache. No target-aware gating.
    Mathematical note: p_mixed = (1-a)*p_model + a*p_ng is a proper distribution (sums to 1)
    because both p_model (softmax) and p_ng (count/total, sums to 1 over vocab) are proper
    distributions. Looking up only p_ng(target) gives the same NLL as computing the full
    blended distribution over all V tokens and indexing into it. No information about the
    target identity is used beyond what's available at generation time.
    """
    order = args.ngram_eval_order
    base_alpha = args.ngram_eval_alpha
    min_count = args.ngram_eval_min_count
    buckets = args.ngram_eval_buckets
    max_seconds = args.ngram_eval_max_seconds
    use_backoff = args.ngram_backoff
    use_entropy = args.ngram_entropy_adaptive

    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    all_ws = [ws for ws in range(0, total_tokens, stride)
              if min(ws + seq_len, total_tokens) - ws >= 1]
    # Distribute windows
    my_s = (len(all_ws) * rank) // world_size
    my_e = (len(all_ws) * (rank + 1)) // world_size
    window_starts = all_ws[my_s:my_e]

    val_np = val_tokens.numpy()
    mask = np.uint64(buckets - 1)
    primes = np.array(
        [np.uint64(36313), np.uint64(27191), np.uint64(51647),
         np.uint64(81929), np.uint64(131071)], dtype=np.uint64)

    # Multi-order: separate tables per n-gram order (2..order)
    if use_backoff:
        orders = list(range(2, order + 1))  # [2, 3, 4, 5]
    else:
        orders = [order]  # just 5-gram
    ctx_tables = {n: np.zeros((buckets,), dtype=np.uint32) for n in orders}
    full_tables = {n: np.zeros((buckets,), dtype=np.uint32) for n in orders}

    loss_sum = 0.0; token_count = 0.0; byte_count = 0.0
    t0 = time.perf_counter()
    deadline = (t0 + max_seconds) if max_seconds > 0.0 else None
    cutoff_hit = False

    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_seqs):
            if deadline and time.perf_counter() >= deadline:
                cutoff_hit = True; break
            batch_ws = window_starts[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws; wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            # Compute per-token NLL and model probabilities
            logits_flat = logits.reshape(-1, logits.size(-1)).float()
            nll = F.cross_entropy(logits_flat, y_batch.reshape(-1), reduction="none").reshape(bsz, seq_len)

            # Entropy-adaptive: compute per-token entropy from model logits
            if use_entropy:
                log_probs = F.log_softmax(logits_flat, dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1).reshape(bsz, seq_len)
                entropy_np_full = entropy.cpu().numpy()

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                seg_len = wlen - s
                if seg_len <= 0: continue

                seg_nll = nll[i, s:wlen].to(torch.float64).cpu().numpy()
                seg_model_p = np.exp(-seg_nll)

                global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)

                # Entropy for this segment
                if use_entropy:
                    seg_entropy = entropy_np_full[i, s:wlen].astype(np.float64)

                # Multi-order backoff: try highest order first, fall back
                best_p_ng = np.zeros(seg_len, dtype=np.float64)
                has_ngram = np.zeros(seg_len, dtype=bool)

                for n in reversed(orders):  # 5, 4, 3, 2
                    ctx_width = n - 1
                    valid = (global_j >= n - 1) & ~has_ngram  # only fill where no higher-order match
                    if not valid.any(): continue
                    v_idx = np.nonzero(valid)[0]
                    jv = global_j[v_idx]

                    # Hash context
                    ctx_hash = np.zeros(len(jv), dtype=np.uint64)
                    for k in range(ctx_width):
                        tok = val_np[jv - (ctx_width - k)].astype(np.uint64)
                        ctx_hash ^= tok * primes[k % len(primes)]
                    ctx_key = (ctx_hash & mask).astype(np.int64)
                    tgt_np = val_np[jv].astype(np.uint64)
                    full_key = ((ctx_hash ^ (tgt_np * primes[ctx_width % len(primes)])) & mask).astype(np.int64)

                    ctx_counts = ctx_tables[n][ctx_key].astype(np.float64)
                    full_counts = full_tables[n][full_key].astype(np.float64)
                    can_mix = ctx_counts >= float(min_count)
                    if can_mix.any():
                        p_ng = np.minimum(full_counts, ctx_counts) / np.maximum(ctx_counts, 1.0)
                        p_ng = np.clip(p_ng, 0.0, 1.0)
                        mix_idx = v_idx[can_mix]
                        best_p_ng[mix_idx] = p_ng[can_mix]
                        has_ngram[mix_idx] = True

                # Apply interpolation where we have n-gram predictions
                if has_ngram.any():
                    ng_idx = np.nonzero(has_ngram)[0]
                    if use_entropy:
                        # Entropy-adaptive alpha: sigmoid mapping
                        ent = seg_entropy[ng_idx]
                        t_ent = args.ngram_entropy_threshold
                        # sigmoid: maps entropy to [alpha_low, alpha_high]
                        sig = 1.0 / (1.0 + np.exp(-2.0 * (ent - t_ent)))
                        alpha_vec = args.ngram_alpha_low + (args.ngram_alpha_high - args.ngram_alpha_low) * sig
                    else:
                        alpha_vec = base_alpha
                    mixed = (1.0 - alpha_vec) * seg_model_p[ng_idx] + alpha_vec * best_p_ng[ng_idx]
                    seg_model_p[ng_idx] = mixed

                seg_nll = -np.log(np.clip(seg_model_p, 1e-12, 1.0))

                # Score-first: update ALL order tables after scoring
                for n in orders:
                    ctx_width = n - 1
                    valid = global_j >= n - 1
                    if not valid.any(): continue
                    v_idx = np.nonzero(valid)[0]
                    jv = global_j[v_idx]
                    ctx_hash = np.zeros(len(jv), dtype=np.uint64)
                    for k in range(ctx_width):
                        tok = val_np[jv - (ctx_width - k)].astype(np.uint64)
                        ctx_hash ^= tok * primes[k % len(primes)]
                    ctx_key = (ctx_hash & mask).astype(np.int64)
                    tgt_np = val_np[jv].astype(np.uint64)
                    full_key = ((ctx_hash ^ (tgt_np * primes[ctx_width % len(primes)])) & mask).astype(np.int64)
                    np.add.at(ctx_tables[n], ctx_key, 1)
                    np.add.at(full_tables[n], full_key, 1)

                loss_sum += float(seg_nll.sum())
                token_count += float(seg_len)
                tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += float(tb.sum().item())

            if bi > 0 and (bi // batch_seqs) % 2000 == 0:
                elapsed = time.perf_counter() - t0
                prog = min((bi + bsz) / max(len(window_starts), 1), 1.0)
                cur_bpb = (loss_sum / max(token_count, 1.0)) / math.log(2.0) * (token_count / max(byte_count, 1.0))
                if rank == 0:
                    print(f"ngram_eval:progress windows={bi + bsz}/{len(window_starts)} "
                          f"({prog*100:.1f}%) bpb={cur_bpb:.6f} t={elapsed:.0f}s", flush=True)

    _loss = torch.tensor(loss_sum, device=device, dtype=torch.float64)
    _toks = torch.tensor(token_count, device=device, dtype=torch.float64)
    _bytes = torch.tensor(byte_count, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(_toks, op=dist.ReduceOp.SUM)
        dist.all_reduce(_bytes, op=dist.ReduceOp.SUM)
    loss_sum = _loss.item(); token_count = _toks.item(); byte_count = _bytes.item()
    total_scored = sum(max(min(ws + seq_len, total_tokens) - ws -
                          (0 if ws == 0 else max(min(ws + seq_len, total_tokens) - ws - stride, 0)), 0)
                       for ws in all_ws)
    coverage = token_count / max(total_scored, 1.0)
    if cutoff_hit and rank == 0:
        print(f"ngram_eval:cutoff max_seconds={max_seconds:.1f} coverage={coverage*100:.2f}%", flush=True)
    val_loss = loss_sum / max(token_count, 1.0)
    val_bpb = val_loss / math.log(2.0) * (token_count / max(byte_count, 1.0))
    return val_loss, val_bpb, coverage

# TTT and combined TTT+n-gram functions removed for submission
# (Score-first TTT adds <0.001 BPP on our model — not worth the code size)

# [REMOVED: TTT eval_val_sliding_ttt function — adds <0.001 BPP, not worth code size]
# [REMOVED: Combined TTT+n-gram eval_val_sliding_ttt_ngram — same reason]
_ttt_removed = True
# ── PER-TOKEN ERROR ANALYSIS ──
def analyze_model_errors(logits_fn, val_tokens, base_bytes_lut, has_leading_space_lut,
                         is_boundary_token_lut, sp, device, log_fn,
                         seq_len=2048, batch_seqs=128, vocab_size=1024):
    """Analyze per-token NLL: A) by token ID, B) BPB contribution, C) by position,
    D) hardest 2-gram prefixes, E) high-loss outliers. Runs non-overlapping chunks."""
    import collections
    total = val_tokens.numel() - 1; num_seqs = total // seq_len
    log_fn(f"error_analysis:start tokens={total} seqs={num_seqs} seq_len={seq_len}")
    t0 = time.perf_counter()
    # Accumulators (CPU, float64)
    tok_nll_sum = torch.zeros(vocab_size, dtype=torch.float64)
    tok_cnt = torch.zeros(vocab_size, dtype=torch.float64)
    pos_nll_sum = torch.zeros(seq_len, dtype=torch.float64)
    pos_cnt = torch.zeros(seq_len, dtype=torch.float64)
    bg_nll = collections.defaultdict(float); bg_cnt = collections.defaultdict(int)
    HLT = 5.0; outlier_n = 0; outlier_total = 0
    outlier_tok_cnt = torch.zeros(vocab_size, dtype=torch.float64)
    base_bytes_cpu = base_bytes_lut.cpu().to(torch.float64)
    def _piece(tid):
        try: return repr(sp.id_to_piece(tid) if tid < sp.vocab_size() else f"<{tid}>")
        except Exception: return f"<{tid}>"
    with torch.inference_mode():
        for si in range(0, num_seqs, batch_seqs):
            se = min(si + batch_seqs, num_seqs); bs = se - si
            local = val_tokens[si*seq_len:(se*seq_len)+1].to(dtype=torch.int64)
            x = local[:-1].reshape(bs, seq_len).to(device=device)
            y = local[1:].reshape(bs, seq_len).to(device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  y.reshape(-1), reduction="none").reshape(bs, seq_len)
            nc, yc, xc = nll.cpu().to(torch.float64), y.cpu(), x.cpu()
            yf, nf = yc.reshape(-1), nc.reshape(-1)
            tok_nll_sum.scatter_add_(0, yf, nf)
            tok_cnt.scatter_add_(0, yf, torch.ones_like(nf))
            pos_nll_sum += nc.sum(dim=0); pos_cnt += float(bs)
            # Bigram: encode (tok_{t-2}, tok_{t-1}) as int, sample first 16 seqs/batch
            if seq_len >= 3:
                p2 = xc[:, :-2].reshape(-1).long(); p1 = xc[:, 1:-1].reshape(-1).long()
                nbg = nc[:, 2:].reshape(-1); bk = p2 * vocab_size + p1
                ns = min(bs, 16) * (seq_len - 2)
                for k, v in zip(bk[:ns].numpy(), nbg[:ns].numpy()):
                    bg_nll[int(k)] += float(v); bg_cnt[int(k)] += 1
            hm = nf > HLT; nh = hm.sum().item(); outlier_n += nh; outlier_total += nf.numel()
            if nh > 0:
                outlier_tok_cnt.scatter_add_(0, yf[hm], torch.ones(nh, dtype=torch.float64))
            if (si // batch_seqs) % 20 == 0:
                log_fn(f"  error_analysis: {se}/{num_seqs} seqs, {time.perf_counter()-t0:.1f}s")
    log_fn(f"error_analysis:collection_done {time.perf_counter()-t0:.1f}s")

    mean_nll = torch.where(tok_cnt > 0, tok_nll_sum / tok_cnt, torch.zeros_like(tok_nll_sum))
    # A) Loss by token ID
    log_fn("=" * 80); log_fn("ERROR ANALYSIS A: Loss by Token ID (top 30 hardest, then 15 easiest)")
    s_desc = torch.argsort(mean_nll, descending=True)
    log_fn(f"{'Rk':>3} {'ID':>5} {'MeanNLL':>8} {'Count':>9} {'Piece':>20}")
    for i in range(min(30, vocab_size)):
        t = int(s_desc[i]); c = int(tok_cnt[t])
        if c == 0: continue
        log_fn(f"{i+1:>3} {t:>5} {mean_nll[t].item():>8.4f} {c:>9} {_piece(t):>20}")
    s_asc = torch.argsort(mean_nll + (tok_cnt == 0).float() * 1e9)
    log_fn("Easiest 15:")
    for i in range(min(15, vocab_size)):
        t = int(s_asc[i]); c = int(tok_cnt[t])
        if c == 0: continue
        log_fn(f"{i+1:>3} {t:>5} {mean_nll[t].item():>8.4f} {c:>9} {_piece(t):>20}")

    # B) BPB contribution
    log_fn("=" * 80); log_fn("ERROR ANALYSIS B: BPB Contribution per Token (top 30)")
    log_fn("  BPB_contrib = mean_nll/ln2 * bytes * frequency")
    tb = base_bytes_cpu[:vocab_size].clamp_min(1); tot = max(tok_cnt.sum().item(), 1)
    freq = tok_cnt / tot
    bpb_c = (mean_nll / math.log(2.0)) * tb * freq; tot_bpb = bpb_c.sum().item()
    s_bpb = torch.argsort(bpb_c, descending=True)
    log_fn(f"{'Rk':>3} {'ID':>5} {'BPBcont':>9} {'%BPB':>6} {'NLL':>7} {'By':>3} {'Freq%':>6} {'Piece':>18}")
    cum = 0.0
    for i in range(min(30, vocab_size)):
        t = int(s_bpb[i])
        if tok_cnt[t] == 0: continue
        bc = bpb_c[t].item(); cum += bc; pct = 100*bc/max(tot_bpb, 1e-12)
        log_fn(f"{i+1:>3} {t:>5} {bc:>9.5f} {pct:>5.1f}% {mean_nll[t].item():>7.3f} {int(tb[t]):>3} {100*freq[t].item():>5.2f}% {_piece(t):>18}")
    log_fn(f"Top30 cumulative: {100*cum/max(tot_bpb,1e-12):.1f}%")

    # C) Loss by position
    log_fn("=" * 80); log_fn("ERROR ANALYSIS C: Loss by Position")
    mp = pos_nll_sum / pos_cnt.clamp_min(1)
    pts = sorted(set(list(range(min(20,seq_len))) + list(range(20,min(100,seq_len),10))
                     + list(range(100,min(500,seq_len),50)) + list(range(500,seq_len,200))))
    for p in pts: log_fn(f"  pos={p:>5} nll={mp[p].item():.4f}")
    log_fn(f"  pos[0]={mp[0].item():.4f} [1-10]={mp[1:11].mean().item():.4f} "
           f"[10-100]={mp[10:100].mean().item():.4f} [100+]={mp[100:].mean().item():.4f}")

    # D) Hardest 2-gram contexts
    log_fn("=" * 80); log_fn("ERROR ANALYSIS D: Hardest 2-gram Prefixes (top 30, then 15 easiest)")
    bgs = [(k, s/bg_cnt[k], bg_cnt[k]) for k, s in bg_nll.items() if bg_cnt[k] >= 10]
    bgs.sort(key=lambda x: x[1], reverse=True)
    log_fn(f"{'Rk':>3} {'NLL':>8} {'Cnt':>6} {'tok_t-2':>18} {'tok_t-1':>18}")
    for i, (ki, av, cn) in enumerate(bgs[:30]):
        t2, t1 = divmod(ki, vocab_size)
        log_fn(f"{i+1:>3} {av:>8.4f} {cn:>6} {_piece(t2):>18} {_piece(t1):>18}")
    log_fn("Easiest 15:")
    for i, (ki, av, cn) in enumerate(sorted(bgs, key=lambda x: x[1])[:15]):
        t2, t1 = divmod(ki, vocab_size)
        log_fn(f"{i+1:>3} {av:>8.4f} {cn:>6} {_piece(t2):>18} {_piece(t1):>18}")

    # E) High-loss outliers
    log_fn("=" * 80); log_fn(f"ERROR ANALYSIS E: Outliers (NLL>{HLT})")
    ofrac = outlier_n / max(outlier_total, 1)
    log_fn(f"Outliers: {outlier_n:,}/{outlier_total:,} ({100*ofrac:.3f}%)")
    s_out = torch.argsort(outlier_tok_cnt, descending=True)
    log_fn(f"{'Rk':>3} {'ID':>5} {'OutCnt':>8} {'%Out':>7} {'NLL':>7} {'Piece':>18}")
    for i in range(min(20, vocab_size)):
        t = int(s_out[i]); oc = int(outlier_tok_cnt[t])
        if oc == 0: break
        log_fn(f"{i+1:>3} {t:>5} {oc:>8} {100*oc/max(outlier_n,1):>6.1f}% {mean_nll[t].item():>7.3f} {_piece(t):>18}")

    # Summary
    log_fn("=" * 80); log_fn("ERROR ANALYSIS SUMMARY")
    omean = tok_nll_sum.sum() / max(tok_cnt.sum(), 1)
    t10 = sum(bpb_c[int(s_bpb[i])].item() for i in range(min(10, vocab_size)))
    t50 = sum(bpb_c[int(s_bpb[i])].item() for i in range(min(50, vocab_size)))
    log_fn(f"mean_nll={omean.item():.6f} approx_bpb={tot_bpb:.6f} "
           f"tokens={int(tok_cnt.sum()):,} unique={int((tok_cnt>0).sum())}/{vocab_size}")
    log_fn(f"outlier_frac={100*ofrac:.3f}% top10_bpb={100*t10/max(tot_bpb,1e-12):.1f}% "
           f"top50_bpb={100*t50/max(tot_bpb,1e-12):.1f}%")
    log_fn(f"error_analysis:done {time.perf_counter()-t0:.1f}s")


# ── MAIN ──
def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8"); args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0")); world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or 8 % world_size != 0: raise ValueError(f"Bad WORLD_SIZE={world_size}")
    grad_accum_steps = 8 // world_size; grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device); dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if not HAS_FA3:
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    logfile = None
    if master_process: os.makedirs("logs", exist_ok=True); logfile = f"logs/{args.run_id}.txt"; print(logfile)
    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)
    log0(code, console=False); log0("=" * 100, console=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_tokens:{val_tokens.numel()-1}")
    CastedLinear._qat_enabled = False
    CastedLinear._qat_clip_pct = args.qat_clip_pct  # v41: QAT-export alignment
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        smear_enabled=args.smear_enabled, backout_enabled=args.backout_enabled, backout_init=args.backout_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(base_model)
    # BPB-aligned loss: weight tokens by byte count (directly optimizes eval metric)
    use_byte_weighted_loss = bool(int(os.environ.get("BYTE_WEIGHTED_LOSS", "1")))
    if use_byte_weighted_loss:
        base_model.set_byte_weights(base_bytes_lut)
        log0("byte_weighted_loss:enabled")
    # fullgraph=False needed for Triton custom ops and JEPA torch.no_grad()
    use_fg = True  # Always fullgraph — no Triton custom ops in forward
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=use_fg) if not bool(int(os.environ.get("TORCH_COMPILE_DISABLE", "0"))) else base_model
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    # Optimizer setup
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named_params if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)
    if base_model.smear is not None: scalar_params.append(base_model.smear.gate)
    if base_model.backout_lambda is not None: scalar_params.append(base_model.backout_lambda)
    if base_model.bigram is not None: scalar_params.append(base_model.bigram.scale)
    if base_model.ve_shared is not None:
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales: scalar_params.append(s)
    # v42: VRL alphas
    if base_model.vrl_enabled:
        for a in base_model.vrl_alphas: scalar_params.append(a)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_param_groups = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_param_groups.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None: matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_param_groups.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None: matrix_params.append(base_model.ve_shared.proj.weight)
    optimizer_tok = torch.optim.AdamW(tok_param_groups, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, ns_steps=args.muon_ns_steps, wd=args.muon_wd)
    for group in optimizer_muon.param_groups: group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                          betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
                                           betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, optimizer_head)
    n_params = sum(p.numel() for p in base_model.parameters())
    xsa_layers = [i for i in range(args.num_layers) if i >= args.num_layers - args.xsa_last_n] if args.xsa_last_n > 0 else []
    log0(f"model_params:{n_params}"); log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"v42: 11L LeakyReLU(0.5)² Late-QAT@{args.late_qat_threshold} int6-all FullGPTQ EMA({args.ema_decay}) TightSWA XSA-all({args.xsa_last_n}) PartialRoPE({args.rope_dims}/64) LNScale VE128 SmearGate BigramHash({args.bigram_vocab_size}) QATalign({args.qat_clip_pct}) VRL Prune({args.prune_pct}) RawBinary")
    log0(f"XSA:last_{args.xsa_last_n} layers:{xsa_layers}")
    log0(f"FA3:{HAS_FA3} SWA:{args.swa_enabled} warmdown:{args.warmdown_iters} adam_wd:{args.adam_wd}")
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if step >= ws else 1.0
        step_ms = elapsed_ms / max(step, 1); wd_ms = args.warmdown_iters * step_ms
        rem_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    # WARMUP
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (ws+1) % 10 == 0: log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True): opt.load_state_dict(state)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    # MAIN TRAINING LOOP
    training_time_ms, stop_after_step = 0.0, None
    swa_state, swa_count = None, 0
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                              val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:soft_round enabled step:{step} scale:{scale:.4f}")
        # Soft-Round alpha annealing: 1 (soft) → 16 (hard) as scale decreases from threshold to 0
        if CastedLinear._qat_enabled:
            progress = 1.0 - max(scale / max(args.late_qat_threshold, 1e-6), 0.0)  # 0→1 as training progresses
            CastedLinear._qat_alpha = 1.0 + 15.0 * progress  # 1→16
        # Mild byte-weighting: ramp alpha from 0 to 0.3 during warmdown (last 20% of LR schedule)
        if hasattr(base_model, '_byte_weight_alpha') and scale < 0.2:
            base_model._byte_weight_alpha = min(0.3, 0.3 * (0.2 - scale) / 0.2)
        zero_grad_all(); train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): loss = model(x, y)
            train_loss += loss.detach(); (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for group in optimizer_muon.param_groups:
            group["momentum"] = (1-frac)*args.muon_momentum_warmup_start + frac*args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all()
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)
        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_interval == 0:
            if swa_state is None:
                swa_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
                swa_count = 1; log0(f"swa:start step:{step}")
            else:
                for n, t in base_model.state_dict().items(): swa_state[n] += t.detach().cpu()
                swa_count += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rct = torch.tensor(int(reached_cap), device=device); dist.all_reduce(rct, op=dist.ReduceOp.MAX); reached_cap = bool(rct.item())
        if stop_after_step is None and reached_cap: stop_after_step = step
    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")

    # Apply EMA weights
    log0("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)

    # v41: GPTQ calibration — collect Hessians AFTER applying EMA weights
    log0(f"gptq:calibrating with {args.gptq_calib_batches} batches...")
    calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    hessians = collect_hessians(base_model, calib_loader, args, device, grad_accum_steps,
                                num_batches=args.gptq_calib_batches)
    # Map module names to state_dict names for Hessian lookup
    hessian_map = {}
    for name, module in base_model.named_modules():
        if isinstance(module, CastedLinear):
            sd_name = name + ".weight"
            h_name = name + ".weight"
            if h_name in hessians:
                hessian_map[sd_name] = hessians[h_name]
    log0(f"gptq:collected hessians for {len(hessian_map)} layers")

    # QUANTIZE + SAVE (raw binary serialization)
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    code_bytes = len(code.encode("utf-8")); size_limit = 16_000_000
    quant_result, quant_meta = quantize_state_dict_mixed(sd_cpu, hessians=hessian_map)
    # Post-quant magnitude pruning: zero out smallest int6 weights for better compression
    if args.prune_pct > 0:
        all_int6_vals = []
        for name, info in quant_meta.items():
            if isinstance(info, dict) and info.get("type") == "int6":
                qname = name + ".q"
                if qname in quant_result:
                    all_int6_vals.append(quant_result[qname].flatten().abs().float())
        if all_int6_vals:
            all_vals = torch.cat(all_int6_vals)
            k = max(1, int(args.prune_pct * all_vals.numel()))
            threshold = all_vals.kthvalue(k).values.item()
            pruned_count = 0
            for name, info in quant_meta.items():
                if isinstance(info, dict) and info.get("type") == "int6":
                    qname = name + ".q"
                    if qname in quant_result:
                        mask = quant_result[qname].abs() <= int(threshold)
                        pruned_count += mask.sum().item()
                        quant_result[qname][mask] = 0
            total_int6 = sum(quant_result[n + ".q"].numel() for n, i in quant_meta.items() if isinstance(i, dict) and i.get("type") == "int6" and n + ".q" in quant_result)
            log0(f"prune:zeroed {pruned_count}/{total_int6} int6 weights ({100*pruned_count/max(total_int6,1):.1f}%) threshold={threshold:.0f}")
    meta_json = json.dumps(quant_meta).encode("utf-8")
    parts = [struct.pack("<I", len(meta_json)), meta_json]
    tensor_order = sorted(quant_result.keys())
    for tname in tensor_order:
        t = quant_result[tname]
        name_bytes = tname.encode("utf-8")
        dtype_map = {torch.int8: 0, torch.float16: 1, torch.float32: 2, torch.bfloat16: 3}
        dt = dtype_map.get(t.dtype, 2)
        t_np = t.contiguous().numpy() if t.dtype != torch.bfloat16 else t.contiguous().view(torch.uint16).numpy()
        raw = t_np.tobytes()
        parts.append(struct.pack("<H", len(name_bytes)))
        parts.append(name_bytes)
        parts.append(struct.pack("<BB", dt, t.ndim))
        for d in t.shape: parts.append(struct.pack("<I", d))
        parts.append(raw)
    def _serialize_and_compress(qr, qm):
        mj = json.dumps(qm).encode("utf-8")
        pp = [struct.pack("<I", len(mj)), mj]
        for tn in sorted(qr.keys()):
            t = qr[tn]; nb = tn.encode("utf-8")
            dm = {torch.int8: 0, torch.float16: 1, torch.float32: 2, torch.bfloat16: 3}
            dt = dm.get(t.dtype, 2)
            tnp = t.contiguous().numpy() if t.dtype != torch.bfloat16 else t.contiguous().view(torch.uint16).numpy()
            pp.append(struct.pack("<H", len(nb))); pp.append(nb)
            pp.append(struct.pack("<BB", dt, t.ndim))
            for d in t.shape: pp.append(struct.pack("<I", d))
            pp.append(tnp.tobytes())
        raw = b"".join(pp)
        if HAS_ZSTD: return zstd.ZstdCompressor(level=22).compress(raw), "zstd22"
        return zlib.compress(raw, level=9), "zlib9"
    model_blob, comp_name = _serialize_and_compress(quant_result, quant_meta)
    model_bytes = len(model_blob); total_size = code_bytes + model_bytes
    # Adaptive prune: DISABLED — zeroing all ±1 int6 values destroys quality (0.03 BPP regression)
    extra_prune_rounds = 0
    while False and total_size > size_limit and extra_prune_rounds < 5:
        extra_prune_rounds += 1
        all_nonzero = []
        for name, info in quant_meta.items():
            if isinstance(info, dict) and info.get("type") == "int6":
                qname = name + ".q"
                if qname in quant_result:
                    q = quant_result[qname]
                    nz = q[q != 0].abs().float()
                    if nz.numel() > 0: all_nonzero.append(nz)
        if not all_nonzero: break
        all_nz = torch.cat(all_nonzero)
        # Zero the smallest 1% of remaining non-zero weights
        k = max(1, int(0.01 * all_nz.numel()))
        thresh = all_nz.kthvalue(k).values.item()
        extra_zeroed = 0
        for name, info in quant_meta.items():
            if isinstance(info, dict) and info.get("type") == "int6":
                qname = name + ".q"
                if qname in quant_result:
                    mask = (quant_result[qname] != 0) & (quant_result[qname].abs() <= int(thresh))
                    extra_zeroed += mask.sum().item()
                    quant_result[qname][mask] = 0
        log0(f"adaptive_prune:round {extra_prune_rounds} zeroed {extra_zeroed} more weights (threshold={thresh:.0f})")
        model_blob, comp_name = _serialize_and_compress(quant_result, quant_meta)
        model_bytes = len(model_blob); total_size = code_bytes + model_bytes
    log0(f"model:{model_bytes} code:{code_bytes} total:{total_size} ({total_size/1e6:.2f} MB)")
    if total_size > size_limit: log0(f"WARNING: Total size {total_size} exceeds 16MB limit by {total_size - size_limit} bytes!")
    else: log0(f"Size OK: {total_size/1e6:.2f} MB")
    if master_process:
        with open("final_model.int6.ptz", "wb") as f: f.write(model_blob)
    if distributed: dist.barrier()
    # ROUNDTRIP DEQUANTIZE
    with open("final_model.int6.ptz", "rb") as f: model_blob_loaded = f.read()
    if HAS_ZSTD: raw_data = zstd.ZstdDecompressor().decompress(model_blob_loaded)
    else: raw_data = zlib.decompress(model_blob_loaded)
    offset = 0
    meta_len = struct.unpack_from("<I", raw_data, offset)[0]; offset += 4
    loaded_meta = json.loads(raw_data[offset:offset+meta_len].decode("utf-8")); offset += meta_len
    dtype_rmap = {0: (torch.int8, np.int8), 1: (torch.float16, np.float16), 2: (torch.float32, np.float32), 3: (torch.bfloat16, np.uint16)}
    loaded_result = {}
    while offset < len(raw_data):
        name_len = struct.unpack_from("<H", raw_data, offset)[0]; offset += 2
        tname = raw_data[offset:offset+name_len].decode("utf-8"); offset += name_len
        dt, ndim = struct.unpack_from("<BB", raw_data, offset); offset += 2
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack_from("<I", raw_data, offset)[0]); offset += 4
        torch_dt, np_dt = dtype_rmap[dt]
        numel = 1
        for s in shape: numel *= s
        nbytes = numel * np.dtype(np_dt).itemsize
        arr = np.frombuffer(raw_data, dtype=np_dt, count=numel, offset=offset).copy()
        offset += nbytes
        t = torch.from_numpy(arr).reshape(shape)
        if torch_dt == torch.bfloat16: t = t.view(torch.bfloat16)
        loaded_result[tname] = t
    deq_state = dequantize_state_dict_mixed(loaded_result, loaded_meta, sd_cpu)
    base_model.load_state_dict(deq_state, strict=True)
    eval_sl = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_tokens_eval = load_validation_tokens(args.val_files, eval_sl) if eval_sl != args.train_seq_len else val_tokens
    raw_logits_fn = torch.compile(base_model.forward_logits, dynamic=False) if not bool(int(os.environ.get("TORCH_COMPILE_DISABLE", "0"))) else base_model.forward_logits
    warmup_x = torch.zeros(args.eval_batch_seqs, eval_sl, dtype=torch.int64, device=device)
    base_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16): _ = raw_logits_fn(warmup_x)
    torch.cuda.synchronize(); t_eval = time.perf_counter()
    q_vl, q_vb = eval_val_sliding(raw_logits_fn, rank, world_size, device,
        val_tokens_eval, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_sl, args.eval_stride, eval_batch_seqs=args.eval_batch_seqs)
    torch.cuda.synchronize(); eval_time = time.perf_counter() - t_eval
    log0(f"final_int6_sliding_window val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f} eval_time:{eval_time*1000:.0f}ms")
    log0(f"final_int6_sliding_window_exact val_loss:{q_vl:.8f} val_bpb:{q_vb:.8f}")
    # Compat aliases for train.py regex parsing
    log0(f"final_int6_sliding_window_s64 val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f}")
    log0(f"final_int6_sliding_window_s64_exact val_loss:{q_vl:.8f} val_bpb:{q_vb:.8f}")
    # N-gram eval cache (score-first, legal)
    if args.ngram_eval_order >= 2:
        if distributed: dist.barrier()
        torch.cuda.synchronize()
        t_ng = time.perf_counter()
        log0(f"ngram_eval:order={args.ngram_eval_order} alpha={args.ngram_eval_alpha} "
             f"min_count={args.ngram_eval_min_count} buckets={args.ngram_eval_buckets} "
             f"backoff={args.ngram_backoff} entropy_adaptive={args.ngram_entropy_adaptive}")
        ng_loss, ng_bpb, ng_coverage = eval_val_sliding_ngram(
            args, raw_logits_fn, rank, world_size, device,
            val_tokens_eval, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, eval_seq_len=eval_sl, batch_seqs=args.eval_batch_seqs,
        )
        torch.cuda.synchronize()
        ng_ms = 1000.0 * (time.perf_counter() - t_ng)
        if ng_coverage >= 0.999:
            log0(f"final_int6_sliding_window_ngram{args.ngram_eval_order} val_loss:{ng_loss:.4f} "
                 f"val_bpb:{ng_bpb:.4f} eval_time:{ng_ms:.0f}ms")
            log0(f"final_int6_sliding_window_ngram{args.ngram_eval_order}_exact "
                 f"val_loss:{ng_loss:.8f} val_bpb:{ng_bpb:.8f}")
        else:
            log0(f"final_int6_sliding_window_ngram{args.ngram_eval_order}_partial val_loss:{ng_loss:.4f} "
                 f"val_bpb:{ng_bpb:.4f} coverage:{ng_coverage:.4f} eval_time:{ng_ms:.0f}ms")
        if distributed: dist.barrier()
    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
