"""train_gpt_submit.py — Recovery: legal-only approach.
12L gated-attention + value-residual + GPTQ int5 + EMA + legal score-first SGD TTT.
Based on v42 architecture (our best legal result at 1.1272 BPB) with:
  - Warmdown increased to 3000 (from 1200)
  - Bigram vocab increased to 4096 (from 1024)
  - GPTQ int5 with retry loop
  - EMA weight averaging
  - Legal score-first SGD TTT (PR#549 pattern: score chunk → train chunk → never re-score)
"""
from __future__ import annotations
import copy, glob, io, math, os, random, subprocess, sys, time, uuid, zlib
from pathlib import Path
import lzma as _lzma
try:
    import zstandard
except ImportError:
    zstandard = None
_COMPRESSOR = "lzma"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    from flash_attn import flash_attn_func as flash_attn_3_func

# ---- HYPERPARAMETERS ----
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
    iterations = int(os.environ.get("ITERATIONS", 10800))
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
    mlp_mult = float(os.environ.get("MLP_MULT", 3.5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 131072.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 20.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.05))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    muon_wd = float(os.environ.get("MUON_WD", 0.02))
    adam_wd = float(os.environ.get("ADAM_WD", 0.01))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 1024))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 256))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 0))
    rope_dims = int(os.environ.get("ROPE_DIMS", 64))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "8,9,10")
    mlp_activation = os.environ.get("MLP_ACTIVATION", "leaky_relu_sq")
    leaky_slope = float(os.environ.get("LEAKY_SLOPE", 0.5))
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "0")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    gptq_enabled = bool(int(os.environ.get("GPTQ_ENABLED", "1")))
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 512))
    quant_bits = int(os.environ.get("QUANT_BITS", 6))
    prune_pct = float(os.environ.get("PRUNE_PCT", 0.05))
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "1")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "1")))
    # Legal score-first SGD TTT (PR#549 pattern)
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_sgd_epochs = int(os.environ.get("TTT_SGD_EPOCHS", 3))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    ttt_max_secs = float(os.environ.get("TTT_MAX_SECS", 550.0))
    ttt_unfreeze_layers = int(os.environ.get("TTT_UNFREEZE_LAYERS", 4))
    # N-gram cache mixing (entropy-adaptive, legal)
    ngram_enabled = bool(int(os.environ.get("NGRAM_ENABLED", "1")))
    ngram_n = int(os.environ.get("NGRAM_N", 13))
    ngram_alpha = float(os.environ.get("NGRAM_ALPHA", 0.6))
    # AdaMuon optimizer (arxiv 2507.11005)
    adamuon_enabled = bool(int(os.environ.get("ADAMUON_ENABLED", "0")))
    adamuon_beta2 = float(os.environ.get("ADAMUON_BETA2", 0.95))
    adamuon_eps = float(os.environ.get("ADAMUON_EPS", 1e-8))
    # GPTQ tuning knobs
    gptq_percdamp = float(os.environ.get("GPTQ_PERCDAMP", 0.01))
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))

# ---- MUON OPTIMIZER ----
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay))
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
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
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
                    if nesterov: g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0: p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# ---- ADAMUON OPTIMIZER (arxiv 2507.11005) ----
class AdaMuon(torch.optim.Optimizer):
    """Muon + per-parameter second-moment adaptation. Tracks running variance of gradients
    and rescales the orthogonalized update by 1/sqrt(v_hat), giving adaptive per-element
    learning rates while preserving Muon's spectral descent direction."""
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0,
                 beta2: float = 0.95, eps: float = 1e-8):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay,
                                      beta2=beta2, eps=eps))
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
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            beta2, eps = group["beta2"], group["eps"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["v"] = torch.zeros_like(g, dtype=torch.float32)
                        state["step"] = 0
                    state["step"] += 1
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov: g_nest = g.add(buf, alpha=momentum)
                    else: g_nest = buf
                    v = state["v"]
                    v.mul_(beta2).addcmul_(g.float(), g.float(), value=1.0 - beta2)
                    bc = 1.0 - beta2 ** state["step"]
                    v_hat = v / bc
                    g_ortho = zeropower_via_newtonschulz5(g_nest, steps=backend_steps)
                    g_ortho *= max(1, g_ortho.size(0) / g_ortho.size(1)) ** 0.5
                    rms_nest = g_nest.float().pow(2).mean().sqrt().item() + eps
                    g_adapted = g_ortho.float() * rms_nest / (v_hat.sqrt() + eps)
                    updates_flat[curr : curr + p.numel()] = g_adapted.reshape(-1).bfloat16()
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0: p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# ---- TOKENIZER-AGNOSTIC EVALUATION ----
def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id): base_bytes_np[token_id] = 1; continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"): has_leading_space_np[token_id] = True; piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0: raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(f"VAL_BATCH_SIZE too small: {args.val_batch_size}")
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            local = val_tokens[bss * seq_len : bse * seq_len + 1].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            bc = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * bc; val_token_count += bc
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (val_loss_sum, val_token_count, val_byte_count): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)

# ---- QUANTIZATION CONSTANTS ----
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get("CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,ve_shared.scale,attn_gate,v_lambda"
    ).split(",") if p)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get("INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
    ",".join(CONTROL_TENSOR_NAME_PATTERNS)).split(",") if p)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0

def tensor_nbytes(t: Tensor) -> int: return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, po: dict[str, str]) -> Tensor:
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS): return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        po[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        ca = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, ca[:, None]), -ca[:, None])
        scale = (ca / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    ca = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(ca / 127.0 if ca > 0 else 1.0, dtype=torch.float32)
    return (torch.clamp(torch.round(torch.clamp(t32, -ca, ca) / scale), -127, 127).to(torch.int8).contiguous(), scale)

# ---- DATA LOADING ----
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    if file.stat().st_size != header_bytes + num_tokens * np.dtype("<u2").itemsize:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens: raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files for: {pattern}")
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail); chunks.append(self.tokens[self.pos:self.pos+k]); self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        prs = local_tokens + 1; chunk = self.stream.take(prs * self.world_size)
        start = self.rank * prs; local = chunk[start:start+prs].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ---- TRANSFORMER MODULES ----
class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    _qat_bits: int = 5
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            bits = CastedLinear._qat_bits
            max_val = (2 ** (bits - 1)) - 1
            with torch.no_grad():
                w32 = self.weight.float()
                row_clip = torch.quantile(w32.abs(), 0.9995, dim=1)
                scale = (row_clip / float(max_val)).clamp_min(1.0 / float(max_val))
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -max_val, max_val) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim, self.base, self.train_seq_len = dim, base, train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        self.register_buffer("inv_freq", 1.0/(base**(torch.arange(0,self.rope_dims,2,dtype=torch.float32)/self.rope_dims)), persistent=False)
        self._seq_len_cached = 0; self._cos_cached = None; self._sin_cached = None
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                sc = seq_len / self.train_seq_len; nb = self.base * (sc ** (rd / (rd - 2)))
                inv_freq = 1.0 / (nb ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else: inv_freq = self.inv_freq.to(device)
            freqs = torch.outer(torch.arange(seq_len, device=device, dtype=inv_freq.dtype), inv_freq)
            self._cos_cached = freqs.cos()[None,:,None,:]; self._sin_cached = freqs.sin()[None,:,None,:]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        xr, xp = x[...,:rope_dims], x[...,rope_dims:]
        h = rope_dims // 2; x1, x2 = xr[...,:h], xr[...,h:]
        return torch.cat((torch.cat((x1*cos+x2*sin, x1*(-sin)+x2*cos), dim=-1), xp), dim=-1)
    h = x.size(-1) // 2; x1, x2 = x[...,:h], x[...,h:]
    return torch.cat((x1*cos+x2*sin, x1*(-sin)+x2*cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 gated_attention=False, value_residual=False, layer_idx=0):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False); self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0; self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False
        self._layer_idx = layer_idx
        self.use_gated_attention = gated_attention
        if gated_attention:
            self.attn_gate = CastedLinear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)
        self.use_value_residual = value_residual
        if value_residual and layer_idx > 0:
            self.v_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D); vn = F.normalize(v, dim=-1).unsqueeze(-2)
        return (y_g - (y_g * vn).sum(dim=-1, keepdim=True) * vn).reshape(B, T, H, D)
    def forward(self, x, v_embed=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None: v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        if self.use_value_residual and hasattr(self, '_gpt_ref') and self._gpt_ref is not None:
            if self._layer_idx == 0:
                self._gpt_ref._v0_cache = v.detach()
            elif self._gpt_ref._v0_cache is not None and hasattr(self, 'v_lambda'):
                lam = self.v_lambda.to(dtype=v.dtype)
                v = lam[0] * self._gpt_ref._v0_cache.to(dtype=v.dtype) + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_gated_attention:
            gate = torch.sigmoid(self.attn_gate(x))
            y = y * gate.unsqueeze(-1)
        if self.use_xsa: y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))

class SmearGate(nn.Module):
    def __init__(self, dim): super().__init__(); self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        return (1-g)*x + g*torch.cat([torch.zeros_like(x[:,:1]), x[:,:-1]], dim=1)

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__(); self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim); nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32); mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t); out[...,0] = mod
        out[...,1:] = torch.bitwise_xor(36313*t[...,1:], 27191*t[...,:-1]) % mod
        return out.long()
    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, ve_dim, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim); nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids):
        h = self.embed(token_ids)
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, activation="leaky_relu_sq", leaky_slope=0.5):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False); self.proj._zero_init = True
        self.activation, self.leaky_slope = activation, leaky_slope
    def forward(self, x):
        x = self.fc(x)
        if self.activation == "leaky_relu_sq":
            x = F.leaky_relu(x, negative_slope=self.leaky_slope)
        else:
            x = torch.relu(x)
        return self.proj(x.square())

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 layer_idx=0, ln_scale=False, dtg=False, activation="leaky_relu_sq",
                 leaky_slope=0.5, gated_attention=False, value_residual=False):
        super().__init__()
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                         gated_attention=gated_attention, value_residual=value_residual,
                                         layer_idx=layer_idx)
        self.mlp = MLP(dim, mlp_mult, activation=activation, leaky_slope=leaky_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight); nn.init.constant_(self.dtg_gate.bias, 2.0)
        else: self.dtg_gate = None
    def forward(self, x, x0, v_embed=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None,None,:]*x + mix[1][None,None,:]*x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None,None,:] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None,None,:] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init,
                 mtp_num_heads=0, mtp_loss_weight=0.1, bigram_vocab_size=0, bigram_dim=128,
                 xsa_last_n=0, rope_dims=0, ln_scale=False, dtg=False, ve_enabled=False,
                 ve_dim=128, ve_layers="9,10", activation="leaky_relu_sq", leaky_slope=0.5,
                 gated_attention=False, value_residual=False):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)
        assert logit_softcap > 0
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.mtp_num_heads, self.mtp_loss_weight = mtp_num_heads, mtp_loss_weight
        self.value_residual = value_residual
        self._v0_cache = None
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, ln_scale=ln_scale, dtg=dtg, activation=activation,
                  leaky_slope=leaky_slope, gated_attention=gated_attention,
                  value_residual=value_residual)
            for i in range(num_layers)])
        if rope_dims > 0:
            hd = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(hd, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, self._ve_target_dim)
            self.ve_layer_scales = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else: self.ve_shared = None; self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList([CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)])
        for head in self.mtp_heads: head._zero_init = True
        if xsa_last_n == 0:
            for block in self.blocks: block.attn.use_xsa = True
        elif xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers): self.blocks[i].attn.use_xsa = True
        if value_residual:
            for block in self.blocks:
                object.__setattr__(block.attn, '_gpt_ref', self)
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings: nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        nl = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False): nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad(): module.weight.mul_(1.0 / math.sqrt(2 * nl))

    def _get_ve(self, layer_idx, input_ids, ve_cache=None):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices: return None
        if ve_cache is not None and 've' not in ve_cache: ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        return ve_base * self.ve_layer_scales[self.ve_layer_indices.index(layer_idx)].to(dtype=ve_base.dtype)

    def _run_backbone(self, input_ids):
        self._v0_cache = None
        x = self.tok_emb(input_ids)
        if self.bigram is not None: x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),)); x = self.smear(x); x0 = x
        skips, ve_cache = [], {}
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, v_embed=self._get_ve(i, input_ids, ve_cache)); skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None,None,:] * skips.pop()
            x = self.blocks[bi](x, x0, v_embed=self._get_ve(bi, input_ids, ve_cache))
        return self.final_norm(x)

    def forward(self, input_ids, target_ids):
        x = self._run_backbone(input_ids)
        x_flat = x.reshape(-1, x.size(-1)); targets = target_ids.reshape(-1)
        if self.tie_embeddings: logits = F.linear(x_flat, self.tok_emb.weight)
        else: logits = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape; mtp_loss_sum = x.new_zeros(()); mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                vt = seqlen - (k + 1)
                if vt <= 0: continue
                ml = mtp_head(x[:, :vt, :].reshape(-1, dim))
                ml = self.logit_softcap * torch.tanh(ml / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(ml.float(), target_ids[:, k+1:].reshape(-1), reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0: main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss

    def forward_logits(self, input_ids):
        x = self._run_backbone(input_ids)
        if self.tie_embeddings: logits = F.linear(x, self.tok_emb.weight)
        else: logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

# ---- SLIDING WINDOW EVALUATION ----
def eval_val_sliding(args, base_model, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride, batch_seqs=32, eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ws_list = [ws for ws in range(0, total_tokens, stride) if min(ws+seq_len, total_tokens)-ws >= 1]
    tw = len(ws_list); my_s = (tw*rank)//world_size; my_e = (tw*(rank+1))//world_size
    my_ws = ws_list[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_ws), batch_seqs):
            bws = my_ws[bi:bi+batch_seqs]; bsz = len(bws)
            xb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(bws):
                end = min(ws+seq_len, total_tokens); wlen = end-ws; wlens.append(wlen)
                chunk = val_tokens[ws:end+1].to(dtype=torch.int64, device=device)
                xb[i,:wlen] = chunk[:-1]; yb[i,:wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(xb)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none").reshape(bsz, seq_len)
            for i, ws in enumerate(bws):
                wlen = wlens[i]; s = 0 if ws == 0 else max(wlen-stride, 0)
                loss_sum += nll[i, s:wlen].to(torch.float64).sum(); token_count += float(wlen-s)
                tgt, prev = yb[i, s:wlen], xb[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (loss_sum / token_count).item()
    base_model.train()
    return vl, vl / math.log(2.0) * (token_count.item() / byte_count.item())

# ---- N-GRAM EVAL CACHE (PR#809 pattern, pure NumPy, vectorized) ----
_NGRAM_PRIMES = np.array([36313, 27191, 51647, 81929, 131071, 174763, 233017, 283721, 347237, 409993, 479909, 557927, 631853, 720721], dtype=np.uint64)

def _batch_hash_ctx(tokens_np, positions, n, bucket_mask):
    h = np.zeros(len(positions), dtype=np.uint64)
    for k in range(n - 1):
        idx = np.clip(positions - (n - 1) + k, 0, len(tokens_np) - 1)
        h ^= tokens_np[idx].astype(np.uint64) * _NGRAM_PRIMES[k]
    return h & np.uint64(bucket_mask)

def _batch_hash_full(tokens_np, positions, targets, n, bucket_mask):
    h = np.zeros(len(positions), dtype=np.uint64)
    for k in range(n - 1):
        idx = np.clip(positions - (n - 1) + k, 0, len(tokens_np) - 1)
        h ^= tokens_np[idx].astype(np.uint64) * _NGRAM_PRIMES[k]
    h ^= targets.astype(np.uint64) * _NGRAM_PRIMES[min(n - 1, len(_NGRAM_PRIMES) - 1)]
    return h & np.uint64(bucket_mask)

class NgramEvalCache:
    def __init__(self, max_order=9, min_order=2, num_buckets=4194304, min_count=2):
        self.max_order, self.min_order = max_order, min_order
        self.num_buckets = num_buckets
        self.bucket_mask = num_buckets - 1
        self.min_count = min_count
        self.ctx_tables = [np.zeros(num_buckets, dtype=np.int32) for _ in range(max_order + 1)]
        self.full_tables = [np.zeros(num_buckets, dtype=np.int32) for _ in range(max_order + 1)]

    def batch_lookup(self, tokens_np, positions, targets):
        n_pos = len(positions)
        ngram_p = np.zeros(n_pos, dtype=np.float64)
        matched = np.zeros(n_pos, dtype=bool)
        matched_orders = np.zeros(n_pos, dtype=np.int32)
        for n in range(self.max_order, self.min_order - 1, -1):
            eligible = (~matched) & (positions >= n - 1)
            if not eligible.any(): continue
            elig_pos = positions[eligible]
            elig_tgt = targets[eligible]
            ctx_keys = _batch_hash_ctx(tokens_np, elig_pos, n, self.bucket_mask).astype(np.int64)
            ctx_counts = self.ctx_tables[n][ctx_keys]
            has_data = ctx_counts >= self.min_count
            if not has_data.any(): continue
            full_keys = _batch_hash_full(tokens_np, elig_pos[has_data], elig_tgt[has_data], n, self.bucket_mask).astype(np.int64)
            full_counts = self.full_tables[n][full_keys]
            capped = np.minimum(full_counts, ctx_counts[has_data])
            probs = capped.astype(np.float64) / np.maximum(ctx_counts[has_data].astype(np.float64), 1.0)
            elig_indices = np.where(eligible)[0]
            data_indices = elig_indices[has_data]
            ngram_p[data_indices] = probs
            matched[data_indices] = True
            matched_orders[data_indices] = n
        return ngram_p, matched, matched_orders

    def update_batch(self, tokens_np, start_pos, end_pos):
        if end_pos <= start_pos: return
        positions = np.arange(start_pos, end_pos, dtype=np.int64)
        targets = tokens_np[positions].astype(np.int64)
        for n in range(self.min_order, self.max_order + 1):
            valid = positions >= n - 1
            if not valid.any(): continue
            v_pos, v_tgt = positions[valid], targets[valid]
            ctx_keys = _batch_hash_ctx(tokens_np, v_pos, n, self.bucket_mask).astype(np.int64)
            full_keys = _batch_hash_full(tokens_np, v_pos, v_tgt, n, self.bucket_mask).astype(np.int64)
            self.ctx_tables[n] += np.bincount(ctx_keys, minlength=self.num_buckets).astype(np.int32)
            self.full_tables[n] += np.bincount(full_keys, minlength=self.num_buckets).astype(np.int32)

def _build_sliding_segments(total_tokens, seq_len, stride):
    segments = []
    first_valid_len = min(seq_len, total_tokens)
    segments.append((0, first_valid_len, 0, first_valid_len, 1, first_valid_len + 1))
    next_target_start = first_valid_len + 1
    while next_target_start <= total_tokens:
        target_end = min(next_target_start + stride, total_tokens + 1)
        window_end = target_end - 1
        window_start = max(0, window_end - seq_len)
        valid_len = window_end - window_start
        local_score_start = next_target_start - window_start - 1
        local_score_end = target_end - window_start - 1
        segments.append((window_start, valid_len, local_score_start, local_score_end, next_target_start, target_end))
        next_target_start = target_end
    return segments

def eval_val_ngram(args, base_model, rank, world_size, device, val_tokens,
                   base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                   eval_seq_len=None):
    """PR#809-style n-gram eval: chunk-based, vectorized, entropy-adaptive, multi-GPU.
    With PR#888 full-rescore: Pass 1 builds cache + captures model probs,
    Pass 2 rescores all tokens using the COMPLETE n-gram cache (no extra forward passes)."""
    seq_len = eval_seq_len or args.train_seq_len
    stride = args.eval_stride
    total_tokens = val_tokens.numel() - 1
    tokens_np = val_tokens.numpy().astype(np.int64)
    chunk_tokens = 1000000  # 1M tokens per chunk
    # Align chunk_tokens to stride
    if (chunk_tokens - seq_len) % stride != 0:
        chunk_tokens = seq_len + ((chunk_tokens - seq_len) // stride) * stride
    ngram_buckets = int(os.environ.get("NGRAM_BUCKETS", 4194304))
    ngram_min_count = int(os.environ.get("NGRAM_MIN_COUNT", 2))
    cache = NgramEvalCache(max_order=args.ngram_n, min_order=2, num_buckets=ngram_buckets, min_count=ngram_min_count)
    alpha_min, alpha_max = 0.05, args.ngram_alpha
    entropy_center = float(os.environ.get("NGRAM_EC", 3.0))
    entropy_scale = float(os.environ.get("NGRAM_ES", 2.0))
    order_mults = np.array([0.3, 0.3, 0.97, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    full_rescore = bool(int(os.environ.get("NGRAM_FULL_RESCORE", "1")))
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    # Storage for Pass 2 rescore (captured per-segment data from Pass 1)
    captured_segments = [] if full_rescore else None
    base_model.eval()
    t_start = time.perf_counter()
    segments = _build_sliding_segments(total_tokens, seq_len, stride)
    seg_idx = 0
    batch_seqs = 32
    with torch.inference_mode():
        # ---- Pass 1: forward pass + build n-gram cache + capture model probs ----
        for chunk_start in range(1, total_tokens + 1, chunk_tokens):
            chunk_end = min(chunk_start + chunk_tokens, total_tokens + 1)
            chunk_segments = []
            while seg_idx < len(segments) and segments[seg_idx][4] < chunk_end:
                chunk_segments.append(segments[seg_idx]); seg_idx += 1
            rank_segments = chunk_segments[rank::world_size]
            for bi in range(0, len(rank_segments), batch_seqs):
                batch_segments = rank_segments[bi:bi+batch_seqs]
                bsz = len(batch_segments)
                xb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                yb = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                for ri, (ws, vl, _, _, _, _) in enumerate(batch_segments):
                    end = min(ws + seq_len, total_tokens)
                    chunk = val_tokens[ws:end+1].to(device=device, dtype=torch.int64)
                    xb[ri, :vl] = chunk[:-1]; yb[ri, :vl] = chunk[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(xb)
                for ri, (_, _, lss, lse, ts, te) in enumerate(batch_segments):
                    seg_len = te - ts
                    row_logits = logits[ri, lss:lse].float()
                    row_targets = yb[ri, lss:lse]
                    model_probs_full = torch.softmax(row_logits, dim=-1)
                    seg_model_p = torch.gather(model_probs_full, 1, row_targets.unsqueeze(-1)).squeeze(-1)
                    seg_model_p = seg_model_p.clamp(min=1e-10).cpu().numpy().astype(np.float64)
                    log_probs_full = torch.log_softmax(row_logits, dim=-1)
                    seg_entropy = -(model_probs_full * log_probs_full).sum(dim=-1).cpu().numpy()
                    global_positions = np.arange(ts, te, dtype=np.int64)
                    seg_targets_np = row_targets.cpu().numpy().astype(np.int64)
                    ngram_p, ng_matched, ng_orders = cache.batch_lookup(tokens_np, global_positions, seg_targets_np)
                    final_p = seg_model_p.copy()
                    if ng_matched.any():
                        matched_ords = ng_orders[ng_matched].astype(np.float64)
                        centers = entropy_center - 0.25 * (matched_ords - cache.min_order)
                        sig = 1.0 / (1.0 + np.exp(-entropy_scale * (seg_entropy[ng_matched] - centers)))
                        alpha = alpha_min + (alpha_max - alpha_min) * sig
                        mult_indices = np.clip(ng_orders[ng_matched] - cache.min_order, 0, len(order_mults) - 1)
                        alpha = np.clip(alpha * order_mults[mult_indices], 0.0, 0.95)
                        final_p[ng_matched] = (1.0 - alpha) * seg_model_p[ng_matched] + alpha * ngram_p[ng_matched]
                        final_p = np.maximum(final_p, 1e-10)
                    loss_sum += float((-np.log(final_p)).sum())
                    scored_x = xb[ri, lss:lse].reshape(-1)
                    scored_y = yb[ri, lss:lse].reshape(-1)
                    tb = base_bytes_lut[scored_y].to(torch.float64)
                    tb += (has_leading_space_lut[scored_y] & ~is_boundary_token_lut[scored_x]).to(torch.float64)
                    byte_sum += tb.sum(); token_count += seg_len
                    # Capture data for Pass 2 rescore
                    if full_rescore:
                        captured_segments.append((
                            global_positions,   # np.int64 array
                            seg_targets_np,     # np.int64 array
                            seg_model_p,        # np.float64 array (model probs for correct token)
                            seg_entropy.astype(np.float64),  # np.float64 array
                            float(tb.sum().item()),          # byte count for this segment
                            seg_len,                         # token count
                        ))
            cache.update_batch(tokens_np, chunk_start, chunk_end)
    # ---- Pass 2: rescore all tokens with FULL n-gram cache (no forward passes) ----
    if full_rescore and captured_segments:
        loss_sum_p2 = torch.zeros((), device=device, dtype=torch.float64)
        byte_sum_p2 = torch.zeros((), device=device, dtype=torch.float64)
        token_count_p2 = torch.zeros((), device=device, dtype=torch.float64)
        for (positions, targets, model_p, entropy, seg_bytes, seg_len) in captured_segments:
            ngram_p, ng_matched, ng_orders = cache.batch_lookup(tokens_np, positions, targets)
            final_p = model_p.copy()
            if ng_matched.any():
                matched_ords = ng_orders[ng_matched].astype(np.float64)
                centers = entropy_center - 0.25 * (matched_ords - cache.min_order)
                sig = 1.0 / (1.0 + np.exp(-entropy_scale * (entropy[ng_matched] - centers)))
                alpha = alpha_min + (alpha_max - alpha_min) * sig
                mult_indices = np.clip(ng_orders[ng_matched] - cache.min_order, 0, len(order_mults) - 1)
                alpha = np.clip(alpha * order_mults[mult_indices], 0.0, 0.95)
                final_p[ng_matched] = (1.0 - alpha) * model_p[ng_matched] + alpha * ngram_p[ng_matched]
                final_p = np.maximum(final_p, 1e-10)
            loss_sum_p2 += float((-np.log(final_p)).sum())
            byte_sum_p2 += seg_bytes
            token_count_p2 += seg_len
        # Replace Pass 1 results with Pass 2 results
        loss_sum = loss_sum_p2
        byte_sum = byte_sum_p2
        token_count = token_count_p2
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, byte_sum, token_count): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    elapsed = 1000.0 * (time.perf_counter() - t_start)
    return val_loss, val_bpb, elapsed

# ---- GPTQ CALIBRATION + QUANTIZATION ----
def gptq_calibrate(model, train_pattern, device, n_samples=256, seq_len=2048):
    hessians, n_seen, hooks = {}, {}, []
    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=torch.float32)
                n_seen[name] = 0
            hessians[name].addmm_(x.t(), x); n_seen[name] += x.shape[0]
        return hook_fn
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, CastedLinear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    stream = TokenStream(train_pattern)
    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            tokens = stream.take(seq_len + 1).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model.forward_logits(tokens[:-1].unsqueeze(0))
    for h in hooks: h.remove()
    for name in hessians: hessians[name] /= max(n_seen[name], 1)
    model.train()
    return hessians

def _find_best_row_scales(W, clip_range=15):
    t32 = W.float()
    best_s = (t32.abs().amax(dim=1) / clip_range).clamp_min(1.0 / clip_range)
    best_err = torch.full((t32.shape[0],), float('inf'))
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range)
        q = torch.clamp(torch.round(t32 / s[:, None]), -clip_range, clip_range)
        err = (t32 - q * s[:, None]).pow(2).mean(dim=1)
        improved = err < best_err
        best_s[improved] = s[improved]; best_err[improved] = err[improved]
    return best_s

def gptq_quantize_weight(W, H, clip_range=15, block_size=128, percdamp=0.01):
    W = W.float().clone()
    rows, cols = W.shape
    row_scale = _find_best_row_scales(W, clip_range)
    H = H.float().clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1.0; W[:, dead] = 0.0
    damp = percdamp * H.diag().mean(); H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True); invperm = torch.argsort(perm)
    W, H = W[:, perm], H[perm][:, perm]
    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
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
            q_col = torch.clamp(torch.round(w_col / row_scale), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col * row_scale) / Hinv_block[j, j].clamp_min(1e-8)
            Err[:, j] = err
            if j + 1 < i2 - i1:
                W_block[:, j + 1:] -= err.unsqueeze(1) * Hinv_block[j, j + 1:].unsqueeze(0)
        if i2 < cols: W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], row_scale.to(torch.float16)

def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    return "other"

def quantize_int_per_row(t, clip_range=15):
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err: best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    return torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8), scale

def mixed_quantize(state_dict, int_cats, hessians=None, clip_range=15, percdamp=0.01, block_size=128):
    result, meta = {}, {}
    gptq_count, naive_count = 0, 0
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"; continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float(); meta[name] = "passthrough_ctrl"; continue
        if cat in int_cats and t.ndim >= 1:
            module_name = name.rsplit(".weight", 1)[0] if name.endswith(".weight") else name
            H = hessians.get(module_name) if hessians else None
            if H is not None and t.ndim == 2:
                q, s = gptq_quantize_weight(t, H.cpu(), clip_range=clip_range, block_size=block_size, percdamp=percdamp); gptq_count += 1
            else:
                q, s = quantize_int_per_row(t, clip_range=clip_range); naive_count += 1
            result[name + ".q"], result[name + ".scale"] = q, s
            meta[name] = {"type": "intN"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"], result[name + ".scale"] = q, s
            meta[name] = {"type": "int8"}
    print(f"quantize: {gptq_count} GPTQ layers, {naive_count} naive layers, clip_range={clip_range}", flush=True)
    return result, meta

def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        od = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and od in (torch.float32, torch.bfloat16): t = t.to(od)
            out[name] = t; continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0: out[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(od)
        else: out[name] = (q.float() * float(s.item())).to(od)
    return out

# ---- LEGAL SCORE-FIRST TTT (PR#549 pattern) ----
BOS_ID = 1

def _find_docs(all_tokens):
    """Find document boundaries by BOS token positions."""
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else all_tokens.numel()
        if i + 1 < len(bos_positions): end += 1
        if end - start >= 2: docs.append((start, end - start))
    return docs

def _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len):
    """Compute the context window for a given chunk index."""
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len

def eval_val_ttt_sgd(args, eval_model, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """Legal score-first SGD TTT (PR#549 pattern):
    For each document:
      1. Restore base weights
      2. For each chunk (in order):
         a. Score chunk with current weights (BEFORE training) → accumulate BPB
         b. Train SGD epochs on this chunk's tokens
      3. Move to next document
    This is legal because tokens are scored BEFORE the model trains on them.
    """
    t_start = time.perf_counter()
    docs = _find_docs(val_tokens)
    if not docs: return None, None

    rank_docs = docs[(len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size]
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len
    sgd_epochs = args.ttt_sgd_epochs
    max_secs = args.ttt_max_secs

    # Identify parameters to unfreeze for TTT
    num_blocks = len(eval_model.blocks)
    unfreeze_start = max(0, num_blocks - args.ttt_unfreeze_layers)
    unfrozen_params = []
    for i in range(unfreeze_start, num_blocks):
        for p in eval_model.blocks[i].parameters():
            if p.ndim >= 2:
                unfrozen_params.append(p)

    # Save base values of unfrozen parameters only (memory efficient)
    base_values = [p.data.detach().clone() for p in unfrozen_params]

    # Freeze everything by default
    for p in eval_model.parameters():
        p.requires_grad_(False)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    n_docs_processed = 0
    fallback_start_idx = len(rank_docs)

    for doc_idx, (doc_start, doc_len) in enumerate(rank_docs):
        if time.perf_counter() - t_start > max_secs:
            fallback_start_idx = doc_idx
            if rank == 0:
                print(f"  ttt: time limit hit at doc {doc_idx}/{len(rank_docs)}", flush=True)
            break

        pred_len = doc_len - 1
        if pred_len < 1: continue
        num_chunks = (pred_len + chunk_size - 1) // chunk_size

        # Restore base weights at start of each document
        with torch.no_grad():
            for p, saved in zip(unfrozen_params, base_values):
                p.data.copy_(saved)

        # Unfreeze and set up SGD optimizer
        for p in unfrozen_params:
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(unfrozen_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

        for ci in range(num_chunks):
            ws, wl, co, cl = _compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len)
            toks = val_tokens[doc_start + ws: doc_start + ws + wl + 1].to(dtype=torch.int64, device=device)
            x = toks[:-1].unsqueeze(0)
            y = toks[1:].unsqueeze(0)

            # SCORE this chunk BEFORE training on it
            eval_model.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = eval_model.forward_logits(x)
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                       y.reshape(-1), reduction="none")

            chunk_nll = nll[co:co + cl].to(torch.float64)
            loss_sum += chunk_nll.sum()
            token_count += cl
            tgt = y[0, co:co + cl]; px = x[0, co:co + cl]
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[px]).to(torch.float64)
            byte_sum += tb.sum()

            # TRAIN on this chunk (skip last chunk — nothing left to benefit from adaptation)
            if ci < num_chunks - 1:
                eval_model.train()
                for _ in range(sgd_epochs):
                    optimizer.zero_grad()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits_t = eval_model.forward_logits(x)
                        chunk_logits = logits_t[0, co:co + cl]
                        train_loss = F.cross_entropy(chunk_logits.float(), y[0, co:co + cl], reduction="mean")
                    train_loss.backward()
                    optimizer.step()

        # Freeze parameters again
        for p in unfrozen_params:
            p.requires_grad_(False)

        n_docs_processed += 1
        if rank == 0 and n_docs_processed % 100 == 0:
            elapsed = time.perf_counter() - t_start
            bpb = float((loss_sum.item() / math.log(2.0)) / max(byte_sum.item(), 1))
            print(f"  ttt [{100*doc_idx/len(rank_docs):.1f}%] {doc_idx}/{len(rank_docs)} docs bpb={bpb:.6f} time={elapsed:.0f}s", flush=True)

    # Fallback: score remaining docs without TTT (base model only)
    if fallback_start_idx < len(rank_docs):
        with torch.no_grad():
            for p, saved in zip(unfrozen_params, base_values):
                p.data.copy_(saved)
        eval_model.eval()
        remaining = rank_docs[fallback_start_idx:]
        if rank == 0:
            print(f"  ttt: scoring {len(remaining)} fallback docs without TTT", flush=True)
        for ds, dl in remaining:
            pred_len = dl - 1
            if pred_len < 1: continue
            toks = val_tokens[ds:ds + dl].to(dtype=torch.int64, device=device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = eval_model.forward_logits(toks[:-1].unsqueeze(0))
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                       toks[1:].reshape(-1), reduction="none").to(torch.float64)
            tgt = toks[1:]; px = toks[:-1]
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[px]).to(torch.float64)
            loss_sum += nll.sum(); byte_sum += tb.sum(); token_count += pred_len

    # Final restore of base weights
    with torch.no_grad():
        for p, saved in zip(unfrozen_params, base_values):
            p.data.copy_(saved)

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, byte_sum, token_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb

# ---- TRAINING ----
def _build_gpt(args, **ov):
    kw = dict(vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads, mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
        ve_layers=args.ve_layers, activation=args.mlp_activation, leaky_slope=args.leaky_slope,
        gated_attention=args.gated_attention, value_residual=args.value_residual)
    kw.update(ov)
    return GPT(**kw)

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
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device); dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    logfile = None
    if master_process: os.makedirs("logs", exist_ok=True); logfile = f"logs/{args.run_id}.txt"; print(logfile)
    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)
    log0(code, console=False); log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    assert args.tokenizer_path.endswith(".model"), f"Need .model file: {args.tokenizer_path}"
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    assert int(sp.vocab_size()) == args.vocab_size, f"Vocab mismatch: {args.vocab_size} vs {sp.vocab_size()}"
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    # MODEL + OPTIMIZER SETUP
    CastedLinear._qat_enabled = args.qat_enabled
    CastedLinear._qat_bits = args.quant_bits
    base_model = _build_gpt(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear): module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    # Optimizer parameter groups
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.mtp_num_heads > 0:
        matrix_params.extend([p for p in base_model.mtp_heads.parameters() if p.ndim == 2])
    scalar_params = [p for n, p in block_named_params
                     if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None: scalar_params.append(base_model.bigram.scale)
    if base_model.ve_shared is not None:
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales: scalar_params.append(s)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None: matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None: matrix_params.append(base_model.ve_shared.proj.weight)
    optimizer_tok = torch.optim.AdamW(tok_params, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    if args.adamuon_enabled:
        log0(f"optimizer:AdaMuon beta2={args.adamuon_beta2} eps={args.adamuon_eps}")
        optimizer_muon = AdaMuon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                                  backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd,
                                  beta2=args.adamuon_beta2, eps=args.adamuon_eps)
    else:
        optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for group in optimizer_muon.param_groups: group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizers.insert(1, torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True))
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{[i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"matrix_lr:{args.matrix_lr} mlp_activation:{args.mlp_activation} leaky_slope:{args.leaky_slope}")
    log0(f"ema_enabled:{args.ema_enabled} ema_decay:{args.ema_decay} swa_enabled:{args.swa_enabled}")
    log0(f"gptq_enabled:{args.gptq_enabled} quant_bits:{args.quant_bits} prune_pct:{args.prune_pct}")
    log0(f"gated_attention:{args.gated_attention} value_residual:{args.value_residual}")
    log0(f"ttt_enabled:{args.ttt_enabled} ttt_lr:{args.ttt_lr} ttt_momentum:{args.ttt_momentum} "
         f"ttt_sgd_epochs:{args.ttt_sgd_epochs} ttt_chunk_size:{args.ttt_chunk_size} "
         f"ttt_unfreeze_layers:{args.ttt_unfreeze_layers}")
    log0(f"warmdown_iters:{args.warmdown_iters} bigram_vocab_size:{args.bigram_vocab_size}")
    log0(f"late_qat_threshold:{args.late_qat_threshold}")
    log0(f"seed:{args.seed}")
    # DATA LOADER & MODEL WARMUP
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            wds = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if wds <= step < args.iterations else 1.0
        sms = elapsed_ms / max(step, 1); wdms = args.warmdown_iters * sms; rms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rms / max(wdms, 1e-9) if rms <= wdms else 1.0
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step+1) % 10 == 0 or warmup_step+1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step+1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True): opt.load_state_dict(state)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    # EMA + SWA STATE
    swa_state, swa_count = None, 0
    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()} if args.ema_enabled else None
    training_time_ms = 0.0; stop_after_step = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True; log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all(); train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True): loss = model(x, y)
            train_loss += loss.detach(); (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        mm = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups: group["momentum"] = mm
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all()
        # EMA update
        if ema_state is not None:
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(args.ema_decay).add_(t.detach().float(), alpha=1.0 - args.ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}; swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for n, t in base_model.state_dict().items(): swa_state[n] += t.detach().cpu()
                swa_count += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/step:.2f}ms")
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rct = torch.tensor(int(reached_cap), device=device); dist.all_reduce(rct, op=dist.ReduceOp.MAX); reached_cap = bool(rct.item())
        if stop_after_step is None and reached_cap: stop_after_step = step
    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
    # Apply EMA (preferred) or SWA
    if args.ema_enabled and ema_state is not None:
        log0("ema:applying EMA weights")
        current_state = base_model.state_dict()
        base_model.load_state_dict({name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}, strict=True)
    elif args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"swa:applying averaged {swa_count} checkpoints")
        base_model.load_state_dict({n: (t/swa_count).to(dtype=base_model.state_dict()[n].dtype) for n, t in swa_state.items()}, strict=True)
    # DIAGNOSTIC: post-EMA eval
    torch.cuda.synchronize(); t_diag = time.perf_counter()
    diag_vl, diag_vb = eval_val(args, compiled_model, rank, world_size, device, grad_accum_steps,
                                 val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"DIAGNOSTIC post_ema val_loss:{diag_vl:.4f} val_bpb:{diag_vb:.4f} eval_time:{1000.0*(time.perf_counter()-t_diag):.0f}ms")
    # SERIALIZATION: pruning + GPTQ quantization with retry loop
    full_sd = base_model.state_dict()
    export_sd = {k: v for k, v in full_sd.items() if "mtp_heads" not in k}
    excluded_mtp = sum(int(t.numel()) for k, t in full_sd.items() if "mtp_heads" in k)
    if excluded_mtp > 0: log0(f"export_excluding_mtp_params:{excluded_mtp}")
    if master_process:
        torch.save(export_sd, "final_model.pt")
        log0(f"Serialized model: {os.path.getsize('final_model.pt')} bytes")
        log0(f"Code size: {len(code.encode('utf-8'))} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    # Pruning: zero out smallest weights
    if args.prune_pct > 0:
        for k, v in sd_cpu.items():
            if v.ndim == 2 and v.numel() > 65536:
                thresh = torch.quantile(v.abs().float(), args.prune_pct)
                v[v.abs() < thresh] = 0.0
        log0(f"pruning:{args.prune_pct*100:.1f}% magnitude pruning applied")
    # GPTQ quantization with retry loop for size constraint
    quant_clip_range = (2 ** (args.quant_bits - 1)) - 1  # 15 for int5, 31 for int6
    max_artifact_bytes = 15_900_000  # safety margin below 16MB
    quant_tag = f"int{args.quant_bits}"
    prune_pct_cur = args.prune_pct
    for gptq_attempt in range(4):
        if gptq_attempt > 0:
            prune_pct_cur = args.prune_pct + 0.01 * gptq_attempt
            sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
            for k, v in sd_cpu.items():
                if v.ndim == 2 and v.numel() > 65536:
                    thresh = torch.quantile(v.abs().float(), prune_pct_cur)
                    v[v.abs() < thresh] = 0.0
            log0(f"gptq:retry {gptq_attempt} with prune_pct={prune_pct_cur:.3f}")
        if args.gptq_enabled:
            log0(f"gptq:calibrating with training data (bits={args.quant_bits}, clip_range={quant_clip_range})...")
            t_gptq = time.perf_counter()
            gptq_hessians = gptq_calibrate(base_model, args.train_files, device,
                                            n_samples=args.gptq_calibration_batches, seq_len=args.train_seq_len)
            log0(f"gptq:calibrated {len(gptq_hessians)} layers in {time.perf_counter()-t_gptq:.1f}s")
            quant_result, quant_meta = mixed_quantize(sd_cpu, {"mlp", "attn"}, gptq_hessians, clip_range=quant_clip_range, percdamp=args.gptq_percdamp, block_size=args.gptq_block_size)
        else:
            quant_result, quant_meta = mixed_quantize(sd_cpu, {"mlp", "attn"}, clip_range=quant_clip_range, percdamp=args.gptq_percdamp, block_size=args.gptq_block_size)
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        if _COMPRESSOR == "lzma":
            quant_blob = _lzma.compress(quant_raw, preset=8)
        elif _COMPRESSOR == "zstd":
            quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
        else:
            quant_blob = zlib.compress(quant_raw, 9)
        artifact_bytes = len(quant_blob)
        log0(f"gptq:attempt {gptq_attempt} artifact={artifact_bytes} bytes (limit={max_artifact_bytes})")
        if artifact_bytes <= max_artifact_bytes:
            break
    if master_process:
        with open(f"final_model.{quant_tag}.ptz", "wb") as f: f.write(quant_blob)
        log0(f"Serialized model {quant_tag}+{_COMPRESSOR}: {len(quant_blob)} bytes")
        log0(f"Total submission size {quant_tag}+{_COMPRESSOR}: {len(quant_blob) + len(code.encode('utf-8'))} bytes")
    if distributed: dist.barrier()
    with open(f"final_model.{quant_tag}.ptz", "rb") as f: quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_lzma.decompress(quant_blob_disk) if _COMPRESSOR == "lzma" else zstandard.ZstdDecompressor().decompress(quant_blob_disk) if _COMPRESSOR == "zstd" else zlib.decompress(quant_blob_disk)),
        map_location="cpu")
    deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu)
    eval_model = _build_gpt(args, mtp_num_heads=0, mtp_loss_weight=0.0).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    # Legal score-first SGD TTT
    if args.ttt_enabled:
        log0(f"ttt:starting legal score-first SGD TTT lr={args.ttt_lr} momentum={args.ttt_momentum} "
             f"sgd_epochs={args.ttt_sgd_epochs} chunk_size={args.ttt_chunk_size} "
             f"eval_seq_len={args.ttt_eval_seq_len} unfreeze_layers={args.ttt_unfreeze_layers} "
             f"max_secs={args.ttt_max_secs}")
        t_ttt = time.perf_counter()
        CastedLinear._qat_enabled = False
        ttt_loss, ttt_bpb = eval_val_ttt_sgd(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        if ttt_loss is not None:
            log0(f"ttt:score-first-sgd val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f}")
            log0(f"ttt:score-first-sgd_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")
        else:
            log0("ttt:no documents found, skipped")
        log0(f"ttt:completed in {time.perf_counter()-t_ttt:.1f}s")
    # Post-quantization evaluation
    sw_seq_len = effective_eval_seq_len
    if not args.ngram_enabled:
        # Standard eval only when n-gram is disabled
        compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
        torch.cuda.synchronize(); t_qe = time.perf_counter()
        qvl, qvb = eval_val(args, compiled_eval, rank, world_size, device, grad_accum_steps, val_tokens,
                              base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, eval_seq_len=effective_eval_seq_len)
        torch.cuda.synchronize()
        log0(f"final_{quant_tag}_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} eval_time:{1000.0*(time.perf_counter()-t_qe):.0f}ms")
        if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
            torch.cuda.synchronize(); t_sl = time.perf_counter()
            swl, swb = eval_val_sliding(args, eval_model, rank, world_size, device, val_tokens,
                                         base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                                         stride=args.eval_stride, eval_seq_len=sw_seq_len)
            torch.cuda.synchronize()
            log0(f"final_{quant_tag}_sliding_window val_loss:{swl:.4f} val_bpb:{swb:.4f} stride:{args.eval_stride} eval_time:{1000.0*(time.perf_counter()-t_sl):.0f}ms")
            log0(f"final_int8_zlib_roundtrip_exact val_loss:{swl:.8f} val_bpb:{swb:.8f}")
    # N-gram cache eval (PR#809 pattern: chunk-based, vectorized, entropy-adaptive)
    if args.ngram_enabled:
        log0(f"ngram:starting n={args.ngram_n} alpha_max={args.ngram_alpha} stride={args.eval_stride}")
        torch.cuda.synchronize(); t_ng = time.perf_counter()
        ng_loss, ng_bpb, ng_ms = eval_val_ngram(
            args, eval_model, rank, world_size, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            eval_seq_len=sw_seq_len)
        torch.cuda.synchronize()
        log0(f"ngram:val_loss:{ng_loss:.4f} val_bpb:{ng_bpb:.4f} eval_time:{ng_ms:.0f}ms")
        log0(f"ngram_exact:val_loss:{ng_loss:.8f} val_bpb:{ng_bpb:.8f}")
    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
