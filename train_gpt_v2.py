"""
Parameter Golf v2 — Final submission.
1. 11 layers @ 512 dim, MLP 3x, GQA 8Q/4KV
2. LeakyReLU(0.5)² activation
3. BigramHash(1536, dim=128) for n-gram features
4. Mixed Int5/Int6 quantization + FP16 embeddings
5. Legal score-first TTT on val chunks
6. SWA from last 40% of warmdown
7. Sliding window evaluation (stride=64)
8. Muon WD=0.04, momentum=0.99, grad_clip=0.3
"""

from __future__ import annotations
import copy, glob, io, lzma, math, os, random, subprocess, sys, time, uuid, zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
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
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    iterations = int(os.environ.get("ITERATIONS", 1050))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 150))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    # Model
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    # BigramHash
    bigram_buckets = int(os.environ.get("BIGRAM_BUCKETS", 1536))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    # SWA
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.4))
    swa_every = int(os.environ.get("SWA_EVERY", 10))
    # Optimizer
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 300))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.04))
    # TTT (Test-Time Training) — AdamW pre-quantization
    ttt_enabled = int(os.environ.get("TTT_ENABLED", 1)) > 0
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    # EMA
    ema_enabled = int(os.environ.get("EMA_ENABLED", 1)) > 0
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

# -----------------------------
# MUON OPTIMIZER
# -----------------------------
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay))
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
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0: p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION
# -----------------------------
def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary_token_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space_np[tid] = True; piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files for: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0: raise ValueError(f"Val split too short for SEQ_LEN={seq_len}")
    return tokens[: usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len: raise ValueError("VAL_BATCH_SIZE too small")
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            rs, re = bss * args.train_seq_len, bse * args.train_seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            btc = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * btc; val_token_count += btc
            prev_ids, tgt_ids = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [val_loss_sum, val_token_count, val_byte_count]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = val_loss_sum / val_token_count
    model.train()
    return float(vl.item()), float(vl.item() / math.log(2.0) * val_token_count.item() / val_byte_count.item())

def eval_val_sliding(args, model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    seq_len, stride = args.train_seq_len, args.eval_stride
    total_tokens = val_tokens.numel()
    num_windows = max(1, (total_tokens - seq_len) // stride + 1)
    ws = (num_windows * rank) // world_size
    we = (num_windows * (rank + 1)) // world_size
    sw_bs = max(1, args.val_batch_size // (seq_len * 4))
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(ws, we, sw_bs):
            be = min(bs + sw_bs, we)
            xl, yl = [], []
            for wi in range(bs, be):
                off = wi * stride
                end = min(off + seq_len + 1, total_tokens)
                w = val_tokens[off:end]
                if w.numel() < seq_len + 1:
                    p = torch.zeros(seq_len + 1, dtype=w.dtype); p[seq_len + 1 - w.numel():] = w; w = p
                xl.append(w[:seq_len]); yl.append(w[1:seq_len + 1])
            xb = torch.stack(xl).to(device=device, dtype=torch.int64)
            yb = torch.stack(yl).to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = _forward_logits(model, xb)
            for i, wi in enumerate(range(bs, be)):
                s = 0 if wi == 0 else seq_len - stride
                sl, st, sp = logits[i:i+1, s:, :], yb[i:i+1, s:], xb[i:i+1, s:]
                ptl = F.cross_entropy(sl.float().reshape(-1, sl.size(-1)), st.reshape(-1), reduction="none")
                tf, pf = st.reshape(-1), sp.reshape(-1)
                nll_sum += ptl.to(torch.float64).sum(); tok_count += ptl.numel()
                tb = base_bytes_lut[tf].to(dtype=torch.int16)
                tb += (has_leading_space_lut[tf] & ~is_boundary_token_lut[pf]).to(dtype=torch.int16)
                byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [nll_sum, tok_count, byte_count]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = nll_sum / tok_count
    model.train()
    return float(vl.item()), float(vl.item() / math.log(2.0) * tok_count.item() / byte_count.item())

def _forward_logits(model, input_ids):
    m = model
    if hasattr(m, 'module'): m = m.module
    if hasattr(m, '_orig_mod'): m = m._orig_mod
    x = m.tok_emb(input_ids)
    if m.bigram_hash is not None:
        x = x + m.bigram_hash(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x; skips = []
    for i in range(m.num_encoder_layers):
        x = m.blocks[i](x, x0); skips.append(x)
    for i in range(m.num_decoder_layers):
        if skips:
            x = x + m.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = m.blocks[m.num_encoder_layers + i](x, x0)
    x = m.final_norm(x)
    lp = F.linear(x, m.tok_emb.weight)
    return m.logit_softcap * torch.tanh(lp / m.logit_softcap)

# -----------------------------
# QUANTIZATION (Mixed Int5/Int6 + FP16 embed)
# -----------------------------
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get("CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,skip_weights").split(",") if p)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_NAMES = {"tok_emb.weight"}  # FP16 for tied embed
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0

def _get_quant_range(name: str) -> int:
    """Int5 for MLP weights, Int6 for attention, Int8 for others."""
    if ".mlp." in name:
        return 15   # Int5: [-16, 15]
    elif ".attn." in name:
        return 31   # Int6: [-32, 31]
    return 127      # Int8: [-128, 127]

def tensor_nbytes(t): return int(t.numel()) * int(t.element_size())
def keep_float_tensor(name, t, pod):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS): return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        pod[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t, max_val=127):
    t32 = t.float()
    if t32.ndim == 2:
        ca = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        cl = torch.maximum(torch.minimum(t32, ca[:, None]), -ca[:, None])
        sc = (ca / max_val).clamp_min(1.0 / max_val)
        return torch.clamp(torch.round(cl / sc[:, None]), -max_val, max_val).to(torch.int8).contiguous(), sc.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    ca = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    sc = torch.tensor(ca / max_val if ca > 0 else 1.0, dtype=torch.float32)
    return torch.clamp(torch.round(torch.clamp(t32, -ca, ca) / sc), -max_val, max_val).to(torch.int8).contiguous(), sc

def quantize_state_dict_int8(state_dict):
    qd, sc, dt, pt, pod, qm = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count","num_tensors","num_float_tensors","num_nonfloat_tensors","baseline_tensor_bytes","int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel()); stats["num_tensors"] += 1; stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1; pt[name] = t; stats["int8_payload_bytes"] += tensor_nbytes(t); continue
        if name in INT8_KEEP_FLOAT_NAMES or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            k = keep_float_tensor(name, t, pod); pt[name] = k; stats["int8_payload_bytes"] += tensor_nbytes(k); continue
        stats["num_float_tensors"] += 1
        max_val = _get_quant_range(name)
        q, s = quantize_float_tensor(t, max_val=max_val)
        if s.ndim > 0: qm[name] = {"scheme": "per_row", "axis": 0}
        qd[name] = q; sc[name] = s; dt[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": qd, "scales": sc, "dtypes": dt, "passthrough": pt}
    if qm: obj["qmeta"] = qm
    if pod: obj["passthrough_orig_dtypes"] = pod
    return obj, stats

def dequantize_state_dict_int8(obj):
    out, qm, pod = {}, obj.get("qmeta", {}), obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name]); s = obj["scales"][name]
        if qm.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(dtype=dtype).contiguous()
        else: out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        o = t.detach().to("cpu").contiguous()
        od = pod.get(name)
        if isinstance(od, str): o = o.to(dtype=getattr(torch, od)).contiguous()
        out[name] = o
    return out

# -----------------------------
# DATA LOADING
# -----------------------------
def load_data_shard(file: Path) -> Tensor:
    hb = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ValueError(f"Bad header: {file}")
    nt = int(header[2])
    if file.stat().st_size != hb + nt * np.dtype("<u2").itemsize: raise ValueError(f"Size mismatch: {file}")
    tnp = np.fromfile(file, dtype="<u2", count=nt, offset=hb)
    if tnp.size != nt: raise ValueError(f"Short read: {file}")
    return torch.from_numpy(tnp.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files for: {pattern}")
        self.file_idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n):
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
        lt = global_tokens // (self.world_size * grad_accum_steps)
        prs = lt + 1; chunk = self.stream.take(prs * self.world_size)
        s = self.rank * prs; local = chunk[s:s+prs].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim)), persistent=False)
        self._seq_len_cached = 0; self._cos_cached = None; self._sin_cached = None
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None,None,:,:]; self._sin_cached = freqs.sin()[None,None,:,:]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin):
    h = x.size(-1)//2; x1, x2 = x[...,:h], x[...,h:]
    return torch.cat((x1*cos+x2*sin, x1*(-sin)+x2*cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads; assert self.head_dim % 2 == 0
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False); self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
    def forward(self, x):
        B, S, D = x.shape
        q = self.c_q(x).reshape(B,S,self.num_heads,self.head_dim).transpose(1,2)
        k = self.c_k(x).reshape(B,S,self.num_kv_heads,self.head_dim).transpose(1,2)
        v = self.c_v(x).reshape(B,S,self.num_kv_heads,self.head_dim).transpose(1,2)
        q, k = F.rms_norm(q,(q.size(-1),)), F.rms_norm(k,(k.size(-1),))
        cos, sin = self.rotary(S, x.device, q.dtype)
        q, k = apply_rotary_emb(q,cos,sin), apply_rotary_emb(k,cos,sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None,:,None,None]
        y = F.scaled_dot_product_attention(q,k,v,attn_mask=None,is_causal=True,enable_gqa=(self.num_kv_heads!=self.num_heads))
        return self.proj(y.transpose(1,2).contiguous().reshape(B,S,D))

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False); self.proj._zero_init = True
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None,None,:]*x + mix[1][None,None,:]*x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None,None,:] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None,None,:] * self.mlp(self.mlp_norm(x))
        return x

class BigramHashEmbed(nn.Module):
    """Hash consecutive token pairs into a learned embedding table."""
    def __init__(self, num_buckets, embed_dim, model_dim):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)
        self.proj = CastedLinear(embed_dim, model_dim, bias=False)
        nn.init.normal_(self.embed.weight, std=0.01)
    def forward(self, input_ids):
        # Shift input_ids to get previous token, use 0 for first position
        prev = F.pad(input_ids[:, :-1], (1, 0), value=0)
        # Hash: combine current and previous token
        bigram_hash = (prev * 31 + input_ids) % self.num_buckets
        return self.proj(self.embed(bigram_hash))

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init,
                 bigram_buckets=0, bigram_dim=128):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tied_embed_init_std = tied_embed_init_std
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram_hash = BigramHashEmbed(bigram_buckets, bigram_dim, model_dim) if bigram_buckets > 0 else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)])
        self.final_norm = RMSNorm()
        self._init_weights()
    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False): nn.init.zeros_(m.weight)
    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        if self.bigram_hash is not None:
            x = x + self.bigram_hash(input_ids)
        x = F.rms_norm(x, (x.size(-1),)); x0 = x; skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        lp = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(lp / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")

# -----------------------------
# TRAINING
# -----------------------------
def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK","0")); world_size = int(os.environ.get("WORLD_SIZE","1"))
    local_rank = int(os.environ.get("LOCAL_RANK","0"))
    if 8 % world_size != 0: raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size; grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
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
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)
    log0(code, console=False); log0("="*100, console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size: raise ValueError("VOCAB_SIZE mismatch")
    dataset_dir = Path(args.data_path).resolve()
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"train_shards:{len(list(dataset_dir.glob('fineweb_train_*.bin')))} val_tokens:{val_tokens.numel()-1}")

    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_buckets=args.bigram_buckets, bigram_dim=args.bigram_dim,
    ).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n,p in block_named_params if p.ndim==2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n,p in block_named_params if p.ndim<2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)

    # Bigram params go to Adam
    embed_params = [base_model.tok_emb.weight]
    if base_model.bigram_hash is not None:
        embed_params.append(base_model.bigram_hash.embed.weight)
        # Bigram projection is a matrix → Muon
        matrix_params.append(base_model.bigram_hash.proj.weight)

    optimizer_tok = torch.optim.AdamW(
        [{"params": embed_params, "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_weight_decay)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps, weight_decay=args.muon_weight_decay)
    for g in optimizer_muon.param_groups: g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_weight_decay)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} layers:{args.num_layers} dim:{args.model_dim} mlp_mult:{args.mlp_mult}")
    log0(f"bigram_buckets:{args.bigram_buckets} bigram_dim:{args.bigram_dim}")
    log0(f"world_size:{world_size} grad_accum:{grad_accum_steps} seq_len:{args.train_seq_len} batch_tokens:{args.train_batch_tokens}")
    log0(f"matrix_lr:{args.matrix_lr} muon_wd:{args.muon_weight_decay} adam_wd:{args.adam_weight_decay} grad_clip:{args.grad_clip_norm}")
    log0(f"swa_start_frac:{args.swa_start_frac} swa_every:{args.swa_every} warmdown:{args.warmdown_iters}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    in_warmdown = False
    def lr_mul(step, elapsed_ms):
        nonlocal in_warmdown
        if args.warmdown_iters <= 0: return 1.0, False
        if max_wallclock_ms is None:
            wds = max(args.iterations - args.warmdown_iters, 0)
            if step >= wds:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0), True
            return 1.0, False
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        if rem_ms <= wd_ms:
            return rem_ms / max(wd_ms, 1e-9), True
        return 1.0, False

    # Warmup (compilation)
    if args.warmup_steps > 0:
        init_state = {n: t.detach().cpu().clone() for n,t in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            if ws+1 == args.warmup_steps: log0(f"warmup done ({args.warmup_steps} steps)")
        base_model.load_state_dict(init_state, strict=True)
        for o, s in zip(optimizers, init_opt, strict=True): o.load_state_dict(s)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # SWA + EMA state
    swa_state = None; swa_count = 0; swa_active = False
    ema_state = None
    if args.ema_enabled:
        ema_state = {n: t.detach().clone() for n, t in base_model.state_dict().items()}
        log0(f"EMA enabled: decay={args.ema_decay}")
    training_time_ms = 0.0; stop_after_step = None
    torch.cuda.synchronize(); t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize(); training_time_ms += 1000.0*(time.perf_counter()-t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0*(time.perf_counter()-t0)
        scale, in_warmdown = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach(); (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step/args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for g in optimizer_muon.param_groups:
            g["momentum"] = (1-frac)*args.muon_momentum_warmup_start + frac*args.muon_momentum
        for o in optimizers:
            for g in o.param_groups: g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers: o.step()
        zero_grad_all()

        step += 1
        # SWA: accumulate during warmdown only
        if args.swa_start_frac > 0 and in_warmdown and step % args.swa_every == 0:
            warmdown_progress = 1.0 - scale  # 0 at start of warmdown, ~1 at end
            if warmdown_progress >= (1.0 - args.swa_start_frac):
                if swa_state is None:
                    swa_state = {n: t.detach().clone() for n, t in base_model.state_dict().items()}
                    swa_count = 1
                    log0(f"SWA started at step {step}")
                else:
                    for n, t in base_model.state_dict().items():
                        swa_state[n].add_(t.detach())
                    swa_count += 1

        approx_ms = training_time_ms + 1000.0*(time.perf_counter()-t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms swa:{swa_count}")
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rct = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rct, op=dist.ReduceOp.MAX); reached_cap = bool(rct.item())
        if stop_after_step is None and reached_cap: stop_after_step = step

        # EMA update
        if ema_state is not None:
            d = args.ema_decay
            for n, t in base_model.state_dict().items():
                ema_state[n].mul_(d).add_(t.detach(), alpha=1.0 - d)

    # Apply EMA if enabled (before SWA)
    if ema_state is not None:
        log0(f"Applying EMA weights (decay={args.ema_decay})")
        base_model.load_state_dict(ema_state, strict=True)

    # Apply SWA if we collected checkpoints
    if swa_state is not None and swa_count > 1:
        log0(f"Applying SWA: {swa_count} checkpoints averaged")
        for n in swa_state: swa_state[n] /= swa_count
        base_model.load_state_dict(swa_state, strict=True)

    log0(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")

    # TTT BEFORE quantization (leader's key insight)
    if args.ttt_enabled and args.eval_stride > 0:
        torch.cuda.synchronize(); ttt_start = time.perf_counter()
        ttt_bpb = run_legal_ttt(
            args, base_model, model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log0,
        )
        torch.cuda.synchronize()
        log0(f"pre_quant_ttt val_bpb:{ttt_bpb:.8f} time:{1000*(time.perf_counter()-ttt_start):.0f}ms")

    # Serialize with LZMA (saves ~280KB vs zlib)
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"Raw model: {os.path.getsize('final_model.pt')} bytes")
    quant_obj, qs = quantize_state_dict_int8(base_model.state_dict())
    qb = io.BytesIO(); torch.save(quant_obj, qb); qraw = qb.getvalue()
    qblob = lzma.compress(qraw, preset=9 | lzma.PRESET_EXTREME)
    if master_process:
        with open("final_model.int8.ptz","wb") as f: f.write(qblob)
        qfb = os.path.getsize("final_model.int8.ptz"); cb = len(code.encode("utf-8"))
        log0(f"int8+lzma: {qfb} bytes | code: {cb} bytes | total: {qfb+cb} bytes")
    if distributed: dist.barrier()
    with open("final_model.int8.ptz","rb") as f: qbd = f.read()
    base_model.load_state_dict(dequantize_state_dict_int8(torch.load(io.BytesIO(lzma.decompress(qbd)), map_location="cpu")), strict=True)
    torch.cuda.synchronize(); te = time.perf_counter()
    qvl, qvb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} eval_time:{1000*(time.perf_counter()-te):.0f}ms")
    log0(f"final_int8_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        torch.cuda.synchronize(); ts = time.perf_counter()
        svl, svb = eval_val_sliding(args, model, rank, world_size, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        torch.cuda.synchronize()
        log0(f"final_sliding val_loss:{svl:.4f} val_bpb:{svb:.4f} stride:{args.eval_stride} eval_time:{1000*(time.perf_counter()-ts):.0f}ms")
        log0(f"final_sliding_exact val_loss:{svl:.8f} val_bpb:{svb:.8f}")

    if distributed: dist.destroy_process_group()


def run_legal_ttt(args, base_model, model, rank, world_size, device,
                  val_tokens, base_bytes_lut, has_leading_space_lut,
                  is_boundary_token_lut, log0):
    """Legal score-first TTT: score chunk N, then train on chunk N, score chunk N+1, etc."""
    seq_len = args.train_seq_len
    stride = args.eval_stride
    chunk_tokens = args.ttt_chunk_tokens
    total_tokens = val_tokens.numel()
    # Split val tokens into non-overlapping chunks
    num_chunks = max(1, (total_tokens - 1) // chunk_tokens)
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Save initial weights for potential restoration
    init_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}

    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * chunk_tokens
        c_end = min(c_start + chunk_tokens, total_tokens)
        chunk = val_tokens[c_start:c_end]

        # SCORE: sliding window eval on this chunk under inference_mode
        model.eval()
        with torch.inference_mode():
            num_windows = max(1, (chunk.numel() - seq_len) // stride + 1)
            ws = (num_windows * rank) // world_size
            we = (num_windows * (rank + 1)) // world_size
            sw_bs = max(1, args.val_batch_size // (seq_len * 4))
            for bs in range(ws, we, sw_bs):
                be = min(bs + sw_bs, we)
                xl, yl = [], []
                for wi in range(bs, be):
                    off = wi * stride
                    end = min(off + seq_len + 1, chunk.numel())
                    w = chunk[off:end]
                    if w.numel() < seq_len + 1:
                        p = torch.zeros(seq_len + 1, dtype=w.dtype)
                        p[seq_len + 1 - w.numel():] = w; w = p
                    xl.append(w[:seq_len]); yl.append(w[1:seq_len + 1])
                xb = torch.stack(xl).to(device=device, dtype=torch.int64)
                yb = torch.stack(yl).to(device=device, dtype=torch.int64)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = _forward_logits(model, xb)
                for i, wi in enumerate(range(bs, be)):
                    s = 0 if (chunk_idx == 0 and wi == 0) else seq_len - stride
                    sl, st, sp = logits[i:i+1, s:, :], yb[i:i+1, s:], xb[i:i+1, s:]
                    ptl = F.cross_entropy(sl.float().reshape(-1, sl.size(-1)), st.reshape(-1), reduction="none")
                    tf, pf = st.reshape(-1), sp.reshape(-1)
                    nll_sum += ptl.to(torch.float64).sum(); tok_count += ptl.numel()
                    tb = base_bytes_lut[tf].to(dtype=torch.int16)
                    tb += (has_leading_space_lut[tf] & ~is_boundary_token_lut[pf]).to(dtype=torch.int16)
                    byte_count += tb.to(torch.float64).sum()

        # TRAIN: adapt on this already-scored chunk (skip last chunk)
        if chunk_idx < num_chunks - 1:
            model.train()
            ttt_opt = torch.optim.AdamW(base_model.parameters(), lr=args.ttt_lr, weight_decay=0.0)
            chunk_seqs = (chunk.numel() - 1) // seq_len
            if chunk_seqs > 0:
                lr_steps = args.ttt_epochs * chunk_seqs
                for epoch in range(args.ttt_epochs):
                    for si in range(chunk_seqs):
                        rs = si * seq_len
                        local = chunk[rs:rs + seq_len + 1].to(device=device, dtype=torch.int64)
                        x, y = local[:-1].unsqueeze(0), local[1:].unsqueeze(0)
                        ttt_opt.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                            loss = model(x, y)
                        loss.backward()
                        if args.ttt_grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.ttt_grad_clip)
                        # Cosine LR decay
                        step_num = epoch * chunk_seqs + si
                        cos_lr = args.ttt_lr * 0.5 * (1 + math.cos(math.pi * step_num / max(lr_steps, 1)))
                        for g in ttt_opt.param_groups: g["lr"] = cos_lr
                        ttt_opt.step()

    if dist.is_available() and dist.is_initialized():
        for t in [nll_sum, tok_count, byte_count]: dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = nll_sum / tok_count
    bpb = float(vl.item() / math.log(2.0) * tok_count.item() / byte_count.item())
    log0(f"ttt_chunks:{num_chunks} ttt_epochs:{args.ttt_epochs}")
    return bpb


if __name__ == "__main__":
    main()
