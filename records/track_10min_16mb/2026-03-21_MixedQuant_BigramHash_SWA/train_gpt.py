"""Parameter Golf 025_optimized — based on top leaderboard recipes
10L ReLU² MLP3x, BigramHash, OrthoInit, Mixed INT5/INT6 STE, SWA, Muon WD=0.04,
zstd-22, grad_clip=0.3. Stripped of slow features (SwiGLU, XSA, window attn, partial RoPE).
"""
from __future__ import annotations
import copy, glob, io, math, os, random, subprocess, sys, time, uuid, zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
USE_ZSTD = HAS_ZSTD and bool(int(os.environ.get("USE_ZSTD", "1" if HAS_ZSTD else "0")))

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
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_hidden = int(os.environ.get("MLP_HIDDEN", 1536))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    bigram_hash_size = int(os.environ.get("BIGRAM_HASH_SIZE", 10240))
    bigram_embed_dim = int(os.environ.get("BIGRAM_EMBED_DIM", 128))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.5))
    swa_every = int(os.environ.get("SWA_EVERY", 50))

# --- STE QUANTIZATION ---
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, bits):
        half = (1 << (bits - 1)) - 1
        sc = w.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / half
        return ((w / sc).round().clamp(-half, half) * sc).to(w.dtype)
    @staticmethod
    def backward(ctx, grad):
        return grad, None

def ste_quantize(w, bits):
    return STEQuantize.apply(w, bits) if bits > 0 else w

# --- MUON OPTIMIZER ---
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
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov, wd = group["backend_steps"], group["nesterov"], group["weight_decay"]
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
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# --- TOKENIZER EVAL ---
def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size())
    sz = max(sp_vs, vocab_size)
    base_np = np.zeros((sz,), dtype=np.int16)
    space_np = np.zeros((sz,), dtype=np.bool_)
    bound_np = np.ones((sz,), dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        bound_np[tid] = False
        if sp.is_byte(tid):
            base_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            space_np[tid] = True
            piece = piece[1:]
        base_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_np, dtype=torch.int16, device=device),
            torch.tensor(space_np, dtype=torch.bool, device=device),
            torch.tensor(bound_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files for {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]

def _count_bytes(x_flat, y_flat, base_lut, space_lut, bound_lut):
    tb = base_lut[y_flat].to(torch.int16)
    tb += (space_lut[y_flat] & ~bound_lut[x_flat]).to(torch.int16)
    return tb.to(torch.float64).sum()

def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_lut, space_lut, bound_lut):
    local_bt = args.val_batch_size // (world_size * grad_accum_steps)
    local_bs = local_bt // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    s0 = (total_seqs * rank) // world_size
    s1 = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(s0, s1, local_bs):
            be = min(bs + local_bs, s1)
            r0, r1 = bs * args.train_seq_len, be * args.train_seq_len + 1
            local = val_tokens[r0:r1].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                bl = model(x, y).detach()
            n = float(y.numel())
            loss_sum += bl.to(torch.float64) * n
            tok_cnt += n
            byte_cnt += _count_bytes(x.reshape(-1), y.reshape(-1), base_lut, space_lut, bound_lut)
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_cnt, byte_cnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = loss_sum / tok_cnt
    bpt = vl.item() / math.log(2.0)
    tpb = tok_cnt.item() / byte_cnt.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)

# --- QUANTIZATION (Mixed INT5/INT6/INT8 + zstd/zlib) ---
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get("CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,skip_weights").split(",") if p)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def tensor_nbytes(t):
    return int(t.numel()) * int(t.element_size())

def _storage_bits(name):
    if 'tok_emb' in name or 'lm_head' in name or 'bigram_embed' in name:
        return 8
    return 6

def keep_float_tensor(name, t, pt_dtypes):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        pt_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t, bits):
    half = (1 << (bits - 1)) - 1
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                    if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / half).clamp_min(1.0 / half)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -half, half).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / half if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -half, half).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_mixed(state_dict):
    quantized, scales, dtypes = {}, {}, {}
    passthrough, pt_dtypes, qmeta = {}, {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors",
                           "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
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
            kept = keep_float_tensor(name, t, pt_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        bits = _storage_bits(name)
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t, bits)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0, "bits": bits}
        else:
            qmeta[name] = {"bits": bits}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {"__quant_format__": "mixed_int_v1", "quantized": quantized,
           "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta:
        obj["qmeta"] = qmeta
    if pt_dtypes:
        obj["passthrough_orig_dtypes"] = pt_dtypes
    return obj, stats

def dequantize_state_dict(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    pt_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s32 = s.to(torch.float32)
            out[name] = (q.float() * s32.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        od = pt_dtypes.get(name)
        if isinstance(od, str):
            out_t = out_t.to(dtype=getattr(torch, od)).contiguous()
        out[name] = out_t
    return out

def compress_bytes(data):
    if USE_ZSTD:
        return zstandard.ZstdCompressor(level=22).compress(data)
    return zlib.compress(data, level=9)

def decompress_bytes(data):
    if len(data) >= 4 and data[:4] == b'\x28\xb5\x2f\xfd':
        if not HAS_ZSTD:
            raise RuntimeError("zstd data but zstandard not installed")
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=len(data) * 20)
    return zlib.decompress(data)

# --- DATA LOADING ---
def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    expected = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n):
        chunks = []
        rem = n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank = local_tokens + 1
        chunk = self.stream.take(per_rank * self.world_size)
        start = self.rank * per_rank
        local = chunk[start:start + per_rank].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# --- TRANSFORMER MODULES ---
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=False, ste_bits=0):
        super().__init__(in_f, out_f, bias)
        self.ste_bits = ste_bits
    def forward(self, x):
        w = self.weight
        if self.ste_bits > 0 and self.training:
            w = ste_quantize(w, self.ste_bits)
        return F.linear(x, w.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0
        self._cos = None
        self._sin = None
    def forward(self, seq_len, device, dtype):
        if self._cos is None or self._cache_len != seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
            self._cache_len = seq_len
        return self._cos.to(dtype), self._sin.to(dtype)

def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False, ste_bits=6)
        self.c_k = CastedLinear(dim, kv_dim, bias=False, ste_bits=6)
        self.c_v = CastedLinear(dim, kv_dim, bias=False, ste_bits=6)
        self.proj = CastedLinear(dim, dim, bias=False, ste_bits=6)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        B, S, D = x.shape
        H, Hkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q = self.c_q(x).reshape(B, S, H, hd).transpose(1, 2)
        k = self.c_k(x).reshape(B, S, Hkv, hd).transpose(1, 2)
        v = self.c_v(x).reshape(B, S, Hkv, hd).transpose(1, 2)
        q = F.rms_norm(q, (hd,))
        k = F.rms_norm(k, (hd,))
        cos, sin = self.rotary(S, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if H != Hkv:
            k = k.repeat_interleave(H // Hkv, dim=1)
            v = v.repeat_interleave(H // Hkv, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(B, S, D)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc = CastedLinear(dim, hidden, bias=False, ste_bits=6)
        self.proj = CastedLinear(hidden, dim, bias=False, ste_bits=6)
        self.proj._zero_init = True
    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_hidden, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_hidden, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, bigram_hash_size, bigram_embed_dim):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.bigram_hash_size = bigram_hash_size
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram_embed = nn.Embedding(bigram_hash_size, bigram_embed_dim)
        self.bigram_proj = CastedLinear(bigram_embed_dim, model_dim, bias=False, ste_bits=0)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(Block(model_dim, num_heads, num_kv_heads, mlp_hidden,
                                     rope_base, qk_gain_init))
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False, ste_bits=0)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            with torch.no_grad():
                U, _, _ = torch.linalg.svd(torch.randn_like(self.tok_emb.weight), full_matrices=False)
                self.tok_emb.weight.copy_(U * self.tied_embed_init_std)
        nn.init.normal_(self.bigram_embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, CastedLinear)) and module is not self.bigram_proj:
                if getattr(module, '_zero_init', False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and not isinstance(module, nn.Embedding):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def forward(self, input_ids, target_ids):
        prev = F.pad(input_ids[:, :-1], (1, 0), value=0)
        bh = (prev * 7919 + input_ids) % self.bigram_hash_size
        x = self.tok_emb(input_ids) + self.bigram_proj(self.bigram_embed(bh))
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
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                              target_ids.reshape(-1), reduction="mean")

# --- TRAINING ---
def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if int(os.environ.get("TORCH_COMPILE", 1)):
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.optimize_ddp = False
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_lut, space_lut, bound_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_path={args.tokenizer_path}")
    log0(f"train_shards:{actual_train_files} val_tokens:{val_tokens.numel() - 1}")

    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        mlp_hidden=args.mlp_hidden, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_hash_size=args.bigram_hash_size, bigram_embed_dim=args.bigram_embed_dim,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    if int(os.environ.get("TORCH_COMPILE", 1)):
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    else:
        compiled_model = base_model
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named
                     if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    bigram_params = list(base_model.bigram_embed.parameters()) + list(base_model.bigram_proj.parameters())
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    embed_params_list = [base_model.tok_emb.weight] + [p for p in bigram_params]
    opt_tok = torch.optim.Adam(
        [{"params": embed_params_list, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps, weight_decay=args.muon_weight_decay)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar]
    if base_model.lm_head is not None:
        opt_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, opt_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} mlp_hidden:{args.mlp_hidden}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"features: ReLU²MLP3x BigramHash({args.bigram_hash_size}x{args.bigram_embed_dim}) "
         f"OrthoInit MixedINT5/6STE SWA({args.swa_start_frac}@{args.swa_every}) "
         f"GradClip({args.grad_clip_norm}) MuonWD({args.muon_weight_decay})")
    log0(f"compress:{'zstd-22' if USE_ZSTD else 'zlib-9'} quant:INT5(MLP)/INT6(attn)/INT8(embed)")
    log0(f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} matrix_lr:{args.matrix_lr}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} seq_len:{args.train_seq_len} "
         f"iterations:{args.iterations} warmup:{args.warmup_steps} wallclock:{args.max_wallclock_seconds:.0f}s")
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wc_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wc_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem_ms = max(max_wc_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    # Warmup (JIT compile)
    if args.warmup_steps > 0:
        init_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_states = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for wstep in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for o in optimizers:
                o.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (wstep + 1) % 10 == 0 or wstep + 1 == args.warmup_steps:
                log0(f"warmup_step:{wstep + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_model_state, strict=True)
        for o, s in zip(optimizers, init_opt_states, strict=True):
            o.load_state_dict(s)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # SWA setup
    swa_state = None
    swa_count = 0
    swa_start_step = None  # will be calculated dynamically based on wallclock

    # Main loop
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_val = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_val:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                              val_tokens, base_lut, space_lut, bound_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        mm = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups:
            g["momentum"] = mm
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers:
            o.step()
        zero_grad_all()

        # SWA: determine start dynamically based on wallclock fraction
        if swa_start_step is None and max_wc_ms is not None:
            approx_now = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            if approx_now >= max_wc_ms * args.swa_start_frac:
                swa_start_step = step
                log0(f"swa_started step:{step}")
        elif swa_start_step is None and step >= int(args.iterations * args.swa_start_frac):
            swa_start_step = step
            log0(f"swa_started step:{step}")

        if swa_start_step is not None and step >= swa_start_step and step % args.swa_every == 0:
            with torch.no_grad():
                if swa_state is None:
                    swa_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
                    swa_count = 1
                else:
                    for k, v in base_model.state_dict().items():
                        swa_state[k] += v.detach()
                    swa_count += 1

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0)
        if should_log:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")

        reached_cap = max_wc_ms is not None and approx_ms >= max_wc_ms
        if distributed and max_wc_ms is not None:
            cap_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak_mem alloc:{torch.cuda.max_memory_allocated() // 1024 // 1024}MiB "
         f"reserved:{torch.cuda.max_memory_reserved() // 1024 // 1024}MiB")

    # Load SWA weights
    if swa_state is not None and swa_count > 0:
        with torch.no_grad():
            for k in swa_state:
                swa_state[k] /= swa_count
        base_model.load_state_dict(swa_state, strict=True)
        log0(f"loaded SWA weights (averaged {swa_count} snapshots)")

    # Serialization
    if master:
        torch.save(base_model.state_dict(), "final_model.pt")
        mb = os.path.getsize("final_model.pt")
        cb = len(code.encode("utf-8"))
        log0(f"raw model:{mb} code:{cb} total:{mb + cb}")

    quant_obj, quant_stats = quantize_state_dict_mixed(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = compress_bytes(quant_raw)
    comp_label = "zstd" if USE_ZSTD else "zlib"
    fname = "final_model.mixed.ptz"
    if master:
        with open(fname, "wb") as f:
            f.write(quant_blob)
        qfb = os.path.getsize(fname)
        cb = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"quant+{comp_label}: {qfb} bytes (payload:{quant_stats['int8_payload_bytes']} ratio:{ratio:.2f}x)")
        log0(f"artifact: {qfb + cb} bytes {'PASS' if qfb + cb <= 16_000_000 else 'FAIL:OVER_16MB'}")

    # Roundtrip validation
    if distributed:
        dist.barrier()
    with open(fname, "rb") as f:
        blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(decompress_bytes(blob_disk)), map_location="cpu", weights_only=False)
    base_model.load_state_dict(dequantize_state_dict(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qe = time.perf_counter()
    qvl, qvb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                        val_tokens, base_lut, space_lut, bound_lut)
    log0(f"roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} eval:{1000.0 * (time.perf_counter() - t_qe):.0f}ms")
    log0(f"final_roundtrip val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
