"""
train_gpt.py — Shared-Weight Recurrent Memory Transformer (RLM)
OpenAI Parameter Golf Competition — Non-Record Submission

Architecture: One transformer block looped N_LOOPS=16 times (weight tying).
  - Differential Attention V2 + per-loop low-rank Q delta (weight specialization)
  - resid_mix: learned carry-forward vs input-reset per loop
  - Loop position embeddings (activation-level depth signal)
  - Memory tokens: learned persistent cross-loop state (RoPE excluded)
  - Deep supervision: gamma-weighted loss at every loop iteration
  - U-Net skip connections between encoder/decoder loop halves
  - Gradient checkpointing: memory-efficient backprop through 16 loops
  - QAT: per-row percentile fake-quantization during training
  - relu² MLP, logit softcap, QK RMSNorm, q_gain (from baseline)

At dim=512, 16 loops: ~4.24M params, ~4.24MB int8 — 26.5% of 16MB budget.
Effective compute depth of a ~63M parameter standard transformer.

Thermal protection: checkpoints saved every SAVE_EVERY steps, auto-resumed.
"""

from __future__ import annotations
import copy, glob, io, math, os, subprocess, sys, time, uuid, zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as grad_ckpt

# ── Hyperparameters ───────────────────────────────────────────────────────────

class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE",  524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY",  500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    save_every      = int(os.environ.get("SAVE_EVERY",      500))

    iterations            = int(os.environ.get("ITERATIONS",       5000))
    warmdown_iters        = int(os.environ.get("WARMDOWN_ITERS",    600))
    warmup_steps          = int(os.environ.get("WARMUP_STEPS",      20))
    train_batch_tokens    = int(os.environ.get("TRAIN_BATCH_TOKENS", 65_536))
    train_seq_len         = int(os.environ.get("TRAIN_SEQ_LEN",     1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0.0))

    vocab_size    = int(os.environ.get("VOCAB_SIZE",     1024))
    model_dim     = int(os.environ.get("MODEL_DIM",       512))
    num_heads     = int(os.environ.get("NUM_HEADS",          8))
    num_kv_heads  = int(os.environ.get("NUM_KV_HEADS",       4))
    n_loops       = int(os.environ.get("N_LOOPS",           16))
    n_memory      = int(os.environ.get("N_MEMORY",          32))
    mlp_mult      = int(os.environ.get("MLP_MULT",           4))
    lora_rank     = int(os.environ.get("LORA_RANK",         16))
    rope_base     = float(os.environ.get("ROPE_BASE",    10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP",  30.0))
    deep_sup_gamma= float(os.environ.get("DEEP_SUP_GAMMA",  0.9))

    embed_lr     = float(os.environ.get("EMBED_LR",    0.05))
    matrix_lr    = float(os.environ.get("MATRIX_LR",   0.04))
    scalar_lr    = float(os.environ.get("SCALAR_LR",   0.04))
    muon_momentum= float(os.environ.get("MUON_MOMENTUM",  0.95))
    muon_backend_steps         = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS",   300))
    beta1        = float(os.environ.get("BETA1", 0.9))
    beta2        = float(os.environ.get("BETA2", 0.95))
    adam_eps     = float(os.environ.get("ADAM_EPS", 1e-8))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.1))

# ── Muon Optimizer (from modded-nanogpt / competition baseline) ───────────────

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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size  = dist.get_world_size() if distributed else 1
        rank        = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum, backend_steps, nesterov = (
                group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            )
            total = sum(int(p.numel()) for p in params)
            flat  = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr  = 0
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
                    flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                p.add_(flat[curr:curr + p.numel()].view_as(p).to(p.dtype), alpha=-lr)
                curr += p.numel()
        return loss

# ── Control tensor patterns (kept fp32, excluded from Muon) ──────────────────

CONTROL_TENSOR_NAME_PATTERNS = tuple(p for p in os.environ.get(
    "CONTROL_TENSOR_NAME_PATTERNS",
    "attn_scale,mlp_scale,resid_mix,q_gain,lambda_q1,lambda_k1,lambda_q2,lambda_k2,"
    "loop_embed,skip_weight,memory",
).split(",") if p)

INT8_KEEP_FLOAT_MAX_NUMEL    = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE  = torch.float16
INT8_PER_ROW_SCALE_DTYPE     = torch.float16
INT8_CLIP_PERCENTILE         = 99.99984
INT8_CLIP_Q                  = INT8_CLIP_PERCENTILE / 100.0

# ── Tokenizer-aware BPB (from competition baseline) ───────────────────────────

def build_sentencepiece_luts(sp, vocab_size: int, device: torch.device):
    sp_vocab = int(sp.vocab_size())
    table    = max(sp_vocab, vocab_size)
    base_np  = np.zeros((table,), dtype=np.int16)
    space_np = np.zeros((table,), dtype=np.bool_)
    bnd_np   = np.ones((table,),  dtype=np.bool_)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        bnd_np[tid] = False
        if sp.is_byte(tid): base_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"): space_np[tid] = True; piece = piece[1:]
        base_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_np,  dtype=torch.int16, device=device),
            torch.tensor(space_np, dtype=torch.bool,  device=device),
            torch.tensor(bnd_np,   dtype=torch.bool,  device=device))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

@torch.no_grad()
def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_lut, space_lut, bnd_lut):
    local_tok   = args.val_batch_size // (world_size * grad_accum_steps)
    local_seqs  = local_tok // args.train_seq_len
    total_seqs  = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start   = (total_seqs * rank) // world_size
    seq_end     = (total_seqs * (rank + 1)) // world_size
    loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    tok_count   = torch.zeros((), device=device, dtype=torch.float64)
    byte_count  = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    val_seq_limit = min(seq_end, seq_start + 4096)  # capped for T4 speed
    with torch.inference_mode():
        for s in range(seq_start, val_seq_limit, local_seqs):
            se  = min(s + local_seqs, seq_end)
            raw = val_tokens[s * args.train_seq_len:(se * args.train_seq_len + 1)]
            raw = raw.to(device=device, dtype=torch.int64, non_blocking=True)
            x   = raw[:-1].reshape(-1, args.train_seq_len)
            y   = raw[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y).detach()
            n = float(y.numel())
            loss_sum   += loss.to(torch.float64) * n
            tok_count  += n
            prev_ids    = x.reshape(-1)
            tgt_ids     = y.reshape(-1)
            tb = base_lut[tgt_ids].to(torch.int16)
            tb += (space_lut[tgt_ids] & ~bnd_lut[prev_ids]).to(torch.int16)
            byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    model.train()
    val_loss = float((loss_sum / tok_count).item())
    bpb      = float((loss_sum / tok_count / math.log(2.0) * tok_count / byte_count).item())
    return val_loss, bpb

# ── Quantization (from competition baseline) ──────────────────────────────────

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def _keep_float(name: str, t: Tensor, pt_dtypes: dict) -> Tensor:
    if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        pt_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def _quantize_tensor(t: Tensor):
    t32 = t.float()
    if t32.ndim == 2:
        clip = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty(t32.shape[0])
        cl   = torch.maximum(torch.minimum(t32, clip[:, None]), -clip[:, None])
        sc   = (clip / 127.0).clamp_min(1.0 / 127.0)
        q    = torch.clamp(torch.round(cl / sc[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, sc.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    sc   = torch.tensor(clip / 127.0 if clip > 0 else 1.0, dtype=torch.float32)
    q    = torch.clamp(torch.round(torch.clamp(t32, -clip, clip) / sc), -127, 127).to(torch.int8).contiguous()
    return q, sc

def quantize_state_dict_int8(sd: dict):
    quant, scales, dtypes, passthrough, pt_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count","num_tensors","num_float_tensors",
                           "num_nonfloat_tensors","baseline_tensor_bytes","int8_payload_bytes"), 0)
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"]         += int(t.numel())
        stats["num_tensors"]         += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = _keep_float(name, t, pt_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = _quantize_tensor(t)
        if s.ndim > 0: qmeta[name] = {"scheme": "per_row", "axis": 0}
        quant[name]  = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict = {"__quant_format__": "int8_clean_per_row_v1",
                 "quantized": quant, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta:   obj["qmeta"] = qmeta
    if pt_dtypes: obj["passthrough_orig_dtypes"] = pt_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict) -> dict:
    out  = {}
    qm   = obj.get("qmeta", {})
    ptod = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s     = obj["scales"][name]
        if qm.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        ot = t.detach().cpu().contiguous()
        if isinstance(ptod.get(name), str):
            ot = ot.to(getattr(torch, ptod[name])).contiguous()
        out[name] = ot
    return out

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    hdr = np.fromfile(file, dtype="<i4", count=256)
    if hdr.size != 256 or int(hdr[0]) != 20240520 or int(hdr[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    n   = int(hdr[2])
    tok = np.fromfile(file, dtype="<u2", count=n, offset=256 * 4)
    return torch.from_numpy(tok.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No shards: {pattern}")
        self.fi, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])
    def _next(self):
        self.fi = (self.fi + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.fi])
        self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks, rem = [], n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._next(); continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k; rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum: int):
        local = global_tokens // (self.world_size * grad_accum)
        span  = local + 1
        chunk = self.stream.take(span * self.world_size)
        s     = self.rank * span
        raw   = chunk[s:s + span].to(torch.int64)
        x     = raw[:-1].reshape(-1, seq_len)
        y     = raw[1:].reshape(-1,  seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ── Model ─────────────────────────────────────────────────────────────────────

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), b)

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__(); self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) \
               and p.dtype != torch.float32:
                p.data = p.data.float()

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._sl, self._cos, self._sin = 0, None, None
    def forward(self, sl: int, device, dtype):
        if self._cos is None or self._sl != sl or self._cos.device != device:
            t = torch.arange(sl, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos = f.cos()[None, None, :, :]; self._sin = f.sin()[None, None, :, :]
            self._sl = sl
        return self._cos.to(dtype=dtype), self._sin.to(dtype=dtype)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

def fake_quant_per_row(w: Tensor) -> Tensor:
    """Per-row percentile int8 QAT with straight-through estimator."""
    if w.ndim < 2 or w.numel() == 0: return w
    w32  = w.float()
    clip = torch.quantile(w32.abs(), INT8_CLIP_Q, dim=1)
    sc   = (clip / 127.0).clamp_min(1.0 / 127.0)
    wq   = (w32.clamp(-clip[:, None], clip[:, None]) / sc[:, None]).round().clamp(-127, 127) * sc[:, None]
    return w + (wq.to(w.dtype) - w).detach()

class SharedDiffAttention(nn.Module):
    """Differential Attention V2 + per-loop low-rank Q delta + GQA."""
    def __init__(self, dim: int, nh: int, nkv: int, n_loops: int,
                 n_mem: int, rope_base: float, rank: int):
        super().__init__()
        assert dim % nh == 0 and nh % nkv == 0
        self.nh, self.nkv, self.n_rep = nh, nkv, nh // nkv
        self.hd   = dim // nh
        self.nmem = n_mem

        self.c_q  = CastedLinear(dim, dim * 2,          bias=False)
        self.c_k  = CastedLinear(dim, nkv * self.hd * 2, bias=False)
        self.c_v  = CastedLinear(dim, nkv * self.hd,    bias=False)
        self.proj = CastedLinear(dim, dim,               bias=False)
        self.proj._zero_init = True

        li = 0.8 - 0.6 * math.exp(-0.3)
        self.lambda_init = li
        self.lambda_q1 = nn.Parameter(torch.randn(nh) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(nh) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(nh) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(nh) * 0.1)

        self.q_gain  = nn.Parameter(torch.ones(nh, dtype=torch.float32))
        self.delta_A = nn.Parameter(torch.zeros(n_loops, dim * 2, rank))
        self.delta_B = nn.Parameter(torch.zeros(n_loops, rank, dim))
        nn.init.normal_(self.delta_A, std=0.01)

        self.rotary = Rotary(self.hd, base=rope_base)

    def forward(self, x: Tensor, loop_idx: int) -> Tensor:
        B, T, C = x.shape
        H, KVH, D, NM = self.nh, self.nkv, self.hd, self.nmem
        sl = T - NM

        # Q with per-loop low-rank delta + QAT on effective weight
        wq  = self.c_q.weight + (self.delta_A[loop_idx] @ self.delta_B[loop_idx])
        wq  = fake_quant_per_row(wq)
        qp  = F.linear(x, wq.to(x.dtype))
        q1  = qp[..., :C].reshape(B, T, H,   D).transpose(1, 2)
        q2  = qp[..., C:].reshape(B, T, H,   D).transpose(1, 2)

        # K, V: fresh every loop from updated hidden state
        kv  = self.c_k(x)
        k1  = kv[..., :KVH*D].reshape(B, T, KVH, D).transpose(1, 2)
        k2  = kv[..., KVH*D:].reshape(B, T, KVH, D).transpose(1, 2)
        v   = self.c_v(x).reshape(B, T, KVH, D).transpose(1, 2)

        # QK RMSNorm before RoPE (baseline technique)
        q1 = F.rms_norm(q1, (D,)); q2 = F.rms_norm(q2, (D,))
        k1 = F.rms_norm(k1, (D,)); k2 = F.rms_norm(k2, (D,))

        # RoPE: sequence positions ONLY — memory tokens excluded
        cos, sin = self.rotary(sl, x.device, q1.dtype)
        def rope_seq(q, k):
            qm, qs = q[:, :, :NM], q[:, :, NM:]
            km, ks = k[:, :, :NM], k[:, :, NM:]
            return (torch.cat([qm, apply_rope(qs, cos, sin)], dim=2),
                    torch.cat([km, apply_rope(ks, cos, sin)], dim=2))
        q1, k1 = rope_seq(q1, k1)
        q2, k2 = rope_seq(q2, k2)

        # Q gain + GQA
        g   = self.q_gain.to(q1.dtype)[None, :, None, None]
        q1  = q1 * g; q2 = q2 * g
        k1  = k1.repeat_interleave(self.n_rep, dim=1)
        k2  = k2.repeat_interleave(self.n_rep, dim=1)
        v   = v.repeat_interleave(self.n_rep, dim=1)

        # Differential attention: cancel noise via subtraction
        lam = (torch.exp(self.lambda_q1) * torch.exp(self.lambda_k1)
             - torch.exp(self.lambda_q2) * torch.exp(self.lambda_k2)
             + self.lambda_init).to(q1.dtype)[None, :, None, None]
        a1  = F.scaled_dot_product_attention(q1, k1, v, is_causal=True)
        a2  = F.scaled_dot_product_attention(q2, k2, v, is_causal=True)
        out = (a1 - lam * a2).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class ReluSquaredMLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        h = dim * mlp_mult
        self.fc   = CastedLinear(dim, h,   bias=False)
        self.proj = CastedLinear(h,   dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())

class RecurrentBlock(nn.Module):
    """Shared block with all per-loop parameters."""
    def __init__(self, dim: int, nh: int, nkv: int, n_loops: int,
                 n_mem: int, mlp_mult: int, rope_base: float, rank: int):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = SharedDiffAttention(dim, nh, nkv, n_loops, n_mem, rope_base, rank)
        self.mlp  = ReluSquaredMLP(dim, mlp_mult)

        # Per-loop scale factors (learned attention/MLP gate per depth)
        self.attn_scale = nn.Parameter(torch.ones(n_loops, dim,  dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(n_loops, dim,  dtype=torch.float32))

        # resid_mix: learned interpolation between current x and original x0
        # [0] = weight on current state, [1] = weight on original input
        # Initialised to carry forward (mix[0]=1, mix[1]=0), learns to reset selectively
        self.resid_mix  = nn.Parameter(
            torch.stack([torch.ones(n_loops, dim), torch.zeros(n_loops, dim)]).float()
        )

        # Loop position embedding: activation-level depth signal
        # Complementary to weight-level low-rank delta
        self.loop_embed = nn.Embedding(n_loops, dim)
        nn.init.normal_(self.loop_embed.weight, std=0.02)

    def forward(self, x: Tensor, x0: Tensor, loop_idx: int) -> Tensor:
        # resid_mix: learned balance of current state vs original input (generalises input injection)
        mix = self.resid_mix[:, loop_idx, :].to(x.dtype)
        x   = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # Loop position embedding (activation-level specialisation)
        x = x + self.loop_embed.weight[loop_idx].to(x.dtype)[None, None, :]

        # Attention + MLP with per-loop scale
        a = self.attn(self.attn_norm(x), loop_idx)
        x = x + self.attn_scale[loop_idx].to(x.dtype)[None, None, :] * a
        x = x + self.mlp_scale[loop_idx].to(x.dtype)[None, None,  :] * self.mlp(self.mlp_norm(x))
        return x

class RLM(nn.Module):
    """
    Recurrent Language Model.
    Single shared block × N_LOOPS with deep supervision and U-Net skips.
    """
    def __init__(self, args: Hyperparameters):
        super().__init__()
        dim, nl, nm = args.model_dim, args.n_loops, args.n_memory
        self.n_loops, self.n_memory = nl, nm
        self.dim      = dim
        self.logit_softcap  = args.logit_softcap
        self.deep_sup_gamma = args.deep_sup_gamma

        self.n_enc  = nl // 2
        self.n_skip = min(self.n_enc, nl - self.n_enc)

        self.tok_emb = nn.Embedding(args.vocab_size, dim)
        nn.init.normal_(self.tok_emb.weight, std=0.005)

        self.memory = nn.Parameter(torch.randn(1, nm, dim) * 0.02)

        self.block  = RecurrentBlock(
            dim, args.num_heads, args.num_kv_heads, nl, nm,
            args.mlp_mult, args.rope_base, args.lora_rank
        )

        self.skip_weights = nn.Parameter(
            torch.ones(self.n_skip, dim, dtype=torch.float32)
            if self.n_skip > 0 else torch.zeros(0, dim, dtype=torch.float32)
        )
        self.final_norm = RMSNorm()

        # Zero-init output projections for stability at loop depth
        nn.init.zeros_(self.block.attn.proj.weight)
        nn.init.zeros_(self.block.mlp.proj.weight)

    def _logits(self, x: Tensor) -> Tensor:
        h = self.final_norm(x[:, self.n_memory:, :])
        l = F.linear(h.reshape(-1, self.dim), self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(l / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        B, T = input_ids.shape
        tok  = F.rms_norm(self.tok_emb(input_ids), (self.dim,))
        x    = torch.cat([self.memory.expand(B, -1, -1), tok], dim=1)
        x0   = x

        gamma      = self.deep_sup_gamma
        total_loss = torch.zeros((), device=input_ids.device)
        weight_sum = 0.0
        targets    = target_ids.reshape(-1)
        skips      = []

        for i in range(self.n_loops):
            # Gradient checkpointing: recompute activations during backward
            x = grad_ckpt(self.block, x, x0, i, use_reentrant=False)

            # U-Net: encoder half saves, decoder half consumes in reverse
            if i < self.n_enc:
                skips.append(x)
            elif self.n_skip > 0 and skips:
                di = i - self.n_enc
                if di < self.n_skip:
                    sw = self.skip_weights[di].to(x.dtype)[None, None, :]
                    x  = x + sw * skips[self.n_enc - 1 - di]

            # Deep supervision: weighted loss at every loop
            # Later loops weighted higher (gamma^(N-1-i)), but all contribute
            w           = gamma ** (self.n_loops - 1 - i)
            total_loss  = total_loss + w * F.cross_entropy(
                self._logits(x).float(), targets, reduction="mean"
            )
            weight_sum += w

        return total_loss / weight_sum

# ── Training ──────────────────────────────────────────────────────────────────

def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    global zeropower_via_newtonschulz5
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # ── Distributed + CUDA setup ──────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. For CPU testing use train_gpt_local.py")
    distributed  = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank         = int(os.environ.get("RANK",       "0"))
    world_size   = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank   = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale       = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                     enable_math_sdp, enable_mem_efficient_sdp)
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(True); enable_math_sdp(True)

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt" if master else None

    def log0(msg: str, console: bool = True) -> None:
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)

    # ── Seed + tokenizer ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Tokenizer vocab {sp.vocab_size()} != VOCAB_SIZE {args.vocab_size}")
    val_tokens          = load_validation_tokens(args.val_files, args.train_seq_len)
    base_lut, space_lut, bnd_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_tokens:{val_tokens.numel()-1} train_seq_len:{args.train_seq_len}")

    # ── Model ─────────────────────────────────────────────────────────────────
    base_model = RLM(args).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled  = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = (DDP(compiled, device_ids=[local_rank], broadcast_buffers=False)
                        if distributed else compiled)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} (~{n_params/1e6:.2f}M)")
    log0(f"n_loops:{args.n_loops} n_memory:{args.n_memory} dim:{args.model_dim} "
         f"seq_len:{args.train_seq_len}")

    # ── Optimizers ────────────────────────────────────────────────────────────
    block_named   = list(base_model.block.named_parameters())
    matrix_params = [p for n, p in block_named
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = ([p for n, p in block_named
                      if p.ndim != 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
                     + [base_model.skip_weights, base_model.memory, base_model.final_norm.eps
                        if hasattr(base_model.final_norm, 'eps') and
                        isinstance(base_model.final_norm.eps, nn.Parameter) else None])
    scalar_params = [p for p in scalar_params if p is not None]

    opt_embed  = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon   = Muon(matrix_params, lr=args.matrix_lr,
                      momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_embed, opt_muon, opt_scalar]

    log0(f"matrix_params:{sum(p.numel() for p in matrix_params):,}  "
         f"scalar_params:{sum(p.numel() for p in scalar_params):,}")

    # ── Checkpoint resume (thermal crash protection) ──────────────────────────
    ckpt_path = f"checkpoint_{args.run_id}.pt"
    start_step = 0
    training_time_ms = 0.0
    if os.path.exists(ckpt_path) and master:
        log0(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        base_model.load_state_dict(ckpt["model"])
        for opt, sd in zip(optimizers, ckpt["optimizers"]):
            opt.load_state_dict(sd)
        start_step       = ckpt["step"]
        training_time_ms = ckpt.get("training_time_ms", 0.0)
        log0(f"Resumed at step {start_step}")
    if distributed: dist.barrier()

    # ── Data loader ───────────────────────────────────────────────────────────
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad():
        for o in optimizers: o.zero_grad(set_to_none=True)

    max_wc_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_scale(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0: return 1.0
        if max_wc_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                   if ws <= step < args.iterations else 1.0
        step_ms   = elapsed_ms / max(step, 1)
        wd_ms     = args.warmdown_iters * step_ms
        rem_ms    = max(max_wc_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    # ── Warmup-then-restore ───────────────────────────────────────────────────
    if args.warmup_steps > 0 and start_step == 0:
        init_model = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opts  = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = (ms == grad_accum_steps - 1)
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast("cuda", torch.bfloat16):
                    (model(x, y) * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad()
        log0(f"warmup_complete:{args.warmup_steps} steps")
        base_model.load_state_dict(init_model, strict=True)
        for o, sd in zip(optimizers, init_opts): o.load_state_dict(sd)
        zero_grad()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── Main training loop ────────────────────────────────────────────────────
    stop_after: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(start_step, args.iterations + 1):
        last = step == args.iterations or (stop_after is not None and step >= stop_after)

        # Validation
        if ((args.val_loss_every > 0 and step % args.val_loss_every == 0 and step >= 200) or last):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                val_tokens, base_lut, space_lut, bnd_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vbpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        if last: break

        # Checkpoint save (thermal crash protection)
        if args.save_every > 0 and step > start_step and step % args.save_every == 0 and master:
            torch.cuda.synchronize()
            elapsed_now = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            torch.save({"step": step, "model": base_model.state_dict(),
                        "optimizers": [o.state_dict() for o in optimizers],
                        "training_time_ms": elapsed_now}, ckpt_path)
            log0(f"checkpoint_saved:step={step}")

        # LR update
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale      = lr_scale(step, elapsed_ms)
        for o in optimizers:
            for g in o.param_groups: g["lr"] = g["base_lr"] * scale

        # Muon momentum warmup
        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        mum  = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups: g["momentum"] = mum

        # Forward + backward with gradient accumulation
        zero_grad()
        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = (ms == grad_accum_steps - 1)
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        for o in optimizers: o.step()
        zero_grad()

        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step+1}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/(step+1):.2f}ms lr_scale:{scale:.4f}")

        reached = max_wc_ms is not None and approx_ms >= max_wc_ms
        if distributed and max_wc_ms is not None:
            rt = torch.tensor(int(reached), device=device)
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
            reached = bool(rt.item())
        if stop_after is None and reached:
            stop_after = step + 1

    log0(f"peak_memory:{torch.cuda.max_memory_allocated()//1024//1024}MiB "
         f"reserved:{torch.cuda.max_memory_reserved()//1024//1024}MiB")

    # ── Serialisation + roundtrip validation ──────────────────────────────────
    if master:
        torch.save(base_model.state_dict(), "final_model.pt")
        code_bytes  = len(code.encode("utf-8"))
        model_bytes = os.path.getsize("final_model.pt")
        log0(f"raw_model:{model_bytes} code:{code_bytes} total:{model_bytes+code_bytes}")

    quant_obj, qstats = quantize_state_dict_int8(base_model.state_dict())
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    if master:
        with open("final_model.int8.ptz", "wb") as f: f.write(blob)
        qbytes     = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio      = qstats["baseline_tensor_bytes"] / max(qstats["int8_payload_bytes"], 1)
        log0(f"int8_zlib:{qbytes} code:{code_bytes} total:{qbytes+code_bytes} "
             f"payload_ratio:{ratio:.2f}x budget_pct:{(qbytes+code_bytes)/16e6*100:.1f}%")

    if distributed: dist.barrier()
    with open("final_model.int8.ptz", "rb") as f: blob_disk = f.read()
    qs2 = torch.load(io.BytesIO(zlib.decompress(blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(qs2), strict=True)
    torch.cuda.synchronize(); te = time.perf_counter()
    qvl, qvbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                           val_tokens, base_lut, space_lut, bnd_lut)
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{qvl:.4f} val_bpb:{qvbpb:.4f} "
         f"eval_time:{1000.0*(time.perf_counter()-te):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvbpb:.8f}")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
