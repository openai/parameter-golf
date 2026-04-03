#!/usr/bin/env python3
"""
RAKI v5 CUDA — H100 için PyTorch versiyonu.
Tüm teknikler: Markov hybrid, stochastic depth, EMA, BigramHash eval, layer grad scaling.

Kullanım:
  RUN_ID=raki_v5 torchrun --standalone --nproc_per_node=1 train_raki_v5_cuda.py
"""
from __future__ import annotations
import glob, math, os, pickle, struct, sys, time, uuid, zlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sentencepiece as spm

# ==============================================================================
# CONFIG
# ==============================================================================
RAKI_POWER    = float(os.environ.get("RAKI_POWER", 0.25))
STOCH_DEPTH   = float(os.environ.get("STOCH_DEPTH_MAX", 0.12))
EMA_START     = float(os.environ.get("EMA_START", 0.90))
EMA_DECAY     = float(os.environ.get("EMA_DECAY", 0.995))
MUON_WD       = float(os.environ.get("MUON_WD", 0.04))
BH_EVAL_W     = float(os.environ.get("BH_EVAL_WEIGHT", 0.4))
WARMDOWN_FRAC = float(os.environ.get("WARMDOWN_FRAC", 0.15))
GRAD_SCALE_P  = float(os.environ.get("GRAD_SCALE_POWER", 1.5))

class HP:
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    iterations     = int(os.environ.get("ITERATIONS", 20_000))
    batch_tokens   = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum     = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    seq_len        = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    val_every      = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch      = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    log_every      = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    warmup         = int(os.environ.get("WARMUP_STEPS", 20))
    max_wall       = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    V    = int(os.environ.get("VOCAB_SIZE", 1024))
    L    = int(os.environ.get("NUM_LAYERS", 10))
    D    = int(os.environ.get("MODEL_DIM", 512))
    NH   = int(os.environ.get("NUM_HEADS", 8))
    NKV  = int(os.environ.get("NUM_KV_HEADS", 4))
    MULT = int(os.environ.get("MLP_MULT", 2))
    softcap   = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain   = float(os.environ.get("QK_GAIN_INIT", 1.5))
    embed_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    embed_lr  = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_mom  = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    out_dir = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self): return f"{self.data_path}/fineweb_train_*.bin"
    @property
    def val_files(self): return f"{self.data_path}/fineweb_val_*.bin"
    @property
    def micro(self): return self.batch_tokens // self.grad_accum
    @property
    def warmdown(self): return max(1, int(self.iterations * WARMDOWN_FRAC))

    def lr_mul(self, step, ms):
        wi = self.warmdown
        if self.max_wall <= 0:
            ws = max(self.iterations - wi, 0)
            return max((self.iterations - step) / max(wi, 1), 0.0) if step >= ws else 1.0
        sms = ms / max(step, 1)
        rms = max(1000 * self.max_wall - ms, 0.0)
        wms = wi * sms
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

CTRL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")

# ==============================================================================
# MARKOV TABLE
# ==============================================================================
class MarkovTable:
    def __init__(self, V, device="cpu"):
        self.V = V
        self.counts = np.zeros((V, V), dtype=np.float64)
        self.device = device

    def update(self, tokens):
        np.add.at(self.counts, (tokens[:-1], tokens[1:]), 1.0)

    def finalize(self):
        smoothed = self.counts + 0.01
        probs = smoothed / smoothed.sum(axis=1, keepdims=True)
        log_probs = np.log(probs)
        self.table_np = log_probs.astype(np.float16)
        self.table_t = torch.tensor(log_probs, dtype=torch.float16, device=self.device)
        # Entropy
        ent = -(probs * log_probs).sum(axis=1).astype(np.float32)
        emin, emax = ent.min(), ent.max()
        self.entropy_norm = (ent - emin) / (emax - emin) if emax > emin else np.full_like(ent, 0.5)
        del self.counts

    def hybrid_score(self, x_np, y_np):
        lp = self.table_np[x_np.ravel(), y_np.ravel()].astype(np.float32)
        surprise = -lp.reshape(x_np.shape)
        ent = self.entropy_norm[x_np.ravel()].reshape(x_np.shape)
        return (surprise * ent).mean(axis=1)

# ==============================================================================
# RAKI SCHEDULE
# ==============================================================================
def raki_state(progress):
    p = max(0.0, min(1.0, progress))
    if p <= 0.25:
        t = p / 0.25
        return {"phase": "ayik", "sw": RAKI_POWER * 0.1 * t, "sd": STOCH_DEPTH * 0.5}
    elif p <= 0.65:
        t = (p - 0.25) / 0.40
        return {"phase": "keyifli", "sw": RAKI_POWER * (0.1 + 0.9*t), "sd": STOCH_DEPTH * (0.5 + 0.5*t)}
    elif p <= 0.90:
        return {"phase": "kivam", "sw": RAKI_POWER, "sd": STOCH_DEPTH}
    else:
        t = (p - 0.90) / 0.10
        return {"phase": "ayilma", "sw": RAKI_POWER * (1.0 - 0.5*t), "sd": STOCH_DEPTH * (1.0 - 0.8*t)}

# ==============================================================================
# DATA
# ==============================================================================
def load_shard(path):
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520 and header[1] == 1
        n = int(header[2])
        tokens = np.frombuffer(f.read(n * 2), dtype=np.uint16)
    return tokens.astype(np.int32)

class TokenStream:
    def __init__(self, pattern):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"No files: {pattern}"
        self.fi, self.pos, self.epoch = 0, 0, 1
        self.tok = load_shard(self.files[0])
    def _next(self):
        self.fi = (self.fi + 1) % len(self.files)
        if self.fi == 0: self.epoch += 1
        self.tok = load_shard(self.files[self.fi]); self.pos = 0
    def take(self, n):
        parts, left = [], n
        while left > 0:
            if self.pos >= len(self.tok): self._next()
            k = min(left, len(self.tok) - self.pos)
            parts.append(self.tok[self.pos:self.pos+k])
            self.pos += k; left -= k
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

class Loader:
    def __init__(self, pattern, device):
        self.s = TokenStream(pattern)
        self.device = device
    def batch(self, bt, sl):
        u = (bt // sl) * sl
        c = self.s.take(u + 1)
        xn, yn = c[:-1].reshape(-1, sl), c[1:].reshape(-1, sl)
        x = torch.tensor(xn, dtype=torch.long, device=self.device)
        y = torch.tensor(yn, dtype=torch.long, device=self.device)
        return x, y, xn, yn

# ==============================================================================
# MODEL
# ==============================================================================
def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)

def precompute_rope(dim, max_len, base=10000.0, device="cpu"):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos_s = cos[:x.shape[-2], :d]
    sin_s = sin[:x.shape[-2], :d]
    return torch.cat([x1 * cos_s - x2 * sin_s, x2 * cos_s + x1 * sin_s], dim=-1)

class CastedLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.weight = nn.Parameter(nn.Linear(i, o, bias=False).weight.float())
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class Attn(nn.Module):
    def __init__(self, D, nh, nkv, cos, sin, qk_gain):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, D // nh
        self.cq = CastedLinear(D, D)
        self.ck = CastedLinear(D, nkv * self.hd)
        self.cv = CastedLinear(D, nkv * self.hd)
        self.proj = CastedLinear(D, D)
        self.q_gain = nn.Parameter(torch.ones(nh) * qk_gain)
        self.cos, self.sin = cos, sin

    def forward(self, x):
        B, T, D = x.shape
        q = self.cq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.ck(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.cv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q = apply_rope(rms_norm(q).to(x.dtype), self.cos, self.sin)
        k = apply_rope(rms_norm(k).to(x.dtype), self.cos, self.sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.hd**-0.5)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

class MLP(nn.Module):
    def __init__(self, D, mult):
        super().__init__()
        self.fc = CastedLinear(D, D * mult)
        self.proj = CastedLinear(D * mult, D)
    def forward(self, x):
        h = F.relu(self.fc(x))
        return self.proj(h * h)

class Block(nn.Module):
    def __init__(self, D, nh, nkv, mult, cos, sin, qk_gain):
        super().__init__()
        self.attn = Attn(D, nh, nkv, cos, sin, qk_gain)
        self.mlp = MLP(D, mult)
        self.attn_scale = nn.Parameter(torch.ones(D))
        self.mlp_scale = nn.Parameter(torch.ones(D))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(D), torch.zeros(D)]))
        # Zero init output projections
        nn.init.zeros_(self.attn.proj.weight)
        nn.init.zeros_(self.mlp.proj.weight)

    def forward(self, x, x0):
        m = self.resid_mix.to(x.dtype)
        x = m[0][None, None, :] * x + m[1][None, None, :] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(rms_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(rms_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, hp, device="cuda"):
        super().__init__()
        self.hp = hp
        self.softcap = hp.softcap
        self.tok_emb = nn.Embedding(hp.V, hp.D)
        nn.init.normal_(self.tok_emb.weight, std=hp.embed_std)

        cos, sin = precompute_rope(hp.D // hp.NH, hp.seq_len * 2, hp.rope_base, device)
        self.n_enc = hp.L // 2
        self.n_dec = hp.L - self.n_enc
        self.blocks = nn.ModuleList([
            Block(hp.D, hp.NH, hp.NKV, hp.MULT, cos, sin, hp.qk_gain)
            for _ in range(hp.L)
        ])
        self.skip_weights = nn.Parameter(torch.ones(min(self.n_enc, self.n_dec), hp.D))

    def forward(self, ids):
        x = rms_norm(self.tok_emb(ids).to(torch.bfloat16))
        x0, skips = x, []
        for i in range(self.n_enc):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.n_dec):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.n_enc + i](x, x0)
        return rms_norm(x)

    def logits(self, ids):
        h = self.forward(ids)
        raw = h @ self.tok_emb.weight.to(h.dtype).T
        return self.softcap * torch.tanh(raw / self.softcap)

    def loss(self, x, y, weights=None):
        logits = self.logits(x).view(-1, self.hp.V).float()
        tgt = y.view(-1)
        if weights is None:
            return F.cross_entropy(logits, tgt)
        per_tok = F.cross_entropy(logits, tgt, reduction="none")
        w = weights.view(-1)
        return (per_tok * w).sum() / w.sum()

    def loss_with_bias(self, x, y, bias_table):
        logits = self.logits(x)
        bias = bias_table[x].to(logits.dtype)
        logits = logits + BH_EVAL_W * bias
        return F.cross_entropy(logits.view(-1, self.hp.V).float(), y.view(-1))

# ==============================================================================
# MUON OPTIMIZER
# ==============================================================================
def zeropower_ns5(g, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float(); x = x / (x.norm() + eps)
    tr = x.shape[0] > x.shape[1]
    if tr: x = x.T
    for _ in range(steps):
        am = x @ x.T
        x = a * x + (b * am + c * (am @ am)) @ x
    if tr: x = x.T
    return x.to(g.dtype)

class MuonOptimizer:
    def __init__(self, model, hp):
        self.hp = hp
        self.embed_key = "tok_emb.weight"
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}

        self.matrix_keys = [n for n, p in params.items()
                           if "blocks." in n and p.ndim == 2
                           and not any(c in n for c in CTRL_PATTERNS)]
        self.scalar_keys = [n for n, p in params.items()
                           if n == "skip_weights" or ("blocks." in n
                           and (p.ndim < 2 or any(c in n for c in CTRL_PATTERNS)))]

        self.muon_bufs = {n: torch.zeros_like(params[n]) for n in self.matrix_keys}
        self.adam_embed = torch.optim.Adam([params[self.embed_key]], lr=hp.embed_lr, betas=(0.9, 0.95))
        self.adam_scalar = torch.optim.Adam([params[n] for n in self.scalar_keys], lr=hp.scalar_lr, betas=(0.9, 0.95))

    def step(self, model, step_num, lr_mul):
        hp = self.hp
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}

        # Muon momentum warmup
        if hp.muon_warmup_steps:
            t = min(step_num / hp.muon_warmup_steps, 1.0)
            mom = (1 - t) * hp.muon_warmup_start + t * hp.muon_mom
        else:
            mom = hp.muon_mom

        lr = hp.matrix_lr * lr_mul

        # Matrix params: Muon
        for n in self.matrix_keys:
            p = params[n]
            if p.grad is None: continue
            g = p.grad.data
            if MUON_WD > 0: g = g + MUON_WD * p.data
            buf = mom * self.muon_bufs[n] + g
            self.muon_bufs[n] = buf
            g_ortho = zeropower_ns5(g + mom * buf, hp.muon_steps)
            scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
            p.data -= lr * (g_ortho * scale).to(p.dtype)

        # Embed: Adam
        self.adam_embed.param_groups[0]["lr"] = hp.embed_lr * lr_mul
        self.adam_embed.step()

        # Scalars: Adam
        self.adam_scalar.param_groups[0]["lr"] = hp.scalar_lr * lr_mul
        self.adam_scalar.step()

# ==============================================================================
# EMA
# ==============================================================================
class EMATracker:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = None
        self.active = False
    def activate(self, model):
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}
        self.active = True
    def update(self, model):
        if not self.active: return
        d = self.decay
        for n, p in model.named_parameters():
            self.shadow[n] = d * self.shadow[n] + (1 - d) * p.data
    def apply(self, model):
        if self.shadow:
            for n, p in model.named_parameters():
                p.data.copy_(self.shadow[n])

# ==============================================================================
# GRADIENT OPS: Stochastic Depth + Layer Scaling
# ==============================================================================
def apply_grad_ops(model, progress, rs):
    sd_prob = rs["sd"]
    n_layers = model.hp.L
    n_enc = n_layers // 2
    uniformity = min(1.0, progress / 0.9)
    active = n_layers

    for i, block in enumerate(model.blocks):
        # Stochastic depth
        if sd_prob > 0.01 and i > 0 and i < n_layers - 1:
            if np.random.random() < sd_prob:
                for p in block.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                active -= 1
                continue

        # Layer gradient scaling
        if i < n_enc:
            depth = i / max(n_enc - 1, 1)
        else:
            depth = 1.0 - (i - n_enc) / max(n_layers - n_enc - 1, 1)
        early_s = max(0.3, depth ** (1.0 / GRAD_SCALE_P))
        scale = early_s * (1.0 - uniformity) + 1.0 * uniformity
        scale = max(0.3, min(1.5, scale))

        if abs(scale - 1.0) > 0.02:
            for p in block.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

    return active

# ==============================================================================
# QUANTIZATION
# ==============================================================================
def quant_int8(state_dict):
    Q, S, D, P, PD, QM = {}, {}, {}, {}, {}, {}
    stats = {"params": 0, "bytes": 0}
    for n, t in state_dict.items():
        arr = t.cpu().float().numpy()
        stats["params"] += arr.size
        if arr.dtype not in (np.float32, np.float16, np.float64):
            P[n] = np.ascontiguousarray(arr)
            stats["bytes"] += P[n].nbytes; continue
        if arr.size <= 65536:
            P[n] = np.ascontiguousarray(arr.astype(np.float16))
            PD[n] = "float32"
            stats["bytes"] += P[n].nbytes; continue
        if arr.ndim == 2:
            ca = np.quantile(np.abs(arr), 0.9999984, axis=1)
            s = np.maximum(ca / 127, 1 / 127).astype(np.float32)
            q = np.clip(np.round(np.clip(arr, -ca[:, None], ca[:, None]) / s[:, None]), -127, 127).astype(np.int8)
            QM[n] = {"scheme": "per_row", "axis": 0}
            S[n] = s.astype(np.float16)
        else:
            ca = float(np.quantile(np.abs(arr.ravel()), 0.9999984)) if arr.size else 0.0
            s = np.array(ca / 127 if ca > 0 else 1.0, dtype=np.float32)
            q = np.clip(np.round(np.clip(arr, -ca, ca) / s), -127, 127).astype(np.int8)
            S[n] = s
        Q[n] = np.ascontiguousarray(q)
        D[n] = "float32"
        stats["bytes"] += q.nbytes + S[n].nbytes
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": Q, "scales": S, "dtypes": D, "passthrough": P}
    if QM: obj["qmeta"] = QM
    if PD: obj["passthrough_orig_dtypes"] = PD
    return obj, stats

# ==============================================================================
# EVAL
# ==============================================================================
def build_luts(sp, V):
    sv = int(sp.vocab_size()); ts = max(sv, V)
    bb, hl, ib = np.zeros(ts, dtype=np.int16), np.zeros(ts, dtype=np.bool_), np.ones(ts, dtype=np.bool_)
    for t in range(sv):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
        ib[t] = False
        if sp.is_byte(t): bb[t] = 1; continue
        pc = sp.id_to_piece(t)
        if pc.startswith("\u2581"): hl[t] = True; pc = pc[1:]
        bb[t] = len(pc.encode("utf-8"))
    return bb, hl, ib

def load_val(pattern, sl):
    files = sorted(glob.glob(pattern))
    assert files, f"No val files: {pattern}"
    tok = np.concatenate([load_shard(f) for f in files])
    u = ((tok.size - 1) // sl) * sl
    return tok[:u + 1]

@torch.no_grad()
def eval_model(model, vt, hp, bb, hl, ib, markov=None, device="cuda"):
    model.eval()
    sl = hp.seq_len
    vbs = hp.val_batch // sl
    ts = (vt.size - 1) // sl
    ls, tt, tb = 0.0, 0.0, 0.0

    for s in range(0, ts, vbs):
        e = min(s + vbs, ts)
        c = vt[s * sl:e * sl + 1]
        xn, yn = c[:-1].reshape(-1, sl), c[1:].reshape(-1, sl)
        x = torch.tensor(xn, dtype=torch.long, device=device)
        y = torch.tensor(yn, dtype=torch.long, device=device)

        if markov is not None:
            l = model.loss_with_bias(x, y, markov.table_t)
        else:
            l = model.loss(x, y)

        n = float(y.numel())
        ls += l.item() * n
        bn = bb[yn.ravel()].astype(np.int16, copy=True)
        bn += (hl[yn.ravel()] & ~ib[xn.ravel()]).astype(np.int16)
        tt += n; tb += float(bn.astype(np.float64).sum())

    model.train()
    vl = ls / tt
    return vl, (vl / math.log(2)) * (tt / tb)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # Setup
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(1337)

    hp = HP()
    is_main = rank == 0
    Path(hp.out_dir).mkdir(parents=True, exist_ok=True)
    logf = Path(hp.out_dir) / f"{hp.run_id}.txt"

    def log(m):
        if is_main:
            print(m)
            with logf.open("a") as f: print(m, file=f)

    log("=" * 60)
    log(" RAKI v5 CUDA — H100")
    log("=" * 60)

    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    vt = load_val(hp.val_files, hp.seq_len)
    bb, hl, ib = build_luts(sp, hp.V)

    # Markov
    log("Markov tablosu...")
    t0 = time.time()
    mk = MarkovTable(hp.V, device)
    for sf in sorted(glob.glob(hp.train_files)):
        mk.update(load_shard(sf))
    mk.finalize()
    log(f"  Hazir: {time.time()-t0:.1f}s")

    # Model
    model = GPT(hp, device).to(device)
    model = torch.compile(model)
    opt = MuonOptimizer(model, hp)
    ema = EMATracker(model, EMA_DECAY)
    np_ = sum(p.numel() for p in model.parameters())
    log(f"  Model: {np_:,} params, {hp.L}L {hp.D}D")
    log(f"  Raki power={RAKI_POWER} SD={STOCH_DEPTH} EMA={EMA_DECAY}")
    log(f"  BH eval={BH_EVAL_W} MuonWD={MUON_WD}")

    # Data
    loader = Loader(hp.train_files, device)

    # Training
    log(f"\nTraining: {hp.iterations} iters, {hp.max_wall}s cap")
    log("-" * 60)

    tms = 0.0
    cap = 1000 * hp.max_wall if hp.max_wall > 0 else None
    torch.cuda.synchronize()

    for step in range(1, hp.iterations + 1):
        t0 = time.perf_counter()
        progress = min(tms / cap, 1.0) if cap else step / hp.iterations
        rs = raki_state(progress)
        sw = rs["sw"]

        # EMA
        if progress >= EMA_START and not ema.active:
            ema.activate(model)
            log(f"  >>> EMA ON (step {step})")

        # Grad accum
        total_loss = 0.0
        model.zero_grad()

        for micro_step in range(hp.grad_accum):
            x, y, xn, yn = loader.batch(hp.micro // world, hp.seq_len)

            if sw > 0.02:
                surp = mk.hybrid_score(xn, yn)
                smin, smax = surp.min(), surp.max()
                ns = (surp - smin) / (smax - smin) if smax > smin else np.full_like(surp, 0.5)
                tw = torch.tensor(
                    np.broadcast_to((1.0 + sw * ns)[:, None], xn.shape).copy(),
                    dtype=torch.float32, device=device
                )
                loss = model.loss(x, y, weights=tw) / hp.grad_accum
            else:
                loss = model.loss(x, y) / hp.grad_accum

            loss.backward()
            total_loss += loss.item() * hp.grad_accum

        # Gradient ops
        active = apply_grad_ops(model, progress, rs)

        # Optimizer
        lr_m = hp.lr_mul(step, tms)
        opt.step(model, step, lr_m)
        model.zero_grad()
        ema.update(model)

        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000
        tms += step_ms

        if is_main and (step == 1 or step % hp.log_every == 0 or step == hp.iterations):
            tps = hp.batch_tokens / (step_ms / 1000)
            log(f"  {step:5d}/{hp.iterations} loss:{total_loss:.4f} "
                f"{rs['phase']:8s} sw:{sw:.2f} sd:{rs['sd']:.2f} "
                f"lr:{lr_m:.3f} ema:{'ON' if ema.active else '--'} "
                f"L:{active}/{hp.L} {tps:.0f}t/s {tms/1000:.1f}s")

        if hp.val_every > 0 and step % hp.val_every == 0 and is_main:
            vl, vb = eval_model(model, vt, hp, bb, hl, ib, device=device)
            log(f"  >>> val_loss:{vl:.4f} val_bpb:{vb:.4f}")

        if cap and tms >= cap:
            log(f"  Wallclock {hp.max_wall}s doldu."); break

    # EMA
    if ema.active:
        log("EMA uygulaniyor...")
        ema.apply(model)

    # Eval
    if is_main:
        log(f"\n{'='*60}")
        log("FINAL EVALUATION")

        vl, vb = eval_model(model, vt, hp, bb, hl, ib, device=device)
        log(f"  Plain    → val_loss:{vl:.4f} val_bpb:{vb:.4f}")

        vl2, vb2 = eval_model(model, vt, hp, bb, hl, ib, markov=mk, device=device)
        log(f"  +Bigram  → val_loss:{vl2:.4f} val_bpb:{vb2:.4f}")

        best = min(vb, vb2)
        log(f"  BEST     → val_bpb:{best:.4f}")

        # Compress
        sd = {n: p.data for n, p in model.named_parameters()}
        qo, st = quant_int8(sd)
        qo["bigram_table"] = mk.table_np
        comp = zlib.compress(pickle.dumps(qo), level=9)
        cb = len(Path(__file__).read_text().encode())
        total = cb + len(comp)
        log(f"\n  Model: {len(comp)/1e6:.2f}MB Code: {cb/1e6:.2f}MB TOTAL: {total/1e6:.2f}MB "
            f"{'OK' if total < 16e6 else 'OVER!'}")
        log(f"  Params: {st['params']:,}")

        mp = Path(hp.out_dir) / f"{hp.run_id}_model.pkl.zlib"
        mp.write_bytes(comp)
        log(f"  Saved: {mp}")
        log("=" * 60)

    if world > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
