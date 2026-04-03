#!/usr/bin/env python3
"""
RAKI v7 CUDA — H100.
Fixed: dtype casting, loss display, warmdown LR, torch.autocast, torch.compile.
Unique: Hybrid Markov curriculum, Stochastic Depth, BigramHash eval-only, EMA.

  RUN_ID=raki_v7 torchrun --standalone --nproc_per_node=1 train_gpt.py
"""
import glob, math, os, pickle, sys, time, uuid, zlib
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import sentencepiece as spm

# ==============================================================================
# CONFIG — Rakı-specific
# ==============================================================================
RAKI_POWER  = float(os.environ.get("RAKI_POWER", "0.20"))
STOCH_DEPTH = float(os.environ.get("STOCH_DEPTH", "0.08"))
EMA_START   = float(os.environ.get("EMA_START", "0.85"))
EMA_DECAY   = float(os.environ.get("EMA_DECAY", "0.995"))
BH_WEIGHT   = float(os.environ.get("BH_WEIGHT", "0.3"))
WD_FRAC     = float(os.environ.get("WD_FRAC", "0.15"))

# ==============================================================================
# HYPERPARAMETERS — matches baseline defaults
# ==============================================================================
run_id      = os.environ.get("RUN_ID", str(uuid.uuid4()))
seed        = int(os.environ.get("SEED", "1337"))
data_path   = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
tok_path    = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
ITERS       = int(os.environ.get("ITERATIONS", "20000"))
BATCH_TOK   = int(os.environ.get("TRAIN_BATCH_TOKENS", "524288"))
GACC        = int(os.environ.get("GRAD_ACCUM_STEPS", "8"))
SEQ         = int(os.environ.get("TRAIN_SEQ_LEN", "1024"))
LOG_EVERY   = int(os.environ.get("TRAIN_LOG_EVERY", "50"))
WARMUP      = int(os.environ.get("WARMUP_STEPS", "20"))
MAX_WALL    = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "600"))
VAL_EVERY   = int(os.environ.get("VAL_LOSS_EVERY", "0"))
VAL_BATCH   = int(os.environ.get("VAL_BATCH_SIZE", "524288"))
V           = int(os.environ.get("VOCAB_SIZE", "1024"))
L           = int(os.environ.get("NUM_LAYERS", "10"))
D           = int(os.environ.get("MODEL_DIM", "512"))
NH          = int(os.environ.get("NUM_HEADS", "8"))
NKV         = int(os.environ.get("NUM_KV_HEADS", "4"))
MULT        = int(os.environ.get("MLP_MULT", "2"))
SOFTCAP     = float(os.environ.get("LOGIT_SOFTCAP", "30.0"))
ROPE_BASE   = float(os.environ.get("ROPE_BASE", "10000.0"))
QK_GAIN     = float(os.environ.get("QK_GAIN_INIT", "1.5"))
EMBED_STD   = float(os.environ.get("TIED_EMBED_INIT_STD", "0.005"))
EMBED_LR    = float(os.environ.get("TIED_EMBED_LR", "0.05"))
MATRIX_LR   = float(os.environ.get("MATRIX_LR", "0.04"))
SCALAR_LR   = float(os.environ.get("SCALAR_LR", "0.04"))
MUON_MOM    = float(os.environ.get("MUON_MOMENTUM", "0.95"))
MUON_STEPS  = int(os.environ.get("MUON_BACKEND_STEPS", "5"))
MUON_WS     = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", "0.85"))
MUON_WN     = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", "500"))
MUON_WD     = float(os.environ.get("MUON_WD", "0.04"))
OUT_DIR     = os.environ.get("OUT_DIR", "logs")
MICRO       = BATCH_TOK // GACC
HD          = D // NH
CTRL        = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")
train_pat   = f"{data_path}/fineweb_train_*.bin"
val_pat     = f"{data_path}/fineweb_val_*.bin"

# ==============================================================================
# FIX #3: LR warmdown uses WALLCLOCK FRACTION, not step count
# v6 bug: WARMDOWN=3000 steps * 338ms = 1014s > 600s total → LR never reaches 1.0
# v7 fix: LR = 1.0 for first (1-WD_FRAC) of wall time, then linear decay
# ==============================================================================
def lr_mul(step, ms):
    if MAX_WALL > 0:
        total_ms = 1000.0 * MAX_WALL
        wd_ms = total_ms * WD_FRAC  # last 15% of wall time
        rms = total_ms - ms          # remaining ms
        if rms <= wd_ms:
            return max(rms / wd_ms, 0.0)
        return 1.0
    else:
        # no wallclock cap — use iteration-based warmdown
        wd_iters = max(1, int(ITERS * WD_FRAC))
        ws = max(ITERS - wd_iters, 0)
        if step >= ws:
            return max((ITERS - step) / max(wd_iters, 1), 0.0)
        return 1.0

# ==============================================================================
# DATA
# ==============================================================================
def load_shard(p):
    with open(p, "rb") as f:
        h = np.frombuffer(f.read(1024), dtype=np.int32)
        assert h[0] == 20240520
        return np.frombuffer(f.read(int(h[2]) * 2), dtype=np.uint16).astype(np.int32)

class Loader:
    def __init__(self, pat, dev):
        self.files = sorted(glob.glob(pat))
        assert self.files
        self.fi = self.pos = 0
        self.tok = load_shard(self.files[0])
        self.dev = dev
    def _nxt(self):
        self.fi = (self.fi + 1) % len(self.files)
        self.tok = load_shard(self.files[self.fi]); self.pos = 0
    def take(self, n):
        parts, left = [], n
        while left > 0:
            if self.pos >= len(self.tok): self._nxt()
            k = min(left, len(self.tok) - self.pos)
            parts.append(self.tok[self.pos:self.pos+k])
            self.pos += k; left -= k
        return parts[0] if len(parts) == 1 else np.concatenate(parts)
    def batch(self, bt, sl):
        u = (bt // sl) * sl
        c = self.take(u + 1)
        xn, yn = c[:-1].reshape(-1, sl), c[1:].reshape(-1, sl)
        return (torch.tensor(xn, dtype=torch.long, device=self.dev),
                torch.tensor(yn, dtype=torch.long, device=self.dev), xn, yn)

# ==============================================================================
# MARKOV — hybrid entropy × surprise scoring
# ==============================================================================
class Markov:
    def __init__(self, pat, vocab, dev):
        shards = sorted(glob.glob(pat))
        counts = np.zeros((vocab, vocab), dtype=np.float64)
        tok = load_shard(shards[0])  # 1 shard only = fast
        np.add.at(counts, (tok[:-1], tok[1:]), 1.0)
        sm = counts + 0.01
        probs = sm / sm.sum(axis=1, keepdims=True)
        lp = np.log(probs)
        self.lp16 = lp.astype(np.float16)
        self.bias = torch.tensor(lp, dtype=torch.float32, device=dev)
        # Entropy per prev token
        ent = -(probs * lp).sum(axis=1).astype(np.float32)
        mn, mx = ent.min(), ent.max()
        self.ent_n = (ent - mn) / (mx - mn) if mx > mn else np.full_like(ent, 0.5)

    def hybrid(self, xn, yn):
        """Returns (batch_size,) hybrid score = entropy × surprise."""
        s = -self.lp16[xn.ravel(), yn.ravel()].astype(np.float32).reshape(xn.shape)
        e = self.ent_n[xn.ravel()].reshape(xn.shape)
        return (s * e).mean(axis=1)

# ==============================================================================
# RAKI SCHEDULE — 4-phase confidence curve
# ==============================================================================
def raki_schedule(progress):
    """Returns (phase_name, surprise_weight, stochastic_depth_prob)."""
    p = max(0.0, min(1.0, progress))
    if p <= 0.25:
        t = p / 0.25
        return "ayik", RAKI_POWER * 0.1 * t, STOCH_DEPTH * 0.5
    elif p <= 0.65:
        t = (p - 0.25) / 0.40
        return "keyifli", RAKI_POWER * (0.1 + 0.9*t), STOCH_DEPTH * (0.5 + 0.5*t)
    elif p <= 0.90:
        return "kivam", RAKI_POWER, STOCH_DEPTH
    else:
        t = (p - 0.90) / 0.10
        return "ayilma", RAKI_POWER*(1-0.5*t), STOCH_DEPTH*(1-0.8*t)

# ==============================================================================
# FIX #1: CastedLinear — weights stay float32, auto-cast to input dtype
# v6 bug: F.linear(x_bf16, weight_fp32) → crash
# v7 fix: weight.to(x.dtype) in every forward, same as baseline
# ==============================================================================
class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)

def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)

# ==============================================================================
# MODEL — U-Net GPT with proper dtype handling
# ==============================================================================
class GPT(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.emb = nn.Embedding(V, D, device=dev)
        nn.init.normal_(self.emb.weight, std=EMBED_STD)
        # RoPE
        inv = 1.0 / (ROPE_BASE ** (torch.arange(0, HD, 2, device=dev).float() / HD))
        t = torch.arange(SEQ * 2, device=dev).float()
        fr = torch.outer(t, inv)
        self.register_buffer("rope_cos", torch.cos(fr))
        self.register_buffer("rope_sin", torch.sin(fr))
        # U-Net structure
        n_enc = L // 2
        self.n_enc = n_enc
        self.n_dec = L - n_enc
        self.skip_w = nn.Parameter(torch.ones(min(n_enc, self.n_dec), D, device=dev))
        # FIX #1: Use CastedLinear instead of raw ParameterList
        self.q_lin = nn.ModuleList([CastedLinear(D, D, bias=False, device=dev) for _ in range(L)])
        self.k_lin = nn.ModuleList([CastedLinear(D, NKV*HD, bias=False, device=dev) for _ in range(L)])
        self.v_lin = nn.ModuleList([CastedLinear(D, NKV*HD, bias=False, device=dev) for _ in range(L)])
        self.o_lin = nn.ModuleList([CastedLinear(D, D, bias=False, device=dev) for _ in range(L)])
        self.fc1   = nn.ModuleList([CastedLinear(D, D*MULT, bias=False, device=dev) for _ in range(L)])
        self.fc2   = nn.ModuleList([CastedLinear(D*MULT, D, bias=False, device=dev) for _ in range(L)])
        # Scale/control params
        self.a_sc = nn.ParameterList([nn.Parameter(torch.ones(D, device=dev)) for _ in range(L)])
        self.m_sc = nn.ParameterList([nn.Parameter(torch.ones(D, device=dev)) for _ in range(L)])
        self.r_mix = nn.ParameterList([nn.Parameter(torch.stack([torch.ones(D,device=dev), torch.zeros(D,device=dev)])) for _ in range(L)])
        self.qk_g = nn.ParameterList([nn.Parameter(torch.ones(NH, device=dev) * QK_GAIN) for _ in range(L)])
        # Init: zero output projections, kaiming for rest
        for i in range(L):
            nn.init.kaiming_normal_(self.q_lin[i].weight)
            nn.init.kaiming_normal_(self.k_lin[i].weight)
            nn.init.kaiming_normal_(self.v_lin[i].weight)
            nn.init.zeros_(self.o_lin[i].weight)
            nn.init.kaiming_normal_(self.fc1[i].weight)
            nn.init.zeros_(self.fc2[i].weight)

    def _rope(self, x):
        T = x.shape[2]
        d = x.shape[-1] // 2
        c, s = self.rope_cos[:T, :d], self.rope_sin[:T, :d]
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1)

    def _attn(self, x, i):
        B, T, _ = x.shape
        xn = rms_norm(x)
        q = self.q_lin[i](xn).view(B, T, NH, HD).transpose(1, 2)
        k = self.k_lin[i](xn).view(B, T, NKV, HD).transpose(1, 2)
        v = self.v_lin[i](xn).view(B, T, NKV, HD).transpose(1, 2)
        q = self._rope(rms_norm(q))
        k = self._rope(rms_norm(k))
        q = q * self.qk_g[i][None, :, None, None]
        # GQA expand
        reps = NH // NKV
        if reps > 1:
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=HD**-0.5)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.o_lin[i](y)

    def _mlp(self, x, i):
        xn = rms_norm(x)
        h = F.relu(self.fc1[i](xn))
        return self.fc2[i](h * h)  # relu²

    def _block(self, x, x0, i):
        m = self.r_mix[i]
        x = m[0][None, None, :] * x + m[1][None, None, :] * x0
        x = x + self.a_sc[i][None, None, :] * self._attn(x, i)
        x = x + self.m_sc[i][None, None, :] * self._mlp(x, i)
        return x

    def forward(self, ids):
        x = rms_norm(self.emb(ids))
        x0, skips = x, []
        for i in range(self.n_enc):
            x = self._block(x, x0, i); skips.append(x)
        for i in range(self.n_dec):
            if i < len(skips) and skips:
                x = x + self.skip_w[min(i, self.skip_w.shape[0]-1)][None, None, :].to(x.dtype) * skips.pop()
            x = self._block(x, x0, self.n_enc + i)
        return rms_norm(x)

    def logits(self, ids):
        h = self.forward(ids)
        raw = h @ self.emb.weight.to(h.dtype).T
        return SOFTCAP * torch.tanh(raw / SOFTCAP)

    def loss_fn(self, x, y, w=None):
        lg = self.logits(x).reshape(-1, V).float()
        tgt = y.reshape(-1)
        if w is None:
            return F.cross_entropy(lg, tgt)
        pt = F.cross_entropy(lg, tgt, reduction="none")
        wf = w.reshape(-1)
        return (pt * wf).sum() / wf.sum()

    def loss_biased(self, x, y, bias_table):
        lg = self.logits(x)
        b = bias_table[x].to(lg.dtype) * BH_WEIGHT
        return F.cross_entropy((lg + b).reshape(-1, V).float(), y.reshape(-1))

# ==============================================================================
# MUON OPTIMIZER — same as baseline
# ==============================================================================
def ns5(g, steps=5, eps=1e-7):
    x = g.bfloat16(); x = x / (x.norm() + eps)
    tr = x.shape[0] > x.shape[1]
    if tr: x = x.T
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        am = x @ x.T
        x = a*x + (b*am + c*(am@am)) @ x
    if tr: x = x.T
    return x

class Optimizer:
    def __init__(self, model):
        self.bufs = {}
        self.mat_keys, self.sca_keys = [], []
        pdict = dict(model.named_parameters())
        for n, p in pdict.items():
            if n == "emb.weight": continue
            # CastedLinear weights have paths like "q_lin.0.weight"
            if ".weight" in n and p.ndim == 2 and p.numel() > 65536:
                self.mat_keys.append(n)
                self.bufs[n] = torch.zeros_like(p)
            elif n != "skip_w" and p.ndim < 2:
                self.sca_keys.append(n)
            elif n == "skip_w":
                self.sca_keys.append(n)
        self.adam_e = torch.optim.Adam([pdict["emb.weight"]],
                                       lr=EMBED_LR, betas=(0.9, 0.95))
        sca_params = [pdict[n] for n in self.sca_keys]
        self.adam_s = torch.optim.Adam(sca_params, lr=SCALAR_LR, betas=(0.9, 0.95))

    def step(self, model, step_n, lrm):
        pdict = dict(model.named_parameters())
        if MUON_WN > 0:
            t = min(step_n / MUON_WN, 1.0)
            mom = (1-t)*MUON_WS + t*MUON_MOM
        else:
            mom = MUON_MOM
        lr = MATRIX_LR * lrm
        for n in self.mat_keys:
            p = pdict[n]
            if p.grad is None: continue
            g = p.grad.data
            if MUON_WD > 0: g = g + MUON_WD * p.data
            buf = mom * self.bufs[n] + g
            self.bufs[n] = buf
            go = ns5(g + mom * buf, MUON_STEPS)
            sc = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
            p.data -= lr * (go * sc).to(p.dtype)
        self.adam_e.param_groups[0]["lr"] = EMBED_LR * lrm
        self.adam_e.step()
        self.adam_s.param_groups[0]["lr"] = SCALAR_LR * lrm
        self.adam_s.step()

# ==============================================================================
# EMA
# ==============================================================================
class EMA:
    def __init__(self):
        self.shadow = None; self.on = False
    def start(self, model):
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}
        self.on = True
    def update(self, model):
        if not self.on: return
        for n, p in model.named_parameters():
            self.shadow[n].lerp_(p.data, 1 - EMA_DECAY)
    def apply(self, model):
        if self.shadow:
            for n, p in model.named_parameters():
                p.data.copy_(self.shadow[n])

# ==============================================================================
# STOCHASTIC DEPTH — gradient-level, skip first and last layer
# ==============================================================================
def apply_stochastic_depth(model, sd_prob):
    if sd_prob < 0.01:
        return L
    active = L
    for i in range(1, L - 1):  # skip first and last
        if np.random.random() < sd_prob:
            # Zero gradients for this layer's linear weights
            for mod_name in ("q_lin", "k_lin", "v_lin", "o_lin", "fc1", "fc2"):
                mod_list = getattr(model, mod_name)
                if i < len(mod_list) and mod_list[i].weight.grad is not None:
                    mod_list[i].weight.grad.zero_()
            active -= 1
    return active

# ==============================================================================
# EVAL + QUANT + LUTS — same as baseline
# ==============================================================================
def build_luts(sp, V_):
    sv = int(sp.vocab_size()); ts = max(sv, V_)
    bb = np.zeros(ts, dtype=np.int16)
    hl = np.zeros(ts, dtype=np.bool_)
    ib = np.ones(ts, dtype=np.bool_)
    for t in range(sv):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
        ib[t] = False
        if sp.is_byte(t): bb[t] = 1; continue
        pc = sp.id_to_piece(t)
        if pc.startswith("\u2581"): hl[t] = True; pc = pc[1:]
        bb[t] = len(pc.encode("utf-8"))
    return bb, hl, ib

def load_val(pat, sl):
    files = sorted(glob.glob(pat))
    assert files
    tok = np.concatenate([load_shard(f) for f in files])
    u = ((tok.size - 1) // sl) * sl
    return tok[:u + 1]

@torch.no_grad()
def evaluate(model, vt, bb, hl, ib, dev, markov=None):
    model.eval()
    vbs = VAL_BATCH // SEQ
    ts = (vt.size - 1) // SEQ
    ls, tt, tb = 0.0, 0.0, 0.0
    for s in range(0, ts, vbs):
        e = min(s + vbs, ts)
        c = vt[s*SEQ:e*SEQ+1]
        xn, yn = c[:-1].reshape(-1, SEQ), c[1:].reshape(-1, SEQ)
        x = torch.tensor(xn, dtype=torch.long, device=dev)
        y = torch.tensor(yn, dtype=torch.long, device=dev)
        # FIX #4: use autocast for eval too
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            if markov is not None:
                l = model.loss_biased(x, y, markov.bias)
            else:
                l = model.loss_fn(x, y)
        n = float(y.numel())
        ls += l.item() * n
        bn = bb[yn.ravel()].astype(np.int16, copy=True)
        bn += (hl[yn.ravel()] & ~ib[xn.ravel()]).astype(np.int16)
        tt += n; tb += float(bn.astype(np.float64).sum())
    model.train()
    vl = ls / tt
    return vl, (vl / math.log(2)) * (tt / tb)

def quant_int8(sd):
    Q, S, D_, P, PD, QM = {}, {}, {}, {}, {}, {}
    st = {"params": 0, "bytes": 0}
    for n, t in sd.items():
        a = t.cpu().float().numpy()
        st["params"] += a.size
        if a.size <= 65536:
            P[n] = a.astype(np.float16); PD[n] = "float32"
            st["bytes"] += P[n].nbytes; continue
        if a.ndim == 2:
            ca = np.quantile(np.abs(a), 0.9999984, axis=1)
            s = np.maximum(ca/127, 1/127).astype(np.float32)
            q = np.clip(np.round(np.clip(a,-ca[:,None],ca[:,None])/s[:,None]),-127,127).astype(np.int8)
            QM[n] = {"scheme":"per_row"}; S[n] = s.astype(np.float16)
        else:
            ca = float(np.quantile(np.abs(a.ravel()), 0.9999984)) if a.size else 0.0
            s = np.array(ca/127 if ca>0 else 1.0, dtype=np.float32)
            q = np.clip(np.round(np.clip(a,-ca,ca)/s),-127,127).astype(np.int8); S[n]=s
        Q[n] = q; D_[n] = "float32"; st["bytes"] += q.nbytes + S[n].nbytes
    obj = {"__quant_format__":"int8_clean_per_row_v1","quantized":Q,"scales":S,"dtypes":D_,"passthrough":P}
    if QM: obj["qmeta"] = QM
    if PD: obj["passthrough_orig_dtypes"] = PD
    return obj, st

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1: dist.init_process_group("nccl")
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.manual_seed(seed)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    logf = Path(OUT_DIR) / f"{run_id}.txt"
    main_proc = rank == 0

    def log(m):
        if main_proc:
            print(m, flush=True)
            with logf.open("a") as f: print(m, file=f)

    log("=" * 60)
    log(" RAKI v7 CUDA — H100")
    log("=" * 60)

    sp = spm.SentencePieceProcessor(model_file=tok_path)
    vt = load_val(val_pat, SEQ)
    bb, hl, ib = build_luts(sp, V)

    # Markov (1 shard = ~2s)
    log("Markov...")
    t0 = time.time()
    mk = Markov(train_pat, V, dev)
    log(f"  OK: {time.time()-t0:.1f}s")

    # Model
    model = GPT(dev)
    np_ = sum(p.numel() for p in model.parameters())
    log(f"  Model: {np_:,} params, {L}L {D}D")
    log(f"  Raki={RAKI_POWER} SD={STOCH_DEPTH} EMA={EMA_DECAY} WD={MUON_WD}")
    log(f"  BH_eval={BH_WEIGHT} Warmdown={WD_FRAC*100:.0f}% of wallclock")

    # Create optimizer FIRST (needs raw parameter references)
    opt = Optimizer(model)
    ema = EMA()

    # FIX #5: torch.compile AFTER optimizer — CastedLinear is a proper nn.Module
    try:
        model = torch.compile(model)
        log("  torch.compile: ON")
    except Exception as e:
        log(f"  torch.compile: OFF ({e})")
    loader = Loader(train_pat, dev)

    log(f"\nTraining: {ITERS} iters, {MAX_WALL}s cap")
    log("-" * 60)

    tms = 0.0
    cap = 1000 * MAX_WALL if MAX_WALL > 0 else None

    for step in range(1, ITERS + 1):
        t0 = time.perf_counter()
        progress = min(tms / cap, 1.0) if cap else step / ITERS
        phase, sw, sd = raki_schedule(progress)

        if progress >= EMA_START and not ema.on:
            ema.start(model)
            log(f"  >>> EMA ON (step {step})")

        model.zero_grad()
        total_loss = 0.0

        for _ in range(GACC):
            x, y, xn, yn = loader.batch(MICRO // world, SEQ)

            # FIX #4: torch.autocast for mixed precision
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                if sw > 0.02:
                    sc = mk.hybrid(xn, yn)
                    mn, mx = sc.min(), sc.max()
                    ns = (sc - mn) / (mx - mn) if mx > mn else np.full_like(sc, 0.5)
                    w = torch.tensor(
                        np.broadcast_to((1.0 + sw * ns)[:, None], xn.shape).copy(),
                        dtype=torch.float32, device=dev)
                    loss = model.loss_fn(x, y, w) / GACC
                else:
                    loss = model.loss_fn(x, y) / GACC

            loss.backward()
            # FIX #2: total_loss accumulates correctly (each loss already /GACC)
            total_loss += loss.item()

        # Stochastic depth (gradient-level)
        active = apply_stochastic_depth(model, sd)

        # Optimizer
        lrm = lr_mul(step, tms)
        opt.step(model, step, lrm)
        model.zero_grad()
        ema.update(model)

        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000
        tms += step_ms

        # FIX #2: display total_loss directly (it IS the mean CE loss)
        if main_proc and (step == 1 or step % LOG_EVERY == 0 or step == ITERS):
            tps = BATCH_TOK / (step_ms / 1000)
            log(f"  {step:5d}/{ITERS} loss:{total_loss:.4f} {phase:8s} "
                f"sw:{sw:.2f} sd:{sd:.2f} lr:{lrm:.3f} "
                f"ema:{'ON' if ema.on else '--'} L:{active}/{L} "
                f"{tps:.0f}t/s {tms/1000:.1f}s")

        if VAL_EVERY > 0 and step % VAL_EVERY == 0 and main_proc:
            vl, vb = evaluate(model, vt, bb, hl, ib, dev)
            log(f"  >>> val_loss:{vl:.4f} val_bpb:{vb:.4f}")

        if cap and tms >= cap:
            log(f"  Wallclock {MAX_WALL}s doldu."); break

    if ema.on:
        log("EMA uygulaniyor...")
        ema.apply(model)

    if main_proc:
        log(f"\n{'='*60}")
        log("FINAL EVALUATION")
        vl, vb = evaluate(model, vt, bb, hl, ib, dev)
        log(f"  Plain   → val_loss:{vl:.4f} val_bpb:{vb:.4f}")
        vl2, vb2 = evaluate(model, vt, bb, hl, ib, dev, markov=mk)
        log(f"  +Bigram → val_loss:{vl2:.4f} val_bpb:{vb2:.4f}")
        best = min(vb, vb2)
        log(f"  BEST    → {best:.4f}")

        sd = {n: p.data for n, p in model.named_parameters()}
        qo, st = quant_int8(sd)
        qo["bigram_table"] = mk.lp16
        comp = zlib.compress(pickle.dumps(qo), level=9)
        cb = len(Path(__file__).read_text().encode())
        total = cb + len(comp)
        log(f"  Model:{len(comp)/1e6:.2f}MB Code:{cb/1e6:.2f}MB TOTAL:{total/1e6:.2f}MB "
            f"{'OK' if total < 16e6 else 'OVER!'}")
        log(f"  Params:{st['params']:,}")
        Path(OUT_DIR).mkdir(exist_ok=True)
        Path(OUT_DIR, f"{run_id}_model.pkl.zlib").write_bytes(comp)
        log(f"  Saved: {OUT_DIR}/{run_id}_model.pkl.zlib")
        log("=" * 60)

    if world > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
