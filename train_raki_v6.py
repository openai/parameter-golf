#!/usr/bin/env python3
"""
RAKI v6 CUDA — H100. Bug-free, hızlı.
Unique teknikler: Hybrid Markov curriculum, Stochastic Depth, BigramHash eval-only, EMA.

  RUN_ID=raki_v6 torchrun --standalone --nproc_per_node=1 train_raki_v6.py
"""
import glob, math, os, pickle, sys, time, uuid, zlib
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
import sentencepiece as spm

# ==============================================================================
# CONFIG
# ==============================================================================
C = type("C", (), {
    "RAKI_POWER":  float(os.environ.get("RAKI_POWER", "0.25")),
    "STOCH_DEPTH": float(os.environ.get("STOCH_DEPTH", "0.10")),
    "EMA_START":   float(os.environ.get("EMA_START", "0.90")),
    "EMA_DECAY":   float(os.environ.get("EMA_DECAY", "0.995")),
    "MUON_WD":     float(os.environ.get("MUON_WD", "0.04")),
    "BH_WEIGHT":   float(os.environ.get("BH_WEIGHT", "0.4")),
    "WD_FRAC":     float(os.environ.get("WD_FRAC", "0.15")),
    "GRAD_SCALE":  float(os.environ.get("GRAD_SCALE", "1.5")),
})()

# ==============================================================================
# HYPERPARAMETERS
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
OUT_DIR     = os.environ.get("OUT_DIR", "logs")
MICRO       = BATCH_TOK // GACC
WARMDOWN    = max(1, int(ITERS * C.WD_FRAC))
HD          = D // NH
CTRL        = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")
train_pat   = f"{data_path}/fineweb_train_*.bin"
val_pat     = f"{data_path}/fineweb_val_*.bin"

def lr_mul(step, ms):
    if MAX_WALL <= 0:
        ws = max(ITERS - WARMDOWN, 0)
        return max((ITERS - step) / max(WARMDOWN, 1), 0.0) if step >= ws else 1.0
    sms = ms / max(step, 1)
    rms = max(1000 * MAX_WALL - ms, 0.0)
    wms = WARMDOWN * sms
    return rms / max(wms, 1e-9) if rms <= wms else 1.0

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
# MARKOV — 1 shard only (hızlı), hybrid scoring
# ==============================================================================
class Markov:
    def __init__(self, pat, V, dev):
        shards = sorted(glob.glob(pat))
        counts = np.zeros((V, V), dtype=np.float64)
        tok = load_shard(shards[0])  # sadece 1 shard = 2 saniye
        np.add.at(counts, (tok[:-1], tok[1:]), 1.0)
        sm = counts + 0.01
        probs = sm / sm.sum(axis=1, keepdims=True)
        lp = np.log(probs)
        self.lp16 = lp.astype(np.float16)
        self.bias = torch.tensor(lp, dtype=torch.float32, device=dev)
        ent = -(probs * lp).sum(axis=1).astype(np.float32)
        mn, mx = ent.min(), ent.max()
        self.ent_n = (ent - mn) / (mx - mn) if mx > mn else np.full_like(ent, 0.5)

    def hybrid(self, xn, yn):
        """(batch,) hybrid score = entropy × surprise."""
        s = -self.lp16[xn.ravel(), yn.ravel()].astype(np.float32).reshape(xn.shape)
        e = self.ent_n[xn.ravel()].reshape(xn.shape)
        return (s * e).mean(axis=1)

# ==============================================================================
# RAKI SCHEDULE
# ==============================================================================
def raki(p):
    p = max(0.0, min(1.0, p))
    if p <= 0.25:
        t = p / 0.25
        return "ayik", C.RAKI_POWER * 0.1 * t, C.STOCH_DEPTH * 0.5
    elif p <= 0.65:
        t = (p - 0.25) / 0.40
        return "keyifli", C.RAKI_POWER * (0.1 + 0.9*t), C.STOCH_DEPTH * (0.5+0.5*t)
    elif p <= 0.90:
        return "kivam", C.RAKI_POWER, C.STOCH_DEPTH
    else:
        t = (p - 0.90) / 0.10
        return "ayilma", C.RAKI_POWER*(1-0.5*t), C.STOCH_DEPTH*(1-0.8*t)

# ==============================================================================
# MODEL
# ==============================================================================
def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)

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
        # Blocks
        n_enc = L // 2
        self.n_enc = n_enc
        self.n_dec = L - n_enc
        self.skip_w = nn.Parameter(torch.ones(min(n_enc, self.n_dec), D, device=dev))
        # Each block: q,k,v,proj weights + attn_scale, mlp_scale, resid_mix, q_gain + mlp fc,proj
        self.q_w = nn.ParameterList([nn.Parameter(torch.empty(D, D, device=dev)) for _ in range(L)])
        self.k_w = nn.ParameterList([nn.Parameter(torch.empty(NKV*HD, D, device=dev)) for _ in range(L)])
        self.v_w = nn.ParameterList([nn.Parameter(torch.empty(NKV*HD, D, device=dev)) for _ in range(L)])
        self.o_w = nn.ParameterList([nn.Parameter(torch.zeros(D, D, device=dev)) for _ in range(L)])
        self.fc_w = nn.ParameterList([nn.Parameter(torch.empty(D*MULT, D, device=dev)) for _ in range(L)])
        self.fc_o = nn.ParameterList([nn.Parameter(torch.zeros(D, D*MULT, device=dev)) for _ in range(L)])
        self.a_sc = nn.ParameterList([nn.Parameter(torch.ones(D, device=dev)) for _ in range(L)])
        self.m_sc = nn.ParameterList([nn.Parameter(torch.ones(D, device=dev)) for _ in range(L)])
        self.r_mix = nn.ParameterList([nn.Parameter(torch.stack([torch.ones(D,device=dev), torch.zeros(D,device=dev)])) for _ in range(L)])
        self.qk_g = nn.ParameterList([nn.Parameter(torch.ones(NH, device=dev) * QK_GAIN) for _ in range(L)])
        # Init weights (kaiming)
        for i in range(L):
            nn.init.kaiming_normal_(self.q_w[i])
            nn.init.kaiming_normal_(self.k_w[i])
            nn.init.kaiming_normal_(self.v_w[i])
            nn.init.kaiming_normal_(self.fc_w[i])

    def _rope(self, x):
        T = x.shape[2]
        d = x.shape[-1] // 2
        c, s = self.rope_cos[:T, :d], self.rope_sin[:T, :d]
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1*c - x2*s, x2*c + x1*s], dim=-1)

    def _attn(self, x, i):
        B, T, _ = x.shape
        xn = rms_norm(x)
        q = F.linear(xn, self.q_w[i]).view(B, T, NH, HD).transpose(1, 2)   # (B,NH,T,HD)
        k = F.linear(xn, self.k_w[i]).view(B, T, NKV, HD).transpose(1, 2) # (B,NKV,T,HD)
        v = F.linear(xn, self.v_w[i]).view(B, T, NKV, HD).transpose(1, 2) # (B,NKV,T,HD)
        # RoPE on normalized q,k
        q = self._rope(rms_norm(q))
        k = self._rope(rms_norm(k))
        # Q gain
        q = q * self.qk_g[i][None, :, None, None]
        # GQA: expand KV heads
        reps = NH // NKV
        if reps > 1:
            k = k.repeat_interleave(reps, dim=1)  # (B,NH,T,HD)
            v = v.repeat_interleave(reps, dim=1)
        # Ensure same dtype
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=HD**-0.5)
        y = y.transpose(1, 2).reshape(B, T, D)
        return F.linear(y, self.o_w[i])

    def _mlp(self, x, i):
        xn = rms_norm(x)
        h = F.relu(F.linear(xn, self.fc_w[i]))
        return F.linear(h * h, self.fc_o[i])  # relu²

    def _block(self, x, x0, i):
        m = self.r_mix[i]
        x = m[0][None, None, :] * x + m[1][None, None, :] * x0
        x = x + self.a_sc[i][None, None, :] * self._attn(x, i)
        x = x + self.m_sc[i][None, None, :] * self._mlp(x, i)
        return x

    def forward(self, ids):
        x = rms_norm(self.emb(ids).to(torch.bfloat16))
        x0, skips = x, []
        for i in range(self.n_enc):
            x = self._block(x, x0, i); skips.append(x)
        for i in range(self.n_dec):
            if skips:
                x = x + self.skip_w[i][None, None, :].to(x.dtype) * skips.pop()
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
        b = bias_table[x].to(lg.dtype) * C.BH_WEIGHT
        return F.cross_entropy((lg + b).reshape(-1, V).float(), y.reshape(-1))

# ==============================================================================
# MUON OPTIMIZER
# ==============================================================================
def ns5(g, steps=5, eps=1e-7):
    x = g.float(); x = x / (x.norm() + eps)
    tr = x.shape[0] > x.shape[1]
    if tr: x = x.T
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        am = x @ x.T
        x = a*x + (b*am + c*(am@am)) @ x
    if tr: x = x.T
    return x.to(g.dtype)

class Optimizer:
    def __init__(self, model):
        self.bufs = {}
        self.mat_keys, self.sca_keys = [], []
        for n, p in model.named_parameters():
            if n == "emb.weight": continue
            if n.startswith(("q_w.", "k_w.", "v_w.", "o_w.", "fc_w.", "fc_o.")):
                self.mat_keys.append(n)
                self.bufs[n] = torch.zeros_like(p)
            elif any(c in n for c in CTRL) or n == "skip_w":
                self.sca_keys.append(n)
        self.adam_e = torch.optim.Adam([dict(model.named_parameters())["emb.weight"]],
                                       lr=EMBED_LR, betas=(0.9, 0.95))
        sca_params = [dict(model.named_parameters())[n] for n in self.sca_keys]
        self.adam_s = torch.optim.Adam(sca_params, lr=SCALAR_LR, betas=(0.9, 0.95))

    def step(self, model, step_n, lrm):
        pdict = dict(model.named_parameters())
        # Muon momentum
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
            if C.MUON_WD > 0: g = g + C.MUON_WD * p.data
            buf = mom * self.bufs[n] + g
            self.bufs[n] = buf
            go = ns5(g + mom * buf, MUON_STEPS)
            sc = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
            p.data -= lr * (go * sc).to(p.dtype)
        # Adam
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
            self.shadow[n].lerp_(p.data, 1 - C.EMA_DECAY)
    def apply(self, model):
        if self.shadow:
            for n, p in model.named_parameters():
                p.data.copy_(self.shadow[n])

# ==============================================================================
# GRADIENT OPS: Stochastic Depth + Layer Scaling
# ==============================================================================
def grad_ops(model, progress, sd_prob):
    n_enc = L // 2
    uni = min(1.0, progress / 0.9)
    active = L
    for i in range(L):
        # Stochastic depth (skip first and last layer)
        if sd_prob > 0.01 and 0 < i < L-1 and np.random.random() < sd_prob:
            # Zero all grads for this layer
            for n in (f"q_w.{i}", f"k_w.{i}", f"v_w.{i}", f"o_w.{i}",
                      f"fc_w.{i}", f"fc_o.{i}", f"a_sc.{i}", f"m_sc.{i}",
                      f"r_mix.{i}", f"qk_g.{i}"):
                p = dict(model.named_parameters()).get(n)
                if p is not None and p.grad is not None:
                    p.grad.zero_()
            active -= 1
            continue
        # Layer gradient scaling
        if i < n_enc:
            depth = i / max(n_enc - 1, 1)
        else:
            depth = 1.0 - (i - n_enc) / max(L - n_enc - 1, 1)
        es = max(0.3, depth ** (1.0 / C.GRAD_SCALE))
        sc = es * (1 - uni) + uni
        sc = max(0.3, min(1.5, sc))
        if abs(sc - 1.0) > 0.02:
            for n in (f"q_w.{i}", f"k_w.{i}", f"v_w.{i}", f"o_w.{i}",
                      f"fc_w.{i}", f"fc_o.{i}"):
                p = dict(model.named_parameters()).get(n)
                if p is not None and p.grad is not None:
                    p.grad.mul_(sc)
    return active

# ==============================================================================
# EVAL + QUANT + LUTS
# ==============================================================================
def build_luts(sp, V_):
    sv = int(sp.vocab_size()); ts = max(sv, V_)
    bb, hl, ib = np.zeros(ts, dtype=np.int16), np.zeros(ts, dtype=np.bool_), np.ones(ts, dtype=np.bool_)
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
    log(" RAKI v6 CUDA")
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
    opt = Optimizer(model)
    ema = EMA()
    np_ = sum(p.numel() for p in model.parameters())
    log(f"  Model: {np_:,} params, {L}L {D}D")
    log(f"  Raki={C.RAKI_POWER} SD={C.STOCH_DEPTH} EMA={C.EMA_DECAY} WD={C.MUON_WD}")
    log(f"  BH_eval={C.BH_WEIGHT} Warmdown={WARMDOWN} ({C.WD_FRAC*100:.0f}%)")

    loader = Loader(train_pat, dev)

    log(f"\nTraining: {ITERS} iters, {MAX_WALL}s cap")
    log("-" * 60)

    tms = 0.0
    cap = 1000 * MAX_WALL if MAX_WALL > 0 else None

    for step in range(1, ITERS + 1):
        t0 = time.perf_counter()
        progress = min(tms / cap, 1.0) if cap else step / ITERS
        phase, sw, sd = raki(progress)

        if progress >= C.EMA_START and not ema.on:
            ema.start(model)
            log(f"  >>> EMA ON (step {step})")

        model.zero_grad()
        total_loss = 0.0

        for _ in range(GACC):
            x, y, xn, yn = loader.batch(MICRO // world, SEQ)

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
            total_loss += loss.item()

        # Gradient ops
        active = grad_ops(model, progress, sd)

        # Optimizer
        lrm = lr_mul(step, tms)
        opt.step(model, step, lrm)
        model.zero_grad()
        ema.update(model)

        torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000
        tms += step_ms

        if main_proc and (step == 1 or step % LOG_EVERY == 0 or step == ITERS):
            real_loss = total_loss * GACC  # undo the /GACC division
            tps = BATCH_TOK / (step_ms / 1000)
            log(f"  {step:5d}/{ITERS} loss:{real_loss:.4f} {phase:8s} "
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
        log(f"  BEST    → {min(vb, vb2):.4f}")

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
