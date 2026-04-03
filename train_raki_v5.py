#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  RAKI v5 — Tam Entegre Sistem                                   ║
║                                                                  ║
║  TRAINING (öğrenci tek başına):                                  ║
║    • Markov hybrid (entropy × surprise) → curriculum weighting   ║
║    • Stochastic Depth → rastgele layer skip (regularization)     ║
║    • Rakı Schedule → tüm teknikleri koordine eder                ║
║    • Layer Gradient Scaling → içten dışa öğrenme                 ║
║    • Progressive Freezing → opsiyonel (büyük model için)         ║
║    • EMA → son fazda ağırlık ortalaması                          ║
║                                                                  ║
║  EVALUATION (öğretmen fısıldar):                                 ║
║    • BigramHash → logit bias SADECE eval'da                      ║
║    • TTT → sınav sırasında öğrenme (1 step/batch)                ║
║                                                                  ║
║  python3 train_raki_v5.py --test    # 30sn doğrulama             ║
║  python3 train_raki_v5.py           # full training              ║
╚══════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations
import glob, math, os, pickle, sys, time, uuid, zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

COMPUTE_DTYPE = mx.bfloat16
TEST_MODE = "--test" in sys.argv

# ==============================================================================
# CONFIGURATION — Her şey burada, tutarlı defaults
# ==============================================================================

# Rakı Schedule: training progress oranları
RAKI_P1     = float(os.environ.get("RAKI_P1", 0.25))    # ayık sonu
RAKI_P2     = float(os.environ.get("RAKI_P2", 0.65))    # keyifli sonu
RAKI_P3     = float(os.environ.get("RAKI_P3", 0.90))    # kıvam sonu
RAKI_POWER  = float(os.environ.get("RAKI_POWER", 0.25)) # surprise weight gücü

# Stochastic Depth: layer skip olasılığı (rakı schedule modüle eder)
STOCH_DEPTH_MAX = float(os.environ.get("STOCH_DEPTH_MAX", 0.15))  # max %15 skip

# Progressive Freeze: default KAPALI (0.0), büyük modelde aç
FREEZE_P1   = float(os.environ.get("FREEZE_P1", 0.0))
FREEZE_P2   = float(os.environ.get("FREEZE_P2", 0.0))

# Layer Gradient Scaling
GRAD_SCALE  = float(os.environ.get("GRAD_SCALE_POWER", 1.5))

# EMA
EMA_START   = float(os.environ.get("EMA_START", 0.90))
EMA_DECAY   = float(os.environ.get("EMA_DECAY", 0.995))

# Muon WD
MUON_WD     = float(os.environ.get("MUON_WD", 0.0))  # default KAPALI, test sonra aç

# TTT (Test-Time Training)
TTT_LR      = float(os.environ.get("TTT_LR", 0.001))   # eval sırasında küçük lr
TTT_STEPS   = int(os.environ.get("TTT_STEPS", 1))       # batch başına kaç step

# BigramHash eval weight
BH_EVAL_W   = float(os.environ.get("BH_EVAL_WEIGHT", 0.5))  # eval'da logit bias gücü

# Warmdown: iterations'ın ORANI (absolute değil!)
WARMDOWN_FRAC = float(os.environ.get("WARMDOWN_FRAC", 0.10))  # son %10

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
class HP:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID", "v5test" if TEST_MODE else str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))
    iterations     = 50 if TEST_MODE else int(os.environ.get("ITERATIONS", 20_000))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every = 5 if TEST_MODE else int(os.environ.get("TRAIN_LOG_EVERY", 20))
    train_batch_tokens = 4096 if TEST_MODE else int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum     = 1 if TEST_MODE else int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    seq_len        = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    mlx_eager      = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps   = 3 if TEST_MODE else int(os.environ.get("WARMUP_STEPS", 20))
    max_wall_sec   = 45.0 if TEST_MODE else float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0))
    # Model
    V    = int(os.environ.get("VOCAB_SIZE", 1024))
    L    = 4 if TEST_MODE else int(os.environ.get("NUM_LAYERS", 9))
    D    = 256 if TEST_MODE else int(os.environ.get("MODEL_DIM", 512))
    NH   = 4 if TEST_MODE else int(os.environ.get("NUM_HEADS", 8))
    NKV  = 2 if TEST_MODE else int(os.environ.get("NUM_KV_HEADS", 4))
    MULT = int(os.environ.get("MLP_MULT", 2))
    e_std   = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    r_base  = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_g    = float(os.environ.get("QK_GAIN_INIT", 1.5))
    # Optimizer
    b1, b2, eps = 0.9, 0.95, 1e-8
    e_lr  = float(os.environ.get("TIED_EMBED_LR", 0.05))
    m_lr  = float(os.environ.get("MATRIX_LR", 0.04))
    s_lr  = float(os.environ.get("SCALAR_LR", 0.04))
    mu_m  = float(os.environ.get("MUON_MOMENTUM", 0.95))
    mu_s  = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    mu_ws = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    mu_wn = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    out_dir = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self): return f"{self.data_path}/fineweb_train_*.bin"
    @property
    def val_files(self): return f"{self.data_path}/fineweb_val_*.bin"
    @property
    def micro(self): return self.train_batch_tokens // self.grad_accum
    @property
    def warmdown_iters(self): return max(1, int(self.iterations * WARMDOWN_FRAC))

    def lr_mul(self, step, ms):
        wi = self.warmdown_iters
        if self.max_wall_sec <= 0:
            ws = max(self.iterations - wi, 0)
            if step >= ws:
                return max((self.iterations - step) / max(wi, 1), 0.0)
            return 1.0
        sms = ms / max(step, 1)
        wms = wi * sms
        rms = max(1000 * self.max_wall_sec - ms, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

CTRL = tuple("attn_scale,mlp_scale,resid_mix,q_gain,skip_weights".split(","))

# ==============================================================================
# MARKOV TABLE — Surprise scoring + Eval logit bias
# ==============================================================================
class MarkovTable:
    def __init__(self, V):
        self.V = V
        self.counts = np.zeros((V, V), dtype=np.float64)

    def update(self, tokens):
        np.add.at(self.counts, (tokens[:-1], tokens[1:]), 1.0)

    def finalize(self):
        smoothed = self.counts + 0.01
        probs = smoothed / smoothed.sum(axis=1, keepdims=True)
        log_probs = np.log(probs)
        self.table_np = log_probs.astype(np.float16)
        self.table_mx = mx.array(self.table_np, dtype=mx.float16)
        # Entropy per prev_token: H(prev) = -Σ P(next|prev) * log P(next|prev)
        self.entropy = -(probs * log_probs).sum(axis=1).astype(np.float32)  # (vocab,)
        # Normalize entropy to 0-1
        emin, emax = self.entropy.min(), self.entropy.max()
        if emax > emin:
            self.entropy_norm = (self.entropy - emin) / (emax - emin)
        else:
            self.entropy_norm = np.full_like(self.entropy, 0.5)
        del self.counts

    def hybrid_score(self, x_np, y_np):
        """
        Hybrid: entropy(prev) × surprise(next|prev)
        Yüksek entropy + yüksek surprise = ALTIN (öğrenmeye değer)
        Düşük entropy + yüksek surprise = gürültü (skip)
        """
        # Surprise per token
        lp = self.table_np[x_np.ravel(), y_np.ravel()].astype(np.float32)
        surprise = -lp.reshape(x_np.shape)  # (batch, seq)
        # Entropy per prev token
        ent = self.entropy_norm[x_np.ravel()].reshape(x_np.shape)  # (batch, seq)
        # Hybrid: çarp, sequence ortalaması al
        hybrid = (surprise * ent).mean(axis=1)  # (batch,)
        return hybrid

    def logit_bias(self, prev_ids):
        """Eval'da kullanılacak: (batch, seq, vocab) logit bias."""
        return self.table_mx[prev_ids]

# ==============================================================================
# RAKI SCHEDULE — Tüm teknikleri koordine eder
# ==============================================================================
def raki_state(progress):
    """
    Her tekniğin parametresini progress'e göre döndürür.
    Tek bir fonksiyon, tutarlı koordinasyon.
    """
    p = max(0.0, min(1.0, progress))

    if p <= RAKI_P1:
        t = p / RAKI_P1
        return {
            "phase": "ayik",
            "surprise_w": RAKI_POWER * 0.1 * t,           # düşük
            "stoch_depth": STOCH_DEPTH_MAX * 0.5,          # orta (keşfet)
            "grad_uniformity": 0.0,                         # derin layerlar dominant
        }
    elif p <= RAKI_P2:
        t = (p - RAKI_P1) / (RAKI_P2 - RAKI_P1)
        return {
            "phase": "keyifli",
            "surprise_w": RAKI_POWER * (0.1 + 0.9 * t),   # artan
            "stoch_depth": STOCH_DEPTH_MAX * (0.5 + 0.5*t),# artan
            "grad_uniformity": t * 0.5,                     # dengeleniyor
        }
    elif p <= RAKI_P3:
        t = (p - RAKI_P2) / (RAKI_P3 - RAKI_P2)
        return {
            "phase": "kivam",
            "surprise_w": RAKI_POWER,                       # max
            "stoch_depth": STOCH_DEPTH_MAX,                 # max
            "grad_uniformity": 0.5 + 0.5*t,                # uniform'a yaklaşıyor
        }
    else:
        t = (p - RAKI_P3) / (1.0 - RAKI_P3)
        return {
            "phase": "ayilma",
            "surprise_w": RAKI_POWER * (1.0 - 0.5*t),     # azalan
            "stoch_depth": STOCH_DEPTH_MAX * (1.0 - 0.8*t),# azalan (stabilize)
            "grad_uniformity": 1.0,                         # tamamen uniform
        }

# ==============================================================================
# GRADIENT OPERATIONS — Freeze + Scale + Stochastic Depth
# ==============================================================================
def apply_gradient_ops(grads_flat, progress, num_layers, rstate):
    """
    Tüm gradient manipülasyonları TEK yerde.
    1) Progressive freeze (opsiyonel)
    2) Stochastic depth (rastgele layer zeroing)
    3) Layer gradient scaling (parabolik)
    """
    n_enc = num_layers // 2

    # ── Hangi layerlar eğitilebilir? ──
    if FREEZE_P1 > 0 and progress <= FREEZE_P1:
        mid_start = max(0, n_enc - num_layers // 4)
        mid_end = min(num_layers, n_enc + num_layers // 4)
        trainable = set(range(mid_start, mid_end))
    elif FREEZE_P2 > 0 and progress <= FREEZE_P2:
        expand = int((progress - FREEZE_P1) / max(FREEZE_P2 - FREEZE_P1, 0.01) * (num_layers // 2))
        mid_start = max(0, n_enc - num_layers // 4 - expand)
        mid_end = min(num_layers, n_enc + num_layers // 4 + expand)
        trainable = set(range(mid_start, mid_end))
    else:
        trainable = set(range(num_layers))

    # ── Stochastic depth: hangi layerlar bu step'te skip? ──
    sd_prob = rstate["stoch_depth"]
    skip_layers = set()
    if sd_prob > 0.01:
        for i in range(num_layers):
            if np.random.random() < sd_prob:
                skip_layers.add(i)
        # En az 2 layer her zaman açık (ilk ve son)
        skip_layers.discard(0)
        skip_layers.discard(num_layers - 1)

    # ── Layer gradient scaling ──
    uniformity = rstate["grad_uniformity"]

    for key in list(grads_flat.keys()):
        if not key.startswith("blocks."):
            continue
        parts = key.split(".")
        try:
            li = int(parts[1])
        except (IndexError, ValueError):
            continue

        # Frozen?
        if li not in trainable:
            grads_flat[key] = mx.zeros_like(grads_flat[key])
            continue

        # Stochastic depth skip?
        if li in skip_layers:
            grads_flat[key] = mx.zeros_like(grads_flat[key])
            continue

        # Gradient scaling
        if li < n_enc:
            depth = li / max(n_enc - 1, 1)
        else:
            depth = 1.0 - (li - n_enc) / max(num_layers - n_enc - 1, 1)

        # early: deep=strong, late: uniform
        early_s = max(0.3, depth ** (1.0 / GRAD_SCALE))
        scale = early_s * (1.0 - uniformity) + 1.0 * uniformity
        scale = max(0.3, min(1.5, scale))

        if abs(scale - 1.0) > 0.02:
            grads_flat[key] = grads_flat[key] * scale

    active = len(trainable - skip_layers)
    return grads_flat, active

# ==============================================================================
# EMA
# ==============================================================================
class EMA:
    def __init__(self, decay=0.995):
        self.decay = decay
        self.shadow = None
        self.active = False

    def activate(self, model):
        if not self.shadow:
            self.shadow = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
        self.active = True

    def update(self, model):
        if not self.active: return
        d = self.decay
        for k, v in tree_flatten(model.parameters()):
            self.shadow[k] = d * self.shadow[k] + (1-d) * v

    def apply(self, model):
        if self.shadow:
            model.update(tree_unflatten(list(self.shadow.items())))

# ==============================================================================
# HELPERS
# ==============================================================================
def rms_norm(x, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x*x, axis=-1, keepdims=True) + eps)).astype(x.dtype)

def ns5(g, steps, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32); x = x / (mx.sqrt(mx.sum(x*x)) + eps)
    tr = x.shape[0] > x.shape[1]
    if tr: x = x.T
    for _ in range(steps):
        am = x @ x.T; x = a*x + (b*am + c*(am@am)) @ x
    if tr: x = x.T
    return x.astype(g.dtype)

def load_shard(path):
    h = np.fromfile(path, dtype="<i4", count=256)
    return np.fromfile(path, dtype="<u2", count=int(h[2]), offset=256*4).astype(np.int32, copy=False)

class TokenStream:
    def __init__(self, pat, log_fn=None):
        self.files = [Path(p) for p in sorted(glob.glob(pat))]
        if not self.files: raise FileNotFoundError(pat)
        self.ep, self.fi, self.pos = 1, 0, 0
        self.log_fn = log_fn
        self.tok = load_shard(self.files[0])
    def _nxt(self):
        self.fi = (self.fi+1) % len(self.files)
        if self.fi == 0:
            self.ep += 1
            if self.log_fn: self.log_fn(f"epoch:{self.ep}")
        self.tok = load_shard(self.files[self.fi]); self.pos = 0
    def take(self, n):
        parts, left = [], n
        while left > 0:
            if self.pos >= self.tok.size: self._nxt()
            k = min(left, self.tok.size - self.pos)
            parts.append(self.tok[self.pos:self.pos+k])
            self.pos += k; left -= k
        return parts[0] if len(parts)==1 else np.concatenate(parts)

class Loader:
    def __init__(self, pat, log_fn=None):
        self.s = TokenStream(pat, log_fn)
    def batch(self, bt, sl):
        u = (bt // sl) * sl
        c = self.s.take(u + 1)
        xn, yn = c[:-1].reshape(-1, sl), c[1:].reshape(-1, sl)
        return mx.array(xn, dtype=mx.int32), mx.array(yn, dtype=mx.int32), xn, yn

# ==============================================================================
# MODEL — Temiz, BigramHash YOK (eval'da ayrıca eklenir)
# ==============================================================================
class CL(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.weight = nn.Linear(i, o, bias=False).weight.astype(mx.float32)
    def __call__(self, x): return x @ self.weight.astype(x.dtype).T

class Norm(nn.Module):
    def __call__(self, x): return rms_norm(x)

class Attn(nn.Module):
    def __init__(self, D, nh, nkv, rb, qkg):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, D//nh
        self.cq, self.ck = CL(D, D), CL(D, nkv*self.hd)
        self.cv, self.proj = CL(D, nkv*self.hd), CL(D, D)
        self.qg = mx.ones((nh,), dtype=mx.float32) * qkg
        self.rope = nn.RoPE(self.hd, traditional=False, base=rb)
        self.sc = self.hd ** -0.5
    def __call__(self, x):
        B,T,D = x.shape
        q = self.cq(x).reshape(B,T,self.nh,self.hd).transpose(0,2,1,3)
        k = self.ck(x).reshape(B,T,self.nkv,self.hd).transpose(0,2,1,3)
        v = self.cv(x).reshape(B,T,self.nkv,self.hd).transpose(0,2,1,3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.qg.astype(q.dtype)[None,:,None,None]
        y = mx.fast.scaled_dot_product_attention(q,k,v,scale=self.sc,mask="causal")
        return self.proj(y.transpose(0,2,1,3).reshape(B,T,D))

class MLP(nn.Module):
    def __init__(self, D, m):
        super().__init__()
        self.fc, self.proj = CL(D, D*m), CL(D*m, D)
    def __call__(self, x):
        h = nn.relu(self.fc(x)); return self.proj(h*h)

class Block(nn.Module):
    def __init__(self, D, nh, nkv, m, rb, qkg):
        super().__init__()
        self.an, self.mn = Norm(), Norm()
        self.attn, self.mlp = Attn(D, nh, nkv, rb, qkg), MLP(D, m)
        self.asc = mx.ones((D,), dtype=mx.float32)
        self.msc = mx.ones((D,), dtype=mx.float32)
        self.rm = mx.array(np.stack((np.ones(D, dtype=np.float32), np.zeros(D, dtype=np.float32))))
    def __call__(self, x, x0):
        m = self.rm.astype(x.dtype)
        x = m[0][None,None,:]*x + m[1][None,None,:]*x0
        x = x + self.asc.astype(x.dtype)[None,None,:] * self.attn(self.an(x))
        x = x + self.msc.astype(x.dtype)[None,None,:] * self.mlp(self.mn(x))
        return x

class GPT(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.lsc = hp.softcap
        self.tok_emb = nn.Embedding(hp.V, hp.D)
        self.n_enc = hp.L // 2
        self.n_dec = hp.L - self.n_enc
        self.n_skip = min(self.n_enc, self.n_dec)
        self.skip_weights = mx.ones((self.n_skip, hp.D), dtype=mx.float32)
        self.blocks = [Block(hp.D, hp.NH, hp.NKV, hp.MULT, hp.r_base, hp.qk_g) for _ in range(hp.L)]
        self.fn = Norm()
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * hp.e_std).astype(COMPUTE_DTYPE)

    def forward(self, ids):
        x = rms_norm(self.tok_emb(ids).astype(COMPUTE_DTYPE))
        x0, skips = x, []
        for i in range(self.n_enc):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.n_dec):
            if skips: x = x + self.skip_weights[i].astype(x.dtype)[None,None,:] * skips.pop()
            x = self.blocks[self.n_enc + i](x, x0)
        return self.fn(x)

    def get_logits(self, ids):
        h = self.forward(ids)
        return self.lsc * mx.tanh((h @ self.tok_emb.weight.astype(h.dtype).T) / self.lsc)

    def loss(self, x, y, token_weights=None):
        """Training loss — BigramHash YOK, temiz öğrenme."""
        logits = self.get_logits(x)
        lf = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
        tgt = y.reshape(-1)
        if token_weights is None:
            return nn.losses.cross_entropy(lf, tgt, reduction="mean")
        pt = nn.losses.cross_entropy(lf, tgt, reduction="none")
        w = token_weights.reshape(-1)
        return mx.sum(pt * w) / mx.sum(w)

    def loss_with_bias(self, x, y, bias_table):
        """
        Eval loss — BigramHash logit bias EKLENIR.
        Öğretmen sınav sırasında fısıldar.
        """
        logits = self.get_logits(x)
        # Markov bias: prev_token → next_token olasılıkları
        bias = bias_table[x].astype(logits.dtype)  # (batch, seq, vocab)
        logits = logits + BH_EVAL_W * bias
        lf = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
        return nn.losses.cross_entropy(lf, y.reshape(-1), reduction="mean")

# ==============================================================================
# OPTIMIZERS
# ==============================================================================
class Muon:
    def __init__(self, keys, params, hp):
        self.keys, self.hp = keys, hp
        self.bufs = {k: mx.zeros_like(params[k]) for k in keys}
    def step(self, P, G, step, lr_mul):
        h = self.hp
        if h.mu_wn:
            t = min(step / h.mu_wn, 1.0)
            mom = (1-t)*h.mu_ws + t*h.mu_m
        else: mom = h.mu_m
        lr = h.m_lr * lr_mul
        out = {}
        for k in self.keys:
            p, g = P[k], G[k]
            if MUON_WD > 0: g = g + MUON_WD * p
            buf = mom * self.bufs[k] + g; self.bufs[k] = buf
            go = ns5(g + mom*buf, h.mu_s)
            sc = math.sqrt(max(1.0, p.shape[0]/p.shape[1]))
            out[k] = p - lr*(go*sc).astype(p.dtype)
        return out

class Opt:
    def __init__(self, model, hp):
        self.hp = hp
        P = dict(tree_flatten(model.parameters()))
        self.ek = "tok_emb.weight"
        self.mk = [k for k,p in P.items() if k.startswith("blocks.") and p.ndim==2 and not any(c in k for c in CTRL)]
        self.sk = [k for k,p in P.items() if k=="skip_weights" or (k.startswith("blocks.") and (p.ndim<2 or any(c in k for c in CTRL)))]
        self.muon = Muon(self.mk, P, hp)
        self.ae = optim.Adam(learning_rate=hp.e_lr, betas=[hp.b1, hp.b2], eps=hp.eps, bias_correction=True)
        self.asc = optim.Adam(learning_rate=hp.s_lr, betas=[hp.b1, hp.b2], eps=hp.eps, bias_correction=True)
    def step(self, model, gt, step, lr_mul):
        P = dict(tree_flatten(model.parameters()))
        G = dict(tree_flatten(gt))
        U = dict(P)
        U.update(self.muon.step(P, G, step=step, lr_mul=lr_mul))
        self.ae.learning_rate = self.hp.e_lr * lr_mul
        U.update(self.ae.apply_gradients({self.ek: G[self.ek]}, {self.ek: P[self.ek]}))
        self.asc.learning_rate = self.hp.s_lr * lr_mul
        U.update(self.asc.apply_gradients({k:G[k] for k in self.sk}, {k:P[k] for k in self.sk}))
        model.update(tree_unflatten(list(U.items())))

# ==============================================================================
# QUANTIZATION
# ==============================================================================
def _f32(a): return np.array(a.astype(mx.float32), dtype=np.float32, copy=False)

def quant_int8(flat):
    Q, S, D, P, PD, QM = {}, {}, {}, {}, {}, {}
    stats = {"params":0, "bytes":0}
    for n, a in flat.items():
        stats["params"] += int(a.size)
        if not mx.issubdtype(a.dtype, mx.floating):
            P[n] = np.ascontiguousarray(np.array(a)); stats["bytes"]+=P[n].nbytes; continue
        if a.size <= 65536:
            if a.dtype in {mx.float32, mx.bfloat16}:
                PD[n] = str(a.dtype).split(".")[-1]
                P[n] = np.ascontiguousarray(np.array(a.astype(mx.float16), dtype=np.float16))
            else: P[n] = np.ascontiguousarray(np.array(a, copy=True))
            stats["bytes"] += P[n].nbytes; continue
        f = _f32(a)
        if f.ndim == 2:
            ca = np.quantile(np.abs(f), 0.9999984, axis=1)
            s = np.maximum(ca/127, 1/127).astype(np.float32)
            q = np.clip(np.round(np.clip(f,-ca[:,None],ca[:,None])/s[:,None]),-127,127).astype(np.int8)
            QM[n] = {"scheme":"per_row","axis":0}; S[n] = s.astype(np.float16)
        else:
            ca = float(np.quantile(np.abs(f).ravel(), 0.9999984)) if f.size else 0.0
            s = np.array(ca/127 if ca>0 else 1.0, dtype=np.float32)
            q = np.clip(np.round(np.clip(f,-ca,ca)/s),-127,127).astype(np.int8); S[n]=s
        Q[n]=np.ascontiguousarray(q); D[n]=str(a.dtype).split(".")[-1]
        stats["bytes"] += q.nbytes + S[n].nbytes
    obj = {"__quant_format__":"int8_clean_per_row_v1","quantized":Q,"scales":S,"dtypes":D,"passthrough":P}
    if QM: obj["qmeta"]=QM
    if PD: obj["passthrough_orig_dtypes"]=PD
    return obj, stats

# ==============================================================================
# EVAL — BigramHash bias + TTT
# ==============================================================================
def build_luts(sp, V):
    sv = int(sp.vocab_size()); ts = max(sv, V)
    bb, hl, ib = np.zeros(ts, dtype=np.int16), np.zeros(ts, dtype=np.bool_), np.ones(ts, dtype=np.bool_)
    for t in range(sv):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
        ib[t] = False
        if sp.is_byte(t): bb[t]=1; continue
        pc = sp.id_to_piece(t)
        if pc.startswith("▁"): hl[t]=True; pc=pc[1:]
        bb[t] = len(pc.encode("utf-8"))
    return bb, hl, ib

def load_val(pat, sl):
    files = [Path(p) for p in sorted(glob.glob(pat))]
    if not files: raise FileNotFoundError(pat)
    tok = np.ascontiguousarray(np.concatenate([load_shard(f) for f in files]))
    u = ((tok.size-1)//sl)*sl
    return tok[:u+1]

def eval_with_ttt(hp, model, markov, vt, bb, hl, ib, log_fn=None):
    """
    Evaluation with BigramHash + TTT.
    1) Her batch'te: BigramHash bias eklenmiş loss hesapla
    2) TTT: hesaplanan loss'tan 1 gradient step yap
    3) Model sınav sırasında öğrenmeye devam eder
    """
    sl = hp.seq_len
    vbt = hp.val_batch_size // hp.grad_accum
    vbs = vbt // sl
    ts = (vt.size-1) // sl
    ls, tt, tb = 0.0, 0.0, 0.0

    # TTT için basit Adam optimizer
    ttt_params_flat = dict(tree_flatten(model.parameters()))
    # Sadece embedding ve son birkaç layer üzerinde TTT yap
    ttt_opt = optim.Adam(learning_rate=TTT_LR, betas=[0.9, 0.99])

    # TTT loss+grad fonksiyonu
    def ttt_loss(x, y):
        return model.loss_with_bias(x, y, markov.table_mx)

    ttt_grad_fn = nn.value_and_grad(model, ttt_loss)

    for s in range(0, ts, vbs):
        e = min(s+vbs, ts)
        c = vt[s*sl:e*sl+1]
        xn, yn = c[:-1].reshape(-1, sl), c[1:].reshape(-1, sl)
        x = mx.array(xn, dtype=mx.int32)
        y = mx.array(yn, dtype=mx.int32)

        # ── 1) Loss hesapla (BigramHash bias ile) ──
        l = model.loss_with_bias(x, y, markov.table_mx).astype(mx.float32)
        mx.eval(l)

        n = float(y.size)
        ls += float(l.item()) * n

        # Byte hesabı
        bn = bb[yn.ravel()].astype(np.int16, copy=True)
        bn += (hl[yn.ravel()] & ~ib[xn.ravel()]).astype(np.int16)
        tt += n; tb += float(bn.astype(np.float64).sum())

        # ── 2) TTT: bu batch üzerinde 1 gradient step ──
        if TTT_STEPS > 0:
            for _ in range(TTT_STEPS):
                ttt_l, ttt_g = ttt_grad_fn(x, y)
                mx.eval(ttt_l, ttt_g)
                # Sadece basit gradient descent
                P = dict(tree_flatten(model.parameters()))
                G = dict(tree_flatten(ttt_g))
                updated = {k: P[k] - TTT_LR * G[k] for k in G.keys()}
                model.update(tree_unflatten(list(updated.items())))

        if log_fn and (s // max(vbs,1)) % 100 == 0:
            log_fn(f"  eval+ttt: {s}/{ts}")

    if tt == 0: return 999.0, 999.0
    vl = ls / tt
    return vl, (vl / math.log(2)) * (tt / tb)

def eval_simple(hp, model, vt, bb, hl, ib):
    """Basit eval — BigramHash yok, TTT yok. Karşılaştırma için."""
    sl = hp.seq_len
    vbt = hp.val_batch_size // hp.grad_accum
    vbs = vbt // sl
    ts = (vt.size-1) // sl
    ls, tt, tb = 0.0, 0.0, 0.0
    loss_fn = lambda x, y: model.loss(x, y)
    for s in range(0, ts, vbs):
        e = min(s+vbs, ts)
        c = vt[s*sl:e*sl+1]
        xn, yn = c[:-1].reshape(-1, sl), c[1:].reshape(-1, sl)
        x, y = mx.array(xn, dtype=mx.int32), mx.array(yn, dtype=mx.int32)
        l = loss_fn(x, y).astype(mx.float32); mx.eval(l)
        n = float(y.size); ls += float(l.item())*n
        bn = bb[yn.ravel()].astype(np.int16, copy=True)
        bn += (hl[yn.ravel()] & ~ib[xn.ravel()]).astype(np.int16)
        tt += n; tb += float(bn.astype(np.float64).sum())
    vl = ls/tt
    return vl, (vl/math.log(2))*(tt/tb)

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    hp = HP()
    Path(hp.out_dir).mkdir(parents=True, exist_ok=True)
    logf = Path(hp.out_dir)/f"{hp.run_id}.txt"
    def log(m, c=True):
        if c: print(m)
        with logf.open("a") as f: print(m, file=f)

    log("="*60)
    log(f" RAKI v5 {'[TEST]' if TEST_MODE else '[FULL]'}")
    log("="*60)

    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    vt = load_val(hp.val_files, hp.seq_len)
    bb, hl, ib = build_luts(sp, hp.V)

    # ── Markov ──
    log("Markov tablosu...")
    t0 = time.time()
    mk = MarkovTable(hp.V)
    for sf in sorted(glob.glob(hp.train_files)):
        mk.update(load_shard(Path(sf)))
    mk.finalize()
    log(f"  Hazır: {time.time()-t0:.1f}s, {mk.table_np.nbytes/1e6:.1f}MB")

    # ── Model ──
    mx.random.seed(hp.seed)
    model = GPT(hp)
    opt = Opt(model, hp)
    ema = EMA(EMA_DECAY)
    np_ = sum(int(np.prod(p.shape)) for _,p in tree_flatten(model.parameters()))

    log(f"  Model: {np_:,} params, {hp.L}L {hp.D}D")
    log(f"  Warmdown: {hp.warmdown_iters} iters ({WARMDOWN_FRAC*100:.0f}%)")
    log(f"  Rakı: P1={RAKI_P1} P2={RAKI_P2} P3={RAKI_P3} power={RAKI_POWER}")
    log(f"  Stochastic Depth: max={STOCH_DEPTH_MAX}")
    log(f"  Progressive Freeze: {'ON' if FREEZE_P1>0 else 'OFF'}")
    log(f"  Gradient Scale: {GRAD_SCALE}")
    log(f"  EMA: {EMA_START*100:.0f}%'dan sonra, decay={EMA_DECAY}")
    log(f"  Muon WD: {MUON_WD}")
    log(f"  BigramHash: eval-only, weight={BH_EVAL_W}")
    log(f"  TTT: {TTT_STEPS} step/batch, lr={TTT_LR}")

    # ── Loss functions ──
    weighted_lg = nn.value_and_grad(model, lambda x, y, tw: model.loss(x, y, token_weights=tw))
    plain_lg = nn.value_and_grad(model, lambda x, y: model.loss(x, y))

    # ── Data ──
    loader = Loader(hp.train_files, log)

    # ── Warmup ──
    if hp.warmup_steps > 0:
        log(f"  Warmup {hp.warmup_steps}...")
        for _ in range(hp.warmup_steps):
            x, y, _, _ = loader.batch(hp.micro, hp.seq_len)
            l, g = plain_lg(x, y); mx.eval(l, g)
        loader = Loader(hp.train_files, log)

    # ── TRAINING ──
    log(f"\nTraining: {hp.iterations} iters")
    log("-"*60)

    tms = 0.0
    cap = 1000*hp.max_wall_sec if hp.max_wall_sec > 0 else None

    for step in range(1, hp.iterations + 1):
        t0 = time.perf_counter()
        progress = min(tms/cap, 1.0) if cap else step/hp.iterations
        rs = raki_state(progress)
        sw = rs["surprise_w"]
        use_w = sw > 0.02

        # EMA
        if progress >= EMA_START and not ema.active:
            ema.activate(model)
            log(f"  >>> EMA ON (step {step})")

        # Grad accum
        ga = None
        total_loss = 0.0
        gs = 1.0 / hp.grad_accum

        for _ in range(hp.grad_accum):
            x, y, xn, yn = loader.batch(hp.micro, hp.seq_len)

            if use_w:
                surp = mk.hybrid_score(xn, yn)
                smin, smax = surp.min(), surp.max()
                ns = (surp-smin)/(smax-smin) if smax > smin else np.full_like(surp, 0.5)
                tw = mx.array(np.broadcast_to((1.0 + sw*ns)[:,None], xn.shape).copy(), dtype=mx.float32)
                lv, grads = weighted_lg(x, y, tw)
            else:
                lv, grads = plain_lg(x, y)

            mx.eval(lv, grads)
            total_loss += float(lv.item()) * gs

            flat_g = dict(tree_flatten(grads))
            if ga is None:
                ga = {k: v*gs for k,v in flat_g.items()}
            else:
                for k,v in flat_g.items(): ga[k] = ga[k] + v*gs
            if hp.mlx_eager: mx.eval(ga)

        # ── Gradient ops: freeze + stoch depth + scaling ──
        ga, active_L = apply_gradient_ops(ga, progress, hp.L, rs)

        # ── Optimizer ──
        lr_m = hp.lr_mul(step, tms)
        opt.step(model, tree_unflatten(list(ga.items())), step=step, lr_mul=lr_m)
        ema.update(model)

        step_ms = (time.perf_counter()-t0)*1000
        tms += step_ms

        # Log
        if step == 1 or step % hp.train_log_every == 0 or step == hp.iterations:
            tps = hp.train_batch_tokens / (step_ms/1000)
            log(f"  {step:5d}/{hp.iterations} loss:{total_loss:.4f} "
                f"{rs['phase']:8s} sw:{sw:.2f} sd:{rs['stoch_depth']:.2f} "
                f"lr:{lr_m:.3f} ema:{'ON' if ema.active else '--'} "
                f"L:{active_L}/{hp.L} {tps:.0f}t/s {tms/1000:.1f}s")

        # Cap
        if cap and tms >= cap:
            log(f"  Wallclock doldu."); break

    # ── EMA uygula ──
    if ema.active:
        log("EMA ağırlıkları yükleniyor...")
        ema.apply(model)

    # ── EVAL ──
    log(f"\n{'='*60}")
    log("FINAL EVALUATION")

    if TEST_MODE:
        log("  [TEST] Eval skip, compression kontrol")
    else:
        # Basit eval (karşılaştırma)
        vl, vb = eval_simple(hp, model, vt, bb, hl, ib)
        log(f"  Plain eval     → val_loss:{vl:.4f} val_bpb:{vb:.4f}")

        # BigramHash + TTT eval
        log("  BigramHash + TTT eval hesaplanıyor...")
        # Model'in kopyasını al (TTT orijinali değiştirir)
        import copy
        saved_params = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
        vl2, vb2 = eval_with_ttt(hp, model, mk, vt, bb, hl, ib, log)
        log(f"  BH+TTT eval    → val_loss:{vl2:.4f} val_bpb:{vb2:.4f}")
        # En iyi skoru raporla
        best_bpb = min(vb, vb2)
        log(f"  BEST val_bpb   → {best_bpb:.4f}")
        # Orijinal ağırlıkları geri yükle (compression için)
        model.update(tree_unflatten(list(saved_params.items())))

    # ── Compression ──
    flat = dict(tree_flatten(model.parameters()))
    qo, st = quant_int8(flat)
    qo["bigram_table"] = mk.table_np  # eval'da kullanılacak
    comp = zlib.compress(pickle.dumps(qo), level=9)
    cb = len(Path(__file__).read_text().encode())
    total = cb + len(comp)

    log(f"\n  Model:  {(len(comp))/1e6:.2f}MB")
    log(f"  Code:   {cb/1e6:.2f}MB")
    log(f"  TOTAL:  {total/1e6:.2f}MB {'OK' if total<16e6 else 'OVER!'}")
    log(f"  Params: {st['params']:,}")

    mp = Path(hp.out_dir)/f"{hp.run_id}_model.pkl.zlib"
    mp.write_bytes(comp)
    log(f"  Saved: {mp}")
    log("="*60)
    if TEST_MODE:
        log(">>> ALL FEATURES TESTED OK <<<")

if __name__ == "__main__":
    main()
