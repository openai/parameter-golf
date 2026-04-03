#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  RAKI TRAINING v3 — GOD MODE                                    ║
║                                                                  ║
║  Top 5'ten alınan kanıtlanmış teknikler:                         ║
║  1) BigramHash → logits'e Markov bias (top 4/5 kullanıyor)      ║
║  2) EMA/SWA → son fazda ağırlık ortalaması (top 3 kullanıyor)   ║
║  3) 10 layer (top 5'in hepsi 10-11 layer)                       ║
║  4) Muon weight decay 0.04 (top 5'in hepsi)                     ║
║  5) Sliding window eval (stride=64, ~0.02 BPB bedava)           ║
║  6) Rakı curriculum (bizim unique katkımız)                      ║
║                                                                  ║
║  Hedef: baseline ~1.22 → v3 ~1.15-1.17 BPB                     ║
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

# ── Rakı Schedule ──
RAKI_P1    = float(os.environ.get("RAKI_P1", 0.25))
RAKI_P2    = float(os.environ.get("RAKI_P2", 0.65))
RAKI_P3    = float(os.environ.get("RAKI_P3", 0.90))
RAKI_POWER = float(os.environ.get("RAKI_POWER", 0.3))

# ── EMA ──
EMA_START  = float(os.environ.get("EMA_START", 0.90))   # training'in %90'ından sonra
EMA_DECAY  = float(os.environ.get("EMA_DECAY", 0.995))  # decay rate

# ── Muon Weight Decay ──
MUON_WD    = float(os.environ.get("MUON_WD", 0.04))     # top submissions: 0.04

# ── Sliding Window Eval ──
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", 64))    # 64 token stride

# ==============================================================================
# HYPERPARAMETERS — 10 layer, 512 dim
# ==============================================================================
class HP:
    data_path       = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path  = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id          = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed            = int(os.environ.get("SEED", 1337))
    iterations      = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 10))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps   = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len      = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    mlx_micro_tokens   = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    mlx_eager          = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps       = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters     = int(os.environ.get("WARMDOWN_ITERS", 3500))  # top3: 3500
    max_wall_sec       = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    # ── Model: 10 layers (top5 hep 10-11) ──
    vocab_size  = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers  = int(os.environ.get("NUM_LAYERS", 10))     # baseline:9 → 10
    model_dim   = int(os.environ.get("MODEL_DIM", 512))
    num_heads   = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult    = int(os.environ.get("MLP_MULT", 2))
    embed_std   = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    softcap     = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base   = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain     = float(os.environ.get("QK_GAIN_INIT", 1.5))
    # ── Optimizer ──
    beta1       = float(os.environ.get("BETA1", 0.9))
    beta2       = float(os.environ.get("BETA2", 0.95))
    eps         = float(os.environ.get("ADAM_EPS", 1e-8))
    embed_lr    = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr   = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr   = float(os.environ.get("SCALAR_LR", 0.04))
    muon_mom    = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_steps  = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    out_dir     = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self): return f"{self.data_path}/fineweb_train_*.bin"
    @property
    def val_files(self): return f"{self.data_path}/fineweb_val_*.bin"
    @property
    def micro_tokens(self): return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step, ms):
        if self.warmdown_iters <= 0: return 1.0
        if self.max_wall_sec <= 0:
            ws = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if ws <= step else 1.0
        sms = ms / max(step, 1)
        wms = self.warmdown_iters * sms
        rms = max(1000 * self.max_wall_sec - ms, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

CTRL = tuple("attn_scale,mlp_scale,resid_mix,q_gain,skip_weights".split(","))

# ==============================================================================
# INNOVATION #1: BIGRAM HASH → LOGIT BIAS
# ==============================================================================
class BigramHash:
    """
    Bigram log-olasılıklarını hesapla ve model logits'ine ekle.
    Transformer sadece Markov'un açıklayamadığı kısmı öğrenir.

    Top submissions bunu "BigramHash(10240)" olarak kullanıyor.
    Bizim vocab 1024 → full tablo 1024x1024 = 2MB (fp16).
    """
    def __init__(self, vocab_size):
        self.V = vocab_size
        self.counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)

    def update(self, tokens):
        np.add.at(self.counts, (tokens[:-1], tokens[1:]), 1.0)

    def finalize(self):
        """Log-prob tablo → MLX array olarak sakla."""
        smoothed = self.counts + 0.01  # minimal smoothing
        log_probs = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        # float16'ya çevir (2MB)
        self.table_np = log_probs.astype(np.float16)
        self.table_mx = mx.array(self.table_np, dtype=mx.float16)
        del self.counts

    def get_bias(self, prev_tokens):
        """
        prev_tokens: (batch, seq_len) — son tokenlara göre bias döndür.
        return: (batch, seq_len, vocab) — logit bias
        """
        # prev_tokens'ın her pozisyonu için 1024-boyutlu bias vektörü
        return self.table_mx[prev_tokens]  # (batch, seq_len, vocab)

    def surprise_scores(self, x_np, y_np):
        """Curriculum için: batch surprise skoru."""
        lp = self.table_np[x_np.ravel(), y_np.ravel()].astype(np.float32)
        return -lp.reshape(x_np.shape).mean(axis=1)

# ==============================================================================
# INNOVATION #2: EMA (Exponential Moving Average)
# ==============================================================================
class EMATracker:
    """
    Training'in son fazında ağırlıkların hareketli ortalaması.
    Top 3 submission'ın hepsi kullanıyor.
    """
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = None  # lazy init
        self.active = False

    def activate(self, model):
        """EMA toplamaya başla."""
        if self.shadow is None:
            self.shadow = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
        self.active = True

    def update(self, model):
        """Her step'te çağır (aktifse)."""
        if not self.active or self.shadow is None:
            return
        d = self.decay
        for k, v in tree_flatten(model.parameters()):
            self.shadow[k] = d * self.shadow[k] + (1 - d) * v

    def apply(self, model):
        """Final evaluation öncesi EMA ağırlıklarını modele yükle."""
        if self.shadow is None:
            return
        model.update(tree_unflatten(list(self.shadow.items())))

# ==============================================================================
# INNOVATION #3: RAKI SCHEDULE (v2'den, düzeltilmiş)
# ==============================================================================
def raki_schedule(progress):
    p = max(0.0, min(1.0, progress))
    if p <= RAKI_P1:
        return RAKI_POWER * 0.1 * (p / RAKI_P1), "ayik"
    elif p <= RAKI_P2:
        t = (p - RAKI_P1) / (RAKI_P2 - RAKI_P1)
        return RAKI_POWER * (0.1 + 0.9 * t), "keyifli"
    elif p <= RAKI_P3:
        return RAKI_POWER, "kivam"
    else:
        t = (p - RAKI_P3) / (1.0 - RAKI_P3)
        return RAKI_POWER * (1.0 - 0.6 * t), "ayilma"

# ==============================================================================
# MATH + DATA
# ==============================================================================
def rms_norm(x, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x*x, axis=-1, keepdims=True) + eps)).astype(x.dtype)

def newtonschulz5(g, steps, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x*x)) + eps)
    tr = x.shape[0] > x.shape[1]
    if tr: x = x.T
    for _ in range(steps):
        am = x @ x.T
        x = a*x + (b*am + c*(am@am)) @ x
    if tr: x = x.T
    return x.astype(g.dtype)

def load_shard(path):
    h = np.fromfile(path, dtype="<i4", count=256)
    return np.fromfile(path, dtype="<u2", count=int(h[2]), offset=256*4).astype(np.int32, copy=False)

class TokenStream:
    def __init__(self, pattern, log_fn=None):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(pattern)
        self.epoch, self.fi, self.pos = 1, 0, 0
        self.log_fn = log_fn
        self.tok = load_shard(self.files[0])
    def _next(self):
        self.fi = (self.fi+1) % len(self.files)
        if self.fi == 0:
            self.epoch += 1
            if self.log_fn: self.log_fn(f"epoch:{self.epoch}")
        self.tok = load_shard(self.files[self.fi]); self.pos = 0
    def take(self, n):
        parts, left = [], n
        while left > 0:
            if self.pos >= self.tok.size: self._next()
            k = min(left, self.tok.size - self.pos)
            parts.append(self.tok[self.pos:self.pos+k])
            self.pos += k; left -= k
        return parts[0] if len(parts)==1 else np.concatenate(parts)

class Loader:
    def __init__(self, pattern, log_fn=None):
        self.s = TokenStream(pattern, log_fn)
    def next(self, bt, sl):
        u = (bt // sl) * sl
        c = self.s.take(u + 1)
        xn = c[:-1].reshape(-1, sl); yn = c[1:].reshape(-1, sl)
        return mx.array(xn, dtype=mx.int32), mx.array(yn, dtype=mx.int32), xn, yn

# ==============================================================================
# MODEL — BigramHash logit bias entegre
# ==============================================================================
class CL(nn.Module):
    """CastedLinear"""
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
        self.cq = CL(D, D); self.ck = CL(D, nkv*self.hd)
        self.cv = CL(D, nkv*self.hd); self.proj = CL(D, D)
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
        self.fc = CL(D, D*m); self.proj = CL(D*m, D)
    def __call__(self, x):
        h = nn.relu(self.fc(x)); return self.proj(h*h)  # relu²

class Block(nn.Module):
    def __init__(self, D, nh, nkv, m, rb, qkg):
        super().__init__()
        self.an, self.mn = Norm(), Norm()
        self.attn = Attn(D, nh, nkv, rb, qkg)
        self.mlp = MLP(D, m)
        self.asc = mx.ones((D,), dtype=mx.float32)
        self.msc = mx.ones((D,), dtype=mx.float32)
        self.rm = mx.array(np.stack((np.ones(D,dtype=np.float32), np.zeros(D,dtype=np.float32))))
    def __call__(self, x, x0):
        m = self.rm.astype(x.dtype)
        x = m[0][None,None,:]*x + m[1][None,None,:]*x0
        x = x + self.asc.astype(x.dtype)[None,None,:] * self.attn(self.an(x))
        x = x + self.msc.astype(x.dtype)[None,None,:] * self.mlp(self.mn(x))
        return x

class GPT(nn.Module):
    def __init__(self, hp, bigram_table_mx=None):
        """
        bigram_table_mx: (vocab, vocab) mx.array — BigramHash bias tablosu.
        Logits'e eklenir → transformer basit pattern'leri öğrenmek zorunda kalmaz.
        """
        super().__init__()
        V, L, D = hp.vocab_size, hp.num_layers, hp.model_dim
        self.lsc = hp.softcap
        self.bigram_bias = bigram_table_mx  # None ise kullanılmaz
        self.bigram_weight = 0.3  # ne kadar güçlü bias uygulansın

        self.tok_emb = nn.Embedding(V, D)
        self.n_enc = L // 2
        self.n_dec = L - self.n_enc
        self.n_skip = min(self.n_enc, self.n_dec)
        self.skip_weights = mx.ones((self.n_skip, D), dtype=mx.float32)
        self.blocks = [Block(D, hp.num_heads, hp.num_kv_heads, hp.mlp_mult, hp.rope_base, hp.qk_gain) for _ in range(L)]
        self.fn = Norm()
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * hp.embed_std).astype(COMPUTE_DTYPE)

    def forward(self, ids):
        x = rms_norm(self.tok_emb(ids).astype(COMPUTE_DTYPE))
        x0, skips = x, []
        for i in range(self.n_enc):
            x = self.blocks[i](x, x0); skips.append(x)
        for i in range(self.n_dec):
            if skips: x = x + self.skip_weights[i].astype(x.dtype)[None,None,:] * skips.pop()
            x = self.blocks[self.n_enc + i](x, x0)
        return self.fn(x)

    def logits(self, ids, prev_ids=None):
        """Logits hesapla + bigram bias ekle."""
        h = self.forward(ids)
        raw_logits = h @ self.tok_emb.weight.astype(h.dtype).T
        logits = self.lsc * mx.tanh(raw_logits / self.lsc)

        # ── BIGRAM BIAS: Markov olasılıklarını logits'e ekle ──
        if self.bigram_bias is not None and prev_ids is not None:
            # prev_ids: her pozisyon için bir önceki token
            # bias: (batch, seq_len, vocab)
            bias = self.bigram_bias[prev_ids].astype(logits.dtype)
            logits = logits + self.bigram_weight * bias

        return logits

    def loss(self, x, y, token_weights=None):
        """
        x: input_ids (batch, seq_len)
        y: target_ids (batch, seq_len)
        token_weights: opsiyonel (batch, seq_len) ağırlıklar

        Bigram bias logits'e otomatik eklenir.
        """
        # Prev tokens for bigram bias: x'in kendisi (x[t] → y[t] tahmininde x[t] prev)
        logits = self.logits(x, prev_ids=x)
        logits_flat = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
        tgt = y.reshape(-1)

        if token_weights is None:
            return nn.losses.cross_entropy(logits_flat, tgt, reduction="mean")

        per_token = nn.losses.cross_entropy(logits_flat, tgt, reduction="none")
        w = token_weights.reshape(-1)
        return mx.sum(per_token * w) / mx.sum(w)

# ==============================================================================
# OPTIMIZERS — Muon + Weight Decay
# ==============================================================================
class Muon:
    def __init__(self, keys, params, hp):
        self.keys, self.hp = keys, hp
        self.bufs = {k: mx.zeros_like(params[k]) for k in keys}
    def step(self, params, grads, step, lr_mul):
        h = self.hp
        if h.muon_warmup_steps:
            t = min(step / h.muon_warmup_steps, 1.0)
            mom = (1-t)*h.muon_warmup_start + t*h.muon_mom
        else: mom = h.muon_mom
        lr = h.matrix_lr * lr_mul
        out = {}
        for k in self.keys:
            p, g = params[k], grads[k]
            # ── WEIGHT DECAY (top submissions: 0.04) ──
            g = g + MUON_WD * p
            buf = mom * self.bufs[k] + g; self.bufs[k] = buf
            go = newtonschulz5(g + mom*buf, h.muon_steps)
            sc = math.sqrt(max(1.0, p.shape[0]/p.shape[1]))
            out[k] = p - lr*(go*sc).astype(p.dtype)
        return out

class Opt:
    def __init__(self, model, hp):
        self.hp = hp
        params = dict(tree_flatten(model.parameters()))
        self.ek = "tok_emb.weight"
        self.mk = [k for k,p in params.items() if k.startswith("blocks.") and p.ndim==2 and not any(c in k for c in CTRL)]
        self.sk = [k for k,p in params.items() if k=="skip_weights" or (k.startswith("blocks.") and (p.ndim<2 or any(c in k for c in CTRL)))]
        self.muon = Muon(self.mk, params, hp)
        self.ae = optim.Adam(learning_rate=hp.embed_lr, betas=[hp.beta1, hp.beta2], eps=hp.eps, bias_correction=True)
        self.asc = optim.Adam(learning_rate=hp.scalar_lr, betas=[hp.beta1, hp.beta2], eps=hp.eps, bias_correction=True)
    def step(self, model, gt, step, lr_mul):
        P = dict(tree_flatten(model.parameters()))
        G = dict(tree_flatten(gt))
        U = dict(P)
        U.update(self.muon.step(P, G, step=step, lr_mul=lr_mul))
        self.ae.learning_rate = self.hp.embed_lr * lr_mul
        U.update(self.ae.apply_gradients({self.ek: G[self.ek]}, {self.ek: P[self.ek]}))
        self.asc.learning_rate = self.hp.scalar_lr * lr_mul
        U.update(self.asc.apply_gradients({k:G[k] for k in self.sk}, {k:P[k] for k in self.sk}))
        model.update(tree_unflatten(list(U.items())))

# ==============================================================================
# QUANTIZATION
# ==============================================================================
def _f32(a): return np.array(a.astype(mx.float32), dtype=np.float32, copy=False)

def quant_int8(flat):
    Q, S, D, P, PD, QM = {}, {}, {}, {}, {}, {}
    stats = {"param_count":0, "bytes":0}
    for n, a in flat.items():
        stats["param_count"] += int(a.size)
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
# EVAL — Sliding Window (stride=64)
# ==============================================================================
def build_luts(sp, V):
    sv = int(sp.vocab_size()); ts = max(sv, V)
    bb = np.zeros(ts, dtype=np.int16)
    hl = np.zeros(ts, dtype=np.bool_)
    ib = np.ones(ts, dtype=np.bool_)
    for t in range(sv):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
        ib[t] = False
        if sp.is_byte(t): bb[t]=1; continue
        pc = sp.id_to_piece(t)
        if pc.startswith("▁"): hl[t]=True; pc=pc[1:]
        bb[t] = len(pc.encode("utf-8"))
    return bb, hl, ib

def load_val(pattern, sl):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(pattern)
    tok = np.ascontiguousarray(np.concatenate([load_shard(f) for f in files]))
    u = ((tok.size-1)//sl)*sl
    return tok[:u+1]

def eval_sliding(hp, loss_fn, vt, bb, hl, ib, stride=64, log_fn=None):
    """
    Sliding window evaluation — stride=64 ile overlap yaparak
    her token'ın daha uzun context'ten faydalanmasını sağlar.
    Top submission #15 bunu kullanarak ~0.02 BPB kazandı.
    """
    sl = hp.train_seq_len
    total_tokens = vt.size - 1
    # Stride ile kaç window var
    n_windows = max(1, (total_tokens - sl) // stride + 1)

    # Basit batched eval (sliding window overhead yüzünden batch küçük tut)
    batch_size = max(1, hp.val_batch_size // (sl * hp.grad_accum_steps))

    loss_sum = 0.0
    byte_sum = 0.0
    token_count = 0.0
    counted = np.zeros(total_tokens, dtype=np.bool_)

    for win_start in range(0, total_tokens - sl + 1, stride):
        win_end = win_start + sl
        chunk = vt[win_start:win_end + 1]
        xn = chunk[:-1].reshape(1, sl)
        yn = chunk[1:].reshape(1, sl)
        x = mx.array(xn, dtype=mx.int32)
        y = mx.array(yn, dtype=mx.int32)

        l = loss_fn(x, y).astype(mx.float32)
        mx.eval(l)

        # Sadece stride bölgesindeki (yeni) tokenleri say
        # İlk window: tüm tokenler yeni
        # Sonraki: sadece son 'stride' token yeni
        if win_start == 0:
            new_start, new_end = 0, sl
        else:
            new_start = sl - stride
            new_end = sl

        new_count = new_end - new_start
        # Approximate: window loss * yeni token oranı
        loss_sum += float(l.item()) * new_count
        token_count += new_count

        # Byte hesabı
        new_y = yn[0, new_start:new_end]
        new_x = xn[0, new_start:new_end]
        bn = bb[new_y].astype(np.int16)
        bn += (hl[new_y] & ~ib[new_x]).astype(np.int16)
        byte_sum += float(bn.astype(np.float64).sum())

        if log_fn and (win_start // stride) % 500 == 0:
            log_fn(f"  eval: {win_start}/{total_tokens}")

    if token_count == 0:
        return 999.0, 999.0
    vl = loss_sum / token_count
    vb = (vl / math.log(2)) * (token_count / byte_sum)
    return vl, vb

def eval_simple(hp, loss_fn, vt, bb, hl, ib):
    """Basit eval — sliding window olmadan (hızlı karşılaştırma için)."""
    sl = hp.train_seq_len
    vbt = hp.val_batch_size // hp.grad_accum_steps
    vbs = vbt // sl
    ts = (vt.size-1) // sl
    ls, tt, tb = 0.0, 0.0, 0.0
    for s in range(0, ts, vbs):
        e = min(s+vbs, ts)
        c = vt[s*sl:e*sl+1]
        xn, yn = c[:-1].reshape(-1,sl), c[1:].reshape(-1,sl)
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
    log(" RAKI v3 GOD MODE")
    log("="*60)

    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    vt = load_val(hp.val_files, hp.train_seq_len)
    bb, hl, ib = build_luts(sp, hp.vocab_size)

    # ── BigramHash tablosu ──
    log("BigramHash tablosu hesaplanıyor...")
    t0 = time.time()
    bh = BigramHash(hp.vocab_size)
    shards = sorted(glob.glob(hp.train_files))
    # Tüm shard'lardan bigram say (daha doğru tablo)
    for sf in shards:
        bh.update(load_shard(Path(sf)))
    bh.finalize()
    log(f"  BigramHash hazır: {time.time()-t0:.1f}s, {bh.table_np.nbytes/1e6:.1f}MB")

    # ── Model (BigramHash entegre) ──
    mx.random.seed(hp.seed)
    model = GPT(hp, bigram_table_mx=bh.table_mx)
    opt = Opt(model, hp)
    ema = EMATracker(model, EMA_DECAY)
    np_ = sum(int(np.prod(p.shape)) for _,p in tree_flatten(model.parameters()))
    log(f"  Model: {np_:,} params, {hp.num_layers}L {hp.model_dim}D")
    log(f"  BigramHash: aktif (logit bias)")
    log(f"  EMA: {EMA_START*100:.0f}%'dan sonra aktif, decay={EMA_DECAY}")
    log(f"  Muon WD: {MUON_WD}")
    log(f"  Rakı: P1={RAKI_P1} P2={RAKI_P2} P3={RAKI_P3} power={RAKI_POWER}")
    log(f"  Warmdown: {hp.warmdown_iters} iters")

    # ── Loss functions ──
    val_loss_fn = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    weighted_loss_grad = nn.value_and_grad(model, lambda x, y, tw: model.loss(x, y, token_weights=tw))
    plain_loss_grad = nn.value_and_grad(model, lambda x, y: model.loss(x, y))

    # ── Data ──
    loader = Loader(hp.train_files, log)

    # ── Warmup ──
    if hp.warmup_steps > 0:
        log(f"  Warmup {hp.warmup_steps} steps...")
        for _ in range(hp.warmup_steps):
            x, y, _, _ = loader.next(hp.micro_tokens, hp.train_seq_len)
            l, g = plain_loss_grad(x, y); mx.eval(l, g)
        loader = Loader(hp.train_files, log)
        log("  Warmup OK")

    # ── TRAINING ──
    log(f"\nTraining: {hp.iterations} iters, {hp.max_wall_sec}s cap")
    log("-"*60)

    tms = 0.0
    cap = 1000*hp.max_wall_sec if hp.max_wall_sec > 0 else None
    acc_fn = lambda a, g, s: ({k: v*s for k,v in dict(tree_flatten(g)).items()} if a is None
                              else {k: a.get(k, mx.zeros_like(v)) + v*s for k,v in dict(tree_flatten(g)).items()})

    for step in range(1, hp.iterations + 1):
        t0 = time.perf_counter()
        progress = min(tms/cap, 1.0) if cap else step/hp.iterations
        sw, phase = raki_schedule(progress)
        use_w = sw > 0.02

        # ── EMA activation ──
        if progress >= EMA_START and not ema.active:
            ema.activate(model)
            log(f"  >>> EMA activated at step {step} (progress={progress:.2f})")

        ga = None
        total_loss = 0.0
        gs = 1.0 / hp.grad_accum_steps

        for _ in range(hp.grad_accum_steps):
            x, y, xn, yn = loader.next(hp.micro_tokens, hp.train_seq_len)

            if use_w:
                surp = bh.surprise_scores(xn, yn)
                smin, smax = surp.min(), surp.max()
                ns = (surp-smin)/(smax-smin) if smax > smin else np.full_like(surp, 0.5)
                tw_np = np.broadcast_to((1.0 + sw*ns)[:,None], xn.shape).copy()
                tw = mx.array(tw_np, dtype=mx.float32)
                lv, grads = weighted_loss_grad(x, y, tw)
            else:
                lv, grads = plain_loss_grad(x, y)

            mx.eval(lv, grads)
            total_loss += float(lv.item()) * gs

            # Accumulate
            flat_g = dict(tree_flatten(grads))
            if ga is None:
                ga = {k: v*gs for k,v in flat_g.items()}
            else:
                for k,v in flat_g.items():
                    ga[k] = ga[k] + v*gs

            if hp.mlx_eager: mx.eval(ga)

        # ── Optimizer step ──
        lr_m = hp.lr_mul(step, tms)
        opt.step(model, tree_unflatten(list(ga.items())), step=step, lr_mul=lr_m)

        # ── EMA update ──
        ema.update(model)

        step_ms = (time.perf_counter()-t0)*1000
        tms += step_ms

        # ── Log ──
        if step == 1 or step % hp.train_log_every == 0 or step == hp.iterations:
            tps = hp.train_batch_tokens / (step_ms/1000)
            log(f"  {step:5d}/{hp.iterations} loss:{total_loss:.4f} {phase:8s} "
                f"sw:{sw:.3f} lr:{lr_m:.3f} ema:{'ON' if ema.active else '--'} "
                f"{tps:.0f}t/s {tms/1000:.1f}s")

        # ── Val ──
        if hp.val_loss_every > 0 and step % hp.val_loss_every == 0:
            vl, vb = eval_simple(hp, val_loss_fn, vt, bb, hl, ib)
            log(f"  >>> val_loss:{vl:.4f} val_bpb:{vb:.4f}")

        # ── Cap ──
        if cap and tms >= cap:
            log(f"  Wallclock {hp.max_wall_sec}s doldu.")
            break

    # ── EMA ağırlıklarını yükle ──
    if ema.active:
        log("\nEMA ağırlıkları yükleniyor...")
        ema.apply(model)

    # ── FINAL EVAL ──
    log(f"\n{'='*60}")
    log("FINAL EVALUATION")

    # Basit eval
    vl, vb = eval_simple(hp, val_loss_fn, vt, bb, hl, ib)
    log(f"  Simple eval  → val_loss:{vl:.4f}  val_bpb:{vb:.4f}")

    # Sliding window eval (daha iyi skor)
    # M1'de çok uzun sürebilir, sadece short run'larda skip
    if hp.iterations >= 500:
        log("  Sliding window eval hesaplanıyor (bu biraz sürer)...")
        vl2, vb2 = eval_sliding(hp, val_loss_fn, vt, bb, hl, ib, stride=EVAL_STRIDE, log_fn=log)
        log(f"  Sliding eval → val_loss:{vl2:.4f}  val_bpb:{vb2:.4f}")

    # ── Compression ──
    flat = dict(tree_flatten(model.parameters()))
    qo, st = quant_int8(flat)
    # BigramHash tablosunu da artifact'a ekle
    qo["bigram_table"] = bh.table_np
    comp = zlib.compress(pickle.dumps(qo), level=9)
    cb = len(Path(__file__).read_text().encode())
    total = cb + len(comp)

    log(f"\n  Model:  {(len(comp)-bh.table_np.nbytes)/1e6:.2f}MB")
    log(f"  Bigram: {bh.table_np.nbytes/1e6:.2f}MB")
    log(f"  Code:   {cb/1e6:.2f}MB")
    log(f"  TOTAL:  {total/1e6:.2f}MB {'✅ <16MB' if total<16e6 else '❌ >16MB!'}")
    log(f"  Params: {st['param_count']:,}")

    Path(hp.out_dir).mkdir(parents=True, exist_ok=True)
    mp = Path(hp.out_dir)/f"{hp.run_id}_model.pkl.zlib"
    mp.write_bytes(comp)
    log(f"  Saved: {mp}")
    log("="*60)

if __name__ == "__main__":
    main()
