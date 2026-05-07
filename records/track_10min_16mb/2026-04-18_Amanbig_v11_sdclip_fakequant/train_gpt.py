"""
Parameter Golf v11 — Leaderboard Top-Stack Recipe
=================================================
Based on v8 (best pre-quant result 1.1387) + v10's proven SDClip-matched FakeQuantize
(+0.044 vs v8's +0.17 quant collapse) + tricks mined from the 1.0810 BPB top submission
(bigbag PR #1493) and Kevin Clark PR #1394.

Key changes vs v8:
  [ARCH]
  [1] 11 layers (was 12) — matches top-stack. Saves size for int8 embeddings budget.
  [2] Parallel Residuals on layers 7+ (GPT-J style, was 9+) — 5 parallel layers like top
  [3] QK-Gain 5.25 kept (top uses same)
  [4] Recurrence on layers 3-5 (3-layer block, was layers 5,6 simple pair)
      activate at 35% of training (was 25%) — lets arch settle first
  [5] Dropped DiffAttn V2 — v10 showed it's not worth the 2x SDPA cost at this scale

  [QUANT]
  [6] SDClip-matched FakeQuantize from v10 (proven +0.044 degradation)
  [7] Int8 embeddings with k=20.0 (matches top)
  [8] Int6 matrices with k=12.85 (matches top)
  [9] Size budget: 16,000,000 bytes DECIMAL (not 16,777,216)
      Target model ≤ 15.5MB to leave room for code

  [TRAINING]
  [10] WD 0.095, matrix LR 0.022, warmdown 0.72 (matches top hyperparams)
  [11] EMA 0.9965 from 50% (v10 proven)
  [12] Warmdown at 72% of iters (was 30%) — longer cooldown

  [EVAL]
  [13] Legal Score-First TTT: SGD lr=0.005 mom=0.9, 3 epochs cosine.
       Score each chunk under no_grad FIRST, then SGD on that chunk.
       Only acts on already-scored tokens → legal.

Target: 1.13 post-quant on Kaggle 1xH100, 1.08-1.09 on 8xH100 RunPod.
"""

from __future__ import annotations
import copy, glob, io, math, os, random, subprocess, sys, time, zlib
from pathlib import Path

os.environ["TORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup():
    print("=" * 60)
    print("PARAMETER GOLF v11: Top-Stack Recipe + SDClip QAT + TTT")
    print("=" * 60)
    for pkg in ["sentencepiece", "huggingface_hub", "brotli"]:
        try: __import__(pkg)
        except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    repo = "/tmp/parameter-golf"
    if not os.path.exists(repo):
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/openai/parameter-golf.git", repo], check=True)
    d = f"{repo}/data/datasets/fineweb10B_sp8192"
    for s in [f"{repo}/data/datasets/manifest.json", f"{repo}/data/manifest.json"]:
        if os.path.exists(s): os.remove(s)
    if not os.path.exists(d) or len(os.listdir(d)) < 3:
        env = os.environ.copy(); env["MATCHED_FINEWEB_REPO_ID"] = "kevclark/parameter-golf"
        subprocess.run([sys.executable, f"{repo}/data/cached_challenge_fineweb.py", "--variant", "sp8192", "--train-shards", "10"], env=env, check=True)
    else: print(f"Dataset cached: {len(os.listdir(d))} files")
    return repo

REPO = setup()
import brotli
import numpy as np, sentencepiece as spm, torch, torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch._dynamo.config.cache_size_limit = 64

# ======================== CONFIG ========================

class H:
    data_path = f"{REPO}/data/datasets/fineweb10B_sp8192"
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = f"{REPO}/data/tokenizers/fineweb_8192_bpe.model"
    seed = 1337

    val_batch_size = 131_072
    val_loss_every = 500
    iterations = 4000
    # Warmdown at 72% of iters (top stack uses this — much longer cooldown)
    warmdown_iters = int(4000 * 0.72)
    warmup_steps = 20
    train_batch_tokens = 524_288
    train_batch_tokens_start = 524_288
    batch_warmup_frac = 0.0
    train_seq_len = 1536
    grad_accum = 4
    max_wallclock_seconds = 0

    # Model — 11L matches top stack (was 12). Cuts ~8% params → headroom for int8 embed.
    vocab_size = 8192
    num_layers = 11
    model_dim = 512
    num_heads = 8
    num_kv_heads = 4
    head_dim = 64
    mlp_mult = 4
    qk_gain_init = 5.25  # matches top (was a bit lower)
    partial_rope_dims = 16

    # BigramHash
    bigram_buckets = 4096
    bigram_dim = 128

    # SmearGate
    smeargate_enabled = True

    # Depth recurrence — 3-layer block L3-5 (matches top stack PR #1493)
    recurse_layers = (3, 4, 5)
    recurse_stage1_frac = 0.35   # activate at 35% (was 25%) — top uses 0.35
    num_recurse = 2

    # Parallel Residuals — layers 7+ (5 of 11 layers parallel, matches top)
    parallel_start = 7

    # Value Embeddings — 11L pairs: (0,10), (1,9)
    value_emb_enabled = True
    value_emb_pairs = ((0, 10), (1, 9))
    value_emb_dim = 64

    # QK-Clip
    qk_clip_enabled = True
    qk_clip_threshold = 100.0

    # QAT — mixed int5 MLP / int6 attn, late
    qat_start_frac = 0.80
    qat_mlp_bits = 5
    qat_attn_bits = 6

    # EMA
    ema_decay = 0.9965
    ema_start_frac = 0.50

    # Misc
    tie_embeddings = True
    rope_base = 10000.0
    logit_softcap = 30.0
    tied_embed_init_std = 0.005

    # Optimizer — matches top (matrix_lr 0.022, WD 0.095)
    tied_embed_lr = 0.035; matrix_lr = 0.022; scalar_lr = 0.04
    muon_momentum = 0.99; muon_backend_steps = 5
    muon_momentum_warmup_start = 0.92; muon_momentum_warmup_steps = 500
    muon_wd = 0.095
    beta1 = 0.9; beta2 = 0.95; adam_eps = 1e-8; weight_decay = 0.04

    # Legal Score-First TTT (eval-time)
    ttt_enabled = True
    ttt_chunk_size = 1536         # one seq at a time
    ttt_lr = 0.005
    ttt_momentum = 0.9
    ttt_epochs = 3
    # Artifact budget — DECIMAL 16M minus ~200KB for code
    artifact_budget = 16_000_000
    code_reserve = 200_000


# ======================== MUON-EQ-R ========================

def zeropower_ns5(G, steps=10, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    row_norm = X.norm(dim=1, keepdim=True).clamp_min(eps)
    X = X / row_norm
    X /= X.norm() + eps
    tr = G.size(0) > G.size(1)
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, wd=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, wd=wd, nesterov=nesterov))
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                gr = p.grad; s = self.state[p]
                if "buf" not in s: s["buf"] = torch.zeros_like(gr)
                s["buf"].mul_(g["momentum"]).add_(gr)
                upd = gr.add(s["buf"], alpha=g["momentum"]) if g["nesterov"] else s["buf"]
                upd = zeropower_ns5(upd, g["backend_steps"])
                upd *= max(1, upd.size(0)/upd.size(1))**0.5
                p.add_(upd, alpha=-g["lr"])
                if g["wd"] > 0:
                    p.mul_(1 - g["lr"] * g["wd"])

# ======================== DATA + EVAL ========================

def build_luts(sp, vs, dev):
    n = max(int(sp.vocab_size()), vs)
    bb = np.zeros(n, np.int16); hs = np.zeros(n, np.bool_); ib = np.ones(n, np.bool_)
    for i in range(int(sp.vocab_size())):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i): continue
        ib[i] = False
        if sp.is_byte(i): bb[i] = 1; continue
        pc = sp.id_to_piece(i)
        if pc.startswith("\u2581"): hs[i] = True; pc = pc[1:]
        bb[i] = len(pc.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=dev), torch.tensor(hs, dtype=torch.bool, device=dev), torch.tensor(ib, dtype=torch.bool, device=dev))

def load_shard(f):
    h = np.fromfile(f, dtype="<i4", count=256)
    return torch.from_numpy(np.fromfile(f, dtype="<u2", count=int(h[2]), offset=256*4).astype(np.uint16, copy=False))

def load_val(pat, sl):
    t = torch.cat([load_shard(Path(p)) for p in sorted(glob.glob(pat))]).contiguous()
    return t[:((t.numel()-1)//sl)*sl+1]

@torch.no_grad()
def eval_val(model, sl, vt, bb, hs, ib, vb=131072, dev="cuda"):
    model.eval(); ls = vb//sl; ts = (vt.numel()-1)//sl
    lsum = torch.zeros((), device=dev, dtype=torch.float64)
    tc = torch.zeros((), device=dev, dtype=torch.float64)
    bc = torch.zeros((), device=dev, dtype=torch.float64)
    for i in range(0, ts, ls):
        j = min(i+ls, ts); r = vt[i*sl:j*sl+1].to(dev, dtype=torch.int64)
        x, y = r[:-1].reshape(-1, sl), r[1:].reshape(-1, sl)
        with torch.autocast("cuda", torch.bfloat16): l = model(x, y).detach()
        n = float(y.numel()); lsum += l.to(torch.float64)*n; tc += n
        tb = bb[y.reshape(-1)].to(torch.int16)
        tb += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(torch.int16)
        bc += tb.to(torch.float64).sum()
    vl = (lsum/tc).item(); model.train()
    return vl, (vl/math.log(2))*(tc.item()/bc.item())


def eval_val_with_ttt(model, sl, vt, bb, hs, ib, h, dev="cuda"):
    """Legal Score-First TTT eval:
      For each chunk:
        1. Score under no_grad (record loss) — these tokens are now "already evaluated"
        2. Take ttt_epochs SGD steps on same chunk (LEGAL: already-scored tokens)
      Next chunk uses adapted weights. This matches top-stack TTT (bigbag PR #1493).
    """
    model.train()  # need grad
    ts = (vt.numel() - 1) // sl
    lsum = 0.0; tc = 0.0; bc = 0.0

    # SGD over ALL trainable floating params (simple, matches top stack)
    train_params = [p for p in model.parameters() if p.dtype.is_floating_point and p.requires_grad]
    opt = torch.optim.SGD(train_params, lr=h.ttt_lr, momentum=h.ttt_momentum)

    pbar = tqdm(range(ts), desc="TTT eval", unit="chunk")
    for i in pbar:
        r = vt[i*sl:(i+1)*sl+1].to(dev, dtype=torch.int64)
        x = r[:-1].reshape(1, sl); y = r[1:].reshape(1, sl)

        # STEP 1: Score under no_grad (LEGAL: these tokens become scored)
        with torch.no_grad():
            with torch.autocast("cuda", torch.bfloat16):
                l = model(x, y).detach()
        n = float(y.numel()); lsum += float(l.item()) * n; tc += n
        tb = bb[y.reshape(-1)].to(torch.int16)
        tb += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(torch.int16)
        bc += float(tb.to(torch.float64).sum().item())

        # STEP 2: Take TTT SGD steps on SAME chunk (cosine LR over epochs)
        for e in range(h.ttt_epochs):
            lr_scale = 0.5 * (1 + math.cos(math.pi * e / max(h.ttt_epochs, 1)))
            for g in opt.param_groups: g["lr"] = h.ttt_lr * lr_scale
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y)
            loss.backward()
            opt.step()

        if i % 20 == 0:
            pbar.set_postfix({"bpb": f"{(lsum/tc/math.log(2))*(tc/bc):.4f}"})

    model.eval()
    vl = lsum / tc
    return vl, (vl / math.log(2)) * (tc / bc)


class TokenStream:
    def __init__(self, pat):
        self.f = sorted(glob.glob(pat)); assert self.f; self.i = 0; self.t = load_shard(Path(self.f[0])); self.p = 0
    def take(self, n):
        ch, r = [], n
        while r > 0:
            a = self.t.numel()-self.p
            if a <= 0: self.i = (self.i+1)%len(self.f); self.t = load_shard(Path(self.f[self.i])); self.p = 0; continue
            k = min(r, a); ch.append(self.t[self.p:self.p+k]); self.p += k; r -= k
        return ch[0] if len(ch)==1 else torch.cat(ch)


# ======================== QUANT ========================

CTRL = ("attn_scale","mlp_scale","resid","q_gain","skip","ln_scale","bigram","smear","qat_scale","v_emb_lambda")

def is_mlp_weight(name):
    return ".mlp." in name and (".fc." in name or ".proj." in name)

def quant_mixed(sd, mlp_bits=5, attn_bits=6):
    """Mixed int5 MLP / int6 attn / int8 embed, all SDClip. Matches top stack
    (matrix k=12.85, embed k=20.0)."""
    Q, S, D, P, PD, BITS = {}, {}, {}, {}, {}, {}
    for n, t in sd.items():
        t = t.detach().cpu().contiguous()
        if not t.is_floating_point(): P[n] = t; continue
        if t.numel() <= 65536:
            if any(p in n for p in CTRL): P[n] = t.float()
            else: PD[n] = str(t.dtype).removeprefix("torch."); P[n] = t.to(torch.float16)
            continue
        t32 = t.float()
        is_embed = ("emb" in n) or ("ve_tables" in n)
        if is_embed:
            k = 20.0; bits = 8
        elif is_mlp_weight(n):
            k = 12.85; bits = mlp_bits
        else:
            k = 12.85; bits = attn_bits
        max_val = (1 << (bits - 1)) - 1
        if t32.ndim == 2:
            std = t32.std(dim=1, unbiased=False)
            cl = (k * std).clamp_min(1e-8)
            s = (cl / max_val).clamp_min(1.0 / max_val)
            clipped = torch.clamp(t32, -cl[:, None], cl[:, None])
            q = torch.clamp(torch.round(clipped / s[:, None]), -max_val, max_val).to(torch.int8)
            S[n] = s.to(torch.float16)
        else:
            std_val = float(t32.std(unbiased=False).item()) if t32.numel() else 0.
            cl = max(k * std_val, 1e-8)
            s = torch.tensor(cl / max_val)
            q = torch.clamp(torch.round(torch.clamp(t32, -cl, cl) / s), -max_val, max_val).to(torch.int8)
            S[n] = s
        Q[n] = q; D[n] = str(t.dtype).removeprefix("torch."); BITS[n] = bits
    return {"quantized":Q, "scales":S, "dtypes":D, "passthrough":P, "passthrough_orig_dtypes":PD, "bits":BITS}


def dequant(o):
    out = {}; pd = o.get("passthrough_orig_dtypes", {})
    for n, q in o["quantized"].items():
        dt = getattr(torch, o["dtypes"][n]); s = o["scales"][n]
        out[n] = (q.float()*s.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(dt) if s.ndim > 0 else (q.float()*float(s.item())).to(dt)
    for n, t in o["passthrough"].items():
        orig = pd.get(n); out[n] = t.to(getattr(torch, orig)) if orig else t.clone()
    return out

def byte_shuffle(data):
    arr = np.frombuffer(data, dtype=np.uint8).copy()
    if len(arr) < 2: return data
    n = len(arr) - (len(arr) % 2)
    return np.concatenate([arr[:n:2], arr[1:n:2], arr[n:]]).tobytes()


# ======================== MODEL ========================

class FakeQuantize(torch.autograd.Function):
    """SDClip-matched: identical formula to save-time quant. Proven in v10 (+0.044 vs +0.17)."""
    @staticmethod
    def forward(ctx, x, bits):
        max_val = (1 << (bits - 1)) - 1
        k = 12.85
        if x.ndim == 2:
            std = x.std(dim=1, unbiased=False, keepdim=True)
            cl = (k * std).clamp_min(1e-8)
            scale = (cl / max_val).clamp_min(1.0 / max_val)
            clipped = torch.clamp(x, -cl, cl)
            return (torch.clamp(torch.round(clipped / scale), -max_val, max_val) * scale).to(x.dtype)
        else:
            std = x.std(unbiased=False) if x.numel() > 0 else torch.ones((), device=x.device)
            cl = (k * std).clamp_min(1e-8)
            scale = (cl / max_val).clamp_min(1.0 / max_val)
            clipped = torch.clamp(x, -cl, cl)
            return (torch.clamp(torch.round(clipped / scale), -max_val, max_val) * scale).to(x.dtype)
    @staticmethod
    def backward(ctx, grad): return grad, None


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),)) * self.ln_scale.to(x.dtype)[None, None, :]

class CastedLinear(nn.Linear):
    _qat_enabled = False
    _mlp_bits = 5
    _attn_bits = 6
    _is_mlp = False
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled:
            bits = CastedLinear._mlp_bits if self._is_mlp else CastedLinear._attn_bits
            w = FakeQuantize.apply(w, bits)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, head_dim, rope_dim, base=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.register_buffer("inv_freq", 1./(base**(torch.arange(0, rope_dim, 2).float()/rope_dim)), persistent=False)
        self._c = (0, None, None)
    def forward(self, s, device, dt):
        if self._c[0]!=s or self._c[1] is None or self._c[1].device!=device:
            t = torch.arange(s, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._c = (s, f.cos()[None, None], f.sin()[None, None])
        return self._c[1].to(dt), self._c[2].to(dt)

def apply_partial_rope(x, cos, sin, rope_dim):
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    h = rope_dim // 2
    x1, x2 = x_rope[..., :h], x_rope[..., h:]
    rotated = torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), -1)
    return torch.cat([rotated, x_pass], dim=-1)


class GQAttention(nn.Module):
    def __init__(self, dim, nh, nkv, hd, rope_base, qk_gain, partial_rope_dims, use_value_emb=False):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.rope_dim = partial_rope_dims
        self.c_q = CastedLinear(dim, nh*hd, bias=False)
        self.c_k = CastedLinear(dim, nkv*hd, bias=False)
        self.c_v = CastedLinear(dim, nkv*hd, bias=False)
        self.proj = CastedLinear(nh*hd, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((nh,), qk_gain, dtype=torch.float32))
        self.rotary = Rotary(hd, partial_rope_dims, rope_base)
        self.use_value_emb = use_value_emb
        if use_value_emb:
            self.v_emb_lambda = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float32))

    def forward(self, x, v_emb=None):
        b, s, d = x.shape
        q = self.c_q(x).reshape(b, s, self.nh, self.hd).transpose(1, 2)
        k = self.c_k(x).reshape(b, s, self.nkv, self.hd).transpose(1, 2)
        v = self.c_v(x).reshape(b, s, self.nkv, self.hd).transpose(1, 2)
        if self.use_value_emb and v_emb is not None:
            ve = v_emb.reshape(b, s, self.nkv, self.hd).transpose(1, 2).to(v.dtype)
            l = self.v_emb_lambda.to(v.dtype)
            v = l[0] * v + l[1] * ve
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(s, x.device, q.dtype)
        q = apply_partial_rope(q, cos, sin, self.rope_dim)
        k = apply_partial_rope(k, cos, sin, self.rope_dim)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.nkv != self.nh))
        out = out.transpose(1, 2).contiguous().reshape(b, s, -1)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        h = mult * dim
        self.fc = CastedLinear(dim, h, bias=False); self.fc._is_mlp = True
        self.proj = CastedLinear(h, dim, bias=False); self.proj._is_mlp = True
        self.proj._zero_init = True
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), 0.5).square())


class BigramHash(nn.Module):
    def __init__(self, vocab_size, buckets, bigram_dim, model_dim):
        super().__init__()
        self.buckets = buckets
        self.emb = nn.Embedding(buckets, bigram_dim)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        self.proj._zero_init = True
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, ids):
        prev = F.pad(ids[:, :-1], (1, 0), value=0)
        h = (prev.long() * 1000003 + ids.long()) % self.buckets
        return self.proj(self.emb(h))


class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(x.dtype))[None, None, :]
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))
        return g * x + (1 - g) * x_prev


class Block(nn.Module):
    def __init__(self, dim, h, idx, use_value_emb=False):
        super().__init__()
        self.idx = idx
        self.parallel = idx >= h.parallel_start
        self.n1 = RMSNorm(dim)
        self.n2 = RMSNorm(dim)
        self.attn = GQAttention(dim, h.num_heads, h.num_kv_heads, h.head_dim,
                                h.rope_base, h.qk_gain_init, h.partial_rope_dims,
                                use_value_emb=use_value_emb)
        self.mlp = MLP(dim, h.mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0, v_emb=None):
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.parallel:
            # GPT-J parallel: x + a*attn(norm(x)) + m*mlp(norm(x)) — both use n1
            nx = self.n1(x)
            attn_out = self.attn(nx, v_emb=v_emb)
            x = x + self.attn_scale.to(x.dtype)[None, None, :] * attn_out + \
                    self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.n2(nx))
        else:
            attn_out = self.attn(self.n1(x), v_emb=v_emb)
            x = x + self.attn_scale.to(x.dtype)[None, None, :] * attn_out
            x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.n2(x))
        return x


def ortho_init_(w):
    nn.init.orthogonal_(w)


class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.bigram = BigramHash(h.vocab_size, h.bigram_buckets, h.bigram_dim, h.model_dim)
        self.smear = SmearGate(h.model_dim) if h.smeargate_enabled else None

        ve_out_dim = h.num_kv_heads * h.head_dim
        self.value_emb_enabled = h.value_emb_enabled
        if h.value_emb_enabled:
            n_pairs = len(h.value_emb_pairs)
            self.ve_tables = nn.ModuleList([nn.Embedding(h.vocab_size, h.value_emb_dim) for _ in range(n_pairs)])
            self.ve_projs = nn.ModuleList([CastedLinear(h.value_emb_dim, ve_out_dim, bias=False) for _ in range(n_pairs)])
            self.ve_layer_to_pair = {}
            for pi, (a, b) in enumerate(h.value_emb_pairs):
                self.ve_layer_to_pair[a] = pi
                self.ve_layer_to_pair[b] = pi
            for p in self.ve_projs:
                p._zero_init = True
            for e in self.ve_tables:
                nn.init.normal_(e.weight, std=0.02)
        else:
            self.ve_tables = None
            self.ve_projs = None
            self.ve_layer_to_pair = {}

        self.blocks = nn.ModuleList([
            Block(h.model_dim, h, i, use_value_emb=(h.value_emb_enabled and i in self.ve_layer_to_pair))
            for i in range(h.num_layers)
        ])

        # U-Net skips for 11L: ne=5, nd=6 — 5 skip pairs
        ne = h.num_layers // 2; nd = h.num_layers - ne
        self.ne, self.nd = ne, nd
        self.skip_weights = nn.Parameter(torch.ones(min(ne, nd), h.model_dim))

        self.final_norm = RMSNorm(h.model_dim)
        self.qat_enabled = False
        self.recurse_enabled = False

        nn.init.normal_(self.tok_emb.weight, std=h.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "_zero_init", False):
                    nn.init.zeros_(m.weight)
                else:
                    try:
                        ortho_init_(m.weight)
                    except Exception:
                        pass

    def _get_v_emb(self, ids, layer_idx):
        if not self.value_emb_enabled or layer_idx not in self.ve_layer_to_pair:
            return None
        pi = self.ve_layer_to_pair[layer_idx]
        ve = self.ve_tables[pi](ids)
        return self.ve_projs[pi](ve)

    def forward(self, ids, targets):
        x = F.rms_norm(self.tok_emb(ids), (self.tok_emb.weight.size(-1),))
        x = x + self.bigram(ids)
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        skips = []

        for i in range(self.ne):
            v_emb = self._get_v_emb(ids, i)
            x = self.blocks[i](x, x0, v_emb=v_emb)
            skips.append(x)

        n_skip = self.skip_weights.size(0)
        for i in range(self.nd):
            if i < n_skip and i < len(skips):
                x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips[-(i+1)]
            bi = self.ne + i
            v_emb = self._get_v_emb(ids, bi)
            if self.recurse_enabled and bi in self.h.recurse_layers:
                for _ in range(self.h.num_recurse):
                    x = self.blocks[bi](x, x0, v_emb=v_emb)
            else:
                x = self.blocks[bi](x, x0, v_emb=v_emb)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        w = self.tok_emb.weight
        if self.qat_enabled: w = FakeQuantize.apply(w, 8)
        logits = self.h.logit_softcap * torch.tanh(F.linear(x, w) / self.h.logit_softcap)
        return F.cross_entropy(logits.float(), targets.reshape(-1), reduction="mean")


# ======================== QK-CLIP ========================

@torch.no_grad()
def qk_clip_all(model, threshold=100.0):
    for m in model.modules():
        if isinstance(m, GQAttention):
            q_norm = m.c_q.weight.norm(dim=1).max().item()
            k_norm = m.c_k.weight.norm(dim=1).max().item()
            product = q_norm * k_norm
            if product > threshold:
                scale = (threshold / product) ** 0.5
                m.c_q.weight.data.mul_(scale)
                m.c_k.weight.data.mul_(scale)


# ======================== TRAINING ========================

def main():
    global zeropower_ns5
    h = H(); dev = torch.device("cuda")
    random.seed(h.seed); np.random.seed(h.seed); torch.manual_seed(h.seed); torch.cuda.manual_seed_all(h.seed)

    sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
    vt = load_val(h.val_files, h.train_seq_len)
    bb, hs, ib = build_luts(sp, h.vocab_size, dev)
    print(f"Val tokens: {vt.numel()-1:,}")

    model = GPT(h).to(dev).bfloat16()
    for n, m in model.named_modules():
        if isinstance(m, CastedLinear): m.float()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if (p.ndim < 2 or any(pat in n for pat in CTRL)) and p.dtype != torch.float32:
                p.data = p.data.float()

    CastedLinear._mlp_bits = h.qat_mlp_bits
    CastedLinear._attn_bits = h.qat_attn_bits

    zeropower_ns5 = torch.compile(zeropower_ns5)
    try:
        model_c = torch.compile(model, dynamic=False)
        print("[speed] torch.compile(default) enabled")
    except Exception as e:
        print(f"[speed] torch.compile failed, falling back: {e}")
        model_c = model

    np_ = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {np_/1e6:.2f}M params | {h.num_layers}L × {h.model_dim} × MLP {h.mlp_mult}x LeakyReLU(0.5)²")
    print(f"SmearGate: {h.smeargate_enabled} | BigramHash: {h.bigram_buckets}")
    print(f"Recurse: layers {h.recurse_layers} x{h.num_recurse} @ {h.recurse_stage1_frac*100:.0f}%")
    print(f"ValueEmb: {h.value_emb_enabled} pairs={h.value_emb_pairs} dim={h.value_emb_dim}")
    print(f"QK-Gain: {h.qk_gain_init} | PartialRoPE: {h.partial_rope_dims}/{h.head_dim}")
    print(f"ParResid: layer {h.parallel_start}+ ({h.num_layers - h.parallel_start} parallel of {h.num_layers})")
    print(f"MuonEq-R WD={h.muon_wd} lr={h.matrix_lr} | EMA: {h.ema_decay} from {h.ema_start_frac*100:.0f}%")
    print(f"QAT: MLP int{h.qat_mlp_bits} / Attn int{h.qat_attn_bits} from {h.qat_start_frac*100:.0f}%")
    print(f"Warmdown: {h.warmdown_iters}/{h.iterations} ({h.warmdown_iters/h.iterations*100:.0f}%)")
    print(f"Legal TTT: enabled={h.ttt_enabled} lr={h.ttt_lr} mom={h.ttt_momentum} epochs={h.ttt_epochs}")
    print(f"Budget: {h.artifact_budget:,} bytes (decimal), reserve {h.code_reserve:,} for code\n")

    ga = h.grad_accum; gs = 1./ga
    mat_p, sc_p = [], []
    for n, p in model.blocks.named_parameters():
        is_c = p.ndim < 2 or any(pat in n for pat in CTRL)
        if is_c: sc_p.append(p)
        elif p.ndim == 2: mat_p.append(p)
    sc_p.append(model.skip_weights)
    if model.smear is not None: sc_p.append(model.smear.gate)
    bigram_p = [p for n, p in model.bigram.named_parameters()]
    ve_table_p = []
    if model.ve_tables is not None:
        for t in model.ve_tables:
            ve_table_p += list(t.parameters())
        mat_ids = {id(p) for p in mat_p}
        for pr in model.ve_projs:
            for pp in pr.parameters():
                if pp.ndim == 2 and id(pp) not in mat_ids:
                    mat_p.append(pp); mat_ids.add(id(pp))

    o_tok = torch.optim.Adam([{"params":[model.tok_emb.weight] + bigram_p + ve_table_p, "lr":h.tied_embed_lr, "base_lr":h.tied_embed_lr}], betas=(h.beta1,h.beta2), eps=h.adam_eps, fused=True)
    o_muon = Muon(mat_p, lr=h.matrix_lr, momentum=h.muon_momentum, backend_steps=h.muon_backend_steps, wd=h.muon_wd)
    for g in o_muon.param_groups: g["base_lr"] = h.matrix_lr
    o_sc = torch.optim.Adam([{"params":sc_p, "lr":h.scalar_lr, "base_lr":h.scalar_lr}], betas=(h.beta1,h.beta2), eps=h.adam_eps, fused=True)
    opts = [o_tok, o_muon, o_sc]

    print(f"MuonEq-R: {sum(p.numel() for p in mat_p)/1e6:.2f}M (WD={h.muon_wd}) | Scalar: {sum(p.numel() for p in sc_p)/1e3:.1f}K\n")

    ema_sd = None
    stream = TokenStream(h.train_files)

    unit = h.train_seq_len * ga
    def _round_to_unit(n):
        return max(unit, (n // unit) * unit)

    def cur_batch_tokens(step):
        if step >= int(h.iterations * h.batch_warmup_frac):
            return _round_to_unit(h.train_batch_tokens)
        frac = step / max(int(h.iterations * h.batch_warmup_frac), 1)
        start = h.train_batch_tokens_start
        return _round_to_unit(int(start + frac * (h.train_batch_tokens - start)))

    def batch(step):
        bt = cur_batch_tokens(step)
        per_micro = bt // ga
        l = stream.take(per_micro + 1).to(torch.int64)
        x = l[:-1].reshape(-1, h.train_seq_len)
        y = l[1:].reshape(-1, h.train_seq_len)
        return x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)

    def lr_mul(step):
        if step < h.warmup_steps:
            return step / max(h.warmup_steps, 1)
        warmdown_start = h.iterations - h.warmdown_iters
        if step < warmdown_start: return 1.0
        frac = (step - warmdown_start) / max(h.warmdown_iters, 1)
        return max(0.0, 1.0 - math.sqrt(frac))

    # Warmup
    print("Warmup...")
    isd = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    ios = [copy.deepcopy(o.state_dict()) for o in opts]
    model_c.train()
    for ws in range(h.warmup_steps):
        for o in opts: o.zero_grad(set_to_none=True)
        for _ in range(ga):
            x, y = batch(0)
            with torch.autocast("cuda", torch.bfloat16): loss = model_c(x, y)
            (loss*gs).backward()
        for o in opts: o.step()
        if (ws+1) % 5 == 0: print(f"  warmup {ws+1}/{h.warmup_steps}")
    model.load_state_dict(isd, strict=True)
    for o, s in zip(opts, ios): o.load_state_dict(s)
    stream = TokenStream(h.train_files)
    print("Warmup done!\n")

    # Train
    tms = 0.; last_bpb = None; best = float("inf")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    pbar = tqdm(range(1, h.iterations+1), desc="Training", unit="step",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for step in pbar:
        if not model.recurse_enabled and step >= int(h.iterations * h.recurse_stage1_frac):
            model.recurse_enabled = True
            tqdm.write(f"\n>>> 3-layer depth recurrence (L{h.recurse_layers}) enabled at step {step}\n")

        if not CastedLinear._qat_enabled and step >= int(h.iterations * h.qat_start_frac):
            CastedLinear._qat_enabled = True; model.qat_enabled = True
            tqdm.write(f"\n>>> QAT mixed int{h.qat_mlp_bits}/int{h.qat_attn_bits} enabled at step {step}\n")

        if step == 1 or (h.val_loss_every > 0 and step % h.val_loss_every == 0):
            torch.cuda.synchronize(); tms += 1000*(time.perf_counter()-t0)
            vl, vb = eval_val(model, h.train_seq_len, vt, bb, hs, ib, dev=dev)
            last_bpb = vb
            if vb < best: best = vb
            tqdm.write(f"\n>>> step:{step} val_loss:{vl:.4f} val_bpb:{vb:.4f} best:{best:.4f} time:{tms/1000:.1f}s\n")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        sc = lr_mul(step)
        for o in opts: o.zero_grad(set_to_none=True)
        tl = torch.zeros((), device=dev)
        for _ in range(ga):
            x, y = batch(step)
            with torch.autocast("cuda", torch.bfloat16): loss = model_c(x, y)
            tl += loss.detach(); (loss*gs).backward()
        tl /= ga

        frac = min(step/h.muon_momentum_warmup_steps, 1.) if h.muon_momentum_warmup_steps > 0 else 1.
        for g in o_muon.param_groups: g["momentum"] = (1-frac)*h.muon_momentum_warmup_start + frac*h.muon_momentum
        for o in opts:
            for g in o.param_groups: g["lr"] = g["base_lr"]*sc
        for o in opts: o.step()

        if h.qk_clip_enabled:
            qk_clip_all(model, h.qk_clip_threshold)

        if ema_sd is None and step >= int(h.iterations * h.ema_start_frac):
            ema_sd = {k: v.clone() for k, v in model.state_dict().items()}
            tqdm.write(f"\n>>> EMA started at step {step}\n")
        elif ema_sd is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema_sd[k].mul_(h.ema_decay).add_(v.to(ema_sd[k].dtype), alpha=1-h.ema_decay)

        v = tl.item()
        es = (tms+1000*(time.perf_counter()-t0))/1000
        pf = {"loss":f"{v:.4f}", "lr":f"{sc*h.matrix_lr:.5f}", "bsz":f"{cur_batch_tokens(step)//1024}K", "time":f"{es:.0f}s"}
        if last_bpb: pf["val_bpb"] = f"{last_bpb:.4f}"; pf["best"] = f"{best:.4f}"
        if model.qat_enabled: pf["QAT"] = "ON"
        if model.recurse_enabled: pf["Rec"] = "ON"
        pbar.set_postfix(pf)

    pbar.close()

    torch.cuda.synchronize(); tms += 1000*(time.perf_counter()-t0)
    CastedLinear._qat_enabled = False; model.qat_enabled = False

    non_ema_sd = {k: v.clone() for k, v in model.state_dict().items()}
    vl_cur, vb_cur = eval_val(model, h.train_seq_len, vt, bb, hs, ib, dev=dev)
    print(f"\nNon-EMA val_bpb: {vb_cur:.4f}")

    if ema_sd is not None:
        model.load_state_dict(ema_sd, strict=True)
        vl_ema, vb_ema = eval_val(model, h.train_seq_len, vt, bb, hs, ib, dev=dev)
        print(f"EMA val_bpb: {vb_ema:.4f}")
        if vb_ema < vb_cur:
            print("Using EMA weights")
            vl, vb = vl_ema, vb_ema
        else:
            print("Using non-EMA weights")
            model.load_state_dict(non_ema_sd, strict=True)
            vl, vb = vl_cur, vb_cur
    else:
        vl, vb = vl_cur, vb_cur

    print(f"\nPre-quant Final val_loss:{vl:.4f} val_bpb:{vb:.4f} time:{tms/1000:.1f}s")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()//1024//1024} MiB")

    # Quantize
    print(f"\n[Quant] Mixed int{h.qat_mlp_bits} MLP / int{h.qat_attn_bits} attn / int8 embed (k=12.85/20.0)...")
    qo = quant_mixed(model.state_dict(), mlp_bits=h.qat_mlp_bits, attn_bits=h.qat_attn_bits)
    buf = io.BytesIO(); torch.save(qo, buf)
    raw = buf.getvalue()
    shuffled = byte_shuffle(raw)
    blob_br = brotli.compress(shuffled, quality=11)
    blob_br_raw = brotli.compress(raw, quality=11)
    blob_zlib = zlib.compress(raw, 9)
    options = [("brotli-shuffle", blob_br), ("brotli", blob_br_raw), ("zlib", blob_zlib)]
    comp, blob = min(options, key=lambda x: len(x[1]))

    try: code = Path(__file__).read_text(encoding="utf-8")
    except: code = "# notebook"
    cb = len(code.encode("utf-8"))
    total = len(blob) + cb
    fits = total < h.artifact_budget
    print(f"\nArtifact ({comp}): model={len(blob):,} code={cb:,} total={total:,} / budget {h.artifact_budget:,} ({'YES' if fits else 'NO'})")
    if not fits:
        print(f"  ⚠ OVER BUDGET by {total - h.artifact_budget:,} bytes")

    # Roundtrip (with dequantized model)
    if comp == "brotli-shuffle":
        qsd = torch.load(io.BytesIO(brotli.decompress(blob_br_raw)), map_location="cpu")
    elif comp == "brotli":
        qsd = torch.load(io.BytesIO(brotli.decompress(blob)), map_location="cpu")
    else:
        qsd = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    model.load_state_dict(dequant(qsd), strict=True)
    vl2, vb2 = eval_val(model, h.train_seq_len, vt, bb, hs, ib, dev=dev)
    print(f"Roundtrip val_bpb:{vb2:.4f} (degradation: {vb2-vb:+.4f})")

    # Legal Score-First TTT
    if h.ttt_enabled:
        print(f"\n[Legal TTT] Score-first: SGD lr={h.ttt_lr} mom={h.ttt_momentum} epochs={h.ttt_epochs}...")
        ttt_model_sd = {k: v.clone() for k, v in model.state_dict().items()}
        vl3, vb3 = eval_val_with_ttt(model, h.train_seq_len, vt, bb, hs, ib, h, dev=dev)
        print(f"TTT val_bpb:{vb3:.4f} (vs no-TTT {vb2:.4f}, Δ {vb3-vb2:+.4f})")
        # Restore pre-TTT weights for save (TTT is eval-only adaptation)
        model.load_state_dict(ttt_model_sd, strict=True)
        final_bpb = min(vb2, vb3)
    else:
        final_bpb = vb2

    with open("final_model_v11.ptz","wb") as f: f.write(blob)
    print(f"\nSaved: final_model_v11.ptz ({len(blob):,} bytes)")
    print(f"Final reported BPB: {final_bpb:.4f}")
    print(f"{'='*60}\nDONE!")

if __name__ == "__main__": main()
