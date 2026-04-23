"""
Weight-Shared Dynamic-L + INT2 TinyLoop — Parameter Golf Submission
Architecture matches TinyLoop C++ engine exactly:
  tokens → Embed[V,D] → Pre0 → Pre1 → [Loop × L] → LN → Head[D,V]
  SwiGLU activation, LayerNorm+bias, fused QKV, MHA, no RoPE
  Dynamic-L training: L ~ Poisson(L_mean), clipped [L_min, L_max]

Based on: "Quantization as the Final Training Step" (Yang 2026)
  - β < 1 for bottleneck weights → INT2 errors cancel across loop iterations
  - Dynamic-L prevents overthinking (stable L=4..48)
  - Quantization acts as regularization when model overfits
"""
from __future__ import annotations
import copy, glob, io, math, os, random, struct, subprocess, sys, time, uuid, zlib
from pathlib import Path
try:
    import zstandard; _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# ── FA3/FA2/SDPA ─────────────────────────────────────
try:
    from flash_attn_interface import flash_attn_func as _fa3
    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False

def flash_attn(q, k, v, causal=True):
    if _HAS_FA3:
        return _fa3(q, k, v, causal=causal)
    return F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
        is_causal=causal
    ).transpose(1,2)

# ── Config ───────────────────────────────────────────
class H:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))
    # Training
    iterations        = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters    = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps      = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens= int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len     = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock     = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    val_loss_every    = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    val_batch_size    = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every   = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    # Model — matches TinyLoop engine
    vocab_size  = int(os.environ.get("VOCAB_SIZE", 8192))
    dim         = int(os.environ.get("DIM", 896))
    n_heads     = int(os.environ.get("N_HEADS", 14))
    ffn_mult    = float(os.environ.get("FFN_MULT", 4.0))
    n_pre       = int(os.environ.get("N_PRE", 2))
    L_mean      = int(os.environ.get("L_MEAN", 12))
    L_min       = int(os.environ.get("L_MIN", 4))
    L_max       = int(os.environ.get("L_MAX", 40))
    L_eval      = int(os.environ.get("L_EVAL", 24))
    tie_weights = bool(int(os.environ.get("TIE_WEIGHTS", "1")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    # Optimizer
    lr          = float(os.environ.get("LR", 0.025))
    embed_lr    = float(os.environ.get("EMBED_LR", 0.035))
    wd          = float(os.environ.get("WD", 0.04))
    muon_mom    = float(os.environ.get("MUON_MOM", 0.97))
    muon_warmup = int(os.environ.get("MUON_WARMUP", 1500))
    muon_start  = float(os.environ.get("MUON_START", 0.92))
    grad_clip   = float(os.environ.get("GRAD_CLIP", 0.3))
    ema_decay   = float(os.environ.get("EMA_DECAY", 0.997))
    # Eval
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    # TTT
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr      = float(os.environ.get("TTT_LR", 0.0001))
    ttt_epochs  = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk   = int(os.environ.get("TTT_CHUNK", 65536))
    ttt_batch   = int(os.environ.get("TTT_BATCH", 32))
    # Export
    factor_dim  = int(os.environ.get("FACTOR_DIM", 96))

# ── Muon ─────────────────────────────────────────────
def ns5(G: Tensor, steps=5, eps=1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    tr = G.size(0) > G.size(1)
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, wd=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd))
    @torch.no_grad()
    def step(self):
        ws = dist.get_world_size() if dist.is_initialized() else 1
        rk = dist.get_rank() if dist.is_initialized() else 0
        for group in self.param_groups:
            ps = group["params"]
            if not ps: continue
            total = sum(p.numel() for p in ps)
            flat = torch.zeros(total, device=ps[0].device, dtype=torch.bfloat16)
            cur = 0
            for i, p in enumerate(ps):
                if i % ws == rk and p.grad is not None:
                    g = p.grad
                    s = self.state[p]
                    if "buf" not in s: s["buf"] = torch.zeros_like(g)
                    s["buf"].mul_(group["momentum"]).add_(g)
                    u = g.add(s["buf"], alpha=group["momentum"])
                    u = ns5(u) * max(1, u.size(0)/u.size(1))**0.5
                    flat[cur:cur+p.numel()] = u.reshape(-1)
                cur += p.numel()
            if ws > 1: dist.all_reduce(flat)
            cur = 0
            for p in ps:
                u = flat[cur:cur+p.numel()].view_as(p).to(p.dtype)
                if group["wd"] > 0: p.data.mul_(1 - group["lr"] * group["wd"])
                p.add_(u, alpha=-group["lr"])
                cur += p.numel()

# ── Data ─────────────────────────────────────────────
def load_shard(f):
    h = np.fromfile(f, dtype="<i4", count=256)
    assert h[0]==20240520 and h[1]==1
    return torch.from_numpy(np.fromfile(f, dtype="<u2", count=h[2], offset=1024))

class TokenStream:
    def __init__(self, pat):
        self.files = sorted(glob.glob(pat))
        assert self.files, f"No files: {pat}"
        self.fi = 0; self.tok = load_shard(self.files[0]); self.pos = 0
    def take(self, n):
        chunks = []; rem = n
        while rem > 0:
            avail = self.tok.numel() - self.pos
            if avail <= 0:
                self.fi = (self.fi+1) % len(self.files)
                self.tok = load_shard(self.files[self.fi]); self.pos = 0; continue
            k = min(rem, avail)
            chunks.append(self.tok[self.pos:self.pos+k]); self.pos += k; rem -= k
        return chunks[0] if len(chunks)==1 else torch.cat(chunks)

class DataLoader:
    def __init__(self, pat, rank, ws, dev):
        self.rank=rank; self.ws=ws; self.dev=dev; self.stream=TokenStream(pat)
    def next(self, gtok, sl, gas):
        lt = gtok // (self.ws * gas) + 1
        chunk = self.stream.take(lt * self.ws)
        s = self.rank * lt
        local = chunk[s:s+lt].to(torch.int64)
        x, y = local[:-1].reshape(-1, sl), local[1:].reshape(-1, sl)
        return x.to(self.dev, non_blocking=True), y.to(self.dev, non_blocking=True)

def load_val(pat, sl):
    files = sorted(glob.glob(pat))
    assert files
    tok = torch.cat([load_shard(f) for f in files])
    u = ((tok.numel()-1)//sl)*sl
    return tok[:u+1]

# ── BPB metrics ──────────────────────────────────────
def build_sp_luts(sp, vs, dev):
    svs = int(sp.vocab_size()); ts = max(svs, vs)
    bb = np.zeros(ts, dtype=np.int16)
    hs = np.zeros(ts, dtype=np.bool_)
    ib = np.ones(ts, dtype=np.bool_)
    for i in range(svs):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i): continue
        ib[i] = False
        if sp.is_byte(i): bb[i]=1; continue
        p = sp.id_to_piece(i)
        if p.startswith("\u2581"): hs[i]=True; p=p[1:]
        bb[i] = len(p.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=dev),
            torch.tensor(hs, dtype=torch.bool, device=dev),
            torch.tensor(ib, dtype=torch.bool, device=dev))

# ── Model (TinyLoop-compatible) ──────────────────────
# Weight names: embed.weight, pre.{i}.{attn_qkv,attn_out,mlp_gate,mlp_up,mlp_down}.weight
#               pre.{i}.ln{1,2}.{weight,bias}, loop.*, ln_out.{weight,bias}, head.weight

class Block(nn.Module):
    """SwiGLU transformer block with LayerNorm+bias, fused QKV, MHA."""
    def __init__(self, dim, n_heads, ffn_dim):
        super().__init__()
        self.dim = dim; self.n_heads = n_heads; self.hd = dim // n_heads
        self.ln1 = nn.LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3*dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.mlp_up   = nn.Linear(dim, ffn_dim, bias=False)
        self.mlp_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.attn_qkv(h).reshape(B, T, 3, self.n_heads, self.hd)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        y = flash_attn(q, k, v, causal=True).reshape(B, T, D)
        x = x + self.attn_out(y)
        h = self.ln2(x)
        x = x + self.mlp_down(F.silu(self.mlp_gate(h)) * self.mlp_up(h))
        return x

class SharedGPT(nn.Module):
    """2 pre + 1 loop × L. TinyLoop-compatible weight names."""
    def __init__(self, h):
        super().__init__()
        self.h = h
        ffn = int(h.dim * h.ffn_mult)
        self.embed = nn.Embedding(h.vocab_size, h.dim)
        self.pre = nn.ModuleList([Block(h.dim, h.n_heads, ffn) for _ in range(h.n_pre)])
        self.loop = Block(h.dim, h.n_heads, ffn)
        self.ln_out = nn.LayerNorm(h.dim)
        self.head = nn.Linear(h.dim, h.vocab_size, bias=False)
        if h.tie_weights:
            self.head.weight = self.embed.weight
        self._init()

    def _init(self):
        n = self.h.n_pre + 1
        s = 1.0 / math.sqrt(2 * n)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.ndim == 2 and min(m.weight.shape) >= 64:
                nn.init.orthogonal_(m.weight)
        for b in list(self.pre) + [self.loop]:
            b.attn_out.weight.data.mul_(s)
            b.mlp_down.weight.data.mul_(s)

    def forward(self, x_ids, y_ids=None, L=None):
        if L is None: L = max(self.h.L_min, min(self.h.L_max, int(np.random.poisson(self.h.L_mean))))
        x = self.embed(x_ids)
        for b in self.pre: x = b(x)
        for _ in range(L): x = self.loop(x)
        x = self.ln_out(x)
        logits = self.head(x)
        if self.h.logit_softcap > 0:
            logits = self.h.logit_softcap * torch.tanh(logits / self.h.logit_softcap)
        if y_ids is None: return logits
        return F.cross_entropy(logits.reshape(-1, self.h.vocab_size).float(), y_ids.reshape(-1))

# ── Eval ─────────────────────────────────────────────
def eval_val(h, model, rk, ws, dev, gas, vt, bb, hs, ib):
    sl = h.train_seq_len; lbt = h.val_batch_size // (ws*gas)
    lbs = lbt // sl; ts = (vt.numel()-1)//sl
    ss, se = (ts*rk)//ws, (ts*(rk+1))//ws
    ls = torch.zeros((), device=dev, dtype=torch.float64)
    tc = torch.zeros((), device=dev, dtype=torch.float64)
    bc = torch.zeros((), device=dev, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(ss, se, lbs):
            be = min(bs+lbs, se)
            loc = vt[bs*sl:be*sl+1].to(dev, dtype=torch.int64)
            x, y = loc[:-1].reshape(-1,sl), loc[1:].reshape(-1,sl)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y, L=h.L_eval).detach()
            n = float(y.numel()); ls += loss.to(torch.float64)*n; tc += n
            tb = bb[y.reshape(-1)].to(torch.int16)
            tb += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(torch.int16)
            bc += tb.to(torch.float64).sum()
    if dist.is_initialized():
        for t in [ls,tc,bc]: dist.all_reduce(t)
    vl = (ls/tc).item(); bpt = vl/math.log(2); tpb = tc.item()/bc.item()
    model.train()
    return vl, bpt*tpb

def eval_sliding(h, model, rk, ws, dev, vt, bb, hs, ib, stride, bsz=32):
    sl = h.train_seq_len; tot = vt.numel()-1
    wins = [w for w in range(0, tot, stride) if min(w+sl, tot)-w >= 1]
    ms, me = (len(wins)*rk)//ws, (len(wins)*(rk+1))//ws
    mw = wins[ms:me]
    ls = torch.zeros((), device=dev, dtype=torch.float64)
    tc = torch.zeros((), device=dev, dtype=torch.float64)
    bc = torch.zeros((), device=dev, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(mw), bsz):
            bw = mw[bi:bi+bsz]; B = len(bw)
            xb = torch.zeros(B, sl, dtype=torch.int64, device=dev)
            yb = torch.zeros(B, sl, dtype=torch.int64, device=dev)
            wl = []
            for i, w in enumerate(bw):
                e = min(w+sl, tot); wn = e-w; wl.append(wn)
                c = vt[w:e+1].to(torch.int64, device=dev)
                xb[i,:wn] = c[:-1]; yb[i,:wn] = c[1:]
            with torch.autocast("cuda", torch.bfloat16):
                lg = model(xb, L=h.L_eval)
            nll = F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(), yb.reshape(-1), reduction="none").reshape(B, sl)
            for i, w in enumerate(bw):
                wn = wl[i]; s = 0 if w==0 else max(wn-stride, 0)
                ls += nll[i,s:wn].to(torch.float64).sum()
                tc += float(wn-s)
                t,p = yb[i,s:wn], xb[i,s:wn]
                tb = bb[t].to(torch.float64)
                tb += (hs[t] & ~ib[p]).to(torch.float64)
                bc += tb.sum()
    if dist.is_initialized():
        for t in [ls,tc,bc]: dist.all_reduce(t)
    vl = (ls/tc).item()
    model.train()
    return vl, vl/math.log(2)*(tc.item()/bc.item())

def eval_ttt(h, model, rk, ws, dev, vt, bb, hs, ib, stride, log0=print):
    """Score-first TTT. TTT on loop block = TTT on ALL virtual layers."""
    sl = h.train_seq_len; tot = vt.numel()-1; chunk = h.ttt_chunk
    wins = [w for w in range(0,tot,stride) if min(w+sl,tot)-w>=stride or w==0]
    nc = (tot+chunk-1)//chunk
    cw = [[] for _ in range(nc)]
    for w in wins:
        e = min(w+sl,tot); wn=e-w; s=0 if w==0 else max(wn-stride,0)
        ci = min((w+s)//chunk, nc-1); cw[ci].append(w)
    log0(f"ttt:start chunks={nc} lr={h.ttt_lr} epochs={h.ttt_epochs}")
    ls = torch.zeros((),device=dev,dtype=torch.float64)
    tc = torch.zeros((),device=dev,dtype=torch.float64)
    bc = torch.zeros((),device=dev,dtype=torch.float64)
    # Freeze all, unfreeze loop + ln_out + head
    for p in model.parameters(): p.requires_grad_(False)
    tp = []
    for p in model.loop.parameters(): p.requires_grad_(True); tp.append(p)
    for n,p in model.named_parameters():
        if any(k in n for k in ("ln_out","head","embed")):
            p.requires_grad_(True); tp.append(p)
    opt = torch.optim.AdamW(tp, lr=h.ttt_lr, weight_decay=0.0)
    t0 = time.perf_counter()
    for ci in range(nc):
        if not cw[ci]: continue
        ms,me = (len(cw[ci])*rk)//ws, (len(cw[ci])*(rk+1))//ws
        myw = cw[ci][ms:me]
        model.eval()
        with torch.inference_mode():
            for bi in range(0,len(myw),h.ttt_batch):
                bw=myw[bi:bi+h.ttt_batch]; B=len(bw)
                xb=torch.zeros(B,sl,dtype=torch.int64,device=dev)
                yb=torch.zeros(B,sl,dtype=torch.int64,device=dev); wl=[]
                for i,w in enumerate(bw):
                    e=min(w+sl,tot); wn=e-w; wl.append(wn)
                    c=vt[w:e+1].to(torch.int64,device=dev)
                    xb[i,:wn]=c[:-1]; yb[i,:wn]=c[1:]
                with torch.autocast("cuda",torch.bfloat16):
                    lg=model(xb,L=h.L_eval)
                nll=F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),yb.reshape(-1),reduction="none").reshape(B,sl)
                for i,w in enumerate(bw):
                    wn=wl[i]; s=0 if w==0 else max(wn-stride,0)
                    ls+=nll[i,s:wn].to(torch.float64).sum(); tc+=float(wn-s)
                    t,p=yb[i,s:wn],xb[i,s:wn]
                    tb=bb[t].to(torch.float64); tb+=(hs[t]&~ib[p]).to(torch.float64); bc+=tb.sum()
        if ci < nc-1 and h.ttt_epochs > 0:
            cs,ce = ci*chunk, min((ci+1)*chunk, tot)
            model.train(); ns=(ce-cs)//sl
            if ns > 0:
                clr = h.ttt_lr*0.5*(1+math.cos(math.pi*ci/max(nc-1,1)))
                for pg in opt.param_groups: pg['lr']=clr
                mss,mse = (ns*rk)//ws, (ns*(rk+1))//ws
                for _ in range(h.ttt_epochs):
                    for bs in range(0, mse-mss, h.ttt_batch):
                        be=min(bs+h.ttt_batch, mse-mss)
                        st=cs+(mss+bs)*sl; et=cs+(mss+be)*sl+1
                        if et>vt.numel(): continue
                        loc=vt[st:et].to(dev,torch.int64)
                        x,y=loc[:-1].reshape(-1,sl),loc[1:].reshape(-1,sl)
                        opt.zero_grad(set_to_none=True)
                        with torch.autocast("cuda",torch.bfloat16):
                            loss=model(x,y,L=h.L_eval)
                        loss.backward()
                        if ws>1:
                            for p in tp:
                                if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(tp, 1.0)
                        opt.step()
        if rk==0 and (ci%10==0 or ci==nc-1):
            rl=ls.item()/max(tc.item(),1)
            rbpb=rl/math.log(2)*(tc.item()/max(bc.item(),1)) if tc.item()>0 else 0
            log0(f"  ttt[{ci+1}/{nc}] bpb={rbpb:.6f} t={time.perf_counter()-t0:.0f}s")
    if dist.is_initialized():
        for t in [ls,tc,bc]: dist.all_reduce(t)
    vl=(ls/tc).item(); vbpb=vl/math.log(2)*(tc.item()/bc.item())
    for p in model.parameters(): p.requires_grad_(True)
    log0(f"ttt:done bpb={vbpb:.6f} t={time.perf_counter()-t0:.0f}s")
    return vl, vbpb

# ── Export for TinyLoop ──────────────────────────────
def export_for_tinyloop(model, h, path, log0=print):
    """Save checkpoint in format that convert_pytorch.py expects."""
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # Rename: head.weight might be tied — save embed separately always
    if h.tie_weights and "head.weight" not in sd:
        sd["head.weight"] = sd["embed.weight"].clone()
    torch.save(sd, path)
    sz = os.path.getsize(path)
    log0(f"Saved TinyLoop-compatible checkpoint: {sz/1e6:.1f}MB")
    return sz

def export_compressed(sd, code, h, log0=print):
    """INT2 quantize + compress for 16MB artifact submission."""
    def q2(w):
        N,K = w.shape; w = w.float()
        pad = (4-K%4)%4
        if pad: w = F.pad(w, (0,pad)); K = w.shape[1]
        mn = w.min(1,keepdim=True).values
        mx = w.max(1,keepdim=True).values
        sc = ((mx-mn)/3).clamp(min=1e-8)
        q = ((w-mn)/sc).round().clamp(0,3).to(torch.uint8)
        q4 = q.reshape(N,K//4,4)
        pk = q4[:,:,0]|(q4[:,:,1]<<2)|(q4[:,:,2]<<4)|(q4[:,:,3]<<6)
        return pk, sc.squeeze(1).half(), mn.squeeze(1).half()

    result = {}
    for name, t in sd.items():
        t = t.detach().cpu()
        if t.ndim == 2 and t.numel() > 65536 and t.shape[1] % 4 == 0:
            pk, sc, zp = q2(t)
            result[name+".packed"] = pk
            result[name+".scale"] = sc
            result[name+".zero"] = zp
        else:
            result[name] = t.half() if t.is_floating_point() else t

    buf = io.BytesIO()
    torch.save({"w": result, "h": {
        "dim": h.dim, "n_heads": h.n_heads, "ffn_dim": int(h.dim*h.ffn_mult),
        "vocab_size": h.vocab_size, "n_pre": h.n_pre, "L_eval": h.L_eval,
        "tie_weights": h.tie_weights, "factor_dim": h.factor_dim,
    }}, buf)
    raw = buf.getvalue()
    if _COMPRESSOR == "zstd":
        blob = zstandard.ZstdCompressor(level=22).compress(raw)
    else:
        blob = zlib.compress(raw, 9)
    code_bytes = len(code.encode("utf-8"))
    total = len(blob) + code_bytes
    log0(f"Artifact: {len(blob)/1e6:.2f}MB + code {code_bytes/1e3:.0f}KB = {total/1e6:.2f}MB")
    log0(f"{'PASS' if total <= 16_000_000 else 'FAIL — OVER 16MB'}")
    return blob, total

# ── Main ─────────────────────────────────────────────
def main():
    code = Path(__file__).read_text()
    h = H()
    ddp = "RANK" in os.environ
    rk = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    lr_k = int(os.environ.get("LOCAL_RANK", 0))
    gas = 8 // ws
    dev = torch.device("cuda", lr_k)
    torch.cuda.set_device(dev)
    if ddp: dist.init_process_group("nccl", device_id=dev); dist.barrier()
    r0 = rk == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if r0:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{h.run_id}.txt"
    def log0(msg, con=True):
        if not r0: return
        if con: print(msg)
        if logfile:
            with open(logfile, "a") as f: print(msg, file=f)
    log0(code, con=False)

    random.seed(h.seed); np.random.seed(h.seed)
    torch.manual_seed(h.seed); torch.cuda.manual_seed_all(h.seed)

    sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
    assert int(sp.vocab_size()) == h.vocab_size, f"Vocab mismatch: {sp.vocab_size()} vs {h.vocab_size}"
    vt = load_val(h.val_files, h.train_seq_len)
    bb, hs, ib = build_sp_luts(sp, h.vocab_size, dev)
    dl = DataLoader(h.train_files, rk, ws, dev)

    model = SharedGPT(h).to(dev).bfloat16()
    # Keep linear weights in fp32 for optimizer quality
    for m in model.modules():
        if isinstance(m, nn.Linear): m.float()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim < 2 and p.dtype != torch.float32: p.data = p.data.float()

    np_total = sum(p.numel() for p in model.parameters())
    ffn = int(h.dim * h.ffn_mult)
    log0(f"=== TinyLoop Weight-Shared GPT ===")
    log0(f"dim={h.dim} heads={h.n_heads} ffn={ffn} pre={h.n_pre} L~Poi({h.L_mean}) L_eval={h.L_eval}")
    log0(f"params={np_total/1e6:.1f}M vocab={h.vocab_size} tie={h.tie_weights}")
    log0(f"est_int2_artifact={np_total*0.25*0.85/1e6:.1f}MB")
    log0(f"ws={ws} gas={gas} FA3={'Y' if _HAS_FA3 else 'N'}")

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[lr_k])
    base = model.module if ddp else model
    compiled = torch.compile(model, dynamic=False, fullgraph=True)

    # Optimizer split
    mat_p, sca_p, emb_p = [], [], []
    for n, p in base.named_parameters():
        if "embed" in n or "head" in n:
            emb_p.append(p)
        elif p.ndim >= 2 and min(p.shape) >= 64:
            mat_p.append(p)
        else:
            sca_p.append(p)

    opt_muon = Muon(mat_p, lr=h.lr, momentum=h.muon_mom, wd=h.wd)
    for g in opt_muon.param_groups: g["base_lr"] = h.lr
    opt_emb = torch.optim.AdamW([{"params": emb_p, "lr": h.embed_lr, "base_lr": h.embed_lr}],
                                 betas=(0.9,0.95), weight_decay=h.wd, fused=True)
    opt_sca = torch.optim.AdamW([{"params": sca_p, "lr": h.lr, "base_lr": h.lr}],
                                 betas=(0.9,0.95), weight_decay=h.wd, fused=True)
    opts = [opt_muon, opt_emb, opt_sca]
    def zero():
        for o in opts: o.zero_grad(set_to_none=True)

    ema = {n: t.detach().float().clone() for n, t in base.state_dict().items()}

    max_ms = 1000*h.max_wallclock if h.max_wallclock > 0 else None
    def lr_scale(step, ems):
        if h.warmdown_iters <= 0: return 1.0
        if max_ms is None:
            wds = max(h.iterations - h.warmdown_iters, 0)
            return max((h.iterations-step)/max(h.warmdown_iters,1), 0.0) if step >= wds else 1.0
        sms = ems / max(step, 1)
        wdms = h.warmdown_iters * sms
        rem = max(max_ms - ems, 0.0)
        return rem / max(wdms, 1e-9) if rem <= wdms else 1.0

    # Warmup
    if h.warmup_steps > 0:
        init_sd = {n: t.cpu().clone() for n, t in base.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in opts]
        model.train()
        for ws_i in range(h.warmup_steps):
            zero()
            for _ in range(gas):
                x, y = dl.next(h.train_batch_tokens, h.train_seq_len, gas)
                with torch.autocast("cuda", torch.bfloat16):
                    loss = compiled(x, y)
                (loss / gas).backward()
            for o in opts: o.step()
        base.load_state_dict(init_sd)
        for o, s in zip(opts, init_opt): o.load_state_dict(s)
        zero()
        dl = DataLoader(h.train_files, rk, ws, dev)

    tms = 0.0; stop = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    L_t = torch.tensor([0], device=dev, dtype=torch.int64)
    step = 0

    while True:
        last = step == h.iterations or (stop is not None and step >= stop)
        do_val = last or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if do_val:
            torch.cuda.synchronize(); tms += 1000*(time.perf_counter()-t0)
            vl, vbpb = eval_val(h, compiled if not ddp else base, rk, ws, dev, gas, vt, bb, hs, ib)
            log0(f"step:{step}/{h.iterations} val_loss:{vl:.4f} val_bpb:{vbpb:.4f} t:{tms:.0f}ms avg:{tms/max(step,1):.1f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last: break

        ems = tms + 1000*(time.perf_counter()-t0)
        sc = lr_scale(step, ems)

        # Dynamic L broadcast
        if rk == 0:
            L_t[0] = max(h.L_min, min(h.L_max, int(np.random.poisson(h.L_mean))))
        if ddp: dist.broadcast(L_t, 0)
        curL = int(L_t.item())

        zero()
        tl = torch.zeros((), device=dev)
        for _ in range(gas):
            x, y = dl.next(h.train_batch_tokens, h.train_seq_len, gas)
            with torch.autocast("cuda", torch.bfloat16):
                loss = compiled(x, y, L=curL)
            tl += loss.detach()
            (loss / gas).backward()
        tl /= gas

        frac = min(step/h.muon_warmup, 1.0) if h.muon_warmup > 0 else 1.0
        mm = (1-frac)*h.muon_start + frac*h.muon_mom
        for g in opt_muon.param_groups: g["momentum"] = mm; g["lr"] = g["base_lr"]*sc
        for o in [opt_emb, opt_sca]:
            for g in o.param_groups: g["lr"] = g["base_lr"]*sc

        if h.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(base.parameters(), h.grad_clip)
        for o in opts: o.step()
        zero()

        with torch.no_grad():
            for n, t in base.state_dict().items():
                ema[n].mul_(h.ema_decay).add_(t.float(), alpha=1-h.ema_decay)

        step += 1
        ams = tms + 1000*(time.perf_counter()-t0)
        if h.train_log_every > 0 and (step <= 10 or step % h.train_log_every == 0):
            log0(f"step:{step} loss:{tl.item():.4f} L={curL} t:{ams:.0f}ms avg:{ams/step:.1f}ms")

        hit = max_ms is not None and ams >= max_ms
        if ddp and max_ms:
            ht = torch.tensor(int(hit), device=dev)
            dist.all_reduce(ht, op=dist.ReduceOp.MAX)
            hit = bool(ht.item())
        if stop is None and hit: stop = step

    # Apply EMA
    log0("ema:apply")
    cur = base.state_dict()
    base.load_state_dict({n: t.to(cur[n].dtype) for n, t in ema.items()})

    # Diagnostic
    torch.cuda.synchronize(); td = time.perf_counter()
    vl, vbpb = eval_val(h, base, rk, ws, dev, gas, vt, bb, hs, ib)
    log0(f"post_ema val_bpb:{vbpb:.4f} t:{1000*(time.perf_counter()-td):.0f}ms")

    # Export TinyLoop-compatible checkpoint
    if r0:
        export_for_tinyloop(base, h, "final_model.pt", log0)

    # Compressed artifact
    sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    blob, total = export_compressed(sd, code, h, log0)
    if r0:
        with open("artifact.ptz", "wb") as f: f.write(blob)

    # Sliding window eval
    if h.eval_stride > 0 and h.eval_stride < h.train_seq_len:
        torch.cuda.synchronize(); ts = time.perf_counter()
        svl, sbpb = eval_sliding(h, base, rk, ws, dev, vt, bb, hs, ib, h.eval_stride)
        log0(f"sliding val_bpb:{sbpb:.4f} stride:{h.eval_stride} t:{1000*(time.perf_counter()-ts):.0f}ms")

    # TTT
    if h.ttt_enabled:
        torch.cuda.synchronize()
        tvl, tbpb = eval_ttt(h, base, rk, ws, dev, vt, bb, hs, ib, h.eval_stride, log0)

    if ddp: dist.destroy_process_group()

if __name__ == "__main__":
    main()
