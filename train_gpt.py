#!/usr/bin/env python3
"""Parameter Golf — Production v4  (contest-adapted)
8×H100 SXM | Universal Transformer | Ternary weights | Muon optimizer

Architecture : NB=4 unique blocks × NR=6 recurrences = 24 effective layers
               QK-Norm · SwiGLU · RoPE · weight-tied output · spectral init
Optimizer    : Muon (NS-ortho) for 2-D weights  +  fused AdamW for emb/norms
Training     : warmup → hold → 35% linear warmdown · bfloat16 · Flash-Attn 2
Evaluation   : sliding-window BPB (stride=256), tokenizer-agnostic
Packing      : base-3 ternary (5 vals/byte) – fully vectorised save & load

Contest env-vars (all optional, with sensible defaults):
  DATA_PATH               – path to pre-tokenised .bin shards  (default: stream raw bytes)
  TOKENIZER_PATH          – SentencePiece .model file          (default: byte-level)
  VOCAB_SIZE              – vocabulary size                     (default: 256)
  RUN_ID                  – run identifier for log file         (default: run_00)
  MAX_WALLCLOCK_SECONDS   – 0 = unlimited, >0 = hard cutoff    (default: 575)
  VAL_LOSS_EVERY          – validate every N steps, 0 = off    (default: 0)
  VAL_BATCH_SIZE          – batch size during validation        (default: 16)
"""
import os, glob, time, math, pickle, json, zlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# ── Read contest env-vars ─────────────────────────────────────────────────────
RUN_ID       = os.environ.get('RUN_ID',             'run_00')
DATA_PATH    = os.environ.get('DATA_PATH',           None)
TOK_PATH     = os.environ.get('TOKENIZER_PATH',      None)
VOCAB_SIZE   = int(os.environ.get('VOCAB_SIZE',      '256'))
_mwc         = os.environ.get('MAX_WALLCLOCK_SECONDS','575')
MAX_WC       = float('inf') if _mwc == '0' else int(_mwc)
VLE          = int(os.environ.get('VAL_LOSS_EVERY',  '0'))
VAL_BS       = int(os.environ.get('VAL_BATCH_SIZE',  '16'))

# ── Global backend flags ──────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32   = True
torch.backends.cudnn.allow_tf32         = True
torch.backends.cudnn.benchmark          = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

# NCCL for H100 SXM NVLink / InfiniBand
os.environ.setdefault('NCCL_IB_DISABLE',    '0')
os.environ.setdefault('NCCL_IB_GID_INDEX',  '3')
os.environ.setdefault('NCCL_SOCKET_IFNAME', 'eth')
os.environ.setdefault('NCCL_NET_GDR_LEVEL', '5')
os.environ.setdefault('NCCL_P2P_LEVEL',     'NVL')

HARD_LIMIT = 16_000_000
WARN_LIMIT = 15_950_000

# ─── Hardware detection ───────────────────────────────────────────────────────
def detect_hw():
    info = dict(gpu='cpu',sm=0,bf16=False,flash=False,compile=False,vram_gb=0)
    if not torch.cuda.is_available(): return info
    maj, _ = torch.cuda.get_device_capability()
    sm = maj * 10 + _
    info.update(gpu=torch.cuda.get_device_name(0), sm=sm,
                bf16=maj>=8, flash=maj>=8, compile=maj>=8,
                vram_gb=round(torch.cuda.get_device_properties(0).total_memory/1e9,1))
    return info

# ─── Newton-Schulz orthogonalisation ─────────────────────────────────────────
def ns_ortho(G: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """Approximate polar factor via degree-5 Newton-Schulz iteration (bfloat16).
    Matches official zeropower_via_newtonschulz5 implementation."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed: X = X.T
    return X

# ─── Muon Optimizer ──────────────────────────────────────────────────────────
class Muon(torch.optim.Optimizer):
    """Momentum + Orthogonalised Update for 2-D weight matrices.
    Nesterov momentum, per-matrix scale correction, DDP all_reduce.
    Embeddings and 1-D tensors (gains/biases) should use AdamW."""
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=10,
                 rank=0, world_size=1):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      ns_steps=ns_steps, rank=rank,
                                      world_size=world_size))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']; mu = group['momentum']
            ns = group['ns_steps']; rk = group['rank']; ws = group['world_size']
            params = [p for p in group['params'] if p.grad is not None]

            # Build Nesterov momentum for each param
            nesterov_grads = []
            for p in params:
                g = p.grad
                state = self.state[p]
                if 'buf' not in state: state['buf'] = g.clone()
                else:                  state['buf'].mul_(mu).add_(g)
                # Nesterov: g + mu * buf
                nesterov_grads.append(g.add(state['buf'], alpha=mu))

            # Distribute NS computation across DDP ranks
            updates = []
            for i, (p, g) in enumerate(zip(params, nesterov_grads)):
                if i % ws == rk:
                    u = ns_ortho(g, steps=ns)
                    # Scale correction: sqrt(max(1, rows/cols))
                    u = u * max(1, p.shape[0] / p.shape[1]) ** 0.5
                else:
                    u = torch.zeros_like(g, dtype=torch.bfloat16)
                updates.append(u)

            # All-reduce to gather each rank's NS result
            if ws > 1:
                flat = torch.cat([u.flatten() for u in updates])
                dist.all_reduce(flat)
                offset = 0
                for i, u in enumerate(updates):
                    sz = u.numel()
                    updates[i] = flat[offset:offset+sz].view(u.shape)
                    offset += sz

            for p, u in zip(params, updates):
                p.add_(u.to(p.dtype), alpha=-lr)

# ─── Ternary packing — fully vectorised ──────────────────────────────────────
def pack_t(w: torch.Tensor):
    """Quantise float → {-1,0,+1}; pack 5 values per byte (base-3).
    Encoding: byte = v0·81 + v1·27 + v2·9 + v3·3 + v4  where vi ∈ {0,1,2}."""
    s = w.abs().mean().clamp(1e-8).item()
    t = ((w.detach()/s).round().clamp(-1,1)+1).to(torch.uint8).flatten()
    n = t.numel(); pad = (-n)%5
    if pad: t = torch.cat([t, t.new_zeros(pad)])
    v = t.view(-1,5)
    b = (v[:,0]*81 + v[:,1]*27 + v[:,2]*9 + v[:,3]*3 + v[:,4]).to(torch.uint8)
    return b.cpu().numpy(), n, s

def unpack_t(b: np.ndarray, n: int, s: float, device, shape) -> torch.Tensor:
    """Vectorised unpack — no Python loops (single broadcast div + mod)."""
    bt  = torch.from_numpy(b).to(device, dtype=torch.int32)
    div = torch.tensor([81,27,9,3,1], device=device, dtype=torch.int32)
    trits = (bt.unsqueeze(1) // div) % 3        # [N,5] ∈ {0,1,2}
    return trits.to(torch.int8).sub_(1).flatten()[:n].float().mul_(s).view(shape)

def save_packed(sd: dict, path: str):
    out = {}
    for k, v in sd.items():
        if v.dim() == 2 and 'emb' not in k:
            b, n, s = pack_t(v)
            out[k] = ('t', b, n, s, tuple(v.shape))
        else:
            out[k] = ('f', v.cpu().to(torch.float16).numpy())
    with open(path,'wb') as f: pickle.dump(out, f, protocol=4)

def load_packed(path: str, device='cpu') -> dict:
    with open(path,'rb') as f: out = pickle.load(f)
    sd = {}
    for k, v in out.items():
        if v[0]=='t': sd[k] = unpack_t(v[1],v[2],v[3],device,v[4])
        else:         sd[k] = torch.from_numpy(v[1]).float().to(device)
    return sd

def zlib_roundtrip_size(sd: dict) -> int:
    """Estimate contest-compatible size: zlib-compress all weight bytes."""
    buf = bytearray()
    for v in sd.values():
        arr = v.cpu().numpy()
        # Quantise to int8 (per-tensor absmax), then zlib-compress
        amax = np.abs(arr).max()
        if amax > 0:
            q = np.round(arr / amax * 127).astype(np.int8)
        else:
            q = arr.astype(np.int8)
        buf.extend(zlib.compress(q.tobytes(), level=9))
    return len(buf)

# ─── Parameter count + predicted file size ───────────────────────────────────
def model_stats(model, script_path=None):
    rows=[]; total_p=0; total_b=0
    for name, p in model.named_parameters():
        n    = p.numel()
        tern = p.dim()==2 and 'emb' not in name
        rb   = math.ceil(n/5)+80 if tern else n*2
        rows.append((name,n,'ternary' if tern else 'fp16',rb))
        total_p+=n; total_b+=rb
    total_b += 4096
    W = 48
    sep = '+' + '-'*W + '+' + '-'*13 + '+' + '-'*9 + '+' + '-'*10 + '+'
    print(sep)
    print(f"|{'Parameter':<{W}}|{'Count':>13}|{'Storage':>9}|{'Packed':>10}|")
    print(sep)
    for name,n,kind,rb in rows:
        nm = name if len(name)<=W else '..' + name[-(W-2):]
        print(f"|{nm:<{W}}|{n:>13,}|{kind:>9}|{rb/1024:>8.1f}K|")
    print(sep)
    print(f"|{'TOTAL':>{W}}|{total_p:>13,}|{'':9}|{total_b/1e6:>8.2f}M|")
    print(sep)
    sz_s = os.path.getsize(script_path) if script_path else 0
    pred = total_b+sz_s
    ok   = 'OK' if pred<=WARN_LIMIT else ('WARN' if pred<=HARD_LIMIT else 'OVER')
    print(f"\n  model ~= {total_b/1e6:.3f} MB | script = {sz_s/1e3:.1f} KB | "
          f"total ~= {pred/1e6:.3f} MB / {HARD_LIMIT/1e6:.3f} MB  [{ok}]")
    if pred>WARN_LIMIT:
        print(f"  !! WARNING: predicted {pred/1e6:.3f} MB > {WARN_LIMIT/1e6:.3f} MB!")
    return total_p, total_b

# ─── Bytes-per-token for BPB conversion ──────────────────────────────────────
def get_bytes_per_token(tok_path, vocab_size) -> float:
    """Compute average bytes/token from SentencePiece model, or default."""
    if vocab_size == 256:
        return 1.0   # byte-level: each token IS one byte
    if tok_path and os.path.exists(tok_path):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=tok_path)
            total_b = sum(len(sp.id_to_piece(i).encode('utf-8','replace'))
                         for i in range(vocab_size))
            bpt = total_b / vocab_size
            print(f"  bytes/token (from tokenizer) = {bpt:.4f}")
            return bpt
        except Exception as e:
            print(f"  SP tokenizer load failed: {e}")
    # Empirical estimate for FineWeb + sp1024: ~3.8 bytes/token
    bpt = 3.8
    print(f"  bytes/token (hardcoded estimate) = {bpt}")
    return bpt

# ─── RoPE ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def rope_freqs(T: int, dh: int, device) -> torch.Tensor:
    theta = 1./(10000**(torch.arange(0,dh,2,device=device).float()/dh))
    f = torch.outer(torch.arange(T,device=device).float(), theta)
    return torch.cat([f,f], dim=-1)   # [T, dh]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1]//2
    return torch.cat([-x[...,h:], x[...,:h]], dim=-1)

def apply_rope(q, k, freqs):
    c, s = freqs.cos()[None,None], freqs.sin()[None,None]
    return q*c+rotate_half(q)*s,  k*c+rotate_half(k)*s

# ─── Ternary Linear with STE ──────────────────────────────────────────────────
class TL(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(o,i))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s  = self.w.abs().mean().clamp(1e-8)
        wq = self.w + ((self.w/s).round().clamp(-1,1)*s - self.w).detach()
        return F.linear(x, wq)

# ─── Block with QK-Norm ───────────────────────────────────────────────────────
class Block(nn.Module):
    """Transformer block with QK-Norm (prevents logit explosion in recurrence)."""
    def __init__(self, d: int, h: int, ff: int = 4, use_flash: bool = True):
        super().__init__()
        self.d, self.h, self.dh = d, h, d//h
        self.use_flash = use_flash
        self.n1  = nn.RMSNorm(d);      self.n2  = nn.RMSNorm(d)
        self.qkv = TL(d, 3*d);         self.op  = TL(d, d)
        self.q_norm = nn.RMSNorm(self.dh)     # QK-Norm: per-head
        self.k_norm = nn.RMSNorm(self.dh)
        self.u   = TL(d, d*ff);        self.g   = TL(d, d*ff)
        self.dw  = TL(d*ff, d)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        r = self.n1(x)
        q, k, v = self.qkv(r).chunk(3, dim=-1)
        q = q.view(B,T,self.h,self.dh).transpose(1,2)
        k = k.view(B,T,self.h,self.dh).transpose(1,2)
        v = v.view(B,T,self.h,self.dh).transpose(1,2)
        q = self.q_norm(q);  k = self.k_norm(k)   # QK-Norm before RoPE
        q, k = apply_rope(q, k, freqs)
        if self.use_flash:
            a = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        else:
            sc  = self.dh**-0.5
            att = (q@k.transpose(-2,-1))*sc
            mask = torch.ones(T,T,device=x.device,dtype=torch.bool).tril().logical_not()
            a   = F.softmax(att.masked_fill(mask,float('-inf')),dim=-1) @ v
        x = x + self.op(a.transpose(1,2).reshape(B,T,C))
        r = self.n2(x)
        x = x + self.dw(F.silu(self.u(r)) * self.g(r))
        return x

# ─── Universal GPT ────────────────────────────────────────────────────────────
class UGPT(nn.Module):
    """NB shared blocks × NR recurrences = NB×NR effective depth.
    Weight-tied output head. Spectral / depth-aware initialisation."""
    def __init__(self, V: int=256, d: int=1024, h: int=16,
                 nb: int=4, nr: int=6, use_flash: bool=True):
        super().__init__()
        self.nr     = nr
        self.emb    = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d,h,use_flash=use_flash)
                                     for _ in range(nb)])
        self.ln     = nn.RMSNorm(d)
        self._spectral_init(nr)

    def _spectral_init(self, nr: int):
        """Orthogonal init with residual-path scaling = 1/√(2·NR)."""
        res_scale = (2*nr)**-0.5
        for name, p in self.named_parameters():
            if p.dim()==2 and 'emb' not in name:
                fan_out, fan_in = p.shape
                if fan_out >= fan_in:
                    nn.init.orthogonal_(p)
                else:
                    tmp = p.data.new_empty(fan_in, fan_out)
                    nn.init.orthogonal_(tmp); p.data.copy_(tmp.T)
                if name.endswith(('.op.w', '.dw.w')):
                    p.data.mul_(res_scale)
        nn.init.normal_(self.emb.weight, std=self.emb.embedding_dim**-0.5)

    def forward(self, idx: torch.Tensor, tgt: torch.Tensor = None):
        B, T = idx.shape
        x     = self.emb(idx)
        freqs = rope_freqs(T, self.blocks[0].dh, idx.device)
        for _ in range(self.nr):
            for blk in self.blocks:
                x = blk(x, freqs)
        x      = self.ln(x)
        logits = x @ self.emb.weight.T
        if tgt is None: return logits
        return logits, F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                        tgt.view(-1))

# ─── Datasets ─────────────────────────────────────────────────────────────────
class BinDS(Dataset):
    """Non-overlapping fixed-length windows over a numpy array of token IDs."""
    def __init__(self, data, T: int):
        self.d=data; self.T=T
    def __len__(self): return (len(self.d)-1)//self.T
    def __getitem__(self, i: int):
        s = i*self.T
        return (torch.from_numpy(self.d[s   :s+self.T  ].astype(np.int64)),
                torch.from_numpy(self.d[s+1 :s+self.T+1].astype(np.int64)))

class SlidingDS(Dataset):
    """Overlapping sliding-window dataset for evaluation."""
    def __init__(self, data, T: int, stride: int):
        self.d=data; self.T=T; self.S=stride
        self.n=max(0,(len(data)-T-1)//stride)
    def __len__(self): return self.n
    def __getitem__(self, i: int):
        s = i*self.S
        return (torch.from_numpy(self.d[s   :s+self.T  ].astype(np.int64)),
                torch.from_numpy(self.d[s+1 :s+self.T+1].astype(np.int64)))

# ─── Sliding-window BPB evaluation ───────────────────────────────────────────
def eval_sliding(model, val_d, T, stride, device, dtype,
                 bytes_per_token=1.0, bs=16, max_win=999_999,
                 rank=0, world=1) -> tuple:
    """Return (val_bpb, val_loss_nats) using sliding-window methodology.

    Only the last `stride` tokens per window count toward the loss;
    the first T-stride tokens provide fully-conditioned context.
    BPB = nats_per_token / (bytes_per_token × log 2).

    Distributed: each rank evaluates windows [rank::world], then all_reduces.
    """
    sds = SlidingDS(val_d, T, stride)
    n   = min(len(sds), max_win)
    ctx = T - stride
    total_nll = 0.0; total_tok = 0
    model.eval()
    amp_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad(), torch.amp.autocast(amp_dev, dtype=dtype):
        # Each rank evaluates every world-th window
        my_wins = list(range(rank, n, world))
        for start in range(0, len(my_wins), bs):
            batch_idx = my_wins[start:start+bs]
            xs, ys = zip(*[sds[i] for i in batch_idx])
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
            logits = model(x)
            lc     = logits[:,ctx:,:].contiguous()
            tc     = y[:,ctx:].contiguous()
            nll    = F.cross_entropy(lc.view(-1, lc.shape[-1]),
                                     tc.view(-1), reduction='sum')
            total_nll += nll.item()
            total_tok += lc.shape[0] * lc.shape[1]
    # Gather across all DDP ranks
    if world > 1:
        t = torch.tensor([total_nll, float(total_tok)],
                         dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_nll = t[0].item(); total_tok = int(t[1].item())
    nats = total_nll / max(total_tok, 1)
    bpb  = nats / (bytes_per_token * math.log(2))
    return bpb, nats

# ─── Data loading (contest .bin shards OR raw-byte fallback) ─────────────────
def _read_bin(path, dtype=np.uint16) -> np.ndarray:
    """Read a contest binary shard.
    Official format: 256 × int32 header (1024 bytes total),
    header[0]=20240520 (decimal magic), header[1]=1 (version),
    followed by uint16 token IDs."""
    raw = np.fromfile(path, dtype=np.uint8)
    if len(raw) >= 4:
        magic = int(np.frombuffer(raw[:4].tobytes(), dtype='<i4')[0])
        if magic == 20240520:
            raw = raw[1024:]   # skip 256 × 4-byte header
    if dtype != np.uint8:
        raw = raw.view(dtype)
    return raw

def load_contest_data(data_path: str, vocab_size: int,
                      max_train_tokens: int = 8_000_000_000):
    """Load pre-tokenised shards from DATA_PATH.
    Returns (train_memmap_or_array, val_array).
    """
    dtype = np.uint16  # contest shards always store uint16 token IDs

    # Locate shards (support multiple naming conventions)
    train_glob = (glob.glob(os.path.join(data_path, 'fineweb_train_*.bin')) or
                  glob.glob(os.path.join(data_path, '*train*.bin')))
    val_glob   = (glob.glob(os.path.join(data_path, 'fineweb_val_*.bin')) or
                  glob.glob(os.path.join(data_path, '*val*.bin')))
    train_glob.sort(); val_glob.sort()

    if not train_glob:
        raise FileNotFoundError(f"No train shards found in {data_path}")
    if not val_glob:
        raise FileNotFoundError(f"No val shards found in {data_path}")

    # Load val completely into RAM (val is small)
    val_arrs = [_read_bin(p, dtype) for p in val_glob]
    val_d    = np.concatenate(val_arrs).astype(np.int32)
    print(f"  val   : {len(val_d):,} tokens from {len(val_glob)} file(s)")

    # Load train shards up to max_train_tokens
    train_arrs = []; total = 0
    for p in train_glob:
        arr = _read_bin(p, dtype).astype(np.int32)
        train_arrs.append(arr); total += len(arr)
        print(f"  train shard loaded: {p}  ({len(arr)/1e6:.0f}M tokens)", flush=True)
        if total >= max_train_tokens: break
    train_d = np.concatenate(train_arrs)
    print(f"  train : {len(train_d)/1e9:.3f}B tokens total")
    return train_d, val_d

def prep_raw_bytes(target_bytes: int = 8_000_000_000):
    """Fallback: stream raw FineWeb bytes when DATA_PATH is not set."""
    os.makedirs('data', exist_ok=True)
    if os.path.exists('data/train.bin') and os.path.exists('data/val.bin'):
        t = os.path.getsize('data/train.bin')
        v = os.path.getsize('data/val.bin')
        print(f"  data/ exists — train={t/1e9:.3f} GB  val={v/1e6:.2f} MB  (skip)")
        return
    print(f"  Streaming FineWeb → data/  (target ≤{target_bytes/1e9:.0f} GB)")
    try:
        from datasets import load_dataset
        tok  = os.environ.get('HF_TOKEN')
        print("  HF_TOKEN: " + ("found ✓" if tok else "not set"))
        ds   = load_dataset('HuggingFaceFW/fineweb', name='sample-10BT',
                            split='train', streaming=True, token=tok)
        SHARD=100_000_000; VAL_DOCS=2_000
        tw=0; nd=0; vi=0; shard=bytearray(); vbuf=bytearray()
        for p in ('data/train.bin','data/val.bin'):
            if os.path.exists(p): os.remove(p)
        for ex in ds:
            txt=(ex['text']+'\n').encode('utf-8','replace'); nd+=1
            if vi<VAL_DOCS: vbuf.extend(txt); vi+=1
            else:
                shard.extend(txt)
                if len(shard)>=SHARD:
                    with open('data/train.bin','ab') as f: f.write(shard)
                    tw+=len(shard); shard=bytearray()
                    print(f"  shard — docs={nd:,}  train={tw/1e9:.2f} GB",flush=True)
            if tw+len(shard)>=target_bytes: break
        if shard:
            with open('data/train.bin','ab') as f: f.write(shard)
            tw+=len(shard)
        with open('data/val.bin','wb') as f: f.write(vbuf)
        print(f"  Done — train={tw/1e9:.3f} GB ({nd:,} docs) val={len(vbuf)/1e6:.2f} MB")
    except Exception as e:
        print(f"  Data prep error: {e}  → synthetic fallback")
        rng=np.random.default_rng(0)
        rng.integers(0,256,50_000_000,dtype=np.uint8).tofile('data/train.bin')
        rng.integers(0,256, 2_000_000,dtype=np.uint8).tofile('data/val.bin')

# ─── Training Logger ──────────────────────────────────────────────────────────
class Logger:
    def __init__(self, path: str):
        self.path=path; self.t0=time.time()
        with open(path,'w') as f:
            f.write(json.dumps({'event':'start',
                                'run_id':RUN_ID,
                                'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                'pid':os.getpid()})+'\n')
    def _w(self, rec):
        rec['wall_s']=round(time.time()-self.t0,2)
        with open(self.path,'a') as f: f.write(json.dumps(rec)+'\n')
    def step(self, step,epoch,loss,bpb,lr_m,lr_a,tokens,tps,gnorm=None):
        r={'event':'train','step':step,'epoch':epoch,
           'loss':round(loss,6),'bpb':round(bpb,6),
           'lr_muon':round(lr_m,8),'lr_adamw':round(lr_a,8),
           'tokens':tokens,'tok_per_s':round(tps,0)}
        if gnorm is not None: r['grad_norm']=round(float(gnorm),4)
        self._w(r)
    def phase(self,name,elapsed,step):
        self._w({'event':'phase','phase':name,'elapsed':round(elapsed,1),'step':step})
    def val(self,bpb,nats,stride):
        self._w({'event':'val','val_bpb':round(bpb,6),
                 'val_nats':round(nats,6),'stride':stride})
    def sizes(self, model_b, script_b, ternary_b):
        self._w({'event':'save','model_mb':round(model_b/1e6,3),
                 'script_kb':round(script_b/1e3,1),
                 'ternary_mb':round(ternary_b/1e6,3),
                 'total_mb':round((model_b+script_b)/1e6,3)})
    def done(self): self._w({'event':'done'})

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # ── DDP ───────────────────────────────────────────────────────────────────
    ddp = 'RANK' in os.environ
    if ddp:
        dist.init_process_group('nccl')
        rank=dist.get_rank(); world=dist.get_world_size()
        device=f'cuda:{rank}'; torch.cuda.set_device(rank)
    else:
        rank=0; world=1
        device='cuda' if torch.cuda.is_available() else 'cpu'
    master = rank==0

    hw = detect_hw()
    if master:
        print('='*72)
        print(f"  RUN_ID  : {RUN_ID}")
        print(f"  GPU     : {hw['gpu']}")
        print(f"  SM={hw['sm']}  VRAM={hw['vram_gb']} GB  "
              f"bf16={hw['bf16']}  flash={hw['flash']}  compile={hw['compile']}")
        print(f"  World   : {world} GPU(s)")
        print(f"  DATA_PATH  = {DATA_PATH or '(streaming raw bytes)'}")
        print(f"  TOK_PATH   = {TOK_PATH  or '(none — byte-level)'}")
        print(f"  VOCAB_SIZE = {VOCAB_SIZE}")
        print(f"  MAX_WC     = {MAX_WC}s")
        print('='*72)

    # ── Config: CPU smoke-test / T4 / H100 production ────────────────────────
    if hw['sm'] == 0:   # CPU only — tiny smoke-test config
        D,H,NB,NR,T  = 128,4,1,1,256
        BS,ACCUM      = 2,1
        MAX_T         = min(MAX_WC, 60)
        DATA_GB       = 0.1
        AMP_DTYPE     = torch.float32
        EVAL_STRIDE   = 64
        if master:
            print("  CPU mode — micro smoke-test: d=128 h=4 nb=1 nr=1 T=256 fp32")
    elif 0 < hw['sm'] < 80:
        D,H,NB,NR,T  = 512,8,2,2,512
        BS,ACCUM      = 4,4
        MAX_T         = min(MAX_WC, 570)
        DATA_GB       = 2.0
        AMP_DTYPE     = torch.float16
        EVAL_STRIDE   = 128
        if master:
            print("  Legacy GPU — smoke-test: d=512 h=8 nb=2 nr=2 T=512 fp16")
    else:
        D,H,NB,NR,T  = 1024,16,4,6,1024
        BS,ACCUM      = 16,4
        MAX_T         = min(MAX_WC, 590)
        DATA_GB       = 1.0
        AMP_DTYPE     = torch.bfloat16
        EVAL_STRIDE   = 256
        if master:
            print(f"  Config: d={D} h={H} nb={NB} nr={NR} eff_depth={NB*NR} T={T} "
                  f"eff_batch={BS*ACCUM*world*T:,} tok/step")

    # LR: warmup → hold → 35 % linear warmdown
    # Muon:  LR_MUON  × s(t)  in [MIN_MUON,  LR_MUON ]
    # AdamW: LR_ADAMW × s(t)  in [MIN_ADAMW, LR_ADAMW]
    # Both share ratio = 0.10  →  same schedule multiplier s(t)
    LR_MUON=0.020; MIN_MUON=2e-3
    LR_ADAMW=3e-3; MIN_ADAMW=3e-4
    WU=200; WD_FRAC=0.40
    WD_START = MAX_T*(1.0-WD_FRAC)   # ≈ 354 s with MAX_T=590
    OUTF = f'{RUN_ID}_model.bin'

    def lr_scale(step: int, elapsed: float) -> float:
        if step<WU:              return (step+1)/WU
        if elapsed<WD_START:    return 1.0
        t = min(1.0,(elapsed-WD_START)/(MAX_T-WD_START))
        return 1.0-0.9*t        # 1.0 → 0.10

    # ── Data ──────────────────────────────────────────────────────────────────
    if DATA_PATH:
        if master:
            print(f"\n  Loading data from DATA_PATH …")
        # All ranks load in parallel (faster than broadcast for large data)
        train_d, val_d = load_contest_data(
            DATA_PATH, VOCAB_SIZE,
            max_train_tokens=int(DATA_GB*1e9))
        if ddp:
            dist.barrier()  # sync after all ranks finished loading
    else:
        if master: prep_raw_bytes(target_bytes=int(DATA_GB*1e9))
        if ddp: dist.barrier()
        train_d = np.memmap('data/train.bin', dtype=np.uint8, mode='r')
        val_d   = np.fromfile('data/val.bin',  dtype=np.uint8)

    # Bytes-per-token factor for BPB calculation
    bpt = get_bytes_per_token(TOK_PATH, VOCAB_SIZE)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UGPT(V=VOCAB_SIZE, d=D, h=H, nb=NB, nr=NR,
                 use_flash=hw['flash']).to(device)
    if master:
        print()
        model_stats(model, script_path=__file__)
        print()

    if not hw['flash'] and torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    # ── Parameter groups ──────────────────────────────────────────────────────
    muon_ps, adamw_ps = [], []
    for name, p in model.named_parameters():
        if p.dim()==2 and 'emb' not in name: muon_ps.append(p)
        else:                                 adamw_ps.append(p)
    if master:
        print(f"  Muon  params: {sum(p.numel() for p in muon_ps)/1e6:.2f}M")
        print(f"  AdamW params: {sum(p.numel() for p in adamw_ps)/1e6:.3f}M")

    # ── DDP ───────────────────────────────────────────────────────────────────
    if ddp:
        model = DDP(model, device_ids=[rank],
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                    static_graph=True)
    raw = model.module if ddp else model

    # ── Optimisers ────────────────────────────────────────────────────────────
    muon_opt = Muon(muon_ps, lr=LR_MUON, momentum=0.95, ns_steps=10,
                    rank=rank, world_size=world)
    try:
        adamw_opt = torch.optim.AdamW(adamw_ps, lr=LR_ADAMW,
                                      betas=(0.9,0.95), weight_decay=0.1,
                                      fused=True)
        if master: print("  Muon + fused AdamW ✓")
    except Exception:
        adamw_opt = torch.optim.AdamW(adamw_ps, lr=LR_ADAMW,
                                      betas=(0.9,0.95), weight_decay=0.1)
        if master: print("  Muon + AdamW")

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_ds = BinDS(train_d, T)
    samp = DistributedSampler(train_ds, world, rank,
                              shuffle=True, drop_last=True) if ddp else None
    nw   = 4 if hw['sm']>=80 else (2 if hw['sm']>0 else 0)
    pin  = hw['sm'] > 0   # pin_memory only works with CUDA
    ldr  = DataLoader(train_ds, BS, sampler=samp, shuffle=(samp is None),
                      num_workers=nw, pin_memory=pin, drop_last=True,
                      persistent_workers=(nw>0), prefetch_factor=(3 if nw>0 else None))

    # ── torch.compile ─────────────────────────────────────────────────────────
    if hw['compile']:
        try:
            model = torch.compile(model, mode='reduce-overhead', dynamic=False)
            if master: print("  torch.compile: reduce-overhead ✓")
        except Exception as e:
            if master: print(f"  torch.compile skipped: {e}")
    else:
        if master: print("  torch.compile: disabled (sm<80)")

    # ── Training ──────────────────────────────────────────────────────────────
    log = Logger(f'{RUN_ID}_train_log.jsonl') if master else None
    t0  = time.time(); step=0; total_tok=0; running_loss=0.
    phase='warmup'; warmup_done=False
    model.train()

    if master:
        print(f"\n{'─'*72}")
        print(f"  Training  MAX_T={MAX_T}s | WD_START≈{WD_START:.0f}s "
              f"({WD_FRAC*100:.0f}% warmdown)")
        print(f"{'─'*72}")

    for epoch in range(9999):
        if samp: samp.set_epoch(epoch)
        for i,(x,y) in enumerate(ldr):
            elapsed = time.time()-t0
            if elapsed>MAX_T: break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sc   = lr_scale(step, elapsed)
            lr_m = LR_MUON*sc;  lr_a = LR_ADAMW*sc
            for pg in muon_opt.param_groups:  pg['lr']=lr_m
            for pg in adamw_opt.param_groups: pg['lr']=lr_a

            new_phase = ('warmup' if step<WU
                         else 'warmdown' if elapsed>=WD_START
                         else 'hold')
            if master and new_phase!=phase:
                phase=new_phase
                if log: log.phase(phase, elapsed, step)
                print(f"  ── Phase: {phase}  t={elapsed:.0f}s  step={step}  "
                      f"lr_m={lr_m:.2e} lr_a={lr_a:.2e} ──")

            with torch.amp.autocast(device.split(':')[0], dtype=AMP_DTYPE):
                _, loss = model(x,y)
                loss    = loss/ACCUM
            loss.backward()
            running_loss += loss.item()
            total_tok    += BS*T*world

            if (i+1)%ACCUM==0:
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                muon_opt.step();  muon_opt.zero_grad(set_to_none=True)
                adamw_opt.step(); adamw_opt.zero_grad(set_to_none=True)
                step += 1

                # Periodic validation (contest VAL_LOSS_EVERY) — master only, fast subset
                if master and VLE>0 and step%VLE==0:
                    raw.eval()
                    bpb_v, nats_v = eval_sliding(
                        raw, val_d, T, EVAL_STRIDE, device, AMP_DTYPE,
                        bytes_per_token=bpt, bs=VAL_BS, max_win=500,
                        rank=0, world=1)
                    print(f"  [val] step={step}  val_bpb={bpb_v:.5f}  "
                          f"val_loss={nats_v:.5f}")
                    if log: log.val(bpb_v, nats_v, EVAL_STRIDE)
                    model.train()

                if master and step%100==0:
                    avg    = running_loss/100; running_loss=0.
                    tps    = total_tok/elapsed
                    bpb_tr = avg/(bpt*math.log(2))
                    eta    = max(0., MAX_T-elapsed)
                    print(f"  step={step:5d}  loss={avg:.4f}  bpb={bpb_tr:.4f}  "
                          f"lr_m={lr_m:.2e} lr_a={lr_a:.2e}  "
                          f"tok/s={tps/1e6:.2f}M  t={elapsed:.0f}s  "
                          f"ETA={eta:.0f}s  [{phase}]")
                    if log: log.step(step,epoch,avg,bpb_tr,lr_m,lr_a,
                                     total_tok,tps,gnorm)
        else:
            continue
        break

    # ── Final sliding-window evaluation — ALL ranks participate ───────────────
    if master:
        print(f"\n  Evaluating (stride={EVAL_STRIDE}, full val set, {world} GPU(s)) …")
    if ddp: dist.barrier()   # sync before final eval
    raw.eval()
    val_bpb, val_nats = eval_sliding(
        raw, val_d, T, EVAL_STRIDE, device, AMP_DTYPE,
        bytes_per_token=bpt, bs=VAL_BS, max_win=999_999,
        rank=rank, world=world)

    if master:
        print(f"\n{'═'*72}")
        print(f"  val_loss = {val_nats:.5f}  |  val_bpb = {val_bpb:.5f}")
        print(f"  (bytes/token={bpt:.3f}, stride={EVAL_STRIDE}, T={T})")
        print(f"{'═'*72}\n")
        if log: log.val(val_bpb, val_nats, EVAL_STRIDE)

        # ── Save model ────────────────────────────────────────────────────────
        sd = raw.state_dict()
        save_packed(sd, OUTF)

        sz_m     = os.path.getsize(OUTF)
        sz_s     = os.path.getsize(__file__)
        total_sz = sz_m+sz_s

        # Also compute zlib-compressed int8 size (informational)
        zlib_sz = zlib_roundtrip_size(sd)

        if log: log.sizes(sz_m, sz_s, zlib_sz)

        # ── Contest output lines ───────────────────────────────────────────────
        print(f"final_int8_zlib_roundtrip "
              f"val_loss={val_nats:.6f} "
              f"val_bpb={val_bpb:.6f} "
              f"model_mb={sz_m/1e6:.4f} "
              f"script_mb={sz_s/1e6:.4f} "
              f"total_mb={total_sz/1e6:.4f} "
              f"zlib_int8_mb={zlib_sz/1e6:.4f}")

        # ── Size guards ───────────────────────────────────────────────────────
        if total_sz>WARN_LIMIT:
            print(f"\n  WARNING: {total_sz/1e6:.3f} MB > {WARN_LIMIT/1e6:.3f} MB!")
        if total_sz>HARD_LIMIT:
            raise RuntimeError(
                f"DISQUALIFIED: {total_sz/1e6:.3f} MB > {HARD_LIMIT/1e6:.3f} MB!")
        print(f"  Size OK: {total_sz/1e6:.3f} MB <= {HARD_LIMIT/1e6:.3f} MB")

        if log: log.done()

    if ddp: dist.destroy_process_group()

if __name__=='__main__': main()
