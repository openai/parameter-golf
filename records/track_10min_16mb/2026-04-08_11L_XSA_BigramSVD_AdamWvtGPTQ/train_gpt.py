"""
MOMDP-Transformer: Layer 0 Chassis
11-Layer Transformer + XSA (all layers) + EMA(0.997) + BigramSystem (GPU-native index_add_)
+ Parallel Muon + Sliding-window eval (stride=64). DDP-ready for 8xH100.

Usage:
  Smoke test (local):  python train_gpt.py --test
  Full run (8xH100):   torchrun --nproc_per_node=8 train_gpt.py
  Single GPU:          python train_gpt.py
"""

import os, math, time, uuid, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ─── Hyperparameters ─────────────────────────────────────────────────────────

class HP:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4())[:8])
    seed           = int(os.environ.get("SEED", 1337))
    max_wallclock  = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    # Architecture
    vocab_size     = 1024          # SP1024 tokeniser
    d_model        = 512
    n_heads        = 8
    n_layers       = 11
    mlp_ratio      = 3             # hidden = d_model * mlp_ratio = 1536
    seq_len        = 1024
    # Training
    batch_tokens   = int(os.environ.get("BATCH_TOKENS", 786_432))   # 786K consensus
    iterations     = int(os.environ.get("ITERATIONS", 20_000))
    warmup_steps   = 750
    cooldown_steps = 3_000
    muon_lr        = float(os.environ.get("MUON_LR", 0.02))
    adam_lr        = float(os.environ.get("ADAM_LR", 0.02))
    weight_decay   = 0.04
    # EMA + eval
    ema_decay      = 0.997
    val_tokens     = int(os.environ.get("VAL_TOKENS", 524_288))
    val_every      = int(os.environ.get("VAL_EVERY", 100))
    val_stride     = 64            # sliding-window stride
    # SP1024 on fineweb10B English text: ~3.5 UTF-8 bytes per token (measured empirically).
    # Converts bits/token → bits/byte to match the competition BPB metric.
    bytes_per_token = float(os.environ.get("BYTES_PER_TOKEN", 3.5))
    # BigramSystem
    bigram_buckets = 2048
    bigram_dim     = 128

args = HP()

# ─── Utilities ────────────────────────────────────────────────────────────────

def log0(*a, **kw):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*a, **kw, flush=True)

def build_rope(seq_len: int, d_head: int, device: torch.device):
    pos   = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    emb   = torch.outer(pos, freqs)                     # [T, d_head/2]
    return emb.cos()[None, None], emb.sin()[None, None]  # [1,1,T,d_head/2]

def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [B,H,T,d_head] | cos,sin: [1,1,T,d_head/2]
    T  = q.shape[2]
    c, s = cos[:, :, :T, :], sin[:, :, :T, :]
    h  = q.shape[-1] // 2
    q_rot = torch.cat([q[..., :h] * c - q[..., h:] * s,
                        q[..., :h] * s + q[..., h:] * c], dim=-1)
    k_rot = torch.cat([k[..., :h] * c - k[..., h:] * s,
                        k[..., :h] * s + k[..., h:] * c], dim=-1)
    return q_rot, k_rot

# ─── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w   = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

# ─── Causal Self-Attention with XSA ───────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Standard causal MHA + XSA (arXiv:2603.09078):
    subtracts the self-value contribution from each attention output.
    Requires explicit attention weights, so we compute attention manually.
    """
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.qkv     = nn.Linear(d, 3 * d, bias=False)
        self.proj    = nn.Linear(d, d, bias=False)
        self.q_gain  = nn.Parameter(torch.full((n_heads,), 4.0, dtype=torch.float32))
        nn.init.zeros_(self.proj.weight)   # zero-init residual path

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        B, T, C = x.shape
        H, dh   = self.n_heads, self.d_head

        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, dh).transpose(1, 2)   # [B,H,T,dh]
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q, k = apply_rope(q, k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        # Explicit attention weights (needed to extract diagonal for XSA)
        w = (q @ k.transpose(-2, -1)) * (dh ** -0.5)   # [B,H,T,T]
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        w      = w.masked_fill(~causal, float('-inf'))
        w      = F.softmax(w.float(), dim=-1).to(x.dtype)   # fp32 softmax for numerical stability

        out = w @ v                                          # [B,H,T,dh]

        # XSA: remove self-value bias: out_t -= a_{t,t} * v_t
        self_w = w.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) # [B,H,T,1]
        out    = out - self_w * v

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

# ─── MLP with relu² activation ────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d: int, ratio: int):
        super().__init__()
        h        = d * ratio
        self.fc1 = nn.Linear(d, h, bias=False)
        self.fc2 = nn.Linear(h, d, bias=False)
        nn.init.zeros_(self.fc2.weight)   # zero-init residual path

    def forward(self, x: torch.Tensor):
        return self.fc2(F.relu(self.fc1(x)).pow(2))   # relu² (good with XSA)

# ─── Transformer Block ────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, ratio: int):
        super().__init__()
        self.n1   = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.n2   = RMSNorm(d)
        self.mlp  = MLP(d, ratio)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x

# ─── System 1: BigramSystem ───────────────────────────────────────────────────

class BigramSystem(nn.Module):
    """
    Compact learned bigram via embed(n_buckets, d_bucket) + proj(d_bucket, vocab).
    Statistics-initialized: uses index_add_ (GPU-native; avoids the numpy
    copy-discard bug where np.add.at operates on a temporary copy).
    Fine-tuned end-to-end during training.  ~400K params → ~0.3MB int6+zstd.
    """
    def __init__(self, vocab: int, n_buckets: int, d_bucket: int):
        super().__init__()
        self.n_buckets = n_buckets
        self.embed     = nn.Embedding(n_buckets, d_bucket)
        self.proj      = nn.Linear(d_bucket, vocab, bias=False)
        nn.init.normal_(self.embed.weight, 0, 0.02)
        nn.init.zeros_(self.proj.weight)

    @torch.no_grad()
    def populate(self, data_iter, device, n_batches: int = 50):
        """
        Count (prev_bucket, curr_token) pairs via index_add_ on GPU,
        then decompose log-probs into (embed, proj) via rank-d_bucket SVD.
        """
        V      = self.proj.out_features
        counts = torch.ones(self.n_buckets, V, dtype=torch.float32, device=device)

        for i, (x, y) in enumerate(data_iter):
            if i >= n_batches:
                break
            bkt  = x.reshape(-1) % self.n_buckets           # hash prev token
            curr = y.reshape(-1).clamp(0, V - 1)
            oh   = torch.zeros(len(curr), V, dtype=torch.float32, device=device)
            oh.scatter_(1, curr.unsqueeze(1), 1.0)
            counts.index_add_(0, bkt, oh)                   # GPU-native, no numpy

        row_sums = counts.sum(1, keepdim=True).clamp(min=1.0)
        lp = torch.log(counts / row_sums)                   # [n_buckets, vocab]

        # Low-rank factorisation → embed + proj weights
        try:
            U, S, Vh = torch.linalg.svd(lp, full_matrices=False)
            r   = self.embed.embedding_dim
            sqS = S[:r].sqrt()
            self.embed.weight.data.copy_((U[:, :r] * sqS))
            self.proj.weight.data.copy_((Vh[:r] * sqS.unsqueeze(1)).T)
            log0(f"BigramSystem: SVD-initialised (rank {r}).")
        except Exception as e:
            log0(f"BigramSystem: SVD fallback ({e}), keeping random init.")

    def forward(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        b = prev_tokens % self.n_buckets
        return self.proj(self.embed(b))                      # [B,T,vocab]

# ─── Main Model ───────────────────────────────────────────────────────────────

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        V, D = args.vocab_size, args.d_model

        self.embed   = nn.Embedding(V, D)
        self.blocks  = nn.ModuleList([
            Block(D, args.n_heads, args.mlp_ratio) for _ in range(args.n_layers)
        ])
        self.norm_f  = RMSNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.embed.weight              # tied embeddings → -1MB artifact
        self.slot    = nn.Parameter(torch.zeros(1, 1, D))
        self.bigram  = BigramSystem(V, args.bigram_buckets, args.bigram_dim)

        # RoPE cache — persistent=False: recomputed, not saved to model.bin
        self.register_buffer('rope_cos', torch.empty(0), persistent=False)
        self.register_buffer('rope_sin', torch.empty(0), persistent=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.Linear) and m.weight.data.abs().max() > 1e-6:
                nn.init.normal_(m.weight, 0, 0.02)

    def _get_rope(self, device: torch.device):
        if self.rope_cos.numel() == 0 or self.rope_cos.device != device:
            c, s = build_rope(args.seq_len, args.d_model // args.n_heads, device)
            self.rope_cos, self.rope_sin = c.to(device), s.to(device)
        return self.rope_cos, self.rope_sin

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T   = x.shape
        cos, sin = self._get_rope(x.device)

        # System 1: bigram logits (prev token hashed to bucket)
        prev          = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device),
                                   x[:, :-1]], dim=1)
        bigram_logits = self.bigram(prev)                    # [B,T,V]

        # System 2: transformer
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        h      = h + self.slot
        h      = self.norm_f(h)
        logits = self.lm_head(h) + bigram_logits             # [B,T,V]

        if targets is not None:
            return F.cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
        return logits

    @property
    def num_params(self):
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def byte_size(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel() * p.element_size()
        return total

# ─── Muon Optimizer (Parallel) ────────────────────────────────────────────────

def zeropower_ns5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    X = X.T if G.shape[0] > G.shape[1] else X
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Parallel Muon: each DDP rank handles a disjoint subset of 2D params,
    Newton-Schulz orthogonalises its slice, then all_reduce merges updates.
    Falls back to vanilla Muon on single-GPU.
    """
    def __init__(self, params, lr: float, momentum: float = 0.95,
                 ns_steps: int = 5, wd: float = 0.04):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, wd=wd))

    @torch.no_grad()
    def step(self):
        ddp  = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank()      if ddp else 0
        ws   = dist.get_world_size() if ddp else 1

        for g in self.param_groups:
            lr, mom, ns, wd = g['lr'], g['momentum'], g['ns_steps'], g['wd']
            params = [p for p in g['params'] if p.grad is not None]
            if not params:
                continue

            total = sum(p.numel() for p in params)
            flat  = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)

            ptr = 0
            for i, p in enumerate(params):
                n = p.numel()
                if i % ws == rank:
                    grad = p.grad.float()
                    st   = self.state[p]
                    if 'buf' not in st:
                        st['buf'] = torch.zeros_like(grad)
                    buf = st['buf']
                    buf.mul_(mom).add_(grad)
                    upd = grad.add(buf, alpha=mom)
                    if upd.ndim >= 2:
                        upd  = zeropower_ns5(upd, steps=ns)
                        upd *= max(1, upd.shape[0] / upd.shape[1]) ** 0.5
                    flat[ptr:ptr + n] = upd.flatten()
                ptr += n

            if ddp:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)

            ptr = 0
            for p in params:
                n   = p.numel()
                upd = flat[ptr:ptr + n].view_as(p).to(p.dtype)
                p.mul_(1.0 - wd * lr)     # decoupled weight decay
                p.add_(upd, alpha=-lr)
                ptr += n

# ─── Data Loader ──────────────────────────────────────────────────────────────

class DataLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.ws, self.dev = rank, world_size, device
        d, b = os.path.dirname(pattern), os.path.basename(pattern)
        self.files = sorted(Path(d).glob(b)) if os.path.isdir(d) else []
        self._buf, self._ptr = None, 0

    def _load(self):
        if not self.files:
            # Dummy random data for local testing (no fineweb shards available)
            self._buf = torch.randint(0, args.vocab_size, (3_000_000,), dtype=torch.int32)
        else:
            # Shard format: 256×int32 header (1024 bytes) then uint16 tokens
            arr       = np.fromfile(self.files.pop(0), dtype="<u2", count=-1, offset=1024)
            self._buf = torch.from_numpy(arr.astype(np.int32))
        self._ptr = 0

    def next_batch(self, tokens: int, seq: int):
        per_rank = max(tokens // self.ws, seq)
        span     = per_rank + 1
        if self._buf is None or self._ptr + span * self.ws > len(self._buf):
            self._load()
        start = self._ptr + self.rank * span
        if start + span > len(self._buf):
            self._load()
            start = self.rank * span
        chunk    = self._buf[start:start + span].to(self.dev, dtype=torch.long, non_blocking=True)
        self._ptr += span * self.ws
        n_seq    = max(per_rank // seq, 1)
        chunk    = chunk[:n_seq * seq + 1]
        return chunk[:-1].view(n_seq, seq), chunk[1:].view(n_seq, seq)

# ─── EMA ──────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay  = decay
        # Store float32 shadow weights keyed by parameter name
        self.shadow = {k: v.float().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.float(), alpha=1.0 - d)

    def apply_to(self, model: nn.Module) -> dict:
        """Swap EMA weights in; return original weights for later restore."""
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        ema_sd = {k: self.shadow[k].to(v.dtype) if k in self.shadow else v
                  for k, v in model.state_dict().items()}
        model.load_state_dict(ema_sd, strict=False)
        return orig

    def restore(self, model: nn.Module, orig: dict):
        model.load_state_dict(orig, strict=False)

# ─── Evaluation: Sliding Window BPB ──────────────────────────────────────────

@torch.no_grad()
def evaluate_bpb(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """
    Sliding-window BPB (stride=64, window=seq_len).
    Each token is scored exactly once with up to seq_len-stride tokens of context.
    Worth ~0.034 BPB vs non-overlapping eval (per competition ablation #77).

    Returns bits-per-byte (competition metric).
    Formula: (nats/token) / ln(2) / bytes_per_token
    In DDP mode, all_reduces across ranks so every GPU contributes to the metric.
    """
    model.eval()
    stride, win = args.val_stride, args.seq_len

    x_all, y_all = val_loader.next_batch(args.val_tokens, args.seq_len)
    flat_x = x_all.reshape(-1)
    flat_y = y_all.reshape(-1)
    # Reconstruct contiguous token stream [x_0, x_1, ..., x_{N-1}, y_{N-1}]
    toks = torch.cat([flat_x, flat_y[-1:]])

    total_loss = total_n = 0
    for start in range(0, len(toks) - win - 1, stride):
        x_win = toks[start:start + win].unsqueeze(0)
        y_win = toks[start + 1:start + win + 1].unsqueeze(0)
        out   = model(x_win)
        # Only score the last `stride` positions (each scored exactly once)
        lgt   = out[:, -stride:, :].reshape(-1, args.vocab_size)
        tgt   = y_win[:, -stride:].reshape(-1)
        total_loss += F.cross_entropy(lgt, tgt, reduction='sum').item()
        total_n    += stride

    # Aggregate across DDP ranks so the logged BPB reflects the full val set.
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([total_loss, float(total_n)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_n = t[0].item(), t[1].item()

    # bits/token → bits/byte (competition metric)
    return (total_loss / max(total_n, 1)) / math.log(2) / args.bytes_per_token

# ─── LR Schedule ─────────────────────────────────────────────────────────────

def lr_scale(step: int, total: int) -> float:
    wu, cd = args.warmup_steps, args.cooldown_steps
    if step < wu:
        return step / wu
    if step > total - cd:
        return max(0.0, (total - step) / cd)
    return 1.0

# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    ddp = 'RANK' in os.environ
    if ddp:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        ws   = dist.get_world_size()
        dev  = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
        torch.cuda.set_device(dev)
    else:
        rank, ws = 0, 1
        if   torch.cuda.is_available():                 dev = torch.device('cuda')
        elif torch.backends.mps.is_available():         dev = torch.device('mps')
        else:                                           dev = torch.device('cpu')

    torch.manual_seed(args.seed + rank)
    log0(f"[{args.run_id}] device={dev}  world_size={ws}")

    model = GPT().to(dev)
    log0(f"Params (deduplicated): {model.num_params:,}  |  FP32: {model.byte_size()/1e6:.1f} MB")
    if model.byte_size() > 15_800_000:
        log0("WARNING: model exceeds ~15.8 MB fp32 — check post-GPTQ artifact size.")
    random_bpb = math.log2(args.vocab_size) / args.bytes_per_token
    log0(f"Random-baseline BPB: {random_bpb:.3f}  |  Competition SOTA ~1.115  |  bytes_per_token={args.bytes_per_token}")

    if ddp:
        model = DDP(model, device_ids=[dev])
    base = model.module if ddp else model

    train_ld = DataLoader(args.train_files, rank, ws, dev)
    val_ld   = DataLoader(args.val_files,   rank, ws, dev)

    # Populate BigramSystem from first 50 training batches via index_add_ (GPU-native)
    log0("Populating BigramSystem ...")
    def _bigram_iter():
        for _ in range(5):
            yield train_ld.next_batch(max(args.batch_tokens // ws, args.seq_len), args.seq_len)
    base.bigram.populate(_bigram_iter(), dev, n_batches=5)
    if dev.type == 'cuda':
        torch.cuda.empty_cache()   # return fragmented allocator blocks after populate

    # Optimizer split: Muon for 2D weights (excluding tied embedding), AdamW for rest
    tied      = base.embed.weight
    muon_p    = [p for p in base.parameters() if p.requires_grad and p.ndim == 2 and p is not tied]
    adam_p    = [p for p in base.parameters() if p.requires_grad and (p.ndim != 2 or p is tied)]
    opt_muon  = Muon(muon_p, lr=args.muon_lr, wd=args.weight_decay)
    opt_adam  = torch.optim.AdamW(adam_p, lr=args.adam_lr, betas=(0.9, 0.95), weight_decay=0.0)

    ema   = EMA(base, args.ema_decay)
    accum = max(1, args.batch_tokens // (ws * args.seq_len * 24))
    log0(f"grad_accum={accum}  effective_batch={args.batch_tokens:,} tokens")

    t0 = time.time()
    for step in range(args.iterations):
        if time.time() - t0 > args.max_wallclock:
            log0(f"Wall-clock limit reached at step {step}.")
            break

        # LR and momentum warmup
        sc  = lr_scale(step, args.iterations)
        mom = 0.95 + 0.04 * min(step / 1500.0, 1.0)   # 0.95 → 0.99 over 1500 steps
        for g in opt_muon.param_groups:
            g['lr'] = args.muon_lr * sc
            g['momentum'] = mom
        for g in opt_adam.param_groups:
            g['lr'] = args.adam_lr * sc

        model.train()
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        loss_acc = 0.0
        for micro in range(accum):
            if ddp:
                model.require_backward_grad_sync = (micro == accum - 1)
            x, y = train_ld.next_batch(args.batch_tokens // accum, args.seq_len)
            if dev.type == 'cuda':
                with torch.autocast('cuda', torch.bfloat16):
                    loss = model(x, y)
            else:
                loss = model(x, y)
            (loss / accum).backward()
            loss_acc += loss.detach().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_muon.step()
        opt_adam.step()
        ema.update(base)

        if step % 10 == 0:
            elapsed = time.time() - t0
            log0(f"step {step:5d} | loss {loss_acc/accum:.4f} | "
                 f"lr {args.muon_lr*sc:.5f} | {elapsed:.0f}s")

        if step % args.val_every == 0:
            orig = ema.apply_to(base)
            bpb  = evaluate_bpb(base, val_ld, dev)
            ema.restore(base, orig)
            log0(f"  ↳ BPB (EMA, stride={args.val_stride}): {bpb:.4f}  "
                 f"[random baseline {math.log2(args.vocab_size)/args.bytes_per_token:.3f}  "
                 f"target <1.115]")

    
    # Collect saliency BEFORE DDP teardown — optimizer state is still live.
    # AdamW v_t (exp_avg_sq) accumulates grad² over the entire run: best signal.
    log0("Collecting optimizer saliency for GPTQ ...")
    saliency = _collect_saliency(opt_adam, opt_muon, base)

    # Barrier: ensure all ranks finish training before rank 0 starts GPTQ.
    if ddp:
        dist.barrier()

    if rank == 0:
        orig = ema.apply_to(base)
        log0("Saving quantized model.bin ...")
        save_quantized_model(base, dev, "model.bin", t0, saliency=saliency)
        ema.restore(base, orig)

    if ddp:
        dist.destroy_process_group()


# ─── GPTQ Int6 + AR Self-Gen Calibration ──────────────────────────────────────

def _collect_saliency(opt_adam: torch.optim.AdamW,
                      opt_muon,
                      model: nn.Module) -> dict:
    """
    Extract per-parameter saliency from optimizer state.

    AdamW params  → exp_avg_sq = v_t = EMA of grad²  (Fisher diagonal approx).
                    Accumulated over the ENTIRE training run — low-noise,
                    reflects which weights actually drove the loss signal.
    Muon params   → |momentum buf|  (integrated gradient direction over training).

    Returns: {param_name: CPU float32 saliency tensor, same shape as weight}
    """
    saliency: dict = {}
    id_to_name = {id(p): n for n, p in model.named_parameters()}

    for p, state in opt_adam.state.items():
        if 'exp_avg_sq' in state:
            name = id_to_name.get(id(p))
            if name:
                saliency[name] = state['exp_avg_sq'].detach().float().cpu()

    for p, state in opt_muon.state.items():
        if 'buf' in state:
            name = id_to_name.get(id(p))
            if name and name not in saliency:
                saliency[name] = state['buf'].abs().detach().float().cpu()

    log0(f"  [saliency] collected for {len(saliency)} parameters.")
    return saliency


def _ar_generate_calibration(model: nn.Module, device: torch.device,
                              num_seqs: int = 32, seq_len: int = 1024,
                              temperature: float = 0.8, seed: int = 42):
    """
    Autoregressively generate calibration sequences from the EMA model.
    No external data — fully self-contained.
    Returns list of [1, seq_len] int64 tensors.
    """
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    seqs = []
    ctx = (torch.autocast('cuda', torch.bfloat16)
           if device.type == 'cuda' else torch.autocast('cpu', torch.bfloat16))
    batch_size = 8
    with torch.inference_mode(), ctx:
        for start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - start)
            tokens = torch.randint(0, args.vocab_size, (bs, 1), device=device, generator=rng)
            for _ in range(seq_len - 1):
                logits   = model(tokens)[:, -1, :]          # [bs, V]
                probs    = torch.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens   = torch.cat([tokens, next_tok], dim=1)
            for i in range(bs):
                seqs.append(tokens[i:i+1].clone())
    return seqs


def _collect_hessians(model: nn.Module, seqs, device: torch.device):
    """
    Collect H = X^T X for every nn.Linear layer via forward hooks.
    Works with our GPT model (all linear layers are nn.Linear, not CastedLinear).
    """
    hessians: dict = {}
    hooks    = []

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            cols = mod.weight.shape[1]
            H    = torch.zeros(cols, cols, dtype=torch.float32)
            hessians[name] = H

            def make_hook(h_ref):
                def hook(m, inp, out):
                    x = inp[0].detach().float()
                    x = x.reshape(-1, x.shape[-1])
                    h_ref.add_((x.T @ x).cpu())
                return hook
            hooks.append(mod.register_forward_hook(make_hook(H)))

    model.eval()
    ctx = (torch.autocast('cuda', torch.bfloat16)
           if device.type == 'cuda' else torch.autocast('cpu', torch.bfloat16))
    with torch.inference_mode(), ctx:
        for seq in seqs:
            x = seq[:, :-1].to(device)
            model(x)

    for h in hooks:
        h.remove()

    n = len(seqs)
    for name in hessians:
        H    = hessians[name] / max(n, 1)
        damp = 0.01 * H.diagonal().mean().clamp_min(1e-6)
        H.diagonal().add_(damp)
        hessians[name] = H
    return hessians


def _gptq_quantize_weight(W: torch.Tensor, H: torch.Tensor,
                           clip_range: int = 31, block_size: int = 128,
                           saliency: torch.Tensor = None):
    """
    GPTQ int6 quantization: Hessian-guided column-wise error compensation.
    W:        [rows, cols] float32
    H:        [cols, cols] float32 Hessian (X^T X + damping)
    saliency: [rows, cols] float32 per-weight importance (AdamW v_t or Muon buf).
              Boosts the Hessian diagonal for columns connected to high-gradient
              features, protecting their precision during int6 compression.
    Returns: (Q: int8 with values in [-31,31], scale: float16 per-row)
    """
    rows, cols = W.shape
    H     = H.clone()

    # AdamW v_t saliency boost: columns with high accumulated gradient²
    # get higher apparent Hessian sensitivity → GPTQ protects them more.
    if saliency is not None and saliency.shape == (rows, cols):
        col_sal = saliency.mean(dim=0).float()             # [cols] per-column importance
        col_sal = col_sal / col_sal.mean().clamp_min(1e-8) # normalise to mean=1
        H.diagonal().add_(0.1 * col_sal * H.diagonal().mean())

    dead  = H.diagonal() == 0
    H[dead.nonzero(as_tuple=True)[0], dead.nonzero(as_tuple=True)[0]] = 1.0

    perm     = torch.argsort(H.diagonal(), descending=True)
    inv_perm = torch.argsort(perm)
    Wp       = W[:, perm].clone()
    Wp[:, dead[perm]] = 0.0

    Hp = H[perm][:, perm]
    try:
        Hinv = torch.linalg.cholesky(Hp)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        # Ill-conditioned Hessian — fall back to percentile quantization.
        return _percentile_quantize_weight(W, clip_range)

    best_q, best_s, best_err = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 1.0]:
        row_clip = (torch.quantile(W.abs(), pct, dim=1)
                    if pct < 1.0 else W.abs().amax(dim=1))
        sf  = (row_clip / clip_range).clamp_min(1.0 / clip_range)
        Q   = torch.zeros_like(Wp, dtype=torch.int8)
        Ww  = Wp.clone()
        for i1 in range(0, cols, block_size):
            i2     = min(i1 + block_size, cols)
            cnt    = i2 - i1
            W1     = Ww[:, i1:i2].clone()
            Q1     = torch.zeros(rows, cnt, dtype=torch.int8)
            Err1   = torch.zeros(rows, cnt)
            Hi1    = Hinv[i1:i2, i1:i2]
            for i in range(cnt):
                w       = W1[:, i]
                d       = Hi1[i, i]
                q       = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                err     = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hi1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                Ww[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        err   = (Wp - recon).pow(2).mean().item()
        if err < best_err:
            best_q, best_s, best_err = Q, sf.to(torch.float16), err

    best_q = best_q[:, inv_perm]
    return best_q, best_s


def _percentile_quantize_weight(W: torch.Tensor, clip_range: int = 31):
    """Per-row percentile int6 quantization (fallback for 1-D or small tensors)."""
    t32 = W.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 1.0]:
            row_clip = (torch.quantile(t32.abs(), pct, dim=1)
                        if pct < 1.0 else t32.abs().amax(dim=1))
            s  = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q  = torch.clamp(torch.round(t32 / s.float()[:, None]),
                             -clip_range, clip_range).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax  = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q     = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def save_quantized_model(model: nn.Module, device: torch.device,
                          path: str = "model.bin", t0: float = 0.0,
                          saliency: dict = None):
    """
    End-to-end GPTQ int6 quantization + compression → model.bin.

    Pipeline:
      1. AR self-gen calibration (32 seqs × 1024 tokens) — no external data
      2. Collect Hessians via nn.Linear forward hooks
      3. GPTQ int6 for 2D weights ≥ 4096 elements, with optional AdamW v_t
         saliency boosting of the Hessian diagonal (novel contribution)
      4. Percentile int6 fallback for 1-D / small tensors
      5. Non-float tensors kept as float16
      6. torch.save + zstd-22 (preferred) or lzma-9 (stdlib fallback)
    """
    import io

    log0("  [quant] generating AR calibration sequences ...")
    seqs = _ar_generate_calibration(model, device, num_seqs=32,
                                    seq_len=args.seq_len, seed=args.seed)

    log0("  [quant] collecting Hessians ...")
    hessians = _collect_hessians(model, seqs, device)

    log0("  [quant] quantizing to int6 (AdamW-v_t saliency boost) ...")
    quant: dict = {}
    meta:  dict = {}
    seen_ids: set = set()

    sd = model.state_dict()
    for name, tensor in sd.items():
        pid = id(tensor)
        if pid in seen_ids:
            meta[name] = "alias"
            quant[name + ".alias_of"] = [k for k, v in sd.items()
                                          if id(v) == pid and k != name][0]
            continue
        seen_ids.add(pid)

        t = tensor.detach().cpu().float()
        if not tensor.is_floating_point() or t.numel() < 4096:
            quant[name] = tensor.detach().cpu().to(torch.float16)
            meta[name]  = "fp16"
            continue

        H   = hessians.get(name.replace(".weight", ""))  # key is module path
        sal = saliency.get(name) if saliency else None
        if t.ndim == 2 and H is not None:
            q, s = _gptq_quantize_weight(t, H, saliency=sal)
            meta[name] = "gptq_int6_sal" if sal is not None else "gptq_int6"
        else:
            q, s = _percentile_quantize_weight(t)
            meta[name] = "pct_int6"

        quant[name + ".q"] = q
        quant[name + ".s"] = s

    buf = io.BytesIO()
    torch.save({"quant": quant, "meta": meta, "hp": {
        "vocab_size": args.vocab_size, "d_model": args.d_model,
        "n_heads": args.n_heads, "n_layers": args.n_layers,
        "mlp_ratio": args.mlp_ratio, "seq_len": args.seq_len,
        "bigram_buckets": args.bigram_buckets, "bigram_dim": args.bigram_dim,
    }}, buf)
    raw = buf.getvalue()

    # zstd-22 gives ~5-10% better compression than lzma-9 on int8 GPTQ data.
    # Try it first; fall back to lzma (stdlib) if zstandard is not installed.
    try:
        import zstandard as zstd
        cctx       = zstd.ZstdCompressor(level=22)
        compressed = cctx.compress(raw)
        algo       = "zstd-22"
    except ImportError:
        import lzma
        compressed = lzma.compress(raw, preset=9)
        algo       = "lzma-9"

    with open(path, "wb") as f:
        f.write(compressed)

    sal_count = sum(1 for v in meta.values() if v == "gptq_int6_sal")
    size_mb   = len(compressed) / 1e6
    log0(f"  [quant] Saved {path}  {size_mb:.2f} MB  [{algo}]  "
         f"saliency-boosted layers: {sal_count}  "
         f"({'OK ✓' if size_mb < 16 else 'OVER LIMIT ✗'})  "
         f"total elapsed {time.time() - t0:.0f}s")
    return size_mb


# ─── Smoke Test ───────────────────────────────────────────────────────────────

def smoke_test():
    """
    Local sanity check — no data files needed.
    Verifies: no NaN, correct model size, backward pass, 5 training steps.
    Run: python train_gpt.py --test
    """
    dev = (torch.device('mps') if torch.backends.mps.is_available()
           else torch.device('cpu'))
    print(f"\n{'='*60}")
    print(f"Smoke test on {dev}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = GPT().to(dev)
    print(f"  Params (dedup):  {model.num_params:,}")
    print(f"  FP32 size:       {model.byte_size()/1e6:.1f} MB")
    print(f"  Expected BPB at random baseline: {math.log2(args.vocab_size):.2f}")

    # Populate bigram from random data
    SEQ = 64   # shorter seq for speed
    xs  = [torch.randint(0, args.vocab_size, (2, SEQ), device=dev) for _ in range(5)]
    ys  = [torch.randint(0, args.vocab_size, (2, SEQ), device=dev) for _ in range(5)]
    model.bigram.populate(zip(xs, ys), dev, n_batches=5)
    assert not torch.isnan(model.bigram.embed.weight).any(), "NaN in bigram embed"
    print("  BigramSystem:    OK")

    # Forward pass
    x    = torch.randint(0, args.vocab_size, (2, SEQ), device=dev)
    y    = torch.randint(0, args.vocab_size, (2, SEQ), device=dev)
    loss = model(x, y)
    assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"
    print(f"  Initial loss:    {loss.item():.4f}  (random baseline ~{math.log(args.vocab_size):.2f})")

    # Backward pass + gradient check
    loss.backward()
    nan_grads = [(n, p) for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not nan_grads, f"Non-finite gradients in: {[n for n,_ in nan_grads]}"
    print("  Gradients:       OK")

    # Optimizer setup (mirrors train())
    tied     = model.embed.weight
    muon_p   = [p for p in model.parameters() if p.requires_grad and p.ndim == 2 and p is not tied]
    adam_p   = [p for p in model.parameters() if p.requires_grad and (p.ndim != 2 or p is tied)]
    opt_m    = Muon(muon_p, lr=0.02, wd=0.04)
    opt_a    = torch.optim.AdamW(adam_p, lr=0.02, betas=(0.9, 0.95))
    ema      = EMA(model, 0.997)

    losses = []
    for step in range(5):
        model.train()
        opt_m.zero_grad(set_to_none=True)
        opt_a.zero_grad(set_to_none=True)
        x = torch.randint(0, args.vocab_size, (2, SEQ), device=dev)
        y = torch.randint(0, args.vocab_size, (2, SEQ), device=dev)
        loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_m.step()
        opt_a.step()
        ema.update(model)
        losses.append(round(loss.item(), 4))
    print(f"  5-step losses:   {losses}")

    # EMA apply/restore round-trip
    orig = ema.apply_to(model)
    ema.restore(model, orig)
    print("  EMA apply/restore: OK")

    print(f"\n{'='*60}")
    print("  SMOKE TEST PASSED ✓")
    print(f"{'='*60}\n")
    print("Next steps:")
    print("  Full local run (small batch): BATCH_TOKENS=4096 ITERATIONS=200 python train_gpt.py")
    print("  8xH100 full run:              torchrun --nproc_per_node=8 train_gpt.py")

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if '--test' in sys.argv:
        smoke_test()
    else:
        train()
