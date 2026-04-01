# ============================================================
# v16 — DEFINITIVE ENGINE
# Every good idea. Zero bad ones.
# ============================================================

import os, math, copy, glob, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.amp import autocast, GradScaler

torch.set_float32_matmul_precision('high')

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
d_model         = 768
n_heads         = 6
n_recur         = 24
rank            = d_model // 4   # 192
vocab_size      = 1024
max_ctx         = 256

batch_size      = 32
accum_steps     = 4
train_steps     = 12000
peak_lr         = 1e-3
min_lr          = 1e-4
warmup          = int(train_steps * 0.1)
weight_decay    = 0.1
label_smoothing = 0.05

# Distillation (set use_distill=False if no teacher available)
use_distill      = False
distill_temp     = 2.0
distill_start    = 1500
distill_alpha_hi = 0.8
distill_alpha_lo = 0.5
distill_pivot    = int(train_steps * 0.5)

# Cross-batch state (requires sequential sampler — always on)
state_weight = 0.15
tbptt_steps  = 4

device  = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
scaler  = GradScaler("cuda", enabled=use_amp)

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────

def load_shard(file: Path) -> torch.Tensor:
    header     = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens     = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256*4)
    return torch.from_numpy(tokens.astype(np.int64))

def load_all(pattern: str) -> torch.Tensor:
    files = sorted(glob.glob(pattern))
    assert len(files) > 0, f"No files found: {pattern}"
    return torch.cat([load_shard(Path(f)) for f in files])

print("Loading data...")
train_data = load_all("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")
eval_data  = load_all("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")
print(f"  train: {len(train_data)/1e6:.1f}M tokens | eval: {len(eval_data)/1e6:.1f}M tokens\n")

# Teacher logits (optional)
teacher_logits = None
if use_distill and os.path.exists("teacher_logits.pt"):
    teacher_logits = torch.load("teacher_logits.pt")
    assert len(teacher_logits) >= len(train_data), \
        f"teacher_logits too short: {len(teacher_logits)} < {len(train_data)}"
else:
    use_distill = False

# Token frequency weighting
counts        = torch.bincount(train_data, minlength=vocab_size).float()
freq          = counts / counts.sum()
token_weights = (1.0 / (freq + 1e-6)) ** 0.3
token_weights = (token_weights / token_weights.mean()).to(device)

# ─────────────────────────────────────────
# SEQUENTIAL SAMPLER
# ─────────────────────────────────────────

class SequentialSampler:
    """
    Splits corpus into batch_size parallel streams.
    Consecutive calls to .next(T) return continuations of each stream —
    making cross-batch hidden state carry meaningful (true BPTT).
    State is detached every tbptt_steps chunks to truncate gradient graph.
    """
    def __init__(self, data: torch.Tensor, B: int):
        self.data  = data
        self.B     = B
        chunk      = len(data) // B
        self.chunk = chunk
        self.pos   = [i * chunk for i in range(B)]
        self._step = 0

    def next(self, T: int):
        xs, ys, ts = [], [], []
        for i in range(self.B):
            s = self.pos[i]
            if s + T + 1 >= (i + 1) * self.chunk:
                self.pos[i] = i * self.chunk
                s = self.pos[i]
            xs.append(self.data[s:s+T])
            ys.append(self.data[s+1:s+T+1])
            if use_distill:
                ts.append(teacher_logits[s:s+T])
            self.pos[i] += T
        self._step += 1
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        t = torch.stack(ts).to(device) if use_distill else None
        return x, y, t

    def should_detach(self) -> bool:
        return self._step % tbptt_steps == 0

sampler = SequentialSampler(train_data, batch_size)

def get_seq_len(step: int) -> int:
    if step < train_steps * 0.2: return 64
    if step < train_steps * 0.5: return 128
    return max_ctx

# ─────────────────────────────────────────
# MODULES
# ─────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class LowRank(nn.Module):
    """Factored linear: in → rank → out. No nonlinearity."""
    def __init__(self, i: int, o: int, r: int):
        super().__init__()
        self.A = nn.Linear(i, r, bias=False)
        self.B = nn.Linear(r, o, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x))


class RoPE(nn.Module):
    """Rotary positional embeddings. Zero parameters. Applied to Q and K only."""
    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        inv  = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t    = torch.arange(max_len).float()
        emb  = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
        self.register_buffer("cos", emb.cos()[None, None])
        self.register_buffer("sin", emb.sin()[None, None])

    @staticmethod
    def _rot(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, T: int):
        cos = self.cos[:, :, :T, :]
        sin = self.sin[:, :, :T, :]
        return q * cos + self._rot(q) * sin, k * cos + self._rot(k) * sin


class Attn(nn.Module):
    """
    Causal multi-head self-attention.
    Q/K/V: low-rank.  Out: full-rank (writes to residual stream).
    RoPE on Q and K.  float('-inf') causal mask.
    No attention score scaling beyond 1/sqrt(hd).
    """
    def __init__(self):
        super().__init__()
        self.hd   = d_model // n_heads
        self.q    = LowRank(d_model, d_model, rank)
        self.k    = LowRank(d_model, d_model, rank)
        self.v    = LowRank(d_model, d_model, rank)
        self.o    = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, n_heads, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, n_heads, self.hd).transpose(1, 2)
        q, k = self.rope(q, k, T)
        att  = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        att  = att.masked_fill(mask, float('-inf'))
        att  = F.softmax(att, dim=-1)
        return self.o((att @ v).transpose(1, 2).contiguous().view(B, T, d_model))


class FF(nn.Module):
    """
    SwiGLU FFN.
    w1/w2: low-rank up-projections (rank*2).
    w3: full-rank down-projection.
    Light dropout (0.05) on the gated activation.
    """
    def __init__(self):
        super().__init__()
        h       = int(d_model * 2.2)
        r       = rank * 2
        self.w1 = LowRank(d_model, h, r)
        self.w2 = LowRank(d_model, h, r)
        self.w3 = nn.Linear(h, d_model, bias=False)
        self.drop = nn.Dropout(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────

class Model(nn.Module):
    """
    Recurrent block model.

    Architecture:
      - n_recur steps of interleaved Attn (every 2 steps) + FF (every step)
      - Per-step learned gates ga[i], gf[i] with tanh squeeze + 1/√n_recur scale
      - Cross-batch hidden state carry (sequential BPTT, detached every tbptt_steps)
      - Weight tying: embed ↔ head

    What is NOT here (and why):
      - No att *= constant    (unconditional sharpening, can't be unlearned)
      - No logit_bias         (conflicts with weight tying)
      - No output * constant  (RMSNorm.scale already handles this)
      - No training-time logit temperature (distorts loss surface)
      - No embedding noise    (random noise on embeddings, not a real regulariser)
      - No progressive depth  (gradient starvation on late-initialised gates)
    """
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn  = Attn()
        self.ff    = FF()
        self.na    = RMSNorm(d_model)
        self.nf    = RMSNorm(d_model)
        self.no    = RMSNorm(d_model)
        self.ga    = nn.Parameter(torch.zeros(n_recur))
        self.gf    = nn.Parameter(torch.zeros(n_recur))
        self.head  = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight   # weight tying
        self._state: torch.Tensor | None = None

    def reset_state(self):
        self._state = None

    def forward(self, idx: torch.Tensor, detach_state: bool = False) -> torch.Tensor:
        B, T = idx.shape
        x    = self.embed(idx)

        # Cross-batch state injection (only when shapes are compatible)
        if self._state is not None and self._state.shape[0] == B \
                and self._state.shape[1] >= T:
            x = x + state_weight * self._state[:, :T, :]

        scale = 1.0 / math.sqrt(n_recur)
        for i in range(n_recur):
            if i % 2 == 0:
                x = x + scale * torch.tanh(self.ga[i]) * self.attn(self.na(x))
            x = x + scale * torch.tanh(self.gf[i]) * self.ff(self.nf(x))

        self._state = x.detach() if detach_state else x
        return self.head(self.no(x))

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new: int  = 200,
        temperature: float = 0.8,   # sharpening lives HERE, not in training
        top_k: int   = 64,
    ) -> torch.Tensor:
        self.reset_state()
        self.eval()
        tokens = prompt.clone().unsqueeze(0)
        for _ in range(max_new):
            ctx    = tokens[:, -max_ctx:]
            logits = self(ctx)[:, -1, :]
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')
            nxt    = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tokens = torch.cat([tokens, nxt], dim=1)
        self.train()
        return tokens[0]

# ─────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────

def compute_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor | None,
    step: int,
) -> torch.Tensor:
    # Token-frequency weighted CE
    ce = (F.cross_entropy(
        logits.view(-1, vocab_size), y.view(-1),
        reduction='none', label_smoothing=label_smoothing,
    ) * token_weights[y.view(-1)]).mean()

    if not (use_distill and t is not None and step >= distill_start):
        return ce

    alpha = distill_alpha_hi if step < distill_pivot else distill_alpha_lo
    T     = distill_temp
    kl    = F.kl_div(
        F.log_softmax(logits / T, dim=-1),
        F.softmax(t / T,          dim=-1),
        reduction='batchmean',
    ) * (T * T)
    return alpha * kl + (1.0 - alpha) * ce

# ─────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────

def get_lr(step: int) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    p = (step - warmup) / (train_steps - warmup)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def eval_bpb(ema: nn.Module) -> float:
    ema.eval()
    ema.reset_state()
    total = 0.0
    for _ in range(20):
        ix = torch.randint(len(eval_data) - max_ctx - 1, (batch_size,))
        x  = torch.stack([eval_data[i:i+max_ctx]     for i in ix]).to(device)
        y  = torch.stack([eval_data[i+1:i+max_ctx+1] for i in ix]).to(device)
        total += F.cross_entropy(ema(x).view(-1, vocab_size), y.view(-1)).item()
    ema.reset_state()
    ema.train()
    return (total / 20) / math.log(2)

# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────

def train():
    model = Model().to(device)
    ema   = copy.deepcopy(model)
    opt   = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr, betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"v16 | {n_params/1e6:.2f}M params | {device} | "
          f"distill={use_distill} | vocab={vocab_size}\n")

    best_bpb = float('inf')

    for step in range(train_steps):
        for pg in opt.param_groups:
            pg['lr'] = get_lr(step)

        T      = get_seq_len(step)
        detach = sampler.should_detach()
        opt.zero_grad(set_to_none=True)

        for acc in range(accum_steps):
            x, y, t   = sampler.next(T)
            do_detach = detach and (acc == accum_steps - 1)
            with autocast(device_type='cuda' if use_amp else 'cpu'):
                logits = model(x, detach_state=do_detach)
                loss   = compute_loss(logits, y, t, step)
            scaler.scale(loss / accum_steps).backward()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        # 3-stage adaptive EMA
        if   step < train_steps * 0.375: decay = 0.999
        elif step < train_steps * 0.875: decay = 0.9995
        else:                            decay = 0.9999
        with torch.no_grad():
            for p, ep in zip(model.parameters(), ema.parameters()):
                ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

        if step % 500 == 0:
            bpb   = eval_bpb(ema)
            phase = "kd" if use_distill and step >= distill_start else "ce"
            print(f"step {step:5d} | bpb {bpb:.4f} | lr {get_lr(step):.2e} | "
                  f"ctx {T} | {phase}")
            if bpb < best_bpb:
                best_bpb = bpb
                torch.save(
                    {k: v.half().cpu() for k, v in ema.state_dict().items()},
                    "best_v16.pt",
                )
                print(f"           → new best {bpb:.4f}")

    print(f"\nDone. Best BPB = {best_bpb:.4f}")

    # Generation sample
    prompt = train_data[:32].to(device)
    out    = ema.generate(prompt, max_new=200, temperature=0.8, top_k=64)
    print("\nSample:\n" + "".join([chr(c) if 32 <= c < 127 else "·"
                                   for c in out.tolist()]))


if __name__ == "__main__":
    train()
