"""Trinity Ternary CPU Trainer — Apple M1 Pro edition.

Non-record submission exploring: can we train a compliant LM **entirely on CPU**?

Architecture:
- BitNet b1.58 style ternary weights {-1, 0, +1} with STE QAT from scratch
- 6L × 384d transformer, vocab=1024 (SP1024 tokenizer)
- Base-3 packing (5 trits/byte = 1.6 bits/trit, near theoretical optimum)
- No GPU, no MPS — pure CPU tensor ops (AMX via torch backend when available)

Target: BPB ≤ 1.3 in 24-48h on M1 Pro (10 cores, 16 GB RAM).

Compliance: Issue #1017 Track A (unlimited compute, non-record).
- Causal attention, standard softmax over full 1024 vocab
- No SLOT, no TTT, no n-gram
- Score-only eval: loss on val tokens without adaptation
"""
import os, sys, math, time, json, io, lzma, argparse, struct
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---- Config ----
class Config:
    # Model (scaled up after M1 Pro speed test: 0.37s/step for 7.5M)
    vocab_size = 1024
    num_layers = 10
    model_dim = 512
    num_heads = 8
    num_kv_heads = 8
    mlp_mult = 2.5
    seq_len = 512  # shorter than GPU for CPU speed
    tie_embeddings = True
    logit_softcap = 30.0

    # Training — v2: longer horizon + smarter schedules
    seed = 42
    batch_size = 8
    grad_accum_steps = 4  # effective batch = 32
    max_steps = int(os.environ.get("MAX_STEPS", 400000))  # 4× headroom
    max_wallclock_hours = float(os.environ.get("MAX_HOURS", 72.0))  # v2: 72h "до упора"
    lr = 3e-4
    lr_min = 3e-5  # v2: cosine decay to 10% of peak
    warmup_steps = 200
    weight_decay = 0.05
    grad_clip = 1.0

    # Ternary QAT (BitNet b1.58) — v3: STEP-based ramp (sleep-proof)
    ternary_warmup_steps = 500  # fp32 first, then ternarize
    # v3: α=1.0 reached at step 60000 (~72h at 0.27 steps/s)
    # Step-based — survives Mac sleep. Alpha only advances with training progress.
    ternary_ramp_end_step = 60000

    # Logging
    log_every = 50
    val_every = 500
    checkpoint_every_hours = 1.0

    # Paths
    train_file = "data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"
    val_file = "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
    tokenizer_file = "data/tokenizers/fineweb_1024_bpe.model"


# ---- BitNet b1.58 ternary quantization (STE) ----

def ternarize_weight(w: Tensor, scale: float = 1.0) -> Tensor:
    """Ternarize weight to {-scale, 0, +scale}. Straight-through estimator: grad passes through."""
    # Use mean absolute value as scale (BitNet b1.58 recipe)
    abs_mean = w.abs().mean().clamp(min=1e-5)
    threshold = 0.7 * abs_mean
    # Quantize: +1 if w > t, -1 if w < -t, else 0
    q = torch.where(w > threshold, torch.ones_like(w),
        torch.where(w < -threshold, -torch.ones_like(w), torch.zeros_like(w)))
    # STE: forward quantized, backward straight-through
    w_q = w + (q * abs_mean - w).detach()
    return w_q


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights at forward, fp32 master weights."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=1.0/math.sqrt(in_features))
        self.ternary_active = False
        self.ternary_alpha = 0.0  # blend factor 0=fp32, 1=full ternary

    def forward(self, x: Tensor) -> Tensor:
        if not self.ternary_active or self.ternary_alpha == 0:
            w_use = self.weight
        elif self.ternary_alpha >= 1.0:
            w_use = ternarize_weight(self.weight)
        else:
            # Blend: (1-alpha)*fp32 + alpha*ternary
            w_t = ternarize_weight(self.weight)
            w_use = (1 - self.ternary_alpha) * self.weight + self.ternary_alpha * w_t
        return F.linear(x, w_use, self.bias)


# ---- Model ----

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.model_dim // cfg.num_heads
        self.qkv = TernaryLinear(cfg.model_dim, cfg.model_dim * 3, bias=False)
        self.proj = TernaryLinear(cfg.model_dim, cfg.model_dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.model_dim * cfg.mlp_mult)
        self.fc = TernaryLinear(cfg.model_dim, hidden, bias=False)
        self.proj = TernaryLinear(hidden, cfg.model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.relu(self.fc(x)).pow(2))  # ReLU² like v3


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.model_dim)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.model_dim)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TrinityTernaryGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.model_dim)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
        # Precompute RoPE
        self.head_dim = cfg.model_dim // cfg.num_heads
        self.register_buffer('_rope_cache', None, persistent=False)

    def _build_rope(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        dim = self.head_dim
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        positions = torch.arange(seq_len, device=device).float()
        angles = torch.outer(positions, freqs)
        cos = torch.cat([angles.cos(), angles.cos()], dim=-1).unsqueeze(0).unsqueeze(0)
        sin = torch.cat([angles.sin(), angles.sin()], dim=-1).unsqueeze(0).unsqueeze(0)
        return cos, sin

    def forward(self, x: Tensor, y: Tensor = None) -> tuple[Tensor, Tensor]:
        B, T = x.shape
        cos, sin = self._build_rope(T, x.device)
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.final_norm(h)
        if self.cfg.tie_embeddings:
            logits = h @ self.tok_emb.weight.t()
        else:
            logits = self.lm_head(h)
        logits = self.cfg.logit_softcap * torch.tanh(logits / self.cfg.logit_softcap)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), y.reshape(-1))
        return logits, loss

    def set_ternary_alpha(self, alpha: float):
        """Blend: 0=fp32, 1=full ternary. Applies to all TernaryLinear."""
        for m in self.modules():
            if isinstance(m, TernaryLinear):
                m.ternary_active = True
                m.ternary_alpha = alpha


# ---- Data loader ----

def load_data_shard(filepath: str) -> np.ndarray:
    """Load a Parameter Golf data shard (256 int32 header + uint16 tokens)."""
    header_bytes = 256 * 4  # 256 int32s = 1024 bytes
    header = np.fromfile(filepath, dtype="<i4", count=256)
    assert header[0] == 20240520 and header[1] == 1, f"Bad header for {filepath}"
    num_tokens = int(header[2])
    tokens = np.fromfile(filepath, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens


class FineWebDataLoader:
    def __init__(self, filepath: str, batch_size: int, seq_len: int, rank: int = 0, world_size: int = 1):
        self.tokens = load_data_shard(filepath)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pos = rank * batch_size * seq_len
        self.stride = batch_size * seq_len * world_size
        assert self.tokens.max() < 1024, f"Found token {self.tokens.max()}, vocab should be 1024"

    def next_batch(self) -> tuple[Tensor, Tensor]:
        chunk = self.tokens[self.pos:self.pos + self.batch_size * self.seq_len + 1]
        if len(chunk) < self.batch_size * self.seq_len + 1:
            self.pos = 0
            chunk = self.tokens[0:self.batch_size * self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64).copy()).view(self.batch_size, self.seq_len)
        y = torch.from_numpy(chunk[1:].astype(np.int64).copy()).view(self.batch_size, self.seq_len)
        self.pos += self.stride
        return x, y


# ---- Training loop ----

def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device('cpu')
    # Use AMX via default backend
    torch.set_num_threads(os.cpu_count())

    print(f"=== Trinity Ternary CPU Trainer ===", flush=True)
    print(f"Device: {device}, threads: {torch.get_num_threads()}", flush=True)
    print(f"Model: {cfg.num_layers}L × {cfg.model_dim}d × {cfg.num_heads}h, vocab={cfg.vocab_size}, seq={cfg.seq_len}", flush=True)

    model = TrinityTernaryGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} ({n_params*4/1024/1024:.1f} MB fp32)", flush=True)
    print(f"Ternary-packed size estimate: {n_params*1.6/8/1024/1024:.2f} MB (5 trits/byte)", flush=True)

    # v3: warm-start from best v2 checkpoint (step 10391, val 2.35)
    warm_start = os.environ.get("WARM_START", "1")
    start_step = 0
    v2_best_ckpt = "/tmp/trinity_ternary_v2_ckpt_10391.pt"
    v1_ckpt = "/Users/ssdm4/Desktop/PROJECTS/CLAUDE/parameter-golf/final_model.pt"
    warm_path = v2_best_ckpt if os.path.exists(v2_best_ckpt) else v1_ckpt
    if warm_start == "1" and os.path.exists(warm_path):
        try:
            ckpt = torch.load(warm_path, map_location='cpu', weights_only=True)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model.load_state_dict(ckpt['model'], strict=True)
                start_step = ckpt.get('step', 0)
            else:
                model.load_state_dict(ckpt, strict=True)
            print(f"✓ Warm-started from {warm_path} (starting at step {start_step})", flush=True)
        except Exception as e:
            print(f"✗ Warm-start failed: {e}, training from scratch", flush=True)
    else:
        print(f"Training from scratch (no warm-start)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    train_loader = FineWebDataLoader(cfg.train_file, cfg.batch_size, cfg.seq_len)
    val_loader = FineWebDataLoader(cfg.val_file, cfg.batch_size, cfg.seq_len)

    # Training — v3: start from loaded checkpoint step
    start_time = time.time()
    last_checkpoint = start_time
    step = start_step  # v3: resume from warm-start step
    loss_ema = None

    print(f"\n=== Training started (max {cfg.max_wallclock_hours}h or {cfg.max_steps} steps) ===", flush=True)

    while step < cfg.max_steps:
        elapsed = time.time() - start_time
        if elapsed > cfg.max_wallclock_hours * 3600:
            print(f"\n⏱ Wallclock limit {cfg.max_wallclock_hours}h reached", flush=True)
            break

        # v3: STEP-based ternary schedule (sleep-proof)
        if step < cfg.ternary_warmup_steps:
            alpha = 0.0
        else:
            ramp_progress = (step - cfg.ternary_warmup_steps) / max(1, cfg.ternary_ramp_end_step - cfg.ternary_warmup_steps)
            alpha = min(1.0, ramp_progress)
        model.set_ternary_alpha(alpha)

        # v3: LR with linear warmup + cosine decay (step-based)
        if step < cfg.warmup_steps:
            lr_now = cfg.lr * (step / cfg.warmup_steps)
        else:
            # Cosine decay from step warmup_steps to ternary_ramp_end_step, then hold at lr_min
            decay_progress = min(1.0, max(0.0, (step - cfg.warmup_steps) / max(1, cfg.ternary_ramp_end_step - cfg.warmup_steps)))
            lr_now = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * decay_progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_loader.next_batch()
            _, loss = model(x, y)
            (loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item() / cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        step += 1
        loss_ema = accum_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * accum_loss

        if step % cfg.log_every == 0 or step == 1:
            mins = elapsed / 60
            rate = step / elapsed if elapsed > 0 else 0
            eta_min = (cfg.max_wallclock_hours * 3600 - elapsed) / 60
            print(f"  step {step}/{cfg.max_steps} loss={loss_ema:.4f} alpha={alpha:.3f} lr={lr_now:.2e} "
                  f"rate={rate:.2f}/s elapsed={mins:.0f}m eta={eta_min:.0f}m", flush=True)

        if step % cfg.val_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_batches = 10
                for _ in range(val_batches):
                    vx, vy = val_loader.next_batch()
                    _, vl = model(vx, vy)
                    val_loss += vl.item()
                val_loss /= val_batches
                val_bpb = val_loss / math.log(2.0) * 1.0  # tokens_per_byte ≈ 1 for SP1024 (exact later)
                print(f"  [VAL] step {step}: val_loss={val_loss:.4f} val_bpb≈{val_bpb:.4f}", flush=True)
            model.train()

        # v2: save best-so-far checkpoint + hourly
        if (time.time() - last_checkpoint) > cfg.checkpoint_every_hours * 3600:
            ckpt_path = f"/tmp/trinity_ternary_v3_ckpt_{step}.pt"
            torch.save({'model': model.state_dict(), 'step': step, 'loss': loss_ema, 'alpha': alpha}, ckpt_path)
            last_checkpoint = time.time()
            print(f"  [CKPT] saved {ckpt_path}", flush=True)

    # Final save — v2: save in submission folder, don't overwrite v1!
    final_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model_v3.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n=== Training done. Final model saved to {final_path} ===", flush=True)
    print(f"Total time: {(time.time()-start_time)/3600:.2f}h, final loss: {loss_ema:.4f}", flush=True)

    return model


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
