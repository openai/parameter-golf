"""
Parameter Golf — Local Research Fork
=====================================
Simplified training script for DGX Spark (GB10).
No Triton/torch.compile dependency. Uses PyTorch native SDPA.
Same model architecture, data, tokenizer, and BPB metric as official.

Usage:
  source .venv/bin/activate
  
  # Baseline (standard 9-layer, no modifications)
  python train_local.py --mode baseline

  # Fractal (weight-shared layers with loops)
  python train_local.py --mode fractal --num-unique-layers 3 --num-loops 3

  # Fractal + Gravity
  python train_local.py --mode fractal --gravity

  # Fractal + Gravity + AttnRes
  python train_local.py --mode fractal --gravity --attnres
"""

from __future__ import annotations
import argparse
import glob
import io
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ─── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "fractal"], default="baseline")
    p.add_argument("--num-layers", type=int, default=9, help="Number of layers for baseline mode")
    p.add_argument("--num-unique-layers", type=int, default=3)
    p.add_argument("--num-loops", type=int, default=3)
    p.add_argument("--model-dim", type=int, default=0, help="0 = auto-size to match baseline param count")
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--mlp-mult", type=int, default=2)
    p.add_argument("--gravity", action="store_true", help="Enable learned gravity aux losses")
    p.add_argument("--attnres", action="store_true", help="Enable attention residuals")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--batch-tokens", type=int, default=32768)
    p.add_argument("--max-seconds", type=float, default=120.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--data-path", type=str, default="./data/datasets/fineweb10B_sp1024")
    p.add_argument("--tokenizer-path", type=str, default="./data/tokenizers/fineweb_1024_bpe.model")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval-tokens", type=int, default=0, help="0 = full val set, >0 = truncated for speed")
    p.add_argument("--run-id", type=str, default="local")
    return p.parse_args()

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_shard(path: Path) -> Tensor:
    header = np.fromfile(path, dtype="<i4", count=256)
    n_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=n_tokens, offset=256 * 4)
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.idx = 0
        self.tokens = load_shard(Path(self.files[0]))
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.idx = (self.idx + 1) % len(self.files)
                self.tokens = load_shard(Path(self.files[self.idx]))
                self.pos = 0
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

# ─── BPB EVALUATION ──────────────────────────────────────────────────────────

def build_bpb_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size())
    table_size = max(sp_vs, vocab_size)
    base_bytes = np.zeros(table_size, dtype=np.int16)
    has_space = np.zeros(table_size, dtype=np.bool_)
    is_boundary = np.ones(table_size, dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary, dtype=torch.bool, device=device),
    )

@torch.no_grad()
def eval_bpb(model, val_tokens, seq_len, batch_tokens, device, base_bytes_lut, has_space_lut, is_boundary_lut):
    model.eval()
    local_batch_seqs = max(1, batch_tokens // seq_len)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0

    for start in range(0, total_seqs, local_batch_seqs):
        end = min(start + local_batch_seqs, total_seqs)
        raw_start = start * seq_len
        raw_end = end * seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
            if isinstance(loss, tuple):
                loss = loss[0]  # gravity returns (total_loss, final_loss)
        n = float(y.numel())
        loss_sum += loss.item() * n
        token_count += n
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        tb = base_bytes_lut[tgt_ids].to(torch.int16)
        tb += (has_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).to(torch.int16)
        byte_count += tb.to(torch.float64).sum().item()

    model.train()
    val_loss = loss_sum / token_count
    bpt = val_loss / math.log(2.0)
    tpb = token_count / byte_count
    return val_loss, bpt * tpb

# ─── MODEL: SHARED COMPONENTS ────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0
        self._cos = None
        self._sin = None

    def forward(self, seq_len, device, dtype):
        if self._cos is None or self._cache_len < seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
            self._cache_len = seq_len
        return self._cos[:, :, :seq_len].to(dtype), self._sin[:, :, :seq_len].to(dtype)

def apply_rope(x, cos, sin):
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, rope_base=10000.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        kv_dim = n_kv_heads * self.head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.rotary = Rotary(self.head_dim, rope_base)

    def forward(self, x):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=(self.n_kv_heads != self.n_heads))
        return self.c_proj(y.transpose(1, 2).contiguous().reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        hidden = dim * mult
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())

class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, mlp_mult):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = Attention(dim, n_heads, n_kv_heads)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x + self.attn_scale * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale * self.mlp(self.mlp_norm(x))
        return x

# ─── MODEL: BASELINE (standard 9-layer) ──────────────────────────────────────

class BaselineGPT(nn.Module):
    def __init__(self, vocab_size, num_layers, dim, n_heads, n_kv_heads, mlp_mult,
                 softcap=30.0):
        super().__init__()
        self.softcap = softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        n_enc = num_layers // 2
        n_dec = num_layers - n_enc
        n_skip = min(n_enc, n_dec)
        self.n_enc = n_enc
        self.n_dec = n_dec
        self.skip_weights = nn.Parameter(torch.ones(n_skip, dim))
        self.blocks = nn.ModuleList([Block(dim, n_heads, n_kv_heads, mlp_mult)
                                     for _ in range(num_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Tie embeddings
        self.lm_head.weight = self.tok_emb.weight
        self._init()

    def _init(self):
        nn.init.normal_(self.tok_emb.weight, std=0.005)
        for block in self.blocks:
            for m in [block.attn.c_q, block.attn.c_k, block.attn.c_v, block.mlp.fc]:
                nn.init.normal_(m.weight, std=0.02)
            for m in [block.attn.c_proj, block.mlp.proj]:
                nn.init.zeros_(m.weight)

    def forward(self, x_ids, targets):
        x = F.rms_norm(self.tok_emb(x_ids), (self.tok_emb.weight.size(-1),))
        x0 = x
        skips = []
        for i in range(self.n_enc):
            x = self.blocks[i](x)
            skips.append(x)
        for i in range(self.n_dec):
            if skips:
                x = x + self.skip_weights[i] * skips.pop()
            x = self.blocks[self.n_enc + i](x)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = self.lm_head(x)
        logits = self.softcap * torch.tanh(logits / self.softcap)
        return F.cross_entropy(logits.float(), targets.reshape(-1))

# ─── MODEL: FRACTAL (weight-shared + gravity + attnres) ──────────────────────

class AttnResModule(nn.Module):
    """Attention over previous loop outputs. One learned query per layer."""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.01)
        self.norm = RMSNorm()

    def forward(self, loop_outputs):
        """
        loop_outputs: list of [B, T, D] tensors (previous loop outputs)
        Returns: [B, T, D] weighted combination
        """
        if len(loop_outputs) == 1:
            return loop_outputs[0]
        V = torch.stack(loop_outputs, dim=0)  # [N, B, T, D]
        K = self.norm(V)
        logits = torch.einsum('d, n b t d -> n b t', self.query, K)
        weights = logits.softmax(dim=0)
        return torch.einsum('n b t, n b t d -> b t d', weights, V)

class FractalGPT(nn.Module):
    def __init__(self, vocab_size, num_unique_layers, num_loops, dim, n_heads,
                 n_kv_heads, mlp_mult, use_gravity=False, use_attnres=False,
                 softcap=30.0):
        super().__init__()
        self.num_loops = num_loops
        self.num_unique_layers = num_unique_layers
        self.use_gravity = use_gravity
        self.use_attnres = use_attnres
        self.softcap = softcap
        self.dim = dim

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, n_kv_heads, mlp_mult)
                                     for _ in range(num_unique_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        # Tie embeddings
        self.lm_head.weight = self.tok_emb.weight

        # Loop position embeddings
        self.loop_pos = nn.Parameter(torch.randn(num_loops, dim) * 0.01)

        # Gravity: learned auxiliary loss weights
        if use_gravity:
            self.gravity_logits = nn.Parameter(torch.tensor(
                [-2.0] * (num_loops - 1) + [0.0]  # softplus → ~[0.13, ..., 0.69]
            ))

        # AttnRes: one module per loop (except first loop which has nothing to attend to)
        if use_attnres:
            total_layers = num_unique_layers * num_loops
            self.attnres = nn.ModuleList([
                AttnResModule(dim) for _ in range(total_layers)
            ])

        self._init()

    def _init(self):
        nn.init.normal_(self.tok_emb.weight, std=0.005)
        for block in self.blocks:
            for m in [block.attn.c_q, block.attn.c_k, block.attn.c_v, block.mlp.fc]:
                nn.init.normal_(m.weight, std=0.02)
            for m in [block.attn.c_proj, block.mlp.proj]:
                nn.init.zeros_(m.weight)

    def _compute_logits(self, x):
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = self.lm_head(x)
        return self.softcap * torch.tanh(logits / self.softcap)

    def forward(self, x_ids, targets):
        x = F.rms_norm(self.tok_emb(x_ids), (self.tok_emb.weight.size(-1),))

        loop_outputs = [x]  # embedding is always available for AttnRes
        gravity_losses = []
        flat_layer_idx = 0

        for loop in range(self.num_loops):
            # Add loop position embedding
            x = x + self.loop_pos[loop]

            # Run shared layers
            for layer_idx in range(self.num_unique_layers):
                # AttnRes: attend over previous loop outputs before this layer
                if self.use_attnres and len(loop_outputs) > 1:
                    x = self.attnres[flat_layer_idx](loop_outputs + [x])

                x = self.blocks[layer_idx](x)
                flat_layer_idx += 1

            # Store this loop's output for future AttnRes
            loop_outputs.append(x)

            # Gravity: compute auxiliary loss at loop boundary
            if self.use_gravity and loop < self.num_loops - 1:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    aux_logits = self._compute_logits(x)
                    aux_loss = F.cross_entropy(aux_logits.float(), targets.reshape(-1))
                weight = F.softplus(self.gravity_logits[loop])
                gravity_losses.append(weight * aux_loss)

        # Final loss (always weight 1.0 equivalent)
        final_logits = self._compute_logits(x)
        final_loss = F.cross_entropy(final_logits.float(), targets.reshape(-1))

        if self.use_gravity and gravity_losses:
            final_weight = F.softplus(self.gravity_logits[-1])
            total_loss = sum(gravity_losses) + final_weight * final_loss
            # Normalize so total weight sums to ~1
            total_weight = sum(F.softplus(self.gravity_logits[i]) for i in range(self.num_loops))
            total_loss = total_loss / total_weight
            return total_loss
        
        return final_loss

# ─── OPTIMIZER ────────────────────────────────────────────────────────────────

def make_optimizer(model, lr):
    """Simple AdamW — we'll add Muon later if needed."""
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=True)

def cosine_lr(step, max_steps, lr, warmup=20, min_frac=0.1):
    if step < warmup:
        return lr * step / warmup
    decay = (step - warmup) / max(max_steps - warmup, 1)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * decay)))

# ─── AUTO-SIZE MODEL DIM ─────────────────────────────────────────────────────

def estimate_params(dim, n_heads, n_kv_heads, mlp_mult, num_unique_layers, vocab_size):
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    per_layer = (
        dim * dim +        # c_q
        dim * kv_dim +     # c_k
        dim * kv_dim +     # c_v
        dim * dim +        # c_proj
        dim * (dim * mlp_mult) +  # fc
        (dim * mlp_mult) * dim +  # proj
        dim * 2            # scales
    )
    total = (
        vocab_size * dim +           # embedding (tied with lm_head)
        num_unique_layers * per_layer  # transformer layers
    )
    return total

def auto_dim(target_params, n_heads, n_kv_heads, mlp_mult, num_unique_layers, vocab_size):
    """Find the largest dim (divisible by 2*n_heads for RoPE) that fits in target_params."""
    step = 2 * n_heads  # must be divisible by 2*n_heads so head_dim is even
    for dim in range(2048, 128, -step):
        if estimate_params(dim, n_heads, n_kv_heads, mlp_mult, num_unique_layers, vocab_size) <= target_params:
            return dim
    return 256

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 70)
    print(f"PARAMETER GOLF LOCAL — mode={args.mode}")
    print("=" * 70)

    # Tokenizer + BPB setup
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_space_lut, is_boundary_lut = build_bpb_luts(sp, args.vocab_size, device)

    # Validation data
    val_files = sorted(glob.glob(os.path.join(args.data_path, "fineweb_val_*.bin")))
    val_tokens = torch.cat([load_shard(Path(f)) for f in val_files])
    usable = ((val_tokens.numel() - 1) // args.seq_len) * args.seq_len
    val_tokens = val_tokens[:usable + 1]
    if args.eval_tokens > 0:
        max_eval = min(args.eval_tokens + 1, val_tokens.numel())
        eval_usable = ((max_eval - 1) // args.seq_len) * args.seq_len
        val_tokens = val_tokens[:eval_usable + 1]
    print(f"Val tokens: {val_tokens.numel():,}{' (truncated)' if args.eval_tokens > 0 else ''}")

    # Train data
    train_stream = TokenStream(os.path.join(args.data_path, "fineweb_train_*.bin"))

    # Baseline param count for auto-sizing
    BASELINE_PARAMS = estimate_params(512, 8, 4, 2, 9, args.vocab_size)

    # Build model
    if args.mode == "baseline":
        dim = args.model_dim if args.model_dim > 0 else 512
        model = BaselineGPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, dim=dim,
            n_heads=args.num_heads, n_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
        ).to(device).bfloat16()
    else:
        # Auto-size dim to match baseline param count
        if args.model_dim > 0:
            dim = args.model_dim
        else:
            dim = auto_dim(BASELINE_PARAMS, args.num_heads, args.num_kv_heads,
                          args.mlp_mult, args.num_unique_layers, args.vocab_size)
            # Ensure divisible by 2*num_heads (RoPE needs even head_dim)
            step = 2 * args.num_heads
            dim = (dim // step) * step

        model = FractalGPT(
            vocab_size=args.vocab_size,
            num_unique_layers=args.num_unique_layers,
            num_loops=args.num_loops,
            dim=dim,
            n_heads=args.num_heads,
            n_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            use_gravity=args.gravity,
            use_attnres=args.attnres,
        ).to(device).bfloat16()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")
    if args.mode == "fractal":
        print(f"  unique_layers={args.num_unique_layers} loops={args.num_loops} dim={dim}")
        print(f"  gravity={args.gravity} attnres={args.attnres}")
        print(f"  effective_depth={args.num_unique_layers * args.num_loops}")
    else:
        print(f"  layers={args.num_layers} dim={dim}")
    print(f"  baseline_params={BASELINE_PARAMS:,}")

    optimizer = make_optimizer(model, args.lr)
    seq_len = args.seq_len
    seqs_per_batch = max(1, args.batch_tokens // seq_len)

    # Training loop
    print(f"\nTraining: {args.iterations} iters, {args.max_seconds:.0f}s max, "
          f"batch={seqs_per_batch * seq_len} tokens")
    model.train()
    t_start = time.time()
    train_time_ms = 0.0

    for step in range(1, args.iterations + 1):
        # LR schedule
        lr = cosine_lr(step, args.iterations, args.lr, args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Get batch
        chunk = train_stream.take(seqs_per_batch * seq_len + 1).to(torch.int64)
        x = chunk[:-1].reshape(seqs_per_batch, seq_len).to(device)
        y = chunk[1:].reshape(seqs_per_batch, seq_len).to(device)

        # Forward / backward
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
            if isinstance(loss, tuple):
                loss = loss[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        elapsed = time.time() - t_start
        train_time_ms = elapsed * 1000

        if step % args.log_every == 0 or step <= 5:
            print(f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} "
                  f"lr:{lr:.2e} time:{train_time_ms:.0f}ms "
                  f"step_avg:{train_time_ms/step:.1f}ms")

        # Wallclock cap
        if args.max_seconds > 0 and elapsed >= args.max_seconds:
            print(f"Wallclock cap reached at step {step} ({elapsed:.1f}s)")
            break

    # Eval
    print("\nEvaluating...")
    val_loss, val_bpb = eval_bpb(
        model, val_tokens, seq_len, args.batch_tokens, device,
        base_bytes_lut, has_space_lut, is_boundary_lut,
    )
    print(f"\nval_loss: {val_loss:.4f}")
    print(f"val_bpb:  {val_bpb:.6f}")
    print(f"params:   {n_params:,}")
    print(f"time:     {train_time_ms:.0f}ms")
    print(f"steps:    {step}")

    # Gravity weights (if applicable)
    if args.mode == "fractal" and args.gravity:
        gw = [F.softplus(model.gravity_logits[i]).item() for i in range(model.num_loops)]
        print(f"gravity_weights: {['%.4f' % w for w in gw]}")

    # Quick size estimate
    state = model.state_dict()
    buf = io.BytesIO()
    torch.save(state, buf)
    raw = len(buf.getvalue())
    compressed = len(zlib.compress(buf.getvalue(), 9))
    print(f"raw_model_size: {raw:,} bytes ({raw/1e6:.1f}MB)")
    print(f"zlib_compressed: {compressed:,} bytes ({compressed/1e6:.1f}MB)")

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"peak_vram: {peak_mem:.0f} MiB")

if __name__ == "__main__":
    main()
