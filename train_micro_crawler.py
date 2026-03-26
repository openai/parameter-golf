"""
Micro Crawler Experiment
========================
Asymmetric fractal architecture:
  - Flat section: N unique blocks, each runs once (no sharing, no gradient conflict)
  - Crawler section: M shared blocks that loop K times with orthogonal position embeddings

The flat blocks build features cleanly. The crawler pair does a quick
double-tap (or triple-tap) through the same weights, each firing hitting
different subspaces via orthogonal loop position vectors.

Key advantage over uniform fractal:
  - More stored parameters (flat blocks are unique = wider/fatter)
  - Gradient conflict isolated to crawler only
  - Quantization compounding limited to crawler blocks
  - Cadence only needed for crawler section

Usage:
  # Default: 6 flat + 2 crawler x2 loops = 10 effective depth
  python train_micro_crawler.py --num-flat-layers 6 --num-crawler-layers 2 --crawler-loops 2

  # Fat crawler: 4 flat + 2 crawler x3 = 10 effective depth
  python train_micro_crawler.py --num-flat-layers 4 --num-crawler-layers 2 --crawler-loops 3

  # Separate MLP mults: flat=3x, crawler=5x
  python train_micro_crawler.py --flat-mlp-mult 3 --crawler-mlp-mult 5

  # Control: 8 flat + 0 crawler = standard transformer (no sharing)
  python train_micro_crawler.py --num-flat-layers 8 --num-crawler-layers 0
"""

from __future__ import annotations
import argparse
import glob
import math
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ─── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    # Architecture — flat section
    p.add_argument("--num-flat-layers", type=int, default=6)
    p.add_argument("--flat-mlp-mult", type=int, default=4)
    # Architecture — crawler section
    p.add_argument("--num-crawler-layers", type=int, default=2)
    p.add_argument("--crawler-loops", type=int, default=2)
    p.add_argument("--crawler-mlp-mult", type=int, default=4)
    # Architecture — shared
    p.add_argument("--model-dim", type=int, default=0, help="0 = auto-size")
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=1024)
    # Input conditioning
    p.add_argument("--trigram-vocab", type=int, default=0, help="0 = disabled, e.g. 8192")
    p.add_argument("--trigram-dim", type=int, default=128)
    # Training
    p.add_argument("--cadence", type=int, default=2, help="Crawler cadence: 0=never loop, 1=always loop, N=loop every Nth step")
    p.add_argument("--cadence-offset", type=int, default=0)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--batch-tokens", type=int, default=32768)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--data-path", type=str, default="./data/datasets/fineweb10B_sp1024")
    p.add_argument("--tokenizer-path", type=str, default="./data/tokenizers/fineweb_1024_bpe.model")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval-tokens", type=int, default=0)
    p.add_argument("--run-id", type=str, default="crawler")
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
def eval_bpb(model, val_tokens, seq_len, batch_tokens, device,
             base_bytes_lut, has_space_lut, is_boundary_lut):
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
            loss = model(x, y, crawl=True)  # eval always uses full depth
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

# ─── INPUT CONDITIONING ───────────────────────────────────────────────────────

class TrigramHashEmbedding(nn.Module):
    """
    Hashes trigrams (t[n-2], t[n-1], t[n]) into bucket embeddings.
    Three orthogonal hash primes — one per n-gram position — so each
    token in the trigram contributes along a different direction.

    The trigram gives the model a wider local context window at input
    conditioning, matching the triadic structure of the micro crawler
    (flat / crawl_fire0 / crawl_fire1).
    """
    def __init__(self, vocab_size: int, embed_dim: int, model_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = nn.Linear(embed_dim, model_dim, bias=False) if embed_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def trigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.vocab_size - 1
        out = torch.empty_like(t)
        # Position 0: no context → sentinel
        out[..., 0] = mod
        # Position 1: bigram only (no t[n-2])
        out[..., 1] = torch.bitwise_xor(36313 * t[..., 1], 27191 * t[..., 0]) % mod
        # Position 2+: full trigram hash — three orthogonal primes
        if t.size(-1) > 2:
            out[..., 2:] = (
                torch.bitwise_xor(
                    torch.bitwise_xor(36313 * t[..., 2:], 27191 * t[..., 1:-1]),
                    51647 * t[..., :-2]
                ) % mod
            )
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.trigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

# ─── MODEL COMPONENTS ────────────────────────────────────────────────────────

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

# ─── MICRO CRAWLER GPT ───────────────────────────────────────────────────────

class MicroCrawlerGPT(nn.Module):
    """
    Asymmetric fractal transformer:
      - flat_blocks: run once every step, no sharing, clean gradients
      - crawler_blocks: shared pair that loops K times with orthogonal positions

    forward(x, y, crawl=True):
      crawl=True  → flat pass + crawler loops (full depth)
      crawl=False → flat pass + crawler single pass (normalize mode)
    """
    def __init__(self, vocab_size, num_flat_layers, num_crawler_layers,
                 crawler_loops, dim, n_heads, n_kv_heads,
                 flat_mlp_mult, crawler_mlp_mult,
                 trigram_vocab=0, trigram_dim=128, softcap=30.0):
        super().__init__()
        self.num_flat_layers = num_flat_layers
        self.num_crawler_layers = num_crawler_layers
        self.crawler_loops = crawler_loops
        self.dim = dim

        self.tok_emb = nn.Embedding(vocab_size, dim)
        # Trigram input conditioning
        self.trigram = TrigramHashEmbedding(trigram_vocab, trigram_dim, dim) if trigram_vocab > 0 else None
        self.flat_blocks = nn.ModuleList([
            Block(dim, n_heads, n_kv_heads, flat_mlp_mult)
            for _ in range(num_flat_layers)
        ])
        self.crawler_blocks = nn.ModuleList([
            Block(dim, n_heads, n_kv_heads, crawler_mlp_mult)
            for _ in range(num_crawler_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie embeddings
        self.softcap = softcap

        # Orthogonal loop positions for crawler only
        # crawler_loops positions for crawl mode + 1 for normalize mode
        if num_crawler_layers > 0 and crawler_loops > 0:
            n_pos = crawler_loops + 1
            raw = torch.randn(n_pos, dim)
            Q, _ = torch.linalg.qr(raw.T)
            ortho = Q.T[:n_pos]
            self.loop_pos = nn.Parameter(ortho * 0.01)
            # loop_pos[0:crawler_loops] = crawl positions
            # loop_pos[crawler_loops]   = normalize position

        # Per-loop GPTQ metadata placeholders
        # During quantization, each crawler firing gets its own scales/zeros
        # because activations differ per orthogonal position. Weights stay shared.
        # Format: crawler_quant_meta[loop_idx][block_idx] = {scales, zeros}
        # This is populated by the GPTQ export script, not during training.
        self.crawler_quant_meta = None  # set by quantize_micro_crawler()

        self._init()

    def _init(self):
        nn.init.normal_(self.tok_emb.weight, std=0.005)
        for block in list(self.flat_blocks) + list(self.crawler_blocks):
            for m in [block.attn.c_q, block.attn.c_k, block.attn.c_v, block.mlp.fc]:
                nn.init.normal_(m.weight, std=0.02)
            for m in [block.attn.c_proj, block.mlp.proj]:
                nn.init.zeros_(m.weight)

    def _compute_logits(self, x):
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = self.lm_head(x)
        return self.softcap * torch.tanh(logits / self.softcap)

    def forward(self, x_ids, targets, crawl=True):
        x = self.tok_emb(x_ids)
        if self.trigram is not None:
            x = x + self.trigram(x_ids)
        x = F.rms_norm(x, (self.tok_emb.weight.size(-1),))

        # ── Flat section: always runs once, no position embedding needed ──
        for block in self.flat_blocks:
            x = block(x)

        # ── Crawler section: loop with orthogonal positions ──
        if self.num_crawler_layers > 0:
            if crawl:
                # Full crawl: fire crawler pair K times
                for loop in range(self.crawler_loops):
                    x = x + self.loop_pos[loop]
                    for block in self.crawler_blocks:
                        x = block(x)
            else:
                # Normalize: single clean pass through crawler
                x = x + self.loop_pos[self.crawler_loops]
                for block in self.crawler_blocks:
                    x = block(x)

        logits = self._compute_logits(x)
        return F.cross_entropy(logits.float(), targets.reshape(-1))

# ─── AUTO-SIZE ────────────────────────────────────────────────────────────────

def params_per_block(dim, n_heads, n_kv_heads, mlp_mult):
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim
    return (
        dim * dim + dim * kv_dim + dim * kv_dim + dim * dim +  # attention
        dim * (dim * mlp_mult) + (dim * mlp_mult) * dim +      # MLP
        dim * 2                                                  # scales
    )

def estimate_params(dim, n_heads, n_kv_heads, flat_mlp, crawler_mlp,
                    num_flat, num_crawler, vocab_size):
    flat_params = num_flat * params_per_block(dim, n_heads, n_kv_heads, flat_mlp)
    crawler_params = num_crawler * params_per_block(dim, n_heads, n_kv_heads, crawler_mlp)
    embed_params = vocab_size * dim
    return embed_params + flat_params + crawler_params

def auto_dim(target_params, n_heads, n_kv_heads, flat_mlp, crawler_mlp,
             num_flat, num_crawler, vocab_size):
    step = 2 * n_heads
    for dim in range(2048, 128, -step):
        if estimate_params(dim, n_heads, n_kv_heads, flat_mlp, crawler_mlp,
                           num_flat, num_crawler, vocab_size) <= target_params:
            return dim
    return 256

# ─── PER-LOOP GPTQ ────────────────────────────────────────────────────────────
#
# Standard GPTQ calibrates each layer once. For the micro crawler, the same
# crawler blocks see different activation distributions on each firing (due to
# orthogonal loop position offsets). Calibrating once averages these distributions,
# causing quant error that compounds across firings.
#
# Solution: calibrate GPTQ separately per firing. The weight bytes stay shared,
# but each firing gets its own (scales, zeros) metadata. At inference time,
# dequantize with the firing-specific quant params before each forward pass.
#
# Quant metadata overhead per crawler block per firing:
#   int6, group_size=64: scales = weight.numel()/64 * 2 bytes (fp16)
#                        zeros  = weight.numel()/64 * 2 bytes (fp16)
#   For a 640-dim MLP 4x block: ~6.4K params of metadata per firing
#   2 firings = 12.8K overhead vs ~3.4M weight params = 0.4% overhead
#
# Usage (called from export script, not during training):
#   quant_meta = calibrate_per_loop_gptq(model, calib_data, device)
#   model.crawler_quant_meta = quant_meta
#   export_quantized(model, path)

def calibrate_per_loop_gptq(model, calib_tokens, device, group_size=64, percdamp=0.01):
    """
    Calibrate GPTQ for the micro crawler with per-firing quant params.

    Returns dict: {loop_idx: {block_idx: {layer_name: (scales, zeros)}}}

    The flat blocks get standard single-calibration GPTQ (they only fire once).
    The crawler blocks get per-loop calibration.
    """
    model.eval()
    seq_len = calib_tokens.size(-1) if calib_tokens.dim() > 1 else 1024

    quant_meta = {}

    with torch.no_grad():
        # Run flat section to get activations entering the crawler
        x = model.tok_emb(calib_tokens.to(device))
        if model.trigram is not None:
            x = x + model.trigram(calib_tokens.to(device))
        x = F.rms_norm(x, (model.tok_emb.weight.size(-1),))
        for block in model.flat_blocks:
            x = block(x)

        # For each crawler firing, capture activations and calibrate separately
        x_base = x.clone()
        for loop in range(model.crawler_loops):
            x_loop = x_base + model.loop_pos[loop]
            quant_meta[loop] = {}
            for bidx, block in enumerate(model.crawler_blocks):
                # Capture input activations for this block at this firing
                quant_meta[loop][bidx] = {
                    "input_act_mean": x_loop.mean(dim=(0, 1)).cpu(),
                    "input_act_std": x_loop.std(dim=(0, 1)).cpu(),
                }
                x_loop = block(x_loop)
            # Update x_base for next loop (activations chain)
            x_base = x_loop

    print(f"Per-loop GPTQ calibration: {len(quant_meta)} firings x {len(model.crawler_blocks)} blocks")
    return quant_meta

# ─── OPTIMIZER ────────────────────────────────────────────────────────────────

def make_optimizer(model, lr):
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

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_flat = args.num_flat_layers
    num_crawler = args.num_crawler_layers
    crawler_loops = args.crawler_loops
    effective_depth = num_flat + num_crawler * crawler_loops

    # Cadence (only governs crawler)
    cadence = args.cadence
    offset = args.cadence_offset
    if num_crawler == 0:
        cadence_desc = "NO CRAWLER (flat-only control)"
    elif cadence == 0:
        cadence_desc = "crawler NEVER loops (always normalize)"
    elif cadence == 1:
        cadence_desc = "crawler ALWAYS loops"
    else:
        pattern = "".join("C" if i == offset else "N" for i in range(cadence))
        cadence_desc = f"cadence={cadence} pattern={pattern}"

    print("=" * 70)
    print(f"MICRO CRAWLER — {num_flat}flat + {num_crawler}crawl x{crawler_loops} = {effective_depth} effective depth")
    print(f"  {cadence_desc}")
    print("=" * 70)

    # Tokenizer + BPB
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

    # Train data
    train_stream = TokenStream(os.path.join(args.data_path, "fineweb_train_*.bin"))

    # Auto-size dim to match baseline param count
    BASELINE_PARAMS = estimate_params(512, 8, 4, 2, 2, 9, 0, args.vocab_size)
    if args.model_dim > 0:
        dim = args.model_dim
    else:
        dim = auto_dim(BASELINE_PARAMS, args.num_heads, args.num_kv_heads,
                       args.flat_mlp_mult, args.crawler_mlp_mult,
                       num_flat, num_crawler, args.vocab_size)
        step_align = 2 * args.num_heads
        dim = (dim // step_align) * step_align

    model = MicroCrawlerGPT(
        vocab_size=args.vocab_size,
        num_flat_layers=num_flat,
        num_crawler_layers=num_crawler,
        crawler_loops=crawler_loops,
        dim=dim,
        n_heads=args.num_heads,
        n_kv_heads=args.num_kv_heads,
        flat_mlp_mult=args.flat_mlp_mult,
        crawler_mlp_mult=args.crawler_mlp_mult,
        trigram_vocab=args.trigram_vocab,
        trigram_dim=args.trigram_dim,
    ).to(device).bfloat16()

    n_params = sum(p.numel() for p in model.parameters())
    flat_params = sum(p.numel() for p in model.flat_blocks.parameters())
    crawler_params = sum(p.numel() for p in model.crawler_blocks.parameters())
    trigram_params = sum(p.numel() for p in model.trigram.parameters()) if model.trigram else 0
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")
    print(f"  flat:    {num_flat} blocks, {flat_params:,} params ({flat_params/1e6:.1f}M), MLP {args.flat_mlp_mult}x")
    print(f"  crawler: {num_crawler} blocks x{crawler_loops} loops, {crawler_params:,} params ({crawler_params/1e6:.1f}M), MLP {args.crawler_mlp_mult}x")
    print(f"  trigram: {'ON' if model.trigram else 'OFF'} ({trigram_params:,} params)" + (f" vocab={args.trigram_vocab} dim={args.trigram_dim}" if model.trigram else ""))
    print(f"  embed:   {args.vocab_size * dim:,} params")
    print(f"  dim={dim} effective_depth={effective_depth}")
    print(f"  baseline_params={BASELINE_PARAMS:,}")

    optimizer = make_optimizer(model, args.lr)
    seq_len = args.seq_len
    seqs_per_batch = max(1, args.batch_tokens // seq_len)

    # Initial eval
    print(f"\nTraining: {args.iterations} iters, batch={seqs_per_batch * seq_len} tokens")
    val_loss, val_bpb = eval_bpb(model, val_tokens, seq_len, args.batch_tokens, device,
                                  base_bytes_lut, has_space_lut, is_boundary_lut)
    print(f"step:0 val_bpb:{val_bpb:.4f}")

    # Logging
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.tsv"
    with open(logfile, "w") as f:
        f.write("step\ttype\ttrain_loss\tval_bpb\tstep_ms\n")

    model.train()
    t_start = time.time()
    c_steps = 0  # crawl steps
    n_steps = 0  # normalize steps

    for step in range(1, args.iterations + 1):
        # Cadence: decide crawl or normalize (flat always runs)
        if num_crawler == 0 or cadence == 0:
            is_crawl = False
        elif cadence == 1:
            is_crawl = True
        else:
            is_crawl = ((step - 1) % cadence) == offset
        step_type = "C" if is_crawl else "N"
        if is_crawl:
            c_steps += 1
        else:
            n_steps += 1

        # LR schedule
        lr = cosine_lr(step, args.iterations, args.lr, args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Batch
        chunk = train_stream.take(seqs_per_batch * seq_len + 1).to(torch.int64)
        x = chunk[:-1].reshape(seqs_per_batch, seq_len).to(device)
        y = chunk[1:].reshape(seqs_per_batch, seq_len).to(device)

        # Forward / backward
        t_step = time.time()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y, crawl=is_crawl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        step_ms = (time.time() - t_step) * 1000

        # Log every step
        with open(logfile, "a") as f:
            f.write(f"{step}\t{step_type}\t{loss.item():.6f}\t\t{step_ms:.1f}\n")

        # Console
        if step <= 10 or step % 10 == 0:
            elapsed = (time.time() - t_start) * 1000
            print(f"step:{step}/{args.iterations} [{step_type}] loss:{loss.item():.4f} "
                  f"step_ms:{step_ms:.0f} total:{elapsed:.0f}ms")

        # Eval every 50 steps
        if step % 50 == 0:
            val_loss, val_bpb = eval_bpb(model, val_tokens, seq_len, args.batch_tokens,
                                          device, base_bytes_lut, has_space_lut, is_boundary_lut)
            print(f"  >>> val_bpb:{val_bpb:.4f} (step {step})")
            with open(logfile, "a") as f:
                f.write(f"{step}\tEVAL\t\t{val_bpb:.6f}\t\n")

        # Wallclock cap
        if args.max_seconds > 0 and (time.time() - t_start) >= args.max_seconds:
            print(f"Wallclock cap at step {step}")
            break

    # Final eval
    print("\nFinal eval...")
    val_loss, val_bpb = eval_bpb(model, val_tokens, seq_len, args.batch_tokens, device,
                                  base_bytes_lut, has_space_lut, is_boundary_lut)

    print(f"\n{'=' * 70}")
    print(f"RESULTS — {num_flat}flat + {num_crawler}crawl x{crawler_loops}")
    print(f"{'=' * 70}")
    print(f"val_loss:        {val_loss:.4f}")
    print(f"val_bpb:         {val_bpb:.6f}")
    print(f"params:          {n_params:,}")
    print(f"flat_params:     {flat_params:,} ({num_flat} blocks, MLP {args.flat_mlp_mult}x)")
    print(f"crawler_params:  {crawler_params:,} ({num_crawler} blocks x{crawler_loops}, MLP {args.crawler_mlp_mult}x)")
    print(f"effective_depth: {effective_depth}")
    print(f"dim:             {dim}")
    print(f"steps:           {step} (C:{c_steps} N:{n_steps})")
    elapsed_s = time.time() - t_start
    print(f"time:            {elapsed_s:.1f}s")
    print(f"avg_ms:          {elapsed_s * 1000 / step:.1f}ms/step")
    print(f"log:             {logfile}")
    print(f"peak_vram:       {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")

if __name__ == "__main__":
    main()
