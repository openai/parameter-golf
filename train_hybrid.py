"""
train_hybrid.py — Hybrid Hypergraph + Transformer for Parameter Golf

Two-phase training:
  Phase 1 (0-2 min): Scan FineWeb shards → build hypergraph pattern store
  Phase 2 (2-10 min): Train residual transformer with hypergraph-guided loss

At evaluation:
  P(next) = λ·P_hyper + (1-λ)·P_neural
  where λ = sigmoid(log(binding_confidence) - log(threshold))

Artifact budget (16MB):
  ~5MB  → Hypergraph pattern store (zlib compressed)
  ~9MB  → Transformer weights (int8 + zlib)
  ~2MB  → Code + overhead
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import struct
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 0))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # Hypergraph-specific
    hyper_budget_bytes = int(os.environ.get("HYPER_BUDGET_BYTES", 5_000_000))
    hyper_scan_shards = int(os.environ.get("HYPER_SCAN_SHARDS", 5))
    hyper_min_count = int(os.environ.get("HYPER_MIN_COUNT", 10))
    hyper_top_k = int(os.environ.get("HYPER_TOP_K", 32))
    hyper_lambda_init = float(os.environ.get("HYPER_LAMBDA_INIT", 0.3))
    hyper_scan_time_budget = float(os.environ.get("HYPER_SCAN_TIME_BUDGET", 90.0))


# ============================================================================
# HYPERGRAPH PATTERN STORE (embedded for single-file submission)
# ============================================================================

class HypergraphStore:
    """
    Multi-level n-gram store with binding-energy-weighted selection.

    Levels:
      1: bigram   (1 context token → next distribution)
      2: trigram   (2 context tokens → next distribution)
      3: 5-gram    (4 context tokens → next distribution)

    Each pattern's binding B determines inclusion and interpolation weight.
    """

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.token_freq = np.zeros(vocab_size, dtype=np.float64)
        self.total_tokens = 0

        # Raw counters (scan phase)
        self._bi_counts: Dict[int, Counter] = defaultdict(Counter)
        self._bi_totals: Counter = Counter()
        self._tri_counts: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self._tri_totals: Counter = Counter()
        self._five_counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
        self._five_totals: Counter = Counter()

        # Built tables: context_tuple → (next_probs_dict, binding_energy)
        self.bigrams: Dict[Tuple[int], Tuple[Dict[int, float], float]] = {}
        self.trigrams: Dict[Tuple[int, int], Tuple[Dict[int, float], float]] = {}
        self.fivegrams: Dict[Tuple[int, ...], Tuple[Dict[int, float], float]] = {}
        self._built = False

    def scan(self, tokens: np.ndarray):
        """Scan a shard of uint16 tokens."""
        n = len(tokens)
        if n < 5:
            return

        # Frequencies — vectorized
        counts = np.bincount(tokens.astype(np.int32), minlength=self.vocab_size)
        self.token_freq[:min(len(counts), self.vocab_size)] += counts[:self.vocab_size]
        self.total_tokens += n

        # Bigrams — vectorized
        prev = tokens[:-1].astype(np.int32)
        nxt = tokens[1:].astype(np.int32)
        keys = prev * self.vocab_size + nxt
        for key, count in Counter(keys.tolist()).items():
            p, nx = divmod(key, self.vocab_size)
            self._bi_counts[p][nx] += count
            self._bi_totals[p] += count

        # Trigrams — vectorized
        t0 = tokens[:-2].astype(np.int64)
        t1 = tokens[1:-1].astype(np.int64)
        t2 = tokens[2:].astype(np.int64)
        vs = self.vocab_size
        tri_keys = t0 * vs * vs + t1 * vs + t2
        for key, count in Counter(tri_keys.tolist()).items():
            t2v = key % vs
            rem = key // vs
            t1v = rem % vs
            t0v = rem // vs
            ctx = (int(t0v), int(t1v))
            self._tri_counts[ctx][int(t2v)] += count
            self._tri_totals[ctx] += count

        # 5-grams — subsample for speed
        step = max(1, n // 1_000_000)
        for i in range(0, n - 4, step):
            ctx = (int(tokens[i]), int(tokens[i+1]),
                   int(tokens[i+2]), int(tokens[i+3]))
            nx = int(tokens[i+4])
            self._five_counts[ctx][nx] += step
            self._five_totals[ctx] += step

    def _specificity(self, tok: int) -> float:
        f = self.token_freq[tok]
        return 1.0 / f if f > 0 else 0.0

    def _binding_bigram(self, prev: int) -> float:
        """B for bigram: specificity × predictability × evidence."""
        sigma = self._specificity(prev)
        total = self._bi_totals[prev]
        if total == 0:
            return 0.0
        # Entropy of next-token distribution
        entropy = 0.0
        for c in self._bi_counts[prev].values():
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)
        max_ent = math.log2(self.vocab_size)
        return sigma * total * (1.0 - entropy / max_ent)

    def _binding_ngram(self, ctx: tuple) -> float:
        """B for n-gram: pairwise specificity × predictability × evidence."""
        n = len(ctx)
        # Pairwise specificity
        pair_sum = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_sum += self._specificity(ctx[i]) * self._specificity(ctx[j])
                n_pairs += 1
        avg_pair = pair_sum / max(1, n_pairs)

        # Get counts for this context
        if n == 2:
            counts = self._tri_counts.get(ctx, {})
            total = self._tri_totals.get(ctx, 0)
        elif n == 4:
            counts = self._five_counts.get(ctx, {})
            total = self._five_totals.get(ctx, 0)
        else:
            return avg_pair

        if total == 0:
            return 0.0

        entropy = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)
        max_ent = math.log2(self.vocab_size)
        certainty = 1.0 - entropy / max_ent

        return avg_pair * certainty * math.log1p(total)

    def build(self, budget_bytes: int = 5_000_000, min_count: int = 10,
              top_k: int = 32):
        """Build finalized stores, selecting by binding energy within budget."""
        # Budget split: 35% bigram, 40% trigram, 25% 5-gram
        bi_budget = int(budget_bytes * 0.35)
        tri_budget = int(budget_bytes * 0.40)
        five_budget = int(budget_bytes * 0.25)

        # --- Bigrams ---
        entries = []
        for prev, dist in self._bi_counts.items():
            total = self._bi_totals[prev]
            if total < min_count:
                continue
            b = self._binding_bigram(prev)
            if b <= 0:
                continue
            top = dist.most_common(top_k)
            probs = {tok: cnt / total for tok, cnt in top}
            entries.append(((prev,), probs, b))
        entries.sort(key=lambda e: -e[2])
        used = 0
        for ctx, probs, b in entries:
            size = 2 + len(probs) * 4 + 8
            if used + size > bi_budget:
                break
            self.bigrams[ctx] = (probs, b)
            used += size

        # --- Trigrams ---
        entries = []
        for ctx, dist in self._tri_counts.items():
            total = self._tri_totals[ctx]
            if total < min_count:
                continue
            b = self._binding_ngram(ctx)
            if b <= 0:
                continue
            top = dist.most_common(top_k)
            probs = {tok: cnt / total for tok, cnt in top}
            entries.append((ctx, probs, b))
        entries.sort(key=lambda e: -e[2])
        used = 0
        for ctx, probs, b in entries:
            size = 4 + len(probs) * 4 + 8
            if used + size > tri_budget:
                break
            self.trigrams[ctx] = (probs, b)
            used += size

        # --- 5-grams ---
        entries = []
        for ctx, dist in self._five_counts.items():
            total = self._five_totals[ctx]
            if total < min_count:
                continue
            b = self._binding_ngram(ctx)
            if b <= 0:
                continue
            top = dist.most_common(top_k)
            probs = {tok: cnt / total for tok, cnt in top}
            entries.append((ctx, probs, b))
        entries.sort(key=lambda e: -e[2])
        used = 0
        for ctx, probs, b in entries:
            size = 8 + len(probs) * 4 + 8
            if used + size > five_budget:
                break
            self.fivegrams[ctx] = (probs, b)
            used += size

        # Free raw counters
        self._bi_counts.clear()
        self._bi_totals.clear()
        self._tri_counts.clear()
        self._tri_totals.clear()
        self._five_counts.clear()
        self._five_totals.clear()

        self._built = True

    def predict_logits(self, context_ids: Tensor, vocab_size: int) -> Tuple[Tensor, Tensor]:
        """
        Given context_ids (batch, seq_len), produce hypergraph log-probs
        and confidence for the LAST token prediction of each sequence.

        Returns:
            hyper_log_probs: (batch, vocab_size) — log probabilities
            confidence: (batch,) — binding confidence per sample
        """
        batch_size = context_ids.shape[0]
        device = context_ids.device
        log_probs = torch.full((batch_size, vocab_size), -20.0,
                               device=device, dtype=torch.float32)
        confidence = torch.zeros(batch_size, device=device, dtype=torch.float32)

        ctx_np = context_ids.cpu().numpy()

        for i in range(batch_size):
            seq = ctx_np[i]
            n = len(seq)
            result = np.full(vocab_size, 1e-10, dtype=np.float64)
            total_weight = 0.0

            # Level 3: 5-gram
            if n >= 4:
                key = (int(seq[-4]), int(seq[-3]), int(seq[-2]), int(seq[-1]))
                entry = self.fivegrams.get(key)
                if entry is not None:
                    probs, b = entry
                    for tok, p in probs.items():
                        result[tok] += b * p
                    total_weight += b

            # Level 2: trigram
            if n >= 2:
                key = (int(seq[-2]), int(seq[-1]))
                entry = self.trigrams.get(key)
                if entry is not None:
                    probs, b = entry
                    for tok, p in probs.items():
                        result[tok] += b * p
                    total_weight += b

            # Level 1: bigram
            if n >= 1:
                key = (int(seq[-1]),)
                entry = self.bigrams.get(key)
                if entry is not None:
                    probs, b = entry
                    for tok, p in probs.items():
                        result[tok] += b * p
                    total_weight += b

            if total_weight > 0:
                result /= total_weight
                result = np.clip(result, 1e-10, None)
                result /= result.sum()
                log_probs[i] = torch.tensor(np.log(result), device=device,
                                            dtype=torch.float32)
                confidence[i] = total_weight

        return log_probs, confidence

    def serialize(self) -> bytes:
        """Serialize to compact binary for artifact."""
        buf = io.BytesIO()

        def write_table(table, ctx_len):
            buf.write(struct.pack('<I', len(table)))
            buf.write(struct.pack('<B', ctx_len))
            for ctx, (probs, binding) in table.items():
                for t in ctx:
                    buf.write(struct.pack('<H', t))
                buf.write(struct.pack('<f', binding))
                buf.write(struct.pack('<H', len(probs)))
                for tok, p in probs.items():
                    buf.write(struct.pack('<H', tok))
                    buf.write(struct.pack('<H', min(65535, int(p * 65535))))

        write_table(self.bigrams, 1)
        write_table(self.trigrams, 2)
        write_table(self.fivegrams, 4)

        raw = buf.getvalue()
        compressed = zlib.compress(raw, level=9)
        return struct.pack('<I', len(raw)) + compressed

    @classmethod
    def deserialize(cls, data: bytes, vocab_size: int = 1024) -> 'HypergraphStore':
        store = cls(vocab_size=vocab_size)
        raw_size = struct.unpack('<I', data[:4])[0]
        raw = zlib.decompress(data[4:])
        buf = io.BytesIO(raw)

        def read_table(target):
            n_patterns = struct.unpack('<I', buf.read(4))[0]
            ctx_len = struct.unpack('<B', buf.read(1))[0]
            for _ in range(n_patterns):
                ctx = tuple(struct.unpack('<H', buf.read(2))[0] for _ in range(ctx_len))
                binding = struct.unpack('<f', buf.read(4))[0]
                n_next = struct.unpack('<H', buf.read(2))[0]
                probs = {}
                for _ in range(n_next):
                    tok = struct.unpack('<H', buf.read(2))[0]
                    p = struct.unpack('<H', buf.read(2))[0] / 65535.0
                    probs[tok] = p
                target[ctx] = (probs, binding)

        read_table(store.bigrams)
        read_table(store.trigrams)
        read_table(store.fivegrams)
        store._built = True
        return store

    def stats(self) -> dict:
        ser = self.serialize()
        return {
            'bigrams': len(self.bigrams),
            'trigrams': len(self.trigrams),
            'fivegrams': len(self.fivegrams),
            'total_patterns': len(self.bigrams) + len(self.trigrams) + len(self.fivegrams),
            'serialized_bytes': len(ser),
            'tokens_scanned': self.total_tokens,
        }


# ============================================================================
# TRANSFORMER MODEL (same as baseline train_gpt.py)
# ============================================================================

class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias)
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype))

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.register_buffer("inv_freq", (1.0 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = 0
        self.cos_cached: Tensor | None = None
        self.sin_cached: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            self.cos_cached = freqs.cos().to(x.dtype)
            self.sin_cached = freqs.sin().to(x.dtype)
            self.seq_len_cached = seq_len
        cos, sin = self.cos_cached, self.sin_cached
        assert cos is not None and sin is not None
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, n_kv_head: int, rope_base: float,
                 qk_gain_init: float):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = dim // n_head
        total_kv = 2 * n_kv_head * self.head_dim
        self.c_attn = CastedLinear(dim, dim + total_kv)
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj._zero_init = True
        self.rotary = Rotary(self.head_dim, rope_base)
        self.q_gain = nn.Parameter(torch.full((n_head, 1, self.head_dim), qk_gain_init))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q = qkv[..., :C].reshape(B, T, self.n_head, self.head_dim)
        kv = qkv[..., C:].reshape(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv.unbind(dim=2)
        q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
        q = self.rotary(q.transpose(1, 2)).contiguous()
        k = self.rotary(k.transpose(1, 2)).contiguous()
        v = v.transpose(1, 2).contiguous()
        q = q * self.q_gain.to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, dim: int, mult: int):
        super().__init__()
        hdim = dim * mult
        self.c_fc = CastedLinear(dim, 2 * hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        a, gate = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(a * F.silu(gate))

class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, n_kv_head: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, n_head, n_kv_head, rope_base, qk_gain_init)
        self.mlp_norm = RMSNorm()
        self.mlp = MLP(dim, mlp_mult)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class HybridGPT(nn.Module):
    """
    GPT with optional hypergraph-guided prediction.

    During training: standard cross-entropy loss (transformer learns residual).
    During eval: interpolate transformer logits with hypergraph predictions.
    """

    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: int,
                 tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 hyper_store: Optional[HypergraphStore] = None,
                 hyper_lambda: float = 0.3):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size
        self.hyper_store = hyper_store
        self.hyper_lambda = hyper_lambda

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult,
                  rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def get_logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning logits (for eval with hypergraph interpolation)."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Training forward: standard cross-entropy."""
        logits = self.get_logits(input_ids)
        logits = logits.reshape(-1, logits.size(-1))
        targets = target_ids.reshape(-1)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_hybrid(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """
        Eval forward with hypergraph interpolation.

        P(next) = λ·P_hyper + (1-λ)·softmax(neural_logits)
        Loss = -log P(next)[target]
        """
        neural_logits = self.get_logits(input_ids)  # (B, T, V)
        B, T, V = neural_logits.shape

        if self.hyper_store is None or not self.hyper_store._built:
            # Fallback to pure neural
            return F.cross_entropy(
                neural_logits.reshape(-1, V).float(),
                target_ids.reshape(-1),
                reduction="mean",
            )

        # Neural probabilities
        neural_probs = F.softmax(neural_logits.float(), dim=-1)  # (B, T, V)

        # Hypergraph predictions for each position
        # For position t, context is input_ids[:, :t+1]
        hyper_probs = torch.full_like(neural_probs, 1.0 / V)
        hyper_conf = torch.zeros(B, T, device=input_ids.device)

        # Batch process: for each position, look up patterns
        input_np = input_ids.cpu().numpy()
        for t in range(T):
            # Context up to position t
            for b in range(B):
                ctx = input_np[b, max(0, t-3):t+1]
                result = np.full(V, 1e-10, dtype=np.float64)
                total_w = 0.0

                # 5-gram
                if len(ctx) >= 4:
                    key = tuple(int(x) for x in ctx[-4:])
                    entry = self.hyper_store.fivegrams.get(key)
                    if entry:
                        probs, bnd = entry
                        for tok, p in probs.items():
                            result[tok] += bnd * p
                        total_w += bnd

                # Trigram
                if len(ctx) >= 2:
                    key = (int(ctx[-2]), int(ctx[-1]))
                    entry = self.hyper_store.trigrams.get(key)
                    if entry:
                        probs, bnd = entry
                        for tok, p in probs.items():
                            result[tok] += bnd * p
                        total_w += bnd

                # Bigram
                if len(ctx) >= 1:
                    key = (int(ctx[-1]),)
                    entry = self.hyper_store.bigrams.get(key)
                    if entry:
                        probs, bnd = entry
                        for tok, p in probs.items():
                            result[tok] += bnd * p
                        total_w += bnd

                if total_w > 0:
                    result /= total_w
                    result = np.clip(result, 1e-10, None)
                    result /= result.sum()
                    hyper_probs[b, t] = torch.tensor(result, device=input_ids.device,
                                                      dtype=torch.float32)
                    hyper_conf[b, t] = total_w

        # Adaptive lambda: sigmoid(log(conf) - threshold)
        lam = torch.sigmoid(hyper_conf - 1.0) * self.hyper_lambda  # (B, T)
        lam = lam.unsqueeze(-1)  # (B, T, 1)

        # Interpolate
        combined = lam * hyper_probs + (1.0 - lam) * neural_probs
        combined = combined.clamp(min=1e-10)

        # Cross-entropy from combined distribution
        log_probs = torch.log(combined)  # (B, T, V)
        targets = target_ids.unsqueeze(-1)  # (B, T, 1)
        loss = -log_probs.gather(dim=-1, index=targets).squeeze(-1)  # (B, T)
        return loss.mean()


# ============================================================================
# DATA LOADING (from train_gpt.py)
# ============================================================================

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype=np.int32, count=256)
    if header[0] != 20240520:
        raise ValueError(f"Bad magic in {file}: {header[0]}")
    n_tokens = int(header[2])
    with open(file, "rb") as f:
        f.seek(256 * 4)
        tokens = np.frombuffer(f.read(n_tokens * 2), dtype=np.uint16)
    return torch.from_numpy(tokens.astype(np.int32))

def load_data_shard_numpy(file: Path) -> np.ndarray:
    """Load shard as numpy uint16 for hypergraph scanning."""
    header = np.fromfile(file, dtype=np.int32, count=256)
    if header[0] != 20240520:
        raise ValueError(f"Bad magic in {file}: {header[0]}")
    n_tokens = int(header[2])
    with open(file, "rb") as f:
        f.seek(256 * 4)
        tokens = np.frombuffer(f.read(n_tokens * 2), dtype=np.uint16)
    return tokens.copy()

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files found: {pattern}")
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self._file_idx = rank
        self._offset = 0
        self._tokens = load_data_shard(Path(self.files[self._file_idx]))

    def next_batch(self, total_batch_tokens: int, seq_len: int,
                   grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = total_batch_tokens // (self.world_size * grad_accum_steps)
        needed = local_tokens + 1
        while self._offset + needed > len(self._tokens):
            self._file_idx = (self._file_idx + self.world_size) % len(self.files)
            self._tokens = load_data_shard(Path(self.files[self._file_idx]))
            self._offset = 0
        chunk = self._tokens[self._offset:self._offset + needed].to(
            device=self.device, dtype=torch.int64)
        self._offset += local_tokens
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return x, y


# ============================================================================
# BPB METRIC (from train_gpt.py)
# ============================================================================

def build_sentencepiece_luts(sp, vocab_size: int, device):
    base_bytes = torch.zeros(vocab_size, dtype=torch.int32, device=device)
    has_leading_space = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        if sp.is_unknown(i) or sp.is_control(i):
            is_boundary[i] = True
            base_bytes[i] = 0
            continue
        raw = piece.replace("\u2581", " ")
        nb = len(raw.encode("utf-8"))
        if piece.startswith("\u2581"):
            has_leading_space[i] = True
            nb -= 1
        base_bytes[i] = nb
    return base_bytes, has_leading_space, is_boundary


# ============================================================================
# QUANTIZATION (from train_gpt.py)
# ============================================================================

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0

def quantize_state_dict_int8(state_dict):
    obj = {"__format__": "int8+scales", "tensors": {}}
    passthrough_dtypes = {}
    baseline_bytes = 0
    int8_bytes = 0

    for name, t in state_dict.items():
        baseline_bytes += t.numel() * t.element_size()
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            obj["tensors"][name] = {"kind": "float", "data": t.float().contiguous()}
            int8_bytes += t.numel() * 4
            continue
        if t.numel() <= 65_536:
            store_t = t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
            obj["tensors"][name] = {"kind": "float", "data": store_t}
            int8_bytes += store_t.numel() * store_t.element_size()
            continue

        t32 = t.float()
        if t32.ndim == 2:
            clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
            scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
            obj["tensors"][name] = {
                "kind": "int8", "data": q.contiguous(),
                "scale": scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous(),
            }
        else:
            clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
            scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0)
            q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale),
                           -127, 127).to(torch.int8)
            obj["tensors"][name] = {"kind": "int8", "data": q.contiguous(), "scale": scale}
        int8_bytes += q.numel() + (scale.numel() * scale.element_size() if isinstance(scale, Tensor) else 4)

    return obj, {"baseline_tensor_bytes": baseline_bytes, "int8_payload_bytes": int8_bytes}

def dequantize_state_dict_int8(obj):
    state_dict = {}
    for name, entry in obj["tensors"].items():
        if entry["kind"] == "float":
            state_dict[name] = entry["data"]
        elif entry["kind"] == "int8":
            q = entry["data"].float()
            scale = entry["scale"]
            if scale.ndim == 1 and q.ndim == 2:
                state_dict[name] = q * scale.float()[:, None]
            else:
                state_dict[name] = q * scale.float()
    return state_dict

def restore_low_dim_params_to_fp32(module):
    for p in module.parameters():
        if p.ndim < 2 or p.numel() <= 65_536:
            p.data = p.data.float()


# ============================================================================
# MUON OPTIMIZER (from train_gpt.py)
# ============================================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim >= 2:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = buf
                    g = g.view(g.size(0), -1) if g.ndim > 2 else g
                    g = zeropower_via_newtonschulz5(g.unsqueeze(0),
                                                     group["backend_steps"]).squeeze(0)
                    g = g.view_as(p.data)
                p.data.add_(g, alpha=-lr)


# ============================================================================
# EVALUATION WITH HYBRID INTERPOLATION
# ============================================================================

def eval_val_hybrid(
    args: Hyperparameters,
    model: HybridGPT,
    hyper_store: Optional[HypergraphStore],
    rank: int, world_size: int, device: torch.device,
    grad_accum_steps: int, val_tokens: Tensor,
    base_bytes_lut: Tensor, has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    use_hybrid: bool = True,
) -> tuple[float, float]:
    """Eval with optional hypergraph interpolation."""
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Get the base model from DDP wrapper if needed
    base = model.module if hasattr(model, 'module') else model

    base.eval()
    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, seq_end)
            raw_start = batch_start * args.train_seq_len
            raw_end = batch_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64,
                                                      non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                if use_hybrid and hyper_store is not None and hyper_store._built:
                    batch_loss = base.forward_hybrid(x, y)
                else:
                    batch_loss = base(x, y)

            batch_loss = batch_loss.detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    base.train()
    return float(val_loss.item()), float(bpt * tpb)


# ============================================================================
# MAIN
# ============================================================================

def main():
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # --- Distributed setup ---
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"

    def log0(msg, console=True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a") as f:
                print(msg, file=f)

    log0(f"=== HYBRID HYPERGRAPH + TRANSFORMER ===")
    log0(f"seed:{args.seed}")

    # --- Tokenizer + validation setup ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    # ================================================================
    # PHASE 1: Build hypergraph pattern store
    # ================================================================
    log0(f"\n{'='*60}")
    log0(f"PHASE 1: Building hypergraph pattern store")
    log0(f"{'='*60}")

    t_phase1 = time.perf_counter()
    hyper_store = HypergraphStore(vocab_size=args.vocab_size)

    train_shard_files = sorted(glob.glob(args.train_files))
    scan_shards = min(args.hyper_scan_shards, len(train_shard_files))

    for i in range(scan_shards):
        if time.perf_counter() - t_phase1 > args.hyper_scan_time_budget:
            log0(f"  Time budget reached after {i} shards")
            break
        tokens = load_data_shard_numpy(Path(train_shard_files[i]))
        hyper_store.scan(tokens)
        log0(f"  Scanned shard {i+1}/{scan_shards}: {len(tokens):,} tokens "
             f"({time.perf_counter() - t_phase1:.1f}s)")

    hyper_store.build(
        budget_bytes=args.hyper_budget_bytes,
        min_count=args.hyper_min_count,
        top_k=args.hyper_top_k,
    )

    stats = hyper_store.stats()
    phase1_time = time.perf_counter() - t_phase1
    log0(f"\nHypergraph store built in {phase1_time:.1f}s:")
    log0(f"  Bigrams:   {stats['bigrams']:,}")
    log0(f"  Trigrams:  {stats['trigrams']:,}")
    log0(f"  5-grams:   {stats['fivegrams']:,}")
    log0(f"  Total:     {stats['total_patterns']:,} patterns")
    log0(f"  Serialized: {stats['serialized_bytes']:,} bytes "
         f"({stats['serialized_bytes']/1e6:.2f} MB)")
    log0(f"  Tokens scanned: {stats['tokens_scanned']:,}")

    # ================================================================
    # PHASE 2: Train residual transformer
    # ================================================================
    log0(f"\n{'='*60}")
    log0(f"PHASE 2: Training residual transformer")
    log0(f"{'='*60}")

    base_model = HybridGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        hyper_store=hyper_store,
        hyper_lambda=args.hyper_lambda_init,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank],
                broadcast_buffers=False) if distributed else compiled_model

    # Optimizer setup
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named_params
                     if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"Model params: {n_params:,}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # Adjust wallclock for phase 1 time
    remaining_seconds = args.max_wallclock_seconds - phase1_time
    max_wallclock_ms = 1000.0 * remaining_seconds if remaining_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup
    if args.warmup_steps > 0:
        initial_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_opt_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len,
                                                grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(initial_state, strict=True)
        for opt, state in zip(optimizers, initial_opt_states):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # --- Main training loop ---
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)

            # Eval WITHOUT hybrid (pure neural baseline)
            val_loss, val_bpb = eval_val_hybrid(
                args, base_model, None, rank, world_size, device,
                grad_accum_steps, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                use_hybrid=False)

            # Eval WITH hybrid interpolation
            val_loss_h, val_bpb_h = eval_val_hybrid(
                args, base_model, hyper_store, rank, world_size, device,
                grad_accum_steps, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                use_hybrid=True)

            log0(f"step:{step}/{args.iterations} "
                 f"neural_bpb:{val_bpb:.4f} hybrid_bpb:{val_bpb_h:.4f} "
                 f"delta:{val_bpb - val_bpb_h:.4f} "
                 f"train_time:{training_time_ms:.0f}ms")

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for ms in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len,
                                            grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)  # Standard CE for training
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_mom

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            cap_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    # ================================================================
    # SERIALIZATION
    # ================================================================
    log0(f"\n{'='*60}")
    log0(f"SERIALIZATION")
    log0(f"{'='*60}")

    if master_process:
        # Save hypergraph store
        hyper_blob = hyper_store.serialize()
        with open("hyper_store.bin", "wb") as f:
            f.write(hyper_blob)
        hyper_bytes = len(hyper_blob)
        log0(f"Hypergraph store: {hyper_bytes:,} bytes ({hyper_bytes/1e6:.2f} MB)")

        # Save model
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        log0(f"Raw model: {model_bytes:,} bytes")

    # Quantize + compress
    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)

    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        model_compressed = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))

        total = hyper_bytes + model_compressed + code_bytes
        log0(f"\nArtifact budget:")
        log0(f"  Hypergraph:  {hyper_bytes:>10,} bytes ({hyper_bytes/1e6:.2f} MB)")
        log0(f"  Model (int8): {model_compressed:>10,} bytes ({model_compressed/1e6:.2f} MB)")
        log0(f"  Code:        {code_bytes:>10,} bytes ({code_bytes/1e6:.2f} MB)")
        log0(f"  TOTAL:       {total:>10,} bytes ({total/1e6:.2f} MB)")
        log0(f"  Under 16MB:  {'YES' if total <= 16_000_000 else 'NO !!!'}")

    # Roundtrip validation
    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_state = torch.load(io.BytesIO(zlib.decompress(f.read())), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    # Final eval: pure neural on quantized weights
    q_val_loss, q_val_bpb = eval_val_hybrid(
        args, base_model, None, rank, world_size, device,
        grad_accum_steps, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        use_hybrid=False)
    log0(f"\nFinal (neural only, int8 roundtrip): val_bpb={q_val_bpb:.4f}")

    # Final eval: hybrid on quantized weights
    q_val_loss_h, q_val_bpb_h = eval_val_hybrid(
        args, base_model, hyper_store, rank, world_size, device,
        grad_accum_steps, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        use_hybrid=True)
    log0(f"Final (hybrid, int8 roundtrip):      val_bpb={q_val_bpb_h:.4f}")
    log0(f"Hypergraph improvement:              {q_val_bpb - q_val_bpb_h:.4f} BPB")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
