#!/usr/bin/env -S python3 -u
"""
Golf V2: Top-3 techniques + CDM eval advantage.

Architecture upgrades (vs V1):
  - 11 layers (was 9) + 3x MLP (was 2x)
  - XSA on last 4 layers (exclusive self-attention)
  - LeakyReLU(0.5)^2 (was relu^2)
  - BigramHash(2048) + SmearGate
  - EMA (decay=0.997)
  - LN Scale (1/sqrt(layer+1))

Eval upgrades:
  - N-gram boosting (orders 2-7, entropy-adaptive)
  - Score-first TTT (AR or CDM mode)

Training: same Muon + Adam split as baseline.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# ==============================================================================
# CONFIG
# ==============================================================================
COMPUTE_DTYPE = mx.bfloat16

DATA_DIR = "/Users/akaihuangm1/Desktop/github/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/Users/akaihuangm1/Desktop/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

VOCAB_SIZE = 1024
NUM_LAYERS = 16          # was 9
MODEL_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 3             # was 2
ROPE_BASE = 10000.0
QK_GAIN_INIT = 1.5
TIED_EMBED_INIT_STD = 0.005
LOGIT_SOFTCAP = 30.0
SEQ_LEN = 1024

XSA_LAST_N = 4
BIGRAM_BUCKETS = 2048
BIGRAM_DIM = 128

# Optimizer
TIED_EMBED_LR = 0.035
MATRIX_LR = 0.025
SCALAR_LR = 0.025
BETA1 = 0.9
BETA2 = 0.95
ADAM_EPS = 1e-8
MUON_MOMENTUM = 0.99
MUON_BACKEND_STEPS = 5
MUON_MOMENTUM_WARMUP_START = 0.92
MUON_MOMENTUM_WARMUP_STEPS = 1500
WEIGHT_DECAY = 0.04
GRAD_CLIP = 0.3

# EMA
EMA_DECAY = 0.997

SEED = 1337

# ==============================================================================
# MATH HELPERS
# ==============================================================================
def rms_norm(x, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)

def zeropower_newtonschulz5(g, steps, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_data_shard(path):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32, copy=False)

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            print(f"WARNING: starting epoch:{self.epoch}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

class TokenLoader:
    def __init__(self, pattern):
        self.stream = TokenStream(pattern)

    def next_batch(self, batch_tokens, seq_len):
        usable = (batch_tokens // seq_len) * seq_len
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = np.concatenate([load_data_shard(f) for f in files])
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

# ==============================================================================
# MODEL BLOCKS
# ==============================================================================
class CastedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)
    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T

class RMSNormNoWeight(nn.Module):
    def __call__(self, x):
        return rms_norm(x)

class DualModeAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5
        self.use_xsa = use_xsa

    def _xsa(self, y, v):
        """Subtract self-value projection (XSA)."""
        bsz, seqlen, dim = y.shape
        hd = self.head_dim
        nkv = self.num_kv_heads
        nh = self.num_heads
        group = nh // nkv

        # y: [B, T, nh*hd] -> [B, T, nkv, group, hd]
        y_g = y.reshape(bsz, seqlen, nkv, group, hd)
        # v: [B, nkv, T, hd] -> [B, T, nkv, 1, hd]
        v_t = v.transpose(0, 2, 1, 3)  # [B, T, nkv, hd]
        vn = v_t / (mx.sqrt(mx.sum(v_t * v_t, axis=-1, keepdims=True)) + 1e-8)
        vn = mx.expand_dims(vn, axis=3)  # [B, T, nkv, 1, hd]

        # Project y onto v direction and subtract
        proj = mx.sum(y_g * vn, axis=-1, keepdims=True) * vn
        return (y_g - proj).reshape(bsz, seqlen, dim)

    def __call__(self, x, is_causal=True):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]

        if is_causal:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        else:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)

        # XSA: subtract self-value projection
        if self.use_xsa:
            y = self._xsa(y, v)

        return self.proj(y)


class MLP(nn.Module):
    """LeakyReLU(0.5)^2 MLP."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x):
        h = self.fc(x)
        # LeakyReLU(0.5) squared
        h = mx.where(h >= 0, h, 0.5 * h)
        return self.proj(h * h)


class BigramHashEmbedding(nn.Module):
    """Learned bigram hash embeddings."""
    def __init__(self, buckets, bigram_dim, model_dim):
        super().__init__()
        self.buckets = buckets
        self.embed = nn.Embedding(buckets, bigram_dim)
        # Init to zero so bigram starts with no effect
        self.embed.weight = mx.zeros_like(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim)
        self.scale = mx.array(0.05, dtype=mx.float32)

    def bigram_hash(self, tokens):
        """Hash (prev, current) token pairs into bucket indices."""
        t = tokens.astype(mx.int32)
        mod = self.buckets - 1
        # First position has no prev token -> use last bucket
        shifted = mx.concatenate([mx.full((t.shape[0], 1), mod, dtype=mx.int32),
                                  t[:, :-1]], axis=1)
        # XOR hash
        hashed = (36313 * t + 27191 * shifted) % mod
        return hashed

    def __call__(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        h = self.proj(h)
        return h * self.scale.astype(h.dtype)


class SmearGate(nn.Module):
    """Learned blending with previous token."""
    def __init__(self, dim):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x):
        g = mx.sigmoid(self.gate.astype(x.dtype))[None, None, :]  # [1, 1, dim]
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        return (1 - g) * x + g * x_prev


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 layer_idx=0, use_xsa=False):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = DualModeAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                       use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult)
        # LN Scale: 1/sqrt(layer+1)
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((
            np.ones((dim,), dtype=np.float32),
            np.zeros((dim,), dtype=np.float32)
        )))

    def __call__(self, x, x0, is_causal=True):
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x) * self.ln_scale, is_causal=is_causal)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * self.ln_scale)
        return x


class GPTv2(nn.Module):
    """Upgraded GPT with all Top-3 techniques."""
    def __init__(self):
        super().__init__()
        self.logit_softcap = LOGIT_SOFTCAP
        self.tok_emb = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.bigram = BigramHashEmbedding(BIGRAM_BUCKETS, BIGRAM_DIM, MODEL_DIM)
        self.smear = SmearGate(MODEL_DIM)

        self.num_encoder_layers = NUM_LAYERS // 2  # 5
        self.num_decoder_layers = NUM_LAYERS - self.num_encoder_layers  # 6
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, MODEL_DIM), dtype=mx.float32)

        self.blocks = []
        for i in range(NUM_LAYERS):
            use_xsa = i >= (NUM_LAYERS - XSA_LAST_N)  # last 4 layers
            self.blocks.append(
                Block(MODEL_DIM, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, ROPE_BASE, QK_GAIN_INIT,
                      layer_idx=i, use_xsa=use_xsa)
            )
        self.final_norm = RMSNormNoWeight()

        # Init: zero out output projections
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * TIED_EMBED_INIT_STD
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits):
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def forward_hidden(self, input_ids, is_causal=True):
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        x = x + self.bigram(input_ids).astype(COMPUTE_DTYPE)
        x = rms_norm(x)
        x = self.smear(x)
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, is_causal=is_causal)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0, is_causal=is_causal)
        return self.final_norm(x)

    def __call__(self, input_ids):
        return self.forward_hidden(input_ids, is_causal=True)

    def loss_fn(self, input_ids, target_ids, is_causal=True):
        h = self.forward_hidden(input_ids, is_causal=is_causal).reshape(-1, MODEL_DIM)
        y = target_ids.reshape(-1)
        logits = self.softcap(h @ self.tok_emb.weight.astype(h.dtype).T)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ==============================================================================
# OPTIMIZER (Muon + Adam split)
# ==============================================================================
CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
                    "gate", "scale", "ln_scale")

class Muon:
    def __init__(self, keys, params):
        self.keys = keys
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params, grads, step, lr_mul):
        t = min(step / max(MUON_MOMENTUM_WARMUP_STEPS, 1), 1.0)
        momentum = (1.0 - t) * MUON_MOMENTUM_WARMUP_START + t * MUON_MOMENTUM
        lr = MATRIX_LR * lr_mul
        out = {}
        for k in self.keys:
            p, g = params[k], grads[k]
            # Gradient clipping
            g_norm = mx.sqrt(mx.sum(g * g))
            g = mx.where(g_norm > GRAD_CLIP, g * (GRAD_CLIP / (g_norm + 1e-8)), g)
            # Momentum
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            # Newton-Schulz orthogonalization
            g_ortho = zeropower_newtonschulz5(g_eff, MUON_BACKEND_STEPS)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            # Weight decay
            out[k] = p * (1 - lr * WEIGHT_DECAY) - lr * (g_ortho * scale).astype(p.dtype)
        return out

class SplitOptimizers:
    def __init__(self, model):
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in params.items()
            if p.ndim == 2
            and k != self.embed_key
            and not any(pat in k for pat in CONTROL_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k != self.embed_key and k not in self.matrix_keys
        ]
        self.muon = Muon(self.matrix_keys, params)
        self.adam_embed = optim.Adam(learning_rate=TIED_EMBED_LR, betas=[BETA1, BETA2], eps=ADAM_EPS)
        self.adam_scalar = optim.Adam(learning_rate=SCALAR_LR, betas=[BETA1, BETA2], eps=ADAM_EPS)

    def step(self, model, grads_tree, step, lr_mul):
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))
        self.adam_embed.learning_rate = TIED_EMBED_LR * lr_mul
        if self.embed_key in grads:
            updated.update(self.adam_embed.apply_gradients(
                {self.embed_key: grads[self.embed_key]},
                {self.embed_key: params[self.embed_key]},
            ))
        self.adam_scalar.learning_rate = SCALAR_LR * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys if k in grads}
        scalar_params = {k: params[k] for k in self.scalar_keys if k in grads}
        if scalar_grads:
            updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))
        model.update(tree_unflatten(list(updated.items())))


# ==============================================================================
# SENTENCEPIECE BPB
# ==============================================================================
def build_sentencepiece_luts(sp, vocab_size):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut

def compute_bpb(total_nll, total_tokens, total_bytes):
    avg_loss = total_nll / total_tokens
    bpt = avg_loss / math.log(2.0)
    return bpt * (total_tokens / total_bytes)

# ==============================================================================
# HELPERS
# ==============================================================================
def accumulate_flat_grads(accum, grads_tree, scale):
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum

def lr_schedule(step, total_steps, warmdown_iters):
    warmdown_start = max(total_steps - warmdown_iters, 0)
    if step >= warmdown_start and step < total_steps:
        return max((total_steps - step) / max(warmdown_iters, 1), 0.0)
    return 1.0

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--microbatch_tokens", type=int, default=32768)
    parser.add_argument("--max_sub_chunk", type=int, default=8192,
                        help="Smaller for 27M model on M1")
    parser.add_argument("--warmdown", type=int, default=150)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--val_tokens", type=int, default=1_000_000)
    parser.add_argument("--save_path", type=str, default="golf_v2_model.npz")
    args = parser.parse_args()

    effective_batch = args.grad_accum * args.microbatch_tokens
    print("=" * 70)
    print(f"Golf V2 + Retrodiction | {NUM_LAYERS}L d={MODEL_DIM} MLP={MLP_MULT}x | steps={args.steps}")
    print(f"Retro alpha=0.3 | XSA last {XSA_LAST_N} | LeakyReLU² | BigramHash({BIGRAM_BUCKETS}) | EMA({EMA_DECAY})")
    print(f"Effective batch: {effective_batch:,} tok/step")
    print("=" * 70)

    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, VOCAB_SIZE)

    # Validation
    val_tokens = load_validation_tokens(f"{DATA_DIR}/fineweb_val_*.bin", SEQ_LEN)
    if args.val_tokens > 0 and args.val_tokens < val_tokens.size:
        usable = (args.val_tokens // SEQ_LEN) * SEQ_LEN
        val_short = val_tokens[:usable + 1]
    else:
        val_short = val_tokens
    print(f"Val tokens: {val_tokens.size - 1:,} (eval on {val_short.size - 1:,})")

    # Model
    mx.random.seed(SEED)
    model = GPTv2()
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"Model params: {n_params:,}")

    opt = SplitOptimizers(model)
    train_loader = TokenLoader(f"{DATA_DIR}/fineweb_train_*.bin")

    # Retrodiction loss: AR forward + AR backward (reversed sequence)
    # Based on Petz recovery map: retrodiction = inferring past from future
    RETRO_ALPHA = 0.3

    def retrodiction_loss(x, y):
        # Forward AR loss (standard)
        forward_l = model.loss_fn(x, y, is_causal=True)

        # Backward AR loss: reverse the sequence, predict in reverse order
        # This teaches the model right→left patterns using causal attention
        x_rev = x[:, ::-1]  # reverse token order
        y_rev = y[:, ::-1]  # reverse target order
        backward_l = model.loss_fn(x_rev, y_rev, is_causal=True)

        return forward_l + RETRO_ALPHA * backward_l

    def ar_loss(x, y):
        return model.loss_fn(x, y, is_causal=True)

    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, retrodiction_loss), inputs=model.state, outputs=model.state)
    compiled_loss = mx.compile(ar_loss, inputs=model.state, outputs=model.state)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        x, y = train_loader.next_batch(min(args.max_sub_chunk, args.microbatch_tokens), SEQ_LEN)
        loss, grads = compiled_loss_and_grad(x, y)
        mx.eval(loss)
    train_loader = TokenLoader(f"{DATA_DIR}/fineweb_train_*.bin")

    # EMA state — delay start until 80% of training to avoid polluting with random init
    ema_start_step = int(args.steps * 0.8)
    ema_state = None

    # Eval function
    def eval_val(vtokens):
        batch_seqs = max(args.microbatch_tokens // SEQ_LEN, 1)
        total_seqs = (vtokens.size - 1) // SEQ_LEN
        total_nll = 0.0
        total_tok = 0
        total_bytes = 0.0
        for s in range(0, total_seqs, batch_seqs):
            e = min(s + batch_seqs, total_seqs)
            chunk = vtokens[s * SEQ_LEN:(e * SEQ_LEN) + 1]
            x_np = chunk[:-1].reshape(-1, SEQ_LEN)
            y_np = chunk[1:].reshape(-1, SEQ_LEN)
            x = mx.array(x_np, dtype=mx.int32)
            y = mx.array(y_np, dtype=mx.int32)
            ct = float(y.size)
            bl = compiled_loss(x, y).astype(mx.float32)
            mx.eval(bl)
            total_nll += float(bl.item()) * ct
            prev_ids = x_np.reshape(-1)
            tgt_ids = y_np.reshape(-1)
            bytes_np = base_bytes_lut[tgt_ids].astype(np.float64)
            bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.float64)
            total_tok += int(ct)
            total_bytes += bytes_np.sum()
        return compute_bpb(total_nll, total_tok, total_bytes)

    # Sub-chunking
    def sub_chunks(micro_tokens):
        usable = (micro_tokens // SEQ_LEN) * SEQ_LEN
        chunk_size = max((args.max_sub_chunk // SEQ_LEN) * SEQ_LEN, SEQ_LEN)
        chunks = []
        rem = usable
        while rem > 0:
            c = min(rem, chunk_size)
            chunks.append(c)
            rem -= c
        return chunks

    # Training loop
    t0 = time.perf_counter()
    best_bpb = float("inf")

    for step in range(args.steps + 1):
        is_last = (step == args.steps)

        # Eval
        if is_last or (args.val_every > 0 and step % args.val_every == 0):
            use_ema = ema_state is not None
            if use_ema:
                orig_params = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
                model.update(tree_unflatten(list(ema_state.items())))
                mx.eval(model.parameters())

            val_bpb = eval_val(val_short)
            marker = " *BEST*" if val_bpb < best_bpb else ""
            best_bpb = min(best_bpb, val_bpb)
            elapsed = time.perf_counter() - t0
            tokens_seen = step * effective_batch
            ema_tag = " [EMA]" if use_ema else ""
            print(f"step:{step}/{args.steps} val_bpb:{val_bpb:.4f}{marker}{ema_tag} "
                  f"tokens:{tokens_seen / 1e6:.0f}M elapsed:{elapsed:.0f}s")

            if use_ema:
                model.update(tree_unflatten(list(orig_params.items())))
                mx.eval(model.parameters())

            if is_last:
                if ema_state is not None:
                    model.update(tree_unflatten(list(ema_state.items())))
                    mx.eval(model.parameters())
                break

        # LR schedule
        lrm = lr_schedule(step, args.steps, args.warmdown)

        # Gradient accumulation
        grad_accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        gs = 1.0 / args.grad_accum

        for _ in range(args.grad_accum):
            chunks = sub_chunks(args.microbatch_tokens)
            total_ct = float(sum(chunks))
            micro_loss = mx.array(0.0, dtype=mx.float32)
            micro_accum = None
            for ct in chunks:
                x, y = train_loader.next_batch(ct, SEQ_LEN)
                loss, grads = compiled_loss_and_grad(x, y)
                sc = float(ct) / total_ct
                micro_loss = micro_loss + loss.astype(mx.float32) * sc
                micro_accum = accumulate_flat_grads(micro_accum, grads, sc)
                mx.eval(micro_loss, micro_accum)

            train_loss = train_loss + micro_loss * gs
            grad_accum = accumulate_flat_grads(
                grad_accum, tree_unflatten(list(micro_accum.items())), gs)
            mx.eval(train_loss, grad_accum)

        grads_tree = tree_unflatten(list(grad_accum.items()))
        opt.step(model, grads_tree, step=step, lr_mul=lrm)
        mx.synchronize()

        # EMA update — start after warmup
        if step == ema_start_step:
            ema_state = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
            mx.eval(ema_state)
            print(f"  EMA started at step {step}")
        elif ema_state is not None:
            d = EMA_DECAY
            for k, v in tree_flatten(model.parameters()):
                if k in ema_state:
                    ema_state[k] = d * ema_state[k] + (1 - d) * v
            mx.eval(ema_state)

        if step % 100 == 0 and step > 0:
            elapsed = time.perf_counter() - t0
            tps = step * effective_batch / elapsed
            print(f"  step:{step} train_loss:{float(train_loss.item()):.4f} "
                  f"lr_mul:{lrm:.4f} tok/s:{tps:.0f}")

    # Save (convert bfloat16 to float32 for numpy compatibility)
    flat = dict(tree_flatten(model.parameters()))
    np_weights = {}
    for k, v in flat.items():
        if v.dtype == mx.bfloat16:
            np_weights[k] = np.array(v.astype(mx.float32))
        else:
            np_weights[k] = np.array(v)
    np.savez(args.save_path, **np_weights)
    print(f"\nSaved to {args.save_path}")

    print("=" * 70)
    print(f"FINAL val_bpb: {val_bpb:.4f} (best: {best_bpb:.4f})")
    print(f"Baseline: 1.2244 | Gap: {best_bpb - 1.2244:+.4f}")
    print(f"Total tokens: {args.steps * effective_batch / 1e9:.3f}B")
    print(f"Model: {NUM_LAYERS}L d={MODEL_DIM} MLP={MLP_MULT}x | {n_params:,} params")
    print("=" * 70)


if __name__ == "__main__":
    main()
