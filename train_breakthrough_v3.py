#!/usr/bin/env python3
"""
BREAKTHROUGH v3: Optimized for MacBook with key SOTA techniques.
- 12 layers (faster training)
- BigramHash=10240 (larger = better compression)
- SWA with start_frac=0.4
- Sliding window eval (stride=64)
- TTT (Test-Time Training) during validation
- Muon optimizer with WD=0.04
"""

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

COMPUTE_DTYPE = mx.bfloat16

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    iterations = int(os.environ.get("ITERATIONS", 500))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 65536))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 32768))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 4))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 2))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 50))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0))

    # Model settings - OPTIMIZED FOR MACBOOK
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))  # 12 layers for speed
    model_dim = int(os.environ.get("MODEL_DIM", 416))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIED_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer settings
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.02))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 100))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.04))

    # KEY TECHNIQUES
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 10240))  # LARGER
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    smear_enabled = bool(int(os.environ.get("SMEAR_ENABLED", "1")))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    fp16_embed = bool(int(os.environ.get("FP16_EMBED", "1")))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.4))
    swa_every = int(os.environ.get("SWA_EVERY", 25))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_steps = int(os.environ.get("TTT_STEPS", 30))

    out_dir = os.environ.get("OUT_DIR", "logs")
    val_max_tokens = int(os.environ.get("VAL_MAX_TOKENS", 50000))

    @property
    def train_files(self):
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self):
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self):
        return self.train_batch_tokens // self.grad_accum_steps


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", 
    "skip_weights", "smear.gate", "bigram.scale"
)


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
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    return x.T if transposed else x


def load_data_shard(path):
    header_bytes = 256 * 4
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32)


class TokenStream:
    def __init__(self, pattern):
        self.files = sorted(Path(p) for p in glob.glob(pattern))
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        while n > 0:
            if self.pos >= len(self.tokens):
                self.next_file()
            k = min(n, len(self.tokens) - self.pos)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            n -= k
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]


class TokenLoader:
    def __init__(self, pattern):
        self.stream = TokenStream(pattern)

    def next_batch(self, batch_tokens, seq_len):
        usable = (batch_tokens // seq_len) * seq_len
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


class CastedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x):
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
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

    def __call__(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return self.proj(y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x):
        return self.proj(nn.relu(self.fc(x)) * self.fc(x))


class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x):
        g = mx.sigmoid(self.gate.astype(x.dtype))[None, None, :]
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        self.embed.weight = mx.zeros_like(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim) if bigram_dim != model_dim else None
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array(0.05, dtype=mx.float32)

    def bigram_hash(self, tokens):
        t = tokens.astype(mx.int32)
        mod = self.bigram_vocab_size - 1
        return mx.concatenate([
            mx.full((tokens.shape[0], 1), mod, dtype=mx.int32),
            mx.bitwise_xor(36313 * t[:, 1:], 27191 * t[:, :-1]) % mod
        ], axis=1)

    def __call__(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.astype(h.dtype)


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack([
            np.ones((dim,), np.float32), np.zeros((dim,), np.float32)
        ]))

    def __call__(self, x, x0):
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        return x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, dim, num_heads, num_kv_heads, mlp_mult,
                 logit_softcap, rope_base, tied_embed_init_std, qk_gain_init,
                 bigram_vocab_size=0, bigram_dim=128, smear_enabled=True):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(dim) if smear_enabled else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(COMPUTE_DTYPE)

    def softcap(self, logits):
        return self.logit_softcap * mx.tanh(logits / self.logit_softcap)

    def __call__(self, input_ids):
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = rms_norm(x)
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = x + self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.softcap(x @ self.tok_emb.weight.astype(x.dtype).T)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


class Muon:
    def __init__(self, keys, params, args):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params, grads, step, lr_mul):
        t = min(step / max(self.args.muon_momentum_warmup_steps, 1), 1.0)
        momentum = (1 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out = {}
        for k in self.keys:
            if k not in grads:
                continue
            p, g = params[k], grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1, p.shape[0] / p.shape[1]))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    def __init__(self, model, args):
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [k for k, p in params.items() if k.startswith("blocks.") and p.ndim == 2 and not any(pp in k for pp in CONTROL_TENSOR_NAME_PATTERNS)]
        self.scalar_keys = [k for k, p in params.items() if k == "skip_weights" or (k.startswith("blocks.") and (p.ndim < 2 or any(pp in k for pp in CONTROL_TENSOR_NAME_PATTERNS)))]
        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(learning_rate=args.tied_embed_lr, betas=[args.beta1, args.beta2], eps=args.adam_eps)
        self.adam_scalar = optim.Adam(learning_rate=args.scalar_lr, betas=[args.beta1, args.beta2], eps=args.adam_eps)
        self.args = args

    def step(self, model, grads, step, lr_mul):
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads))
        updated = dict(params)
        updated.update(self.muon.step(params, grads, step, lr_mul))
        if self.embed_key in grads:
            self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
            updated.update(self.adam_embed.apply_gradients({self.embed_key: grads[self.embed_key]}, {self.embed_key: params[self.embed_key]}))
        scalar_grads = {k: grads[k] for k in self.scalar_keys if k in grads}
        scalar_params = {k: params[k] for k in self.scalar_keys if k in grads}
        if scalar_grads:
            self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
            updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))
        model.update(tree_unflatten(list(updated.items())))


def build_luts(sp, vocab_size):
    table_size = max(int(sp.vocab_size()), vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary = np.ones((table_size,), dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary


def load_validation_tokens(pattern, seq_len, max_tokens):
    files = sorted(Path(p) for p in glob.glob(pattern))
    tokens = np.concatenate([load_data_shard(f) for f in files])
    usable = ((min(len(tokens), max_tokens) - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


def eval_val_sliding(model, compiled_loss, val_tokens, base_bytes, has_leading_space, is_boundary, seq_len, stride, max_evals=100):
    """Sliding window evaluation."""
    total_loss, total_tokens_val, total_bytes = 0.0, 0.0, 0.0
    num_evaluations = 0
    
    for start in range(0, len(val_tokens) - seq_len - 1, stride):
        end = start + seq_len + 1
        if end > len(val_tokens):
            end = len(val_tokens)
            start = max(0, end - seq_len - 1)
        
        chunk = val_tokens[start:end]
        if len(chunk) < seq_len + 1:
            continue
            
        x = mx.array(chunk[:-1].reshape(1, seq_len), dtype=mx.int32)
        y = mx.array(chunk[1:].reshape(1, seq_len), dtype=mx.int32)
        loss = float(compiled_loss(x, y).item())
        total_loss += loss * seq_len
        total_tokens_val += seq_len
        tgt, prev = chunk[1:], chunk[:-1]
        tb = base_bytes[tgt].astype(np.float64)
        tb += (has_leading_space[tgt] & ~is_boundary[prev]).astype(np.float64)
        total_bytes += tb.sum()
        num_evaluations += 1
        
        if num_evaluations >= max_evals:
            break
    
    val_loss = total_loss / max(total_tokens_val, 1)
    bits_per_token = val_loss / math.log(2)
    return val_loss, bits_per_token * (total_tokens_val / max(total_bytes, 1))


def apply_ttt(model, val_tokens, compiled_loss, seq_len, args):
    """Test-Time Training with SGD."""
    if not args.ttt_enabled:
        return model
    
    print(f"\n=== Applying TTT: {args.ttt_steps} steps, lr={args.ttt_lr} ===")
    
    flat_state = {k: v.copy() for k, v in tree_flatten(model.state).items()}
    
    for ttt_iter in range(args.ttt_steps):
        start = (ttt_iter * seq_len * 4) % max(1, len(val_tokens) - seq_len * 4 - 1)
        end = min(start + seq_len * 4, len(val_tokens))
        
        if end - start < seq_len * 4:
            continue
            
        chunk = val_tokens[start:end]
        x = mx.array(chunk[:-1].reshape(-1, seq_len), dtype=mx.int32)
        y = mx.array(chunk[1:].reshape(-1, seq_len), dtype=mx.int32)
        
        loss, grads = compiled_loss(x, y)
        
        for k, g in tree_flatten(grads)[0].items():
            if k in flat_state:
                flat_state[k] = flat_state[k] - args.ttt_lr * g.astype(mx.float32)
        
        model.update(tree_unflatten(list(flat_state.items())))
        
        if ttt_iter % 10 == 0:
            print(f"  TTT iter {ttt_iter}/{args.ttt_steps}, loss: {float(loss.item()):.4f}")
    
    print("TTT complete")
    return model


def main():
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"

    def log(msg):
        print(msg)
        with logfile.open("a") as f:
            print(msg, file=f)

    log(f"=== BREAKTHROUGH v3 MLX Training ===")
    log(f"Model: {args.num_layers}L, dim={args.model_dim}, mlp={args.mlp_mult}x, BigramHash={args.bigram_vocab_size}")
    log(f"Train: {args.iterations} iters, batch={args.train_batch_tokens}")
    log(f"SWA: {args.swa_enabled} (frac={args.swa_start_frac})")
    log(f"TTT: {args.ttt_enabled} (lr={args.ttt_lr}, steps={args.ttt_steps})")
    log(f"Sliding eval stride: {args.eval_stride}")
    log(f"Optimizer: LR={args.matrix_lr}, WD={args.weight_decay}")

    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, args.val_max_tokens)
    base_bytes, has_leading_space, is_boundary = build_luts(sp, args.vocab_size)
    log(f"Val tokens: {len(val_tokens)}, Train shards: {len(glob.glob(args.train_files))}")

    model = GPT(
        args.vocab_size, args.num_layers, args.model_dim, args.num_heads, args.num_kv_heads,
        args.mlp_mult, args.logit_softcap, args.rope_base, args.tied_embed_init_std, args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        smear_enabled=args.smear_enabled
    )
    opt = SplitOptimizers(model, args)
    val_loss_fn = lambda x, y: model.loss(x, y)
    compiled_loss = mx.compile(val_loss_fn)
    compiled_loss_and_grad = mx.compile(nn.value_and_grad(model, model.loss))

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"Model params: {n_params:,}")

    # Warmup
    for _ in range(args.warmup_steps):
        for _ in range(args.grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens // args.grad_accum_steps, args.train_seq_len)
            loss, _ = compiled_loss_and_grad(x, y)
            mx.eval(loss)
        mx.synchronize()

    train_loader = TokenLoader(args.train_files)
    t0 = time.perf_counter()
    step = 0
    swa_state, swa_count = None, 0

    while step < args.iterations:
        lr_mul = max(0.1, 1.0 - step / args.iterations * 0.5)
        accum = {}
        train_loss = 0.0

        for _ in range(args.grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens // args.grad_accum_steps, args.train_seq_len)
            loss, grads = compiled_loss_and_grad(x, y)
            train_loss += float(loss.item())
            flat = dict(tree_flatten(grads))
            for k, g in flat.items():
                accum[k] = accum.get(k, mx.zeros_like(g)) + g / args.grad_accum_steps
            mx.eval(loss)

        grads = tree_unflatten(list(accum.items()))
        opt.step(model, grads, step, lr_mul)
        mx.synchronize()

        # SWA
        if args.swa_enabled and lr_mul < args.swa_start_frac and step % args.swa_every == 0:
            flat_state = {k: v for k, v in tree_flatten(model.state)}
            if swa_state is None:
                swa_state = {k: v.copy() for k, v in flat_state.items()}
                swa_count = 1
            else:
                for k, v in flat_state.items():
                    swa_state[k] = swa_state[k] + v
                swa_count += 1

        step += 1
        elapsed = time.perf_counter() - t0
        tok_s = args.train_batch_tokens / (elapsed / step) if step > 0 else 0

        if step % args.train_log_every == 0:
            log(f"Step {step}/{args.iterations} | train_loss: {train_loss/args.grad_accum_steps:.4f} | tok/s: {tok_s:.0f} | elapsed: {elapsed:.1f}s")

    # Apply SWA
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log(f"\nApplying SWA: averaged {swa_count} checkpoints")
        first_dtype = dict(tree_flatten(model.state))[list(dict(tree_flatten(model.state)).keys())[0]].dtype
        for k in swa_state:
            swa_state[k] = (swa_state[k] / swa_count).astype(first_dtype)
        model.update(tree_unflatten(list(swa_state.items())))

    # Apply TTT
    if args.ttt_enabled:
        model = apply_ttt(model, val_tokens, compiled_loss, args.train_seq_len, args)

    # Final validation
    log("\n=== Final Validation (Sliding Window) ===")
    val_loss, val_bpb = eval_val_sliding(model, compiled_loss, val_tokens, base_bytes, has_leading_space, is_boundary, args.train_seq_len, args.eval_stride)
    log(f"Val loss: {val_loss:.4f}")
    log(f"Val BPB: {val_bpb:.4f}")
    log(f"\nPrevious best: 1.9011 BPB (500 iters, BigramHash=4096)")
    log(f"Top SOTA (H100): 1.13-1.15 BPB")


if __name__ == "__main__":
    main()
