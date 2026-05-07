#!/usr/bin/env -S python3 -u
"""
Test-Time Training (TTT) evaluation script.

Loads a saved .npz model (train_depth_recurrent.py or train_depth_768.py architecture)
and compares two evaluation modes:

  1. Standard BPB: forward pass only, no weight updates.
  2. TTT BPB: process val tokens sequentially; after each batch, do N gradient steps
     on that same batch to adapt the model, then eval on the next batch.

TTT Rationale:
  Each validation batch is treated as a small "training signal". The model adapts
  to the local distribution seen so far. This is a form of online learning at eval time.
  Useful because FineWeb has document-level coherence: adapting to early tokens in a
  document may improve prediction of later tokens.

Usage:
  # dim=512 model (default)
  python eval_ttt.py --model_path golf_v2_model.npz

  # dim=768 model
  python eval_ttt.py --model_path golf_768_model.npz --model_dim 768

  # Tune TTT hyperparams
  python eval_ttt.py --model_path golf_v2_model.npz --ttt_lr 1e-4 --ttt_steps 1
  python eval_ttt.py --model_path golf_v2_model.npz --ttt_lr 3e-4 --ttt_steps 2
  python eval_ttt.py --model_path golf_v2_model.npz --ttt_lr 1e-3 --ttt_steps 3
"""
from __future__ import annotations

import argparse
import copy
import glob
import math
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

# MLX imports — guarded so this Apple-Silicon-only pre-flight script can be
# IMPORTED on Linux CPU smoke-test environments (Python 3.10 + torch CPU,
# where `pip install mlx` is not possible). The stub class below lets module-
# level definitions like `class Foo(nn.Module):` and `COMPUTE_DTYPE = mx.bfloat16`
# parse without crashing; any actual MLX call only happens inside main() and
# fails clearly when the script is run as __main__ on a non-Apple machine.
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False
    class _MlxStub:
        """Permissive stub: any attribute access or call returns another stub.
        Lets `class Foo(nn.Module):` and `COMPUTE_DTYPE = mx.bfloat16` parse on
        a non-Apple machine without raising at module load time."""
        Module = type("Module", (object,), {})
        def __getattr__(self, name): return _MlxStub()
        def __call__(self, *a, **k): return _MlxStub()
    mx = _MlxStub()
    nn = _MlxStub()
    optim = _MlxStub()
    def tree_flatten(*a, **k):
        raise RuntimeError("MLX not available; this script is Apple Silicon only.")
    def tree_unflatten(*a, **k):
        raise RuntimeError("MLX not available; this script is Apple Silicon only.")

# ==============================================================================
# DEFAULTS (overridden by --model_dim)
# ==============================================================================
DATA_DIR = "/Users/akaihuangm1/Desktop/github/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/Users/akaihuangm1/Desktop/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

VOCAB_SIZE = 1024
NUM_LAYERS = 17
NUM_UNIQUE_LAYERS = 6
LAYER_MAP = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
ROPE_BASE = 10000.0
QK_GAIN_INIT = 1.5
LOGIT_SOFTCAP = 30.0
XSA_LAST_N = 4
BIGRAM_BUCKETS = 2048
COMPUTE_DTYPE = mx.bfloat16

# Per-dim config
DIM_CONFIGS = {
    512: dict(model_dim=512, num_heads=8, num_kv_heads=4, mlp_mult=3, bigram_dim=128),
    768: dict(model_dim=768, num_heads=8, num_kv_heads=4, mlp_mult=3, bigram_dim=192),
}

# ==============================================================================
# MATH HELPERS
# ==============================================================================
def rms_norm(x, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)

# ==============================================================================
# MODEL (same architecture as train_depth_recurrent.py / train_depth_768.py)
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

    def _xsa(self, y, v):
        bsz, seqlen, dim = y.shape
        hd = self.head_dim
        nkv = self.num_kv_heads
        nh = self.num_heads
        group = nh // nkv
        y_g = y.reshape(bsz, seqlen, nkv, group, hd)
        v_t = v.transpose(0, 2, 1, 3)
        vn = v_t / (mx.sqrt(mx.sum(v_t * v_t, axis=-1, keepdims=True)) + 1e-8)
        vn = mx.expand_dims(vn, axis=3)
        proj = mx.sum(y_g * vn, axis=-1, keepdims=True) * vn
        return (y_g - proj).reshape(bsz, seqlen, dim)

    def __call__(self, x, is_causal=True, use_xsa=False):
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
        if use_xsa:
            y = self._xsa(y, v)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x):
        h = self.fc(x)
        h = mx.where(h >= 0, h, 0.5 * h)
        return self.proj(h * h)

class BigramHashEmbedding(nn.Module):
    def __init__(self, buckets, bigram_dim, model_dim):
        super().__init__()
        self.buckets = buckets
        self.embed = nn.Embedding(buckets, bigram_dim)
        self.proj = CastedLinear(bigram_dim, model_dim)
        self.scale = mx.array(0.05, dtype=mx.float32)

    def bigram_hash(self, tokens):
        t = tokens.astype(mx.int32)
        mod = self.buckets - 1
        shifted = mx.concatenate([mx.full((t.shape[0], 1), mod, dtype=mx.int32), t[:, :-1]], axis=1)
        hashed = (36313 * t + 27191 * shifted) % mod
        return hashed

    def __call__(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        h = self.proj(h)
        return h * self.scale.astype(h.dtype)

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x):
        g = mx.sigmoid(self.gate.astype(x.dtype))[None, None, :]
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        return (1 - g) * x + g * x_prev

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = DualModeAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((
            np.ones((dim,), dtype=np.float32),
            np.zeros((dim,), dtype=np.float32)
        )))

    def __call__(self, x, x0, exec_pos=0, is_causal=True):
        ln_scale = 1.0 / math.sqrt(exec_pos + 1)
        use_xsa = exec_pos >= (NUM_LAYERS - XSA_LAST_N)
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x) * ln_scale, is_causal=is_causal, use_xsa=use_xsa)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * ln_scale)
        return x

class GPTv2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg["model_dim"]
        self.logit_softcap = LOGIT_SOFTCAP
        self.tok_emb = nn.Embedding(VOCAB_SIZE, dim)
        self.bigram = BigramHashEmbedding(BIGRAM_BUCKETS, cfg["bigram_dim"], dim)
        self.smear = SmearGate(dim)
        self.num_encoder_layers = NUM_LAYERS // 2
        self.num_decoder_layers = NUM_LAYERS - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, cfg["num_heads"], cfg["num_kv_heads"], cfg["mlp_mult"], ROPE_BASE, QK_GAIN_INIT)
            for _ in range(NUM_UNIQUE_LAYERS)
        ]
        self.depth_bias = mx.zeros((NUM_LAYERS, dim), dtype=mx.float32)
        self.final_norm = RMSNormNoWeight()

    def softcap(self, logits):
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def forward_hidden(self, input_ids, is_causal=True):
        dim = self.cfg["model_dim"]
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        x = x + self.bigram(input_ids).astype(COMPUTE_DTYPE)
        x = rms_norm(x)
        x = self.smear(x)
        x0 = x
        skips = []
        dec_idx = 0
        for exec_pos in range(NUM_LAYERS):
            unique_idx = LAYER_MAP[exec_pos]
            x = x + self.depth_bias[exec_pos].astype(x.dtype)[None, None, :]
            if exec_pos < self.num_encoder_layers:
                x = self.blocks[unique_idx](x, x0, exec_pos=exec_pos, is_causal=is_causal)
                skips.append(x)
            else:
                if skips:
                    x = x + self.skip_weights[dec_idx % self.num_skip_weights].astype(x.dtype)[None, None, :] * skips.pop()
                    dec_idx += 1
                x = self.blocks[unique_idx](x, x0, exec_pos=exec_pos, is_causal=is_causal)
        return self.final_norm(x)

    def __call__(self, input_ids):
        return self.forward_hidden(input_ids, is_causal=True)

    def loss_fn(self, input_ids, target_ids):
        dim = self.cfg["model_dim"]
        h = self.forward_hidden(input_ids, is_causal=True).reshape(-1, dim)
        y = target_ids.reshape(-1)
        logits = self.softcap(h @ self.tok_emb.weight.astype(h.dtype).T)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_data_shard(path):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32, copy=False)

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No val files found: {pattern}")
    tokens = np.concatenate([load_data_shard(f) for f in files])
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

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
# LOAD MODEL FROM NPZ
# ==============================================================================
def load_model_from_npz(path, cfg):
    """Load saved .npz weights into a GPTv2 model with the given config."""
    model = GPTv2(cfg)
    # Initialize with dummy data to create parameter shapes
    dummy = mx.zeros((1, 4), dtype=mx.int32)
    _ = model(dummy)
    mx.eval(model.parameters())

    # Load weights
    data = np.load(path)
    flat_params = dict(tree_flatten(model.parameters()))
    new_params = {}
    for k in flat_params:
        if k in data:
            arr = data[k]
            # Match dtype of existing param
            target_dtype = flat_params[k].dtype
            new_params[k] = mx.array(arr).astype(target_dtype)
        else:
            print(f"WARNING: key '{k}' not found in checkpoint, keeping init value")
            new_params[k] = flat_params[k]

    model.update(tree_unflatten(list(new_params.items())))
    mx.eval(model.parameters())
    return model

# ==============================================================================
# EVAL HELPERS
# ==============================================================================
def eval_batch_loss(model, x_np, y_np, seq_len):
    """Compute NLL and token/byte counts for a batch."""
    x = mx.array(x_np, dtype=mx.int32)
    y = mx.array(y_np, dtype=mx.int32)
    loss = model.loss_fn(x, y)
    mx.eval(loss)
    return float(loss.item()) * float(y.size)

def copy_params(model):
    """Deep copy model parameters as a flat dict of numpy arrays."""
    return {k: np.array(v) for k, v in tree_flatten(model.parameters())}

def restore_params(model, saved_params):
    """Restore model parameters from a flat dict of numpy arrays."""
    flat_current = dict(tree_flatten(model.parameters()))
    restored = {}
    for k, v_np in saved_params.items():
        restored[k] = mx.array(v_np).astype(flat_current[k].dtype)
    model.update(tree_unflatten(list(restored.items())))
    mx.eval(model.parameters())

# ==============================================================================
# TTT UPDATE
# ==============================================================================
def ttt_update(model, x_np, y_np, ttt_lr, ttt_steps):
    """
    Perform TTT gradient steps on the given batch.
    Uses simple AdamW (no Muon needed for TTT — just need a few fast steps).
    """
    x = mx.array(x_np, dtype=mx.int32)
    y = mx.array(y_np, dtype=mx.int32)

    # Use Adam with a modest weight decay
    ttt_optimizer = optim.AdamW(learning_rate=ttt_lr, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.01)

    loss_and_grad_fn = nn.value_and_grad(model, model.loss_fn)

    for _ in range(ttt_steps):
        loss, grads = loss_and_grad_fn(x, y)
        ttt_optimizer.update(model, grads)
        mx.eval(model.parameters(), loss)

# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Test-Time Training BPB Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .npz model checkpoint")
    parser.add_argument("--model_dim", type=int, default=512, choices=[512, 768],
                        help="Model dimension (512 for train_depth_recurrent, 768 for train_depth_768)")
    parser.add_argument("--seq_len", type=int, default=1024,
                        help="Sequence length")
    parser.add_argument("--val_tokens", type=int, default=500_000,
                        help="Number of val tokens to evaluate (0 = all)")
    parser.add_argument("--batch_seqs", type=int, default=8,
                        help="Number of sequences per eval batch (also the TTT adapt batch)")
    parser.add_argument("--ttt_lr", type=float, default=3e-4,
                        help="TTT learning rate (default: 3e-4)")
    parser.add_argument("--ttt_steps", type=int, default=1,
                        help="Number of TTT gradient steps per batch (default: 1)")
    parser.add_argument("--ttt_batch_seqs", type=int, default=-1,
                        help="Sequences per TTT update (-1 = same as batch_seqs)")
    parser.add_argument("--skip_standard", action="store_true",
                        help="Skip standard eval (only run TTT)")
    parser.add_argument("--skip_ttt", action="store_true",
                        help="Skip TTT eval (only run standard)")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    args = parser.parse_args()

    seq_len = args.seq_len
    ttt_batch_seqs = args.ttt_batch_seqs if args.ttt_batch_seqs > 0 else args.batch_seqs

    cfg = DIM_CONFIGS[args.model_dim]
    print("=" * 70)
    print(f"TTT Eval | model_dim={args.model_dim} | {args.model_path}")
    print(f"TTT: lr={args.ttt_lr} steps={args.ttt_steps} batch_seqs={ttt_batch_seqs}")
    print("=" * 70)

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, VOCAB_SIZE)

    # Load val tokens
    val_tokens = load_validation_tokens(f"{args.data_dir}/fineweb_val_*.bin", seq_len)
    if args.val_tokens > 0 and args.val_tokens < val_tokens.size:
        usable = (args.val_tokens // seq_len) * seq_len
        val_tokens = val_tokens[:usable + 1]
    total_seqs = (val_tokens.size - 1) // seq_len
    print(f"Val tokens: {val_tokens.size - 1:,} | total_seqs: {total_seqs:,}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model_from_npz(args.model_path, cfg)
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    # =========================================================================
    # STANDARD EVAL (no TTT)
    # =========================================================================
    standard_bpb = None
    if not args.skip_standard:
        print("\n--- Standard Eval (no TTT) ---")
        t0 = time.perf_counter()
        total_nll = 0.0
        total_tok = 0
        total_bytes = 0.0

        compiled_loss = mx.compile(model.loss_fn, inputs=model.state, outputs=model.state)

        for s in range(0, total_seqs, args.batch_seqs):
            e = min(s + args.batch_seqs, total_seqs)
            chunk = val_tokens[s * seq_len:(e * seq_len) + 1]
            x_np = chunk[:-1].reshape(-1, seq_len)
            y_np = chunk[1:].reshape(-1, seq_len)
            x = mx.array(x_np, dtype=mx.int32)
            y = mx.array(y_np, dtype=mx.int32)
            ct = float(y.size)
            bl = compiled_loss(x, y)
            mx.eval(bl)
            total_nll += float(bl.item()) * ct
            prev_ids = x_np.reshape(-1)
            tgt_ids = y_np.reshape(-1)
            bytes_np = base_bytes_lut[tgt_ids].astype(np.float64)
            bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.float64)
            total_tok += int(ct)
            total_bytes += bytes_np.sum()

            if (s // args.batch_seqs) % 50 == 0:
                interim_bpb = compute_bpb(total_nll, total_tok, total_bytes)
                print(f"  seqs {e}/{total_seqs} interim_bpb={interim_bpb:.4f}")

        standard_bpb = compute_bpb(total_nll, total_tok, total_bytes)
        elapsed = time.perf_counter() - t0
        print(f"Standard BPB: {standard_bpb:.4f}  (elapsed: {elapsed:.1f}s)")

    # =========================================================================
    # TTT EVAL
    # =========================================================================
    ttt_bpb = None
    if not args.skip_ttt:
        print(f"\n--- TTT Eval (lr={args.ttt_lr}, steps={args.ttt_steps}) ---")
        print("Strategy: adapt on batch[i], eval on batch[i+1]")

        # Save original weights to restore after TTT
        original_params = copy_params(model)
        restore_params(model, original_params)

        t0 = time.perf_counter()
        total_nll_ttt = 0.0
        total_tok_ttt = 0
        total_bytes_ttt = 0.0

        # Process in TTT batches
        # We adapt on batch[i] then measure loss on batch[i+1]
        # First batch: eval without adaptation (cold start)
        batch_boundaries = list(range(0, total_seqs, ttt_batch_seqs))
        num_batches = len(batch_boundaries)

        adapt_params_saved = copy_params(model)  # track adapted state across batches

        for batch_i, s in enumerate(batch_boundaries):
            e = min(s + ttt_batch_seqs, total_seqs)
            chunk = val_tokens[s * seq_len:(e * seq_len) + 1]
            x_np = chunk[:-1].reshape(-1, seq_len)
            y_np = chunk[1:].reshape(-1, seq_len)

            if batch_i == 0:
                # First batch: eval without any adaptation
                x = mx.array(x_np, dtype=mx.int32)
                y = mx.array(y_np, dtype=mx.int32)
                ct = float(y.size)
                loss_val = model.loss_fn(x, y)
                mx.eval(loss_val)
                total_nll_ttt += float(loss_val.item()) * ct
                prev_ids = x_np.reshape(-1)
                tgt_ids = y_np.reshape(-1)
                bytes_np = base_bytes_lut[tgt_ids].astype(np.float64)
                bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.float64)
                total_tok_ttt += int(ct)
                total_bytes_ttt += bytes_np.sum()

                # Now adapt on this batch for the next eval
                ttt_update(model, x_np, y_np, args.ttt_lr, args.ttt_steps)
                adapt_params_saved = copy_params(model)

            else:
                # Eval on current batch (model was adapted on previous batch)
                x = mx.array(x_np, dtype=mx.int32)
                y = mx.array(y_np, dtype=mx.int32)
                ct = float(y.size)
                loss_val = model.loss_fn(x, y)
                mx.eval(loss_val)
                total_nll_ttt += float(loss_val.item()) * ct
                prev_ids = x_np.reshape(-1)
                tgt_ids = y_np.reshape(-1)
                bytes_np = base_bytes_lut[tgt_ids].astype(np.float64)
                bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.float64)
                total_tok_ttt += int(ct)
                total_bytes_ttt += bytes_np.sum()

                # Adapt on current batch (cumulative adaptation)
                ttt_update(model, x_np, y_np, args.ttt_lr, args.ttt_steps)
                adapt_params_saved = copy_params(model)

            if batch_i % 20 == 0:
                interim_ttt_bpb = compute_bpb(total_nll_ttt, total_tok_ttt, total_bytes_ttt)
                elapsed = time.perf_counter() - t0
                print(f"  batch {batch_i+1}/{num_batches} seqs {e}/{total_seqs} "
                      f"interim_ttt_bpb={interim_ttt_bpb:.4f} elapsed:{elapsed:.0f}s")

        ttt_bpb = compute_bpb(total_nll_ttt, total_tok_ttt, total_bytes_ttt)
        elapsed = time.perf_counter() - t0
        print(f"TTT BPB: {ttt_bpb:.4f}  (elapsed: {elapsed:.1f}s)")

        # Restore original weights
        restore_params(model, original_params)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if standard_bpb is not None:
        print(f"  Standard BPB: {standard_bpb:.4f}")
    if ttt_bpb is not None:
        print(f"  TTT BPB:      {ttt_bpb:.4f}  (lr={args.ttt_lr}, steps={args.ttt_steps})")
    if standard_bpb is not None and ttt_bpb is not None:
        delta = standard_bpb - ttt_bpb
        pct = delta / standard_bpb * 100
        if delta > 0:
            print(f"  TTT improvement: -{delta:.4f} ({pct:.2f}% better)")
            if delta > 0.02:
                print("  VERDICT: TTT is EFFECTIVE — integrate into final pipeline")
            elif delta > 0.005:
                print("  VERDICT: TTT shows modest improvement — worth testing at scale")
            else:
                print("  VERDICT: TTT improvement is marginal — may not be worth eval cost")
        else:
            print(f"  TTT degradation: +{-delta:.4f} ({-pct:.2f}% worse)")
            print("  VERDICT: TTT is NOT effective with these hyperparams — try lower lr or fewer steps")
    print("=" * 70)

if __name__ == "__main__":
    main()
