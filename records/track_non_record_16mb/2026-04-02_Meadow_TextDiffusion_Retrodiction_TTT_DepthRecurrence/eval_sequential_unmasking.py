#!/usr/bin/env -S python3 -u
"""
Sequential Unmasking BPB Eval — 正確的 CDM 量尺

不改模型。用 chain rule 精確計算 log P(x)：

  Step 1: [rand, rand, rand, ...] → P(x_1)
  Step 2: [x_1,  rand, rand, ...] → P(x_2 | x_1)
  Step 3: [x_1,  x_2,  rand, ...] → P(x_3 | x_1, x_2)
  ...
  Step L: [x_1,  x_2,  ..., x_{L-1}, rand] → P(x_L | x_{<L})

BPB = -Σ log₂ P(x_t | x_{<t}, rand_{≥t}) / total_bytes

比較：
  A: CDM model + sequential unmasking (新尺)
  B: AR model + standard causal eval (傳統尺)
"""
import glob, math, time, os
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
    def tree_flatten(*a, **k):
        raise RuntimeError("MLX not available; this script is Apple Silicon only.")
    def tree_unflatten(*a, **k):
        raise RuntimeError("MLX not available; this script is Apple Silicon only.")

COMPUTE_DTYPE = mx.bfloat16
DATA_DIR = "/Users/akaihuangm1/Desktop/github/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/Users/akaihuangm1/Desktop/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

VOCAB_SIZE = 1024
MODEL_DIM = 512
NUM_LAYERS = 11
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 3
ROPE_BASE = 10000.0
QK_GAIN_INIT = 1.5
LOGIT_SOFTCAP = 30.0
SEQ_LEN = 1024
XSA_LAST_N = 4
BIGRAM_BUCKETS = 2048
BIGRAM_DIM = 128

# ============================================================================
# Model (same as golf_v2)
# ============================================================================
def rms_norm(x, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x*x, axis=-1, keepdims=True) + eps)).astype(x.dtype)

class CastedLinear(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.weight = nn.Linear(i, o, bias=False).weight.astype(mx.float32)
    def __call__(s, x): return x @ s.weight.astype(x.dtype).T

class SmearGate(nn.Module):
    def __init__(s, d):
        super().__init__(); s.gate = mx.zeros((d,), dtype=mx.float32)
    def __call__(s, x):
        g = mx.sigmoid(s.gate.astype(x.dtype))[None, None, :]
        return (1-g)*x + g*mx.concatenate([mx.zeros_like(x[:,:1]), x[:,:-1]], axis=1)

class BigramHashEmbedding(nn.Module):
    def __init__(s):
        super().__init__()
        s.embed = nn.Embedding(BIGRAM_BUCKETS, BIGRAM_DIM)
        s.embed.weight = mx.zeros_like(s.embed.weight)
        s.proj = CastedLinear(BIGRAM_DIM, MODEL_DIM)
        s.scale = mx.array(0.05, dtype=mx.float32)
    def __call__(s, t):
        ti = t.astype(mx.int32); m = BIGRAM_BUCKETS - 1
        sh = mx.concatenate([mx.full((ti.shape[0],1), m, dtype=mx.int32), ti[:,:-1]], axis=1)
        return s.proj(s.embed((36313*ti + 27191*sh) % m)) * s.scale.astype(COMPUTE_DTYPE)

class DualModeAttention(nn.Module):
    def __init__(s, use_xsa=False):
        super().__init__()
        s.nh=NUM_HEADS; s.nkv=NUM_KV_HEADS; s.hd=MODEL_DIM//NUM_HEADS; s.use_xsa=use_xsa
        s.c_q=CastedLinear(MODEL_DIM, MODEL_DIM)
        s.c_k=CastedLinear(MODEL_DIM, NUM_KV_HEADS*s.hd)
        s.c_v=CastedLinear(MODEL_DIM, NUM_KV_HEADS*s.hd)
        s.proj=CastedLinear(MODEL_DIM, MODEL_DIM)
        s.q_gain=mx.ones((NUM_HEADS,), dtype=mx.float32)*QK_GAIN_INIT
        s.rope=nn.RoPE(s.hd, traditional=False, base=ROPE_BASE)
        s.scale=s.hd**-0.5
    def __call__(s, x, is_causal=True):
        B,T,D = x.shape
        q=s.c_q(x).reshape(B,T,s.nh,s.hd).transpose(0,2,1,3)
        k=s.c_k(x).reshape(B,T,s.nkv,s.hd).transpose(0,2,1,3)
        v=s.c_v(x).reshape(B,T,s.nkv,s.hd).transpose(0,2,1,3)
        q=s.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k=s.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q=q*s.q_gain.astype(q.dtype)[None,:,None,None]
        y=mx.fast.scaled_dot_product_attention(q,k,v,scale=s.scale,
            mask="causal" if is_causal else None)
        return s.proj(y.transpose(0,2,1,3).reshape(B,T,D))

class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.fc=CastedLinear(MODEL_DIM, MODEL_DIM*MLP_MULT)
        s.proj=CastedLinear(MODEL_DIM*MLP_MULT, MODEL_DIM)
    def __call__(s, x):
        h=s.fc(x); h=mx.where(h>=0,h,0.5*h); return s.proj(h*h)

class Block(nn.Module):
    def __init__(s, li=0, use_xsa=False):
        super().__init__()
        s.attn=DualModeAttention(use_xsa=use_xsa); s.mlp=MLP()
        s.ln_scale=1.0/math.sqrt(li+1)
        s.attn_scale=mx.ones((MODEL_DIM,), dtype=mx.float32)
        s.mlp_scale=mx.ones((MODEL_DIM,), dtype=mx.float32)
        s.resid_mix=mx.array(np.stack((np.ones(MODEL_DIM,dtype=np.float32),
                                        np.zeros(MODEL_DIM,dtype=np.float32))))
    def __call__(s, x, x0, is_causal=True):
        m=s.resid_mix.astype(x.dtype)
        x=m[0][None,None,:]*x+m[1][None,None,:]*x0
        x=x+s.attn_scale.astype(x.dtype)[None,None,:]*s.attn(rms_norm(x)*s.ln_scale, is_causal=is_causal)
        x=x+s.mlp_scale.astype(x.dtype)[None,None,:]*s.mlp(rms_norm(x)*s.ln_scale)
        return x

class GPTv2(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok_emb=nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        s.bigram=BigramHashEmbedding(); s.smear=SmearGate(MODEL_DIM)
        ne=NUM_LAYERS//2; nd=NUM_LAYERS-ne; s.ne=ne; s.nd=nd
        s.skip_weights=mx.ones((min(ne,nd),MODEL_DIM), dtype=mx.float32)
        s.blocks=[Block(li=i, use_xsa=i>=(NUM_LAYERS-XSA_LAST_N)) for i in range(NUM_LAYERS)]
        s.tok_emb.weight=(mx.random.normal(s.tok_emb.weight.shape)*0.005).astype(COMPUTE_DTYPE)

    def forward_hidden(s, ids, is_causal=True):
        x=s.tok_emb(ids).astype(COMPUTE_DTYPE)+s.bigram(ids).astype(COMPUTE_DTYPE)
        x=rms_norm(x); x=s.smear(x); x0=x; skips=[]
        for i in range(s.ne):
            x=s.blocks[i](x, x0, is_causal=is_causal); skips.append(x)
        for i in range(s.nd):
            if skips: x=x+s.skip_weights[i].astype(x.dtype)[None,None,:]*skips.pop()
            x=s.blocks[s.ne+i](x, x0, is_causal=is_causal)
        return rms_norm(x)

    def get_logits(s, ids, is_causal=True):
        h = s.forward_hidden(ids, is_causal=is_causal)
        return LOGIT_SOFTCAP * mx.tanh(h @ s.tok_emb.weight.astype(h.dtype).T / LOGIT_SOFTCAP)

# ============================================================================
# Load model
# ============================================================================
def load_model(path):
    mx.random.seed(42)
    model = GPTv2()
    w = dict(np.load(path))
    loaded = {}
    for k, v in w.items():
        if v.dtype.str == '|V2':
            loaded[k] = mx.array(v.view(np.uint16)).view(mx.bfloat16)
        else:
            loaded[k] = mx.array(v)
    model_keys = set(k for k, _ in tree_flatten(model.parameters()))
    matched = {k: v for k, v in loaded.items() if k in model_keys}
    model.update(tree_unflatten(list(matched.items())))
    mx.eval(model.parameters())
    print(f"  Loaded {len(matched)}/{len(model_keys)} keys from {os.path.basename(path)}")
    return model

# ============================================================================
# Sequential Unmasking BPB
# ============================================================================
def eval_sequential_unmasking(model, tokens, sp, n_seqs=20, seq_len=256, batch_positions=32, n_random=3):
    """
    Sequential Unmasking: exact BPB via chain rule.

    For each position t:
      input = [x_1..x_{t-1} (real), rand_t..rand_L (random)]
      output P(x_t | input) at position t

    Batch: process `batch_positions` positions per forward pass.
    n_random: average over multiple random fills for variance reduction.
    """
    print(f"  Sequential Unmasking eval ({n_seqs} seqs, seq_len={seq_len}, batch={batch_positions}, R={n_random})...")

    # Byte counting
    base_bytes = np.zeros(int(sp.vocab_size()), dtype=np.int16)
    has_space = np.zeros(int(sp.vocab_size()), dtype=np.bool_)
    is_boundary = np.ones(int(sp.vocab_size()), dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary[tid] = False
        if sp.is_byte(tid): base_bytes[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): has_space[tid] = True; piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))

    total_nll = 0.0
    total_bytes = 0.0
    t0 = time.time()

    for s in range(n_seqs):
        idx = np.random.randint(0, len(tokens) - seq_len - 1)
        seq = tokens[idx:idx + seq_len + 1]  # +1 for prev token
        x = seq[1:]   # tokens to predict (positions 0..seq_len-1)
        prev = seq[:-1]  # previous tokens for byte counting

        # Byte counting for this sequence
        for t in range(seq_len):
            b = float(base_bytes[x[t]])
            if has_space[x[t]] and not is_boundary[prev[t]]:
                b += 1.0
            total_bytes += max(b, 1.0)

        # Sequential unmasking: process in chunks of batch_positions
        for start in range(0, seq_len, batch_positions):
            end = min(start + batch_positions, seq_len)
            n_pos = end - start

            avg_nll = np.zeros(n_pos)

            for r in range(n_random):
                # Build batch: each sample reveals up to position t
                batch = np.zeros((n_pos, seq_len), dtype=np.int32)
                for i in range(n_pos):
                    t = start + i
                    # Positions < t: real tokens
                    batch[i, :t] = x[:t]
                    # Positions >= t: random tokens
                    batch[i, t:] = np.random.randint(0, VOCAB_SIZE, size=seq_len - t)

                # Forward pass (bidirectional!)
                batch_mx = mx.array(batch)
                logits = model.get_logits(batch_mx, is_causal=False)  # [n_pos, seq_len, vocab]
                log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)  # log_softmax
                mx.eval(log_probs)
                log_probs_np = np.array(log_probs.astype(mx.float32))

                # Extract P(x_t) at position t for each sample
                for i in range(n_pos):
                    t = start + i
                    target = int(x[t])
                    nll = -log_probs_np[i, t, target]
                    avg_nll[i] += nll / n_random

            total_nll += avg_nll.sum()

        if (s + 1) % 5 == 0:
            elapsed = time.time() - t0
            bpb = total_nll / total_bytes / math.log(2)
            print(f"    {s+1}/{n_seqs} | BPB: {bpb:.4f} | {elapsed:.1f}s")

    bpb = total_nll / total_bytes / math.log(2)
    return bpb

# ============================================================================
# Standard Causal BPB (for comparison)
# ============================================================================
def eval_causal_bpb(model, tokens, sp, n_seqs=20, seq_len=256):
    """Standard AR causal BPB."""
    print(f"  Causal AR eval ({n_seqs} seqs, seq_len={seq_len})...")

    base_bytes = np.zeros(int(sp.vocab_size()), dtype=np.int16)
    has_space = np.zeros(int(sp.vocab_size()), dtype=np.bool_)
    is_boundary = np.ones(int(sp.vocab_size()), dtype=np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary[tid] = False
        if sp.is_byte(tid): base_bytes[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): has_space[tid] = True; piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))

    total_nll = 0.0
    total_bytes = 0.0
    t0 = time.time()

    for s in range(n_seqs):
        idx = np.random.randint(0, len(tokens) - seq_len - 1)
        x = mx.array(tokens[idx:idx+seq_len].reshape(1, -1))
        y = tokens[idx+1:idx+seq_len+1]
        prev = tokens[idx:idx+seq_len]

        logits = model.get_logits(x, is_causal=True)  # [1, seq_len, vocab]
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(log_probs)
        lp = np.array(log_probs.astype(mx.float32))[0]

        for t in range(seq_len):
            total_nll -= lp[t, int(y[t])]
            b = float(base_bytes[y[t]])
            if has_space[y[t]] and not is_boundary[prev[t]]:
                b += 1.0
            total_bytes += max(b, 1.0)

    bpb = total_nll / total_bytes / math.log(2)
    elapsed = time.time() - t0
    print(f"    Done | BPB: {bpb:.4f} | {elapsed:.1f}s")
    return bpb

# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  Sequential Unmasking vs Causal Eval")
    print("  正確的 CDM 量尺 vs 傳統 AR 量尺")
    print("=" * 60)

    np.random.seed(42); mx.random.seed(42)

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)

    # Load validation data
    header = np.fromfile(f"{DATA_DIR}/fineweb_val_000000.bin", dtype="<i4", count=256)
    val_tokens = np.fromfile(f"{DATA_DIR}/fineweb_val_000000.bin", dtype="<u2",
                             count=int(header[2]), offset=256*4).astype(np.int32)
    print(f"Val: {len(val_tokens):,} tokens")

    # === CDM model (right brain, bidirectional) ===
    print(f"\n[1] Loading CDM model (right brain)...")
    cdm = load_model(os.path.join(SCRIPT_DIR, "diffusion_right_brain_27M.npz"))

    # === AR model (left brain, causal) ===
    print(f"\n[2] Loading AR model (left brain)...")
    ar = load_model(os.path.join(SCRIPT_DIR, "golf_v2_retro_5000step.npz"))

    N_SEQS = 20
    SEQ = 256  # shorter for speed

    # === Eval ===
    print(f"\n[3] CDM + Sequential Unmasking (新尺):")
    cdm_su_bpb = eval_sequential_unmasking(cdm, val_tokens, sp,
                                             n_seqs=N_SEQS, seq_len=SEQ,
                                             batch_positions=32, n_random=3)

    print(f"\n[4] CDM + Causal Eval (舊的錯誤尺):")
    cdm_causal_bpb = eval_causal_bpb(cdm, val_tokens, sp, n_seqs=N_SEQS, seq_len=SEQ)

    print(f"\n[5] AR + Causal Eval (傳統尺):")
    ar_causal_bpb = eval_causal_bpb(ar, val_tokens, sp, n_seqs=N_SEQS, seq_len=SEQ)

    print(f"\n[6] AR + Sequential Unmasking (AR 用新尺會怎樣):")
    ar_su_bpb = eval_sequential_unmasking(ar, val_tokens, sp,
                                            n_seqs=N_SEQS, seq_len=SEQ,
                                            batch_positions=32, n_random=3)

    # === Results ===
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"")
    print(f"  {'Model':<20s} {'Eval':<25s} {'BPB':>8s}")
    print(f"  {'-'*55}")
    print(f"  {'CDM (right brain)':<20s} {'Sequential Unmasking':<25s} {cdm_su_bpb:>8.4f}  ← 新尺")
    print(f"  {'CDM (right brain)':<20s} {'Causal (wrong ruler)':<25s} {cdm_causal_bpb:>8.4f}  ← 舊尺")
    print(f"  {'AR (left brain)':<20s} {'Causal (standard)':<25s} {ar_causal_bpb:>8.4f}  ← 傳統")
    print(f"  {'AR (left brain)':<20s} {'Sequential Unmasking':<25s} {ar_su_bpb:>8.4f}  ← AR 用新尺")
    print(f"")

    if cdm_su_bpb < ar_causal_bpb:
        improvement = (1 - cdm_su_bpb / ar_causal_bpb) * 100
        print(f"  ✅ CDM + 正確量尺 WINS!")
        print(f"     CDM {cdm_su_bpb:.4f} < AR {ar_causal_bpb:.4f}")
        print(f"     改善: {improvement:.1f}%")
    else:
        gap = (cdm_su_bpb / ar_causal_bpb - 1) * 100
        print(f"  AR still leads by {gap:.1f}%")
        print(f"     CDM {cdm_su_bpb:.4f} vs AR {ar_causal_bpb:.4f}")

if __name__ == "__main__":
    main()
