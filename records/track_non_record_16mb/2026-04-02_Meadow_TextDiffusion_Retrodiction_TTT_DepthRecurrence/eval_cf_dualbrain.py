#!/usr/bin/env -S python3 -u
"""
Coarse-to-Fine BPB Eval — TRUE Shared Dual-Brain

Uses ONE model (shared_ar_cdm.npz, 5L d=256) in TWO modes:
  - Left brain  (AR):  is_causal=True,  predicts skeleton positions
  - Right brain (CDM): is_causal=False, fills gaps bidirectionally

Both modes share the SAME weights. This is the real "R+L brain" setup.

Configurable via env vars:
  N_SEQS=20            # number of sequences to sample (20 = smoke, 200 = subset, 2000 = scaled, full val = 242K)
  SEQ_LEN=256          # sequence length (matches prior runs)
  STRIDE=2             # AR predicts every N-th position
  ROUNDS=2             # CDM refinement passes over gap positions
  N_RANDOM=3           # random fills per CDM round (variance reduction)
  MODEL_PATH=.../shared_ar_cdm.npz
  DATA_DIR=.../fineweb10B_sp1024
  TOKENIZER_PATH=.../fineweb_1024_bpe.model
  LOG_PATH=eval_cf_dualbrain.log   # if set, tee to file
  SEED=42
"""
import os, sys, math, time, json
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
        a non-Apple machine without raising at module load time. Real MLX work
        is gated on _HAS_MLX inside main()."""
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

# --- 5L d=256 config (matches shared_ar_cdm.npz) ---
VOCAB_SIZE = 1024
NUM_LAYERS = 5
MODEL_DIM = 256
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 3
ROPE_BASE = 10000.0
QK_GAIN_INIT = 1.5
LOGIT_SOFTCAP = 30.0
XSA_LAST_N = 2
BIGRAM_BUCKETS = 2048
BIGRAM_DIM = 64

# --- env config ---
N_SEQS       = int(os.environ.get("N_SEQS", "20"))
SEQ_LEN      = int(os.environ.get("SEQ_LEN", "256"))
STRIDE       = int(os.environ.get("STRIDE", "2"))
ROUNDS       = int(os.environ.get("ROUNDS", "2"))
N_RANDOM     = int(os.environ.get("N_RANDOM", "3"))
MODEL_PATH   = os.environ.get("MODEL_PATH",
                              os.path.expanduser("~/Desktop/github/meadow-parameter-golf/h100_results/shared_ar_cdm.npz"))
DATA_DIR     = os.environ.get("DATA_DIR",
                              os.path.expanduser("~/Desktop/github/parameter-golf/data/datasets/fineweb10B_sp1024"))
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH",
                                os.path.expanduser("~/Desktop/github/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"))
LOG_PATH     = os.environ.get("LOG_PATH", "")
SEED         = int(os.environ.get("SEED", "42"))

# --- tee logger ---
_log_fh = None
def log(msg=""):
    print(msg, flush=True)
    if _log_fh:
        _log_fh.write(msg + "\n"); _log_fh.flush()

if LOG_PATH:
    _log_fh = open(LOG_PATH, "w")

# ============================================================================
# Model (5L d=256 GPTv2 with U-net skip, matches shared_ar_cdm training)
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
def load_model(path):
    mx.random.seed(SEED)
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
    log(f"  Loaded {len(matched)}/{len(model_keys)} keys from {os.path.basename(path)}")
    return model

# ============================================================================
# Byte counting (tokenizer-agnostic BPB denominator)
# ============================================================================
def build_byte_luts(sp):
    sz = int(sp.vocab_size())
    bb = np.zeros(sz, dtype=np.int16)
    hs = np.zeros(sz, dtype=np.bool_)
    ib = np.ones(sz, dtype=np.bool_)
    for t in range(sz):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
        ib[t] = False
        if sp.is_byte(t): bb[t] = 1; continue
        p = sp.id_to_piece(t)
        if p.startswith("\u2581"): hs[t] = True; p = p[1:]
        bb[t] = len(p.encode("utf-8"))
    return bb, hs, ib

def count_bytes(tokens, prev_tokens, bb, hs, ib):
    total = 0.0
    for i in range(len(tokens)):
        b = float(bb[tokens[i]])
        if hs[tokens[i]] and not ib[prev_tokens[i]]:
            b += 1.0
        total += max(b, 1.0)
    return total

def split_rounds(cdm_positions, cdm_rounds):
    rounds = [[] for _ in range(cdm_rounds)]
    for i, pos in enumerate(cdm_positions):
        rounds[i % cdm_rounds].append(pos)
    return rounds

# ============================================================================
# Pure AR eval (single-mode baseline, same shared model in is_causal=True)
# ============================================================================
def eval_pure_ar(model, tokens, sp, n_seqs, seq_len):
    bb, hs, ib = build_byte_luts(sp)
    total_nll = 0.0; total_bytes = 0.0
    rng = np.random.RandomState(SEED)
    t0 = time.time()
    for s in range(n_seqs):
        idx = rng.randint(0, len(tokens) - seq_len - 1)
        seq = tokens[idx:idx+seq_len+1]
        inp = mx.array(seq[:-1].reshape(1,-1))
        tgt = seq[1:]
        prev = seq[:-1]
        logits = model.get_logits(inp, is_causal=True)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(lp)
        lp_np = np.array(lp.astype(mx.float32))[0]
        for t in range(seq_len):
            total_nll -= lp_np[t, int(tgt[t])]
        total_bytes += count_bytes(tgt, prev, bb, hs, ib)
        if (s+1) % max(1, n_seqs//10) == 0:
            elapsed = time.time() - t0
            bpb = total_nll / total_bytes / math.log(2)
            log(f"    AR {s+1}/{n_seqs} | BPB:{bpb:.4f} | {elapsed:.1f}s")
    return total_nll / total_bytes / math.log(2)

# ============================================================================
# Coarse-to-Fine eval (L brain: causal / R brain: bidirect, SAME weights)
# ============================================================================
def eval_coarse_to_fine(model, tokens, sp, n_seqs, seq_len, stride, n_random, cdm_rounds):
    bb, hs, ib = build_byte_luts(sp)
    log(f"  CF eval (stride={stride}, rounds={cdm_rounds}, {n_seqs} seqs × {seq_len} tok, R={n_random})")

    total_ar_nll = 0.0
    total_cdm_nll = 0.0
    total_bytes = 0.0
    rng = np.random.RandomState(SEED + 1)
    t0 = time.time()

    for s in range(n_seqs):
        idx = rng.randint(0, len(tokens) - seq_len - 1)
        seq = tokens[idx:idx + seq_len + 1]
        x = seq[1:]
        prev = seq[:-1]
        total_bytes += count_bytes(x, prev, bb, hs, ib)

        ar_positions = list(range(0, seq_len, stride))
        cdm_positions = [i for i in range(seq_len) if i not in ar_positions]
        round_groups = split_rounds(cdm_positions, cdm_rounds)

        # === L brain: causal AR over full input, only score skeleton positions ===
        input_ids = seq[:-1]
        input_mx = mx.array(input_ids.reshape(1, -1))
        ar_logits = model.get_logits(input_mx, is_causal=True)      # ← is_causal=True
        ar_lp = ar_logits - mx.logsumexp(ar_logits, axis=-1, keepdims=True)
        mx.eval(ar_lp)
        ar_lp_np = np.array(ar_lp.astype(mx.float32))[0]
        for pos in ar_positions:
            total_ar_nll -= ar_lp_np[pos, int(x[pos])]

        # === R brain: bidirectional CDM, rounds of gap filling ===
        for ridx, current_round in enumerate(round_groups):
            if not current_round: continue
            unresolved = set()
            for g in round_groups[ridx:]:
                unresolved.update(g)

            avg_round_nll = np.zeros(len(current_round))
            for r in range(n_random):
                cdm_input = x.copy()
                for pos in unresolved:
                    cdm_input[pos] = rng.randint(0, VOCAB_SIZE)
                cdm_input_mx = mx.array(cdm_input.reshape(1, -1))
                cdm_logits = model.get_logits(cdm_input_mx, is_causal=False)   # ← is_causal=False, SAME model
                cdm_lp = cdm_logits - mx.logsumexp(cdm_logits, axis=-1, keepdims=True)
                mx.eval(cdm_lp)
                cdm_lp_np = np.array(cdm_lp.astype(mx.float32))[0]
                for i, pos in enumerate(current_round):
                    avg_round_nll[i] -= cdm_lp_np[pos, int(x[pos])] / n_random

            total_cdm_nll += avg_round_nll.sum()

        if (s + 1) % max(1, n_seqs // 20) == 0:
            elapsed = time.time() - t0
            ar_bpb = total_ar_nll / total_bytes / math.log(2)
            cdm_bpb = total_cdm_nll / total_bytes / math.log(2)
            total_bpb = (total_ar_nll + total_cdm_nll) / total_bytes / math.log(2)
            rate = (s+1) / elapsed if elapsed > 0 else 0
            eta = (n_seqs - (s+1)) / rate if rate > 0 else 0
            log(f"    CF {s+1}/{n_seqs} | AR:{ar_bpb:.4f} + CDM:{cdm_bpb:.4f} = {total_bpb:.4f} | {elapsed:.0f}s ETA:{eta:.0f}s")

    ar_bpb = total_ar_nll / total_bytes / math.log(2)
    cdm_bpb = total_cdm_nll / total_bytes / math.log(2)
    total_bpb = (total_ar_nll + total_cdm_nll) / total_bytes / math.log(2)
    return ar_bpb, cdm_bpb, total_bpb

# ============================================================================
def main():
    log("=" * 68)
    log(f"  Shared Dual-Brain CF Eval  |  N_SEQS={N_SEQS}  stride={STRIDE}  rounds={ROUNDS}")
    log(f"  Model: {os.path.basename(MODEL_PATH)}  (5L d=256, ONE set of weights, TWO modes)")
    log("=" * 68)

    np.random.seed(SEED); mx.random.seed(SEED)
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)

    header = np.fromfile(f"{DATA_DIR}/fineweb_val_000000.bin", dtype="<i4", count=256)
    val = np.fromfile(f"{DATA_DIR}/fineweb_val_000000.bin", dtype="<u2",
                      count=int(header[2]), offset=256*4).astype(np.int32)
    log(f"Val tokens available: {len(val):,}")
    log(f"Sampling {N_SEQS} sequences of length {SEQ_LEN} = {N_SEQS*SEQ_LEN:,} tokens scored")

    log("\nLoading shared AR+CDM model (ONE model, used in both modes)...")
    model = load_model(MODEL_PATH)

    log("\n[1] Pure AR baseline (single-mode, left brain only, is_causal=True)")
    t0 = time.time()
    ar_only = eval_pure_ar(model, val, sp, N_SEQS, SEQ_LEN)
    log(f"  Pure AR BPB: {ar_only:.4f}  ({time.time()-t0:.0f}s)")

    log(f"\n[2] Coarse-to-Fine dual-brain (stride={STRIDE}, rounds={ROUNDS})")
    t0 = time.time()
    ar_bpb, cdm_bpb, cf_bpb = eval_coarse_to_fine(
        model, val, sp, N_SEQS, SEQ_LEN, STRIDE, N_RANDOM, ROUNDS)
    dt = time.time() - t0

    ar_frac = 1.0 / STRIDE
    cdm_frac = 1.0 - ar_frac
    delta_pct = (cf_bpb/ar_only - 1) * 100

    log("\n" + "=" * 68)
    log(f"  RESULTS  (n_seqs={N_SEQS}, seq_len={SEQ_LEN}, total scored={N_SEQS*SEQ_LEN:,} tokens)")
    log("=" * 68)
    log(f"  Pure AR (baseline):              {ar_only:.4f}")
    log(f"  CF AR part ({ar_frac*100:.0f}% of positions):       {ar_bpb:.4f}")
    log(f"  CF CDM part ({cdm_frac*100:.0f}% of positions):     {cdm_bpb:.4f}")
    log(f"  CF Total (dual-brain):           {cf_bpb:.4f}")
    log(f"  CF vs Pure AR:                   {delta_pct:+.2f}%  ({'-' if delta_pct<0 else '+'}{abs(cf_bpb-ar_only):.4f} BPB)")
    log(f"  CF eval time:                    {dt:.0f}s  ({N_SEQS/dt:.1f} seqs/s)")
    log("=" * 68)

    # machine-readable result
    result = {
        "n_seqs": N_SEQS, "seq_len": SEQ_LEN, "tokens_scored": N_SEQS*SEQ_LEN,
        "stride": STRIDE, "rounds": ROUNDS, "n_random": N_RANDOM,
        "pure_ar_bpb": float(ar_only),
        "cf_ar_part": float(ar_bpb),
        "cf_cdm_part": float(cdm_bpb),
        "cf_total": float(cf_bpb),
        "cf_vs_ar_pct": float(delta_pct),
        "cf_time_sec": float(dt),
        "model": os.path.basename(MODEL_PATH),
        "seed": SEED,
    }
    log("\nJSON:")
    log(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
    if _log_fh:
        _log_fh.close()
