#!/usr/bin/env -S python3 -u
"""
Coarse-to-Fine BPB Eval (PyTorch/CUDA) — TRUE Shared Dual-Brain, 11L variant

Counterpart of eval_cf_dualbrain.py (MLX/5L) for H100.

Loads a PyTorch 11L shared AR+CDM checkpoint and runs it in BOTH modes
(is_causal=True for L-brain, is_causal=False for R-brain) via the SAME weights.

Expects train_cdm_ar_v4096_pytorch.py to be importable (copy to same dir on pod).

Env vars:
  N_SEQS=20            # smoke=20, scaled=10000, full val ~ 242000
  SEQ_LEN=1024         # 11L model was trained at seq_len=1024
  STRIDE=2             # AR every N-th position
  ROUNDS=2             # CDM refinement rounds
  N_RANDOM=3           # random fills per round
  MODEL_PATH=.../11L_shared_cdm_bf16.pt
  DATA_DIR=/workspace/data_v4096_full          # v4096 tokenized FineWeb
  TOKENIZER_PATH=/workspace/bpe_v4096.model
  LOG_PATH=eval_cf_11L.log                     # tee to file
  SEED=42
  DEVICE=cuda
  BATCH=1              # sequences per forward pass (kept small for now)
"""
import os, sys, math, time, json
import numpy as np
import torch
import sentencepiece as spm

# --- env config ---
N_SEQS       = int(os.environ.get("N_SEQS", "20"))
SEQ_LEN      = int(os.environ.get("SEQ_LEN", "1024"))
STRIDE       = int(os.environ.get("STRIDE", "2"))
ROUNDS       = int(os.environ.get("ROUNDS", "2"))
N_RANDOM     = int(os.environ.get("N_RANDOM", "3"))
MODEL_PATH   = os.environ.get("MODEL_PATH", "/workspace/models/11L_shared_cdm_bf16.pt")
DATA_DIR     = os.environ.get("DATA_DIR", "/workspace/data_v4096_full")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/workspace/bpe_v4096.model")
LOG_PATH     = os.environ.get("LOG_PATH", "")
SEED         = int(os.environ.get("SEED", "42"))
DEVICE       = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_log_fh = None
def log(msg=""):
    print(msg, flush=True)
    if _log_fh:
        _log_fh.write(msg + "\n"); _log_fh.flush()

if LOG_PATH:
    _log_fh = open(LOG_PATH, "w")

# ============================================================================
# Import the model class from the training file (must be in PYTHONPATH)
# ============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_cdm_ar_v4096_pytorch as trmod

# Sanity check
assert trmod.VOCAB_SIZE == 4096, f"Expected VOCAB_SIZE=4096, got {trmod.VOCAB_SIZE}"
assert trmod.NUM_LAYERS == 11,   f"Expected NUM_LAYERS=11, got {trmod.NUM_LAYERS}"
assert trmod.MODEL_DIM == 512,   f"Expected MODEL_DIM=512, got {trmod.MODEL_DIM}"

VOCAB_SIZE = trmod.VOCAB_SIZE
MODEL_DIM = trmod.MODEL_DIM

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
# Logits from hidden state (matches train_cdm_ar_v4096_pytorch.forward)
# ============================================================================
@torch.no_grad()
def get_logits(model, input_ids, is_causal):
    h = model.forward_hidden(input_ids, is_causal=is_causal)  # [B, T, D]
    logits = model.softcap(torch.nn.functional.linear(h, model.tok_emb.weight.to(h.dtype)))
    return logits

@torch.no_grad()
def get_log_probs(model, input_ids, is_causal):
    logits = get_logits(model, input_ids, is_causal)
    return torch.log_softmax(logits.float(), dim=-1)

# ============================================================================
# Pure AR eval (shared model in single-mode is_causal=True)
# ============================================================================
@torch.no_grad()
def eval_pure_ar(model, tokens, sp, n_seqs, seq_len):
    bb, hs, ib = build_byte_luts(sp)
    total_nll = 0.0; total_bytes = 0.0
    rng = np.random.RandomState(SEED)
    t0 = time.time()
    for s in range(n_seqs):
        idx = rng.randint(0, len(tokens) - seq_len - 1)
        seq = tokens[idx:idx+seq_len+1]
        inp = torch.from_numpy(seq[:-1].reshape(1,-1).astype(np.int64)).to(DEVICE)
        tgt = seq[1:]
        prev = seq[:-1]
        lp = get_log_probs(model, inp, is_causal=True)[0].cpu().numpy()
        for t in range(seq_len):
            total_nll -= lp[t, int(tgt[t])]
        total_bytes += count_bytes(tgt, prev, bb, hs, ib)
        if (s+1) % max(1, n_seqs//10) == 0:
            elapsed = time.time() - t0
            bpb = total_nll / total_bytes / math.log(2)
            log(f"    AR {s+1}/{n_seqs} | BPB:{bpb:.4f} | {elapsed:.1f}s")
    return total_nll / total_bytes / math.log(2)

# ============================================================================
# Coarse-to-Fine dual-brain eval
# ============================================================================
@torch.no_grad()
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

        # === L brain: causal AR, score skeleton positions ===
        input_ids = seq[:-1]
        input_t = torch.from_numpy(input_ids.reshape(1, -1).astype(np.int64)).to(DEVICE)
        ar_lp = get_log_probs(model, input_t, is_causal=True)[0].cpu().numpy()
        for pos in ar_positions:
            total_ar_nll -= ar_lp[pos, int(x[pos])]

        # === R brain: bidirectional CDM, rounds of gap fill, SAME weights ===
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
                cdm_input_t = torch.from_numpy(cdm_input.reshape(1,-1).astype(np.int64)).to(DEVICE)
                cdm_lp = get_log_probs(model, cdm_input_t, is_causal=False)[0].cpu().numpy()
                for i, pos in enumerate(current_round):
                    avg_round_nll[i] -= cdm_lp[pos, int(x[pos])] / n_random

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
def load_val_tokens():
    """Parameter-golf binary format: 256 × int32 header + uint16 tokens."""
    val_path = os.path.join(DATA_DIR, "fineweb_val_000000.bin")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Val data not found: {val_path}")
    header = np.fromfile(val_path, dtype="<i4", count=256)
    val = np.fromfile(val_path, dtype="<u2",
                      count=int(header[2]), offset=256*4).astype(np.int32)
    return val

def load_model():
    log(f"Instantiating GPTv2 (V={VOCAB_SIZE}, L={trmod.NUM_LAYERS}, D={MODEL_DIM})")
    model = trmod.GPTv2().to(DEVICE)
    log(f"Loading state_dict from {MODEL_PATH}")
    sd = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # strip common prefixes
    clean = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."): k2 = k2[7:]
        if k2.startswith("_orig_mod."): k2 = k2[10:]
        clean[k2] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    log(f"  loaded {len(clean)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:  log(f"  missing first 5:   {list(missing)[:5]}")
    if unexpected: log(f"  unexpected first 5:{list(unexpected)[:5]}")
    model.eval()
    return model

def main():
    log("=" * 70)
    log(f"  CUDA Shared Dual-Brain CF Eval | N_SEQS={N_SEQS} stride={STRIDE} rounds={ROUNDS}")
    log(f"  Device: {DEVICE}")
    log(f"  Model:  {MODEL_PATH}")
    log(f"  Seq len: {SEQ_LEN}   Seed: {SEED}")
    log("=" * 70)

    torch.manual_seed(SEED); np.random.seed(SEED)

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    log(f"Tokenizer: vocab_size={sp.vocab_size()}")

    val = load_val_tokens()
    log(f"Val tokens available: {len(val):,}")
    log(f"Sampling {N_SEQS} seqs × {SEQ_LEN} tok = {N_SEQS*SEQ_LEN:,} tokens scored\n")

    model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  params: {n_params:,}")

    log("\n[1] Pure AR baseline (shared model, is_causal=True)")
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

    log("\n" + "=" * 70)
    log(f"  RESULTS (n_seqs={N_SEQS}, seq_len={SEQ_LEN}, total={N_SEQS*SEQ_LEN:,} tokens)")
    log("=" * 70)
    log(f"  Pure AR (baseline):              {ar_only:.4f}")
    log(f"  CF AR part ({ar_frac*100:.0f}% positions):       {ar_bpb:.4f}")
    log(f"  CF CDM part ({cdm_frac*100:.0f}% positions):     {cdm_bpb:.4f}")
    log(f"  CF Total (dual-brain):           {cf_bpb:.4f}")
    log(f"  CF vs Pure AR:                   {delta_pct:+.2f}%  ({'-' if delta_pct<0 else '+'}{abs(cf_bpb-ar_only):.4f} BPB)")
    log(f"  CF eval time:                    {dt:.0f}s  ({N_SEQS/dt:.1f} seqs/s)")
    log("=" * 70)

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
        "n_params": n_params,
        "device": DEVICE,
        "seed": SEED,
    }
    log("\nJSON:")
    log(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
    if _log_fh: _log_fh.close()
