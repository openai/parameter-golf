#!/usr/bin/env python3
"""
Trivial future-token leakage test for the shared AR+CDM model.

Procedure:
  1. Load a shared model checkpoint (via MODEL_PATH env var)
  2. Create two sequences identical for positions 0..t and different for t+1..L-1
  3. Forward both with is_causal=True
  4. Assert: logits[position < t] are identical between sequences
     (if not, future tokens are leaking into earlier positions)

Usage:
  MODEL_PATH=./shared_ar_cdm.npz python3 leakage_test.py
  # or from the PR folder with the checkpoint downloaded alongside:
  MODEL_PATH=../../../../models/shared_ar_cdm.npz python3 leakage_test.py

Requires mlx >= 0.31 and eval_cf_dualbrain.py in the same directory (the test
re-uses the GPTv2 class from that script).
"""
import os, sys, numpy as np
# MLX imports — guarded so this Apple-Silicon-only pre-flight script can be
# IMPORTED on Linux CPU smoke-test environments (Python 3.10 + torch CPU,
# where `pip install mlx` is not possible). When the file is imported (not
# executed as __main__), nothing below the imports actually runs. When the
# file is executed as __main__ on a non-Apple-Silicon machine without MLX
# installed, it exits cleanly with a clear message.
try:
    import mlx.core as mx
    import mlx.nn as nn
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False
    mx = None
    nn = None


def _main() -> int:
    """Run the leakage test. Requires MLX to be available."""
    if not _HAS_MLX:
        print("leakage_test.py requires Apple MLX (Apple Silicon only).")
        print("It is included in the PR folder as M1 pre-flight reproducibility")
        print("evidence; the H100 production CF eval uses eval_cf_ablation.py.")
        print("Install MLX with `pip install mlx` on an Apple Silicon Mac and")
        print("re-run with `python3 leakage_test.py`.")
        return 0

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(SCRIPT_DIR, "shared_ar_cdm.npz"))
    EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_cf_dualbrain.py")

    # Pull the GPTv2 class from eval_cf_dualbrain.py without running its main()
    with open(EVAL_SCRIPT) as f:
        src = f.read()
    src_prelude = src.split("def main()")[0]
    ns = {"__name__": "leakage_test_module", "__file__": EVAL_SCRIPT}
    exec(src_prelude, ns)

    GPTv2 = ns["GPTv2"]
    load_model = ns["load_model"]
    MODEL_DIM = ns["MODEL_DIM"]
    VOCAB_SIZE = ns["VOCAB_SIZE"]

    # ---- Load model ----
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: MODEL_PATH not found: {MODEL_PATH}")
        print(f"Set MODEL_PATH to a shared_ar_cdm.npz checkpoint.")
        print(f"Download from: https://huggingface.co/datasets/akaiii/meadow-golf-checkpoints")
        return 1

    model = load_model(MODEL_PATH)
    print(f"Loaded 5L d={MODEL_DIM} model with vocab={VOCAB_SIZE}")

    # ---- Build two test sequences ----
    T = 32       # short is fine for a leakage test
    SPLIT = 16   # position 0..SPLIT-1 identical, SPLIT..T-1 different

    rng = np.random.default_rng(1337)
    prefix = rng.integers(0, VOCAB_SIZE, size=SPLIT, dtype=np.int32)

    suffix_A = rng.integers(0, VOCAB_SIZE, size=(T - SPLIT), dtype=np.int32)
    # Make suffix_B intentionally different from suffix_A
    suffix_B = rng.integers(0, VOCAB_SIZE, size=(T - SPLIT), dtype=np.int32)
    while np.array_equal(suffix_A, suffix_B):
        suffix_B = rng.integers(0, VOCAB_SIZE, size=(T - SPLIT), dtype=np.int32)

    seq_A = np.concatenate([prefix, suffix_A])
    seq_B = np.concatenate([prefix, suffix_B])

    assert np.array_equal(seq_A[:SPLIT], seq_B[:SPLIT])
    assert not np.array_equal(seq_A, seq_B)

    print(f"\nSequences constructed:")
    print(f"  T={T}, prefix length (identical part)={SPLIT}")
    print(f"  positions 0..{SPLIT-1}: identical")
    print(f"  positions {SPLIT}..{T-1}: different")

    # ---- Forward both with is_causal=True ----
    x_A = mx.array(seq_A.reshape(1, -1))
    x_B = mx.array(seq_B.reshape(1, -1))

    logits_A = model.get_logits(x_A, is_causal=True)
    logits_B = model.get_logits(x_B, is_causal=True)
    mx.eval(logits_A, logits_B)

    lp_A = np.array(logits_A.astype(mx.float32))[0]  # [T, V]
    lp_B = np.array(logits_B.astype(mx.float32))[0]

    print(f"\nLogits shape: {lp_A.shape}")

    # ---- Check identity on positions 0..SPLIT-1 ----
    diff_prefix = np.abs(lp_A[:SPLIT] - lp_B[:SPLIT])
    max_diff_prefix = diff_prefix.max()
    mean_diff_prefix = diff_prefix.mean()
    print(f"\nPrefix positions 0..{SPLIT-1} (should be identical under causal):")
    print(f"  max  |logits_A - logits_B| = {max_diff_prefix:.6e}")
    print(f"  mean |logits_A - logits_B| = {mean_diff_prefix:.6e}")

    # ---- Check that suffix positions DO differ (sanity: the model is not constant) ----
    diff_suffix = np.abs(lp_A[SPLIT:] - lp_B[SPLIT:])
    max_diff_suffix = diff_suffix.max()
    print(f"\nSuffix positions {SPLIT}..{T-1} (should differ, as inputs differ):")
    print(f"  max  |logits_A - logits_B| = {max_diff_suffix:.6e}")

    # ---- Verdict ----
    THRESHOLD_LEAK = 1e-3   # bfloat16 precision is ~3e-3 per operation so we allow 1e-3
    print("\n" + "="*60)
    if max_diff_prefix < THRESHOLD_LEAK and max_diff_suffix > THRESHOLD_LEAK:
        print("  PASS: no future-token leakage detected")
        print(f"  (prefix max diff {max_diff_prefix:.2e} < threshold {THRESHOLD_LEAK:.0e})")
        print(f"  (suffix max diff {max_diff_suffix:.2e} confirms model is not constant)")
    elif max_diff_prefix >= THRESHOLD_LEAK:
        print("  FAIL: possible future-token leakage")
        print(f"  (prefix max diff {max_diff_prefix:.2e} >= threshold {THRESHOLD_LEAK:.0e})")
        print("  Changing tokens at position >= {SPLIT} affects logits at position < {SPLIT}.")
        print("  This indicates the causal mask is NOT correctly applied.")
    elif max_diff_suffix < THRESHOLD_LEAK:
        print("  INCONCLUSIVE: model appears constant across different inputs")
        print("  (this is a model problem, not a leakage test problem)")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
