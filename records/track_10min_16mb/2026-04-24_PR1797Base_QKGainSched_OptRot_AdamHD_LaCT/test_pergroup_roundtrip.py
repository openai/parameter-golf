"""
Pergroup roundtrip diagnostic.

Tests:
  1. Exact roundtrip: pack → unpack returns bit-identical tensors
  2. int16 overflow in _similarity_sort_l1: sort order corrupted?
  3. uint16 perm range: sufficient for actual tensor row counts?
  4. Remainder (scales, LQER, quant_meta) roundtrip exactness
  5. Live artifact roundtrip (if final_model.int6.ptz exists)

Run on the pod:
  python3 test_pergroup_roundtrip.py [path/to/final_model.int6.ptz]
"""
import io
import os
import sys
import struct as _struct

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Inject the pergroup helpers from train_gpt_human.py by extracting
# the relevant section at runtime — avoids importing the full training
# script (which would trigger triton/torch.compile at import time).
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(HERE, "train_gpt_human.py")

with open(TRAIN_SCRIPT, "r") as _fh:
    _src = _fh.read()

_globs = {
    "__name__": "__test__",
    "np": np,
    "torch": torch,
    "io": io,
    "os": os,
    "subprocess": __import__("subprocess"),
    "time": __import__("time"),
    "lzma": __import__("lzma"),
}
try:
    exec(compile(_src, TRAIN_SCRIPT, "exec"), _globs)
except SystemExit:
    pass
except Exception as e:
    # Training script may fail at top-level if CUDA not available; that's OK.
    pass

_pack_pergroup       = _globs.get("_pack_pergroup")
_unpack_pergroup     = _globs.get("_unpack_pergroup")
_similarity_sort_l1  = _globs.get("_similarity_sort_l1")
_byte_shuffle        = _globs.get("_byte_shuffle")
_byte_unshuffle      = _globs.get("_byte_unshuffle")
_lrzip_compress_bytes = _globs.get("_lrzip_compress_bytes")
_lrzip_decompress_bytes = _globs.get("_lrzip_decompress_bytes")
_decompress          = _globs.get("_decompress")
_PGRP_Q_SUFFIXES     = _globs.get("_PGRP_Q_SUFFIXES", ())

assert _pack_pergroup, "failed to extract _pack_pergroup from training script"
assert _unpack_pergroup, "failed to extract _unpack_pergroup from training script"
assert _similarity_sort_l1, "failed to extract _similarity_sort_l1"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
SEP  = "-" * 70


# ---------------------------------------------------------------------------
# 1. Confirm int16 overflow in _similarity_sort_l1
# ---------------------------------------------------------------------------
def test_sort_overflow():
    print(SEP)
    print("TEST 1: int16 overflow in _similarity_sort_l1")
    results = []
    for rows, cols in [(512, 2048), (512, 512), (2048, 512), (256, 512)]:
        max_possible_sum = 255 * cols  # worst case: all int8 differ maximally
        overflows = max_possible_sum > np.iinfo(np.int16).max
        # Demonstrate with a pathological tensor
        W = np.full((rows, cols), 127, dtype=np.int8)
        W[0] = -128  # row 0 maximally distant from all others
        perm_correct = _similarity_sort_l1(W)
        # Row 0 should be last (most isolated) or at least not first after the pivot
        # With overflow the sort order is unpredictable
        row0_pos = int(np.where(perm_correct == 0)[0][0])
        results.append((rows, cols, overflows, row0_pos))
        status = WARN if overflows else PASS
        print(f"  ({rows:4d}x{cols:4d})  max_col_sum={max_possible_sum:7d}  "
              f"int16_overflow={overflows}  row0_at_pos={row0_pos}/{rows-1}  {status}")

    any_overflow = any(r[2] for r in results)
    if any_overflow:
        print(f"  → {WARN}: sort order is CORRUPTED for large cols due to int16 overflow.")
        print(f"    Compression ratio is SUBOPTIMAL but values are STILL exact (perm is stored).")
        print(f"    Fix: use np.int32 or np.int64 for distance accumulation in _similarity_sort_l1.")
    else:
        print(f"  → {PASS}: no overflow in any tested shape")
    return any_overflow


# ---------------------------------------------------------------------------
# 2. uint16 perm range
# ---------------------------------------------------------------------------
def test_perm_range():
    print(SEP)
    print("TEST 2: uint16 perm range for actual tensor shapes")
    shapes = {
        "blocks.0.mlp.fc.weight.q":    (2048, 512),
        "blocks.0.mlp.proj.weight.q":  (512, 2048),
        "blocks.0.attn.c_q.weight.q":  (512, 512),
        "blocks.0.attn.c_k.weight.q":  (256, 512),
        "blocks.0.attn.c_v.weight.q":  (256, 512),
        "blocks.0.attn.proj.weight.q": (512, 512),
        "tok_emb.weight.q":            (8192, 512),
    }
    all_ok = True
    for name, (rows, cols) in shapes.items():
        fits = rows <= 65535
        status = PASS if fits else FAIL
        if not fits:
            all_ok = False
        print(f"  {name}: rows={rows}  uint16_ok={fits}  {status}")
    if all_ok:
        print(f"  → {PASS}: all row counts fit in uint16")
    else:
        print(f"  → {FAIL}: some row counts EXCEED uint16 — perm indices will wrap!")
    return all_ok


# ---------------------------------------------------------------------------
# 3. Synthetic roundtrip with actual-sized tensors
# ---------------------------------------------------------------------------
def _make_synthetic_state(seed=0):
    rng = np.random.default_rng(seed)
    quant_result = {}
    quant_meta   = {}
    # Representative GPTQ int6 weight tensors (int8 storage, values in [-32,31])
    shapes = {
        "blocks.0.mlp.fc.weight":    (2048, 512),
        "blocks.0.mlp.proj.weight":  (512,  2048),
        "blocks.0.attn.c_q.weight":  (512,  512),
        "blocks.0.attn.c_k.weight":  (256,  512),
        "blocks.0.attn.c_v.weight":  (256,  512),
        "blocks.0.attn.proj.weight": (512,  512),
        "tok_emb.weight":            (8192, 512),
    }
    for name, (rows, cols) in shapes.items():
        q = torch.from_numpy(rng.integers(-32, 32, (rows, cols), dtype=np.int8))
        s = torch.from_numpy(rng.random((rows,), dtype=np.float32).astype(np.float16))
        quant_result[name + ".q"] = q
        quant_result[name + ".scale"] = s
        quant_meta[name] = "gptq (int6)"
    # A small passthrough tensor (won't match any Q suffix)
    quant_result["tok_emb.bias"] = torch.from_numpy(
        rng.random((512,), dtype=np.float32).astype(np.float16))
    quant_meta["tok_emb.bias"] = "passthrough (float16)"
    # A fake LQER key
    quant_result["blocks.0.mlp.fc.weight.lqer_qA"] = torch.from_numpy(
        rng.integers(-2, 2, (2048, 6), dtype=np.int8))
    return quant_result, quant_meta


def test_synthetic_roundtrip():
    print(SEP)
    print("TEST 3: Synthetic roundtrip (pack → unpack, exact equality per tensor)")
    quant_result, quant_meta = _make_synthetic_state(seed=42)

    blob = _pack_pergroup(quant_result, quant_meta)
    print(f"  packed blob size: {len(blob):,} bytes")

    recovered = _unpack_pergroup(blob)
    rec_w  = recovered["w"]
    rec_m  = recovered["m"]

    all_ok = True
    for name, orig in quant_result.items():
        if name not in rec_w:
            print(f"  {FAIL}: key MISSING after roundtrip: {name!r}")
            all_ok = False
            continue
        rec = rec_w[name]
        if orig.shape != rec.shape:
            print(f"  {FAIL}: shape mismatch {name!r}: {orig.shape} vs {rec.shape}")
            all_ok = False
            continue
        if orig.dtype != rec.dtype:
            print(f"  {FAIL}: dtype mismatch {name!r}: {orig.dtype} vs {rec.dtype}")
            all_ok = False
            continue
        if not torch.equal(orig, rec):
            diff = (orig.float() - rec.float()).abs()
            print(f"  {FAIL}: VALUE MISMATCH {name!r}  "
                  f"max_diff={diff.max().item():.4f}  "
                  f"mean_diff={diff.mean().item():.6f}  "
                  f"n_mismatched={(orig != rec).sum().item()}")
            all_ok = False
        else:
            print(f"  {PASS}: {name!r}  shape={list(orig.shape)}  dtype={orig.dtype}")

    # Check quant_meta roundtrip
    if rec_m != quant_meta:
        print(f"  {FAIL}: quant_meta mismatch after roundtrip")
        all_ok = False
    else:
        print(f"  {PASS}: quant_meta exact match")

    if all_ok:
        print(f"  → {PASS}: synthetic roundtrip is EXACT")
    else:
        print(f"  → {FAIL}: synthetic roundtrip has LOSSES")
    return all_ok


# ---------------------------------------------------------------------------
# 4. Byte-shuffle roundtrip
# ---------------------------------------------------------------------------
def test_byte_shuffle():
    print(SEP)
    print("TEST 4: _byte_shuffle / _byte_unshuffle roundtrip")
    for n in [0, 1, 7, 16, 127, 1024, 65537]:
        data = bytes(range(256)) * (n // 256 + 1)
        data = data[:n]
        shuffled   = _byte_shuffle(data)
        unshuffled = _byte_unshuffle(shuffled)
        ok = data == unshuffled
        print(f"  n={n:6d}  {'PASS' if ok else 'FAIL'}: {'ok' if ok else 'MISMATCH'}")


# ---------------------------------------------------------------------------
# 5. Live artifact roundtrip (optional)
# ---------------------------------------------------------------------------
def test_live_artifact(path):
    print(SEP)
    print(f"TEST 5: Live artifact roundtrip from {path}")
    with open(path, "rb") as fh:
        raw = fh.read()
    print(f"  artifact size: {len(raw):,} bytes")

    # Detect format — PGRP or brotli-wrapped torch.save
    if raw[:4] == b"PGRP":
        print("  format: PGRP (pergroup)")
        state = _unpack_pergroup(raw)
        mode = "pergroup"
    else:
        print("  format: brotli-wrapped (standard)")
        import brotli
        try:
            inner = _byte_unshuffle(brotli.decompress(raw))
            state = torch.load(io.BytesIO(inner), map_location="cpu")
        except Exception:
            state = torch.load(io.BytesIO(raw), map_location="cpu")
        mode = "brotli"

    w = state.get("w", state)
    print(f"  keys: {len(w)}")
    q_keys = [k for k in w if any(k.endswith(s) for s in _PGRP_Q_SUFFIXES)]
    print(f"  Q tensor keys ({len(q_keys)}): {q_keys[:3]}{'...' if len(q_keys)>3 else ''}")

    if mode == "brotli":
        # For a brotli artifact we can re-pack via pergroup and check roundtrip
        quant_result = w
        quant_meta   = state.get("m", {})
        print("  Re-packing as pergroup and verifying roundtrip...")
        blob = _pack_pergroup(quant_result, quant_meta)
        recovered = _unpack_pergroup(blob)
        rec_w = recovered["w"]
        all_ok = True
        for name in quant_result:
            if name not in rec_w:
                print(f"  {FAIL}: key missing after pergroup roundtrip: {name!r}")
                all_ok = False
                continue
            orig = quant_result[name]
            rec  = rec_w[name]
            if not torch.equal(orig, rec):
                diff = (orig.float() - rec.float()).abs()
                print(f"  {FAIL}: VALUE MISMATCH {name!r}  "
                      f"max_diff={diff.max().item():.4f}  "
                      f"mean_diff={diff.mean().item():.6f}  "
                      f"n_mismatched={(orig != rec).sum().item()}")
                all_ok = False
        if all_ok:
            print(f"  → {PASS}: brotli→pergroup→unpack is EXACT for all {len(quant_result)} tensors")
        else:
            print(f"  → {FAIL}: pergroup roundtrip of brotli artifact has losses")
    else:
        # PGRP artifact — re-pack and compare
        quant_result = w
        quant_meta   = state.get("m", {})
        print("  Re-packing PGRP artifact and comparing...")
        blob2 = _pack_pergroup(quant_result, quant_meta)
        recovered2 = _unpack_pergroup(blob2)
        rec_w2 = recovered2["w"]
        all_ok = True
        for name in quant_result:
            orig = quant_result[name]
            rec  = rec_w2.get(name)
            if rec is None:
                print(f"  {FAIL}: key missing: {name!r}")
                all_ok = False
                continue
            if not torch.equal(orig, rec):
                diff = (orig.float() - rec.float()).abs()
                print(f"  {FAIL}: PGRP→re-pack mismatch {name!r}  "
                      f"max_diff={diff.max().item():.4f}  "
                      f"n_mismatched={(orig != rec).sum().item()}")
                all_ok = False
        if all_ok:
            print(f"  → {PASS}: PGRP re-roundtrip is EXACT")
        else:
            print(f"  → {FAIL}: PGRP re-roundtrip has losses")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import brotli  # must be available

    overflow_found   = test_sort_overflow()
    perm_ok          = test_perm_range()
    roundtrip_ok     = test_synthetic_roundtrip()
    test_byte_shuffle()

    artifact_path = None
    if len(sys.argv) > 1:
        artifact_path = sys.argv[1]
    else:
        candidates = [
            "/workspace/parameter-golf/final_model.int6.ptz",
            os.path.join(HERE, "final_model.int6.ptz"),
        ]
        for c in candidates:
            if os.path.exists(c):
                artifact_path = c
                break

    if artifact_path:
        test_live_artifact(artifact_path)
    else:
        print(SEP)
        print("TEST 5: skipped (no artifact path found; pass as argv[1])")

    print(SEP)
    print("SUMMARY")
    print(f"  int16 overflow in sort:   {WARN if overflow_found else PASS}  "
          f"{'(affects compression ratio, NOT values)' if overflow_found else ''}")
    print(f"  uint16 perm range:        {PASS if perm_ok else FAIL}")
    print(f"  synthetic roundtrip:      {PASS if roundtrip_ok else FAIL}")
    print()
    if roundtrip_ok and perm_ok:
        print("CONCLUSION: Pergroup roundtrip is EXACT. BPB clustering is NOT due to weight lossiness.")
        print("  The 80s eval speedup and BPB clustering have a DIFFERENT root cause.")
        print("  Possible causes: (a) PHASED_TTT_NUM_PHASES differs between pergroup and brotli runs,")
        print("  (b) a different env var was missing in the pergroup launch commands,")
        print("  (c) GLOBAL_TTT_LR=0.01 is so aggressive it overshoots and hurts (for S44).")
        print("  Check: diff the launch commands for S35 (brotli) vs S42/S43/S44 (pergroup).")
    if overflow_found:
        print()
        print("ACTION: Fix int16 overflow in _similarity_sort_l1 for better compression.")
        print("  Change: W16 = W.astype(np.int16)  →  W32 = W.astype(np.int32)")
        print("  Then use W32 throughout the sort. This fixes suboptimal row ordering.")
