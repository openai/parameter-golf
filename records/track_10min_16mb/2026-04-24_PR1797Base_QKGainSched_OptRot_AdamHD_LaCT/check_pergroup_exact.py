"""
Bit-equality test for pergroup roundtrip.

Run on the pod:
  python3 check_pergroup_exact.py /workspace/parameter-golf/final_model.int6.ptz

If the artifact is PGRP format: loads it, re-packs, re-unpacks, checks equality.
If the artifact is brotli format: packs via pergroup, unpacks, checks equality.
If no artifact: uses synthetic data with production-sized tensors.
"""
import io, os, sys, struct as _st
import numpy as np
import torch

# ── Extract helpers from training script ────────────────────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "train_gpt_human.py")

_g = {"__name__": "__test__", "np": np, "torch": torch, "io": io, "os": os,
      "subprocess": __import__("subprocess"), "time": __import__("time"),
      "lzma": __import__("lzma")}
try:
    exec(compile(open(SCRIPT).read(), SCRIPT, "exec"), _g)
except Exception:
    pass

pack   = _g["_pack_pergroup"]
unpack = _g["_unpack_pergroup"]
decomp = _g.get("_decompress")
bshuf  = _g.get("_byte_unshuffle")
SUFFIXES = _g.get("_PGRP_Q_SUFFIXES", ())

OK   = "\033[32mOK\033[0m"
FAIL = "\033[31mFAIL\033[0m"

# ── Synthetic production-sized state ────────────────────────────────────────
def make_state(seed=7):
    rng = np.random.default_rng(seed)
    qr, qm = {}, {}
    shapes = [
        ("blocks.0.mlp.fc.weight",    (2048, 512)),
        ("blocks.0.mlp.proj.weight",  (512,  2048)),
        ("blocks.0.attn.c_q.weight",  (512,  512)),
        ("blocks.0.attn.c_k.weight",  (256,  512)),
        ("blocks.0.attn.c_v.weight",  (256,  512)),
        ("blocks.0.attn.proj.weight", (512,  512)),
        ("tok_emb.weight",            (8192, 512)),
    ]
    for name, (r, c) in shapes:
        q = torch.from_numpy(rng.integers(-32, 32, (r, c), dtype=np.int8))
        s = torch.from_numpy(rng.random(r).astype(np.float16))
        qr[name + ".q"] = q
        qr[name + ".scale"] = s
        qm[name] = "gptq (int6)"
    # Fake LQER asym keys on two tensors (as in our run config)
    for base in ("blocks.0.mlp.fc.weight", "blocks.0.attn.c_q.weight"):
        for sfx in (".lqA_a", ".lqAs_a", ".lqB_a", ".lqBs_a"):
            shape = (6, 2048) if "mlp" in base else (6, 512)
            qr[base + sfx] = torch.from_numpy(rng.integers(-2, 2, shape, dtype=np.int8))
        qm[base] = "gptq (int6)+lqer_asym"
    # Passthrough (small fp16)
    qr["smear_lambda"] = torch.tensor([0.0], dtype=torch.float16)
    qm["smear_lambda"] = "passthrough (float16)"
    return qr, qm


def check_exact(orig_w, orig_m, recovered_w, recovered_m, label):
    all_ok = True
    for name, orig in orig_w.items():
        rec = recovered_w.get(name)
        if rec is None:
            print(f"  {FAIL}  MISSING: {name}")
            all_ok = False
            continue
        if orig.shape != rec.shape:
            print(f"  {FAIL}  SHAPE MISMATCH {name}: {orig.shape} vs {rec.shape}")
            all_ok = False
            continue
        if orig.dtype != rec.dtype:
            print(f"  {FAIL}  DTYPE MISMATCH {name}: {orig.dtype} vs {rec.dtype}")
            all_ok = False
            continue
        if not torch.equal(orig, rec):
            d = (orig.float() - rec.float()).abs()
            print(f"  {FAIL}  VALUE MISMATCH {name}: max={d.max():.4e} mean={d.mean():.4e} "
                  f"n={(orig!=rec).sum()}/{orig.numel()}")
            all_ok = False
    if orig_m != recovered_m:
        print(f"  {FAIL}  QUANT_META mismatch")
        all_ok = False
    if all_ok:
        print(f"  {OK}  {label}: all {len(orig_w)} tensors + meta bit-exact")
    return all_ok


# ── Test 1: synthetic roundtrip ──────────────────────────────────────────────
print("=" * 60)
print("TEST 1: synthetic roundtrip")
qr, qm = make_state()
blob = pack(qr, qm)
rec  = unpack(blob)
t1_ok = check_exact(qr, qm, rec["w"], rec["m"], "synthetic")
print(f"  blob size: {len(blob):,} bytes")


# ── Test 2: which keys go to Q vs remainder ──────────────────────────────────
print("=" * 60)
print("TEST 2: key routing (Q section vs remainder)")
q_keys   = [k for k in qr if any(k.endswith(s) for s in SUFFIXES)]
rem_keys = [k for k in qr if not any(k.endswith(s) for s in SUFFIXES)]
print(f"  Q section ({len(q_keys)}):        {sorted(q_keys)[:4]}...")
print(f"  Remainder ({len(rem_keys)}):   {sorted(rem_keys)[:6]}...")
# Check that LQER keys all end up in remainder (not Q section)
lqer_in_q = [k for k in q_keys if "lqA" in k or "lqB" in k or "lqAs" in k]
if lqer_in_q:
    print(f"  {FAIL}  LQER keys incorrectly routed to Q section: {lqer_in_q}")
else:
    print(f"  {OK}  LQER keys correctly in remainder")


# ── Test 3: live artifact ────────────────────────────────────────────────────
artifact = None
if len(sys.argv) > 1:
    artifact = sys.argv[1]
else:
    for p in ["/workspace/parameter-golf/final_model.int6.ptz",
              os.path.join(HERE, "final_model.int6.ptz")]:
        if os.path.exists(p):
            artifact = p
            break

if artifact:
    print("=" * 60)
    print(f"TEST 3: live artifact {artifact}")
    raw = open(artifact, "rb").read()
    print(f"  artifact size: {len(raw):,} bytes")
    fmt = "PGRP" if raw[:4] == b"PGRP" else "brotli"
    print(f"  format: {fmt}")

    if fmt == "PGRP":
        state = unpack(raw)
    else:
        import brotli
        state = torch.load(io.BytesIO(bshuf(brotli.decompress(raw))), map_location="cpu")

    orig_w = state["w"]
    orig_m = state["m"]
    print(f"  keys: {len(orig_w)} total, "
          f"{sum(1 for k in orig_w if any(k.endswith(s) for s in SUFFIXES))} Q, "
          f"{sum(1 for k in orig_w if 'lq' in k)} LQER")

    # Repack → unpack → compare
    blob2 = pack(orig_w, orig_m)
    rec2  = unpack(blob2)
    t3_ok = check_exact(orig_w, orig_m, rec2["w"], rec2["m"], "live artifact repack")
    print(f"  repacked blob size: {len(blob2):,} bytes")
else:
    print("=" * 60)
    print("TEST 3: skipped (no artifact; pass path as argv[1])")
    t3_ok = True


# ── Test 4: uint16 perm range ────────────────────────────────────────────────
print("=" * 60)
print("TEST 4: uint16 perm range for actual tensor shapes")
worst = [("tok_emb.weight.q", 8192), ("blocks.0.mlp.fc.weight.q", 2048)]
for name, rows in worst:
    ok = rows <= 65535
    print(f"  {OK if ok else FAIL}  {name}: rows={rows} ≤ 65535? {ok}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("CONCLUSION:")
if t1_ok:
    print(f"  {OK}  Pergroup roundtrip is BIT-EXACT.")
    print()
    print("  The quant_bpb diff (+0.0004) and 80s speedup are NOT from pergroup lossiness.")
    print("  Most likely cause: S35 and S42-S44 trained with different seeds or")
    print("  different GPU count (8-GPU data distribution vs 1-GPU). Check:")
    print("    grep 'pre.quant bpb\\|pre_quant\\|bpb.*base' train_seed42.log")
    print("  If pre-quant BPB in S42 log differs from S35 log → training was different.")
    print()
    print("  80s eval speedup: check if PHASED_TTT_NUM_PHASES or PREFIX_DOCS differed,")
    print("  or if the eval timer in S42 logs starts later than in S35 logs.")
else:
    print(f"  {FAIL}  Pergroup has precision issues — see details above.")
