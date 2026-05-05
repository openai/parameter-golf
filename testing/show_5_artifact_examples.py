"""Walk through 5 specific artifact examples from a saved per-byte compare dump.

For each: shows nn_uniform, nn_proper, ppm log-probs and the resulting mix
charges under both spec 055 (uniform+PPM) and proper+PPM, plus context bytes.
"""
import argparse
import numpy as np
import math
from pathlib import Path

DEFAULT_CANDIDATES = [
    Path("eval/data/2026-04-29_047B_200k_per_byte_compare.npz"),
    Path("/tmp/per_byte_compare.npz"),
]


def resolve_input(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    for path in DEFAULT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No per-byte compare dump found. Pass --input or place one of these files:\n"
        + "\n".join(f"  - {p}" for p in DEFAULT_CANDIDATES)
    )


parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to per-byte compare .npz dump")
args = parser.parse_args()

input_path = resolve_input(args.input)
d = np.load(input_path)
bytes_arr = d["bytes"]
nn_proper = d["nn_proper_nats"].astype(np.float64)
nn_uniform = d["nn_uniform_nats"].astype(np.float64)
ppm = d["ppm_nats"].astype(np.float64)
gate_hi = d["gate_hi"]
n = len(bytes_arr)

mask = gate_hi == 1
savings = nn_uniform - nn_proper

def show(idx, label):
    idx = int(idx)
    b = int(bytes_arr[idx])
    ch = chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
    nu = nn_uniform[idx]; np_ = nn_proper[idx]; pp = ppm[idx]
    p_uni = math.exp(-nu); p_ppm = math.exp(-pp); p_pro = math.exp(-np_)
    mix_uni = -math.log(0.05*p_uni + 0.95*p_ppm)
    mix_pro = -math.log(0.05*p_pro + 0.95*p_ppm)
    pre = bytes(bytes_arr[max(0, idx-12):idx+1]).decode("utf-8", errors="replace")
    nxt = bytes(bytes_arr[idx+1:min(n, idx+5)]).decode("utf-8", errors="replace")
    print(f"\n[{label}] idx={idx} realized byte={ch!r}")
    print(f"  context (last 13 bytes incl. realized): {pre!r}")
    print(f"  next 4 bytes:                            {nxt!r}")
    print(f"  ----")
    print(f"  nn_uniform-spread  = {nu:6.3f} nats   (p ≈ {p_uni:.4g})")
    print(f"  nn_proper-margin   = {np_:6.3f} nats   (p ≈ {p_pro:.4g})")
    print(f"  ppm                = {pp:6.3f} nats   (p ≈ {p_ppm:.4g})")
    print(f"  spec055 mix charge = {mix_uni:.3f} nats   →  spec055 \"saves\" {nu - mix_uni:+.3f} vs uniform")
    print(f"  proper+ppm charge  = {mix_pro:.3f} nats   →  proper+PPM gains {np_ - mix_pro:+.3f} vs proper-alone")
    print(f"  TRUTH (proper) - SPEC055_MIX: {np_ - mix_uni:+.3f} nats  (positive = spec 055 even charges more than truth!)")

ranked = np.argsort(-(savings * mask.astype(np.float64)))

print("=" * 100)
print("TOP 5 ARTIFACT BYTES — gate_hi=1, ranked by (nn_uniform − nn_proper)")
print("These are bytes where uniform-spread *fakes* a high NN cost that PPM then 'rescues'.")
print(f"  input: {input_path}")
print(f"  Total bytes: {n:,}   gate_hi rate: {gate_hi.mean():.4f}")
print("=" * 100)
for k in range(5):
    show(ranked[k], f"#{k+1}")

# bonus: aggregate impact
gate_high_mask = gate_hi == 1
spec055_savings = (nn_uniform - np.minimum(nn_uniform, ppm))  # rough proxy under λ_lo=0.05
total_uniform = nn_uniform.sum()
total_proper = nn_proper.sum()
print(f"\n{'='*100}")
print(f"AGGREGATE (across {n:,} bytes):")
print(f"  Sum nn_uniform NLL:   {total_uniform:11.1f} nats  → bpb = {total_uniform/n/math.log(2):.5f}")
print(f"  Sum nn_proper  NLL:   {total_proper:11.1f} nats  → bpb = {total_proper/n/math.log(2):.5f}")
print(f"  (these MUST be equal by bit-conservation, they total to the same — just redistributed)")
print(f"  diff = {total_uniform - total_proper:+.6f} nats")
print(f"  fraction of bytes with gate_hi=1: {gate_hi.mean():.4f}")
print(f"  on those bytes: avg(uniform-proper) = {savings[mask].mean():.3f} nats per byte")
