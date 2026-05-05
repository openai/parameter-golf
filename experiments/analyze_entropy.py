"""Analyze entropy of quantized int6 weight distribution."""
import torch, numpy as np, math, collections, struct, json, sys

try:
    import brotli
except ImportError:
    brotli = None

path = sys.argv[1] if len(sys.argv) > 1 else "final_model.int6.ptz"
with open(path, "rb") as f:
    blob = f.read()

print(f"File size: {len(blob):,} bytes ({len(blob)/1e6:.2f} MB)")

# Try brotli decompress
dec = None
if brotli:
    try:
        dec = brotli.decompress(blob)
        print(f"Decompressed: {len(dec):,} bytes ({len(dec)/1e6:.2f} MB)")
    except:
        pass

if dec is None:
    dec = blob  # not compressed

# Try torch format first
try:
    import io
    sd = torch.load(io.BytesIO(dec), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "w" in sd:
        result = sd["w"]
        all_q = []
        for name, t in result.items():
            if name.endswith(".q") and t.dtype == torch.int8:
                all_q.append(t.numpy().flatten())
        all_q = np.concatenate(all_q)
        print(f"Format: torch dict with {len(result)} keys")
    else:
        print("Unknown torch format")
        sys.exit(1)
except Exception as e:
    print(f"Torch load failed ({e}), trying manual format...")
    header_len = struct.unpack("<I", dec[:4])[0]
    header = json.loads(dec[4:4+header_len])
    meta = header.pop("__meta__")
    data_start = 4 + header_len
    all_q = []
    for name, info in header.items():
        if name.endswith(".q") and info.get("dtype") == "torch.int8":
            n = info.get("n_values", info["nbytes"])
            if info.get("bitpacked"):
                # Skip bitpacked for now
                continue
            raw = dec[data_start + info["offset"]:data_start + info["offset"] + info["nbytes"]]
            vals = np.frombuffer(raw, dtype=np.int8)
            all_q.append(vals)
    all_q = np.concatenate(all_q)

# Compute histogram and entropy
counts = collections.Counter(all_q.tolist())
total = len(all_q)
entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())

print(f"\nTotal int8 quantized values: {total:,}")
print(f"Unique values used: {len(counts)}")
print(f"Entropy: {entropy:.4f} bits/value")
print(f"Theoretical minimum (entropy coding): {total * entropy / 8:,.0f} bytes ({total * entropy / 8 / 1e6:.2f} MB)")
print(f"At 8 bits/value (current): {total:,} bytes ({total/1e6:.2f} MB)")
print(f"At 6 bits/value (bitpack): {total * 6 // 8:,} bytes ({total * 6 / 8 / 1e6:.2f} MB)")
print(f"Compression ratio vs 8-bit: {entropy/8:.3f}")
print(f"Brotli achieves: {len(blob)/total:.3f} bytes/value")

print(f"\nValue distribution (top 15):")
for val, cnt in sorted(counts.items(), key=lambda x: -x[1])[:15]:
    bar = "#" * int(50 * cnt / counts.most_common(1)[0][1])
    print(f"  val={val:+3d}: {cnt:>8,} ({100*cnt/total:5.1f}%) {bar}")

print(f"\nDistribution stats:")
q_arr = all_q.astype(np.float64)
print(f"  Mean: {q_arr.mean():.4f}")
print(f"  Std: {q_arr.std():.4f}")
print(f"  |val|<=5: {np.sum(np.abs(all_q) <= 5)/total*100:.1f}%")
print(f"  |val|<=10: {np.sum(np.abs(all_q) <= 10)/total*100:.1f}%")
print(f"  |val|<=15: {np.sum(np.abs(all_q) <= 15)/total*100:.1f}%")
print(f"  |val|<=20: {np.sum(np.abs(all_q) <= 20)/total*100:.1f}%")
