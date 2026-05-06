"""
Post-training analysis pipeline. Run after training completes.
Does everything needed to prepare a competition submission.

Usage: CUDA_VISIBLE_DEVICES=0 python post_training.py best_model_8B.pt
"""
import os, sys, torch, io, zlib, time
sys.path.insert(0, '.')
from train_gpt import quantize_state_dict_int8, dequantize_state_dict_int8

model_path = sys.argv[1] if len(sys.argv) > 1 else 'best_model_8B.pt'
print(f'=== Post-Training Pipeline for {model_path} ===', flush=True)

# Step 1: Check raw model size
raw_size = os.path.getsize(model_path)
print(f'\n1. Raw model: {raw_size/1e6:.2f} MB', flush=True)

# Step 2: Quantize with competition code
print(f'\n2. Quantizing with competition int8...', flush=True)
state = torch.load(model_path, map_location='cpu')
quant_obj, stats = quantize_state_dict_int8(state)
buf = io.BytesIO()
torch.save(quant_obj, buf)
quant_raw = buf.getvalue()

# Step 3: Compress
print(f'3. Compressing...', flush=True)
compressed_zlib = zlib.compress(quant_raw, level=9)
try:
    import zstandard as zstd
    cctx = zstd.ZstdCompressor(level=22)
    compressed_zstd = cctx.compress(quant_raw)
    best_compressed = min(compressed_zlib, compressed_zstd, key=len)
    method = 'zstd-22' if len(compressed_zstd) < len(compressed_zlib) else 'zlib-9'
except ImportError:
    best_compressed = compressed_zlib
    method = 'zlib-9'

code_size = 60000  # estimate
total = len(best_compressed) + code_size
print(f'   Quantized raw: {len(quant_raw)/1e6:.2f} MB', flush=True)
print(f'   Compressed ({method}): {len(best_compressed)/1e6:.2f} MB', flush=True)
print(f'   Code estimate: {code_size/1e3:.0f} KB', flush=True)
print(f'   Total artifact: {total/1e6:.2f} MB', flush=True)
print(f'   Under 16MB: {total < 16e6}', flush=True)
print(f'   Headroom: {(16e6 - total)/1e6:.2f} MB', flush=True)

# Step 4: Save compressed artifact
artifact_path = model_path.replace('.pt', '.int8.ptz')
with open(artifact_path, 'wb') as f:
    f.write(best_compressed)
print(f'\n4. Artifact saved: {artifact_path} ({len(best_compressed)/1e6:.2f} MB)', flush=True)

# Step 5: Roundtrip validation (quantize -> decompress -> eval)
print(f'\n5. Roundtrip validation...', flush=True)
roundtrip_state = dequantize_state_dict_int8(quant_obj)
roundtrip_path = model_path.replace('.pt', '_roundtrip.pt')
torch.save(roundtrip_state, roundtrip_path)
print(f'   Roundtrip model saved: {roundtrip_path}', flush=True)
print(f'   Run full_eval.py on BOTH original and roundtrip to measure quant gap', flush=True)

print(f'\n=== Pipeline Complete ===', flush=True)
print(f'Next steps:', flush=True)
print(f'  1. python full_eval.py {model_path}          # pre-quant BPB', flush=True)
print(f'  2. python full_eval.py {roundtrip_path}  # post-quant BPB', flush=True)
print(f'  3. Quant gap = post - pre (should be < 0.01 BPB)', flush=True)
