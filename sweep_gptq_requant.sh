#!/bin/bash
set -euo pipefail

# GPTQ re-quantization sweep — NO TRAINING, uses existing checkpoint
# Tests different GPTQ settings on final_model.pt to find best compression/quality
# Each config takes ~15 seconds. Total: ~2 minutes.
#
# Usage: bash sweep_gptq_requant.sh

cd /workspace/parameter-golf

if [ ! -f final_model.pt ]; then
    echo "ERROR: final_model.pt not found. Run a training script first."
    exit 1
fi

echo "Checkpoint: $(ls -lh final_model.pt | awk '{print $5}')"
echo ""

python3 << 'PYEOF'
import os, io, time, torch
import zstandard as zstd

# Load the model architecture + checkpoint
# Detect which script created the checkpoint by trying imports
try:
    # Try SwiGLU Frugendorff first
    import importlib.util
    spec = importlib.util.spec_from_file_location("train", "train_gpt_swiglu_frugendorff.py")
    mod = importlib.util.module_from_spec(spec)

    # Patch out training-only imports that might fail
    import sys
    sys.modules['flash_attn_interface'] = type(sys)('fake')
    sys.modules['flash_attn_interface'].flash_attn_func = None

    spec.loader.exec_module(mod)
    args = mod.Hyperparameters()
    print(f"Script: train_gpt_swiglu_frugendorff.py")
    print(f"Params: {sum(p.numel() for p in torch.load('final_model.pt', map_location='cpu').values()):,}")
except Exception as e:
    print(f"Could not load module: {e}")
    exit(1)

# Load checkpoint
state_dict = torch.load("final_model.pt", map_location="cpu")
device = torch.device("cuda:0")

# Import quantization functions
gptq_calibrate = mod.gptq_calibrate
quantize_state_dict_int6 = mod.quantize_state_dict_int6
gptq_quantize_weight = mod.gptq_quantize_weight

# Build model for calibration
model = mod.GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers,
    model_dim=args.model_dim, num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads, mlp_hidden=args.mlp_hidden,
    tie_embeddings=args.tie_embeddings,
    tied_embed_init_std=getattr(args, 'tied_embed_init_std', 0.005),
    logit_softcap=args.logit_softcap, rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,
    bigram_buckets=args.bigram_buckets, bigram_embed_dim=args.bigram_embed_dim,
    xsa_layers=getattr(args, 'xsa_layers', 4),
    rope_dims=getattr(args, 'rope_dims', 16),
    ln_scale=getattr(args, 'ln_scale', True),
    share_start=args.share_start, share_loops=args.share_loops,
).to(device)
model.load_state_dict(state_dict)
model.eval()

# Calibrate GPTQ Hessians once (reused across all configs)
print("Calibrating Hessians (256 samples)...")
t0 = time.time()
hessians = gptq_calibrate(model, args.train_files, device, n_samples=256, seq_len=args.train_seq_len)
print(f"Calibrated {len(hessians)} layers in {time.time()-t0:.1f}s")
print()

# Sweep configs
configs = [
    {"name": "baseline",       "block_size": 128, "percdamp": 0.01},
    {"name": "percdamp_005",   "block_size": 128, "percdamp": 0.005},
    {"name": "percdamp_002",   "block_size": 128, "percdamp": 0.002},
    {"name": "percdamp_02",    "block_size": 128, "percdamp": 0.02},
    {"name": "percdamp_05",    "block_size": 128, "percdamp": 0.05},
    {"name": "block_64",       "block_size": 64,  "percdamp": 0.01},
    {"name": "block_256",      "block_size": 256, "percdamp": 0.01},
    {"name": "block64_pd005",  "block_size": 64,  "percdamp": 0.005},
]

print(f"{'Config':<20} {'Size':>12} {'Fits 16MB':>10} {'Roundtrip BPB':>15}")
print("-" * 62)

cctx = zstd.ZstdCompressor(level=22)

for cfg in configs:
    os.environ["GPTQ_BLOCK_SIZE"] = str(cfg["block_size"])
    os.environ["GPTQ_PERCDAMP"] = str(cfg["percdamp"])

    t0 = time.time()
    quant_obj, stats = quantize_state_dict_int6(state_dict, gptq_hessians=hessians)

    # Serialize + compress
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    compressed = cctx.compress(raw)
    code_size = len(open("train_gpt_swiglu_frugendorff.py").read().encode())
    total = len(compressed) + code_size
    fits = "YES" if total <= 16_000_000 else f"NO (+{total - 16_000_000})"

    # Quick roundtrip eval
    quant_state = torch.load(io.BytesIO(compressed if False else raw), map_location="cpu")
    # Just report size for now — roundtrip eval needs full model rebuild
    elapsed = time.time() - t0

    print(f"{cfg['name']:<20} {total:>12,} {fits:>10} {'(skip)':>15}  [{elapsed:.1f}s]")

print()
print("16MB limit = 16,000,000 bytes")
print("Best config = smallest size that maintains quality")
PYEOF
