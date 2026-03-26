"""
GPTQ Re-quantization Sweep for GS v7 — runs locally, no 8xH100 needed.
Loads the trained fp32 checkpoint, re-quantizes with different GPTQ params,
measures compressed artifact size and optionally roundtrip BPB.

Usage:
  python3 sweep_gptq_requant_v7.py
"""

import io
import os
import sys
import time
from functools import partial

import torch
import torch.nn.functional as F

# Shim flash_attn_interface for non-Hopper GPUs (DGX Spark uses SDPA)
def _sdpa_shim(q, k, v, causal=True):
    # FA3: (B, T, H, D) -> SDPA: (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    # GQA: expand KV heads to match Q heads
    if k.shape[1] != q.shape[1]:
        group = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)  # back to (B, T, H, D)

_fake_fa = type(sys)("flash_attn_interface")
_fake_fa.flash_attn_func = _sdpa_shim
sys.modules["flash_attn_interface"] = _fake_fa

# Now safe to import the GS script
import importlib.util
GS_SCRIPT = os.environ.get("GS_SCRIPT", "GS/GS_train_gpt_v7_1.1206.py")
CHECKPOINT = os.environ.get("CHECKPOINT", "final_model.pt")
DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")

spec = importlib.util.spec_from_file_location("gs_model", GS_SCRIPT)
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)

args = gs.Hyperparameters()
args.data_path = DATA_PATH
args.train_files = os.path.join(DATA_PATH, "fineweb_train_*.bin")

import zstandard as zstd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"GS Script: {GS_SCRIPT}")
print()

# Load fp32 checkpoint
print("Loading checkpoint...")
state_dict = torch.load(CHECKPOINT, map_location="cpu")
print(f"Params: {sum(t.numel() for t in state_dict.values()):,}")

# Build model for calibration
gs.CastedLinear._qat_enabled = False
model = gs.GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
).to(device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Calibrate Hessians once
print(f"Calibrating Hessians (128 samples from {args.train_files})...")
t0 = time.time()
hessians = gs.gptq_calibrate(model, args.train_files, device, n_samples=128, seq_len=args.train_seq_len)
print(f"Calibrated {len(hessians)} layers in {time.time()-t0:.1f}s")
print()

# Code size (for total artifact calculation)
code_bytes = len(open(GS_SCRIPT).read().encode())
print(f"Code size: {code_bytes:,} bytes")

# Sweep configs
configs = [
    {"name": "baseline",        "block_size": 128, "percdamp": 0.01},
    {"name": "percdamp_005",    "block_size": 128, "percdamp": 0.005},
    {"name": "percdamp_002",    "block_size": 128, "percdamp": 0.002},
    {"name": "percdamp_02",     "block_size": 128, "percdamp": 0.02},
    {"name": "percdamp_05",     "block_size": 128, "percdamp": 0.05},
    {"name": "percdamp_10",     "block_size": 128, "percdamp": 0.10},
    {"name": "block_64",        "block_size": 64,  "percdamp": 0.01},
    {"name": "block_256",       "block_size": 256, "percdamp": 0.01},
    {"name": "block64_pd005",   "block_size": 64,  "percdamp": 0.005},
    {"name": "block64_pd002",   "block_size": 64,  "percdamp": 0.002},
]

cctx = zstd.ZstdCompressor(level=22)

print()
print(f"{'Config':<20} {'Payload':>12} {'Compressed':>12} {'Total':>12} {'Fits 16MB':>10} {'Time':>8}")
print("-" * 78)

results = []

for cfg in configs:
    t0 = time.time()

    # Monkey-patch gptq_quantize_weight to use sweep params
    orig_fn = gs.gptq_quantize_weight
    def patched_gptq(W, H, clip_range=31, block_size=cfg["block_size"], percdamp=cfg["percdamp"]):
        return orig_fn(W, H, clip_range=clip_range, block_size=block_size, percdamp=percdamp)
    gs.gptq_quantize_weight = patched_gptq

    # Re-quantize
    sd_cpu = {k: v.detach().cpu() for k, v in state_dict.items()}
    quant_result, quant_meta = gs.mixed_quantize_int6_gptq(sd_cpu, {"mlp", "attn"}, hessians)

    # Restore original function
    gs.gptq_quantize_weight = orig_fn

    # Serialize + compress
    buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, buf)
    raw = buf.getvalue()
    compressed = cctx.compress(raw)
    total = len(compressed) + code_bytes
    fits = "YES" if total <= 16_000_000 else f"+{total - 16_000_000:,}"
    elapsed = time.time() - t0

    print(f"{cfg['name']:<20} {len(raw):>12,} {len(compressed):>12,} {total:>12,} {fits:>10} {elapsed:>7.1f}s")
    results.append({**cfg, "raw": len(raw), "compressed": len(compressed), "total": total})

print()
print(f"16MB limit = 16,000,000 bytes")
print(f"GS baseline artifact: 15,564,772 bytes")
print()

# Sort by total size
results.sort(key=lambda r: r["total"])
print("Ranked by size (smallest first):")
for i, r in enumerate(results):
    delta = r["total"] - 15_564_772
    sign = "+" if delta >= 0 else ""
    print(f"  {i+1}. {r['name']:<20} {r['total']:>12,}  ({sign}{delta:,} vs GS)")
