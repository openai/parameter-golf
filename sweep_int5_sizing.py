"""
Int5 Sizing Sweep — how many params can we fit at int5?

Tests int5 (clip_range=15) vs int6 (clip_range=31) and mixed int5/int6
on our existing 27M checkpoint, then extrapolates to larger architectures.

This tells us exactly how much headroom int5 gives before we burn pod money.

Usage:
  .venv/bin/python3 sweep_int5_sizing.py
"""

import io
import os
import sys
import time

import torch
import torch.nn.functional as F

# Shim flash_attn for DGX Spark
def _sdpa_shim(q, k, v, causal=True):
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if k.shape[1] != q.shape[1]:
        group = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)

_fake_fa = type(sys)("flash_attn_interface")
_fake_fa.flash_attn_func = _sdpa_shim
sys.modules["flash_attn_interface"] = _fake_fa

import importlib.util
GS_SCRIPT = os.environ.get("GS_SCRIPT", "GS/GS_train_gpt_v7_1.1206.py")
CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/gs_v7_final_model.pt")
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
print()

# Load checkpoint
print("Loading checkpoint...")
state_dict = torch.load(CHECKPOINT, map_location="cpu")
n_params = sum(t.numel() for t in state_dict.values())
print(f"Params: {n_params:,}")

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

# Calibrate
print(f"Calibrating Hessians (128 samples)...")
t0 = time.time()
hessians = gs.gptq_calibrate(model, args.train_files, device, n_samples=128, seq_len=args.train_seq_len)
print(f"Calibrated {len(hessians)} layers in {time.time()-t0:.1f}s\n")

code_bytes = len(open(GS_SCRIPT).read().encode())
cctx = zstd.ZstdCompressor(level=22)

# Custom quantization function that supports different clip ranges per category
def quantize_mixed(state_dict, hessians, attn_clip, mlp_clip, block_size, percdamp):
    """Quantize with potentially different clip ranges for attn vs mlp."""
    orig_fn = gs.gptq_quantize_weight
    result = {}
    meta = {}
    gptq_count = naive_count = 0

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = gs._classify_param(name)

        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in gs.CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue

        # Pick clip range based on category
        if cat == "attn":
            clip = attn_clip
        elif cat == "mlp":
            clip = mlp_clip
        else:
            clip = max(attn_clip, mlp_clip)  # embeddings etc use the larger range

        if cat in {"mlp", "attn"} and t.ndim == 2:
            module_name = name.rsplit(".weight", 1)[0] if name.endswith(".weight") else name
            H = hessians.get(module_name)
            if H is not None and H.shape[0] == t.shape[1]:
                q, s = orig_fn(t, H.cpu(), clip_range=clip, block_size=block_size, percdamp=percdamp)
                gptq_count += 1
            else:
                q, s = gs.quantize_int6_per_row(t, clip_range=clip)
                naive_count += 1
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": f"int_clip{clip}"}
        elif cat in {"mlp", "attn"} and t.ndim >= 1:
            q, s = gs.quantize_int6_per_row(t, clip_range=clip)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": f"int_clip{clip}"}
            naive_count += 1
        else:
            q, s = gs.quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}

    return result, meta, gptq_count, naive_count

# ── CONFIGS ──────────────────────────────────────────────────────────────────
configs = [
    # Current baseline: int6 everywhere
    {"name": "int6_baseline",     "attn_clip": 31, "mlp_clip": 31, "block_size": 64, "percdamp": 0.002},

    # Pure int5: everything at clip_range=15
    {"name": "int5_pure",         "attn_clip": 15, "mlp_clip": 15, "block_size": 128, "percdamp": 0.01},
    {"name": "int5_b64",          "attn_clip": 15, "mlp_clip": 15, "block_size": 64,  "percdamp": 0.01},
    {"name": "int5_b64_pd002",    "attn_clip": 15, "mlp_clip": 15, "block_size": 64,  "percdamp": 0.002},
    {"name": "int5_b128_pd002",   "attn_clip": 15, "mlp_clip": 15, "block_size": 128, "percdamp": 0.002},

    # Mixed: int5 for MLP (bulk of params), int6 for attention (sensitive)
    {"name": "mixed_mlp5_attn6",       "attn_clip": 31, "mlp_clip": 15, "block_size": 64,  "percdamp": 0.002},
    {"name": "mixed_mlp5_attn6_b128",  "attn_clip": 31, "mlp_clip": 15, "block_size": 128, "percdamp": 0.01},

    # Aggressive: int4 for MLP, int5 for attention
    {"name": "mixed_mlp4_attn5",  "attn_clip": 15, "mlp_clip": 7,  "block_size": 64,  "percdamp": 0.002},
]

print(f"{'Config':<25} {'Compressed':>12} {'Total':>12} {'Headroom':>10} {'Extra params':>14} {'Time':>8}")
print("-" * 85)

results = []
for cfg in configs:
    t0 = time.time()
    sd_cpu = {k: v.detach().cpu() for k, v in state_dict.items()}
    qr, qm, gc, nc = quantize_mixed(sd_cpu, hessians, cfg["attn_clip"], cfg["mlp_clip"], cfg["block_size"], cfg["percdamp"])

    buf = io.BytesIO()
    torch.save({"w": qr, "m": qm}, buf)
    raw = buf.getvalue()
    compressed = cctx.compress(raw)
    total = len(compressed) + code_bytes
    headroom = 16_000_000 - total
    # Estimate extra params: headroom bytes / (compressed bits per param)
    bytes_per_param = len(compressed) / n_params
    extra_params = int(headroom / bytes_per_param) if headroom > 0 else 0
    elapsed = time.time() - t0

    hr_str = f"{headroom:,}" if headroom > 0 else f"OVER {-headroom:,}"
    ep_str = f"+{extra_params:,}" if extra_params > 0 else "—"

    print(f"{cfg['name']:<25} {len(compressed):>12,} {total:>12,} {hr_str:>10} {ep_str:>14} {elapsed:>7.1f}s")
    results.append({**cfg, "compressed": len(compressed), "total": total, "headroom": headroom, "extra_params": extra_params, "gptq": gc, "naive": nc})

print()
print("=" * 85)
print("ANALYSIS")
print("=" * 85)
print()

# Find int6 and int5 baselines
int6_base = next(r for r in results if r["name"] == "int6_baseline")
int5_best = min([r for r in results if "int5" in r["name"]], key=lambda r: r["total"])
mixed_best = min([r for r in results if "mixed" in r["name"] and r["headroom"] > 0], key=lambda r: r["total"], default=None)

print(f"Current (int6 b64/pd002):  {int6_base['total']:,} bytes, headroom: {int6_base['headroom']:,}")
print(f"Best int5:                 {int5_best['total']:,} bytes, headroom: {int5_best['headroom']:,}, extra params: +{int5_best['extra_params']:,}")
if mixed_best:
    print(f"Best mixed:                {mixed_best['total']:,} bytes, headroom: {mixed_best['headroom']:,}, extra params: +{mixed_best['extra_params']:,}")
print()

# Architecture projections
print("ARCHITECTURE PROJECTIONS (what fits in 16MB):")
print()
for r in sorted(results, key=lambda r: -r["headroom"]):
    if r["headroom"] <= 0:
        continue
    total_params = n_params + r["extra_params"]
    print(f"  {r['name']:<25} → {total_params:,} total params (+{r['extra_params']:,})")

print()
print("CANDIDATE ARCHITECTURES:")
print(f"  Current:  11L/512d/8H/4KV/MLP1536  = {n_params:,} params")
print(f"  +MHA 8/8: 11L/512d/8H/8KV/MLP1536  = ~29.9M params (+2.9M from KV)")
print(f"  +MLP3.5x: 11L/512d/8H/8KV/MLP1792  = ~33.6M params (+3.7M from bigger MLP)")
print(f"  +12 layers: 12L/512d/8H/8KV/MLP1792 = ~36.6M params")
print()
print("Match these against the headroom above to find what fits.")
