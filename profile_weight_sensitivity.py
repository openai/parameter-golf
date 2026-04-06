"""Profile per-layer weight sensitivity for SpQR-style mixed precision.

Downloads the Exp 2 model from HuggingFace, quantizes to int6,
and measures per-weight quantization error weighted by Hessian diagonal.
Outputs: for each layer, what % of weights cause what % of total error.
"""
import os, sys, torch, math
import torch.nn.functional as F
from torch import Tensor
import types

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["SWA_WINDOW_SIZE"] = "256"
os.environ["SWA_FULL_ATTN_LAYERS"] = "5"
os.environ["BIGRAM_VOCAB_SIZE"] = "3072"
os.environ["BIGRAM_DIM"] = "112"
os.environ["NUM_LAYERS"] = "11"
os.environ["XSA_LAST_N"] = "11"
os.environ["ROPE_DIMS"] = "16"
os.environ["LN_SCALE"] = "1"
os.environ["VE_ENABLED"] = "1"
os.environ["VE_LAYERS"] = "9,10"

# SDPA fallback
def sdpa_fallback(q, k, v, causal=True, window_size=(-1, -1), **kwargs):
    q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    y = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=None, is_causal=causal,
                                        enable_gqa=(q.size(2) != k.size(2)))
    return y.transpose(1, 2)

fake_fa = types.ModuleType("flash_attn_interface")
fake_fa.flash_attn_func = sdpa_fallback
sys.modules["flash_attn_interface"] = fake_fa

sys.path.insert(0, ".")
import train_gpt_swa as tgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

args = tgs.Hyperparameters()
model_kwargs = dict(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    gated_attention=args.gated_attention, value_residual=args.value_residual,
)

# Download model from HuggingFace
print("Downloading model from HuggingFace...")
from huggingface_hub import hf_hub_download
pt_path = hf_hub_download("shikhar007/parameter-golf-gram-ns", "exp2_w256_full5_seed1337.pt")
print(f"Downloaded: {pt_path}")

# Load banked state dict
banked_sd = torch.load(pt_path, map_location="cpu")
print(f"Loaded state dict: {len(banked_sd)} keys")

# Unbank for per-weight analysis
unbanked_sd = tgs._unbank_state_dict(banked_sd, args.num_layers)
print(f"Unbanked: {len(unbanked_sd)} keys")

# For each 2D weight, quantize to int6 and measure per-weight error
print("\n" + "=" * 80)
print("PER-LAYER WEIGHT SENSITIVITY ANALYSIS")
print("=" * 80)

layer_stats = {}
all_errors = []

for name, tensor in sorted(unbanked_sd.items()):
    t = tensor.float()
    cat = tgs._classify_param(name)
    if t.ndim != 2 or cat not in ("mlp", "attn"):
        continue

    # Quantize to int6
    q, s = tgs.quantize_int6_per_row(t, clip_range=31)
    # Dequantize
    recon = q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
    # Per-weight absolute error
    error = (t - recon).abs()

    total_err = error.sum().item()
    max_err = error.max().item()
    mean_err = error.mean().item()
    num_weights = t.numel()

    # What % of weights cause 50%, 80%, 90% of the total error?
    flat_err = error.flatten()
    sorted_err, _ = flat_err.sort(descending=True)
    cumsum = sorted_err.cumsum(0)
    total = cumsum[-1].item()

    pct_for_50 = (cumsum <= 0.5 * total).sum().item() / num_weights * 100
    pct_for_80 = (cumsum <= 0.8 * total).sum().item() / num_weights * 100
    pct_for_90 = (cumsum <= 0.9 * total).sum().item() / num_weights * 100

    # Extract layer index
    layer_idx = -1
    if name.startswith("blocks."):
        try:
            layer_idx = int(name.split(".")[1])
        except:
            pass

    print(f"{name:45s} [{t.shape[0]:5d}x{t.shape[1]:5d}] "
          f"mean_err={mean_err:.6f} max_err={max_err:.4f} "
          f"top_for_50%err={pct_for_50:.1f}% top_for_80%={pct_for_80:.1f}% top_for_90%={pct_for_90:.1f}%")

    layer_stats[name] = {
        "shape": list(t.shape),
        "cat": cat,
        "layer": layer_idx,
        "mean_err": mean_err,
        "max_err": max_err,
        "total_err": total_err,
        "num_weights": num_weights,
        "pct_for_50": pct_for_50,
        "pct_for_80": pct_for_80,
        "pct_for_90": pct_for_90,
    }
    all_errors.append((name, flat_err))

# Summary by layer
print("\n" + "=" * 80)
print("SUMMARY BY LAYER")
print("=" * 80)
for layer_idx in range(args.num_layers):
    layer_names = [n for n, s in layer_stats.items() if s["layer"] == layer_idx]
    total_err = sum(layer_stats[n]["total_err"] for n in layer_names)
    total_weights = sum(layer_stats[n]["num_weights"] for n in layer_names)
    mean_pct50 = sum(layer_stats[n]["pct_for_50"] for n in layer_names) / len(layer_names)
    print(f"Layer {layer_idx:2d}: {total_weights:>10,} weights, total_err={total_err:.2f}, "
          f"avg top_for_50%err={mean_pct50:.1f}%")

# Summary by component type
print("\n" + "=" * 80)
print("SUMMARY BY COMPONENT")
print("=" * 80)
for comp in ["c_q", "c_k", "c_v", "proj", "fc", "mlp.proj"]:
    comp_names = [n for n in layer_stats if comp in n]
    if not comp_names:
        continue
    total_err = sum(layer_stats[n]["total_err"] for n in comp_names)
    total_weights = sum(layer_stats[n]["num_weights"] for n in comp_names)
    avg_mean_err = sum(layer_stats[n]["mean_err"] for n in comp_names) / len(comp_names)
    avg_pct50 = sum(layer_stats[n]["pct_for_50"] for n in comp_names) / len(comp_names)
    print(f"{comp:12s}: {total_weights:>10,} weights, total_err={total_err:.2f}, "
          f"avg_mean_err={avg_mean_err:.6f}, avg top_for_50%err={avg_pct50:.1f}%")

# Key insight: if top X% of weights cause Y% of error,
# then keeping those X% in fp16 removes Y% of quantization error.
print("\n" + "=" * 80)
print("GLOBAL ERROR CONCENTRATION")
print("=" * 80)
# Concatenate all errors
all_flat = torch.cat([e for _, e in all_errors])
sorted_all, _ = all_flat.sort(descending=True)
cumsum_all = sorted_all.cumsum(0)
total_all = cumsum_all[-1].item()
n_all = all_flat.numel()

for pct in [1, 2, 3, 5, 8, 10, 15, 20]:
    k = int(n_all * pct / 100)
    err_captured = cumsum_all[k-1].item() / total_all * 100 if k > 0 else 0
    fp16_bytes = k * 2
    int5_bytes = (n_all - k) * 1
    total_bytes = fp16_bytes + int5_bytes
    print(f"Top {pct:2d}% weights ({k:>10,}): captures {err_captured:.1f}% of error. "
          f"Storage: {total_bytes/1024/1024:.1f}MB raw "
          f"(fp16:{fp16_bytes/1024/1024:.1f}MB + int5:{int5_bytes/1024/1024:.1f}MB)")
