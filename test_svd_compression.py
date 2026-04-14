"""Test post-training SVD compression on the trained baseline model.
Compares: int8-only vs SVD+int8 at various ranks.
"""
import io
import os
import sys
import zlib
import torch
import torch.nn.functional as F
import numpy as np

# Load baseline code to get model class and quantization functions
# We need to avoid running main(), so we exec only the class/function definitions
code = open("train_gpt.py", encoding="utf-8").read()
exec(code.split("def main")[0])

# Load the trained model
print("Loading trained baseline model...")
args = Hyperparameters()
device = torch.device("cuda", 0)
torch.cuda.set_device(device)

base_model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
).to(device).bfloat16()

state = torch.load("final_model.pt", map_location=device)
base_model.load_state_dict(state)
print(f"Model loaded: {sum(p.numel() for p in base_model.parameters()):,} params")


def svd_compress_state_dict(state_dict, target_rank_ratio=0.5):
    """Apply SVD compression to large 2D weight matrices.
    target_rank_ratio: fraction of min(m,n) to keep as rank.
    """
    compressed = {}
    total_orig = 0
    total_compressed = 0
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float()
        if t.ndim == 2 and t.numel() > 65536:
            m, n = t.shape
            rank = max(1, int(min(m, n) * target_rank_ratio))
            U, S, Vh = torch.linalg.svd(t, full_matrices=False)
            # Keep top-r components, distribute S between U and V
            sqrt_S = S[:rank].sqrt()
            A = (U[:, :rank] * sqrt_S[None, :])   # (m, rank)
            B = (Vh[:rank, :] * sqrt_S[:, None])   # (rank, n)
            # Reconstruct and measure error
            W_approx = A @ B
            rel_error = (t - W_approx).norm() / t.norm()
            orig_params = m * n
            new_params = m * rank + rank * n
            total_orig += orig_params
            total_compressed += new_params
            # Store as two separate tensors
            compressed[name + ".__svd_A"] = A.contiguous()
            compressed[name + ".__svd_B"] = B.contiguous()
            print(f"  {name:50s} {m}x{n} -> rank {rank:3d}  ratio={orig_params/new_params:.2f}x  rel_err={rel_error:.4f}")
        else:
            compressed[name] = t.contiguous()
    print(f"  SVD compression: {total_orig:,} -> {total_compressed:,} params ({total_orig/total_compressed:.2f}x)")
    return compressed


def reconstruct_from_svd(compressed):
    """Reconstruct state_dict from SVD-compressed format."""
    reconstructed = {}
    svd_keys = set()
    for key in compressed:
        if key.endswith(".__svd_A"):
            base = key[:-8]
            svd_keys.add(base)
    for key in compressed:
        if key.endswith(".__svd_A") or key.endswith(".__svd_B"):
            continue
        base = key
        if base in svd_keys:
            continue
        reconstructed[key] = compressed[key]
    for base in svd_keys:
        A = compressed[base + ".__svd_A"]
        B = compressed[base + ".__svd_B"]
        reconstructed[base] = (A @ B).contiguous()
    return reconstructed


def measure_compressed_size(state_dict):
    """Measure int8+zlib compressed size of a state dict."""
    quant_obj, stats = quantize_state_dict_int8(state_dict)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    return len(blob), stats


# Test various SVD rank ratios
print("\n=== SVD Compression Analysis ===\n")

orig_sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}

# Baseline: int8 only
int8_size, int8_stats = measure_compressed_size(orig_sd)
print(f"Baseline int8+zlib: {int8_size:,} bytes ({int8_size/1e6:.2f} MB)")

for ratio in [1.0, 0.75, 0.5, 0.375, 0.25]:
    print(f"\n--- SVD rank ratio = {ratio} ---")
    svd_sd = svd_compress_state_dict(orig_sd, target_rank_ratio=ratio)
    recon_sd = reconstruct_from_svd(svd_sd)

    # Measure reconstruction error
    total_err = 0.0
    total_norm = 0.0
    for name in orig_sd:
        if orig_sd[name].ndim == 2 and orig_sd[name].numel() > 65536:
            err = (orig_sd[name].float() - recon_sd[name].float()).norm().item()
            nrm = orig_sd[name].float().norm().item()
            total_err += err**2
            total_norm += nrm**2
    overall_rel_err = (total_err / total_norm) ** 0.5 if total_norm > 0 else 0

    # Measure int8+zlib size of SVD-reconstructed model
    svd_int8_size, svd_stats = measure_compressed_size(recon_sd)

    # Also measure direct SVD storage (store A,B as int8)
    svd_direct_size, svd_direct_stats = measure_compressed_size(svd_sd)

    print(f"  Overall relative error: {overall_rel_err:.6f}")
    print(f"  Reconstructed int8+zlib: {svd_int8_size:,} bytes ({svd_int8_size/1e6:.2f} MB)")
    print(f"  Direct SVD+int8+zlib:    {svd_direct_size:,} bytes ({svd_direct_size/1e6:.2f} MB)")
    print(f"  Size savings vs baseline: {(1 - svd_direct_size/int8_size)*100:.1f}%")

print("\n=== Done ===")
