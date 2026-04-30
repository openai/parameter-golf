"""Compilerized Model Artifacts — Cheapest Test

Can a tiny decoder network reconstruct MLP weight matrices from latent codes?
If yes, 11 MLP layers (~8.6M params) could be stored as 1 decoder + 11 latent codes.

Test: Train a small MLP decoder to reconstruct MLP proj weights from per-layer latents.
Compare reconstruction MSE vs int6 quantization MSE.
"""
import torch
import torch.nn as nn
import math
import sys
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load checkpoint
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/code/parameter-golf/checkpoints/final_model_baseline.pt")
print(f"Loading {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location="cpu")

# Extract all MLP proj weights (512 x 1536) and fc weights (1536 x 512)
proj_weights = {}
fc_weights = {}
for name, tensor in ckpt.items():
    if "mlp.proj.weight" in name:
        proj_weights[name] = tensor.float()
    elif "mlp.fc.weight" in name:
        fc_weights[name] = tensor.float()

print(f"\nMLP proj weights: {len(proj_weights)} layers, shape {list(next(iter(proj_weights.values())).shape)}")
print(f"MLP fc weights: {len(fc_weights)} layers, shape {list(next(iter(fc_weights.values())).shape)}")

# Stack all proj weights for analysis
proj_stack = torch.stack(list(proj_weights.values()))  # [11, 512, 1536]
fc_stack = torch.stack(list(fc_weights.values()))  # [11, 1536, 512]
n_layers = proj_stack.shape[0]

# === Baseline: Int6 quantization MSE ===
def int6_mse(t):
    row_max = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = row_max / 31.0
    quantized = (t / scale).round().clamp(-31, 31)
    reconstructed = quantized * scale
    return ((t - reconstructed) ** 2).mean().item()

proj_int6_mse = sum(int6_mse(w) for w in proj_weights.values()) / len(proj_weights)
fc_int6_mse = sum(int6_mse(w) for w in fc_weights.values()) / len(fc_weights)
print(f"\n=== Int6 baseline MSE ===")
print(f"  MLP proj: {proj_int6_mse:.8f}")
print(f"  MLP fc:   {fc_int6_mse:.8f}")

# === Cross-layer Procrustes analysis ===
print(f"\n=== Procrustes Analysis (cross-layer similarity) ===")
print("MLP proj (512×1536):")
for i in range(min(n_layers, 5)):
    for j in range(i+1, min(n_layers, 5)):
        A = proj_stack[i]
        B = proj_stack[j]
        # Compute optimal rotation: R = V @ U^T from SVD(A^T @ B)
        U, S, Vt = torch.linalg.svd(A.T @ B, full_matrices=False)
        R = Vt.T @ U.T
        aligned = A @ R
        residual_mse = ((aligned - B) ** 2).mean().item()
        original_mse = ((A - B) ** 2).mean().item()
        reduction = 1.0 - residual_mse / original_mse
        print(f"  layers {i}↔{j}: {reduction*100:.1f}% reduction (residual MSE: {residual_mse:.6f})")

# === Decoder Test ===
print(f"\n=== Decoder Reconstruction Test ===")

# Flatten weights for reconstruction
# Each layer's proj weight is 512*1536 = 786432 params
flat_size = proj_stack[0].numel()
targets = proj_stack.reshape(n_layers, flat_size).to(device)

# Normalize targets for training stability
target_scale = targets.abs().mean()
targets_normed = targets / target_scale

for latent_dim in [32, 64, 128, 256]:
    for hidden_dim in [512, 1024, 2048]:
        # Decoder: latent_code → weight matrix
        decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_size),
        ).to(device)

        # Per-layer latent codes
        latent_codes = nn.Parameter(torch.randn(n_layers, latent_dim, device=device) * 0.1)

        decoder_params = sum(p.numel() for p in decoder.parameters())
        latent_params = latent_codes.numel()
        total_params = decoder_params + latent_params
        original_params = n_layers * flat_size

        # Size comparison
        # Decoder at fp16 + latents at fp16 vs original at int6
        decoder_bytes = total_params * 2  # fp16
        original_bytes = original_params * 6 / 8  # int6
        compression = original_bytes / decoder_bytes

        optimizer = torch.optim.Adam(
            list(decoder.parameters()) + [latent_codes], lr=1e-3)

        # Train for 2000 steps
        best_mse = float("inf")
        for step in range(2000):
            preds = decoder(latent_codes)  # [11, flat_size]
            loss = ((preds - targets_normed) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse = loss.item() * (target_scale.item() ** 2)
            if mse < best_mse:
                best_mse = mse

            if step == 1999:
                vs_int6 = best_mse / proj_int6_mse
                status = "BETTER" if vs_int6 < 1.0 else f"{vs_int6:.1f}x worse"
                print(f"  latent={latent_dim:3d} hidden={hidden_dim:4d} | "
                      f"decoder={decoder_params:,} latent={latent_params:,} total={total_params:,} | "
                      f"MSE={best_mse:.8f} ({status} vs int6) | "
                      f"compression={compression:.2f}x")

        del decoder, latent_codes, optimizer
        torch.cuda.empty_cache()

# === Shared prototype + per-layer transform test ===
print(f"\n=== Prototype + Transform Test ===")
print("Store 1 prototype MLP proj + per-layer low-rank corrections")

prototype = proj_stack.mean(dim=0).to(device)  # [512, 1536]
proto_mse_total = 0

for rank in [8, 16, 32, 64, 128]:
    total_correction_params = 0
    total_mse = 0

    for i in range(n_layers):
        target = proj_stack[i].to(device)
        residual = target - prototype

        # Low-rank approximation of residual
        U, S, Vt = torch.linalg.svd(residual, full_matrices=False)
        approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
        reconstructed = prototype + approx
        mse = ((reconstructed - target) ** 2).mean().item()
        total_mse += mse

        # Params: U[:, :rank] (512*rank) + S[:rank] (rank) + Vt[:rank, :] (rank*1536)
        correction_params = 512 * rank + rank + rank * 1536
        total_correction_params += correction_params

    avg_mse = total_mse / n_layers
    proto_params = prototype.numel()
    total_params = proto_params + total_correction_params
    original_params = n_layers * flat_size

    # Size at fp16
    total_bytes = total_params * 2
    original_bytes = original_params * 6 / 8
    compression = original_bytes / total_bytes

    vs_int6 = avg_mse / proj_int6_mse
    status = "BETTER" if vs_int6 < 1.0 else f"{vs_int6:.1f}x worse"

    print(f"  rank={rank:3d} | proto={proto_params:,} corrections={total_correction_params:,} "
          f"total={total_params:,} | MSE={avg_mse:.8f} ({status} vs int6) | "
          f"compression={compression:.2f}x")

print("\nDone.")
