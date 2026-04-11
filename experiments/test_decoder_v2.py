"""Compilerized Model Artifacts v2 — smaller decoders + prototype test

Test if a tiny decoder can beat int6 while being smaller than int6.
Also test prototype + low-rank corrections.
"""
import torch
import torch.nn as nn
import math
import sys
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/code/parameter-golf/checkpoints/final_model_baseline.pt")
print(f"Loading {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location="cpu")

# Extract MLP weights
proj_weights = []
fc_weights = []
for name, tensor in sorted(ckpt.items()):
    if "mlp.proj.weight" in name:
        proj_weights.append(tensor.float())
    elif "mlp.fc.weight" in name:
        fc_weights.append(tensor.float())

proj_stack = torch.stack(proj_weights)  # [11, 512, 1536]
n_layers = proj_stack.shape[0]
flat_size = proj_stack[0].numel()  # 786432
print(f"Layers: {n_layers}, flat size: {flat_size:,}")

# Int6 baseline
def int6_mse(t):
    row_max = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = row_max / 31.0
    q = (t / scale).round().clamp(-31, 31)
    return ((q * scale - t) ** 2).mean().item()

proj_int6_mse = sum(int6_mse(w) for w in proj_weights) / len(proj_weights)
print(f"Int6 MSE: {proj_int6_mse:.8f}")
print(f"Original size (int6): {n_layers * flat_size * 6 / 8 / 1024:.0f} KB")

# === Test 1: Prototype + low-rank corrections ===
print(f"\n=== Prototype + Low-Rank Corrections ===")
prototype = proj_stack.mean(dim=0)

for rank in [4, 8, 16, 32, 64, 128, 256]:
    total_mse = 0
    total_correction_params = 0

    for i in range(n_layers):
        residual = proj_weights[i] - prototype
        U, S, Vt = torch.linalg.svd(residual, full_matrices=False)
        approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
        mse = ((prototype + approx - proj_weights[i]) ** 2).mean().item()
        total_mse += mse
        # Store U_r (512*rank) + S_r (rank) + Vt_r (rank*1536) per layer
        total_correction_params += 512 * rank + rank + rank * 1536

    avg_mse = total_mse / n_layers
    proto_params = prototype.numel()
    total_params = proto_params + total_correction_params
    # fp16 for proto+corrections vs int6 for original
    total_kb = total_params * 2 / 1024
    original_kb = n_layers * flat_size * 6 / 8 / 1024
    vs_int6 = avg_mse / proj_int6_mse
    print(f"  rank={rank:3d} | {total_kb:.0f}KB (vs {original_kb:.0f}KB int6) | "
          f"MSE={avg_mse:.8f} | {vs_int6:.2f}x vs int6 | "
          f"{'VIABLE' if total_kb < original_kb and vs_int6 <= 1.0 else 'no'}")

# === Test 2: Per-layer SVD (no shared prototype) ===
print(f"\n=== Per-Layer SVD (independent low-rank per layer) ===")
for rank in [4, 8, 16, 32, 64, 128, 256]:
    total_mse = 0
    total_params = 0

    for i in range(n_layers):
        w = proj_weights[i]
        U, S, Vt = torch.linalg.svd(w, full_matrices=False)
        approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
        mse = ((approx - w) ** 2).mean().item()
        total_mse += mse
        total_params += 512 * rank + rank + rank * 1536

    avg_mse = total_mse / n_layers
    total_kb = total_params * 2 / 1024
    original_kb = n_layers * flat_size * 6 / 8 / 1024
    vs_int6 = avg_mse / proj_int6_mse
    print(f"  rank={rank:3d} | {total_kb:.0f}KB (vs {original_kb:.0f}KB int6) | "
          f"MSE={avg_mse:.8f} | {vs_int6:.2f}x vs int6 | "
          f"{'VIABLE' if total_kb < original_kb and vs_int6 <= 1.0 else 'no'}")

# === Test 3: Tiny decoder on CPU (avoid OOM) ===
print(f"\n=== Tiny Decoder (CPU, 3000 steps) ===")

targets = proj_stack.reshape(n_layers, flat_size)
target_scale = targets.abs().mean()
targets_normed = targets / target_scale

# Process in chunks to fit in memory
chunk_size = flat_size // 4  # Reconstruct in 4 chunks

for latent_dim in [64, 128, 256]:
    for hidden_dim in [256, 512]:
        # 4 small decoders, each reconstructing 1/4 of the weight matrix
        total_decoder_params = 0
        total_mse = 0
        latent_codes = torch.randn(n_layers, latent_dim) * 0.1
        latent_codes.requires_grad_(True)

        for chunk_idx in range(4):
            start = chunk_idx * chunk_size
            end = start + chunk_size
            chunk_targets = targets_normed[:, start:end]  # [11, chunk_size]

            decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, chunk_size),
            )
            total_decoder_params += sum(p.numel() for p in decoder.parameters())

            opt = torch.optim.Adam(list(decoder.parameters()) + [latent_codes], lr=1e-3)
            best = float("inf")
            for step in range(3000):
                preds = decoder(latent_codes)
                loss = ((preds - chunk_targets) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                mse = loss.item() * (target_scale.item() ** 2)
                if mse < best:
                    best = mse
            total_mse += best
            del decoder, opt

        avg_mse = total_mse / 4
        total_params = total_decoder_params + latent_codes.numel()
        total_kb = total_params * 2 / 1024
        original_kb = n_layers * flat_size * 6 / 8 / 1024
        vs_int6 = avg_mse / proj_int6_mse
        print(f"  latent={latent_dim:3d} hidden={hidden_dim:3d} | "
              f"{total_kb:.0f}KB (vs {original_kb:.0f}KB int6) | "
              f"MSE={avg_mse:.8f} | {vs_int6:.2f}x vs int6 | "
              f"{'VIABLE' if total_kb < original_kb and vs_int6 <= 1.0 else 'no'}")

print("\nDone.")
