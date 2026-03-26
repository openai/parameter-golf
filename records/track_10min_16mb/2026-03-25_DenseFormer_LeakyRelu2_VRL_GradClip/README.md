# DenseFormer + LeakyReLU(0.5)² + VRL + XSA + Grad Clip

## Score: val_bpb = 1.3036 (1×H100, pending 8×H100 multi-seed eval)

Trained on 1×H100 in 600 seconds. int8+zlib artifact (pending final size measurement on 8×H100).

## Approach

Five techniques stacked on the baseline 9-layer, 512-dim GPT:

### 1. DenseFormer (Depth-Weighted Average)
After each transformer block, compute a learned weighted average of current and ALL past layer representations (including the embedding output). Adds only d(d+3)/2 = 54 scalar parameters for 9 layers. Based on [DenseFormer (Pagliardini et al., 2024)](https://arxiv.org/abs/2402.02622). Enables cross-layer feature reuse without increasing model width or depth.

### 2. LeakyReLU(0.5)²
Replaces relu² activation with leaky_relu(negative_slope=0.5)². The large negative slope (0.5) keeps gradient flowing through negative activations while squaring still provides sparsity. Zero extra parameters, negligible wallclock cost on CUDA. Provides -0.010 BPB over relu² in local testing.

### 3. Value Residual Learning (VRL)
Caches the value (V) tensor from layer 0 and blends it into all subsequent layers' V via learned softmax-normalized scalars: `v = λ1 * v_first + λ2 * v_current`. Prevents attention concentration collapse in deeper layers. Complementary to DenseFormer (DWA operates on residual stream; VRL operates inside attention). Adds 2 scalar parameters per layer (16 total). Initialized as identity (use current V only).

### 4. Cross-Self Attention (XSA)
Applied to the last 4 layers. Projects out the self-value component from attention output, encouraging the model to focus on cross-token information rather than self-reinforcement. Zero extra parameters.

### 5. Gradient Clipping
Global gradient norm clipping at 0.3. Stabilizes training with the increased gradient flow from DenseFormer + VRL + LeakyReLU pathways. Proven technique from top leaderboard submissions.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 9 |
| model_dim | 512 |
| mlp_mult | 2 (hidden=1024) |
| train_seq_len | 1024 |
| train_batch_tokens | 524,288 |
| warmdown_iters | 1200 |
| matrix_lr | 0.04 |
| scalar_lr | 0.04 |
| tied_embed_lr | 0.05 |
| muon_momentum | 0.95 (warmup from 0.85 over 500 steps) |
| grad_clip_norm | 0.3 |
| xsa_layers | 4 (last 4 layers) |
| vrl | enabled (layers 1-8) |
| compressor | zlib (level 9) |

## Key Metrics (1×H100, preliminary)

- **val_bpb: 1.3036** (int8+zlib roundtrip)
- Pre-quant val_bpb: 1.3024
- Quantization penalty: 0.0012 bpb (int8 vs fp16)
- Training: 1,504 steps in 600s (399 ms/step)
- Artifact size: pending 8×H100 run

## Ablation (1×H100 RunPod)

| Config | Steps | val_bpb (int8+zlib) | Delta |
|--------|-------|---------------------|-------|
| Baseline (vanilla) | 1209 | 1.3448 | — |
| + DenseFormer (relu²) | 1151 | 1.3439 | -0.0009 |
| + LeakyReLU(0.5)² + VRL + GC | 1228 | 1.3215 | -0.0233 |
| **+ XSA (last 4 layers)** | **1504** | **1.3036** | **-0.0412** |

## Reproducibility

Pending 8×H100 multi-seed evaluation. Draft PR submitted for GPU credit request.
