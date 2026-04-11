# Non-Record Submission: 1.3319 BPB — 10L Spatio-Temporal SNN Transformer

**10L + Spiking-MLP + ArcTan Surrogate Gradients + RBF Temporal Encoding + Muon WD + sp1024**

**val_bpb: 1.3319** | **15.79 MB** artifact | 8×H100 SXM, 13,565 steps (10m)

> **This is a non-record submission. It is submitted to demonstrate the viability of biologically-inspired Spiking Neural Networks (SNN) for autoregressive language modeling under strict temporal and size constraints. Despite the computational overhead of the discrete time-step simulation loop ($T_{sim}=2$), the model converges stably and learns language patterns effectively within the 10-minute wallclock limit.**

## Results (8×H100 SXM)

| Metric | Value |
|--------|-------|
| val_loss | 2.2488 |
| val_bpb | **1.3319** |
| Steps | 13,565 |
| Training time | 600,012 ms (10m) |
| Artifact | 15,794,883 bytes (15.79MB) |
| Code Size | 51,275 bytes |

## Architecture

- 10 transformer layers, dim=768, 12 heads, 4 KV heads (GQA)
- **Spatio-Temporal Integration:** Replaced the standard MLP with a Spiking-MLP running a discrete Leaky Integrate-and-Fire (LIF) simulation over $T_{sim}=2$ time steps.
- **RBF Temporal Encoding:** Continuous token embeddings are converted into spike trains using a parameter-free Radial Basis Function (RBF) Gaussian kernel layer.
- **Surrogate Gradients:** `ArcTan` surrogate derivatives used to allow backpropagation-through-time (BPTT) over binary step functions.
- **Homeostatic Regularization:** Dynamic threshold adaptation mechanism that smoothly maintains neuron firing rates at ~5%.
- FlashAttention-2 / GQA native handling
- Zlib compression + Int8 Post-Training Quantization

## Key Techniques

### Architecture
- **Spiking-MLP:** Proves that non-differentiable, binary-spiking activations can replace standard non-linearities (like GELU/ReLU) in Transformers for text generation.
- **RBF Encoding:** Uses exactly 0 parameters while effectively spreading temporal features, maximizing the representational capacity within the 16MB limit.

### Training
- **Muon Optimizer:** Maintained stability despite the noisy gradients inherent to surrogate BPTT.
- **GQA Handling:** Handled `enable_gqa` compatibility manually for older PyTorch versions inside the compilation graph.

## Compliance

- [x] Artifact <= 16,000,000 bytes (15,794,883 bytes)
- [x] Train time <= 10m (600,012 ms)
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] No external compute
