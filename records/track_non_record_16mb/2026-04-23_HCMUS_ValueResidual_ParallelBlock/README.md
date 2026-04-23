# Value Residuals + Parallel Block + Stochastic Depth

**9 Layers (4 Encoder - 5 Decoder) + Value Residuals + Parallel Transformer + Stochastic Depth + EMA(0.997) + XSA + SmearGate + BigramHash + int8/zlib**

**val_bpb: 1.2081** (sliding stride=64) | **14.63 MB** artifact | RTX Pro 6000, 20,000 steps (~5.9h)

> **Non-record submission from Lab 1 - FIT HCMUS.** This model explores optimizing information flow and computational efficiency to maximize layer depth within a strict 16MB parameter budget.

## Results (seed=1337)

| Metric | Value |
|--------|-------|
| **Sliding BPB (s64)** | **1.2081** |
| val_bpb (roundtrip) | 1.2417 |
| val_loss | 2.0398 |
| Training Steps | 20,000 |
| Total Parameters | 17,650,825 |
| Training Time | 212,370s (~5.9h) |
| Artifact Size | 14,633,147 bytes (14.63 MB) |

## Architecture

The architecture is based on a U-Net Transformer structure enhanced by three core techniques designed to improve convergence and parameter efficiency:

- **Value Residuals (VR):** Each layer propagates its $V$ (Value) state to the next layer via a sigmoid-gated residual chain. This facilitates long-range value propagation across the network depth.
- **Parallel Attention + MLP Block:** Unlike sequential blocks, both the attention and MLP branches receive the same pre-norm input. This saves ~15% wall-clock time, allowing the model to fit 9 layers instead of the baseline 7 within the compute budget.
- **Stochastic Depth:** A linear drop-rate schedule (0.0 to 0.1) is applied across layers. This forces the model to be less dependent on specific layers, acting as a strong regularizer before int8 quantization.

### Additional Features
- **XSA (Extended Self-Attention):** Applied to the final 4 layers to refine feature extraction.
- **SmearGate:** Learned causal smoothing that blends the current position with a cumulative mean of past positions.
- **BigramHash:** A secondary embedding layer that hashes bigrams to compensate for the small 1024 vocab size.
- **EMA (Exponential Moving Average):** A decay of 0.997 is used to smooth weights, significantly reducing quantization noise during the int8 roundtrip.

## Compression & Compliance

- **Int-8 Quantization:** Weights are quantized to 8-bit using row-wise or scalar scaling modes.
- **zlib Compression:** Final artifact is compressed at level 9, reducing the size from ~17.6MB to 14.63MB.
- [x] Artifact <= 16,000,000 bytes (14,633,147).
- [x] Sliding window evaluation (stride=64).
- [x] No test-time training or external compute.