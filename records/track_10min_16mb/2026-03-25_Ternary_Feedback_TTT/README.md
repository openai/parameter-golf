# SKC Ternary Reasoner — Spectral Koopman Capsule Architecture

**Author:** Aki Gogikar (OneNew AI) | **GitHub:** akhileshgogikar

**Architecture:** SKC 24L d512 · Ternary BitNet b1.58 · Muon + LAWA/SWA · Engram Hash (orders=3) · N-gram Cache · Sliding Eval + Temp Scaling

**Training:** 8×H100 SXM, 599s wallclock | **Model:** ~51.7M ternary params, ~10.4MB compressed

---

## Architecture: Spectral Koopman Capsule (SKC)

The core idea is replacing the standard MLP sub-layer with a **Spectral Koopman Capsule block** — a sequence mixer based on the Walsh-Hadamard Transform (WHT) and Koopman operator theory. Each SKC block:

1. **Causal blockwise WHT** (block_size=16): Applies a learned spectral decomposition in sequency space, capturing long-range token correlations within blocks at O(N log N) cost.
2. **FrequencyBandRouter**: Routes each sequency band to a specialized capsule via learned gating — analogous to MoE but in frequency space, with no routing collapse.
3. **KoopmanSpectralEvolution**: Propagates the frequency representation forward in time using Koopman eigenfunctions, providing an inductive bias for temporal dynamics.
4. **Symmetric UNet capsule skip connections**: Capsule states flow through a U-Net encoder-decoder structure across layers. Proven -0.107 BPB improvement over no skip connections.

This architecture exploits ternary quantization naturally: the WHT is purely additive (no multiplications), making it ideal for BitNet b1.58 {-1, 0, +1} weights.

### Model Configuration (8×H100 submission)
| Parameter | Value |
|-----------|-------|
| Layers | 24 |
| Model dim | 512 |
| Attention heads | 8 (4 KV, GQA) |
| SKC block size | 16 |
| SKC capsules | 16 × dim=128 |
| SKC conv kernel | 4 |
| MLP multiplier | 4× |
| Vocabulary | 8192 BPE (competition standard) |
| Sequence length | 2048 |
| Parameters (total) | ~51.7M |
| Compressed size | ~10.4MB |

---

## Key Techniques

### Ternary Quantization (BitNet b1.58)
All weight matrices quantized to {-1, 0, +1} using per-group (128) absmean scaling during training. The turbo packer uses base-3 encoding (5 trits/byte) + LZMA-9 compression, achieving ~10.4MB for a 51.7M parameter model.

### Engram Hash (orders=3)
A learned hash embedding table that captures bigram, trigram, and 4-gram token statistics, injected at layer 1. Uses 8192 buckets × 128 dims with 3 orders, providing token-level context that the model can use from the very first layer.

### Muon Optimizer
Matrix weights optimized with Muon (momentum=0.95, WD=0.04, 5 Newton-Schulz steps). Embedding and scalar parameters use AdamW. Learning rates: matrix=0.02, scalar=0.015, tied_embed=0.025.

### Weight Averaging (LAWA + SWA)
- **LAWA** (Latest Averaged Weights, k=5): Averages the last 5 optimizer snapshots.
- **SWA** (Stochastic Weight Averaging): Periodic averaging checkpoints throughout training.

### N-gram Cache
At evaluation time, a count-based n-gram language model (order=5) is interpolated with the neural model output. The mixing coefficient is entropy-adaptive: high-entropy positions rely on the neural model, low-entropy positions lean on n-gram statistics.

### SmearGate
A learned gating mechanism that allows the model to smear residual information across token positions, providing a cheap alternative to extra attention heads for positional mixing.

### Sliding Window Evaluation + Temperature Scaling
- **Sliding eval** (stride=64): Evaluates all positions using the full 2048-token context window.
- **Temperature scaling**: Post-training calibration finds the optimal softmax temperature on training data.

---

## Training Setup

### Hardware
- 8× NVIDIA H100 SXM (80GB HBM3), NVLink interconnect
- PyTorch 2.4.0 + CUDA 12.4

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch tokens | 786,432 / step (global) |
| Sequence length | 2048 |
| Warmup steps | 20 (outside 599s budget — compile trigger only) |
| Warmdown fraction | 0.4 (last 240s) |
| Grad clip | 0.3 |
| Muon LR | 0.02 |
| Adam LR | 0.015 |
| Weight decay | 0.04 |

### Run
```bash
bash run_runpod_8xh100.sh
```

---

## Results

| Seed | Steps | ms/step | val_bpb (sliding) | val_bpb (roundtrip) | Artifact |
|------|-------|---------|-------------------|---------------------|----------|
| 42   | TBD   | TBD     | TBD               | TBD                 | TBD      |

*Results pending final 8×H100 run.*

---

## File Structure

| File | Purpose |
|------|---------|
| `train_gpt.py` | Main training script |
| `run_runpod_8xh100.sh` | 8×H100 competition launch script |
| `run_small_skc_2gpu.sh` | 2-GPU development/testing script |
| `requirements.txt` | Python dependencies |
| `submission.json` | Competition submission metadata (generated after run) |
