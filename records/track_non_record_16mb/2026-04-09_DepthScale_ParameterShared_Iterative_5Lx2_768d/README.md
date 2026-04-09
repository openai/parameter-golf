# Non-Record Submission: DepthScale — Parameter-Shared Iterative Transformer with I4 Training

**val_bpb: 1.1962** (3-seed mean, std 0.0005) | **~30 MB** artifact (int8+zlib, exceeds 16MB — see notes) | 8×H100 SXM, 600s

> **Non-record submission** demonstrating a novel architecture: parameter-shared iterative reasoning applied to language model compression. 5 physical transformer layers × N iterations = 5N effective depth at constant parameter cost. Combined with 4-bit Straight-Through Estimator (STE) training from the anoLLM research framework.

## Architecture: DepthScale

### Core Innovation: Parameter-Shared Iterative Depth

Instead of stacking independent transformer layers, DepthScale reuses the **same 5 physical layers** across multiple iterations. Each iteration processes the full sequence through all 5 layers, with **iteration-aware RoPE** that distinguishes pass 1 from pass N.

```
Standard 10-layer transformer:
  Layer 0 → Layer 1 → ... → Layer 9
  10 sets of unique weights, 10 layers of depth

DepthScale (5 layers × 2 iterations):
  Iteration 0: Layer 0 → Layer 1 → Layer 2 → Layer 3 → Layer 4
  Iteration 1: Layer 0 → Layer 1 → Layer 2 → Layer 3 → Layer 4
  SAME weights reused, 10 effective layers of depth, 5 sets of weights
```

This is related to depth recurrence (PR #745) but with a key difference: **iteration-aware positional encoding**. Standard depth recurrence repeats layers with identical positional context. DepthScale shifts the RoPE frequencies by `ε × iteration`, allowing the model to learn iteration-specific attention patterns.

```python
# Iteration-aware RoPE (from DepthScale research)
angles = positions × theta + ROPE_ITER_FACTOR × iteration
# ε = 0.1: ~10% influence of iteration vs position
```

### Parameter Efficiency

At 768d, 8 heads, 4x MLP:
- 5 physical layers: ~36M parameters
- 2 iterations: 10 effective layers
- 4 iterations: 20 effective layers
- Parameters stay constant regardless of iteration count

### I4 Straight-Through Estimator Training

During training, weight matrices undergo 4-bit quantization simulation via STE:

```python
# Forward: use quantized weights (4-bit range [-7, 7])
w_quantized = clamp(round(w / scale), -7, 7) * scale

# Backward: gradient flows through original weights (straight-through)
w = w + (w_quantized - w).detach()
```

This trains the model to be robust to extreme quantization. Derived from the anoLLM (nanollm) research framework's `I4TrainableLinear` implementation.

### Results

| Config | Pre-Quant BPB | Post-Quant BPB | Steps | ms/step | Notes |
|--------|:----------:|:----------:|------:|--------:|-------|
| 5L × 4 iter, I4 training | 1.2497 | 1.6728 (I4) | 2,605 | 230ms | I4 post-quant too aggressive |
| 5L × 2 iter, float training | 1.1902 | **1.1962** (int8) | ~5,500 | ~106ms | **Architecture validated** |

### 3-Seed Reproducibility (8×H100 SXM, PyTorch 2.4.1)

| Seed | Steps | val_bpb (int8 roundtrip) | Artifact Size |
|------|------:|:------------------------:|:-------------:|
| 1337 | 5,583 | **1.19674** | 30,111,815 |
| 42 | ~5,500 | **1.19595** | 30,130,131 |
| 2025 | ~5,500 | **1.19581** | 30,170,827 |
| **Mean** | | **1.19617 (std 0.0005)** | |

The **pre-quantization BPB of 1.1902** from 5 shared layers demonstrates that parameter-shared iterative reasoning is a viable architecture for small language models. For comparison, the 9-layer baseline achieves 1.2244 with nearly 2x the parameter count.

### What Works

- **Parameter sharing reduces parameters by ~2x** at comparable effective depth
- **Iteration-aware RoPE** allows the model to learn distinct per-iteration attention patterns
- **I4 STE training** enables the model to learn under quantization pressure (loss converges normally)

### What Needs Work

- **I4 post-training quantization** is too aggressive (0.42 BPB penalty vs ~0.007 for int8)
- **Step time overhead**: 4 iterations = 4x forward pass cost per step, losing training steps
- **No SOTA techniques integrated yet** (XSA, EMA, sliding window, TTT)
- **torch.compile compatibility** not verified on all PyTorch versions

### Connection to Published Research

This submission draws on three research projects:

1. **DepthScale (YOCO)**: Universal self-decoder for memory-constant depth scaling. Parameter-shared iterative reasoning with convergence-based stopping. [Research: Automate Capture Research, 2026]

2. **anoLLM (nanollm)**: I4 quantized linear layers with STE, 4-bit fixed-point activations, sparse attention kernels. 27 tests passing. [Lab build: osp-72527819]

3. **Model Garage**: Component-level neural network surgery toolkit. Used for activation profiling and neuron importance analysis during development. [Published: Apache 2.0, Lumi-node/model-garage]

### Experiment Journey

This submission is the result of 30+ controlled experiments across 5 GPU sessions (~$130 total compute):

- **Session 1**: Baseline verification (1.2304 BPB), ADRQ negative result
- **Session 2**: MLLA (Multi-Layer Latent Attention) — noise-level, not useful
- **Session 3**: Gearbox experiments — curriculum learning validated (-0.017 BPB on baseline)
- **Session 4**: SOTA reproduction (1.1243 BPB), curriculum on SOTA (noise-level)
- **Session 5**: DepthScale + I4 — architecture validated (1.1902 pre-quant)

Full experiment documentation available in the DMEDI/ folder of our research repository.

### Configuration

```bash
DEPTH_ITERS=2 \
I4_ENABLED=0 \
NUM_PHYSICAL_LAYERS=5 \
MODEL_DIM=768 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=4 \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Artifact Size Note

The current artifact is ~30MB at int8+zlib, exceeding the 16MB limit. This is because 36.2M params × 1 byte (int8) + scales + metadata = ~30MB. To fit 16MB, the model needs either:
- **Int6 quantization** (~22.5MB raw → ~16MB compressed)
- **Reduced dimensions** (768d → ~544d, fewer params)
- **LZMA compression** (better ratio than zlib)

This is submitted as a **non-record** to demonstrate the architecture's viability. The BPB of 1.1962 from only 5 physical layers validates parameter-shared depth as a compression strategy worth pursuing.

### Compliance

- [x] 3 seeds run on 8×H100 SXM
- [x] All seeds train in ≤600s
- [ ] All artifacts ≤16,000,000 bytes (**NOT MET** — 30MB, needs int6 or dim reduction)
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] No external compute

### Author

Andrew Young (@Lumi-node) | Automate Capture Research
