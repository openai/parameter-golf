# 10L CountInitBigram + XSA + PartialRoPE + LN Scale

**val_bpb: 1.1522** (sliding window stride=64, post int5/int6+zstd quantization roundtrip)

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed.

## Key Techniques

### 1. Count-Initialized Exact Bigram Logit Head (Novel)
A 1024x1024 lookup table providing exact bigram logit biases with zero hash collisions.
Initialized from corpus transition probabilities before training:

```
B[a,b] = log p(b|a) - log p(b)
```

Computed from the first 16M training tokens with additive smoothing (alpha=0.25), clipped to [-4, 4].
Applied BEFORE logit softcap so the bias is properly bounded.
The table is quantized to int4 with nibble packing (524KB vs 1MB at int8).

This gives the model a strong count-based language model prior from step 0, which the neural
network only needs to refine.

### 2. Int4 Nibble Packing (Novel)
Custom `pack_i4` / `unpack_i4` functions pack signed int4 values [-8,7] into uint8 bytes
(two values per byte). Applied to the bigram logit table, halving its storage cost.

### 3. XSA (Exclusive Self Attention) - Last 4 Layers
Removes the self-value component from attention output (arxiv:2603.09078).

### 4. Partial RoPE (16 of 64 dims)
Apply rotary position embeddings to only 25% of head dimensions. The remaining 75%
attend without positional bias, acting as position-independent feature detectors.

### 5. LN Scale
Block outputs scaled by `1/sqrt(layer_idx + 1)`. Damps deeper layers' contributions,
stabilizing training. Zero parameters.

### 6. Higher Learning Rates
- matrix_lr: 0.025 (up from 0.02)
- scalar_lr: 0.025
- tied_embed_lr: 0.035

## Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu squared activation
- SmearGate + exact BigramLogitHead (count-initialized, int4 packed)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Training Hyperparameters
- Muon optimizer: matrix_lr=0.025, WD=0.04, momentum=0.99
- AdamW for embeddings/scalars: WD=0.04
- warmdown=2800 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- SWA: start_frac=0.4, every=50 steps (22 checkpoints)
- Sliding window eval: stride=64

## Quantization
- Int5 [-16,15] for MLP weights
- Int6 [-32,31] for attention weights
- Int4 [-8,7] nibble-packed for bigram logit table
- FP16 for tied embeddings
- zstd-22 compression

## Results
```
Steps completed: 6267 (wallclock capped at 600s)
Step time: 95.75 ms/step
Peak memory: 19609 MiB allocated, 19878 MiB reserved

Pre-SWA val_bpb: 1.1563
Post-SWA+quant val_bpb: 1.1522
Quant gap: 0.004 bpb

Artifact size: 15,322,709 bytes (int6+zstd)
Code size: 61,523 bytes
Total: 15,384,232 bytes (under 16,000,000 limit)
```

Built on the baseline by @thwu1 (PR #180). Adopts XSA from arxiv:2603.09078,
Partial RoPE and LN Scale from PR #315.
