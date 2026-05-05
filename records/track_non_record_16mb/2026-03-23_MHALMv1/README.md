# MHALM — Multi-Head Atlas Language Model

## Summary

MHALM is a language model that replaces the standard output projection with **multiple mathematical kernel heads**. Instead of a single linear layer mapping hidden states to vocabulary logits, five kernel heads — each using a different similarity measure — produce independent predictions that are combined by a learned mixer. The model was built over a weekend by adapting the [Intrinsic Green's Learning (IGL)](https://quemy.info/2026-03-21-intrinsic-greens-learning.html) framework, originally designed for manifold geometry, to language modeling. It works, but is not competitive with optimised transformer variants.

For a detailed writeup of the submission, see [MHALM: Parameter Golf Submission](https://quemy.info/2026-03-22-mhalm-parameter-golf.html).

### Key result

| Metric | Value |
|--------|-------|
| **Competition bpb** | **1.4574** |
| Val loss (nats) | 2.4607 |
| Artifact size | 10.8 MB / 16 MB |
| Stored params | 13.6M |
| Training steps | 6,857 |
| Training time | 594s on 8×H100 |
| Step time (compiled) | 87 ms/step |
| SWA checkpoints | 201 |

## Architecture

### Overall flow

```
tokens → Embedding (V=1024, d=512) + BigramHash (10240 buckets)
       → HybridAtlasBlock 0
       → U-Net skip connection (encoder outputs from Block 0 feed into Block 1)
       → HybridAtlasBlock 1
       → Output projection (weight-tied with embedding) → logits
```

### Inside each HybridAtlasBlock

**Encoders.** Three independent MLPs map the input to intermediate representations. Each has a different width, matched to the complexity of the kernel it feeds: H=700 (Spherical), H=256 (Gabor), H=384 (Laplacian). All produce 128-dimensional outputs.

**Five kernel heads.** Each head computes a different similarity measure and produces vocabulary logits through a learned readout matrix:

| Head | What it does |
|------|-------------|
| **Spherical** | Cosine similarity to 256 learned landmark positions (128 global + 128 local). Closest in spirit to attention — compares tokens to reference points. Row-sum normalised. |
| **Gabor** | Localised oscillatory patterns (Gaussian window × cosine) around 128 learned anchor points. Captures frequency-like structure. |
| **Laplacian** | Smooth proximity via radial basis functions around 128 learned anchor points. Captures which tokens are "nearby" in representation space. |
| **Tucker** | Element-wise product of Gabor × Laplacian outputs. Captures conjunctions ("oscillatory AND proximate") with no extra parameters. |
| **Linear** | Raw pass-through — no kernel, just a linear readout from the input. A simple baseline that turns out to be the single most valuable head. |

**Mixer.** A learned softmax-weighted combination of all five heads' logits, with a soft cap to prevent any single head from dominating.

**Temporal processing.** The mixed signal passes through:
1. A **ComplexSSM** (state-space model with complex eigenvalues, parallel scan) for long-range context
2. **2× causal self-attention** (RoPE, 8 heads, query gain) for local token interactions

### Training

All parameters trained end-to-end with Muon (for encoder matrices) + AdamW (for everything else). Stochastic Weight Averaging (SWA) over the last 40% of training. Single forward-backward pass per step.

## Running

```bash
# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train on 8×H100 (golf submission)
torchrun --standalone --nproc_per_node=8 train_gpt.py --mode golf \
    --data-dir ../../data/datasets/fineweb10B_sp1024/ \
    --tokenizer-path ../../data/tokenizers/fineweb_1024_bpe.model

# Single GPU test
torchrun --standalone --nproc_per_node=1 train_gpt.py --mode golf \
    --data-dir ../../data/datasets/fineweb10B_sp1024/ \
    --tokenizer-path ../../data/tokenizers/fineweb_1024_bpe.model

# Quick smoke test (synthetic data, 50 steps)
python train_gpt.py --mode smoke
```

## Ablations

Experiments run during development on FineWeb (V=1024). Each row is a single change from the final configuration.

| Experiment | Result | Decision |
|------------|--------|----------|
| Learned readout vs analytical solve | 3.40 vs 3.58 nats @500 steps | Learned wins |
| 1 block vs 2 blocks | 2.97 vs 3.24 nats @1500 steps | 2 blocks wins |
| Remove each kernel head | Each contributes 0.04–0.80 nats | All kept |
| Normalise Spherical head by row sum | −0.20 nats | Adopted |
| Normalise Laplacian head by row sum | +0.07 nats (worse) | Rejected |
| Increase encoder output dim (8 → 128) | −0.030 bpb, widening gap over training | Adopted |
| BigramHash 10,240 vs 4,096 buckets | Competition standard | Adopted |
| 3 attention layers instead of 2 | +0.009 bpb (worse) | Rejected |
