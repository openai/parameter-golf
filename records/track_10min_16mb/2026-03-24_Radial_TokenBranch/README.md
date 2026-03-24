# FRO + Radial Token Branch + 1024-Bucket Bigram Hash

This submission presents a compressed language-model design for the Parameter Golf 16MB track built around three interacting components:

- **FRO (Fractal Resonant Optimization)** as the main optimizer for the compressed transformer core,
- **Radial token geometry** as a lightweight token-level geometric feature branch,
- **A 1024-bucket bigram-hash branch** to provide fast short-horizon lexical context under the strict artifact budget.

The final exported artifact remains within the official `16,000,000` byte submission limit after post-training compression and audit.

## Summary

This submission uses a compressed dual-branch transformer with:

- BitNet-style ternary-weight forward behavior in major internal projections,
- split optimization between **FRO** and AdamW,
- EMA during training,
- mixed post-training export (`int8` / `int6`) with light export-time pruning,
- a **radial token branch** derived directly from token IDs,
- a **1024-bucket bigram-hash branch** to improve early short-budget convergence.

The design goal is to maximize useful model capacity and fast convergence under the 16MB artifact constraint and a 10-minute wall-clock training budget.

## Author Signatures

### 1. FRO (Fractal Resonant Optimization)
FRO is the primary optimizer contribution in this submission. It is used on the main compressed transformer parameter set and modulates update strength through a multi-scale directional resonance signal between gradients and momentum.

### 2. Radial Token Geometry
The radial component is used as a **token-level geometric representation**, not as a positional-attention bias. A lightweight radial feature is constructed from token IDs via a multi-angle geometric mapping and projected into the model fusion space through a learnable scalar gain.

### 3. Bigram Hash Branch
A lightweight hashed bigram embedding branch (1024 buckets) is added to provide short-horizon lexical context at very low artifact cost. This branch dramatically improves convergence in the 10-minute regime while remaining compatible with the radial token branch and the compressed backbone.

## Architecture

### Compressed Dual-Branch Backbone
The model uses a compressed dual-branch transformer backbone with a shared fusion space:

- Fusion dimension: 448
- Branch A: 8 layers, dim 384, 6 heads
- Branch B: 5 layers, dim 320, 5 heads
- Vocabulary size: 1024

### BitLinear Expansion
Major internal projections use BitNet-style ternary-weight forward behavior to reduce storage pressure while preserving width and depth under the artifact budget.

### Radial Token Branch
For each token ID, a radial feature vector is built from its binary representation through a lightweight geometric transform and projected into the fusion space.

### Bigram Hash Branch
A 1024-bucket hashed bigram embedding is computed from adjacent token IDs and injected into the same fusion space through a learnable gain.

## Optimization

The training recipe uses:

- **FROStable** on the main compressed transformer parameter set,
- **AdamW** on embeddings / bridge / fusion / output parameters and auxiliary lightweight branches,
- cosine-style decay after warmup,
- EMA checkpoint tracking during training.

## Export and Artifact Accounting

The final exported artifact uses:

- mixed `int8` / `int6` serialization for selected weights,
- light export-time pruning of values below `0.0025`,
- zlib compression,
- decimal-byte accounting against the official artifact limit.

Final audited artifact:

- **Compressed model:** `15,918,192` bytes
- **Source code:** `24,987` bytes
- **Total artifact:** `15,943,179` bytes
- **Headroom:** `56,821` bytes

## Observed Development Result

Observed development-run result for this configuration:

- **Validation BPB:** `1.6130`

This result was obtained from the included development configuration and log on Kaggle Dual T4 hardware.

## Reproducibility Notes

The repository entry includes:

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`

The training script includes:

- distributed execution support,
- tokenizer-aware BPB evaluation,
- EMA evaluation,
- mixed export serialization,
- final compressed artifact measurement.

## Notes

This submission represents the definitive development result of the author's **FRO + radial token branch + bigram-hash branch** line under the 16MB track constraints.
