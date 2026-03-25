# FRO + Radial Token Branch + Disciplined 1024-Bucket Bigram Hash

This submission presents a compressed language-model design for the Parameter Golf 16MB track built around three interacting components:

- **FRO (Fractal Resonant Optimization)** as the main optimizer for the compressed transformer core,
- **Radial token geometry** as a lightweight token-level geometric feature branch,
- **A 1024-bucket bigram-hash branch** to provide fast short-horizon lexical context under the strict artifact budget.

This specific "Disciplined" variant achieved the definitive record result by refining the optimization trajectory and branch injection gain clamping.

## Summary

This submission uses a compressed dual-branch transformer with:

- BitNet-style ternary-weight forward behavior in major internal projections,
- split optimization between **FROStable** and AdamW,
- EMA during training,
- mixed post-training export (`int8` / `int6`) with light export-time pruning,
- a **radial token branch** derived directly from token IDs (limited to bits-representation),
- a **1024-bucket bigram-hash branch** to maximize early convergence.

The design goal is to maximize useful model capacity and fast convergence under the 16MB artifact constraint and a 10-minute wall-clock training budget.

## Author Signatures

### 1. FRO (Fractal Resonant Optimization)
FRO is the primary optimizer contribution. It is used on the main compressed transformer parameter set and modulates update strength through a multi-scale directional resonance signal between gradients and momentum.

### 2. Radial Token Geometry
The radial component is used as a **token-level geometric representation**. A lightweight radial feature is constructed from token IDs via a multi-angle mapping and projected into the model fusion space.

### 3. Bigram Hash Branch
A lightweight hashed bigram embedding branch (1024 buckets) is added to provide short-horizon lexical context at very low artifact cost. This branch dramatically improves convergence in the 10-minute regime.

## Architecture

- **Fusion Dimension:** 448
- **Branch A:** 8 layers, dim 384, 6 heads
- **Branch B:** 5 layers, dim 320, 5 heads
- **Vocabulary Size:** 1024
- **Radial Bits:** 10
- **Bigram Hash Buckets:** 1024

## Optimization (Disciplined Tuning)

The training recipe uses:

- **FROStable** (LR 9e-4, alpha 0.12, gamma 0.66) on the main compressed transformer parameters,
- **AdamW** (LR 1.4e-3) on auxiliary branches and output heads,
- Cosine decay starting at step 1000,
- EMA checkpointing (`decay=0.997`),
- Gradient clipping: `1.8`,
- Dynamic clamping for radial and hash gains to ensure stability.

## Export and Artifact Accounting

The final exported artifact uses:

- mixed `int8` / `int6` serialization,
- export-time pruning: `0.0030`,
- zlib compression.

Final audited artifact:

- **Total artifact (approx):** `15,943,179` bytes
- **PASS ✅** (within 16,000,000 byte limit)

## Observed Development Result

Observed development-run result:

- **Validation BPB:** `1.5852`

This result was obtained on Kaggle Dual T4 hardware under a 10-minute constraint.

## Reproducibility Notes

The repository entry includes:

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py` (distributed DDP support, EMA evaluation, mixed export serialization).
