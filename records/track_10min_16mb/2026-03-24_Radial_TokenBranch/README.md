# Radial Token Branch + FRO

This submission presents a compressed language-model design for the Parameter Golf 16MB track built around two original components by the author:

- **FRO (Fractal Resonant Optimization)** as the primary optimizer,
- **Radial token geometry** as a lightweight representational branch.

The final model remains within the official `16,000,000` byte artifact limit after post-training export and compression.

## Summary

This submission uses a dual-branch compressed transformer with:

- BitNet-style ternary-weight linear behavior in major internal projections,
- split optimization between a custom optimizer (**FRO**) and AdamW,
- EMA during training,
- mixed post-training export (`int8` / `int6`) with light export-time pruning,
- a **radial token branch** that injects geometric features derived from token IDs.

The design goal is to maximize useful model capacity under the strict artifact limit while preserving fast convergence under a 10-minute wall-clock budget.

## Author Signatures

### 1. FRO (Fractal Resonant Optimization)
FRO is the primary optimization contribution in this submission.  
It replaces a standard AdamW-only setup for the majority of parameters and uses a multi-scale directional resonance signal between gradients and momentum to adapt update strength under short-horizon training constraints.

### 2. Radial Token Geometry
The radial component is used here as a **token-level geometric feature branch**, not as a positional-attention bias.

A lightweight radial encoding is computed directly from token IDs using a phi-scaled multi-angle construction, projected into the model fusion space, and blended through a small learnable gain. In this configuration, the radial branch is retained by training and contributes positively instead of collapsing to zero.

## Architecture

### Backbone
- Dual-branch compressed transformer
- Fusion dimension: 448
- Branch A: 8 layers, dim 384, 6 heads
- Branch B: 5 layers, dim 320, 5 heads
- Vocabulary size: 1024

### Radial Token Branch
For each token ID, a radial feature vector is constructed from its binary representation using a geometric mapping over multiple angular components. The resulting low-dimensional radial feature is projected into the fusion space and added to the token embedding through a learnable scalar gain.

## Optimization

The training recipe uses:

- **FRO** on the main compressed transformer parameter set,
- **AdamW** on embedding / bridge / fusion / output parameters and the radial token branch,
- cosine-style decay after warmup,
- EMA checkpoint tracking during training.

## Export and Artifact Accounting

The final exported artifact uses:

- mixed `int8` / `int6` serialization for selected weights,
- light export-time pruning of very small values,
- zlib compression,
- decimal-byte accounting against the official artifact cap.

Final audited artifact:

- **Compressed model:** `15,086,685` bytes
- **Estimated code bytes:** `47,000`
- **Total artifact:** `15,133,685` bytes
- **Headroom:** `866,315` bytes

## Observed Development Result

Observed development-run result for this configuration:

- **Validation BPB:** `1.8716`

This result was obtained from the included development configuration and log. The submission is presented as the current best version of this architecture line.

## Reproducibility Notes

The repository entry includes:

- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`

The training script includes:

- distributed execution support,
- explicit post-training artifact audit,
- tokenizer-aware BPB evaluation,
- mixed export serialization,
- final compressed artifact measurement.

## Notes

This submission should be read as the current best compressed-model result of the author’s **FRO + radial token branch** design line under the 16MB track constraints.
