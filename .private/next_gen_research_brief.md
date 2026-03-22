# Next-Gen Parameter Golf Script: Research Questions

## Context
We're at 1.1401 bpb (verified SOTA on merged leaderboard). PR #374 claims 1.1246 with techniques we need to understand and implement. Competition deadline: April 30, 2026.

## Questions for Research Agents

### 1. XSA (Cross-Segment Attention)
PR #374 and #379 both use "XSA on last 4 layers" and claim it's a key improvement.
- What exactly is XSA? Is this the same as cross-document attention or something else?
- How does it differ from standard causal attention?
- What's the implementation? Is it a change to the attention mask, a separate attention mechanism, or something else?
- Why only on the last 4 layers?
- How does it interact with GQA (grouped-query attention)?
- Is there a reference implementation in any of the competition PRs?

### 2. Partial RoPE (16/64 dims)
Both top PRs apply RoPE to only 16 of 64 head dimensions.
- What's the rationale? Does limiting RoPE to fewer dims help with extrapolation?
- How is this implemented? Do the remaining 48 dims use absolute positional information or nothing?
- What paper/technique is this based on?
- Does this interact with NTK-aware scaling?

### 3. Late QAT with STE
Both top PRs do "STE fake-quantization when LR scale < 0.1" — quantization-aware training in the final phase.
- What's the exact implementation of STE (Straight-Through Estimator) for int6?
- How do you add fake-quantize nodes during training? Is it `torch.fake_quantize_per_channel_affine` or custom?
- Does this work with Muon optimizer or only Adam?
- What's the training overhead (+28% step time was mentioned)?
- Can we do this JUST for the warmdown phase to minimize overhead?

### 4. Shared Value Embedding
Both top PRs mention "Shared Value Embedding (dim=128, on layers 9-10)" with per-layer learned scales.
- How does this work? Is the embedding table reused as an additional value projection?
- What's the architecture change in the attention layer?
- How many additional parameters does this add?
- Why only on the last 2 layers?

### 5. LN Scale Factor 1/sqrt(layer_idx+1)
- Is this applied to the output of each block (like a residual scaling)?
- Or is it a modification to the RMSNorm itself?
- What's the theoretical justification?
- Is this related to muP (maximal update parameterization)?

### 6. GPTQ-lite Clip Percentile Search
PR #379 mentions per-layer optimal clip percentile search during int6 quantization.
- How does this work? Try N clip ratios per weight matrix, pick the one minimizing reconstruction error?
- What's the search space? How many candidates?
- Does it require a calibration dataset or just the weight statistics?
- What's the wall-clock cost of this search? (It's post-training, so it's "free" in the 10-min budget)

### 7. Tight SWA (scale < 0.2, last ~600 steps)
PR #374 achieves "zero SWA penalty" by only averaging checkpoints in the very final phase.
- What's the exact trigger? `swa_start_frac = 0.2` instead of our 0.5?
- How many checkpoints get averaged? (~600 steps / swa_every=50 = ~12 checkpoints)
- Our SWA with warmdown=3000 on 7400 steps starts at step 4400 and averages ~60 checkpoints. Is that too many?

### 8. U-Net Skip Connections for 11L
PR #374 uses "5 encoder, 6 decoder" with skip connections.
- Our 9L model already has U-Net skips (from PR #162). How do we extend this to 11L?
- Is the encoder/decoder split always floor(L/2) encoder + ceil(L/2) decoder?
- What happens to skip weights when we go from 9L to 11L?

### 9. Logit Softcap 30.0
Both top PRs use logit softcap = 30.0.
- Our model already uses this. Confirm it's `softcap * tanh(logits / softcap)`.
- Is there any benefit to tuning this value?

### 10. Fitting 11L under 16MB without int4
PR #374 fits 11L with "int6 (MLP+attention), int8 (embeddings), zstd-22" at ~15.7MB.
- Our 11L int6+zstd produces 19.1MB. How do they achieve 15.7MB?
- Is their int6 implementation different from ours?
- Do they use a custom serialization format instead of torch.save?
- Could Late QAT be the key? (QAT-trained weights may compress better)
