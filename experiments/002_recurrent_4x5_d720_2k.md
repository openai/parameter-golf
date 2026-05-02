# Experiment 002: Depth Recurrence 4x5 at dim=720, 2K Steps

## Status: RUNNING

## Hypothesis
**Based on**: Looped Transformers (ICLR 2024), MoEUT (NeurIPS 2024), Recurrent Depth Transformer (2025)

Depth recurrence (weight sharing) will improve val_bpb vs baseline at the same step count by:
1. **More effective depth** (20 layers vs 9) provides better feature extraction
2. **Wider model** (dim=720 vs 512) captures more information per layer
3. **QAT with STE** recovers quantization loss
4. **Logit softcap 15** (from modded-nanogpt) improves training dynamics
5. **Adam eps 1e-10** improves loss by ~0.001

**Prediction**: val_bpb < 1.25 at 2000 steps (vs baseline 1.2963).

## Configuration
- **Model**: 4 unique blocks x 5 loops = 20 effective layers
- **Dimensions**: dim=720, 10 heads, 5 KV heads, 2x MLP expansion
- **Vocab**: 1024 (unchanged from baseline)
- **Logit softcap**: 15 (vs baseline 30)
- **Adam eps**: 1e-10 (vs baseline 1e-8)
- **QAT**: enabled from step 0 (STE for int8 fake quantization)
- **Warmdown**: 1200 steps (same as baseline for fair comparison)
- **Training**: 2000 iterations, 524K tokens/step
- **Eval extra loops**: 7 loops at eval time (vs 5 during training)
- **GPU**: 1x H100 prototyping
- **wandb**: disabled (TODO: enable next run)

## Parameter Budget Analysis
- 4 shared blocks at dim=720: ~14.5M params (matrices only)
- Embedding (tied): 1024 x 720 = 737K
- Per-iteration scalars (20 layers): ~115K (attn_scales + mlp_scales + resid_mixes)
- Skip weights: ~7K
- **Total**: ~15.3M params
- **Expected int8+zlib artifact**: ~14.3MB (comfortably under 16MB)

## Literature Grounding
- **Looped Transformers**: "<10% of parameters for comparable in-context learning"
- **MoEUT**: "slightly outperforms non-shared transformers" with per-layer specialization
- **Recurrent Depth**: "dynamic scaling of computational depth" via iterative latent computation
- **Per-iteration scalars**: Essential for symmetry breaking (MoEUT's key contribution)

## Changes from Baseline
1. `SharedBlock` class separates heavy weights from per-iteration scalars
2. `GPT.forward()` loops through shared blocks with per-iteration scalars
3. U-Net skip connections work across effective layers
4. `quantize_dequantize_ste()` for QAT
5. `_QAT_ACTIVE` global flag for torch.compile compatibility
6. Eval-time extra depth via `eval_num_loops`

## Results

### FAILED - Bug in eval path

| Step | train_loss | val_loss | val_bpb | train_time |
|------|-----------|----------|---------|------------|
| 100  | 3.22      | -        | -       | 44s        |
| 200  | 2.77      | -        | -       | 212s       |
| 500  | 2.50      | 4.24     | **2.5101** | 915s    |

**val_bpb = 2.51 at step 500 vs baseline 1.48** — dramatically worse!

### Root Cause
Bug in eval path: `eval_num_loops=7` caused the model to run 28 effective layers at eval time
(vs 20 during training). With `torch.compile(fullgraph=True)`, changing the loop count between
train and eval creates a graph mismatch. The U-Net skip structure at 28 layers is completely
different from 20 layers, and the extra loops use cyclically-wrapped scalars that weren't trained
for those positions.

### Fix Applied
Changed GPT.forward to always use `self.num_loops` (no train/eval split). Extra eval loops
will be a post-quantization feature only, not during regular eval.

### Additional Issue: Throughput
Step time ~1525ms vs baseline ~440ms = 3.5x slower. On 8xH100 this would yield only ~5.3K steps
in 10 min vs baseline 20K. The model would need to be 4x more efficient per step to compensate.

### Lesson Learned
1. Don't change loop count between train/eval with torch.compile
2. 20 effective layers at dim=720 is too much compute for the 10-min budget
3. Need to find the right depth/width tradeoff
