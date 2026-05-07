# Distill + IntraLoop: SP1024 9x512

## Result

| Metric | Value |
|--------|-------|
| **Post-quant roundtrip val_bpb** | **1.19421634** |
| Pre-quant val_bpb | 1.1871 |
| Quantization penalty | 0.0071 |
| Artifact size (int8+zstd) | 15,624,290 bytes |
| Budget headroom | 375,710 bytes |
| Training time | 600s (wallclock cap) |
| Steps completed | 10,577 / 20,000 |
| Tokens seen | ~5.54B |

Compared to the Naive Baseline (1.2244), this is a **0.030 BPB improvement**.

---

## Techniques

### 1. Partial Depth Recurrence ("Intra-Loop")

Rather than full recurrence (which shrinks the model to fit the parameter budget), we loop only a subset of layers. Layers 3 and 4 (0-indexed, out of 9 physical layers) execute twice each, yielding **11 effective layers from 9 physical layers** at near-zero parameter cost.

- Ablation showed middle-layer looping outperforms front-layer looping (contradicting the ILR paper arXiv:2505.01855 at this scale)
- Compute overhead: +9% per step
- BPB gain measured at 4k fixed-step ablation: **-0.005 BPB** vs no loop

Configuration: `INTRA_LOOP_START=3 INTRA_LOOP_END=4 INTRA_LOOP_STEPS=2`

### 2. EMA Self-Distillation

An exponential moving average (EMA) copy of the model serves as a teacher during the final 30% of training. The student is trained with a combination of:
- Standard cross-entropy loss on the target tokens
- KL divergence against the teacher's soft predictions (temperature=2.0, weight=0.08)

The teacher is never explicitly trained; it accumulates a smoothed version of the student's weights (decay=0.999). This provides a form of regularization and knowledge consolidation, improving generalization without any external data or models.

Configuration: `DISTILL_ENABLED=1 DISTILL_WEIGHT=0.08 DISTILL_TEMP=2.0 DISTILL_START_FRAC=0.70 DISTILL_EMA_DECAY=0.999`

### 3. GPTQ Post-Training Quantization

After training completes, weights are quantized to int8 using GPTQ (Frantar et al., 2022). GPTQ uses Hessian information (collected from 128 calibration samples) to optimally compensate remaining weights for each column's rounding error via a Cholesky-based inverse Hessian solve.

- 64 weight matrices quantized
- Roundtrip penalty: only **0.0071 BPB** (from 1.1871 to 1.1942)
- Compressed with zstd for final artifact

### 4. QK-Gain Initialization

Query and key projections are scaled by a learnable per-head gain parameter initialized to 5.0 (following the leaderboard's QK-Gain technique). This amplifies attention sharpness early in training without requiring warmup, and the gain adapts during training.

Configuration: `QK_GAIN_INIT=5.0`

### 5. Stochastic Weight Averaging (SWA)

Model weight snapshots are collected periodically during training and averaged at the end. This submission averaged **282 snapshots**, smoothing out late-training oscillations and improving generalization.

### 6. Muon Optimizer

Matrix-shaped parameters (attention projections, MLP weights) use the Muon optimizer, while embeddings and scalar parameters (gains, norms) use Adam. Muon applies Newton's method in the orthogonal group, providing better conditioning for matrix parameters.

### 7. Architecture Details

- **GQA (Grouped Query Attention)**: 8 query heads, 4 KV heads. Reduces KV cache and parameter count while maintaining quality.
- **SwiGLU activation**: Gated linear unit with SiLU activation in the MLP, replacing standard GeLU. Slightly more parameters per MLP but better expressiveness.
- **Tied embeddings**: Input and output embeddings share weights, saving ~512K parameters.
- **Residual bigram head**: A rank-32 bigram matrix (prev_token -> next_token) mixed with the model's logits at inference time, providing a cheap frequency-based prior.
- **U-Net skip connections**: Encoder-decoder style skip connections between the first and second half of the layer stack.

---

## Negative Results / What Didn't Work

| Technique | Result | Notes |
|-----------|--------|-------|
| **Parallel residuals** | +0.004 BPB worse | Attn+MLP on same norm input. Gradient flow benefit didn't materialize at this scale. |
| **Ouroboros loop conditioning** | +0.005 BPB worse | Input-conditioned (scale,shift) per loop step. Controller needs more training time to learn; hurts at short schedules. |
| **Full recurrence** | Much worse | Looping ALL layers forces a tiny model (2 physical layers). Partial recurrence is strictly better. |
| **Front-layer looping** | +0.001 BPB worse | Looping layers 0-2 underperforms middle layers 3-4. |
| **Curriculum learning** | Neutral/worse | Short-to-long sequence curriculum didn't help at this training budget. |

---

## Reproduction

```bash
# From the repository root, with data already downloaded:
# python3 data/cached_challenge_fineweb.py --variant sp1024

# All winning hyperparameters are baked into train_gpt.py defaults.
# Only data paths and run ID need to be specified:
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
RUN_ID=submission_8gpu_v1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script is fully self-contained (single file, no local imports). All winning hyperparameters are baked into the defaults -- no env var overrides needed for reproduction. All configuration is still overridable via environment variables.

## Included Files

- `train_gpt.py` -- complete training script (standalone, no dependencies beyond standard PyTorch + sentencepiece)
- `train.log` -- exact training log from the 8xH100 submission run
- `submission.json` -- leaderboard metadata
- `README.md` -- this file
