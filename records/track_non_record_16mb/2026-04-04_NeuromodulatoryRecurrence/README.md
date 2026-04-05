# Neuromodulatory Depth-Recurrent Transformer: Resolving the TTT-Recurrence Conflict via Targeted Parameter Adaptation

**val_bpb: 1.3151** (sliding window, 1xH100 80GB) | **~12.87 MB** | 4000 steps

## Summary

We introduce a depth-recurrent transformer that shares weight banks across select layers and disambiguates their function via learnable FiLM (Feature-wise Linear Modulation) conditioning vectors. Built on top of the PR #549 SOTA stack (LeakyReLU^2, Parallel Muon, XSA, BigramHash, EMA), our architecture reduces the physical block count from 11 to 9 while maintaining 11 virtual layers through partial weight sharing. This frees 4.7M parameters and yields a -0.021 BPB improvement over baseline at 500 iterations on matched hardware. We propose FiLM-only TTT as a solution to the documented TTT-recurrence conflict: at test time, only the modulatory vectors (not shared weights) are updated, preventing gradient compounding across shared layers. FiLM-only TTT was implemented but crashed due to a tensor comparison bug before credits ran out; the fix is identified and a rerun is pending.

## Motivation: Cortical Column Reuse and Neuromodulation

The mammalian neocortex reuses the same six-layer cortical column circuit approximately 150,000 times across functionally distinct regions. Visual cortex, motor cortex, and prefrontal cortex all share the same canonical microcircuit, yet they process fundamentally different information. The difference is not in the wiring but in neuromodulation: acetylcholine, dopamine, noradrenaline, and serotonin modulate how cortical circuits process information without altering the circuits themselves.

This maps directly to our architecture:
- **Shared transformer blocks** = cortical columns (same circuit, reused)
- **FiLM conditioning vectors** = neuromodulatory signals (per-region modulation)
- **FiLM-only TTT** = adjusting neuromodulatory tone rather than rewiring synapses (fast, targeted adaptation)

The key insight is that biological brains achieve functional diversity through modulation of shared circuits, not through maintaining entirely separate circuits for each function. This is precisely what depth recurrence with FiLM conditioning achieves in a parameter-constrained setting.

## The Problem: TTT Breaks Depth Recurrence

Test-time training (TTT) updates model weights using gradient descent on already-scored validation data. When weights are shared across multiple virtual layers (depth recurrence), gradient updates compound: a single SGD step on a shared weight matrix effectively applies the same update multiple times during forward propagation. This is analogous to administering a systemic drug when you wanted a targeted local effect.

The loveless2001 submission documented this conflict empirically. Our FiLM-only TTT is proposed as a solution: update only the per-loop conditioning vectors, which are unique to each virtual layer and do not compound.

## Architecture

Built on PR #549 SOTA stack with the following modifications:

| Component | SOTA (PR #549) | Ours |
|-----------|---------------|------|
| Physical blocks | 11 | 9 |
| Virtual layers | 11 | 11 |
| Weight sharing | None | Blocks 3-4 share, Blocks 9-10 share |
| FiLM conditioning | None | 4 scale/shift pairs (shared layers) |
| Parameters | 26.9M | 22.2M |
| TTT target | All weights | FiLM vectors only (shared blocks) |

### Recurrence Layout

```
Virtual layer:  0  1  2  3  4  5  6  7  8  9  10
Physical block: 0  1  2  3  3  4  5  6  7  8   8
Bank index:     0  1  2  3  3  4  5  6  7  8   8
FiLM applied:   -  -  -  Y  Y  -  -  -  -  Y   Y
```

Blocks 3 and 8 are each executed twice with different FiLM conditioning:
- `x = scale_i * x + shift_i` applied after each shared block execution
- Scales initialized to 1.0, shifts to 0.0
- FiLM parameters optimized by Adam (scalar optimizer), not Muon

### Preserved SOTA Features

- LeakyReLU(0.5)^2 activation
- XSA on last 4 virtual layers (physical blocks 6, 7, 8)
- BigramHash(1536)
- EMA(0.997) + SWA(every 50, when scale < 0.2)
- Partial RoPE (16/64 dims)
- LN Scale (1/sqrt(layer+1))
- Value Embedding (VE128) on virtual layers 9, 10
- Int6 QAT + GPTQ-lite + lzma compression
- Sliding window eval (stride=64)
- Parameter Banking + Parallel Muon optimizer
- U-Net encoder/decoder with skip connections

## Results

### Ablation at 500 iterations (1x RTX 4090, TRAIN_BATCH_TOKENS=524288)

| Experiment | Config | Params | val_bpb | Delta |
|-----------|--------|--------|---------|-------|
| Exp 0 | SOTA baseline (PR #549) | 26.9M | 1.7075 | -- |
| Exp 1 | Recurrence + FiLM | 22.2M | **1.6864** | **-0.0211** |
| Exp 3 | FiLM only (no recurrence) | 26.9M | 1.7446 | +0.0371 |

The improvement comes from depth recurrence (parameter sharing), not FiLM alone. FiLM on all layers without recurrence hurts performance, confirming that the conditioning vectors serve their intended purpose of disambiguating shared blocks rather than providing general-purpose modulation.

### Full training (1x H100 80GB, 4000 iters, TRAIN_BATCH_TOKENS=786432)

| Metric | Value |
|--------|-------|
| Raw val_bpb (step 4000) | 1.3410 |
| EMA val_bpb | 1.3134 |
| Int6 quantized val_bpb | 1.3371 |
| **Sliding window val_bpb** | **1.3151** |
| Artifact size | 12.87 MB |
| Step time | 792 ms |
| Total training time | 53 min |

### FiLM-only TTT (not completed)

TTT was implemented with the following design:
- Unique blocks: full weight TTT (standard behavior)
- Shared blocks: freeze block parameters, update only FiLM scale/shift vectors
- FiLM vectors always unfrozen and included in TTT optimizer

The TTT eval crashed with `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` due to using `if p not in ttt_params` (element-wise tensor comparison) instead of `if id(p) not in ttt_param_ids`. The fix is one line. A rerun would likely yield an additional -0.002 to -0.003 BPB improvement based on SOTA TTT gains.

## Discussion

**What worked:** Partial depth recurrence with FiLM conditioning improves BPB while using 17% fewer parameters. The freed parameters allow the model to train faster per step (fewer bank weights to update) without sacrificing effective depth. The U-Net skip connections appear to interact well with shared blocks, as the encoder-decoder structure provides distinct gradient paths for different iterations of the same block.

**What surprised us:** FiLM conditioning on all layers without recurrence (Exp 3) hurts performance significantly (+0.037 BPB). We expected it to be neutral or slightly positive. This suggests that per-layer modulation adds optimization noise when the layers already have unique weights. The modulation is only beneficial when it serves a disambiguation purpose.

**What we could not test:** Due to GPU credit constraints, we could not: (a) complete the TTT eval to verify FiLM-only TTT resolves the recurrence-TTT conflict, (b) try different recurrence patterns (e.g., sharing only MLP weights, or 3 shared blocks instead of 2), (c) reinvest freed parameters into wider layers rather than keeping model_dim=512.

## Limitations

- Trained on 1x H100, not 8x H100 SXM. Scores are not directly comparable to the leaderboard.
- Only 4000 training iterations (vs ~7200 on 8xH100 within 10 min).
- FiLM-only TTT was not successfully evaluated due to a bug + credit exhaustion.
- Limited hyperparameter search. The recurrence pattern (which blocks to share) was chosen heuristically, not searched.
- The sliding window BPB of 1.3151 is measured on 1xH100; on 8xH100 with full TTT the result would differ.

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
RECUR_ENABLED=1 FILM_ENABLED=1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=4000 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## References

- Universal Transformer (Dehghani et al., 2019)
- FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., 2018)
- Doya, K. (2002). Metalearning and neuromodulation. Neural Networks.
- PR #549: LeakyReLU^2 + Legal TTT + Parallel Muon (abaybektursun)
- PR #461: TTT recipe (Christopher-Lee-McClendon)
- PR #399: Parameter Banking + Parallel Muon (abaybektursun)
- PR #414: Base model stack (signalrush)
- PR #493: LeakyReLU^2 activation (parinzee)

## Credits

This submission is built entirely on top of PR #549 by @abaybektursun, which itself integrates work from PRs #399, #414, #461, and #493. The novel contribution is the depth recurrence pattern, FiLM conditioning, and FiLM-only TTT design. Submission by Nir Mathur (@nirmathur), medical student at King's College London, applying neurophysiology principles to ML architecture design.
