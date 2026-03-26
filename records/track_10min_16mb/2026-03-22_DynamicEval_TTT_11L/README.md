# Record: Dynamic Evaluation on SOTA Pipeline (val_bpb = 1.1364, 3-seed mean 1.1371)

## Summary

We apply dynamic evaluation (Krause et al., ICML 2018) to the current SOTA pipeline without modifying training. The model takes periodic gradient steps during sliding window scoring, adapting to local text distribution. This produces a consistent 2.0% bpb improvement across 3 seeds at zero artifact cost. 3-seed mean: **1.1371 bpb** (merged SOTA: 1.1428).

## Results (3-seed, 8xH100 SXM, SDPA backend)

| Seed | Steps | Int6 Roundtrip | + TTT + Dynamic Eval | Delta | Artifact |
|------|-------|----------------|----------------------|-------|----------|
| 42   | 5,604 | 1.1607         | **1.1364**           | -0.0243 | 15.65 MB |
| 7    | 5,590 | 1.1618         | **1.1369**           | -0.0249 | 15.80 MB |
| 2024 | 5,620 | 1.1613         | **1.1380**           | -0.0233 | 15.35 MB |
| **Mean** | | **1.1613** | **1.1371** | **-0.0242** | |

Note: FA3 (`flash_attn_interface`) is not available on the official RunPod template ([issue #280](https://github.com/openai/parameter-golf/issues/280)). All results use PyTorch SDPA.

## Novel Contribution: Dynamic Evaluation

After training and TTT adaptation, we score the validation stream using sliding windows (stride=64). Between batches of scored windows, we take an SGD gradient step on the model weights using one window's loss. The model adapts to the local distribution of the text — a passage about physics shifts the model's predictions toward that domain's token distribution before the next passage is scored. Later windows benefit from all preceding adaptation, so the effect gets stronger as eval progresses.

To our knowledge, no prior submission has applied dynamic evaluation. TTT (PR #254, #338) adapts weights *before* scoring; dynamic eval adapts weights *during* scoring. The two techniques are complementary.

**Implementation.** Windows scored in batches of 32 (no gradient, fast). Every 4 batches, one gradient step on a single window (SGD, lr=0.001). Each of 8 DDP ranks processes its partition independently — gradient steps are rank-local. Final loss/byte counts are all_reduced.

**Hyperparameters.** lr=0.001 was selected based on initial experiments; we did not perform a systematic sweep. Higher learning rates may yield larger gains. `adapt_every=4` balances adaptation strength against eval speed. SGD without momentum is used so each step reacts to the current window rather than carrying signal from earlier, unrelated text.

**Cost.** Zero additional artifact bytes. ~344s eval time on 8xH100 SXM.

**Reference.** Krause, B., Kahembwe, E., Murray, I., & Renals, S. (2018). Dynamic Evaluation of Neural Sequence Models. ICML 2018.

## Adopted Techniques (SOTA-ADOPT)

| Technique | Source |
|-----------|--------|
| 11L, dim=512, 8 heads, 4 KV heads, MLP 3x | PR #315 (jfprincz) |
| XSA (Exclusive Self Attention), last 4 layers | PR #315 |
| EMA (decay=0.997) | PR #315 |
| Partial RoPE (16 dims), LN Scale | PR #315 |
| Late QAT + Int6 + zstd | PR #315 |
| SmearGate, BigramHash, OrthoInit | PR #315 (originally by unnir) |
| TTT: full-weight SGD, lr=0.002, 3 epochs, freeze 2 blocks | PR #338 (alertcat) |
| Muon + Adam optimizer | Community |

## Ablation

Seed 42, 8xH100 SXM:

| Configuration | val_bpb | Delta |
|--------------|---------|-------|
| Int6 roundtrip (non-overlapping) | 1.1607 | baseline |
| + TTT + Dynamic eval (sliding, stride=64) | 1.1364 | **-0.0243** |

We were unable to isolate TTT's individual contribution in this run due to an inference-mode tensor incompatibility when skipping TTT. alertcat's PR #338 measured TTT as ~0.002 bpb improvement on this base. Subtracting that, dynamic eval contributes approximately 0.022 of the total 0.024 improvement.

## Eval Time Budget (8xH100 SXM)

| Phase | Time |
|-------|------|
| TTT (3 epochs, lr=0.002) | ~83s |
| Dynamic eval sliding window (stride=64) | ~344s |
| **Total eval** | **~427s** |

Standard sliding window eval is skipped when dynamic eval is enabled — dynamic eval already performs sliding window scoring with identical stride and byte accounting.

## What Didn't Work

**N-gram interpolation on BPE tokens.** We built a trigram frequency table (506K entries, 2.61 MB) and interpolated model logprobs with n-gram logprobs. Result: -0.56% (hurt performance). The 1024-token BPE vocabulary already captures bigram structure. N-gram tables add noise, not information.

**TTT at lr=1e-4.** Our initial TTT used a learning rate 20x lower than FarnsworthEngine's proven lr=0.002. Zero measurable effect. The technique works; we had the wrong hyperparameters.

**Depth recurrence.** We tested weight sharing (6 blocks x 2 loops, 12 effective layers). Two findings: (1) Vanilla sharing produces degenerate recurrence. The model's `resid_mix` weights showed blocks actively suppressing the second pass, with no mechanism to distinguish loop iterations. (2) Per-loop LoRA adapters (Bae et al., ICLR 2025) fix this, but we discovered a critical bug: 3D LoRA tensors fell through both optimizer parameter groups, receiving zero updates. After fixing, LoRA showed crossover at step 1500, but `fullgraph=False` imposed a 2.3x step penalty, making recurrence non-competitive at fixed wallclock. Independently corroborated by PR #363.

## Reproduction

```bash
# 8xH100 SXM, ~600s training + ~427s eval
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 \
DYNEVAL_ENABLED=1 DYNEVAL_LR=0.001 DYNEVAL_ADAPT_EVERY=4 \
MUON_WD=0.042 ADAM_WD=0.042 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Attribution

This submission builds on:
- **jfprincz** — PR #315 (1.1248): XSA, EMA, Partial RoPE, LN Scale, Late QAT
- **unnir** — SmearGate, BigramHash, OrthoInit (PRs #102, #135)
- **alertcat** — PR #338 (1.1254): TTT integration on the PR #315 base
- **thwu1** — Merged SOTA (1.1428): The stacking meta baseline

Reference: Krause et al., "Dynamic Evaluation of Neural Sequence Models," ICML 2018.

## Future Work

Dynamic eval hyperparameters were not swept. We expect higher learning rates and more frequent adaptation steps to yield further gains. Combining dynamic eval with stronger base architectures (e.g., PR #374) is a natural next step.
