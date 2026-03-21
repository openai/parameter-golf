# Ablation Runbook v2 — Execute in Order

When you get 8xH100 access, follow this EXACTLY. Budget: ~7 runs in $25 credit.

## Shared base config (NEVER change these between runs)

```bash
BASE="NUM_LAYERS=11 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64"
```

## Phase 1: Reproduce SOTA baseline (1 run)

Match PR #287 exactly (XSA + EMA, no TTT). Expect: ~1.1271

```bash
eval $BASE \
BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
TTT_ENABLED=0 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Decision gate:** If this gives > 1.130, STOP and debug. If close to 1.127, proceed.

## Phase 2: Add vanilla TTT (1 run)

Simplest TTT on top of #287's base. Expect: ~1.120-1.124

```bash
eval $BASE \
BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 \
TTT_FREEZE_BLOCKS=2 TTT_SURPRISE_WEIGHT=0 TTT_CURRICULUM_LR=0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Decision gate:** If TTT helps (BPB < 1.1271), TTT is validated. Proceed with enhancements.
If TTT hurts, try TTT_LR=0.001 and TTT_EPOCHS=2.

## Phase 3: Best TTT config (1 run)

Add our novel TTT improvements + temperature search. DISABLE surprise weighting (PR #294 evidence).

```bash
eval $BASE \
BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 \
TTT_FREEZE_BLOCKS=2 TTT_SURPRISE_WEIGHT=0 \
TTT_COSINE=1 TTT_LOSS_CLIP=5.0 TTT_ACCUM=4 \
TTT_EMA_DECAY=0.95 TTT_LABEL_SMOOTH=0.05 TTT_L2_ANCHOR=0.001 \
TTT_UNFREEZE_SKIPS=1 \
TEMP_SEARCH=1 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Phase 4: Add eval-time enhancements (1 run)

Full stack: OLB + freq bias + everything.

```bash
eval $BASE \
BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 \
TTT_FREEZE_BLOCKS=2 TTT_SURPRISE_WEIGHT=0 \
TTT_COSINE=1 TTT_LOSS_CLIP=5.0 TTT_ACCUM=4 \
TTT_EMA_DECAY=0.95 TTT_LABEL_SMOOTH=0.05 TTT_L2_ANCHOR=0.001 \
TTT_UNFREEZE_SKIPS=1 \
TEMP_SEARCH=1 \
FREQ_BIAS_WEIGHT=0.05 CONF_GATE_FREQ=1 \
OLB_LR=0.1 OLB_MOMENTUM=0.9 OLB_WARMUP_STRIDE=512 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Phase 5: Try bigger BigramHash (1 run)

If artifact from Phase 1 is < 15.5MB, try 10240.

```bash
# Same as best config from Phase 3 or 4, but:
BIGRAM_VOCAB_SIZE=10240
```

**CHECK:** artifact must be < 16MB (16,777,216 bytes). If over, abort and use 2048.

## Phase 6: Multi-seed validation (2 runs)

Take the best config from above. Run with SEED=42 and SEED=2025.

## Decision tree

| If... | Then... |
|-------|---------|
| Phase 1 > 1.130 | Debug base code, don't proceed |
| Phase 2 TTT hurts | Try lower LR (0.001) and fewer epochs (2) |
| Phase 3 worse than Phase 2 | TTT enhancements hurt — submit Phase 2 config |
| Phase 4 worse than Phase 3 | OLB/freq hurt — submit Phase 3 config |
| BigramHash(10240) > 16MB | Use 2048 |
| Best BPB < 1.1271 | Submit! Multi-seed validate first |

## Priority ranking of techniques

1. XSA + EMA training (proven SOTA)
2. Vanilla TTT (proven by multiple PRs)
3. Temperature search (principled, zero risk)
4. TTT loss clipping (evidence-based)
5. TTT cosine annealing (standard optimization)
6. TTT EMA (stabilization)
7. OLB (novel, highest risk/reward)
8. Frequency bias (low risk, possibly redundant with OLB)
