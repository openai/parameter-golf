# Parameter Golf — Definitive Model Specs

**Single source of truth for all models.** Updated after every test.
**Last Updated:** 2026-03-25 23:15 CDT

---

## Model 1: Codec 🔥 CURRENT LEADER

**Concept:** Dictionary + n-gram context + transformer. Bigram embedding feeds into transformer, unigram projection on output.
**Nature analog:** Audio codec — compress signal into learned dictionary, reconstruct with context.

| Field | Value |
|-------|-------|
| Best score | **val_bpb 2.63** (Nick manual iteration) |
| Params | ~17M |
| Compressed size | 8.0 MB |
| Steps built | 5/5 — COMPLETE |
| Latest file | `train_gpt_m1_step5.py` |
| Original spec | `specs/model1-codec.md` |
| Status | **COMPLETE — BEST MODEL** |

**Architecture:** Bigram embedding (token pair lookup) → standard transformer blocks → unigram log-prob projection. The n-gram context gives the model pre-computed statistical priors that the transformer refines.

**What makes it special:** The n-gram features are essentially "free information" — bigram/unigram statistics computed from the training data feed directly into the model, giving it a strong prior before the transformer even starts reasoning.

**Known issues:** 8.0MB compressed — over half the 16MB budget. Leaves less room for scaling.

---

## Model 2: Recursive (Shared Weights)

**Concept:** One transformer block applied N times (weight sharing). Same weights, different layer position embeddings.
**Nature analog:** Fractal / recursive function — simple rule applied repeatedly creates complex behavior.

| Field | Value |
|-------|-------|
| Best score | 4.01 loss (9×512d) |
| Params | ~5.7 MB compressed |
| Steps built | 3/3 — COMPLETE |
| Latest file | `train_gpt_m2_step3.py` |
| Original spec | `specs/model2-recursive.md` |
| Status | **COMPLETE** |

**Architecture:** Single transformer block (512d, 8 heads) applied 9-12 times. Each application gets a different position signal so the model can differentiate "which iteration am I on." Dramatically fewer unique parameters than a standard model.

**What makes it special:** Extreme parameter efficiency. One block does the work of 9-12 blocks. Compresses very small.

**Known issues:** OOM at 12×640d and 12×768d on 4070 Super (needs H100). Loss plateaus earlier than non-shared models.

---

## Model 3: Hybrid (GatedRNN + Attention)

**Concept:** First 3 layers are GatedRNN (recurrent), remaining 6 layers are standard attention. Combines recurrent efficiency with attention's long-range capability.
**Nature analog:** Brain stem (fast reflexes/RNN) + cortex (slow deliberation/attention).

| Field | Value |
|-------|-------|
| Best score | **val_bpb 2.529** (213 steps, 30s on 4070 Super) |
| Params | ~17M |
| Compressed size | ~5.1 MB |
| Steps built | 3/3 — COMPLETE |
| Latest file | `train_gpt_m3_step3.py` |
| Original spec | `specs/model3-hybrid.md` |
| Status | **COMPLETE** |

**Architecture:** 3 GatedRNN layers (fast local patterns) → 6 attention layers (long-range dependencies). GatedRNN has gating mechanism similar to LSTM but simpler. Warmdown 3500, grad clip 0.3.

**What makes it special:** Non-transformer layers competing with pure attention. If this scores well at scale, it's a genuine architectural finding.

**Known issues:** GatedRNN placement matters — must be first layers, not last. RNN layers are sequential (less parallelizable on 8×GPU).

---

## Model 4: Optimized Transformer

**Concept:** Take the baseline `train_gpt.py` and apply every known optimization technique incrementally.
**Nature analog:** None — pure engineering optimization.

| Field | Value |
|-------|-------|
| Best score | 3.83 bpb (roundtrip, 200 steps on 4070 Super) |
| Params | ~17M (5.1MB compressed), ~62M at 768d (OOM) |
| Compressed size | 5.1 MB |
| Steps built | 15/15 — COMPLETE |
| Latest file | `train_gpt_step15.py` (or `train_gpt_model4.py`) |
| Original spec | `specs/model4-optimized-transformer.md` |
| Status | **COMPLETE** |

**Techniques applied (15 steps):**
1. 11 layers (up from 9)
2. 3x MLP expansion
3. EMA (decay=0.997)
4. Warmdown 3500 iterations
5. Sequence length 2048
6. Batch 786,432 tokens
7. Gradient clipping 0.3
8. Muon weight decay 0.04
9. BigramHash 2048 buckets
10. SmearGate (current/previous token blending)
11. Sliding window eval stride=64
12. GPTQ-lite int6/int8 + zstd-22
13. PolarQuant (code written, NOT wired)
14. Scale to fill 16MB (OOM on 4070)
15. TTT + curriculum learning

**What makes it special:** Maximum technique stacking. Public submission candidate for leaderboard visibility.

**Known issues:** 12L/768d OOMs on 4070 Super (needs H100). PolarQuant not integrated. Full 10-min score unknown.

---

## Model 5: Frankenstein

**Concept:** Take the best-performing techniques from ALL models and combine them into one.
**Status:** NOT STARTED — built after M6-M8 are tested.

---

## Model 6: Hive (Frozen Backbone + LoRA)

**Concept:** 90% of params frozen (random orthogonal init, never trained). Only 10% trainable via LoRA adapters. The frozen weights act as a fixed random projection / feature extractor.
**Nature analog:** Bee brain — 90% hardwired pattern detectors, 10% plastic learning.

| Field | Value |
|-------|-------|
| Best score | 5.36 train_loss @ 200 steps (30s smoke test, no val) |
| Params | 17M total (16.5M frozen, 545K trainable, 96.8% frozen) |
| Steps built | 2/3 |
| Latest file | `train_gpt_m6_step2.py` |
| Original spec | `specs/model6-hive.md` |
| Status | **IN PROGRESS — Step 3 needed** |

**Architecture:** Standard transformer backbone with ALL attention and MLP weights frozen after random orthogonal init. LoRA adapters (low-rank) at each layer are the only trainable parameters. The frozen backbone is a random feature space that the LoRA layers learn to read.

**Step 2 adds:** Warmdown 3500, grad clip 0.3, EMA 0.997.
**Step speed:** 114ms/step (fastest — less gradient computation from frozen params).

**Step 3 plan:** Tune LoRA rank, tune frozen/trainable ratio, add router warmup.

**Known issues:** No int4 quantization of frozen backbone yet (original spec calls for it). No gating/routing mechanism yet.

---

## Model 7: Immune System / Template Codebook

**Concept:** Library of 16 small weight templates shared across all layers. Per-token router selects and mixes templates to generate effective representations. Like V(D)J recombination.
**Nature analog:** Immune system — 400 gene segments combine to create millions of unique antibodies.

| Field | Value |
|-------|-------|
| Best score | 4.49 train_loss @ 200 steps (30s smoke test, no val) |
| Params | 17M |
| Steps built | 2/3 |
| Latest file | `train_gpt_m7_step2.py` |
| Original spec | `specs/model7-immune.md` |
| Status | **IN PROGRESS — Step 3 needed** |

**Architecture:** Standard transformer + `TemplateCodebook` module. 16 templates (512d each), router (linear layer) produces softmax weights per token, mixed templates added to hidden state. Applied after the transformer blocks.

**Step 2 adds:** Warmdown 3500, grad clip 0.3, EMA 0.997.

**Step 3 plan:** Router warmup, monitor template utilization, try different template counts (8, 16, 32).

**Known issues:** High initial loss spikes (steps 2-3 spike to 17.5 before dropping). Router collapse risk (always picking same templates). Original spec called for int6 quantization — not implemented.

---

## Model 8: Crystal (Seed + Growth Rules)

**Concept:** One small "seed" block + a growth rule network that generates per-layer modifications. The architecture's shape encodes learned knowledge.
**Nature analog:** Crystal/snowflake — simple local rules create complex global structure.

| Field | Value |
|-------|-------|
| Best score | 4.36 train_loss @ 200 steps (30s smoke test, no val) |
| Params | 17M |
| Steps built | 2/3 |
| Latest file | `train_gpt_m8_step2.py` |
| Original spec | `specs/model8-crystal.md` |
| Status | **IN PROGRESS — Step 3 needed** |

**Architecture:** Standard transformer + `GrowthRule` module. Growth rule has layer embeddings (32d) + MLP (32→64→256) that generates per-layer scaling factors. **⚠️ CRITICAL: GrowthRule exists as a module but is NOT called in the forward pass. Currently trains as a baseline transformer with dead parameters.**

**Step 2 adds:** Warmdown 3500, grad clip 0.3, EMA 0.997.

**Step 3 plan:** **MUST wire growth_rule into forward pass** — generate per-layer modifications from seed. Without this, M8 is just a baseline with wasted params. Monitor gradient flow through growth path.

**Known issues:** Growth rule not wired (critical). If growth rule is too simple, all layers are identical (defeats purpose). Inference cost from running growth rules at each forward pass.

---

## Cross-Model Comparison (200-step smoke tests on 4070 Super)

| Model | Score | Speed | Size | Novel? |
|-------|-------|-------|------|--------|
| **M1 Codec** 🔥 | **2.63 bpb** | ~140ms | 8.0 MB | ✅ |
| **M3 Hybrid** | **2.53 bpb** | 141ms | ~5.1 MB | ✅ |
| **M4 Optimized** | 3.83 bpb | ~140ms | 5.1 MB | ❌ |
| **M8 Crystal** | 4.36 loss | 140ms | — | ⚠️ (not wired) |
| **M7 Template** | 4.49 loss | 141ms | — | ✅ |
| **M6 Hive** | 5.36 loss | 114ms | — | ✅ |
| **M2 Recursive** | 4.01 loss | ~140ms | ~5.7 MB | ✅ |

**NOTE:** M1 and M3 scores are val_bpb (comparable). M6/M7/M8 scores are train_loss at 200 steps without validation (not directly comparable to val_bpb). Full training runs needed to compare properly.

---

## Smoke Test Command (IMPORTANT)

Short tests MUST skip validation (eval_val eats entire wallclock):
```bash
RUN_ID=test DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=30 \
TRAIN_BATCH_TOKENS=16384 TRAIN_SEQ_LEN=256 \
WARMUP_STEPS=2 VAL_LOSS_EVERY=0 \
python3 -u train_gpt_mX_stepY.py
```

---

*Update this file after EVERY model test or build step.*
