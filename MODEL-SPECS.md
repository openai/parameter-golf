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

**Concept:** Combine the best-performing techniques from all models into one architecture.

| Field | Value |
|-------|-------|
| Best score | **val_bpb 3.257** (pre-quant), 3.322 (roundtrip) |
| Params | 17.4M |
| Compressed size | 6.4 MB |
| Steps built | 2/2 — COMPLETE |
| Latest file | `train_gpt_m5_step2.py` |
| Status | **COMPLETE** |

**Components combined:**
- BigramEmbed from M1 Codec (2048-bucket hash of token pairs)
- GrowthRule from M8 Crystal (per-layer ±5% learned scaling)
- Standard transformer backbone with all M4 optimizations (warmdown, grad clip, EMA)

**What was tried and removed:**
- GatedRNN from M3 concept: 7× slower (993ms/step vs 140ms baseline) due to sequential loop. Only got 31 steps in 30s. Unusable for competition.

**Key finding:** BigramEmbed + GrowthRule together (3.257) beat either alone (M1 codec without growth: 2.631, M8 growth without bigram: 3.342). But still behind M1 Codec's full architecture.

**The surprise:** M3 Hybrid's 2.529 bpb was NOT from the GatedRNN — it was dead code. M3 is really just an optimized baseline, same as M4 but with warmdown/grad_clip tuned slightly differently.

---

## Model 6: Hive (Frozen Backbone + LoRA)

**Concept:** 90% of params frozen (random orthogonal init, never trained). Only 10% trainable via LoRA adapters. The frozen weights act as a fixed random projection / feature extractor.
**Nature analog:** Bee brain — 90% hardwired pattern detectors, 10% plastic learning.

| Field | Value |
|-------|-------|
| Best score | **val_bpb 4.031** (pre-quant), 4.109 (roundtrip) |
| Params | 17M total (11.5M frozen, 5.75M trainable, 66.8% frozen) |
| Steps built | 3/3 — COMPLETE |
| Latest file | `train_gpt_m6_step3.py` |
| Original spec | `specs/model6-hive.md` |
| Status | **COMPLETE** |

**Architecture:** Standard transformer backbone. First 6 layers frozen (random init, act as fixed feature extractors). Last 3 layers fully trainable. LoRA rank-8 adapters on c_q, c_v, fc across ALL layers. Norms, embeddings, q_gain all trainable.

**Step 3 results:** Hybrid freeze at 66.8%. val_bpb 4.031 (pre-quant), 4.109 (roundtrip). 9.4 MB compressed. 217 steps in 30s.

**Key lesson:** 99% frozen + LoRA on RANDOM weights = no learning. The Hive concept needs pre-trained weights to work. Compromised by unfreezing last 3 layers.

**Known issues:** Still one of our weakest models (bpb 4.03 vs M1's 2.63). The frozen-backbone-with-adapters idea may be fundamentally limited for train-from-scratch competitions.

---

## Model 7: Immune System / Template Codebook

**Concept:** Library of 16 small weight templates shared across all layers. Per-token router selects and mixes templates to generate effective representations. Like V(D)J recombination.
**Nature analog:** Immune system — 400 gene segments combine to create millions of unique antibodies.

| Field | Value |
|-------|-------|
| Best score | **val_bpb 3.464** (pre-quant), 3.657 (roundtrip) |
| Params | 17M |
| Steps built | 3/3 — COMPLETE |
| Latest file | `train_gpt_m7_step3.py` |
| Original spec | `specs/model7-immune.md` |
| Status | **COMPLETE** |

**Architecture:** Standard transformer + `TemplateCodebook` module. 16 templates (512d each), router (linear layer) produces softmax weights per token, mixed templates added to hidden state. Applied after the transformer blocks.

**Step 3 results:** 32 templates (up from 16), router temp warmup (5→1), diversity reg (0.01 cosine penalty). val_bpb 3.464 pre-quant, 3.657 roundtrip. 5.3 MB compressed. 206 steps in 30s.

**Key finding:** Temperature warmup is critical — uniform mixing early gives all templates gradient signal, then sharpening enables specialization. Diversity reg prevents router collapse.

**Known issues:** Initial loss spike to 17.9 (recovers by step 50). Int6 quantization not implemented.

---

## Model 8: Crystal (Seed + Growth Rules)

**Concept:** One small "seed" block + a growth rule network that generates per-layer modifications. The architecture's shape encodes learned knowledge.
**Nature analog:** Crystal/snowflake — simple local rules create complex global structure.

| Field | Value |
|-------|-------|
| Best score | **val_bpb 3.342** (pre-quant) |
| Params | 17M |
| Steps built | 3/3 — COMPLETE |
| Latest file | `train_gpt_m8_step3.py` |
| Original spec | `specs/model8-crystal.md` |
| Status | **COMPLETE** |

**Architecture:** Standard transformer + `GrowthRule` module. Growth rule has layer embeddings (32d) + MLP (32→64→256) that generates per-layer scaling factors. **⚠️ CRITICAL: GrowthRule exists as a module but is NOT called in the forward pass. Currently trains as a baseline transformer with dead parameters.**

**Step 3 results:** Growth rule wired into forward pass. Per-layer scaling: `x * (1 + 0.05 * tanh(MLP(layer_emb)))`. val_bpb 3.342 pre-quant. 125 steps in 30s (242ms/step — slower due to per-layer MLP). 5.2 MB compressed.

**Key finding:** Growth rule helps! Best of the nature-inspired models. Per-layer learned scaling gives each layer a unique identity without many extra params.

**Known issues:** Slower per-step (242ms vs ~140ms) due to growth MLP computation. Initial loss spike to 17. No int6 quantization.

---

## Cross-Model Comparison (30s smoke tests on 4070 Super)

| Model | val_bpb (pre-quant) | val_bpb (roundtrip) | Speed | Size | Novel? |
|-------|-------------------|-------------------|-------|------|--------|
| **M3 Hybrid** 🔥 | **2.529** | — | 141ms | ~5.1 MB | ✅ |
| **M1 Codec** 🔥 | **2.631** | **2.631** | ~140ms | 8.0 MB | ✅ |
| **M8 Crystal** | **3.342** | — | 242ms | 5.2 MB | ✅ |
| **M7 Template** | **3.464** | **3.657** | 146ms | 5.3 MB | ✅ |
| **M4 Optimized** | 3.83 | 3.83 | ~140ms | 5.1 MB | ❌ |
| **M2 Recursive** | — | — | ~140ms | ~5.7 MB | ✅ |
| **M6 Hive** | **4.031** | **4.109** | 138ms | 9.4 MB | ✅ |

**All scores are val_bpb from 30-second smoke tests.** Full 10-min 8×H100 runs will score much lower (better).
**M3 Hybrid is now the smoke-test leader** at val_bpb 2.529.

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
