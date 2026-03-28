# X-WING Night Session — Discoveries & Hypotheses
## 2026-03-26

## Proven Results (tonight)

| Variant | BPB | Delta vs baseline | Key change |
|---------|-----|-------------------|------------|
| Podracer III (old SOTA) | 0.9362 | — | rank-local tables |
| X-WING v1 (cubric) | **0.5640** | -0.372 | shared tables + 1D cubric |
| X-WING v2 (cubric + per-order) | 0.5637 | -0.0003 vs v1 | per-order entropy centers |
| X-WING brown (per-order only) | 0.6218 | +0.058 vs v1 | cubric removed — WORSE |
| X-WING fast (speed boosts) | 0.5644 | +0.000 vs v1 | no measurable gain |
| PR #803 (competitor) | **0.4416** | -0.122 vs v1 | complementary training |

## Key Lessons

1. **Shared tables = the unlock** (-0.372). All ranks seeing all data is worth more than everything else combined.
2. **Cubric is essential** (-0.058 vs flat alpha). Per-order entropy centers do NOT stack — cubric already captures that axis.
3. **Training loop is maxed** at 88ms/step. Safe boosts add ~0 steps.
4. **Complementary training is the next frontier.** PR #803 proves it: train the model to be WEAK where n-grams are strong → crank alpha → 0.44.

---

## Hypotheses to Test

### H1: Complementary Training + 3D Cubric Synergy
**Prediction:** Combined score < 0.44 (beat #803)

**Why:** Complementary training changes the model's entropy landscape — it becomes more uncertain on bigram-predictable tokens. 3D cubric adapts its 54 multipliers to THIS SPECIFIC landscape. PR #803 uses flat backoff (no cubric). Our cubric should extract more from the complementary model than their flat mixing does.

**Risk:** Low. Both mechanisms are independently proven. Worst case they don't interact.

**Test:** Yellow II (already built, pending run)

---

### H2: More Buckets for Higher Orders (8M → 16M)
**Prediction:** -0.005 to -0.01 BPB

**Why:** Orders 8-9 have longer context hashes. With 8M buckets and 62M tokens, high-order collision rate is ~7.4 collisions/bucket. At 16M: ~3.7. Fewer collisions = purer probability estimates for orders that matter most (cubric gives them 2.0x weight).

**Risk:** Zero. Memory is 20.7GB of 80GB. 16M uint32 tables = +128MB.

**Test:** Change NGRAM_EVAL_BUCKETS=16777216 in Yellow II run.

---

### H3: Complement Alpha Sweep (0.3 / 0.5 / 0.7)
**Prediction:** Optimal is NOT 0.5 when cubric is present

**Why:** PR #803 tuned alpha=0.5 for flat backoff. Cubric already suppresses orders 2-3 (the same ones bigram complementarity targets). With cubric doing partial suppression, the model doesn't need to be AS complementary. Optimal may be lower (0.3-0.4) or higher (0.6-0.7 to fully specialize).

**Risk:** Low. Each test is a full training run (14 min). Run 3 on eval-only after first full run.

**Test:** Sweep via COMPLEMENT_ALPHA env var.

---

### H4: Raise Cubric Ceiling (2.0 → 2.5 or 3.0) with Complementary Training
**Prediction:** Safe now. -0.005 to -0.01 BPB.

**Why:** Green2 catastrophe (ceiling=4.0) happened because model was STRONG everywhere — high alpha on confident tokens destroyed predictions. With complementary training, the model is deliberately WEAK on easy tokens. High cubric multipliers push alpha up on tokens where n-grams genuinely dominate. The failure mode (alpha too high on confident model) no longer applies.

**Risk:** Medium. Green2 trauma is real. Start with 2.5, not 4.0.

**Test:** Change ceiling in cubric adaptation code. Eval-only test possible.

---

### H5: Adaptive Complement Alpha (Ramp During Training)
**Prediction:** -0.002 to -0.005 BPB vs fixed alpha

**Why:** Early training needs normal gradients to learn language structure. Late training (warmdown phase) should specialize for n-gram complementarity. Like QAT and SWA that phase in late, complementary training could ramp from 0→0.5 during the last 30% of steps.

**Risk:** Low. If ramp hurts, the fixed-alpha version is the fallback.

**Test:** ~5 line code change in training loop.

---

### H6: Remove Bigram Embedding When Using Complementary Training
**Prediction:** -0.001 to -0.003 BPB, or neutral

**Why:** The BigramHashEmbedding (1536 vocab) teaches the model bigram patterns during training. But complementary training DOWNWEIGHTS those same tokens. The embedding is pushing the model to learn what we're telling it to ignore. Removing it frees parameters and avoids the conflict.

**Risk:** Low. BIGRAM_VOCAB_SIZE=0 to disable. Easy A/B.

**Test:** Single env var change.

---

### H7: TTT on Top of Everything
**Prediction:** -0.005 to -0.02 BPB

**Why:** TTT was only +0.005 on the old setup. But with complementary training, the model is designed for n-gram complementarity at the POPULATION level. TTT adapts it to the SPECIFIC val data distribution. The delta could be larger now because the model has more room to adapt (it's deliberately uncertain on predictable tokens → TTT can sharpen those predictions).

**Risk:** Time budget. TTT adds ~600s eval. PR #803 fits it in 458s eval time.

**Test:** TTT_EVAL_ENABLED=1 with tuned epochs.

---

### H8: Chunk Size Sweep (512K / 1M / 2M)
**Prediction:** Optimal may shift with complementary training

**Why:** Smaller chunks = more frequent table updates = fresher statistics. But also = less data per scoring pass. With complementary training, the model's predictions are different (more uncertain on easy tokens) → the optimal freshness/accuracy tradeoff may shift.

**Risk:** Zero. Env var change.

**Test:** NGRAM_CHUNK_TOKENS sweep.

---

## Priority Ranking

| Priority | Hypothesis | Expected gain | Cost | Dependencies |
|----------|-----------|---------------|------|-------------|
| **1** | H1: CT + 3D cubric | -0.10+ | 1 run (14 min) | Yellow II (built) |
| **2** | H2: 16M buckets | -0.005 to -0.01 | env var | None |
| **3** | H4: Ceiling 2.5 | -0.005 to -0.01 | code + run | H1 result first |
| **4** | H3: Alpha sweep | find optimal | 3 eval-only | H1 result first |
| **5** | H7: TTT | -0.005 to -0.02 | 1 run | H1 result first |
| **6** | H6: Kill bigram embed | -0.001 to -0.003 | env var | H1 result first |
| **7** | H5: Ramp alpha | -0.002 to -0.005 | 5 lines + run | H1 result first |
| **8** | H8: Chunk sweep | find optimal | 3 eval-only | H1 result first |

**Critical path:** H1 first. Everything else depends on whether complementary training + cubric synergize. If Yellow II beats 0.50, we're in the hunt. If it beats 0.45, we're winning.
