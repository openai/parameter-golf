# Parameter Golf — Progress Chart

**Last Updated:** 2026-03-25 22:00 CDT

---

## Model Build Status

| Model | Name | Steps | Smoke Test | Best Loss (200 steps) | Size | Status |
|-------|------|-------|-----------|----------------------|------|--------|
| **4** | Optimized Transformer | 15/15 | ✅ ALL PASS | 3.74 bpb (roundtrip) | 5.1 MB | **COMPLETE** |
| **1** | Codec | 5/5 | ✅ ALL PASS | 3.88 loss / 8.0 MB | 8.0 MB | **COMPLETE** |
| **2** | Recursive (Shared Weights) | 3/3 | ✅ ALL PASS | 4.01 loss (9×512d) | ~5.7 MB | **COMPLETE** |
| **3** | Hybrid (GatedRNN + Attention) | 3/3 | ✅ ALL PASS | 2.529 bpb (213 steps) | ~5.1 MB | **COMPLETE** |
| **6** | Hive (Frozen + LoRA) | 1/3 | ✅ Step 1 PASS | 96.8% frozen, 545K trainable | — | **IN PROGRESS** |
| **7** | Immune (Template Library) | 1/3 | ✅ Step 1 PASS | 17M params, ~295ms/step | — | **IN PROGRESS** |
| **8** | Crystal (Seed + Growth) | 1/3 | ✅ Step 1 PASS | 17M params, ~295ms/step | — | **IN PROGRESS** |
| **5** | Frankenstein (Best of all) | — | — | — | — | **AFTER ALL TESTED** |

---

## Scaling Limitations (Need H100 / Compute Grant)

| Issue | Model | What Happened | What We Need |
|-------|-------|---------------|-------------|
| **OOM at 768d** | Model 4 Step 14 | 12L/768d/3x (62M params) OOM'd on 4070 Super 12GB | H100 80GB for scaled model |
| **OOM at 640d shared** | Model 2 Step 3 | 12×640d shared block OOM during activation storage | H100 for wider recursive model |
| **OOM at 768d shared** | Model 2 Step 3 | 12×768d shared block — 48M params, OOM immediately | H100 only |
| **Can't test 8×GPU** | All | Official scores require 8×H100 SXM | Need compute grant or Vast 8×H100 (~$12/hr) |
| **PolarQuant not wired** | All | Code written, needs save/load integration | More dev time |
| **TurboQuant not integrated** | All | Spec complete, needs implementation | More dev time |

### What the Compute Grant Unlocks:
1. **Scaled models** — test 12L/768d (62M params, 99.6% of 16MB)
2. **8×H100 official scores** — get real bpb numbers for leaderboard
3. **PolarQuant at scale** — 3-bit compression = even more params in 16MB
4. **Multiple submission runs** — 3-seed verification for PR
5. **Final submission** — April 28-30

---

## Smoke Test Infrastructure

| Platform | GPU | Cost | Speed | Used For |
|----------|-----|------|-------|----------|
| **Colab** | T4 16GB | Free | ~4.5s/step | Baseline verification |
| **Vast.ai** | RTX 4070 Super 12GB | $0.09/hr | ~145ms/step | All smoke tests |
| **Vast.ai** | H100 80GB | ~$1.50/hr | ~100ms/step | Scaled model tests |
| **RunPod** | 8×H100 SXM | $21.52/hr | ~107ms/step | Official submission runs |

**Total Vast.ai spend:** ~$0.80 (instance running ~9 hours at $0.09/hr)
**Total RunPod spend:** ~$7 (baseline + failed v1)
**Budget remaining:** Vast $19.20 / RunPod $13

---

## Techniques Verified Working (Model 4 Incremental Build)

| Step | Technique | Impact | Status |
|------|-----------|--------|--------|
| 1 | 11 layers (up from 9) | More depth | ✅ |
| 2 | 3x MLP expansion | More capacity per layer | ✅ |
| 3 | EMA (decay=0.997) | Smoother eval weights | ✅ |
| 4 | Warmdown 3500 iterations | Better LR scheduling | ✅ |
| 5 | Sequence length 2048 | More context | ✅ |
| 6 | Batch 786,432 tokens | More stable gradients | ✅ |
| 7 | Gradient clipping 0.3 | Prevents divergence | ✅ |
| 8 | Muon weight decay 0.04 | Regularization | ✅ (best pre-quant bpb) |
| 9 | BigramHash 2048 buckets | Token pair patterns | ✅ (needs full training) |
| 10 | SmearGate | Current/previous token blending | ✅ (needs full training) |
| 11 | Sliding window eval stride=64 | Better eval context | ✅ |
| 12 | GPTQ-lite int6/int8 + zstd-22 | Better compression | ✅ |
| 13 | PolarQuant (code written) | 26% more params in 16MB | ⏳ Needs wiring |
| 14 | Scale to fill 16MB | 2× parameters | ❌ OOM on 4070 Super |
| 15 | TTT + curriculum learning | Eval-time adaptation | ✅ |

---

## Novel Architectures Verified

| Architecture | Status | Key Finding |
|-------------|--------|-------------|
| **Codec** (dictionary + n-gram + transformer) | ✅ Working | Bigram embedding + unigram projection improves loss |
| **Recursive** (shared weight block × N) | ✅ Working | 12× application of 1 block trains well at 512d |
| **Hybrid** (GatedRNN + attention) | ✅ Working | 3 RNN + 6 attention layers — no degradation vs pure attention |
| **Hive** (90% frozen + 10% LoRA) | ✅ Step 1 Working | 96.8% frozen, 545K trainable, trains on 4070 Super |
| **Immune** (template library) | ✅ Step 1 Working | 17M params, loss dropping, ~295ms/step |
| **Crystal** (seed + growth rules) | ✅ Step 1 Working | 17M params, loss dropping, ~295ms/step |

---

## Research Completed (17+ files)

See `TOOLS-AND-RESOURCES.md` for full inventory.

Key findings:
- TTT with AdamW is the biggest single technique (0.02-0.05 bpb gain)
- PolarQuant/TurboQuant enables 26% more params per MB
- Nobody in competition is using non-transformer architectures
- Pending PRs are at 1.05-1.07 bpb (leaderboard shows 1.12)

---

## Next Steps (Priority Order)

1. ~~Finish Model 3~~ ✅ DONE (val_bpb 2.529)
2. **Build Models 6-8 Step 2** (warmdown + grad clip + EMA)
3. **Wire PolarQuant** into save/load pipeline
4. **Apply for compute grant** with our model diversity as evidence
5. **Scale up on H100** — test 12L/768d models
6. **8×H100 submission runs** — get official leaderboard scores
7. **Submit Model 4** (public, for leaderboard visibility)
8. **Drop best novel model** (private, April 28-30)
