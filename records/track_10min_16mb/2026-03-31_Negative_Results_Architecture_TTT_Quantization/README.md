# Non-record: Negative Results — Architecture, TTT Variants, Quantization, and Eval-Time Experiments

## Summary

A collection of negative and marginal results from ~15 experiments on the LeakyReLU(0.5)^2 stack (PR #518 architecture). All runs on 1xH100 or 8xH100 SXM, seed 1337 unless noted.

- **Architecture**: depth recurrence, TrigramHash, MLP expansion, XSA coverage
- **TTT variants**: MLP-only TTT, SGD vs AdamW, layer freezing, LR schedules
- **Quantization**: int5 post-training swap, MLP 3.25x size budget
- **Eval-time**: n-gram caches (proven illegal), XSA-all at MLP 3.0

## Key finding

At this model scale (27M params, 16MB budget), most architectural changes are neutral or negative. The dominant lever is eval-time TTT with the right optimizer (AdamW + cosine LR). Quantization-aware training needs sufficient steps to be effective, and post-training quantization changes are catastrophic.

## Architecture Experiments

| Approach | BPB | Delta vs Baseline | GPU | Why It Failed |
|----------|:---:|:-----------------:|:---:|---------------|
| Baseline (PR #518 stack) | 1.1449 | — | 8xH100 | Reference (post-int6, no TTT) |
| Depth recurrence (Huginn-style) | 1.345 | +0.20 | 1xH100 | Compute overhead dominates; int6 quantization is a better way to add effective params |
| TrigramHash addition | 1.190 | +0.045 | 1xH100 | Quantization destroys small trigram weights; BigramHash works because weights are larger |
| MLP 3.25x expansion | 1.1408 | -0.004 | 8xH100 | Marginal BPB gain but **artifact is 17.0MB — over 16MB limit**. int6+zstd can't compress the extra params enough |
| XSA-all (11 layers vs 4) | 1.1440 | -0.001 | 8xH100 | Neutral at MLP 3.0. The last 4 layers capture most of the XSA benefit |

## TTT Experiments

| Approach | BPB | Delta vs Cosine AdamW | GPU | Why It Failed |
|----------|:---:|:---------------------:|:---:|---------------|
| Cosine AdamW TTT 30ep (baseline) | 1.0781 | — | 8xH100 | Our best legal result (PR #672) |
| Cosine AdamW TTT 20ep | 1.1101 | +0.032 | 8xH100 | 10 fewer epochs costs ~0.03 BPB; diminishing returns but still significant |
| MLP-only TTT (from TTT-E2E paper) | 1.315 | +0.237 | 1xH100 | Needs meta-learning (gradients-of-gradients) to work; naive fine-tuning of MLP alone underfits |
| SGD+momentum TTT 20ep | 1.1435 | +0.065 | 8xH100 | **Made it worse than no TTT** (1.1408). SGD lacks the adaptive LR that AdamW provides for per-parameter scaling |
| TTT with LR floor 0.05 | 1.0932 | +0.015 | 8xH100 | Cosine should decay to 0; the floor prevents the model from fully converging in later epochs |
| TTT freeze first 5 layers | 2.52 | +1.44 | 1xH100 | Tested on 1xH100 where base model was already badly quantized (2.36 BPB); freezing made it worse. May work on 8xH100 but untested |
| Chained TTT + multi-pass min(NLL) | 1.0366 | -0.042 | 8xH100 | **Illegal** — min(NLL) across passes violates single-pass rule (PR #685, rejected) |

## Quantization Experiments

| Approach | BPB | Delta | GPU | Why It Failed |
|----------|:---:|:-----:|:---:|---------------|
| int6+zstd (baseline) | 1.1449 | — | 8xH100 | Optimal for this model scale |
| int5 post-training swap | catastrophic | >>+1.0 | 1xH100 | Must train with int5 QAT from the start; post-training precision reduction destroys weights |
| Earlier QAT onset | 1.260 | +0.115 | 1xH100 | Training with QAT too early reduces effective model capacity during critical learning phase |
| 1xH100 training (860 steps) | 2.36 | +1.22 | 1xH100 | Only 860 optimizer steps (vs ~6600 on 8xH100). EMA averages poorly, QAT gets only 518 steps. **1xH100 is not viable for this architecture** without reducing batch size |

## N-gram Cache (Proven Illegal)

| Approach | BPB | Notes |
|----------|:---:|-------|
| Hashed n-gram cache (2-5gram, 4M buckets) | 0.9850 | **Illegal**: only computes blend for correct token, distribution sums to ~410 not 1.0. "Improvement" tracks hash collision density, not prediction quality. See issue #677 and abaybektursun's PR #886 analysis. Our PR #741 was closed for this reason. |

## Meta-Lessons

1. **AdamW >> SGD for TTT.** Per-parameter adaptive learning rates matter when fine-tuning a quantized model where different layers have different sensitivity.
2. **Don't change quantization post-training.** int5/int6 decisions must be baked into QAT. Post-training swaps are catastrophic.
3. **Architecture changes have diminishing returns at 16MB.** The converged optimum (11L, 512d, MLP 3x, LeakyReLU(0.5)^2) is hard to beat. Eval-time adaptation (TTT) is where the remaining gains live.
4. **Hashed n-gram caches are fundamentally broken**, not just rule-violating. The math doesn't work — hash collisions inflate all token probabilities, producing unnormalized distributions. Any BPB improvement is an artifact of the scoring bug.
5. **1xH100 is not a useful proxy for 8xH100.** With grad_accum_steps=8, you get 8x fewer optimizer updates in the same wallclock, producing a model too undertrained for meaningful comparison.
6. **30 TTT epochs > 20 epochs**, even with diminishing returns. The last 10 epochs contribute ~0.03 BPB, which is significant at the frontier.

## Test plan

- [x] All experiments ran to completion
- [x] BPB numbers verified from training logs
- [x] Illegal approaches clearly labeled
- [x] No artifacts or code included (documentation only)

Generated with [Claude Code](https://claude.com/claude-code)
