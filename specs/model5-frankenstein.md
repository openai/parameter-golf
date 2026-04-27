# Model 5: "The Frankenstein" — Build Spec

**Classification:** PRIVATE — FINAL SUBMISSION
**Target bpb:** Best of everything
**Approach:** Combine winning components from Models 1-4

---

## This Model Doesn't Get Built Until Models 1-4 Are Tested

Model 5 is assembled from empirical results, not theory.

## Assembly Rules

1. Run Models 1-4 on 8×H100
2. Score each model's bpb
3. For each model, identify which COMPONENTS contributed most to the score
4. Combine the best components into Model 5

## Possible Configurations (to be determined by results)

### If Codec (Model 1) wins:
- Codec layers 1+2 + best transformer architecture from Model 2 or 4
- TTT adapted for codec (per-document n-gram adaptation)

### If Recursive (Model 2) wins:
- Adaptive recursion + codec preprocessing from Model 1
- Frequency codebook + shared block + LPC preprocessing

### If Hybrid (Model 3) wins:
- SSM layers + attention top layers + codec preprocessing
- Hash-routed MoE + n-gram cache

### If Optimized Transformer (Model 4) wins:
- Standard architecture + codec preprocessing + curriculum learning
- TTT + hard example mining + ANS-aware loss from Model 1

## The Synthesis Questions
After testing Models 1-4:
1. Which preprocessing helps? (LPC, n-gram cache, frequency codebook)
2. Which base architecture is fastest to train? (transformer, recursive, SSM hybrid)
3. Which quantization is best? (int6, BitNet 1.58, GPTQ vs GPTQ-lite)
4. Which eval enhancement works? (sliding window, TTT, ensemble)
5. Do any components CONFLICT with each other?

## Output
- `train_gpt_model5.py` — final submission candidate
- Comprehensive README for the PR (interview audition)
- 3-seed reproducibility verification
