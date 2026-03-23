# Phase 1: Legal Score-First TTT + Meta-TTT (FOMAML)

**Author:** George Teifel (@george11642)
**Status:** Awaiting compute for validation
**Base:** PR #462 architecture (1.0672 BPB)
**Track:** non_record_16mb

## Approach

Building on PR #462's GEPA-discovered architecture (Star-ReLU + U-Net + XSA + AdamW TTT), this submission adds:

### Proven Techniques (Phase 1)
1. **XSA on all 11 layers** (PR #462 uses last 4 only). PR #478 shows XSA-all improves BPB.
2. **Cosine TTT scheduling with 30 epochs** (PR #462 uses 10). PR #481 shows cosine + more epochs helps.
3. **Per-layer TTT learning rate groups**: 3x LR for MLP output projections (highest quantization error), 0.5x for input projections, 1x for everything else. Based on PR #481's quantization damage analysis.
4. **GPTQ-lite optimal clip percentile search**: Per-row sweep of 6 percentile candidates to minimize reconstruction error before final quantization.
5. **Legal score-first TTT**: Evaluate tokens BEFORE training on them, complying strictly with the rule "you are only allowed to test-time train on validation set tokens you've already evaluated your model on."

### Novel Technique (Phase 2 - In Development)
6. **Meta-TTT (FOMAML)**: During training, periodically simulate the quantize-TTT-eval pipeline via first-order MAML. This teaches the model to produce weight configurations that are maximally adaptable during test-time training. No existing submission uses meta-learning for TTT optimization.

## Architecture

- 11 layers (5 encoder + 6 decoder, U-Net gated skips)
- dim=512, heads=8/8, MLP hidden=1792
- Star-ReLU activation with learned scale/bias
- BigramHash (8192 buckets, 128d), SmearGate
- Partial RoPE (16/64 dims), XSA on all layers
- Int6 QAT with GPTQ-lite clip search, zstd-22

## Expected Results

Awaiting 8xH100 compute to validate. Expected improvement over PR #462 baseline (1.0672):
- Phase 1 techniques: ~-0.012 BPB
- Meta-TTT (if successful): ~-0.015 BPB additional

## Confirmed Dead Ends (Not Attempted)
- Depth recurrence (PR #386: 1.4061 BPB)
- MoE (sparsity=0 optimal below 500M params)
- LoRA TTT (10x worse than full-param TTT)
