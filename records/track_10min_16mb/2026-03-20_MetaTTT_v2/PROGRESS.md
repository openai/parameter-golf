# MetaTTT v2: Meta-Learned Test-Time Training for Parameter Golf

## Core Idea
Store not "English knowledge" but "the ability to quickly learn any document."
Train with Reptile meta-learning so the model is optimized as a starting point
for fast per-document adaptation at eval time.

## Architecture
- 11 layers, 512 dim, 3x MLP, int6 + zstd-22
- SmearGate + BigramHash (from PR #162)
- SWA during training

## Training (10 min on 8xH100)
- Phase 1 (5 min): Normal Muon + AdamW training
- Phase 2 (5 min): Reptile meta-learning on last 3 blocks' MLP params
  - Inner loop: 3 SGD steps on document chunk (simulates TTT)
  - Outer loop: Move base toward adapted weights

## Eval (10 min budget)
1. Standard sliding window eval (baseline score)
2. TTT eval: For each document chunk, score then adapt via SGD on MLP params

## Key Results So Far
- v1 (buggy TTT eval): sliding window val_bpb = 1.1768
- Baseline (10L, no extras): 1.2090
- SOTA PR #179 (11L): 1.1472
- SOTA PR #162 (SmearGate): 1.1483
- Expected with TTT: ~1.144 (based on ~0.033 constant TTT gain)
- Expected with meta-learned TTT: ~1.11-1.13 (if Reptile works)

## Known Issues (Fixed in v2)
1. TTT eval crashed with "Inference tensors cannot be saved for backward"
   - Root cause: Rotary cache contains inference tensors from quantize roundtrip
   - Fix: Create fresh model copy, reset all caches before TTT eval
2. Reptile only ran 69 steps (too few)
   - Fix: Dedicate 50% of training time to Reptile, not 5%

## GPU Costs
- 1xH100 experiments: ~$10
- 8xH100 experiments: ~$50 (4 runs at ~$3.6 each + idle time)
- Total spent: ~$60

## Literature
- TTT-E2E (Sun et al., 2025): Meta-learning for TTT, proven at 3B scale
- TTT-NN (Hardt & Sun, ICLR 2024): 20% bpb reduction with nearest neighbors
- Reptile (Nichol et al., OpenAI): First-order MAML, ~50 lines to implement
