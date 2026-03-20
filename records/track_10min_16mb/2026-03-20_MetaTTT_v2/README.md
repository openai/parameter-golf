# Meta-Learned Test-Time Training for Parameter Golf

**Non-record research submission** | val_bpb: 1.1661 (sliding window) | Artifact: 18.77MB (over budget — see analysis)

## Summary

This submission explores **meta-learned test-time training (Meta-TTT)** as a theoretically-motivated approach to the Parameter Golf challenge. Instead of treating the 16MB artifact as a static model, we train it as a **fast adaptation starting point** — optimized via Reptile meta-learning to rapidly specialize to each validation document at eval time.

While our final score (1.1661) does not beat the current SOTA (~1.145), and our artifact exceeds the 16MB limit, we document several novel findings and a complete implementation of Reptile meta-learning + document-adaptive TTT evaluation for compressed language models.

## Motivation

The Parameter Golf challenge asks: *what is the most efficient way to encode knowledge about language in 16MB?*

Current approaches answer: **store ~22M quantized weights in a transformer.** Every submission on the leaderboard optimizes the same paradigm — architecture tweaks, quantization schemes, and hyperparameter tuning.

We propose a different answer: **store a learning algorithm, not frozen knowledge.** The 16MB artifact should encode the ability to rapidly learn any document, not static predictions averaged over all possible documents.

This is grounded in two theoretical frameworks:
1. **Solomonoff induction**: The optimal predictor is the shortest program that explains the data. A model that adapts per-document is a shorter "program" than one that must handle all documents statically.
2. **TTT-E2E** (Sun et al., 2025): Meta-learning the initialization for test-time gradient descent provably outperforms both static models and naive dynamic evaluation.

## Method

### Architecture (Engineering Base)
- 11 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA)
- 3x MLP expansion (hidden=1536), relu^2 activation
- SmearGate: learned gate blending each token with previous token's embedding (~512 params)
- BigramHash: 4096-bucket hash table for token-pair features (~524K params)
- Int6 per-row quantization + zstd-22 compression
- FP16 tied embeddings, sliding window eval (stride=64)
- Stochastic Weight Averaging (every 50 steps during warmdown)
- 27.1M total parameters

### Training Phase 1: Standard Pre-training (80% of 10-minute budget)
Standard Muon optimizer (momentum=0.99, warmup from 0.92) + AdamW for embeddings/scalars. Training on FineWeb with seq_len=1024 for ~7,800 steps.

### Training Phase 2: Reptile Meta-Learning (20% of 10-minute budget)
After standard training, we switch to **Reptile** (Nichol & Schulman, 2018) — a first-order meta-learning algorithm that optimizes the model's initialization for fast test-time adaptation.

**Which parameters to adapt at test time?** Following TTT-E2E's finding that updating the last 1/4 of blocks is optimal, we designate the MLP layers of the final 2 blocks (blocks 9-10) as "TTT parameters" (~4 tensors).

**Reptile outer loop:**
```
for each meta-step:
    1. Save base TTT params
    2. Sample a training document chunk
    3. Inner loop: 3 SGD steps (lr=0.1) on TTT params only
    4. Outer loop: base += 0.01 * (adapted - base)
```

In 120 seconds, we complete **1,576 Reptile meta-steps** — each one simulating the test-time adaptation process and pushing the base weights toward initializations that adapt quickly.

### Evaluation: Test-Time Training
At eval time, for each sliding window chunk:
1. **Score** the chunk (accumulate BPB, no gradients)
2. **Adapt**: 1 SGD step (lr=0.01) on the last 2 blocks' MLP parameters using next-token prediction loss
3. Move to next chunk with adapted weights

The model progressively specializes to the local distribution of the validation text.

## Results

### Experiment 1: 13 Layers Beat Baseline on 8xH100 (Depth Frontier)

Early experiments validated that deeper models improve on 8xH100 despite fewer training steps:

| Config | Steps | val_bpb | Artifact |
|--------|-------|---------|----------|
| 10L baseline (int8) | 12,157 | 1.2090 | 15.4MB |
| **13L (int8)** | **9,385** | **1.1884** | 19.8MB (over) |
| 13L (int6) | 9,200 | 1.1973 | 15.1MB |

**Finding**: 13 layers with int6 quantization improves val_bpb by 0.012 over the 10L baseline while fitting in 16MB. This was the first evidence that the "depth frontier" — more layers with aggressive quantization — is viable.

### Experiment 2: Engineering Stack Combination

| Config | val_bpb | Artifact |
|--------|---------|----------|
| 10L baseline | 1.2090 | 15.4MB |
| 11L + SmearGate + BigramHash + SWA | 1.1768 | 14.85MB |
| 11L + SmearGate + BigramHash + SWA + Reptile(69 steps) | 1.1776 | 14.85MB |

**Finding**: SmearGate and BigramHash provide ~0.03 bpb improvement. 69 Reptile steps (from an earlier implementation where Reptile only ran for a few seconds after training) showed no measurable benefit — confirming that meaningful meta-learning requires substantial compute allocation.

### Experiment 3: Reptile at Scale (1,576 meta-steps)

| Config | Phase 1 Steps | Reptile Steps | Sliding val_bpb | Artifact |
|--------|--------------|---------------|-----------------|----------|
| No Reptile (v1) | 12,286 | 0 | 1.1768 | 14.85MB |
| Reptile 50% (v2a) | 4,788 | 3,967 | 1.1866 | 17.75MB |
| **Reptile 20% (v2b)** | **7,882** | **1,576** | **1.1661** | **18.77MB** |

**Finding**: Reptile 20% achieves the best sliding window val_bpb (1.1661), a 0.011 improvement over the no-Reptile version. However, the artifact size exceeds 16MB due to the model's 27M parameters being too large for int6+zstd compression.

### TTT Eval: Implementation Complete but Too Slow

We implemented a complete TTT evaluation pipeline that:
- Constructs a fresh model (avoiding inference tensor issues from quantization roundtrip)
- Resets Rotary position encoding caches
- Performs per-window SGD adaptation on the last 2 blocks' MLP layers

The implementation runs correctly but is too slow without `torch.compile` (~60 minutes for full validation set). Using `torch.compile` conflicts with gradient tracking needed for TTT. This is a fundamental engineering tension that future work must resolve — potentially via custom CUDA kernels for the TTT forward-backward loop.

## Analysis: Why Meta-TTT Didn't Beat SOTA

### 1. Artifact Size vs. Model Capacity Tradeoff
Adding SmearGate (512 params) and BigramHash (~524K params) pushed the model to 27.1M parameters. At int6 + zstd-22, this compresses to ~18.8MB — over the 16MB limit. The SOTA submissions carefully balance model size against compression budget. Our focus on maximizing TTT benefit led us to prioritize model expressiveness over compressibility.

**Fix**: Use int5 quantization (as in PR #180) or remove BigramHash to fit within budget.

### 2. Reptile Time Allocation Dilemma
More Reptile steps improve the initialization for TTT, but fewer training steps mean a worse base model that compresses poorly. At 50% Reptile, the artifact was 17.75MB; at 20%, it was 18.77MB (worse because more training steps created larger weights). The optimal allocation remains unclear.

### 3. TTT Eval Speed
Without `torch.compile`, TTT eval takes ~60 minutes — far exceeding the 10-minute eval budget. This means TTT must be implemented more efficiently (larger chunks, fewer windows, or custom kernels) to be practical.

### 4. Naive vs. Meta-Learned TTT Gap
The competition data shows naive TTT (PR #152) provides ~0.033 bpb improvement. Our meta-learned approach could not be properly benchmarked due to the eval speed issue. The theoretical advantage of meta-learning (TTT-E2E paper shows naive TTT is nearly useless while meta-learned TTT matches full attention) may require more training compute than 10 minutes allows.

## Key Takeaways

1. **Depth beats width under aggressive quantization**: 13L int6 (1.1973) beats 10L int8 (1.2090) on 8xH100.

2. **Reptile meta-learning is feasible in 2 minutes**: 1,576 meta-steps in 120 seconds, with measurable sliding window improvement (1.1768 → 1.1661).

3. **The artifact size constraint is binding**: Engineering the quantization-compression-capacity tradeoff matters more than any individual technique.

4. **TTT eval needs custom infrastructure**: The forward-backward loop for per-window adaptation is too slow with standard PyTorch. Future work should explore fused kernels or chunked evaluation.

5. **Meta-learning requires minimum compute**: 69 Reptile steps showed no benefit; 1,576 steps showed 0.011 bpb improvement. The relationship between meta-learning steps and TTT quality deserves systematic study.

## Theoretical Contribution

This submission represents (to our knowledge) the **first attempt to apply Reptile meta-learning to optimize a compressed language model's initialization for test-time adaptation** in a competitive setting. While the engineering execution fell short of SOTA, the framework — train for adaptability, not just accuracy — offers a principled alternative to the dominant paradigm of optimizing static model quality.

The key open question: **At what training compute budget does meta-learned TTT overtake naive TTT for small models?** Our results suggest 2 minutes (1,576 steps) is insufficient, but the TTT-E2E paper shows the gap widens with more compute. The 10-minute budget may be on the wrong side of this threshold.

## Reproduction

```bash
# Training + eval (requires 8xH100)
REPTILE_TIME_FRAC=0.2 torchrun --nproc_per_node=8 train_gpt.py

# Key environment variables:
# REPTILE_ENABLED=1       Meta-learning on/off
# REPTILE_TIME_FRAC=0.2   Fraction of training time for Reptile
# REPTILE_INNER_STEPS=3   SGD steps per meta-step
# REPTILE_INNER_LR=0.1    Inner loop learning rate
# REPTILE_OUTER_LR=0.01   Outer loop learning rate
# TTT_ENABLED=1           Test-time training on/off
# TTT_LR=0.01             TTT learning rate at eval
# TTT_STEPS_PER_CHUNK=1   Gradient steps per eval window
```

## References

- Sun et al., "End-to-End Test-Time Training for Long Context," arXiv 2512.23675, 2025.
- Hardt & Sun, "Test-Time Training on Nearest Neighbors for Large Language Models," ICLR 2024.
- Nichol & Schulman, "Reptile: A Scalable Metalearning Algorithm," 2018.
- Behrouz et al., "Titans: Learning to Memorize at Test Time," ICML 2025.

## Hardware

All experiments on RunPod 8x NVIDIA H100 80GB SXM. Total GPU cost: ~$120 across all experiments.

## Author

Xiaoan Liu (NYU) | GitHub: @sseanliu
