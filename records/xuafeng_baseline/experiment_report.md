# Parameter Golf Experiment Report: QAT + Int5/Int6

## Executive Summary

We tested adding Quantization-Aware Training (QAT) with STE fake-quantization to the current #1 leaderboard entry. Results show QAT produces slightly worse BPB than #1's post-training quantization approach. **QAT did not beat the #1 entry.**

| Run | Seed | val_bpb | Artifact Size | Under 16MB? |
|-----|------|---------|--------------|-------------|
| QAT + Trigram | 42 | 1.14423 | 16,215,286 | **No** (over by 215KB) |
| QAT only | 1337 | 1.14476 | 15,793,963 | Yes |
| QAT only | 2024 | — | — | SSH dropped at step 4500 |
| **#1 (baseline)** | **mean** | **1.14276** | **~15.9MB** | **Yes** |

**Conclusion**: QAT added ~0.002 BPB penalty vs #1's post-training quantization, the opposite of our hypothesis.

---

## Experiment Details

### What We Changed

Starting from the #1 entry (`10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04`), we added:

1. **QAT with STE fake-quantization** in `CastedLinear.forward()`:
   - MLP layers: int5 (clip_range=15)
   - Attention layers: int6 (clip_range=31)
   - Straight-Through Estimator: `w + (w_quantized - w).detach()`
   - Enabled from step 0 (no QAT warmup)

2. **TrigramHash(4096, dim=32)** — tested in seed 42 only, disabled for later runs due to size constraint

### Run Configuration

| Parameter | Value |
|-----------|-------|
| GPU | 8x NVIDIA H100 80GB SXM |
| Training time | 600s wallclock cap |
| Steps completed | ~6600-6650 |
| Step avg | ~90.2 ms/step |
| Model params | 25.5M (QAT only) / 25.7M (QAT + trigram) |
| Compression | int5 MLP / int6 attn + zstd-22 |
| SWA | 23-24 checkpoints averaged |
| Eval | Sliding window, stride=64 |

---

## Detailed Results

### Run 1: QAT + TrigramHash (seed=42)

```
model_params: 25,664,594
steps: 6614 / 20000
training_time: 600,056ms

Validation progression:
  step 500:  val_bpb = 1.3937
  step 1000: val_bpb = 1.3200
  step 2000: val_bpb = 1.2691
  step 3000: val_bpb = 1.2448
  step 4000: val_bpb = 1.2313
  step 5000: val_bpb = 1.2052
  step 6000: val_bpb = 1.1783
  step 6500: val_bpb = 1.1638 (pre-quant)
  step 6614: val_bpb = 1.1623 (pre-quant, final)

Post-quantization (int5/int6 + zstd + SWA + sliding eval):
  val_bpb = 1.14423
  artifact = 16,215,286 bytes (OVER 16MB LIMIT)
```

**Issue**: TrigramHash pushed artifact 215KB over the 16MB limit.

### Run 2: QAT only (seed=1337)

```
model_params: 25,517,137
steps: 6649 / 20000
training_time: 600,057ms

Validation progression:
  step 500:  val_bpb = 1.3944
  step 1000: val_bpb = 1.3211
  step 2000: val_bpb = 1.2691
  step 3000: val_bpb = 1.2455
  step 4000: val_bpb = 1.2322
  step 5000: val_bpb = 1.2063
  step 6000: val_bpb = 1.1799
  step 6500: val_bpb = 1.1652 (pre-quant)
  step 6649: val_bpb = 1.1628 (pre-quant, final)

Post-quantization (int5/int6 + zstd + SWA + sliding eval):
  val_bpb = 1.14476
  artifact = 15,793,963 bytes (under 16MB)
```

### Run 3: QAT only (seed=2024) — INCOMPLETE

SSH connection dropped at step 4500 (pod terminated). Last logged val_bpb = 1.2166 at step 4500. Run was on pace to match seeds 42 and 1337.

---

## Analysis: Why QAT Didn't Help

### The Quantization Penalty Paradox

Our hypothesis was that QAT would reduce the quantization penalty (gap between pre-quant and post-quant BPB). Let's compare:

| Metric | #1 (no QAT) | Ours (QAT) |
|--------|-------------|------------|
| Pre-quant val_bpb (step ~6600) | ~1.162* | 1.1628 |
| Post-quant val_bpb | 1.14276 | 1.14476 |
| **Quantization "bonus"** | **-0.019** | **-0.018** |

*Estimated from #1's pre-quant metrics.

**Key insight**: Post-training quantization with SWA actually **improves** BPB in this setup (negative penalty). The SWA weight averaging + magnitude pruning + sliding window eval creates a regime where quantization acts as regularization. QAT fights this by making weights "quantization-aware" during training, which ironically removes the beneficial regularization effect of post-training quantization.

### Why #1's Approach Works Better

1. **SWA already produces smooth weights**: Weight averaging across 24 checkpoints creates distributions that quantize cleanly
2. **Magnitude pruning (3% threshold)**: Zeroing small weights improves both compression and generalization
3. **Post-quant as regularization**: The quantization noise from int5/int6 acts as implicit regularization at eval time
4. **QAT reduces weight diversity**: Training with quantization noise makes weights cluster around quantization levels, which may hurt the model's expressiveness

### TrigramHash Assessment

The trigram run (seed 42) achieved val_bpb 1.14423 vs the non-trigram run (seed 1337) at 1.14476. The difference (0.0005) is within noise, suggesting the trigram's contribution is negligible at this size. It also pushed the artifact over the 16MB limit.

---

## Cost Summary

| Pod | Config | Duration | Cost |
|-----|--------|----------|------|
| pgolf-8xh100-qat | 8x H100 SXM | ~50 min | ~$17.93 |
| pgolf-experiments | 1x H100 SXM (unused) | ~5 min | ~$0.22 |
| **Total** | | | **~$18.15** |

---

## Lessons Learned

1. **Post-training quantization can outperform QAT** when combined with SWA and magnitude pruning. The "quantization penalty" can actually be negative (a bonus).

2. **TrigramHash is not worth the size cost** at these parameter counts. BigramHash(10240) already captures most local patterns.

3. **The #1 entry is very well-optimized**. The combination of int5/int6 mixed quantization, SWA, magnitude pruning, and sliding window eval creates a synergistic system that's hard to improve piecemeal.

4. **To beat #1**, the most promising directions are:
   - **Better architectures**: MoE, different attention patterns, or novel layer designs that are more parameter-efficient
   - **Better compression**: Custom learned compression instead of zstd could pack more information into 16MB
   - **Longer effective training**: Reduce per-step overhead to get more steps in 600s
   - **Better tokenizers**: A tokenizer optimized for this specific dataset could improve BPB directly

---

## Ranked Ideas (Updated Post-Experiment)

| Rank | Idea | Expected Impact | Feasibility |
|------|------|----------------|-------------|
| 1 | Custom weight compression (learned codebook instead of zstd) | High | Hard |
| 2 | Sparse MoE with shared experts | High | Hard |
| 3 | Int4 for select layers + 11th layer | Medium | Medium |
| 4 | Dataset-optimized tokenizer | Medium | Medium |
| 5 | ~~QAT + Int5~~ | ~~High~~ **Disproven** | ~~Medium~~ |
| 6 | ~~TrigramHash~~ | ~~Low~~ **Negligible** | ~~Easy~~ |
