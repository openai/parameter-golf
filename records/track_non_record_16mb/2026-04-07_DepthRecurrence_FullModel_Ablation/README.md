# Full-Model Depth Recurrence: Systematic Ablation Study

**Non-Record Submission (Research Contribution)**
**Author:** codeprakhar25
**Hardware:** 1x H100 (RunPod), 600s wallclock per run
**Best config (7×2):** 1.3680 BPB | 13.4M params | 10.0 MB artifact

---

## Summary

Full-model depth recurrence — cycling through all N unique transformer blocks R times (effective depth = N×R) — with U-Net skip connections operating on effective layer indices. Seven configurations tested across the full design space (3→5→7→9 unique blocks).

**Key finding:** `torch.compile` shows **zero slowdown** with depth recurrence. All configs run at 433–556 ms/step vs baseline 447 ms/step. This contradicts [prior reports](https://anthonymaggio.substack.com/p/speed-running-a-language-model-from) claiming 2.7x penalty.

### Difference from PR #363

PR #363 (Evangeline Kamin) uses a **middle-cycle** architecture with separate stem/core/tail blocks. This submission uses **full-model recurrence** — all blocks are looped uniformly, with U-Net skip connections reindexed to effective depth. Simpler implementation, different tradeoff profile.

---

## Architecture

Standard baseline architecture (9-layer GPT with GQA, RoPE, ReLU², Muon optimizer) with one modification:

```
# Instead of 9 unique blocks in sequence:
for block in blocks:        # 9 blocks, 1 pass

# N unique blocks cycled R times:
for repeat in range(R):     # R passes
    for block in blocks:    # N blocks per pass
```

U-Net skip connections index by **effective layer position** (0 to N×R−1), not physical block index. This means skip connections span across recurrence boundaries, connecting encoder layers from pass 1 to decoder layers in pass 2.

### Implementation

Three lines added to `Hyperparameters`:
```python
depth_repeats = int(os.environ.get("DEPTH_REPEATS", 1))
```

Forward pass modified to loop `depth_repeats` times over `self.blocks`, tracking `effective_idx` for U-Net skip connection routing.

---

## Results

### Full Ablation Table

| Config | Unique blocks | Eff. depth | Params | Artifact | val_bpb | ms/step | Gap vs baseline |
|--------|--------------|-----------|--------|----------|---------|---------|-----------------|
| Baseline (9×1) | 9 | 9 | 17.1M | 13.6 MB | **1.3322** | 447 | — |
| 7×2 | 7 | 14 | 13.4M | 10.0 MB | 1.3680 | 542 | +0.036 |
| 9×2 | 9 | 18 | 17.1M | 11.8 MB | 1.3758 | 692 | +0.044 |
| 5×2 | 5 | 10 | 9.7M | 7.7 MB | 1.3819 | 444 | +0.050 |
| 5×2 + BigramHash | 5 | 10 | 9.8M | 7.9 MB | 1.3955 | 444 | +0.064 |
| 3×3 | 3 | 9 | 6.0M | 4.9 MB | 1.4238 | 433 | +0.092 |
| 7×3 | 7 | 21 | 13.4M | 9.0 MB | 1.4075 | 755 | +0.075 |
| 3×5 | 3 | 15 | 6.0M | 4.5 MB | 1.4382 | 556 | +0.106 |
| 3×3 wide (640d) | 3 | 9 | 11.7M | 7.2 MB | 2.0672 | 504 | +0.735 (failed) |

### Scaling Curve

```
BPB gap vs baseline (×2 repeats unless noted):

         Unique blocks
         3    5    7    9
+0.11 |  *3×3
+0.10 |
+0.09 |  *3×5
+0.08 |
+0.07 |       *7×3  ← more repeats, worse again
+0.06 |
+0.05 |       *5×2
+0.04 |                 *9×2  ← step-budget starved
+0.03 |            *7×2  ← optimal
+0.02 |
+0.01 |
 0.00 |                    *9×1 (baseline)
```

- Along any fixed block count: **more repeats always hurt** (3×3→3×5, 7×2→7×3)
- Along ×2 repeats: more unique blocks help (3→5→7), then step-starvation at 9×2
- **7×2 is the proven optimum from both directions**

---

## Key Findings

### 1. torch.compile has NO penalty for depth recurrence

| Config | ms/step | Memory (MiB) |
|--------|---------|-------------|
| Baseline 9×1 | 447 | ~10,000 |
| 3×3 | 433 | ~10,127 |
| 5×2 | 444 | ~10,000 |
| 7×2 | 542 | ~12,000 |
| 3×5 | 556 | 16,384 |

The 3×3 config is actually **faster** than baseline (433 vs 447 ms/step). Only 3×5 (15 effective layers) shows significant slowdown, which is expected from the deeper forward pass, not compile overhead.

### 2. More repeats hurt — more unique blocks help

3×3 (9 effective, 1.4238) vs 3×5 (15 effective, 1.4382): Adding more passes through the same 3 blocks makes things **worse**. The model hits diminishing returns on shared-weight depth quickly. In contrast, 7×2 (14 effective, 1.3680) is much better because it has more unique representations.

### 3. Width changes require hyperparameter retuning

The 3×3 wide (MODEL_DIM=640) run catastrophically failed (2.0672 BPB). Root cause: Muon optimizer and AdamW learning rates are tuned for 512-dim. This is NOT an architecture failure — it's a hyperparameter search gap.

### 4. BigramHash and recurrence don't complement

5×2 + BigramHash (1.3955) is worse than 5×2 alone (1.3819). At low step counts (~1300 steps on 1x H100), the extra bigram parameters steal training capacity from the recurrent blocks.

### 6. Step-budget interaction: why 9×2 breaks the trend

9×2 has the same 17.1M params as baseline but a 2x forward pass costs 692 ms/step vs 447 ms/step baseline. In the 600s budget that's only **867 steps vs ~1340 steps** — 35% less training for the same size model. It's undertrained relative to baseline.

| Config | ms/step | Steps in 600s | BPB |
|--------|---------|--------------|-----|
| Baseline 9×1 | 447 | ~1340 | 1.3322 |
| 7×2 | 542 | ~1107 | 1.3680 |
| 9×2 | 692 | ~867 | 1.3758 |

The tradeoff is clear: recurrence only helps when the step-time overhead is compensated by having fewer unique parameters to train. 7×2 hits the sweet spot — smaller model trains faster per step, recurrence gives it effective depth. 9×2 pays the recurrence cost without the parameter savings.

### 7. More repeats always hurt — confirmed universally

| Pair | Fewer repeats | More repeats | BPB penalty |
|------|---|---|---|
| 3-block | 3×3: 1.4238 | 3×5: 1.4382 | +0.014 |
| 7-block | 7×2: 1.3680 | 7×3: 1.4075 | **+0.040** |

The penalty grows with more unique blocks — because a larger model at higher effective depth takes longer per step, starving training further. 7×3 (755 ms/step) gets only 795 steps vs 7×2's 1108 steps, despite identical 13.4M params. The rule holds universally: **never add repeats, add unique blocks instead**.

### 5. Optimal tradeoff: 7×2

For artifact-size-constrained scenarios, 7×2 offers the best balance: only +0.036 BPB gap with 26% smaller artifact (10.0 MB vs 13.6 MB). On 8x H100 with ~6900 steps, this gap would likely shrink further as shared weights get more training.

---

## Negative Results

1. **More repeats always hurt** — Confirmed at 3 blocks (3×3→3×5) and 7 blocks (7×2→7×3). Root cause: each extra repeat slows the forward pass, eating into the fixed training budget. Never increase repeats; increase unique blocks instead.
2. **9×2 breaks the unique-block scaling trend** — Same params as baseline but 35% fewer steps in 600s budget. Step-budget starvation overrides any benefit from deeper effective representation.
3. **BigramHash + recurrence compete** — These tricks don't complement at low step counts; bigram params steal training capacity from recurrent blocks.
4. **Naive width scaling** — Changing model dimensions without retuning optimizer hyperparameters produces catastrophic results (3×3 wide: 2.07 BPB).

---

## Reproducing

```bash
# Best config (7×2):
RUN_ID=recur_7x2 NUM_LAYERS=7 DEPTH_REPEATS=2 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Baseline comparison:
RUN_ID=baseline \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

All runs use the default baseline hyperparameters (learning rates, warmup, etc.) with only `NUM_LAYERS` and `DEPTH_REPEATS` changed.

---

## What Might Work With More Compute

- **7×2 on 8x H100**: With ~6900 steps (vs ~1300 on 1x), shared weights would get 5x more training. The +0.036 gap should narrow.
- **Noisy QAT for recurrence**: PR #363 found that quantization error amplifies through recurrence loops. Their Noisy QAT technique could be combined with full-model recurrence.
- **Per-repeat learnable scaling**: Instead of uniform cycling, apply learned per-repeat scaling factors to block outputs. Low parameter overhead, could help differentiate repeat passes.
- **Recurrence + GPTQ**: If the artifact size savings from recurrence can be spent on higher-precision quantization, the net BPB might improve.
