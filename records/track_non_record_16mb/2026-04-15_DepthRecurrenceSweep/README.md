# Depth Recurrence Sweep: Mapping the Layer Loop Design Space

**Non-Record Submission (Research Contribution)**
**Author:** [@krishs0404](https://github.com/krishs0404)
**Date:** April 15–17, 2026
**Hardware:** RunPod 1×H100 SXM 80GB, 6 runs × 10-min wallclock cap ≈ ~60 GPU-minutes total
**Base:** Current SOTA stack (sp8192, int6 block weights, int8 embeddings, brotli, depth recurrence)
**Best result:** 1.4689 post-quant bpb (SOTA baseline, included as reference)

---

## Summary

Systematic ablation of depth recurrence loop configuration in the current SOTA training stack. Five variants tested against baseline across three axes: which layers to loop (`LOOP_START`, `LOOP_END`) and when to activate the loop (`ENABLE_LOOPING_AT`). Every variant was worse than the SOTA config. The SOTA authors found the right hyperparameters.

The key finding: the middle-layer sweet spot (layers 3–5) is genuine, not arbitrary. Moving the loop to earlier layers, later layers, expanding it, or activating it early all hurt. The minimal 2-layer variant (layers 5–6 only) is surprisingly competitive at +0.006 bpb, suggesting most of the recurrence benefit is concentrated in those two layers specifically.

---

## Motivation

Depth recurrence is one of the distinguishing features of current top submissions, but no public ablation documents which layers to reuse, how many to loop, or when to activate the loop. The current SOTA uses `LOOP_START=3`, `LOOP_END=5`, `ENABLE_LOOPING_AT=0.35` — but these look like they could equally well be 2–6 or 4–7. This sweep answers: does the specific layer range actually matter?

The short answer is yes, and the answer is non-obvious: the middle layers (3–5) are uniquely important, early/late alternatives are significantly worse, and the activation timing matters almost as much as the layer selection.

---

## Experimental Setup

All runs used the SOTA training script with identical configuration except the looping parameters:

- **Hardware**: RunPod 1×H100 SXM 80GB (132 SMs, HBM3)
- **Tokenizer**: sp8192 (8192-vocab SentencePiece)
- **Model**: 11 layers, 512d, 8 heads / 4 KV heads, tied embeddings
- **Training budget**: `MAX_WALLCLOCK_SECONDS=600` (10 min, same as competition runs)
- **GPTQ reserve**: 12s, so effective training = 588s
- **Quantization**: GPTQ int6 block weights, GPTQ int8 embeddings, brotli compression
- **Baseline looping config**: `LOOP_START=3`, `LOOP_END=5`, `ENABLE_LOOPING_AT=0.35`, `NUM_LOOPS=2`
- **Evaluation**: Standard val_bpb + sliding window val_bpb, after dequantization

How the loop index lists are constructed from `LOOP_START`/`LOOP_END`: the looped segment `[loop_start, loop_end]` is repeated `NUM_LOOPS` times. The remaining layers (`[0, loop_start)` and `(loop_end, num_layers)`) are split into encoder (first half) and decoder (second half) with U-Net skip connections. For the baseline, this yields `encoder:[0,1,2,3,4,5,3,4]` and `decoder:[5,3,4,5,6,7,8,9,10]`.

---

## Results

| Experiment | `LOOP_START` | `LOOP_END` | `ENABLE_LOOPING_AT` | Steps | val_bpb (pre-quant) | val_bpb (post-quant) | Δ vs baseline |
|---|---|---|---|---|---|---|---|
| **Baseline (SOTA)** | 3 | 5 | 0.35 | **568** | **1.2885** | **1.4689** | — |
| A — minimal reuse | 5 | 6 | 0.35 | 565 | 1.2971 | 1.4750 | +0.006 |
| D — early layers | 1 | 4 | 0.35 | 538 | 1.2930 | 1.5072 | +0.038 |
| C — late layers | 7 | 10 | 0.35 | 541 | 1.3052 | 1.5181 | +0.049 |
| E — early activation | 3 | 5 | 0.15 | 522 | 1.2985 | 1.5190 | +0.050 |
| B — heavy reuse | 2 | 7 | 0.35 | 451 | 1.3189 | 1.6321 | +0.163 |

The encoder/decoder index lists for each experiment:

| Experiment | Encoder indices | Decoder indices |
|---|---|---|
| Baseline | [0,1,2,3,4,5,3,4] | [5,3,4,5,6,7,8,9,10] |
| A — minimal (LS=5, LE=6) | [0,1,2,3,4,5,6] | [5,6,5,6,7,8,9,10] |
| D — early (LS=1, LE=4) | [0,1,2,3,4,1,2,3,4] | [1,2,3,4,5,6,7,8,9,10] |
| C — late (LS=7, LE=10) | [0,1,2,3,4,5,6,7,8] | [9,10,7,8,9,10,7,8,9,10] |
| E — early act (LS=3, LE=5, ELA=0.15) | [0,1,2,3,4,5,3,4] | [5,3,4,5,6,7,8,9,10] |
| B — heavy (LS=2, LE=7) | [0,1,2,3,4,5,6,7,2,3,4] | [5,6,7,2,3,4,5,6,7,8,9,10] |

---

## Key Findings

### The SOTA config is genuinely optimal, not a lucky guess

Every variant tested was worse than the baseline. The ordering — minimal reuse (A) is closest, then early/late shifts (D, C), then early activation (E), then heavy reuse (B) catastrophically worse — forms a coherent picture of what makes depth recurrence work at the 10-minute training budget.

This is not a case where the SOTA config was picked arbitrarily and any similar config would do. Moving the loop range by just a few layers (early: D, late: C) costs roughly +0.04–0.05 bpb. That's a large penalty for a small change.

### Minimal reuse (2 layers, Exp A) is surprisingly competitive at +0.006

Experiment A loops only layers 5–6 instead of the baseline's 3–5. Despite reusing one fewer layer, performance drops by only 0.006 post-quant bpb. This is the closest any variant came to the baseline, and the gap is small enough to be within run-to-run variance at this training budget.

The implication is that the recurrence benefit is concentrated specifically in layers 5–6. Layers 3 and 4 contribute only marginally. Why layers 5–6? Speculative, but these are mid-depth layers where the model has built reasonable representations from the embedding and early layers, but hasn't yet committed to the final abstract features. Reusing them lets the model refine intermediate representations without perturbing early feature extraction or final classification layers.

On an 8×H100 run where far more training steps are possible, this minimal configuration might be preferable: fewer looped layers means more training steps per wall-clock minute, and the accuracy gap may close with more iterations.

### Heavy reuse is catastrophically worse (+0.163) due to throughput loss

Experiment B expands the loop to layers 2–7 — six layers instead of three. The result is a disaster: post-quant bpb of 1.6321, the worst result by a wide margin, and only 451 training steps completed versus 568 for the baseline.

The step count difference is the key. Looping 6 layers instead of 3 means each forward pass takes roughly twice as long. In a 10-minute budget, this costs ~117 training steps. At this early stage of training (sub-600 steps vs. a 20,000-step schedule), each step matters enormously — the model is still rapidly descending from the initial loss. Losing 117 steps to compute overhead is a severe penalty that the additional depth cannot compensate for.

Heavier looping is only justified if the accuracy-per-step improvement exceeds the step-count penalty. At the 1×H100 / 10-minute scale tested here, it clearly does not. This might change at longer training budgets where the model has already extracted most of the easy gradient signal and additional depth becomes more valuable.

### Early/late layer shifts are symmetrically bad (+0.038 to +0.049)

Experiments C (late: 7–10) and D (early: 1–4) both hurt significantly. The losses are roughly symmetric around the baseline, with late layers slightly worse than early layers. Neither extreme is good.

The early layers (D) fail because layers 1–4 are closest to the raw embedding. Reusing them means the model runs embedding-proximal computation multiple times, but the abstract representations needed for useful recurrence haven't formed yet. The late layers (C) fail for the opposite reason: layers 7–10 are already computing high-level features close to the output. Reusing them duplicates computation that should be done at most once before the final projection.

The middle layers (3–5 in the SOTA) sit at the point where the model has built enough abstraction to benefit from recurrence without those abstractions being so finalized that recomputation is wasteful.

### Early loop activation hurts: the model needs stable representations first

Experiment E uses the same LOOP_START=3, LOOP_END=5 as baseline but activates the loop at 15% of training progress instead of 35%. This yields 522 steps and 1.5190 post-quant bpb — a +0.050 penalty and 46 fewer training steps.

Two effects combine here. First, activating earlier introduces loop overhead earlier, costing steps. Second, and likely more important, the loop activates before the model has learned stable intermediate representations. At 15% progress (roughly step 90), the model's layer-5 outputs are still changing rapidly. Reusing them via the U-Net encoder/decoder causes the looped representations to be built on shifting foundations, degrading the benefit.

The SOTA's 35% threshold appears to be calibrated for when representations have stabilized sufficiently for reuse to be helpful. Earlier than this, recurrence introduces noise rather than refinement.

---

## Implications for Future Work

### Asymmetric recurrence: loop only the highest-impact layers

The +0.006 gap for minimal reuse (layers 5–6 only) compared to the full baseline (layers 3–5) suggests a promising direction: identify the single highest-impact layer within the loop, loop only that one, and spend the recovered wall-clock time on additional training steps.

On 8×H100 where ~4,500–5,500 steps complete in 10 minutes, the tradeoff changes dramatically. The step-count savings from a smaller loop matter less as a fraction of total training, while the accuracy-per-depth benefit of using middle-layer recurrence could accumulate over more steps. This sweep was run at 1×H100 scale; the Pareto frontier between loop width and step count will be different at full competition scale.

### Adaptive loop activation scheduling

The `ENABLE_LOOPING_AT` parameter is currently a fixed fraction of total training. A more principled approach would monitor validation loss, gradient norms, or representation similarity (CKA between layers) and activate the loop when representations have stabilized. This would be especially valuable in runs with different training budgets or batch sizes, where the fixed 35% threshold may not correspond to the same stage of model maturation.

### 8×H100 validation

All findings here are from 1×H100 runs at sub-600-step training. The rankings may hold at 8×H100 scale, but given that heavy reuse's main failure mode is step-count loss (which is a smaller relative penalty with more steps), the hierarchy is not guaranteed to be stable. In particular, Exp B (heavy reuse) might be less catastrophic at full scale if the 117-step loss is a smaller fraction of 5,000+ total steps. Exp A (minimal reuse) might close the gap further with more steps to leverage the saved per-step compute.

---

## Hardware

- **Pod**: RunPod 1×H100 SXM 80GB (HBM3)
- **CUDA**: 12.8
- **Total GPU time**: ~60 minutes across 6 experiments (baseline + 5 ablations)
- **Total cost**: ~$3 at RunPod spot rates

---

## Logs

Raw training logs for all experiments are included in this directory:

| File | Description |
|---|---|
| `baseline.txt` | SOTA baseline (LOOP_START=3, LOOP_END=5, ELA=0.35) |
| `exp_a_minimal.txt` | Exp A: minimal reuse (LOOP_START=5, LOOP_END=6) |
| `exp_b_heavy.txt` | Exp B: heavy reuse (LOOP_START=2, LOOP_END=7) |
| `exp_c_late.txt` | Exp C: late layers (LOOP_START=7, LOOP_END=10) |
| `exp_d_early.txt` | Exp D: early layers (LOOP_START=1, LOOP_END=4) |
| `exp_e_early_act.txt` | Exp E: early activation (ELA=0.15, same loop as baseline) |
| `gptq_ablation.log` | Bonus: simple int8 vs GPTQ int8 for embeddings (+0.003 bpb for GPTQ) |
| `ref_1gpu.txt` | Reference run log from April 14 (575 steps, 1.4684 post-quant bpb) |

---

## Reproducing

All experiments use `train_gpt_sota.py` (the competition SOTA script) with `MAX_WALLCLOCK_SECONDS=600`:

```bash
# Baseline
MAX_WALLCLOCK_SECONDS=600 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
  RUN_ID=baseline torchrun --standalone --nproc_per_node=1 train_gpt_sota.py

# Exp A — minimal reuse
MAX_WALLCLOCK_SECONDS=600 LOOP_START=5 LOOP_END=6 ENABLE_LOOPING_AT=0.35 \
  RUN_ID=exp_a_minimal torchrun --standalone --nproc_per_node=1 train_gpt_sota.py

# Exp B — heavy reuse (warning: ~117 fewer training steps at 1xH100)
MAX_WALLCLOCK_SECONDS=600 LOOP_START=2 LOOP_END=7 ENABLE_LOOPING_AT=0.35 \
  RUN_ID=exp_b_heavy torchrun --standalone --nproc_per_node=1 train_gpt_sota.py

# Exp C — late layers
MAX_WALLCLOCK_SECONDS=600 LOOP_START=7 LOOP_END=10 ENABLE_LOOPING_AT=0.35 \
  RUN_ID=exp_c_late torchrun --standalone --nproc_per_node=1 train_gpt_sota.py

# Exp D — early layers
MAX_WALLCLOCK_SECONDS=600 LOOP_START=1 LOOP_END=4 ENABLE_LOOPING_AT=0.35 \
  RUN_ID=exp_d_early torchrun --standalone --nproc_per_node=1 train_gpt_sota.py

# Exp E — early activation
MAX_WALLCLOCK_SECONDS=600 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.15 \
  RUN_ID=exp_e_early_act torchrun --standalone --nproc_per_node=1 train_gpt_sota.py
```

The `run_sweep.sh` script (included in the sweep_results directory) runs all six sequentially on a single GPU.

---

*Baseline: 568 steps, 1.2885 pre-quant bpb, 1.4689 post-quant bpb, 16,005,909 bytes | Best ablation: Exp A at 1.4750 (+0.006) | Worst: Exp B at 1.6321 (+0.163)*
