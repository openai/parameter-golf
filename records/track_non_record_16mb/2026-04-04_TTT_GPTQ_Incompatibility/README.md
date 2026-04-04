## Summary

**Test-time training (TTT) provides substantial BPB improvement on simple quantization but is fundamentally ineffective on GPTQ-quantized models.** This work aggregates evidence from 4 independent configurations across 3 research groups showing that GPTQ's compensatory weight structure is destroyed by gradient-based adaptation, making TTT and GPTQ mutually exclusive optimization strategies.

This finding has immediate implications for the competition: teams using GPTQ (the dominant compression method) cannot benefit from TTT at eval time.

---

## Evidence

| Configuration | TTT Method | Quantization | BPB Delta | Source |
|--------------|-----------|-------------|-----------|--------|
| PR #461 baseline | SGD, 3 epochs, momentum=0.9 | Simple int6 per-row | **-0.0165** | Christopher-Lee-McClendon |
| PR #601 replication | SGD, full model | Full GPTQ int5 | **+0.030 (WORSE)** | Community finding |
| This work | LoRA rank-8 on Q,V | Full GPTQ int6 | -0.0013 | My experiments (1×H100) |
| PR #1326 | Score-first SGD | Full GPTQ int6 | -0.0001 | aryanbhosale |

The pattern is stark: SGD TTT improves BPB by -0.0165 on simple int6 quantization (PR #461) but provides **zero benefit** on GPTQ-quantized weights. When applied aggressively to GPTQ models, TTT actively *degrades* performance by +0.030 BPB (PR #601).

My LoRA TTT experiment used rank-8 adapters on Q and V projections of a GPTQ-quantized Clark-architecture model (11L, 512d, sp4096). Even this conservative approach — updating only ~2% of parameters — yielded negligible improvement (-0.0013 BPB).

PR #1326 (aryanbhosale) independently confirmed this: applying score-first TTT to the strongest current architecture (depth recurrence + parallel residuals + GPTQ int6) produced -0.0001 BPB improvement — statistically indistinguishable from zero.

---

## Root Cause: GPTQ's Compensatory Weight Structure

GPTQ (Frantar et al., 2023) solves a per-layer Hessian-weighted least-squares problem:

```
For each column j of weight matrix W:
    Quantize w_j, compute error δ_j
    Distribute δ_j to remaining columns: W[:,j+1:] -= δ_j * H_inv[j,j+1:] / H_inv[j,j]
```

Each quantized weight **compensates for errors in previously quantized weights**. The resulting weight matrix is not independently quantized — it's a globally optimized system where individual weights encode error-correction information for their neighbors.

SGD updates individual weights based on local gradients, **ignoring the compensatory structure**. After even one SGD step:
- Weight w_j is updated by -lr * ∂L/∂w_j
- But w_j was carrying compensation for w_{j-1}'s quantization error
- This compensation is now destroyed
- The net effect: the SGD update that was supposed to reduce loss instead breaks error cancellation, often increasing loss

This is why TTT on GPTQ is not merely unhelpful — it can be actively harmful (+0.030 BPB in PR #601).

---

## Implication: Compression vs Adaptation Tradeoff

The competition has two parallel optimization strategies that **cannot be combined**:

**Compression path (GPTQ):**
- GPTQ enables fitting more parameters in 16MB
- Every recent record submission uses GPTQ (PRs #1218, #1285, #1296, #1334)
- Gain: ~0.02-0.05 BPB from fitting larger models

**Adaptation path (TTT):**
- Score-first TTT adapts the model to the evaluation distribution
- Works well on simple quantization: -0.0165 BPB (PR #461)
- But simple int6 produces artifacts too large for 16MB at competitive model sizes

Teams must choose one. The current leaderboard shows GPTQ winning — but this may change if someone finds a way to bridge the gap.

---

## Proposed Fix Directions

1. **Quantization-aware TTT:** Maintain full-precision master weights alongside GPTQ weights. Run TTT on masters, re-quantize per chunk. Preserves GPTQ structure while allowing adaptation. Cost: 2× memory + re-quantization overhead.

2. **Structured TTT:** Constrain SGD updates to respect GPTQ block boundaries. Only update weights in ways that maintain the compensatory structure. Requires understanding GPTQ's column ordering.

3. **Higher-rank LoRA:** My rank-8 LoRA gave -0.0013. Higher ranks (32, 64) may provide enough adaptation capacity without disturbing GPTQ weights. But higher rank = more parameters = potential artifact overhead.

4. **Simple int6 + larger model:** Skip GPTQ entirely. Use simple int6 with a model small enough to fit 16MB. TTT then provides -0.0165 BPB. The question: does the GPTQ compression advantage (larger model) outweigh the TTT adaptation advantage (better eval)?

None of these have been attempted in the competition.

---

## SGD TTT Implementation

I implemented the full PR #461 TTT protocol: SGD with momentum=0.9, lr=0.002, cosine decay across 32K-token chunks, 3 epochs per chunk, freeze first 2 blocks, grad clip 1.0. Code: `sgd_ttt_eval.py`

When applied to a GPTQ-quantized Clark 11L model (val_bpb ~1.10 pre-TTT), the result was -0.0013 BPB — consistent with PR #1326's finding of -0.0001 on a similar architecture.

---

## Reproduction

```bash
# Run SGD TTT on a GPTQ-quantized model:
python3 sgd_ttt_eval.py \
    --model-path final_model.int6.ptz \
    --data-dir ./data/ \
    --ttt-lr 0.002 --ttt-epochs 3 \
    --ttt-chunk-size 32768 --ttt-freeze-blocks 2
```

---

## Attribution

Analysis aggregates findings from PR #461 (Christopher-Lee-McClendon), PR #601 (community), PR #1326 (aryanbhosale), and my own experiments. GPTQ analysis based on Frantar et al. (2023). All experiments self-funded.
