# RecurLoRA: Quantization-Stable Shallow Recurrence with Low-Rank Corrective Adapters

**val_bpb: TBD** (3-seed runs pending) | **~15.9 MB artifact** | 8xH100 SXM

## Summary

**RecurLoRA** introduces bounded-depth recurrence with per-pass low-rank attention corrections, enabling increased effective model depth within a fixed parameter budget without triggering quantization error amplification.

Layers 4 and 5 of an 11-layer transformer are each repeated once, yielding **13 virtual layers from 11 physical layers** at a cost of just **28KB** (0.18% of the 16MB budget). The second pass applies rank-2 LoRA corrections to attention projections (Q, K, V, O) only, with RMSNorm gating and a learnable scaling factor.

Built on the PR #1179 stack (1.1105 BPB) with validated hyperparameter improvements, targeting sub-1.10 BPB.

## Status

Implementation complete and validated for:
- Forward/backward correctness
- Gradient flow across recurrent passes (warm-initialized LoRA: A and B active from step 1)
- Parameter budget compliance (28KB overhead, fp16 passthrough)

Full training runs (3 seeds + ablations) queued pending compute. Additional compute should further improve shared weight optimization and adapter specialization.

## Motivation

Weight sharing is one of the most natural ways to increase effective depth under a parameter budget. Yet it has been the most reliably failing technique in this competition:

| PR | Approach | Result |
|----|----------|--------|
| #363 | 3-cycle recurrence | +4.3 BPB (quant error amplifies ~900x) |
| #344 | Full weight sharing | 2x slower, no quality gain |
| #579 | 6x2 loop recurrence | 1.1478 BPB, GPTQ compounds multiplicatively |
| **#686** | **2-layer shallow recurrence** | **1.1182 BPB (competitive)** |

The pattern is clear: **aggressive recurrence (3+ cycles) is catastrophic, but shallow recurrence (1-2 repeats) survives** int6 GPTQ quantization. PR #686 demonstrated this but used identical weights for both passes, leaving the model unable to specialize its behavior per pass.

## RecurLoRA: The Approach

RecurLoRA addresses the "both passes are identical" limitation by introducing per-pass low-rank corrective adapters on attention projections, while keeping the expensive MLP weights fully shared.

### 1. Low-rank attention corrections (rank 2)

For each recurrent layer, the second pass adds learned corrections to the attention weight matrices:

```
Q_pass2 = Q_shared + alpha * (B_q @ A_q)
K_pass2 = K_shared + alpha * (B_k @ A_k)
V_pass2 = V_shared + alpha * (B_v @ A_v)
O_pass2 = O_shared + alpha * (B_o @ A_o)
```

MLP weights remain fully shared -- they are the dominant parameter cost and empirically more stable under sharing than attention weights.

Both A and B are warm-initialized with N(0, 1e-3) so corrections are active from step 1 -- critical under a 600s training budget where cold-start delays waste optimization steps.

### 2. RMSNorm before repeat

The residual stream is normalized before re-entering the shared block, preventing distribution drift between passes. Without this, the second pass sees a shifted input distribution that the shared weights were not optimized for.

### 3. Learnable scaling factor (alpha)

A per-layer scalar (initialized at 0.6) controls correction magnitude, allowing the model to:
- Start with moderate corrections and adjust during training
- Shrink corrections if sharing is already sufficient
- Avoid overcorrection that could destabilize the shared structure

### Design rationale

- **Rank 2** (not higher): At 512d, rank-2 LoRA provides 4 directions of correction per matrix. Higher ranks risk overfitting per-pass corrections and destroying shared structure.
- **Layers 4, 5** (encoder middle): Intermediate representations benefit most from recurrence. Early layers handle token embedding (low value), late decoder layers already have U-Net skip connections providing representational flexibility.
- **No MLP LoRA**: MLPs at 3x width (1536) are the dominant parameter cost. Adding LoRA there would bloat the budget without proportional gain, and MLPs are empirically more stable under sharing than attention.

### Parameter overhead

| Component | Params | Bytes (fp16) |
|-----------|--------|-------------|
| LoRA (Q,K,V,O) x 2 layers | 14,334 | 28,668 |
| Alpha scalars x 2 | 2 | 4 |
| RMSNorm (no learnable params) | 0 | 0 |
| **Total** | **14,336** | **28,672 (28KB)** |

Kept as fp16 passthrough in the quantized artifact (below the 65536-element threshold), avoiding additional quantization error -- the exact failure mode that kills deeper recurrence.

## Why This Matters Under 16MB / 600s Constraints

In this regime, increasing depth via standard parameter scaling is prohibitively expensive -- each additional transformer layer costs ~2.5MB in the compressed artifact.

RecurLoRA reallocates parameters by:
- Sharing expensive transformer blocks across passes
- Adding minimal low-rank corrections (0.18% overhead) for per-pass specialization

This effectively reallocates parameters from duplicated layers into increased depth, improving expressivity under a fixed 16MB budget:
- Higher effective depth at constant parameter budget
- Improved expressivity without increasing quantization pressure

The approach should scale favorably with additional compute, as shared weights benefit from more optimization steps (receiving 2x gradient signal per forward pass) while adapters specialize per-pass behavior as base weights stabilize.

## Supporting Optimizations

Independently validated changes applied on top of the PR #1179 base stack:

| Change | Evidence | Expected gain |
|--------|----------|---------------|
| LeakyReLU slope 0.5 -> 0.9 | Issue #140 sweep: monotonic improvement, 0.9 beats 0.5 by 0.013 BPB | ~0.010 |
| QK-Gain 1.5 -> 4.0 | PR #1176 ablation | ~0.006 |
| Score-first Muon-TTT | PRs #549, #1176: legal variant, score then train on scored tokens | ~0.003 |
| Warmdown 3500 -> 4000 | PRs #1145, #1179 | ~0.001 |

## Architecture

```
Input -> Embedding + BigramHash(2816x160) -> RMSNorm -> SmearGate
  -> [Encoder: layers 0-4, with recurrence on layer 4]
  -> [Decoder: layers 5-10, with recurrence on layer 5, U-Net skip connections]
  -> Final RMSNorm -> Tied LM Head
```

- 11 physical layers, **13 virtual layers** (layers 4, 5 repeated with RecurLoRA)
- 512d, 8 GQA heads / 4 KV heads, MLP 3x (1536)
- LeakyReLU(0.9)^2 activation
- Partial RoPE (16/64 dims), XSA on all layers
- QK-Gain 4.0, LayerNorm scale 1/sqrt(layer+1)
- EMA(0.997) + SWA(every 50), Parallel Muon with Split-LR
- Full Hessian GPTQ with AR self-generated calibration data
- Brotli-11 + byte-shuffle compression

## Ablation Plan

| Step | Experiment | Purpose |
|------|-----------|---------|
| 1 | Hyperparameters only (no recurrence) | Establish control |
| 2 | Recurrence without LoRA | Confirm parity with PR #686 (~1.1182 BPB) |
| 3 | RecurLoRA: recurrence + rank-2 LoRA | Measure LoRA contribution (3 seeds) |
| 4 | Alpha sweep: [0.3, 0.5, 0.6, 0.8] | Optimal correction strength |
| 5 | Layer sweep: [3,4], [4,5], [5,6] | Optimal recurrence position |
| 6 | Report mean +/- std across seeds | Statistical validation |

## Run Command

```bash
NEGATIVE_SLOPE=0.9 QK_GAIN_INIT=4.0 TTT_ENABLED=1 \
BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=160 WARMDOWN_ITERS=4000 \
RECUR_LAYERS=4,5 RECUR_LORA_RANK=2 RECUR_ALPHA_INIT=0.6 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To disable recurrence (for ablation):
```bash
RECUR_LAYERS="" torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Base stack**: PR #1179 by @dexhunter (Split-LR + BigramHash + Full GPTQ + Brotli)
- **Shallow recurrence insight**: PR #686 by @msisovic (demonstrated 2-repeat quantization survival)
- **LeakyReLU sweep**: @MatoTeziTanka (issue #140)
- **QK-Gain**: PR #1125/#1176
- **TTT**: PR #549 by @abaybektursun
