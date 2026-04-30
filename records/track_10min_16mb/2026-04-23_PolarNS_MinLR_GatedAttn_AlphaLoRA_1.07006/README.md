# Polar NS + MIN_LR + GatedAttn + Alpha LoRA — 1.07006 BPB

**val_bpb: 1.07005686** (3-seed mean: seeds 1337, 42, 314)

## Results

| Seed | BPB | Train time | Eval time | Artifact |
|------|-----|------------|-----------|----------|
| 1337 | 1.07026727 | 599.6s | 480.7s | 15,977,086 B |
| 42   | 1.06964040 | 599.6s | 474.4s | 15,975,968 B |
| 314  | 1.07026291 | 599.6s | 475.8s | 15,975,620 B |
| **Mean** | **1.07005686** | | | |

All runs: train ≤600s, eval ≤600s, artifact ≤16MB.

## What this submission adds on top of PR #1768

This submission stacks three independently-validated techniques from other authors
onto our PR #1768 stack:

### (1) Polar Express NS coefficients (ported from PR #1344)

Replaces Muon's fixed Newton-Schulz coefficients `(3.4445, -4.775, 2.0315)` (applied
identically 5 times per Muon step) with 5 per-iteration minimax-optimal tuples:

```python
_PE_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
```

Same backend_steps=5, but the per-iteration minimax coefficients produce a
higher-quality polar factor approximation per Muon step.

### (2) MIN_LR=0.10 warmdown floor (from PR #1787)

Floors the LR warmdown at 10% of max instead of 0 — the final ~25% of training
keeps delivering meaningful gradient updates instead of winding down to near-zero.

### (3) Tight budget polish (from PR #1787)

- `GPTQ_RESERVE_SECONDS=0.5` (was 4.0)
- `VAL_LOSS_EVERY=0` (was 4000, disables periodic mid-training val)

Together these reclaim ~15s of the 600s training budget for additional depth-3
training steps, visible in the higher step counts vs prior submissions.

## Stack summary

All techniques and their origins:

| Component | Origin |
|-----------|--------|
| SP8192 + triple depth recurrence + parallel residuals | @bigbag PR #1493, @EthanYangTW PR #1523 |
| VarLen attention + Fused Triton MLP + doc-independent LoRA TTT | @samacqua PR #1530 |
| Phased TTT | @romeerp PR #1610 |
| Multi-Phase Global SGD + Trimmed GPTQ + MATRIX_LR=0.026 | @dexhunter |
| Gated Attention | @dexhunter PR #1736 |
| Alpha/rank LoRA scaling + Warm-start A + WD=1.0 + alpha=144 | **this author, PR #1767** |
| Gate mirror in LoRA-TTT forward path + per-row int8 gate quant | **this author, PR #1768** |
| Polar Express NS coefficients | Ported from PR #1344 |
| MIN_LR=0.10 + GPTQ_RESERVE=0.5 + VAL_LOSS_EVERY=0 | Ported from @nprime06 PR #1787 |

## 3-seed trajectory

| Seed | 1.07326 (PR #1767 mean-reproduction) | PR #1767 | PR #1768 | **This PR** |
|------|---:|---:|---:|---:|
| 1337 | 1.07423 | 1.07189 | 1.07146 | **1.07027** |
| 42   | 1.07341 | 1.07248 | 1.07014 | **1.06964** |
| 314  | 1.07214 | 1.07189 | 1.07082 | **1.07026** |
| Mean | 1.07326 | 1.07209 | 1.07081 | **1.07006** |

Every seed improves monotonically across each submission.

## Legality (Issue #1017)

- **Condition 1 (Causal)**: single left-to-right pass.
- **Condition 2 (Full normalized distribution)**: standard softmax over 8192 SP tokens.
- **Condition 3 (Score-before-update)**: each chunk scored in `forward_ttt_train` before the optimizer step on it.
- **Condition 4 (Single pass)**: one left-to-right pass, no rescoring.

## Reproduction

```bash
export DATA_DIR=/path/to/parameter-golf/data
torchrun --standalone --nproc_per_node=8 train_gpt.py        # seed 1337
SEED=42  torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters hardcoded as defaults in `train_gpt.py`:
`TTT_LORA_RANK=128`, `TTT_LORA_ALPHA=144`, `TTT_WARM_START_A=1`, `TTT_WEIGHT_DECAY=1.0`,
`GATED_ATTN_ENABLED=1`, `GATED_ATTN_INIT_STD=0.005`, `POLAR_EXPRESS_NS=1`, `MIN_LR=0.10`,
`GPTQ_RESERVE_SECONDS=0.5`, `VAL_LOSS_EVERY=0`, `PHASED_TTT_ENABLED=1`, `PHASED_TTT_NUM_PHASES=3`.
