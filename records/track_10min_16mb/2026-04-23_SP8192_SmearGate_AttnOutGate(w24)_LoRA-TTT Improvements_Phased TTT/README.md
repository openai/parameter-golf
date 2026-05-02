# Record: SP8192 + SmearGate + AttnOutGate(w24) + LoRA-TTT Improvements + Phased TTT — val_bpb 1.06991 (3-seed mean)

## Summary

**val_bpb = 1.06991** (3-seed mean, std 0.00061) | **~15.9 MB** | 8xH100 SXM

Combines four previously validated techniques on the SP8192 base, **no Casefold/CaseOps**:
1. **SmearGate** (PR #1667) — forward token smear gate, zero-init transparent
2. **Attention Output Gate width=24** (PR #1667, extended) — per-head multiplicative gate on attention output with 24-dim input (vs default 12), zero-init transparent
3. **LoRA TTT improvements** (PR #1767) — alpha=144 scaling, rank=128, warm-start A (keep A matrix, reset only B), WD=1.0
4. **Multi-Phase Global SGD TTT** (PR #1700/#1767) — 3 global SGD phases on 2000 prefix docs at eval time

Key combination not previously submitted: PR #1667 used standard TTT; we apply phased TTT (PR #1700/#1767 style) on top of the SmearGate+AttnGate architecture. This combination improves both pre-quant quality (SmearGate+AttnGate) and post-quant TTT recovery (phased global SGD).

## 3-Seed Results (8xH100 SXM)

| Seed | Steps | Train time | Post-phased-TTT val_bpb | Eval time | Artifact (bytes) |
|------|------:|-----------:|------------------------:|----------:|-----------------:|
| 42   | ~4853 | ~596s      | 1.06921                 | ~500s     | 15,944,988       |
| 1337 | ~4686 | ~596s      | 1.07037                 | ~401s     | 15,946,853       |
| 0    | ~4803 | ~596s      | 1.07015                 | ~434s     | 15,943,381       |
| **Mean** | | | **1.06991** | | |
| **Std** | | | **0.00061** | | |

## Key Techniques

### 1. SmearGate (from PR #1667 by @MarioPaerle)
`x_t += λ·σ(W·x_t[:12])·x_{t-1}` — forward token smear via sigmoid-gated residual.
- Model-level SmearGate (shared across all layers, not per-block)
- Zero-init (W=0, λ=0) → transparent at start
- Width: 12 dims (smear_gate_width=12)

### 2. Attention Output Gate (extended, from PR #1667 by @MarioPaerle)
Per-head multiplicative gate on attention output: `y = y * 2·σ(W·x[:12])`
- CastedLinear(24, num_heads) per block, zero-init
- Applied before out_proj, also in TTT LoRA path (`_block_with_lora`, `_parallel_block_with_lora`)
- 96 params per layer × 11 layers = 1,056 params total

### 3. LoRA TTT Improvements (from PR #1767 by @renqianluo)
- `TTT_LORA_ALPHA=144` (vs no alpha scaling previously)
- `TTT_LORA_RANK=128` (vs 96)
- Warm-start A: keep A matrix across TTT batches, reset only B
- `TTT_LORA_WD=1.0` (stronger regularization)

### 4. Phased Global SGD TTT (from PR #1700 by @jorge-asenjo, via #1767 by @renqianluo)
- `PHASED_TTT_NUM_PHASES=3`
- `PHASED_TTT_PREFIX_DOCS=2000`
- `GLOBAL_TTT_LR=0.001`
- At phase boundaries, run global SGD on 2000 prefix docs

## Hyperparameters

```bash
SEED=<42|1337|0> \
QK_GAIN_INIT=5.25 \
SMEAR_GATE=1 \
GATE_ATTN_OUT=1 \
GATE_ATTN_WIDTH=24 \
GPTQ_RESERVE_SECONDS=4 \
GPTQ_CALIBRATION_BATCHES=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Other defaults from base code (PR #1767):
- `MATRIX_LR=0.026`, `EMBED_BITS=7`, `MLP_CLIP_SIGMAS=12.0`, `ATTN_CLIP_SIGMAS=13.0`
- `PHASED_TTT_NUM_PHASES=3`, `GLOBAL_TTT_LR=0.001`, `TTT_LORA_RANK=128`, `TTT_LORA_ALPHA=144`

## Rule Compliance (Issue #1017 Track B)

1. **Strict causal dependence**: LoRA state built from prefix tokens only. Global SGD at phase boundaries operates on already-scored prefix docs only.
2. **Full normalized distribution**: Standard softmax over SP8192 vocab. No n-gram cache, no logit biasing.
3. **Score before update**: Per-chunk: full forward pass and loss accumulation BEFORE any LoRA gradient step. Global SGD: invoked after phase boundary, on already-scored docs only. Last chunk of each phase explicitly skipped.
4. **Single left-to-right pass**: Each token scored exactly once, no rescoring.

- Artifact ≤ 16,000,000 bytes: ✅ (all seeds ~15.94 MB)
- Training ≤ 600s: ✅ (all seeds ~596s)
- Eval ≤ 600s: ✅ (all seeds ~401-500s TTT)
- No val data during training: ✅
- Note: sliding window eval is not included in this stack (PR #1700 base disables it)

## Attribution

- @MarioPaerle — PR #1667 for SmearGate and AttnOutGate
- @renqianluo — PR #1767 for LoRA TTT improvements (alpha/rank/warm-start A/WD) and phased TTT integration
- @jorge-asenjo — PR #1700 for Multi-Phase Global SGD TTT concept
- @bigbag — PR #1493 prior merged SOTA

🤖 Generated with [Claude Code](https://claude.com/claude-code)
