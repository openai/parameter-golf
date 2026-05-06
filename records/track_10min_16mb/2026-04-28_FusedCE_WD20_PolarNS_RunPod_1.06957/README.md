# Fused softcap CE + WD=2.0 (warm-start stability fix) — 1.06957 BPB

**val_bpb: 1.06957227** (3-seed mean: seeds 1337, 42, 314)

## Results

| Seed | BPB | Train | Eval | Artifact |
|------|-----|-------|------|----------|
| 1337 | 1.07009653 | 599.6s | 393.0s | 15,975,995 B |
| 42   | 1.06919893 | 599.4s | 478.4s | 15,978,915 B |
| 314  | 1.06942136 | 599.7s | 397.5s | 15,979,029 B |
| **Mean** | **1.06957227** | | | |

All runs: train ≤600s, eval ≤600s, artifact ≤16MB.

## Novel contribution: WD=2.0 unlocks Fused CE + warm-start LoRA

Adding the Fused softcap CE Triton kernel (ported from PR #1787) on top of our PR
#1768 stack (which uses warm-start LoRA A) caused **TTT divergence on seeds 314 and 1337**
while seed 42 trained cleanly:

| Config | Seed 42 | Seed 314 | Seed 1337 |
|--------|---------|----------|-----------|
| Fused CE + warm-start + WD=1.0 | 1.06923 | **1.12144 (diverged)** | **1.12145 (diverged)** |
| Fused CE + no warm-start + WD=1.0 | 1.07078 | 1.07078 | 1.07162 |
| **Fused CE + warm-start + WD=2.0** (this) | **1.06920** | **1.06942** | **1.07010** |

The diagnosis: the Fused CE kernel does fp32 accumulation in-register vs the eager
`softcap*tanh + F.cross_entropy` path's bf16/fp32 mix. The micro-numerical-difference
shifts the optimizer state at the end of training in a way that, combined with
warm-start A's accumulated feature directions, pushes the LoRA optimizer into a
bad attractor on seeds 314/1337. Disabling warm-start eliminates the divergence
but also loses the warm-start gain we got in PR #1768 (~−0.001 BPB).

Raising `TTT_WEIGHT_DECAY` from 1.0 → 2.0 regularizes the LoRA enough that the
across-batch A drift stays bounded for all 3 seeds. With WD=2.0:

- All three seeds beat their PR #1768 record results (per-seed deltas −0.00017 to −0.00084).
- 3-seed mean drops from PR #1768's 1.07081 → **1.06957 (−0.00124)**.

## Stack summary

| Component | Origin |
|-----------|--------|
| **Novel: WD=2.0 for Fused CE + warm-start stability** | this author |
| Fused softcap CE Triton kernel | @nprime06 (PR #1787) |
| Polar Express NS coefficients | PR #1344 |
| MIN_LR=0.10, GPTQ_RESERVE=0.5, VAL_LOSS_EVERY=0 | @nprime06 (PR #1787) |
| Per-head GatedAttn + per-row int8 gate quant + gate mirror in LoRA-TTT path | this author (PR #1768) |
| Alpha/rank LoRA scaling, warm-start A, alpha=144 | this author (PR #1767) |
| Multi-phase global SGD, trimmed GPTQ, MATRIX_LR=0.026 | @dexhunter |
| VarLen attention, Fused Triton MLP, doc-independent LoRA TTT | @samacqua (PR #1530) |
| Phased TTT | @romeerp (PR #1610), @dexhunter |
| Triple recurrence, parallel residuals | @bigbag (PR #1493), @EthanYangTW (PR #1523) |
| Legal TTT framework | @abaybektursun (PR #549) |

## Hardware

Trained on **RunPod 8xH100 80GB SXM** (not Zoom MLP cluster). PyTorch 2.9.1+cu128, FA3, Triton 3.5.1. Identical SP8192 SentencePiece tokenizer and FineWeb document selection as upstream HF dataset `willdepueoai/parameter-golf`. Validation set is the standard `fineweb_val_*.bin` shard from the SP8192 tokenization.

## Legality (Issue #1017)

- **Condition 1 (Causal)**: single left-to-right pass; LoRA at `t` depends only on earlier tokens of the same doc.
- **Condition 2 (Full normalized distribution)**: standard softcap-tanh + softmax over 8192 SP tokens.
- **Condition 3 (Score-before-update)**: each chunk scored before the LoRA grad step.
- **Condition 4 (Single pass)**: one left-to-right pass, no rescoring.
- **Fused CE is training-only.** The `forward_logits` eval path keeps eager `logit_softcap * torch.tanh(logits/softcap)` numerics — only the training forward uses the fused kernel. Per-token byte counting is unchanged.

## Reproduction

```bash
export DATA_DIR=/path/to/parameter-golf/data

torchrun --standalone --nproc_per_node=8 train_gpt.py        # seed 1337
SEED=42  torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are hardcoded as defaults: `TTT_WEIGHT_DECAY=2.0`, `FUSED_CE_ENABLED=1`,
`POLAR_EXPRESS_NS=1`, `MIN_LR=0.10`, `GPTQ_RESERVE_SECONDS=0.5`, `VAL_LOSS_EVERY=0`,
`TTT_LORA_RANK=128`, `TTT_LORA_ALPHA=144`, `TTT_WARM_START_A=1`, `GATED_ATTN_ENABLED=1`,
`PHASED_TTT_ENABLED=1`, `PHASED_TTT_NUM_PHASES=3`.
