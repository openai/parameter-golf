# PR #1738 + PreQuant TTT LR=1e-3 + Unfrozen

**val_bpb 1.02767 (3-seed mean, std 0.00049)** on the 10 min / 16 MB track.

## Summary

This PR applies PR #1738 (alertcat) unchanged and retunes two hyperparameters of the **pre-quant TTT phase** introduced in PR #1735 (AjAnubolu):

| env var | PR #1738 default | this PR |
|---|---|---|
| `PREQUANT_TTT_LR` | `5e-4` | **`1e-3`** |
| `PREQUANT_TTT_FREEZE_BLOCKS` | `2` | **`0`** |

No changes to architecture, tokenizer, main training, or evaluation. The submitted `train_gpt.py` is PR #1738's `train_gpt.py` with two `os.environ.setdefault` lines prepended that flip those two defaults.

## Why it works

PR #1735 introduced an 8-GPU parallel pre-quant AdamW TTT pass (21 epochs, epoch-level cosine LR, federated averaging across GPUs). PR #1738 inherited the defaults. Two observations during a small sweep on the alertcat base:

1. **TTT was undertrained at the default LR.** At `PREQUANT_TTT_LR=5e-4` the TTT loss was still descending at epoch 21 (final epoch val_bpb ~1.019). Doubling to `1e-3` drove the final TTT val_bpb to ~1.015 — a 0.004-nat improvement pre-quantization.
2. **Freezing the first 2 blocks during TTT was unnecessary.** With only 21 epochs on held-out legal tokens there is no overfitting regime to protect against; freezing just reduces adapt capacity. Setting `PREQUANT_TTT_FREEZE_BLOCKS=0` dropped the final TTT val_bpb to ~1.012 (another -0.003 pre-quant) and passed through quantization.

Both effects stack monotonically. Higher LRs (1.5e-3, 2e-3) diverged under the 21-epoch budget; smaller `freeze_blocks` values (3, 2, 1, 0) improved monotonically, with `0` winning. Size stays under the 16 MB limit in all three confirm runs.

## 3-seed results (8× H100 80GB SXM, 10-min train / 10-min eval budgets)

| Seed | val_loss | val_bpb (sliding) | val_bpb (fixed) | artifact bytes |
|------|---------:|------------------:|----------------:|---------------:|
| 43   | 2.24778  | **1.02715**       | 1.03633         | 15,997,720     |
| 44   | 2.24909  | **1.02775**       | 1.03706         | 15,996,585     |
| 45   | 2.24992  | **1.02812**       | 1.03744         | 15,998,726     |
| **mean** | **2.24893** | **1.02767** | 1.03694 | 15,997,677 |
| **std**  |           | **0.00049**  |         |               |

All artifact sizes pass the 16 MB constraint. `val_bpb` reported above is the sliding-window (stride-64) eval used by the current PR #1735/#1738 lineage.

## Statistical significance

Claim: beats PR #1738 (`val_bpb` 1.03540, 3-seed mean) by ≥ 0.005 nats at p < 0.01.

- observed Δ = 1.03540 − 1.02767 = **0.00773 nats** (vs required 0.005)
- our sample std = 0.00049 on n=3 → standard error 0.00028
- one-sided t-test vs μ₀ = 1.03540 − 0.005 = 1.03040: t = (1.03040 − 1.02767) / 0.00028 ≈ 9.7, df = 2 → **p ≈ 0.005**

Note that PR #1738's reported 3-seed mean (1.03540, std 0.00057) was obtained on pytorch 2.9.1+cu128, whereas these runs were performed on pytorch 2.5.1+cu124 (vast.ai `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`). On the same hardware and pytorch 2.5.1, a reproduction of the PR #1738 defaults landed at 1.03612 single-seed (seed 42), 0.0007 above PR #1738's claim — the fixed stack drift is smaller than the improvement reported here.

## How to reproduce

```bash
# 8× H100 SXM, /workspace/parameter-golf = this repo root
# (also works with PR #1738's template inputs unchanged)
export DATA_DIR=/workspace/data
# defaults baked into train_gpt.py; exporting explicitly is not required
export PREQUANT_TTT_LR=1e-3
export PREQUANT_TTT_FREEZE_BLOCKS=0
export MAX_WALLCLOCK_SECONDS=600
export TTT_ENABLED=0    # eval-time TTT stays off, as in PR #1738

for SEED in 43 44 45; do
  env RUN_ID=frz0_s${SEED} SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 \
    | tee train_seed${SEED}.log
done
```

Dataset (CaseOps tokenizer + byte sidecar) must be pre-downloaded into `$DATA_DIR` per PR #1729 (`romeerp/parameter-golf-caseops-v1`).

## Attribution

- PR #1738 base (unchanged): @alertcat
- PR #1735 (Parallel Pre-Quant AdamW TTT): @AjAnubolu
- PR #1729 (CaseOps tokenizer + byte sidecar): @romeerp
- PR #1493 (QK-Gain 5.25): @bigbag
- PR #1412 (parallel residuals): @Robby955
- PR #1331 (depth recurrence): @dexhunter
- PR #1394 (SP8192 + GPTQ SDClip): @clarkkev
- This PR: two env-var tunes (Julian Quick / @kilojoules)
