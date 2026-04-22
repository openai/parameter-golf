# Spec 021h — Learnable α in fp32 (match 017's precision), 8×H100 JP

**Slug:** `learnable-alpha-fp32-8xh`
**Created:** 2026-04-22
**Status:** READY (launch after 021g completes; depends on 021g's α trajectory data)
**Links to:** `research/specs/021g-017-redux-learnable-alpha-ttt-fix-8xh.md`, `research/specs/017-recur-alpha-full.md`

## Context

021g (learnable Parameter α init=1.0, bf16 dtype, algebraic blend, TTT fix) was launched as the attempted reproduction of 017's pre-quant advantage. Early data (steps 2200-3100) shows:

- **021g per-step train_loss +0.007 above 017** at matched steps
- **α converges to a different basin** than 017:
  - pass-2 L5: 021g 1.383 vs 017 1.430 (−0.047)
  - pass-2 L3: 021g 1.102 vs 017 1.078 (+0.024)
  - pass-2 L4: 021g 1.305 vs 017 1.273 (+0.032)

The trajectory shape matches (same peaks/valleys per data order), but 021g is systematically offset. Learning is happening, just not reaching the same endpoint as 017.

**Hypothesis: bf16 precision at α≈1.0 is the cause.**
- bf16 LSB at 1.0: 1/128 = 0.0078125
- AdamW per-step update on α: ~1e-4 to 1e-5 (grad_norm × lr × momentum factor)
- Per-step update is **100-1000× smaller than bf16's LSB**
- Every update mostly rounds to zero; α is trapped on a coarse 1/128 grid
- α reaches a "nearby but not optimal" basin

## Hypothesis

fp32 storage of α's 6 scalar values restores update precision to 2⁻²³ ≈ 1.2e-7 — well below any AdamW step on these parameters. α can reach 017's exact basin, reproducing its per-step quality.

**Expected results:**
- Pre-quant post-EMA: **~1.06861** (matches 017)
- Post-quant: **~1.0778**
- Post-TTT (with 021 stack's TTT fix giving 019b-style delta): **~1.0653**
- Beats #1736 (1.06610) by 0.00078, beats 019b (1.06628) by 0.00096 — decisive.

## Baseline

- **Primary:** 017 (4dd2d63) pre-quant 1.06861, projected post-TTT-with-fix ~1.06532.
- **Secondary:** 021g (fab6e7f, in flight) pre-quant TBD (~1.069 projected), post-TTT TBD.
- **Target:** #1736 post-TTT 1.06610.

## Expected Δ

Credence distribution (informed by 021g's early data):

| Bucket | Probability | Predicted post-TTT |
|---|---|---|
| Clean beat (fp32 restores 017 quality) | **55%** | 1.0640-1.0660 |
| Tight beat | 25% | 1.0660-1.0665 |
| Marginal miss (017 advantage was pod-luck, not dtype) | 15% | 1.0665-1.068 |
| Regress | 5% | > 1.068 |

Higher confidence than 021g because 021g's failure mode is now understood (bf16 precision), and the fix is targeted and mechanistic.

## Accept criteria

**Primary (post-TTT):**

| Post-TTT bpb | Bucket | Next action |
|---|---|---|
| ≤ 1.06400 | Clear beat | **3-seed 43/44 on same pod** → submission |
| (1.06400, 1.06610] | Beats #1736 | 3-seed for confirmation |
| (1.06610, 1.0665] | Ties #1736 | Same-pod comparison to 021g and 021e; pick best |
| > 1.0665 | Dtype wasn't the fix | Pivot |

**Scientific signal — α reproduces 017's endpoint:**

| α at step ~3500 | Interpretation |
|---|---|
| Matches 017 within 0.01 on all sites | Dtype was the issue; mechanism confirmed |
| Still offset like 021g (differs by 0.02+ on L5) | Not dtype alone; bigger confound |

## Config diff

Identical to 021g — same env block, `RECUR_ALPHA_ENABLED=1`, `MATRIX_LR=0.026`, default `ENABLE_LOOPING_AT` (0.35), 596s wallclock.

Only change is the code commit.

## Code changes

**Branch:** `exp/recur-alpha-buffer`
**Commit:** **`5906820`** (already pushed)

Cumulative stack:
1. `dc0b5f8`: α correct values (irrelevant — α learnable here)
2. `d070df3`: bf16 dtype (reverted below)
3. ~~`8b2d791`~~ / ~~`0ad5269`~~ (frozen — replaced)
4. `931bd7c`: TTT α fix
5. `d761a22`: Algebraic blend form
6. `fab6e7f`: Learnable Parameter α (021g)
7. `5906820`: **dtype bf16 → fp32 + add `.to(x_new.dtype)` casts** ← 021h

Diff from 021g (`fab6e7f`):
```python
# α init:
torch.ones(..., dtype=torch.bfloat16) → torch.ones(..., dtype=torch.float32)

# 4 blend sites:
alpha = self.recur_alpha[pass_off, local_idx]
  → alpha = self.recur_alpha[pass_off, local_idx].to(x_new.dtype)
```

## Hardware ladder

**8×H100 AP-JP-1.** Same pod as 021e/021g preferred for pod-variance-matched comparison. Sequential after 021g completes.

## Seed plan

Seed 42 first. **3-seed (42/43/44) conditional on post-TTT ≤ 1.06610.**

## Run protocol

```bash
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 5906820

# Sanity-verify
grep "nn.Parameter" train_gpt.py | grep recur_alpha                # must match
grep "torch.ones.*dtype=torch.float32" train_gpt.py                # must match (α is fp32)
grep -c "alpha = self.recur_alpha\[.*\].to(x_new.dtype)" train_gpt.py  # must be 4
grep -c "x_before + alpha \* (x_new - x_before)" train_gpt.py      # must be 4 (algebraic)

mkdir -p /runpod/runs/021h-learnable-alpha-fp32-8xh/seed_42

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/021h-learnable-alpha-fp32-8xh/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/021h-learnable-alpha-fp32-8xh/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_021h_8h_jp \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/021h-learnable-alpha-fp32-8xh/seed_42/train.log 2>&1

kill $NVSMI_PID
```

## Checkpoints / artifacts

Standard — `final_model.pt`, `final_model.int6.ptz`, `train.log`, `diag_nvsmi.csv`, `final.json`.

**Additional ask:** capture `recur_alpha: values=...` log lines across entire run so we can compare α trajectory to 017 and 021g at matched steps. This is diagnostic-critical.

## Stop-early criteria (hard safety)

- NaN/inf in train_loss → halt
- Step time > 2× 019b's → halt
- `layer_loop_enabled_at_step` outside [2000, 2300] → halt
- α divergence (|value| > 3.0 after step 3000) → halt

## Cost estimate

~$10 single-seed. ~$30 if 3-seed promotes.

## Dependencies

1. **021g completes first** (captured post-TTT, α trajectory logged)
2. **JP 8×H stock** available
3. **Brotli preflight confirmed**

## Open questions for executor

1. **Watch α trajectory vs 017.** Log α at matched steps (2200, 2300, ..., 4000) and compare. If 021h's α values match 017's within 0.01, dtype was the cause. If still offset, 017's basin was pod-dependent too.

2. **Watch for fp32 performance cost.** α is only 6 scalars; the fp32 → bf16 cast per step is trivial (<1μs). Should have ~identical tok/s to 021g.

3. **Same pod as 021g preferred** for pod-variance-matched comparison. If stock dries, any JP 8×H pod acceptable.

## What this does NOT test

- α init from values other than 1.0 (e.g., 015's α=0 init).
- α evolving during TTT (base params frozen for TTT phase).
- Per-α-site dtype mixing (some fp32, some bf16) — not explored.

## Decision tree after 021h

| 021h post-TTT | Action |
|---|---|
| ≤ 1.064 | Clear beat. 3-seed. Submit as primary. |
| 1.064-1.066 | Beats #1736 and 019b. 3-seed if gap to 021e/021g > 0.001. |
| 1.066-1.0665 | Ties #1736 within noise. Pick winner among 021e/021g/021h via mean. |
| > 1.0665 | Dtype wasn't the cause. 017's advantage was pod-specific. Pivot. |

## Priority

**021h is the current highest-EV variant in the buffer-α arc.** If 021g lands as projected (tied with 021e near 1.066), 021h is the final shot at reproducing 017's decisive-beat projection.

If 021h also falls short, the buffer-α arc is definitively exhausted and the submission question becomes: "3-seed whichever of 021e/021g/021h has the best single-seed."
