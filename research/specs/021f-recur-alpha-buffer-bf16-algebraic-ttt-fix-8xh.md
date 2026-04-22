# Spec 021f — Recur-α register_buffer+bf16+algebraic+TTT-fix, 8×H100 JP single-seed

**Slug:** `recur-alpha-buffer-bf16-algebraic-ttt-fix-8xh`
**Created:** 2026-04-22
**Status:** READY (conditional — launch after 021e completes and depending on outcome)
**Links to:** `research/specs/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh.md`, `research/specs/021c-recur-alpha-param-frozen-mini.md`

## Context

021e stacks TTT α fix + algebraic blend form + bf16 + Parameter container over the original spec 021 baseline. At 8H it's tracking 019b-rerun favorably (−0.0007 mean per-step delta at 2200-3900).

**021f tests whether the Parameter container specifically matters, or if the more idiomatic `register_buffer` form gives equivalent results** once the TTT fix and algebraic blend form are in place. Only code delta vs 021e:

```python
# 021e (d761a22):
self.recur_alpha = nn.Parameter(_recur_alpha_017_endpoint, requires_grad=False)

# 021f (0ad5269):
self.register_buffer("recur_alpha", _recur_alpha_017_endpoint)
```

The optimizer guard (`isinstance(...nn.Parameter)`) short-circuits correctly for buffers — no other code changes needed.

## Hypothesis

`register_buffer` and `nn.Parameter(requires_grad=False)` are numerically equivalent at 8H once TTT α is applied in forward_ttt and the algebraic blend form is used. Today's 4H mini (spec 021c Arm A vs B vs C) already showed no differentiation at 4H. 021f confirms or refutes this at 8H.

**Expected:** 021f matches 021e within pod-variance noise (±0.0005 post-TTT).

## Baseline

- **Primary:** 021e (commit `d761a22`) on same pod. Post-TTT TBD but trending ≤ 1.066.
- **Reference:** 019b-rerun (commit `e93d77d`, same pod) and 019b-original (commit `9517a3b`, post-TTT 1.06628).
- **Target to beat:** #1736 post-TTT 1.06610.

## Expected Δ

Credence distribution (relative to 021e's outcome):

| Bucket | Probability | Outcome |
|---|---|---|
| Matches 021e (±0.0005) | 75% | Buffer form works equally; ship buffer as clean form |
| Regresses from 021e (+0.001-0.003) | 15% | Parameter was subtly load-bearing; keep 021e |
| Beats 021e (−0.001+) | 10% | Buffer form slightly better; ship buffer |

## Accept criteria

**Primary (post-TTT):**

Identical buckets to 021e:
| Post-TTT bpb | Bucket | Next action |
|---|---|---|
| ≤ 1.06400 | Clear beat #1736 | 3-seed if 021f beats 021e, else use 021e 3-seed |
| (1.06400, 1.06610] | Tight beat | Compare to 021e; submit whichever is better |
| (1.06610, 1.06710] | Borderline | Compare to 021e; pick winner |
| > 1.06710 | Miss | Keep 021e as submission |

**Primary scientific question:** does 021f match 021e?

| 021f vs 021e post-TTT | Conclusion |
|---|---|
| Within ±0.0005 | Buffer is equivalent. **Container choice doesn't matter.** Ship whichever is cleaner (buffer). |
| 021f worse by 0.001-0.003 | Parameter container has a small real benefit at 8H. Mechanism unclear but actionable. |
| 021f better by 0.001+ | Buffer is better. Ship buffer. |

## Config diff

Identical to 021e — same env block, same `RECUR_ALPHA_ENABLED=1`, `MATRIX_LR=0.026`, default `ENABLE_LOOPING_AT` (0.35), 596s wallclock.

Only change is the code commit.

## Code changes

**Branch:** `exp/recur-alpha-buffer`
**Commit:** **`0ad5269`** (already pushed)

Stack:
1. `dc0b5f8`: α pass-3 L4 fix (0.97265625)
2. `d070df3`: bf16 dtype + drop .to() cast
3. ~~`8b2d791`~~ skipped (that's the Parameter change 021f reverts)
4. `931bd7c`: TTT α bug fix
5. `d761a22`: Algebraic blend form
6. `0ad5269`: **register_buffer** (reverts 8b2d791's Parameter)

## Hardware ladder

**8×H100 AP-JP-1** — same pod as 021e if possible (warm cache). Sequential after 021e completes.

**Do NOT substitute** other hardware.

## Seed plan

Seed 42 first. If it matches or beats 021e, can 3-seed on same commit for submission.

## Inputs

- Data: CaseOps dataset, JP `/runpod/data/...`
- Tokenizer: `fineweb_8192_bpe.model` bundled
- Hotstart: none

## Run protocol

**Prerequisite:** 021e full pipeline must complete first. Capture 021e's post-TTT number before launching 021f.

```bash
# Preflight (brotli should be installed from 021e but confirm)
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 0ad5269

# Sanity-verify: must have register_buffer for recur_alpha, NOT nn.Parameter
grep "register_buffer.*recur_alpha" train_gpt.py      # must match
grep "nn.Parameter" train_gpt.py | grep recur_alpha   # must return nothing
grep "dtype=torch.bfloat16" train_gpt.py | head -1    # must match (bf16 still there)
grep "0.97265625" train_gpt.py                        # must match (α fix still there)
grep -c "x_before + alpha \* (x_new - x_before)" train_gpt.py  # must be 4 (algebraic form)

mkdir -p /runpod/runs/021f-recur-alpha-buffer-bf16-algebraic-ttt-fix-8xh/seed_42
# Reuse inductor cache from 021e — most kernels cache-hit
# (only the buffer-vs-Parameter graph nodes need re-specialization)

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/021f-recur-alpha-buffer-bf16-algebraic-ttt-fix-8xh/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/021f-recur-alpha-buffer-bf16-algebraic-ttt-fix-8xh/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_021e_8h_jp \
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
  > /runpod/runs/021f-recur-alpha-buffer-bf16-algebraic-ttt-fix-8xh/seed_42/train.log 2>&1

kill $NVSMI_PID
```

## Checkpoints / artifacts

- `final_model.pt` — post-EMA FP state dict
- `final_model.int6.ptz` — quantized submission artifact
- `train.log` — every-100-step tok/s, val_bpb, train_loss, TTT trace
- `diag_nvsmi.csv` — per-GPU telemetry
- `final.json` — `val_bpb_pre_gptq_post_ema`, `val_bpb_post_gptq`, **`val_bpb_post_ttt`**, `stopping_early_at_step`, `layer_loop_enabled_at_step`, `recur_alpha_is_buffer: true`, `recur_alpha_dtype: "bfloat16"`, `blend_form: "algebraic"`, `ttt_alpha_applied: true`

## Stop-early criteria (hard safety)

- NaN/inf in train_loss → halt
- Step time > 2× 019b-rerun's → halt
- `layer_loop_enabled_at_step` outside [2000, 2300] → halt

## Cost estimate

~$10 single-seed. Conditional 3-seed only if 021f is the winner vs 021e.

## Decision tree when both numbers in hand

| 021e post-TTT | 021f post-TTT | Action |
|---|---|---|
| ≤ 1.064 | ≤ 1.064 (matches) | Ship either as 3-seed. Prefer buffer for cleanliness. |
| ≤ 1.064 | > 1.065 | Parameter was load-bearing. Ship 021e 3-seed. |
| 1.064-1.066 | similar | Ship 3-seed of best. |
| > 1.067 | > 1.067 | Both miss. Pivot. |
| ≤ 1.064 | ≤ 1.064 | Pick cleaner form (buffer). |

## Open questions for executor interview

1. **Launch only AFTER 021e post-TTT is captured.** 021f only makes sense in context of comparing to 021e.
2. **Inductor cache reuse:** 021f's graph should cache-hit on most non-α kernels (they're identical to 021e). Expect minimal recompile, ~10-30s.
3. **Same pod preferred** for pod-variance-matched comparison vs 021e.
4. **If 021e clearly beats #1736 by >0.002** (clean bucket), 021f is still worth running for scientific completeness — cheap, and if it matches, buffer is the cleaner form to ship.
5. **If 021e misses #1736,** 021f might be a last-gasp shot — if it happens to win by some pod/numerical quirk. Very low expected value; mostly skip and pivot.
