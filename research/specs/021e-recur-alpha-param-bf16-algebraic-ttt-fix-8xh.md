# Spec 021e — Recur-α Parameter+bf16+algebraic+TTT-fix, 8×H100 JP single-seed

**Slug:** `recur-alpha-param-bf16-algebraic-ttt-fix-8xh`
**Created:** 2026-04-22
**Status:** READY (pending 4×H Arm A' + Arm E sanity check per spec 021c update)
**Links to:** `research/specs/021d-recur-alpha-param-frozen-8xh.md`, `research/specs/021c-recur-alpha-param-frozen-mini.md`

## Context

021d (commit `8b2d791`) shipped to 8×H JP with two unknown bugs:

1. **TTT α bug:** the entire 021 lineage branched from 017 (commit `4dd2d63`), which had the TTT α bug (α applied in forward_logits but NOT in forward_ttt). 019 fixed this; 021 family didn't inherit the fix. Result: 021d's TTT delta was expected to track 017's buggy −0.01013 rather than 019b's fixed −0.01249 — costing ~0.0025 of post-TTT.

2. **Wrong blend form:** we've been using `x = α·x_new + (1−α)·x_before` (manual add) vs 019b-submission's actual form `x = x_before + α·(x_new − x_before)` (algebraic lerp). Mathematically equivalent, numerically different in bf16. Our reference "019b" commit `e93d77d` (which we'd been calling 019b's code) is actually the pre-OOM-fix variant; the real 019b-submission at `9517a3b` uses the algebraic form. Root cause: commit-hash confusion across branches.

This spec is 021d with both fixes applied.

## Hypothesis

With:
- Parameter container (021d's key change — doesn't break at 4H)
- bf16 dtype (021-bf16's key change — halved 8H per-step gap vs 019b)
- Algebraic blend form (019b-submission's actual form)
- TTT α fix (019's fix applied to forward_ttt in the 021 lineage)

Expected outcome at 8×H:
- Pre-quant post-EMA: tracks 019b (~1.06951) or slightly better (Parameter may marginally const-fold).
- Post-quant: ~1.0788 (similar quant delta to 019b).
- Post-TTT with fix: ~**1.0664** (applying 019b's TTT delta of −0.01249).
- Step count: ~4880 (Parameter's throughput advantage over literal).

**Projection: post-TTT ~1.0655-1.0665.** Beats #1736 (1.06610) by 0.0005-0.0015, likely beats 019b (1.06628).

## Baseline

- **Primary:** 019b original (commit `9517a3b`) on 8×H JP: post-TTT 1.06628, pre-quant 1.06951.
- **Prior 021 variants at 8×H:** 021-buggy 1.06900, 021d projected ~1.066-1.068 (TTT pending).
- **Target to beat:** #1736 post-TTT 1.06610.

## Expected Δ

Credence distribution:

| Bucket | Probability | Predicted post-TTT |
|---|---|---|
| Clean beat | 35% | 1.0650-1.0660 |
| Tight beat | 40% | 1.0660-1.0665 (ties 019b) |
| Marginal miss | 15% | 1.0665-1.068 |
| Arc closes | 10% | > 1.068 |

Credence shift relative to 021d: more optimistic because we're fixing two independent issues simultaneously, both of which have clear mechanistic stories for how they hurt.

## Accept criteria

**Primary (post-TTT):**

| Post-TTT bpb | Bucket | Next action |
|---|---|---|
| ≤ 1.06400 | Clear beat #1736 by ≥ 0.002 | **3-seed 43/44 on same pod/commit → submission candidate** |
| (1.06400, 1.06610] | Beats #1736 | 3-seed for confirmation |
| (1.06610, 1.06710] | Borderline | Compare to 019b-rerun on same pod; may skip 3-seed and pivot |
| > 1.06710 | Arc closed | Shelve buffer-α arc; final pivot to Trunk C or accept 019b 3-seed |

## Early-signal checkpoints (informational ONLY)

Policy: **let the full pipeline complete regardless of mid-training signal.**

| Step 3000 train_loss | Bucket |
|---|---|
| ≤ 2.562 | Clean beat (matches 019b-original) |
| 2.562-2.568 | Tight beat |
| > 2.568 | Likely miss |

| Step 4000 val_bpb | Meaning |
|---|---|
| ≤ 1.108 | On track for clean beat |
| 1.108-1.112 | Marginal |
| > 1.112 | Off-track |

## Config diff

Identical to spec 021d — same env block, same `RECUR_ALPHA_ENABLED=1`, `MATRIX_LR=0.026`, default `ENABLE_LOOPING_AT` (0.35), 596s wallclock.

Only change is the code commit.

## Code changes

**Branch:** `exp/recur-alpha-buffer`
**Commit:** **`d761a22`** (already pushed)

Stack on top of spec 021 base (`cb5cd78`):
1. `dc0b5f8`: α pass-3 L4 fix (0.97265625)
2. `d070df3`: bf16 dtype + drop .to() cast
3. `8b2d791`: register_buffer → nn.Parameter(requires_grad=False)
4. `931bd7c`: **TTT α bug fix** (apply recur_alpha blend in forward_ttt)
5. `d761a22`: **Algebraic blend form** (x = x_before + α·(x_new − x_before) at all 4 sites)

## Hardware ladder

**8×H100 AP-JP-1 required.** Do not substitute.

**Preceding validation:** 4×H NE-1 mini per spec 021c Arm A' + Arm E (commit `9517a3b` vs `d761a22`). If Arm E regresses vs Arm A' at 4×H by >+0.003 on step-3000 train_loss, reconsider before launching this.

## Seed plan

Seed 42 first. **3-seed (42/43/44) conditional on post-TTT ≤ 1.06610** (beats #1736). Same pod for all three seeds.

## Inputs

- Data: CaseOps dataset, JP `/runpod/data/...`
- Tokenizer: `fineweb_8192_bpe.model` bundled
- Hotstart: none

## Run protocol

```bash
# Preflight — brotli is non-negotiable
pip install --break-system-packages brotli
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout d761a22

# Sanity-verify all five changes landed
grep "nn.Parameter" train_gpt.py | grep recur_alpha      # must match (Parameter)
grep "dtype=torch.bfloat16" train_gpt.py | head -1       # must match (bf16)
grep "0.97265625" train_gpt.py                            # must match (α fix)
grep -c "x_before + alpha \* (x_new - x_before)" train_gpt.py  # must be 4 (algebraic form, all sites)
grep -A1 "def forward_ttt" train_gpt.py | head -5        # forward_ttt has α application (TTT fix)

mkdir -p /runpod/runs/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh/seed_42
mkdir -p /tmp/torch_inductor_cache_021e_8h_jp

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh/seed_42 \
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
  > /runpod/runs/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh/seed_42/train.log 2>&1

kill $NVSMI_PID
```

## Checkpoints / artifacts

- `final_model.pt` — post-EMA FP state dict
- `final_model.int6.ptz` — quantized submission artifact
- `train.log` — every-100-step tok/s, val_bpb, train_loss, TTT trace
- `diag_nvsmi.csv` — per-GPU telemetry
- `final.json` — `val_bpb_pre_gptq_post_ema`, `val_bpb_post_gptq`, **`val_bpb_post_ttt`**, `stopping_early_at_step`, `layer_loop_enabled_at_step`, `recur_alpha_is_parameter: true`, `recur_alpha_requires_grad: false`, `recur_alpha_dtype: "bfloat16"`, `blend_form: "algebraic"`, `ttt_alpha_applied: true`

## Stop-early criteria (hard safety)

- NaN/inf in train_loss → halt
- Step time > 2× 019b-original's → halt
- `layer_loop_enabled_at_step` outside [2000, 2300] → halt

## Cost estimate

| item | cost |
|---|---|
| 8×H JP × ~25 min (compile + 596s train + GPTQ + TTT) | ~$10 |
| Rsync + pod stop | ~$0.20 |
| **Single-seed total** | **~$10** |
| Conditional 3-seed × 2 on same pod | ~$20 |
| **If 3-seed promotes** | **~$30 total** |

## Dependencies and prerequisites

1. **4×H NE-1 mini (spec 021c Arm A' + Arm E) completes cleanly.** Arm E at `d761a22` must not regress vs Arm A' at `9517a3b` by more than +0.003 on train_loss at step 3000. If Arm E is clearly worse at 4×H, reconsider before paying $10 for 8×H.
2. **JP 8×H stock availability.** Last probe showed "Low." Provision with ceiling $24/hr.
3. **Brotli in preflight** — confirm `python -c "import brotli"` before launch.

## Open questions for executor interview

1. **If 4×H mini shows Arm E ≈ Arm A' (both algebraic form match):** confident launch.
2. **If 4×H mini shows Arm E slightly worse:** interview user — the container penalty may be real but small enough to still win post-TTT (because TTT fix compensates). Decision: user-dependent.
3. **If 4×H mini shows Arm E significantly worse (>+0.005):** halt promotion; pivot.
4. **Running order vs 021d:** 021d's TTT is in flight at time of writing. If 021d post-TTT lands > 1.068, 021e supersedes immediately. If 021d lands ≤ 1.066, 021e is still valuable as a confirmatory-plus-improvement variant.

## What this does NOT test

- Per-seed α basin adaptation (that's a warmstart+freeze variant, not in scope).
- Other tokenizer/TTT stackings (022, 023 still pending).
- Trunk C-style alternative TTT paradigm (separate arc, not related).

## Reflection

Two real bugs in 021's code path were discovered today: TTT α missing, wrong blend form. Both were inherited from branching errors (021 forked from 017 instead of 019; "019b commit" label pointed to pre-OOM-fix variant). 021e is the cumulative fix.

If 021e wins post-TTT, the submission-path story is: **"019b's recipe with Parameter container and bf16 dtype for throughput."** The buffer-α arc's final form.

If 021e loses, the mechanism story we've been chasing for the last session was dominated by these two bugs, not by container/dtype choices. In that case, the real lesson is to verify commit hashes before any cross-run comparison and to audit TTT coverage in every new code path.
