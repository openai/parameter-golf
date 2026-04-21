# Spec 017 — Recur-Alpha matched-throughput full-pipeline run

**Slug:** `recur-alpha-full`
**Created:** 2026-04-21
**Links to:** spec 016, `research/ideas/beating-1736-note.md`, spec 016b (throughput diagnostic)

## Hypothesis

On a matched-throughput 8×H100 NA pod running the full training → GPTQ → TTT pipeline end-to-end (no EVAL_ONLY_CHECKPOINT shortcut), spec 016's commit (`4dd2d63`, α=1 init + grad_norm logging fix) produces a post-TTT val_bpb **≤ 1.06550**, beating #1736's claimed 1.06610 by ≥0.0005.

Works backward from the target via the chain:
- Target post-TTT: 1.06610 (beat = ≤ 1.06550)
- Assuming #1736-typical TTT recovery (−0.01237), target post-GPTQ: ≤ 1.07847
- Assuming observed GPTQ cost (+0.00947), target pre-quant post-EMA: ≤ 1.06900
- 016's JP pre-quant post-EMA at step 4708 was 1.07083; at matched 008-step (4828), projected ~1.06471 — well under the target.

## Baseline

Primary reference: **#1736's claimed post-TTT val_bpb = 1.06610** (from `records/track_10min_16mb/.../submission.log` or equivalent).
Our 008 post-TTT number: **not measured** (008's execution stopped before TTT eval ran). For training-endpoint comparison, 008 pre-quant post-EMA = 1.06922.

## Expected Δ (conditional on spec 016b result)

| 016b outcome | NA throughput vs JP 008 | Expected 017 margin over #1736 |
|---|---|---|
| ≥99% of 008 baseline (no tax) | likely matched-or-better | −0.0005 to −0.003 bpb better than #1736 (clear beat) |
| 97-99% (partial tax) | mild NA variance | ±0.001 of #1736 (coinflip, needs 3-seed) |
| <97% (architectural tax) | 016 genuinely slower | +0.001 to +0.002 worse than #1736 (miss at matched-wallclock; shelve) |

**Note:** 017 runs **regardless** of 016b outcome. The diagnostic changes our expectation, not the plan. Even in the worst case (architectural tax), 017 provides:
- Real post-TTT number on recur-alpha (currently unmeasured — 016 post-hoc OOM'd)
- Validates TTT × recur-alpha composition (the other big unknown)
- Sets up a proper spec 017 evaluation record

## Accept criteria

- Training completes without NaN/divergence
- `final_model.pt`, `final_model.int6.ptz` both emit
- Post-GPTQ diagnostic val_bpb captured
- **Phased-TTT val_bpb captured** (the submission-gate number; this is the bit 016's post-hoc missed)
- α grad_norm ≠ 0 post-activation (from the 4dd2d63 logging fix)
- `recur_alpha_final` populated in final.json (top-level, per spec 016 convention)

**Decision criterion (post-TTT val_bpb):**
| Post-TTT | Bucket | Next action |
|---|---|---|
| ≤ 1.06550 | Clear promote | 3-seed confirmation (~$30), then submission |
| (1.06550, 1.06710] | Close, within seed std | 3-seed to resolve, budget $30 |
| (1.06710, 1.06910] | Inside the accept gate but worse than #1736 | Shelve submission path; keep mechanistic findings. Likely reflects TTT partial absorption. |
| > 1.06910 | Outside gate | Hard shelve; investigate what went wrong in pipeline |

## Config diff vs spec 016

No config change. Spec 016 screened with `PHASED_TTT_ENABLED=1` but we killed training before TTT eval ran (screening mode). Spec 017 **lets the full pipeline run to completion** — everything after `stopping_early: wallclock_cap` including post-EMA eval, GPTQ, TTT.

Implicit: `pyminify` must be installed on the pod for the submission-size measurement at end. This is in the post-`304c552` preflight, but verify on pod.

## Code changes

None. Same commit as spec 016: **`4dd2d63` on `fork/exp/recur-alpha-ones`**.

## Hardware ladder

- **Skip smoke** — cite spec 016's JP smoke + the upcoming spec 016b NA 2×H100 throughput diagnostic as sufficient recent validation.
- **8×H100 NA-1** (primary). Seed 42.
  - If 016b confirms no throughput tax → high confidence, single-seed is enough for now
  - If 016b shows tax → single-seed is diagnostic; decide 3-seed based on result
- **(Conditional) 3-seed NA** (seeds 43, 44): only if spec 017 seed 42 lands in "clear promote" or "close" bucket. ~$30.

## Seed plan

Single seed (42) first. 3-seed (42/43/44) only if seed 42 promotes or lands in the ambiguous range.

## Inputs

- Data: CaseOps dataset on NA-1 volume. **OPEN QUESTION**: NA volume's CaseOps data was being prepped as of spec 015 (via prep pod `q52slv996d3uqx`); verify completion before launching. If NA data isn't ready, fall back to JP but note the throughput-vs-017 comparison gets noisier.
- Tokenizer: `fineweb_8192_bpe.model`, bundled with submission dir.
- Hotstart: **none — fresh from-scratch training.** We considered hotstarting from 016's saved `final_model.pt` on JP volume, but fresh-train on NA gives cleaner apples-to-apples comparison against 008's reference pipeline. Hotstart is a separate (cheaper) experiment if we want it later.

## Execution protocol

Standard #1736 full training + eval flow. Explicitly **do not use EVAL_ONLY_CHECKPOINT bypass** — that shortcut caused the OOM in 016's post-hoc. The normal end-of-train flow runs GPTQ and TTT with properly-primed CUDA allocator.

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git checkout 4dd2d63

mkdir -p /workspace/runs/017-recur-alpha-full/seed_42
mkdir -p /workspace/.torch_inductor_cache

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/017-recur-alpha-full/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache \
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
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/017-recur-alpha-full/seed_42/train.log 2>&1
```

## Checkpoints / artifacts to emit

- `final_model.pt` (~135 MB, post-EMA pre-GPTQ) — inherited from commit `304c552` baseline fix
- `final_model.int6.ptz` (~16 MB, quantized+compressed submission artifact)
- `train.log` (full training + post-training eval sequence)
- `final.json` — must include:
  - `val_bpb` (endpoint bare)
  - `val_bpb_pre_gptq_post_ema`
  - `val_bpb_post_gptq`
  - **`val_bpb_post_ttt`** (the gate number — this is the one that was missing in 016)
  - `recur_alpha_final` (top-level array, per spec 016 convention)
  - `stopping_early_at_step`, `layer_loop_enabled_at_step`
  - Tok/s snapshots at steps 100/1000/2000/3000/4000/4500 for post-hoc throughput comparison
- `notes.md` (execution narrative)

## Stop-early criteria

Unconditional:
- NaN/inf in train_loss → halt
- Step time > 2× spec 008 at matched step → halt

Conditional on `looping_active=True` (step ≥ ~1700):
- α grad_norm exactly 0 for 5+ consecutive log entries **after activation** → halt (plumbing broken; should never fire now that `4dd2d63` fixes the logging bug)
- Training loss > 008's matched-step loss + 0.03 for 5+ consecutive log entries → halt (convergence off)

**Do NOT halt based on:**
- α grad_norm = 0 pre-activation (expected; α out of circuit)
- tok/s being a few percent below 008 (expected per 016b finding regardless of direction — that's data, not a halt condition)

## Cost estimate

| item | cost |
|---|---|
| 8×H100 NA × ~25 min (compile + 10 min training + ~10 min post-training pipeline) | ~$10 |
| Rsync + pod stop | ~$0.10 |
| **Single-seed total** | **~$10-12** |
| (Conditional) 3-seed × additional 2 runs | ~$20-24 |
| (Conditional) Full 3-seed total | **~$30-36** |

Budget check: we have ~$148 remaining from the $200 hard budget. Single-seed 017 fits comfortably; 3-seed leaves us ~$110 for follow-ups if the result promotes.

## Open questions for interview

1. **Is CaseOps data synced to NA volume?** Critical preflight check. If not, either fall back to JP (losing throughput clarity) or run the prep script on NA first (adds ~30 min, ~$0.25). Execution should verify via `ls /workspace/data/datasets/fineweb10B_sp8192/` on NA before launching.
2. **Order of operations vs 016b?** Preferred sequence: 016b first (~15 min, ~$1) → decision → 017. If 016b is running while spec 017 is drafted, that's fine.
3. **Does #1736's TTT config transfer cleanly to recur-alpha?** Assumption is yes (TTT LoRAs are on Q/K/V/MLP/O, not on recur_alpha scalars). But we haven't verified. Risk: if TTT LoRA assembly expects specific parameter shapes, adding recur_alpha to the state_dict could trip assertion. Spec 017 is the test.
4. **3-seed timing:** run 3-seed back-to-back on one NA pod, or parallel on three? Parallel is faster (~25 min), sequential is cheaper (same wallclock cost but single-pod boot overhead amortized). Default to parallel if NA capacity allows; sequential otherwise.

## What 017 does NOT do

- Does not change code (same commit as 016)
- Does not test α=0 init (that's 015; shelved for submission but keep for diagnostic)
- Does not test cross-pass XSA or other follow-up ideas
- Does not run on JP (specifically avoids the JP variance that confounded 015/016)
- Does not skip TTT (this is the point — we need the TTT composition number)
- Does not run 3-seed unconditionally (that's an upgrade, not baked in)

## Conditional decision tree (post-016b)

**Branch 1 — 016b says ≥99% (no tax):**
- Run 017 seed 42 with high confidence of beat-1736
- If seed 42 lands in "clear promote" bucket → 3-seed NA
- Track ~$10 + $20-24 = ~$30-34 total

**Branch 2 — 016b says 97-99% (ambiguous):**
- Run 017 seed 42 knowing margin may be razor-thin
- Single-seed result is more likely to land in the "close" bucket needing 3-seed to resolve
- Track ~$10 + (probably $20-24) = ~$30-34 total

**Branch 3 — 016b says <97% (architectural tax):**
- Run 017 seed 42 as diagnostic, not submission bet
- Primary value: measuring TTT × recur-alpha composition (does TTT absorb the matched-step gain?)
- If post-TTT is surprisingly good despite throughput tax → investigate further
- If post-TTT shows tax carrying through to submission → hard shelve recur-alpha for submission, write findings up as mechanistic-only result
- Track ~$10, no 3-seed
