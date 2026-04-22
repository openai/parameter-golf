# Spec 020 — Alpha-throughput full-scale replication (Phase 0)

**Slug:** `alpha-throughput-replication`
**Created:** 2026-04-21
**Status:** READY
**Links to:** `research/ideas/throughput-alpha-proxy-gap.md` (Phase 0)

## Hypothesis

The ~130K tok/s gap between 019 (constant α + lerp) and 017 inferred across separate pods is real architectural overhead, not node variance. On a **single 8×H100 pod** running both configs back-to-back with identical compile flags, we will observe Run B (α on) at ~3-4% lower tok/s than Run A (α off), matching the gap inferred from the 019 vs 008 pod-to-pod comparison.

If no gap appears on same-pod A/B, the 019 regression premise itself is wrong and the idea file's Phase 2+ plan needs revisiting.

## Baseline

Reference numbers (from prior runs, different pods):
- 008 (no α), JP 8×H100: ~3.3M tok/s regime at full
- 019 (constant α + lerp), JP 8×H100: ~130K below 017 throughout

Expected same-pod ratio: B/A ≈ 0.96–0.97.

## Accept criteria

Three tok/s numbers (steps 100-200 mean), one per run, from the same pod, in order **A₁ (α off) → B (α on) → A₂ (α off)**. No eval, no GPTQ, no TTT.

**Warm-effect sanity check:** A₁ vs A₂ must agree within ±1.0% tok/s. If they don't, warm-state drift (thermal, page cache, something else) is non-negligible and the B/A ratio needs correction. If |A₂ − A₁| / A₁ > 1%, report this and use A̅ = (A₁ + A₂) / 2 as the corrected baseline, with an explicit noise note.

**Decision criterion (B / A̅):**
| B / A̅ ratio | Interpretation | Next step |
|---|---|---|
| ≥ 0.99 | No replicable gap | Revisit idea file premise; investigate what made 019 slow (cache? env?) |
| 0.96 – 0.99 | Gap replicated, smaller than expected | Proceed to Phase 2 proxy search with this as the target ratio to reproduce |
| 0.93 – 0.96 | Gap replicated, matches expected | Proceed to Phase 2 |
| < 0.93 | Gap larger than expected | Still proceed to Phase 2; note that our prior estimate was conservative |

## Config diff

Same source tree (branch below), two runs with different env:

**Run A (α off, looping on):**
```
RECUR_ALPHA_ENABLED=0
ENABLE_LOOPING_AT=0
MAX_STEPS=200
THROUGHPUT_DIAG=1
```

**Run B (α on, looping on):**
```
RECUR_ALPHA_ENABLED=1
ENABLE_LOOPING_AT=0
MAX_STEPS=200
THROUGHPUT_DIAG=1
```

All other flags (CaseOps, GatedAttn, QuantGate, TTT, etc.) remain at spec 019's values — but **eval/GPTQ/TTT code paths never execute** because `MAX_STEPS=200` short-circuits before them.

`ENABLE_LOOPING_AT=0` ensures looping + α are live from step 1 (analogous to 016b). Without this, steps 1-200 would run pre-loop and the α blend ops would never fire.

## Code changes

**Branch:** `exp/alpha-throughput-diag` forking from `3c3a134` (spec 019).
**Commit:** TBD — to be created during execution session.

Changes (additive, behind env flags):

1. `MAX_STEPS` env var. In the training loop, after the loss/optimizer step, `if MAX_STEPS is set and step >= MAX_STEPS: break`. Placed so the break skips the eval/GPTQ/TTT pipeline entirely. ~5 LOC.
2. Per-step tok/s logging. If not already per-step, log tok/s each step to train.log so we can compute 100-200 mean precisely. ~3 LOC.
3. **CUDA-event timing on α blend sites** (behind `THROUGHPUT_DIAG=1`). For each blend-op call site, wrap with `torch.cuda.Event(enable_timing=True)` start/end pair; log μs per site per step. Gives direct per-site walltime cost independent of global tok/s. ~20 LOC.

Keep all diag code behind the `THROUGHPUT_DIAG` flag so it doesn't affect normal runs.

## Hardware ladder

- Skip mini (proxy already done in 016b / 018c — those are the runs that *failed* to predict; a new mini-test tells us nothing here).
- **8×H100 JP** (matches 019's region, per user answer). Single pod, two sequential runs.

## Seed plan

Single seed (42) per run. Throughput test — seed doesn't meaningfully matter.

## Inputs

- Data: CaseOps dataset on JP `/runpod/data/...`
- Tokenizer: `fineweb_8192_bpe.model`, bundled
- Hotstart: none (fresh from step 0; we're only training 200 steps)

## Run protocol

Both runs on the **same pod**. Clear Inductor cache between runs so compile state doesn't carry over.

```bash
cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git checkout <exp/alpha-throughput-diag commit>

# --- Run A1 (alpha OFF) ---
mkdir -p /runpod/runs/020-alpha-throughput-repl/run_a1_alpha_off
mkdir -p /runpod/.torch_inductor_cache_a1
rm -rf /runpod/.torch_inductor_cache_a1/*

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/020-alpha-throughput-repl/run_a1_alpha_off \
TORCHINDUCTOR_CACHE_DIR=/runpod/.torch_inductor_cache_a1 \
CASEOPS_ENABLED=1 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=0 \
ENABLE_LOOPING_AT=0 \
MAX_STEPS=200 THROUGHPUT_DIAG=1 \
TRAIN_LOG_EVERY=1 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/020-alpha-throughput-repl/run_a1_alpha_off/train.log 2>&1

# --- Run B (alpha ON) ---
# Same as above with:
#   ARTIFACT_DIR=.../run_b_alpha_on
#   TORCHINDUCTOR_CACHE_DIR=.../.torch_inductor_cache_b  (fresh)
#   RECUR_ALPHA_ENABLED=1

# --- Run A2 (alpha OFF, repeat) ---
# Same as Run A1 with:
#   ARTIFACT_DIR=.../run_a2_alpha_off
#   TORCHINDUCTOR_CACHE_DIR=.../.torch_inductor_cache_a2  (fresh)
```

Order: **A₁ → B → A₂**. A₂ catches warm-state drift (thermal, page cache) and lets us compute a corrected baseline A̅ = (A₁ + A₂) / 2 if the two A runs differ by >1%.

## Checkpoints / artifacts to emit

Per run:
- `train.log` — per-step tok/s, α-blend CUDA-event μs
- `final.json` — summary with: `tok_per_sec_mean_100_200`, `tok_per_sec_per_step` (array), `alpha_blend_site_us_mean` (per-site μs from CUDA events), `config` (env snapshot), `commit`

No model checkpoints. Artifact size is negligible.

## Stop-early criteria

- NaN / inf in loss → halt (unlikely in 200 steps but guard anyway)
- First-step compile > 20 min → halt (something wrong with cache setup)

## Cost estimate

| item | cost |
|---|---|
| 8×H100 JP × ~22 min (3× compile + 3× 200 steps, A₁ / B / A₂) | ~$7 |
| Buffer / diag overhead | ~$1 |
| **020 total** | **~$7–9** |

## Extra artifacts

- Diff summary: B/A ratio, per-site blend μs, sanity check that alpha-blend μs × sites × steps ≈ the walltime gap.

## Open questions for interview (execution)

1. **Ensure instrumentation doesn't itself introduce overhead**: CUDA events on blend sites should be cheap, but if they show in Run B's tok/s as an artifact, we'd mis-attribute. Suggest: timing events only on every 10th step, or run a third "diag-off" version of Run B as a sanity check if Run B's tok/s looks suspiciously worse than historical 019.
2. **Compile-cache contamination**: verify the two runs really do have cold caches (no `.torch_inductor_cache` in unexpected locations, no `/tmp/torchinductor_*` carryover between runs).
3. **What if Run A's tok/s doesn't match historical 008?** If Run A itself comes in way off the ~3.3M regime, something about this pod or this instrumentation is off and the A/B ratio is untrustworthy. Halt and investigate before doing Run B.
