# Spec 016b — Recur-Alpha throughput diagnostic

**Slug:** `recur-alpha-throughput`
**Created:** 2026-04-21
**Links to:** `research/ideas/beating-1736-note.md` (context on why the throughput tax matters)

## Hypothesis

Specs 015 and 016 both ran 2-4% below 008 baseline tok/s on the same JP pool. Cause is unresolved: pod variance OR a real throughput tax from the blend op `x = α * x_new + (1 − α) * x_before` in the recur-alpha forward path. Testing this directly via `ENABLE_LOOPING_AT=0` (forces looping-and-α active from step 1) on identical hardware answers the question cheaply.

## Baseline

Spec 008's late-training tok/s on JP pool: ~6.45M @ step 4500. Spec 015 / 016 are 2.4% / 3.6% below that respectively.

## Decision criterion

- **016 tok/s ≥ 99% of 008 tok/s** → no tax, pod variance only. Greenlight spec 017 full-pipeline run with high confidence of beating #1736 given matched throughput.
- **016 tok/s ∈ [97%, 99%]** → ambiguous. Partial tax. Proceed to spec 017 with reduced expectations; margin over #1736 may be near-zero.
- **016 tok/s < 97%** → real architectural tax. Recur-alpha is a net regression at matched wallclock. Shelve submission path; optimize blend op or pivot.

## Code changes

None. Uses existing commits:
- **008 side:** `154c9b8` (spec 008 pinned commit on `research`) — no recur-alpha code.
- **016 side:** `4dd2d63` (`exp/recur-alpha-ones`) — recur-alpha enabled via `RECUR_ALPHA_ENABLED=1`.

## Model config (proxy — NOT full submission model)

The full 11L/512d model does not fit on 2×H100 under DDP (each GPU holds full model + optimizer state, ~78 GB used). We run a **6L/256d proxy model** via env vars only — no code change:

```
NUM_LAYERS=6    MODEL_DIM=256    XSA_LAST_N=6    PARALLEL_START_LAYER=99
```

- `XSA_LAST_N=6` — must match NUM_LAYERS (default 11 would reference non-existent layers)
- `PARALLEL_START_LAYER=99` — disables parallel residuals (default 8 > num_layers=6, set explicitly for safety)
- Loop config unchanged: `LOOP_START=3 LOOP_END=5 NUM_LOOPS=2` — looped layers 3,4,5 still exist in a 6-layer model ✓
- All other flags (CaseOps, GatedAttn, QuantGate) unchanged

**Caveat:** smaller model means cheaper layers, so the 6-scalar blend op is a larger *fraction* of compute. Any overhead measured here is an upper bound on real overhead at full model size. If we see <1% difference here, the full model is definitely safe.

## Hardware

Single **2×H100 US-NE-1** pod, NA volume `hvpdph5i3g` mounted at `/workspace`. CaseOps data already on volume at `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/`. ~$6/hr.

## Execution protocol

Two sequential runs on the same pod (no warmup — small model compiles fast). Each runs to the 596s wallclock cap.

### Run A — 008 baseline (commit 154c9b8, no recur-alpha)

```bash
NCCL_NET=Socket DATA_DIR=/workspace/parameter-golf/data \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache \
ARTIFACT_DIR=/workspace/runs/016b-throughput/run-a-2gpu \
CASEOPS_ENABLED=1 GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
ENABLE_LOOPING_AT=0 TRAIN_LOG_EVERY=25 SEED=42 \
NUM_LAYERS=6 MODEL_DIM=256 XSA_LAST_N=6 PARALLEL_START_LAYER=99 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

### Run B — 016 recur-alpha (commit 4dd2d63, RECUR_ALPHA_ENABLED=1)

Same env as Run A plus `RECUR_ALPHA_ENABLED=1`. Alpha active from step 0 (`ENABLE_LOOPING_AT=0`).

### After all three runs

- Rsync all three `train.log` files to local `runs/016b-throughput/`
- **Stop the pod immediately.** Save the full pipeline run for spec 017 decision.

## Expected artifacts

```
runs/016b-throughput/
  run-0-warmup/train.log    # discarded, kept for audit
  run-a-008/train.log
  run-b-016/train.log
  notes.md                   # execution writeup: ratio, raw tok/s, decision
```

No final.json needed — this isn't a submission experiment, just a diagnostic. `notes.md` must include:
- Raw tok/s at steps 50/100/150 for Run A and Run B (ignore Run 0)
- Ratio 016/008 at each matched step
- Any anomalies (step-time spikes, compile hiccups)
- Decision: which bucket we landed in per the criteria above

## Stop-early criteria

- NaN / inf in step time (not loss — loss WILL be huge; ignore) → halt, investigate
- Compile failure on either commit → halt, investigate (shouldn't happen, both commits previously compiled fine)
- Step 150 reached on Run B → done, measurement complete

## Cost estimate

| item | cost |
|---|---|
| Pod boot + SSH (~2 min) | ~$0.10 |
| Run 0 warmup: compile ~5 min + 50 steps (~10 sec) | ~$0.30 |
| Run A 008: warm compile cache hit ~1-2 min + 150 steps (~25 sec) | ~$0.15 |
| Run B 016: fresh compile ~5 min + 150 steps (~25 sec) | ~$0.40 |
| Rsync + pod stop | ~$0.05 |
| **Total 016b diagnostic** | **~$1** |

Total wallclock: ~15-20 min.

## Open questions for interview

1. **Does 2×H100 faithfully represent 8×H100 blend overhead?** Per-GPU memory bandwidth is similar; the blend op is per-token per-layer, not NCCL-dependent. Should hold. If we want belt-and-suspenders, we can re-run on 8×H100 for same diagnostic, ~$4 extra.
2. **Does `ENABLE_LOOPING_AT=0` perfectly emulate post-activation cost?** Yes — looping_active is a binary switch; setting it True from step 1 puts the blend op in the forward graph identically to what happens post-activation in normal runs.
3. **Compile cache reuse:** since 008 and 016 have different commits / different graphs, no cache reuse between the two runs. Each pays its own 5 min. Acceptable.

## What 016b does NOT do

- Does not produce val_bpb (loss is catastrophic by design)
- Does not test TTT or GPTQ composition
- Does not verify seed-to-seed α shape reproduction
- Does not beat #1736. Only tells us if #1736 is *beatable* at matched throughput.

## Sequencing

**Before spec 017 (matched-clock NA full-pipeline run):** 016b should run first. It's the cheap diagnostic that decides whether spec 017 is worth $15 or not. If 016b says "real tax, >3%", spec 017 gets recast as a diagnostic for "does TTT compose" rather than "do we beat #1736".

Ideally both run in a single execution session: 016b diagnostic (~30 min) → decision → 017 or pivot.
