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

## Hardware

Single **2×H100 NA** pod for the whole diagnostic. Smaller hardware is fine because:
- We're measuring the *ratio* 016/008 on identical pod, not absolute numbers
- Blend-op overhead is per-GPU memory-bandwidth, scales similarly at 2-GPU and 8-GPU
- Much cheaper (~$3/hr vs ~$24/hr)

## Execution protocol

Single pod, three sequential runs: one warmup (discard) + two measurement. Use NA-1 region explicitly (avoid JP). Use `TORCHINDUCTOR_CACHE_DIR` to cache compile artifacts.

### Run 0 — Warmup (throwaway, 008 commit, ~50 steps)

Primes pod-level state (NCCL fabric, CUDA drivers, memory allocator patterns) so that Run A and Run B both start from an equally-warm pod. Output is discarded.

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git checkout 154c9b8

mkdir -p /workspace/runs/016b-throughput/run-0-warmup
mkdir -p /workspace/.torch_inductor_cache

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/016b-throughput/run-0-warmup \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache \
CASEOPS_ENABLED=1 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
ENABLE_LOOPING_AT=0 \
ITERATIONS=50 \
TRAIN_LOG_EVERY=25 \
SEED=42 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  > /workspace/runs/016b-throughput/run-0-warmup/train.log 2>&1
```

Discard this run's tok/s numbers. Its purpose is exclusively to warm the pod.

### Run A — 008 baseline (no recur-alpha, 150 steps, measurement)

```bash
# Still on 154c9b8 — no need to re-checkout
mkdir -p /workspace/runs/016b-throughput/run-a-008

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/016b-throughput/run-a-008 \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache \
CASEOPS_ENABLED=1 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
ENABLE_LOOPING_AT=0 \
ITERATIONS=150 \
TRAIN_LOG_EVERY=25 \
SEED=42 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  > /workspace/runs/016b-throughput/run-a-008/train.log 2>&1
```

Second time on this commit on this pod — torch.compile cache hits, compile drops to ~1-2 min. Read tok/s at steps 50, 100, 150. Training loss will be catastrophic (like #1739) — **do not panic, bpb is not the measurement here.**

### Run B — 016 with recur-alpha (150 steps, measurement)

```bash
git checkout 4dd2d63

mkdir -p /workspace/runs/016b-throughput/run-b-016

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/016b-throughput/run-b-016 \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache \
CASEOPS_ENABLED=1 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
ENABLE_LOOPING_AT=0 \
ITERATIONS=150 \
TRAIN_LOG_EVERY=25 \
SEED=42 \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  > /workspace/runs/016b-throughput/run-b-016/train.log 2>&1
```

Different commit than Run A → different graph hash → fresh ~5 min compile. Read tok/s at matched steps 50, 100, 150.

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
