# Evaluation — Spec 019b (Recur-Alpha manual+constant, full pipeline)

**Run dir:** `runs/019b-recur-alpha-manual-constant-full/seed_42/`
**Commit:** `9517a3b` on `exp/recur-alpha-manual-constant-full`
**Pod:** `6pyy9q7aatvgpb` — 8×H100 SXM, AP-JP-1, JP volume `jlxvxeiol4`
**Eval date:** 2026-04-21

## Hypothesis recap

019 used `torch.lerp` with literal-constant α, which unexpectedly regressed throughput −2.7% at full scale (despite 018c proxy showing 92% overhead recovery). Hypothesis: `torch.lerp`'s primitive template disrupts Inductor fusion heuristics at full scale. Fix: replace with algebraic lerp form `x_before + α * (x_new - x_before)` — three elementary ops that fuse as pointwise arithmetic at any scale.

## Result

| metric | #1736 (target) | 008 | 017 | 019 | **019b** |
|--------|---------------|-----|-----|-----|---------|
| final step | 4,854 | 4,828 | 4,784 | 4,697 | **4,716** |
| pre-quant post-EMA val_bpb | 1.06907 | 1.06922 | 1.07083 | 1.07063 | **1.06951** |
| post-GPTQ val_bpb | 1.07847 | 1.08010 | — | 1.07989 | **1.07877** |
| **post-TTT val_bpb** | **1.06610** | **1.06728** | **1.06733** | **1.06744** | **1.06628** |
| submission size (bytes) | 15,978,834 | 15,946,577 | — | 15,980,998 | **15,981,844** |

**019b post-TTT 1.06628 — misses #1736 by 0.00018, within single-seed std.**

## Noise/signal judgment

**Bucket: (1.06550, 1.06710] — close, within seed std.**

019b improves over 019 by 0.00116 post-TTT. However, the notes confirm this is attributable to **19 extra training steps** (pod lottery at the wallclock boundary), not the lerp → algebraic code change. Step-4000 val_bpb is identical between 019 and 019b (both 1.1071), confirming the algebraic form is a true no-op at the loss level.

The 0.00018 miss vs #1736 is within the SOTA single-seed std floor (~0.0002). It could be noise in either direction.

## Matched-step quality

| step | 008 val_bpb | 017 val_bpb | 019 val_bpb | 019b val_bpb |
|------|-------------|-------------|-------------|--------------|
| 4000 | 1.1110 | 1.1088 | 1.1071 | **1.1071** |

019b matches 019 exactly at step 4000 — best per-step quality across all recur-alpha runs, and better than 008 by 0.0039. The per-step signal is real and consistent.

## Throughput — still regressed, algebraic form did not help

Same JP drift pattern as 019:

| step | tok/s |
|------|-------|
| 600 | 8,115,451 |
| 1300 | 8,082,236 |
| 1900 | 8,082,974 |
| 2400 | 7,695,149 |
| 3200 | 6,784,807 |
| 4000 | 6,431,417 |
| 4500 | 6,262,589 |
| 4700 | 6,225,615 |

Starts at ~8.08M tok/s (matching 008/017's best), drifts to ~6.2M by minute 8. This is the same JP pod cooling/contention pattern seen in 019. Final step 4716 vs 019's 4697 — essentially identical. **The algebraic lerp form did not recover throughput at full scale.** The root cause of the throughput regression remains undiagnosed.

## Path to clean run (two failed attempts)

1. **OOM attempt (commit `e93d77d`, pod `cuh1g7lxkdhi5z`):** Initial manual form `α * x_new + (1-α) * x_before` produced CUDA OOM in `_run_cu_bucket_warmup` on all 8 ranks. Root cause: 4 FX nodes per blend site × 6 sites × ~512 MiB = 3–6 GiB extra activation memory from AoT partitioner's min-cut saving two intermediate tensors. Fixed by switching to algebraic form (3 ops, 1 intermediate, partitioner saves nothing).
2. **Tokenizer path bug (first attempt on successful pod):** `DATA_DIR=/workspace/parameter-golf/data` wrong for CaseOps on this volume. Correct: `/workspace/data/`. Log saved as `train.tokenizer_path_bug.log`.
3. **Clean run (commit `9517a3b`, pod `6pyy9q7aatvgpb`):** completed.

## What 019b answered

| Question | Answer |
|---|---|
| Does algebraic lerp fix full-scale throughput vs torch.lerp? | **No** — same drift pattern, same final step range |
| Does the code change affect loss quality? | **No** — step-4000 val_bpb identical to 019 |
| Does constant-α improve over tensor-α (017)? | **Marginally** — 019b post-TTT 1.06628 vs 017's 1.06733, but step-count confounded |
| Does TTT fix from 019 help over 017? | **Noise level** — 1.06628 vs 1.06733 is 0.00105, too close to call with step variance |

## Decision — 3-seed or pivot?

**Bucket outcome is (1.06550, 1.06710]: per spec, 3-seed is the prescribed next action (~$20–24).**

However, caution is warranted before committing:

1. **019b's gain over 019 is pod lottery, not code.** A 3-seed study would establish the mean/std of the current recipe but wouldn't tell us if the throughput problem is solvable.
2. **Per-step quality (1.1071 @ step 4000) is the best we've achieved.** If we could run to step 4828, Method A extrapolation puts post-TTT at ~1.063–1.065, a clear beat of #1736.
3. **The throughput problem is the bottleneck.** Until we understand why constant-α + full scale regresses, we're leaving ~130 steps (and ~0.007 post-TTT bpb) on the table every run.

**Recommended path:**

- **Option A (de-risk current recipe):** 3-seed (seeds 43+44) at 019b's recipe. Establishes whether 1.06628 is representative or a lucky single-seed outcome. Cost ~$20–24. If mean ≤ 1.06610: submit. If mean > 1.06610: constant-α is exhausted; shelve and pivot.
- **Option B (diagnose throughput first):** Small diagnostic run (TORCH_LOGS=output_code or proxy A/B at 6L/256d) to understand why full-scale drifts. If fixable, a clean run could reach step 4828+ and post-TTT ~1.063.
- **Option C (stack now):** Accept the throughput tax and combine recur-alpha with the next highest-EV lever. Per-step quality stacks; if the next lever also adds per-step gains, the combined run post-TTT could be decisive even with the step deficit.

**Recommendation: Option A first (cheapest, fastest resolution of the immediate question), then Option C if it doesn't promote.**

## Cost

| item | cost |
|---|---|
| `6vc98xb3qwpzri` 2×H100 smoke (killed by user) | ~$1.00 |
| `cuh1g7lxkdhi5z` 8×H100 JP OOM attempt (~10 min) | ~$4.00 |
| `imfw8twiq2ipsw` 8×H100 NA create+delete (user preferred JP) | ~$0.50 |
| `6pyy9q7aatvgpb` 8×H100 JP successful run (~36 min) | ~$14.50 |
| **Spec 019b total** | **~$20** |

## Artifacts

- `runs/019b-recur-alpha-manual-constant-full/seed_42/train.log` — full pipeline log
- `runs/019b-recur-alpha-manual-constant-full/seed_42/final_model.int6.ptz` — 15.95 MB, under cap ✓
- `final_model.pt` on JP volume `jlxvxeiol4` at `/workspace/runs/019b-recur-alpha-manual-constant-full/seed_42/` (135 MB, not git-synced)

## Cross-references

- Spec: `research/specs/019b-recur-alpha-manual-constant-full.md`
- Prior full pipeline: `research/evaluations/019-recur-alpha-constant-full.md`
- Throughput diagnostic: `research/evaluations/016b-recur-alpha-throughput.md`
- Baseline: `runs/008-1736-reproduction/seed_42/`
