# Spec 018 — Recur-Alpha blend-op optimization via `torch.lerp`

**Slug:** `recur-alpha-lerp`
**Created:** 2026-04-21
**Links to:** spec 016b (throughput diagnostic), spec 017 (full-pipeline run), `research/ideas/beating-1736-note.md`

## Hypothesis

Spec 017 showed the blend op `x = alpha * x_new + (1.0 - alpha) * x_before` costs ~1.8 ms per step on the full model at step 4500 (~1.5% overhead). The cost breaks down roughly as:

- ~1.34 ms from unfused memory traffic (~112 MB per site × 6 sites × 2 passes)
- ~300 μs from kernel-launch overhead (4 unfused ops per site)
- ~180 μs from fusion loss in surrounding ops

`torch.lerp(x_before, x_new, alpha)` is mathematically identical (`lerp(a,b,w) = a + w(b−a) = (1−w)a + wb`) but fuses into a single CUDA primitive. Expected: reduce blend memory traffic to 1 read + 1 read + 1 write ≈ 48 MB/site, collapse 4 kernel launches to 1, allow surrounding-op fusion to reform. Projected **~50-60% reduction in total blend overhead** on the full model.

Testing at mini-model scale (6L/256d) amplifies blend fraction of total compute by ~6× (smaller matmuls relative to same-shape blend), making the throughput delta measurable on cheap 2×H100 hardware.

## Baseline

- **Spec 016b** (when it lands): provides "no recur-alpha" vs "current blend" tok/s on mini-model at 2×H100. Run B of 016b = control for this spec.
- If 016b hasn't run yet by execution time, 018 should include a control run (016 commit, same mini-model config) to establish the baseline.

## Decision criterion

Let Δ = (current blend tok/s − lerp tok/s) / current blend tok/s, measured on mini-model 2×H100 at steady-state (step 100-150).

| Δ | interpretation | next action |
|---|---|---|
| Δ ≥ 40% of blend overhead recovered | Lerp is the win | Spec 019+ applies lerp on full model; re-run 017's full pipeline with lerp, expect ~0.0010 bpb improvement at training endpoint |
| 20-40% recovered | Partial win | Apply lerp + continue to bake-into-block refactor for bigger gain |
| <20% recovered | Disappointing | Skip lerp, go direct to bake-into-block (spec 019 candidate) |

Mini-model "blend overhead" baseline: whatever 016b (Run B vs Run A) measures. Or, on the mini-model, approximately 2-5% relative tok/s delta (proxy amplifies).

## Code changes

**Branch:** `exp/recur-alpha-lerp` forking from `4dd2d63` (spec 016's commit, which already has α=1 init and the grad_norm logging fix).
**Commit:** `ede7895` on `fork/exp/recur-alpha-lerp`.

Two-commit stack on the branch:
- `97d9854`: torch.lerp replacement in forward_logits (encoder + decoder blend sites)
- `ede7895`: TTT forward-path fix — apply torch.lerp blend in forward_ttt too (closes the α=1-at-TTT bug discovered after 017 landed)

**Note:** spec 018's throughput test uses `ENABLE_LOOPING_AT=0` + `ITERATIONS=150` + training-only (no TTT phase). So the TTT fix is *included* in the commit but *not exercised* by 018's diagnostic. It's shipped here so downstream full-pipeline specs can fork from this branch with the fix intact.

**Patch:** Two one-line changes in `forward_logits` of `records/.../train_gpt.py`:

```python
# Encoder loop (line ~1204):
- x = alpha * x_new + (1.0 - alpha) * x_before
+ x = torch.lerp(x_before, x_new, alpha)

# Decoder loop (line ~1265):
- x = alpha * x_new + (1.0 - alpha) * x_before
+ x = torch.lerp(x_before, x_new, alpha)
```

Argument order: `torch.lerp(start, end, weight)` = `start + weight * (end − start)`. With `start=x_before, end=x_new, weight=alpha`, we get `x_before + alpha * (x_new − x_before) = (1 − alpha) * x_before + alpha * x_new`. Mathematically identical to the current code.

No other changes. Commit will be small (~4 lines diff).

## Model config (same proxy as 016b)

```
NUM_LAYERS=6  MODEL_DIM=256  XSA_LAST_N=6  PARALLEL_START_LAYER=99
LOOP_START=3  LOOP_END=5  NUM_LOOPS=2  (unchanged)
CASEOPS_ENABLED=1  GATED_ATTN_ENABLED=1  GATED_ATTN_QUANT_GATE=1  (unchanged)
ENABLE_LOOPING_AT=0  (forces looping + blend-op active from step 1)
ITERATIONS=150  TRAIN_LOG_EVERY=25  SEED=42
```

Proxy caveat: mini-model overstates blend overhead vs full model by ~6×. If we see 5% throughput gain from lerp at proxy scale, expect ~0.8% on full model — still worth it given the endpoint-step implications.

## Hardware

**2×H100 NA US-NE-1** pod, NA volume `hvpdph5i3g` mounted at `/workspace` (same config 016b used successfully).

## Execution protocol

Single pod, multiple sequential runs. If 016b hasn't run yet, include Run A and Run B here. If 016b already ran, start from Run C (the lerp variant).

### Run A (baseline, 008 commit, no recur-alpha) — skip if already done in 016b

```bash
git checkout 154c9b8
# Launch as in 016b Run A
```

### Run B (current blend, 4dd2d63) — skip if already done in 016b

```bash
git checkout 4dd2d63
# RECUR_ALPHA_ENABLED=1, rest same as 016b Run B
```

### Run C (lerp blend, new commit on exp/recur-alpha-lerp)

```bash
git checkout 97d9854

mkdir -p /workspace/runs/018-recur-alpha-lerp/run-c-lerp

NCCL_NET=Socket DATA_DIR=/workspace/parameter-golf/data \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache \
ARTIFACT_DIR=/workspace/runs/018-recur-alpha-lerp/run-c-lerp \
CASEOPS_ENABLED=1 GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
ENABLE_LOOPING_AT=0 TRAIN_LOG_EVERY=25 SEED=42 ITERATIONS=150 \
NUM_LAYERS=6 MODEL_DIM=256 XSA_LAST_N=6 PARALLEL_START_LAYER=99 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  > /workspace/runs/018-recur-alpha-lerp/run-c-lerp/train.log 2>&1
```

Different commit than Run B → fresh compile (~5 min). Read tok/s at steps 50, 100, 150.

## Expected artifacts

```
runs/018-recur-alpha-lerp/
  run-c-lerp/train.log
  (run-a-008, run-b-current-blend here if 016b hadn't run)
  notes.md   # ratios, decision
```

`notes.md` must include:
- Raw tok/s at steps 50/100/150 for Run C (lerp) and control (Run B or 016b Run B)
- Ratio lerp/current at each matched step
- Blend overhead recovered (as % of the original blend overhead from 016b's B vs A)
- Decision: which bucket we landed in

## Stop-early criteria

Same as 016b:
- NaN / inf in step time → halt
- Compile failure → halt (one-line change shouldn't break anything, but possible)
- Step 150 reached → done

## Cost estimate

| item | cost |
|---|---|
| Pod boot + SSH | ~$0.10 |
| Run C (lerp): ~5 min compile + 25s training | ~$0.40 |
| (if needed) Run A baseline: ~$0.15 | — |
| (if needed) Run B current blend: ~$0.40 | — |
| Rsync + pod stop | ~$0.05 |
| **Total 018 diagnostic (assuming 016b provided Runs A, B)** | **~$0.55** |
| **Total if 018 has to run all three** | **~$1.10** |

Wallclock: 10-20 min depending on whether 016b data is available.

## Open questions for interview

1. **016b status at launch time?** If 016b already ran, 018 is just Run C (cheapest path). If not, 018 should absorb Runs A and B as well.
2. **Does torch.compile fully fuse `torch.lerp`?** Expected yes — lerp is a first-class aten op. If the measured throughput gain is significantly less than expected, Dynamo might be breaking fusion anyway, and we'd pivot to bake-into-block faster.
3. **Should we also test Run D = in-place `x.lerp_(x_new, alpha)`?** Could save another memory op. Simple extension, adds one more run ~$0.40. Defer to execution judgment.

## Sequencing

- Run before: 016b (provides baseline Run A and current-blend Run B data)
- Run after: if lerp wins at proxy scale, spec 017-variant with lerp on full model (~$10, real submission test)
- Orthogonal: spec 019 (user's "cool idea") is independent and doesn't depend on 018's outcome

## What 018 does NOT do

- Does not test on full 8×H100 production model (that's spec 017-with-lerp if 018 promotes)
- Does not measure val_bpb (loss is catastrophic by design with ENABLE_LOOPING_AT=0)
- Does not test bake-into-block refactor (that's a follow-up if lerp underperforms)
- Does not run 3-seed (single-seed diagnostic sufficient for throughput measurement)
