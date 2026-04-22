# Spec 018d — Recur-Alpha with manual addition + constant α

**Slug:** `recur-alpha-manual-constant`
**Created:** 2026-04-21
**Links to:** spec 018c (`research/evaluations/018c-recur-alpha-constant.md`), spec 019 (full-scale regression of 018c's recipe)

## Hypothesis

Spec 018c's recipe (`torch.lerp(x_before, x_new, α_literal)`) recovered 92% of blend overhead at proxy scale but **regressed −2.7% at full scale** (spec 019 vs 008 baseline). The proxy → full inversion implicates `torch.lerp`'s primitive-template specialization: at proxy, Inductor inlines lerp cleanly; at full, the per-literal-specialized lerp kernels apparently fail to fuse with surrounding ops.

**Manual addition with literal α** sidesteps the lerp template entirely:

```python
alpha = 1.4296875  # Python float
x = alpha * x_new + (1.0 - alpha) * x_before
```

After Dynamo's constant-folding, this is `mul, mul, add` — three pointwise ops with two literal scalars. Inductor's pointwise-fusion pass is **not template-dependent** and should chew this into surrounding pointwise ops (block residual sum). Pointwise fusion is one of Inductor's most reliable passes; we expect it to behave consistently across scales.

**Expected outcome at proxy:** somewhere between Run B (manual + tensor α, −2.9%) and Run E (lerp + literal α, −0.24%). Probably ~−0.5% to −1%. Not expected to beat Run E at proxy — lerp's per-site specialization is pure win when there's no scale penalty.

The real test is at full scale (spec 019b, conditional on 018d).

## Baseline

Add as Run F to the 016b/018/018c family:
- Run A (016b): no recur-alpha → 3,333K tok/s
- Run B (016b): manual + tensor α → 3,234K (−2.9%)
- Run C (018):  lerp + tensor α → 3,252K (−2.4%)
- Run D (018b): bake-in + tensor α → 3,174K (−4.8%) [shelved]
- Run E (018c): lerp + literal α → 3,325K (−0.24%) [proxy winner, full regressed]
- **Run F (018d): manual + literal α → ?** ← this spec

Primary comparison: **Run F vs Run E** (same literal-α scaffold, different blend op).
Secondary: **Run F vs Run B** (same manual blend op, different α type).

## Decision criterion

Let M = Run F tok/s, K = Run E tok/s (lerp+literal), B = Run B tok/s (manual+tensor).

| Scenario | Interpretation | Next action |
|---|---|---|
| M ≥ 0.99 × K (≥ 3,292K) | Manual+literal essentially matches lerp+literal at proxy | Promote to spec 019b — likely robust at scale |
| 0.97 × K ≤ M < 0.99 × K (3,225K–3,292K) | Manual+literal is between Run B and Run E — partial fusion | Worth a 019b spec; downside is bounded |
| M < 0.97 × K (< 3,225K) | Manual is materially worse than lerp at proxy | Skip 019b. The pointwise-fusion theory was wrong. |
| M > K (> 3,325K) | Manual+literal *beats* lerp+literal at proxy — surprise | Definitely promote to 019b; possibly the right answer all along |

## Code changes

**Branch:** `exp/recur-alpha-manual-constant` forking from `aabfbea` (018c).
**Commit:** new commit on this branch with ~4 LOC change.

Diff scope: replace 2 `torch.lerp` calls in `forward_logits` (encoder loop + decoder loop) with manual addition. `forward_ttt` lines do not exist on aabfbea (TTT-fix lives only on the 019 fork).

Encoder site (line ~1210):
```python
# OLD (018c):
x = torch.lerp(x_before, x_new, alpha)

# NEW (018d):
x = alpha * x_new + (1.0 - alpha) * x_before
```

Same change at decoder site (line ~1240ish in forward_logits).

Everything else from 018c stays: `_ALPHA_CONSTANTS_017` table, `self.recur_alpha = None`, alpha_info populated with Python floats at __init__.

## Model config (same proxy as 018/018b/018c)

```
NUM_LAYERS=6  MODEL_DIM=256  XSA_LAST_N=6  PARALLEL_START_LAYER=99
ENABLE_LOOPING_AT=0  ITERATIONS=150  TRAIN_LOG_EVERY=25
RECUR_ALPHA_ENABLED=1  SEED=42
```

## Hardware

**2×H100 NA US-NE-1** (or whichever is available). Ideally chain onto an existing 018-family pod for warm Inductor cache + zero pod-boot cost.

## Execution protocol

```bash
git checkout <new-commit-sha>

mkdir -p /workspace/runs/018d-recur-alpha-manual-constant/run-f-manual

NCCL_NET=Socket DATA_DIR=/workspace/parameter-golf/data \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache \
ARTIFACT_DIR=/workspace/runs/018d-recur-alpha-manual-constant/run-f-manual \
CASEOPS_ENABLED=1 GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
ENABLE_LOOPING_AT=0 TRAIN_LOG_EVERY=25 SEED=42 ITERATIONS=150 \
NUM_LAYERS=6 MODEL_DIM=256 XSA_LAST_N=6 PARALLEL_START_LAYER=99 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  > /workspace/runs/018d-recur-alpha-manual-constant/run-f-manual/train.log 2>&1
```

Different commit from 018c → fresh compile (~5–6 min). Then 150 steps. Read tok/s at steps 100, 125, 150.

## Expected artifacts

```
runs/018d-recur-alpha-manual-constant/
  run-f-manual/train.log
  notes.md  # tok/s vs Run E (and B), decision bucket
```

## Stop-early criteria

- NaN / inf in step time → halt
- Compile failure → halt (very unlikely; 4-LOC change, just elementary arithmetic)
- Step 150 reached → done

## Cost estimate

| item | cost |
|---|---|
| Pod boot (if fresh) | ~$0.10 |
| Run F: ~6 min compile + 1 min training | ~$0.80 |
| **Total (fresh pod)** | **~$0.90** |
| **Total (chained onto 018 pod)** | **~$0.80** |

## Open questions

1. **Will Inductor actually fuse `mul + mul + add` with literal scalars into surrounding block ops?** Empirical answer from this run. The pointwise-fusion theory predicts yes; bake-in (Run D) is the cautionary counter-example, but Run D used tensor coefficients inside Block.forward — structurally different.
2. **Could manual+literal even beat lerp+literal at proxy?** Maybe, if pointwise fusion is more aggressive than lerp's specialization. Probably not — lerp at proxy worked.
3. **What if Run F lands in the (0.97×K, 0.99×K) bucket?** Promote to 019b anyway. Even if it loses to lerp at proxy, the test is whether it survives scale. The prediction is that pointwise fusion is more scale-robust.

## What 018d does NOT do

- Does not learn α — values are fixed at 017's endpoint (same constants as 018c)
- Does not produce val_bpb (catastrophic loss by design at this proxy scale)
- Does not test TTT or full-pipeline — training-only throughput diagnostic
- Does not test α=1 (identity) elimination — doesn't matter for this α range

## Sequencing

Run on whatever pod is available after 018c/019. If 019's pod is still up, chain. Otherwise spin a fresh 2×H100 NA US-NE-1.

After 018d: if Run F passes (≥0.97×K bucket), spec **019b** as full-pipeline test. See `research/specs/019b-recur-alpha-manual-constant-full.md`.
