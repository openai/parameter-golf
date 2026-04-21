# Spec 018c — Recur-Alpha with α as compile-time constants

**Slug:** `recur-alpha-constant`
**Created:** 2026-04-21
**Links to:** specs 018 (lerp), 018b (bake-in, shelved), `research/evaluations/018-recur-alpha-lerp.md`

## Hypothesis

Spec 018 showed torch.lerp only recovered 18% of the blend overhead — the compiler couldn't aggressively specialize because α is a tensor Parameter (runtime value). Hardcoding α as Python floats (017's endpoint values) should make them compile-time constants, allowing torch.compile to:

- Generate specialized lerp kernels per-site with the known weight baked in
- Fuse more aggressively with surrounding ops (block output → blend → next block input) now that one input is a compile-time constant
- Potentially recognize edge cases (α=1.0 → identity) and eliminate those sites

Expected additional gain over lerp: **some — magnitude unclear.** Based on 018b's experience (compile optimization can backfire), there's a real chance this is also a null or regression. Test is cheap either way.

## Baseline

- 016b Run A (no recur-alpha): 3,333K tok/s (proxy baseline)
- 016b Run B (current 4-op blend): 3,234K tok/s
- 018 Run C (torch.lerp, tensor α): 3,252K tok/s

Primary comparison for 018c: **Run C (lerp with tensor α) vs Run E (lerp with Python-float α).**

## Decision criterion

Let L = lerp tok/s (Run C from 018), K = constant tok/s (new Run E).

| Scenario | Interpretation | Next action |
|---|---|---|
| K ≥ 1.02 × L (≥65K tok/s gain) | Compile meaningfully specializes on constant α | Apply hardcoded α in production. Considerable win. |
| 1.005 × L ≤ K < 1.02 × L | Small but real specialization gain | Modest improvement, still worth applying if low engineering cost. |
| K ≈ L (within 0.5%) | Compile doesn't specialize meaningfully | No gain from hardcoded α. Stick with learnable + lerp. |
| K < L | Constant-α hurts — compile pattern regresses | Shelve; stick with lerp. |

Any outcome informs whether "compile-time constant α" is a real lever.

## Code changes

**Branch:** `exp/recur-alpha-constant` forking from `97d9854` (018 lerp, training path only — no TTT fix since this is training-only throughput test).
**Commit:** `aabfbea` on `fork/exp/recur-alpha-constant`.

**Scope**: ~30 LOC change.
- `GPT.__init__`: replace `self.recur_alpha = nn.Parameter(...)` with None. Hardcode `_ALPHA_CONSTANTS_017 = ((1.078125, 1.2734375, 1.3984375), (1.015625, 0.97265625, 0.83203125))`. Store Python floats directly in `_encoder_alpha_info` / `_decoder_alpha_info` lists.
- `forward_logits`: `alpha = enc_alpha_info[step_idx]` (Python float) instead of tensor index. No `.to(x_new.dtype)` needed.
- Remove p2p cos diagnostic code that referenced `pass_off, local_idx` (not needed for throughput test; diagnostic off by default).

At compile time, torch sees `torch.lerp(x_before, x_new, 1.3984375)` as a literal weight — specializes kernel.

## Model config (same proxy as 018)

```
NUM_LAYERS=6  MODEL_DIM=256  XSA_LAST_N=6  PARALLEL_START_LAYER=99
ENABLE_LOOPING_AT=0  ITERATIONS=150  TRAIN_LOG_EVERY=25
RECUR_ALPHA_ENABLED=1  SEED=42
```

α values from the `_ALPHA_CONSTANTS_017` table are used — no training, just constants.

## Hardware

Same **2×H100 NA US-NE-1** pod as 018/018b (if still live) or a fresh one. Chain onto existing pod if possible to use warm torch compile cache.

## Execution protocol

Single Run E:

```bash
git checkout aabfbea

mkdir -p /workspace/runs/018c-recur-alpha-constant/run-e-constant

NCCL_NET=Socket DATA_DIR=/workspace/parameter-golf/data \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache \
ARTIFACT_DIR=/workspace/runs/018c-recur-alpha-constant/run-e-constant \
CASEOPS_ENABLED=1 GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
ENABLE_LOOPING_AT=0 TRAIN_LOG_EVERY=25 SEED=42 ITERATIONS=150 \
NUM_LAYERS=6 MODEL_DIM=256 XSA_LAST_N=6 PARALLEL_START_LAYER=99 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  > /workspace/runs/018c-recur-alpha-constant/run-e-constant/train.log 2>&1
```

Different commit from 018/018b → fresh compile (~6 min, maybe slightly longer due to 6 specialized lerp kernels). Then 150 steps of training.

Read tok/s at steps 100, 125, 150.

## Expected artifacts

```
runs/018c-recur-alpha-constant/
  run-e-constant/train.log
  notes.md   # ratios, decision
```

`notes.md` should include tok/s comparison vs Run C (018 lerp) and the decision bucket.

## Stop-early criteria

- NaN / inf in step time → halt (very unlikely given code change is trivial)
- Compile failure → halt (could happen if 6 specializations crash compile for some reason)
- Step 150 reached → done

## Cost estimate

| item | cost |
|---|---|
| Pod boot (if fresh) | ~$0.10 |
| Run E: ~6 min compile + 2 min training | ~$0.80 |
| **Total (fresh pod)** | **~$0.90** |
| **Total (chained onto 018/018b pod)** | **~$0.80** |

## Open questions

1. **Compile time inflation**: 6 specialized lerp kernels may add compile time vs 1 generic kernel. Is this tolerable? (150-step test completes quickly, amortization not an issue.)
2. **Does torch.compile actually specialize on Python-float lerp weights?** Empirical answer from this run. If it treats them as runtime args anyway (e.g. Dynamo's guards aren't tight enough), gain will be null.
3. **If gain is real**, do we promote hardcoded-α to spec 019 (full-pipeline)? Means giving up α learning — Flavor 2 of the earlier framing. Path-dependence risk.

## What 018c does NOT do

- Does not learn α — values are fixed at 017's endpoint
- Does not produce val_bpb (catastrophic loss by design)
- Does not test TTT or full-pipeline — training-only throughput diagnostic
- Does not change inference behavior at α=learned vs α=017-endpoint (those are the same numerical values)

## Sequencing

Run on whatever pod is available after 018/018b. If the existing pod is still up, chain onto it for warm cache. If it was stopped, spin fresh — cost is still negligible.
