# Spec 033b — TTT-only: adapt frozen alpha/beta with aggressive LR

**Slug:** `ttt-adapt-alpha-beta-high-lr`
**Created:** 2026-04-23
**Status:** DRAFT
**Branch:** `exp/033-ttt-adapt-alpha-beta`
**Commit:** `3513aac`
**Links to:** `research/ideas/033b-ttt-adapt-alpha-beta-high-lr.md`, `research/specs/033-ttt-adapt-alpha-beta.md`, `research/specs/028-ttt-only-026-seed42.md`, `research/specs/026-cross-layer-carry-frozen-8xh.md`

## Hypothesis

`033` was directionally better than `028B`, but only by a negligible amount:

- `028B`: `1.0664948109`
- `033`: `1.0664878103`

The likely explanation is that the TTT alpha/beta LR was too small to be a real
test:

- base `TTT_LORA_LR = 1e-4`
- `033` used `TTT_ALPHA_BETA_LR_SCALE=0.25`
- effective alpha/beta LR was only `2.5e-5`

This spec asks the narrow follow-up:

- if we raise alpha/beta TTT LR aggressively, do these parameters finally move
  enough to matter?

## Base checkpoint

Same checkpoint as `028` and `033`:

```text
/workspace/runs/026-cross-layer-carry-frozen-8xh/seed_42/final_model.pt
```

## Comparison targets

Primary comparisons:

- `028B`: `1.0664948109`
- `033`: `1.0664878103`

Interpretation target:

- beat `033` by more than trivial noise
- or conclude the line is not worth more cycles

## Mechanism

Same mechanism as `033`:

- frozen checkpoint values for `recur_alpha/recur_beta`
- trainable during TTT only
- same LoRA warm-start behavior
- same hotstart path
- same code commit

Only pinned change from `033`:

- `TTT_ALPHA_BETA_LR_SCALE=10.0`

With base `TTT_LORA_LR=1e-4`, effective alpha/beta LR becomes:

```text
1e-3
```

## Why keep this in the `033` family

This is not a new mechanism and not a new code path.

It is a single-parameter rerun of `033`, so it should stay in the same family:

- `033` = conservative alpha/beta TTT adaptation
- `033b` = aggressive alpha/beta TTT adaptation

## Hardware

4×H100 TTT-only run, same class as `028` and `033`.

One run is enough.

## Accept criteria

| post-TTT bpb | verdict |
|---|---|
| meaningfully below `033` | positive, LR was the bottleneck |
| ≈ `033` | likely low-value line |
| worse than `028B` | aggressive LR hurts |

Also inspect:

- `recur_alpha_max_drift`
- `recur_beta_max_drift`
- eval time increase vs `033`
- any instability in phased TTT

## Run protocol

Same hotstart entrypoint as `033`, with only the alpha/beta LR scale changed.

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 3513aac

# Sanity verify
grep -n "TTT_ALPHA_BETA_ENABLED" train_gpt.py
grep -n "TTT_ALPHA_BETA_LR_SCALE" train_gpt.py
grep -n "ttt_alpha_beta:" train_gpt.py

mkdir -p /workspace/runs/033b-ttt-adapt-alpha-beta-high-lr/seed_42
mkdir -p /tmp/torch_inductor_cache_033b

SPINQUANT_MODE=baseline \
HOTSTART_FP_CKPT=/workspace/runs/026-cross-layer-carry-frozen-8xh/seed_42/final_model.pt \
ARTIFACT_DIR=/workspace/runs/033b-ttt-adapt-alpha-beta-high-lr/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_033b \
DATA_DIR=/workspace/parameter-golf/data \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=3 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=2 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
TTT_ALPHA_BETA_ENABLED=1 TTT_ALPHA_BETA_LR_SCALE=10.0 \
GPTQ_RESERVE_SECONDS=0 GPTQ_CALIBRATION_BATCHES=16 \
SEED=42 \
torchrun --standalone --nproc_per_node=4 spinquant_hotstart.py \
  > /workspace/runs/033b-ttt-adapt-alpha-beta-high-lr/seed_42/ttt.log 2>&1
```

## Required logging

The run must log:

- `ttt_alpha_beta: enabled=1 ...`
- `ttt_alpha_beta: before_beta=...`
- `ttt_alpha_beta: before_alpha=...`
- `ttt_alpha_beta: after_beta=...`
- `ttt_alpha_beta: after_alpha=...`
- `ttt_alpha_beta: recur_alpha_max_drift=... recur_beta_max_drift=...`

## Hotstart validation contract

Use the observed `028`/`033` hotstart metrics as the validation reference:

- `diagnostic_pre_quant_post_rotation val_bpb` ≈ `1.32726185`
- `diagnostic_quantized val_bpb` ≈ `1.08045767`

Do **not** compare the hotstart-side diagnostic to `026`'s training-side
pre-quant EMA; they are different diagnostics.

## Stop-early criteria

- NaN in TTT loss → halt
- `diagnostic_pre_quant_post_rotation` materially different from `~1.32726` → halt
- `diagnostic_quantized` materially different from `~1.08046` → halt
- missing `ttt_alpha_beta:` logs → halt
- post-TTT worse than `028A` old-TTT territory (`> 1.0673`) → flag hard

## Cost estimate

Same class as `033`: one 4×H TTT-only run, low single-digit dollars.

## Execution note

Execution should validate:

- checkpoint path exists
- commit `3513aac` contains the patched alpha/beta TTT path
- data root is `/workspace/parameter-golf/data`
- `spinquant_hotstart.py` is the entrypoint being launched
