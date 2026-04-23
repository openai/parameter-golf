# Spec 033 — TTT-only: adapt frozen alpha/beta on top of `026 seed_42`

**Slug:** `ttt-adapt-alpha-beta`
**Created:** 2026-04-23
**Status:** READY
**Branch:** `exp/033-ttt-adapt-alpha-beta`
**Commit:** `3513aac`
**Links to:** `research/ideas/033-ttt-adapt-alpha-beta.md`, `research/specs/028-ttt-only-026-seed42.md`, `research/specs/026-cross-layer-carry-frozen-8xh.md`

## Hypothesis

`028B` improved TTT quality on the same frozen `026 seed_42` float checkpoint by changing only TTT settings:

- `028A` old TTT: `1.06724`
- `028B` newer TTT: `1.06649`

But `alpha/beta` remained frozen in both runs.

This spec asks the next narrow question:

- if TTT is allowed to adapt the frozen recurrence carry parameters too, does it improve further?

## Base checkpoint

Same checkpoint as `028`:

```text
/workspace/runs/026-cross-layer-carry-frozen-8xh/seed_42/final_model.pt
```

This is the 8×H frozen-alpha/beta float from `026 seed_42`.

## Comparison target

Primary comparison is directly against `028B`:

- same float checkpoint
- same newer TTT settings
- only change is that `alpha/beta` are added to the TTT trainable set

Target to beat:

- `028B post-TTT = 1.0664948109`

## Mechanism

During TTT only:

- load the frozen alpha/beta values from the checkpoint
- set them trainable
- include them in the TTT optimizer

Important:

- do **not** reinitialize them
- do **not** neutral-init them
- do **not** change the training checkpoint itself

Start from the exact frozen values already present in the float checkpoint.

## Optimizer treatment

Keep the same TTT settings as `028B`:

- `TTT_LORA_ALPHA=144`
- `TTT_WEIGHT_DECAY=1.0`

But add a separate smaller LR for alpha/beta.

Pinned choice:

- `TTT_ALPHA_BETA_ENABLED=1`
- `TTT_ALPHA_BETA_LR_SCALE=0.25`

Interpretation:

- LoRA keeps the normal `TTT_LORA_LR`
- alpha/beta use `0.25x` of that LR

Reason:

- alpha/beta are tiny global structural parameters
- we want gentle adaptation
- full LoRA LR is more likely to over-move them

## Hardware

4×H100 TTT-only run, same class as `028`.

One run is enough for the first answer because the comparison target is already available in `028B`.

## Accept criteria

| post-TTT bpb | verdict |
|---|---|
| < 1.0660 | clear positive signal |
| [1.0660, 1.06649) | modest gain vs `028B` |
| ≈ 1.06649 | alpha/beta adaptation unnecessary |
| > 1.06649 | adapting alpha/beta hurts or adds noise |

Also inspect:

- eval time increase vs `028B`
- whether alpha/beta actually move materially
- whether adaptation remains stable

## Run protocol

Use the same hotstart entrypoint class as `028B`, with the new TTT alpha/beta envs enabled.

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 3513aac

# Sanity verify
grep -n "TTT_ALPHA_BETA_ENABLED" train_gpt.py
grep -n "TTT_ALPHA_BETA_LR_SCALE" train_gpt.py
grep -n "ttt_alpha_beta:" train_gpt.py

mkdir -p /workspace/runs/033-ttt-adapt-alpha-beta/seed_42
mkdir -p /tmp/torch_inductor_cache_033

SPINQUANT_MODE=baseline \
HOTSTART_FP_CKPT=/workspace/runs/026-cross-layer-carry-frozen-8xh/seed_42/final_model.pt \
ARTIFACT_DIR=/workspace/runs/033-ttt-adapt-alpha-beta/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_033 \
DATA_DIR=/workspace/parameter-golf/data \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=3 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=2 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
TTT_ALPHA_BETA_ENABLED=1 TTT_ALPHA_BETA_LR_SCALE=0.25 \
GPTQ_RESERVE_SECONDS=0 GPTQ_CALIBRATION_BATCHES=16 \
SEED=42 \
torchrun --standalone --nproc_per_node=4 spinquant_hotstart.py \
  > /workspace/runs/033-ttt-adapt-alpha-beta/seed_42/ttt.log 2>&1
```

## Required logging

The run must log:

- `ttt_alpha_beta: enabled=1 ...`
- `ttt_alpha_beta: before_beta=...`
- `ttt_alpha_beta: before_alpha=...`
- `ttt_alpha_beta: after_beta=...`
- `ttt_alpha_beta: after_alpha=...`
- `ttt_alpha_beta: recur_alpha_max_drift=... recur_beta_max_drift=...`

This run is not useful unless we can see whether the frozen carry parameters actually moved.

## Hotstart validation contract

For this standalone `spinquant_hotstart.py` path on the `026 seed_42` checkpoint, use the
observed `028` metrics as the validation reference rather than the training-side `026`
pre-quant EMA.

Expected baseline range:

- `diagnostic_pre_quant_post_rotation val_bpb` ≈ `1.32726185`
- `diagnostic_quantized val_bpb` ≈ `1.08045767`
- primary gate is `quantized_ttt_phased` vs `028B = 1.0664948109`

Do **not** compare the hotstart-side `diagnostic_pre_quant_post_rotation` metric to `026`'s
training/eval-side pre-quant EMA `1.06893`; they are different diagnostics from different
pipelines.

## Stop-early criteria

- NaN in TTT loss → halt
- `diagnostic_pre_quant_post_rotation` materially different from `~1.32726` → halt
- `diagnostic_quantized` materially different from `~1.08046` → halt
- missing `ttt_alpha_beta:` logs → halt
- post-TTT worse than `028A` old-TTT territory (`> 1.0673`) → flag hard

## Cost estimate

Same class as `028`: about one 4×H TTT-only run, low single-digit dollars.

## Execution note

The executioner should validate:

- the checkpoint path exists
- the pinned commit contains the new TTT alpha/beta envs
- `spinquant_hotstart.py` is using the patched TTT path from this branch
