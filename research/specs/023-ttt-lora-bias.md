# Spec 023 — TTT LoRA Bias

**Slug:** `ttt-lora-bias`
**Created:** 2026-04-22
**Status:** SHELVED — 2026-04-22. Deprioritised in favour of learnable-α warmdown-delay arc (spec 024). TTT-only eval path still valid; revisit post-deadline if time allows.
**Branch:** `exp/ttt-extra-depth` (same branch as spec 022)
**Commit:** `b17cc65`

## Hypothesis

LoRA adapters (A×B) can only express rank-r corrections to each projection. They cannot express a per-document mean shift — a constant offset to all token representations in a document. Adding a bias vector `b` (init zeros) alongside each A×B gives the TTT optimizer a direct channel for per-document mean adaptation. Since `b` starts at zero, it's an exact no-op at TTT start and cannot disrupt the baseline.

## Baseline

Same as spec 022:
- **Spec 008 checkpoint:** `runs/008-1736-reproduction/seed_42/final_model.int6.ptz`
- **#1736 reference post-TTT:** 1.06610

## Expected Δ

Unknown. Mean shift adaptation is complementary to low-rank correction — could help documents with strong domain shift from the base model's average representation.

## Accept criteria

| Post-TTT bpb | Decision |
|---|---|
| < 1.06610 | Beats #1736 — stack with spec 022 and run combined |
| 1.06610 – 1.06700 | Marginal — combine with spec 022 to check stacking |
| > 1.06700 | Bias doesn't help on its own |

## Config diff

| var | value | note |
|---|---|---|
| `TTT_LORA_BIAS` | `1` | adds zero-init bias to all LoRA projections |
| `TTT_EXTRA_DEPTH` | `0` | off — isolate bias signal |
| `TTT_ONLY` | `1` | skip training, load spec 008 checkpoint |
| `QUANTIZED_MODEL_PATH` | `<spec 008 int6 path>` | use trained checkpoint |

## Code changes

**~19 lines in `BatchedLinearLoRA` and `BatchedTTTLoRA`** (same commit as spec 022's `TTT_ONLY` addition):

- `BatchedLinearLoRA` gains optional `bias=False` param; when True adds `nn.Parameter(zeros(bsz,1,out))`, zeroed in `reset()`, added in `forward()`
- `BatchedTTTLoRA.__init__` threads `lora_bias=h.ttt_lora_bias` through to all projection LoRAs
- `TTT_LORA_BIAS=0` (default) → identical to prior behavior

## Hardware ladder

Skip mini — eval-only, same checkpoint as spec 022.

## Run protocol

```bash
cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git checkout b17cc65

mkdir -p /runpod/runs/023-ttt-lora-bias/seed_42

# Restore inductor cache if available (skips ~12s compile)
rsync -a /runpod/torch_compile_caches/ttt_eval/ /tmp/torch_inductor_cache_ttt_eval/ 2>/dev/null || true

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/023-ttt-lora-bias/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_ttt_eval \
QUANTIZED_MODEL_PATH=/runpod/runs/008-1736-reproduction/seed_42/final_model.int6.ptz \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
TTT_ONLY=1 TTT_LORA_BIAS=1 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/023-ttt-lora-bias/seed_42/ttt.log 2>&1

# Save inductor cache for future runs
rsync -a /tmp/torch_inductor_cache_ttt_eval/ /runpod/torch_compile_caches/ttt_eval/
```

## Artifacts to emit

- `ttt.log`
- `final.json` with `val_bpb_post_ttt`, `ttt_eval_time_s`, `ttt_lora_bias: true`

## Stop-early criteria

- NaN in TTT loss → halt

## Cost estimate

~$8 (8×H100 JP, eval-only, ~20 min)

## Run alongside spec 022

Specs 022 and 023 are independent eval-only runs on the same checkpoint. Can run in parallel on separate pods to get both signals in one session (~$16 total).

## Open questions for interview

1. **Stacking**: if both 022 and 023 show improvement, run spec 024 with `TTT_EXTRA_DEPTH=1 TTT_LORA_BIAS=1` to check if they stack additively.
2. **TTT time**: bias adds `out_features` params per slot, negligible vs rank-96 LoRA. No meaningful time increase expected.
