# Spec 026 — Cross-layer carry frozen, 8×H100 JP full pipeline

**Slug:** `cross-layer-carry-frozen-8xh`
**Created:** 2026-04-22
**Status:** READY
**Branch:** `exp/recur-alpha-buffer`
**Commit:** `d70888f` (025b shared-frozen + LoRA warm-start-A; see open question 1 re: 025c)
**Links to:** `research/specs/025b-cross-layer-carry-frozen.md`

## Hypothesis

025b (frozen cross-layer blend at 024b converged values) produced val@4000=**1.1079** at 4×H
— a Δ=**−0.0098** vs 021c's 1.1177, the largest positive signal in the entire arc, beating
every 4×H and 8×H run to date. The structural routing (L4 self-subtract, L5 pulls from
L3+L4) baked in from step 0 eliminates the "discovery cost" learnable alpha pays in early
training. At 8×H the full step budget amplifies this advantage, and phased TTT should extract
the same ~−0.013 delta as 021e. Additionally, LoRA warm-start-A (keep A across batch resets,
scale output by alpha/rank) improves TTT expressivity per renqianluo/#1767 and bigbag/#1771
(Δ ~−0.001). Projection: post-TTT near **1.051–1.057**.

## Baselines

| run | val@4000 | pre-quant EMA | post-TTT |
|---|---|---|---|
| 021e (frozen diagonal, 8×H) | 1.1134 | 1.06944 | **1.06622** |
| 025b (frozen cross, 4×H) | **1.1079** | TBD | TBD |
| #1736 (SOTA baseline) | — | — | 1.06610 |

Target to beat: 021e post-TTT **1.06622**.

## Accept criteria

| post-TTT bpb | Bucket | Action |
|---|---|---|
| < 1.062 | Clear beat, >0.004 margin | 3-seed (42/43/44) same pod → submission |
| [1.062, 1.066] | Beats 021e by 0–4 milli | 3-seed to confirm |
| (1.066, 1.070] | Borderline / noise | Compare to 021e seed 43/44; decide |
| > 1.070 | Cross-layer arc closed | Kill; fallback to 021e 3-seed |

Early signal (step 4000, informational only — let full pipeline run):

| val@4000 | Meaning |
|---|---|
| ≤ 1.105 | On track for clear beat |
| 1.105–1.112 | Marginal |
| > 1.112 | Off-track |

## Config diff vs 025b mini

| | 025b mini | 026 full |
|---|---|---|
| nproc_per_node | 4 | 8 |
| MAX_WALLCLOCK_SECONDS | 1200 | (none — full run) |
| PHASED_TTT_ENABLED | 0 | 1 |
| PHASED_TTT_PREFIX_DOCS | — | 2000 |
| PHASED_TTT_NUM_PHASES | — | 3 |
| paths | /workspace | /runpod |
| `TTT_LORA_ALPHA` | (absent, default 96) | **`144`** |
| `TTT_WEIGHT_DECAY` | `0.5` | **`1.0`** |

Commit: `d70888f` (adds LoRA warm-start-A on top of `950af24`).

## Hardware ladder

**8×H100 AP-JP-1 required.** Do not substitute.
Mini already validated at 4×H (025b, val@4000=1.1079). Skip mini rung.
LoRA warm-start-A is TTT-only — no training-path code changes, no new mini required.

## Seed plan

Seed 42 first. **Seed 43/44 conditional on post-TTT ≤ 1.066** (matches or beats 021e).
Same pod, sequential.

## Run protocol

```bash
pip install --break-system-packages brotli
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout d70888f

# Sanity verify
grep "1.5973426" train_gpt.py                              # beta[L3]
grep "\-0.34765625" train_gpt.py                           # alpha[L4,L4] self-subtract
grep -c "recur_beta\[local_idx\]" train_gpt.py             # must be 4
grep -c "recur_alpha\[local_idx, j\]" train_gpt.py         # must be 4
grep "recur_beta.*requires_grad\|recur_alpha.*requires_grad" train_gpt.py  # must be empty
grep "warm-start A" train_gpt.py                           # must be present
grep "_scale = alpha / rank" train_gpt.py                  # must be present

mkdir -p /runpod/runs/026-cross-layer-carry-frozen-8xh/seed_42
mkdir -p /tmp/torch_inductor_cache_026_8h_jp

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/026-cross-layer-carry-frozen-8xh/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/026-cross-layer-carry-frozen-8xh/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_026_8h_jp \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/026-cross-layer-carry-frozen-8xh/seed_42/train.log 2>&1

kill $NVSMI_PID
```

## Checkpoints / artifacts

- `final_model.pt` — post-EMA FP state dict
- `final_model.int6.ptz` — quantized submission artifact
- `train.log` — every-100-step tok/s, val_bpb, train_loss, TTT trace
- `diag_nvsmi.csv` — per-GPU telemetry
- `final.json` — `val_bpb_pre_gptq_post_ema`, `val_bpb_post_gptq`, `val_bpb_post_ttt`

## Stop-early criteria

- NaN/inf in train_loss → halt immediately
- Step time > 2× 021e's (~1.35 s/step at 8×H) → halt
- `layer_loop_enabled_at_step` outside [2000, 2300] → halt

## Cost estimate

| item | cost |
|---|---|
| 8×H JP × ~25 min (compile + train + GPTQ + TTT) | ~$10 |
| Conditional 3-seed × 2 | ~$20 |
| **Max total** | **~$30** |

## Open questions for executor interview

1. **025c result available?** If 025c (per-pass frozen, commit `414cbc3`) has completed
   with val@4000 < 1.1079 before this pod is provisioned, switch to commit `414cbc3` and
   update ARTIFACT_DIR to `026-cross-layer-carry-frozen-per-pass-8xh`. The sanity verify
   changes to: `grep -c "recur_beta\[pass_off, local_idx\]" train_gpt.py` must be 4.
   Ask user before switching.

2. **JP stock?** Last run used AP-JP-1. Provision with `--template-id y5cejece4j`
   (parameter-golf image). Do not use other regions.

3. **Monitoring cadence?** Ask user: poll every 30s or leave to notification?
