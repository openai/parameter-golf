# Spec 026 — Cross-layer carry frozen, 8×H100 JP full pipeline

**Slug:** `cross-layer-carry-frozen-8xh`
**Created:** 2026-04-22
**Status:** READY
**Branch:** `exp/recur-alpha-buffer`
**Commit:** `d70888f` (025c per-pass frozen + LoRA warm-start-A; builds on `414cbc3` not `950af24`)
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

Benchmark: **#1769** (dexhunter, clip=12, 5-seed mean 1.06453, seed 314 = 1.06357).
Seed 42 established 1.06582 — gap was entirely in training seed, not GPTQ/TTT.

| post-TTT bpb | Bucket | Action |
|---|---|---|
| < 1.062 | Beats #1769 seed 314 (1.06357) — clear new SOTA | Run seeds 2025 + 777 same pod → 3-seed submission |
| [1.062, 1.065] | Matches #1769 mean, LoRA warm-start-A contributing | Run seeds 2025 + 777 to confirm mean |
| (1.065, 1.068] | Seed 314 didn't help as expected | Debug: check val@4000 vs seed 42's 1.1128 |
| > 1.068 | Regression vs seed 42 | Kill; investigate LoRA warm-start-A interaction |

Early signal (step 4000, informational only — let full pipeline run):

| val@4000 | Meaning |
|---|---|
| ≤ 1.108 | Seed 314 producing better float than seed 42 (1.1128) — on track |
| 1.108–1.115 | Marginal improvement |
| > 1.115 | Seed 314 not helping at training level |

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

**4×H100 JP screen first (seed 314 only), then 8×H100 JP full pipeline.**

Architecture already validated at 4×H (025b). This screen is purely a seed check —
confirm seed 314 produces a better float than seed 42 (1.06893) before spending $12 on
the full pipeline. LoRA warm-start-A is TTT-only so does not affect the screen result.

## Seed plan

Seed 42 already run (post-TTT 1.06582 — seed 42 is a known mediocre training seed inherited
from #1736). Next run: **seed 314** — dexhunter's best seed in #1769 (float 1.06637,
post-TTT 1.06357). Seeds 2025 and 777 conditional on seed 314 result.
Same pod, sequential.

## Run protocol

### Screen (4×H100 JP) — seed 314 float check

```bash
pip install --break-system-packages brotli
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout d70888f

# Sanity verify
grep "0.9453125" train_gpt.py
grep "\-0.3046875" train_gpt.py
grep "warm-start A" train_gpt.py

mkdir -p /runpod/runs/026-cross-layer-carry-frozen-8xh/screen_seed_314
mkdir -p /tmp/torch_inductor_cache_026_screen

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/026-cross-layer-carry-frozen-8xh/screen_seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_026_screen \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /runpod/runs/026-cross-layer-carry-frozen-8xh/screen_seed_314/train.log 2>&1
```

**Screen pass/fail** — compare against 025b (seed 42, 4×H, same architecture):

| metric | 025b seed 42 (baseline) | seed 314 target | action if missed |
|---|---|---|---|
| val@4000 | 1.1079 | ≤ 1.105 | note but don't halt — float is the gate |
| pre-quant EMA | 1.06917 | < 1.068 | try seed 2025 screen before 8×H |
| steps | 4756 | ~4750–4870 | normal range |

Pass → proceed to 8×H full pipeline. Fail → try seed 2025 screen (~$4) before committing to 8×H.

### Full pipeline (8×H100 JP) — seed 314

```bash
pip install --break-system-packages brotli
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout d70888f

# Sanity verify (025c per-pass values + LoRA warm-start-A)
grep "0.9453125" train_gpt.py                              # recur_beta[0,0] from 025c
grep "\-0.3046875" train_gpt.py                            # recur_alpha self-subtract term
grep -c "recur_beta\[pass_off, local_idx\]" train_gpt.py   # must be 4 (per-pass indexing)
grep -c "recur_alpha\[pass_off, local_idx, j\]" train_gpt.py  # must be 4
grep "recur_beta.*requires_grad\|recur_alpha.*requires_grad" train_gpt.py  # must be empty
grep "warm-start A" train_gpt.py                           # must be present
grep "_scale = alpha / rank" train_gpt.py                  # must be present

mkdir -p /runpod/runs/026-cross-layer-carry-frozen-8xh/seed_314
mkdir -p /tmp/torch_inductor_cache_026_8h_jp

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/026-cross-layer-carry-frozen-8xh/seed_314/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/026-cross-layer-carry-frozen-8xh/seed_314 \
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
SEED=314 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/026-cross-layer-carry-frozen-8xh/seed_314/train.log 2>&1

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
| 4×H JP screen seed 314 (~25 min, no TTT) | ~$4 |
| 8×H JP seed 314 full pipeline (conditional on screen pass) | ~$12 |
| Conditional seeds 2025 + 777 × 8×H JP | ~$24 |
| **Max total** | **~$40** |

## Open questions for executor interview

1. **JP stock?** Provision with `--template-id y5cejece4j` (parameter-golf image).
   Do not use other regions.

2. **Monitoring cadence?** Ask user: poll every 30s or leave to notification?

_Note: 025c already ran and was shelved (025b beats it on all metrics). Commit stays `d70888f`._
