# Spec 021b — Recur-α buffer in bfloat16 (dtype-matched blend)

**Slug:** `recur-alpha-buffer-bf16`
**Created:** 2026-04-22
**Status:** READY
**Links to:** `research/specs/021-recur-alpha-buffer.md`, `research/evaluations/021-recur-alpha-buffer.md`

## Context

Spec 021 (buffer-α, commit `cb5cd78`) regressed train loss by +0.005–0.023 vs 019b starting at loop activation (step ~2156). The α-value-typo hypothesis was tested by spec 021-fix (commit `dc0b5f8`, one-digit correction at pass-3 L4): fix had **zero measurable effect** on train loss. Ruled out.

Remaining mechanism candidate: **buffer-read vs Python-literal numerical/fusion path**.

- **019b literal-α**: `α = 1.2734375` is a Python float, Inductor const-folds it into a single fused bf16 pointwise kernel covering `α·x_new + (1−α)·x_before`.
- **021 buffer-α (float32)**: `α = self.recur_alpha[pi, li].to(x_new.dtype)` — a float32 buffer read, a `.to(bf16)` cast, then blend. Not const-foldable; likely breaks fusion; introduces a cast kernel; may cause extra-precision intermediates.

This spec tests the fix: **store the buffer in bfloat16 directly, remove the runtime `.to()` cast**. If fusion/dtype was the issue, post-loop losses should match 019b within pod variance. Throughput profile should be preserved (bf16 buffer reads are no slower than fp32 buffer reads at scalar size).

## Hypothesis

With α storage matching activation dtype (bf16) and no cast kernel at use site, the buffer-α blend fuses like 019b's literal-α blend. Post-loop train loss tracks 019b within ±0.003. Throughput stays at 021's clean profile (zero Type B spikes, ~8.13M tok/s pre-loop, flat post-loop, +167 steps vs 019b). Resulting pre-quant post-EMA lands ~1.062–1.064 (019b + step-count improvement).

## Baseline

- **Primary per-step reference:** `runs/019b-recur-alpha-manual-constant-full/seed_42/` (8×H JP, post-TTT 1.06628).
- **Prior-attempt reference:** `runs/021-recur-alpha-buffer/seed_42_8h_jp/` (post-TTT 1.06900, buggy α) and `runs/021-recur-alpha-buffer/seed_42_fix/` (post-TTT TBD, α-fixed but buffer-f32).

## Expected Δ

Projection matches spec 021's original projection (pre-loss-gap correction):
- Per-step loss: match 019b at every matched step.
- Final step: ~4880 (stall-free throughput preserved).
- Pre-quant post-EMA: **~1.062–1.064** (019b's 1.06951 minus ~0.007 from +167 step-count gain on matched trajectory).
- **Post-TTT: ~1.053–1.060** if TTT gains compose.

Confidence: medium. Dtype/fusion is the leading remaining candidate but not proven. ~50% this closes the gap; ~30% it partially closes; ~20% it doesn't move, meaning buffer-α has some deeper autograd/numerical penalty.

## Accept criteria

**Primary (post-TTT):**

| Post-TTT | Bucket | Next action |
|---|---|---|
| ≤ 1.06400 | Clear beat #1736 | 3-seed (43/44) confirm → submit |
| (1.06400, 1.06610] | Tight beat | 3-seed |
| (1.06610, 1.06710] | Borderline (tie/miss #1736 by ≤0.001) | Hold; compare to 019b 3-seed replication |
| > 1.06710 | Arc closed | Shelve buffer-α entirely; pivot |

**Secondary (step-match to 019b, early-signal):**

| Matched-step criterion | Interpretation |
|---|---|
| step 3000 train_loss ≤ 2.5643 (019b + 0.003) | dtype/fusion was the cause → proceed to TTT |
| step 3000 train_loss > 2.5680 (still +0.007 elevated) | dtype didn't help → early-abort, save TTT cost |

## Config diff

Identical to spec 021 — same env block, same `RECUR_ALPHA_ENABLED=1`, same `MATRIX_LR=0.026`, same `ENABLE_LOOPING_AT` default (0.35).

The only change is in the code.

## Code changes

**Branch:** `exp/recur-alpha-buffer`
**Commit:** **`d070df3`** (builds on `dc0b5f8` which built on `cb5cd78`)

3-line effective diff vs `dc0b5f8`:

```python
# At buffer init (line ~1060):
_recur_alpha_017_endpoint = torch.tensor(
    [[1.078125, 1.2734375, 1.4296875],
     [1.015625, 0.97265625, 0.83203125]],
    dtype=torch.bfloat16,   # was torch.float32
)

# At blend site, encoder (line ~1214):
alpha = self.recur_alpha[pass_off, local_idx]   # was .to(x_new.dtype)

# At blend site, decoder (line ~1275):
alpha = self.recur_alpha[pass_off, local_idx]   # was .to(x_new.dtype)
```

All six α values (multiples of 1/128 ≤ 1.5) fit bf16's 7-bit mantissa exactly → zero precision loss from dtype narrowing.

## Hardware ladder

**8×H100 JP preferred** (matched to all reference runs). NE-1 acceptable if JP unavailable. **Do NOT run on 4×H100** — hardware-variance confound would obscure the dtype signal (see prior 4×H 021 run's wildly different trajectory).

Run **sequentially on the seed_42_fix pod** if it's still alive: warm inductor cache for non-blend kernels, brotli pre-installed, data mounted.

## Seed plan

Seed 42 first. **3-seed (42/43/44) conditional on post-TTT ≤ 1.06610** (clear beat or tie #1736). Same pod for all three seeds.

## Inputs

- Data: CaseOps dataset, JP `/runpod/data/...`
- Tokenizer: `fineweb_8192_bpe.model` bundled
- Hotstart: none

## Run protocol

```bash
cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout d070df3

# Verify the fix
grep "dtype=torch.bfloat16" train_gpt.py | head -1   # must match
grep ".to(x_new.dtype)" train_gpt.py                  # must return nothing

# Verify brotli installed (should already be from seed_42_fix)
python -c "import brotli"

mkdir -p /runpod/runs/021-recur-alpha-buffer-8xh100/seed_42_bf16

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/021-recur-alpha-buffer-8xh100/seed_42_bf16/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/021-recur-alpha-buffer-8xh100/seed_42_bf16 \
TORCHINDUCTOR_CACHE_DIR=/runpod/.torch_inductor_cache_021_8xh100 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/021-recur-alpha-buffer-8xh100/seed_42_bf16/train.log 2>&1

kill $NVSMI_PID
```

## Checkpoints / artifacts

- `final_model.pt` — post-EMA FP state dict
- `final_model.int6.ptz` — quantized submission artifact
- `train.log` — every-100-step tok/s, val_bpb, train_loss, TTT trace
- `diag_nvsmi.csv` — per-GPU telemetry
- `final.json` — `val_bpb_pre_gptq_post_ema`, `val_bpb_post_gptq`, **`val_bpb_post_ttt`**, `stopping_early_at_step`, `layer_loop_enabled_at_step`, `recur_alpha_is_buffer: true`, `recur_alpha_dtype: "bfloat16"`, `recur_alpha_values_hardcoded`

## Stop-early criteria

- NaN/inf in train_loss → halt
- Step time > 2× 019b's (> ~240ms) → halt
- `layer_loop_enabled_at_step` outside [2000, 2300] → halt
- **Step 3000 train_loss > 2.5680** (still +0.007 elevated vs 019b) → halt before TTT; saves ~$2

## Cost estimate

| item | cost |
|---|---|
| 8×H100 × ~12 min on warm pod | ~$5 |
| Extra Inductor compile (blend graph changed) | ~$0.50 |
| (Conditional) 3-seed extension | ~$10 additional |
| **Single-seed total** | **~$5.50** |
| **3-seed total if promoted** | **~$15.50** |

## Extra artifacts

None beyond standard. Spec 021's nvsmi + final.json fields suffice.

## Open questions for interview

1. **Recompile scope:** bf16 buffer changes blend input dtype → blend graph is structurally different from fp32-buffer+`.to()` graph. Inductor will re-codegen blend kernels. Attention/MLP/quant kernels cache-hit. Expected compile overhead: ~60-120s (not 5-10s like a cache-hit rerun). Acceptable.

2. **If bf16 variant ALSO fails to close the gap:** dtype/fusion is not the mechanism. Likely remaining candidates: (a) autograd graph differences from buffer vs literal even in fp16 path, (b) some interaction with `isinstance(..., nn.Parameter)` optimizer guard. In that case: shelve buffer-α and pivot to 019b 3-seed replication (best remaining shot at beating #1736 — its 1.06628 misses by 0.00018, within seed-std).

3. **Early-abort at step 3000:** if train_loss is still +0.007 above 019b at step 3000, the mechanism is not dtype and TTT won't rescue it. Halting saves ~$2 of TTT+quant time. Interview: halt-on-abort-criterion policy = halt.

## What this does NOT test

- Whether literal-α (019b) + longer wallclock / different loop schedule can reach the same step count as 021 → that's a separate 019b variant, not in scope here.
- Whether buffer-α interacts badly with TTT independent of training → controllable only after training-loss gap is closed.
