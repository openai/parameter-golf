# Spec 022 — TTT Extra Depth

**Slug:** `ttt-extra-depth`
**Created:** 2026-04-22
**Status:** SHELVED — 2026-04-22. Deprioritised in favour of learnable-α warmdown-delay arc (spec 024). TTT-only eval path still valid; revisit post-deadline if time allows.
**Branch:** `exp/ttt-extra-depth`
**Commit:** `b17cc65`

## Hypothesis

The Loop45 model runs 3 recurrence passes during training and TTT. Since the loop block weights are shared and the INT6 artifact is dequantized to BF16 at TTT time, we can extend to 4+ passes at TTT eval time at zero submission cost. New LoRA slots for the extra passes initialize B=0 (exact identity — no correction), so TTT gradient descent learns per-document whether to activate the extra capacity. Documents that benefit from deeper recurrence will push B nonzero; others leave it near zero.

## Baseline

- **Spec 008 checkpoint:** `runs/008-1736-reproduction/seed_42/final_model.int6.ptz`
- **#1736 reference post-TTT:** 1.06610
- **Spec 008 projected post-TTT:** ~1.06626 (never measured — pod stopped before TTT)

## Expected Δ

Unknown. Nobody has tried eval-depth > training-depth. Could be +/− 0.001. The GPTQ calibration was done for 3 passes; the 4th pass runs base weights that were not calibrated for that depth, but LoRA can correct for this.

## Accept criteria

| Post-TTT bpb | Decision |
|---|---|
| < 1.06610 | Beats #1736 — promote to full pipeline run |
| 1.06610 – 1.06700 | Marginal — try depth=2 (5 passes total) before deciding |
| > 1.06700 | Neutral or worse — depth extension doesn't help on this checkpoint |

## Config diff

| var | value | note |
|---|---|---|
| `TTT_EXTRA_DEPTH` | `1` | 3→4 recurrence passes at TTT time |
| `PHASED_TTT_ENABLED` | `1` | same as baseline |
| `PHASED_TTT_PREFIX_DOCS` | `2000` | same as baseline |
| `PHASED_TTT_NUM_PHASES` | `3` | same as baseline |

All other TTT params unchanged from #1736.

## Code changes

**12 lines in `train_gpt.py`:**

1. Env var added to hyperparams: `ttt_extra_depth = int(os.environ.get("TTT_EXTRA_DEPTH", 0))`
2. After `ttt_model.looping_active = True`, extend indices:
```python
if h.ttt_extra_depth > 0 and h.num_loops > 0:
    _loop_seg = list(range(h.loop_start, h.loop_end + 1))
    _new_all = (
        list(range(h.loop_start))
        + _loop_seg * (h.num_loops + 1 + h.ttt_extra_depth)
        + list(range(h.loop_end + 1, h.num_layers))
    )
    _n = len(_new_all) // 2
    ttt_model.encoder_indices = _new_all[:_n]
    ttt_model.decoder_indices = _new_all[_n:]
    log(f"ttt_extra_depth:{h.ttt_extra_depth} slots:{len(_new_all)} ...")
```

`BatchedTTTLoRA` reads `encoder_indices + decoder_indices` to compute `num_slots`, so extra slots auto-allocate with B=0.

## Hardware ladder

**Skip mini — eval-only.** No training. Load checkpoint → TTT → report bpb.

**Rung: 8×H100 JP** (or 4×H100 JP fallback). Eval-only run takes ~15–20 min including TTT.

## Inputs

- Checkpoint: `runs/008-1736-reproduction/seed_42/final_model.int6.ptz`
- Data: CaseOps dataset, JP mount at `/runpod/data/...`
- Tokenizer: bundled

## Run protocol

```bash
cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git checkout 135d7af

mkdir -p /runpod/runs/022-ttt-extra-depth/seed_42

# Restore inductor cache if available (skips ~15min first-compile on TTT_ONLY path)
rsync -a /runpod/torch_compile_caches/ttt_eval/ /tmp/torch_inductor_cache_ttt_eval/ 2>/dev/null || true

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/022-ttt-extra-depth/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_ttt_eval \
QUANTIZED_MODEL_PATH=/runpod/runs/008-1736-reproduction/seed_42/final_model.int6.ptz \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
TTT_ONLY=1 TTT_EXTRA_DEPTH=1 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/022-ttt-extra-depth/seed_42/ttt.log 2>&1

# Save inductor cache for future runs (023, depth sweep variants)
mkdir -p /runpod/torch_compile_caches/ttt_eval
rsync -a /tmp/torch_inductor_cache_ttt_eval/ /runpod/torch_compile_caches/ttt_eval/
```

## Artifacts to emit

- `ttt.log` — full TTT output
- `final.json` — `val_bpb_post_ttt`, `ttt_eval_time_s`, `ttt_extra_depth`, slot counts from log

## Stop-early criteria

- If log shows `ttt_extra_depth:1 slots:N` with N ≠ 20 (expected: 3 pre-loop + 3×4 loop + 5 post-loop = 20) → halt, index math wrong
- NaN in TTT loss → halt

## Cost estimate

| item | cost |
|---|---|
| 8×H100 JP × ~20 min (TTT only, no training) | ~$8 |
| **Total** | **~$8** |

## Open questions for interview

1. **EVAL_ONLY path**: does the baseline script support skipping training and running only post-training pipeline from an existing INT6 checkpoint? If not, execution needs to either (a) set `MAX_WALLCLOCK_SECONDS=0` to skip training immediately, or (b) use a separate eval-only entry point.
2. **Slot count sanity**: confirm log line `ttt_extra_depth:1 slots:17` appears before TTT begins.
3. **TTT time increase**: with 4 passes instead of 3, TTT will be ~33% slower (~650s vs ~500s). Still uncapped — acceptable.
