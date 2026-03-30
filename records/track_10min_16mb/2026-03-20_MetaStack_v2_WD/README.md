# MetaStack v2 WD

## Thesis

This branch keeps the current strong Parameter Golf recipe family intact and adds one schedule-side transfer from the modded-nanogpt ecosystem: decoupled weight decay tied to the existing LR schedule. The main hypothesis is that modest decay on Muon-managed matrices and scalar/control parameters will improve post-quant quality and artifact friendliness without requiring a larger model or a more complex architecture.

## Current Status

Batch #1 complete. 2 Sobol-sampled runs finished on GB10. Weight decay is **directionally neutral to mildly positive** — excellent quant gap but no clear bpb improvement signal yet. Confounded by iteration count differences.

- **Smoke run**: completed (val_bpb=2.6019, int6=2.6360, 50 iters)
- **Batch #1**: **completed** (2 runs, sliding-window eval, 2026-03-20)
- **Best run**: `metastack_wd_slide_local_0000` (sliding_bpb=2.4594, int6=1.4269, quant_gap=0.0021)
- **Search harness**: functional, Sobol + GP pipeline validated end-to-end

Paths:
- Trainer: `records/track_10min_16mb/2026-03-20_MetaStack_v2_WD/train_gpt.py`
- Local smoke preset: `search_configs/metastack_v2_wd_smoke.yaml`
- Serious local sliding preset: `search_configs/metastack_v2_wd_sliding_local.yaml`
- Serious remote sliding preset: `search_configs/metastack_v2_wd_sliding_remote.yaml`
- Search output: `search_runs/metastack_v2_wd_sliding_local/`

Target-hardware leaderboard runs and confirmation reruns are still pending.

## Results

### Smoke Baseline (GB10, 50 iterations, non-sliding eval)

| Metric | Value |
|---|---|
| val_bpb (prequant, terminal) | 2.6019 |
| int8 roundtrip bpb | 2.6029 |
| int6 roundtrip bpb | 2.6360 |
| int6 quant gap | +0.0341 |
| int6 artifact size | 5,424,774 bytes |
| int8 artifact size | 10,387,267 bytes |
| Train time | 10,302 ms (50 steps, ~206 ms/step) |
| Peak GPU memory | 1,059 MiB allocated / 1,098 MiB reserved |
| Model params | 21,778,504 |
| Log | `logs/wd_v2_smoke_direct.txt` |

Smoke knobs: MATRIX_LR=0.02, SCALAR_LR=0.03, EMBED_LR=0.03, MUON_WEIGHT_DECAY=0.001, SCALAR_WEIGHT_DECAY=0.0001, ITERATIONS=50.

The quant gap (0.0341) is reasonable for 50 iterations. The main purpose of the smoke run was to confirm the trainer + quantization + artifact pipeline works end-to-end, not to produce a competitive score.

### Batch #1 (completed 2026-03-20)

2 Sobol-sampled runs over 11 dimensions. Sliding-window eval (seq_len=2048, stride=256).

| Run | MUON_WD | SCALAR_WD | Iters | Prequant bpb | int6 bpb | Quant gap | int6 size | Sliding bpb | Status |
|-----|---------|-----------|-------|-------------|----------|-----------|-----------|-------------|--------|
| 0000 | 7.96e-5 | 1.33e-5 | 3750 | 1.4248 | 1.4269 | 0.0021 | 13.9 MB | **2.4594** | completed |
| 0001 | 0.0386 | 0.00585 | 2500 | 1.4997 | 1.5069 | 0.0072 | 9.8 MB | N/A (killed@71%) | killed |

**Observations:**
- Run 0000 (low WD) has better bpb but also had 50% more iterations
- Run 0001 (high WD, 485x Muon) produced a **30% smaller artifact** (9.8 vs 13.9 MB) — WD improves weight compressibility
- Both runs have very tight quant gaps (0.0021 and 0.0072) — WD at any level is friendly to quantization
- Comparison is confounded: run 0001 had fewer iterations AND different LRs, so we cannot isolate the WD effect
- No baseline (WD=0) run in this batch — needed to measure actual WD contribution

## Next Steps

1. **One more local iteration** — run 2-4 more Sobol samples including a WD=0 baseline to isolate the WD effect from iteration count / LR confounds.
2. **Decision gate** — if WD shows a clear directional improvement vs the WD=0 baseline on matched iterations, escalate to 8xH100.
3. **Confirmation reruns** — any promising configuration gets 3 seeds on target hardware before claiming a result.
4. **Next branch scout** — batch-size scheduling identified as the most promising next transfer (low risk, synergizes with warmdown). Defer until WD thesis is resolved.

## What Changed

- Added decoupled weight decay knobs to the MetaStack optimizer split:
  - `MUON_WEIGHT_DECAY`
  - `SCALAR_WEIGHT_DECAY`
  - `TOKEN_WEIGHT_DECAY` (default `0`)
  - `HEAD_WEIGHT_DECAY` (default `0`)
- Kept the recipe family otherwise narrow:
  - `int6`
  - `MLP 3x`
  - `selective precision / tied embedding special handling`
  - `sliding-window eval` in the serious lane
  - `no STE` in the default lane
- Added a dedicated search lane for the new knobs under `search_configs/`.
- Added periodic sliding-window progress logging so long GB10 or single-node eval phases expose window-level progress and throughput.

## Why This Should Work

- Parameter Golf rewards post-quant quality, not only pre-quant loss.
- Weight decay is one of the lowest-risk ways to shape weight distributions toward cleaner export behavior.
- The current recipe already has strong LR/warmdown machinery, so LR-tied decoupled decay is a natural extension rather than a branchy architectural change.

## Experimental Protocol

Use the protocol in `docs/parameter_golf_pr_pack/experiment_protocol.md`.

Immediate order:

1. GB10 smoke validation
2. target-hardware calibration
3. warm-start search batch
4. guided search batch
5. confirmation reruns

## Experiment Ledger

See `EXPERIMENT_LEDGER.md` in this folder.

## Reproducibility

Local smoke search:

```bash
source /home/spark-advantage/.venv/bin/activate
python search/run_search.py --config search_configs/metastack_v2_wd_smoke.yaml --max-runs 1
```

Local sliding dry-run:

```bash
source /home/spark-advantage/.venv/bin/activate
python search/run_search.py --config search_configs/metastack_v2_wd_sliding_local.yaml --dry-run
```

Sliding progress logging is controlled by `EVAL_SLIDING_LOG_EVERY` and now defaults to `2048` windows in the serious sliding presets. Older runs from Batch #1 may still show `8192` if they were launched before that preset change.

Remote sliding search:

```bash
python search/run_search.py --config search_configs/metastack_v2_wd_sliding_remote.yaml
```

## Notes For Reviewers

- This branch is intentionally conservative at the mechanism level.
- The novel value is the combination of:
  - a clean Parameter Golf-native search harness
  - an explicit transfer thesis from modded-nanogpt
  - disciplined logging of both successful and failed runs
- If the schedule-side thesis lands, more novel branches such as asymmetric logit rescaling or context-isolation mechanisms can be layered on top later.
