# Nitrust Target Lock — Crawler Mainline (Medusa Delta)
Date: 2026-03-29

## Baseline Contract
Optimization target is crawler-only mainline:
- Canonical launcher: `experiments/Crawler_Leg_1/run.sh`
- Compatibility alias: `experiments/Medusa/run.sh`

Baseline runtime knobs:
- `NGRAM_EVAL_ORDER=0`
- `USE_CRAWLER=1`
- `NUM_FLAT_LAYERS=4`
- `NUM_CRAWLER_LAYERS=1`
- `CRAWLER_LOOPS=4`
- `INST_DIM=32`
- `CRAWLER_QUANT_INT8=1`
- `DELTA_NET_HEADS=0`
- `SKIP_EMA=1`
- `SKIP_GPTQ=1`

## Confirmed Code Seams (train_gpt.py)
- Data shard parse/load: `load_data_shard` and `TokenStream`/`DistributedTokenLoader`.
- Training hot path: `DistributedTokenLoader.next_batch`.
- Eval hot path: `eval_val_sliding` window assembly loop.
- Export hot path: int6 pack/compress path near final export.
- EMA/GPTQ cut points: `SKIP_EMA` and `SKIP_GPTQ` gates in finalization section.

## Nitrust Compatibility Note
Crawler shard files are headered, not raw `u16`; they contain a 256x`i32` prefix:
- `magic=20240520`
- `version=1`
- `num_tokens` in header slot 2

`nitrust-mmap-loader` now supports this format with strict size checks,
while retaining raw `u16` fallback.

## Complexity-Ordered Knockout Plan (NGRAM-Free)
1. NIT-A1: Swap shard reads to Rust mmap path (no model math changes).
2. NIT-A2: Rust batch assembly + pinned host buffer handoff.
3. NIT-B1: Rust sliding-window index builder for eval path.
4. NIT-C1: Rust quant/export pack pipeline.
5. NIT-D1: CUDA graph replay wrapper for fixed-shape training steps.

## Quarantine Rule
DeltaNet remains sandbox-only during this leg (`experiments/Medusa/run_delta_sandbox.sh`).

## Success Gates
- Primary: lower `step_avg` and higher tokens/sec at equal wallclock.
- Quality: `final_int6_roundtrip_exact` and `final_int6_sliding_window_exact`
  within tolerance (`val_bpb` delta <= +0.01 unless explicitly traded for speed).
- Reproducibility: deterministic run logs with fixed seed.
