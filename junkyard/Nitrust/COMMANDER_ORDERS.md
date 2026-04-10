# Nitrust Commander Orders — Crawler-Only Sprint Sequence
Date: 2026-03-29

## Command Intent
Increase end-to-end speed via Rust hardware modules outside crawler internals.
Do not depend on ngram systems for wins.
Bandit is current SOTA reference while crawler-only leg is rebuilt.

## Sprint Queue (In Order)

| Sprint | Modules | Goal | Gate |
|---|---|---|---|
| A | NR-01 + NR-02 | Remove Python data path bottlenecks and overlap H2D transfers | >=10% throughput gain, no metric regression |
| B | NR-03 | Accelerate sliding-window eval infra | >=25% eval wallclock reduction |
| C | NR-04 | Compress/export faster with deterministic pack pipeline | >=2x export speedup, bit-exact roundtrip |
| D | NR-05 | Reduce launch overhead with CUDA graph replay | >=10% train step reduction |
| E | NR-06 | Stabilize topology-level performance | lower p95 step jitter and +3% throughput |
| F | NR-07 | Online parameter tuning | additional >=5% gain over Sprint E |

## Non-Negotiables
1. Every sprint ships with A/B benchmark evidence.
2. No sprint proceeds if parity checks fail.
3. Any speed gain that harms baseline quality beyond tolerance is rejected.

## Benchmark Baseline Spec
Use `experiments/Crawler_Leg_1/run.sh` profile with:
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
- fixed seed and wallclock

## Immediate Next Action
Execute Sprint A/B/C on crawler-only lane:
1. Keep `nitrust-py` import path optional with strict parity checks.
2. Benchmark Rust mmap + pinned batcher on crawler-only ablation grid.
3. Add eval/export Rust path tests only after crawler baseline is stable across two seeds.
