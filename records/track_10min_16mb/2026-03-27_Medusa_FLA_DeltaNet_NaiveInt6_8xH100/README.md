# Medusa: Unstable — DeltaNet Crawler, Frugendorff Continuation

**val_bpb: PENDING** (3-seed mean) | **~9.96MB** | 8xH100 SXM | Successor to PR #990 (ClownCar, 1.1813)

> **Catalyst:** PR #875 (@shalyhinpavel, Pure Neural GDN, 1.0226 BPB) proved that Gated DeltaNet
> is the dominant architecture for this competition. Medusa's DeltaNet integration is directly
> symbiotic: the same `chunk_delta_rule` kernel powering GDN's state updates is active inside
> the Frugendorff crawler topology here. Different architectures, same foundational mechanism.

> **Stability note:** This submission shows significant cross-seed variance (see results table).
> The DeltaNet heads introduce sensitivity not present in ClownCar (variance 0.00015).
> Best seed is a genuine improvement. Research into stabilization is ongoing — Medusa_VII next.

## Results

| Seed | BPB (sliding window) | Size (int6+zstd) | Post-EMA BPB | Steps |
|------|---------------------:|-----------------:|-------------:|------:|
| 42   | **0.8104** ← best | 9.96MB | 0.2519 | 4872 |
| 300  | 0.9578 | 9.97MB | 0.3882 | 4880 |
| 1337 | 1.2269 | 9.96MB | 0.7126 | 4876 |
| **Mean** | **0.9984** | | | |
| **Std dev** | **0.1724** | | | |

## What Changed vs PR #990 (ClownCar)

| Change | Reason |
|--------|--------|
| `DELTA_NET_HEADS=4` | Canonical FLA DeltaNet enabled (vs 0 in ClownCar) |
| `LOOP_AWARE_GPTQ=1` | 2-phase GPTQ calibration: phase 1 collects flat-layer Hessians, phase 2 collects crawler Hessians with quantized-flat activations — better approximation of inference conditions |
| `EMA_START_STEP=4400` + `EMA_DECAY=0.99` | Late-start EMA re-initialized at warmdown onset, fast decay tracks warmdown weights closely |

## Architecture

- **Topology**: 4 flat layers + 1 crawler layer × 4 loops (Frugendorff compression)
- **INST_DIM**: 32 (flow instructions)
- **DeltaNet**: 4 heads, canonical `chunk_delta_rule` from `fla.ops.delta_rule`
- **Quantization**: int6+zstd + CRAWLER_QUANT_INT8=1, loop-aware GPTQ (41 layers)
- **Dims**: XSA_LAST_N=11, BIGRAM_VOCAB_SIZE=2048, ROPE_DIMS=16
- **Schedule**: WARMDOWN_ITERS=2000, SWA_EVERY=50, EMA_START_STEP=4400
- **N-gram eval**: DISABLED (sliding window only)

## Known Issues

The DeltaNet heads introduce cross-seed instability. Investigation identified two causes:
1. **State dtype bug**: `chunk_delta_rule` returns Float32 `new_state` in BF16 training — fixed in follow-on work (Medusa_V: `new_state.to(dtype)`)
2. **Quantization unravel**: DeltaNet weight errors compound through 4 crawler loops — active research area

## Legality

1. No n-gram eval — sliding window only
2. No val data used during training
3. int6 quantization runs inside training wallclock
4. Score-first protocol not applicable (no n-gram cache)

## Reproduce

```bash
SEED=300 bash experiments/Medusa_IV/run.sh
SEED=1337 bash experiments/Medusa_IV/run.sh
SEED=42 bash experiments/Medusa_IV/run.sh
```

8xH100 SXM, 600s training per seed.

## Credits

- **Gated DeltaNet (GDN) — primary catalyst**: @shalyhinpavel (PR #875) — proved GDN is the architecture for this competition at 1.0226 BPB pure neural. Medusa's DeltaNet integration is directly symbiotic: same `chunk_delta_rule` mechanism, applied inside the crawler topology.
- **Canonical DeltaNet kernel**: `fla.ops.delta_rule` (flash-linear-attention)
- **Loop-aware GPTQ**: @newjordan (Medusa series)
- **Frugendorff crawler architecture + flow instructions**: @newjordan (PR #990)
- **FX_Wing_Delta base**: @newjordan
