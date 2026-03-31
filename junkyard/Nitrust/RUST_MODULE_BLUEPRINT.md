# Nitrust Commander Blueprint — Rust Hardware Modules (Outside Crawler)
Date: 2026-03-27

## Scope
This plan targets speedups outside crawler architecture logic:
1. Host-to-device pipeline
2. Runtime orchestration overhead
3. Export/eval infrastructure
4. Hardware telemetry and tuning

NGRAM is explicitly out of scope for this phase.

## North-Star Metrics
1. `step_avg_ms` reduction at fixed model config
2. tokens/sec increase (train + eval)
3. no regression in `final_int6_roundtrip_exact` and `final_int6_sliding_window_exact`
4. stable artifact bytes and deterministic reproducibility

---

## Module Stack (Ordered by Complexity)

| ID | Complexity | Module | Primary Job | Target Gain | Integration |
|---|---:|---|---|---|---|
| NR-01 | 1 | `nitrust-mmap-loader` | Zero-copy shard reads + lock-free prefetch ring on CPU | +5% to +12% step throughput | Python extension (`pyo3`) replacing current Python shard iteration |
| NR-02 | 2 | `nitrust-pinned-batcher` | Build pinned host batches and async H2D staging | +6% to +15% step throughput | Called before each forward pass; returns CUDA-ready tensors |
| NR-03 | 2 | `nitrust-window-engine` | Sliding-window index generation + byte-count LUT acceleration | +20% to +40% eval wallclock | Eval path only; no model math changes |
| NR-04 | 3 | `nitrust-quantpack` | SIMD int6/int8 packing + parallel zstd pipeline | 2x to 5x faster export step | Replaces Python-side quant blob packaging |
| NR-05 | 4 | `nitrust-cudagraph-runner` | Static-shape step replay and launch amortization | +10% to +25% step throughput | Orchestrator wrapper around train/eval step calls |
| NR-06 | 5 | `nitrust-affinity` | NUMA/core pinning policy + dataloader/comm thread affinity | +3% to +10% throughput stability | Runtime bootstrap module |
| NR-07 | 6 | `nitrust-autotune` | Online tuning for batching/prefetch/chunk sizes | +5% to +15% over static config | Consumes telemetry, writes tuned profile |

---

## Module Contracts

### NR-01 `nitrust-mmap-loader`
- Inputs: dataset shard glob, sequence length, batch-token target
- Outputs: contiguous token spans (CPU), deterministic with seed
- Hard requirements:
  - no Python data parsing in hot path
  - bounded memory ring buffer
  - shard rollover without stalls

### NR-02 `nitrust-pinned-batcher`
- Inputs: token spans from NR-01
- Outputs: pinned host buffers + async transfer handles
- Hard requirements:
  - overlap copy with compute
  - configurable prefetch depth
  - zero realloc in steady state

### NR-03 `nitrust-window-engine`
- Inputs: val token buffer, stride, sequence length
- Outputs: precomputed windows and scoring metadata
- Hard requirements:
  - deterministic window partitioning across ranks
  - no Python loops for window bookkeeping

### NR-04 `nitrust-quantpack`
- Inputs: quantized tensors and metadata
- Outputs: final `.ptz` blob and size report
- Hard requirements:
  - bit-exact roundtrip checks
  - parallel compression pipeline

### NR-05 `nitrust-cudagraph-runner`
- Inputs: static-shape step function + tensor buffers
- Outputs: replay handle for train/eval loops
- Hard requirements:
  - graph-safe memory ownership
  - fallback path when shape changes

### NR-06 `nitrust-affinity`
- Inputs: machine topology (CPU sockets, GPU mapping)
- Outputs: pinning policy for workers/threads
- Hard requirements:
  - explicit CPU set management
  - no cross-NUMA batch assembly

### NR-07 `nitrust-autotune`
- Inputs: live telemetry stream
- Outputs: tuned config profile (`json`)
- Hard requirements:
  - bounded exploration budget
  - rollback to stable profile on regressions

---

## Build and Integration Shape

Proposed workspace:
- `Nitrust/rust/Cargo.toml` (workspace)
- `Nitrust/rust/crates/nitrust-mmap-loader`
- `Nitrust/rust/crates/nitrust-pinned-batcher`
- `Nitrust/rust/crates/nitrust-window-engine`
- `Nitrust/rust/crates/nitrust-quantpack`
- `Nitrust/rust/crates/nitrust-cudagraph-runner`
- `Nitrust/rust/crates/nitrust-affinity`
- `Nitrust/rust/crates/nitrust-autotune`
- `Nitrust/rust/crates/nitrust-py` (`pyo3` bridge)

Python boundary rule:
- Python keeps model definition and optimizer logic.
- Rust owns high-frequency orchestration/data/export hot paths.

---

## Commander Rollout Order

1. NR-01 `nitrust-mmap-loader`
2. NR-02 `nitrust-pinned-batcher`
3. NR-03 `nitrust-window-engine`
4. NR-04 `nitrust-quantpack`
5. NR-05 `nitrust-cudagraph-runner`
6. NR-06 `nitrust-affinity`
7. NR-07 `nitrust-autotune`

## Acceptance Gates Per Stage

For each stage:
1. Pass deterministic data/metric parity checks against baseline.
2. Show isolated speed gain in A/B run with fixed seed/config.
3. Keep model metrics within tolerance (`val_bpb` delta <= +0.01 unless intentionally trading for speed).
4. Record benchmark in Nitrust changelog before moving to next stage.

## First Execution Ticket

Start with NR-01 + NR-02 together as Sprint A:
- Deliverable: Rust-backed dataloader + pinned batcher via `pyo3`.
- Exit criteria: at least 10% train throughput improvement on Medusa baseline config with NGRAM disabled.
