# GEMM-Hopper Regenerated Adapter B2B: A Documented Negative ML Result with Passing H100 Kernel Proofs

**PR [#2115](https://github.com/openai/parameter-golf/pull/2115) | Non-Record Submission**
**Author:** [@Sacmaj](https://github.com/Sacmaj)
**Lineage:** Follow-up to [`2026-04-18_HopperPersistentSeededLowRankGEMM`](../2026-04-18_HopperPersistentSeededLowRankGEMM/) (val_bpb 1.42375499, validated)
**Hardware:** 1× NVIDIA H100 80GB HBM3 (Runpod) for headline runs; 1× RTX 4090 24GB for local iteration
**Headline:** smart-role regen adapter `val_bpb = 2.05623253` strict / 8.10 MB, vs matched base-only `1.68527241` / 8.32 MB at 297 steps. Adapter hurts.

---

## The Short Version

I tried to extend my predecessor submission, [`HopperPersistentSeededLowRankGEMM`](../2026-04-18_HopperPersistentSeededLowRankGEMM/) (val_bpb 1.42), with three changes at once: **bigger rank** (8 → 16), **tighter ternary regen** (Rademacher → Rule-30 cellular automaton + Achlioptas {1/4, 1/2, 1/4}), and a **real H100 native kernel** (CUTLASS WGMMA / WMMA bridge). The composite went backwards. At matched 297-step runs on a single H100 with seed 1337:

| run | val_bpb (strict int8+zlib) | bytes_total | Δ vs base |
|---|---:|---:|---:|
| smart-role adapter (skip Q/K) | **2.05623253** | 8,099,240 | +0.371 |
| all-role adapter, native bridge | 2.28083634 | 7,439,076 | +0.596 |
| base, `USE_REGEN_ADAPTER=0` | **1.68527241** | 8,323,872 | — |

The adapter consistently hurts quality at matched updates. **This is submitted as a documented negative ML result**, not a leaderboard contender.

The reusable parts are real, though:

- A **zero-byte-artifact adapter primitive**: A and B regenerate from a 64-bit master seed via Rule-30 + Achlioptas; only `(seed, alpha)` lives in `state_dict`. The materialized adapter weights add zero artifact bytes. Bit-for-bit identical between Python NumPy regen (`train_gpt.py`) and the C++ reference (`src/reference.cpp`).
- A working **H100 kernel prototype**: byte-equality Rule-30 (sm_89 ≡ sm_90, aggregate `bbb1498b31d0ecfe`), kernel parity for `wmma_bridge`, `cutlass_base_adapter_epilogue`, `cutlass_fused_v1`, `cutlass_fused_v2` across M ∈ {32, 64, 128, 256, 512, 1024}, 100-rerun ping-pong determinism, 1000-rerun synthetic e2e route stability, SASS HGMMA/UTMA/setmaxreg-family proof. NCU counter proof was waived (`ERR_NVGPUCTRPERM` on the Runpod fixed template); the ≤ 5% adapter-overhead gate was missed (75–103% across M-buckets). The proof logs and standalone runner are not bundled in this PR — they're held in the working tree and can be added on request.
- A **specific compute ask**: ~6 8×H100-hours to ablate `REGEN_RANK ∈ {4, 8, 16, 32}`, `REGEN_ALPHA_INIT ∈ {0.0625, 0.125, 0.25}`, and placement, plus a 3-seed 8×H100 600s gate at the best config. Full rationale in [Compute Request](#compute-request) below.

The point of writing this up at length is the same as [DepthRecurrence](../2026-03-21_DepthRecurrence_MixedPrecisionQuant/): if someone else is thinking about ternary regen-adapters or zero-byte-storage LoRAs, **read this first**. It will save you days.

---

## Table of Contents

1. [Lineage](#lineage)
2. [The Technique](#the-technique)
3. [What Worked (Engineering)](#what-worked-engineering)
4. [What Didn't Work (ML)](#what-didnt-work-ml)
5. [Hopper Kernel Engineering](#hopper-kernel-engineering)
6. [Compute Request](#compute-request)
7. [Reproducing These Results](#reproducing-these-results)
8. [Compliance](#compliance)
9. [Files Included](#files-included)
10. [Acknowledgments](#acknowledgments)

---

## Lineage

The predecessor [`2026-04-18_HopperPersistentSeededLowRankGEMM`](../2026-04-18_HopperPersistentSeededLowRankGEMM/) reached **val_bpb 1.42375499** end-to-end on WSL Ubuntu 24.04 + RTX 4090 with compiled PyTorch 2.11.0+cu128, full validation split, ~3,145 steps in 600 s. It used:

- **Rank 8** seeded low-rank residual projections.
- **Rademacher** seeded factors with **base seed 1729**.
- A single LoRA-style adapter formula `y = xW0^T + alpha * (xA^T)B^T`, three trainable scalars per projection (`alpha`, `adapter_latent`, `adapter_bias`).
- No native CUDA kernel — pure PyTorch GEMMs.

This submission changes three things at once:

| Knob | Predecessor | This submission |
|---|---|---|
| Adapter rank | 8 | 16 |
| Ternary mechanism | Rademacher (base seed 1729) | Rule-30 multi-word CA + Achlioptas {1/4, 1/2, 1/4} |
| Adapter shape | Single LoRA | **B2B** (back-to-back) `Y = X@W^T + alpha*(X@A^T)@B^T` with separate small GEMMs |
| Native kernel | None (pure PyTorch) | CUTLASS WGMMA + WMMA bridge prototype on H100 |

The honest reading: the predecessor *worked*, this attempt at "more aggressive ternary regen + bigger rank + native kernel" went backwards. Investigating *why* is the point of the writeup. The most plausible suspect is that I changed too many things at once — the right next experiment is to back off to rank 8 and Rademacher and re-introduce only the B2B shape and the kernel, isolating which knob actually broke quality. That's a compute question, not a code question, which is why the [Compute Request](#compute-request) section is explicit and specific.

---

## The Technique

### The Decomposition

Each replaced projection in the transformer block computes:

```
Y = X @ W^T + alpha * (X @ A^T) @ B^T
```

- `W` ∈ FP16/BF16, the normal learned base projection.
- `A` ∈ {-1, 0, +1}^(rank × in_features) — frozen ternary, regenerated.
- `B` ∈ {-1, 0, +1}^(out_features × rank) — frozen ternary, regenerated.
- `alpha` is a single trainable scalar per replaced projection.

This is two back-to-back small GEMMs (the "B2B" in the title) glued onto the base GEMM by a scalar.

### Why "Regenerated"

`A` and `B` are **non-persistent buffers** in the PyTorch sense — they live in module memory but are excluded from `state_dict`. At module init, each replaced linear derives its own `(seed_a, seed_b)` pair via SplitMix64 from `(REGEN_MASTER_SEED, layer_idx, role)`, then runs:

1. **Rule-30 multi-word evolution**: 256-word elementary cellular automaton state, initialized from the seed via SplitMix64, evolved for `_RULE30_WARMUP=256` steps before output is sampled. Multi-word avoids the `0xAAAAAAAAAAAAAAAB` attractor a single-word Rule-30 falls into.
2. **Achlioptas ternary unpack**: each pair of CA bits maps to {-1, 0, 0, +1} → {-1, 0, +1} with the {1/4, 1/2, 1/4} distribution Achlioptas's projection theorem requires. Two-bits-per-ternary, packed.

The pure-NumPy implementation in `train_gpt.py` (~150 lines) is **bit-for-bit equivalent** to the C++ reference at `src/reference.cpp::generate_ternary_matrix`. This is verified by aggregate checksum: 4 fixed seeds × 10,000 generations × `{u64_baseline, u32_halves}` variants produce the same `bbb1498b31d0ecfe` aggregate on local sm_89 (RTX 4090) and on H100 sm_90.

The total artifact cost of the adapter is **(seed, alpha) per replaced layer**: 64 + 32 bits. The materialized A and B add **zero bytes** to the saved model.

### Why This Should Work in Theory

- **Achlioptas projections** are a standard tool: random ±1 sparse projections with known concentration bounds, used everywhere from LSH to JL embeddings.
- **Ternary LoRAs** have been explored in the quantization literature; representational ceiling is below FP16 LoRA at the same rank but the storage cost is dramatically lower.
- **The predecessor result** shows seeded low-rank residuals *can* coexist with the rest of the parameter-golf pipeline (Muon + Adam, int8+zlib export, the SP1024 FineWeb tokenizer) without breaking convergence — at val_bpb 1.42, comfortably under the 16 MB cap.

The hypothesis was that a tighter regen mechanism (Rule-30 over Rademacher) plus a doubled rank plus the structural change to B2B should give the model more useful "free" capacity per byte. Empirically it did not.

### Three Knobs

- `REGEN_RANK` — adapter rank. **16** in this submission.
- `REGEN_ALPHA_INIT` — initial value of the per-layer trainable scalar. **0.125** here, inherited from the predecessor without sweeping.
- `REGEN_INCLUDE_ROLES` — which transformer-block linears get the adapter. The smart-role default skips Q/K: `attn_c_v,attn_proj,mlp_fc,mlp_proj`. With this filter, 36 of the 54 candidate linears get the adapter; 18 (Q/K plus tiny linears under `REGEN_MIN_DIM=64`) stay plain `CastedLinear`.

The trainer also has a backend-selection knob, `USE_HOPPER_REGEN_GEMM` ∈ {`off`, `triton`, `native`}. The submitted headline run uses `off` (pure PyTorch math); the all-role native run exercises the H100 native bridge.

---

## What Worked (Engineering)

These are the gates the engineering side of the submission cleared. None of these are the headline ML number — they're correctness and reproducibility evidence.

**Rule-30 byte equality across implementations.** The same aggregate checksum `bbb1498b31d0ecfe` over 4 fixed seeds × 10,000 generations × `{u64_baseline, u32_halves}` variants on **local sm_89** (RTX 4090) and **H100 sm_90**. The Python NumPy regen, the C++ CPU reference (`src/reference.cpp`), and the CUDA kernel (`src/generator_kernels.cu` in the working tree) all produce the same bits. This means a model serialized on one machine regenerates the same A, B on any other machine that runs this code.

**H100 kernel parity, M ∈ {32, 64, 128, 256, 512, 1024}.** The standalone `gemm_hopper_runner` compares five backends — `wmma_bridge`, `isolated_base_stub`, `cutlass_base_adapter_epilogue`, `cutlass_fused_v1`, `cutlass_fused_v2` — and they all parity-pass against the CPU reference at K=N=4096, r=16 across every M-bucket. Logs at `runpod_kernel_parity.log` and `runpod_kernel_proofs.log` (in the working tree).

**Determinism gates.** 100-rerun ping-pong determinism across M ∈ {32, 64, 128} (`runpod_pingpong_100rerun.log`). 1000-rerun synthetic e2e route stability with 256 routes (`runpod_e2e_synthetic_1000rerun.log`).

**SASS family inspection.** The CUTLASS path produces HGMMA, UTMA (TMA), USETMAXREG, and async-fence family instructions on H100 (`profiles/sm90/gemm_hopper_runner.sass` in the working tree). This is real Hopper-family code, not a forced fallback.

**Exact int8+zlib roundtrip.** Every training log shows `final_int8_zlib_roundtrip_exact` at 8 decimals, matching the in-memory `final_int8_zlib_roundtrip` — confirming the quantization + zlib + de-quant + load + regen-buffers cycle reproduces the trained model losslessly within int8 quantization. The reported `val_bpb` numbers in this README are the strict roundtrip values, not the optimistic in-memory ones.

**Two real caveats.** (1) The NCU counter proof (occupancy, register usage, achieved-vs-peak FLOPs) is **waived** — the Runpod fixed template denies `nvidia-cuprofile` GPU performance counters with `ERR_NVGPUCTRPERM`. The waiver applies only to the WMMA bridge currently used by the trainer; final fused-kernel profiler proof is still open. (2) The ≤ 5% adapter-overhead gate is **missed**: `cutlass_fused_v2` shows 75.26% / 82.93% / 102.63% / 100.16% / 99.20% / 100.63% overhead across M ∈ {32..1024} (`runpod_perf.log`). Synthetic TFLOPS are only 9.67–14.91 — Tier-2 perf remains open until the production CUTLASS WGMMA mainloop integration lands.

The proof logs and standalone runner sources are not in this PR's 15-file list because they're not transitive dependencies of the trainer's native bridge. They're held in the working tree and can be added to the PR if a reviewer asks.

---

## What Didn't Work (ML)

This is the headline section. Every number traces to a log file in the PR.

### The Matched-Step Ablation

Same hardware (1× H100 80GB HBM3 on Runpod), same data (SP1024 FineWeb, 1 train shard for the bench), same seed (`1337`), same `TRAIN_BATCH_TOKENS=524288`, same `TRAIN_SEQ_LEN=1024`, same `EVAL_MAX_TOKENS=524288` capped validation, same fixed `ITERATIONS=297`, same `MAX_WALLCLOCK_SECONDS=0` (step-cap, not wall-cap). The only thing varying is the adapter configuration:

| run | adapter | layers w/ regen | strict val_bpb | bytes_total | wallclock | log |
|---|---|---:|---:|---:|---:|---|
| smart-role (skip Q/K), backend off | `USE_REGEN_ADAPTER=1`, `REGEN_INCLUDE_ROLES=attn_c_v,attn_proj,mlp_fc,mlp_proj`, `USE_HOPPER_REGEN_GEMM=off` | 36 | **2.05623253** | 8,099,240 | 299.2 s | `train_runpod_1h100_smart_roles_ablation_297step.log` |
| all-role, native bridge | `USE_REGEN_ADAPTER=1`, `REGEN_INCLUDE_ROLES=` (all), `USE_HOPPER_REGEN_GEMM=native` | 54 | 2.28083634 | 7,439,076 | 601.5 s | `train_runpod_1h100_native_10m.log` |
| base (technique disabled) | `USE_REGEN_ADAPTER=0` | 0 | **1.68527241** | 8,323,872 | 261.2 s | `train_runpod_1h100_base_ablation_297step.log` |

Skipping Q/K helps substantially (`2.28 → 2.06`, a 0.22 bpb improvement) but **the filtered adapter still loses by 0.371 bpb to base-only**. Whatever benefit the regen-adapter adds in expressivity is more than overwhelmed by some other cost — and the cost is consistent (worse, not noisy-worse) across the two adapter configurations.

### A Second Tax: Wallclock

Compare wallclock at fixed step count:

| Run | wallclock (s) | step_avg (ms) | Notes |
|---|---:|---:|---|
| base | 261.2 | ~880 | technique disabled |
| smart-role (off backend) | 299.2 | ~1008 | adapter on, native bridge OFF, 14% slower than base |
| all-role native bridge | 601.5 | ~2025 | adapter on, native bridge ON, **2.3× slower** than base |

The native bridge is currently slower than the pure-PyTorch path for this trainer. The standalone-runner perf result (75–103% overhead vs base GEMM) is consistent with that — the bridge isn't actually fused with the base GEMM yet, and the trainer-path call requires `torch.compile(..., fullgraph=False)` so Dynamo can graph-break around the pybind extension. **At a fixed wallclock budget rather than a fixed step count, the adapter penalty would be even larger**, because the slower step rate means fewer steps in the same 600 s.

### Hypotheses for Why It Hurts

These are candidates, not certainty. The point of the [Compute Request](#compute-request) is to test them.

**1. Ternary representational floor.** A ∈ {-1, 0, +1} caps adapter expressivity well below an FP16 LoRA at the same rank. The predecessor used Rademacher (also ternary in spirit, but with base seed 1729 chosen carefully) at rank 8 and worked. Rule-30 produces a different statistical structure than Rademacher even with matched ternary marginals; possibly the autocorrelation of CA-evolved bits hurts the projection's "approximately orthogonal" property that Achlioptas relies on.

**2. Training-step tax.** Each step now backprops through `alpha * (X@A^T)@B^T` for every replaced linear. At 17M params and 297 steps that's a real fraction of the total update budget. The smart-role config at 36 replaced linears spends ~14% more wallclock per step than base (above), which translates to fewer effective gradient updates of the *base* W's at the same step count — and the base W's are what's actually doing the work.

**3. Q/K sensitivity.** All-role being worse than smart-role (which skips Q/K) suggests the adapter especially perturbs attention scores, where small noise has outsized downstream effect via the softmax. Smart-role helps a lot (+0.22 bpb improvement) but doesn't recover to base.

**4. Rank/alpha mismatch.** Rank 16 and alpha_init 0.125 were inherited from the predecessor without sweeping. The predecessor used rank 8 — doubling the rank doubles the adapter contribution magnitude at fixed alpha, which may overwhelm the base path early in training. **This is the most testable hypothesis**, and the first thing the [Compute Request](#compute-request) asks for.

**5. Native bridge is slow.** Independent of quality, `USE_HOPPER_REGEN_GEMM=native` is 2× slower wallclock at 297 steps than `=off`. The WMMA bridge isn't fused with the base GEMM, and the Dynamo graph-break around the pybind extension probably costs more than the kernel saves at these tiny adapter shapes (rank 16). The bridge is correctness-validated; it is not a perf win in its current form.

---

## Hopper Kernel Engineering

The "B2B kernel prototype" in the PR title refers to the H100-side work: a route runtime that dispatches small-K micro-GEMMs through one of five backends, with a specific design contract for fusing the base GEMM and the rank-r adapter epilogue.

### Architecture

The route runtime takes M ∈ {32, 64, 128, 256, 512, 1024} buckets and dispatches to either a **ping-pong** schedule (M ≤ 128, two warps alternating producer/consumer with mbarriers) or a **cooperative** schedule (M ≥ 256, multi-warp producer/consumer with TMA). The standalone runner `gemm_hopper_runner --compare-backends` benchmarks five candidates per M-bucket:

- `wmma_bridge` — current trainer path, separate base GEMM + adapter WMMA, used by the native PyTorch extension.
- `isolated_base_stub` — base GEMM only, no adapter, baseline for overhead measurement.
- `cutlass_base_adapter_epilogue` — CUTLASS WGMMA mainloop with adapter as a separate epilogue pass.
- `cutlass_fused_v1` — first attempt at a single-launch fused base+adapter; large shapes regress to v2's path to avoid timeout (recorded as `large_shape_uses_fused_v2_tile`).
- `cutlass_fused_v2` — one-launch WMMA fused candidate; parity-passes everywhere but doesn't hit the perf gate.

### Three Contract Corrections (Read These Before Touching H100 Code)

These surfaced during the work, all three from real H100 PTX/SASS evidence vs the original design draft. They're enforced throughout the source and tests:

1. **WGMMA accumulator is F32 only.** Never BF16 — that path doesn't exist in the H100 PTX ISA. Cast to BF16 in the epilogue before TMA store.
2. **H100 SXM5 dense FP8 peak = 1979 TFLOPS, not 989.** The 989 TFLOPS number people cite is the BF16 peak. With FP8 inputs, F32 accum, the peak doubles. Compute/memory crossover for K=N=4096 r=16 lands at M ≈ 166.
3. **`KernelTmaWarpSpecializedCooperative` requires TileM ≥ 128.** M ∈ {32, 64} buckets cannot use the cooperative schedule — they must dispatch to ping-pong. Build-time CUTLASS asserts catch this if violated.

`docs/design-corrections.md` (in the working tree, not in this PR) tracks the rationale and which files enforce each correction.

### Status

| Gate | Result | Evidence |
|---|---|---|
| Kernel parity (M ∈ {32..1024}, all 5 backends) | ✓ pass on H100 | `runpod_kernel_parity.log`, `runpod_kernel_proofs.log` |
| Rule-30 byte equality (sm_89 == sm_90) | ✓ pass, `bbb1498b31d0ecfe` | `rule30_checksum_local.log` (local), `runpod_kernel_proofs.log` (H100) |
| 100-rerun ping-pong determinism | ✓ pass for M ∈ {32, 64, 128} | `runpod_pingpong_100rerun.log` |
| 1000-rerun synthetic e2e route stability | ✓ pass for 256 routes | `runpod_e2e_synthetic_1000rerun.log` |
| SASS HGMMA / UTMA / setmaxreg / async-fence families | ✓ pass | `profiles/sm90/gemm_hopper_runner.sass` |
| `cutlass_fused_v2` ≤ 5% adapter-overhead gate | ✗ miss (75–103% across M-buckets) | `runpod_perf.log` |
| Rule-30 P1 throughput gate (256 cells/cycle/warp) | ✗ miss (107.789 best for `u32_halves`); waived per ROADMAP | `generator_device_bench_local.log` |
| NCU counter proof (occupancy/registers/achieved-vs-peak) | ✗ waived (ERR_NVGPUCTRPERM on Runpod fixed template) | n/a |

### What's Deferred

The production target — a **single-launch CUTLASS WGMMA mainloop with in-kernel Rule-30 regen and adapter epilogue, ≤ 5% overhead vs base** — is open. The current `cutlass_fused_v2` path uses a bounded WMMA tile rather than the full WGMMA mainloop. Closing this would let the native bridge actually be a perf win on the trainer path, which would in turn change the wallclock-tax calculation in [What Didn't Work (ML)](#what-didnt-work-ml).

These proof logs (`runpod_kernel_*.log`, `runpod_perf.log`, `runpod_pingpong_100rerun.log`, `runpod_e2e_synthetic_1000rerun.log`, `triton_wrapper_runpod_parity.log`) and the standalone runner sources (`src/cutlass_fused_sm90.cu`, `src/persistent_kernel_sm90.cu`, `src/runtime.cpp`, `src/runner_main.cpp`, the rest of `tests/`) are **not bundled in this PR's 15 files** because they are not transitive dependencies of the trainer's native bridge. The PR ships only what `train_gpt.py` actually compiles or imports against. If a reviewer wants the full kernel evidence checked into the submission, I'm happy to add it.

---

## Compute Request

**Submitting non-record because I cannot self-fund the 8×H100 leaderboard regime.** Local hardware is one RTX 4090 (24 GB) and hourly Runpod 1×H100 rentals; the leaderboard validation gate needs 8×H100 SXM at ~$20/hr, plus iteration overhead.

The 297-step 1×H100 ablation above is a convergence signal at partial-batch step count, not the production regime. To turn this into a publishable confirmation-or-refutation, the experiments below are needed in roughly this order of expected information gain:

| # | Experiment | Configs | Seeds | Runs | Wallclock |
|---|---|---|---:|---:|---:|
| 1 | Rank sweep at smart-role placement | `REGEN_RANK ∈ {4, 8, 16, 32}` | 3 | 12 | ~120 min 8×H100 |
| 2 | Alpha sweep at best rank from #1 | `REGEN_ALPHA_INIT ∈ {0.0625, 0.125, 0.25}` | 3 | 9 | ~90 min 8×H100 |
| 3 | Placement sweep at best rank/alpha | `all` / `smart-role` / `mlp-only` / `attn-only` | 3 | 12 | ~120 min 8×H100 |
| 4 | Best-config 8×H100 600s production gate | best of #1-#3 | 3 | 3 | ~30 min 8×H100 |
| | **Total** | | | **36 runs** | **~6 hours 8×H100** |

At Runpod's 8×H100 SXM rate of ~$20/hr that's roughly **$120 of compute**. Any formal compute-credit request would go through OpenAI's grant form; this PR is the technical justification.

**What a positive result would mean.** A zero-byte-artifact LoRA primitive, deterministically regenerable from a 64-bit seed, that closes some or all of the +0.37 bpb gap at the production step budget. Even a partial close justifies the kernel work as a primitive the next contributor can build on (the CUTLASS WGMMA mainloop integration is open and well-specified — see [Hopper Kernel Engineering](#hopper-kernel-engineering)).

**What a negative result would mean.** A published-quality refutation that ternary regen-adapters at this rank/alpha don't help in the 10-minute budget. That saves the next would-be contributor four days, the way [DepthRecurrence](../2026-03-21_DepthRecurrence_MixedPrecisionQuant/) saved me.

---

## Reproducing These Results

Use the official Parameter Golf Runpod template (id `y5cejece4j`) and SP1024 cached FineWeb:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
cd records/track_non_record_16mb/2026-04-28_HopperRegenAdapterB2B
```

**Headline run — smart-role adapter, 297 steps, 1×H100** (`val_bpb=2.05623253`):

```bash
RUN_ID=hopper_regen_smart_roles_ablation_297step \
SEED=1337 \
USE_REGEN_ADAPTER=1 \
USE_HOPPER_REGEN_GEMM=off \
REGEN_INCLUDE_ROLES=attn_c_v,attn_proj,mlp_fc,mlp_proj \
ITERATIONS=297 \
MAX_WALLCLOCK_SECONDS=0 \
EVAL_MAX_TOKENS=524288 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**Matched base-only control, 297 steps, 1×H100** (`val_bpb=1.68527241`):

```bash
RUN_ID=hopper_base_1h100_ablation_297step \
SEED=1337 \
USE_REGEN_ADAPTER=0 \
USE_HOPPER_REGEN_GEMM=off \
ITERATIONS=297 \
MAX_WALLCLOCK_SECONDS=0 \
EVAL_MAX_TOKENS=524288 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**All-role native bridge, 297 steps, 1×H100** (`val_bpb=2.28083634`, exercises the SM90 native extension):

```bash
RUN_ID=hopper_regen_all_role_native_10m \
SEED=1337 \
USE_REGEN_ADAPTER=1 \
USE_HOPPER_REGEN_GEMM=native \
ITERATIONS=297 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_MAX_TOKENS=524288 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

The native bridge requires `torch.compile(..., fullgraph=False)`, which `train_gpt.py` selects automatically when `USE_HOPPER_REGEN_GEMM=native`.

For predecessor context, see [`../2026-04-18_HopperPersistentSeededLowRankGEMM/README.md`](../2026-04-18_HopperPersistentSeededLowRankGEMM/README.md).

---

## Compliance

- [x] Submitted to `records/track_non_record_16mb/` as a documented negative result. No SOTA claim.
- [x] No 8×H100 leaderboard claim. Headline evidence is 1×H100 capped-validation only.
- [x] No p-value claim. Three single-seed 1×H100 ablations; statistical significance gating is for record submissions and would require the 3-seed 8×H100 runs in [Compute Request](#compute-request).
- [x] Best submitted regen artifact is under the 16,000,000-byte decimal cap: **8,099,240 bytes**.
- [x] `train_gpt.py` is **1440 lines**, under the 1500-line hard cap.
- [x] Validation is not accessed during training. Logs report post-training capped validation only. No paid-prefix tricks; validation tokens are not compressed into the artifact.
- [x] No tokenizer or dataset scoring change is claimed (SP1024 FineWeb root pipeline unchanged).
- [x] Folder is additive — no edits to root files or other record folders.
- [x] Predecessor is cited but not modified.

---

## Files Included

This PR ships exactly the files the trainer compiles or imports against, plus the three ablation logs and metadata:

```
2026-04-28_HopperRegenAdapterB2B/
├── README.md                                              this writeup
├── submission.json                                         metadata + comparisons + lineage + compute_request
├── requirements.txt                                        root-requirements copy
├── train_gpt.py                                            1440 lines; RegenAdapterLinear + Muon + int8+zlib export
├── python/
│   ├── __init__.py
│   ├── gemm_hopper.py                                      Triton wrapper + native-extension dispatch
│   ├── gemm_hopper_native.cpp                              pybind11 binding to native CUDA kernel
│   └── gemm_hopper_native_kernel.cu                        seed-based SM90-gated adapter kernel
├── src/
│   └── reference.cpp                                       C++ Rule-30 + ternary regen, byte-equivalent to NumPy path
├── include/gemm_hopper/
│   ├── reference.hpp                                       CPU reference declarations
│   ├── generator.hpp                                       Rule-30 stepping primitives
│   └── ternary.hpp                                         2-bit-packed → ternary expander
├── train_runpod_1h100_smart_roles_ablation_297step.log     headline (val_bpb 2.05623253)
├── train_runpod_1h100_native_10m.log                       all-role native bridge (val_bpb 2.28083634)
└── train_runpod_1h100_base_ablation_297step.log            matched base-only control (val_bpb 1.68527241)
```

The C++ headers and `src/reference.cpp` are present because the native PyTorch extension at `python/gemm_hopper_native.cpp` compiles against them at install time.

**Not bundled** (held in the working tree, available on request):

- The standalone runner sources (`src/cutlass_fused_sm90.cu`, `src/persistent_kernel_sm90.cu`, `src/runtime.cpp`, `src/runner_main.cpp`, `src/generator_kernels.cu`, `src/micro_gemm_sm90.cu`).
- `CMakeLists.txt`, `cutlass_commit.txt` (CUTLASS pinned to `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`).
- The 30+ `scripts/run_runpod_*.sh` reproducers and 10 C++ unit tests under `tests/`.
- The `docs/` design-and-corrections tree (`architecture.md`, `design-corrections.md`, `route-specialization.md`, `risks.md`, etc.).
- The kernel proof logs (`runpod_kernel_*.log`, `runpod_perf.log`, `runpod_pingpong_100rerun.log`, `runpod_e2e_synthetic_1000rerun.log`, `triton_wrapper_runpod_parity.log`).
- The `.int8.ptz` quantized model checkpoints from each ablation.
- Internal-process docs (`AGENTS.md`, `ROADMAP.md`, `TASKS.md`).

If a reviewer wants the full kernel evidence in-tree, ping me on the PR and I'll add it.

---

## Acknowledgments

- **Predecessor submission**: [`2026-04-18_HopperPersistentSeededLowRankGEMM`](../2026-04-18_HopperPersistentSeededLowRankGEMM/) for the seeded-LoRA baseline that motivated this attempt.
- **[DepthRecurrence](../2026-03-21_DepthRecurrence_MixedPrecisionQuant/)** by Evangeline Kamin: for showing how to write a thorough negative-results submission and for the explicit "save the next contributor four days" framing.
- **modded-nanogpt**: Muon optimizer and the trainer scaffolding lineage. See `THIRD_PARTY_NOTICES.md` at repo root.
- **NVIDIA CUTLASS** (pinned to `f3fde58`): the CUTLASS WGMMA / TMA primitives the H100 kernel work builds on.
- **Achlioptas (2003)**: the {1/4, 1/2, 1/4} ternary projection theorem the regen mechanism relies on.
- **OpenAI / Will DePue**: for the Parameter Golf challenge, the Runpod template that made the 1×H100 evidence runs cheap, and the explicit invitation to submit non-records — including negative results — to `track_non_record_16mb/`.
