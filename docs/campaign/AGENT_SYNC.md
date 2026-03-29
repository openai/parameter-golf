# Agent Sync

Date: 2026-03-29

## Current Objective

Build a portable frontier base, then add novelty. Do not replicate old merged #1 (PR #549) verbatim.

Session 04 is closed (Delta 1 failed, Delta 2 neutral). Session 05 strategy has been revised
based on competitive landscape analysis (2026-03-29 17:00 UTC+2).

### Competitive Landscape (2026-03-29)

The validated merged #1 is still PR #549 at `1.1194` BPB (with TTT).
But the open frontier has moved:

- **PR #1060**: `1.1122` BPB, 3-seed mean, no TTT — coprime-stride loader + full Hessian GPTQ + XSA-all. Most credible new claim.
- **PR #1072**: `1.117` BPB, 1-seed only — fused Triton MLP (70ms/step) + online Hessian GPTQ + parallel Muon. Interesting but unconfirmed.
- **PR #1077**: `1.1130` BPB, 3-seed — looks template-like, internally inconsistent. Treat with caution.
- **PR #1070**: `1.1190` BPB — clean PR #549 reproduction, not frontier.

Key insight: every competitive submission shares the same architectural backbone (11L U-Net, GQA, LeakyReLU², XSA, VE128, BigramHash). Differentiation is in throughput, quantization, and data loading. TTT is losing ground — top unvalidated claims beat #1 without it.

### Revised Session 05 Plan (3 phases)

**Phase 1 — Throughput (FA3 port)**
- Port anchor from SDPA to direct `flash_attn_interface` on NGC 25.02 container
- Microbenchmark shows 11.44x kernel speedup; full training impact TBD
- Saved Pegasus FA3 container now exists at `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`
- First full `8xH100` run on that saved-container path FAILED to beat the anchor
- Goal: close the `91.37 ms → ~70-80 ms/step` gap, gain extra training steps

**Phase 2 — Quantization upgrade (Full Hessian GPTQ)**
- Replace current int6+zstd with full Hessian GPTQ (Cholesky error compensation)
- PRs #1060 and #1072 both use this; proven `0.003-0.007 BPB` gain over GPTQ-lite
- Also fixes the artifact compression issue seen in Session 04 Delta 1

**Phase 3 — Novelty**
- Only after phases 1-2 give a competitive `1.12x` base
- Candidate areas: fused Triton kernels, loader optimization, or novel contribution from XAI/RFN background
- TTT is parked — revisit only if needed after phases 1-2

## In Scope

- FA3 port of anchor attention on Pegasus saved NGC 25.02 FA3 container
- Full Hessian GPTQ implementation to replace int6+zstd
- Throughput optimization to maximize training steps in 600s
- Building a modular frontier base, not hard-binding to one historical winner
- Verified Pegasus `8xH100` Slurm-native launch path
- Preserved launcher, artifact, and metric logging discipline

## Out Of Scope

- Verbatim replication of PR #549
- TTT as the center of the plan (parked, optional later)
- More Session 04 micro-deltas
- Treating RFN as the mainline strategy
- RunPod budget except for final validation
- Bundling unrelated changes into one unattributable run

## Current Hardware Stance

- Parity target: Pegasus `8xH100` on one node
- Active development target: Pegasus `8xH100`
- Fallback target: Pegasus `A100-80GB` only when H100 access is blocked
- Current measured evidence tiers:
  - `A100-80GB`: solid development evidence
  - `1xH100`: early parity-adjacent evidence
  - `8xH100`: operationally verified baseline evidence

## Status Snapshot

- Pegasus operator path: confirmed working
- A100 smoke run: complete
- A100 `600s` baseline run: complete
- A100 `600s` `LowerLR` comparison: complete
- A100 `600s` baseline seed-42 reproducibility run: complete
- A100 `600s` warmdown-only variant: complete
- `1xH100` `600s` root baseline: complete
- `8xH100` `600s` root baseline: complete
- Session 03 pre-TTT anchor run: complete
- Current best measured A100 result: root baseline (`val_bpb=1.37140771`)
- Current best measured H100 result: `1xH100` root baseline (`val_bpb=1.30594735`)
- Current best measured `8xH100` baseline/reference: root baseline (`val_bpb=1.23368511`)
- Current best measured `8xH100` competition result: Session 03 anchor sliding s64 (`val_bpb=1.12904446`)
- Baseline seed spread on A100 is small (`+0.00319322` BPB from seed `1337` to seed `42`)
- `8xH100` launch via `torchrun --standalone` is blocked by rendezvous timeout on `serv-3342`
- `8xH100` launch via Slurm-native `srun` works on `serv-3342`
- Session 05 audit artifact: complete (`docs/campaign/artifacts/05_ttt_correctness_audit.md`)
- FA3 isolated attention microbenchmark: complete
- Saved FA3 Pegasus container build: complete
- `1xH100` FA3 smoke: complete
- `8xH100` FA3 timing run on saved `25.02` container: complete
- Stock NGC `25.02` + `--no-deps` FA3 install path: rejected (PyTorch ABI mismatch)
- `8xH100` FA3 runs must include `--nodes=1`
- Immediate next deliverable: decide whether to pursue FA3 on a vendor-tuned NGC runtime or pivot to Phase 2

## Canonical Workspaces

- Local repo: `/home/amay/Work/parameter-golf`
- Remote repo: `/netscratch/$USER/parameter-golf`

Use `git clone` and `git pull` as the default remote sync path.
Use `rsync` only when local uncommitted changes need to be pushed quickly.

## Latest Measured Results

Date: 2026-03-27
Node: `serv-3333`
GPU: `NVIDIA A100-SXM4-80GB`
Run: `a100_baseline_smoke`

Measured outputs:

- Train setup: `1` shard, `200` iterations, `TRAIN_BATCH_TOKENS=65536`
- `amp_dtype: bf16`
- Step average at finish: `154.57 ms`
- Pre-roundtrip eval: `val_loss=3.6186`, `val_bpb=2.1432`
- Post-roundtrip exact eval: `val_loss=3.67022861`, `val_bpb=2.17371612`
- Post-roundtrip eval time: `250881 ms`
- Peak memory: `1548 MiB allocated`, `1566 MiB reserved`
- Total submission size `int8+zlib`: `7066088` bytes

Interpretation:

- Pegasus execution path is working end to end on available hardware.
- Artifact size is comfortably below the `16,000,000` byte cap.
- This run is development evidence only, not H100-equivalent validation.

Date: 2026-03-28
Node: `serv-3333`
GPU: `NVIDIA A100-SXM4-80GB`
Run: `a100_baseline_600s`

Measured outputs:

- Train setup: `10` shards, `600s` wallclock cap
- `amp_dtype: bf16`
- Stopped at `907` steps in `600119 ms`
- Pre-roundtrip eval: `val_loss=2.3117`, `val_bpb=1.3691`
- Post-roundtrip exact eval: `val_loss=2.31556447`, `val_bpb=1.37140771`
- Post-roundtrip eval time: `22204 ms`
- Peak memory: `10253 MiB allocated`, `10578 MiB reserved`
- Total submission size `int8+zlib`: `12046627` bytes

Date: 2026-03-28
Node: `serv-3333`
GPU: `NVIDIA A100-SXM4-80GB`
Run: `a100_lowerlr_600s`

Measured outputs:

- Train setup: `10` shards, `600s` wallclock cap
- `amp_dtype: bf16`
- Stopped at `908` steps in `600185 ms`
- Pre-roundtrip eval: `val_loss=2.3142`, `val_bpb=1.3706`
- Post-roundtrip exact eval: `val_loss=2.32600988`, `val_bpb=1.37759407`
- Post-roundtrip eval time: `22206 ms`
- Peak memory: `10253 MiB allocated`, `10530 MiB reserved`
- Total submission size `int8+zlib`: `10723611` bytes

Comparison:

- On this 1xA100 600s setup, `LowerLR` is worse than the root baseline by `+0.00618636` BPB post-roundtrip.
- `LowerLR` does reduce artifact size by `1323016` bytes, but size was not the limiting factor here.
- For grant evidence, the useful conclusion is that the operator path is reproducible and controlled comparisons are already discriminating between variants.

Date: 2026-03-28
Node: `serv-3338`
GPU: `NVIDIA A100-SXM4-80GB`
Run: `a100_baseline_seed42_600s`

Measured outputs:

- Train setup: `10` shards, `600s` wallclock cap
- `amp_dtype: bf16`
- Stopped at `900` steps in `600537 ms`
- Pre-roundtrip eval: `val_loss=2.3165`, `val_bpb=1.3720`
- Post-roundtrip exact eval: `val_loss=2.32095610`, `val_bpb=1.37460093`
- Post-roundtrip eval time: `22483 ms`
- Peak memory: `10253 MiB allocated`, `10578 MiB reserved`
- Total submission size `int8+zlib`: `12018778` bytes

Reproducibility read:

- Baseline seed `42` is worse than baseline seed `1337` by `+0.00319322` BPB post-roundtrip.
- Step time and memory are effectively unchanged across baseline seeds.
- This is strong enough to claim the baseline behavior is reproducible on Pegasus `A100-80GB`.

Date: 2026-03-28
Node: `serv-3338`
GPU: `NVIDIA A100-SXM4-80GB`
Run: `a100_warmdown3600_600s`

Measured outputs:

- Train setup: `10` shards, `600s` wallclock cap, `WARMDOWN_ITERS=3600`
- `amp_dtype: bf16`
- Stopped at `903` steps in `600661 ms`
- Pre-roundtrip eval: `val_loss=2.3568`, `val_bpb=1.3958`
- Post-roundtrip exact eval: `val_loss=2.38171775`, `val_bpb=1.41058741`
- Post-roundtrip eval time: `22360 ms`
- Peak memory: `10253 MiB allocated`, `10530 MiB reserved`
- Total submission size `int8+zlib`: `9951155` bytes

Schedule read:

- Warmdown-only is worse than the root baseline by `+0.03917970` BPB post-roundtrip.
- Warmdown-only is also worse than `LowerLR` by `+0.03299334` BPB post-roundtrip.
- It does reduce artifact size by `2095472` bytes versus the root baseline, but size was not the bottleneck.
- Current evidence says the root schedule should remain the A100 anchor.

Date: 2026-03-28
Node: `serv-3343`
GPU: `NVIDIA H100 80GB HBM3`
Run: `h100_baseline_600s`

Measured outputs:

- Train setup: `10` shards, `600s` wallclock cap
- `amp_dtype: bf16`
- Stopped at `1795` steps in `600092 ms`
- Pre-roundtrip eval: `val_loss=2.2028`, `val_bpb=1.3046`
- Post-roundtrip exact eval: `val_loss=2.20503740`, `val_bpb=1.30594735`
- Post-roundtrip eval time: `10931 ms`
- Peak memory: `10303 MiB allocated`, `10730 MiB reserved`
- Total submission size `int8+zlib`: `14684525` bytes

H100 read:

- The exact same root baseline improves materially moving from `1xA100` to `1xH100`.
- Step average drops from about `661.65 ms` on A100 to about `334.31 ms` on H100.
- Post-roundtrip `val_bpb` improves by `-0.06546036` versus the best A100 baseline.
- Artifact size remains under the `16,000,000` byte challenge cap.

Date: 2026-03-28
Node: `serv-3342`
GPU: `8x NVIDIA H100 80GB HBM3`
Runs: `h100_8gpu_baseline_600s` attempt, `nccl_test.py` smoke, `h100_8gpu_baseline_600s`

Observed behavior:

- All `8` GPUs are visible in the allocation.
- `torchrun --standalone --nproc_per_node=8 train_gpt.py` never prints `logs/h100_8gpu_baseline_600s.txt`.
- Minimal `8`-rank `nccl_test.py` under `torchrun --standalone` also hangs before any rank prints.
- `torch.distributed.elastic` reports `RendezvousTimeoutError`.
- A fresh Slurm-shaped allocation with `--ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6` succeeds on the same node.
- Slurm-native smoke using `srun --gpu-bind=none` prints `rank 0..7 ok` and `rank 0..7 barrier ok`.
- Slurm-native trainer launch reaches `11611` steps in `599780 ms`.
- Final post-roundtrip exact eval from `8xH100` root baseline: `val_bpb=1.23368511`.
- Peak memory: `10184 MiB allocated`, `10358 MiB reserved`.
- Total submission size `int8+zlib`: `15871532` bytes.

Interpretation:

- This is not currently a model-code failure.
- The specific blocked path is `torchrun --standalone` rendezvous on `serv-3342`.
- Slurm-native `srun --ntasks=8 --gpus-per-task=1 --gpu-bind=none` is currently the working launch path for `8xH100` jobs on Pegasus.
- The root baseline is now challenge-shaped and reproducibly under the artifact cap on real `8xH100`.

Date: 2026-03-28
Node: `serv-3342`
GPU: `8x NVIDIA H100 80GB HBM3 (SXM5)`
Run: `pre_ttt_anchor_8xh100_600s`

Measured outputs:

- Train setup: `600s` wallclock cap, Session 03 pre-TTT anchor
- `amp_dtype: bf16`
- Stopped at `6564` steps in `599759 ms`
- Pre-quant EMA exact eval: `val_loss=1.93281857`, `val_bpb=1.14472403`
- Post-roundtrip exact eval: `val_loss=1.94590192`, `val_bpb=1.15247273`
- Sliding-window exact eval (`stride=64`): `val_loss=1.90633923`, `val_bpb=1.12904446`
- Step average: `91.37 ms`
- Peak memory: `21274 MiB allocated`, `22070 MiB reserved`
- Total submission size `int6+zstd`: `15751324` bytes
- Model bytes: `15692752`, code bytes: `58572`

Interpretation:

- This is the first real competition-phase architecture result on Pegasus `8xH100`.
- The Session 03 anchor improves on the root `8xH100` baseline by `0.10464065` BPB on the final sliding metric.
- The remaining gap to the public 2026-03-21 donor is small enough to justify narrow Session 04 deltas rather than a redesign.
- Throughput is one plausible contributor to the residual gap, but export fidelity also remains worth isolated measurement.

Date: 2026-03-28
Node: `serv-3342`
GPU: `8x NVIDIA H100 80GB HBM3 (SXM5)`
Run: `delta1_gptq_lite`

Measured outputs:

- Train setup: `600s` wallclock cap, Session 04 Delta 1: GPTQ-lite percentile clip search
- `amp_dtype: bf16`
- Stopped at `6565` steps
- Pre-quant EMA exact eval: `val_loss=1.93362522`, `val_bpb=1.14520403`
- Post-roundtrip exact eval: `val_loss=1.94640899`, `val_bpb=1.15277272`
- Sliding-window exact eval (`stride=64`): `val_loss=1.90696266`, `val_bpb=1.12941356`
- Step average: `91.37 ms`
- Total submission size `int6+zstd`: `16219752` bytes — OVER CAP

Interpretation:

- GPTQ-lite percentile clip search FAILED.
- All three eval metrics are worse than the Session 03 anchor.
- Artifact size exceeds the `16000000` byte cap by `219752` bytes.
- The clip search hurts zstd compressibility more than it helps quantization quality.
- Clean negative result: the export gap is not caused by clip suboptimality.
- Pivot to Delta 2 (LeakyReLU^2).

Date: 2026-03-29
Node: `serv-3342`
GPU: `8x NVIDIA H100 80GB HBM3 (SXM5)`
Run: `delta2_leakyrelu2_8xh100`

Measured outputs:

- Train setup: `600s` wallclock cap, Session 04 Delta 2: LeakyReLU^2
- `amp_dtype: bf16`
- Stopped at `6511` steps in `599586 ms`
- Pre-quant EMA exact eval: `val_loss=1.93224692`, `val_bpb=1.14438546`
- Post-roundtrip exact eval: `val_loss=1.94547854`, `val_bpb=1.15222198`
- Sliding-window exact eval (`stride=64`): `val_loss=1.90633378`, `val_bpb=1.12904123`
- Step average: `92.09 ms`
- Peak memory: `21274 MiB allocated`, `22070 MiB reserved`
- Total submission size `int6+zstd`: `15582968` bytes
- Model bytes: `15524210`, code bytes: `58758`

Interpretation:

- Delta 2 LeakyReLU^2 is a **NEUTRAL** result — not a clear win, not a failure.
- Sliding s64 val_bpb improved by only `-0.00000323` vs anchor — effectively zero.
- Pre-quant and roundtrip metrics improved slightly (`-0.00034` and `-0.00025` respectively).
- Artifact is `168356` bytes smaller — marginally better quantization-friendliness.
- Step time is `+0.72 ms` slower (`92.09` vs `91.37`), costing `53` steps. Slower throughput roughly cancels the small per-step quality gain.
- Measured anchor state for comparison was `enable_math_sdp(True)`. Delta 2 preserved that isolation.
- LeakyReLU^2 is **not a standalone graduating delta**, but may be a useful stack component later (slightly helps quantization + artifact size without hurting headline metric).

Date: 2026-03-29
Node: `serv-3343`
GPU: `1x NVIDIA H100 80GB HBM3`
Run: `scripts/bench_fa3_vs_sdpa.py` attention microbenchmark

Measured outputs:

- Workload: isolated attention kernel benchmark matching anchor dimensions
  - `B=16`, `T=2048`, `H=8`, `Hkv=4`, `D=64`
  - `50` warmup iterations, `200` timed iterations
- NGC `26.03` container (`PyTorch 2.11.0a0+a6c236b9fd.nv26.03.46836102`, `CUDA 13.2`)
  - SDPA flash: `1.967 ms/iter`
- NGC `25.02` container + installed `flash_attn_3` wheel (`PyTorch 2.11.0+cu130`, `CUDA 13.0`)
  - SDPA flash: `1.889 ms/iter`
  - direct `flash_attn_interface` FA3: `0.165 ms/iter`
  - relative kernel speedup vs SDPA flash in the same container: `11.44x`

Interpretation:

- This is an **isolated attention microbenchmark**, not end-to-end training throughput.
- The result is still strong enough to justify an isolated FA3 training delta next.
- NGC `26.03` remains the standard stable container path for normal runs.
- The saved Pegasus `25.02` FA3 container is the current explicit-FA3 experiment path.
- The next real question is whether direct FA3 materially improves `step_avg` in the full anchor training loop.

Date: 2026-03-29
Node: `serv-3343`
GPU: `1x NVIDIA H100 80GB HBM3`
Run: `2026-03-29_fa3_port` smoke

Measured outputs:

- Train setup: `world_size=1`, `grad_accum_steps=8`, `600s` wallclock cap
- Runtime path: explicit FA3 path later frozen into `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`
- Warmup completed normally through `20/20`
- Stable post-warmup training reached at least step `400`
- Step average stabilized around `640.17-640.52 ms`
- Training loss dropped from `6.9307` to `2.5696` by step `400`
- No NaN, FA3 import, or training-stability failures observed

Interpretation:

- The FA3 port is healthy on Pegasus `1xH100`.
- The remaining open question is `8xH100` single-node throughput, not FA3 correctness.
- The operationally correct Pegasus path is now the saved FA3 container, not ad hoc per-job pip installs.

Date: 2026-03-29
Node: `serv-3342`
GPU: `8x NVIDIA H100 80GB HBM3 (SXM5)`
Run: `2026-03-29_fa3_port`

Measured outputs:

- Train setup: `600s` wallclock cap, Session 05 Phase 1 FA3 port on saved Pegasus `25.02` FA3 container
- `amp_dtype: bf16`
- Stopped at `6474` steps in `599967 ms`
- Pre-quant EMA exact eval: `val_loss=1.93384137`, `val_bpb=1.14532979`
- Post-roundtrip exact eval: `val_loss=1.94672710`, `val_bpb=1.15296145`
- Sliding-window exact eval (`stride=64`): `val_loss=1.90726009`, `val_bpb=1.12958984`
- Step average: `92.67 ms`
- Peak memory: `20825 MiB allocated`, `21198 MiB reserved`
- Total submission size `int6+zstd`: `15529557` bytes
- Model bytes: `15471357`, code bytes: `58200`

Interpretation:

- Session 05 Phase 1 on the current saved-container FA3 runtime is a **FAILED** delta.
- It is slower than the Session 03 anchor by `+1.30 ms/step` and reaches `90` fewer steps.
- All quality metrics regress:
  - sliding s64 `+0.00054538`
  - pre-quant EMA `+0.00060576`
  - roundtrip `+0.00048872`
- Memory and artifact size improve (`-449 MiB`, `-221767 bytes`), but that is not enough to offset the throughput loss.
- The current best explanation is runtime-level regression from the pip-installed generic PyTorch stack replacing the vendor-tuned NGC build.
- Do not rerun the current saved-container FA3 path as a throughput candidate.

## Next Actions

### 1. Freeze the Session 03 facts

- Root `8xH100` baseline is the reference point:
  - `val_bpb=1.23368511`
  - `step_avg=51.66 ms`
  - `artifact=15871532 bytes`
- Session 03 anchor is the new competition reference:
  - sliding s64 `val_bpb=1.12904446`
  - roundtrip `val_bpb=1.15247273`
  - pre-quant EMA `val_bpb=1.14472403`
  - `step_avg=91.37 ms`
  - `artifact=15751324 bytes`
- Launcher lesson is locked:
  - do not use `torchrun --standalone` on Pegasus `8xH100`
  - use Slurm-native `srun --gpu-bind=none` with `LOCAL_RANK=$SLURM_LOCALID`, `RANK=$SLURM_PROCID`, `WORLD_SIZE=$SLURM_NTASKS`

### 2. Session 04 Delta 1: GPTQ-lite clip search — FAILED

Delta 1 measured results vs Session 03 anchor:

- Sliding s64 val_bpb: `1.12941356` (WORSE by `+0.00036910`)
- Roundtrip val_bpb: `1.15277272` (WORSE by `+0.00029999`)
- Pre-quant EMA val_bpb: `1.14520403` (effectively identical, `+0.00048000`)
- Artifact size: `16219752` bytes — OVER the `16000000` byte cap (anchor was `15751324`)
- Steps: `6565`, step_avg: `91.37 ms` (identical to anchor as expected)

Conclusion: GPTQ-lite percentile clip search is a clean negative result. It hurts zstd compressibility more than it helps quantization quality. The export gap is not caused by clip suboptimality.

### 3. Session 04 Delta 2: LeakyReLU^2 — NEUTRAL

Delta 2 measured results vs Session 03 anchor:

- Sliding s64 val_bpb: `1.12904123` (effectively identical, `-0.00000323`)
- Roundtrip val_bpb: `1.15222198` (slightly better, `-0.00025075`)
- Pre-quant EMA val_bpb: `1.14438546` (slightly better, `-0.00033857`)
- Artifact size: `15582968` bytes (smaller by `168356` bytes)
- Steps: `6511`, step_avg: `92.09 ms` (`+0.72 ms` slower, `-53` steps vs anchor)

Conclusion: LeakyReLU^2 is a neutral/tie result. Not a standalone graduating delta. Keep as a possible stack component — slightly better quantization-friendliness and artifact headroom, but slower throughput cancels the small per-step quality gain.

### 4. Session 04 closeout

Session 04 is complete.

- Delta 1 (GPTQ-lite percentile clip search): FAILED
- Delta 2 (LeakyReLU^2): NEUTRAL / tie

Interpretation:

- The isolated micro-delta sweep did its job: it ruled out one export-side hypothesis and showed one model-side tweak is stackable but not standalone.
- Session 04 should be treated as closed rather than extended into more tiny deltas by default.

### 5. Session 05 revised plan

Session 05 follows a 3-phase strategy based on competitive landscape analysis (2026-03-29):

1. **Phase 1: FA3 throughput port** (ACTIVE)
   - Port anchor attention from SDPA to direct `flash_attn_interface`
   - Current saved-container runtime at `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh` is a measured negative result
   - Full `8xH100` result: `92.67 ms/step`, sliding s64 `1.12958984` vs anchor `91.37 ms` and `1.12904446`
   - Further FA3 work is gated on a vendor-tuned NGC runtime
   - Candidate paths: build FA3 against an NGC torch build, or find a newer NGC image where FA3 works without replacing the tuned stack

2. **Phase 2: Full Hessian GPTQ**
   - Replace int6+zstd with Cholesky-compensated GPTQ
   - Reference: PR #1060 and #1072 implementations
   - Expected gain: `0.003-0.007 BPB` + better artifact compression
   - This is now the cleaner next model-side delta if a good FA3 runtime is not found quickly

3. **Phase 3: Novelty contribution**
   - Gated on phases 1-2 reaching competitive `1.12x` base
   - Candidates: fused Triton MLP kernel, coprime-stride loader, XAI/RFN-derived approach
   - TTT: parked, revisit only if phases 1-2 leave insufficient gap closure

### 6. Grant/application stance

- Pegasus `8xH100` access is already usable, so the grant is no longer the immediate blocker.
- The better case is to use the current access window to strengthen the pre-TTT and TTT story first, then refresh the compute request from a stronger measured position.
- Grant documentation remains worth keeping current, but it is no longer the mainline optimization target.

## Evidence Required From Each Run

- GPU model
- Exact command
- Train wallclock / `step_avg`
- Final post-roundtrip `val_bpb`
- Artifact size
- Peak memory
- Any compile or export warnings

## Decision Rule

The evidence package is now:

- `A100` smoke
- `A100` baseline
- `A100` `LowerLR` negative control
- `A100` seed-repeat reproducibility check
- `A100` warmdown negative control
- `1xH100` baseline
- `8xH100` `torchrun --standalone` rendezvous blocker
- `8xH100` Slurm-native NCCL smoke success
- `8xH100` Slurm-native trainer success
- `8xH100` Session 03 anchor success
- `8xH100` Session 04 Delta 1 GPTQ-lite failure
- `8xH100` Session 04 Delta 2 LeakyReLU^2 neutral

Do not spend more time on repeated `torchrun --standalone` retries or more root-baseline reruns.
Session 05 audit is complete (`docs/campaign/artifacts/05_ttt_correctness_audit.md`).

Key decisions from the audit, competitive analysis, and FA3 benchmark:
- Strategy pivot: build portable frontier base, not replicate old #1 verbatim
- FA3 is the first implementation target (Phase 1)
- Full Hessian GPTQ is the second target (Phase 2), replacing GPTQ-lite
- TTT is parked — top open PRs beat #1 without it
- Explicit FA3 runtime: saved NGC `25.02` FA3 container
- Treat the `11.44x` result as attention-kernel-only evidence, not full training speedup
- The first full `8xH100` FA3 run on the saved `25.02` container is a clean negative result
- Future FA3 work is gated on vendor-tuned NGC runtime compatibility
- Parameter Banking / Parallel Muon are deferred until after phases 1-2
- LeakyReLU² re-test is gated on FA3 (throughput-coupling hypothesis)
- Never use `| tail -1` or buffered stdout for Pegasus training jobs
- Never drop `--nodes=1` on challenge-shaped `8xH100` runs just to get a faster allocation

If a fresh session starts now, it should:
1. Do **not** rerun the current saved-container FA3 path
2. Check whether FA3 can run against a vendor-tuned NGC runtime without replacing the optimized torch stack
3. If not, proceed to Phase 2 (Full Hessian GPTQ) on the standard stable container path
