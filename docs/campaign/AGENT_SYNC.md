# Agent Sync

Date: 2026-03-29

## Current Objective

Use the completed Session 03 anchor as the fixed Pegasus competition reference while opening Session 05.

Session 04 is now closed:
- Delta 1 GPTQ-lite clip search: failed
- Delta 2 LeakyReLU^2: neutral

The immediate goal is Session 05:
- audit the throughput gap to the faster public stack
- audit portable pre-TTT stack improvements
- audit and plan score-first TTT integration

## In Scope

- Keep the measured Session 03 anchor as the fixed competition reference
- Use the verified Pegasus `8xH100` Slurm-native launch path as the main execution base
- Preserve exact launcher, artifact, and metric logging discipline from the baseline phase
- Investigate throughput contributors to the `91.37 ms -> 83.4 ms` gap, starting with FA3 portability
- Audit portable stack differences between the anchor and the local `1.1194` public record
- Audit and plan TTT integration with explicit legality and cost checks
- Refresh the compute request later using stronger H100 evidence if results improve

## Out Of Scope

- Claiming parity with the public `1.1194` record before measuring a stronger pre-TTT base
- More Session 04 micro-deltas as the default mainline
- Treating RFN as the mainline strategy
- Spending RunPod budget except for final validation later
- Bundling unrelated throughput, export, model, and eval changes into one unattributable run without a plan
- Arbitrary trainer edits unrelated to the Session 05 throughput / pre-TTT / TTT program

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
- Immediate next deliverable: Session 05 throughput + pre-TTT + TTT audit, not more Session 04 micro-deltas

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

### 5. Session 05 opening stance

Session 05 has three linked tracks:

1. **Throughput audit**
   - explain the gap from anchor `91.37 ms` to the public `83.4 ms` stack
   - start with FA3 portability
   - separate FA3 from harder changes like parameter banking and Parallel Muon

2. **Pre-TTT base enhancement audit**
   - compare the anchor to the local `1.1194` public record
   - rank portable stack features by cost and likely upside
   - likely first-wave candidates: VE128, warmdown3500, Bigram 1536, tight SWA

3. **TTT correctness + portability audit**
   - inspect the public score-first TTT path
   - verify legality, evaluation budget, and portability to the anchor stack
   - treat TTT as necessary but not sufficient from the current `1.1290` anchor

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
If a fresh session starts now, it should begin by closing Session 04 and planning Session 05 from:
- the measured Session 03 anchor
- the Session 04 closeout artifact
- the local `1.1194` public record
- the existing Session 05 audit doc
