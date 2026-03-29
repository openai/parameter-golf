# Agent Sync

Date: 2026-03-29

## Current Objective

Fix the Session 05b Full Hessian GPTQ correctness bug on top of the Session 03 anchor.

This is now the mainline. Do not spend more H100 time on the saved-container FA3 path, TTT,
or broader redesigns until the GPTQ export path is either fixed or explicitly abandoned.

## Challenge Reality

- Official leaderboard entry is **record-gated**, not top-5-open-entry.
- A record submission must beat the current official SOTA by at least `0.005` nats and show `p < 0.01`.
- Operationally, this means the project is a **threshold-crossing problem**, not a rank-climbing problem.
- Current official merged #1 is still PR #549 at `1.1194` BPB.
- Current open frontier is lower:
  - PR #1089: `1.1086` BPB, 3-seed mean
  - PR #1060: `1.1122` BPB, 3-seed mean

## Current Mainline Plan

### Phase 1: GPTQ correctness and parity with working PR code

1. Debug Session 05b on the same checkpoint, without retraining first.
2. Run the new export-only diagnostics on identical weights and inspect `gptq_layer_diagnostics.json`.
3. If GPTQ is still wrong, ablate `actorder=False` and `block_size=d_col` on the same checkpoint.
4. Only after the roundtrip gap is sane, re-run `1xH100` smoke with enough post-train time budget.
5. Only then run `8xH100`.

### Phase 2: training-side quality bundle

After GPTQ is healthy:
- XSA `4 -> 11`
- VE128
- tight SWA
- warmdown `3500`

### Parked for now

- saved-container FA3 throughput path
- TTT
- broad novelty probes
- broad compression/model redesigns before GPTQ correctness is resolved

## In Scope

- Session 05b GPTQ correctness debugging
- PR-code-first comparison against working GPTQ submissions
- Export-only A/B diagnostics on the same checkpoint
- `1xH100` re-smoke after the export path is fixed
- `8xH100` run only after the smoke gate passes

## Out Of Scope

- More saved-container FA3 reruns
- TTT work
- New A100 baseline work as the mainline
- Large stacked changes before GPTQ is sane
- Paper-first over-research when repo-local PR code exists

## Canonical Files

- Shared mutable state: `docs/campaign/AGENT_SYNC.md`
- Stable rules: `CLAUDE.md`
- Codex memory:
  - `docs/codex-memory/decisions.md`
  - `docs/codex-memory/project-state.md`
  - `docs/codex-memory/next-session.md`
  - `docs/codex-memory/session-handoff.md`
- GPTQ experiment:
  - `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py`
  - `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/README.md`
- GPTQ plan artifact:
  - `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md`
- Fresh-session restart prompt:
  - `docs/campaign/prompts/session_05b_gptq_debug_restart.md`

## Fixed Reference Results

- Session 03 anchor (`8xH100`, `serv-3342`)
  - sliding s64 `val_bpb=1.12904446`
  - pre-quant EMA `val_bpb=1.14472403`
  - int6 roundtrip `val_bpb=1.15247273`
  - steps `6564`
  - step_avg `91.37 ms`
  - artifact `15751324` bytes

- Saved-container FA3 negative result (`8xH100`, `serv-3342`)
  - sliding s64 `val_bpb=1.12958984`
  - pre-quant EMA `val_bpb=1.14532979`
  - int6 roundtrip `val_bpb=1.15296145`
  - steps `6474`
  - step_avg `92.67 ms`
  - artifact `15529557` bytes
  - conclusion: do not rerun this runtime as a throughput candidate

## Latest Measured Result: Session 05b GPTQ Smoke

Date: 2026-03-29
Node: `serv-3340`
GPU: `1xH100 80GB HBM3`
Run: `records/track_non_record_16mb/2026-03-29_full_hessian_gptq`

Measured outputs:

- stopped at `906` steps in `600202 ms`
- step_avg `662.47 ms`
- pre-quant EMA exact `val_bpb=1.47753094`
- roundtrip exact `val_bpb=1.68963326`
- GPTQ stats:
  - `67` Hessians collected
  - `66` GPTQ layers used
  - `0` naive fallbacks
  - `0` Cholesky fallbacks
  - Hessian collection `815 ms`
  - GPTQ quantization `4236 ms`
- artifact:
  - code `66907` bytes
  - model `7687970` bytes
  - total `7754877` bytes
- job hit the Slurm time limit before sliding eval finished

Interpretation:

- The training-side numbers from this `1xH100` smoke are **not comparable** to the `8xH100` anchor.
- The export result is still clearly wrong: the roundtrip gap is about `0.2121` BPB, far worse than the anchor gap of `0.00775`.
- This means the GPTQ pipeline mechanics work, but the current quantizer implementation is not trustworthy.

## Current Bug Assessment

High-confidence facts:

- The current local GPTQ loop is **not a faithful copy** of the working competition PR quantizer.
- PRs `#634`, `#1019`, and `#1060` all update the within-block residual from `j:`, while the local broken loop used `j+1:`.
- Working PRs run the full GPTQ loop across multiple clip percentiles and keep the best reconstruction MSE.
- The old local implementation used only fixed `row_max / 31` scaling and clamped to `[-32, 31]`, while the PRs use percentile search plus symmetric `[-31, 31]`.
- The old local hook filter pulled an extra `bigram.proj` Hessian because `.proj.` was treated as attention in `_classify_param`.
- The current smoke allocation left no reserve for full post-train eval.

Safest current conclusion:

- The main problem was the local GPTQ implementation drifting from the known-good PR code in the quantization loop itself.
- A PR-grounded repair is now landed locally, but it has **not** been runtime-verified on a real checkpoint yet.
- The first proof point is still export-only A/B, not more retraining.
- The first replay after the repair still failed: `gptq_diag` showed GPTQ worse than both naive baselines on all `66/66` layers.
- That result makes the bug look systematic and points upstream of the inner loop, most likely Hessian construction / interpretation.
- An export-only replay mode is now landed so the next checks can reuse a saved `final_model.pt` instead of retraining.

## Immediate Next Actions

1. **No more retraining yet.**
   - First debug on the same export path.

2. **Run export-only diagnostics in this order:**
   - run replay from a saved `final_model.pt`
   - inspect `gptq_layer_diagnostics*.json` from the repaired export path
   - check layer names where GPTQ is worse than:
     - legacy row-max int6
     - percentile-naive int6
   - if needed, ablate `actorder=False` and `block_size=d_col`

3. **Do not keep hand-debugging the old rewrite.**
   - the PR-grounded loop transplant is already in the local file

4. **After correctness is restored:**
   - keep the PR-style 5-percentile search
   - keep the symmetric `[-31, 31]` clamp
   - keep the tightened hook target set to the actual int6-exported weights

5. **Then re-run `1xH100` smoke with enough time for post-train export + eval.**

6. **Only after the smoke gap is sane:**
   - run full `8xH100`
   - then move to the 05c training bundle

## Workspace

- Local repo: `/home/amay/Work/parameter-golf`
- Remote repo: `/netscratch/$USER/parameter-golf`

Use `git clone` and `git pull` by default.
Use `rsync` only when local uncommitted changes must be pushed quickly.

## Historical Measurements

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

1. **Phase 1: FA3 throughput port** — NEGATIVE, PARKED
   - Full `8xH100` result: `92.67 ms/step`, sliding s64 `1.12958984` — worse than anchor
   - Root cause: pip torch replacement killed vendor-tuned NGC performance
   - Gated on vendor-tuned NGC runtime with native FA3 support

2. **Phase 2: Full Hessian GPTQ** — IMPLEMENTED, CORRECTNESS BUG
   - Code: `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py`
   - Plan: `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md`
   - Commit: `e00bc0a` pushed to origin/main
   - Implementation: 4 new functions (~200 lines), rank-0-only Hessian collection + GPTQ quantization
   - Algorithm: post-training calibration (128 seqs × 2048 tokens), Cholesky error compensation, block_size=128, actorder, percdamp=0.01
   - **1xH100 smoke test (2026-03-29 20:43 UTC+2, serv-3340): BUG FOUND**
     - GPTQ pipeline ran cleanly: 66 layers GPTQ'd, 0 Cholesky fallbacks, 4236ms quantization
     - But roundtrip quality is catastrophically bad: gap = **0.212 BPB** (anchor gap = 0.00775)
     - Pre-quant EMA: `1.4775` → Roundtrip: `1.6896` (906 steps, 1xH100)
     - Artifact: `7,754,877` bytes (under 16MB cap)
     - **Must debug before 8xH100 run**
   - Suspects: error compensation formula, Hinv_chol row/column indexing, actorder un-permute, per-row scale interaction with column-wise GPTQ updates

3. **Phase 3: Novelty contribution**
   - Gated on phases 1-2 reaching competitive `1.12x` base
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
1. **Debug the GPTQ correctness bug** — export-only, no retraining needed
   - A/B per-layer MSE (naive vs GPTQ on same checkpoint)
   - Inner-loop cascade diagnostics
   - Actorder ablation
   - If stuck: port from PR #1060 directly
2. After fix: add multi-percentile search + symmetric clamping
3. Re-smoke on 1xH100 (roundtrip gap < 0.01)
4. Full 8xH100 run
5. Then quality stack: brotli compression → MLP 3.5x → EngramLite → mixed-precision GPTQ
6. FA3 probe on NGC 26.03 is a quick background check (untested ABI compatibility)
