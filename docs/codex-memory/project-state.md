# Project State

Date: 2026-04-19

## Objective

Primary:

- current target: `#1610`-direct, Gate A reproduced (BPB `1.07218477`), corrector lane closed, Fallback 1A active
- source base: `#1610` `train_gpt.py` at SHA `ca191953` (replaces the earlier `#1530`-first plan)
- execution plan remains `docs/campaign/PLAN_PR1610_CORRECTOR.md` (locked Revision 3, 2026-04-14) as historical reference
- Session 4 handoff: `docs/campaign/SESSION_4_PREP.md`
- fallback cascade now active at Level 1A: export-only `#1586`-style requant levers on the preserved Gate-A checkpoint

Secondary:

- keep non-record PR `#1598` open and frozen; do not edit unless reviewers ask
- D / R1 evidence bundle is frozen as local evidence base
- keep `07c1` background-only
- cleanup still pending:
  - delete idle RunPod pod `utwe9wnuze72ds`
  - rotate the leaked RunPod API key
  - rotate the leaked HF token

## Session 3 result

- Gate A: scientifically reproduced PR `#1610` at BPB `1.07218477` vs published `1.07216564` (delta `+1.913×10⁻⁵`), eval `455.9 s`, artifact `15,999,394 B`
- Gate A administrative `FAIL` was false: stale internal headroom threshold `15,997,520 B` tripped even though the artifact was `606 B` under the competition cap
- Gate B: not attempted
- Corrector ablations (all eval-only on the preserved Gate-A seed-0 checkpoint):
  - baseline: `1.07218477`
  - `1a`: `α=0.3`, orders `[8]`, BPB `1.08876294`, delta `+0.01658`, eval `462.8 s`
  - `1b`: `α=0.3`, orders `[5,8,12]`, BPB `1.08891256`, delta `+0.01673`, eval `472.4 s`
  - `1c`: `α=0.1`, orders `[5,8,12]`, BPB `1.07430360`, delta `+0.00212`, eval `465.8 s`
- Locked interpretation: corrector damage scaled monotonically with `α`; no tested configuration improved BPB; corrector lane is closed for this TTT-phased eval pipeline
- Active next move: Fallback Cascade Level 1A (`clip_sigmas` + int7 embeddings), 1–2 requant-only runs, kill criterion `<0.001 BPB gain` or artifact exceeds cap
- Artifact preservation:
  - `amay01/parameter-golf-session3-artifacts/runs/runs_20260418_2204.tar.gz`
  - MD5 `caf8adf63d8c80965f6671beba95d7aa`
  - contains Gate-A checkpoint, all 3 ablation logs, summary JSONs, and provenance
- Budget state:
  - productive Session 3 sub-session: `~$40` total (`~$22` productive + `~$18` deployment waste)
  - earlier infra-thrash sub-session: `~$10.40` total (`~$6.80` preventable waste)
  - remaining RunPod credit: `$76.64`

Historical sections below are archival pre-Session-3 context and should not be treated as the active plan.

## Current campaign state

- campaign scaffolding exists under `docs/campaign/`
- shared handoff file is `docs/campaign/AGENT_SYNC.md`
- evidence summary is `docs/campaign/artifacts/2026-03-28_a100_evidence_summary.md`
- coordination entry points exist:
  - `AGENTS.md`
  - `CLAUDE.md`
- Session 03 anchor run is complete
- Session 05b GPTQ implementation exists but is parked on the current anchor after 7 conclusive ablations
- Session 05c-plus code and smoke harness are implemented and pushed
- Session 05e same-checkpoint export-only replay completed and closed the GPTQ question for this model family
- Offline analysis utilities now live under `scripts/diagnostics/`

## Verified hardware state

- Pegasus `A100-80GB` path works
- Pegasus `1xH100` path works
- Pegasus `8xH100` path works when launched with Slurm-native `srun`
- Pegasus `8xH100` path does **not** work reliably with `torchrun --standalone` on `serv-3342`
- NGC 26.03 container on Pegasus confirmed working with fscratch setup
- Saved Pegasus FA3 container exists at `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`
- `1xH100` FA3 smoke is confirmed healthy
- Stock `25.02` + `--no-deps` FA3 import is not viable on Pegasus

## Locked baseline facts

- `1xA100` 600s baseline post-roundtrip exact: `val_bpb=1.37140771`
- `1xH100` 600s baseline post-roundtrip exact: `val_bpb=1.30594735`
- `8xH100` 600s baseline post-roundtrip exact: `val_bpb=1.23368511`
- `8xH100` baseline step average: `51.66 ms`
- `8xH100` baseline artifact size: `15871532` bytes

## Current measured anchors

- `8xH100` root baseline: `val_bpb=1.23368511` (step_avg `51.66 ms`, artifact `15871532` bytes)
- `8xH100` Session 03 anchor:
  - sliding s64 val_bpb: `1.12904446`
  - pre-quant EMA val_bpb: `1.14472403`
  - int6 roundtrip val_bpb: `1.15247273`
  - steps: `6564`, step_avg: `91.37 ms`
  - artifact: `15751324` bytes (model `15692752` + code `58572`)
  - GPU: `8xH100 SXM5`, `serv-3342`, NGC 26.03 container
- `8xH100` Session 05c-plus (quality-positive, throughput regressed):
  - sliding s64 val_bpb: `1.12557920` (anchor delta: **-0.00347**)
  - pre-quant EMA val_bpb: `1.14186715`
  - int6 roundtrip val_bpb: `1.14933197`
  - steps: `5977`, step_avg: `100.39 ms` (+9.02ms vs anchor)
  - artifact: `15589271` bytes
  - GPU: `8xH100`, NGC 26.03 container
- `8xH100` Session 05f (negative vs 05c-plus):
  - sliding s64 val_bpb: `1.12660664` (05c-plus delta: **+0.00103**)
  - pre-quant EMA val_bpb: `1.14190308`
  - int6 roundtrip val_bpb: `1.15026661`
  - steps: `5977`, step_avg: `100.51 ms` (+0.12ms vs 05c-plus)
  - artifact: `15630854` bytes (+41,583 bytes vs 05c-plus)
  - GPU: `8xH100`, NGC 26.03 container
- `8xH100` Session 05g (negative vs 05c-plus):
  - sliding s64 val_bpb: `1.12584234` (05c-plus delta: **+0.00026**)
  - pre-quant EMA val_bpb: `1.14203044`
  - int6 roundtrip val_bpb: `1.14963535`
  - steps: `6080`, step_avg: `98.67 ms` (-1.72ms vs 05c-plus)
  - artifact: `16475467` bytes (+886,196 bytes vs 05c-plus, over cap on old export path)
  - GPU: `8xH100`, NGC 26.03 container

## Launcher lesson

Use:
- Slurm-shaped allocation with `--ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6`
- Slurm-native `srun`
- env mapping inside the launch:
  - `LOCAL_RANK=$SLURM_LOCALID`
  - `RANK=$SLURM_PROCID`
  - `WORLD_SIZE=$SLURM_NTASKS`

Do not use:
- `torchrun --standalone` for Pegasus `8xH100`

## What has been demonstrated

- end-to-end training, evaluation, compression, and roundtrip validation
- controlled negative results (`LowerLR`, `Warmdown3600`)
- small A100 seed spread
- first challenge-shaped root baseline on real `8xH100`
- Session 03 pre-TTT anchor port: sliding s64 val_bpb `1.12904446` on `8xH100`
- int6+zstd roundtrip under the 16MB cap with `248676` bytes headroom
- small remaining donor gap with both throughput and export fidelity still worth isolated measurement
- NGC container + fscratch confirmed as optimized Pegasus path
- GPTQ-lite percentile clip search does not help at this scale (Session 04 Delta 1 negative result: worse BPB + artifact cap violation)
- LeakyReLU^2 activation is neutral (Session 04 Delta 2: sliding s64 val_bpb effectively identical at `1.12904123`, but slightly better quantization metrics and 168KB smaller artifact; slower step time cancels quality gain)
- The local public `1.1194` record is not “TTT only”: its pre-TTT base is already `1.1218` at `83.4 ms`, so stronger pre-TTT work and throughput matter before TTT can close the remaining gap
- Direct FA3 on Pegasus was benchmark-backed as a hypothesis, but the saved-container end-to-end path is now a measured negative result.
- The FA3 deployment path is operationally understood, but the current saved-container runtime is not a throughput candidate.
- The first `1xH100` GPTQ smoke successfully exercised Hessian collection, quantization, compression, reload, and eval.
- That same smoke also exposed a correctness failure in the current GPTQ quantizer: roundtrip exact `1.68963326` vs pre-quant exact `1.47753094`.
- The first replay-based Hessian repair also failed: `replay_ref_hfix` reached `2.15770170` from pre-quant `1.82064877`, with `gptq_diag` still reporting `66/66` layers worse than both naive baselines.
- The 05e architecture probe also failed to rescue GPTQ on the new stack:
  - pre-quant exact `3.95543154`
  - naive roundtrip exact `3.96902897`
  - GPTQ roundtrip exact `3.96902897`
  - `worse_than_naive_rowmax = 44/66`
- The checkpoint-diagnostics workflow is now operational on the best measured branch:
  - backed up `final_model.pt`, `final_model.int6.ptz`, and `train.log` under `diagnostics/2026-03-31_05c_plus/`
  - pulled `diagnostics_float.txt` and `diagnostics_int6.txt` locally
  - `scripts/diagnostics/diagnose_weights.py` now supports either:
    - a single-model weight statistics report, or
    - float-vs-int6 comparison mode on the same checkpoint
- The compression-path feasibility workflow is operational:
  - `scripts/diagnostics/compress_probe.py` tested 13 compression strategies on saved artifacts
  - best measured path on both 05c-plus and 05g is `custom-shuffle + brotli-10`
  - on 05c-plus it saved `149,991` bytes vs the current export baseline
  - on 05g it saved `1,032,921` bytes and brought the branch back under the cap
  - byte-shuffle itself contributes only `~8-10 KB`; custom serialization + brotli is the real win

## Session 05b: Full Hessian GPTQ (2026-03-29)

- Implementation: `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py`
- Plan: `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md`
- Commit: `e00bc0a` pushed to origin/main
- Algorithm: post-training calibration (128 seqs), Cholesky error compensation, block_size=128, actorder, percdamp=0.01
- 4 new functions (~200 lines): `_make_hessian_hook`, `collect_hessians`, `gptq_quantize_layer`, `gptq_mixed_quantize_int6`
- Export path restructured: rank-0-only GPTQ, barrier, all ranks read file for eval
- **1xH100 smoke test: CORRECTNESS BUG** — roundtrip gap 0.212 BPB (27x worse than anchor's 0.00775)
  - 66 layers GPTQ'd, 0 Cholesky fallbacks, 4.2s quantization, 7.75MB artifact
  - Pipeline mechanics work, but quantized weights reconstruct poorly
  - Must debug before 8xH100 run
  - The `1xH100` training metrics are not anchor-comparable because the smoke run uses a different `WORLD_SIZE` and therefore different `grad_accum_steps`
- **2026-03-29 code repair landed, rerun pending**
  - local PR diff found the key loop mismatch: `W_block[:, j + 1:]` vs PR `W_block[:, j:]`
  - the repaired code now matches the PR structure for:
    - within-block residual propagation
    - 5-percentile reconstruction search
    - symmetric `[-31, 31]` clamp
    - block-only `attn` / `mlp` Hessian targeting
  - export now writes `gptq_layer_diagnostics.json` with per-layer naive-vs-GPTQ MSE and worst-block summaries
  - this repo does not currently contain a saved checkpoint for same-checkpoint replay
  - this local shell does not have `torch`, so verification here only reached `py_compile`
- **2026-03-29 server replay still failed**
  - `gptq_diag: worse_than_legacy_rowmax=66 worse_than_percentile_naive=66`
  - roundtrip exact `2.15604597` vs pre-quant exact `1.82064982`
  - the remaining failure is systematic
  - export-only replay mode is now landed so the next ablations can use the saved `final_model.pt` directly
- **2026-03-30 smaller Hessian-path repair also failed**
  - commit on `main`: `9cea7e9`
  - `replay_ref_hfix`: `1.82064877 -> 2.15770170`, gap `+0.33705293`
  - `gptq_diag` remains `66/66` worse than both naive baselines
  - the smaller forward-hook + average+damp patch is not enough
  - next step should be a more faithful single-PR Hessian/quantization transplant

## What has not happened yet

- no vendor-tuned NGC FA3 runtime result yet
- no top-tier leaderboard-adjacent result yet
- no seed-validation run yet for 05c-plus (throughput regression makes it premature)
- no committed compression-path upgrade yet
- no corrected width-feasibility rerun yet after the initial probe

## Best next move

- **Rerun the corrected compression probe, then choose one coherent larger fork**
- Keep 05c-plus as the best measured branch for now
- 05f and 05g are clean negatives and should not receive more 8xH100 time
- Preferred diagnostic approaches going forward:
  - `python scripts/diagnostics/diagnose_weights.py final_model.pt` for single-checkpoint weight stats
  - `python scripts/diagnostics/diagnose_weights.py final_model.pt final_model.int6.ptz` for float-vs-int6 comparison
  - `python scripts/diagnostics/compress_probe.py diagnostics/2026-03-31_05c_plus/final_model.int6.ptz` for export-path feasibility
  - correlate those reports with the measured 05c-plus / 05f / 05g logs before proposing the next branch
- Use the corrected compression probe to decide whether the next big fork is:
  - compression-path upgrade + modest width, or
  - a different larger fork that does not depend on width unlock

## 2026-04-17 PR #1610 RunPod pipeline state

- Branch `submission/pr1610-corrector` is launch-pinned at commit
  `876bb3603eaeb9213d23e555645b49ed30d66738`.
- Session 1 warmup-fix commit `a33191f572430566b88c4d61badb0369e1e6f9a3`
  remains in ancestry and is enforced by `scripts/runpod_pipeline/00_verify_pod.sh`.
- `scripts/runpod_pipeline/` is committed and tracked (11 files).
- The final pre-launch doc mismatch was a stale S3 upload option in
  `scripts/runpod_pipeline/README.md`; removed. Stage 5 now documents only
  `hf:` and `rsync:` upload targets, matching `05_preserve_artifacts.sh`.
- Targeted re-audit after the fixes confirmed:
  - local HEAD == `origin/submission/pr1610-corrector`
  - Gate A checkpoint persistence happens before parsing
  - Stage 3 / Stage 4 decision logic includes the 0.001–0.002 hold band
  - fallback variants fail closed on missing BPB/results

Current status: Session 3 pod workflow is launch-ready from repo state.
