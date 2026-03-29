# Project State

Date: 2026-03-29

## Objective

Primary:
- repair the Session 05b Full Hessian GPTQ export path on top of the Session 03 anchor
- regain a sane roundtrip gap before spending more `8xH100` training budget

Secondary:
- keep the Session 03 anchor as the new fixed reference
- preserve exact launch, logging, artifact, and evaluation discipline

Stretch:
- recover a valid GPTQ path, then stack training-side quality improvements toward a leaderboard-entry-capable result

## Current campaign state

- campaign scaffolding exists under `docs/campaign/`
- shared handoff file is `docs/campaign/AGENT_SYNC.md`
- evidence summary is `docs/campaign/artifacts/2026-03-28_a100_evidence_summary.md`
- coordination entry points exist:
  - `AGENTS.md`
  - `CLAUDE.md`
- Session 03 anchor run is complete
- Session 05b GPTQ implementation exists but currently fails its smoke-test correctness gate

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

## What has not happened yet

- no correct Full Hessian GPTQ result yet
- no same-checkpoint naive-vs-GPTQ export A/B yet
- no runtime validation of the repaired PR-grounded quantizer yet
- no vendor-tuned NGC FA3 runtime result yet
- no top-tier leaderboard-adjacent result yet
- no measured VE128 delta yet

## Best next move

- **Debug the GPTQ roundtrip quality regression** — top priority
- Run the repaired export path on a real checkpoint and inspect `gptq_layer_diagnostics.json`
- If diagnosis is not immediate, try `actorder=False` and `block_size=d_col` on that same checkpoint
- After fix: re-smoke on `1xH100` with more post-train wallclock headroom, then full `8xH100`
- Then Session 05c training bundle (XSA-all + VE128 + SWA + warmdown3500)
- Do not spend time on FA3 or TTT until GPTQ is fixed
