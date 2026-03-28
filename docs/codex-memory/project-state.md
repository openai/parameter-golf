# Project State

Date: 2026-03-28

## Objective

Primary:
- use the verified Pegasus `8xH100` path to advance from the Session 03 anchor into Session 04 isolated deltas
- move from the first competition-phase anchor to targeted throughput and fidelity improvements

Secondary:
- keep the Session 03 anchor as the new fixed reference
- preserve exact launch, logging, artifact, and evaluation discipline

Stretch:
- reach a clearly improved `8xH100` result that justifies a stronger compute request or leaderboard-adjacent claim

## Current campaign state

- campaign scaffolding exists under `docs/campaign/`
- shared handoff file is `docs/campaign/AGENT_SYNC.md`
- evidence summary is `docs/campaign/artifacts/2026-03-28_a100_evidence_summary.md`
- coordination entry points exist:
  - `AGENTS.md`
  - `CLAUDE.md`
- Session 03 anchor run is complete

## Verified hardware state

- Pegasus `A100-80GB` path works
- Pegasus `1xH100` path works
- Pegasus `8xH100` path works when launched with Slurm-native `srun`
- Pegasus `8xH100` path does **not** work reliably with `torchrun --standalone` on `serv-3342`
- NGC 26.03 container on Pegasus confirmed working with fscratch setup

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
- throughput bottleneck identified: SDPA vs FA3, not model fidelity
- NGC container + fscratch confirmed as optimized Pegasus path

## What has not happened yet

- no FA3 integration (primary throughput unlock)
- no GPTQ-lite compression delta
- no LeakyReLU^2 activation delta
- no top-tier leaderboard-adjacent result yet

## Best next move

- start a fresh Codex session from `docs/codex-memory/BOOTSTRAP.md`
- execute Session 04: FA3 integration, GPTQ-lite, LeakyReLU^2 as isolated deltas
- focus on throughput unlock (FA3) as the highest-leverage single change
