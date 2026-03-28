# Hardware And Constraints

## Challenge constraints

- train in under `10 minutes` on `8xH100 SXM`
- artifact under `16,000,000` bytes total
- final metric is `val_bpb`

## Pegasus verified state

- `H100` allocation works for `1xH100`
- `8xH100` single-node allocation works
- `8xH100` NCCL works under Slurm-native `srun`
- `torchrun --standalone` is currently not the correct launcher for Pegasus `8xH100`
- NGC 26.03 container confirmed working on Pegasus
- `/fscratch` confirmed as optimized data staging path (avoids `/netscratch` I/O bottlenecks)

## Current measured anchors

- `1xA100` root baseline: `1.37140771`
- `1xH100` root baseline: `1.30594735`
- `8xH100` root baseline: `1.23368511`
- `8xH100` Session 03 anchor (sliding s64): `1.12904446`
- `8xH100` Session 03 anchor (int6 roundtrip): `1.15247273`

## Artifact pressure

- `8xH100` root baseline artifact: `15871532` bytes
- `8xH100` Session 03 int6+zstd artifact: `15751324` bytes (model `15692752` + code `58572`)
- remaining headroom under cap: `248676` bytes
- size discipline is now important for any competition-phase change
- GPTQ-lite may recover some headroom; must be measured

## RunPod

- reserve for final validation only unless external credits are granted

## Practical implication

- infrastructure uncertainty is no longer the blocker
- throughput (SDPA vs FA3) is now the primary bottleneck
- the secondary blocker is model quality under a very tight artifact budget
- FA3 integration is the highest-leverage single change for Session 04
