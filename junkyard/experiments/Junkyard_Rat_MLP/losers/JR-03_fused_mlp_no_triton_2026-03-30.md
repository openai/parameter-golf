# JR-03 Loser — Fused MLP (No Triton Kernel), 2026-03-30

## Verdict
Loser for the JR lane under current setup.

## Run
- Script lane: `experiments/Junkyard_Rat_MLP`
- Seed: `1337`
- Loader: `coprime`
- `MLP_KERNEL_MODE=fused_mlp`
- Compile: `enabled=1 mode=default fullgraph=1`
- World size: `8`
- Wallclock cap: `600s`
- Runtime log reference (from run output): `logs/9a025f71-b0b4-4691-b8aa-7a88fc54cf9a.txt`

## Observed Metrics
- Stop step at wallclock cap: `6598`
- Mid-run eval: `val_bpb=1.1354` at step `6598`
- Post-EMA diagnostic: `val_bpb=1.1345`
- Final sliding window exact: `val_bpb=1.11105181`
- Peak memory: `allocated 22850 MiB`, `reserved 23004 MiB`

## Why Marked Loser
The fused MLP lane did not show a competitive quality gain versus the JR control path and does not justify promotion.

## Exception / Revisit Condition
This test used the **non-Triton** fused path (`fused_mlp`).  
It may still be worth a future revisit on Triton-capable nodes with a true kernel-level MLP fusion path if we explicitly target throughput-only gains and re-measure end quality drift.

