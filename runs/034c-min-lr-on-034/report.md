# 034c Run Report

This directory contains the synced execution artifacts for the `034c` ladder on
top of the validated `034` baseline.

## Artifacts

- `034cB/seed_314/`: corrected full run, completed through TTT
- `034cA/seed_314/`: corrected run, stopped during TTT at user request
- `034cB-invalid/`: earlier invalid launches kept for audit only

Both corrected rungs include:

- `config.json`
- `config_diff.json`
- training log
- `final_model.pt`
- `final_model.int6.ptz`

For both corrected rungs, `config_diff.json` is `{}`, meaning the effective
config matched the validated `034` baseline except for the intended `MIN_LR`
change.

## Baseline

Validated comparison target: `034` from
`/workspace/runs/034-frozen-direct-carry-from-031a/seed_314/train.log`

- stop step: `4732`
- stop `val_bpb`: `1.0704`
- post-EMA pre-quant `val_bpb`: `1.06975736`
- quantized diagnostic `val_bpb`: `1.07912355`
- final TTT `val_bpb`: `1.06669374`

## Corrected 034cB

Run:

- rung: `034cB`
- `MIN_LR=0.10`
- stop step: `5041`
- wallclock stop: `1196265ms`

Metrics:

- stop `val_bpb`: `1.0782`
- post-EMA pre-quant `val_bpb`: `1.06720956`
- quantized diagnostic `val_bpb`: `1.07642793`
- final TTT `val_bpb`: `1.06391360`

Comparison vs baseline `034`:

- post-EMA pre-quant: `-0.00254780`
- quantized diagnostic: `-0.00269562`
- final TTT: `-0.00278014`

Summary:

- `034cB` is the successful rung in this ladder.
- The online training curve looked weaker than `034`, but the final checkpoint
  improved at the metrics that matter.

## Corrected 034cA

Run:

- rung: `034cA`
- `MIN_LR=0.05`
- stop step: `4819`
- wallclock stop: `1196192ms`
- user-requested stop during TTT warmup; no final TTT result

Metrics captured before stop:

- stop `val_bpb`: `1.0724`
- post-EMA pre-quant `val_bpb`: `1.06977915`
- quantized diagnostic `val_bpb`: `1.07885075`

Comparison vs baseline `034`:

- stop `val_bpb`: `+0.0020`
- post-EMA pre-quant: `+0.00002179`
- quantized diagnostic: `-0.00027280`

Comparison vs corrected `034cB`:

- stop `val_bpb`: `-0.0058`
- post-EMA pre-quant: `+0.00256959`
- quantized diagnostic: `+0.00242282`

Summary:

- `034cA` was much better than `034cB` on stop-val.
- But at post-EMA and quantized diagnostics, it was roughly flat to baseline
  `034` and clearly worse than `034cB`.
- Because TTT was interrupted, `034cA` remains incomplete and should not be
  treated as a final official rung result.

## Operational Notes

- The bad first `034cB` failure mode was config drift. That is preserved under
  `034cB-invalid/`.
- The corrected launches inherited the validated `034` stack and only changed
  `MIN_LR`.
- The pod was freed after the user requested the live `034cA` run be stopped.
