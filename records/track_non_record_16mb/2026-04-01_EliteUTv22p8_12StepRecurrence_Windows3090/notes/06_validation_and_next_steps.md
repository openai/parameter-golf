## 06 — Validation and Next Steps

## Validation checklist

Before claiming a result as final/public:

1. Confirm run has a valid final evaluation line (`[FINAL STRIDE 64]`).
2. Confirm no known causal/metric leak in configuration.
3. Confirm result family is comparable (same eval assumptions and protocol).
4. Record source log path for each table row.

## Current caveats

- `bpb_full_journey.csv` is the main narrative source, but some run families use different architectural regimes and should not be mixed without context.
- One journey entry is explicitly invalid (`SmearGate_BUG`) and should remain flagged.
- User-mentioned names `traininglog2` / `traininglog3` were not found under those exact names in the workspace; if they exist elsewhere, add them as supplemental sources.

## Immediate next experiments

1. Re-run top configurations with strict comparable protocol and deterministic data mode.
2. Re-validate best AB2/AB3-style configs under the same measurement envelope.
3. Perform targeted ablations on the current winning feature combination.
4. Extend journey CSV with explicit validity status and comparable-group tags.

## Documentation maintenance

- Keep `README.md` as the executive summary only.
- Keep detailed tables and source references in `notes/04_experimental_results.md`.
- Update journey milestones from `logs/bpb_full_journey.csv` as new runs are validated.
