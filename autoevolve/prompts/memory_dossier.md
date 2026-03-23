# AutoEvolve Memory Dossier

- Mode: `scout`
- Committed experiment rows (all modes): 0
- Committed experiment rows for `scout`: 0
- Infrastructure failures for `scout`: 0
- Research outcomes for `scout`: 0

## Read This First
- Local incumbent val_bpb for this mode: not yet benchmarked
- There is no local incumbent benchmark for this mode yet.
- There is no active exploration frontier right now.
- There are no infrastructure-only failures in the committed history.
- There are no committed research outcomes yet.

## Research Family Summary
- No research families recorded yet.

## Recent Research Outcomes
- No research outcomes yet.

## Infrastructure Failures
- None.

## Active Guard
- No active repeat-family guard.

## Scout Heuristics
- Prefer low-overhead probes that preserve step throughput. Completed scout runs beat ambitious ideas that never finish.
- Treat throughput-heavy quantization alignment and per-step fake-quant as promotion-track ideas unless timing telemetry says otherwise.
- If a family has already produced two non-kept outcomes in a row, pivot to a different family before trying a more complicated variant.
