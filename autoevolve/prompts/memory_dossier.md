# AutoEvolve Memory Dossier

- Mode: `proxy`
- Committed experiment rows (all modes): 0
- Committed experiment rows for `proxy`: 0
- Infrastructure failures for `proxy`: 0
- Research outcomes for `proxy`: 0

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

## Mode Heuristics
- Proxy mode should rank ideas by likely transfer to the official 8xH100 / 600s setting, not by cheap single-GPU shortcuts.
- Preserve official-like evaluation behavior in proxy runs so local wins are more meaningful when promoted.
- Proxy mode is still single-GPU, so some throughput and eval timing noise remains; promote meaningful local wins to final validation.
