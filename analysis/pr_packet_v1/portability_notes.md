# Portability Notes (V5.9 Packet)

## Leakage / Portability Audit Findings
- Hardcoded machine path leakage existed in several V5.9 docs (`/Users/michael/...`).
- Launch script token expansion relied on `config_path.parents[3]` assumptions.
- Record folder contains generated outputs and archives with machine-specific absolute paths and local-only traces.

## Portability Strategy Used
- Keep launch behavior unchanged.
- Improve root discovery robustness in packet execution code.
- Normalize documentation command examples to repo-relative paths.
- Separate commit-now vs hold-later artifacts to avoid leaking local-machine traces into public draft PR.

## Non-Goals In This Pass
- No redesign of launch graph.
- No threshold updates.
- No new model-family search.
- No paid-run simulation masquerading as validation.
