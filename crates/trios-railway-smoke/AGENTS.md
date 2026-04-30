# AGENTS.md — trios-railway-smoke

## Scope

Smoke-test crate for the IGLA RACE pipeline. Validates the full cycle:
queue → claim → trainer → JSONL stdout → parse → DB insert.

Uses synthetic data, no GPU, no external binaries. Runs in CI in <60s.

## Rules

- **R1** Rust-only.
- **R5** Honest exit codes.
- **L4** Tests pass; new code carries new tests.
- No network calls except Neon (configurable via env var).
- No GPU, no CUDA, no external binaries.

Anchor: `phi^2 + phi^-2 = 3`
