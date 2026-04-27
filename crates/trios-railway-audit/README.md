# trios-railway-audit

Drift detection between Railway reality and the Neon ledger.

- `migrations::ddl_statements()` — idempotent DDL for the audit tables.
- `detect(real, ledger)` — emits `DriftEvent` for D1..D7.
- `verdict(real, events, target_bpb)` — Gate-2 PASS / NOT YET / DRIFT.
