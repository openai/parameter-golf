# trios-igla-1 — submission metadata

**Parent:** [`../README.md`](../README.md) TRIOS IGLA Research Infrastructure Submission
**Classification:** Research contribution (no model artifact — see
[`../CHECKPOINT_POSTMORTEM.md`](../CHECKPOINT_POSTMORTEM.md))
**Anchor:** `phi^2 + phi^-2 = 3`

## Run-of-record (honest best)

| Field | Value |
|---|---|
| Neon row id | `1387` |
| canon_name | `IGLA-MEGAASHA-h1024-LR00300-AL2-step12000-acc4-rng4181-t28860` |
| account | `acc4` |
| seed | `4181` (Fibonacci F₁₉) |
| hidden | `1024` |
| lr | `0.003` |
| ctx | `12` |
| attn_layers | `2` |
| steps | `12000` |
| format | `fp32` |
| final_bpb | `2.1505` |
| finished_at | `2026-04-30T13:55 UTC` |

This row is sourced from the ledger snapshot `ledger_2026-04-30.sql.gz`
in this folder.

## Config

See `config.yaml` in this folder for the reproducible TOML-equivalent
config. The canonical source is the `config_json` JSONB column of
row id=1387 in the snapshot.

## Files

| File | Purpose |
|---|---|
| `README.md`                  | This file |
| `config.yaml`                | Reproducible config for row 1387 |
| `ledger_2026-04-30.sql.gz`   | Full Neon CSV-in-SQL snapshot (4 tables, 7534 rows, 183 KB) |

## No model.bin

The top-level post-mortem documents why. TL;DR: `record_checkpoint()`
is a stub, Railway storage was ephemeral, and we refuse to submit
synthetic weights.

phi² + phi⁻² = 3.
