# Multi-Account Fan-Out — IGLA RACE acceleration

Anchor: `phi^2 + phi^-2 = 3`. Issue: [#143](https://github.com/gHashTag/trios/issues/143).

## Current accounts (verified via Railway GraphQL on 2026-04-27)

| Acc | Email | Token secret | Status | Projects |
|---|---|---|---|---|
| 0 | `kaglerslomaansc@hotmail.com` | — | ❌ Not Authorized — token revoked / banned | — |
| 1 | `rumbodzalaclhdv0@hotmail.com` | `RAILWAY_TOKEN_ACC1` | ✅ OK | IGLA (member), `artistic-beauty`, `dazzling-blessing`, `surprising-enthusiasm` |
| 2 | `brabbtjubindt5cug@hotmail.com` | `RAILWAY_TOKEN_ACC2` | ✅ OK | `reasonable-perception`, `thriving-eagerness` |

GitHub Secrets in this repo (set via `gh secret set`):

```
RAILWAY_TOKEN_ACC1
RAILWAY_TOKEN_ACC2
RAILWAY_PROJECT_ACC1_IGLA   = e4fe33bb-3b09-4842-9782-7d2dea1abc9b
RAILWAY_PROJECT_ACC1_AB     = e6f02957-4703-45b1-bfbd-cbc04bbca149
RAILWAY_ENV_ACC1_IGLA       = 54e293b9-00a9-4102-814d-db151636d96e
RAILWAY_ENV_ACC1_AB         = 549d283b-3f76-4ac9-855c-acf2c10f5817
RAILWAY_PROJECT_ACC2_RP     = 12c508c7-1196-468d-b06d-d8de8cb77e93
RAILWAY_PROJECT_ACC2_TE     = 39d833c1-4cb6-4af9-b61b-c204b6733a98
RAILWAY_ENV_ACC2_RP         = 441bd3a6-f6d8-455e-b567-376b7538e9f1
```

## Lane assignment (4 configs × 3 seeds = 12 parallel containers)

| Lane | Config | Services | Acc | Project | Env |
|---|---|---|---|---|---|
| **A** | champion + GF16 (h=384, lr=0.003, 60K) | `igla-final-seed-{42,43,44}` | Acc1 | IGLA | production |
| **B** | + muP transfer (h=512, 40K) | `iglaB-seed-{42,43,44}` | Acc2 | thriving-eagerness | production (`bce42949-…`) |
| **C** | Schedule-Free + WSD (60K) | `iglaC-seed-{42,43,44}` | Acc1 | artistic-beauty | production |
| **D** | post-hoc EMA sweep over A checkpoints | `iglaD-eval-{42,43,44}` | Acc2 | reasonable-perception | production |

Single source of ledger truth: Neon DB `igla_race_trials` — every container appends its
`agent_id`, `branch`, `seed`, `bpb`, `step`, `sha`. `trios-igla gate --target 1.50 --quorum 3`
aggregates across all four lanes.

## Live deploy (2026-04-27 — already on Railway)

Lane A — Acc1, project IGLA:

| Service | UUID | Triplet |
|---|---|---|
| `igla-final-seed-42` | `89e5243d-…` | `RAIL=deploy @ project=e4fe33bb service=89e5243d sha=42da805cd1ddcbe8 ts=2026-04-27T14:35:44Z` |
| `igla-final-seed-43` | `e9779d8f-…` | `RAIL=deploy @ project=e4fe33bb service=e9779d8f sha=e1871ad47e7a8197 ts=2026-04-27T14:35:38Z` |
| `igla-final-seed-44` | `10994ca5-…` | `RAIL=deploy @ project=e4fe33bb service=10994ca5 sha=711099e7a885bc1f ts=2026-04-27T14:35:36Z` |

Lane B — Acc2, project thriving-eagerness (IGLA-MIRROR-2):

| Service | UUID | Deploy ID |
|---|---|---|
| `iglaB-seed-42` | `c6675e56-…` | `b592a83c-…` |
| `iglaB-seed-43` | `63a8a01d-…` | `aa8be524-…` |
| `iglaB-seed-44` | `f95cedf4-…` | `cc5c8a1a-…` |

## How a workflow picks the right account

```yaml
env:
  RAILWAY_TOKEN: ${{
    inputs.lane == 'A' && secrets.RAILWAY_TOKEN_ACC1 ||
    inputs.lane == 'B' && secrets.RAILWAY_TOKEN_ACC2 ||
    inputs.lane == 'C' && secrets.RAILWAY_TOKEN_ACC1 ||
    inputs.lane == 'D' && secrets.RAILWAY_TOKEN_ACC2 ||
    secrets.RAILWAY_TOKEN }}
```

For MCP-driven runs, a future `trios-railway-mcp` v0.0.2 will accept an
optional `account_alias` (`acc1` / `acc2`) on every tool, choosing the token
internally. For now, run two MCP instances (`trios-mcp-acc1`,
`trios-mcp-acc2`) — each takes a single `RAILWAY_TOKEN` env var.

## DR + multi-account interplay

`restore-fleet.json` v1 is single-account. v1.1 will add an `accounts[]`
section and per-service `account_id`. After a ban, the operator only needs to:

1. Replace the token of the banned account in GitHub Secrets.
2. Run `tri-railway restore --new-token "$NEW" --account-alias acc1`.
3. Other accounts continue running uninterrupted.

phi^2 + phi^-2 = 3 | TRINITY | NEVER STOP
