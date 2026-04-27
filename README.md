# trios-railway

## Disaster Recovery — one click

After a Railway ban or account loss, restore the entire IGLA fleet from this
repo. **Three trigger paths**, all documented in
[`docs/DISASTER_RECOVERY.md`](docs/DISASTER_RECOVERY.md):

[![Deploy IGLA Fleet on Railway](https://railway.com/button.svg)](https://railway.com/template/igla-fleet)

1. **Web button (above)** — published from [`railway-template.json`](railway-template.json). Provisions all 6 control-plane services (1 MCP + 3 champion seeds + dwagent + Neon backup-to-R2 sidecar).
2. **GitHub Actions** — `Actions → DR Deploy from template → account_alias=accN, confirm=PHI`. Workflow: [`deploy-from-template.yml`](.github/workflows/deploy-from-template.yml).
3. **CLI** — `tri-railway service deploy …` for each service in [`disaster-recovery/fleet-snapshot.json`](disaster-recovery/fleet-snapshot.json) (refreshed hourly by [`fleet-snapshot.yml`](.github/workflows/fleet-snapshot.yml)).
4. **MCP chat** — say “восстанови флот на acc3, подтверждаю PHI” to any client connected to the `trios-railway-mcp` server. Tools: `railway_dr_snapshot`, `railway_dr_restore` (issue [#17](https://github.com/gHashTag/trios-railway/issues/17)).

Fleet shape, audit ledger, and champion BPB rows survive any single-account ban
— see the survives-table in [`docs/DISASTER_RECOVERY.md`](docs/DISASTER_RECOVERY.md).

Anchor: `phi^2 + phi^-2 = 3`.


Manage Railway services for the **IGLA** project (`e4fe33bb-3b09-4842-9782-7d2dea1abc9b`)
+ online audit, with the source of truth in `.trinity/experience/` and Neon.

Anchor: `phi^2 + phi^-2 = 3`
Companion to:
- [`gHashTag/trios-trainer-igla`](https://github.com/gHashTag/trios-trainer-igla) — trainer
- [`gHashTag/trios-mcp`](https://github.com/gHashTag/trios-mcp) — MCP wrapper
- [`gHashTag/trios#143`](https://github.com/gHashTag/trios/issues/143) — eternal IGLA RACE dashboard

## Status

v0.0.1 — bootstrap. Rings landed:

| Ring | What | Issue |
|---|---|---|
| `RW-00` | Identity types + R7 `RailwayHash` | [#2](https://github.com/gHashTag/trios-railway/issues/2) |
| `RW-01` | GraphQL transport (skeleton) | [#3](https://github.com/gHashTag/trios-railway/issues/3) |
| `AU-00` | Neon DDL migrations | [#6](https://github.com/gHashTag/trios-railway/issues/6) |
| `AU-01` | Drift detector D1..D7 (in-memory) | [#7](https://github.com/gHashTag/trios-railway/issues/7) |
| `EX-00` | `.trinity/experience` writer | [#10](https://github.com/gHashTag/trios-railway/issues/10) |
| `BR-CLI` | `tri-railway` subcommand router | [#11](https://github.com/gHashTag/trios-railway/issues/11) |

Remaining: [#4](https://github.com/gHashTag/trios-railway/issues/4)..[#5](https://github.com/gHashTag/trios-railway/issues/5)
(typed read/write GraphQL),
[#8](https://github.com/gHashTag/trios-railway/issues/8)..[#9](https://github.com/gHashTag/trios-railway/issues/9)
(audit ledger writer + Gate-2 verdict),
[#12](https://github.com/gHashTag/trios-railway/issues/12)..[#17](https://github.com/gHashTag/trios-railway/issues/17)
(integration + cron + MCP).

## Build

```bash
cargo build --release
# binary: ./target/release/tri-railway
```

## Subcommands available today

```bash
# Print version
tri-railway version

# Print idempotent Neon DDL (issue #6)
tri-railway audit migrate-sql | psql "$NEON_DATABASE_URL"

# Append one L7 experience line
tri-railway experience append \
    --issue '#1' --phi-step EXPERIENCE \
    --task 'bootstrap repo' --status OK \
    --soul-name DustyDeployer --agent GENERAL
```

## Constitutional alignment

| Rule | How `trios-railway` honours it |
|---|---|
| **R1** Rust-only | One workspace, three crates, one binary; no Python, no `.sh`. |
| **R5** Honesty  | Audit verdict exit codes 0 / 1 / 2 reflect Gate-2, drift, NOT YET. |
| **R7** Triplet  | Every mutation seals `RAIL=<verb> @ project=<8c> service=<8c> sha=<8c> ts=<rfc3339>` via [`hash::RailwayHash::seal`](crates/trios-railway-core/src/hash.rs). |
| **R9** Embargo  | Mutation surface (issue #5) is gated behind `igla check <sha>`. |
| **L1** No `.sh` | Self-checked in CI. |
| **L2** Closes # | Every commit references an issue (`Closes #N`). |
| **L3** Clippy 0 | CI runs `cargo clippy -- -D warnings`. |
| **L4** Tests    | `cargo test --all` enforced in CI. |
| **L7** Experience | `EX-00` is append-only; the writer never truncates. |
| **L8** Push first | This README, the workspace, and the CI are pushed in the same commit that closes #1. |
| **L20** Sessions → tools | Every operator step is a `tri-railway …` subcommand. |
| **L21** Context immutability | The experience writer never seeks; it only opens in append mode. |

## Architecture

```
operator / agent
       │
       ▼
  tri-railway <verb>
       │
       ├── trios-railway-core         GraphQL transport + identity types
       ├── trios-railway-audit        drift detector + Neon DDL
       └── trios-railway-experience   .trinity/experience writer
       │
       ▼
  Railway GraphQL  ←──────────  Neon (igla schema, audit tables)
```

## Neon schema

`tri-railway audit migrate-sql` emits the DDL for these tables in the
existing `public` schema next to the IGLA RACE ledger:

- `railway_projects`
- `railway_services`
- `railway_audit_runs`
- `railway_audit_events`
- `v_railway_drift_open` (latest run, open events only)

## Development

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
```

## License

Apache-2.0
