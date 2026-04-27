# Disaster Recovery Runbook — IGLA fleet

Anchor: `phi^2 + phi^-2 = 3` · refs [trios#143](https://github.com/gHashTag/trios/issues/143).

This runbook lets you bring the IGLA training fleet back online **in
under 15 minutes** after a Railway-account ban, payment lapse, or
catastrophic project deletion.

## TL;DR — three commands

```bash
# 1. Provision new Railway account, generate a fresh Personal API token.
gh secret set RAILWAY_TOKEN_ACC3 --repo gHashTag/trios-railway     # paste token

# 2. Click the Deploy on Railway button in the README → fill 3 secrets.

# 3. Migrate the audit ledger schema (idempotent).
./target/release/tri-railway audit migrate-sql | psql "$NEON_DATABASE_URL"
```

You are back online.

## What survives, what does not

| Asset                       | Survives a Railway ban? | Why |
|-----------------------------|---|---|
| Source code                 | ✅ | GitHub: `gHashTag/trios-trainer-igla`, `gHashTag/trios-railway`, `gHashTag/trinity-clara` |
| Container images            | ✅ | GHCR: `ghcr.io/ghashtag/trios-trainer-igla:sha-<commit>` is independent of Railway |
| Champion ledger             | ✅ | git: [`assertions/seed_results.jsonl`](https://github.com/gHashTag/trios-trainer-igla/blob/main/assertions/seed_results.jsonl) |
| Embargo SHAs                | ✅ | git: `assertions/embargo.txt` |
| Coq proofs                  | ✅ | git: `gHashTag/trinity-clara/proofs/igla/` |
| Fleet shape                 | ✅ | this repo: [`disaster-recovery/fleet-snapshot.json`](../disaster-recovery/fleet-snapshot.json) (refreshed hourly via [`fleet-snapshot.yml`](../.github/workflows/fleet-snapshot.yml)) |
| Audit ledger rows           | ✅ | Neon (separate account) **+** hourly `pg_dump` to R2 (via the `neon-backup-r2` service in this template) |
| `RAILWAY_TOKEN`             | ❌ | Issued per-account; regenerate after recovery |
| Project / service UUIDs     | ❌ | Recreated by the template; new IDs land back in `fleet-snapshot.json` next hour |
| Public MCP URL              | ❌ | New domain on the new project; update `trios-perplexity` MCP config |

## Trigger paths

You have **three** ways to recover. Use whichever is fastest given the
state of your access.

### A. Railway template button (web UI · ~5 min)

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/template/igla-fleet)

The button links to a published template (manifest:
[`railway-template.json`](../railway-template.json)). Filling in the
required secrets (`RAILWAY_TOKEN`, `NEON_DATABASE_URL`, R2 credentials)
provisions all 6 control-plane services with `ghcr.io` image pins.

### B. CLI from a fresh shell (~3 min)

```bash
gh repo clone gHashTag/trios-railway && cd trios-railway
cargo build --release --bin tri-railway --locked

RAILWAY_TOKEN=<new-acct-token> RAILWAY_TOKEN_AUTH=team \
  ./target/release/tri-railway service deploy \
      --project <NEW_PROJECT_UUID> \
      --environment <NEW_ENV_UUID> \
      --name trios-mcp-public \
      --image ghcr.io/ghashtag/trios-railway-mcp:latest \
      --var NEON_DATABASE_URL=<...> \
      --var RAILWAY_TOKEN=<same-as-above> \
      --var PORT=8080

# Repeat for each service in fleet-snapshot.json.
```

### C. GitHub Actions ([`deploy-from-template.yml`](../.github/workflows/deploy-from-template.yml)) (~5 min)

```bash
# Operator: store the new account's token first.
gh secret set RAILWAY_TOKEN_ACC3 --repo gHashTag/trios-railway

gh workflow run deploy-from-template.yml --repo gHashTag/trios-railway --ref main \
    -f account_alias=acc3 \
    -f project_name=IGLA-DR \
    -f confirm=PHI
```

The workflow reads `railway-template.json`, calls Railway GraphQL to
create one project + N services, and writes the new IDs back to
`disaster-recovery/last-restore.json`.

## Required secrets for full recovery

Stored in `gHashTag/trios-railway` Actions Secrets (`gh secret list`):

| Secret                           | Purpose                                | Where to get |
|---|---|---|
| `RAILWAY_TOKEN_ACC1`             | Acc1 Personal API token                | <https://railway.com/account/tokens> on `rumbodzalaclhdv0@hotmail.com> |
| `RAILWAY_TOKEN_ACC2`             | Acc2 Personal API token                | same on `brabbtjubindt5cug@hotmail.com> |
| `RAILWAY_TOKEN_ACC3`             | Acc3 token (created during recovery)   | new account |
| `RAILWAY_PROJECT_ACC*_*`         | Project UUIDs (auto-discovered)        | `tri-railway service list` |
| `RAILWAY_ENV_ACC*_*`             | Environment UUIDs                      | `tri-railway service list` |
| `NEON_DATABASE_URL`              | `postgres://...?sslmode=require`       | <https://console.neon.tech/> |
| `TRIOS_REPO_PAT`                 | Fine-grained PAT, `issues:write` on `gHashTag/trios` | <https://github.com/settings/personal-access-tokens> |
| `R2_ACCESS_KEY_ID`               | Cloudflare R2 access-key               | <https://dash.cloudflare.com/r2> |
| `R2_SECRET_ACCESS_KEY`           | R2 secret-key                          | same |
| `R2_BUCKET`                      | Bucket name (e.g. `igla-ledger-backups`) | same |
| `R2_ENDPOINT`                    | `https://<acct>.r2.cloudflarestorage.com` | same |

## Recovery sequence in detail

1. **Snapshot is current.** Verify `disaster-recovery/fleet-snapshot.json`
   was written within the last hour (the `fleet-snapshot.yml` workflow
   commits it). The file lists every service across every account, so
   you have the canonical shape even if Railway is wholly inaccessible.

2. **Open new Railway account.** Sign up, accept email, add billing
   (DR is meaningless without it). Generate a Personal API token at
   <https://railway.com/account/tokens>.

3. **Push token + R2 creds to GitHub Secrets** (so the template can
   read them). One-liner per secret:

   ```bash
   gh secret set RAILWAY_TOKEN_ACC3 --repo gHashTag/trios-railway
   ```

4. **Deploy the template** via path A, B, or C above.

5. **Run the audit-ledger migration** (idempotent — safe to re-run):

   ```bash
   ./target/release/tri-railway audit migrate-sql | psql "$NEON_DATABASE_URL"
   ```

6. **Restore Neon ledger** (only if Neon itself was lost). Pull the
   most recent `pg_dump` from R2:

   ```bash
   aws s3 cp s3://igla-ledger-backups/igla/audit-ledger/<latest>.sql.gz . \
       --endpoint-url "$R2_ENDPOINT"
   gunzip -c <latest>.sql.gz | psql "$NEON_DATABASE_URL"
   ```

7. **Update MCP config.** Point the `trios-perplexity` MCP servers at
   the new project UUIDs (they will appear in the next
   `fleet-snapshot.json` commit).

8. **Verify with watchdog.** Trigger
   [`audit-watchdog.yml`](../.github/workflows/audit-watchdog.yml)
   manually:

   ```bash
   gh workflow run audit-watchdog.yml --repo gHashTag/trios-railway --ref main \
       -f target=1.85 -f skip_comment=false
   ```

   The next comment on `trios#143` should show the new account/project
   labels. If it does, recovery is complete.

## Cost expectations

- **Railway**: 6 services on the hobby tier ≈ $5–$15/mo per account; the 3
  trainer services dominate (2 vCPU each).
- **Neon**: free tier is enough for the ledger (low row volume).
- **R2**: hourly 1-MB dumps at 14-day retention is well under the 10 GB
  free quota.

## Why the template only contains 6 services, not 18

Railway template marketplace caps practical templates at ~10 services
and balks at copy-pasted seeds (the trainer-seed-100..102 fleet). The
template ships exactly the services needed to **resume the race from a
ban**:

- 1 × MCP control-plane (orchestrates everything else)
- 3 × champion seeds (42, 43, 44 — the configuration that produced the
  current best BPB 2.21 according to `seed_results.jsonl`)
- 1 × dwagent (auto-claims further trials)
- 1 × neon-backup-r2 (closes the last single-point-of-failure)

Once the control-plane is online, additional seeds (45/46/.../102) can
be spawned in seconds via the MCP `railway_service_deploy` tool reading
from `disaster-recovery/fleet-snapshot.json`.

phi^2 + phi^-2 = 3 · TRINITY · NEVER STOP
