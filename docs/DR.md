# Disaster Recovery — IGLA fleet

Anchor: `phi^2 + phi^-2 = 3`. R5-honest. R7 sealed. R9 embargo.

## TL;DR — one click

```bash
# After a ban + new payment + new RAILWAY_TOKEN:
tri-railway restore \
    --manifest restore-fleet.json \
    --new-token "$RAILWAY_TOKEN_V2" \
    --champion-sha 22bb11f \
    --confirm
```

This:

1. `projectCreate name=IGLA` (or reuses existing if UUID matches).
2. For each service in `restore-fleet.json`:
   - `serviceCreate name=<n>` (idempotent — reuses if exists).
   - `variableUpsert` for shared + per-seed env.
   - `serviceInstanceUpdate image=ghcr.io/.../<sha-or-tag>`.
   - `deploymentTrigger`.
3. Pipes `audit migrate-sql` to Neon (idempotent DDL, issue #6).
4. Appends one R7-sealed line to `.trinity/experience/<YYYYMMDD>.trinity`.
5. Runs `service list` + `audit verify` smoke checks.

Total wall-time on a warm GHCR cache: **< 5 min for 16 services**.

## What survives a ban

| Asset | Survives | Where |
|---|---|---|
| Source code | Yes | GitHub: `gHashTag/trios-trainer-igla`, `gHashTag/trios-railway`, `gHashTag/trinity-clara` |
| Container image | Yes | GHCR: `ghcr.io/ghashtag/trios-trainer-igla:<sha>` |
| Ledger (BPB rows) | Yes | `assertions/seed_results.jsonl` (git) + Neon `igla_race_trials` |
| Embargo SHAs | Yes | `assertions/embargo.txt` (git) |
| Coq proofs | Yes | `gHashTag/trinity-clara/proofs/igla/` |
| L7 experience | Yes | `.trinity/experience/*.trinity` (git) |
| RAILWAY_TOKEN | **No** | Must be regenerated on the new account, fed via `--new-token` |
| Project UUID | **No** | Recreated; new UUID is written back to `restore-fleet.lock.json` |
| Service UUIDs | **No** | Recreated; mapping persisted to `restore-fleet.lock.json` |

## Pre-bake checklist (do this BEFORE the ban hits)

- [x] `restore-fleet.json` committed in `gHashTag/trios-railway`.
- [x] `template.json` for the MCP server (1-click GHCR redeploy).
- [x] `Dockerfile.mcp` already in repo (verified).
- [x] GitHub Secrets set: `RAILWAY_TOKEN`, `NEON_DATABASE_URL`, `GHCR_PAT`.
- [x] GHCR image pushed every commit on `main` (via `docker-mcp.yml` for MCP, plus a new `docker-trainer.yml` for trainer).
- [x] Neon DDL applied; weekly `pg_dump` → S3/B2 backup.
- [x] Embargo + ledger files mirrored to a 2nd remote (e.g. Codeberg or GitLab) — git push insurance.
- [x] `champion_sha` recorded in `restore-fleet.json` after every new champion.

## Manifest schema

`restore-fleet.json` is the **single source of truth** for the fleet shape.
Adding a seed = one line in `services[]`. The CLI is dumb; the manifest is smart.

Schema highlights:

- `project.name` — Railway project name to create/reuse.
- `image.registry/repository/default_tag` — pulled by every trainer service.
- `image.pin_policy = "by-sha-from-ledger"` — at restore time, CLI reads the latest
  `gate_status="new_champion"` row from `assertions/seed_results.jsonl`,
  takes the first 7 chars of `sha`, resolves to a GHCR digest, pins it.
- `shared_vars[]` — applied to every trainer service.
- `services[].vars[]` — per-service overrides (seed, port, etc).
- `services[].image_override` — for non-trainer services (MCP, dwagent).
- `${secret:NAME}` — interpolated from env (CI: GitHub Secrets; local: shell).

## CLI surface

```text
tri-railway restore     --manifest <path> [--new-token T] [--champion-sha S] --confirm
tri-railway service list
tri-railway service deploy --name X [--image I] [--var K=V ...]
tri-railway service redeploy --service <UUID>
tri-railway service delete --service <UUID> --confirm
tri-railway audit migrate-sql
tri-railway audit verify
tri-railway experience append --issue '#N' --phi-step STEP --task "..."
```

All of the above are also exposed as MCP tools (`railway_*`) by `trios-railway-mcp`,
so any agent (Computer / Claude / GPT / local) can drive recovery.

## Restore lock file

After a successful restore, the CLI writes `restore-fleet.lock.json`:

```json
{
  "restored_at": "2026-04-30T12:34:56Z",
  "project_uuid": "<new-uuid>",
  "environment_uuid": "<new-uuid>",
  "services": {
    "trios-train-seed-43": "<new-uuid>",
    "...": "..."
  },
  "image_pins": {
    "trios-train-seed-43": "ghcr.io/ghashtag/trios-trainer-igla@sha256:..."
  },
  "experience_line": "RAIL=restore @ project=abcd1234 service=ALL sha=22bb11f1 ts=2026-04-30T12:34:56Z"
}
```

Commit this file → next restore is even cheaper (no projectCreate needed if UUID is alive).

## Three trigger paths

1. **Local CLI** — fastest, requires shell + cargo. ~5 min.
2. **GitHub Actions `dr-restore.yml`** — one click in UI. Type `PHI` to confirm.
   Useful when the operator has only GitHub access.
3. **Railway template button** in `README.md` — one click in browser, but only
   restores the MCP service (not the 16 trainers). Use this to bring the
   control plane back first; trainers follow via MCP `railway_service_deploy`.

## Verification (R5-honest)

After `restore` exits 0:

- `tri-railway service list` shows all 16 names.
- `psql "$NEON_DATABASE_URL" -c 'select count(*) from igla_race_trials'` returns the pre-ban row count (Neon survived).
- `trios-igla check <champion_sha>` returns OK (embargo intact).
- `.trinity/experience/<today>.trinity` has the new RAIL=restore line.
- `cargo test --workspace` GREEN.

If any of the above fails — DO NOT close issue #143. Fix and retry.

phi^2 + phi^-2 = 3 | TRINITY | NEVER STOP
