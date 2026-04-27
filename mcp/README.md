# trios-railway-mcp

MCP (Model Context Protocol) stdio server that wraps the
[`tri-railway`](../bin/tri-railway) CLI so MCP-aware clients (Claude Desktop,
VS Code MCP, custom hosts) can drive Railway deployments and audits for the
[TRAINER-IGLA-SOT](https://github.com/gHashTag/trios-trainer-igla) mission.

> Anchor: `phi^2 + phi^-2 = 3` · Gate-2 deadline `2026-04-30 23:59 UTC` ·
> Tracking: [#18](https://github.com/gHashTag/trios-railway/issues/18)

## Tools (7)

| Tool | Purpose |
| --- | --- |
| `railway_service_list` | List services in IGLA project, with deploy status + image digest. |
| `railway_service_deploy` | Create/update a service from an OCI image with env vars. |
| `railway_service_redeploy` | Trigger redeploy by `service_id`. |
| `railway_service_delete` | Delete a service. Requires `confirm: true`. |
| `railway_audit_run` | Run D1..D7 drift audit. |
| `neon_hive_status` | Read-only freshness check of Hive tables in Neon. |
| `railway_experience_append` | Append L7 row with R7 triplet to `.trinity/experience/`. |

## Build

```bash
cd mcp
npm install
npm run build
```

Produces `dist/index.js` (ES2022 ESM, executable).

## Run (stdio)

```bash
TRI_RAILWAY_BIN=/path/to/tri-railway \
RAILWAY_TOKEN=... \
TRIOS_NEON_DSN=postgres://... \
node dist/index.js
```

The server logs to **stderr** only (stdout is reserved for MCP frames).

## Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "trios-railway": {
      "command": "node",
      "args": ["/absolute/path/to/trios-railway/mcp/dist/index.js"],
      "env": {
        "TRI_RAILWAY_BIN": "/absolute/path/to/cargo/bin/tri-railway",
        "RAILWAY_TOKEN": "447e97bf-8c32-42c9-a585-c7e359f7458f",
        "RAILWAY_PROJECT_ID": "e4fe33bb-3b09-4842-9782-7d2dea1abc9b",
        "RAILWAY_ENVIRONMENT_ID": "54e293b9-00a9-4102-814d-db151636d96e",
        "TRIOS_DEFAULT_IMAGE": "ghcr.io/ghashtag/trios-trainer-igla:latest",
        "TRIOS_NEON_DSN": "postgres://USER:PASS@HOST/neondb?sslmode=require"
      }
    }
  }
}
```

Then restart Claude Desktop. The 7 tools should appear under the `trios-railway`
namespace.

## Configure Cursor / VS Code MCP

```json
{
  "servers": {
    "trios-railway": {
      "type": "stdio",
      "command": "node",
      "args": ["/absolute/path/to/trios-railway/mcp/dist/index.js"],
      "env": { "TRI_RAILWAY_BIN": "tri-railway", "RAILWAY_TOKEN": "..." }
    }
  }
}
```

## Environment variables

| Var | Default | Purpose |
| --- | --- | --- |
| `TRI_RAILWAY_BIN` | `tri-railway` | Path to the compiled CLI. |
| `RAILWAY_TOKEN` | — | Project-scoped UUID or team JWT. |
| `RAILWAY_PROJECT_ID` | `e4fe33bb-…` | IGLA project. |
| `RAILWAY_ENVIRONMENT_ID` | `54e293b9-…` | `production`. |
| `TRIOS_DEFAULT_IMAGE` | `ghcr.io/ghashtag/trios-trainer-igla:latest` | Default image for `service_deploy`. |
| `TRIOS_NEON_DSN` | — | Required for `neon_hive_status`. |
| `TRI_RAILWAY_TIMEOUT_MS` | `120000` | Per-CLI-invocation timeout. |

## Constitutional notes

- **R1** — TypeScript on stdio, no Python.
- **R5** — Exit codes from `tri-railway` are forwarded; non-zero sets `isError: true`.
- **R7** — Triplet `BPB=<v> @ step=<N> seed=<S> sha=<7c> jsonl_row=<L> gate_status=<g>`
  is required for `railway_experience_append`.
- **R9** — No tool exposes a path that bypasses `ledger::is_embargoed`.

## Status

- v0.0.1 — initial wrapper. CI hookup tracked in
  [#18](https://github.com/gHashTag/trios-railway/issues/18).
- `audit hive-status` subcommand on `tri-railway` is referenced but not yet
  shipped — `neon_hive_status` will return non-zero until that lands.
