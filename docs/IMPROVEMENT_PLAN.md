# trios-railway-mcp Improvement Plan
## Deep Investigation & Decomposed Roadmap

**Date**: 2026-04-28  
**Anchor**: φ² + φ⁻² = 3  
**Current state**: 12 tools, clippy clean, 48 services injected with NEON_DATABASE_URL

---

## Executive Summary

The MCP gateway is **functional** but has critical gaps in 5 areas:
1. **Observability** — no container logs, no tracing spans
2. **Reliability** — no connection pooling, no rate limiting, no auth
3. **Test coverage** — 0 tests in `trios-railway-mcp` (L4 violation)
4. **Tool completeness** — 5 high-impact tools missing for autonomous gardener
5. **Code hygiene** — 825-line `tools.rs`, dead code annotations, hardcoded constants

---

## Phase 1: Critical Fixes (P0 — 2-3 hours)

### 1.1 Neon connection pooling
**Problem**: [`db_connect()`](crates/trios-railway-mcp/src/tools.rs:786) creates a new TCP+TLS connection for EVERY tool call. Each call = DNS resolve + TLS handshake + auth = ~500ms overhead + connection exhaustion risk on Neon pooler.

**Fix**: Add `deadpool-postgres` or `bb8-postgres` connection pool, initialized once at server startup.

```rust
// crates/trios-railway-mcp/src/db.rs (NEW FILE)
use deadpool_postgres::{Config, Pool, Runtime};

static DB_POOL: OnceLock<Pool> = OnceLock::new();

pub fn init_pool() -> Result<Pool, McpError> {
    let url = neon_url()?;
    // strip channel_binding, keep sslmode
    let cfg = Config::from_str(&cleaned_url)?;
    let pool = cfg.create_pool(Some(Runtime::Tokio1), tls_connector()?)?;
    DB_POOL.set(pool).ok();
    Ok(pool)
}

pub async fn db_client() -> Result<Client, McpError> {
    DB_POOL.get().unwrap().get().await.map_err(internal_err)
}
```

**Files**: `crates/trios-railway-mcp/src/db.rs` (new), `Cargo.toml` (add `deadpool-postgres`)

**Effort**: 1 hour  
**Issue ref**: #62 (bpb_samples DDL blocked by connection instability)

### 1.2 Move CryptoProvider to startup
**Problem**: `rustls::crypto::aws_lc_rs::default_provider().install_default()` called in every [`db_connect()`](crates/trios-railway-mcp/src/tools.rs:799) call. Should be once at server startup.

**Fix**: Move to [`main()`](crates/trios-railway-mcp/src/main.rs:34) before axum serve.

```rust
// main.rs — add before TcpListener::bind
let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
tracing::info!("rustls crypto provider installed");
```

**Files**: `crates/trios-railway-mcp/src/main.rs`, `crates/trios-railway-mcp/src/tools.rs`  
**Effort**: 15 min

### 1.3 Add unit tests for MCP tools (L4 compliance)
**Problem**: `trios-railway-mcp` has **zero** tests. L4 says "new code carries new tests".

**Fix**: Add tests for:
- `neon_url()` — env var present/missing
- URL cleaning (strip `channel_binding`, keep `sslmode`)
- `load_accounts()` — parses 4 accounts from env
- `build_client_for_project()` — whitelist validation
- Request struct serialization/deserialization

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_channel_binding_keeps_sslmode() {
        let url = "postgres://u:p@h/d?sslmode=require&channel_binding=require";
        let cleaned: String = url.split('&')
            .filter(|p| !p.starts_with("channel_binding="))
            .collect::<Vec<_>>()
            .join("&");
        assert!(cleaned.contains("sslmode=require"));
        assert!(!cleaned.contains("channel_binding"));
    }

    #[test]
    fn allowed_project_ids_contains_igla() {
        assert!(ALLOWED_PROJECT_IDS.contains(&"e4fe33bb-3b09-4842-9782-7d2dea1abc9b"));
    }
}
```

**Files**: `crates/trios-railway-mcp/src/tools.rs`  
**Effort**: 1 hour

### 1.4 Remove dead code annotations
**Problem**: `#[allow(dead_code)]` on [`neon_url()`](crates/trios-railway-mcp/src/tools.rs:779), [`BatchRedeployRequest`](crates/trios-railway-mcp/src/tools.rs:132), [`ExperimentInsertRequest`](crates/trios-railway-mcp/src/tools.rs:142). These ARE used — the annotations suppress false positives from rmcp's macro expansion.

**Fix**: Either remove annotations (if clippy is clean without them) or add module-level `#[allow(dead_code)]` once.

**Files**: `crates/trios-railway-mcp/src/tools.rs`  
**Effort**: 5 min

---

## Phase 2: Missing Tools (P0 — 4-5 hours)

### 2.1 `railway_service_logs` — Container stderr reader
**Impact**: 🔴 CRITICAL — 348 dead workers are undiagnosable without logs

**Implementation**: Use Railway GraphQL `deploymentLogs` subscription or REST proxy.

```rust
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ServiceLogsRequest {
    /// Service UUID
    pub service: String,
    /// Account index (0-3)
    pub account: u8,
    /// Number of log lines to fetch (default 200)
    #[serde(default = "default_tail")]
    pub tail: Option<u32>,
}

#[tool(description = "Fetch recent container logs for a Railway service. Returns stderr/stdout for diagnosing crash loops.")]
async fn railway_service_logs(
    &self,
    Parameters(req): Parameters<ServiceLogsRequest>,
) -> Result<CallToolResult, McpError> {
    // Use Railway GraphQL: query { deploymentLogs(deploymentId, filter, limit) }
    // or REST: GET /project/{pid}/service/{sid}/env/{eid}/logs
}
```

**Railway API**: `query { deployments(input: {projectId, environmentId, serviceId}) { edges { node { id logs { edges { node { message timestamp severity } } } } } } }`

**Files**: `crates/trios-railway-core/src/queries.rs` (add `QUERY_LOGS`), `crates/trios-railway-mcp/src/tools.rs`  
**Effort**: 2 hours  
**Issue ref**: #78 (fleet tools)

### 2.2 `experiment_queue_update` — Requeue/modify experiments
**Impact**: 🔴 P0 — stuck/failed experiments need requeuing; priority changes needed for gardener

```rust
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ExperimentQueueUpdateRequest {
    /// Experiment ID to update
    pub id: i64,
    /// New status (pending, running, done, failed, pruned)
    #[serde(default)]
    pub status: Option<String>,
    /// New priority (0-100)
    #[serde(default)]
    pub priority: Option<i32>,
    /// Prune reason (for status=pruned)
    #[serde(default)]
    pub prune_reason: Option<String>,
    /// Clear worker_id (requeue)
    #[serde(default)]
    pub clear_worker: Option<bool>,
}

#[tool(description = "Update experiment queue entry: change status, priority, requeue stuck experiments.")]
async fn experiment_queue_update(
    &self,
    Parameters(req): Parameters<ExperimentQueueUpdateRequest>,
) -> Result<CallToolResult, McpError> {
    let client = db_client().await?;
    // UPDATE experiment_queue SET ... WHERE id = $1
    // Audit via experience_append
}
```

**Files**: `crates/trios-railway-mcp/src/tools.rs`  
**Effort**: 1 hour

### 2.3 `service_variable_upsert` — Set env vars on existing service
**Impact**: 🔴 P0 — currently done manually via curl; needed for NEON_DATABASE_URL injection on new services

```rust
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct VariableUpsertRequest {
    /// Account index (0-3)
    pub account: u8,
    /// Service UUID
    pub service: String,
    /// Variable name
    pub name: String,
    /// Variable value
    pub value: String,
}

#[tool(description = "Upsert an environment variable on an existing Railway service. Triggers no redeploy (call service_redeploy separately).")]
async fn service_variable_upsert(
    &self,
    Parameters(req): Parameters<VariableUpsertRequest>,
) -> Result<CallToolResult, McpError> {
    // Uses M::variable_upsert from core
}
```

**Files**: `crates/trios-railway-mcp/src/tools.rs`  
**Effort**: 45 min

### 2.4 `service_batch_deploy` — Bulk create workers
**Impact**: 🟡 P2 — currently 1 deploy = 1 call; needed for scaling

```rust
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct BatchDeployRequest {
    /// Account index (0-3)
    pub account: u8,
    /// List of service configs (name + vars)
    pub services: Vec<BatchDeployService>,
}

#[tool(description = "Create and deploy multiple services in one call. Each service gets NEON_DATABASE_URL auto-injected.")]
async fn service_batch_deploy(
    &self,
    Parameters(req): Parameters<BatchDeployRequest>,
) -> Result<CallToolResult, McpError> {
    // Iterate services, create + set image + upsert vars + redeploy
}
```

**Files**: `crates/trios-railway-mcp/src/tools.rs`  
**Effort**: 1.5 hours

### 2.5 `bpb_samples_query` — Trajectory analysis
**Impact**: 🟠 P1 — gardener needs this for plateau detection

```rust
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct BpbSamplesQueryRequest {
    /// Canon name filter
    pub canon_name: String,
    /// Max rows (default 100)
    #[serde(default = "default_limit")]
    pub limit: Option<i32>,
}

#[tool(description = "Query bpb_samples table for trajectory analysis. Returns step, bpb, timestamp for a given experiment.")]
async fn bpb_samples_query(
    &self,
    Parameters(req): Parameters<BpbSamplesQueryRequest>,
) -> Result<CallToolResult, McpError> {
    let client = db_client().await?;
    // SELECT step, bpb, created_at FROM bpb_samples WHERE canon_name = $1 ORDER BY step LIMIT $2
}
```

**Files**: `crates/trios-railway-mcp/src/tools.rs`  
**Effort**: 45 min  
**Issue ref**: #62 (bpb_samples DDL)

---

## Phase 3: Architecture Hardening (P1 — 3-4 hours)

### 3.1 Split `tools.rs` into modules
**Problem**: [`tools.rs`](crates/trios-railway-mcp/src/tools.rs) is 825 lines. Hard to navigate, hard to review.

**Fix**: Split into:
```
crates/trios-railway-mcp/src/
├── main.rs          — axum server, port binding
├── tools.rs         — TriosRailwayMcp struct + tool_router
├── tools/
│   ├── mod.rs       — shared helpers (build_client, internal_err)
│   ├── railway.rs   — service_list, deploy, redeploy, delete, batch_redeploy
│   ├── fleet.rs     — fleet_health, seed_list
│   ├── database.rs  — experiment_queue_status, experiment_queue_insert, worker_status, bpb_samples_query
│   ├── audit.rs     — experience_append, audit_migrate_sql
│   └── types.rs     — all request/response structs
└── db.rs            — connection pool, neon_url, db_connect
```

**Effort**: 2 hours

### 3.2 Add bearer auth to MCP endpoint
**Problem**: MCP endpoint is fully public — anyone can call deploy/delete/redeploy.

**Fix**: Add `Authorization: Bearer <MCP_API_KEY>` header check in axum middleware.

```rust
// main.rs — add middleware
async fn auth_middleware(
    State(key): State<String>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(auth) = req.headers().get("authorization") {
        if auth.to_str().unwrap_or("") == format!("Bearer {}", key) {
            return Ok(next.run(req).await);
        }
    }
    // Also allow unauthenticated access for health/tools/list
    if req.uri().path() == "/health" {
        return Ok(next.run(req).await);
    }
    Err(StatusCode::UNAUTHORIZED)
}
```

**Files**: `crates/trios-railway-mcp/src/main.rs`  
**Effort**: 1 hour  
**Issue ref**: #72 (bearer auth)

### 3.3 Rate limiting on mutations
**Problem**: No rate limiting — a single MCP client can trigger 100+ redeployments/sec.

**Fix**: Add `governor` or simple token bucket per session.

```rust
use governor::{Quota, RateLimiter};
// Max 10 mutations per minute per session
let quota = Quota::per_minute(nonzero!(10u32));
```

**Files**: `crates/trios-railway-mcp/src/main.rs`  
**Effort**: 1 hour

### 3.4 Kill switch (MCP_FROZEN env var)
**Problem**: No way to freeze all mutations without redeploying.

**Fix**: Check `MCP_FROZEN` env var at the start of every mutation tool.

```rust
fn check_frozen() -> Result<(), McpError> {
    if std::env::var("MCP_FROZEN").as_deref() == Ok("true") {
        return Err(McpError::internal_error(
            "MCP_FROZEN=true — all mutations are suspended", None
        ));
    }
    Ok(())
}
```

**Effort**: 30 min

---

## Phase 4: Observability (P1 — 2-3 hours)

### 4.1 Tracing spans per tool call
**Problem**: No structured logging — can't trace which tool was called, how long it took, or if it failed.

**Fix**: Add `tracing::instrument` to each tool method.

```rust
#[tool(description = "...")]
#[tracing::instrument(skip(self))]
async fn railway_service_deploy(
    &self,
    Parameters(req): Parameters<DeployRequest>,
) -> Result<CallToolResult, McpError> {
    tracing::info!(name = %req.name, "deploying service");
    // ...
}
```

**Effort**: 1 hour

### 4.2 Health check endpoint
**Problem**: No `/health` endpoint — Railway can't detect if the server is responsive.

**Fix**: Add `GET /health` returning `{"status": "ok", "tools": 17, "accounts": 4}`.

**Effort**: 30 min

### 4.3 Metrics endpoint (optional)
**Fix**: Add `GET /metrics` with Prometheus-format counters for tool calls, errors, latency.

**Effort**: 1 hour

---

## Phase 5: Code Quality (P2 — 2-3 hours)

### 5.1 Remove hardcoded constants
**Problem**: [`IGLA_PROJECT_ID`](crates/trios-railway-mcp/src/tools.rs:24), [`DEFAULT_TRAINER_IMAGE`](crates/trios-railway-mcp/src/tools.rs:26) are hardcoded. Should be configurable via env vars.

**Fix**: Load from env with fallback:
```rust
fn default_project() -> String {
    std::env::var("DEFAULT_PROJECT_ID")
        .unwrap_or_else(|_| "e4fe33bb-3b09-4842-9782-7d2dea1abc9b".to_string())
}
```

**Effort**: 30 min

### 5.2 Error type unification
**Problem**: Mixed error handling — `McpError::internal_error(format!(...))` everywhere. No structured error types.

**Fix**: Create `enum ToolError` with variants for each failure mode.

**Effort**: 1 hour

### 5.3 Improve `experiment_queue_insert` seed validation
**Problem**: The tool's doc says "Only sanctioned seeds are allowed" but doesn't validate client-side. The SQL trigger catches violations but gives a raw Postgres error.

**Fix**: Add client-side seed validation before INSERT:
```rust
const SANCTIONED_SEEDS: &[i32] = &[42, 43, 44, 1597, 2584, 4181, 6765, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10009, 10010, 10946];

if !SANCTIONED_SEEDS.contains(&params.seed) {
    return Err(McpError::invalid_params(
        format!("seed {} not in sanctioned_seeds. Allowed: {:?}", params.seed, SANCTIONED_SEEDS),
        None,
    ));
}
```

**Effort**: 30 min

### 5.4 Improve `service_batch_redeploy` with concurrency
**Problem**: [`service_batch_redeploy`](crates/trios-railway-mcp/src/tools.rs:601) redeployes services sequentially. 48 services × ~1s = 48s.

**Fix**: Use `futures::future::join_all` with a concurrency limiter:
```rust
let futures = services.iter().map(|s| {
    let sid = ServiceId::from(s.id.as_str());
    let client = client.clone();
    async move {
        M::service_redeploy(&client, &sid, &eid).await
    }
});
let results = futures::future::join_all(futures).await;
```

**Effort**: 30 min

---

## Phase 6: CI/CD (P2 — 2-3 hours)

### 6.1 GitHub Actions CI pipeline
**Problem**: No CI — tests, clippy, fmt are run manually.

**Fix**: Add `.github/workflows/ci.yml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo fmt --check
      - run: cargo clippy --all-targets -- -D warnings
      - run: cargo test --workspace
```

**Effort**: 1 hour  
**Issue ref**: #57 (GHCR pipeline)

### 6.2 Automated Docker build + push
**Problem**: Manual `docker build` + `docker push` + Railway redeploy.

**Fix**: Add GitHub Actions workflow triggered by tags `v*`:
```yaml
- run: docker build -f Dockerfile.mcp -t ghcr.io/ghashtag/trios-railway-mcp:${{ github.ref_name }} .
- run: docker push ghcr.io/ghashtag/trios-railway-mcp:${{ github.ref_name }}
```

**Effort**: 1 hour

### 6.3 Railway auto-deploy on GHCR push
**Problem**: After pushing a new image, must manually redeploy the MCP service.

**Fix**: Add GitHub Actions step that calls Railway GraphQL `serviceInstanceRedeploy` after push.

**Effort**: 30 min

---

## Phase 7: Documentation (P2 — 1-2 hours)

### 7.1 MCP_TOOL_CATALOG.md
**Problem**: No tool documentation for agents/operators.

**Fix**: Create `docs/MCP_TOOL_CATALOG.md` listing all 17 tools with:
- Name, description, parameters
- Example request/response
- Error cases

**Issue ref**: #73

### 7.2 ARCHITECTURE.md
**Problem**: No architecture doc for new contributors.

**Fix**: Create `docs/ARCHITECTURE.md` with:
- Ring layout diagram
- Data flow (MCP → Railway API, MCP → Neon)
- Auth model (4 accounts, token modes)
- Deployment topology

### 7.3 Runbook
**Fix**: Create `docs/RUNBOOK.md` with:
- How to add a new tool
- How to add a new account
- How to debug dead workers
- How to inject env vars fleet-wide

---

## Priority Matrix

| Phase | Tasks | Impact | Effort | Blocks |
|-------|-------|--------|--------|--------|
| **Phase 1** | Critical fixes | 🔴 | 2-3h | L4 compliance, Neon stability |
| **Phase 2** | Missing tools | 🔴 | 4-5h | Autonomous gardener |
| **Phase 3** | Architecture | 🟠 | 3-4h | Production hardening |
| **Phase 4** | Observability | 🟠 | 2-3h | Operational visibility |
| **Phase 5** | Code quality | 🟡 | 2-3h | Maintainability |
| **Phase 6** | CI/CD | 🟡 | 2-3h | Deployment velocity |
| **Phase 7** | Documentation | 🟢 | 1-2h | Onboarding |

**Total estimated effort**: 16-23 hours

---

## Recommended Execution Order

1. **Phase 1.2** (CryptoProvider to startup) — 15 min, immediate
2. **Phase 1.4** (Remove dead code annotations) — 5 min, immediate
3. **Phase 1.1** (Neon connection pooling) — 1h, unblocks reliability
4. **Phase 1.3** (Unit tests) — 1h, L4 compliance
5. **Phase 2.3** (service_variable_upsert) — 45 min, most used manually
6. **Phase 2.1** (railway_service_logs) — 2h, unblocks debugging
7. **Phase 2.2** (experiment_queue_update) — 1h, unblocks gardener
8. **Phase 2.5** (bpb_samples_query) — 45 min, unblocks gardener
9. **Phase 3.4** (Kill switch) — 30 min, safety net
10. **Phase 4.2** (Health check) — 30 min, Railway needs this
11. **Phase 3.1** (Split tools.rs) — 2h, maintainability
12. **Phase 3.2** (Bearer auth) — 1h, security
13. **Phase 5.3** (Seed validation) — 30 min, UX improvement
14. **Phase 5.4** (Concurrent batch redeploy) — 30 min, performance
15. **Phase 6.1** (CI pipeline) — 1h, automation
16. **Phase 7.1-7.3** (Documentation) — 1-2h, onboarding

---

## Tool Count After All Phases

| # | Tool | Phase | Status |
|---|------|-------|--------|
| 1 | `railway_service_list` | existing | ✅ |
| 2 | `railway_service_deploy` | existing | ✅ |
| 3 | `railway_service_redeploy` | existing | ✅ |
| 4 | `railway_service_delete` | existing | ✅ |
| 5 | `railway_experience_append` | existing | ✅ |
| 6 | `railway_audit_migrate_sql` | existing | ✅ |
| 7 | `fleet_health` | existing | ✅ |
| 8 | `seed_list` | existing | ✅ |
| 9 | `experiment_queue_status` | existing | ✅ |
| 10 | `worker_status` | existing | ✅ |
| 11 | `service_batch_redeploy` | existing | ✅ |
| 12 | `experiment_queue_insert` | existing | ✅ |
| 13 | `railway_service_logs` | Phase 2.1 | 🆕 |
| 14 | `experiment_queue_update` | Phase 2.2 | 🆕 |
| 15 | `service_variable_upsert` | Phase 2.3 | 🆕 |
| 16 | `service_batch_deploy` | Phase 2.4 | 🆕 |
| 17 | `bpb_samples_query` | Phase 2.5 | 🆕 |

**Total: 17 tools** (12 existing + 5 new)

---

*Agent: GENERAL · Soul: RailRangerOne · φ² + φ⁻² = 3*
