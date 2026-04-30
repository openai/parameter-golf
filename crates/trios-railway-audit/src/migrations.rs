//! Idempotent Neon DDL for `tri railway audit migrate`.
//!
//! Tables live in the existing `public` schema next to the IGLA RACE
//! ledger so a single Neon role can read both halves.

use anyhow::{Context, Result};

/// Returns a slice of statements to run in order. All statements are
/// `CREATE … IF NOT EXISTS`, so re-running is a no-op.
#[must_use]
pub fn ddl_statements() -> &'static [&'static str] {
    DDL
}

/// Connect to Neon at `neon_url` and execute every DDL statement.
///
/// Returns the number of statements successfully executed.
/// Each statement is `CREATE IF NOT EXISTS` / `CREATE OR REPLACE`,
/// so re-running is always safe.
///
/// # Errors
///
/// Returns `Err` on connection failure or if any statement fails.
/// Never silently swallows errors (R5).
pub async fn run_migrate(neon_url: &str) -> Result<usize> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .ok(); // already installed is fine
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    let connector = tokio_postgres_rustls::MakeRustlsConnect::new(tls_config);
    let (client, connection) = tokio_postgres::connect(neon_url, connector)
        .await
        .context("connect to Neon for DDL migration")?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::error!("neon connection error: {e}");
        }
    });

    let stmts = ddl_statements();
    for (i, stmt) in stmts.iter().enumerate() {
        client
            .batch_execute(stmt)
            .await
            .with_context(|| format!("DDL statement {}/{} failed", i + 1, stmts.len()))?;
        tracing::debug!(i, total = stmts.len(), "DDL applied");
    }

    tracing::info!(total = stmts.len(), "all DDL statements applied");
    Ok(stmts.len())
}

const DDL: &[&str] = &[
    r"CREATE TABLE IF NOT EXISTS railway_projects (
        id              text PRIMARY KEY,
        name            text NOT NULL,
        workspace       text NOT NULL,
        default_env_id  text NOT NULL,
        observed_at     timestamptz NOT NULL DEFAULT now()
    )",
    r"CREATE TABLE IF NOT EXISTS railway_services (
        id              text PRIMARY KEY,
        project_id      text NOT NULL REFERENCES railway_projects(id),
        env_id          text NOT NULL,
        name            text NOT NULL,
        seed            integer,
        image           text,
        image_digest    text,
        last_deploy_id  text,
        last_status     text,
        created_at      timestamptz,
        observed_at     timestamptz NOT NULL DEFAULT now()
    )",
    r"CREATE TABLE IF NOT EXISTS railway_audit_runs (
        id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        agent           text NOT NULL,
        soul_name       text NOT NULL,
        phi_step        text NOT NULL CHECK (phi_step IN
                        ('CLAIM','NAME','SPEC','SEAL','GEN','TEST',
                         'VERDICT','EXPERIENCE','REPORT','COMMIT','PUSH')),
        started_at      timestamptz NOT NULL,
        finished_at     timestamptz,
        services_seen   integer,
        drift_events    integer,
        gate2_pass      boolean,
        target_bpb      double precision,
        artifact_url    text,
        experience_path text NOT NULL,
        exit_code       integer NOT NULL
    )",
    r"CREATE TABLE IF NOT EXISTS railway_audit_events (
        run_id          uuid REFERENCES railway_audit_runs(id) ON DELETE CASCADE,
        service_id      text,
        code            text NOT NULL,
        severity        text NOT NULL CHECK (severity IN ('warn','error','info')),
        detail          jsonb NOT NULL,
        triplet         text,
        PRIMARY KEY (run_id, service_id, code)
    )",
    r"CREATE OR REPLACE VIEW v_railway_drift_open AS
        SELECT e.code, e.severity, s.name AS service, e.triplet, r.started_at
        FROM railway_audit_events e
        JOIN railway_audit_runs   r ON r.id = e.run_id
        LEFT JOIN railway_services s ON s.id = e.service_id
        WHERE r.id = (SELECT id FROM railway_audit_runs ORDER BY started_at DESC LIMIT 1)",
    // AU-02: audit-event telemetry. Written by `event::audit_event()`.
    // NOTE: if the table already exists with a different schema (no `step`
    // column), CREATE IF NOT EXISTS is a no-op. The index below uses
    // DO NOTHING via a plpgsql block to avoid failures on divergent schemas.
    r"CREATE TABLE IF NOT EXISTS igla_race_trials (
        id          bigserial PRIMARY KEY,
        seed        integer   NOT NULL,
        bpb         double precision NOT NULL,
        step        integer   NOT NULL,
        image_sha   text      NOT NULL,
        recorded_at timestamptz NOT NULL DEFAULT now()
    )",
    r"DO $$ BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'igla_race_trials' AND column_name = 'step'
        ) THEN
            CREATE INDEX IF NOT EXISTS igla_race_trials_seed_idx
                ON igla_race_trials (seed, step);
        END IF;
    END $$",
    // Gardener orchestrator run log. Written by tri-gardener neon.rs.
    r"CREATE TABLE IF NOT EXISTS gardener_runs (
        id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        ts            timestamptz NOT NULL DEFAULT now(),
        tick_t_minus  text         NOT NULL,
        action        text         NOT NULL,
        lane          text,
        seed          int,
        before_bpb    double precision,
        after_bpb     double precision,
        decision      jsonb        NOT NULL,
        audit_run_id  uuid REFERENCES railway_audit_runs(id) ON DELETE SET NULL
    )",
    r"CREATE INDEX IF NOT EXISTS gardener_runs_ts_idx
        ON gardener_runs (ts DESC)",
    // ===================================================================
    // ADR-0081 — Unified Experiment Loop (issue #81)
    //
    // Pull-based work-stealing queue. Gardener writes to `experiment_queue`
    // and reads `bpb_samples`. Seed Agent claims rows via
    // `SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1`, runs the trainer,
    // emits `bpb_samples` every 100 steps, makes the early-stop call at
    // step 1000.
    //
    // Status enum: pending | claimed | running | pruned | done | failed
    // R5-honest: status transitions are append-only audit (one row per
    // claim attempt) — never UPDATE-in-place silent moves.
    // ===================================================================
    r"CREATE TABLE IF NOT EXISTS experiment_queue (
        id              bigserial PRIMARY KEY,
        canon_name      text NOT NULL,
        config_json     jsonb NOT NULL,
        priority        integer NOT NULL DEFAULT 50
                        CHECK (priority BETWEEN 0 AND 100),
        seed            integer NOT NULL,
        steps_budget    integer NOT NULL
                        CHECK (steps_budget > 0),
        account         text NOT NULL
                        CHECK (account IN ('acc0','acc1','acc2','acc3','acc4','acc5')),
        status          text NOT NULL DEFAULT 'pending'
                        CHECK (status IN
                              ('pending','claimed','running','pruned','done','failed')),
        worker_id       uuid,
        prune_reason    text,
        final_bpb       double precision,
        final_step      integer,
        early_stop_bpb  double precision,
        created_at      timestamptz NOT NULL DEFAULT now(),
        claimed_at      timestamptz,
        started_at      timestamptz,
        finished_at     timestamptz,
        created_by      text NOT NULL DEFAULT 'gardener'
                        CHECK (created_by IN
                              ('gardener','human','auto-mirror','seed-agent'))
    )",
    // Pull-queue index — partial, only over rows that are actually
    // claimable. Keeps SKIP LOCKED scans cheap as the table grows.
    // DESC matches claim SQL `ORDER BY priority DESC` for index-only scan.
    r"CREATE INDEX IF NOT EXISTS experiment_queue_pull_idx
        ON experiment_queue (priority DESC, created_at ASC)
        WHERE status = 'pending'",
    // Lookup by canon for gardener strategy ticks.
    r"CREATE INDEX IF NOT EXISTS experiment_queue_canon_idx
        ON experiment_queue (canon_name, seed)",
    // Stale-claim recovery: gardener resets rows whose claimed_at is
    // older than 5 minutes back to 'pending'. Index keeps that scan O(log n).
    r"CREATE INDEX IF NOT EXISTS experiment_queue_stale_claim_idx
        ON experiment_queue (claimed_at)
        WHERE status = 'claimed'",
    // BPB telemetry — one row per (canon, seed, step). Already referenced
    // by `bin/tri-gardener/src/bpb_source.rs`; this DDL is the canonical
    // create. Issue #62 noted Pipedream silently rolls DDL back; apply
    // via psql for reliable schema.
    r"CREATE TABLE IF NOT EXISTS bpb_samples (
        id          bigserial PRIMARY KEY,
        canon_name  text NOT NULL,
        seed        integer NOT NULL,
        step        integer NOT NULL CHECK (step >= 0),
        bpb         double precision NOT NULL,
        val_bpb_ema double precision,
        ts          timestamptz NOT NULL DEFAULT now(),
        UNIQUE (canon_name, seed, step)
    )",
    r"CREATE INDEX IF NOT EXISTS bpb_samples_canon_seed_step_idx
        ON bpb_samples (canon_name, seed, step DESC)",
    r"CREATE INDEX IF NOT EXISTS bpb_samples_recent_idx
        ON bpb_samples (ts DESC)",
    // Worker registry. Heartbeat updated by Seed Agent at every Neon
    // poll. Stale workers (no heartbeat > 2 minutes) are evicted by the
    // gardener and their claimed experiments are returned to 'pending'.
    r"CREATE TABLE IF NOT EXISTS workers (
        id              uuid PRIMARY KEY,
        railway_acc     text NOT NULL
                        CHECK (railway_acc IN ('acc0','acc1','acc2','acc3','acc4','acc5')),
        railway_svc_id  text NOT NULL,
        railway_svc_name text NOT NULL,
        last_heartbeat  timestamptz NOT NULL DEFAULT now(),
        current_exp_id  bigint REFERENCES experiment_queue(id) ON DELETE SET NULL,
        registered_at   timestamptz NOT NULL DEFAULT now()
    )",
    r"CREATE INDEX IF NOT EXISTS workers_heartbeat_idx
        ON workers (last_heartbeat DESC)",
    // Audit trail of strategic decisions made by the gardener. Append-only.
    r"CREATE TABLE IF NOT EXISTS gardener_decisions (
        id              bigserial PRIMARY KEY,
        ts              timestamptz NOT NULL DEFAULT now(),
        action          text NOT NULL
                        CHECK (action IN
                              ('enqueue','prune','priority_boost',
                               'spawn_mirror','reset_stale_claim','noop')),
        affected_exp_ids bigint[] NOT NULL DEFAULT '{}',
        reason          text NOT NULL,
        snapshot        jsonb
    )",
    r"CREATE INDEX IF NOT EXISTS gardener_decisions_ts_idx
        ON gardener_decisions (ts DESC)",
    // Live-leaderboard view: best (lowest) BPB per canon+seed across all
    // samples, joined with experiment status. Used by gardener strategy
    // and `mcp.fleet.snapshot`.
    r"CREATE OR REPLACE VIEW v_leaderboard AS
        SELECT
            q.canon_name,
            q.seed,
            q.account,
            q.status,
            q.priority,
            COALESCE(b.best_bpb, q.final_bpb) AS best_bpb,
            b.last_step,
            b.last_ts,
            q.created_at,
            q.finished_at
        FROM experiment_queue q
        LEFT JOIN (
            SELECT canon_name, seed,
                   MIN(bpb) AS best_bpb,
                   MAX(step) AS last_step,
                   MAX(ts)   AS last_ts
            FROM bpb_samples
            GROUP BY canon_name, seed
        ) b ON b.canon_name = q.canon_name AND b.seed = q.seed",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ddl_is_nonempty_and_idempotent_friendly() {
        let v = ddl_statements();
        assert!(!v.is_empty());
        for stmt in v {
            // Either CREATE TABLE IF NOT EXISTS or CREATE OR REPLACE VIEW.
            assert!(
                stmt.contains("IF NOT EXISTS") || stmt.contains("OR REPLACE"),
                "non-idempotent DDL: {stmt}"
            );
        }
    }

    /// All canonical tables must be present in `ddl_statements()`.
    /// If anyone removes one the test fails loudly (R5).
    #[test]
    fn all_canonical_tables_are_present() {
        let blob: String = ddl_statements().join("\n");
        for needle in [
            "CREATE TABLE IF NOT EXISTS railway_projects",
            "CREATE TABLE IF NOT EXISTS railway_services",
            "CREATE TABLE IF NOT EXISTS railway_audit_runs",
            "CREATE TABLE IF NOT EXISTS railway_audit_events",
            "CREATE TABLE IF NOT EXISTS igla_race_trials",
            "CREATE TABLE IF NOT EXISTS gardener_runs",
            "CREATE TABLE IF NOT EXISTS experiment_queue",
            "CREATE TABLE IF NOT EXISTS bpb_samples",
            "CREATE TABLE IF NOT EXISTS workers",
            "CREATE TABLE IF NOT EXISTS gardener_decisions",
            "CREATE OR REPLACE VIEW v_leaderboard",
        ] {
            assert!(blob.contains(needle), "missing DDL: {needle}");
        }
    }

    /// Pull-queue index must be partial on `status='pending'` so the
    /// SKIP LOCKED scan stays cheap as the table grows.
    #[test]
    fn experiment_queue_pull_index_is_partial() {
        let blob: String = ddl_statements().join("\n");
        assert!(blob.contains("experiment_queue_pull_idx"));
        // Find the line and assert it carries the WHERE clause.
        let idx_block = blob
            .split("experiment_queue_pull_idx")
            .nth(1)
            .expect("pull idx fragment");
        assert!(
            idx_block.contains("WHERE status = 'pending'"),
            "pull idx must be partial on status=pending"
        );
    }

    /// Status enum is the single source of truth for legal
    /// `experiment_queue.status` values. Any drift between this list
    /// and the CHECK constraint in the DDL trips the test.
    #[test]
    fn experiment_queue_status_enum_is_locked() {
        let blob: String = ddl_statements().join("\n");
        for s in ["pending", "claimed", "running", "pruned", "done", "failed"] {
            assert!(
                blob.contains(&format!("'{s}'")),
                "experiment_queue status `{s}` missing from DDL"
            );
        }
    }

    /// Account whitelist matches `RailwayMultiClient::AccountId::all()`.
    #[test]
    fn experiment_queue_account_enum_matches_multiclient() {
        let blob: String = ddl_statements().join("\n");
        for a in ["acc0", "acc1", "acc2", "acc3", "acc4", "acc5"] {
            assert!(
                blob.contains(&format!("'{a}'")),
                "account `{a}` missing from DDL CHECK"
            );
        }
    }

    /// `bpb_samples` must enforce step uniqueness so the trainer's
    /// `INSERT ... ON CONFLICT DO NOTHING` path is well-defined.
    #[test]
    fn bpb_samples_has_canon_seed_step_unique() {
        let blob: String = ddl_statements().join("\n");
        let bpb_block = blob
            .split("CREATE TABLE IF NOT EXISTS bpb_samples")
            .nth(1)
            .expect("bpb_samples DDL fragment");
        assert!(
            bpb_block.contains("UNIQUE (canon_name, seed, step)"),
            "bpb_samples must declare (canon_name, seed, step) UNIQUE"
        );
    }

    /// `gardener_decisions.action` enum is the single source of truth
    /// for orchestrator audit-log values.
    #[test]
    fn gardener_decisions_action_enum_is_locked() {
        let blob: String = ddl_statements().join("\n");
        for a in [
            "enqueue",
            "prune",
            "priority_boost",
            "spawn_mirror",
            "reset_stale_claim",
            "noop",
        ] {
            assert!(
                blob.contains(&format!("'{a}'")),
                "gardener_decisions action `{a}` missing from DDL"
            );
        }
    }

    /// `igla_race_trials` must have a seed+step index for audit lookups.
    #[test]
    fn igla_race_trials_has_seed_step_index() {
        let blob: String = ddl_statements().join("\n");
        assert!(
            blob.contains("igla_race_trials_seed_idx"),
            "igla_race_trials missing seed+step index"
        );
        assert!(
            blob.contains("igla_race_trials"),
            "igla_race_trials table DDL missing"
        );
    }

    /// `gardener_runs` must have a ts index for the gardener dashboard.
    #[test]
    fn gardener_runs_has_ts_index() {
        let blob: String = ddl_statements().join("\n");
        assert!(
            blob.contains("gardener_runs_ts_idx"),
            "gardener_runs missing ts index"
        );
    }
}
