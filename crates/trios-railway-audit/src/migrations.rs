//! Idempotent Neon DDL for `tri railway audit migrate`.
//!
//! Tables live in the existing `public` schema next to the IGLA RACE
//! ledger so a single Neon role can read both halves.

/// Returns a slice of statements to run in order. All statements are
/// `CREATE … IF NOT EXISTS`, so re-running is a no-op.
#[must_use]
pub fn ddl_statements() -> &'static [&'static str] {
    DDL
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
}
