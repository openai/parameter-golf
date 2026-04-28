-- 0001: Railway audit base tables + drift view.
-- All statements are idempotent (IF NOT EXISTS / OR REPLACE).
-- Apply via: psql $NEON_DATABASE_URL -f migrations/0001_railway_audit.sql
--            or: tri-railway audit migrate-sql | psql $NEON_DATABASE_URL
--            or: tri-railway audit migrate

CREATE TABLE IF NOT EXISTS railway_projects (
    id              text PRIMARY KEY,
    name            text NOT NULL,
    workspace       text NOT NULL,
    default_env_id  text NOT NULL,
    observed_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS railway_services (
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
);

CREATE TABLE IF NOT EXISTS railway_audit_runs (
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
);

CREATE TABLE IF NOT EXISTS railway_audit_events (
    run_id          uuid REFERENCES railway_audit_runs(id) ON DELETE CASCADE,
    service_id      text,
    code            text NOT NULL,
    severity        text NOT NULL CHECK (severity IN ('warn','error','info')),
    detail          jsonb NOT NULL,
    triplet         text,
    PRIMARY KEY (run_id, service_id, code)
);

CREATE OR REPLACE VIEW v_railway_drift_open AS
    SELECT e.code, e.severity, s.name AS service, e.triplet, r.started_at
    FROM railway_audit_events e
    JOIN railway_audit_runs   r ON r.id = e.run_id
    LEFT JOIN railway_services s ON s.id = e.service_id
    WHERE r.id = (SELECT id FROM railway_audit_runs ORDER BY started_at DESC LIMIT 1);

CREATE TABLE IF NOT EXISTS igla_race_trials (
    id          bigserial PRIMARY KEY,
    seed        integer   NOT NULL,
    bpb         double precision NOT NULL,
    step        integer   NOT NULL,
    image_sha   text      NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS igla_race_trials_seed_idx
    ON igla_race_trials (seed, step);

CREATE TABLE IF NOT EXISTS gardener_runs (
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
);

CREATE INDEX IF NOT EXISTS gardener_runs_ts_idx
    ON gardener_runs (ts DESC);
