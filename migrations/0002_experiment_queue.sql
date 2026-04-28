-- 0002: ADR-0081 — Unified Experiment Loop (issue #81).
-- Pull-based work-stealing queue. Gardener writes to experiment_queue
-- and reads bpb_samples. Seed Agent claims rows via
-- SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1, runs the trainer,
-- emits bpb_samples every 100 steps, makes the early-stop call at
-- step 1000.
--
-- All statements are idempotent (IF NOT EXISTS / OR REPLACE).
-- Apply via: psql $NEON_DATABASE_URL -f migrations/0002_experiment_queue.sql
--            or: tri-railway audit migrate-sql | psql $NEON_DATABASE_URL
--            or: tri-railway audit migrate

-- Status enum: pending | claimed | running | pruned | done | failed
-- R5-honest: status transitions are append-only audit (one row per
-- claim attempt) — never UPDATE-in-place silent moves.
CREATE TABLE IF NOT EXISTS experiment_queue (
    id              bigserial PRIMARY KEY,
    canon_name      text NOT NULL,
    config_json     jsonb NOT NULL,
    priority        integer NOT NULL DEFAULT 50
                    CHECK (priority BETWEEN 0 AND 100),
    seed            integer NOT NULL,
    steps_budget    integer NOT NULL
                    CHECK (steps_budget > 0),
    account         text NOT NULL
                    CHECK (account IN ('acc0','acc1','acc2','acc3')),
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
);

-- Pull-queue index — partial, only over rows that are actually
-- claimable. Keeps SKIP LOCKED scans cheap as the table grows.
CREATE INDEX IF NOT EXISTS experiment_queue_pull_idx
    ON experiment_queue (priority ASC, created_at ASC)
    WHERE status = 'pending';

-- Lookup by canon for gardener strategy ticks.
CREATE INDEX IF NOT EXISTS experiment_queue_canon_idx
    ON experiment_queue (canon_name, seed);

-- Stale-claim recovery: gardener resets rows whose claimed_at is
-- older than 5 minutes back to 'pending'. Index keeps that scan O(log n).
CREATE INDEX IF NOT EXISTS experiment_queue_stale_claim_idx
    ON experiment_queue (claimed_at)
    WHERE status = 'claimed';

-- BPB telemetry — one row per (canon, seed, step).
CREATE TABLE IF NOT EXISTS bpb_samples (
    id          bigserial PRIMARY KEY,
    canon_name  text NOT NULL,
    seed        integer NOT NULL,
    step        integer NOT NULL CHECK (step >= 0),
    bpb         double precision NOT NULL,
    val_bpb_ema double precision,
    ts          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (canon_name, seed, step)
);

CREATE INDEX IF NOT EXISTS bpb_samples_canon_seed_step_idx
    ON bpb_samples (canon_name, seed, step DESC);

CREATE INDEX IF NOT EXISTS bpb_samples_recent_idx
    ON bpb_samples (ts DESC);

-- Worker registry. Heartbeat updated by Seed Agent at every Neon
-- poll. Stale workers (no heartbeat > 2 minutes) are evicted by the
-- gardener and their claimed experiments are returned to 'pending'.
CREATE TABLE IF NOT EXISTS workers (
    id              uuid PRIMARY KEY,
    railway_acc     text NOT NULL
                    CHECK (railway_acc IN ('acc0','acc1','acc2','acc3')),
    railway_svc_id  text NOT NULL,
    railway_svc_name text NOT NULL,
    last_heartbeat  timestamptz NOT NULL DEFAULT now(),
    current_exp_id  bigint REFERENCES experiment_queue(id) ON DELETE SET NULL,
    registered_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS workers_heartbeat_idx
    ON workers (last_heartbeat DESC);

-- Audit trail of strategic decisions made by the gardener. Append-only.
CREATE TABLE IF NOT EXISTS gardener_decisions (
    id              bigserial PRIMARY KEY,
    ts              timestamptz NOT NULL DEFAULT now(),
    action          text NOT NULL
                    CHECK (action IN
                          ('enqueue','prune','priority_boost',
                           'spawn_mirror','reset_stale_claim','noop')),
    affected_exp_ids bigint[] NOT NULL DEFAULT '{}',
    reason          text NOT NULL,
    snapshot        jsonb
);

CREATE INDEX IF NOT EXISTS gardener_decisions_ts_idx
    ON gardener_decisions (ts DESC);

-- Live-leaderboard view: best (lowest) BPB per canon+seed across all
-- samples, joined with experiment status. Used by gardener strategy
-- and `mcp.fleet.snapshot`.
CREATE OR REPLACE VIEW v_leaderboard AS
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
    ) b ON b.canon_name = q.canon_name AND b.seed = q.seed;
