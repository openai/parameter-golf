//! Atomic experiment claim against `experiment_queue`.
//!
//! The single SQL statement below is the heart of the pull loop. It
//! finds the highest-priority pending row, locks it with
//! `SKIP LOCKED` so concurrent workers never collide, and flips its
//! status to `claimed` in one round-trip.
//!
//! R5: any RowsAffected mismatch surfaces as an error — never silent.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// One claimed experiment row, hydrated from `experiment_queue`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimedExperiment {
    pub id: i64,
    pub canon_name: String,
    pub config: serde_json::Value,
    pub seed: i32,
    pub steps_budget: i32,
    pub account: String,
    pub priority: i32,
}

/// SQL fragment exposed for testability. The double `WITH` is required
/// because Postgres does not support `FOR UPDATE` in a top-level
/// `UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED)` for
/// readability — we use a CTE for the same effect.
pub const CLAIM_SQL: &str = r"
    WITH pick AS (
        SELECT id FROM experiment_queue
        WHERE status = 'pending' AND account = $2
        ORDER BY priority DESC, id ASC
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    )
    UPDATE experiment_queue q
    SET status = 'claimed',
        worker_id = $1,
        claimed_at = now()
    FROM pick
    WHERE q.id = pick.id
    RETURNING q.id, q.canon_name, q.config_json, q.seed,
              q.steps_budget, q.account, q.priority
";

/// Atomic claim by `(worker_id, account)`. Returns `None` when the
/// queue is empty for that account — callers sleep and retry.
pub async fn claim_next(
    client: &tokio_postgres::Client,
    worker_id: Uuid,
    account: &str,
) -> Result<Option<ClaimedExperiment>> {
    let row = client
        .query_opt(CLAIM_SQL, &[&worker_id, &account])
        .await
        .with_context(|| "claim_next: SKIP LOCKED query failed")?;
    let Some(row) = row else { return Ok(None) };
    Ok(Some(ClaimedExperiment {
        id: row.get(0),
        canon_name: row.get(1),
        config: row.get(2),
        seed: row.get(3),
        steps_budget: row.get(4),
        account: row.get(5),
        priority: row.get(6),
    }))
}

/// Promote a claimed row to `running` once the trainer has actually
/// started consuming budget. Separates the two states so a crashed
/// worker between `claim` and `start` is recoverable by gardener
/// stale-claim eviction.
pub async fn mark_running(client: &tokio_postgres::Client, id: i64) -> Result<()> {
    let n = client
        .execute(
            "UPDATE experiment_queue \
             SET status='running', started_at=now() \
             WHERE id=$1 AND status='claimed'",
            &[&id],
        )
        .await
        .with_context(|| "mark_running")?;
    if n != 1 {
        anyhow::bail!(
            "mark_running: expected 1 row affected, got {n} — race or stale claim"
        );
    }
    Ok(())
}

/// Mark an experiment `done` with its final BPB/step.
pub async fn mark_done(
    client: &tokio_postgres::Client,
    id: i64,
    final_bpb: f64,
    final_step: i32,
) -> Result<()> {
    let n = client
        .execute(
            "UPDATE experiment_queue \
             SET status='done', finished_at=now(), final_bpb=$2, final_step=$3 \
             WHERE id=$1 AND status IN ('running','claimed')",
            &[&id, &final_bpb, &final_step],
        )
        .await
        .with_context(|| "mark_done")?;
    if n != 1 {
        anyhow::bail!(
            "mark_done: expected 1 row affected, got {n} — concurrent gardener prune?"
        );
    }
    Ok(())
}

/// Mark an experiment `pruned` with an early-stop reason and the BPB
/// that triggered the prune.
pub async fn mark_pruned(
    client: &tokio_postgres::Client,
    id: i64,
    reason: &str,
    early_stop_bpb: f64,
) -> Result<()> {
    let n = client
        .execute(
            "UPDATE experiment_queue \
             SET status='pruned', finished_at=now(), \
                 prune_reason=$2, early_stop_bpb=$3 \
             WHERE id=$1 AND status IN ('running','claimed')",
            &[&id, &reason, &early_stop_bpb],
        )
        .await
        .with_context(|| "mark_pruned")?;
    if n != 1 {
        anyhow::bail!("mark_pruned: expected 1 row affected, got {n}");
    }
    Ok(())
}

/// Mark an experiment `failed` (trainer crashed / unrecoverable).
pub async fn mark_failed(
    client: &tokio_postgres::Client,
    id: i64,
    reason: &str,
) -> Result<()> {
    let _ = client
        .execute(
            "UPDATE experiment_queue \
             SET status='failed', finished_at=now(), prune_reason=$2 \
             WHERE id=$1 AND status IN ('running','claimed')",
            &[&id, &reason],
        )
        .await
        .with_context(|| "mark_failed")?;
    Ok(())
}

/// Release a claimed/running row back to `pending` (e.g. graceful
/// shutdown). Idempotent — safe even if already terminal.
pub async fn release(client: &tokio_postgres::Client, id: i64) -> Result<()> {
    let _ = client
        .execute(
            "UPDATE experiment_queue \
             SET status='pending', worker_id=NULL, claimed_at=NULL, started_at=NULL \
             WHERE id=$1 AND status IN ('claimed','running')",
            &[&id],
        )
        .await
        .with_context(|| "release")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_sql_uses_skip_locked() {
        assert!(CLAIM_SQL.contains("FOR UPDATE SKIP LOCKED"));
    }

    #[test]
    fn claim_sql_filters_by_status_pending() {
        assert!(CLAIM_SQL.contains("status = 'pending'"));
    }

    #[test]
    fn claim_sql_orders_by_priority_desc_then_id_asc() {
        // ADR-0081 ONE-SHOT BRIEF (#81) mandates `priority DESC, id ASC` —
        // higher numeric priority means higher precedence. ALPHA's runner
        // claimed exp 31 (priority=95) over the priority=0 Fibonacci probes,
        // confirming this ordering is canon.
        assert!(CLAIM_SQL.contains("ORDER BY priority DESC, id ASC"));
    }

    #[test]
    fn claim_sql_returns_canonical_columns() {
        // Ensure the RETURNING list maps positionally to ClaimedExperiment.
        for c in [
            "q.id",
            "q.canon_name",
            "q.config_json",
            "q.seed",
            "q.steps_budget",
            "q.account",
            "q.priority",
        ] {
            assert!(CLAIM_SQL.contains(c), "RETURNING missing {c}");
        }
    }

    #[test]
    fn claim_sql_is_account_scoped() {
        // Each worker filters by its own account so two acc0 workers
        // don't fight an acc1 worker over the same global queue head.
        assert!(CLAIM_SQL.contains("account = $2"));
    }
}
