//! AU-02 — Neon `audit_event` writer.
//!
//! Writes one row to `igla_race_trials` and returns the inserted `row_id`.
//! The DDL is owned by `migrations.rs` (issue #6, already in main).
//!
//! # Usage
//!
//! ```no_run
//! use trios_railway_audit::event::audit_event;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let row_id = audit_event(
//!         100,       // seed
//!         1.83,      // bpb
//!         2000,      // step
//!         "abc123",  // image sha (first 8 chars is fine)
//!         &std::env::var("NEON_DATABASE_URL")?,
//!     ).await?;
//!     println!("inserted row_id={row_id}");
//!     Ok(())
//! }
//! ```
//!
//! # Table schema (already applied via `tri-railway audit migrate-sql`)
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS igla_race_trials (
//!     id         BIGSERIAL PRIMARY KEY,
//!     seed       INTEGER   NOT NULL,
//!     bpb        DOUBLE PRECISION NOT NULL,
//!     step       INTEGER   NOT NULL,
//!     image_sha  TEXT      NOT NULL,
//!     recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
//! );
//! ```

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

/// Opaque row identifier returned after a successful insert.
pub type RowId = i64;

/// JSON artifact emitted to stdout by `tri railway audit run --neon`.
///
/// Allows the hourly watchdog to parse the result without re-querying Neon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditArtifact {
    /// UNIX timestamp of the write.
    pub ts: i64,
    /// Seed that was audited.
    pub seed: i32,
    /// BPB value recorded.
    pub bpb: f64,
    /// Training step at the time of the audit.
    pub step: i32,
    /// First 8 chars of the Docker image SHA.
    pub image_sha: String,
    /// Neon-assigned row id.
    pub row_id: RowId,
    /// Target BPB used for gate evaluation.
    pub target_bpb: f64,
    /// Whether this row contributes to Gate-2 PASS.
    pub gate2_contribution: bool,
}

impl AuditArtifact {
    /// Build an artifact from writer output.
    #[must_use]
    pub fn new(
        seed: i32,
        bpb: f64,
        step: i32,
        image_sha: &str,
        row_id: RowId,
        target_bpb: f64,
    ) -> Self {
        Self {
            ts: Utc::now().timestamp(),
            seed,
            bpb,
            step,
            image_sha: image_sha[..image_sha.len().min(8)].to_string(),
            row_id,
            target_bpb,
            gate2_contribution: bpb < target_bpb,
        }
    }
}

/// Insert one audit row into `igla_race_trials` and return the row id.
///
/// # Arguments
///
/// * `seed`      – integer seed (100, 101, 102, …)
/// * `bpb`       – bits-per-byte from the trainer log
/// * `step`      – training step at which BPB was measured
/// * `image_sha` – Docker image digest / git sha (first 8 chars minimum)
/// * `neon_url`  – full `postgres://` connection string (`NEON_DATABASE_URL`)
///
/// # Errors
///
/// Returns `Err` on connection failure, SQL error, or missing env.
/// Never silently swallows errors (R5).
pub async fn audit_event(
    seed: i32,
    bpb: f64,
    step: i32,
    image_sha: &str,
    neon_url: &str,
) -> Result<RowId> {
    let (client, connection) = tokio_postgres::connect(neon_url, tokio_postgres::NoTls)
        .await
        .context("connect to Neon")?;

    // Drive the connection in a background task (tokio-postgres pattern).
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::error!("neon connection error: {e}");
        }
    });

    let row = client
        .query_one(
            "INSERT INTO igla_race_trials (seed, bpb, step, image_sha) \
             VALUES ($1, $2, $3, $4) \
             RETURNING id",
            &[&seed, &bpb, &step, &image_sha],
        )
        .await
        .context("insert igla_race_trials")?;

    let row_id: RowId = row.get(0);
    tracing::info!(seed, bpb, step, row_id, "audit_event written to Neon");
    Ok(row_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_gate2_contribution_true_below_target() {
        let a = AuditArtifact::new(100, 1.83, 2000, "abc123def", 42, 1.85);
        assert!(a.gate2_contribution);
        assert_eq!(a.image_sha, "abc123de");
        assert_eq!(a.seed, 100);
        assert_eq!(a.row_id, 42);
    }

    #[test]
    fn artifact_gate2_contribution_false_above_target() {
        let a = AuditArtifact::new(101, 2.10, 1500, "xyz", 43, 1.85);
        assert!(!a.gate2_contribution);
    }

    #[test]
    fn artifact_image_sha_truncated_to_8() {
        let a = AuditArtifact::new(102, 1.50, 3000, "0123456789abcdef", 1, 1.85);
        assert_eq!(a.image_sha.len(), 8);
        assert_eq!(a.image_sha, "01234567");
    }

    #[test]
    fn artifact_image_sha_short_kept_as_is() {
        let a = AuditArtifact::new(100, 1.50, 100, "abc", 1, 1.85);
        assert_eq!(a.image_sha, "abc");
    }

    #[test]
    fn artifact_serialises_to_json() {
        let a = AuditArtifact::new(100, 1.83, 2000, "deadbeef", 7, 1.85);
        let json = serde_json::to_string(&a).expect("serialize");
        assert!(json.contains("\"gate2_contribution\":true"));
        assert!(json.contains("\"seed\":100"));
        assert!(json.contains("\"row_id\":7"));
    }
}
