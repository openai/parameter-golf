//! BPB telemetry inserts into `bpb_samples`.
//!
//! Idempotent — `(canon_name, seed, step)` is UNIQUE so duplicate
//! inserts (worker retried after transient Neon error) are no-ops.
//!
//! L-R8: every sample also goes to stdout in the canonical
//! `BPB=X.XXXX` form so Railway logs remain parseable by the
//! audit-watchdog.

use anyhow::{Context, Result};

pub async fn push_sample(
    client: &tokio_postgres::Client,
    canon_name: &str,
    seed: i32,
    step: i32,
    bpb: f64,
    val_bpb_ema: Option<f64>,
) -> Result<()> {
    // L-R8 stdout discipline.
    println!("BPB={bpb:.4} seed={seed} step={step} canon={canon_name}");

    client
        .execute(
            "INSERT INTO bpb_samples (canon_name, seed, step, bpb, val_bpb_ema) \
             VALUES ($1, $2, $3, $4, $5) \
             ON CONFLICT (canon_name, seed, step) DO NOTHING",
            &[&canon_name, &seed, &step, &bpb, &val_bpb_ema],
        )
        .await
        .with_context(|| "push_sample insert")?;
    Ok(())
}

/// Read the latest BPB samples for a given experiment. Used by the
/// early-stop decision to fit a power-law over the trajectory so far.
pub async fn read_history(
    client: &tokio_postgres::Client,
    canon_name: &str,
    seed: i32,
    limit: i64,
) -> Result<Vec<(i32, f64)>> {
    let rows = client
        .query(
            "SELECT step, bpb FROM bpb_samples \
             WHERE canon_name = $1 AND seed = $2 \
             ORDER BY step ASC \
             LIMIT $3",
            &[&canon_name, &seed, &limit],
        )
        .await
        .with_context(|| "read_history")?;
    Ok(rows
        .into_iter()
        .map(|r| (r.get::<_, i32>(0), r.get::<_, f64>(1)))
        .collect())
}

// L-R8 stdout-format integration test removed: it referenced
// `trios_railway_core::canon::parse_bpb_line`, but that module does not
// exist in this crate. Re-add once the canon parser lands.
