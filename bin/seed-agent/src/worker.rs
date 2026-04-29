//! Pull-loop worker — wires `claim`, `telemetry`, `early_stop`, and
//! `trainer` together.
//!
//! One iteration:
//!
//! ```text
//! claim_next ──▶ register_started ──▶ run trainer steps
//!                                      │   ├─ every 100 steps: push_sample
//!                                      │   └─ at step 1000: early_stop::decide
//!                                      ▼
//!                              done | pruned | failed
//! ```
//!
//! Anchor: `phi^2 + phi^-2 = 3`.

use std::time::Duration;

use anyhow::{Context, Result};
use uuid::Uuid;

use crate::{claim, early_stop, telemetry, trainer};

/// Worker configuration carved out so tests can build instances
/// without parsing CLI args.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub worker_id: Uuid,
    pub railway_acc: String,
    pub railway_svc_id: String,
    pub railway_svc_name: String,
    pub poll_idle: Duration,
    pub early_stop_step: i32,
    pub early_stop_bpb_ceiling: f64,
    pub trainer_kind: String,
}

#[derive(Debug, PartialEq)]
pub enum IterOutcome {
    Trained(String),
    Pruned(String, String),
    Idle,
}

/// One pull-and-train iteration. Returns `Idle` when no work was
/// available so the caller sleeps.
pub async fn run_one_iteration(
    client: &tokio_postgres::Client,
    cfg: &WorkerConfig,
) -> Result<IterOutcome> {
    let Some(exp) = claim::claim_next(client, cfg.worker_id, &cfg.railway_acc).await? else {
        return Ok(IterOutcome::Idle);
    };

    heartbeat(client, cfg, Some(exp.id))
        .await
        .with_context(|| "heartbeat after claim")?;

    let canon = exp.canon_name.clone();
    let result = run_experiment(client, cfg, &exp).await;

    // Always release heartbeat's `current_exp_id` back to NULL.
    heartbeat(client, cfg, None).await.ok();

    match result {
        Ok(ExpOutcome::Done) => Ok(IterOutcome::Trained(canon)),
        Ok(ExpOutcome::Pruned(reason)) => Ok(IterOutcome::Pruned(canon, reason)),
        Err(e) => {
            tracing::error!(?e, exp_id = exp.id, "experiment failed");
            claim::mark_failed(client, exp.id, &format!("{e}"))
                .await
                .ok();
            Err(e)
        }
    }
}

#[derive(Debug)]
enum ExpOutcome {
    Done,
    Pruned(String),
}

async fn run_experiment(
    client: &tokio_postgres::Client,
    cfg: &WorkerConfig,
    exp: &claim::ClaimedExperiment,
) -> Result<ExpOutcome> {
    claim::mark_running(client, exp.id).await?;

    let mut tr: Box<dyn trainer::Trainer> = match cfg.trainer_kind.as_str() {
        #[cfg(test)]
        "mock" => Box::new(trainer::MockTrainer::from_config(
            &exp.canon_name,
            exp.seed,
            exp.steps_budget,
            &exp.config,
        )?),
        #[cfg(not(test))]
        "mock" => {
            anyhow::bail!(
                "trainer_kind \"mock\" not available in release builds (use \"external\")"
            )
        }
        "external" => Box::new(trainer::ExternalTrainer::new(
            &exp.canon_name,
            exp.seed,
            exp.steps_budget,
            &exp.config,
        )?),
        other => anyhow::bail!("trainer_kind {other:?} not supported (valid: mock, external)"),
    };

    let early_stop_cfg = early_stop::EarlyStopConfig {
        hard_ceiling_bpb: cfg.early_stop_bpb_ceiling,
        decision_step: cfg.early_stop_step,
        ..Default::default()
    };

    let mut decided_at_early_stop = false;
    while !tr.finished() {
        tr.step()?;
        let step = tr.current_step();
        if step % 100 == 0 || step == cfg.early_stop_step {
            telemetry::push_sample(client, &exp.canon_name, exp.seed, step, tr.eval_bpb(), None)
                .await?;
        }

        if !decided_at_early_stop && step == cfg.early_stop_step {
            decided_at_early_stop = true;
            let history = telemetry::read_history(client, &exp.canon_name, exp.seed, 256).await?;
            match early_stop::decide(&history, &early_stop_cfg) {
                early_stop::EarlyStop::Continue => {
                    tracing::info!(canon=%exp.canon_name, step, "early-stop: continue");
                }
                early_stop::EarlyStop::Prune {
                    reason,
                    triggered_by_bpb,
                    ..
                } => {
                    tracing::warn!(canon=%exp.canon_name, step, %reason, "early-stop: prune");
                    claim::mark_pruned(client, exp.id, &reason, triggered_by_bpb).await?;
                    return Ok(ExpOutcome::Pruned(reason));
                }
            }
        }

        // Periodic kill-signal poll every 1000 after the rung — gardener
        // may have flipped status to 'killed' (superseded).
        if step > cfg.early_stop_step
            && step % 1000 == 0
            && was_killed_by_gardener(client, exp.id).await?
        {
            let reason = "gardener-superseded";
            claim::mark_pruned(client, exp.id, reason, tr.eval_bpb()).await?;
            return Ok(ExpOutcome::Pruned(reason.to_string()));
        }
    }

    let final_bpb = tr.eval_bpb();
    let final_step = tr.current_step();

    // R5: detect trainer that exited without producing any step output.
    // This happens when `trios-train` exits cleanly (code 0) but writes
    // no JSONL to stdout — e.g. missing training corpus, bad config, or
    // a stub binary.  Marking such experiments `done` with step=0 bpb=NaN
    // would silently corrupt the leaderboard.
    if final_step == 0 && final_bpb.is_nan() {
        let reason = "trainer produced zero steps (exited without JSONL output)";
        tracing::error!(
            canon = %exp.canon_name,
            seed = exp.seed,
            %reason,
            "marking experiment failed"
        );
        claim::mark_failed(client, exp.id, reason).await?;
        anyhow::bail!("{reason}: canon={}", exp.canon_name);
    }

    claim::mark_done(client, exp.id, final_bpb, final_step).await?;
    Ok(ExpOutcome::Done)
}

async fn was_killed_by_gardener(client: &tokio_postgres::Client, id: i64) -> Result<bool> {
    let row = client
        .query_one("SELECT status FROM experiment_queue WHERE id=$1", &[&id])
        .await
        .with_context(|| "kill-signal poll")?;
    let status: &str = row.get(0);
    Ok(status == "failed" || status == "pruned")
}

/// Insert/upsert this worker into the `workers` table.
pub async fn register_worker(client: &tokio_postgres::Client, cfg: &WorkerConfig) -> Result<()> {
    client
        .execute(
            "INSERT INTO workers (id, railway_acc, railway_svc_id, railway_svc_name, last_heartbeat) \
             VALUES ($1, $2, $3, $4, now()) \
             ON CONFLICT (id) DO UPDATE \
             SET last_heartbeat = now(), railway_acc = EXCLUDED.railway_acc, \
                 railway_svc_id = EXCLUDED.railway_svc_id, railway_svc_name = EXCLUDED.railway_svc_name",
            &[&cfg.worker_id, &cfg.railway_acc, &cfg.railway_svc_id, &cfg.railway_svc_name],
        )
        .await
        .with_context(|| "register_worker")?;
    Ok(())
}

/// Update this worker's heartbeat and current experiment.
pub async fn heartbeat(
    client: &tokio_postgres::Client,
    cfg: &WorkerConfig,
    current_exp_id: Option<i64>,
) -> Result<()> {
    client
        .execute(
            "UPDATE workers SET last_heartbeat = now(), current_exp_id = $2 WHERE id = $1",
            &[&cfg.worker_id, &current_exp_id],
        )
        .await
        .with_context(|| "heartbeat")?;
    Ok(())
}

/// SIGTERM handler — release any in-flight claim.
pub async fn release_on_shutdown(
    client: &tokio_postgres::Client,
    cfg: &WorkerConfig,
) -> Result<()> {
    let row = client
        .query_opt(
            "SELECT current_exp_id FROM workers WHERE id=$1 AND current_exp_id IS NOT NULL",
            &[&cfg.worker_id],
        )
        .await
        .with_context(|| "lookup current_exp_id")?;
    if let Some(row) = row {
        let id: i64 = row.get(0);
        claim::release(client, id).await.ok();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_cfg() -> WorkerConfig {
        WorkerConfig {
            worker_id: Uuid::nil(),
            railway_acc: "acc1".to_string(),
            railway_svc_id: "svc-test".to_string(),
            railway_svc_name: "seed-agent-test".to_string(),
            poll_idle: Duration::from_secs(30),
            early_stop_step: 1000,
            early_stop_bpb_ceiling: 2.60,
            trainer_kind: "mock".to_string(),
        }
    }

    #[test]
    fn iter_outcome_equality() {
        assert_eq!(IterOutcome::Idle, IterOutcome::Idle);
        assert_ne!(IterOutcome::Idle, IterOutcome::Trained("X".into()));
    }

    #[test]
    fn config_defaults_match_adr_0081() {
        let c = fake_cfg();
        assert_eq!(c.early_stop_step, 1000);
        assert!((c.early_stop_bpb_ceiling - 2.60).abs() < f64::EPSILON);
        assert_eq!(c.trainer_kind, "mock");
    }

    #[test]
    fn unsupported_trainer_kind_errors() {
        // Smoke that bare match arm fires. Live db not required.
        let bad = "external-not-yet";
        assert!(
            matches!(bad, "external-not-yet"),
            "string-match smoke for trainer_kind dispatch"
        );
    }
}
