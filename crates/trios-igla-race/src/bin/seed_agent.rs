use anyhow::Result;
use clap::Parser;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{error, info, warn};
use uuid::Uuid;

use trios_igla_race::pull_queue::{ExperimentConfig, PullQueueDb, SelfDecision};

const SELF_CHECK_STEP: i32 = 1000;
const KILL_CHECK_INTERVAL: i32 = 1000;
const REPORT_INTERVAL: i32 = 100;
const ABANDON_GAP: f32 = 2.0;
const PREDICTED_INF_GATE2: f32 = 1.95;
const DEFAULT_TRAINER_BIN: &str =
    "/Users/playom/trios-trainer-igla/target/release/trios-train";
const DEFAULT_WORKDIR: &str = "/Users/playom/trios-trainer-igla";

#[derive(Parser)]
#[command(
    name = "seed-agent",
    about = "ADR-001: Pull-based self-orchestrating trainer worker"
)]
struct Cli {
    #[arg(long, env = "NEON_DATABASE_URL")]
    neon_url: String,
    #[arg(long, env = "RAILWAY_ACC", default_value = "acc0")]
    railway_acc: String,
    #[arg(long, env = "RAILWAY_SERVICE_NAME", default_value = "opencode-seed-agent")]
    railway_svc: String,
    #[arg(long, default_value = DEFAULT_TRAINER_BIN)]
    trainer_bin: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter("seed_agent=info,trios_igla_race=info")
        .init();

    let cli = Cli::parse();
    let worker_id = Uuid::new_v4();

    info!(
        "seed-agent starting | worker={worker_id} | acc={}",
        cli.railway_acc
    );

    let db = PullQueueDb::connect(&cli.neon_url).await?;
    db.health_check().await?;
    info!("Neon health check OK");

    let tables_ok = db.check_table_exists("experiment_queue").await.unwrap_or(false);
    if !tables_ok {
        error!("experiment_queue table not found — run DDL migration first");
        std::process::exit(1);
    }

    db.register_worker(&worker_id, &cli.railway_acc, &cli.railway_svc)
        .await?;
    info!("Worker registered: {worker_id}");

    let _hb = trios_igla_race::pull_queue::spawn_heartbeat(db.clone_handle(), worker_id);

    loop {
        match worker_tick(&db, &worker_id, &cli.trainer_bin).await {
            Ok(true) => {
                info!("experiment completed, pulling next");
            }
            Ok(false) => {
                info!("no experiments in queue, sleeping 30s");
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            }
            Err(e) => {
                error!("worker tick error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            }
        }
    }
}

async fn worker_tick(db: &PullQueueDb, worker_id: &Uuid, trainer_bin: &str) -> Result<bool> {
    let exp = match db.pull_experiment(worker_id).await? {
        Some(e) => e,
        None => return Ok(false),
    };

    info!(
        "pulled experiment: id={} name={} priority={:.2}",
        exp.id, exp.canon_name, exp.priority
    );

    let config = ExperimentConfig::from_json_str(&exp.config_blob)?;
    let steps = exp.steps_budget as usize;
    let seed = exp.seed as u64;
    info!(
        "config: seed={} h={} ctx={} lr={:.4} steps={} (budget={})",
        seed, config.hidden, config.ctx, config.lr, config.steps, steps
    );

    db.mark_running(exp.id).await?;
    db.update_heartbeat(worker_id, Some(exp.id)).await?;

    let mut best_bpb: f32 = f32::MAX;
    let mut last_step: i32 = 0;
    let mut bpb_history: Vec<(i32, f32)> = Vec::new();
    let mut abandoned = false;

    let mut child = tokio::process::Command::new(trainer_bin)
        .arg("--seed")
        .arg(seed.to_string())
        .arg("--steps")
        .arg(steps.to_string())
        .arg("--hidden")
        .arg(config.hidden.to_string())
        .arg("--lr")
        .arg(format!("{:.6}", config.lr))
        .current_dir(DEFAULT_WORKDIR)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to spawn {trainer_bin}: {e}"))?;

    let stdout = child.stdout.take().expect("stdout piped");
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        if let Some(parsed) = parse_step_line(&line) {
            let (step, bpb) = parsed;
            last_step = step;

            if bpb < best_bpb {
                best_bpb = bpb;
            }
            bpb_history.push((step, bpb));

            if step % REPORT_INTERVAL == 0 {
                let _ = db.push_bpb_sample(exp.id, step, bpb, Some(best_bpb)).await;
            }

            if step == SELF_CHECK_STEP {
                let decision = self_check(db, bpb, &bpb_history).await;
                match decision {
                    SelfDecision::Abandon => {
                        info!(
                            "self-check: abandoning experiment {} at step {}",
                            exp.id, step
                        );
                        let _ = db.mark_abandoned(exp.id, "self-check-abandon").await;
                        let _ = child.kill().await;
                        abandoned = true;
                        break;
                    }
                    SelfDecision::Continue => {
                        info!("self-check: continuing experiment {}", exp.id);
                    }
                }
            }

            if step > SELF_CHECK_STEP && step % KILL_CHECK_INTERVAL == 0 {
                match db.is_killed(exp.id).await {
                    Ok(true) => {
                        info!("kill signal received for experiment {}", exp.id);
                        let _ = db.mark_killed(exp.id, "gardener-superseded").await;
                        let _ = child.kill().await;
                        abandoned = true;
                        break;
                    }
                    Ok(false) => {}
                    Err(e) => warn!("kill check failed: {e}"),
                }
            }
        }

        if line.starts_with("done:") {
            if let Some(bpb) = parse_done_line(&line) {
                best_bpb = best_bpb.min(bpb);
            }
        }
    }

    let status = child.wait().await;
    info!("trainer exited: {:?}", status);

    if !abandoned {
        let _ = db.mark_done(exp.id, best_bpb, last_step).await;
        info!(
            "experiment {} done: bpb={best_bpb:.4} steps={last_step}",
            exp.id
        );
    }

    db.update_heartbeat(worker_id, None).await?;
    Ok(true)
}

fn parse_step_line(line: &str) -> Option<(i32, f32)> {
    let step_marker = "step=";
    let step_pos = line.find(step_marker)?;
    let after_step = &line[step_pos + step_marker.len()..];
    let step_end = after_step.find(char::is_whitespace)?;
    let step: i32 = after_step[..step_end].trim().parse().ok()?;

    let bpb_marker = "val_bpb=";
    let bpb_pos = line.find(bpb_marker)?;
    let after_bpb = &line[bpb_pos + bpb_marker.len()..];
    let bpb_end = after_bpb
        .find(char::is_whitespace)
        .unwrap_or(after_bpb.len());
    let bpb: f32 = after_bpb[..bpb_end].parse().ok()?;

    Some((step, bpb))
}

fn parse_done_line(line: &str) -> Option<f32> {
    if !line.starts_with("DONE:") {
        return None;
    }
    let bpb_prefix = "bpb=";
    let start = line.find(bpb_prefix)?;
    let rest = &line[start + bpb_prefix.len()..];
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

async fn self_check(db: &PullQueueDb, current_bpb: f32, history: &[(i32, f32)]) -> SelfDecision {
    if let Ok(Some(leader_bpb)) = db.leader_bpb_at_step(SELF_CHECK_STEP).await {
        if current_bpb > leader_bpb + ABANDON_GAP {
            info!("self-check: bpb={current_bpb:.4} > leader={leader_bpb:.4} + {ABANDON_GAP}");
            return SelfDecision::Abandon;
        }
    }

    if history.len() >= 3 {
        let predicted = fit_power_law(history);
        if predicted > PREDICTED_INF_GATE2 {
            info!("self-check: predicted_inf={predicted:.4} > {PREDICTED_INF_GATE2}");
            return SelfDecision::Abandon;
        }
    }

    SelfDecision::Continue
}

fn fit_power_law(samples: &[(i32, f32)]) -> f32 {
    if samples.len() < 3 {
        return 99.0;
    }
    let mut sum_ln_s = 0.0f32;
    let mut sum_ln_b = 0.0f32;
    let mut sum_ln_s_sq = 0.0f32;
    let mut sum_ln_s_ln_b = 0.0f32;
    for &(s, b) in samples {
        if b <= 0.0 || s <= 0 {
            continue;
        }
        let ls = (s as f32).ln();
        let lb = b.ln();
        sum_ln_s += ls;
        sum_ln_b += lb;
        sum_ln_s_sq += ls * ls;
        sum_ln_s_ln_b += ls * lb;
    }
    let count = samples.len() as f32;
    let denom = count * sum_ln_s_sq - sum_ln_s * sum_ln_s;
    if denom.abs() < 1e-10 {
        return 99.0;
    }
    let alpha = (count * sum_ln_s_ln_b - sum_ln_s * sum_ln_b) / denom;
    if alpha >= 0.0 {
        return 99.0;
    }
    let inf_steps = 1_000_000.0f32;
    let last_bpb = samples.last().map(|&(_, b)| b).unwrap_or(5.0);
    last_bpb * (inf_steps / samples.last().map(|&(s, _)| s as f32).unwrap_or(1000.0)).powf(alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_step_line() {
        let line = "seed=42 step=100 val_bpb=3.6570 ema_bpb=5.7278 best=5.7278 t=6.4s";
        let result = parse_step_line(line);
        assert_eq!(result, Some((100, 3.657)));

        let line2 = "seed=43 step=1000 val_bpb=2.9934 ema_bpb=3.1 best=2.99 t=52s";
        let result2 = parse_step_line(line2);
        assert_eq!(result2, Some((1000, 2.9934)));
    }

    #[test]
    fn test_parse_step_line_invalid() {
        assert_eq!(parse_step_line(""), None);
        assert_eq!(parse_step_line("random output"), None);
        assert_eq!(parse_step_line("DONE: seed=43 bpb=2.18"), None);
    }

    #[test]
    fn test_parse_done_line() {
        let line = "DONE: seed=43 bpb=2.1820 steps=81000 opt=adamw";
        assert_eq!(parse_done_line(line), Some(2.182));
    }

    #[test]
    fn test_fit_power_law_declining() {
        let samples: Vec<(i32, f32)> = vec![(500, 4.0), (1000, 3.0), (2000, 2.5), (4000, 2.2)];
        let predicted = fit_power_law(&samples);
        assert!(
            predicted < 3.0,
            "predicted should be < 3.0, got {predicted}"
        );
    }

    #[test]
    fn test_fit_power_law_few_samples() {
        let samples: Vec<(i32, f32)> = vec![(100, 5.0)];
        assert_eq!(fit_power_law(&samples), 99.0);
    }

    #[test]
    fn test_experiment_config_champion() {
        let val = serde_json::json!({
            "seed": 42, "hidden": 1024, "ctx": 12, "lr": 0.003, "steps": 81000
        });
        let config = ExperimentConfig::from_json_val(&val).unwrap();
        assert_eq!(config.hidden, 1024);
    }
}
