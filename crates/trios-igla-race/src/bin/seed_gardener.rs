use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, warn};

use trios_igla_race::pull_queue::{BpbSample, ExperimentConfig, PullQueueDb};

const GATE2_BPB: f32 = 1.85;
const DIVERGING_GAP: f32 = 0.3;
const MIRROR_COUNT: usize = 3;

#[derive(Parser)]
#[command(name = "seed-gardener", about = "ADR-001: Gardener orchestrator tick")]
struct Cli {
    #[command(subcommand)]
    command: GardenerCommand,
}

#[derive(Subcommand)]
enum GardenerCommand {
    Tick {
        #[arg(long, env = "NEON_DATABASE_URL")]
        neon_url: String,
    },
    Seed {
        #[arg(long, env = "NEON_DATABASE_URL")]
        neon_url: String,
        #[arg(long, default_value = "42")]
        seed: u64,
        #[arg(long, default_value = "1024")]
        hidden: usize,
        #[arg(long, default_value = "12")]
        ctx: usize,
        #[arg(long, default_value = "0.003")]
        lr: f64,
        #[arg(long, default_value = "81000")]
        steps: usize,
        #[arg(long, default_value = "1.0")]
        priority: f32,
    },
    Enqueue {
        #[arg(long, env = "NEON_DATABASE_URL")]
        neon_url: String,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        hidden: usize,
        #[arg(long)]
        ctx: usize,
        #[arg(long)]
        lr: f64,
        #[arg(long)]
        steps: usize,
        #[arg(long, default_value = "1.0")]
        priority: f32,
    },
    Status {
        #[arg(long, env = "NEON_DATABASE_URL")]
        neon_url: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter("seed_gardener=info,trios_igla_race=info")
        .init();

    let cli = Cli::parse();

    match cli.command {
        GardenerCommand::Tick { neon_url } => gardener_tick(&neon_url).await,
        GardenerCommand::Seed {
            neon_url,
            seed,
            hidden,
            ctx,
            lr,
            steps,
            priority,
        } => {
            let db = PullQueueDb::connect(&neon_url).await?;
            let config = serde_json::json!({"seed": seed, "hidden": hidden, "ctx": ctx, "lr": lr, "steps": steps}).to_string();
            let name = format!("h{hidden}-ctx{ctx}-lr{lr:.4}-s{seed}");
            let id = db
                .insert_experiment(&name, &config, priority, "human", None)
                .await?;
            info!("seeded experiment id={id} name={name}");
            Ok(())
        }
        GardenerCommand::Enqueue {
            neon_url,
            seed,
            hidden,
            ctx,
            lr,
            steps,
            priority,
        } => {
            let db = PullQueueDb::connect(&neon_url).await?;
            let config = serde_json::json!({"seed": seed, "hidden": hidden, "ctx": ctx, "lr": lr, "steps": steps}).to_string();
            let name = format!("q-h{hidden}-ctx{ctx}-lr{lr:.4}-s{seed}");
            let id = db
                .insert_experiment(&name, &config, priority, "gardener", None)
                .await?;
            info!("enqueued experiment id={id} name={name}");
            Ok(())
        }
        GardenerCommand::Status { neon_url } => show_status(&neon_url).await,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeedState {
    Leading,
    CatchingUp,
    Diverging,
}

async fn gardener_tick(neon_url: &str) -> Result<()> {
    info!("gardener tick starting");
    let db = PullQueueDb::connect(neon_url).await?;
    db.health_check().await?;

    let running = db.running_experiments().await?;
    info!("running experiments: {}", running.len());

    if running.is_empty() {
        info!("no running experiments — seeding defaults");
        seed_defaults(&db).await?;
        return Ok(());
    }

    let mut leader_curve: Vec<(i64, Vec<BpbSample>)> = Vec::new();
    for exp in &running {
        let samples = db.fetch_bpb_samples(exp.id).await?;
        leader_curve.push((exp.id, samples));
    }

    let best_running_bpb = leader_curve
        .iter()
        .filter_map(|(_, samples)| samples.last().map(|s| s.bpb))
        .fold(f32::MAX, f32::min);

    info!("best running bpb: {best_running_bpb:.4}");

    for (exp_id, samples) in &leader_curve {
        if samples.is_empty() {
            continue;
        }

        let current_bpb = samples.last().map(|s| s.bpb).unwrap_or(f32::MAX);
        let state = classify(current_bpb, best_running_bpb);

        info!("exp {exp_id}: bpb={current_bpb:.4} state={state:?}");

        match state {
            SeedState::Diverging => {
                let reason =
                    format!("diverging: bpb={current_bpb:.4} vs best={best_running_bpb:.4}");
                warn!("{reason}");
                if let Err(e) = db.mark_killed(*exp_id, &reason).await {
                    warn!("failed to kill {exp_id}: {e}");
                }
                let _ = db
                    .log_gardener_decision(
                        "prune",
                        &reason,
                        &[*exp_id],
                    )
                    .await;
            }
            SeedState::Leading => {
                let exp = running.iter().find(|e| e.id == *exp_id);
                if let Some(exp) = exp {
                    let config = ExperimentConfig::from_json_str(&exp.config_blob)?;
                    spawn_mirrors(&db, exp.id, &config).await?;
                }
            }
            SeedState::CatchingUp => {}
        }
    }

    let done_count = running.len();
    let quorum = check_gate2_quorum(&db).await?;
    info!("gate2 quorum: {quorum} experiments below {GATE2_BPB}");

    if !quorum && done_count < 12 {
        let suggestions = suggest_next(best_running_bpb);
        for sug in &suggestions {
            let config = serde_json::json!({"seed": sug.seed, "hidden": sug.hidden, "ctx": sug.ctx, "lr": sug.lr, "steps": sug.steps}).to_string();
            let name = format!(
                "g-h{}-ctx{}-lr{:.4}-s{}",
                sug.hidden, sug.ctx, sug.lr, sug.seed
            );
            match db
                .insert_experiment(&name, &config, 0.5, "gardener", None)
                .await
            {
                Ok(id) => info!("suggested experiment id={id}"),
                Err(e) => warn!("insert failed: {e}"),
            }
        }
    }

    info!("gardener tick complete");
    Ok(())
}

fn classify(current_bpb: f32, best_running_bpb: f32) -> SeedState {
    if current_bpb <= best_running_bpb + 0.05 {
        SeedState::Leading
    } else if current_bpb > best_running_bpb + DIVERGING_GAP {
        SeedState::Diverging
    } else {
        SeedState::CatchingUp
    }
}

async fn spawn_mirrors(db: &PullQueueDb, parent_id: i64, config: &ExperimentConfig) -> Result<()> {
    let mirror_seeds = [42u64, 43, 44]
        .iter()
        .filter(|&&s| s != config.seed)
        .copied()
        .collect::<Vec<_>>();

    for seed in mirror_seeds.iter().take(MIRROR_COUNT) {
        let mirror_config = serde_json::json!({
            "seed": seed,
            "hidden": config.hidden,
            "ctx": config.ctx,
            "lr": config.lr,
            "steps": config.steps,
        })
        .to_string();
        let name = format!("mirror-h{}-ctx{}-s{seed}", config.hidden, config.ctx);
        match db
            .insert_experiment(&name, &mirror_config, 1.0, "auto-mirror", Some(parent_id))
            .await
        {
            Ok(id) => info!("spawned mirror id={id} seed={seed}"),
            Err(e) => warn!("mirror insert failed: {e}"),
        }
    }
    Ok(())
}

async fn check_gate2_quorum(db: &PullQueueDb) -> Result<bool> {
    let running = db.running_experiments().await?;
    let mut below = 0usize;
    for exp in &running {
        let samples = db.fetch_bpb_samples(exp.id).await?;
        if let Some(best) = samples.iter().map(|s| s.bpb).reduce(f32::min) {
            if best < GATE2_BPB {
                below += 1;
            }
        }
    }
    Ok(below >= 3)
}

struct Suggestion {
    seed: u64,
    hidden: usize,
    ctx: usize,
    lr: f64,
    steps: usize,
}

fn suggest_next(best_bpb: f32) -> Vec<Suggestion> {
    let mut suggestions = Vec::new();
    let seeds = [42u64, 43, 44];

    let lr_variants = if best_bpb < 2.5 {
        vec![0.003, 0.002, 0.004]
    } else {
        vec![0.003, 0.006, 0.004]
    };

    let hidden_variants = [512, 1024, 828];
    let ctx_variants = [8, 12, 16];

    for &seed in &seeds {
        suggestions.push(Suggestion {
            seed,
            hidden: *hidden_variants.choose().unwrap_or(&1024),
            ctx: *ctx_variants.choose().unwrap_or(&12),
            lr: *lr_variants.choose().unwrap_or(&0.003),
            steps: 81000,
        });
    }

    suggestions
}

trait Choose<T> {
    fn choose(&self) -> Option<&T>;
}

impl<T> Choose<T> for [T] {
    fn choose(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        Some(&self[self.len() % 3.min(self.len())])
    }
}

async fn seed_defaults(db: &PullQueueDb) -> Result<()> {
    let defaults = [
        (42, 1024, 12, 0.003, 81000),
        (43, 1024, 12, 0.003, 81000),
        (44, 1024, 12, 0.003, 81000),
        (42, 512, 8, 0.003, 27000),
        (43, 828, 12, 0.004, 81000),
    ];

    for (seed, hidden, ctx, lr, steps) in defaults {
        let config = serde_json::json!({"seed": seed, "hidden": hidden, "ctx": ctx, "lr": lr, "steps": steps}).to_string();
        let name = format!("default-h{hidden}-ctx{ctx}-s{seed}");
        match db
            .insert_experiment(&name, &config, 1.0, "gardener", None)
            .await
        {
            Ok(id) => info!("seeded default id={id}"),
            Err(e) => warn!("seed failed: {e}"),
        }
    }
    Ok(())
}

async fn show_status(neon_url: &str) -> Result<()> {
    let db = PullQueueDb::connect(neon_url).await?;

    let rows = db.query_raw(
        "SELECT id, canon_name, status, priority, seed, steps_budget, account, final_bpb, final_step \
         FROM experiment_queue ORDER BY status, priority DESC, id",
    ).await?;

    println!("═══ EXPERIMENT QUEUE ({}) ═══", rows.len());
    println!(
        "{:<4} {:<45} {:<10} {:<3} {:<5} {:<7} {:<5} {:<10} {:<6}",
        "ID", "Name", "Status", "P", "Seed", "Budget", "Acc", "BestBPB", "Step"
    );

    for r in &rows {
        let id: i64 = r.get(0);
        let name: String = r.get(1);
        let status: String = r.get(2);
        let pri: i32 = r.get(3);
        let seed: Option<i32> = r.get(4);
        let budget: Option<i32> = r.get(5);
        let account: Option<String> = r.get(6);
        let final_bpb: Option<f64> = r.get(7);
        let final_step: Option<i32> = r.get(8);

        let name_short = if name.len() > 44 {
            format!("{}..", &name[..42])
        } else {
            name.clone()
        };

        println!(
            "{:<4} {:<45} {:<10} {:<3} {:<5} {:<7} {:<5} {:<10} {:<6}",
            id,
            name_short,
            status,
            pri,
            seed.map(|s| s.to_string()).unwrap_or("-".into()),
            budget.map(|b| b.to_string()).unwrap_or("-".into()),
            account.unwrap_or("-".into()),
            final_bpb
                .map(|b| format!("{b:.4}"))
                .unwrap_or("-".into()),
            final_step
                .map(|s| s.to_string())
                .unwrap_or("-".into()),
        );
    }

    let w_rows = db.query_raw(
        "SELECT id, railway_acc, railway_svc_name, last_heartbeat, current_exp_id FROM workers ORDER BY registered_at",
    ).await?;
    println!("\n═══ WORKERS ({}) ═══", w_rows.len());
    for r in &w_rows {
        let id: uuid::Uuid = r.get(0);
        let acc: String = r.get(1);
        let svc: Option<String> = r.get(2);
        let hb: Option<chrono::DateTime<chrono::Utc>> = r.get(3);
        let exp: Option<i64> = r.get(4);
        println!(
            "  {} acc={} svc={} hb={} exp={}",
            id,
            acc,
            svc.unwrap_or("-".into()),
            hb.map(|h| h.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
                .unwrap_or("-".into()),
            exp.map(|e| e.to_string())
                .unwrap_or("-".into()),
        );
    }

    let bpb_count: i64 = db.query_raw("SELECT COUNT(*) FROM bpb_samples")
        .await?
        .into_iter()
        .next()
        .map(|r| r.get::<_, i64>(0))
        .unwrap_or(0);
    println!("\n═══ BPB SAMPLES: {} ═══", bpb_count);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_leading() {
        assert_eq!(classify(1.8, 1.8), SeedState::Leading);
        assert_eq!(classify(1.84, 1.8), SeedState::Leading);
    }

    #[test]
    fn test_classify_diverging() {
        assert_eq!(classify(3.0, 1.8), SeedState::Diverging);
        assert_eq!(classify(2.2, 1.8), SeedState::Diverging);
    }

    #[test]
    fn test_classify_catching_up() {
        assert_eq!(classify(2.05, 1.8), SeedState::CatchingUp);
    }

    #[test]
    fn test_suggest_next() {
        let suggestions = suggest_next(2.5);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions.len(), 3);
    }

    #[test]
    fn test_gate2_constant() {
        assert!((GATE2_BPB - 1.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mirror_count() {
        assert_eq!(MIRROR_COUNT, 3);
    }
}
