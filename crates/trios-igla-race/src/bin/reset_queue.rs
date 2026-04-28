use anyhow::Result;
use trios_igla_race::pull_queue::PullQueueDb;

#[tokio::main]
async fn main() -> Result<()> {
    let neon_url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "postgresql://neondb_owner:npg_NHBC5hdbM0Kx@ep-curly-math-ao51pquy-pooler.c-2.ap-southeast-1.aws.neon.tech/neondb?sslmode=require".to_string());

    let db = PullQueueDb::connect(&neon_url).await?;
    eprintln!("Connected to Neon");

    if std::env::args().any(|a| a == "--cpu-batch") {
        let lrs: Vec<f64> = vec![0.001, 0.0015, 0.003, 0.004];
        let seeds: Vec<i32> = vec![42, 43, 44];
        let mut batch = Vec::new();
        for lr in &lrs {
            for seed in &seeds {
                batch.push((*lr, *seed));
            }
        }

        for (lr, seed) in &batch {
            let canon = format!("CPU-LR-SWEEP-h1024-lr{:.4}-s{}", lr, seed);
            let config = serde_json::json!({
                "h": 1024, "ctx": 12, "lr": lr,
                "seed": seed, "steps": 120000,
                "phase": "CPU-LR-SWEEP",
                "d_model": 64, "hidden": 1024,
                "weight_tying": true, "bottleneck_residual": true,
                "model": "TRAIN_V2", "optimizer": "AdamW"
            });
            let sql = format!(
                "INSERT INTO experiment_queue (canon_name, config_json, priority, seed, steps_budget, account, status, created_by) \
                 VALUES ('{}', '{}', 99, {}, 120000, 'acc0', 'pending', 'seed-agent') \
                 ON CONFLICT (canon_name) DO NOTHING",
                canon.replace('\'', "''"),
                serde_json::to_string(&config)?.replace('\'', "''"),
                seed
            );
            match db.raw_execute(&sql).await {
                Ok(n) => eprintln!("  OK lr={lr:.4} seed={seed} rows={n}"),
                Err(e) => eprintln!("  ERR lr={lr:.4} seed={seed}: {e}"),
            }
        }
        eprintln!("CPU LR sweep inserted ({} experiments, P=99)", batch.len());
    }

    let rows = db
        .raw_query(
            "SELECT id, canon_name, status, steps_budget, seed, priority FROM experiment_queue WHERE status IN ('pending','claimed','running') ORDER BY priority DESC, id",
        )
        .await?;
    eprintln!("\n--- Active experiments ---");
    for r in &rows {
        let id: i64 = r.get(0);
        let name: String = r.get(1);
        let status: String = r.get(2);
        let budget: i32 = r.get(3);
        let seed: i32 = r.get(4);
        let prio: i32 = r.get(5);
        eprintln!("{id:>3} P={prio:<3} {status:<10} budget={budget:>6} seed={seed} {name}");
    }

    Ok(())
}
