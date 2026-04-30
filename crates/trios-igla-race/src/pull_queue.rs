use anyhow::Result;
use openssl::ssl::{SslConnector, SslMethod, SslVerifyMode};
use postgres_openssl::MakeTlsConnector;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: i64,
    pub canon_name: String,
    pub config_blob: String,
    pub priority: f32,
    pub seed: i32,
    pub steps_budget: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub seed: u64,
    pub hidden: usize,
    pub ctx: usize,
    pub lr: f64,
    pub steps: usize,
}

impl ExperimentConfig {
    pub fn from_json_str(s: &str) -> Result<Self> {
        let val: serde_json::Value = serde_json::from_str(s)?;
        Self::from_json_val(&val)
    }

    pub fn from_json_val(val: &serde_json::Value) -> Result<Self> {
        Ok(Self {
            seed: val["seed"].as_u64().unwrap_or(43),
            hidden: val["d_model"]
                .as_u64()
                .or_else(|| val["hidden"].as_u64())
                .or_else(|| val["h"].as_u64())
                .unwrap_or(512) as usize,
            ctx: val["ctx_len"]
                .as_u64()
                .or_else(|| val["ngram"].as_u64())
                .or_else(|| val["ctx"].as_u64())
                .unwrap_or(12) as usize,
            lr: val["lr"].as_f64().unwrap_or(0.002),
            steps: val["steps"].as_u64().unwrap_or(120000) as usize,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfDecision {
    Continue,
    Abandon,
}

#[derive(Debug, Clone)]
pub struct BpbSample {
    pub step: i32,
    pub bpb: f32,
    pub val_bpb_ema: Option<f32>,
}

pub struct PullQueueDb {
    client: Arc<Mutex<tokio_postgres::Client>>,
}

impl PullQueueDb {
    pub async fn connect(conn_str: &str) -> Result<Self> {
        let mut builder = SslConnector::builder(SslMethod::tls())?;
        builder.set_verify(SslVerifyMode::NONE);
        let connector = MakeTlsConnector::new(builder.build());
        let (client, conn) = tokio_postgres::connect(conn_str, connector).await?;
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                tracing::error!("connection error: {e}");
            }
        });
        info!("PullQueueDb: connected to Neon (openssl TLS)");
        Ok(Self {
            client: Arc::new(Mutex::new(client)),
        })
    }

    pub async fn query_raw(
        &self,
        sql: &str,
    ) -> Result<Vec<tokio_postgres::Row>> {
        let client = self.client.lock().await;
        Ok(client.query(sql, &[]).await?)
    }

    pub async fn health_check(&self) -> Result<()> {
        let client = self.client.lock().await;
        let row = client.query_one("SELECT 1", &[]).await?;
        let val: i32 = row.get(0);
        anyhow::ensure!(val == 1, "health check failed");
        Ok(())
    }

    pub async fn pull_experiment(&self, worker_id: &Uuid) -> Result<Option<Experiment>> {
        let client = self.client.lock().await;
        let rows = client
            .query(
                "UPDATE experiment_queue \
                 SET status='claimed', worker_id=$1, claimed_at=now() \
                 WHERE id = (\
                     SELECT id FROM experiment_queue \
                     WHERE status='pending' \
                     ORDER BY priority DESC, created_at ASC \
                     LIMIT 1 \
                     FOR UPDATE SKIP LOCKED\
                 ) \
                 RETURNING id, canon_name, config_json::text, priority, seed, steps_budget",
                &[worker_id],
            )
            .await?;
        if rows.is_empty() {
            return Ok(None);
        }
        let row = &rows[0];
        Ok(Some(Experiment {
            id: row.get(0),
            canon_name: row.get(1),
            config_blob: row.get(2),
            priority: row.get::<_, i32>(3) as f32,
            seed: row.get(4),
            steps_budget: row.get(5),
        }))
    }

    pub async fn mark_running(&self, exp_id: i64) -> Result<()> {
        let client = self.client.lock().await;
        client
            .execute(
                "UPDATE experiment_queue SET status='running', started_at=now() WHERE id=$1",
                &[&exp_id],
            )
            .await?;
        Ok(())
    }

    pub async fn mark_done(&self, exp_id: i64, bpb: f32, steps: i32) -> Result<()> {
        let client = self.client.lock().await;
        client
            .execute(
                "UPDATE experiment_queue SET status='done', finished_at=now(), final_bpb=$1, final_step=$2 WHERE id=$3",
                &[&(bpb as f64), &steps, &exp_id],
            )
            .await?;
        Ok(())
    }

    pub async fn mark_abandoned(&self, exp_id: i64, reason: &str) -> Result<()> {
        let client = self.client.lock().await;
        client
            .execute(
                "UPDATE experiment_queue SET status='failed', finished_at=now(), prune_reason=$1 WHERE id=$2",
                &[&reason, &exp_id],
            )
            .await?;
        Ok(())
    }

    pub async fn mark_killed(&self, exp_id: i64, reason: &str) -> Result<()> {
        let client = self.client.lock().await;
        client
            .execute(
                "UPDATE experiment_queue SET status='pruned', finished_at=now(), prune_reason=$1 WHERE id=$2",
                &[&reason, &exp_id],
            )
            .await?;
        Ok(())
    }

    pub async fn push_bpb_sample(
        &self,
        exp_id: i64,
        step: i32,
        bpb: f32,
        val_bpb_ema: Option<f32>,
    ) -> Result<()> {
        let client = self.client.lock().await;
        let row = client
            .query_one(
                "SELECT canon_name, seed FROM experiment_queue WHERE id=$1",
                &[&exp_id],
            )
            .await?;
        let canon_name: String = row.get(0);
        let seed: i32 = row.get(1);
        client
            .execute(
                "INSERT INTO bpb_samples (canon_name, seed, step, bpb, val_bpb_ema, ts) \
                 VALUES ($1, $2, $3, $4, $5, now()) \
                 ON CONFLICT (canon_name, seed, step) DO UPDATE SET bpb=$4, val_bpb_ema=$5",
                &[&canon_name, &seed, &step, &(bpb as f64), &val_bpb_ema.map(|v| v as f64)],
            )
            .await?;
        Ok(())
    }

    pub async fn register_worker(
        &self,
        worker_id: &Uuid,
        railway_acc: &str,
        railway_svc: &str,
    ) -> Result<()> {
        let client = self.client.lock().await;
        client
            .execute(
                "INSERT INTO workers (id, railway_acc, railway_svc_id, railway_svc_name, last_heartbeat, registered_at) \
                 VALUES ($1, $2, $3, $3, now(), now()) \
                 ON CONFLICT (id) DO UPDATE SET last_heartbeat=now(), railway_acc=$2, railway_svc_id=$3, railway_svc_name=$3",
                &[&worker_id, &railway_acc, &railway_svc],
            )
            .await?;
        Ok(())
    }

    pub async fn update_heartbeat(
        &self,
        worker_id: &Uuid,
        current_exp_id: Option<i64>,
    ) -> Result<()> {
        let client = self.client.lock().await;
        client
            .execute(
                "UPDATE workers SET last_heartbeat=now(), current_exp_id=$1 WHERE id=$2",
                &[&current_exp_id, &worker_id],
            )
            .await?;
        Ok(())
    }

    pub async fn leader_bpb_at_step(&self, step: i32) -> Result<Option<f32>> {
        let client = self.client.lock().await;
        let row = client
            .query_opt(
                "SELECT MIN(bpb)::real FROM bpb_samples WHERE step=$1",
                &[&step],
            )
            .await?;
        match row {
            Some(r) => Ok(r.get::<_, Option<f32>>(0)),
            None => Ok(None),
        }
    }

    pub async fn is_killed(&self, exp_id: i64) -> Result<bool> {
        let client = self.client.lock().await;
        let row = client
            .query_opt(
                "SELECT status FROM experiment_queue WHERE id=$1",
                &[&exp_id],
            )
            .await?;
        match row {
            Some(r) => {
                let status: String = r.get(0);
                Ok(status == "pruned" || status == "failed")
            }
            None => Ok(true),
        }
    }

    pub async fn fetch_bpb_samples(&self, exp_id: i64) -> Result<Vec<BpbSample>> {
        let client = self.client.lock().await;
        let rows = client
            .query(
                "SELECT s.step, s.bpb::real, s.val_bpb_ema::real \
                 FROM bpb_samples s \
                 JOIN experiment_queue e ON e.canon_name = s.canon_name AND e.seed = s.seed \
                 WHERE e.id=$1 \
                 ORDER BY s.step",
                &[&exp_id],
            )
            .await?;
        Ok(rows
            .iter()
            .map(|r| BpbSample {
                step: r.get(0),
                bpb: r.get(1),
                val_bpb_ema: r.get(2),
            })
            .collect())
    }

    pub async fn running_experiments(&self) -> Result<Vec<Experiment>> {
        let client = self.client.lock().await;
        let rows = client
            .query(
                "SELECT id, canon_name, config_json::text, priority::real, seed, steps_budget \
                 FROM experiment_queue WHERE status='running' ORDER BY id",
                &[],
            )
            .await?;
        Ok(rows
            .iter()
            .map(|r| Experiment {
                id: r.get(0),
                canon_name: r.get(1),
                config_blob: r.get(2),
                priority: r.get::<_, Option<f32>>(3).unwrap_or(0.0),
                seed: r.get(4),
                steps_budget: r.get(5),
            })
            .collect())
    }

    pub async fn insert_experiment(
        &self,
        canon_name: &str,
        config_blob: &str,
        priority: f32,
        created_by: &str,
        _parent_exp_id: Option<i64>,
    ) -> Result<i64> {
        let client = self.client.lock().await;
        let config_val: serde_json::Value = serde_json::from_str(config_blob)?;
        let seed: i32 = config_val["seed"].as_i64().unwrap_or(43) as i32;
        let steps_budget: i32 = config_val["steps"].as_i64().unwrap_or(27000) as i32;
        let priority_i32 = priority as i32;
        let row = client
            .query_one(
                "INSERT INTO experiment_queue (canon_name, config_json, priority, seed, steps_budget, account, status, created_by) \
                 VALUES ($1, $2::jsonb, $3, $4, $5, 'acc0', 'pending', $6) \
                 ON CONFLICT DO NOTHING \
                 RETURNING id",
                &[&canon_name, &config_blob, &priority_i32, &seed, &steps_budget, &created_by],
            )
            .await?;
        Ok(row.get::<_, i64>(0))
    }

    pub async fn log_gardener_decision(
        &self,
        action: &str,
        reason: &str,
        affected_exp_ids: &[i64],
    ) -> Result<()> {
        let client = self.client.lock().await;
        client
            .execute(
                "INSERT INTO gardener_decisions (action, reason, affected_exp_ids, ts) VALUES ($1, $2, $3, now())",
                &[&action, &reason, &affected_exp_ids],
            )
            .await?;
        Ok(())
    }

    pub async fn check_table_exists(&self, table_name: &str) -> Result<bool> {
        let client = self.client.lock().await;
        let row = client
            .query_one(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name=$1)",
                &[&table_name],
            )
            .await?;
        Ok(row.get::<_, bool>(0))
    }

    pub fn clone_handle(&self) -> Self {
        Self {
            client: self.client.clone(),
        }
    }

    pub async fn raw_query(&self, sql: &str) -> Result<Vec<tokio_postgres::Row>> {
        let client = self.client.lock().await;
        Ok(client.query(sql, &[]).await?)
    }

    pub async fn raw_execute(&self, sql: &str) -> Result<u64> {
        let client = self.client.lock().await;
        Ok(client.execute(sql, &[]).await?)
    }
}

pub fn spawn_heartbeat(db: PullQueueDb, worker_id: Uuid) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            if let Err(e) = db.update_heartbeat(&worker_id, None).await {
                warn!("heartbeat failed for {}: {e}", worker_id);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_config_defaults() {
        let val = serde_json::json!({});
        let config = ExperimentConfig::from_json_val(&val).unwrap();
        assert_eq!(config.seed, 43);
        assert_eq!(config.hidden, 512);
        assert_eq!(config.ctx, 12);
        assert!((config.lr - 0.002).abs() < f64::EPSILON);
        assert_eq!(config.steps, 120000);
    }

    #[test]
    fn test_experiment_config_from_json() {
        let val = serde_json::json!({
            "seed": 42,
            "hidden": 1024,
            "ctx": 12,
            "lr": 0.004,
            "steps": 81000
        });
        let config = ExperimentConfig::from_json_val(&val).unwrap();
        assert_eq!(config.seed, 42);
        assert_eq!(config.hidden, 1024);
        assert_eq!(config.ctx, 12);
        assert!((config.lr - 0.004).abs() < f64::EPSILON);
        assert_eq!(config.steps, 81000);
    }

    #[test]
    fn test_self_decision_variants() {
        assert_eq!(SelfDecision::Continue, SelfDecision::Continue);
        assert_eq!(SelfDecision::Abandon, SelfDecision::Abandon);
        assert!(SelfDecision::Continue != SelfDecision::Abandon);
    }

    #[test]
    fn test_experiment_config_champion() {
        let val = serde_json::json!({
            "seed": 42,
            "hidden": 1024,
            "ctx": 12,
            "lr": 0.003,
            "steps": 81000
        });
        let config = ExperimentConfig::from_json_val(&val).unwrap();
        assert_eq!(config.hidden, 1024);
        assert_eq!(config.ctx, 12);
    }
}
