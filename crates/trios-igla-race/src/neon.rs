use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::info;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialConfig {
    pub arch: String,
    #[serde(rename = "d_model")]
    pub hidden: usize,
    #[serde(rename = "n_gram")]
    pub context: usize,
    pub lr: f64,
    pub seed: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMeta {
    pub agent_id: String,
    pub branch: String,
    pub machine_id: String,
    pub worker_id: String,
}

impl Default for DashboardMeta {
    fn default() -> Self {
        Self {
            agent_id: "ALPHA".to_string(),
            branch: "main".to_string(),
            machine_id: "unknown".to_string(),
            worker_id: "w0".to_string(),
        }
    }
}

impl DashboardMeta {
    pub fn new(agent_id: &str, machine_id: &str, worker_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            branch: "main".to_string(),
            machine_id: machine_id.to_string(),
            worker_id: worker_id.to_string(),
        }
    }

    pub fn with_branch(mut self, branch: &str) -> Self {
        self.branch = branch.to_string();
        self
    }
}

#[derive(Debug, Clone)]
pub struct LessonEntry {
    pub lesson: String,
    pub lesson_type: String,
    pub pattern_count: i32,
}

pub struct NeonDb {
    _dummy: (),
}

impl NeonDb {
    pub async fn connect(conn_str: &str) -> Result<Self> {
        info!("Connecting to Neon: {conn_str} (STUB MODE)");
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        info!("Connected to Neon (STUB)");
        Ok(Self { _dummy: () })
    }

    pub fn client(&self) -> &Self {
        self
    }

    pub async fn register_trial(
        &self,
        trial_id: &Uuid,
        machine_id: &str,
        worker_id: i32,
        config_json: &str,
    ) -> Result<()> {
        info!("Trial registered (STUB): trial_id={trial_id} machine={machine_id} worker={worker_id} config={config_json}");
        Ok(())
    }

    pub async fn record_checkpoint(&self, trial_id: &Uuid, rung: i32, bpb: f64) -> Result<()> {
        info!("Checkpoint recorded (STUB): trial={trial_id} rung={rung} BPB={bpb:.4}");
        Ok(())
    }

    pub async fn update_rung(&self, trial_id: &str, rung_steps: usize, bpb: f64) -> Result<()> {
        info!("Rung updated (STUB): trial={trial_id} rung={rung_steps} BPB={bpb:.4}");
        Ok(())
    }

    pub async fn update_heartbeat(&self, trial_id: &str) -> Result<()> {
        info!("Heartbeat (STUB): trial_id={trial_id}");
        Ok(())
    }

    pub async fn mark_pruned(&self, trial_id: &Uuid, at_step: i32, bpb: f64) -> Result<()> {
        info!("Trial pruned (STUB): trial={trial_id} step={at_step} bpb={bpb:.4}");
        Ok(())
    }

    pub async fn mark_completed(&self, trial_id: &Uuid, bpb: f64, steps: i32) -> Result<()> {
        info!("Trial completed (STUB): trial={trial_id} BPB={bpb:.4} steps={steps}");
        Ok(())
    }

    pub async fn mark_winner(&self, trial_id: &str, bpb: f64, steps: usize) -> Result<()> {
        info!("IGLA FOUND (STUB): trial={trial_id} BPB={bpb:.4} steps={steps}");
        Ok(())
    }

    pub async fn is_config_running(&self, machine_id: &str, config_json: &str) -> Result<bool> {
        let _ = (machine_id, config_json);
        Ok(false)
    }

    pub async fn get_median_bpb_at_rung(&self, rung_steps: usize) -> Result<Option<f64>> {
        let _ = rung_steps;
        Ok(None)
    }

    pub async fn store_lesson(
        &self,
        trial_id: &Uuid,
        outcome: &str,
        pruned_at_rung: i32,
        bpb_at_pruned: f64,
        lesson: &str,
        lesson_type: &str,
    ) -> Result<()> {
        info!("Lesson stored (STUB): trial={trial_id} outcome={outcome} rung={pruned_at_rung} bpb={bpb_at_pruned:.4} type={lesson_type} lesson={lesson}");
        Ok(())
    }

    pub async fn get_top_lessons(&self, limit: i32) -> Result<Vec<LessonEntry>> {
        let _ = limit;
        Ok(vec![])
    }

    pub async fn query(
        &self,
        query: &str,
        _params: &[&(dyn tokio_postgres::types::ToSql + Sync)],
    ) -> Result<Vec<tokio_postgres::Row>> {
        info!(
            "Query (STUB): {}",
            query.trim().chars().take(80).collect::<String>()
        );
        Err(anyhow::anyhow!("stub: no connection"))
    }

    pub async fn query_one(
        &self,
        query: &str,
        _params: &[&(dyn tokio_postgres::types::ToSql + Sync)],
    ) -> Result<tokio_postgres::Row> {
        info!(
            "Query one (STUB): {}",
            query.trim().chars().take(80).collect::<String>()
        );
        Err(anyhow::anyhow!("stub: no rows"))
    }
}

pub fn spawn_heartbeat(db: NeonDb, trial_id: String) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
            if let Err(e) = db.update_heartbeat(&trial_id).await {
                tracing::warn!("Heartbeat failed for {}: {}", trial_id, e);
            }
        }
    })
}

pub const SCHEMA_MIGRATION: &str = r#"
ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS branch TEXT DEFAULT 'main';
ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS agent_id TEXT;
ALTER TABLE igla_race_trials ADD COLUMN IF NOT EXISTS last_heartbeat TIMESTAMPTZ DEFAULT NOW();
"#;

pub mod queries {
    pub const LEADERBOARD: &str = r#"
SELECT
  agent_id,
  branch,
  config->>'arch' as arch,
  config->>'d_model' as d_model,
  rung_1000_bpb,
  rung_3000_bpb,
  final_bpb,
  status,
  last_heartbeat,
  EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) as heartbeat_lag_sec
FROM igla_race_trials
ORDER BY COALESCE(final_bpb, rung_3000_bpb, rung_1000_bpb, 999) ASC
LIMIT 20;
"#;

    pub const ACTIVE_AGENTS: &str = r#"
SELECT agent_id, machine_id, branch, COUNT(*) as active_trials
FROM igla_race_trials
WHERE status='running'
  AND last_heartbeat > NOW() - INTERVAL '2 minutes'
GROUP BY agent_id, machine_id, branch;
"#;

    pub const BEST_BY_ARCH: &str = r#"
SELECT config->>'arch' as arch, MIN(final_bpb) as best_bpb, COUNT(*) as trials
FROM igla_race_trials
WHERE status IN ('completed', 'winner')
GROUP BY config->>'arch'
ORDER BY best_bpb ASC NULLS LAST;
"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_config_serialization() {
        let config = TrialConfig {
            arch: "ngram".to_string(),
            hidden: 384,
            context: 6,
            lr: 0.004,
            seed: 42,
            optimizer: Some("adamw".to_string()),
            wd: Some(0.01),
            activation: Some("relu".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("ngram"));
        assert!(json.contains("384"));
    }

    #[test]
    fn test_dashboard_meta_default() {
        let meta = DashboardMeta::default();
        assert_eq!(meta.agent_id, "ALPHA");
        assert_eq!(meta.branch, "main");
    }

    #[test]
    fn test_dashboard_meta_custom() {
        let meta = DashboardMeta::new("BETA", "mac-studio-2", "w1");
        assert_eq!(meta.agent_id, "BETA");
        assert_eq!(meta.machine_id, "mac-studio-2");
        assert_eq!(meta.worker_id, "w1");
        assert_eq!(meta.branch, "main");
    }

    #[test]
    fn test_dashboard_meta_with_branch() {
        let meta = DashboardMeta::new("GAMMA", "macbook-pro-1", "w0").with_branch("feat/jepa");
        assert_eq!(meta.branch, "feat/jepa");
    }

    #[test]
    fn test_schema_migration_contains_columns() {
        assert!(SCHEMA_MIGRATION.contains("branch"));
        assert!(SCHEMA_MIGRATION.contains("agent_id"));
        assert!(SCHEMA_MIGRATION.contains("last_heartbeat"));
    }

    #[test]
    fn test_queries_exist() {
        assert!(!queries::LEADERBOARD.is_empty());
        assert!(!queries::ACTIVE_AGENTS.is_empty());
        assert!(!queries::BEST_BY_ARCH.is_empty());
    }
}
