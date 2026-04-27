//! Append-only writer for `.trinity/experience/<YYYYMMDD>.trinity`.
//!
//! Enforces L7 (experience log) + L21 (context immutability): the
//! writer only ever opens files in append mode and never seeks. We
//! refuse to truncate; any caller that asks is rejected at compile
//! time because no truncating method is exposed.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;

use trios_railway_core::RailwayHash;

/// Allowed PHI LOOP steps. Mirrors `NOW.json`.
pub const PHI_STEPS: &[&str] = &[
    "CLAIM",
    "NAME",
    "SPEC",
    "SEAL",
    "GEN",
    "TEST",
    "VERDICT",
    "EXPERIENCE",
    "REPORT",
    "COMMIT",
    "PUSH",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceLine {
    pub ts: String,
    pub agent: String,
    pub soul_name: String,
    pub issue: String,
    pub task: String,
    pub status: String,
    pub phi_step: String,
    pub triplet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

impl ExperienceLine {
    pub fn from_hash(
        agent: &str,
        soul_name: &str,
        issue: &str,
        task: &str,
        status: &str,
        phi_step: &str,
        hash: &RailwayHash,
    ) -> Result<Self> {
        if !PHI_STEPS.contains(&phi_step) {
            anyhow::bail!("invalid phi_step `{phi_step}`");
        }
        Ok(Self {
            ts: Utc::now().to_rfc3339(),
            agent: agent.to_string(),
            soul_name: soul_name.to_string(),
            issue: issue.to_string(),
            task: task.to_string(),
            status: status.to_string(),
            phi_step: phi_step.to_string(),
            triplet: hash.triplet(),
            evidence: None,
        })
    }
}

/// Append a single line to the daily experience file, creating
/// directories as needed. Returns the path written.
pub async fn append_line(root: &Path, line: &ExperienceLine) -> Result<PathBuf> {
    let dir = root.join("experience");
    fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("create_dir_all {}", dir.display()))?;

    let day = Utc::now().format("%Y%m%d").to_string();
    let path = dir.join(format!("trios_railway_{day}.trinity"));

    let mut serialized = serde_json::to_string(line)?;
    serialized.push('\n');

    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .with_context(|| format!("open append {}", path.display()))?;
    f.write_all(serialized.as_bytes()).await?;
    f.flush().await?;

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trios_railway_core::{ProjectId, RailwayHash};

    #[tokio::test]
    async fn append_creates_and_appends() {
        let tmp = tempfile::tempdir().unwrap();
        let p = ProjectId::from("e4fe33bb-3b09-4842-9782-7d2dea1abc9b");
        let hash = RailwayHash::seal("test", &p, None, "fp");
        let line = ExperienceLine::from_hash(
            "GENERAL",
            "DustyDeployer",
            "#1",
            "bootstrap",
            "OK",
            "EXPERIENCE",
            &hash,
        )
        .unwrap();

        let out1 = append_line(tmp.path(), &line).await.unwrap();
        let out2 = append_line(tmp.path(), &line).await.unwrap();
        assert_eq!(out1, out2, "same day → same file");

        let body = tokio::fs::read_to_string(&out1).await.unwrap();
        assert_eq!(body.lines().count(), 2, "two appends");
    }

    #[test]
    fn rejects_unknown_phi_step() {
        let p = ProjectId::from("p");
        let h = RailwayHash::seal("v", &p, None, "fp");
        let res = ExperienceLine::from_hash("a", "s", "#1", "t", "ok", "WALK", &h);
        assert!(res.is_err());
    }
}
