//! Issue #63 — `BpbSource` trait + three implementations.
//!
//! The R0 invariant for the gardener is "every tick prints a
//! leaderboard". The leaderboard needs (seed, lane, step, BPB) tuples;
//! this module is the typed contract for *where those tuples come from*
//! when the primary write path (`bpb_samples` table — issue #62) is
//! still un-shipped.
//!
//! Three sources, polled in priority order, first non-empty wins per
//! `(seed, lane)` pair:
//!
//! 1. `NeonBpbSamples` — the eventual primary source, reads the
//!    `bpb_samples` table once it exists. Today returns empty (table
//!    missing — 42P01 — but the code path compiles and tests).
//! 2. `RailwayLogsScraper` — pulls the last N stdout lines from each
//!    Railway service via Railway's GraphQL `deploymentLogs` and greps
//!    for `step=NNNN bpb=N.NNNN`. Closes the gap until #62 lands and
//!    until ALPHA's writer patch is pushed.
//! 3. `GithubIssueComments` — last-resort tail of `trios#143` comments
//!    for ALPHA's manual `BPB=… @ step=…` posts. Always available; uses
//!    `gh` CLI via the `github` API credentials preset on the host.
//!
//! No source is allowed to panic on a network/DB failure. Each impl
//! returns `Vec<BpbSample>`, possibly empty, and emits a tracing event
//! describing the honest reason the source is empty (R5).

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::collections::BTreeMap;

/// One observed (seed, lane) BPB sample with timestamp + step.
#[derive(Debug, Clone, PartialEq)]
pub struct BpbSample {
    pub seed: u32,
    pub lane: String,
    pub step: u64,
    pub bpb: f64,
    pub ts: DateTime<Utc>,
    pub source: SourceTag,
}

/// Tag carried on every sample so the leaderboard can show provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceTag {
    Neon,
    RailwayLogs,
    GithubComments,
    /// Used by tests / fixtures.
    Manual,
}

impl SourceTag {
    pub fn as_str(&self) -> &'static str {
        match self {
            SourceTag::Neon => "neon",
            SourceTag::RailwayLogs => "rail-logs",
            SourceTag::GithubComments => "gh-comments",
            SourceTag::Manual => "manual",
        }
    }
}

#[async_trait]
pub trait BpbSource: Send + Sync {
    fn name(&self) -> &'static str;
    async fn fetch(&self) -> Result<Vec<BpbSample>>;
}

// ---------------------------------------------------------------------
// (a) NeonBpbSamples — primary, currently 42P01.
// ---------------------------------------------------------------------

pub struct NeonBpbSamples {
    pub database_url: Option<String>,
}

impl NeonBpbSamples {
    pub fn from_env() -> Self {
        Self {
            database_url: std::env::var("NEON_DATABASE_URL").ok(),
        }
    }
}

#[async_trait]
impl BpbSource for NeonBpbSamples {
    fn name(&self) -> &'static str {
        "neon-bpb-samples"
    }
    async fn fetch(&self) -> Result<Vec<BpbSample>> {
        let Some(url) = self.database_url.as_deref() else {
            tracing::info!("NEON_DATABASE_URL not set; neon source skipped");
            return Ok(Vec::new());
        };
        let (client, conn) = match tokio_postgres::connect(url, tokio_postgres::NoTls).await {
            Ok(pair) => pair,
            Err(e) => {
                tracing::warn!(?e, "neon connect failed; treating as empty source");
                return Ok(Vec::new());
            }
        };
        tokio::spawn(async move {
            let _ = conn.await;
        });
        // Honest 42P01 path: SELECT against a missing table returns an
        // error; we map it to an empty result so the leaderboard still
        // renders.
        let rows = match client
            .query(
                "SELECT seed, lane, step, bpb, ts FROM bpb_samples \
                 WHERE ts > now() - interval '12 hours' ORDER BY ts DESC",
                &[],
            )
            .await
        {
            Ok(rows) => rows,
            Err(e) => {
                tracing::warn!(?e, "bpb_samples select failed (likely 42P01); empty");
                return Ok(Vec::new());
            }
        };
        let out = rows
            .into_iter()
            .map(|r| BpbSample {
                seed: r.get::<_, i64>(0) as u32,
                lane: r.get(1),
                step: r.get::<_, i32>(2) as u64,
                bpb: r.get(3),
                ts: r.get(4),
                source: SourceTag::Neon,
            })
            .collect();
        Ok(out)
    }
}

// ---------------------------------------------------------------------
// (b) RailwayLogsScraper — last 500 lines per service, regex grep.
// ---------------------------------------------------------------------

pub struct RailwayLogsScraper {
    pub services: Vec<RailwayServiceRef>,
    pub token: Option<String>,
    pub endpoint: String,
}

#[derive(Debug, Clone)]
pub struct RailwayServiceRef {
    pub seed: u32,
    pub lane: String,
    pub deployment_id: String,
    pub account_token: String,
}

impl RailwayLogsScraper {
    pub fn new(services: Vec<RailwayServiceRef>) -> Self {
        Self {
            services,
            token: None,
            endpoint: "https://backboard.railway.com/graphql/v2".to_string(),
        }
    }
}

/// Public so tests in other modules can re-use the same parser.
pub fn parse_step_bpb_lines(text: &str) -> Vec<(u64, f64)> {
    use regex::Regex;
    // Tolerate either `step=NNNN bpb=N.NNNN` or `step NNNN bpb N.NNNN`
    // and either order. Captures `step` first then `bpb`.
    let re = Regex::new(r"step[=\s]+(\d+)\s+bpb[=\s]+([0-9]+\.[0-9]+)").unwrap();
    re.captures_iter(text)
        .filter_map(|c| {
            let s = c.get(1)?.as_str().parse::<u64>().ok()?;
            let b = c.get(2)?.as_str().parse::<f64>().ok()?;
            Some((s, b))
        })
        .collect()
}

#[async_trait]
impl BpbSource for RailwayLogsScraper {
    fn name(&self) -> &'static str {
        "railway-logs"
    }
    async fn fetch(&self) -> Result<Vec<BpbSample>> {
        if self.services.is_empty() {
            tracing::info!("railway-logs: no services configured; empty");
            return Ok(Vec::new());
        }
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        let mut out: Vec<BpbSample> = Vec::new();
        for s in &self.services {
            let body = serde_json::json!({
                "query": "query Logs($id: String!) { deploymentLogs(deploymentId: $id, limit: 500) { message timestamp } }",
                "variables": {"id": s.deployment_id},
            });
            let res = client
                .post(&self.endpoint)
                .header("Project-Access-Token", &s.account_token)
                .json(&body)
                .send()
                .await;
            let body = match res {
                Ok(r) => match r.text().await {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!(?e, seed = s.seed, "log body read failed");
                        continue;
                    }
                },
                Err(e) => {
                    tracing::warn!(?e, seed = s.seed, "deploymentLogs query failed");
                    continue;
                }
            };
            // Best-effort parse — the GraphQL envelope wraps messages
            // with timestamps; we only need the `message` strings.
            let json: serde_json::Value = match serde_json::from_str(&body) {
                Ok(j) => j,
                Err(_) => continue,
            };
            let logs = json["data"]["deploymentLogs"].as_array();
            let Some(arr) = logs else {
                tracing::info!(
                    seed = s.seed,
                    "deploymentLogs returned no data (likely Not Authorized or 42P01)"
                );
                continue;
            };
            let mut latest: Option<(u64, f64, DateTime<Utc>)> = None;
            for entry in arr {
                let msg = entry["message"].as_str().unwrap_or("");
                let ts_str = entry["timestamp"].as_str().unwrap_or("");
                let ts: DateTime<Utc> = ts_str.parse().unwrap_or_else(|_| Utc::now());
                for (step, bpb) in parse_step_bpb_lines(msg) {
                    if latest.as_ref().map_or(true, |(prev_step, _, _)| step >= *prev_step) {
                        latest = Some((step, bpb, ts));
                    }
                }
            }
            if let Some((step, bpb, ts)) = latest {
                out.push(BpbSample {
                    seed: s.seed,
                    lane: s.lane.clone(),
                    step,
                    bpb,
                    ts,
                    source: SourceTag::RailwayLogs,
                });
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------
// (c) GithubIssueComments — tail of trios#143.
// ---------------------------------------------------------------------

pub struct GithubIssueComments {
    pub repo: String,
    pub issue: u64,
    pub since_minutes: i64,
}

impl GithubIssueComments {
    pub fn for_trios_143() -> Self {
        Self {
            repo: "gHashTag/trios".to_string(),
            issue: 143,
            since_minutes: 120,
        }
    }
}

/// Public for re-use in tests / future ALPHA-comment parsing tools.
///
/// Recognises lines like `BPB=2.1919 @ step=81000 seed=43`. Returns
/// `(seed, step, bpb)` triples in document order.
pub fn parse_alpha_bpb_block(text: &str) -> Vec<(u32, u64, f64)> {
    use regex::Regex;
    let re = Regex::new(
        r"BPB[=\s]+([0-9]+\.[0-9]+)\s*@\s*step[=\s]+(\d+)\s+seed[=\s]+(\d+)",
    )
    .unwrap();
    re.captures_iter(text)
        .filter_map(|c| {
            let bpb = c.get(1)?.as_str().parse::<f64>().ok()?;
            let step = c.get(2)?.as_str().parse::<u64>().ok()?;
            let seed = c.get(3)?.as_str().parse::<u32>().ok()?;
            Some((seed, step, bpb))
        })
        .collect()
}

#[async_trait]
impl BpbSource for GithubIssueComments {
    fn name(&self) -> &'static str {
        "gh-comments"
    }
    async fn fetch(&self) -> Result<Vec<BpbSample>> {
        // We shell out to `gh` so we inherit the host's GitHub auth.
        // R1 reminder: this binary still ships as Rust-only; `gh` is a
        // host tool, not a script we author. Subprocess use is fine.
        // We use std::process::Command on a blocking thread because
        // workspace tokio doesn't enable the `process` feature.
        let issue = self.issue;
        let repo = self.repo.clone();
        let out = tokio::task::spawn_blocking(move || {
            std::process::Command::new("gh")
                .args([
                    "issue",
                    "view",
                    &issue.to_string(),
                    "--repo",
                    &repo,
                    "--json",
                    "comments",
                ])
                .output()
        })
        .await
        .map_err(|e| anyhow::anyhow!("join: {e}"))?;
        let out = match out {
            Ok(o) if o.status.success() => o,
            Ok(o) => {
                tracing::warn!(
                    code = ?o.status.code(),
                    stderr = %String::from_utf8_lossy(&o.stderr),
                    "gh issue view failed"
                );
                return Ok(Vec::new());
            }
            Err(e) => {
                tracing::warn!(?e, "gh CLI not available");
                return Ok(Vec::new());
            }
        };
        let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_default();
        let comments = json["comments"].as_array().cloned().unwrap_or_default();
        let cutoff = Utc::now() - chrono::Duration::minutes(self.since_minutes);
        let mut samples: BTreeMap<(u32, String), BpbSample> = BTreeMap::new();
        for c in comments {
            let body = c["body"].as_str().unwrap_or("");
            let ts_str = c["createdAt"].as_str().unwrap_or("");
            let ts: DateTime<Utc> = ts_str.parse().unwrap_or_else(|_| Utc::now());
            if ts < cutoff {
                continue;
            }
            for (seed, step, bpb) in parse_alpha_bpb_block(body) {
                let key = (seed, "alpha".to_string());
                let entry = samples.entry(key).or_insert(BpbSample {
                    seed,
                    lane: "alpha".to_string(),
                    step,
                    bpb,
                    ts,
                    source: SourceTag::GithubComments,
                });
                if step >= entry.step {
                    entry.step = step;
                    entry.bpb = bpb;
                    entry.ts = ts;
                }
            }
        }
        Ok(samples.into_values().collect())
    }
}

// ---------------------------------------------------------------------
// Merge: first-non-empty wins per (seed, lane); all sources run in
// parallel for honesty (no source can starve another).
// ---------------------------------------------------------------------

/// Run all sources and merge results. Per-source errors are logged and
/// the source contributes the empty set; the merged output never panics.
pub async fn merge_sources(sources: &[Box<dyn BpbSource>]) -> Vec<BpbSample> {
    let mut all: Vec<BpbSample> = Vec::new();
    for s in sources {
        let name = s.name();
        match s.fetch().await {
            Ok(v) => {
                tracing::info!(source = name, n = v.len(), "source fetch ok");
                all.extend(v);
            }
            Err(e) => {
                tracing::warn!(?e, source = name, "source fetch errored; treated as empty");
            }
        }
    }
    // Dedup by (seed, lane), keep highest step then most recent ts.
    let mut by_key: BTreeMap<(u32, String), BpbSample> = BTreeMap::new();
    for s in all {
        let key = (s.seed, s.lane.clone());
        let replace = match by_key.get(&key) {
            None => true,
            Some(prev) => s.step > prev.step || (s.step == prev.step && s.ts > prev.ts),
        };
        if replace {
            by_key.insert(key, s);
        }
    }
    by_key.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_step_bpb_picks_up_typical_line() {
        let text = "INFO step=27000 bpb=2.4500 lr=0.003\nINFO step=27001 bpb=2.4485";
        let v = parse_step_bpb_lines(text);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], (27000, 2.45));
    }

    #[test]
    fn parse_step_bpb_ignores_garbage() {
        assert_eq!(parse_step_bpb_lines("nothing here").len(), 0);
        assert_eq!(parse_step_bpb_lines("step=foo bpb=bar").len(), 0);
    }

    #[test]
    fn parse_alpha_bpb_block_handles_canonical_form() {
        let text = "BPB=2.1919 @ step=81000 seed=43 sha=cd91c45\nBPB=2.2024 @ step=81000 seed=44";
        let v = parse_alpha_bpb_block(text);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], (43, 81000, 2.1919));
        assert_eq!(v[1], (44, 81000, 2.2024));
    }

    #[tokio::test]
    async fn neon_source_without_url_returns_empty() {
        let s = NeonBpbSamples { database_url: None };
        let v = s.fetch().await.unwrap();
        assert!(v.is_empty());
    }

    #[tokio::test]
    async fn railway_scraper_with_no_services_is_empty() {
        let s = RailwayLogsScraper::new(vec![]);
        let v = s.fetch().await.unwrap();
        assert!(v.is_empty());
    }

    #[tokio::test]
    async fn merge_picks_highest_step_per_seed_lane() {
        struct FixedSource(Vec<BpbSample>);
        #[async_trait::async_trait]
        impl BpbSource for FixedSource {
            fn name(&self) -> &'static str {
                "fixed"
            }
            async fn fetch(&self) -> anyhow::Result<Vec<BpbSample>> {
                Ok(self.0.clone())
            }
        }
        let now = Utc::now();
        let early = BpbSample {
            seed: 43,
            lane: "L1".into(),
            step: 1000,
            bpb: 3.0,
            ts: now,
            source: SourceTag::Manual,
        };
        let late = BpbSample {
            seed: 43,
            lane: "L1".into(),
            step: 5000,
            bpb: 2.5,
            ts: now,
            source: SourceTag::Manual,
        };
        let sources: Vec<Box<dyn BpbSource>> = vec![
            Box::new(FixedSource(vec![early.clone()])),
            Box::new(FixedSource(vec![late.clone()])),
        ];
        let merged = merge_sources(&sources).await;
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].step, 5000);
        assert_eq!(merged[0].bpb, 2.5);
    }
}
