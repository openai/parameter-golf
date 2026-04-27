//! Drift detection between Railway reality, the Neon `igla_*` ledger,
//! and `.trinity/experience` lines.
//!
//! This v0.0.1 ships:
//!   - `DriftCode` enum (D1..D7)
//!   - `DriftEvent` struct + JSON Schema
//!   - `Severity`
//!   - in-memory `detect()` over typed inputs
//!
//! AU-02 (issue #8): `event` module adds the Neon writer.
//! The Neon writer consumes `DriftEvent` rows and inserts
//! them via `tri railway audit run --neon`. Cron + GitHub Actions integration
//! lives behind issue #16.

pub mod event;
pub mod migrations;

use serde::{Deserialize, Serialize};
use trios_railway_core::ServiceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Severity {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DriftCode {
    /// Service exists in Railway but no ledger row for that seed.
    D1OrphanService,
    /// Ledger row exists but service was removed in Railway.
    D2LostLedger,
    /// Real BPB diverges from the ledger BPB by more than 0.01.
    D3BpbMismatch,
    /// Logs include the canonical fallback-data marker AND BPB ~ 0.
    D4FallbackData,
    /// BPB > 1e30 (`f32::MAX` overflow).
    D5Overflow,
    /// `igla_agents_heartbeat.last_seen` is older than 1 hour.
    D6NoHeartbeat,
    /// Image digest differs from the canonical winning digest.
    D7ImageDrift,
}

impl DriftCode {
    pub const fn severity(self) -> Severity {
        match self {
            Self::D1OrphanService
            | Self::D2LostLedger
            | Self::D6NoHeartbeat
            | Self::D7ImageDrift => Severity::Warn,
            Self::D3BpbMismatch | Self::D4FallbackData | Self::D5Overflow => Severity::Error,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftEvent {
    pub service_id: ServiceId,
    pub code: DriftCode,
    pub severity: Severity,
    pub detail: serde_json::Value,
    pub triplet: Option<String>,
}

/// Minimal ledger and reality views needed for v0.0.1 detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealService {
    pub service_id: ServiceId,
    pub name: String,
    pub seed: Option<i32>,
    pub last_log_excerpt: Option<String>,
    pub last_bpb: Option<f64>,
    pub image_digest: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerRow {
    pub seed: i32,
    pub bpb: f64,
    pub canonical_image_digest: Option<String>,
}

const FALLBACK_MARKER: &str = "Failed to load data/tiny_shakespeare.txt";

#[must_use]
pub fn detect(real: &[RealService], ledger: &[LedgerRow]) -> Vec<DriftEvent> {
    let mut out = Vec::new();

    for svc in real {
        // D5: overflow
        if let Some(b) = svc.last_bpb {
            if b > 1e30 {
                out.push(event(svc, DriftCode::D5Overflow, &format!("bpb={b}")));
                continue;
            }
        }

        // D4: fallback-data
        if let (Some(log), Some(b)) = (&svc.last_log_excerpt, svc.last_bpb) {
            if log.contains(FALLBACK_MARKER) && b < 0.001 {
                out.push(event(
                    svc,
                    DriftCode::D4FallbackData,
                    "fallback corpus + BPB~0",
                ));
                continue;
            }
        }

        // D1: orphan service (no ledger row for its seed)
        if let Some(seed) = svc.seed {
            let row = ledger.iter().find(|r| r.seed == seed);
            match row {
                None => out.push(event(svc, DriftCode::D1OrphanService, "no ledger row")),
                Some(r) => {
                    if let Some(b) = svc.last_bpb {
                        if (b - r.bpb).abs() > 0.01 {
                            out.push(event(
                                svc,
                                DriftCode::D3BpbMismatch,
                                &format!("real={b} ledger={}", r.bpb),
                            ));
                        }
                    }
                    if let (Some(real_d), Some(canon)) =
                        (&svc.image_digest, &r.canonical_image_digest)
                    {
                        if real_d != canon {
                            out.push(event(
                                svc,
                                DriftCode::D7ImageDrift,
                                &format!("real={real_d} canon={canon}"),
                            ));
                        }
                    }
                }
            }
        }
    }

    // D2: lost ledger (seed in ledger, no service)
    for row in ledger {
        let exists = real.iter().any(|s| s.seed == Some(row.seed));
        if !exists {
            out.push(DriftEvent {
                service_id: ServiceId::from(format!("ledger-seed-{}", row.seed)),
                code: DriftCode::D2LostLedger,
                severity: DriftCode::D2LostLedger.severity(),
                detail: serde_json::json!({ "seed": row.seed, "ledger_bpb": row.bpb }),
                triplet: None,
            });
        }
    }

    out
}

fn event(svc: &RealService, code: DriftCode, msg: &str) -> DriftEvent {
    DriftEvent {
        service_id: svc.service_id.clone(),
        code,
        severity: code.severity(),
        detail: serde_json::json!({ "service": svc.name, "msg": msg }),
        triplet: None,
    }
}

/// Verdict for `tri railway audit run`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditVerdict {
    /// At least three live services with `bpb < target` and no error-level drift.
    Gate2Pass,
    /// No errors but Gate-2 not yet met.
    NotYet,
    /// At least one error-level drift event.
    Drift,
}

impl AuditVerdict {
    pub const fn exit_code(self) -> i32 {
        match self {
            Self::Gate2Pass => 0,
            Self::NotYet => 2,
            Self::Drift => 1,
        }
    }
}

#[must_use]
pub fn verdict(real: &[RealService], events: &[DriftEvent], target_bpb: f64) -> AuditVerdict {
    if events.iter().any(|e| e.severity == Severity::Error) {
        return AuditVerdict::Drift;
    }
    let live_pass = real
        .iter()
        .filter(|s| s.last_bpb.is_some_and(|b| b < target_bpb))
        .count();
    if live_pass >= 3 {
        AuditVerdict::Gate2Pass
    } else {
        AuditVerdict::NotYet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn svc(seed: i32, bpb: Option<f64>, log: Option<&str>) -> RealService {
        RealService {
            service_id: ServiceId::from(format!("svc-{seed}")),
            name: format!("trios-train-seed-{seed}"),
            seed: Some(seed),
            last_log_excerpt: log.map(str::to_string),
            last_bpb: bpb,
            image_digest: None,
        }
    }

    #[test]
    fn detects_d4_fallback_data() {
        let real = vec![svc(
            100,
            Some(0.0001),
            Some("Failed to load data/tiny_shakespeare.txt + ..."),
        )];
        let events = detect(&real, &[]);
        assert!(events.iter().any(|e| e.code == DriftCode::D4FallbackData));
    }

    #[test]
    fn detects_d5_overflow() {
        let real = vec![svc(43, Some(3.4e38), None)];
        let events = detect(&real, &[]);
        assert!(events.iter().any(|e| e.code == DriftCode::D5Overflow));
    }

    #[test]
    fn detects_d2_lost_ledger() {
        let real = vec![svc(100, Some(2.0), None)];
        let ledger = vec![LedgerRow {
            seed: 999,
            bpb: 1.5,
            canonical_image_digest: None,
        }];
        let events = detect(&real, &ledger);
        assert!(events.iter().any(|e| e.code == DriftCode::D2LostLedger));
    }

    #[test]
    fn verdict_pass_requires_three_live_below_target() {
        let real = vec![
            svc(100, Some(1.5), None),
            svc(101, Some(1.7), None),
            svc(102, Some(1.84), None),
        ];
        assert_eq!(verdict(&real, &[], 1.85), AuditVerdict::Gate2Pass);
    }

    #[test]
    fn verdict_drift_when_error_event() {
        let real = vec![svc(
            100,
            Some(0.0),
            Some("Failed to load data/tiny_shakespeare.txt"),
        )];
        let events = detect(&real, &[]);
        assert_eq!(verdict(&real, &events, 1.85), AuditVerdict::Drift);
    }
}
