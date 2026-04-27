//! R7-style triplet hash for audit events.
//!
//! Format: `RAIL=<verb> @ project=<8c> service=<8c> sha=<8c> ts=<rfc3339>`
//!
//! The hash digest is `sha256(verb || project || service || token_fp)[..16]`.
//! We never embed the bearer token; only its fingerprint.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ids::{ProjectId, ServiceId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RailwayHash {
    pub verb: String,
    pub project: ProjectId,
    pub service: Option<ServiceId>,
    pub digest: String,
    pub ts: DateTime<Utc>,
}

impl RailwayHash {
    pub fn seal(
        verb: &str,
        project: &ProjectId,
        service: Option<&ServiceId>,
        token_fingerprint: &str,
    ) -> Self {
        let mut h = Sha256::new();
        h.update(verb.as_bytes());
        h.update(project.as_str().as_bytes());
        if let Some(s) = service {
            h.update(s.as_str().as_bytes());
        }
        h.update(token_fingerprint.as_bytes());
        let bytes = h.finalize();
        let digest = hex::encode(&bytes[..8]); // 16 hex chars

        Self {
            verb: verb.to_string(),
            project: project.clone(),
            service: service.cloned(),
            digest,
            ts: Utc::now(),
        }
    }

    pub fn triplet(&self) -> String {
        format!(
            "RAIL={} @ project={} service={} sha={} ts={}",
            self.verb,
            self.project.short(),
            self.service.as_ref().map_or("-", |s| s.short()),
            self.digest,
            self.ts.to_rfc3339(),
        )
    }
}

/// 7-char fingerprint of a bearer token. The token itself is never
/// stored.
#[must_use]
pub fn token_fingerprint(token: &str) -> String {
    let mut h = Sha256::new();
    h.update(token.as_bytes());
    let bytes = h.finalize();
    hex::encode(&bytes[..4]) // 8 hex chars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_is_deterministic_modulo_timestamp() {
        let p = ProjectId::from("e4fe33bb-3b09-4842-9782-7d2dea1abc9b");
        let s = ServiceId::from("0f0a948f-c457-4f4c-b5c7-a5ef96fcf9e9");
        let fp = token_fingerprint("dummy-token");

        let a = RailwayHash::seal("create", &p, Some(&s), &fp);
        let b = RailwayHash::seal("create", &p, Some(&s), &fp);
        assert_eq!(a.digest, b.digest, "digest must be content-addressed");
        assert_ne!(
            a.ts,
            b.ts.checked_sub_signed(chrono::Duration::days(1)).unwrap()
        );
    }

    #[test]
    fn token_fingerprint_does_not_leak_token() {
        let fp = token_fingerprint("super-secret-bearer");
        assert_eq!(fp.len(), 8);
        assert!(!fp.contains("secret"));
    }

    #[test]
    fn triplet_format_includes_all_fields() {
        let p = ProjectId::from("e4fe33bb-3b09-4842-9782-7d2dea1abc9b");
        let s = ServiceId::from("0f0a948f-c457-4f4c-b5c7-a5ef96fcf9e9");
        let h = RailwayHash::seal("deploy", &p, Some(&s), &token_fingerprint("t"));
        let line = h.triplet();
        assert!(line.starts_with("RAIL=deploy @ project=e4fe33bb service=0f0a948f sha="));
    }
}
