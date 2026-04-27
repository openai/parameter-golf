//! Strongly typed identifiers for Railway resources.
//!
//! All identifiers are opaque, lowercase, hyphen-delimited UUIDs in
//! Railway's data plane. We keep them as `String` newtypes rather than
//! `Uuid` so we can round-trip Railway's free-form short IDs (logs,
//! environment slugs) without lossy parsing.

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! id_newtype {
    ($name:ident, $kind:literal) => {
        /// Opaque Railway identifier.
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            #[must_use]
            pub fn new(s: impl Into<String>) -> Self {
                Self(s.into())
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// Short prefix (first 8 chars) for log lines.
            #[must_use]
            pub fn short(&self) -> &str {
                let n = self.0.len().min(8);
                &self.0[..n]
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", $kind, self.0)
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_string())
            }
        }
    };
}

id_newtype!(ProjectId, "project");
id_newtype!(EnvironmentId, "env");
id_newtype!(ServiceId, "service");
id_newtype!(DeployId, "deploy");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_serde() {
        let s = ServiceId::from("0f0a948f-c457-4f4c-b5c7-a5ef96fcf9e9");
        let j = serde_json::to_string(&s).unwrap();
        assert_eq!(j, "\"0f0a948f-c457-4f4c-b5c7-a5ef96fcf9e9\"");
        let back: ServiceId = serde_json::from_str(&j).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn short_prefix() {
        let s = ProjectId::from("e4fe33bb-3b09-4842-9782-7d2dea1abc9b");
        assert_eq!(s.short(), "e4fe33bb");
    }

    #[test]
    fn display_kind_marker() {
        let d = DeployId::from("abc1234");
        assert_eq!(format!("{d}"), "deploy(abc1234)");
    }
}
