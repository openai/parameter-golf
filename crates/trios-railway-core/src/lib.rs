//! `trios-railway-core` — identity types + Railway GraphQL transport.
//!
//! Anchor: `phi^2 + phi^-2 = 3`.
//!
//! Standing rules (binding):
//!   R1 — Rust-only.
//!   R5 — Honest passthrough of upstream errors.
//!   R7 — Every mutation call seals an audit triplet via [`RailwayHash`].
//!   R9 — `igla check <sha>` MUST be invoked by callers before any
//!        mutation; this crate exposes only the typed mutation surface.

pub mod hash;
pub mod ids;
pub mod transport;

pub use hash::RailwayHash;
pub use ids::{DeployId, EnvironmentId, ProjectId, ServiceId};
pub use transport::{Client, ClientError};
