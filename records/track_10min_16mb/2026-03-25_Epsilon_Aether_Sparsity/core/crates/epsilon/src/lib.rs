//! ═══════════════════════════════════════════════════════════════════════════════
//! Epsilon — Geometric State Transfer via Topological Surgery on Hollow Manifolds
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! **Pre-Print Reference Implementation v0.1.0**
//! **Author: Teerth Sharma**
//!
//! # Abstract
//!
//! Current LLM architectures suffer from severe latency bottlenecks during
//! context loading due to sequential token processing and KV-cache matrix
//! multiplication. This crate implements a novel mechanism for Zero-Shot
//! Context Teleportation by structuring the agent's state space as a hollow
//! cubic manifold (an S² boundary with Betti number β₂ = 1).
//!
//! A higher-order manifold can instantaneously inject pre-computed,
//! topologically stable "mental models" directly into the agent's internal
//! void, bypassing the O(N²) token bottleneck and enabling instantaneous
//! knowledge transfer between AEGIS-powered autonomous agents.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                        Epsilon Stack                                 │
//! ├──────────────────┬───────────────────────────────────────────────────┤
//! │  bridge.rs       │ EmbeddingBridge — ℝ^E → S² (JL + L2 normalize)  │
//! │                  │ ProjectionMatrix — seeded, deterministic          │
//! ├──────────────────┼───────────────────────────────────────────────────┤
//! │  teleport.rs     │ sys_teleport_context() — Orchestration syscall   │
//! ├──────────────────┼───────────────────────────────────────────────────┤
//! │  manifold.rs     │ HollowCubeManifold  — S² void receptacle         │
//! │                  │ ManifoldPayload     — Verified payload unit       │
//! ├──────────────────┼───────────────────────────────────────────────────┤
//! │  governor.rs     │ SurgeryGovernor     — PD control + clutch        │
//! │                  │ SurgeryPermit       — One-shot derivative lock    │
//! ├──────────────────┼───────────────────────────────────────────────────┤
//! │  memory.rs       │ LivenessAnchor      — Inherited Chebyshev k-σ    │
//! │                  │ ChebyshevGuard      — GC eviction safety          │
//! └──────────────────┴───────────────────────────────────────────────────┘
//! ```
//!
//! # Mathematical Foundation
//!
//! ## The Hollow Cube (S² Manifold)
//!
//! - Solid Manifold: β₀ = 1, β₁ = 0, β₂ = 0
//! - Hollow Manifold: β₀ = 1, β₁ = 0, β₂ = 1
//!
//! The presence of β₂ = 1 defines an interior "void" — the secure
//! receptacle for instantaneous geometric injection.
//!
//! ## Topological Surgery
//!
//! Context teleportation is modeled as a fiber bundle projection:
//!
//! ```text
//!   f: D ⊂ M_high → Void(M_recv)
//! ```
//!
//! ## Safety Modifications
//!
//! 1. **Surgery Permit** (governor.rs): Zeroes derivative gain β for one
//!    tick to prevent oscillation panic when de/dt → ∞.
//! 2. **Chebyshev Liveness Inheritance** (memory.rs): Pre-ages teleported
//!    data with inherited k-σ bounds to prevent immediate eviction.
//! 3. **Wake-Up Rescan** (manifold.rs): Verifies Betti boundaries of
//!    injected mass against the hollow cube's inner walls.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

pub mod bridge;
pub mod manifold;
pub mod governor;
pub mod memory;
pub mod teleport;
pub mod mock;

// ═══════════════════════════════════════════════════════════════════════════════
// Public API Re-exports
// ═══════════════════════════════════════════════════════════════════════════════

pub use bridge::{EmbeddingBridge, BridgeError, MIN_TOKENS, DEFAULT_SPHERE_EPSILON};
pub use manifold::{
    HollowCubeManifold, ManifoldPayload, SurgeryError,
    EpsilonPoint, SparseGraph,
};
pub use governor::{SurgeryGovernor, SurgeryPermit};
pub use memory::{LivenessAnchor, ChebyshevGuard};
pub use teleport::{TeleportTarget, TeleportResult, sys_teleport_context, RemoteVoidDescriptor};
