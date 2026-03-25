//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Core Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Platform-agnostic mathematical foundation for AEGIS.
//! Works on both `no_std` (bare-metal kernel) and `std` (CLI/apps).
//!
//! Core Modules:
//!   - topology: TDA, Betti numbers, shape verification
//!   - manifold: Time-delay embedding, sparse attention graphs
//!   - aether: AETHER geometric primitives, hierarchical blocks
//!   - ml: Regression engine, convergence detection
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

// ═══════════════════════════════════════════════════════════════════════════════
// Module Exports
// ═══════════════════════════════════════════════════════════════════════════════

pub mod aether;
pub mod governor;
pub mod manifold;
pub mod ml;
pub mod state;
pub mod os;
pub mod topology;
pub mod memory;

// Re-export key types for convenience
pub use aether::{BlockMetadata, DriftDetector, HierarchicalBlockTree};
pub use manifold::{ManifoldPoint, SparseAttentionGraph, TimeDelayEmbedder, TopologicalPipeline};
pub use topology::{
    compute_betti_0, compute_betti_1, compute_shape, verify_shape, TopologicalShape, VerifyResult,
};
