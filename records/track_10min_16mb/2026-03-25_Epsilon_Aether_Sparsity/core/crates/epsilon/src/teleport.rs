//! ═══════════════════════════════════════════════════════════════════════════════
//! Epsilon — Context Teleportation Orchestration
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements the `sys_teleport_context` syscall — the top-level orchestration
//! function that drives the full context-transfer pipeline:
//!
//! ```text
//!   EmbeddingBridge  →  ManifoldPayload  →  sys_teleport_context
//!                                                    │
//!                        ┌───────────────────────────┼───────────────────────────┐
//!                        │   1. SurgeryPermit acquire (governor clutch zeroed)   │
//!                        │   2. inject_into_void(payload)                        │
//!                        │   3. assimilate() — wake-up rescan                    │
//!                        │   4. complete_surgery(permit) — restore β             │
//!                        └───────────────────────────────────────────────────────┘
//! ```
//!
//! # Transfer Targets
//!
//! - [`TeleportTarget::LocalVoid`] — in-process injection. Fully implemented.
//! - [`TeleportTarget::RemoteVoid`] — cross-process / cross-machine transfer.
//!   The variant is defined and the routing stub is in place, but serialization
//!   and network routing are labeled **future work** (see `FUTURE_WORK` note
//!   inside the `RemoteVoid` arm). Returns [`TeleportResult::RemoteUnimplemented`].
//!
//! # Limitation 3 Handling
//!
//! If the receiving manifold's void is occupied the caller receives
//! [`TeleportResult::VoidBusy`] and must retry. This avoids blocking inside a
//! no_std context where async runtimes are unavailable.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use crate::governor::SurgeryGovernor;
use crate::manifold::{HollowCubeManifold, ManifoldPayload, SurgeryError};

// ═══════════════════════════════════════════════════════════════════════════════
// TeleportTarget
// ═══════════════════════════════════════════════════════════════════════════════

/// Descriptor for a remote agent's injection endpoint.
///
/// # Future Work
///
/// The routing and serialization layer for `RemoteVoid` is unimplemented.
/// A complete implementation would require:
/// - Serializing [`ManifoldPayload`] via the `serde` feature flag.
/// - Establishing a transport channel (e.g., shared memory, TCP, QUIC).
/// - Authenticating the target agent (agent_id verification).
///
/// This struct is intentionally minimal so the type system enforces the
/// "future work" boundary: calling `sys_teleport_context` with this variant
/// will always return [`TeleportResult::RemoteUnimplemented`].
#[derive(Debug, Clone)]
pub struct RemoteVoidDescriptor {
    /// Opaque identifier for the target agent process / node.
    pub agent_id: u64,
}

impl RemoteVoidDescriptor {
    /// Create a descriptor for a remote agent.
    pub fn new(agent_id: u64) -> Self {
        Self { agent_id }
    }
}

/// Specifies where the payload should be injected.
///
/// # Variants
///
/// - `LocalVoid` — inject into the provided [`HollowCubeManifold`] in the
///   calling process. Fully implemented.
/// - `RemoteVoid(RemoteVoidDescriptor)` — route to a remote agent. The
///   serialization / transport layer is not yet implemented; this variant
///   causes [`sys_teleport_context`] to return
///   [`TeleportResult::RemoteUnimplemented`].
#[derive(Debug, Clone)]
pub enum TeleportTarget {
    /// In-process injection into the provided [`HollowCubeManifold`].
    LocalVoid,
    /// Cross-agent transfer to the described remote endpoint.
    ///
    /// **Status:** Variant defined; transport mechanism unimplemented for production.
    /// See [`crate::mock::MockRemoteEndpoint`] for the simulated integration
    /// testing stub. See [`RemoteVoidDescriptor`] for target details.
    RemoteVoid(RemoteVoidDescriptor),
}

// ═══════════════════════════════════════════════════════════════════════════════
// TeleportResult
// ═══════════════════════════════════════════════════════════════════════════════

/// The outcome of a [`sys_teleport_context`] call.
#[derive(Debug, Clone, PartialEq)]
pub enum TeleportResult {
    /// Context was successfully transferred and assimilated.
    Success {
        /// Number of manifold points merged into the receiving shell.
        points_assimilated: usize,
    },

    /// The receiving manifold's topological surgery failed.
    ///
    /// The payload was rejected because its Betti signature did not match
    /// the void boundary constraints, or the shell was degenerate.
    TopologyRejected(SurgeryError),

    /// The receiving manifold's void is already occupied by a prior payload.
    ///
    /// The caller should retry after the current payload has been assimilated.
    VoidBusy,

    /// The target was [`TeleportTarget::RemoteVoid`] but the network transport
    /// layer is not yet implemented.
    ///
    /// This is a defined recoverable error, not a panic. Callers can branch
    /// on this variant and fall back to local computation or queue the
    /// transfer for when the transport layer is available.
    RemoteUnimplemented,
}

// ═══════════════════════════════════════════════════════════════════════════════
// sys_teleport_context
// ═══════════════════════════════════════════════════════════════════════════════

/// **Primary entry point for context teleportation.**
///
/// Orchestrates the full topological surgery pipeline in a single atomic
/// sequence:
///
/// 1. **Governor Clutch** — acquire a [`SurgeryPermit`](crate::SurgeryPermit),
///    zeroing the derivative gain β to prevent oscillation panic.
/// 2. **Void Injection** — call `manifold.inject_into_void(payload)`, which
///    verifies Betti constraints and writes the payload into the hollow void.
/// 3. **Wake-Up Rescan** — call `manifold.assimilate()`, merging the payload
///    points into the active shell graph.
/// 4. **Governor Restore** — call `governor.complete_surgery(permit)`,
///    restoring β and the error history.
///
/// # Arguments
///
/// - `manifold`: Mutable reference to the **receiving** agent's manifold.
/// - `payload`: The [`ManifoldPayload<D>`] produced by [`EmbeddingBridge`](crate::EmbeddingBridge).
/// - `governor`: The receiving agent's [`SurgeryGovernor`] (for clutch management).
/// - `target`: Where to inject — `LocalVoid` or `RemoteVoid(descriptor)`.
///
/// # Returns
///
/// - [`TeleportResult::Success`] — pipeline completed; `points_assimilated`
///   reports how many points were merged.
/// - [`TeleportResult::TopologyRejected`] — Betti mismatch or degenerate shell.
/// - [`TeleportResult::VoidBusy`] — void already occupied; retry later.
/// - [`TeleportResult::RemoteUnimplemented`] — remote routing not yet available.
///
/// # Example
///
/// ```rust,ignore
/// use epsilon::{EmbeddingBridge, sys_teleport_context, HollowCubeManifold, SurgeryGovernor};
/// use epsilon::teleport::TeleportTarget;
///
/// const SEED: u64 = 0xEPSILON;
/// let bridge = EmbeddingBridge::<768, 3>::with_seed(SEED);
/// // ... build payload from token embeddings ...
/// let result = sys_teleport_context(
///     &mut agent_b_manifold,
///     payload,
///     &mut agent_b_governor,
///     TeleportTarget::LocalVoid,
/// );
/// ```
pub fn sys_teleport_context<const D: usize>(
    manifold: &mut HollowCubeManifold<D>,
    payload: ManifoldPayload<D>,
    governor: &mut SurgeryGovernor,
    target: TeleportTarget,
) -> TeleportResult {
    // ── Route check ─────────────────────────────────────────────────────────
    match &target {
        TeleportTarget::RemoteVoid(_descriptor) => {
            // FUTURE_WORK: Serialize ManifoldPayload<D> via serde feature, route
            // to descriptor.agent_id over the configured transport layer.
            // For now we return a defined, recoverable error.
            return TeleportResult::RemoteUnimplemented;
        }
        TeleportTarget::LocalVoid => {
            // fall through to local injection
        }
    }

    // ── Void occupancy pre-check ─────────────────────────────────────────────
    if !manifold.void_is_empty() {
        return TeleportResult::VoidBusy;
    }

    // ── Step 1: Governor Clutch — zero β for exactly one tick ────────────────
    let permit = governor.prepare_for_surgery();

    // ── Step 2: Void Injection — Betti-guarded write ─────────────────────────
    let inject_result = manifold.inject_into_void(payload);
    match inject_result {
        Err(err) => {
            // Surgery aborted — still restore governor to avoid stuck β=0
            governor.complete_surgery(permit);
            return match err {
                SurgeryError::VoidOccupied => TeleportResult::VoidBusy,
                other => TeleportResult::TopologyRejected(other),
            };
        }
        Ok(()) => {}
    }

    // ── Step 3: Wake-Up Rescan — assimilate into shell ───────────────────────
    let points_assimilated = manifold.assimilate();

    // ── Step 4: Governor Restore — re-enable derivative damping ─────────────
    governor.complete_surgery(permit);

    TeleportResult::Success { points_assimilated }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::{EpsilonPoint, SparseGraph, ManifoldPayload};

    /// Build a minimal connected shell for receiving tests.
    fn make_connected_shell<const D: usize>(epsilon: f64) -> HollowCubeManifold<D>
    where
        [(); D]:,
    {
        let mut m = HollowCubeManifold::<D>::new(epsilon);
        // Use unit-axis points — close enough to be connected at epsilon=1.5
        m.add_shell_point(EpsilonPoint::new({
            let mut c = [0.0f64; D];
            c[0] = 1.0;
            c
        }));
        m.add_shell_point(EpsilonPoint::new({
            let mut c = [0.0f64; D];
            c[0] = 0.9;
            c[1] = 0.1;
            c
        }));
        m.add_shell_point(EpsilonPoint::new({
            let mut c = [0.0f64; D];
            c[0] = 0.9;
            c[2 % D] = 0.1;
            c
        }));
        m
    }

    /// Build a connected 2-point payload with β₀ = 1.
    fn make_connected_payload() -> ManifoldPayload<3> {
        let mut src = SparseGraph::<3>::new(2.0);
        src.add_point(EpsilonPoint::new([0.0, 0.5, 0.5]));
        src.add_point(EpsilonPoint::new([0.1, 0.5, 0.5]));
        ManifoldPayload::from_graph(&src, 5.0)
    }

    // ─── Test 1: Full pipeline success ────────────────────────────────────────

    #[test]
    fn test_teleport_local_void_success() {
        let mut manifold = make_connected_shell::<3>(1.5);
        let payload = make_connected_payload();
        let mut gov = SurgeryGovernor::new();

        let result = sys_teleport_context(
            &mut manifold,
            payload,
            &mut gov,
            TeleportTarget::LocalVoid,
        );

        match result {
            TeleportResult::Success { points_assimilated } => {
                assert!(points_assimilated > 0,
                    "At least one point must be assimilated");
            }
            other => panic!("Expected Success, got {:?}", other),
        }

        // Governor must be restored (β must be non-zero again)
        assert!(gov.beta() > 0.0, "Governor β must be restored after surgery");
        // Void must be empty again
        assert!(manifold.void_is_empty(), "Void must empty after assimilation");
    }

    // ─── Test 2: Topology rejection ───────────────────────────────────────────

    #[test]
    fn test_teleport_topology_rejected() {
        let mut manifold = make_connected_shell::<3>(1.5);
        let mut gov = SurgeryGovernor::new();

        // Disconnected payload: β₀ = 2 (two isolated points)
        let mut src = SparseGraph::<3>::new(0.01);
        src.add_point(EpsilonPoint::new([0.0, 0.0, 0.0]));
        src.add_point(EpsilonPoint::new([100.0, 100.0, 100.0]));
        let bad_payload = ManifoldPayload::from_graph(&src, 5.0);

        assert_eq!(bad_payload.signature_b0, 2);

        let result = sys_teleport_context(
            &mut manifold,
            bad_payload,
            &mut gov,
            TeleportTarget::LocalVoid,
        );

        assert_eq!(
            result,
            TeleportResult::TopologyRejected(
                SurgeryError::TopologyMismatch { expected_b0: 1, actual_b0: 2 }
            ),
            "Disconnected payload must be rejected"
        );

        // Governor must still be restored even on rejection
        assert!(gov.beta() > 0.0, "Governor β must be restored even on rejection");
        // Void must still be empty
        assert!(manifold.void_is_empty(), "Void must remain empty after rejection");
    }

    // ─── Test 3: Void-busy guard ──────────────────────────────────────────────

    #[test]
    fn test_teleport_void_busy() {
        let mut manifold = make_connected_shell::<3>(1.5);
        let mut gov = SurgeryGovernor::new();

        // Manually occupy the void (direct injection, bypassing governor)
        let payload1 = make_connected_payload();
        manifold.inject_into_void(payload1).expect("First injection must succeed");
        assert!(!manifold.void_is_empty());

        // Now teleport should return VoidBusy without touching the governor
        let payload2 = make_connected_payload();
        let result = sys_teleport_context(
            &mut manifold,
            payload2,
            &mut gov,
            TeleportTarget::LocalVoid,
        );

        assert_eq!(result, TeleportResult::VoidBusy,
            "Occupied void must return VoidBusy");
        // Governor should NOT have issued a permit (it was caught before step 1)
        // β still at default value (not zeroed for surgery)
        assert!(gov.beta() > 0.0);
    }

    // ─── Test 4: RemoteVoid → RemoteUnimplemented ─────────────────────────────

    #[test]
    fn test_teleport_remote_void_unimplemented() {
        let mut manifold = make_connected_shell::<3>(1.5);
        let payload = make_connected_payload();
        let mut gov = SurgeryGovernor::new();

        let result = sys_teleport_context(
            &mut manifold,
            payload,
            &mut gov,
            TeleportTarget::RemoteVoid(RemoteVoidDescriptor::new(0xDEAD_BEEF)),
        );

        assert_eq!(result, TeleportResult::RemoteUnimplemented,
            "RemoteVoid must return RemoteUnimplemented until transport is implemented");
        // Void must remain untouched
        assert!(manifold.void_is_empty());
    }
}
