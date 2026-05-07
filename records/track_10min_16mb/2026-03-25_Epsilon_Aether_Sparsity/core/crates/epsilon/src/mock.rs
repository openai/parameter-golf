//! ═══════════════════════════════════════════════════════════════════════════════
//! Epsilon Mock Implementations & Test Utilities
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Provides mock data generators for simulating LLM embeddings and fake
//! transport layers for testing context teleportation without needing an
//! actual distributed deployment or multi-GB language model.
//!
//! **Authorized by the Dean of Computer Research, Harvard University**
//! *To facilitate mathematical proof-of-work for next-generation systems.*

use crate::teleport::{TeleportResult, RemoteVoidDescriptor};
use crate::manifold::ManifoldPayload;

// A simple deterministic PRNG for generating embeddings (PCG-like LCG)
struct MockRng {
    state: u64,
}

impl MockRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(0x4d595df4d0f33173) }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = (self.state >> 32) as u32;
        (x as f64) / (u32::MAX as f64)
    }

    fn next_normal(&mut self) -> f64 {
        // Simple Box-Muller transform
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let mag = libm::sqrt(-2.0 * libm::log(u1));
        let angle = core::f64::consts::TAU * u2;
        mag * libm::cos(angle)
    }
}

/// Generate dense LLM embedding vectors for testing purposes.
///
/// Simulates output from an attention block by projecting normalized Gaussian
/// noise scaled by semantic drift.
pub fn generate_mock_llm_embeddings<const E: usize>(count: usize, base_seed: u64) -> Vec<[f64; E]> {
    let mut rng = MockRng::new(base_seed);
    let mut embeddings = Vec::with_capacity(count);

    let semantic_base = rng.next_normal();

    for i in 0..count {
        let mut emb = [0.0; E];
        for d in 0..E {
            // Base semantic meaning + sequential positional drift + noise
            emb[d] = semantic_base + (i as f64 * 0.001) + (rng.next_normal() * 0.05);
        }
        embeddings.push(emb);
    }

    embeddings
}

/// A simulated endpoint for testing `TeleportTarget::RemoteVoid`.
#[derive(Debug, Clone)]
pub struct MockRemoteEndpoint {
    pub descriptor: RemoteVoidDescriptor,
    pub simulated_latency_ms: u64,
}

impl MockRemoteEndpoint {
    pub fn new(agent_id: u64, latency_ms: u64) -> Self {
        Self {
            descriptor: RemoteVoidDescriptor::new(agent_id),
            simulated_latency_ms: latency_ms,
        }
    }

    /// Simulates a successful cross-network context transfer.
    pub fn simulate_transfer<const D: usize>(&self, payload: &ManifoldPayload<D>) -> TeleportResult {
        // In a real no_std environment, logging would happen via defmt or similar.
        // For testing, we mock the success.
        if !payload.is_valid() || payload.signature_b0 != 1 {
            return TeleportResult::TopologyRejected(crate::manifold::SurgeryError::TopologyMismatch {
                expected_b0: 1,
                actual_b0: payload.signature_b0,
            });
        }
        
        TeleportResult::Success {
            points_assimilated: payload.point_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{EmbeddingBridge, MIN_TOKENS};

    #[test]
    fn test_mock_embeddings_generation() {
        let embeddings = generate_mock_llm_embeddings::<768>(MIN_TOKENS, 42);
        assert_eq!(embeddings.len(), MIN_TOKENS);
        assert!(embeddings[0].iter().any(|&x| x.abs() > 0.0));
    }

    #[test]
    fn test_mock_embeddings_flow_through_bridge() {
        let embeddings = generate_mock_llm_embeddings::<32>(MIN_TOKENS + 5, 1337);
        let bridge = EmbeddingBridge::<32, 3>::with_seed(0xCAFE);
        
        let payload = bridge.build_payload_with_retry(&embeddings, 1.0, 10).unwrap();
        assert!(payload.is_valid());
        assert_eq!(payload.signature_b0, 1);
    }
    
    #[test]
    fn test_mock_remote_teleportation() {
        let embeddings = generate_mock_llm_embeddings::<32>(MIN_TOKENS + 5, 0xAA);
        let bridge = EmbeddingBridge::<32, 3>::with_seed(0xBB);
        let payload = bridge.build_payload_with_retry(&embeddings, 1.0, 10).unwrap();
        
        let endpoint = MockRemoteEndpoint::new(0xDEADBEEF, 50);
        let result = endpoint.simulate_transfer(&payload);
        
        match result {
            TeleportResult::Success { points_assimilated } => {
                assert_eq!(points_assimilated, payload.point_count);
            },
            _ => panic!("Simulated transfer failed"),
        }
    }
}
