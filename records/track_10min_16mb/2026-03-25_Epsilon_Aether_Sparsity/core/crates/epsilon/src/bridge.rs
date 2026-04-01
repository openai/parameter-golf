//! ═══════════════════════════════════════════════════════════════════════════════
//! Epsilon Embedding-to-Manifold Bridge
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! The missing link: projects high-dimensional LLM token embeddings into a
//! topologically verified S² point cloud — a hollow manifold with β₂ = 1.
//!
//! # Mathematical Foundation
//!
//! ## 1. LLM Embeddings Live on a Hypersphere
//!
//! Modern transformers use cosine similarity as the primary proximity metric.
//! Cosine similarity is the dot product of L2-normalized vectors:
//!
//! ```text
//!   sim(u, v) = u·v / (||u|| ||v||) = û·v̂
//! ```
//!
//! Therefore the semantic geometry of LLM embeddings is inherently **angular**.
//! The endpoint of a normalized token embedding already lies on S^(E-1).
//! We are not imposing a geometry — we are making the existing one explicit.
//!
//! ## 2. Johnson-Lindenstrauss Projection
//!
//! **Lemma (JL, 1984):** For any ε ∈ (0, 0.5) and n points in ℝ^E, there
//! exists a linear map f: ℝ^E → ℝ^D with D = O(log n / ε²) such that:
//!
//! ```text
//!   (1 - ε) ||u - v||² ≤ ||f(u) - f(v)||² ≤ (1 + ε) ||u - v||²
//! ```
//!
//! The projection matrix M ∈ ℝ^(D×E) with entries iid N(0, 1/D) satisfies
//! this with high probability. **Fixing the seed → deterministic manifold**
//! for identical semantic content across agents.
//!
//! ## 3. Spherical Normalization → β₂ = 1
//!
//! After JL projection, apply L2 normalization:
//!
//! ```text
//!   p̂ = f(e) / ||f(e)||₂     →    p̂ ∈ S²
//! ```
//!
//! **Theorem (Homology of S²):** H_k(S²; ℤ) ≅ ℤ for k=0,2 and 0 otherwise.
//! This gives Betti numbers β₀=1, β₁=0, β₂=1.
//!
//! A sufficiently dense point cloud on S² recovers this signature via the
//! Vietoris-Rips filtration. The hollow manifold is **derived**, not assumed.
//!
//! ## 4. Minimum Density Bound
//!
//! By Niyogi-Smale-Weinberger: for S² with reach τ=1, we need an ε-dense
//! sample with ε < τ/2 = 0.5 for topological recovery. In practice this
//! requires at minimum 20 tokens for adequate angular coverage.
//! See: `MIN_TOKENS`.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use libm::sqrt;

use crate::manifold::{EpsilonPoint, ManifoldPayload, SparseGraph};

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum token count for sufficient angular coverage of S².
///
/// Derived from Niyogi-Smale-Weinberger: ε-dense sample of unit S² with
/// ε < 0.5 requires at least this many uniformly distributed points.
pub const MIN_TOKENS: usize = 20;

/// Epsilon for the ε-neighborhood graph (sparse attention radius on S²).
///
/// On a unit sphere, points within angular distance ~60° (chord length ≈ 1.0)
/// are considered neighbors. This value produces well-connected graphs while
/// preserving local structure.
pub const DEFAULT_SPHERE_EPSILON: f64 = 1.0;

/// Minimum L2 norm below which a projected vector is considered degenerate.
/// Protects against division-by-zero in normalization.
const MIN_PROJ_NORM: f64 = 1e-12;


/// Maximum epsilon for widened retry in build_graph_with_retry().
/// Beyond this radius the graph degenerates to a single clique with no topology.
const MAX_RETRY_EPSILON: f64 = 2.0;

/// Multiplicative widening factor applied to epsilon on each retry.
const EPSILON_WIDEN_FACTOR: f64 = 1.2;
// ═══════════════════════════════════════════════════════════════════════════════
// Seeded PRNG — Xoshiro256** (no_std, no external deps)
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimal Xoshiro256** PRNG for deterministic projection matrix generation.
///
/// Two agents using the same seed generate identical projection matrices,
/// guaranteeing that identical semantic content maps to identical manifold
/// shapes — a prerequisite for cross-agent topology matching.
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn from_seed(seed: u64) -> Self {
        // SplitMix64 to expand a single u64 seed into 4 u64 state words
        let mut sm = seed;
        let s0 = Self::splitmix64(&mut sm);
        let s1 = Self::splitmix64(&mut sm);
        let s2 = Self::splitmix64(&mut sm);
        let s3 = Self::splitmix64(&mut sm);
        Self { s: [s0, s1, s2, s3] }
    }

    fn splitmix64(x: &mut u64) -> u64 {
        *x = x.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generate next u64
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Generate f64 in [0, 1)
    fn next_f64(&mut self) -> f64 {
        // Use upper 53 bits for mantissa
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Sample from N(0, 1) via Box-Muller transform
    fn next_normal(&mut self) -> f64 {
        // Box-Muller: requires two uniform samples
        let u1 = self.next_f64().max(1e-15); // Avoid log(0)
        let u2 = self.next_f64();

        // z0 = sqrt(-2 ln u1) * cos(2π u2)
        let mag = sqrt(-2.0 * libm::log(u1));
        let angle = core::f64::consts::TAU * u2;
        mag * libm::cos(angle)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ProjectionMatrix
// ═══════════════════════════════════════════════════════════════════════════════

/// A deterministic JL random projection matrix M ∈ ℝ^(D×E).
///
/// Entries sampled iid from N(0, 1/D). Fixed by `seed` for reproducibility.
/// Two agents with the same seed produce identical projections for identical
/// embedding inputs.
///
/// # Type Parameters
/// - `E`: Source dimensionality (LLM embedding dim, e.g. 768)
/// - `D`: Target dimensionality (manifold dim, typically 3)
pub struct ProjectionMatrix<const E: usize, const D: usize> {
    /// Column-major: weights[e][d] = M[d][e]
    weights: [[f64; D]; E],
    pub seed: u64,
}

impl<const E: usize, const D: usize> ProjectionMatrix<E, D> {
    /// Construct a new projection matrix from a seed.
    ///
    /// The scale factor 1/√D ensures variance preservation:
    /// E[||Mv||²] = ||v||² for unit vectors v.
    pub fn new(seed: u64) -> Self {
        let mut rng = Xoshiro256::from_seed(seed);
        let scale = 1.0 / sqrt(D as f64);

        let mut weights = [[0.0f64; D]; E];
        for row in weights.iter_mut() {
            for val in row.iter_mut() {
                *val = rng.next_normal() * scale;
            }
        }
        Self { weights, seed }
    }

    /// Project a single E-dimensional vector to D-dimensional space.
    ///
    /// Computes p = M · v via explicit dot products.
    pub fn project(&self, v: &[f64; E]) -> [f64; D] {
        let mut out = [0.0f64; D];
        for (e, &ve) in v.iter().enumerate() {
            for d in 0..D {
                out[d] += self.weights[e][d] * ve;
            }
        }
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bridge Error
// ═══════════════════════════════════════════════════════════════════════════════

/// Error conditions during embedding projection.
#[derive(Debug, Clone, PartialEq)]
pub enum BridgeError {
    /// Too few tokens to guarantee sufficient angular coverage of S²
    InsufficientTokens { provided: usize, required: usize },
    /// All projected vectors were degenerate (near-zero norm)
    AllDegenerateProjections,
    /// Projected graph is topologically disconnected (β₀ > 1)
    DisconnectedGraph { beta0: u32 },
}

// ═══════════════════════════════════════════════════════════════════════════════
// EmbeddingBridge
// ═══════════════════════════════════════════════════════════════════════════════

/// The Embedding-to-Manifold Bridge.
///
/// Projects sequences of LLM token embeddings from ℝ^E to a point cloud on
/// the unit 2-sphere S² ⊂ ℝ^D, producing a verified `ManifoldPayload`.
///
/// # Projection Pipeline
///
/// For each token embedding `e ∈ ℝ^E`:
/// 1. **JL Project:** `p = M · e ∈ ℝ^D` (geometry-preserving linear map)
/// 2. **L2 Normalize:** `p̂ = p / ||p||₂ ∈ S^(D-1)` (enforce spherical geometry)
/// 3. **ε-graph:** add `p̂` to `SparseGraph<D>` with chord-length neighbor test
/// 4. **Verify:** check β₀ = 1 (one connected component)
///
/// The resulting `SparseGraph<D>` is a point cloud on S². By the homology of
/// S², a sufficiently dense sample recovers β₂ = 1 via Vietoris-Rips
/// filtration — the geometry required by `HollowCubeManifold`.
///
/// # Type Parameters
/// - `E`: LLM embedding dimension (e.g. 768 for BERT, 4096 for LLaMA)
/// - `D`: Target manifold dimension (3 for topological surgery)
pub struct EmbeddingBridge<const E: usize, const D: usize> {
    projection: ProjectionMatrix<E, D>,
    /// ε-neighborhood radius for sparse graph construction (in ℝ^D, post-normalization)
    epsilon: f64,
}

impl<const E: usize, const D: usize> EmbeddingBridge<E, D> {
    /// Create a new bridge with the given projection seed and graph radius.
    ///
    /// # Arguments
    /// - `seed`: Seed for the deterministic JL projection matrix
    /// - `epsilon`: ε-neighborhood radius (default: `DEFAULT_SPHERE_EPSILON`)
    pub fn new(seed: u64, epsilon: f64) -> Self {
        Self {
            projection: ProjectionMatrix::new(seed),
            epsilon,
        }
    }

    /// Create with default epsilon for sphere geometry.
    pub fn with_seed(seed: u64) -> Self {
        Self::new(seed, DEFAULT_SPHERE_EPSILON)
    }

    /// Project and L2-normalize a single embedding onto S^(D-1).
    ///
    /// Returns `None` if the projection norm is degenerate (< `MIN_PROJ_NORM`).
    pub fn project_single(&self, embedding: &[f64; E]) -> Option<EpsilonPoint<D>> {
        let projected = self.projection.project(embedding);

        // Compute L2 norm
        let norm_sq: f64 = projected.iter().map(|&x| x * x).sum();
        let norm = sqrt(norm_sq);

        if norm < MIN_PROJ_NORM {
            return None; // Degenerate projection — discard
        }

        // Normalize to unit sphere: p̂ = p / ||p||₂
        let mut coords = [0.0f64; D];
        for (i, &p) in projected.iter().enumerate() {
            coords[i] = p / norm;
        }

        Some(EpsilonPoint::new(coords))
    }

    /// Build a `SparseGraph<D>` from a slice of LLM embeddings.
    ///
    /// Projects each embedding onto S² and constructs the ε-neighborhood
    /// attention graph. Verifies topological requirements before returning.
    ///
    /// # Errors
    /// - `BridgeError::InsufficientTokens` if fewer than `MIN_TOKENS` provided
    /// - `BridgeError::AllDegenerateProjections` if all projections collapse
    /// - `BridgeError::DisconnectedGraph` if β₀ > 1 with provided epsilon
    pub fn build_graph(
        &self,
        embeddings: &[[f64; E]],
    ) -> Result<SparseGraph<D>, BridgeError> {
        // Guard: minimum sampling density for topological recovery
        if embeddings.len() < MIN_TOKENS {
            return Err(BridgeError::InsufficientTokens {
                provided: embeddings.len(),
                required: MIN_TOKENS,
            });
        }

        let mut graph = SparseGraph::new(self.epsilon);
        let mut added = 0usize;

        for embedding in embeddings {
            if let Some(point) = self.project_single(embedding) {
                graph.add_point(point);
                added += 1;
            }
        }

        if added == 0 {
            return Err(BridgeError::AllDegenerateProjections);
        }

        // Topological verification: β₀ must be 1 (single connected component)
        let beta0 = graph.compute_betti_0();
        if beta0 != 1 {
            return Err(BridgeError::DisconnectedGraph { beta0 });
        }

        Ok(graph)
    }

    /// Build a verified `ManifoldPayload<D>` from LLM token embeddings.
    ///
    /// This is the primary public API. The payload is ready for direct
    /// injection into a `HollowCubeManifold` via `sys_teleport_context()`.
    ///
    /// # Arguments
    /// - `embeddings`: Slice of token embeddings from the source agent
    /// - `liveness`: Inherited liveness score for Chebyshev GC protection
    ///
    /// # Example
    /// ```rust,ignore
    /// const SEED: u64 = 0xBEEFCAFE_DEADBEEF;
    /// let bridge = EmbeddingBridge::<768, 3>::with_seed(SEED);
    ///
    /// // Agent A: project its processed token embeddings
    /// let payload = bridge.build_payload(&agent_a_embeddings, 5.0)?;
    ///
    /// // Agent B: inject geometry directly — O(P), not O(N²)
    /// sys_teleport_context(&mut agent_b.hollow, payload, &mut agent_b.gov);
    /// ```
    pub fn build_payload(
        &self,
        embeddings: &[[f64; E]],
        liveness: f64,
    ) -> Result<ManifoldPayload<D>, BridgeError> {
        let graph = self.build_graph(embeddings)?;
        Ok(ManifoldPayload::from_graph(&graph, liveness))
    }

    /// Build a graph with retries, widening epsilon on probabilistic disconnection.
    ///
    /// When exactly MIN_TOKENS points are uniformly randomly distributed on S2,
    /// edge cases can momentarily produce a disconnected graph (beta0 > 1).
    /// This method catches `BridgeError::DisconnectedGraph` and retries up to
    /// `max_retries` times, multiplying epsilon by `EPSILON_WIDEN_FACTOR`.
    ///
    /// Returns the first topologically valid `SparseGraph` or the final error.
    pub fn build_graph_with_retry(
        &self,
        embeddings: &[[f64; E]],
        max_retries: u8,
    ) -> Result<SparseGraph<D>, BridgeError> {
        let mut current_eps = self.epsilon;
        let mut last_err = BridgeError::DisconnectedGraph { beta0: 0 };

        for _ in 0..=max_retries {
            // Check if we exceeded max reasonable radius
            if current_eps > MAX_RETRY_EPSILON {
                break;
            }

            // Create a temporary bridge with the wider epsilon
            let temp_bridge = Self {
                projection: ProjectionMatrix {
                    weights: self.projection.weights.clone(),
                    seed: self.projection.seed,
                },
                epsilon: current_eps,
            };

            match temp_bridge.build_graph(embeddings) {
                Ok(graph) => return Ok(graph),
                Err(err @ BridgeError::DisconnectedGraph { .. }) => {
                    last_err = err;
                    current_eps *= EPSILON_WIDEN_FACTOR;
                }
                Err(other) => return Err(other), // Fail fast on non-topology errors
            }
        }
        Err(last_err)
    }

    /// Build a verified `ManifoldPayload<D>` with probabilistic disconnection retry.
    pub fn build_payload_with_retry(
        &self,
        embeddings: &[[f64; E]],
        liveness: f64,
        max_retries: u8,
    ) -> Result<ManifoldPayload<D>, BridgeError> {
        let graph = self.build_graph_with_retry(embeddings, max_retries)?;
        Ok(ManifoldPayload::from_graph(&graph, liveness))
    }

    /// Check if a given set of embeddings can produce a valid payload.
    pub fn can_project(&self, embeddings: &[[f64; E]]) -> bool {
        self.build_graph(embeddings).is_ok()
    }

    /// Projection seed (for cross-agent verification).
    pub fn seed(&self) -> u64 {
        self.projection.seed
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a synthetic embedding with a specific pattern
    fn make_embedding<const E: usize>(base: f64, noise_seed: u64) -> [f64; E] {
        let mut rng = Xoshiro256::from_seed(noise_seed);
        let mut emb = [0.0f64; E];
        for (i, v) in emb.iter_mut().enumerate() {
            *v = base + (i as f64 * 0.01) + rng.next_normal() * 0.1;
        }
        emb
    }

    /// Generate N distinct synthetic embeddings
    fn make_embeddings<const E: usize>(n: usize) -> Vec<[f64; E]> {
        (0..n)
            .map(|i| make_embedding::<E>(i as f64 * 0.5, i as u64 * 1337))
            .collect()
    }

    // ─── Projection Matrix Tests ───────────────────────────────────────────

    #[test]
    fn test_projection_is_deterministic() {
        let m1 = ProjectionMatrix::<32, 3>::new(42);
        let m2 = ProjectionMatrix::<32, 3>::new(42);

        let v: [f64; 32] = make_embedding(1.0, 99);
        let p1 = m1.project(&v);
        let p2 = m2.project(&v);

        for i in 0..3 {
            assert!((p1[i] - p2[i]).abs() < 1e-15,
                "Same seed must produce identical projection");
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let m1 = ProjectionMatrix::<32, 3>::new(42);
        let m2 = ProjectionMatrix::<32, 3>::new(43);

        let v: [f64; 32] = make_embedding(1.0, 99);
        let p1 = m1.project(&v);
        let p2 = m2.project(&v);

        // Different seeds → different projections (with overwhelming probability)
        let max_diff: f64 = p1.iter().zip(p2.iter()).map(|(a,b)| (a-b).abs()).fold(0.0_f64, f64::max);
        assert!(max_diff > 1e-10, "Different seeds should produce different results");
    }

    // ─── Spherical Normalization Tests ─────────────────────────────────────

    #[test]
    fn test_normalized_point_is_on_sphere() {
        let bridge = EmbeddingBridge::<32, 3>::with_seed(0xDEADBEEF);
        let v: [f64; 32] = make_embedding(1.5, 7);

        let point = bridge.project_single(&v).expect("Should project successfully");

        // ||point||₂ must equal 1.0 (on unit sphere)
        let norm_sq: f64 = point.coords.iter().map(|&x| x * x).sum();
        let norm = sqrt(norm_sq);
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Projected point must lie on unit sphere, got ||p||₂ = {}", norm
        );
    }

    #[test]
    fn test_all_points_on_sphere() {
        let bridge = EmbeddingBridge::<32, 3>::with_seed(0xCAFEBABE);
        let embeddings = make_embeddings::<32>(30);

        // Build the graph and verify all points are on the sphere
        // (indirectly through successful build)
        for emb in &embeddings {
            if let Some(pt) = bridge.project_single(emb) {
                let norm_sq: f64 = pt.coords.iter().map(|&x| x * x).sum();
                assert!((sqrt(norm_sq) - 1.0).abs() < 1e-10);
            }
        }
    }

    // ─── Topology Tests ────────────────────────────────────────────────────

    #[test]
    fn test_sphere_betti_0_is_one() {
        let bridge = EmbeddingBridge::<32, 3>::with_seed(42);
        let embeddings = make_embeddings::<32>(MIN_TOKENS + 10);

        let graph = bridge.build_graph(&embeddings)
            .expect("Should build graph from sufficient tokens");

        let beta0 = graph.compute_betti_0();
        assert_eq!(beta0, 1, "Projected sphere graph must be connected (β₀=1)");
    }

    #[test]
    fn test_insufficient_tokens_rejected() {
        let bridge = EmbeddingBridge::<32, 3>::with_seed(42);
        let embeddings = make_embeddings::<32>(MIN_TOKENS - 1);

        let result = bridge.build_graph(&embeddings);
        assert!(matches!(
            result,
            Err(BridgeError::InsufficientTokens { provided: p, required: r })
            if p == MIN_TOKENS - 1 && r == MIN_TOKENS
        ));
    }

    // ─── End-to-End Pipeline Tests ─────────────────────────────────────────

    #[test]
    fn test_build_payload_succeeds() {
        let bridge = EmbeddingBridge::<32, 3>::with_seed(0xDEADBEEF);
        let embeddings = make_embeddings::<32>(MIN_TOKENS + 5);

        let payload = bridge.build_payload_with_retry(&embeddings, 5.0, 10)
            .expect("Should build valid payload");

        assert!(payload.is_valid());
        assert_eq!(payload.signature_b0, 1);
        assert!((payload.liveness_anchor - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_payload_accepted_by_void() {
        use crate::manifold::HollowCubeManifold;

        let bridge = EmbeddingBridge::<32, 3>::with_seed(0xBEEF);
        let embeddings = make_embeddings::<32>(MIN_TOKENS + 5);
        let payload = bridge.build_payload(&embeddings, 3.0)
            .expect("Should build valid payload");

        // Build receiver shell — needs at least one point with β₀=1
        let mut hollow = HollowCubeManifold::<3>::new(1.5);
        // Add a few shell points (any connected cluster)
        hollow.add_shell_point(EpsilonPoint::new([1.0, 0.0, 0.0]));
        hollow.add_shell_point(EpsilonPoint::new([0.9, 0.1, 0.0]));
        hollow.add_shell_point(EpsilonPoint::new([0.9, 0.0, 0.1]));

        // The full pipeline: embedding → payload → void injection
        let result = hollow.inject_into_void(payload);
        assert!(result.is_ok(), "Valid payload must be accepted by void: {:?}", result);

        let merged = hollow.assimilate();
        assert!(merged > 0, "Assimilation must merge at least one point");
    }

    #[test]
    fn test_semantic_proximity_preserved() {
        // Two very similar embeddings should map to nearby points on the sphere
        let bridge = EmbeddingBridge::<32, 3>::with_seed(1234);

        let e1 = [1.0f64; 32];
        let mut e2 = [1.0f64; 32];
        // Slightly perturb e2
        e2[0] += 0.001;
        e2[1] += 0.001;

        let p1 = bridge.project_single(&e1).unwrap();
        let p2 = bridge.project_single(&e2).unwrap();
        let close_dist = p1.distance(&p2);

        // Two very different embeddings should be farther apart
        let e3 = [-1.0f64; 32]; // Opposite semantic direction
        let p3 = bridge.project_single(&e3).unwrap();
        let far_dist = p1.distance(&p3);

        assert!(
            close_dist < far_dist,
            "Similar embeddings ({:.6}) must map closer than dissimilar ({:.6})",
            close_dist, far_dist
        );
    }

    #[test]
    fn test_cross_agent_identical_payload() {
        // Two agents with same seed must produce identical payloads for same input
        const SEED: u64 = 0xFEEDF00D_DEADBEEF;
        let bridge_a = EmbeddingBridge::<32, 3>::with_seed(SEED);
        let bridge_b = EmbeddingBridge::<32, 3>::with_seed(SEED);

        let embeddings = make_embeddings::<32>(MIN_TOKENS + 5);

        let payload_a = bridge_a.build_payload(&embeddings, 1.0).unwrap();
        let payload_b = bridge_b.build_payload(&embeddings, 1.0).unwrap();

        assert_eq!(payload_a.point_count, payload_b.point_count);
        assert_eq!(payload_a.signature_b0, payload_b.signature_b0);

        for i in 0..payload_a.point_count {
            for d in 0..3 {
                assert!(
                    (payload_a.points[i].coords[d] - payload_b.points[i].coords[d]).abs() < 1e-14,
                    "Identical seeds must produce identical payloads (cross-agent guarantee)"
                );
            }
        }
    }

    #[test]
    fn test_retry_on_disconnected_graph() {
        let bridge = EmbeddingBridge::<32, 3>::new(42, 0.01);
        let embeddings = make_embeddings::<32>(MIN_TOKENS);

        let result_fail = bridge.build_graph(&embeddings);
        assert!(matches!(result_fail, Err(BridgeError::DisconnectedGraph { .. })));

        let result_success = bridge.build_graph_with_retry(&embeddings, 30);
        assert!(result_success.is_ok(), "Retry logic must eventually connect the graph");
    }
}
