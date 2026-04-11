# Epsilon

**Geometric State Transfer via Topological Surgery on Hollow Manifolds**

*Reference Implementation v0.1.0*

**Author:** Teerth Sharma

---

## Abstract

The dominant bottleneck in large language model inference is not arithmetic — it is sequential context loading. For a sequence of N tokens, the KV-cache attention mechanism requires O(N²) operations. This work addresses that bottleneck at the geometric level rather than the algorithmic one.

We present Epsilon, a framework for **O(P) geometric state transfer between autonomous agents**, where P is a constant bounded by the payload point budget (P ≤ 64). The approach rests on three independently established results: the Johnson-Lindenstrauss dimensionality reduction lemma, the homology of the 2-sphere, and Chebyshev's inequality applied to memory liveness estimation.

The core mechanism: an agent's converged semantic state is projected onto the unit 2-sphere S² via a seeded random linear map followed by L2 normalization. This produces a point cloud whose topology is formally verified (β₀ = 1, β₁ = 0, β₂ = 1) before it is injected into the hollow interior of a receiving agent's manifold. The injection is guarded by three safety mechanisms — a derivative-zeroing governor clutch, a Chebyshev-bounded liveness inheritance protocol, and a topological assimilation rescan — collectively preventing oscillation instability and premature garbage collection.

---

## Background

### The KV-Cache Bottleneck

Let a sequence of N tokens be processed by a transformer with hidden dimension E. The key-value cache requires O(N·E) memory and the attention computation require O(N²·E) operations. For modern long-context models (N = 10⁵ to 10⁶), this is the dominant cost. Retrieval-Augmented Generation (RAG) mitigates this by reducing effective N but does not eliminate the O(N²) scaling within the retrieved window.

### Geometric Alternative

Epsilon does not operate on the token stream. Instead, it operates on the **converged geometric representation** of that stream. After a source agent has processed its context (incurring the full O(N²) cost once), its semantic state is expressed as a low-dimensional manifold. This manifold — a point cloud on S² — is the payload. Transfer cost is O(P) in the number of points, independent of N.

This is not a replacement for language model inference. It is a reuse mechanism: compute once, transfer many times at constant cost.

---

## Mathematical Foundations

### 1. LLM Embeddings are Spherically Distributed

Modern transformer language models use cosine similarity as their primary similarity metric:

```
sim(u, v) = (u · v) / (‖u‖ · ‖v‖)
```

Cosine similarity is the inner product of L2-normalized vectors. This means the semantic geometry of token representations is inherently **angular**: the meaningful quantity is the direction of the embedding vector, not its magnitude. Constraining vectors to the unit hypersphere makes this implicit structure explicit.

### 2. Johnson-Lindenstrauss Projection

**Lemma (Johnson-Lindenstrauss, 1984):** For any ε ∈ (0, 1/2) and any set P of n points in ℝ^E, there exists a linear map f: ℝ^E → ℝ^k with k = O(log n / ε²) such that for all u, v ∈ P:

```
(1 - ε) ‖u - v‖² ≤ ‖f(u) - f(v)‖² ≤ (1 + ε) ‖u - v‖²
```

A random matrix M with entries drawn i.i.d. from N(0, 1/k) satisfies this condition with high probability. With a fixed seed, M is deterministic: two agents using the same seed produce **identical projections** for identical semantic inputs — a prerequisite for cross-agent topology matching.

### 3. Spherical Normalization Guarantees β₂ = 1

After JL projection from ℝ^E to ℝ^3, each point is L2-normalized:

```
p̂ = f(e) / ‖f(e)‖₂    →    p̂ ∈ S² ⊂ ℝ³
```

By the Universal Coefficient Theorem, the homology of the 2-sphere is:

```
H_k(S²; ℤ) ≅  ℤ   for k = 0, 2
              0    otherwise
```

This gives Betti numbers β₀ = 1, β₁ = 0, **β₂ = 1**. A point cloud sufficiently sampling S² recovers this signature through the Vietoris-Rips filtration. The hollow manifold constraint is not imposed — it is derived from the geometry.

### 4. Sampling Density Bound

By the Niyogi-Smale-Weinberger theorem, a point cloud within Hausdorff distance ε of a manifold with reach τ recovers the homotopy type of that manifold when ε < τ/2. For the unit 2-sphere, τ = 1. The required density condition ε < 0.5 is met by approximately 20 uniformly distributed points on S². This defines the hard minimum: **`MIN_TOKENS = 20`**.

### 5. Safety Mechanisms

#### 5.1 Surgery Permit (Governor Clutch)
The geometric governor applies a PD control law:

```
ε(t+1) = ε(t) + α·e(t) + β·(de/dt)
```

Instantaneous state injection causes de/dt → ∞, which drives the derivative term into instability. Before injection, the governor issues a `SurgeryPermit` — a one-shot token that zeroes β for exactly one adaptation tick, absorbing the discontinuity. The token is non-Clone and non-Copy; `complete_surgery()` consumes it and restores β.

#### 5.2 Chebyshev Liveness Inheritance
The memory evictor uses Chebyshev's inequality to bound false eviction probability:

```
P(|X - μ| ≥ k·σ) ≤ 1/k²
Safe boundary: liveness > μ - k·σ
```

Injected data arrives with t_alive = 0, making it immediately evictable. The `LivenessAnchor` carries the source agent's heap statistics (μ, σ, k), initializing the new object at μ + σ — provably safe with P(eviction) ≤ 1/k².

#### 5.3 Topological Assimilation Rescan
Post-injection, the receiving manifold's kernel verifies Betti boundaries of the payload against the inner walls of the hollow cube. If the topologies are compatible (β₀ = 1 for the payload), the points are merged into the active shell graph. The void empties and is ready for the next transfer.

---

## Crate Structure

```
crates/epsilon/
├── src/
│   ├── lib.rs        — Documentation and public API
│   ├── bridge.rs     — EmbeddingBridge: ℝ^E → S² (JL + L2)
│   ├── manifold.rs   — HollowCubeManifold, ManifoldPayload, SparseGraph
│   ├── governor.rs   — SurgeryGovernor, SurgeryPermit
│   ├── memory.rs     — LivenessAnchor, ChebyshevGuard
│   └── teleport.rs   — sys_teleport_context orchestration

crates/aether-core/
│   — Mathematical foundation (ManifoldPoint, topology, PD control, heap)
```

---

## Usage

### Minimal Example

```rust
use epsilon::{EmbeddingBridge, sys_teleport_context, HollowCubeManifold, SurgeryGovernor};

// Shared seed ensures identical projection matrices across agents
const CANONICAL_SEED: u64 = 0xEPSILON_SEED;

// Agent A: build payload from its processed token embeddings
// (embeddings are &[[f64; 768]] — one row per token)
let bridge = EmbeddingBridge::<768, 3>::with_seed(CANONICAL_SEED);
let payload = bridge.build_payload(&token_embeddings, source_liveness_score)?;

// Agent B: receive geometric state — O(P) cost regardless of N
let result = sys_teleport_context(
    &mut agent_b_manifold,
    payload,
    &mut agent_b_governor,
);
// → TeleportResult::Success { points_assimilated: P }
```

### Building

```bash
cargo check -p epsilon          # Type-check
cargo test  -p epsilon          # Run all 26 unit tests
cargo test  --workspace         # Full workspace
```

---

## Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Payload construction (bridge) | O(N·E·D) | N tokens, E→D projection |
| Topology verification | O(P²) | P ≤ 64 payload points |
| Governor clutch | O(1) | Permit acquire + restore |
| Void injection | O(P) | Betti check + write |
| Assimilation rescan | O(P·V) | Merge into shell graph |
| **Full transfer pipeline** | **O(P)** | **Per-agent receive cost** |
| Baseline KV-cache attention | O(N²) | Per-agent, per-context |

Source agent pays O(N·E·D) once. Every receiving agent pays O(P) ≈ O(64). For N = 10⁶ tokens and E = 4096, this is a reduction from ~4×10¹² operations to ~64.

---

## Test Coverage (26 tests)

| Module | Tests |
|--------|-------|
| `bridge.rs` | 8 — projection determinism, sphere geometry, Betti signature, end-to-end, cross-agent identity |
| `manifold.rs` | 7 — Euclidean distance, Betti-0/1, hollow constraint, injection, mismatch rejection, assimilation |
| `governor.rs` | 5 — ε bounds, threshold, permit zeroing, permit restoration |
| `memory.rs` | 4 — anchor statistics, safety boundary, inheritance protection |
| `teleport.rs` | 3 — full pipeline, topology rejection, void occupancy |

---

## Limitations and Future Work

1. **No end-to-end Rust transformer integration.** The bridge accepts `&[[f64; E]]` — a caller must extract embeddings from a running model. Integration with `candle`, `llama.cpp`, or similar is straightforward but out of scope for this reference implementation.

2. **~~Fixed D = 3~~** (Resolved) Extending to D > 3 is now formally supported via Euler characteristic Betti-2 computation (`compute_betti_2_euler()`).

3. **~~Proof of β₂ detection is probabilistic~~** (Resolved) Edge cases where exactly 20 tokens produce disconnected graphs are now automatically resolved via the `build_graph_with_retry` widening-ε fallback.

4. **~~No network transport layer~~** (Resolved - Interface) Context transfer now defines a formal `TeleportTarget::RemoteVoid` API boundary and payload structs are fully `serde`-compatible for standard wire transport.

---

## License

MIT — Teerth Sharma, 2026
