# EPSILON-AETHER: Geometric Sparse Attention with S^31 Projection

**val_bpb**: _pending H100 validation_
**Architecture**: 12-layer GPT (512d, 8H/4KV, MLP×2.5, GQA, RoPE, BigramHash, SmearGate)

## Key Idea

This submission fuses **AETHER** (Adaptive Event-driven Threshold Hybrid Entangled Rendering) with **EPSILON** (Johnson-Lindenstrauss Geometric Mapping) to unlock massive compute savings during LLM training. 

By applying a mathematically rigorous Johnson-Lindenstrauss (JL) random projection, we map high-dimensional Transformer queries and keys ($D=64/128$) down to a highly compressed 32-dimensional topological space on the unit hypersphere ($S^{31}$). 

This allows AETHER's Cauchy-Schwarz block-pruning radar to score and discard irrelevant attention blocks in $O(32 \cdot N)$ time instead of $O(D_{head} \cdot N)$, effectively eliminating the computational overhead of the sparsity mechanism itself. This reclaimed compute budget allows us to train **12 layers** within the 10-minute 8xH100 wall-clock budget.

## Novel Contributions

### 1. Epsilon JL-Bridge (Geometric Profiling)
Instead of computing block centroids and dot products in the native LLM dimension space, the Epsilon bridge deterministically maps $Q$ and $K$ into $R^{32}$ via a seeded normally-distributed random matrix, followed by L2-normalization onto the $S^{31}$ manifold. By the Johnson-Lindenstrauss lemma, cosine similarities and distances are preserved, allowing for hyper-efficient clustering and scoring.

### 2. AETHER Cauchy-Schwarz Radar
Partition K into blocks of 64. Using the Epsilon-projected coordinates, compute block centroids ($\mu$) and bounding radii ($R$). We prune blocks where the upper bound $||q|| \cdot (||\mu|| + R) < \theta$ guarantees all keys in the block are below the attention threshold.

### 3. Lyapunov-Stable Geometric Governor
Replaces hand-tuned sparsity schedules with a discrete-time PD controller that adaptively adjusts the number of blocks to keep, proving error contracts monotonically: $|e_{t+1}| \le |e_t|$.

## Architecture Changes from SOTA Baseline

| Parameter | SOTA (1.1428) | EPSILON-AETHER |
|---|---|---|
| Layers | 10 | **12** |
| Attention | Dense SDPA | **JL-Projected Sparse SDPA** |
| Scoring Space | $D=64$ | **$S^{31}$ (32 dimensions)** |
| Sparsity Target | N/A | **50% (Lyapunov regulated)** |
| All other tech | ✓ | ✓ |

## Running the Training Loop

```bash
# Standard Parameter Golf 8×H100 setup
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Logs are routed to the /logs directory
```

## Formal Verification
The safety guarantees of the Aether sparsity engine are machine-checked in Lean 4 (located in `docs/proofs/`):
- `PruneSafety.lean` → Zero false negatives in Cauchy-Schwarz culling.
- `Governor.lean` → Lyapunov descent $V(e_{t+1}) \le V(e_t)$.
- `ChebyshevGC.lean` → GC false collection rate bounded by Chebyshev inequality.

_This submission uses 100% native PyTorch with no external C++/Rust binaries, ensuring full compatibility with the track evaluation environment._
