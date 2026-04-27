# TurboQuant Integration Spec — Parameter Golf

**Created:** 2026-03-25
**Source:** Google Research — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874, ICLR 2026)
**Purpose:** Adapt TurboQuant/PolarQuant/QJL techniques for Parameter Golf weight compression and KV cache optimization

---

## Background: Competition Constraints

- **Artifact limit:** ≤16MB total (model weights + code + compression artifacts)
- **Training time:** ≤10 minutes on 8×H100
- **Metric:** bits-per-byte (bpb) — lower is better
- **Current #1:** 1.1428 bpb using int5 MLP + BigramHash(10240) + SWA
- **Baseline:** 1.2244 bpb
- **Our target:** 1.12-1.13 bpb

---

## Three Algorithms from the Paper

### 1. PolarQuant (MSE-Optimal Quantization)

**Core idea:** Random rotation makes weight distribution predictable → enables optimal per-coordinate quantization with zero overhead.

**Algorithm steps:**
1. **Normalize:** Given weight vector `x ∈ ℝ^d`, store `‖x‖₂` separately (float16 = 2 bytes). Normalize to unit norm: `x̂ = x / ‖x‖₂`
2. **Random rotation:** Apply a fixed random orthogonal matrix `R ∈ ℝ^{d×d}` to get `x̃ = R · x̂`. After rotation, each coordinate of `x̃` follows a Beta distribution: `f(t) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-t²)^((d-3)/2)`. For large d, this converges to `N(0, 1/d)`.
3. **Optimal scalar quantization:** For each coordinate independently, apply a Lloyd-Max quantizer precomputed for the Beta/Gaussian distribution. The codebook entries are found by solving a 1D k-means problem on the known distribution.
4. **Store:** b-bit integer codes per coordinate + norm (float16)

**Key property:** No per-group scale/zero-point needed! The rotation makes the distribution known a priori, eliminating the 1-2 bits of overhead that standard quantization wastes on calibration constants.

**Distortion bound at b bits:** `D_mse ≤ (√3·π/2) · 1/4^b`
- b=1: D ≈ 0.36
- b=2: D ≈ 0.117
- b=3: D ≈ 0.03
- b=4: D ≈ 0.009

**Theoretical lower bound:** `D_mse ≥ 1/4^b`, so TurboQuant is within 2.7× of optimal.

### 2. QJL (1-bit Inner Product Quantizer)

**Core idea:** A 1-bit quantizer that preserves inner products exactly in expectation.

**Algorithm:**
1. Generate random Gaussian matrix `S ∈ ℝ^{d×d}` with i.i.d. N(0,1) entries
2. **Quantize:** `Q_qjl(x) = sign(S · x)` → d-dimensional vector of {-1, +1}
3. **Dequantize:** `Q_qjl⁻¹(z) = (√(π/2) / d) · S^T · z`

**Properties:**
- Unbiased: `E[⟨y, Q⁻¹(Q(x))⟩] = ⟨y, x⟩`
- Variance: `Var ≤ (π / 2d) · ‖y‖²`

### 3. TurboQuant (Combined — Inner Product Optimal)

**Two-stage approach for b-bit budget:**
1. Apply PolarQuant at (b-1) bits → get quantized version `x̂` and residual `r = x - x̂`
2. Apply QJL (1-bit) to the residual `r` → corrects bias in inner product estimation

**Total: b bits per coordinate, unbiased inner products, near-optimal distortion.**

---

## Application to Parameter Golf

### Application A: Weight Compression via PolarQuant (HIGH PRIORITY)

**What current leaders do:** Standard int5/int6 per-group quantization with group-wise scale+zero constants.

**What PolarQuant enables:** Eliminates calibration overhead entirely.

**Concrete savings calculation for our model (d=512):**

Standard int5 with group_size=128:
- 5 bits per weight
- + 32 bits scale + 32 bits zero per group of 128 = 0.5 bits overhead per weight
- Effective: 5.5 bits per weight

PolarQuant int4:
- 4 bits per weight
- + 16 bits (float16 norm) per vector of 512 = 0.03 bits overhead per weight
- Effective: ~4.03 bits per weight

**Savings: ~1.47 bits per weight → 26% reduction in model size → can fit ~26% more parameters in 16MB**

For a 512-dim model: that's roughly 2-3 extra transformer layers.

**Implementation plan:**
1. Precompute Lloyd-Max codebooks for Beta(d/2 - 1/2, d/2 - 1/2) distribution at b=3,4,5 bits for d=512 (our model dimension)
2. Generate a fixed random orthogonal rotation matrix R (via QR decomposition of random Gaussian matrix; seed it for reproducibility)
3. During training: train in float32/16, quantize weights via PolarQuant during save
4. During eval: dequantize = apply codebook lookup + inverse rotation + scale by stored norm
5. Key: the rotation matrix R and codebooks are stored once (~512×512×2 bytes = 512KB for R, or use structured rotation like Hadamard which is O(d log d) and requires 0 storage)

**IMPORTANT — Use Hadamard rotation instead of random matrix:**
- A random d×d rotation matrix costs d²×2 bytes = 512KB for d=512. That eats our 16MB budget.
- Use randomized Hadamard transform (RHT): `R = H_d · D` where `H_d` is the Hadamard matrix (applied via fast Walsh-Hadamard transform in O(d log d)) and `D` is a diagonal matrix of random ±1 entries.
- Storage: only d bits for the diagonal signs = 64 bytes for d=512. Negligible.
- The RHT has the same distributional properties needed (coordinates become approximately Gaussian).

### Application B: QJL Residual Correction for Weights (MEDIUM PRIORITY)

After PolarQuant at b-1 bits, apply QJL to the weight residual. This debiases the weight×activation inner products.

**Challenge:** QJL requires storing the random matrix S (d×d Gaussians), which is too expensive. But we can use a pseudorandom generator with a fixed seed to regenerate S on-the-fly.

**Storage cost:** ~0 (just the seed)
**Compute cost:** Regenerating S for each dequantization. For d=512, that's 512×512 Gaussian draws per layer dequantization. Feasible on H100 but adds latency.

**Implementation decision:** Only worth it if PolarQuant alone at b-1 bits + QJL beats PolarQuant at b bits. Paper says the gap is small at b≥3. **Recommend trying PolarQuant at 4 bits first, then A/B test against 3-bit PolarQuant + QJL.**

### Application C: KV Cache Compression During Eval (MEDIUM PRIORITY)

During evaluation, compress the KV cache to 3 bits using TurboQuant. This frees GPU memory for:
- Longer context windows (train at 1024, eval at 2048+ tokens)
- Larger batch sizes during training

**Implementation:**
1. After computing K and V in attention, apply TurboQuant compression before storing in cache
2. When reading from cache, dequantize on-the-fly
3. Use 3.5-bit TurboQuant for zero quality loss (paper proves 100% needle-in-haystack at 104K tokens)

**Expected impact:** Enables ~4× longer context during eval. If we can eval at 4096 tokens instead of 1024, that's a significant bpb improvement from better context utilization.

---

## Implementation Spec for Codex

### File: `turboquant.py` (new module)

```python
"""
TurboQuant / PolarQuant weight compression for Parameter Golf.
Based on arXiv:2504.19874 (Google Research, ICLR 2026).
"""
import torch
import math

# --- Hadamard Transform (replaces random rotation, 0 storage cost) ---

def hadamard_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """
    Randomized Hadamard Transform: H_d @ diag(signs) @ x
    Uses fast Walsh-Hadamard transform in O(d log d).
    x: (..., d) tensor
    signs: (d,) tensor of +1/-1
    """
    x = x * signs  # Apply random sign flip
    # Fast Walsh-Hadamard Transform
    d = x.shape[-1]
    assert d & (d - 1) == 0, "d must be power of 2"
    h = 1
    while h < d:
        # Butterfly operation
        x_even = x[..., 0::2*h]  # Not quite right for in-place
        # Actually implement properly:
        x = x.view(*x.shape[:-1], -1, 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.view(*x.shape[:-3], -1)
        h *= 2
    return x / math.sqrt(d)  # Normalize


def inverse_hadamard_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Inverse RHT = (1/d) * H_d^T @ diag(signs) but H is symmetric so same as forward."""
    # Hadamard is its own inverse (up to scaling), so:
    x = hadamard_transform(x, torch.ones_like(signs))  # H @ x (without signs)
    return x * signs  # Then multiply by signs


# --- Lloyd-Max Codebook for Beta Distribution ---

def compute_lloyd_max_codebook(d: int, bits: int, num_iterations: int = 100) -> tuple:
    """
    Compute optimal Lloyd-Max quantizer for the marginal distribution of 
    coordinates after rotating a unit-norm vector.
    
    For high d, each coordinate ~ N(0, 1/d). We solve 1D k-means on this distribution.
    
    Returns: (boundaries, centroids) where boundaries has 2^bits - 1 entries
             and centroids has 2^bits entries.
    """
    num_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    
    # Initialize uniformly in [-3σ, 3σ]
    lo, hi = -3 * sigma, 3 * sigma
    centroids = torch.linspace(lo, hi, num_levels)
    
    for _ in range(num_iterations):
        # Boundaries = midpoints of adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        
        # Update centroids = conditional expectation of N(0, σ²) within each bin
        # For Gaussian: E[X | a < X < b] = σ² * (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))
        new_centroids = torch.zeros_like(centroids)
        all_bounds = torch.cat([torch.tensor([-float('inf')]), boundaries, torch.tensor([float('inf')])])
        
        for i in range(num_levels):
            a, b = all_bounds[i].item(), all_bounds[i+1].item()
            # Numerical integration for conditional expectation
            t = torch.linspace(max(a, -5*sigma), min(b, 5*sigma), 1000)
            pdf = torch.exp(-0.5 * (t/sigma)**2) / (sigma * math.sqrt(2 * math.pi))
            prob = pdf.sum() * (t[1] - t[0])
            if prob > 1e-10:
                new_centroids[i] = (t * pdf).sum() * (t[1] - t[0]) / prob
            else:
                new_centroids[i] = (a + b) / 2 if math.isfinite(a) and math.isfinite(b) else centroids[i]
        
        centroids = new_centroids
    
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return boundaries, centroids


# --- PolarQuant Weight Compression ---

class PolarQuantizer:
    def __init__(self, d: int, bits: int, seed: int = 42):
        self.d = d
        self.bits = bits
        self.seed = seed
        
        # Generate random signs for Hadamard (stored as bits, negligible)
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.signs = (torch.randint(0, 2, (d,), generator=rng) * 2 - 1).float()
        
        # Precompute Lloyd-Max codebook
        self.boundaries, self.centroids = compute_lloyd_max_codebook(d, bits)
    
    def quantize(self, weights: torch.Tensor) -> dict:
        """
        Quantize weight matrix using PolarQuant.
        weights: (out_features, in_features) or (*, d)
        Returns dict with quantized codes, norms, and metadata.
        """
        original_shape = weights.shape
        w = weights.view(-1, self.d)  # (N, d)
        
        # Step 1: Store norms
        norms = w.norm(dim=1, keepdim=True)  # (N, 1)
        w_normalized = w / (norms + 1e-8)  # Unit norm
        
        # Step 2: Random Hadamard rotation
        w_rotated = hadamard_transform(w_normalized, self.signs)  # (N, d)
        
        # Step 3: Scalar quantize each coordinate
        codes = torch.bucketize(w_rotated, self.boundaries)  # (N, d) of uint8/int
        
        return {
            'codes': codes.to(torch.uint8) if self.bits <= 8 else codes,
            'norms': norms.half(),  # float16
            'shape': original_shape,
        }
    
    def dequantize(self, quantized: dict) -> torch.Tensor:
        """Reconstruct weights from quantized representation."""
        codes = quantized['codes'].long()
        norms = quantized['norms'].float()
        
        # Step 1: Look up centroids
        w_rotated = self.centroids[codes]  # (N, d)
        
        # Step 2: Inverse Hadamard rotation
        w_normalized = inverse_hadamard_transform(w_rotated, self.signs)
        
        # Step 3: Rescale by norms
        w = w_normalized * norms
        
        return w.view(quantized['shape'])


# --- Size Estimation ---

def estimate_compressed_size(num_params: int, d: int, bits: int) -> int:
    """Estimate compressed model size in bytes."""
    num_vectors = num_params // d
    code_bytes = (num_vectors * d * bits + 7) // 8  # bit-packed codes
    norm_bytes = num_vectors * 2  # float16 norms
    codebook_bytes = (2 ** bits) * 4  # float32 centroids
    sign_bytes = (d + 7) // 8  # Hadamard signs
    return code_bytes + norm_bytes + codebook_bytes + sign_bytes
```

### Integration Points in train_gpt.py:

1. **After training completes:** Apply PolarQuant to all weight matrices before saving
2. **At eval time:** Dequantize weights on-the-fly before forward pass
3. **Bit allocation strategy:** Use 4 bits for attention weights (more sensitive), 3 bits for MLP weights (more compressible) — mixed precision across layers

### Size Budget Analysis (d=512, 12-layer model):

| Component | Params | Standard int5 | PolarQuant int4 | PolarQuant int3 |
|-----------|--------|---------------|-----------------|-----------------|
| Attention (Q,K,V,O) | 512² × 4 × 12 = 12.6M | 7.87 MB | 6.30 MB | 4.72 MB |
| MLP (2 layers) | 512 × 2048 × 2 × 12 = 25.2M | 15.75 MB | 12.60 MB | 9.45 MB |
| Embeddings | 512 × 1024 = 0.5M | 0.31 MB | 0.25 MB | 0.19 MB |
| **Total** | **38.3M** | **23.9 MB** ❌ | **19.2 MB** ❌ | **14.4 MB** ✅ |

**Key finding:** PolarQuant at **3 bits** can fit a 12-layer model (38.3M params) into 14.4 MB. Standard int5 can't even fit it. This is 2 extra layers compared to what int5 allows.

Mixed precision (4-bit attention, 3-bit MLP) would be ~16.0 MB — right at the limit with room for code/embedding overhead.

---

## Recommended Experiment Sequence

1. **Baseline reproduction:** Run current best model, verify bpb
2. **PolarQuant 4-bit:** Replace standard int5/int6 with PolarQuant 4-bit. Measure bpb delta.
3. **PolarQuant 3-bit:** Try 3-bit for MLP weights only. If quality holds, use the freed bytes for more layers.
4. **Add layers:** With 3-bit compression, add 2-3 extra transformer layers. Retrain.
5. **KV cache compression:** Add 3-bit TurboQuant to KV cache during eval for context extension.
6. **A/B test QJL residual:** Compare (3-bit PolarQuant + 1-bit QJL) vs (4-bit PolarQuant) for attention weights.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| QAT (quantization-aware training) beats PolarQuant | Medium | PolarQuant is for compression at save time; combine with QAT during training |
| Hadamard overhead during eval | Low | O(d log d) per vector, negligible vs attention compute |
| 3-bit too aggressive for small models | Medium | Fall back to 4-bit; still better than int5 |
| Rotation matrix seed sensitivity | Low | Paper proves data-oblivious — works for any seed |
| Not enough training time to retrain with more layers | Medium | Start with same architecture, just better compression |

---

*This spec is ready for Codex implementation. Priority: Application A (PolarQuant weight compression).*
