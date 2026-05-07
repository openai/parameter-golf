# 🏌️ TURBOQUANT INTEGRATION SPEC — Parameter Golf

**Date:** 2026-03-25 | **Source:** Google Research, arXiv:2504.19874 (ICLR 2026)

---

## 📌 WHAT IS THIS

Google released TurboQuant today — a compression algorithm that quantizes vectors to 3 bits with zero accuracy loss. The paper targets KV cache compression, but the core technique (PolarQuant) applies directly to **weight compression**, which is our primary bottleneck in the 16MB artifact limit.

- **Paper:** https://arxiv.org/abs/2504.19874
- **Google Blog:** https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

---

## 📌 WHY IT MATTERS FOR US

Current leaderboard leaders use int5/int6 quantization with per-group calibration constants (scale + zero-point). Those constants eat 0.5-1.0 bits of overhead per weight. PolarQuant **eliminates this overhead entirely** by rotating weights into a known distribution first.

**The math:**
- Standard int5 with group_size=128: 5 bits/weight + 0.5 bits overhead = **5.5 bits effective**
- PolarQuant int4: 4 bits/weight + 0.03 bits overhead = **4.03 bits effective**
- **Savings: 1.47 bits/weight → 26% more parameters in the same 16MB**

That's 2-3 extra transformer layers we can't fit otherwise.

---

## 📌 THREE TECHNIQUES FROM THE PAPER

### 1. PolarQuant (MSE-Optimal) — HIGH PRIORITY
- Normalize weight vectors to unit norm, store norms separately (float16)
- Apply randomized Hadamard transform (makes coordinate distribution predictable — converges to N(0, 1/d))
- Apply precomputed Lloyd-Max optimal scalar quantizer per coordinate
- No calibration constants needed — the rotation makes the distribution known a priori
- Storage: b-bit codes + float16 norms + 64 bytes Hadamard signs. That's it.
- Distortion at 3 bits: 0.03 MSE (within 2.7× of theoretical lower bound)
- Distortion at 4 bits: 0.009 MSE (essentially lossless)

### 2. QJL (1-bit Residual Correction) — MEDIUM PRIORITY
- After PolarQuant, compute residual error
- Apply sign(S · residual) where S is a random Gaussian matrix (regenerated from seed)
- Dequantize: (√(π/2) / d) · Sᵀ · z
- Gives **unbiased** inner product estimation (PolarQuant alone is biased)
- Costs 1 extra bit per coordinate
- Test: does (3-bit PolarQuant + 1-bit QJL) beat (4-bit PolarQuant)?

### 3. KV Cache Compression During Eval — MEDIUM PRIORITY
- Apply TurboQuant to attention KV cache at 3 bits
- Frees GPU memory → enables longer context during eval
- Paper proves 100% needle-in-haystack accuracy at 104K tokens with 3.5 bits
- Could enable eval at 4096 tokens instead of 1024

---

## 📌 SIZE BUDGET ANALYSIS (d=512, 12-layer model)

### Standard int5:
- Attention (Q,K,V,O): 12.6M params → 7.87 MB
- MLP (2 layers per block): 25.2M params → 15.75 MB
- Embeddings: 0.5M → 0.31 MB
- **Total: 23.9 MB ❌ (over budget)**

### PolarQuant int4:
- Same params → **19.2 MB ❌ (still over, but closer)**

### PolarQuant int3:
- Same 38.3M params → **14.4 MB ✅ (fits with 1.6 MB headroom)**

### Mixed precision (4-bit attention, 3-bit MLP):
- ~16.0 MB — right at limit with room for code/embeddings

**KEY INSIGHT:** PolarQuant at 3 bits fits a 12-layer model that int5 can't. Those extra layers are historically the strongest lever for bpb improvement.

---

## 📌 IMPLEMENTATION — KEY DECISIONS

### Rotation matrix: Use Hadamard, NOT random matrix
- Random d×d matrix = 512KB for d=512. Kills our budget.
- Randomized Hadamard Transform = 64 bytes (just ±1 signs for diagonal)
- Same distributional properties. O(d log d) compute. Negligible overhead.

### Codebook: Precompute once, embed in code
- Lloyd-Max quantizer for N(0, 1/d) distribution
- 2^b centroids per bit-width (8 for 3-bit, 16 for 4-bit)
- Stored as float32 constants in the training script. ~64 bytes.

### Integration with QAT (quantization-aware training):
- PolarQuant is post-training compression (apply after train, before save)
- Can ALSO do QAT during training: forward pass uses quantized weights, backward pass uses STE
- Combining both = best of both worlds

---

## 📌 EXPERIMENT SEQUENCE (Priority Order)

1. **Baseline:** Verify current best model bpb
2. **PolarQuant 4-bit:** Replace our int5/int6 with PolarQuant 4-bit. Measure bpb delta.
3. **PolarQuant 3-bit MLP:** Try 3-bit for MLP weights only. Measure degradation.
4. **Add layers:** With 3-bit compression freeing bytes, add 2-3 extra transformer layers. Retrain.
5. **KV cache compression:** Add 3-bit TurboQuant to KV cache during eval for context extension.
6. **QJL A/B test:** Compare (3-bit + 1-bit QJL) vs (4-bit PolarQuant) for attention.

---

## 📌 RISKS

- **QAT might still beat post-training PolarQuant** → Combine both
- **3-bit too aggressive for small models** → Fallback: 4-bit still beats int5
- **Hadamard overhead during eval** → Negligible: O(d log d) vs O(d²) attention
- **Not enough training time for bigger model** → Same arch first, just better compression

---

## 📌 REFERENCE CODE

Full pseudocode with `PolarQuantizer` class, fast Walsh-Hadamard transform, and Lloyd-Max codebook computation is in the detailed spec: `TURBOQUANT-SPEC.md`

### Core Algorithm (simplified):

```python
import torch, math

def hadamard_transform(x, signs):
    """Randomized Hadamard Transform — O(d log d), 64 bytes storage."""
    x = x * signs  # Random sign flip
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.view(*x.shape[:-1], -1, 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.view(*x.shape[:-3], -1)
        h *= 2
    return x / math.sqrt(d)

class PolarQuantizer:
    def __init__(self, d, bits, seed=42):
        self.d = d
        self.bits = bits
        rng = torch.Generator().manual_seed(seed)
        self.signs = (torch.randint(0, 2, (d,), generator=rng) * 2 - 1).float()
        self.boundaries, self.centroids = lloyd_max_codebook(d, bits)
    
    def quantize(self, weights):
        w = weights.view(-1, self.d)
        norms = w.norm(dim=1, keepdim=True)           # Store norms (float16)
        w_norm = w / (norms + 1e-8)                     # Unit norm
        w_rot = hadamard_transform(w_norm, self.signs)  # Rotate
        codes = torch.bucketize(w_rot, self.boundaries) # Quantize
        return codes.to(torch.uint8), norms.half()
    
    def dequantize(self, codes, norms, shape):
        w_rot = self.centroids[codes.long()]
        w_norm = hadamard_transform(w_rot, self.signs)  # Inverse (Hadamard is self-inverse)
        return (w_norm * norms.float()).view(shape)
```

---

## 📌 BOTTOM LINE

This paper gives us a mathematically proven way to compress **26% more model into the same 16MB budget**, with near-zero overhead and provable distortion bounds. Nobody on the current leaderboard is using it yet — paper just dropped today. First-mover advantage if we implement fast.
