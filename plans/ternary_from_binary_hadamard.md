# Mathematical Foundation: Unifying Binary Hadamard and Ternary HDC Representations

## Overview

This document explains how binary Hadamard vectors can be extended to provide ternary semantics {-1, 0, +1}, enabling instant learning and storage of relationships from token projections.

---

## Part 1: The Binary Hadamard Foundation

### Sylvester Hadamard Matrix

The Sylvester construction generates orthogonal matrices with entries in {-1, +1}:

```
H₁ = [1]
H₂ = [ 1  1]
     [ 1 -1]
H₄ = [ 1  1  1  1]
     [ 1 -1  1 -1]
     [ 1  1 -1 -1]
     [ 1 -1 -1  1]
```

**Key Property**: `H[i,j] = (-1)^(popcount(i AND j))`

### Packed Binary Representation

In the codebase, we pack this into uint64 arrays:
- Bit = 1 → represents +1 in the Hadamard matrix
- Bit = 0 → represents -1 in the Hadamard matrix

```python
def hadamard_row_packed(index: int, dim: int) -> np.ndarray:
    for bit_idx in range(64):
        parity = bin(index & bit_idx).count('1') % 2
        if parity == 0:  # +1 in Hadamard
            block_val |= (1 << bit_idx)  # Set bit to 1
        # else: -1 in Hadamard, bit stays 0
```

---

## Part 2: XOR as Multiplication in Sign Space

### The Fundamental Isomorphism

XOR in {0, 1} space is **isomorphic** to multiplication in {+1, -1} space:

| XOR (binary) | Multiplication (sign) |
|--------------|----------------------|
| 0 ⊕ 0 = 0    | (+1) × (+1) = +1     |
| 0 ⊕ 1 = 1    | (+1) × (-1) = -1     |
| 1 ⊕ 0 = 1    | (-1) × (+1) = -1     |
| 1 ⊕ 1 = 0    | (-1) × (-1) = +1     |

**Mapping**: `bit_0 → +1`, `bit_1 → -1`

### Why This Matters for HDC

In HDC, **binding** is multiplication in sign space:
```
bound = token_vec × position_vec  (element-wise)
```

With binary XOR:
```
bound_bits = token_bits ⊕ position_bits
```

These are **mathematically equivalent**! XOR preserves the algebraic structure.

---

## Part 3: The Missing Ternary State

### Current Ternary Representation

The codebase uses two vectors for ternary:

```python
def seed_to_ternary_hypervector(seed_string: str, dim: int):
    pos_vec = seed_to_hypervector(f"{seed_string}:pos", dim)
    neg_vec = seed_to_hypervector(f"{seed_string}:neg", dim)
    
    # Remove overlap to ensure orthogonality
    overlap = np.bitwise_and(pos_vec, neg_vec)
    pos_vec = np.bitwise_xor(pos_vec, overlap)
    neg_vec = np.bitwise_xor(neg_vec, overlap)
    
    return pos_vec, neg_vec
```

| State | pos_vec | neg_vec | Meaning |
|-------|---------|---------|---------|
| +1    | 1       | 0       | Excited |
| -1    | 0       | 1       | Inhibited |
| 0     | 0       | 0       | Neutral/Unknown |

### The Problem

This requires **2× memory** and **2× computation** for every operation.

---

## Part 4: Deriving Ternary from Binary via Bundling

### The Key Insight: Bundling Creates Confidence

When you XOR-bind and bundle multiple vectors:

```
bundled = v₁ ⊕ v₂ ⊕ v₃ ⊕ ... ⊕ vₙ
```

In sign space, this is:
```
bundled = v₁ × v₂ × v₃ × ... × vₙ
```

### Popcount as Confidence Measure

For a single uint64 element after bundling:
- **popcount = 32** → exactly half 1s, half 0s → **NEUTRAL (0)**
- **popcount = 64** → all 1s → **strong +1**
- **popcount = 0** → all 0s → **strong -1**
- **popcount = 48** → 75% 1s → **moderate +1**

### Confidence Formula

```python
def binary_to_ternary_confidence(packed_uint64):
    """Convert binary packed vector to ternary with confidence."""
    pc = popcount(packed_uint64)
    
    # Distance from neutral (32 out of 64)
    confidence = abs(pc - 32) / 32.0  # 0.0 to 1.0
    
    # Sign: +1 if more 1s, -1 if more 0s
    sign = +1 if pc > 32 else -1 if pc < 32 else 0
    
    return sign, confidence
```

| popcount | sign | confidence | interpretation |
|----------|------|------------|----------------|
| 0        | -1   | 1.0        | Strong -1 |
| 16       | -1   | 0.5        | Moderate -1 |
| 32       | 0    | 0.0        | Neutral/Unknown |
| 48       | +1   | 0.5        | Moderate +1 |
| 64       | +1   | 1.0        | Strong +1 |

---

## Part 5: Application to Instant Projection

### Current Instant Projection Flow

```
1. Project: dataset_vec = ⊕(token_vec[pos] ⊕ pos_vec[pos]) for all positions
2. Verify: unbound = dataset_vec[window] ⊕ pos_vec[window]
3. Compare: unbound == expected_token_vec?
```

### Enhanced Flow with Ternary Semantics

```
1. Project: dataset_vec = ⊕(token_vec[pos] ⊕ pos_vec[pos]) for all positions
2. Unbind: unbound = dataset_vec[window] ⊕ pos_vec[window]
3. Compute confidence: 
   - popcount each uint64 in unbound
   - confidence = |popcount - 32| / 32
4. If confidence > threshold:
   - Strong signal → known relationship
   - Use for instant learning
5. If confidence < threshold:
   - Weak signal → neutral/unknown
   - Mark for additional learning
```

### GPU Kernel Enhancement

```cpp
// In sparse_verify_and_correct kernel
unsigned long long unbound = dataset_val ^ pos_val;

// Compute confidence (popcount distance from 32)
int pc = __popcll(unbound);
float confidence = abs(pc - 32) / 32.0f;

// Store confidence for learning decisions
if (confidence > 0.8f) {
    // High confidence - instant learn
    atomicAdd(&high_confidence_count, 1);
} else if (confidence < 0.2f) {
    // Low confidence - needs more data
    atomicAdd(&neutral_count, 1);
}
```

---

## Part 6: Benefits of Unified Representation

### Memory Efficiency
- **Before**: 2 vectors × uint64_count = 2 × 16384 = 32768 uint64s per hypervector
- **After**: 1 vector × uint64_count = 16384 uint64s per hypervector
- **Savings**: 50% memory reduction

### Computational Efficiency
- **Before**: 2 XOR operations per binding
- **After**: 1 XOR operation per binding
- **Savings**: 50% compute reduction

### Semantic Richness
- Still captures {-1, 0, +1} semantics via confidence
- 0 (neutral) emerges naturally from bundled signals
- No need for explicit two-vector encoding

---

## Part 7: Mathematical Proof of Equivalence

### Theorem: Binary Bundling Preserves Ternary Semantics

**Given**: A set of binary vectors V = {v₁, v₂, ..., vₙ} where each vᵢ ∈ {0,1}^d

**Define**: The bundled vector B = v₁ ⊕ v₂ ⊕ ... ⊕ vₙ

**Claim**: For each bit position j:
- If most vᵢ[j] = 0, then B[j] encodes -1
- If most vᵢ[j] = 1, then B[j] encodes +1
- If equal 0s and 1s, B[j] encodes 0 (neutral)

**Proof**:
1. XOR is associative and commutative
2. XOR of n bits = (count of 1s) mod 2
3. If count(1s) > n/2: result biased toward 1 → +1
4. If count(1s) < n/2: result biased toward 0 → -1
5. If count(1s) = n/2: result random → 0 (neutral)

**Confidence**: |count(1s) - n/2| / (n/2) measures signal strength

---

## Part 8: Implementation Roadmap

### Phase 1: Add Confidence Computation
- Modify `sparse_verify_and_correct` kernel to compute popcount-based confidence
- Store confidence alongside predictions

### Phase 2: Use Confidence for Learning
- High confidence positions: instant learn (store recipe)
- Low confidence positions: defer or require more data

### Phase 3: Unified API
- Create `BinaryTernaryVector` class that wraps binary vectors with confidence
- Provide ternary-like API over binary representation

### Phase 4: Deprecate Two-Vector Ternary
- Migrate all ternary operations to confidence-based binary
- Remove `seed_to_ternary_hypervector` once migration complete

---

## Conclusion

Binary Hadamard vectors can provide ternary semantics through confidence measurement. The key insight is that **bundling naturally creates a spectrum from -1 to +1**, with the neutral point (0) emerging at the balance point. This enables:

1. **50% memory savings**
2. **50% compute savings**
3. **Preserved ternary semantics**
4. **Instant learning from projections**

The mathematical foundation is solid: XOR in binary space is isomorphic to multiplication in sign space, and popcount provides the confidence measure needed to derive the ternary neutral state.
