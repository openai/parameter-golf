# Model 3: "The Hybrid" — Build Spec

**Classification:** PRIVATE — DO NOT SUBMIT UNTIL ENDGAME
**Target bpb:** Unknown (experimental)
**Approach:** SSM/transformer hybrid with hash-routed MoE

---

## Architecture

### Lower Layers: SSM/Recurrent (Layers 1-8)
- 8 layers of state space model (Mamba-style or simplified xLSTM)
- Linear time complexity O(n) vs O(n²) for attention
- 3.5x faster training per step than attention layers
- More steps in 10 min = better convergence
- Each layer: selective state space with gating
- Handles local and mid-range patterns

### Upper Layers: Attention (Layers 9-11)
- 3 standard transformer layers with full attention
- GQA (8 heads, 4 KV heads), 512 dim
- Handles global reasoning and long-range dependencies
- Only 3 layers = small parameter cost

### Hash-Routed MoE on SSM Layers
- Each SSM layer has 4 expert variants
- Hash function routes each token to exactly 1 expert
- No learned router = no routing overhead
- 4x capacity in same compute budget
- Hash ensures load balancing automatically

### ChebyshevKAN (Optional)
- Replace standard MLP in attention layers with Chebyshev KAN
- Learnable activation functions (Chebyshev polynomials, k=3)
- 5-15% quality gain with near-zero overhead
- Only in the 3 attention layers (small cost)

## Why This Could Work

The hypothesis: SSM layers are better than attention for local patterns, and attention is better for global patterns. By using SSM for 8/11 layers, we get:
- Faster training (3.5x per SSM layer)
- More total training steps in 10 min
- Attention only where it's needed (top 3 layers)

Nobody in the competition has tried this. If SSMs match attention quality at small scale (which recent papers suggest), this is a free speedup.

## Build Instructions for Codex

### Phase 1: Implement Basic SSM Layer
- Simplified Mamba/S4 block (selective state space)
- Test: can a stack of 8 SSM layers learn language at all?
- **Verify:** loss decreases on FineWeb

### Phase 2: Hybrid Stack
- 8 SSM layers + 3 attention layers
- Standard training pipeline
- **Verify:** hybrid matches or beats pure-attention baseline

### Phase 3: Hash-Routed MoE
- Add 4 expert variants per SSM layer
- Implement hash-based routing (murmurhash → expert_id % 4)
- **Verify:** MoE improves over single-expert baseline

### Phase 4: KAN (Optional)
- Replace MLP in attention layers with ChebyshevKAN
- **Verify:** quality improvement with acceptable speed cost

### Phase 5: Quantization + Compression
- Standard int6 + zstd pipeline
- **Verify:** fits in 16MB

## Key Risks
- SSM implementation from scratch is complex (Mamba has custom CUDA kernels)
- May need to use simpler S4 variant without hardware-specific optimization
- Hash routing may cause quality issues with imbalanced expert specialization
- ChebyshevKAN may not compose well with SSM layers

## Fallback
If SSM doesn't work at all → fall back to attention (becomes Model 4 variant)
If MoE hurts quality → drop it, use single experts
If KAN is too slow → drop it, use standard MLP

## Output
- `train_gpt_model3.py` — SSM/transformer hybrid
