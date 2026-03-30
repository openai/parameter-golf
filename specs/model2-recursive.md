# Model 2: "The Recursive" — Build Spec

**Classification:** PRIVATE — DO NOT SUBMIT UNTIL ENDGAME
**Target bpb:** 1.08-1.10 on 8×H100
**Approach:** Adaptive depth recurrence with frequency codebook fast-path

---

## Architecture

### Frequency Codebook Fast-Path
- Top 2048 most common byte sequences in FineWeb → dedicated learned embeddings
- Hash-based detection — O(1) per token
- Matching tokens bypass the transformer entirely
- Covers ~30-40% of typical web text
- Size: 2048 × 64 dims × 2 bytes (fp16) = ~256KB

### Shared Transformer Block
- ONE transformer block with full architecture:
  - 512 dim, 8 heads (4 KV heads, GQA)
  - 3x MLP (1536 hidden), ReLU²
  - Layer norm, residual connections
- Applied 1-12 times per token (adaptive recursion)
- Each pass refines the representation
- ~2.6M parameters × 1 block

### Adaptive Depth Router
- Small MLP (512 → 64 → 1) that predicts optimal recursion depth per token
- Input: current token embedding after first pass
- Output: integer 1-12 (how many more passes needed)
- Trained end-to-end with the main model
- Easy tokens (common words, predictable syntax) → 1-2 passes
- Hard tokens (rare words, complex semantics) → 8-12 passes
- ~33K parameters

### Post-Processing
- EMA weight averaging (decay=0.997)
- GPTQ-lite quantization (5 clip percentiles)
- Int6 per-row quantization
- zstd level 22 compression
- Sliding window eval (stride=64)

## Parameter Budget

| Component | Raw Size | Compressed (int6+zstd) |
|-----------|----------|----------------------|
| Codebook | 256 KB | ~100 KB |
| Shared transformer block | ~10.4 MB | ~2.6 MB |
| Depth router | ~130 KB | ~50 KB |
| Embeddings (tied, fp16) | ~1 MB | ~1 MB |
| BigramHash + SmearGate | ~2 MB | ~500 KB |
| **Total** | | **~4.25 MB** |

Wait — that's WAY under 16MB. We have massive headroom. Options:
- Increase to 2-3 shared blocks with different specializations
- Increase model dim to 768
- Add more codebook entries
- Use more blocks with different widths

## Build Instructions for Codex

### Phase 1: Basic Shared Block
- Single transformer block, applied N times (fixed N=6 to start)
- Standard training, no router yet
- **Verify:** loss decreases, model trains stably with weight sharing

### Phase 2: Adaptive Router
- Add depth router MLP
- Train with Gumbel-softmax for discrete depth selection
- **Verify:** router learns to assign different depths to different tokens

### Phase 3: Frequency Codebook
- Analyze FineWeb for most common byte sequences
- Build codebook with learned embeddings
- Route matching tokens through codebook, rest through transformer
- **Verify:** codebook handles ~30% of tokens, bpb improves

### Phase 4: Optimization
- Add EMA, GPTQ-lite, int6, sliding eval
- Tune recursion depth distribution
- **Verify:** compressed model fits in 16MB, bpb competitive

## Output
- `train_gpt_model2.py` — adaptive recursive model
