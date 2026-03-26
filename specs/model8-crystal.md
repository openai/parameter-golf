# Model 8: "The Crystal" — Build Spec

**Classification:** PRIVATE — DO NOT SUBMIT UNTIL ENDGAME
**Target bpb:** Unknown (experimental)
**Approach:** Self-organizing architecture where topology encodes knowledge
**Nature Analog:** Crystal growth — simple local rules create complex global structure

---

## Core Concept

A snowflake is infinitely complex but generated from a single rule applied repeatedly at the molecular level. The crystal's STRUCTURE is the information — not data stored in a container, but the container's shape IS the data.

We build a model where the architecture itself encodes language patterns. Instead of fixed layers with trained weights, we have a single computational "seed" (small rule set) that grows into a full model through recursive self-application.

## Architecture

### The Seed (2 MB)
- One small transformer block: 256 dim, 4 heads, 2x MLP
- Plus a "growth rule" network: MLP (256 → 128 → 256) that determines how the seed modifies itself
- This is ALL that's stored in the 16MB artifact (plus embeddings)

### The Growth Process (at inference time)
1. Start with the seed block
2. Apply growth rules to generate layer-specific modifications:
   - Growth rule takes (layer_index, position_encoding) as input
   - Outputs: attention scaling factors, MLP bias shifts, skip connection weights
3. Effectively creates a unique layer at each depth from the shared seed
4. 12-16 effective layers, each slightly different, all derived from one seed

### Cellular Automata Attention
- Instead of full O(n²) attention, use local CA-style attention
- Each token attends to its k nearest neighbors (k=32-64)
- But CA rules propagate information: after L layers, effective receptive field = k^L
- With k=32 and L=12: effective receptive field covers entire sequence
- Much cheaper per layer → more layers in same compute budget

### Fractal Position Encoding
- Standard RoPE encodes position linearly
- Fractal PE encodes at multiple scales simultaneously
- Like how a river network has structure at every zoom level
- Implemented as: PE = sum(RoPE(pos, freq_i) for freq_i in geometric_series)

## Training Strategy

1. Train the seed block normally (like training one transformer layer)
2. Train the growth rules to make derived layers useful
3. End-to-end: loss backprops through the growth process
4. The seed learns to be a "universal computational primitive"
5. Growth rules learn to specialize each layer appropriately

## Parameter Budget

| Component | Size |
|-----------|------|
| Seed block (256 dim, int6) | ~1.5 MB |
| Growth rule network | ~0.5 MB |
| Embeddings (tied, fp16) | ~1 MB |
| CA attention parameters | ~0.5 MB |
| Fractal PE parameters | ~0.1 MB |
| Quantization overhead | ~12.4 MB of headroom |
| **Total** | **~16 MB** |

NOTE: This model has MASSIVE headroom. Options:
- Increase seed to 512 dim
- Add multiple seed variants
- Increase growth rule complexity
- Use the space for a n-gram cache (like Model 1)

## Why This Could Work

- Extreme parameter efficiency: one block generates 12-16 effective layers
- Growth rules learn layer specialization with minimal parameters
- CA attention is O(n·k) instead of O(n²) — much faster per layer
- More layers in same compute budget = better modeling
- Completely novel — nobody in the competition is doing anything like this

## Key Risks
- Growth rules may not learn useful layer differentiation
- CA attention may miss long-range dependencies
- Very unusual architecture — hard to debug
- May converge slowly because the seed must be a good universal primitive
- The "architecture IS knowledge" idea is philosophically interesting but unproven

## Output
- `train_gpt_model8.py`
