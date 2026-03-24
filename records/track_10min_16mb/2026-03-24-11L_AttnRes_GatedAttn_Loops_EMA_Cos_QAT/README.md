# AttnRes + Gated Attention + Looped Blocks

### Architecture changes
- **11 layers, 3x MLP** — increased capacity matching top submissions
- **Block Attention Residuals** (Moonshot AI, arXiv 2603.15031) — replaces fixed `skip_weights` with softmax attention over all encoder outputs using per-decoder-layer pseudo-queries
- **Per-head gated attention** — learnable `sigmoid(gate)` per attention head, prevents attention-sink pathology
- **Looped middle blocks** — layers 4-7 run twice per forward pass, adding compute depth without parameters

### Training changes
- **EMA** (decay=0.995) — exponential moving average weights for final eval/export
- **Cosine LR decay** — replaces linear warmdown
- **QAT** (last 15%) — simulates int8 per-row quantization to reduce roundtrip degradation

## Local validation (M4 Max, 500 steps)
- val_bpb: 1.475 (float), int8 roundtrip: 1.648
- Artifact size: 13.7MB (2.3MB under 16MB cap)
- Full H100 run pending

## Config
- 26.5M params, 11 layers, 512 dim, 8 heads, 4 KV heads, 3x MLP, tied embeddings
- Estimated artifact: ~13.7MB after int8+zlib
