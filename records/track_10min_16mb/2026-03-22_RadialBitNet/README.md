# Radial-BitNet 16MB Titan

This submission challenges the limits of parameter compression by employing full **BitNet 1.58b** quantization and **Radial Encoding**. 

While the standard baseline manages ~19 Million parameters in 16MB using INT8, the intrinsic ternary entropy of BitNet weights ($\{-1, 0, 1\} \approx 1.58$ bits) combined with aggressive Zstandard compression allows us to scale a model of **~28.5 Million Parameters** into the exact same 16MB boundary.

### Key Architectural Hacks
1. **BitLinear Expansion:** All projections ($Q, K, V, O$ and the $3\times$ MLP expansion) strictly use BitNet ternary weights, scaling up the parameter count while minimizing storage footprint.
2. **Radial Encoding:** We completely discard the traditional Positional Embedding table to save parameters. Instead, absolute geometrical position is analytically injected into the token embeddings via `RadialEncoding(8)`. 
3. **FRO Optimizer (Fractal Resonant Optimization):** A custom directional optimizer replacing AdamW, which calculates gradient/momentum alignment across multi-scale fractal steps for extreme early convergence within the 10-minute compute limit.

### Configuration
*   **Layers:** 16
*   **Model Dim:** 512
*   **Heads:** 8 (with 4 KV Heads)
*   **Target Size:** 28.5M Parameters (~6.5MB Compressed `.zst`)

### Reproducibility
The `train_gpt.py` script automatically verifies the parameter limits post-training using an exact size audit loop. It mimics the OpenAI validation BPB protocol explicitly.
