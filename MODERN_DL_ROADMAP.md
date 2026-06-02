# Modern Deep Learning Learning Roadmap (2024 Edition)

This document is structured for compatibility with **NotebookLLM**, **Slide-Gen**, and **Video-Generation** tools. It bridges the gap between classic "College DL" (2018-2020) and modern SOTA (2024).

---

## 📺 Video Series Curriculum (5 Videos)

### Video 1: The Modern LLM Skeleton
**Focus**: What replaced the components in your old textbook.
*   **Rotary Position Embeddings (RoPE)**: Moving from absolute/sinusoidal positions to relative rotations in complex space.
*   **RMSNorm (No-Weight)**: Why we stopped using BatchNorm/LayerNorm and why we often skip the learnable "gamma" parameter for stability.
*   **GQA & MQA**: Grouped Query Attention and Multi-Query Attention—how we save 80% KV-cache memory without losing quality.
*   **Tied Embeddings**: Sharing parameters between input tokens and output logits to save millions of parameters.

### Video 2: The Optimizer Revolution (Beyond Adam)
**Focus**: The math that makes small models punch like heavyweights.
*   **Newton-Schulz Orthogonalization (Muon)**: Using spectral normalization to ensure every update is maximally "perpendicular" to the last.
*   **Muon vs. Adam**: Why second-order-like behavior is essential for modern high-learning-rate training.
*   **Soap & Prodigy**: Experimental optimizers that auto-tune themselves.

### Video 3: Loss Engineering & Training Dynamics
**Focus**: How to focus your compute only on what matters.
*   **Focal Loss**: How `(1-pt)^gamma` forces the model to ignore "common" words and focus on high-entropy "knowledge" tokens.
*   **Learning Rate Warmdown**: Why "linear warmdown to zero" is the current SOTA schedule for short training runs.
*   **EMA & SWA**: Exponential Moving Average and Stochastic Weight Averaging for smooth convergence.

### Video 4: The Art of Parameter Golf (Quantization)
**Focus**: How to squeeze a giant brain into a tiny box (16MB).
*   **Int8 Per-Row Quantization**: Scaling weights row-by-row to maintain precision while using 1/4 the memory.
*   **Compression Pipelines**: Using zlib and lzma on weights—why "structured weights" are more compressible.
*   **BPB (Bits Per Byte)**: The ultimate metric for compression—agnostic to model size or tokenizer.

### Video 5: M4 & H100 Hardware Optimization
**Focus**: Practical deployment and unified memory.
*   **Unified Memory (MLX)**: Why Apple Silicon is a secret weapon for training (no PCIe transfer bottleneck).
*   **Graph Compilation**: Eager vs. Lazy execution in MLX/JAX.
*   **The 10-Minute Sprint**: Strategies for maximizing learning given a strict wallclock deadline.

---

## 🛠 Self-Learning Roadmap (Weekly Goals)

### Week 1: Architecture Deep Dive
- **Source**: Read the "RoFormer" paper (RoPE) and Llama 3 technical reports.
- **Hands-on**: Modify a standard Attention block to use `mlx.nn.RoPE`.

### Week 2: Optimizer Mastery
- **Source**: Research "Muon" and the "Newton-Schulz" iteration for matrix orthogonalization.
- **Hands-on**: Switch an AdamW optimizer for Muon in a small MNIST-level project.

### Week 3: Efficiency & Quantization
- **Source**: Study the "BitNet" or "llama.cpp" quantization docs.
- **Hands-on**: Take your `E03` model and try to manually compress it under 5MB using `zlib`.

---

## 🤖 Automation Prompts (For NotebookLLM / Claude)

**Prompt for Video Scripts**:
> "Using the 'Modern DL Roadmap' provided, act as a technical scriptwriter. Generate a 5-minute video script for Video 1 (The Modern LLM Skeleton). Ensure you explain RoPE using an analogy that a 2018-era DL student would understand."

**Prompt for Slide Generation**:
> "Extract the key mathematical differences between RMSNorm and LayerNorm from the roadmap and generate 3 bullet points per slide for a technical presentation."
