 Efficient Recurrent GPT — Compact Language Model

🚀 Overview

This submission presents a compact, parameter-efficient language model designed to minimize Bits-Per-Byte (BPB) under strict constraints on model size (<16MB) and training time.

The model adopts a recurrent transformer-style architecture that reuses parameters across multiple refinement steps, enabling deeper computation without increasing parameter count.

---

🧠 Architecture

Key Components

- Recurrent Transformer Core
  Multiple refinement steps ("n_recur") with shared weights for efficient depth

- Low-Rank Attention (Q/K/V)
  Reduces parameter count while maintaining performance

- Full-Rank Output Projection
  Preserves expressivity of attention outputs

- Rotary Positional Embeddings (RoPE)
  Efficient positional encoding without additional parameters

- SwiGLU Feed-Forward Network
  
  - Low-rank up-projections
  - Full-rank down-projection
  - Improved expressivity under tight constraints

- RMSNorm + Residual Gating
  Stabilizes training and prevents exploding activations

---

⚙️ Training Strategy

- Dataset: FineWeb tokenized dataset ("fineweb10B_sp1024")
- Vocabulary Size: 1024 (SentencePiece tokenizer)
- Context Length: Curriculum-based (64 → 128 → 256)
- Optimizer: AdamW (β = 0.9, 0.95)
- Learning Rate: Cosine decay with warmup
- Mixed Precision Training: Enabled (AMP)

Optimizations

- Token-frequency weighted cross-entropy (better compression)
- Gradient clipping for stability
- Adaptive Exponential Moving Average (EMA)
- Dropout regularization in feed-forward layers

---

📦 Model Size

- Final checkpoint: < 16MB (float16)
- Fully compliant with competition constraints

---

📊 Result

- Achieved BPB: <your_result_here>

---

✅ Compliance

This submission:

- Uses only the official dataset and tokenizer
- Does not rely on external data
- Avoids compression tricks or post-processing hacks
- Fully adheres to all competition rules

---

📁 Structure

records/subash_v15/
    train.py
    best.pt
    README.md

---

📝 Notes

The focus of this model is to balance efficiency, stability, and compression performance within strict resource constraints. All design choices prioritize reproducibility and compliance while pushing BPB performance as low as possible.
