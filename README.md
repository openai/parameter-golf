Parameter Golf Challenge Submission

> Overview
This project focuses on building a highly parameter-efficient language model under strict constraints. The goal was to maximize learning performance while minimizing total parameter count.

---

> Approach

1. Vocabulary Reduction (Primary Optimization Lever)
The most impactful optimization was reducing vocabulary size.

- Started with ~50,000 tokens (GPT-style)
- Gradually reduced to:
  - 5000 → 2000 → 1000 → 500 → 200

Insight:
Smaller vocabulary significantly improved learning efficiency for a small model by increasing token frequency and reducing output space complexity.

---

2. Model Architecture

Final architecture:
- Embedding size: 16
- Hidden layer: 64
- Activation: ReLU

Structure:
Embedding → Linear → ReLU → Linear

This introduced non-linearity while keeping parameter count low.

---

3. Parameter Efficiency

Key design principle:
> Maximize performance per parameter, not absolute performance

- Avoided unnecessary depth
- Focused on compact representations
- Achieved ~21K parameters total

---

4. Training Strategy

- Learning rate: 0.005
- Epochs: 30
- Dataset: subset of Wikitext

Insight:
Higher learning rate worked better due to limited training time and small model size.

---

5. Iterative Optimization Process

Systematically explored:
- Embedding sizes (32 → 16 → 8)
- Vocabulary sizes (50K → 200)
- Hidden layer scaling (32 → 64)
- Training duration (20 → 30 → 40 epochs)

Rejected approaches:
- Excessively small embeddings (loss of capacity)
- Weight tying (reduced performance in this regime)
- Over-training (diminishing returns)

---

> Final Result

- Parameters: ~21K
- Efficient training time
- Strong loss reduction through structural and representation optimizations

---

> Key Insight

> For small models, reducing vocabulary size improves both efficiency and learning quality by concentrating probability mass and increasing token frequency.

---

> Reproducibility

Run: python final_model.py


*Tokenizer files are included and required for execution.
