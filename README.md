![Header](https://i.ibb.co/F4QdJpDM/16v.png)

# Parameter Golf Submission: Semantic Atomism & Logic Distillation

## Strategy Overview
This submission leverages **Semantic Atomism & Logic Distillation** to push the boundaries of model efficiency within the 16MB constraint.

### Semantic Atomism
Language is decomposed into minimal, reusable semantic units ("atoms"), reducing redundancy and improving compression.

### Logic Distillation
The model prioritizes learning logical structures (syntax, inference rules) over rote memorization, enabling better generalization with fewer parameters.

---

## Key Features
- **Custom Tokenizer:** Optimized to align with semantic atoms.
- **Lightweight Architecture:** Attention mechanisms designed for logic distillation.
- **Efficient Training:** Focuses on high-value patterns in the FineWeb10B dataset.

## Results
- **Target:** `val_bpb` ~1.2
- **Artifact Size:** <16MB (code + compressed model)

## How to Run
```bash
RUN_ID=semantic_atomism \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
