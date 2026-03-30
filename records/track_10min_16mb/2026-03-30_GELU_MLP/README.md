# GELU MLP Activation

## Summary
Baseline submission replacing ReLU² activation with GELU in the MLP layer.

## Key Changes
- **Activation Function**: Changed from `ReLU(x)² ` to `GELU(x)` in the MLP forward pass
- **Rationale**: GELU is a smooth activation function often used in transformer models. It's been shown to work well in language models and may provide better gradient flow compared to ReLU²
- **Impact**: Likely minimal change in model performance, used as a test for the submission process

## Architecture Details
- Base: 9 layers, 512 dims, 8 heads (4 KV heads)
- Vocab: 1024, Seq Length: 1024
- Tied embeddings, standard Muon optimizer

## How to Train
From the submission folder:
```bash
cd 2026-03-30_GELU_MLP

# Single GPU
RUN_ID=gelu_test \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 (for leaderboard submission)
torchrun --nproc_per_node=8 train_gpt.py
```

## Expected Results
Baseline performance - this is primarily a proof-of-concept submission to demonstrate the submission workflow.
