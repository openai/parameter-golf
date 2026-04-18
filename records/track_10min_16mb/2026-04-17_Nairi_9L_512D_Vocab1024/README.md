# Nairi Model: Parameter Golf Submission

## Approach

This submission uses the baseline architecture with additional encoder-decoder skip connections for improved gradient flow.

## Architecture Details

- **Model:** 9L Transformer, 512 dim, 8 heads, 4 KV heads
- **Vocab:** 1024 (using SentencePiece BPE 8192 tokenizer)
- **Tied embeddings:** Yes
- **Optimizer:** Muon
- **Skip connections:** Encoder-decoder style
- **Normalization:** RMSNorm
- **Attention:** RoPE, Flash Attention
- **Quantization:** int8 + zlib compression

## Key Techniques

1. Baseline architecture from OpenAI parameter-golf
2. Encoder-decoder skip connections for better gradient flow
3. Muon optimizer for fast convergence
4. Early stopping at optimal step
5. Token clamping to prevent OOB errors
6. Vocab size mismatch (1024 model vs 8192 tokenizer) for compression efficiency

## Results

| Seed | val_bpb | Best Step |
|------|----------|-----------|
| 1337 | 0.5116 | 250 |
| 42 | 0.5570 | 500 |
| 2025 | 0.5467 | 500 |

## Hyperparameters

```
vocab_size = 1024
num_layers = 9
model_dim = 512
num_heads = 8
num_kv_heads = 4
mlp_mult = 2
tie_embeddings = True
train_batch_tokens = 524288
train_seq_len = 1024
matrix_lr = 0.04
muon_momentum = 0.95
max_wallclock_seconds = 600
```

## Tokenizer Notes

Using SentencePiece BPE tokenizer with 8192 vocab, but model uses only 1024 dimensions. This creates efficiency through unused parameter space.

## Files

- `train_gpt.py` - Training code
- `submission.json` - Submission metadata
- `README.md` - This file
- `nairi_model_compressed.ptz` - Model weights (included separately)