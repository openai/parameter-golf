# Parameter Golf Solution — Technical Write-Up

## Approach Summary

This solution extends the official baseline `train_gpt.py` with five key innovations while maintaining full infrastructure compatibility (official data format, DDP, Int8+zlib quantization, tokenizer-agnostic BPB).

## Innovations

### 1. SwiGLU MLP (replaces relu²)

The baseline uses `relu(x)²` as the activation. We replace this with **SwiGLU** (`SiLU(W1·x) * W3·x`), which adds an extra gate projection but improves gradient flow and representation quality. With `mlp_mult=3`, the hidden dimension is `3 * model_dim = 1536`, matched by two up-projections and one down-projection.

### 2. SmearGate

A lightweight gating mechanism that blends each token's embedding with the previous token's embedding using a learned sigmoid gate. This injects local bigram context into the input representation before the transformer blocks process it. Cost: `dim * 2 * dim` parameters (~0.5M).

### 3. BigramHash

A hash table of size 4096 that maps bigram pairs `(token[t], token[t-1])` to learned embeddings via `hash(t * 31 + t_prev) % 4096`. This provides a different form of bigram awareness — shared statistical patterns across token pairs with the same hash. Cost: `4096 * dim` parameters (~2M).

### 4. SENT-lite (Entropy-Weighted Loss)

During training, we weight each token's cross-entropy loss by `1 + alpha * detach(loss)`, where `alpha=0.1`. This gives higher weight to tokens the model finds harder to predict, creating a mild curriculum effect without the complexity of full SENT.

### 5. Batched TTT LoRA

At evaluation time, we apply per-document LoRA adapters (rank 8) on the Q, V, and LM head projections. Each document gets its own adapter weights, which are trained for a few gradient steps on earlier chunks before scoring later chunks. Adapters are fully reset between documents.

## Architecture Details

| Component | Value |
|---|---|
| Layers | 9 (4 encoder + 5 decoder) |
| Model dim | 512 |
| Heads (Q / KV) | 8 / 4 |
| MLP multiplier | 3x (SwiGLU) |
| Vocab size | 1024 |
| Sequence length | 1024 |
| Embeddings | Tied (tok_emb = lm_head) |
| Logit softcap | 30.0 |
| Position encoding | RoPE (base 10000) |
| Normalization | RMSNorm (pre-norm) |
| Skip connections | U-Net style (encoder→decoder) |

## Optimizer Configuration

- **Muon** (2D matrices): Newton-Schulz orthogonalization, 5 iterations, lr=0.04
- **Adam** (embeddings): lr=0.05, betas=(0.9, 0.95)
- **Adam** (scalars/control tensors): lr=0.04, betas=(0.9, 0.95)
- **Warmdown**: 1200 iterations, wallclock-aware

## Results

- **Training**: Completes within 600 seconds on 8xH100 (wallclock enforced)
- **Artifact**: Int8+zlib under 16MB (model + code)
- **BPB**: Measured via tokenizer-agnostic SentencePiece LUTs

## Reproduction

```bash
# On 8xH100 machine with official data cached:
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script automatically:
1. Loads official cached shards (256-int header format)
2. Trains with DDP across all GPUs
3. Stops at 600 seconds wallclock
4. Exports Int8+zlib model (`final_model.int8.ptz`)
5. Validates roundtrip accuracy
6. Runs TTT LoRA evaluation (competition score)
