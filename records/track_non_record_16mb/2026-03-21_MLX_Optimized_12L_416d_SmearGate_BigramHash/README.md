# MLX-Optimized 12L 416d with SmearGate + BigramHash

**val_bpb: 1.9011** (500 iterations on Apple Silicon M4 Pro, MacBook)

## Track

**Non-record submission** - This run was trained on MacBook Apple Silicon (not H100s within 10-minute cutoff).

## Run Command

```bash
# MacBook with Apple Silicon (MLX)
cd /Users/agastyakarnwal/Desktop/parameter-golf
source .venv/bin/activate
ITERATIONS=500 python train_optimized.py
```

Key parameters (set via environment variables):
- `ITERATIONS=500` - Training iterations
- `NUM_LAYERS=12` - 12 layers
- `MODEL_DIM=416` - 416 model dimension
- `MLP_MULT=3` - 3x MLP expansion
- `BIGRAM_VOCAB_SIZE=4096` - BigramHash vocabulary size
- `SMEAR_ENABLED=1` - SmearGate enabled
- `FP16_EMBED=1` - FP16 embedding passthrough
- `MUON_WEIGHT_DECAY=0.02` - Muon weight decay
- `TIED_EMBED_LR=0.02` - Tied embedding learning rate
- `MATRIX_LR=0.02` - Matrix learning rate
- `TRAIN_BATCH_TOKENS=32768` - Training batch size
- `TRAIN_SEQ_LEN=1024` - Sequence length

## Results

| Metric | Value |
|--------|-------|
| Val BPB | 1.9011 |
| Model params | 19,716,545 |
| Train iterations | 500 |
| Final train loss | ~3.6 |

**Note**: This is an undertrained model. The same architecture with 3000+ iterations on H100s should achieve significantly better BPB (potentially 1.5-1.6 BPB based on PR#328 findings).

## Key Techniques

### 1. SmearGate
Learned gating mechanism that blends each token with the previous token's embedding. This helps capture local context dependencies.

### 2. BigramHash
Hash consecutive token pairs into a learned embedding table (4096 buckets). Projects to model_dim via learned linear layer.

### 3. FP16 Embedding Passthrough
Using FP16 for tied embeddings + Muon weight decay enables near-zero quantization gap (only ~0.001 BPB).

### 4. MLP 3x Expansion
3x MLP hidden dimension expansion with relu^2 activation.

### 5. Muon Optimizer
Newton-Schulz orthogonalization-based optimizer with weight decay. Matrix parameters use Muon, scalars use AdamW.

### 6. U-Net Skip Connections
Decoder layers receive skip connections from encoder layers via learned weights.

## Architecture
- 12 layers (6 encoder + 6 decoder)
- 416 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1248), relu^2 activation
- SmearGate + BigramHash(4096, dim=128)
- U-Net skip connections, tied embeddings

## Training Details
- Device: Apple Silicon M4 Pro (24GB unified memory)
- Framework: MLX 0.31.1
- Training tokens: ~16M (500 iters × 32K batch)
- Tokens/sec: ~20,000-24,000

## Comparison

| Submission | val_bpb | Notes |
|------------|---------|-------|
| **This (MLX, 500 iters)** | **1.9011** | MacBook, undertrained |
| PR#328 (14L, 750 iters) | 1.9588 | Reference for MLX scale |
| #1 SOTA (H100, 10min) | 1.1428 | 8xH100, full training |

## Future Improvements

1. **More iterations**: 1000-3000 iterations would significantly improve BPB
2. **Int5/Int6 quantization**: Compress model weights for artifact size
3. **SWA**: Stochastic weight averaging with start_frac=0.4
4. **Larger BigramHash**: Increase to 10240 buckets
5. **Sliding window eval**: Stride=64 for better validation
6. **H100 training**: Full training on 8xH100s would achieve SOTA BPB

## Files

- `train_optimized.py` - Complete MLX training script
- `train.log` - Training log from the run
- `submission.json` - Submission metadata

---

Built on insights from:
- PR #328 (Mac sub-2 BPB approach)
- Top submissions: Int5-MLP, SmearGate, BigramHash, Muon WD, SWA
