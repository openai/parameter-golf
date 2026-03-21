# 11L Int5-All + EMA + XSA3 + Causal TTT (12 epochs, 64 chunks)

**val_bpb: 1.13742** (single seed=42, sliding window stride=64, post int5+zstd quantization roundtrip)

## Run Command

```bash
# Setup (once)
bash prepare.sh

# Train + evaluate (default seed=42)
bash eval/eval.sh

# With specific seed
SEED=42 bash eval/eval.sh
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed.

## Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|---------------|-------|
| 42 | 1.13742 | 15,516,237 | yes |

## Key Techniques

### Causal Test-Time Training (TTT)
- Legal TTT: splits validation data into 64 chunks
- For each chunk: **evaluate first** (compute and record loss), **then** train on those already-evaluated tokens
- 12 SGD epochs per chunk (lr=0.004, momentum=0.9)
- All model parameters adapted (no layer freezing)
- Obeys the rule: "when evaluating token N, allowed to train on [0:N-1]"
- Each chunk is scored once and never re-evaluated after training

### Int5-All Quantization
- **Int5 [-16,15]** for ALL weight categories (MLP, attention, other)
- **FP16** for tied embeddings and last-layer key projections
- Int5 quantization enables 11 layers to fit under 16MB

### EMA (Exponential Moving Average)
- EMA decay=0.997 applied during training
- Provides better model averaging than SWA

### Exclusive Self-Attention (XSA)
- Applied to last 3 layers
- Subtracts self-value projection from attention output
- Improves quality at minimal compute cost

## Architecture
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate
- Orthogonal init with scaled output projections (1/sqrt(2*num_layers))
- U-Net skip connections, tied embeddings
- Logit softcap=30

## Training Hyperparameters
- Muon optimizer: matrix_lr=0.02, WD=0.08, momentum=0.99
- AdamW for embeddings/scalars
- warmdown=3000 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- EMA: decay=0.997
- Sliding window eval: stride=64

## TTT Compliance Note

The causal TTT implementation strictly follows the challenge rules:

1. **No training before evaluation**: Each chunk is evaluated first, loss is recorded, then training occurs
2. **No re-evaluation**: Tokens are scored exactly once; training on chunk N cannot affect scores for chunks 0..N
3. **No multiple passes**: The validation set is processed in a single sequential pass (64 chunks)
4. Training time (600s wallclock) is separate from evaluation/TTT time
