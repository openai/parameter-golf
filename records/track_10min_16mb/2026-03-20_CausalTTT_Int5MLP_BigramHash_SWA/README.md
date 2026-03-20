# 10L Int5-MLP + BigramHash(16384) + Causal TTT + SWA(0.3/20) + WD=0.08

**val_bpb: 1.14020** (mean of 3 seeds, sliding window stride=64, post int5/int6+zstd quantization roundtrip)

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

## 3-Seed Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|---------------|-------|
| 42 | 1.13976 | 15,795,375 | yes |
| 1337 | 1.14063 | 15,739,819 | yes |
| 2024 | 1.14020 | ~15.8M | yes |
| **Mean** | **1.14020** | | |
| **Std** | **0.00036** | | |

## Key Techniques

### Causal Test-Time Training (TTT)
- Legal TTT: splits validation data into 32 chunks
- For each chunk: **evaluate first** (compute and record loss), **then** train on those already-evaluated tokens
- 3 SGD epochs per chunk (lr=0.004, momentum=0.9)
- All model parameters adapted (no layer freezing)
- Obeys the rule: "when evaluating token N, allowed to train on [0:N-1]"
- Each chunk is scored once and never re-evaluated after training

### Mixed Int5/Int6 Quantization
- **Int5 [-16,15]** for MLP weights (most compressible, ~1.88x zstd ratio)
- **Int6 [-32,31]** for attention weights (precision-sensitive, ~1.51x zstd ratio)
- **FP16** for tied embeddings and last-layer key projections
- Int5 MLP saves ~1.86MB vs uniform int6, enabling a 10th layer

### BigramHash(16384)
- Hash consecutive token pairs into 16384-bucket embedding table (dim=64)
- Projected to model_dim=512 via learned linear
- Reduces token-pair hash collisions vs smaller tables

### SWA with start_frac=0.3, every=20
- Collect checkpoints from last 30% of warmdown
- Average every 20 steps for more frequent snapshots
- More aggressive averaging yields better generalization

## Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(16384, dim=64)
- Orthogonal init with scaled output projections (1/sqrt(2*num_layers))
- U-Net skip connections, tied embeddings
- Logit softcap=30

## Training Hyperparameters
- Muon optimizer: matrix_lr=0.02, WD=0.08, momentum=0.99
- AdamW for embeddings/scalars
- warmdown=3000 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- SWA: start_frac=0.3, every=20 steps
- Sliding window eval: stride=64

## TTT Compliance Note

The causal TTT implementation strictly follows the challenge rules:

1. **No training before evaluation**: Each chunk is evaluated first, loss is recorded, then training occurs
2. **No re-evaluation**: Tokens are scored exactly once; training on chunk N cannot affect scores for chunks 0..N
3. **No multiple passes**: The validation set is processed in a single sequential pass (32 chunks)
4. Training time (600s wallclock) is separate from evaluation/TTT time
