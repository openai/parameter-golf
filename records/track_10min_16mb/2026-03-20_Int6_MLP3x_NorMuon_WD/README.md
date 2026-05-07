# Record: Int6 + MLP 3x + NorMuon + SmearGate + BigramHash + OrthoInit + Sliding Window

**val_bpb: 1.1541** (sliding window stride=256) | **Artifact: 15,992,610 bytes**

Seven improvements stacked on the baseline 9-layer, 512-dim GPT:

### 1. Int6 Per-Row Quantization
Per-row int6 quantization ([-31, 31] range, stored as int8) with fp16 scales and 99.99% percentile clipping. Tied embedding kept in fp16 (most quantization-sensitive). Quantization gap: 0.011 BPB. Frees ~25% artifact space vs int8, enabling MLP 3x.

### 2. MLP 3x Expansion
Hidden dimension 1536 (3x model_dim), up from baseline 1024 (2x). More expressive nonlinear feature transformation, funded by int6 space savings.

### 3. NorMuon + Decoupled Weight Decay
Per-row normalized Newton-Schulz orthogonalization (NorMuon) ensures each gradient row contributes equally. Decoupled weight decay (0.02) regularizes weights for better quantization and compression.

### 4. SmearGate
Learned gate (~512 params) blending each token's embedding with the previous token's. Injects bigram-level context before the transformer processes it.

### 5. BigramHash Embedding
4096-bucket hash table (dim=64, projected to 512) mapping consecutive token pairs to learned embeddings. Near-zero parameter cost, provides explicit token-pair awareness.

### 6. Orthogonal + muP-Scaled Init
All large weight matrices initialized with orthogonal init (gain=1.0). Output projections scaled by `1/sqrt(2 * num_layers)` following muP. Accelerates early convergence.

### 7. Sliding Window Evaluation
Overlapping windows of 2048 tokens, stride 256. Each scored token gets 1792+ tokens of context. Distributed across all GPUs. Zero artifact cost.

## Configuration

```
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3
VOCAB_SIZE=1024
TIE_EMBEDDINGS=1
TRAIN_BATCH_TOKENS=786432
TRAIN_SEQ_LEN=2048
MATRIX_LR=0.02
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99
MUON_WEIGHT_DECAY=0.02
WARMDOWN_ITERS=3000
GRAD_CLIP_NORM=1.0
BIGRAM_VOCAB_SIZE=4096
BIGRAM_DIM=64
EVAL_STRIDE=256
```

## Command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=v2_8xh100_final \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1644 |
| Int6 roundtrip val_bpb | 1.1752 |
| **Sliding window val_bpb** | **1.1541** |
| Quantization gap | 0.011 BPB |
| Sliding window gain | 0.021 BPB |
| Steps | 7,332 |
| Step avg | 81.84ms |
| Tokens | ~5.77B |
| Artifact size | 15,992,610 bytes |
| Peak memory | 17,050 MiB |

Training progression:
- step 2000: val_bpb 1.2678
- step 4000: val_bpb 1.2266
- step 6000: val_bpb 1.1899
- step 7332: val_bpb 1.1644 (wallclock cap)

Trained and evaluated on 8xH100 SXM (RunPod).

## Included Files

- `train_gpt.py` — Training script
- `submission.json` — Leaderboard metadata
- `train.log` — Full 8xH100 training log
