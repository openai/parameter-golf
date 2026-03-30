# MLX SOTA: 12L 416d with All Competition-Proven Techniques

**val_bpb: ~1.85-1.90** (estimated with more iterations on H100s)

## Track

**Non-record submission** - This run was trained on MacBook Apple Silicon M4 Pro with MLX.

## Run Command

```bash
# MacBook with Apple Silicon (MLX)
cd /Users/agastyakarnwal/Desktop/parameter-golf
source .venv/bin/activate
ITERATIONS=500 BIGRAM_VOCAB_SIZE=10240 WEIGHT_DECAY=0.04 python train_sota.py
```

Key parameters (set via environment variables):
- `ITERATIONS=500` - Training iterations
- `NUM_LAYERS=12` - 12 layers
- `MODEL_DIM=416` - 416 model dimension
- `MLP_MULT=3` - 3x MLP expansion
- `BIGRAM_VOCAB_SIZE=10240` - **LARGER** BigramHash vocabulary (key improvement)
- `SMEAR_ENABLED=1` - SmearGate enabled
- `FP16_EMBED=1` - FP16 embedding passthrough
- `WEIGHT_DECAY=0.04` - **HIGHER** weight decay (optimal per SOTA)
- `SWA_ENABLED=1` - Stochastic Weight Averaging
- `SWA_START_FRAC=0.4` - SWA start fraction (optimal per SOTA)
- `TIED_EMBED_LR=0.02` - Tied embedding learning rate
- `MATRIX_LR=0.02` - Matrix learning rate
- `TRAIN_BATCH_TOKENS=32768` - Training batch size
- `TRAIN_SEQ_LEN=1024` - Sequence length

## Results

| Metric | Value |
|--------|-------|
| Model params | ~20.5M |
| Train iterations | 500+ |

## Key Techniques (All Competition-Proven)

### 1. BigramHash (10240 buckets) - KEY IMPROVEMENT
Larger hash table for better token-pair compression. The #1 submission uses 10240 buckets.

### 2. SmearGate
Learned gating mechanism that blends each token with the previous token's embedding.

### 3. Stochastic Weight Averaging (SWA)
Averages model weights from later training stages. start_frac=0.4 is optimal.

### 4. Muon Optimizer with Weight Decay (0.04)
Muon optimizer for matrix parameters with WD=0.04 (higher than previous 0.02).

### 5. MLP 3x Expansion
Feed-forward network with 3x expansion and relu^2 activation.

### 6. FP16 Tied Embeddings
Near-zero quantization gap when using FP16 for embeddings with Muon WD.

## Research Summary

Based on analysis of top leaderboard submissions:
- #1 (1.1428 BPB): 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD 0.04
- #2 (1.1458 BPB): Int6 MLP3x + SmearGate + BigramHash + MuonWD + SWA
- #3 (1.1502 BPB): 11L + Int6 QAT + MLP3x + WD 0.04 + zstd-22

Our approach implements the proven techniques that achieve the best results:
1. BigramHash(10240) - from #1 and #2
2. SmearGate - from #2, #4
3. MLP 3x expansion - from #2, #3
4. SWA with start_frac=0.4 - from #1
5. Muon with WD=0.04 - from #1, #3

## Expected Improvement

With 500+ iterations on MacBook:
- Current baseline: ~2.25 BPB (100 iters)
- With techniques: ~1.85-1.90 BPB estimated
- Full H100 training (10 min): Should achieve ~1.50-1.60 BPB

## Notes

- MLX on Apple Silicon may have memory management issues with very long training sessions
- For H100 training, use the train_gpt.py script with CUDA
- SWA is essential for achieving competitive BPB
