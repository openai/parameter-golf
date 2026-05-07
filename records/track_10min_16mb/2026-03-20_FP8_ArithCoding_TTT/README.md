# FP8 Training + Arithmetic Coding + SWA

**val_bpb: 1.1511**

## Key Techniques

### 1. FP8 Training via TransformerEngine
- H100 native FP8 tensor cores for ~1.3-1.5x training throughput
- `FP8Linear` wrapper isolates Muon optimizer from TE's internal weight caches
- E4M3 forward / E5M2 backward (HYBRID format) with delayed scaling
- Master weights remain FP32; evaluation runs in BF16 (no FP8 at eval)
- SWA excludes TE internal state (_amax, _scale, _scale_inv)
- Falls back to CastedLinear if TE not installed

### 2. Custom Arithmetic Coder (replaces zstd-22)
- Exploits known peaked distribution of quantized neural network weights
- Per-tensor empirical histogram as probability model (captures zero-spike from 3% magnitude pruning)
- Pure Python 32-bit integer arithmetic coding approaching Shannon entropy limit
- Parallel encoding/decoding via `multiprocessing.Pool` across CPU cores
- Custom binary format eliminates torch.save pickle overhead

### 3. SWA (Stochastic Weight Averaging)
- Step-based early start (SWA_START_STEP=4500) for more averaging steps
- Collects checkpoints every 50 steps during warmdown
- Averages all collected checkpoints at end of training

## Architecture (base from SOTA)
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Run Command

```bash
NUM_LAYERS=10 MODEL_DIM=512 MLP_MULT=3 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=600 \
TTT_ENABLED=0 BIGRAM_VOCAB_SIZE=10240 WEIGHT_DECAY=0.04 \
DATA_PATH=/dev/shm/fineweb10B_sp1024/ \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional: enable LoRA test-time training at evaluation (experimental):
```bash
TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Built On
- SOTA submission by thwu1 (1.1428 BPB): Int5-MLP + BigramHash(10240) + SWA
- LoRA TTT submission by samacqua (1.1928 BPB)
