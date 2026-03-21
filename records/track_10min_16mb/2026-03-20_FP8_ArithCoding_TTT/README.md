# FP8 Training + Arithmetic Coding + LoRA TTT

**val_bpb: TBD** (pending 3-seed validation on 8xH100)

## Key Techniques

### 1. FP8 Training via TransformerEngine
- H100 native FP8 tensor cores for ~1.3-1.5x training throughput
- `MuonSafeLinear` wrapper isolates Muon optimizer from TE's internal weight caches
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
- Expected ~1.0-1.3MB savings over zstd-22

### 3. LoRA Test-Time Training
- Per-document LoRA adapters (rank=8) targeting lm_head, c_q, c_v
- Evaluate chunk -> train on chunk -> evaluate next chunk (no info leakage)
- Reset LoRA between documents
- Compatible with SOTA model architecture (SmearGate, BigramHash, U-Net skips)

## Architecture (base from SOTA)
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Disable individual optimizations:
```bash
USE_FP8=0 TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Built On
- SOTA submission by thwu1 (1.1428 BPB): Int5-MLP + BigramHash(10240) + SWA
- LoRA TTT submission by samacqua (1.1928 BPB)
