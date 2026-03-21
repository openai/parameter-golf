## Submission: 11L NTK-RoPE + FA3 + Batch524K + XSA4 + EMA (val_bpb: 1.1328)

**val_bpb: 1.1328** (sliding window, stride=64, 3-seed mean) | **15.87 MB** (mean) | 8xH100 SXM, 600s

### Key Innovations

| Change | Impact |
|--------|--------|
| **NTK-aware RoPE** | Auto-scales RoPE base frequency when seq_len > train_seq_len (1024→2048 triggers ~4x base scaling). Better long-range context modeling without explicit base tuning. |
| **FlashAttention 3** (Hopper) | 58ms/step vs 99ms with SDPA — enables 10,300+ training steps in 600s |
| **Batch=524K** (from 786K) | Smaller batch = faster steps = 71% more gradient updates |
| **Adaptive Pruning** | Binary search for minimal pruning % that guarantees artifact under 16MB per seed |
| **XSA on last 4 layers** | Exclusive Self Attention removes self-value bias (PR #265) |
| **EMA** (decay=0.997) | Exponential moving average replaces SWA for smoother weight averaging |

### Results (3 seeds, 8xH100 SXM)

| Seed | Steps | Sliding BPB (s64) | Artifact | Prune % |
|------|-------|-------------------|----------|---------|
| 42 | 10,368 | 1.1344 | 15.89 MB | 14% |
| 1337 | 10,346 | 1.1320 | 15.89 MB | 10% |
| **2024** | **10,358** | **1.1319** | **15.83 MB** | **12%** |

**Mean: 1.1328 | Std: 0.0014** | Submitted: seed 2024 (best)

### Architecture

- 11 transformer layers, 512-dim, 8 heads (4 KV heads via GQA)
- 3x MLP expansion (1536 hidden), relu-squared activation
- U-Net skip connections (encoder=5, decoder=6)
- SmearGate + BigramHash (4096 buckets, dim=128)
- Tied embeddings, logit softcap=30.0
- NTK-aware RoPE (auto-scales at seq > train_seq_len=1024)
- XSA on layers 7-10 (deepest 4 of 11)

### Training

- FlashAttention 3 (Hopper-optimized, installed from source with sm80 disabled)
- Muon optimizer: lr=0.025, momentum=0.99 (warmup from 0.92 over 1500 steps)
- AdamW for embeddings/scalars: lr=0.035/0.025
- Weight decay: 0.04 (both Muon and AdamW)
- Warmdown: 3000 iterations, grad clip 0.3
- EMA (decay=0.997, every step)
- OrthoInit + muP-scaled output projections
- Batch: 524,288 tokens/step

### Quantization

- Int5 per-row quantization on MLP weights (clip=15)
- Int6 per-row quantization on attention + bigram weights (clip=31)
- Int8 for embeddings
- Adaptive magnitude pruning (10-16%, auto-selected per seed to fit 16MB)
- zstd level 22 compression

### Run Command

```bash
SEED=2024 bash eval/eval.sh
```

### FA3 Installation Note

FA3 must be built from source with CUDA version check bypassed:
```bash
cd flash-attention/hopper
FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE \
FLASH_ATTENTION_DISABLE_FP8=TRUE TORCH_CUDA_ARCH_LIST="9.0a" \
MAX_JOBS=16 python setup.py install
```
