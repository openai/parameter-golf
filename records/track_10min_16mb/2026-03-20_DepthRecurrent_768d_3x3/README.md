# Ultimate Recurrent Parameter Golf

## Result

| Metric | Value |
|--------|-------|
| val_bpb | 1.2271 |
| model_bytes | 11,125,178 (~11.1MB) |
| code_bytes | 64,664 (~63KB) |
| total_bytes | 11,189,842 (~10.7MB) |
| under_16mb | True |
| param_count | 12,836,768 (~12.8M) |
| steps | 6,956 |
| ms/step | 86.26ms |
| hardware | 8xH100 SXM |
| wallclock | 600s |

## Architecture

Depth-recurrent transformer with 3 unique layers shared across 3 passes
(effective depth 9), dim=768, 8q/2kv GQA, seq_len=2048.

## Techniques

### Proven (14)
1. Depth recurrence (3 unique x 3 passes, effective depth 9)
2. Wider dim (768 vs baseline 512)
3. Seq len 2048 (train + eval)
4. GQA (8q / 2kv heads)
5. Sliding window eval (stride=64)
6. RoPE base 500k
7. Spectral embed init (std=0.1/sqrt(dim))
8. Muon weight decay (0.01)
9. Value embeddings
10. Per-pass control params (attn_scale, mlp_scale, resid_mix)
11. U-Net skip connections
12. Adaptive depth (exit gate per token per pass)
13. Confidence conditioning across passes
14. Compression-aware auxiliary loss

### Novel Original (6)
15. Gradient Memory Recurrence
16. Thermodynamic Compression Loss (F=E-T*S)
17. Temporal Difference Recurrence (low-rank, rank=16)
18. Eigenspace Token Routing
19. Resonant Position Encoding
20. Selective State GRU Carry (low-rank, rank=16)

### Phase 1 Optimizations
- Low-rank K projection (rank=32, saves ~415K params)
- Low-rank TD projection (rank=16, saves ~565K params)
- Low-rank GRU state carry (rank=16, saves ~516K params)

## How to Run
```bash
RUN_ID=ultimate_final \
NUM_UNIQUE_LAYERS=3 \
NUM_PASSES=3 \
MODEL_DIM=768 \
NUM_KV_HEADS=2 \
NUM_EXPERTS=1 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
VAL_STRIDE=64 \
USE_LORA_TTT=0 \
VAL_LOSS_EVERY=1000 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
