First pilot run of Mixture of Softmax (MoS) on 1x H100 SXM, 10-minute wallclock.

Configuration:
- Track: `non-record`, 1x H100 SXM, 10 min wallclock
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- MoS: `USE_MOS=1 MOS_K=2 MOS_RANK=64` (low-rank factorization, ~99K extra params)
- Tied embeddings, seed=42

Command:
```bash
RUN_ID=mos_k2_r64_pilot \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 SEED=42 \
USE_MOS=1 MOS_K=2 MOS_RANK=64 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Key metrics:
- Stopped at step 1113/20000 (wallclock cap)
- Pre-quant: `val_loss:2.3505 val_bpb:1.3921`
- Post-quant (int8+zlib): `val_loss:2.3523 val_bpb:1.3932`
- Quantization degradation: +0.0011 bpb (minimal)
- Model params: 17,159,240
- Artifact: 12,764,492 bytes int8+zlib (12.8MB, 3.2MB under 16MB cap)
- Code: 63,345 bytes
- Total: 12,827,837 bytes
- Peak memory: 11,012 MiB allocated
- Step avg: 539ms/step on 1x H100

Training curve:
| Step | Train Loss | Val BPB | Time |
|------|-----------|---------|------|
| 0 | 6.93 | 4.11 | 0s |
| 100 | 3.27 | — | 54s |
| 500 | 2.58 | 1.52 | 271s |
| 1000 | 2.40 | 1.40 | 542s |
| 1113 | — | 1.39 | 600s |

Notes:
- Loss still dropping at wallclock stop — model had more to learn
- No TTT/LoRA eval was run (only int8 roundtrip)
- No same-conditions baseline for direct comparison (8xH100 baseline: ~1.2244 bpb at 20K steps)
- 1x H100 = ~1/8 throughput → only 1113 steps vs ~20K on 8xH100
