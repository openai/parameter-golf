This record captures the `8192 Vocab + Sliding Window Eval + Selective Quantization` submission.

## Approach

Key modifications from the baseline:
- **SP-8192 tokenizer** — larger vocabulary for better compression per token
- **NorMuon optimizer** — normalized Muon with improved convergence
- **Sliding window evaluation** — stride-256 sliding window for more accurate val_bpb measurement
- **Selective quantization (w6e8)** — INT6 for weight matrices, INT8 for embeddings, achieving better compression while preserving quality
- **8-layer model** with `TRAIN_SEQ_LEN=4096`

Configuration:
- Layout: `VOCAB_SIZE=8192 NUM_LAYERS=8 TRAIN_SEQ_LEN=4096`
- Warmdown: `WARMDOWN_ITERS=3000`
- Sliding window eval: stride=256
- Quantization: selective w6e8 (INT6 weights, INT8 embeddings) + zlib

Command (track-relevant params):
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=v8192_sw \
VOCAB_SIZE=8192 \
NUM_LAYERS=8 \
TRAIN_SEQ_LEN=4096 \
WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt_v8192_sw.py
```

Key metrics (from `train.log`):
- Training stopped at `5467/20000` steps due to wallclock cap
- Pre-quant eval at stop: `val_bpb: 1.1791`
- Post-quant (w6e8+zlib) eval: `val_bpb: 1.1938`
- Step average: `109.76ms`
- Serialized model w6e8+zlib: `14,715,057 bytes` (14.7 MB)
- Code size: `56,813 bytes`
- Total submission size: `14,715,057 bytes`

Hardware:
- 8×H100 SXM (Vast.ai, Iowa)
- 10-minute wallclock cap

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
