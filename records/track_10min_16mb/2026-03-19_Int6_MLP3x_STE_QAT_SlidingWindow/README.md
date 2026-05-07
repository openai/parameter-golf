# Int6 MLP3x + STE QAT + Sliding Window Eval

## Summary

Six-technique stack achieving **val_bpb=1.1594** (post int6+zstd quantization roundtrip, sliding window eval stride=64):

1. **Int6 per-row quantization + zstd-22 compression** — saves ~4MB vs int8+zlib, enabling wider MLP within the 16MB budget
2. **3x MLP expansion** (MLP_HIDDEN=1536) — enabled by int6 savings, 3x wider hidden layer than baseline
3. **STE fake int6 quantization-aware training** — CastedLinear forward pass uses fake-quantized weights via Straight-Through Estimator, teaching weight distributions that survive int6 post-training quantization
4. **fp16 tied embedding passthrough** — tied embedding kept in fp16 (no quantization penalty on the output head)
5. **Sliding window evaluation** (stride=64, seq_len=4096) — each token scored with nearly full context
6. **Co-optimized training dynamics** — MATRIX_LR=0.02, MUON_MOMENTUM=0.99, WARMDOWN_ITERS=3000, TRAIN_SEQ_LEN=4096, TRAIN_BATCH_TOKENS=393216

## Configuration

```
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_MULT=3.0 TIE_EMBEDDINGS=1
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000
TRAIN_BATCH_TOKENS=393216 TRAIN_SEQ_LEN=4096
EVAL_STRIDE=64
MAX_WALLCLOCK_SECONDS=600
```

## Command

```bash
pip install zstandard
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics (seed 1337)

- Training stopped at step **10535/20000** due to 600s wallclock cap
- Average training step time: **56.95ms**
- Model params: ~21.8M (MLP 3x)
- Pre-quant eval: `val_loss:1.9801 val_bpb:1.1727`
- Int6+zstd roundtrip: `val_loss:1.9575 val_bpb:1.1594`
- Sliding window eval time: **207s** (within 10-min eval budget)
- Compressed model (int6+zstd): **15,105,576 bytes**
- Code size: **57,201 bytes**
- **Total artifact: 15,162,777 bytes** (under 16,000,000 limit)
- Peak memory: 8,522 MiB per GPU

## Quantization Strategy

Mixed precision quantization optimized for the 16MB budget:

- **MLP + attention 2D weights**: int6 per-row quantization ([-32, 31]), scale in fp16
- **Tied embedding** (`tok_emb.weight`): fp16 passthrough (no quantization — critical for output head quality)
- **Small tensors** (scales, norms, gains): fp16 or fp32 passthrough
- **Compression**: zstd level 22 (better ratio than zlib-9 on int6 data)

## Approach

This submission composes techniques from several community PRs (#42, #50, #52, #65, #70) into a unified training script. The key insight is that int6 quantization + zstd compression saves enough artifact bytes to fit a 3x MLP expansion, while STE-based QAT during training ensures weights survive the aggressive 6-bit quantization with minimal quality loss (~0.013 BPB quant gap).

Built with Claude Code (Anthropic) as an autonomous research agent.
