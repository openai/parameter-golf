# MetaStack v3 Competitive

## Score

**val_bpb: 1.1792** (sliding window, int6+zstd22, stride=64, seq_len=1024)

## Summary

10-layer GPT with BigramHash embeddings, SmearGate, OrthoInit, SWA (30 checkpoints), decoupled weight decay (Muon WD=0.04), mixed int5(MLP)/int6(attn) quantization, and 2% magnitude pruning. Trained for 600 seconds on 8xH100 SXM, reaching step 7819/20000 before wallclock cap.

## Key Results

| Metric | Value |
|--------|-------|
| Sliding window val_bpb (submission) | **1.1792** |
| Pre-quant terminal val_bpb | 1.1806 |
| int8 roundtrip val_bpb | 1.1830 |
| int6 roundtrip val_bpb | 1.2139 |
| int6 artifact size | 12,099,323 bytes (12.1 MB) |
| int8 artifact size | 20,098,187 bytes (over 16MB cap) |
| Code size | 63,971 bytes |
| Model params | 24,730,705 |
| Training time | 600,090 ms (wallclock cap) |
| Steps completed | 7,819 / 20,000 |
| Step avg | 76.75 ms |
| Peak GPU memory | 18,861 MiB |
| SWA checkpoints | 30 |
| Sliding eval config | seq_len=1024, stride=64, 8-GPU, 81s |

## Architecture

- 10 transformer layers (5 encoder + 5 decoder with U-Net skip connections)
- 512 model dim, 8 attention heads, 4 KV heads (GQA)
- MLP 3x (hidden=1536), relu^2 activation
- BigramHash: 4096-bucket hashed bigram embedding (dim=128, projected to 512)
- SmearGate: per-dim learned sigmoid gate blending token with predecessor
- OrthoInit: orthogonal initialization for large linear layers, muP output scaling
- Tied embeddings with fp16 passthrough during quantization
- Logit softcap = 30

## Optimizer

- Muon for matrix params (lr=0.02, momentum=0.99, WD=0.04)
- AdamW for scalar/token params (scalar_lr=0.04, token WD=0.01)
- Tied embed lr=0.03
- Grad clip norm=0.3
- 20-step warmup, 3000-step warmdown
- SWA: enabled, every 50 steps, start_frac=0.5

## Quantization

- Mixed precision: int5 (clip_range=15) for MLP weights, int6 (clip_range=31) for attention weights
- Per-row quantization with fp16 scales
- 2% magnitude pruning before quantization
- Tied embedding kept in fp16 (not quantized)
- Compression: zstd level 22

## Evaluation

- Sliding window: seq_len=1024 (matching training), stride=64
- Each token scored with near-full context (only last 64 positions per window scored)
- Batched evaluation (256 windows per forward pass) across 8 GPUs
- Total: 969,073 windows, 81 seconds

## What Changed vs MetaStack v2 WD

1. **+BigramHash** (+524K params): hashed bigram context at embedding layer
2. **+SmearGate** (+512 params): learned per-dim token blending gate
3. **+OrthoInit**: orthogonal init for large linears, muP scaling on projections
4. **+10th layer**: funded by mixed int5/int6 quantization savings
5. **+SWA every 50 steps**: 30 checkpoint average during warmdown
6. **+Magnitude pruning 2%**: improves compressibility
7. **+WD=0.04 Muon, 0.01 scalar/token**: from 0.0 default
8. **+Grad clip 0.3**: from 0.0 default
9. **+Batch tokens 786K**: from 524K
10. **Fixed sliding eval**: seq_len=1024 (was 2048, causing RoPE extrapolation failure), stride=64 (was 256)

## Reproducibility

```bash
cd /workspace/parameter-golf
RUN_ID=v3_competitive_h100_001 SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 \
EVAL_SLIDING=1 EVAL_SEQ_LEN=1024 EVAL_STRIDE=64 EVAL_BATCH_SEQS=256 \
MAX_WALLCLOCK_SECONDS=600 TRAIN_BATCH_TOKENS=786432 \
NUM_LAYERS=10 MUON_WEIGHT_DECAY=0.04 SCALAR_WEIGHT_DECAY=0.01 \
TOKEN_WEIGHT_DECAY=0.01 GRAD_CLIP_NORM=0.3 \
SWA_ENABLED=1 SWA_EVERY=50 SWA_START_FRAC=0.5 \
BIGRAM_ENABLED=1 SMEARGATE_ENABLED=1 \
MIXED_QUANT_MLP_BITS=5 PRUNE_FRACTION=0.02 \
VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_MetaStack_v3_competitive/train_gpt.py
```
