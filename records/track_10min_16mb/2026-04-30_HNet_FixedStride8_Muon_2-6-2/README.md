# Fixed-Stride H-Net + Muon | val_bpb = 1.9058

**Track**: 10min_16mb | **Author**: @firfre | **Date**: 2026-04-30

> **Note**: This result was obtained by training for **10 minutes on a single H100 80GB**. The competition evaluates on 8×H100, which means more steps in the same wall-clock budget — the actual submission score is expected to be better than the val_bpb reported here.

## Result

| Metric | Value |
|--------|-------|
| val_bpb (after int6+brotli11) | **1.9058** |
| Artifact size | 13.04 MB |
| Training time | **600s on 1×H100 80GB** (944 steps) |
| Competition eval hardware | 8×H100 80GB |

## Architecture

Fixed-stride hierarchical byte-level model (no learned routing):

```
Input bytes → Embedding
→ Encoder: 2× Mamba2 (byte level, seq_len=1024)
→ Fixed stride select: every 8th token (causal, seq_len=128)
→ Chunk Transformer: 6× Causal Self-Attention (word level)
→ Causal upsample: repeat_interleave(8) + shift right by 1
→ + Residual
→ Decoder: 2× Mamba2 (byte level)
→ LM head
```

**Key hyperparameters**:
- `MODEL_DIM=512`, `CHUNK_MODEL_DIM=576`, `CHUNK_FFN_DIM=2304`
- `MAMBA_STATE=32`, `NUM_HEADS=8`, `CHUNK_ROTARY_DIM=48`
- `VOCAB_SIZE=260` (byte260 tokenizer, 1 token = 1 byte)

## Training

**Optimizer**: Muon for 2D weight matrices + AdamW for embeddings/scalars/head
- Muon: `lr=0.02`, `momentum=0.95`, Newton-Schulz steps=5
- AdamW: `embed_lr=0.6`, `head_lr=0.008`, `scalar_lr=0.04`
- Warmdown: 1200 iters (time-aware)

**torch.compile**: enabled

## Compression

- 2D matrices ≥ 65536 elements: **int6** quantization (per-row scale)
- Mamba SSM dynamics (A_log, dt_proj, conv1d, D): **int8**
- Small tensors: **fp16** passthrough
- Final compression: **brotli-11**
- Compression ratio: ~3.2× on quantized payload

## Reproduce

```bash
FIXED_STRIDE=8 ENCODER_LAYERS=2 CHUNK_LAYERS=6 DECODER_LAYERS=2 \
VOCAB_SIZE=260 \
TOKENIZER_PATH=./data/tokenizers/fineweb_pure_byte_260.json \
DATA_PATH=./data/datasets/fineweb_byte260 \
MAMBA_STATE=32 MODEL_DIM=512 CHUNK_MODEL_DIM=576 CHUNK_ROTARY_DIM=48 \
USE_MUON=1 MUON_LR=0.02 USE_COMPILE=1 \
VAL_LOSS_EVERY=200 \
python train_hnet_v5.py
```
