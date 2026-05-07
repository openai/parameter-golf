# HybridQuantGPT v6.1 — 1.1986 BPB

**val_bpb: 1.1986** (Legal TTT) | **15.13 MB** artifact | 1×RTX 3090, 10K steps (~28h)

## Results

| Metric | Value |
|--------|-------|
| val_bpb (Legal TTT, stride=64) | **1.1986** |
| val_bpb (sliding window, stride=64) | 1.2100 |
| val_bpb (sequential, SWA) | 1.2420 |
| val_bpb (sequential, HMA) | 1.2633 |
| TTT improvement | 0.0114 bpb |
| Steps | 10,000 |
| Wallclock (training) | 101,157s (~28.1h) |
| Wallclock (TTT eval) | 7,457s (~2.1h) |
| Peak memory | 9,205 MiB |
| Total params | 32,760,946 |
| Quantized params | 31,719,424 |
| Artifact size | 15,132,719 bytes |

## Architecture

**HybridQuantGPT v6.1**: 11-layer U-Net Transformer (dim=512, 8 heads, 4 KV heads)

### Mixed-Precision Quantization
| Component | Quantization | Bits |
|-----------|-------------|------|
| Q/K projections | IntNLinear | 6-bit |
| V/O projections | IntNLinear | 5-bit |
| MLP up | PentanaryLinear {-2,-1,0,+1,+2} | ~2.3 bit |
| MLP down | IntNLinear | 4-bit |
| Embeddings | FP16 (tied) | 16-bit |

### Key Techniques
- **rANS entropy coding**: Custom Rust rANS codec for near-Shannon-limit compression
- **U-Net skip connections**: Encoder-decoder with learned skip weights
- **XSA (all layers)**: Cross-Self Attention — remove self-value projection from attention output
- **Value Residual**: First layer V propagated to all subsequent layers via learned lambda
- **SmearGate**: Blend each token with previous token via learned gate
- **BigramHash**: Hash-based bigram embedding (vocab=2048, dim=128)
- **ValueEmbedding (VE128)**: Token identity re-injection at layers 9,10
- **PartialRoPE(16)**: Rotary only on 16 of 64 head dims
- **LN Scale**: Layer-dependent normalization scaling (1/sqrt(layer+1))
- **LeakyReLU(0.5)²**: Activation function for MLP
- **Logit softcap=15, QK gain=2.0**

## Training

- **Optimizer**: Muon (matrix params) + AdamW (embeddings, scalars)
- **LR**: matrix=0.01, tied_embed=0.0125, scalar=0.01
- **Muon momentum**: 0.95 (warmup from 0.85 over 500 steps)
- **Batch tokens**: 524,288
- **Seq len**: 1,024
- **Warmdown**: 17.5% linear decay
- **EMA**: HMA (Hull Moving Average), decay=0.997
- **SWA**: 7 snapshots during warmdown (scale < 0.2), every 50 steps (step 9700-10000)
- **Weight selection**: SWA (1.2420) > HMA (1.2633)
- **GPU**: 1× NVIDIA GeForce RTX 3090 (24 GB)
- **Wallclock**: ~28.1 hours (step_avg ~10.1s)

## Evaluation

### Legal Score-First TTT
- **Method**: SGD fine-tuning on already-evaluated tokens (score-first, fully legal)
- **LR**: 0.002, **Epochs**: 3, **Chunk tokens**: 32,768
- **Frozen**: First 2 blocks (freeze-blocks=2)
- **Sliding window**: stride=64, batch_seqs=32
- **Result**: 1.2100 → **1.1986** (improvement: 0.0114 bpb)
- **TTT eval time**: 7,457s (~2.1h)

### Without TTT
- **Sliding window**: stride=64, batch_seqs=32 → **1.2100 bpb**
- **Pure Python rANS decoder**: No Rust dependency for eval
- Eval data: fineweb10B_sp1024 validation split

## Compression

rANS entropy coding via custom Rust FFI (`rans_codec_rs`):
- Per-layer symbol distribution → near-entropy compression
- rANS compressed weights: 12,807,948 bytes
- Frequency counts: 9,372 bytes
- Per-row scales: 90,112 bytes (FP16)
- Passthrough (embeddings, scalars): 2,083,044 bytes (FP16)
- Model artifact: 15,066,137 bytes (model.rans.ptz)
- Code: 66,582 bytes (train_gpt.py)
- **Total: 15,132,719 bytes** (< 16,000,000 limit, 850 KB headroom)

## Setup and Run

```bash
# Evaluation with Legal TTT (pure Python, no Rust needed)
cd parameter-golf
python records/track_non_record_16mb/2026-03-30_HybridQuantGPT_v61_rANS_SWA_10K/train_gpt.py \
    --eval --checkpoint /path/to/model.rans.ptz --stride 64 \
    --ttt --ttt-lr 0.002 --ttt-epochs 3 --ttt-chunk-tokens 32768 --ttt-freeze-blocks 2

# Evaluation without TTT
python records/track_non_record_16mb/2026-03-30_HybridQuantGPT_v61_rANS_SWA_10K/train_gpt.py \
    --eval --checkpoint /path/to/model.rans.ptz --stride 64

# Training (requires rans_codec_rs for artifact saving)
CUDA_VISIBLE_DEVICES=0 python train_gpt.py --train \
    --iterations 10000 --ema 0.997 --ema-type hma --swa \
    --muon-momentum 0.95 --warmdown-ratio 0.175 \
    --val-every 500 --save-every 2500 --micro-batch 16
```

## Hardware Note

All training and evaluation performed on a **single NVIDIA RTX 3090 (24 GB)**. This submission demonstrates that competitive results (within 0.08 bpb of the 1st place record 1.1194) are achievable on consumer-grade hardware with extended training time, without requiring multi-GPU H100 setups.

## Compliance

- [x] Artifact <= 16,000,000 bytes (15,132,719)
- [x] Non-record submission (unlimited compute)
- [x] Single-file train_gpt.py with full training + eval code
- [x] Pure Python rANS decoder (no external binary dependencies for eval)
- [x] Legal TTT: only fine-tunes on already-evaluated tokens (score-first)
