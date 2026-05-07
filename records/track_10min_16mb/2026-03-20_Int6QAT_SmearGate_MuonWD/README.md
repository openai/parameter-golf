# Int6 QAT + SmearGate + Muon WD

**val_bpb: 1.1669** | **Artifact: 14.77 MB**

## Key Techniques

1. **Int6 Quantization-Aware Training (QAT)**: Straight-through estimator (STE) fake int6 quantization during the forward pass. Per-row symmetric quantization with 6-bit clipping. Nearly eliminates post-quantization degradation — no need for fp16 late-K layer passthrough.

2. **Int6-in-Int8 compression**: Int6 values (-32 to 31) stored in int8 containers rather than bit-packed. zstd-22 compresses the restricted value range ~35%, achieving 14.77MB from 21.8M parameters. Bit-packing destroys byte alignment and defeats compressors.

3. **SmearGate**: Learned gate (~513 params) blending current and previous token embeddings, zero-initialized with very low learning rate (1% of scalar LR). Provides bigram-level context at the embedding layer.

4. **Decoupled Muon weight decay** (0.01): Applied in the Muon optimizer step, improving generalization and quantization robustness.

5. **Sliding window evaluation** (stride=64, batch=32 sequences): Full-context scoring for every token position.

6. **FP16 tied embedding passthrough**: Embedding weights kept in fp16 to avoid compounding int6 errors through both input and output paths.

## Architecture

- 9 layers, 512 dim, 8 heads, 4 KV heads, MLP mult 3x
- 21.8M parameters, tied embeddings
- Vocab size 1024 (sp1024 tokenizer)
- Training sequence length 2048, batch 524288 tokens

## Training Config

| Parameter | Value |
|-----------|-------|
| Matrix LR | 0.06 |
| Scalar LR | 0.06 |
| Tied Embed LR | 0.07 |
| Muon Momentum | 0.99 |
| Muon Weight Decay | 0.01 |
| Warmdown Steps | 8000 |
| QAT Bits | 6 |
| Late K Layers | 0 |

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 1.9703 | 1.1669 | 9706 | 61.80 |

Serialized model int6+zstd22: 14,696,046 bytes
Code size: 71,909 bytes
Total artifact: 14,767,955 bytes (well under 16MB cap)

## Comparison to Current SOTA

| | This submission | Current SOTA (notapplica) |
|--|-----------------|---------------------------|
| val_bpb | **1.1669** | 1.1748 |
| Improvement | **-0.0079** | — |
| Artifact size | 14.77 MB | ~14.7 MB |
| Layers | 9 | 10 |
| Quantization | Int6 QAT | Int8 |

## Command

```bash
NCCL_NVLS_ENABLE=0 \
RUN_ID=int6_qat_smeargate_v5_wd8k \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_HEADS=8 NUM_KV_HEADS=4 MODEL_DIM=512 \
NUM_LAYERS=9 MLP_MULT=3 \
QAT=1 QUANT_BITS=6 FP16_EMBED=1 LATE_K_LAYERS=0 \
EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MATRIX_LR=0.06 SCALAR_LR=0.06 TIED_EMBED_LR=0.07 \
MUON_MOMENTUM=0.99 MUON_WEIGHT_DECAY=0.01 \
SMEAR_GATE=1 BIGRAM_HASH=0 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_STEPS=8000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
