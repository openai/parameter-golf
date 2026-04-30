# 11L Int6 QAT + SmearGate + Decoupled WD 0.038

**val_bpb: 1.1502** | **Artifact: 15.50 MB**

## Key Techniques

1. **11 layers** — More depth for better representation capacity. Enabled by int6 compression fitting under 16MB.

2. **Int6 Quantization-Aware Training (QAT)**: Straight-through estimator (STE) fake int6 quantization during the forward pass. Per-row symmetric quantization with 6-bit clipping.

3. **Int6-in-Int8 compression**: Int6 values (-32 to 31) stored in int8 containers. zstd-22 compresses the restricted value range ~35%.

4. **SmearGate**: Learned gate (~513 params) blending current and previous token embeddings, zero-initialized.

5. **Decoupled Muon weight decay (0.038)**: High WD keeps weights small for better int6 quantization and generalization.

6. **Sliding window evaluation** (stride=64, batch=32 sequences): Full-context scoring for every token position.

7. **FP16 tied embedding passthrough**: Embedding weights kept in fp16.

## Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads, MLP mult 3x
- 26.5M parameters, tied embeddings
- Vocab size 1024 (sp1024 tokenizer)
- Training sequence length 2048, batch 524288 tokens

## Training Config

| Parameter | Value |
|-----------|-------|
| Matrix LR | 0.02 |
| Scalar LR | 0.02 |
| Tied Embed LR | 0.03 |
| Muon Momentum | 0.99 |
| Muon Weight Decay | 0.038 |
| Warmdown Steps | 3000 |
| QAT Bits | 6 |

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 1.9421 | 1.1502 | 7723 | 77.25 |

Serialized model int6+zstd22: 15,411,175 bytes
Code size: ~72,000 bytes
Total artifact: 15,495,792 bytes (under 16MB cap)

## Command

```bash
NCCL_NVLS_ENABLE=0 \
RUN_ID=v3c_11L_match179 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_HEADS=8 NUM_KV_HEADS=4 MODEL_DIM=512 \
NUM_LAYERS=11 MLP_MULT=3 \
QAT=1 QUANT_BITS=6 FP16_EMBED=1 LATE_K_LAYERS=0 \
EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_WEIGHT_DECAY=0.038 \
SMEAR_GATE=1 BIGRAM_HASH=0 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_STEPS=3000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
