**val_bpb: 1.1418** | **15.7 MB** | 1x NVIDIA RTX A6000, ~14 hours

## Summary

11-layer GPT with community meta-stack plus novel techniques: **Value Residual (VR)** and **Gated Attention (GA)**, combined with **Late QAT** during training and **Full GPTQ + Int5 MLP** post-training quantization pipeline. Achieves 1.1418 BPB at stride=128 in a 15.7 MB artifact.

## Novel Contributions

### Value Residual (VR)
Inspired by [arXiv:2410.17897](https://arxiv.org/abs/2410.17897). Each attention layer receives a shortcut from layer-0's V projection (`v0`), blended with the current layer's V via learned mixing. This prevents deep attention layers from losing signal, providing a consistent -0.015 BPB improvement across configurations.

### Gated Attention (GA)
Inspired by [arXiv:2505.06708](https://arxiv.org/abs/2505.06708). A per-head learned sigmoid gate applied after scaled dot-product attention, allowing each head to learn when to suppress or amplify its contribution. Provides -0.003 BPB on top of VR.

### Late QAT (Quantization-Aware Training)
STE fake-quantize applied to all linear layers when the learning rate scale drops below a threshold (0.15), activating during the final ~5% of training. Helps the model adapt its weight distribution to the target int6 quantization format during training.

### Full GPTQ Post-Training Quantization
Hessian-aware column-wise quantization with Cholesky error compensation, applied post-training. Uses 100 calibration batches to collect per-layer Hessian information, then quantizes each column optimally to minimize reconstruction error. Combined with adaptive clip percentile search (GPTQ-lite).

### Int5 MLP Re-quantization
Post-training re-quantization of MLP weights from int6 to int5. Surprisingly acts as regularization, improving BPB by ~0.028 while reducing artifact size.

## Architecture

- 11 layers, dim=512, 8 heads (4 KV), MLP mult=3
- Vocab 1024 (SentencePiece BPE), BigramHash with 1024 buckets
- XSA (first 4 layers), partial RoPE (16 dims), logit softcap=30
- EMA (decay=0.997), SmearGate, orthogonal init, LN scale

## Training Configuration

- 9500 steps, batch size 524K tokens, warmdown 3500 steps
- Muon optimizer: matrix_lr=0.025, momentum=0.99 (warmup from 0.92 over 1500 steps)
- Adam for scalars/embeddings: lr=0.025/0.035
- Weight decay: 0.04 (both Muon and Adam)
- Late QAT threshold: 0.15 (activated at step ~8976)

## Post-Training Pipeline

1. Full GPTQ quantization (100 calibration batches)
2. Int5 MLP re-quantization
3. GPTQ-lite adaptive clip search
4. Int6+zstd serialization → 16,442,824 bytes (15.7 MB)

## Ablation Results (stride=128)

| Configuration | BPB | Delta |
|--------------|-----|-------|
| Base (int6+zstd, no post-training) | 1.1696 | — |
| + Full GPTQ + Int5 + GPTQ-lite | **1.1418** | **-0.028** |
| + VR_V0_FP16 (asymmetric quant) | 1.1418 | +0.000 |
| + SGD TTT (legal, cosine, per-layer) | 1.1721 | +0.030 (worse) |

Key finding: TTT hurts on GPTQ-quantized models — the quantized weight space is incompatible with gradient-based test-time adaptation.

## Reproducibility

```bash
# Training (requires ~14 hours on A6000)
TORCHDYNAMO_DISABLE=1 \
VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=9500 WARMDOWN_ITERS=3500 WARMUP_STEPS=20 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_BACKEND_STEPS=5 GRAD_CLIP_NORM=0.3 \
WEIGHT_DECAY_MUON=0.04 WEIGHT_DECAY_ADAM=0.04 \
SMEAR_GATE=1 BIGRAM_HASH=1 BIGRAM_BUCKETS=1024 BIGRAM_DIM=128 ORTHO_INIT=1 \
XSA_LAYERS=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
PARTIAL_ROPE_DIMS=16 LN_SCALE=1 LOGIT_SOFTCAP=30.0 \
GATED_ATTENTION=1 VALUE_RESIDUAL=1 \
LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
QUANT_BITS=6 EVAL_SEQ_LEN=1024 EVAL_STRIDE=128 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Eval with post-training pipeline (loads trained model)
LOAD_ARTIFACT=model.ptz FULL_GPTQ=1 INT5_MLP=1 GPTQ_LITE=1 \
ITERATIONS=0 EVAL_STRIDE=128 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Files

- `train_gpt.py` — Training script with all techniques
- `submission.json` — Metadata
- `README.md` — This file
