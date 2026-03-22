# 11L EMA + TTT + XSA + Partial RoPE + LN Scale

**val_bpb = 1.1329** (3-seed mean, std=0.0006) | 8×H100 SXM | 600s

| Seed | Steps | val_bpb | Artifact |
|------|-------|---------|----------|
| 1337 | 8,193 | 1.1335 | 15.8 MB |
| 42 | 8,585 | 1.1328 | 15.9 MB |
| 7 | 8,730 | 1.1324 | 15.3 MB |

## Config

```
NUM_LAYERS=11  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4  MLP_MULT=3
QUANT_BITS=6  USE_ZSTD=1  TIE_EMBEDDINGS=1
TRAIN_SEQ_LEN=2048  EVAL_SEQ_LEN=2048  ROPE_BASE=10000
TRAIN_BATCH_TOKENS=524288  WARMDOWN_ITERS=3000
XSA_LAST_N=4  EMA_ENABLED=1  EMA_DECAY=0.997
TTT_ENABLED=1  TTT_LR=0.002  TTT_EPOCHS=3  TTT_FREEZE_BLOCKS=1
SMEAR_GATE=1  BIGRAM_HASH=1  ORTHO_INIT=1  UNET_SKIPS=1
ROPE_DIMS=16  LN_SCALE=1  QAT=0
FP16_EMBED_EXPORT=0  LATE_K_FP16=0
MUON_MOMENTUM=0.99  GRAD_CLIP_NORM=0.3  MUON_WD=0.04  ADAM_WD=0.04
EVAL_STRIDE=32  DOC_ISOLATED_EVAL=0
```

## Reproduction

```bash
git clone https://github.com/mrdavtan/parameter-golf.git
cd parameter-golf && git checkout int6-3xMLP-pr
pip install flash-attn --no-cache-dir --no-build-isolation
pip install zstandard sentencepiece huggingface_hub
python3 data/cached_challenge_fineweb.py --variant sp1024

SEED=7 QAT=0 TTT_MAX_STEPS=500 TTT_FREEZE_BLOCKS=1 \
TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 UNET_SKIPS=1 \
ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000 \
EVAL_STRIDE=32 DOC_ISOLATED_EVAL=0 \
LATE_K_FP16=0 FP16_EMBED_EXPORT=0 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
```

Hardware: 8×H100 SXM (RunPod), PyTorch 2.9.1+cu128, Flash Attention 2
