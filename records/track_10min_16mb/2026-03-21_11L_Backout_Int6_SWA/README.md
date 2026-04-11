# 11L Backout + Int6 + SWA (val_bpb: 1.1364)

**val_bpb: 1.1364** (sliding window, stride=64) | **16.17 MB** | 8xH100 SXM, 600s

## Known Issue

Artifact is 16,170,051 bytes — 170KB over the 16,000,000 byte cap. The code already supports `INT5_MLP=1` which switches MLP quantization from int6 to int5, typically saving 1-2MB. A follow-up run with `INT5_MLP=1` is planned to bring the artifact under the cap.

## Approach

Built on PR #198's 11-layer stack with one new technique:

**Backout Connection** — A learned residual subtraction from a mid-network hidden state. After the U-Net encoder-decoder forward pass, the model subtracts `lambda * h_mid` from the final representation, where `lambda` is a learned scalar (initialized at 0.2) and `h_mid` is the hidden state at layer `num_layers // 2`. This removes redundant mid-layer information, sharpening the final representation. Zero additional matrix parameters — only one learned scalar.

Everything else from PR #198 carries forward: 11 layers, 512 dim, 8 heads / 4 KV heads, MLP 3x, relu-squared, SmearGate, BigramHash(4096), OrthoInit, Muon + AdamW with WD=0.04, SWA, int6 mixed quant + zstd, FA3, seq 2048, sliding window eval stride=64.

## Results

| Metric | Baseline (PR #198 config) | + Backout | Delta |
|--------|---------------------------|-----------|-------|
| **val_bpb (sliding, s=64)** | 1.1435 | **1.1364** | **-0.0071** |
| val_loss | 1.9307 | 1.9188 | -0.0119 |
| Steps (600s) | 5246 | 6642 | +1396 |
| Step time | 114ms | 90ms | -24ms |
| Artifact | 17.1 MB (zlib) | 16.2 MB (zstd) | -0.9 MB |

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1544 |
| Int6 roundtrip val_bpb | 1.1588 |
| **Int6 sliding val_bpb (s64)** | **1.1364** |
| Steps completed (600s cap) | 6642 |
| Step time | 90ms |
| Artifact size | 16,170,051 bytes |
| Code size | 70,854 bytes |
| SWA checkpoints averaged | 6 |

## Run command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=4096 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
BACKOUT_ENABLED=1 BACKOUT_LAMBDA_INIT=0.2 \
LAWA_ENABLED=0 INT5_MLP=0 VE_ENABLED=0 \
python3 -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware

8xH100 SXM 80GB HBM3 (RunPod, EUR-IS-3)
