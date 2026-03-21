# 11L XSA + EMA + TTT

**val_bpb = 1.1401** (seed=1337) | artifact = 15.4 MB | 8×H100 SXM | 600s

11-layer transformer with Exclusive Self Attention, EMA weight averaging, Test-Time Training, U-Net skip connections, Partial RoPE, and LN Scale.

## Config

```
NUM_LAYERS=11  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4  MLP_MULT=3
QUANT_BITS=6  USE_ZSTD=1  TIE_EMBEDDINGS=1  FP16_EMBED_EXPORT=1
TRAIN_SEQ_LEN=2048  EVAL_SEQ_LEN=2048  ROPE_BASE=10000
TRAIN_BATCH_TOKENS=524288  WARMDOWN_ITERS=3000
XSA_LAST_N=4  EMA_ENABLED=1  EMA_DECAY=0.997
TTT_ENABLED=1  TTT_LR=0.002  TTT_EPOCHS=3  TTT_FREEZE_BLOCKS=2
SMEAR_GATE=1  BIGRAM_HASH=1  ORTHO_INIT=1  UNET_SKIPS=1
ROPE_DIMS=16  LN_SCALE=1
MUON_MOMENTUM=0.99  GRAD_CLIP_NORM=0.3  MUON_WD=0.04  ADAM_WD=0.04
MATRIX_LR=0.025  SCALAR_LR=0.025  TIED_EMBED_LR=0.035
EVAL_STRIDE=32  DOC_ISOLATED_EVAL=0
QAT=1 (lr_scale < 0.1 trigger, absmax int6 STE)
```

## Techniques

**XSA** (last 4 layers): Subtracts each head's self-value projection from attention output, forcing reliance on context. GQA-aware reshape avoids `repeat_interleave`. Zero parameters.

**EMA** (decay=0.997): Shadow weights in float32, updated every step. Applied before quantization, replacing last-step weights with smoothed average.

**TTT**: 3-epoch full-weight SGD (lr=0.002, momentum=0.9) on validation data after EMA, before final eval. First 2 blocks frozen for stability. ~75s runtime.

**U-Net skip connections**: Encoder (first 5 layers) saves outputs; decoder (last 6 layers) adds learned per-channel weighted skips before each decoder block. Single `nn.Parameter` tensor for torch.compile compatibility.

**Partial RoPE** (16/64 dims): RoPE applied to first 16 of 64 head dimensions only. Remaining 48 dims are position-free, focusing on content representation.

**LN Scale**: RMSNorm output scaled by `1/sqrt(layer_idx + 1)`. Deeper layers receive progressively dampened inputs, stabilizing gradient flow.

**SmearGate**: Per-channel interpolation between current and previous token embeddings: `torch.lerp(x, prev, sigmoid(gate))`. 512 independent learned gates.

**BigramHash**: XOR-based hash of consecutive token pairs into learned embedding table, projected to model dim, scaled by a learned scalar (init 0.05).

**QAT**: Fake int6 quantization with absmax per-row scale and straight-through estimator. Activates when lr_scale drops below 0.1 (~last 300 steps of warmdown).

## Reproduction

```bash
cd /workspace
git clone https://github.com/mrdavtan/parameter-golf.git
cd parameter-golf && git checkout 11l-xsa-ema-ttt
pip install flash-attn --no-cache-dir --no-build-isolation
pip install zstandard sentencepiece huggingface_hub
python3 data/cached_challenge_fineweb.py --variant sp1024

unset MLP_HIDDEN QUANT_BITS RUN_ID SEED TIER2_MODE && \
ROPE_DIMS=16 LN_SCALE=1 ROPE_BASE=10000 \
EVAL_STRIDE=32 DOC_ISOLATED_EVAL=0 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_11L_XSA_EMA_TTT/train_gpt.py
```

Hardware: 8×H100 SXM (RunPod), PyTorch 2.9.1+cu128, Flash Attention 2

## Development notes

See [SESSION_LOG.md](SESSION_LOG.md) for the full progression from 1.1708 to 1.1401, including implementation fixes and revised ablation findings.
