Int6 mixed quantization with STE fake-int6 QAT, 3x MLP expansion, NorMuon optimizer, SWA checkpoint averaging, and sliding window eval.

## what changed

**MLP 3x expansion (hidden=1536)**: 21.8M params. The extra capacity is paid for by int6 quantization.

**STE fake-int6 QAT**: weights are fake-quantized to int6 and dequantized via straight-through estimator throughout training. The model learns weight distributions that survive 6-bit export, reducing quantization penalty from ~0.008 to ~0.001 BPB.

**NorMuon optimizer**: per-neuron row-wise RMS normalization after Newton-Schulz orthogonalization. Stabilizes updates across neurons with different activation scales.

**SWA checkpoint averaging**: collects model checkpoints every 200 steps during warmdown and uniformly averages them. Finds flatter minima than EMA.

**Mixed quantization**: int6 per-row on MLP and attention weights, fp16 passthrough for the tied embedding, zstd-22 compression.

**Sliding window eval (stride=64)**: each token scored with nearly full context.

**seq_len=2048**, **batch=786K**, **grad_clip=0.3**, **matrix_lr=0.02**, **Muon momentum=0.99** (warmup from 0.92 over 1500 steps), **Muon weight decay=0.01**, **warmdown=3000 iters**, **logit softcap=15**.

## config

```
VOCAB_SIZE=1024  NUM_LAYERS=9  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_MULT=3  TIE_EMBEDDINGS=1  TRAIN_SEQ_LEN=2048  TRAIN_BATCH_TOKENS=786432
EMBED_LR=0.03  MATRIX_LR=0.02  SCALAR_LR=0.02  TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99  MUON_MOMENTUM_WARMUP_START=0.92  MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_WEIGHT_DECAY=0.01  WARMDOWN_ITERS=3000  LOGIT_SOFTCAP=15
EVAL_STRIDE=64  GRAD_CLIP_NORM=0.3  ENABLE_QAT=1  EMA_DECAY=0.998
```

## run command

```bash
pip install zstandard
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## results

8xH100 80GB HBM3 (Modal, 10 min wallclock, seed 1337):

| metric | val_loss | val_bpb | artifact |
|--------|----------|---------|----------|
| pre-quant (raw) | 2.007 | 1.1887 | — |
| post-quant (standard) | 2.0055 | 1.1877 | 15.22 MB |
| **post-quant (sliding window, stride=64)** | **1.9697** | **1.1666** | 15.22 MB |

6,065 steps at 98.9ms/step. Sliding window eval: 156s (under 10 min eval budget).

## quantization details

- `.mlp.` and `.attn.` 2D weights: int6 per-row with STE QAT (54 tensors)
- `tok_emb.weight`: fp16 passthrough
- Small/control tensors: fp16/fp32 passthrough (38 tensors)
- Compression: zstd level 22
- Quant penalty: 0.001 BPB

## files

- `train_gpt.py` — training script
- `train.log` — full 8xH100 run log (seed 1337)
- `submission.json`
