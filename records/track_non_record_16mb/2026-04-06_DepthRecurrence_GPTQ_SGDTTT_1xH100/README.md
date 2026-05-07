# Non-Record: 7-Block Depth Recurrence + GPTQ + SGD TTT (val_bpb=1.1172)

**Author:** swapp1990
**Date:** 2026-04-06
**GPU:** 1xH100 PCIe (RunPod)
**Artifact size:** 15,380,366 bytes (15.38 MB)

## Architecture

- 7 unique transformer blocks, dim=624, 8 heads, 4 KV heads
- Depth recurrence: layer schedule `[0,1,2,3,4,3,4,3,4,5,6,5,6]` = 13 effective layers from 7 unique blocks
- XSA (cross-sample attention) on all 7 blocks
- SwiGLU 3x MLP with SmearGate
- U-Net skip connections
- Tied embeddings, vocab=1024, seq_len=2048
- ~26.5M parameters

## Training

- Muon optimizer (matrix_lr=0.04, scalar_lr=0.04)
- 524K tokens/step, ~7926 steps
- EMA during warmdown phase
- ~80 minutes on 1xH100 PCIe

## Quantization Pipeline

1. **GPTQ** (Generalized Post-Training Quantization): Column-wise Hessian-compensated rounding using 64 calibration sequences from training data. Minimizes output reconstruction error rather than weight error.
2. **Mixed bit-width**: int5 for MLP weights (31 levels), int6 for attention weights (63 levels), int8 for embeddings (255 levels)
3. **zstd-22 compression**: Final artifact compressed with zstandard level 22

GPTQ improved val_bpb by -0.0086 over naive rounding (1.1258 -> 1.1172) and stacks with TTT because it improves the quantized weight values themselves.

## Test-Time Training (TTT)

SGD all-weights adaptation per document at eval time:
- For each document: train ALL 26.5M weights with SGD (lr=0.005, momentum=0.9) on 2048-token chunks
- Score first 2048 tokens after adaptation
- Restore base weights before next document
- Short documents (<32 tokens) scored without adaptation

## Progression (46 experiments)

| Milestone | val_bpb | Key Change |
|-----------|---------|------------|
| Baseline (exp1) | 1.2987 | 9-block, int8 quant, no TTT |
| + LoRA TTT (exp13) | 1.1573 | Score-then-train LoRA on Q/V |
| + Depth recurrence (exp36) | 1.1290 | 9 unique -> shared blocks, SGD all-weights TTT |
| + Architecture D (exp40) | 1.1258 | 7 blocks dim=624, 3x recurrence |
| + GPTQ (exp43) | **1.1172** | Hessian-compensated quantization |

## Technical Report

See [technical_report.md](technical_report.md) for a deep analysis of GPTQ: how Hessian-compensated quantization works, why it stacks with TTT, per-layer error analysis, and comparison with naive rounding.

## Key Findings

- **GPTQ stacks with TTT** unlike QAT, because it improves quantized values directly rather than modifying the training landscape
- **Depth recurrence** (sharing middle blocks) is more parameter-efficient than wider or deeper unique blocks
- **SGD all-weights TTT** (-0.028 BPB) vastly outperforms LoRA TTT (-0.016 BPB)
- **Dead ends**: sliding window scoring, persistent TTT, meta-learning (Reptile/MAML), n-gram cache, logit ensemble -- all add redundant signal that SGD TTT already captures

## Reproducing

```bash
# Train (1xH100, ~80 min)
RUN_ID=depth_recurrence_gptq \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=7 MODEL_DIM=624 MLP_MULT=3 \
LAYER_SCHEDULE=0,1,2,3,4,3,4,3,4,5,6,5,6 \
XSA_LAST_N=7 TRAIN_BATCH_TOKENS=524288 \
GPTQ_ENABLED=1 TTT_ENABLED=1 \
SGD_LR=0.005 SGD_MOMENTUM=0.9 TTT_TRAIN_CHUNK=2048 SCORE_CAP=2048 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
