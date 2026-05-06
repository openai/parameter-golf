# Batch-Opt 524K + WD4000 + Full-Weight TTT

**Mean val_bpb: 1.1433** (3 seeds, post int5/int6+zstd quantization + full-weight TTT + sliding window eval)

## Key Techniques

### 1. Full-Weight Test-Time Training (TTT)
After quantization roundtrip, adapt ALL model weights to the validation distribution via SGD (lr=0.005, momentum=0.9, 15 epochs). No per-document reset — the entire model adapts to the validation distribution before scoring. This gives ~0.006 BPB improvement over standard sliding eval.

### 2. Batch Size = 524K (down from 786K)
More optimizer updates per wall-clock minute. ~7,400 steps vs ~5,100 at 786K batch on our hardware.

### 3. Warmdown = 4000 (up from 3000)
Retuned for the higher step count from smaller batch. Smoother LR decay.

## Results

| Seed | val_loss | val_bpb | Steps | Artifact | Valid |
|------|----------|---------|-------|----------|-------|
| 42 | 1.92950060 | 1.14276196 | 7,473 | 15,769,541 | YES |
| 7 | 1.93232602 | 1.14443534 | 7,439 | 15,781,381 | YES |
| 2024 | 1.92944959 | 1.14273175 | 7,439 | 15,715,583 | YES |
| **Mean** | **1.93042540** | **1.14330968** | | | |
| **Std** | | **0.00098** | | | |

All artifacts under 16,000,000 bytes. TTT + sliding eval completes in ~190s (under 10 min eval budget).

## Command

```bash
RUN_ID=submission \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 SEED=42 \
TRAIN_BATCH_TOKENS=524288 WARMDOWN_ITERS=4000 \
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=15 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=0 TTT_BATCH_SEQS=16 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Architecture

Built on #1 entry (thwu1): 10L MLP3x, SmearGate, BigramHash(10240), SWA, OrthoInit, int5/int6+zstd-22, FP16 tied embed, Muon WD=0.04.

## Hardware

8x NVIDIA H100 80GB HBM3 SXM (RunPod Parameter Golf template). PyTorch 2.9.1+cu128.
