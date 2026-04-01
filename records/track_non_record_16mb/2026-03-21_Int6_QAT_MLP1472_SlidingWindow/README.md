# Int6 QAT MLP1472 SlidingWindow + TTT

**val_bpb: 1.1807** (with TTT) | Post-quant: 1.1991 | Artifact: 15.78MB

## Approach

1. **Int6 QAT:** STE fake-quantize in last 20% of training. Quant gap: 0.0017.
2. **MLP hidden=1472:** 21.2M params in 15.78MB via int6+zstd.
3. **Warmdown 20K iters + FP16 tied embeddings.**
4. **Sliding window eval (stride=64):** 62s on 8xH100.
5. **TTT:** Full-weight SGD (lr=0.002, momentum=0.9), 3 epochs on val data, freeze first 2 blocks. -0.018 BPB. 113s.

## Results

| Stage | val_bpb |
|-------|---------|
| Pre-quant | 1.1974 |
| Post-quant (int6+QAT) | 1.1991 |
| **After TTT** | **1.1807** |

## Config

9L, 512d, 8/4 heads (GQA), MLP 1472, 1024 vocab. 11,937 steps at 50.3ms on 8xH100.

## Reproduction

```
WARMDOWN_ITERS=20000 MLP_HIDDEN=1472 QUANT_BITS=6 QAT_START_FRAC=0.8 \
EVAL_STRIDE=64 MUON_WD=0 ADAM_WD=0 EMA_DECAY=0 SMEARGATE=0 BIGRAM_HASH_BUCKETS=0 \
TTT_EPOCHS=3 TTT_LR=0.002 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
