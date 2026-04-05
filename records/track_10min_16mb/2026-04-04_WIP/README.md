# Record: WIP (SOTA Stack + Depth Recurrence)

**val_bpb: TBD** | **~XX MB** | 8xH100 SXM, 600s

Based on the current SOTA ([2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072](../2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/), 1.1147 BPB), all techniques ported into the clean 1126-line baseline and extended with depth recurrence.

## Techniques Included

All proven techniques from the SOTA lineage, implemented from scratch in the baseline framework:

| Technique | Source | Est. BPB Gain |
|-----------|--------|---------------|
| 11 layers, 3x MLP (1536 hidden) | #493 @parinzee | ~0.02 |
| LeakyReLU(0.5)^2 activation | #493 @parinzee | ~0.003 |
| Partial RoPE (16/64 dims) | #315 @jfprincz | ~0.002 |
| LN Scale 1/sqrt(layer+1) | #315 @jfprincz | ~0.001 |
| XSA on all 11 layers | #478 @gowtham0992 | ~0.002 |
| SmearGate | #65 @aquariouseworkman | ~0.001 |
| BigramHash 3072x112 | #162 @raahilshah | ~0.005-0.01 |
| EMA weight averaging (0.997) | #401 @newjordan | ~0.002 |
| Late QAT (STE at LR < 0.15) | #286 @chris-buckley | ~0.001-0.002 |
| Int6 GPTQ-lite + LZMA-9 | #535 @raahilshah, #160 @ChaseWNorton | ~0.01-0.015 |
| Sliding window eval (stride=64) | Multiple PRs | ~0.015-0.02 |
| Muon weight decay 0.04 | #364 @shikhar1729 | ~0.002 |
| Warmdown 3500 iters | #364 @shikhar1729 | ~0.001 |
| Gradient clipping 0.3 | Standard | ~0.001 |

### Novel: Depth Recurrence (configurable)

Run the 11 physical layers N times (default: off, set `DEPTH_RECURRENCE=2` to enable). This gives 22 effective layers at zero parameter cost. Each recurrence pass gets fresh U-Net skip connections.

## Results

| Seed | Steps | ms/step | **Sliding BPB** | Artifact |
|------|-------|---------|-----------------|----------|
| 42   |       |         |                 |          |
| 314  |       |         |                 |          |
| 999  |       |         |                 |          |
| **Mean** | | | | |

## Requirements

No additional packages beyond the baseline (`torch`, `numpy`, `sentencepiece`). `lzma` is Python stdlib. Evaluation environment has everything pre-installed.

## Run Command

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=3500 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

With depth recurrence (experimental):

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=3500 \
DEPTH_RECURRENCE=2 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

```
Baseline (1.2244 BPB)
    -> Phase 1: 11L, 3xMLP, LeakyReLU^2, partial RoPE, LN scale, WD, warmdown
    -> Phase 2: Sliding eval, EMA, SmearGate, XSA, BigramHash, Late QAT, int6+LZMA
    -> Phase 3: Depth recurrence (novel)
```
