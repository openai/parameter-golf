# Record: Corrupted Context Training (R2-11) + R1-5 Architecture Stack

**val_bpb: TBD** (3-seed mean) | **~TBD MB** | 8xH100 SXM, 600s | No TTT

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **int8+zlib BPB** | Artifact |
|------|-------|---------|---------------|-------------------|----------|
| 42 | TBD | TBD | TBD | **TBD** | TBD |
| 137 | TBD | TBD | TBD | **TBD** | TBD |
| 256 | TBD | TBD | TBD | **TBD** | TBD |
| **Mean** | | | | **TBD** | |

---

## Main Changes

### 1. Corrupted Context Training (R2-11)

During training, 10% of input tokens are replaced with uniformly random tokens (first position never corrupted). This bridges the train/inference exposure bias gap: the model learns to make predictions even when some context tokens are noisy, improving robustness and generalization.

This technique is orthogonal to standard dropout (which operates on feature dimensions). Corrupted context operates on the sequence dimension, forcing the model to build redundant causal paths rather than over-relying on any single context position.

**Screening result**: On 1-shard screening, corrupted context training (val_bpb=1.3004) beat both reproduced SOTA submissions (1.3315, 1.3471) by a significant margin.

### 2. R1-5 Architecture Stack

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 3x (1536) with LeakyReLU(0.5)^2 | [#493](https://github.com/openai/parameter-golf/pull/493) |
| Attention | XSA on last 4 layers | [#198](https://github.com/openai/parameter-golf/pull/198) |
| BigramHash | 3072 x dim=128 | [#162](https://github.com/openai/parameter-golf/pull/162), [#1019](https://github.com/openai/parameter-golf/pull/1019) |
| U-Net skips | Encoder-decoder connections | [#289](https://github.com/openai/parameter-golf/pull/289) |
| Value residual | v + sigmoid(alpha) * v0 | [#549](https://github.com/openai/parameter-golf/pull/549) |
| Quantization | int8 per-row + zlib level 9 | Baseline |

### 3. Muon Optimizer

Newton-Schulz orthogonalization of gradient updates for matrix parameters, with separate Adam optimizers for embeddings and scalars. Momentum warmup from 0.85 to 0.95 over 500 steps.

## Run Command

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All R1-5 features and corrupted context are enabled by default. Override with env vars if needed:
```bash
CORRUPT_RATE=0.1 NUM_LAYERS=11 MLP_MULT=3 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 VALUE_RESIDUAL=1
```

## Lineage

```
Baseline (9L/2x, relu^2)
    +-- R1-1: LeakyReLU(0.5)^2 + 11L + 3x MLP
        +-- R1-2: BigramHash 3072
            +-- R1-3: XSA last 4 layers
                +-- R1-5: Value residual (R1-4 U-Net skips always on)
                    +-- R2-11: Corrupted context training (10%) <-- this submission
```
