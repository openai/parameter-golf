# 5-expert Hedge Mixer + TTT

**val_bpb: 1.0745** (3-seed mean) | **<15.5 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM)

| Seed | steps | step_avg | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | Eval time | Artifact |
|------|-------|----------|-------------|-----------------|----------|-----------|----------|
| 1337 | 5,997 | 97.1ms | 1.1248 | **1.0560** | -0.0688 | 563s | 15.48 MB |
| 42 | 5,997 | 97.1ms | 1.1257 | **1.0970** | -0.0287 | 563s | 15.41 MB |
| 7 | 5,983 | 97.3ms | 1.1251 | **1.0704** | -0.0547 | 561s | 15.43 MB |
| **Mean** | | | **1.1252** | **1.0745** | **-0.0507** | | |

## Key Contribution: 5-expert Logistic Context Mixer

GPU-vectorized online context mixing using the Hedge/multiplicative-weights algorithm. Five experts blend predictions in log-probability space:

| Expert | Source | Description |
|--------|--------|-------------|
| 0 | Neural | Base model log-softmax |
| 1 | Unigram | Token frequency from scored tokens |
| 2 | Bigram | P(next \| prev) from scored tokens |
| 3 | Trigram | Hashed P(next \| prev2, prev1) with 64K buckets |
| 4 | Entropy | Neural model entropy as confidence regularizer |

Expert weights are updated online via Hedge: `log_w -= eta * loss`. N-gram tables are built incrementally from already-scored tokens only (legal).

## Architecture

PR #606 base with the following additions:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 8KV) |
| MLP | 3x with **LeakyReLU(0.5)^2** |
| BigramHash | 6144 (dim=128) |
| XSA | All 11 layers (ws=8) |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) |
| Quantization | Full GPTQ int5 + zstd (level 22) |
| Pruning | 3% magnitude |

## Legal Score-First TTT

Backward-looking adaptation with GPTQ-calibrated model:

1. Validation tokens split into 474 chunks of 131K tokens each
2. For each chunk:
   - **SCORE**: Sliding window eval (stride=32, seq_len=2048) with 5-expert mixer blending
   - **TRAIN**: AdamW(lr=0.0001) on already-scored chunk. 3 epochs, last 2 blocks unfrozen + norms + lm_head, cosine LR decay, Polyak averaging
3. Last chunk scored but never trained on

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 131,072 tokens |
| Optimizer | AdamW (lr=0.0001) |
| Epochs per chunk | 3 |
| Frozen blocks | First 9 (last 2 + norms + head unfrozen) |
| Polyak decay | 0.998 |
| Adaptive LR | max_mult=3.0 |
| Mixer eta | 0.1 |

### Training Budget

GPTQ calibration runs within the 600s training budget (18s reserved from training loop for EMA selection + calibration + quantization).

| Phase | Time |
|-------|------|
| Training loop | 582s |
| EMA + GPTQ calibration + quantization | ~18s |
| **Total training** | **~600s** |
| Sliding window eval | ~165s |
| TTT eval with mixer | ~562s |
| **Total eval** | **~562s** |

## Reproduction

```bash
# Install dependencies
pip install -r requirements.txt
# Build FA3 Hopper kernels (required)
cd /tmp && git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper && python setup.py install

# Run training + eval (single seed)
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
USE_MIXER=1 TTT_LR=0.0001 TTT_CHUNK_TOKENS=131072 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Run all 3 seeds
for SEED in 1337 42 7; do
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  SEED=$SEED MAX_WALLCLOCK_SECONDS=600 \
  USE_MIXER=1 TTT_LR=0.0001 TTT_CHUNK_TOKENS=131072 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Credits

- **Base model**: PR #606 by @gowtham0992
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **Mixer inspiration**: PAQ compression (context mixing) + Hedge algorithm
