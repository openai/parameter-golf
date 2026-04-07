# Non-record: TurboQuant + N-gram Hybrid Eval + TTT (1xH100 NVL)

**val_bpb: 0.3509** | **14.92 MB** | 586 steps, 601s train + 521s eval | 1xH100 NVL

This submission uses TurboQuant random-rotation quantization, entropy-adaptive n-gram backoff cache, complementary training, and LoRA test-time training with tuned temperature and Polyak averaging. All experiments ran on a single H100 NVL with a 10-minute wallclock cap.

The final score (0.3509 val_bpb) is not directly comparable to 8xH100 leaderboard entries due to fewer training steps (~586 vs ~6900). The pre-quant val_bpb is 1.3167.

## Approach

1. **TurboQuant int5** — Random rotation + Lloyd-Max scalar quantization. Each weight row is normalized, random sign-flipped, permuted, then uniformly quantized. Preserves weight structure better than greedy quantization when Hessians are noisy from limited training steps.

2. **TTT temperature 1.1** — Raising test-time training temperature from 0.98 to 1.1 increases model entropy during TTT eval, shifting entropy-adaptive alpha upward so the n-gram cache gets more influence where it's most reliable.

3. **Hyperparameter tuning** — Independently validated improvements that stack linearly:
   - `NGRAM_MIN_COUNT` 1
   - `POLYAK_DECAY` 0.995
   - `DECODER_LR_MULT` 3.0
   - `TTT_LR` 0.005

4. **Complementary training** (alpha=0.5) — Downweights bigram-predictable tokens during training, focusing model capacity on harder predictions.

5. **Order-9 n-gram backoff cache** — Hash-based n-gram statistics collected during eval, mixed with neural probabilities via entropy-adaptive alpha blending with per-order centers and multipliers.

6. **LoRA TTT** — Rank-8 test-time training on Q,V of last 2 blocks with Polyak averaging.

## Results

| Experiment | N-gram BPB | Model BPB | Notes |
|-----------|-----------|-----------|-------|
| **tq5_temp11_fullstack** | **0.3509** | 1.485 | TurboQuant + temp1.1 + all tuning |
| tq5_fullstack | 0.3572 | 1.487 | TurboQuant + all tuning (temp=0.98) |
| gptq_fullstack | 0.3653 | 1.548 | Standard GPTQ, same hyperparams |
| mincount1_only | 0.3737 | 1.579 | Only mincount=1 changed |
| default_config | 0.3864 | 1.560 | Default configuration on 1xH100 |

**Seed variance** (default config): std ~0.0003. The improvement of -0.0355 is ~118x the seed noise.

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 9 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 3x (1536) |
| Attention | XSA on last 4 layers |
| BigramHash | 1536 |
| RoPE | Partial (16/64 dims) |
| Value Residual | Layers 1-8 |
| SmearGate | Position-mixing gate |
| Gated Attention | All 9 layers |
| Weight Averaging | EMA (0.997) |
| N-gram Cache | Order-9 backoff, entropy-adaptive alpha |
| Complementary Training | alpha=0.5 |
| LoRA TTT | Rank-8, Q/V, last 2 blocks, Polyak 0.995 |
| Quantization | TurboQuant int5 + LZMA |
| Optimizer | Parallel Muon (Newton-Schulz) |
| Late QAT | STE at wallclock 0.85 |

**Model parameters:** 27,301,064 | **Artifact:** 14,922,913 bytes (14.92 MB)

## Run Command

```bash
QUANT_METHOD=turboquant TURBO_BITS=5 SLOT_ENABLED=0 \
NGRAM_ENABLED=1 NGRAM_EVAL_ORDER=9 NGRAM_EVAL_MIN_COUNT=1 \
TTT_POLYAK_DECAY=0.995 TTT_LR=0.005 TTT_TEMPERATURE=1.1 \
DECODER_LR_MULT=3.0 MATRIX_LR=0.025 \
COMPLEMENTARY_ALPHA=0.5 EVAL_STRIDE=384 \
RUN_ID=tq5_temp11 SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

SDPA attention is used (FA3 not available on H100 NVL). For H100 SXM with FA3, revert the SDPA patch in the script.

## Negative Results

| Technique | BPB | Why rejected |
|-----------|-----|-------------|
| SLOT + n-gram | 0.461 | Reduces eval chunk coverage 29%, destroys n-gram stats |
| Dirichlet smoothing (c=5.0) | 0.429 | Entropy-adaptive alpha mixing is strictly better |
| 15-gram order | 0.428 | Hash collisions at 96% load factor negate higher-order benefit |
| dim=640 (42M params) | 0.459 | Fewer steps hurts more than bigger model helps |
| 13 layers | 0.422 | Same tradeoff |
| TTT 2 epochs | 0.429 | Overfitting |

## Hardware Note

All experiments ran on a single H100 NVL with a 10-minute wallclock cap. The throughput gap vs 8xH100 (~12x fewer steps) explains the score gap vs leaderboard entries. The techniques documented here are hardware-agnostic.

## Included Files

- `train_gpt.py` — training script (best configuration)
- `train_log.txt` — full training + eval log for best run
- `submission.json` — metadata
