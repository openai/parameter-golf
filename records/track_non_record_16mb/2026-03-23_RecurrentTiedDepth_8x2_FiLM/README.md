# Recurrent Tied-Depth 8×2 + FiLM + TrigramHash

**val_bpb: 1.1634** (seed=42, 8xH100, 10-min wallclock, int6+zstd, 15.34MB artifact)

## Core Idea

Replace 10 unique transformer blocks with **8 unique blocks looped 2 times** (16 effective layers), conditioned by **FiLM scale/shift per iteration**. Augment with **BigramHash + TrigramHash** lexical sidecars for richer local context.

This explores the **L(N) optimization frontier** from a different angle: reuse fewer parameters more times, and spend the freed budget on lexical memory instead of more unique weights.

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Script defaults are pinned to the submission candidate:
- `NUM_UNIQUE_BLOCKS=8`, `NUM_LOOPS=2`
- `BIGRAM_VOCAB_SIZE=20480`, `TRIGRAM_VOCAB_SIZE=8192`
- `MODEL_DIM=512`, LeakyReLU(0.5)² activation

## Architecture

- 8 unique transformer blocks × 2 loops = 16 effective layers
- FiLM conditioning: learned scale+shift per loop iteration (3,072 params)
- U-Net skip connections: collect during loop 0, inject during loop 1
- **BigramHash(20480)** + **TrigramHash(8192)**: hashed 2- and 3-token context features
- LeakyReLU(0.5)² MLP activation
- 512 dim, 8 heads, 4 KV heads (GQA), 3× MLP, tied embeddings
- int6 QAT + zstd-22 compression, SWA (24 checkpoints)

## Results

### Recurrence Scaling (8xH100, 10 shards, seed=42)

| Config | val_bpb | Artifact | Steps/10min | Eff. Depth |
|--------|---------|----------|-------------|------------|
| 3×3 | 1.2469 | 5.9MB | 7405 | 9 |
| 5×2 | 1.2009 | 9.0MB | 6770 | 10 |
| 7×2 | 1.1829 | 11.5MB | 4848 | 14 |
| 8×2 | 1.1752 | 13.2MB | 4268 | 16 |

### Improvement Stack (on 8×2 base)

| Addition | val_bpb | Delta |
|----------|---------|-------|
| 8×2 relu² baseline | 1.1752 | — |
| + LeakyReLU(0.5)² | 1.1723 | −0.003 |
| + BigramHash 20480 | 1.1714 | −0.001 |
| **+ TrigramHash 8192** | **1.1634** | **−0.008** |

### Negative Results

| Attempt | Result | Why |
|---------|--------|-----|
| EMA(0.997) | 1.42 (catastrophic) | Full-run averaging bad for short training |
| Legal TTT | 1.34 (catastrophic) | Recurrence amplifies SGD updates |
| Width d544 | 1.19 (worse) | Slower steps → fewer iterations |
| GPTQ-lite | 1.18 (worse) | Percentile clipping didn't help |
| Trigram 12288 | 1.17 (worse) | Too sparse to learn in available steps |

## Findings

1. **Recurrence is viable** — stable training, competitive BPB at much smaller artifact
2. **Unique capacity > loop depth** — 5×2 beat 4×3 despite fewer total block applications
3. **Trigram hashing is the strongest lexical lever** — −0.008 BPB, 9.4× better ROI/MB than bigram scaling
4. **TTT fails with recurrence** — weight tying amplifies SGD updates catastrophically
5. **EMA fails with short training** — full-run averaging includes early poorly-converged states
6. **Sweet spot exists for hash table size** — trigram 8192 > 12288 (sparser tables harder to learn)

## What This Means

Recurrence + n-gram sidecars is a distinct approach from the converging TTT/EMA/XSA leaderboard meta. The model achieves competitive BPB without test-time training, using only 15.34MB of the 16MB budget. The trigram hash finding (positive result at 8xH100 scale, contrasting PR #571's negative 1xH100 result) suggests lexical memory interacts favorably with the higher-step recurrent regime.
