# Random Linear Maps + Learned LoRA Adapters

## Summary

This submission implements the **"Learning adapters on random linear maps"** idea from the challenge wishlist — a previously unclaimed approach that inverts the standard train→compress paradigm.

**Core idea**: Instead of training all weights and then compressing to fit in 16MB, we:

1. **Freeze most weights as pseudo-random projections** initialized from a deterministic seed (stored in code = 0 bytes in artifact).
2. **Only train small LoRA-style low-rank adapters** (rank 16) on each layer, plus embeddings, norms, and control parameters.
3. **At save time**, serialize only the trained adapter weights + seed.
4. **At load time**, regenerate the full random backbone from the seed and apply the trained adapters.

## Why This Is Interesting

- **Massive model for free**: The frozen random backbone (12 layers, 768 dim, 3x MLP) has ~70M+ parameters but costs **0 bytes** in the artifact since it's reproducible from a seed.
- **Only ~5-10M trainable params**: These fit easily in 16MB even at FP16, leaving headroom for wider/deeper architectures.
- **Theoretically motivated**: Random features are surprisingly powerful (Random Kitchen Sinks, Lottery Ticket Hypothesis). The frozen random projections provide a rich feature basis that the adapters learn to combine.
- **Novel for this challenge**: Nobody has tried this approach — it's fundamentally different from the quantization-focused submissions on the leaderboard.

## Architecture

| Component            | Value                |
| -------------------- | -------------------- |
| Layers               | 12                   |
| Model dim            | 768                  |
| Heads                | 12 (4 KV heads, GQA) |
| MLP mult             | 3x                   |
| LoRA rank            | 16                   |
| Vocab                | 1024 (sp1024)        |
| Backbone seed        | 42                   |
| Trainable params     | ~5-10M               |
| Frozen random params | ~70M+                |

## Key Components

### LoRALinear Module

Each linear layer has:

- A **frozen random base weight** `W` (from deterministic seed, stored as buffer)
- **Trainable low-rank adapters** `A` (rank×in) and `B` (out×rank)
- Output: `W@x + (B@A)@x * scale`
- `B` initialized to zero so initial behavior = pure random projection

### Optimizer Split

- **Muon**: LoRA adapter matrices (lora_A, lora_B)
- **Adam**: Token embeddings, scalar/control parameters, norms

### Serialization

- Only trainable parameters are saved (not frozen buffers)
- Int8 + zlib compression on the trainable subset
- At load time: regenerate random backbone from seed, apply dequantized adapters

## Running

```bash
cd /workspace/parameter-golf

python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

RUN_ID=random_lora_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 records/track_10min_16mb/2026-04-03_RandomLinearMaps_LoRA_Adapters/train_gpt.py
```

## Potential Improvements

- **Selective unfreezing**: Unfreeze first/last layer base weights for better embedding-to-hidden and hidden-to-logit projections.
- **Larger LoRA rank** on critical layers (attention Q/K vs MLP).
- **Different random initialization** schemes (orthogonal, spectral norm matching).
- **Hybrid**: Freeze only MLP base weights (largest), train attention fully.
- **Combine with proven techniques**: BigramHash, sliding eval, EMA/SWA.

## Theoretical Background

- [Random Features for Large-Scale Kernel Machines](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) (Rahimi & Recht, 2007)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) (Frankle & Carlin, 2018)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) (Aghajanyan et al., 2020)
