# Depth-Recurrent U-Net Transformer (Universal Transformer Variant)

## Summary

This submission explores **weight-tied depth recurrence** applied to the baseline U-Net transformer architecture. Instead of N independent transformer blocks (each with unique weights), we use **two shared blocks** — one for the encoder half and one for the decoder half — repeated multiple times. This is a form of the Universal Transformer, adapted to preserve the U-Net skip-connection structure of the baseline.

## Motivation

The baseline model uses 9 independent transformer layers at dim=512, consuming ~17M parameters and compressing to ~9.97MB. With weight tying across depth, the per-layer parameter cost collapses to 2 blocks regardless of recurrence depth. This frees the 16MB budget to be spent on **model width** rather than depth diversity — the hypothesis being that a wider recurrent model may learn better representations per parameter than a narrower non-recurrent one.

This idea is listed as a requested contribution in the challenge README under "Universal Transformer."

## Architecture

- **Base architecture:** U-Net transformer (encoder stores skips, decoder pops in reverse)
- **Modification:** Replace `nn.ModuleList` of N unique blocks with:
  - 1 shared `encoder_block` (repeated `num_recurrences // 2` times)
  - 1 shared `decoder_block` (repeated `num_recurrences - num_recurrences // 2` times)
- **Skip weights** remain unique per layer (small parameter cost, important for U-Net structure)
- **Final config:** `num_recurrences=9`, `model_dim=1024`, `num_heads=16`, `num_kv_heads=8`

### Parameter comparison

| Model | Params | Compressed size |
|-------|--------|-----------------|
| Baseline (9L, dim=512) | 17.1M | 9.97MB |
| This submission (recur=9, dim=1024) | 15.7M | 14.43MB |

Despite similar parameter counts, the recurrent model has significantly wider representations (1024 vs 512 dim) at the cost of weight diversity across depth.

## Results

### Ablation — 500-step proxy runs

| Config | val_bpb (raw) | val_bpb (int8+zlib) | Size |
|--------|--------------|---------------------|------|
| recur=9, dim=512 | 1.5585 | 1.5792 | 2.59MB |
| recur=9, dim=1024 | 1.4608 | 1.4899 | 8.21MB |
| recur=12, dim=1280 | 1.4383 | 1.4707 | 11.94MB |
| recur=9, dim=1280 | 1.4360 | 1.4667 | 11.90MB |

### Final run — 20,000 steps

| Metric | Value |
|--------|-------|
| val_bpb (raw) | **1.2707** |
| val_bpb (int8+zlib) | **1.2919** |
| Compressed size | **14.43MB** |
| Parameters | 15.7M |
| Iterations | 20,000 |
| Hardware | 1× A10 GPU |

## Key Observations

1. Weight tying alone (dim=512, recur=9) hurts performance vs baseline — the model needs width to compensate for loss of layer diversity.
2. Scaling width to dim=1024 recovers most of the gap, approaching baseline performance with a fundamentally different architecture.
3. The U-Net skip structure is preserved exactly — skip weights remain unique and are not tied, which appears important for encoder-decoder information flow.
4. Larger dim (1280) with same recurrences gives the best raw val_bpb across ablations, suggesting width matters more than recurrence depth in this regime.

## Limitations & Future Work

- The recurrent model trains slower per step due to larger width (3664ms vs 1452ms for baseline), meaning fewer effective steps in a 10-minute 8xH100 window.
- Gradient flow through repeated shared weights may benefit from auxiliary losses at each recurrence step.
- Learned per-step recurrence embeddings (as in the original Universal Transformer paper) were not explored and could improve results.
- This is a non-record submission run on a single A10 GPU, not the 8xH100 leaderboard configuration.

## How to Run

```bash
NUM_RECURRENCES=9 MODEL_DIM=1024 NUM_HEADS=16 NUM_KV_HEADS=8 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Author

- **Name:** Muhammad Ahmed Rayyan
- **GitHub:** [@Muhammad-Ahmed-Rayyan](https://github.com/Muhammad-Ahmed-Rayyan)