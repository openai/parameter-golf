# Complementary Training + Backoff N-gram Mixer (Reproduction)

**val_bpb: 0.4377** (2-seed mean 0.4379, std 0.0002) | 8x L20Z (H100) | eval 450s

## Results

| Seed | Steps | val_bpb | eval_time |
|------|-------|---------|-----------|
| 1337 | 7,003 | **0.4377** | 450s |
| 42 | 7,011 | **0.4380** | 450s |

## Approach

Reproduction of PR #803 (pentxayc) on 8x NVIDIA L20Z GPUs with stride=128 optimization.

### Key Techniques

1. **Complementary Training** (COMPLEMENT_ALPHA=0.5): Downweights training loss on tokens that bigram statistics can predict, forcing the neural model to specialize on hard tokens (long-range dependencies, semantic surprises).

2. **BackoffNgramMixer**: Orders 2-10, 4M flat hash buckets. At eval time, entropy-adaptive alpha mixing: `alpha = 0.20 + 0.55 * sigmoid(2 * (entropy - 3.0))`. High-entropy tokens get more n-gram weight.

3. **Legal Score-First TTT**: AdamW (lr=5e-4), 4 epochs per chunk, freeze first 2 blocks, Polyak EMA 0.998. Every token scored BEFORE any update uses it.

4. **Stride=128**: Reduces eval windows from ~30K to ~950, with negligible BPB impact vs stride=32.

### Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads, 3x MLP with LeakyReLU(0.5)^2
- XSA on last 4 layers, VRL enabled
- Int6 mixed quantization + lzma compression
- Artifact: ~15.9MB (under 16MB limit)

## Reproduction

```bash
VRL_ENABLED=1 LEAKY_RELU=1 GATED_ATTENTION=0 \
TTT_ENABLED=1 TTT_OPTIMIZER=adamw TTT_LR=0.0005 TTT_EPOCHS=4 \
TTT_FREEZE_BLOCKS=2 TTT_TEMPERATURE=0.98 \
USE_HEDGE_MIXER=1 NGRAM_ORDER=10 NGRAM_BUCKETS=4194304 \
ALPHA_BASE=0.20 ALPHA_RANGE=0.55 ALPHA_CENTER=3.0 \
COMPLEMENT_ALPHA=0.5 EVAL_STRIDE=128 \
SEED=1337 torchrun --nproc_per_node=8 train_gpt.py
```

## Acknowledgment

Based on PR #803 by pentxayc. The core innovation of complementary training (bigram-weighted loss reweighting) is their contribution.
