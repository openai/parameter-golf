# Sliding Window + WARMDOWN + AttnRes + PhiAlpha Simple

**Mean TTT LoRA val_bpb: 1.1925** (3 seeds)

## Key Techniques

1. **Always-decaying LR schedule** (`WARMDOWN_ITERS=20000`): LR decays from the first step, producing tighter weight distributions with fewer outliers. Reduces int8 quantization penalty significantly.

2. **Sliding window evaluation** (stride=64, seq_len=1024): Every token scored with 960+ context instead of 0-1023 average. Compiled `forward_logits` method for efficient batch inference.

3. **Block AttnRes**: At block boundaries (every 3 layers), learned attention over all previous block outputs replaces fixed residual aggregation. Adds ~1,024 parameters (2 boundary queries × 512 dim).

4. **PhiAlpha Simple**: Per-layer learnable scale on relu² activation: `relu²(x) * (1 + alpha)`, alpha initialized to 0. Near-zero overhead vs baseline.

## Results

| Seed | val_bpb (sliding window) | TTT LoRA val_bpb | Steps | ms/step |
|------|--------------------------|------------------|-------|---------|
| 1337 | 1.1901 | 1.1932 | 12271 | 48.90 |
| 42   | 1.1902 | 1.1929 | 12273 | 48.89 |
| 7    | 1.1889 | 1.1914 | 12279 | 48.86 |
| **Mean** | **1.1897** | **1.1925** | **12274** | **48.88** |

Artifact: ~14.9 MB | Eval time: ~77s (TTT LoRA) + ~37s (sliding window)

## Reproduction

```bash
WARMDOWN_ITERS=20000 EVAL_STRIDE=64 USE_ATTN_RES=1 USE_PHI_ALPHA_SIMPLE=1 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
