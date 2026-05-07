# Discrete Masked Diffusion Language Model (MDLM)

**Mean val_bpb: 1.3600** (3 seeds: 1337, 42, 7)

## Results

| Seed | val_bpb | Steps | ms/step |
|------|---------|-------|---------|
| 1337 | 1.3606 | 7344 | 81.71 |
| 42   | 1.3772 | 7189 | 83.47 |
| 7    | 1.3423 | ~7300 | ~83 |
| **Mean** | **1.3600** | **~7277** | **~83** |

Artifact: ~12.9MB | Eval: pseudo-log-likelihood (8 masked passes)

## Key Idea

True discrete diffusion language model. Unlike autoregressive GPT, the model:
- Trains with **bidirectional attention** — each [MASK] token sees all other tokens
- Evaluates with **pseudo-log-likelihood** — each token scored with full bilateral context
- Uses a learnable [MASK] embedding as the diffusion noise signal

## Techniques

1. **Bidirectional attention** (`USE_BIDIRECTIONAL_TRAIN=1`): full attention during training, each masked position attends to all tokens including future ones. True diffusion signal.

2. **Masked token prediction** (`USE_MASK_LOSS_ONLY=1`): CE loss only on masked positions, rate sampled from Uniform[0.15, 0.85] per sequence. Forces robust representations.

3. **Pseudo-log-likelihood eval** (`USE_MASKED_EVAL=1`): 8 forward passes per sequence, each masking 50% of tokens randomly. Each token predicted with bilateral context. Unbiased estimator of per-token CE under full context.

4. **Always-decaying LR** (`WARMDOWN_ITERS=20000`): LR decays from step 0, better int8 quantization.

## Reproduction

```bash
export USE_BIDIRECTIONAL_TRAIN=1 USE_MASKED_EVAL=1 USE_MASK_LOSS_ONLY=1 MIX_GPT_PROB=0.0 USE_TTT_EVAL=0 WARMDOWN_ITERS=20000 MAX_WALLCLOCK_SECONDS=600 SEED=1337
torchrun --standalone --nproc_per_node=8 train_diffusion.py
```

## Architecture

```
Input: token IDs with some replaced by [MASK] (id = vocab_size)
Embedding: vocab_size+1 × 512 (real tokens + learnable MASK embedding)
9 × Block: bidirectional attention (GQA 8/4 heads, RoPE) + MLP relu²
Output: logits over real vocab only (weight-tied)
```

## Why Diffusion vs GPT

| Aspect | GPT | MDLM |
|--------|-----|------|
| Attention | Causal | Bidirectional |
| Training signal | Next-token | Masked token prediction |
| Eval context | Left only | Bilateral (left + right) |
| Eval method | Next-token CE | Pseudo-log-likelihood |
| Artifact | ~14.9MB | ~12.9MB |
