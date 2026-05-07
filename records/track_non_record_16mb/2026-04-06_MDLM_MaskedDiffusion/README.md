# MDLM: Masked Diffusion Language Model

**val_bpb: 1.3485** (int8+zlib roundtrip) | **15.63 MB** | 8×H100 SXM, 600s

## What is this?

A non-record submission converting the baseline autoregressive GPT into a **Masked Diffusion Language Model** (MDLM), based on Sahoo et al. 2024 ("Simplified and Generalized Masked Diffusion for Discrete Data").

Instead of predicting token i+1 from tokens 0..i (AR), we:
1. Take a complete sequence of tokens
2. Randomly mask fraction `t` of positions (replace with MASK token id=1024)
3. Train the model to predict original tokens at masked positions using **bidirectional** attention
4. Weight loss by `1/t` to form a valid ELBO (upper bound on NLL)
5. At eval, approximate the integral over t using 8-point trapezoidal quadrature

## Results (8×H100 80GB SXM)

| Metric | Value |
|--------|-------|
| Steps | 11,808 |
| ms/step | 50.8 |
| Pre-roundtrip val_bpb | 1.3409 |
| **Post-roundtrip val_bpb** | **1.3485** |
| Artifact size | 15,631,777 bytes |
| Peak memory | 10,251 MiB |

## Key Changes from AR Baseline

| Change | Detail |
|--------|--------|
| Vocab size | 1024 → 1025 (extra MASK token) |
| Attention | `is_causal=True` → `is_causal=False` (bidirectional) |
| Data loading | Removed AR x/y shift; returns single token tensor |
| Forward pass | Computes `(1/t)`-weighted CE loss at masked positions only |
| Eval | 8-point trapezoidal quadrature over mask ratios `[0.05..0.95]` |
| Mask ratio floor | `eps=0.1` (prevents 1/t explosion) |

Everything else is unchanged: Muon optimizer, U-Net skips, RoPE, RMSNorm, relu², int8+zlib quantization.

## Run Command

```bash
RUN_ID=mdlm_8gpu ITERATIONS=20000 VAL_LOSS_EVERY=1000 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Why MDLM is interesting (and worse)

The ELBO bound MDLM optimizes is inherently looser than AR exact log-likelihood. An equally capable AR model will always report better BPB. The ~0.13 BPB gap vs the AR baseline (1.22) is expected.

The value is architectural: bidirectional context, parallel decoding potential, and a connection to continuous diffusion models — all areas of active research.

## Credits

- **MDLM paper**: Sahoo et al. 2024, "Simplified and Generalized Masked Diffusion for Discrete Data"
- **Base model**: Parameter Golf baseline (`train_gpt.py`)

