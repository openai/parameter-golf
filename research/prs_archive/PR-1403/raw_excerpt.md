# PR 1403 — MDLM: Masked Diffusion Language Model

**Author:** Rhoahndur (non-record)
**Claimed BPB:** 1.3485 (1-seed, post-roundtrip)
**Artifact size:** 15,631,777 bytes
**Seeds:** not stated (single run)

## Files retrieved
- `records__track_non_record_16mb__2026-04-06_MDLM_MaskedDiffusion__README.md`
- `records__track_non_record_16mb__2026-04-06_MDLM_MaskedDiffusion__train_gpt.py`

## Environment variables (from run command)
RUN_ID=mdlm_8gpu, ITERATIONS=20000, VAL_LOSS_EVERY=1000, MAX_WALLCLOCK_SECONDS=600

## Claimed changes (from README, verbatim)
> A non-record submission converting the baseline autoregressive GPT into a Masked Diffusion Language Model (MDLM), based on Sahoo et al. 2024. Instead of predicting token i+1 from tokens 0..i, we:
> 1. Take a complete sequence of tokens
> 2. Randomly mask fraction t of positions (replace with MASK token id=1024)
> 3. Train the model to predict original tokens at masked positions using bidirectional attention
> 4. Weight loss by 1/t to form a valid ELBO
> 5. At eval, approximate the integral over t using 8-point trapezoidal quadrature
>
> Key Changes from AR Baseline: Vocab 1024→1025 (MASK token); is_causal=True→False; removed AR x/y shift; (1/t)-weighted CE loss at masked positions only; 8-point trapezoidal quadrature over mask ratios [0.05..0.95]; mask ratio floor eps=0.1.
>
> Everything else unchanged: Muon optimizer, U-Net skips, RoPE, RMSNorm, relu², int8+zlib quantization.
>
> The ELBO bound MDLM optimizes is inherently looser than AR exact log-likelihood.
