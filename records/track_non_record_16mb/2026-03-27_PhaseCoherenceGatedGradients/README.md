# Phase Coherence Gated Gradients

Exploratory PIC-GD experiment folder for `2026-03-27`. This is not a leaderboard submission package yet.

The script in this folder adapts the baseline training loop to a batch-level version of phase-induced coherence-gated gradient descent (PIC-GD) while keeping:

- the real-valued transformer architecture
- the Muon + Adam optimizer split
- tokenizer-agnostic `val_bpb` evaluation
- the int8 + zlib roundtrip export path

## Current Status

This folder should be treated as an experiment source of truth, not as a finished submission.

- current reported result: single exploratory `8x H100` run around `1.3178 val_bpb`
- not competitive with the current `track_10min_16mb` leaderboard
- `submission.json` remains intentionally unbenchmarked
- generated artifacts like `final_model.pt` and `final_model.int8.ptz` should not be part of the eventual PR

## PIC-GD Adaptation

The implementation stays close to the baseline training loop:

- final hidden states are treated as pseudo-complex latents by pairing adjacent channels as `(real, imag)`
- target-token embeddings are paired the same way to provide a reference signal
- a normalized coherence score is computed from the paired latent/reference dot product
- the coherence score is converted into a detached gradient gate

```python
alpha = PICGD_MIN_GATE + (1 - PICGD_MIN_GATE) * sigmoid(PICGD_BETA * coherence)
```

Training backpropagates `loss * alpha`, while validation and final quantized roundtrip evaluation continue to use raw cross-entropy only.

## Current Experimental Defaults

- `PICGD_ENABLED=1`
- `PICGD_BETA=2.0`
- `PICGD_MIN_GATE=0.05`
- `PICGD_EPS=1e-6`
- `PICGD_TOKEN_STRIDE=32`
- `attention_impl` is logged as `native_gqa`, `kv_repeat_fallback`, or `standard_sdpa`

Training logs include:

- `picgd_coherence`
- `picgd_gate`
- `attention_impl`

## Evidence Standard Before Packaging

Do not rewrite this folder as a real submission until the following exists on `8x H100`:

- 1 baseline run with root `train_gpt.py`
- 3 PIC-GD runs with this folder's `train_gpt.py`
- recorded seeds, `step_avg`, final quantized `val_bpb`, artifact size, and peak memory for every run
- a positive baseline-vs-PIC-GD comparison that justifies keeping the method

If PIC-GD does not beat the baseline mean on the same setup, stop pursuing it for submission.
