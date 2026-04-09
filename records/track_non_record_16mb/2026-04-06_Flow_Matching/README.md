# Flow Matching Language Model

This is a non-record submission that replaces the autoregressive `train_gpt.py` baseline with a flow matching language model implemented in `train_gpt.py` (this folder). The model is based on ["Flow Matching for Conditional Text Generation in a Few Sampling Steps"](https://aclanthology.org/2024.eacl-short.33.pdf) (EACL 2024).

The model keeps much of the original training stack (data loading, quantization, distributed training, Muon optimizer infrastructure), but replaces the causal next-token objective with a continuous-flow denoising objective over token embeddings, conditioned on a source context. It was trained on 8 x H100 for 10 minutes.

## What Changed

The baseline GPT is replaced by three new components:

- **`TransformerTimestepModel`** — a bidirectional (non-causal) transformer with sinusoidal timestep embeddings injected at each layer. Position embeddings are added alongside the timestep signal.
- **`TransformerEncoderLayer`** — standard bidirectional multi-head attention + FFN block with LayerNorm, GELU, and dropout=0.1.
- **`Flow`** — the top-level model wrapping the transformer. Implements the flow matching objective and variational BPB evaluation.

The optimizer is switched from Muon to AdamW (lr=1e-5), since the flow matching objective is not a classification cross-entropy and benefits from a simpler optimizer.

## Flow Matching Objective

Each training sequence of length `TRAIN_SEQ_LEN` is split in half:
- **Source**: first `TRAIN_SEQ_LEN // 2` tokens — used as clean context, never noised.
- **Target**: second `TRAIN_SEQ_LEN // 2` tokens — the tokens to predict/generate.

At each training step:
1. Sample `t ~ U[0, 1]` per batch element.
2. Interpolate the target token embeddings toward Gaussian noise: `x_t = (1 - t) * noise + t * target_embs` (linear interpolation flow path).
3. Concatenate clean source embeddings with the noisy target embeddings and pass to the transformer with timestep `t`.
4. The model predicts the velocity field `v_pred` over the full sequence; only the target portion is supervised.
5. Estimate the denoised embedding: `z1_hat = x_t + (1 - t) * v_pred_tgt`.
6. Combined loss: **MSE on the velocity field** (flow regression) + **cross-entropy on `z1_hat` projected through the embedding matrix** (token anchor loss).

## Config

- **Tokenizer/data**: reuses FineWeb SP-1024; no extra tokens needed (embedding space is continuous).
- **Layout**: `vocab_size=1024`, `model_dim=512`, `num_layers=6`, `num_heads=8`, `max_seq_len=1024`. Slightly shallower than the 9-layer baseline to fit within the 16 MB compressed limit.
- **Attention**: bidirectional (no causal mask), standard multi-head attention with dropout=0.1.
- **Timestep conditioning**: sinusoidal embedding of `t` projected through a two-layer MLP (`dims → 4*dims → dims`), broadcast-added to every sequence position before the transformer.
- **TRAIN_SEQ_LEN=1024**: sequence is split 512/512 between source and target.
- **VAR_EVAL_STEPS=32**: validation uses 32 evenly spaced timestep samples for the Monte Carlo BPB estimate.
- **Optimizer**: AdamW, lr=1e-5, β=(0.9, 0.95), weight_decay=0.01.
- **All else** (batch size, distributed setup, quantization, data loading) is identical to the baseline.

## Metrics

- **Training loss** is the sum of MSE velocity loss and CE anchor loss — not directly comparable to the baseline cross-entropy.
- **val_bpb** is a variational upper bound on bits-per-byte obtained by averaging CE(`z1_hat`, target) over `VAR_EVAL_STEPS` evenly spaced timesteps, then converting nats/token → bits/byte using the tokenizer's tokens-per-byte ratio. This is **not apples-to-apples** with the autoregressive baseline BPB for the same reason as masked diffusion models: there is no exact autoregressive factorization, so the variational Monte Carlo estimate is an upper bound.

## Variational BPB

The val_bpb reported here follows the same logic as the MDLM submission's variational metric:

- For each of `VAR_EVAL_STEPS` evenly spaced `t ∈ [0, 1]`, independently sample noise and compute the forward-noised input.
- Run the transformer to get `v_pred`, estimate `z1_hat`, and compute token CE.
- Average CE across timesteps → nats/token → bits/byte via `tokens_per_byte`.

This is an upper bound, not an exact codelength. More steps → tighter bound. At `VAR_EVAL_STEPS=32`, it provides a reasonable estimate at acceptable eval cost (~2 min on 8×H100 NVL).

## Things That Didn't Work / Notes

- Flow matching on token embeddings is fundamentally harder than on continuous data (e.g., images) — the embedding space has no natural density, so the model must both move in the right direction and land near the right token.
- The MSE + CE combined loss is unstable early in training; the loss starts very high (~20) and drops slowly.
- The model only hits ~2500 steps in the 10-minute wallclock cap on 8×H100 NVL, vs ~20 000 steps for the AR baseline — significantly fewer updates per wall-clock second due to the bidirectional attention over double the sequence length.
- val_bpb is not directly comparable to AR baselines.

## Files

- `train_gpt.py` — single-file flow matching training/eval script
- `log_run1.txt`, `log_run2.txt`, `log_run3.txt` — training logs (3 seeds) on 8×H100 NVL
- `submission.json`
- `README.md`

## Metrics

- **val_bpb**: 3.6674 (mean of 3 runs, post-quant int8+zlib roundtrip)
- **step_stop**: 2534 (wallclock cap at 10 min)
- **wallclock_seconds**: 600.379
- **compressed artifact size**: 10,976,330 bytes (int8+zlib)
- **total submission size**: 44,224,625 bytes (raw model + code)
- **model parameters**: 22,063,616

Although the val_bpb is not directly comparable to autoregressive baselines, the model is clearly learning: val_bpb drops from 7.05 → 3.72 → 3.67 over the first 2000 steps (see `log_run1.txt`). However, a flow matching model would likely have to be trained for many more epochs to be a useful language model.