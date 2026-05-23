# Text Diffusion: 1.478 BPB with a 15.8MB Masked Diffusion Language Model

First text diffusion submission to Parameter Golf. 19.2M-parameter bidirectional transformer trained with the MDLM objective (Sahoo et al. 2024), quantized to fp8 e4m3. Non-record: training took 99 minutes on 8xH100, not the 10-minute budget.

The model is 0.18 BPB worse than the AR baseline at matched eval protocol. That gap holds across prefix lengths, training budgets, and evaluation methods.

Initial results imply that it comes from the bidirectional masked-LM objective being less effective at compounding left context than causal training.

| Metric | Value |
|---|---|
| val_bpb (NELBO K=256, 3-seed mean) | **1.483 +/- 0.0004** |
| val_bpb (left-to-right chain-rule, 5000 seqs, 3-seed mean) | **1.420 +/- 0.003** |
| val_bpb (confidence-order chain-rule, 5000 seqs, 3-seed mean) | 1.430 +/- 0.002 |
| AR baseline (9L-512d, 10 min 8xH100, 5000 seqs) | 1.239 |
| Parameters | 19,190,656 |
| Artifact | 15,833,702 bytes mean (15.83 MB / 15.10 MiB) |
| Training | 80,000 steps, 64.4 ms/step, 99 min on 8xH100 |
| Seeds | 314, 42, 999 |

Methodology: try a variety of 3-minute 1-H100 experiments on Modal, expand to 15 minute 1-H100 experiments, invest in the more promising experiments for 8-H100.

Future directions: use introspective diffusion? Looped LMs?

## How it works

Standard autoregressive LMs predict tokens left to right: P(x_i | x_1, ..., x_{i-1}). MDLM instead trains a bidirectional transformer to predict tokens that have been randomly replaced with a [MASK] token. At each training step, a fraction t (sampled uniformly from [0.001, 1]) of positions are masked, and the model predicts the original tokens at those positions. The loss is weighted by 1/t (clamped at 100) so that lightly-masked examples — where the model must make fine-grained predictions from nearly complete context — contribute as much as heavily-masked ones.

The forward pass is the same transformer architecture as the AR baseline (RMSNorm, GQA attention, relu^2 MLP, U-Net skip connections, logit softcap, RoPE), with one difference: `is_causal=False` in the attention. The embedding table has vocab_size+1 rows (the extra row is the [MASK] embedding). The output projection uses only the first vocab_size rows of the tied embedding, excluding [MASK] from the prediction vocabulary.

At eval time, BPB is computed via the NELBO (negative evidence lower bound): for K evenly-spaced values of t in [0.001, 1], mask positions with probability t, run the model, and accumulate the weighted negative log-likelihood at masked positions. The NELBO is a valid upper bound on -log P(x) under the model. With K=256 on the full validation set, it gives 1.4791 BPB.

A tighter estimate comes from the confidence-order chain-rule: start with all positions masked, unmask the position where the model is most confident, score it, repeat. This is an exact chain-rule decomposition of log P(x) under a model-chosen ordering and gives 1.4405 BPB on 2000 validation sequences — 0.048 BPB tighter than NELBO.

## Architecture

```
8 transformer layers, dim=576, 8 heads, 4 KV heads (GQA), head_dim=72
Bidirectional attention (is_causal=False)
Absorbing-state masking: tokens → [MASK] with probability t
Embedding: 1025 x 576 (vocab_size + 1 for [MASK]), tied with output head
U-Net: 4 encoder layers + 4 decoder layers with learned skip weights
relu^2 MLP, 2x expansion (hidden=1152)
RMSNorm, logit softcap=30, RoPE base=10000
No timestep conditioning (per SMDM finding, Nie et al. 2025)
```

19,190,656 parameters. Quantized to fp8 e4m3 (no per-parameter scale; the 8-bit float format's 3 mantissa bits match the narrow magnitude range of trained transformer weights better than int8's uniform grid). Compressed with the better of zlib-9/lzma-6. Artifact: 15,833,702 bytes (mean over 3 seeds).

## Training

Muon optimizer for matrix parameters (lr=0.02, momentum=0.95, 5 Newton-Schulz steps), Adam for embeddings and scalars (tied_embed_lr=0.05, beta1=0.9, beta2=0.95). The Muon learning rate is half the AR baseline default of 0.04 — the MDLM loss's 1/t weighting creates higher gradient variance, and the lower LR was the single biggest scout-measured improvement (-0.003 BPB).

EMA with decay=0.999, starting at 10% of the wallclock budget (540 s ≈ step 8,400 at 64.4 ms/step). The EMA shadow is stored in fp32 regardless of the live model's dtype. This matters: at decay=0.999, the per-step update is ~5e-5 times the weight magnitude, which rounds to zero in bf16. A bf16 shadow silently freezes at initialization, producing a 0.40 BPB regression (measured empirically in a 10,000-step microbenchmark and the 1.95 BPB "int8-ema999" run before the fix).

80,000 iterations (~4.2 epochs of FineWeb 10B), 524,288 tokens per step (batch_size=512, seq_len=1024, grad_accum=1 on 8 GPUs). 64.4 ms/step on 8xH100 SXM. Total wall time 99 minutes including compilation and eval.

Low-discrepancy time sampling during training: instead of drawing t ~ Uniform for each sample independently, the batch of B samples uses evenly-spaced t values offset by a single random draw. This reduces variance in the per-step loss. (Not antithetic sampling, which would pair t with 1-t — this is the stratified-grid variance-reduction trick from MDLM.)

## The diffusion-AR gap

Evaluated head-to-head on 200 validation sequences using the same AR-style chain-rule protocol (mask the current position and everything to its right, sweep left to right):

| Position | AR baseline | Diffusion | Gap |
|---|---|---|---|
| 0 | 2.701 | 2.673 | -0.028 (diffusion wins) |
| 1 | 2.453 | 2.403 | -0.050 (diffusion wins) |
| 4 | 2.032 | 2.063 | +0.031 |
| 16 | 1.618 | 1.745 | +0.127 |
| 64 | 1.422 | 1.589 | +0.167 |
| 256 | 1.297 | 1.486 | +0.189 |
| 1023 | 1.227 | 1.408 | +0.181 |

The diffusion model has a better unconditional prior (positions 0-1), having been trained to predict tokens from any amount of context. But starting at position 4, the AR model pulls ahead and the gap grows to 0.18 BPB by position 256, then stabilizes. The AR model compounds left context more effectively at every scale.

The gap is flat across prefix lengths (0-2048 bytes) when measured as tail BPB, and flat across training budgets (15 min to 4 epochs). It is not a hyperparameter or scale issue — it reflects a structural difference between causal and bidirectional masked-LM training.

## Evaluation methods compared

All three estimators evaluated on the same 5000 validation sequences, 3 seeds each (results are 3-seed means):

| Estimator | BPB (3-seed mean) | Wall time (1 GPU) |
|---|---|---|
| NELBO K=256 | 1.483 | ~15 min |
| Confidence-order chain-rule (step_size=1) | 1.430 | ~60 min |
| **Left-to-right chain-rule (step_size=1)** | **1.420** | **~60 min** |

Left-to-right is the tightest bound — 0.010 BPB tighter than confidence-order, 0.063 BPB tighter than NELBO. This is consistent across all three seeds (per-seed range: 1.418–1.423 for l2r, 1.427–1.432 for conf_order, 1.483 for all three NELBO).

This was initially unclear: an earlier comparison on mismatched sample sizes (200 seqs for l2r, 2000 seqs for conf_order) made it ambiguous which ordering won. The 5000-seq matched evaluation resolved it decisively. Left-to-right wins, which is surprising — a bidirectional model trained with random masking evaluates best under the left-to-right ordering it was never trained on, and worse when allowed to pick its own order.

One possible explanation: left-to-right builds context monotonically (position i always has positions 0..i-1 fully revealed), while confidence-order can leave gaps. A gap at position j means position j+1 was unmasked without seeing j, which may be suboptimal for a model whose attention patterns, despite being bidirectional, still learned to weight nearby left-context heavily from the data distribution (English text flows left to right).

Earlier measurements on a different artifact (2000 seqs) found: NELBO K=64 → K=256 improves only 0.005 BPB, and random-order chain-rule (K_ord=4, step_size=8) gives 1.498 — slightly worse than NELBO. The ordering matters; averaging over random orderings does not help.

AR baseline at matched 5000-seq chain-rule protocol: **1.239 BPB** (ar-baseline-8gpu-adfafe). Diffusion − AR gap at matched l2r chain-rule: 1.420 − 1.239 = **0.181 BPB**, identical to the 200-seq comparison — the gap is robust to slice size.

### What we haven't measured

- **Iterative denoising as a likelihood estimator.** The original motivation for text diffusion was "apply the same weights T times at eval for iterative refinement." We never built a true multi-step denoising chain estimator — the chain-rule evals are the closest proxy, but they unmask one position at a time rather than running a trajectory through noise levels from t=1 to t=0.

## Convergence

| Steps | Epochs | BPB (raw) | Source |
|---|---|---|---|
| 2,300 | 0.12 | 1.704 | 15-min 1xH100 |
| 18,000 | 0.95 | 1.600 | 1-epoch 1xH100 |
| 34,000 | 1.79 | 1.568 | 2-epoch 1xH100 |
| 69,000 | 3.63 | 1.552 | 4-epoch 1xH100 (int8, no EMA/LR fix) |
| 33,700 | 1.78 | 1.510 | 30-min 8xH100 (int8) |
| 67,300 | 3.54 | 1.498 | 4-epoch 8xH100 (int8, K=256) |
| **80,000** | **4.21** | **1.479** | **4-epoch 8xH100 (stacked wins, K=256)** |

No overfitting through 4.2 epochs. Loss follows a power law. The stacked improvements (fp8 e4m3, lr=0.02, EMA 0.999, 8L-576d shape) contributed -0.019 BPB over the int8 9L-512d baseline, decomposing as: ~-0.003 from LR, ~-0.002 from EMA, ~-0.010 from wider shape at convergence, ~0 from fp8 (artifact savings only).

## Dead ends

**Ternary quantization** (1.58 bits/param, ~80M params in 16MB): Two measurement campaigns — 1xH100 epoch-scale and 8xH100 4-epoch — both showed the ternary floor at ~2.1 BPB, 0.5 BPB worse than int8's ~1.5 BPB. The gap didn't close with more training. The 5x parameter density advantage was real but the per-parameter information loss overwhelmed it.

**int4 without non-uniform quantization**: 0.24-0.38 BPB damage. E2M1 (non-uniform 4-bit float: {0, 0.5, 1, 1.5, 2, 3, 4, 6}) was 2.7x better at 0.14 BPB damage, but still worse than int8's 0.002.

**Depth recurrence under quantization**: Using 3 unique layers cycled 3x each (9 effective layers, 3 layers of parameters) collapsed to 4.0 BPB. Quantization errors amplify through weight reuse.

**EMA with bf16 shadow**: Silently no-ops. The 0.001 * weight update rounds to zero in bf16 precision. Produced 1.95 BPB (vs 1.55 baseline). Fixed by storing the shadow in fp32.

**Muon LR 0.04** (AR baseline default): Too hot for the 1/t-weighted MDLM loss. LR sweep on 3-min scouts: 0.01→2.138, 0.02→1.968, **0.04→1.922**, 0.08→1.956, 0.16→2.290. The default happened to be near-optimal for scouts but 0.02 won at convergence.

## Reproducing

```bash
NUM_LAYERS=8 MODEL_DIM=576 QUANT_MODE=fp8 MATRIX_LR=0.02 \
EMA_DECAY=0.999 EMA_START_FRAC=0.1 ITERATIONS=80000 \
MAX_WALLCLOCK_SECONDS=5400 \
FINAL_EVAL_T_SAMPLES=256 FINAL_EVAL_MAX_SEQS=0 \
torchrun --standalone --nproc_per_node=8 train_diffusion.py
```

`MAX_WALLCLOCK_SECONDS=5400` is what the three submission seeds used (actual wall time was ~5,900 s, with the overage absorbed by Modal's per-spawn `max_seconds=5400 + 1800` buffer). The script's default of 600 s would stop training at ~9K steps. Note that `EMA_START_FRAC` is interpreted as a fraction of the wallclock cap, not of `ITERATIONS` — so with `MAX_WALLCLOCK_SECONDS=5400` and `EMA_START_FRAC=0.1`, EMA updates begin at 540 s (~step 8,400), not at step 8,000.

## References

- Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024)
- Nie et al., "Scaling up Masked Diffusion Models on Text" (ICLR 2025)
- Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (2024)
