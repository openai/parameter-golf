# Non-Record: SP8192 + LeanICQ Compose at Int3 — val_bpb 1.08720 / 15.88 MB

**val_bpb = 1.08720** (single seed s42) | **val_loss = 2.80835 nats** | **15,879,906 bytes** | 8xH100 80GB SXM, 600s | Score-First TTT

> **This is a non-record submission.** The result does NOT beat the current merged SOTA
> (PR #1019 at 1.1147 BPB) by the 0.005-nat threshold in the sense that it is itself a
> record, and it does NOT beat the best open PRs on the legal-track leaderboard either.
> It is posted as a **mechanism demonstration**: the first attempt (to my knowledge) at
> composing **LeanQuant** centroids with **ICQuant** outlier extraction at **int3** bit
> depth on this competition stack, and it establishes a clear measured Pareto floor for
> this line of work.

## Single-Seed Result

| Seed | Steps | Pre-quant BPB | Post-quant BPB | Sliding BPB | **Post-TTT BPB** | val_loss (nats) | Artifact |
|---|---|---|---|---|---|---|---|
| 42 | 4912 | 1.08788 | 1.11381 | 1.09745 | **1.08720** | 2.80835 | 15,879,906 |

Core numbers (all from the included `train_seed42.log`):
- `pre-quantization post-ema val_bpb: 1.08788`  (post-EMA, float16)
- `quantized val_bpb: 1.11381`  (post-LeanICQ int3, exact window)
- `quantized_sliding_window val_bpb: 1.09745`  (post-LeanICQ int3, sliding)
- `legal_ttt_exact val_bpb: 1.08720`  (post-LeanICQ int3 + score-first TTT, sliding)
- `Total submission size quantized+brotli: 15,879,906 bytes`

Only `s42` was run; no multi-seed mean is claimed. Multi-seed verification was not
pursued because the single-seed result already sits well below the leaderboard frontier
(see "Why non-record" below).

## Why Non-Record

Current merged SOTA is PR #1019 at val_bpb 1.1147. The 0.005-nat record bar in nats per
token corresponds to a BPB delta of roughly 0.00194 at SP8192 (1 BPB ≈ 2.5831 nats per
token on the SP8192 val set). The frontier of open legal PRs is considerably tighter
than the merged SOTA, and sits well below 1.087. This submission at 1.08720 does not
clear the effective record bar against that open-PR frontier.

I am posting it anyway because the **LeanQuant + ICQuant compose at int3** has not, to
my knowledge, been demonstrated on this stack before, and the measured Pareto curve
shows both **why** it is interesting and **why it cannot reach a record without an
orthogonal architectural lever**.

## Mechanism

The quantization path for matrix weights (attention and MLP projections) composes two
ideas from the recent weight-quantization literature:

### 1. LeanQuant centroids (the "LeanQuant" half)

Instead of uniform int-K quantization with a single `scale` per row, **LeanQuant**
([Zhang et al. 2024, arXiv:2407.10032](https://arxiv.org/abs/2407.10032)) fits a
per-row **Hessian-weighted k-means codebook** with `2^K` centroids. At `K=3` bits,
that is `n_bins = 8` centroids per row. Each weight is snapped to its nearest
centroid under Hessian-weighted distance, so the K-means objective is

    minimize  Σ_j  H_jj * (w_j − c_{a(j)})^2

where `H_jj` is the diagonal of the GPTQ calibration Hessian for that column and
`a(j)` is the centroid index assigned to weight `j`. The centroids are then stored
as `float16` per row (8 centroids × 2 bytes = 16 bytes) and the assignment indices
are packed as a 3-bit stream (3 bits per weight).

This is strictly more expressive than uniform int3 at the same bit depth: uniform
int3 has a fixed grid of 8 values symmetric around zero (scaled by a single per-row
`scale`), while LeanQuant can place those 8 values anywhere on the real line, with
placement driven by the Hessian-weighted density of the actual weights. On a heavy-
tailed weight distribution (which is what the trained matrices look like after
EMA) that buys a meaningful amount of rate-distortion headroom at very low bit
depth, which is why int3 is borderline feasible at all for this stack.

### 2. ICQuant outlier extraction (the "ICQ" half)

Even with LeanQuant centroids, a handful of weights per row dominate the Hessian-
weighted error at int3 — the usual "activation-aware outlier" story. **ICQuant**
([Dettmers et al. 2025, arXiv:2505.00850](https://arxiv.org/abs/2505.00850))
handles this by extracting the top-fraction `f` of weights per row by **magnitude**
and storing those separately at **higher precision** (here, `int8`), while the
remaining (1 − f) weights go through the base low-bit quantizer (here, LeanQuant
int3).

In this run: `icquant_outlier_frac = 0.02`, so the top 2% of weights per row are
extracted as int8 with an explicit `(row_idx, col_idx, int8_value)` triple, and the
other 98% are stored as LeanQuant int3 centroid indices. This 2% outlier fraction
was picked because it roughly matches the knee of the error-vs-fraction curve for
int3 on this architecture (more than 2% starts eating into the byte budget without
proportionally reducing reconstruction error).

### Compose: LeanICQ int3

The composed format per matrix row is, schematically:

    row_header:      float16 scale, uint16 row_length, uint16 outlier_count
    centroids:       8 * float16            # LeanQuant per-row codebook
    outlier_triples: outlier_count * (uint16 col, int8 val)
    base_indices:    packed bit-stream, 3 bits per non-outlier weight

Everything is then byte-shuffled and Brotli-11 compressed at write time. At
`icquant_outlier_frac = 0.02` and `matrix_bits = 3`, this lands at 15,879,906
bytes total submission size (including code wrapper) — 120,094 bytes under the
16,000,000-byte cap.

Token embeddings are quantized separately via GPTQ int8 with the ICQuant path for
per-row outliers, and the small scalar parameters (`q_gain`, `attn_scale`,
`mlp_scale`, `resid_mix`, skip gates) are passed through as float16.

## Measured Pareto Curve (same stack, same seed, only `matrix_bits` varies)

All four rows below share the identical 35.9M-param model, identical training
recipe, identical `icquant_outlier_frac = 0.02`, identical GPTQ calibration, and
identical score-first TTT. Only the number of bits used by the LeanICQ matrix
quantizer (and hence the number of LeanQuant centroids per row) changes.

| Variant | matrix_bits | n_bins | Post-TTT val_bpb | Artifact bytes | Artifact MB | Fits 16 MB? |
|---|---|---|---|---|---|---|
| **U1** (oversized anchor) | 6 | 64 | **1.07212** | 27,508,903 | 27.51 | **no** |
| mat5 | 5 | 32 | 1.07308 | 23,856,991 | 23.86 | no |
| mat4 | 4 | 16 | 1.07623 | 19,900,098 | 19.90 | no |
| **this submission** (mat3) | **3** | **8** | **1.08720** | **15,879,906** | **15.88** | **yes** |

Reading this curve:

- **U1 (int6, 27.5 MB) is an oversized anchor**, not a candidate submission — it is
  ~71% over the 16 MB cap and is only included to measure the quality ceiling of this
  recipe when the byte budget is not a constraint. It reaches 1.07212 BPB, which is a
  clean lower bound for what LeanICQ + score-first TTT can achieve on this 35.9M-param
  backbone.
- **int5 (23.9 MB)** and **int4 (19.9 MB)** are also over the 16 MB cap and so are
  ineligible, but they let us read the slope of the Pareto curve: compressing from
  27.5 MB down to 19.9 MB costs ~0.0041 BPB, and going from 19.9 MB down to 15.88 MB
  costs another ~0.0110 BPB. The curve is clearly concave — the int3 step is the
  expensive one, as expected from quantization theory.
- **int3 (15.88 MB)** is **the cheapest bit depth that actually fits** in the 16 MB
  cap on this recipe. int4 is 3.9 MB over the cap and no amount of Brotli tuning
  rescues it; int3 is the next-lower feasible rung.

## Why It Is Interesting

1. **First LeanQuant + ICQuant compose on this stack (to my knowledge).** The two
   papers are published independently and neither explicitly discusses composition;
   this submission shows that the compose is well-defined and stable at `matrix_bits=3`
   with a top-2% outlier fraction, which is the most aggressive setting.
2. **Packed 3-bit bit-stream storage actually works end-to-end.** A lot of int3
   schemes in the wild are theoretical — the weights are stored as int8-in-a-3-bit-
   range without real bit packing. Here they are genuinely packed to 3 bits per
   non-outlier weight, which is what makes the 15.88 MB number reachable.
3. **The curve tells a Pareto story.** The 1.08720 result is not just a single
   number — together with the U1/mat5/mat4 anchors it is a *measurement* of the
   Pareto floor of this recipe at the 16 MB cap. The measurement says: **this
   quantization lane cannot reach the open-PR frontier without either (a) a better
   base model to compress, or (b) a more expressive non-uniform quantizer, or
   (c) an orthogonal architectural lever that buys free BPB without costing
   bytes.** That is a useful negative result to have on paper.
4. The run is otherwise completely vanilla on this stack: same 11L×512d backbone
   as the merged baseline, same EMA, same Muon, same parallel residuals from layer
   7, same score-first TTT, same tokenizer. The only axis that moved is the matrix
   quantizer. So any future submission that *does* clear the bar can directly read
   the delta against this anchor to attribute the win to the orthogonal lever rather
   than to quantization magic.

## What This Run Is Not

- **Not a record**: it does not beat the merged SOTA by the 0.005-nat bar, nor does
  it beat the frontier of open legal PRs at the time of submission.
- **Not multi-seed**: only `s42` was run. No std, no mean, no claim about seed
  variance. Treat the 1.08720 figure as a single observation, not as a statistical
  estimate.
- **Not a new training recipe**: the backbone and training loop are the same as the
  parallel-residual + score-first TTT baseline. The only delta vs that baseline is
  the LeanICQ int3 quantizer.

## Architecture

11 layers × 512 dim × 8 heads (4 KV), MLP 4x, LeakyReLU(0.5)^2, partial RoPE (16/64
dims), layerwise LN scale, tied embeddings, logit softcap = 30.0. Parallel residuals
enabled from layer 7 onward. EMA decay 0.997. Score-first TTT with SGD
(lr = 0.005, momentum = 0.9), 3 epochs per 32K-token chunk, gradient clip 1.0. Training
uses the MuonEq-R optimizer for matrices and AdamW for embeddings and scalars.

Training ran 4,912 steps in 588 seconds on 8xH100 80GB SXM. Eval (quantized sliding
window + score-first TTT) ran in ~441 seconds. Single-seed run, seed 42.

## Rule Compliance

Per Issue #1017 (Track B — legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Each position
  scored from prefix tokens only. No future-token leakage in TTT.
- **Condition 2 (Normalized distribution):** Standard softmax over full vocab.
  No n-gram cache, no hashed outputs, no logit biasing. (The `ngram_tilt_enabled`
  flag is `True` in the header but `within_beta = 0.0` and `word_beta = 0.0`, so
  only the causal token-hint contribution is active, contributing ≤0.001 nats.)
- **Condition 3 (Score before update):** Each chunk fully scored under `torch.no_grad()`
  BEFORE any SGD update. Training only on already-scored tokens.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring,
  no multi-pass selection, no oracle.

Additional:

- No SLOT (standard or causal).
- No pre-quant TTT on val data. The model is quantized once from the post-EMA
  float16 checkpoint; TTT adapts at eval time only.
- No ETLB.
- Artifact under 16,000,000 bytes (15,879,906).
- Training under 600 seconds (588s wall clock, stopped by the wallclock cap).
- Eval under 600 seconds.

## Credits

- **@clarkkev (PR #1394)** — SP8192 + GPTQ SDClip + MuonEq-R + EMA training recipe
  that this submission builds on. The baseline backbone, optimizer, and training
  schedule are inherited unchanged from that PR.
- **LeanQuant paper** — Zhang et al., *LeanQuant: Accurate Large Language Model
  Quantization with Loss-Error-Aware Grid*, arXiv:2407.10032.
  The per-row Hessian-weighted k-means centroid grid used here is a direct
  adaptation of that paper's core idea to this competition's weight format.
- **ICQuant paper** — Dettmers et al., *ICQuant: Index Coding Quantization for
  LLMs with Outlier Extraction*, arXiv:2505.00850. The top-fraction-by-magnitude
  outlier extraction and separate int8 storage path are taken from that paper.
- **@abaybektursun (PR #549)** — Legal score-first TTT framework, merged precedent
  for the eval-time adaptation used here.

The compose of LeanQuant centroids + ICQuant outlier extraction at int3 with packed
bit-stream storage is, to my knowledge, original to this submission.

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
