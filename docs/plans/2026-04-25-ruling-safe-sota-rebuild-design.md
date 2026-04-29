# Ruling-Safe SOTA Rebuild Design

Date: 2026-04-25

## Goal

Build a ruling-safe Parameter Golf submission that beats the current strongest
non-contested open frontier around PR #1797 (`1.06157` BPB) without relying on
byte-PPM legality or validation-derived predictors. The target operating band is
`<= 1.0595` BPB over at least three seeds, under the decimal `16,000,000` byte
artifact cap, with training and eval both under 600 seconds on 8xH100 SXM.

## Current Gap

Our local `train_gpt_kl.py` is not the current SOTA stack. It still defaults to
SP1024 paths, 3x MLP, GPTQ-lite/zstd serialization, and `QUANT_BITS=4`. It has
some useful experimental hooks, but several are incomplete or unused:

- `LegalTTT` exists but is not connected to final eval.
- Value embedding flags exist but the model constructor does not wire the
  feature through attention.
- The quantization path is GPTQ-lite or uniform int4/int6, not the full
  Hessian GPTQ used by recent frontier submissions.
- The optimizer does not use Polar Express Newton-Schulz coefficients.
- There is no CaseOps/SP8192 byte sidecar accounting path.
- There is no proven phased LoRA-TTT implementation with rank scaling,
  warm-started A, and score-before-update checks.

## SOTA Reference Points

Ruling-safe baseline to reconstruct:

- PR #1787: SP8192 CaseOps, Sparse Attention Gate, Quant Gate, Loop4-5,
  Phased TTT, Polar NS, MIN_LR, Fused CE, PR #1767 TTT improvements:
  `1.06335` BPB.
- PR #1797: PR #1787 base plus SmearGate and LQER asymmetric rank-4
  correction: `1.06157` BPB.

High-risk references to mine but not depend on:

- PR #1795: SP4096 byte-level PPM mixture at `1.01252` BPB, but it requests
  organizer ruling on whether the online byte predictor is legal.
- PR #1813: Scylla tokenizer/data path with QK5.25 and looped recurrence at
  `0.94166` BPB. Treat this as a separate tokenizer/data track until we can
  verify the tokenizer metadata, data prep, BPB accounting, and reviewer stance.

## Approved Strategy

Use a ruling-safe primary branch and keep moonshot ideas as isolated ablations.
The primary submission should be a reconstructed PR #1797-style stack plus
carefully chosen additions that are orthogonal and defensible.

### Primary Stack

- SP8192 CaseOps tokenizer and bijective capitalization transform.
- Per-token byte sidecar BPB accounting on original UTF-8 bytes.
- 11 layers, 512 model dimension, 8 attention heads, 4 KV heads, MLP 4x.
- Train and eval sequence length 2048, sliding eval stride 64.
- XSA on all layers using Flash Attention 3 where available.
- Loop4-5 depth recurrence with two loops, enabled after 35 percent training.
- Parallel residual start layer 8.
- Value embeddings on late layers 9 and 10, VE dimension 128.
- Sparse attention head-output gate, not dense gate.
- SmearGate on the residual stream.
- Full Hessian GPTQ int6, embed bits 7, Brotli quality 11.
- LQER asymmetric rank-4 correction on top quant-error tensors.
- Polar Express Newton-Schulz coefficients for Muon.
- MIN_LR warmdown floor at 0.10.
- Fused softcapped cross entropy for training only.
- Phased score-first LoRA-TTT with rank 96, LoRA alpha 144, warm-started A,
  weight decay 1.0, and document reset.

### Our Stackable Innovations

1. TTT-robust training regularizer.
   Mirror the real BatchedTTTLoRA shape, not the old full-parameter `LegalTTT`
   helper. Apply it sparsely during late training to reduce post-TTT drift
   without increasing artifact size.

2. Value embedding repair.
   The local flags must become a working late-layer value injection path. This
   is already part of the strongest upstream style, so it is not speculative;
   our job is to ensure it is actually active and mirrored in TTT paths.

3. QK gain and recurrence ablations.
   Test `QK_GAIN_INIT=5.25`, Scylla-style recurrence scheduling, and reduced
   bigram dimensions as secondary ablations. Promote only if three-seed and
   artifact numbers are clean.

4. Codec-aware QAT only as a size-margin tool.
   Use it to recover artifact headroom if LQER/CaseOps/code size pushes too
   close to 16 MB. Do not let entropy regularization trade away BPB unless
   ablation proves a net win.

## Approaches Considered

### Recommended: Ruling-Safe Stack First

Reconstruct PR #1797/#1787 and add our stackable fixes. This has the best odds
of an accepted record because all core components are already argued in public
PRs and have logs under the official constraints.

Risk: Requires careful porting because the TTT path manually mirrors model
forward logic. Any gate or value embedding added to training must also be added
to LoRA-TTT helpers.

### High-Risk: PPM/Online Predictor Branch

Add byte-level online PPM mixture. This may produce a large numerical gain, but
the legality category is unresolved. Keep it as a separate experimental branch
or non-record submission, not the primary target.

Risk: Organizer ruling can invalidate the result regardless of BPB.

### Moonshot: Scylla Tokenizer/Data Branch

Rebuild the Scylla tokenizer/data path and try to stack TTT/LQER/sparse gates on
top of the PR #1813 base. This is potentially the highest ceiling, but it
changes the tokenizer/data surface and requires independent validation before it
can be trusted.

Risk: Harder to debug, small artifact margin, and less known reviewer history.

## Data Flow

Training data flows through CaseOps transformation into SP8192 token shards.
Validation uses transformed token shards plus a byte sidecar derived from the
original documents. During final eval, token log probabilities are accumulated
against original byte lengths; no validation token is used before it has been
scored by the model.

The TTT eval path scores each chunk first, snapshots the score, then applies
LoRA updates only from already-scored tokens. Adapters reset per document except
for the approved warm-start behavior on LoRA A. This preserves causal
dependence, full distribution scoring, and single-pass evaluation.

## Error Handling And Compliance

- Abort on tokenizer vocab mismatch.
- Assert BOS markers exist for document-boundary TTT.
- Assert CaseOps decode(encode(x)) equality in data prep tests.
- Fail packaging if artifact plus code exceeds `16,000,000` bytes.
- Log pre-quant BPB, post-quant BPB, post-TTT BPB, artifact bytes, train time,
  eval time, seed, tokenizer path, and core env vars.
- Keep `VAL_LOSS_EVERY=0` in final runs.
- Keep PPM disabled in primary submission.

## Testing Strategy

- Static compile: `python3 -m py_compile train_gpt_v3.py`.
- Unit tests for CaseOps roundtrip, byte sidecar accounting, LQER pack/unpack,
  sparse gate path parity, and score-before-update TTT invariants.
- CPU/small CUDA smoke with tiny iterations when available.
- Artifact-size dry run from a saved small state dict.
- One H100 smoke seed before full 3-seed run.
- Three final 8xH100 SXM seeds with full logs.

## Promotion Criteria

Promote a candidate only if:

- Mean BPB is below `1.0595` across at least three seeds.
- Every individual artifact is below `16,000,000` bytes.
- Training is below 600 seconds.
- Eval is below 600 seconds.
- Logs show score-first TTT and no validation access during training.
- The exact submission folder compiles in isolation.
