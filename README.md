# Non-Record Submission: Arterial Mixer (N-Lane Transformer)

## Architecture

### Overview
- The model consists of `N` parallel Transformer streams ("arteries"), with lane-wise mixers inserted between them for cross-artery fusion.
- The artery mixer is implemented as multi-head linear attention over the artery dimension.
- Each artery is supervised independently with its own LM head during training.
- During validation, predictions from all artery heads are aggregated in probability space by averaging.
- Different arteries can use different attention window sizes. Sliding-window attention is used instead of full attention.

### Implementation Details
- Let `artery_dim = D` and the number of arteries be `N`.
- The embedding tensor has shape `[vocab_size, N, D]`, so each token owns an independent embedding for each artery.
- Each artery runs its own Transformer forward pass with independent parameters.
- Linear attention in the artery mixer uses `ELU` activation, allowing near-zero routing scores to produce negligible interaction and negative scores to express suppression.
- XSA is applied only to selected arteries, usually the ones with smaller attention windows such as `1024` or `2048`.
- U-Net-style residual connections are implemented as additional mixer slots at the target block. Information is fused through the mixer KV path rather than direct addition.
- RoPE is applied to these residual slots to encode slot position / residual-source identity.

### Techniques Not Used
- SmearGate
- n-gram cache
- TTT
- ...

These may be explored in future iterations.

---

## Experimental Findings

### Submitted Configuration and Result
- Main submitted setting:
  - `10L`
  - `2 arteries`
  - `artery_dim = 384`
  - `attention heads = 3`
  - attention windows: `1024, 4096`
  - XSA on artery `0` (the `1024` window artery)
  - `mlp_mult = 2`
  - `ctx = 16384`

- Representative result from an `8xH100`, `10 min` run:
  - `step = 3511`
  - `final val_loss = 2.0275`
  - `final val_bpb = 1.2008`
  - `step_avg = 170.93 ms` (About 2× the latency of a dense model with the same number of parameters)

### Key Findings
- Averaging predictions from parallel LM heads at validation performs better than using any individual artery head alone.
- XSA placement matters:
  - `XSA on 1024 artery > XSA on both 1024/4096 arteries > XSA on 4096 artery only`
- Applying RoPE to U-Net residual slots performs better than leaving residual slots unencoded.
- A causal mask on the artery mixer allows later arteries to avoid affecting earlier arteries. This makes it possible to inspect the incremental effect of adding arteries `1...N` within a single `N`-artery run.
- Under causal mixing, diminishing marginal returns from increasing the number of arteries were observed.
- However, the causal restriction itself appears to hurt final performance relative to non-causal artery mixing.

---

## Motivation

### 1. Extra Hidden-State Slots
This idea was first motivated by `records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence`.

That work's parallel residual design (not the GPT-J style) separates the MLP and attention lanes. My hypothesis was that some of the gain came from maintaining more parallel hidden states and allowing controlled interaction between them.

### 2. Sparse Hidden Dimensions
Model parameters scale roughly as `O(D^2)` as model width grows, because both:
- the number of neurons scales with `D`, and
- the input/output projections also scale with `D`.

MoE introduces sparsity in the number of activated neurons. This work explores a different direction: making the hidden dimension itself structurally sparse by splitting it into multiple parallel subspaces, then selectively mixing them.

---

## Limitations
- Higher latency than a standard dense Transformer with the same effective width.
- The current implementation is not yet optimized for throughput; artery-wise sequential execution introduces overhead.
