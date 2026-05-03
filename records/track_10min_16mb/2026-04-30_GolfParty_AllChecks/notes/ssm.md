# State-Space Models — `KS_SSM_LAST_K`

OpenAI Requests-for-PRs item: *"State-space models, E2E TTT, super long context for evaluation or training."*

## What this is

`ToySSMBlock` in `train_gpt.py` is a Mamba-flavored SSM block that can
replace the last *K* attention layers in the transformer stack. Each
block is:

1. Input projection to `(u, gate)` of size `2D`.
2. Depthwise causal 1-D conv (`kernel_size=4`) on `u`. Stands in for the
   parallel-prefix scan in real Mamba.
3. Per-channel diagonal recurrence `y_t = exp(A_log) * y_{t-1} + u_t`
   in a Python-loop form (no parallel scan).
4. Multiplicative SiLU gate from `gate`.
5. Output projection back to dim `D`.

## Toy vs real

- **Toy:** the recurrence is a Python `for` loop over time, not a parallel
  prefix-sum kernel. Throughput at training time will be terrible —
  `O(T)` sequential loop on GPU. Practical only for `KS_SSM_LAST_K=1`
  on a short eval seq_len, or as a structural ablation that demonstrates
  "yes the block runs end-to-end."
- **Real:** would need (a) Mamba's selective SSM with input-dependent
  `A`, `B`, `C` matrices, (b) a Triton parallel-scan kernel similar to
  `mamba-ssm`, (c) discretization (Δt parameterization), and (d)
  state-passing across cu_seqlens packed-doc boundaries. Roughly the
  amount of code in the public `mamba-ssm` repo.

## Why it's still here

Because the Requests-for-PRs list explicitly asks for it, and even a
stub helps anyone iterating in this direction skip the
boilerplate-integration step (block-stack swap-in, hparam wiring, BOS
handling for cu_seqlens).

## To make it record-worthy

1. Replace the Python loop with a parallel-scan Triton kernel.
2. Move from constant `A_log` to selective `A(u)`, `B(u)`, `C(u)`.
3. Decide which layers to swap (purely SSM stack? hybrid attn+SSM?). The
   hybrid case probably wins given the LQER quantization recipe is tuned
   for attention.
4. Re-tune the `MATRIX_CLIP_SIGMAS` / `ATTN_CLIP_SIGMAS` envelope —
   SSM weight statistics differ from attention.
