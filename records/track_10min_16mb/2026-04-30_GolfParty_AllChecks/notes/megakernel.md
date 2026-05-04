# Megakernels — `KS_MEGAKERNEL`

OpenAI Requests-for-PRs item: *"Megakernels"*.

## What's already shipping

The PR #1953 base already uses **two fused Triton megakernels**, both
inherited from prior PRs:

1. **Fused LeakyReLU² MLP kernel** (PR #1530, samacqua) — single Triton
   kernel that fuses `up_proj → LeakyReLU² → down_proj` into one pass,
   avoiding the round-trip to fp32 between activation and downprojection.
   Visible in `train_gpt.py` as the `_fused_mlp_*` Triton functions.

2. **Fused softcapped CE kernel** (PR #1787, nprime06) — single Triton
   kernel that computes `softcap(logits) → log_softmax → cross-entropy`
   in a fused pass with a custom backward, avoiding materializing the
   `(B*T, V)` softmax matrix in memory. Visible as the
   `_softcapped_ce_kernel` family.

`KS_MEGAKERNEL=1` is the default and surfaces these in the hparam log
(`megakernels: 2 (fused LeakyReLU² MLP + fused softcapped CE)`) so the
contribution is visible to anyone reviewing.

## What's NOT in this submission

A *full*-block Triton megakernel (single kernel for the entire transformer
block: attn + MLP + residual + norm) would be the next step — see e.g.
"Toward Hardware-Friendly Mamba" or NVIDIA's CUTLASS Block-Based GEMM-FA
fusion. Fitting Polar-Express Newton-Schulz Muon updates and FA3
attention into a single block kernel is a significant project; not in
scope for this submission.

## Why it's here

The Requests-for-PRs item says "megakernels." The honest claim is: the
existing recipe **already has two**, and they're load-bearing for the
600s wallclock budget (un-fused versions would push past the cap). The
flag is doc-only — it makes the contribution visible.

## To make it record-worthy

Build a `flash_block` kernel that fuses the entire block forward
(attention + MLP + residual + ScaleNorm) and integrates with FA3's
attention path. ~1-2 weeks of CUDA / Triton work. Requires careful
backward fusion to keep training cost down.
