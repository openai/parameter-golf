# PR 812 — BankLinear: Compositional Weight Sharing via Learned + Random Basis Mixtures

**Author:** Andrew Mouldon (andrewmouldon)
**Claimed BPB:** 1.2236 pre-quant / 1.2297 post-quant (3-seed avg, BankLinear). Baseline: 1.2264 / 1.2331.
**Artifact size:** ~15.73 MB (BankLinear avg 15,734,631 bytes); submission.json lists 15,739,602 / val_bpb 1.2223 / val_bpb_post_quant 1.2280 (seed 2025)
**Seeds:** 1337, 42, 2025

## Files retrieved
- `records__track_non_record_16mb__2026-03-18_BankLinear__README.md`
- `records__track_non_record_16mb__2026-03-18_BankLinear__submission.json`
- `records__track_non_record_16mb__2026-03-18_BankLinear__train_gpt.py`

## Environment variables (from run script if present)
No run_*.sh present.

## Claimed changes (from README, verbatim)
> BankLinear, a linear layer that does not store weights per layer. Instead, each layer constructs its weights as a mixture over a shared bank of matrices. We allocate: a small set of learned basis matrices, a larger set of fixed random projections. In our configuration: 9 total layers, 3 learned basis matrices, 512 fixed random projections. Each layer learns its own set of mixing coefficients: W^(l) = Σ_i α_i^(l) B_i.

> This design can be viewed as a relaxed form of depth recurrence: Recurrence reuses the same weights across layers; BankLinear reuses a shared basis, but allows each layer to construct its own weights.

> The learned basis captures reusable, important directions. The random basis provides a cheap, high-dimensional span.

> We initialize mixing coefficients with a depth-aware profile, where early, middle, and late layers are biased toward different learned bases, with smooth transitions between them. Without this initialization, performance degrades significantly.

> We apply BankLinear to attention projections: Q, K, V are constructed from shared banks. Each layer uses its own mixing coefficients. BankLinear replaces QKV projections, and saved parameters are reinvested into a larger MLP (2.65× vs 2.00× baseline).

> Additional experiments: LoRA adapters less effective than reinvesting into larger MLPs. Per-attention-head mixing coefficients degraded performance. Applying BankLinear to output projections significantly degraded performance (residual stream pollution hypothesis).
