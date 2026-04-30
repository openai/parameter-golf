# TTSM: Typical Ternary State-Space Model

**Author:** Ambivalence (dd_dent)  
**Track:** 10min/16MB  
**val_bpb:** 2.0032 (seed 42)  
**Comparison baseline:** PR #1644 (mradassaad, Mamba-3 Hybrid, 1.1473 bpb)  
**Artifact:** 12,039,626 bytes | **Params:** 11M (7.8M ternary at 1.6 bits/param, 3.3M fp16/fp32 dynamics)  
**Date:** April 30, 2026  

---

## State Is Protected

In selective SSMs, the ternary quantization boundary falls at the B and C projections, not in the hidden state. The state vector h_t never sees the ternary constraint. This is why ternary Mamba works.

In the selective SSM recurrence:

```
h_t = exp(Δ_t ⊙ A) ⊙ h_{t-1} + Δ_t ⊙ B_t ⊙ x_t
y_t = C_t ⊙ h_t + D ⊙ x_t
```

B_t controls *whether* each input is written into state — write at full scale, write nothing, or write negated. C_t controls *which* state channels are read out. Both are gates operating on a continuous fp16 state. Errors in B degrade write selectivity; they do not corrupt existing state. Errors in C degrade read selectivity; they do not touch what is being read.

The contrast with DeltaNet: k_t appears in both the state update and the readout simultaneously. An error in k_t propagates bidirectionally. Ternary B/C is structurally easier than ternary DeltaNet k_t. We haven't tested this.

Δ_t (the discretization step) is *not* the easy case either. Δ enters exp(ΔA), where small errors compound multiplicatively through the recurrence. We kept dt_proj in fp16 and A_log in fp32. mradassaad (PR #1890, same author as our comparison baseline) independently reached the same boundary — their Mamba-3 hybrid collapsed under INT6 until they promoted A/dt to INT8. The ternary boundary is at B and C.

The model confirms this structure empirically. B activations stayed stable across training (std ≈ 0.009 at convergence); C activations were highly variable (std ≈ 27.1). The write gate locked; the readout explored. We didn't engineer it.

---

## Notes on Engineering

**Reversed-scan backward.** The Triton forward kernel for chunk-wise parallel scan runs in ~1ms per step. The naive backward (PyTorch compiled autograd) took 31 seconds — 1,024 sequential Python→CUDA sync points. The backward recurrence `Δh[t] = do[t] + exp(g[t+1]) * Δh[t+1]` is the same operation as the forward scan, reversed in time. Flip the inputs, run the forward kernel, flip the output. 15 lines. 31s → 1.2s per step (26×).

This generalizes: any recurrence whose backward is the same recurrence reversed in time can reuse its forward kernel.

**NS=5 outperforms NS=10.** DeepSeek-V4 uses 10 Muon Newton-Schulz iterations. We confirmed NS=10 gives 52× better orthogonality than NS=5. It also gives worse val_bpb. The less-orthogonal Muon step acts as a diversity regularizer for the ternary weight competition — under STE, imprecision in the optimizer step prevents premature commitment to ternary assignments. Finite-resource optima require finite imprecision.

**Overtraining degrades quality.** Running beyond the 600s cutoff (1×H100 long-burn), val_bpb rises past 2.08 by step 5000, compared to ~1.93 at the competition-equivalent step count. A phase transition around step ~3000 marks over-crystallization: flip rate collapses from ~12% to <1%, and the ternary assignments lock into a suboptimal configuration. The 600s budget is coincidentally near-optimal.

**Frozen conv outperforms trained conv.** Short conv weights frozen at kaiming initialization outperform trained conv by 0.07 bpb on FineWeb. Random initialization provides an unbiased local smoother that task-specific training degrades. The submitted artifact uses trained conv — this was discovered after the submission run, and GPU availability precluded a rerun.

**Trit packing at entropy.** Ternary weights pack at 5 trits per byte (1.6 bits/param). The artifact is compressed with zlib; zstd achieves equivalent ~0% compression — the encoding is already at entropy. Artifact budget is deterministic: `ternary_params = bytes × 5`. At 1.6 bits/param, 16 MB buys ~80M ternary parameters vs ~25M at int5. Whether the capacity gain exceeds the precision loss is the experiment this submission runs.

**Constrained-optimum architecture.** Under the 10min/16MB budget, optimal model size shrinks relative to unconstrained scaling. Steps-per-second dominates parameters-per-bit. We swept d ∈ {384, 512, 576}, blocks ∈ {5, 7, 9}, D_STATE ∈ {32, 64}, and MATRIX_LR over five points. The architecture below is the result of this search.

**Z-loss for STE stability.** Large logits saturate the softmax, reducing CE gradients to near-zero, cascading to near-zero STE gradients. Z-loss (`1e-4 × logsumexp(logits)².mean()`) keeps logits anchored near zero. From CiprianFlorin (PR #640); origin in PaLM/Gemma.

**Triton kernel.** The kernel required three sequential fixes (int32 overflow, gc lifetime, traced conditional). Each bug hid behind the previous.

---

## Architecture

7 SSM-only blocks (no attention), d=576, D_STATE=64. Shared bigram/trigram token-boundary features.

| Component | Precision | Reason |
|-----------|-----------|--------|
| B_proj, C_proj | Ternary ({-1,0,+1}, STE, per-row least-squares scales at serialization) | State is protected |
| dt_proj | fp16 | Discretization hazard |
| A_log | fp32 | Same |
| D (skip) | fp32 | Standard Mamba direct-path skip |
| Short conv (k=4) | fp32, trained | dt/B/C preprocessing |
| in_proj, out_proj | Ternary | Gating |

Training: 10% bf16 warmstart → ternary QAT. Muon optimizer, MATRIX_LR=0.40. Z-loss 1e-4. B/C L2 normalization. Triton chunk-wise parallel scan with reversed-scan backward. Batch 32K tokens, EVAL_STRIDE=96.

Triton scan kernel adapted from fla-org/flash-linear-attention HGRN (MIT license).

<details>
<summary>Run command</summary>

```bash
TTSM_BLOCKS=7 SSM_ONLY=1 D_STATE=64 A_LOG_INIT=diverse \
SSM_SHORT_CONV=1 SSM_NORMALIZE_BC=1 TTSM_TRITON=1 \
EVAL_STRIDE=96 MODEL_DIM=576 NUM_BLOCKS=7 MUON_EQ_R=1 \
WARMSTART_FRAC=0.1 LAYER_TYPE_WARMDOWN=1 \
BIGRAM_BUCKETS=3072 TRIGRAM_HASH=1 \
MATRIX_LR=0.40 MUON_BACKEND_STEPS=5 \
TRAIN_BATCH_TOKENS=32768 MAX_WALLCLOCK_SECONDS=600 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_ternary.py
```

</details>

## Results

| Seed | val_bpb | Steps | Artifact |
|------|---------|-------|----------|
| **42** | **2.0032** | 3889 | 12,039,626 B |

8×H100 SXM, 154 ms/step, 600s wallclock.

Additional seeds (same config, same hardware, separate runs): seed 314 → 2.0062, seed 999 → 2.0419. 3-seed mean: 2.0171 ± 0.018.

## Compliance

- [x] All seeds train in ≤600s
- [x] All artifacts ≤16,000,000 bytes (largest: 12,039,626)
- [x] Sliding window eval, EVAL_STRIDE=96, consistent across seeds
- [x] No test-time training on validation data
- [x] No network calls during evaluation

---

## Attributions

- **Fork base**: @thwu1 (int5/int6 mixed quantization)
- **Ternary QAT + Z-loss**: @CiprianFlorin (PR #640)
- **Mamba SSM**: Albert Gu, Tri Dao (2023)
- **Triton scan kernel**: fla-org/flash-linear-attention HGRN (MIT license, adapted)
- **BigramHash**: @Raahil Shah
- **MuonEq-R**: @clarkkev (PR #1394)
- **SSM baseline**: @mradassaad (PR #1644)
- **Dynamics protection convergence**: @mradassaad (PR #1890)
- **Muon optimizer**: Kosson et al., Jordan et al.
- **DeepSeek-V4**: CSA m=4 convergence with our short conv k=4; NS iteration calibration informed our NS=5 finding
- **Steps > depth insight**: @newjordan

---

*First ternary SSM. The lane is open.*
