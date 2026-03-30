# MoE Exploration + Multi-bit Quantization Analysis

**Dense control val_bpb: 1.1456** (attn6_mlp6, sliding window stride=64, post int6+zstd quantization roundtrip)

This non-record submission documents two negative results under the 16MB artifact cap:

1. **Preliminary MoE negative result at small scale** — a 2-expert soft-routing MoE underperforms the dense control throughout the observed training window.
2. **Multi-bit post-training quantization comparison** — int4 MLP quantization is destructive in this setup, making the MoE parameter-expansion path unattractive at this scale.

The dense control in this analysis reaches **1.1456 val_bpb**, which is within **0.0028 BPB** of the **March 20, 2026** leaderboard leader (**1.1428**). This makes the quantization sensitivity results directly leaderboard-relevant rather than a far-from-SOTA artifact study.

Configuration:
- Track: `non-record-16mb`
- Dense control layout: `NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 TRAIN_SEQ_LEN=2048`
- Quantization comparison: same trained dense model evaluated at int6, int5, int4 for MLP and attention
- Hardware: 8xH100 SXM, 7410 steps, 81ms/step, SWA 30 checkpoints

## Run Command

```bash
# Dense quantization comparison (8xH100 SXM, ~10min training + ~13min eval)
NCCL_IB_DISABLE=1 NUM_EXPERTS=1 MLP_MULT=3 NUM_LAYERS=9 MODEL_DIM=512 TRAIN_SEQ_LEN=2048 MAX_WALLCLOCK_SECONDS=600 RUN_ID=quant_comparison torchrun --standalone --nproc_per_node=8 train_gpt.py

# MoE run (8xH100 SXM)
NCCL_IB_DISABLE=1 NUM_EXPERTS=2 MLP_MULT=1.5 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Multi-bit Quantization Results

Same trained dense model quantized at 5 configurations, each with zstd-22 compression and sliding-window eval (stride=64):

| Config | Attn | MLP | Artifact | val_bpb | vs int6 baseline |
|--------|------|-----|----------|---------|-----------------|
| attn6_mlp6 | int6 | int6 | 15.14 MB | 1.1456 | baseline |
| attn6_mlp5 | int6 | int5 | 13.39 MB | 1.1524 | +0.0068 |
| attn6_mlp4 | int6 | int4 | 11.51 MB | 1.2111 | **+0.0655** |
| attn5_mlp5 | int5 | int5 | 13.05 MB | 1.1559 | +0.0103 |
| attn5_mlp4 | int5 | int4 | 11.29 MB | 1.2183 | +0.0727 |

![Quantization Comparison](quant_comparison.png)

Key findings:

- **int5 MLP is viable** (+0.0068 BPB, saves 1.75 MB) — consistent with the March 20, 2026 leaderboard leader using int5 for MLP
- **int4 MLP is destructive in this setup** (+0.0655 BPB) — it wipes out most of the dense-model gain over the naive baseline
- **int5 attention adds ~+0.0036 BPB** over int6 attention — attention is more sensitive than MLP in this experiment
- **int4 PTQ is destructive in this setup** for both MLP and attention

## MoE at Small Scale (preliminary evidence)

**Setup:** 2nd-place-style architecture with dense MLP 3x replaced by MoE (2 experts × 1.5x, soft gated routing). Total params remain effectively identical (~22M).

We include `moe_train_partial.log`, a partial 8xH100 SXM log from the MoE run. The RunPod pod died at step 2000, so only a partial log survived; the MoE conclusion here should therefore be interpreted as a **preliminary negative signal**, not a fully converged final result.

### Observed checkpoints

| Step | Dense control val_bpb | MoE val_bpb | Delta |
|------|-----------------------|-------------|-------|
| 500  | 1.4058 | 1.4115 | +0.0057 |
| 1000 | 1.3286 | 1.3386 | +0.0100 |
| 1500 | 1.3024 | 1.3163 | +0.0139 |
| 2000 | 1.2709 | 1.2866 | +0.0157 |

**Observed result:** by step 2000, the MoE model is already **+0.0157 BPB** worse than the dense control under the same validation protocol, and the gap widens monotonically over the observed window.

**About the earlier `~1.20-1.22` number:** that figure was an informal extrapolation from this partial learning curve, not a verified final score. The reproducible MoE evidence included in this submission is the partial checkpoint table above plus `moe_train_partial.log`.

Why it likely failed in this setup:
- Soft routing = weighted sum of both experts = effectively one slightly more flexible MLP
- Per-token MLP capacity is halved (1.5x vs 3x) with no demonstrated compensating specialization
- Apple scaling laws (ICML 2025) argue optimal sparsity is 0 below ~500M total params
- 2 experts is likely too coarse for meaningful specialization compared with large-expert-count MoE systems such as DeepSeekMoE

## Why This Matters for MoE Under 16MB

The MoE value proposition under 16MB is: “use aggressive quantization to fit more total expert parameters, then compensate the per-token capacity loss with specialization.”

This data shows:
- int5 saves only 1.75 MB (~12% of the MLP budget) — not enough for meaningful expert expansion
- int4 saves 3.63 MB but costs +0.0655 BPB — much larger than any plausible specialization gain at this scale
- In this ~22M-parameter regime, the available quantization headroom appears too small to fund a compelling MoE expansion path

## References

- Abnar et al., "Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models", ICML 2025
- Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization", ACL 2024
- Kim et al., "MoQE: Mixture of Quantized Experts", 2023
