# Experiment 026: MoE 2-Expert Top-2 — RUNNING

## Step 500: val_bpb = 1.4685 (baseline 1.4805 = 0.012 BETTER!)
## Same total params as baseline (17.1M). ✅ FITS 16MB.
## Step avg: 967ms (2.2x baseline) — slower than hoped due to expert loop overhead.
## 2 experts × 1x expansion each with learned softmax routing.

## Analysis
MoE routing provides 0.012 BPB improvement per step with identical params/artifact.
But the for-loop over experts + routing computation adds ~2.2x overhead that
torch.compile can't fully optimize. On 8xH100 this would give ~6,300 steps vs
baseline's 13,780 — the per-step advantage may not compensate.

The overhead could potentially be reduced by:
1. Implementing experts as a single batched matmul (einsum) instead of for-loop
2. Pre-computing router weights and using scatter/gather operations
