# Compute Grant Application — Reference Draft

Use these to fill the form at https://openai.com/index/parameter-golf/#credit-form

---

## Level of compute support

**Select: Medium** (or the largest tier available — you have a concrete plan and working code)

---

## Current best leaderboard submission score

**New participant — no leaderboard score yet.** Code is built and locally verified; waiting on H100 access for first scored run.

---

## Detailed approach (max 2,500 characters)

I have built a complete submission script that stacks every proven winning technique from the current leaderboard into a single, verified pipeline, plus two novel additions. The code is fully written, syntax-checked, and structurally validated on CPU — it just needs H100s to produce a scored run. Here is the technical breakdown:

**Architecture (10-layer, 512-dim, 24.7M params):**
- 10 transformer blocks (up from baseline 9) with 8 attention heads, 4 KV heads (GQA), U-Net skip connections, tied embeddings
- MLP 3x expansion (hidden=1536) for more expressive nonlinear capacity, enabled by int6 quantization freeing ~4MB of artifact budget
- SmearGate: a learned per-dimension gate (init sigmoid(3.0) ≈ 0.95) that blends each token's embedding with the previous token's, injecting bigram context directly at the embedding layer for ~512 extra params
- Bigram hash embedding: 4096-bucket hash table (dim=128, projected to 512) mapping consecutive token pairs to learned features, providing direct token-pair signal at near-zero parameter cost

**Training:**
- STE int6 quantization-aware training: every forward pass fake-quantizes 2D weight matrices to int6 [-31,31] via straight-through estimator, so the model learns to be robust to its own post-training quantization (measured roundtrip gap: ~0.0001 BPB on test model)
- Muon optimizer with decoupled weight decay (0.01), momentum warmed from 0.92 to 0.99 over 1500 steps
- Orthogonal weight initialization on all non-zero-init matrices for uniform gradient flow from step 1
- Tuned LR schedule: matrix_lr=0.02, warmdown=3000 steps

**Evaluation:**
- Sliding window eval with stride=64: each scored token gets 960+ tokens of prior context, the single biggest free-win technique (~-0.034 BPB)

**Compression:**
- Int6 per-row quantization for all 2D weights, fp16 passthrough for tied embedding
- zstd-22 compression
- Total artifact: ~10.4MB with 5.6MB headroom — room to push to 12 layers or wider dims if step throughput allows

The plan is: run the default 10L config first, measure step time, then sweep to 12L 512d (12.3MB) or 10L 640d (15.6MB) to use the full budget. Each config has been pre-verified to fit under 16MB.

---

## What improvement you expect from additional compute (max 1,500 characters)

With compute credits, I plan a systematic 3-phase approach:

**Phase 1 — Baseline scored run (1-2 hours):** Run the 10L 512d MLP3x config on 8xH100 to establish a real val_bpb score. Based on the stacked techniques (sliding window, int6 QAT, SmearGate, orthogonal init, MLP 3x, 10 layers, Muon WD), I expect ~1.14–1.15 BPB, competitive with the current #1 (1.1556).

**Phase 2 — Model size sweep (3-4 hours):** The int6+zstd-22 pipeline gives massive headroom. I'll sweep 12L 512d (29.5M params, 12.3MB) and 10L 640d (38.2M params, 15.6MB) to find the Pareto-optimal config — more parameters directly improve loss, and I've pre-verified all configs fit under 16MB. I expect 12L to push past 1.14 BPB if step throughput stays under ~55ms.

**Phase 3 — Hyperparameter tuning and novel additions (3-4 hours):** Fine-tune learning rates, momentum schedule, and warmdown across 2-3 seeds to prove statistical significance (p < 0.01 at 0.005-nat improvement as required). Explore seq4096 long-context training to better match the sliding window distribution at eval. Expected final score: **1.13–1.14 BPB**, which would be a new SOTA.

Total estimated compute: ~10 hours of 8xH100 (~$200 in credits).
