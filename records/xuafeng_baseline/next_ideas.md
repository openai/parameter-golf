# Parameter Golf - Ideas for Future Runs

## Baseline Summary

Our first run achieved **1.3274 BPB** on 1x H100 SXM (1404 steps in 600s). The current leaderboard SOTA is **~1.1428 BPB**. That's a gap of ~0.185 BPB to close.

---

## Idea 1: Scale to 8x H100 (High Priority)

**What**: Run baseline with `--nproc_per_node=8` on 8x H100 SXM.

**Why**: The competition target is 8x H100. With 8 GPUs and linear scaling, we'd complete ~11,200 steps instead of 1,404 in the same 600s wall time. Loss was still decreasing at step 1404, so more steps alone should significantly improve BPB.

**Expected impact**: Moderate (~0.05–0.10 BPB improvement from longer training alone).

**Cost**: ~$3.59 per run.

---

## Idea 2: Quantization-Aware Training (QAT) — int6/int5

**What**: Train with quantization in the loop so the model learns to compensate for precision loss. The top 3 leaderboard entries all use int5 or int6 quantization.

**Why**: Our raw model is 67 MB but compresses to 13.7 MB with int8+zlib. Switching to int5/int6 could allow a larger model (more params) while staying under 16 MB. More params at lower precision often beats fewer params at higher precision.

**Approach**:
1. Implement fake-quantize forward passes during training (STE / straight-through estimator)
2. Start with int6 (simpler) then try int5
3. Quantize weights per-channel with learned scale factors

**Expected impact**: High (~0.05–0.10 BPB). This is the single most impactful technique on the leaderboard.

---

## Idea 3: Wider MLP (3x Expansion)

**What**: Increase MLP expansion from 2x to 2.6x–3x.

**Why**: Top entries use 3x MLP. Wider MLPs increase model capacity per layer. Combined with quantization, the compressed size can stay under 16 MB while packing more effective parameters.

**Approach**:
- Modify `mlp_expansion` in Hyperparameters from 2.0 to 3.0
- May need to reduce number of layers to stay within size budget
- Try 7 layers × 3x MLP vs current 9 layers × 2x MLP

**Expected impact**: Moderate (~0.02–0.05 BPB).

---

## Idea 4: More Layers with Smaller Width

**What**: Experiment with deeper, narrower architectures (e.g., 11–12 layers at width 448 vs 9 layers at 512).

**Why**: The leaderboard leader uses 10L. Depth helps with compositional reasoning. With QAT compressing weights, we can afford more layers.

**Expected impact**: Low-moderate, architecture-dependent.

---

## Idea 5: SmearGate / Gating Mechanisms

**What**: Add gating to MLP or attention outputs (e.g., SwiGLU, SmearGate as used by the #2 leaderboard entry).

**Why**: Gating adds minimal parameters but significantly improves expressiveness. SwiGLU is standard in modern LLMs for good reason.

**Approach**:
- Replace standard MLP with SwiGLU: `output = (Wx * sigmoid(Vx)) @ W2`
- The gate projection (V) adds ~50% more MLP params but often yields disproportionate quality gains

**Expected impact**: Moderate (~0.02–0.04 BPB).

---

## Idea 6: BigramHash / Auxiliary Heads

**What**: Add a lightweight bigram hash table or auxiliary prediction head as used by the #1 entry.

**Why**: A bigram hash captures local token co-occurrence patterns essentially for free (small memory, no backprop needed). It complements the transformer's global attention.

**Approach**:
- Maintain a hash table of bigram log-probabilities
- Interpolate transformer predictions with bigram predictions at inference
- The hash table is tiny and compresses well

**Expected impact**: Low-moderate (~0.01–0.03 BPB), but nearly free in terms of compute.

---

## Idea 7: Learning Rate & Schedule Tuning

**What**: Tune learning rate, warmup steps, and decay schedule.

**Why**: The baseline uses fixed LR settings (embed_lr=0.05, matrix_lr=0.04). With 8 GPUs and more steps, the optimal schedule may differ. Cosine decay with longer warmup may help.

**Approach**:
- Grid search: matrix_lr in {0.03, 0.04, 0.05, 0.06}
- Warmup steps in {20, 50, 100}
- Compare cosine vs linear decay

**Expected impact**: Low (~0.01–0.02 BPB), but easy to try.

---

## Idea 8: Mixed-Precision Embeddings

**What**: Keep embeddings at higher precision (fp16/bf16) while quantizing transformer weights to int5/int6.

**Why**: Embeddings are the first and last layer — quantization errors there propagate through the entire model. Leaderboard entries treat embedding precision separately.

**Expected impact**: Low (~0.005–0.01 BPB), but almost free.

---

## Proposed Run Plan

| Run | try_id | Config | Goal |
|-----|--------|--------|------|
| 2 | `8gpu_baseline_try2` | 8x H100, default config | Measure 8-GPU baseline BPB |
| 3 | `8gpu_3xmlp_try3` | 8x H100, 3x MLP expansion | Test wider MLP |
| 4 | `8gpu_swiglu_try4` | 8x H100, SwiGLU gating | Test gated MLP |
| 5 | `8gpu_int6qat_try5` | 8x H100, int6 QAT | Test quantization-aware training |
| 6 | `8gpu_combo_try6` | 8x H100, int6 QAT + 3x MLP + SwiGLU | Combine best techniques |
| 7 | `8gpu_bigram_try7` | 8x H100, combo + bigram hash | Full stack attempt |

Each run costs ~$3.59 (8x H100, 10 min). Total budget for 7 runs: ~$25.

---

## Target

Close the gap from **1.327 BPB** → **<1.15 BPB** through the combination of:
1. 8x GPU scaling (more steps)
2. QAT int6/int5 (more effective params per byte)
3. Wider MLP + gating (better architecture)
4. Bigram hash (free local statistics)
