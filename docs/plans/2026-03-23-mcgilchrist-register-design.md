# Experiment 2: McGilchrist Register Token — Design Document

**Date:** 2026-03-23
**Location:** `records/track_10min_16mb/2026-03-21_McGilchrist_Register/train_gpt.py`
**Baseline:** SOTA 1.1428 bpb (`2026-03-20_10L_Int5MLP_MuonWD04_SWA50`)

---

## 1. Thesis

Current transformer architectures are "left hemisphere machines" (McGilchrist): they decompose inputs into local features via causal attention and process them sequentially. What's missing is a **global context pathway** — the "right hemisphere" — that captures the holistic gestalt of the document and feeds it back to inform local processing.

**Hypothesis:** Adding a causal global context signal to each block, via cumulative mean + FiLM conditioning, improves compression by providing each position with document-level understanding that local causal attention alone cannot capture.

**Expected gain:** +0.003–0.015 bpb improvement over SOTA.

---

## 2. Architecture: Cumulative Mean + FiLM

### The mechanism (per block, after attention + MLP)

```
Block output x: (B, T, D=512)
        │
        ├──── [standard path] ────────────────────────────┐
        │                                                  │
        ▼                                                  │
  cumsum(x, dim=1) / positions   ← causal cumulative mean │
        │ (B, T, D)                                        │
        ▼                                                  │
  RMSNorm                                                  │
        │                                                  │
        ▼                                                  │
  Linear(D → bottleneck=8)  → ReLU → Linear(8 → 2D)       │
        │ (B, T, 2D)                                       │
        ▼                                                  │
  chunk → gamma (B,T,D), beta (B,T,D)                      │
        │                                                  │
        ▼                                                  │
  x_out = x * (1 + scale * gamma) + scale * beta    ◄─────┘
```

### Why cumulative mean?

1. **Fully causal** — position t only sees positions 0..t (no future leakage)
2. **O(T) parallel** — implemented via `torch.cumsum`, no sequential bottleneck
3. **Genuine global signal** — the running average captures document-level statistics that local attention windows miss
4. **Zero attention overhead** — no extra attention computation, just one cumsum + small MLP per block

### Why FiLM?

FiLM (Feature-wise Linear Modulation) is the minimal transformation that allows the global context to both **scale** (emphasize/suppress) and **shift** (bias) each feature dimension. It's multiplicative + additive, giving the model two degrees of freedom per dimension to integrate global context.

---

## 3. Initialization Strategy

**Critical design choice:** the model must start *identical* to SOTA so the register mechanism doesn't disrupt early training.

| Parameter | Shape per block | Init | Rationale |
|---|---|---|---|
| `film_down.weight` | (8, 512) | Kaiming uniform (default) | Non-zero so gradients flow through ReLU |
| `film_up.weight` | (1024, 8) | **Zeros** | Output gamma=0, beta=0 → identity transform |
| `register_scale` | (1,) | **0.01** | Small but non-zero so ∂loss/∂film_up ≠ 0 from step 1 |
| `register_norm` | (none) | N/A | RMSNorm has no learnable params |

**Gradient flow at step 0:**
- `film_up` is zero → gamma=0, beta=0 → output = x (identity)
- But `register_scale = 0.01` → `∂loss/∂(film_up) = 0.01 * (...)` → non-zero gradients
- The FiLM projection starts learning immediately while the output is still near-identity
- `register_scale` can grow as the model learns to use the global context

---

## 4. Parameter Budget

| Component | Params per block | 10 blocks total | Bytes (float16) |
|---|---|---|---|
| `film_down.weight` (8 × 512) | 4,096 | 40,960 | 81,920 |
| `film_up.weight` (1024 × 8) | 8,192 | 81,920 | 163,840 |
| `register_scale` (1) | 1 | 10 | 20 |
| `register_norm` | 0 | 0 | 0 |
| **Total** | **12,289** | **122,890** | **~240 KB** |

After zstd/zlib compression: estimated **~100–150 KB** additional (film_up starts as zeros → compresses extremely well; after training, values are small due to 0.01 scale).

**Budget impact:** SOTA artifact is ~15.5–16.0 MB. Adding ~150 KB compressed → well within 16 MB.

---

## 5. Quantization Path

Register parameters hit the `mixed_quantize_int6` passthrough path:

| Parameter | `numel` | `numel ≤ 8192`? | Quantization |
|---|---|---|---|
| `film_down.weight` | 4096 | Yes | **float16 passthrough** |
| `film_up.weight` | 8192 | Yes | **float16 passthrough** |
| `register_scale` | 1 | Yes | **float16 passthrough** |

All register params are kept at float16 precision — better than int5/int6/int8 used for the main model weights. This is a nice side effect of the small parameter count.

---

## 6. Optimizer Assignment

| Parameter | ndim | Control pattern? | Optimizer |
|---|---|---|---|
| `film_down.weight` | 2 | No | **Muon** (matrix_params) |
| `film_up.weight` | 2 | No | **Muon** (matrix_params) |
| `register_scale` | 1 | Yes ("register_scale") | **AdamW** (scalar_params) |

The film weights use Muon's Newton-Schulz orthogonalization. Despite the small bottleneck dimension (8), the Gram matrix is 8×8 which converges in 5 NS iterations.

---

## 7. torch.compile Compatibility

The register mechanism uses only standard ops:
- `torch.cumsum` — fully supported
- `torch.arange` — supported with static shapes (seq_len=2048 during training)
- RMSNorm → `F.rms_norm` — supported
- CastedLinear → `F.linear` — supported
- `torch.relu`, `.chunk`, multiply, add — all supported

The `if self.has_register:` branch is constant-folded by the compiler (the flag is set in `__init__` and never changes). Should work with `fullgraph=True`.

---

## 8. Testing Plan

### 8.1 Sanity check (1×GPU, ~2 min)

```bash
ITERATIONS=200 MAX_WALLCLOCK_SECONDS=120 REGISTER_BOTTLENECK=8 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**What to verify:**
1. No crash → torch.compile handles the register path
2. Training loss decreases normally (similar trajectory to SOTA baseline)
3. `register_params:122890` appears in logs (confirms params are created)
4. Model serialization succeeds (check "Total submission size" in logs)

### 8.2 Full run (8×H100, ~10 min)

```bash
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
SEED=42 REGISTER_BOTTLENECK=8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Success criteria:**
- `final_int8_zlib_roundtrip_exact val_bpb` < 1.1428 (beats SOTA)
- Total submission size ≤ 16 MB
- Training completes within 10 min wallclock

---

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Cumulative mean too simple (just averaging) | Medium | If no signal, upgrade to learned EMA or causal cross-attention |
| `torch.compile` graph break on `arange` | Low | Fall back to pre-computed position buffer |
| Exceeds 16 MB artifact | Low | Reduce bottleneck to 4, or apply to 5 blocks instead of 10 |
| Register hurts training (too much regularization) | Low | Zero-init + small scale ensures graceful degradation to SOTA |
| Muon doesn't work well for thin (8×512) matrices | Low | Switch film weights to AdamW if needed |

---

## 10. Connection to Experiment 3: Hermeneutic Depth Recurrence

The register mechanism in Exp 2 and depth recurrence in Exp 3 are complementary:

- **Exp 2 (Register):** Adds a *within-block* global context signal. Each block independently computes its own holistic summary and uses it for modulation.
- **Exp 3 (Depth Recurrence):** Adds *across-block* iterative refinement. A small set of blocks (e.g., 3-4) are reused N times with tiny modulation (FiLM or learned gating), implementing Gadamer's "hermeneutic circle" — each pass deepens understanding.

**Key design parallels:**
- Both use FiLM conditioning as the modulation mechanism
- Both need zero/small initialization to start from SOTA-like behavior
- Both add parameters that are small enough to fit in the 16 MB budget
- Both can be stacked: register tokens within each block + depth recurrence across blocks

**What Exp 3 can reuse from Exp 2:**
- The `CastedLinear` bottleneck → FiLM pattern
- The zero-init + small scale initialization strategy
- The control tensor pattern registration for quantization
- The optimizer group assignment logic

**What's different in Exp 3:**
- Instead of cumulative mean, the modulation signal comes from the *cycle index* (which re-read pass are we on?)
- Instead of adding params to each block, we *share* a small set of blocks and add per-cycle modulation
- The parameter savings from weight sharing could allow a larger/deeper effective model

---

## 11. Exp 2 v2 — Fixes for Next Run

**Run results (2026-03-21):**
- val_bpb: 1.16166728 (vs SOTA 1.1428)
- Artifact: 17,132,420 bytes (**over 16MB** ← main blocker)
- Training steps: 4,259 (vs ~4,917 for SOTA; lost 658 steps due to 141ms/step vs 122ms)
- Peak VRAM: **23.7 GB / 80 GB per H100 (~30% utilization)** ← large opportunity
- Eval time: 233s (3.9 min) ✓
- Training time: 599.8s ✓ (met wallclock cap exactly)

### Fix 1 — Artifact size (BLOCKER)

The artifact is 17.1MB because the code fell back to `zlib` instead of `zstd-22`. SOTA uses `zstd-22` which compresses ~20% better than zlib. Switching to zstd-22 should recover ~1–1.5 MB, bringing the artifact back under 16MB.

**Action:** Verify `import zstd` works in the RunPod environment and the compress path uses `zstd.compress(data, 22)`.

### Fix 2 — Step time (bf16 cumsum, already implemented locally)

`torch.cumsum` on fp32 is ~19ms overhead. Casting to bf16 before cumsum reduces this to ~5ms, recovering ~650 training steps.
- 141ms/step → ~127ms/step → ~4,724 steps in 10 min

### Fix 3 — Fill VRAM with bigger batch (OPPORTUNITY)

23.7 GB / 80 GB = 30% VRAM utilization. The H100 is largely idle between steps. Increasing `TRAIN_BATCH_TOKENS` fills the compute pipeline better and gives stronger gradient signal per unit of wallclock time.

**Applied:** Default changed from `786_432` → `1_572_864` (2×) in `train_gpt.py:68`.

Add this env var to unlock larger allocations without OOM from memory fragmentation:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Without `expandable_segments`, PyTorch pre-allocates fixed-size memory blocks. As allocations/frees happen, the pool fragments — PyTorch can see 50 GB free but report OOM because no single contiguous block is large enough. `expandable_segments:True` lets the allocator grow segments on demand, bypassing fragmentation entirely and allowing the full 80 GB to be used.

**Why batch size matters here specifically:** At 4,259 steps × 786K tokens/step = 3.35B tokens seen. At 2× batch (same steps): 6.7B tokens seen — double the training signal in the same wallclock time. The step count stays the same, but each gradient is computed over twice as much data.

**Potential to push further:** If VRAM allows (check peak after first run), could try `2_359_296` (3×) or increase `TRAIN_SEQ_LEN` from 2048 → 4096 for longer context.

### Fix 4 — Eval batch size

Eval used `eval_batch_seqs=32` (default). With 23.7 GB peak VRAM and only 3.9 min eval, we have room to set `eval_batch_seqs=256` which would drop eval to ~15 seconds and leave more headroom if needed.

### Summary: v2 launch command

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
SEED=42 REGISTER_BOTTLENECK=8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/register_v2.log
```

Note: `TRAIN_BATCH_TOKENS` is now hardcoded to `1_572_864` as the default in train_gpt.py. Override with `TRAIN_BATCH_TOKENS=786432` to revert if VRAM OOMs.

**Expected outcome:**
- Artifact ~15.5–16.0 MB (zstd-22 fix)
- ~4,724 training steps (bf16 cumsum fix)
- 2× training tokens per step → better gradient signal
- val_bpb target: < 1.1428 (beats SOTA)

---

## 12. MoE Insights (2026-03-23) — Impact on Exp 2

Community experiments (71 runs, 500 steps each) found:

- **4-expert MoE + leaky ReLU: -0.048 BPB** — clear winner at 500 steps
- **Untied factored embeddings (bn128): -0.031 BPB** — strong second
- Depthwise convolution: dead end (every variant hurts)
- Tied factored embeddings: catastrophic at small bottlenecks

**Impact on Exp 2 (Register mechanism):**

The register is orthogonal to MoE — it adds a *global context pathway* between positions, while MoE adds *capacity/specialization* within each position's MLP. They solve different problems and should combine additively.

**Plan:** Keep Exp 2 v2 clean (register only, with fixes from §11). If it shows improvement, Exp 3 becomes **Register + MoE** to test stacking. This isolates what each mechanism contributes rather than attributing a combined result to one idea.

**Also worth adding to Exp 2 v2:** Untied embeddings (-0.031 BPB) is a simple change that doesn't interact with the register mechanism at all. Low-risk addition to the next run.
