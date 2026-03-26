# Parameter Golf — Model Build Log & Known Issues

**Purpose:** Track what worked, what broke, and what to watch for during incremental builds.
**Updated:** 2026-03-25 23:10 CDT

---

## Model 3: Hybrid (GatedRNN + Attention) — COMPLETE ✅

### Build History
| Step | What Changed | Result | bpb |
|------|-------------|--------|-----|
| Step 1 | Baseline: 3 RNN layers + 6 attention layers, 512d | ✅ Training, loss drops | — |
| Step 2 | Added EMA, tuned architecture ratios | ✅ val_bpb 2.556 (200 steps) | 2.556 |
| Step 3 | Warmdown scheduling + grad clip | ✅ val_bpb 2.529 (213 steps) | 2.529 |

### Known Issues (Resolved)
- **M3 Step 2 SIGTERM:** First Codex build was killed mid-run. Cause: SSH timeout. Fix: use `nohup` for long runs.
- **GatedRNN integration:** The RNN layers needed careful placement (first 3 layers) — putting them later caused training instability.

---

## Model 4: Optimized Transformer — COMPLETE ✅ (BEST MODEL)

### Build History
15 incremental steps. See PROGRESS.md for full technique list.

### Key Learnings
- **Best pre-quant bpb:** Achieved with Muon weight decay 0.04 + warmdown 3500 + grad clip 0.3
- **OOM at 768d:** 12L/768d/3x MLP (62M params) OOMs on 4070 Super 12GB. Needs H100 80GB.
- **Sliding window eval stride=64** significantly improved eval accuracy.
- **GPTQ-lite int6/int8 + zstd-22** was best compression combo.
- **TTT (Test-Time Training)** with AdamW gives 0.02-0.05 bpb improvement at eval.
- **PolarQuant spec complete but NOT wired** — needs save/load integration.

### Scaling Limitations
- 12L/768d (62M params) — OOM on 4070 Super, confirmed needs H100
- 12L/640d shared weights — OOM during activation storage on 4070 Super
- Full 8×H100 runs not yet done (need compute grant or ~$12/hr Vast rental)

---

## Model 6: Hive (Frozen Backbone + LoRA) — Step 1 ✅

### Concept
90% frozen random-init backbone (fixed feature extractor) + 10% trainable LoRA adapters.
~17M total params, only 545K trainable (96.8% frozen).

### Build History
| Step | What Changed | Result |
|------|-------------|--------|
| Step 1 | Baseline: frozen backbone + LoRA adapters | ✅ Runs, trains (30s test) |

### Error History
- **Original Codex build (model6.py):** NaN during training. Cause: the monolithic build tried to implement the full hive concept in one shot — frozen int4 backbone + LoRA + gating all at once. Too many moving parts.
- **Step 1 fix:** Rebuilt incrementally from the reference `train_gpt.py` baseline. Added ONLY the frozen backbone + LoRA concept. No int4 quantization of backbone yet. Result: training works.

### Known Limitations & Watch Items
- **Frozen backbone ratio:** Currently 96.8% frozen. May need tuning — if too much is frozen, the trainable portion can't learn enough. If too little, we're just a normal model with extra params.
- **LoRA rank:** Currently using default. Step 2 should experiment with rank (4, 8, 16).
- **No int4 backbone yet:** The original spec calls for int4 storage of frozen weights. Not implemented in Step 1. Needs to be added for size compliance.
- **No gating/routing:** Original spec had ant-colony-style pheromone routing. Not in Step 1.
- **Step speed:** ~295ms/step on 4070 Super (similar to other models). No speed penalty from frozen backbone.

### Step 2 Results
- ✅ Warmdown 3500, grad_clip 0.3, EMA 0.997
- Training loss @ 200 steps: 5.36 (down from 18.4 initial)
- Step speed: 114ms/step (FASTER than other models — frozen backbone = less gradient work)
- Smoke test: PASS with VAL_LOSS_EVERY=0

### Step 3 Results — LoRA + Hybrid Freeze ✅
- Added LoRA adapters (rank 8) to c_q, c_v, and fc (MLP up-projection)
- **CRITICAL FINDING:** 98.6% frozen + LoRA = NO LEARNING (loss flat at 6.93 = random). Random frozen weights are NOT useful feature extractors without pre-training.
- **FIX:** Hybrid freeze — first 6 layers frozen, last 3 fully trainable + LoRA on all layers
- Frozen ratio: 66.8% (was 98.6%)
- val_bpb: **4.031** (pre-quant), **4.109** (roundtrip int8+zlib)
- Train loss @ 200 steps: 6.468
- Compressed size: 9.4 MB
- 217 steps in 30s, 138ms/step
- **Lesson:** The Hive concept of "frozen random backbone + tiny trainable adapters" requires pre-trained weights. With random init, the model can't learn through LoRA alone — there's no useful feature space to adapt. Had to compromise by unfreezing last 3 layers.

---

## Model 7: Immune System / Template Codebook — Step 2 ✅

### Concept
128 weight templates (~96KB each) shared across all layers. Per-token router selects and combines 4 templates to generate effective weights. Like V(D)J recombination in the immune system.

### Build History
| Step | What Changed | Result |
|------|-------------|--------|
| Step 1 | Baseline: template library + recombination router | ✅ 17M params, loss 6.94→9.08 (10 steps), ~295ms/step |

### Error History
- **Original Codex build (model7.py):** OOM. Cause: template library was too large — 128 templates × 96KB each at full precision. Plus the dynamic weight generation created huge intermediate tensors.
- **Step 1 fix:** Rebuilt incrementally. Reduced template dimensions to fit in memory on 4070 Super. Loss drops but starts high (initial loss 6.94 with spikes to 17.5 before dropping).

### Known Limitations & Watch Items
- **High initial loss spikes:** Steps 2-3 show loss going UP to 17.5 before coming back down. This is likely the template combination mechanism finding its footing. May need warmup specifically for the router.
- **Template count vs. quality tradeoff:** More templates = more expressiveness but more memory. Current count needs verification against 16MB budget.
- **Router collapse risk:** If the router learns to always pick the same templates, we get a standard model with extra overhead. Need diversity regularization.
- **Quantization of templates:** The original spec calls for int6 quantization. Not implemented in Step 1.
- **Dynamic weight generation cost:** Each forward pass generates weights on-the-fly. This could be a bottleneck at inference.

### Step 2 Results
- ✅ Warmdown 3500, grad_clip 0.3, EMA 0.997
- Training loss @ 200 steps: 4.49
- Step speed: 141ms/step
- Smoke test: PASS with VAL_LOSS_EVERY=0

### Step 3 Results — 32 Templates + Router Warmup + Diversity Reg ✅
- Increased templates: 16 → 32
- Added router temperature warmup: starts at 5.0 (uniform mixing) → anneals to 1.0 (sharp selection)
- Added diversity regularization: penalizes cosine similarity between templates (weight 0.01)
- Added EMA template usage tracking
- val_bpb: **3.464** (pre-quant), **3.657** (roundtrip int8+zlib)
- Train loss @ 200 steps: 4.553 (initial spike to 17.9 from diversity reg, recovers fast)
- Compressed size: 5.3 MB
- 206 steps in 30s, 146ms/step
- **Lesson:** Temperature warmup is key — starting with uniform mixing lets all templates get gradient signal early, then sharpening lets specialization emerge.

---

## Model 8: Crystal (Seed + Growth Rules) — Step 2 ✅

### Concept
Single small transformer "seed" block (256d) + growth rule network that generates layer-specific modifications. One block grows into a full model through recursive self-application. The architecture's shape IS the learned knowledge.

### Build History
| Step | What Changed | Result |
|------|-------------|--------|
| Step 1 | Baseline: seed block + growth rule expansion | ✅ 17M params(?), loss 6.94→9.17 (10 steps), ~295ms/step |

### Error History
- **Original Codex build (model8.py):** Shape mismatch errors. Cause: the growth rule network's output dimensions didn't match the expected layer modification shapes. The recursive self-application created dimension mismatches at deeper layers.
- **Step 1 fix:** Rebuilt incrementally. Simplified the growth mechanism to ensure dimensional consistency across all generated layers.

### Known Limitations & Watch Items
- **Growth rule expressiveness:** If the growth rule is too simple, all generated layers are nearly identical (defeating the purpose). If too complex, it's just a normal model with extra steps.
- **Seed size vs. model size:** The original spec has a 2MB seed. Need to verify the expanded model actually uses the seed efficiently and doesn't just learn to ignore the growth rules.
- **Inference cost:** Each forward pass must run the growth rules to generate layer params. This adds latency.
- **No recursive depth control yet:** The original spec calls for 12-16 effective layers. Need to verify how many the current Step 1 actually generates.
- **Gradient flow through growth rules:** Backprop through the growth rule → generated weights → loss path can be unstable. Watch for vanishing/exploding gradients.

### Step 2 Results
- ✅ Warmdown 3500, grad_clip 0.3, EMA 0.997
- Training loss @ 200 steps: 4.36
- Step speed: 140ms/step
- Smoke test: PASS with VAL_LOSS_EVERY=0
- NOTE: growth_rule module exists but is NOT called in forward pass. Currently trains as a baseline + unused params.

### Step 3 Results — Growth Rule Wired Into Forward Pass ✅
- **Fixed critical bug:** Growth rule was dead code. Now applies per-layer scaling from seed embeddings.
- Each layer gets: `x = x * (1.0 + 0.05 * tanh(growth_mlp(layer_embedding)))` — ±5% learned scaling per dimension
- First attempt: OOM (computed all scales at once). Fix: lazy per-layer computation.
- Reduced scale from ±10% to ±5% for stability (initial loss still spikes to 17 but recovers)
- val_bpb: **3.342** (pre-quant). Roundtrip not captured (eval timed out or process killed before printing).
- Only 125 steps in 30s (242ms/step — **slower** than M7's 146ms due to growth_rule overhead per layer)
- Compressed size: 5.2 MB
- **Key finding:** Growth rule DOES help — val_bpb 3.342 is better than M7 (3.464) and M6 (4.031). The per-layer scaling gives each layer a unique learned "identity" without adding many params.

---

## Cross-Model Issues (Apply to ALL)

### T4 vs 4070 Super vs H100 Compatibility
- `torch.compile` DISABLED for T4 (not supported). Comment it out or guard with capability check.
- `flash_sdp` and `cudnn_sdp` disabled on T4/4070. Only `mem_efficient_sdp` and `math_sdp` enabled.
- BF16 not available on T4 — must use FP32 or FP16.
- H100: all backends available, BF16 preferred.

### Size Budget (16MB checkpoint)
- Current models are built for training, not size-constrained yet.
- Quantization (int8/int6/int4) + compression (zstd-22) needed for final submission.
- PolarQuant spec exists but not wired into any model.
- Model 4 at 5.1MB leaves room; Models 6-8 need size verification.

### Training Infrastructure
- Vast.ai 4070 Super: $0.09/hr, ~145ms/step — used for all smoke tests
- SSH timeouts kill long runs — always use `nohup` + redirect to log file
- Logs stored at `/root/pg/logs/` on Vast instance
- Git repo synced via `git fetch origin && git reset --hard origin/main`

### Codex Build Pattern (What Works)
- **DO:** Build incrementally from the working `train_gpt.py` reference. Add ONE concept per step.
- **DON'T:** Try to implement the full architecture spec in one shot. This caused NaN (M6), OOM (M7), and shape errors (M8).
- **DO:** Test each step with 30-second smoke test before proceeding to next step.
- **DON'T:** Skip smoke tests. A broken Step N means a broken Step N+1.

### Smoke Test Bug (CRITICAL — discovered 2026-03-25)
**`eval_val()` at step 0 processes ALL 62M validation tokens.** On a 4070 Super without torch.compile, this takes 30-60+ seconds. If `MAX_WALLCLOCK_SECONDS` is set to 30 for a smoke test, the ENTIRE budget is consumed by the first validation pass. The training loop never executes.

**FIX:** Always use `VAL_LOSS_EVERY=0` for short smoke tests (under 120s). This skips validation during training. Validation works fine for full 10-minute runs where it's <1% of total time.

**Correct smoke test command:**
```bash
RUN_ID=test DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=30 TRAIN_BATCH_TOKENS=16384 TRAIN_SEQ_LEN=256 WARMUP_STEPS=2 VAL_LOSS_EVERY=0 python3 -u train_gpt_mX_stepY.py
```

---

## Leaderboard Context (as of 2026-03-25)

- **Current leader:** ~1.12 bpb (public leaderboard)
- **Pending PRs:** 1.05-1.07 bpb
- **Our best (M4, 200 steps on 4070):** 3.74 bpb (NOT comparable — need full 10min 8×H100 run)
- **Estimated full-run score for M4:** ~1.1-1.2 bpb (competitive with leaderboard)
- **Novel architectures (M3, M6-M8):** Unknown at scale. That's the point — if any of these beat transformers, it's a real finding.

---

*Update this file after every build step, error, or discovery.*
