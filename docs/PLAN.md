# Parameter Golf — Competitive Submission Plan

## Objective

Beat the baseline **1.2244 BPB** by ≥0.005 nats on FineWeb validation (50k docs), with a 16MB artifact training in ≤10 min on 8×H100. Target: **~1.185–1.205 BPB** post-quantization.

---

## Baseline Anatomy

| Component | Value |
|-----------|-------|
| Architecture | 9-layer encoder-decoder (4 enc + 5 dec, U-Net skips) |
| Dims | 512 hidden, 8 Q heads, 4 KV heads, 1024 MLP hidden |
| Vocab | 1024 SentencePiece BPE, tied embeddings |
| Params | ~17M (int8 + zlib → 15.8MB compressed) |
| Code | ~48KB |
| Headroom | ~137KB unused in 16MB budget |
| Score | **1.2244 BPB** (pre-quant 1.2172, quant tax = 0.0072) |
| Training | 13,780 steps in 600s, 524K tok/step, ~7.2B tokens seen, 43.5ms/step |
| Eval | ~2 seconds (trivially fast vs the 10-min eval cap) |

---

## Three Key Exploits

### 1. Depth Recurrence → Wider Model at Same Depth (biggest win)

The baseline stores 9 unique transformer blocks. With weight sharing:
- Store **3 unique blocks**, loop **3×** = 9 effective layers (same depth)
- Per-loop learned scalar gates + tiny LoRA adapters (rank 4-8) to differentiate iterations
- Frees ~60% of block parameters → reinvest in **wider hidden dim (768 vs 512)**

Budget estimate at 768-dim, 3 unique blocks:
- tok_emb: 1024 × 768 = 786K params
- 3 blocks @ 768-dim: ~3.9M each = 11.8M params
- Per-loop adapters (8 adapters × ~4K): ~32K params
- Skip weights, norms, etc.: ~5K params
- **Total: ~12.6M unique params → ~12MB compressed, well under 16MB**

The freed 3-4MB can fund even wider dims (896?) or a 4th unique block.

### 2. QAT → Recover Quantization Tax (easy win, 0.005+ BPB)

The baseline loses 0.0072 BPB from post-training int8 quantization. QAT with straight-through estimator (STE) during training nearly eliminates this:
- Add `torch.fake_quantize_per_channel_affine()` in forward pass from ~step 5000 onward
- Train "through" the quantization noise so weights learn to be robust to int8 rounding
- Expected recovery: 0.005–0.007 BPB (nearly the full tax)

### 3. Test-Time Compute → Free BPB at Eval (unique exploit)

The README explicitly says:
- *"We encourage competitors to push the bounds of evaluation methods as aggressively as with training methods"*
- *"We won't accept submissions that take more than 10 minutes on 8xH100 to evaluate"*
- Baseline eval takes ~2 seconds. **We have ~598 seconds of unused eval compute.**

Options (ranked by safety/impact):
1. **More recurrence loops at eval** — train with 3 loops, eval with 6-8. More iterations → closer to fixed-point equilibrium → better predictions. Zero risk.
2. **Longer eval context** — eval at seq_len 2048-4096 instead of 1024. RoPE supports this. More context = better predictions.
3. **Test-time training** — fine-tune on the val data during eval (explicitly listed as encouraged). Compute gradients on early tokens, update weights, evaluate later tokens. Nuclear option but legal.

---

## Estimated BPB Impact

| Change | Est. BPB gain | Confidence |
|--------|--------------|------------|
| QAT (recover quant tax) | -0.005 to -0.007 | High |
| Depth recurrence + wider model | -0.010 to -0.020 | Medium-High |
| Extra eval loops | -0.002 to -0.005 | Medium |
| Longer eval context | -0.001 to -0.003 | Medium |
| Hyperparam tuning | -0.002 to -0.005 | Medium |
| **Combined** | **-0.020 to -0.040** | |
| **Target BPB** | **~1.185 – 1.205** | |

---

## Phased Execution

### Phase 1: Prototype on MLX (Local Mac, Free)
- Implement depth recurrence (3 blocks × 3 loops) in `train_gpt_mlx.py`
- Add per-loop learned scalars + tiny LoRA adapters
- Widen model to 768-dim, verify compressed size fits 16MB
- Validate directional BPB improvement vs baseline
- **Exit criteria**: recurrence working, compressed artifact under 16MB, BPB trending better than baseline

### Phase 2: Validate on CUDA (1×H100, ~$85)
- Port architecture changes to `train_gpt.py`
- Add QAT with STE (fake quantization in forward pass, enable at step ~5000)
- Verify on 1×H100 with 10-min wallclock cap
- Compare pre-quant and post-quant BPB to confirm QAT recovers the tax
- **Exit criteria**: CUDA training working, QAT reducing quant tax, BPB < 1.215

### Phase 3: Hyperparameter Sweep (1×H100, ~$42)
- Set up autoresearch loop (Karpathy pattern): agent edits config, runs 10-min train, measures BPB, keeps improvements
- Sweep: learning rates, loop count, LoRA rank, width, warmdown schedule, entropy regularization weight
- Run overnight (~100 experiments)
- **Exit criteria**: best config identified, stable BPB improvement

### Phase 4: Eval-Time Tricks (1×H100, ~$12)
- Implement extra recurrence loops at eval (3 train → 6-8 eval)
- Implement longer eval context (2048-4096 seq_len)
- Optionally: test-time training (gradient steps on val data before measuring)
- **Exit criteria**: eval tricks add measurable BPB improvement

### Phase 5: Submit (8×H100, ~$120)
- Final validation on 8×H100 with exact submission config
- Multiple seeds for statistical significance (p < 0.01)
- Prepare submission folder: README.md, submission.json, train.log, train_gpt.py
- PR to `records/track_10min_16mb/`
- **Exit criteria**: BPB beats 1.2244 by ≥0.005 nats, p < 0.01, artifact < 16MB

---

## Compute Requirements

### Development (1×H100 on RunPod)
- ~$2.50/hr (80GB SXM)
- Architecture experiments: ~30 runs × 10 min = 5 hrs = $12
- Hyperparameter sweeps: ~100 runs × 10 min = 17 hrs = $42
- Autoresearch overnight loops: 12 hrs = $30
- **Subtotal: ~$85**

### Validation (8×H100 on RunPod)
- ~$24/hr (8×H100 SXM pod)
- Validation runs: ~20 runs × 10 min = 3.3 hrs = $80
- Final submission (multiple seeds): ~10 runs × 10 min = 1.7 hrs = $40
- **Subtotal: ~$120**

### Total: ~$200

**Priority: Apply for the OpenAI/RunPod compute grant** via https://openai.com/index/parameter-golf/#credit-form to make this free.

### Local (Free)
- Mac with Apple Silicon: MLX script for rapid architecture iteration
- ~10-50× slower than H100 but fine for directional validation
- All Phase 1 work happens here — minimize cloud spend

---

## Compute Grant Application Strategy

When applying via the credit form, we should have ready:
1. **Clear technical approach** — depth recurrence + QAT + eval-time compute (this plan)
2. **Directional results from MLX** — showing the approach is promising before requesting H100 time
3. **Estimated compute needs** — ~200 1×H100-hours + ~50 8×H100-hours
4. **Submission timeline** — concrete phase plan with milestones

Iterate as much as possible on MLX first, arrive at the grant application with validated architecture choices and only need cloud compute for final training and statistical validation.

---

## Key Technical Decisions Still Open

1. **Loop count**: 3×3 vs 4×3 vs 3×4 — need MLX experiments
2. **Width**: 768 vs 896 vs 640 — depends on compressed size budget
3. **LoRA rank**: 4 vs 8 vs 16 — tradeoff between adaptation quality and param cost
4. **QAT schedule**: enable at step 5000? 3000? From start?
5. **Eval loops**: how many extra loops before diminishing returns?
6. **Entropy regularization**: weight and schedule TBD
7. **U-Net skip compatibility**: how do skips work with looped blocks?

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Depth recurrence doesn't improve BPB at this scale | High | Fall back to QAT + hyperparam tuning alone |
| QAT destabilizes training | Medium | Enable QAT late (last 30% of training) |
| Compressed artifact exceeds 16MB | Medium | Reduce width, increase recurrence ratio |
| Extra eval loops slow eval past 10 min | Low | Profile and cap loop count |
| Grant not approved | Medium | Budget ~$200 personal spend |
