# Parameter Golf — Experiment Ideas

Running list of directions worth trying. Added during the comparative read of `train_gpt.py`.

---

## 1. U-Net skip structure / depth recurrence

The baseline already has a U-Net encoder-decoder shape with skip connections (first half feeds into second half in reverse). This is already better than plain sequential — but there's room to push further:

- **Vary the encoder/decoder split ratio** — baseline is 50/50. What if 1/3 encoder + 2/3 decoder?
- **Depth recurrence** — reuse the same Block weights multiple times (e.g., run 4 unique blocks × 3 passes each = 12 "effective" layers at the param cost of 4). More compute, far fewer stored params → direct 16 MB win.
- **Skip weight schedules** — the `skip_weights` are learned per-layer. Could try initializing them differently or constraining them.

**Why interesting for PG:** depth recurrence is probably the highest-leverage param-saving trick available. You get deep effective compute at shallow param cost. Several top leaderboard entries use layer recurrence.

**Refs:** U-Net (Ronneberger 2015), modded-nanoGPT speed-run experiments.

---

## 2. Quantization — push below INT8

The baseline uses INT8 quantization at save time (8 bits, 4× compression vs fp32). The 16 MB budget is measured on the compressed artifact, so more aggressive quantization = more parameters fit = potentially better model.

**Options, roughly easiest → hardest:**

- **INT8 (baseline)** — already in the reference trainer. ~50M params fit in 16 MB.
- **INT4** — 8× compression vs fp32. ~100M params in 16 MB. Post-training quantization (PTQ) is feasible — quantize after normal training. AWQ or GPTQ are the SOTA approaches. Risk: outlier weights hurt quality, need per-channel scaling.
- **INT4 + quantization-aware training (QAT)** — train with fake quantization in the loop so the model learns to be robust to INT4 rounding. Better quality than PTQ but more complex to implement.
- **Ternary / INT2** — `{-1, 0, 1}` weights, 16× compression. ~256M params in 16 MB. Needs training from scratch with quantization in mind (BitNet b1.58 style). High risk, high reward — this is frontier research territory.
- **Mixed precision** — INT4 for large weight matrices (most of the params), INT8 for sensitive layers (embeddings, norms, first/last layers). Good middle ground.

**Key insight:** individual weight precision loss gets averaged away across large dot products — errors partially cancel. This is why aggressive quantization works better than you'd expect.

**Main challenge:** outlier weights. A few large values force a coarse scale for the whole tensor, wasting precision on small values. Solutions: per-channel scales, weight clipping, SmoothQuant.

**Recommended starting point:** try INT4 PTQ after a normal INT8 training run — measure the val_bpb hit. If < 0.01 nats degradation, it's worth pursuing QAT.

**Refs:** BitNet b1.58 (Ma et al. 2024), AWQ (Lin et al. 2023), GPTQ (Frantar et al. 2022).

---

## 3. Tokenizer vocab size (SP1024 → SP4096 / SP8192)

The baseline uses SentencePiece with vocab=1024. Top leaderboard entries use SP4096 or SP8192.

**The tradeoff:**
- Larger vocab → each token covers more bytes → fewer tokens per document → lower val_loss
- Larger vocab → bigger embedding table → more params in 16 MB budget
- For val_bpb to improve, `bits_per_token` must drop faster than `tokens_per_byte` drops

**Why it's worth trying:** several top leaderboard entries explicitly list SP4096/SP8192 in their submission names. It's one of the cleaner levers — change one config value, retrain, measure bpb.

**Interaction with weight tying:** if embeddings are tied (same matrix for input + output), the embedding table is `vocab_size × model_dim`. At SP8192 with dim=512: 8192 × 512 = 4M params = 4MB in fp32, 2MB in bf16, 0.5MB in INT4. Still manageable.

**Recommended:** try SP4096 first — it's the most common upgrade on the leaderboard and the param cost is moderate.

---

## 4. Staged Hyperparameters (change config mid-training)

The idea: different training stages need different configs. Instead of one fixed set of hyperparameters, switch them at specific steps.

**Proven approaches (top submissions use these):**

- **Recurrence activation at step N** — train baseline for ~2000 steps, then activate depth recurrence. Model learns basics first, then gets extra effective depth to refine. Top submission uses `recur_start_step=2016`. Implementation: add `RECUR_START_STEP` env var, check `step > threshold` in GPT.forward to decide whether to use block_schedule or plain sequential.

- **QAT warmdown** — train normally, then switch to quantization-aware training (fake INT8 in the loop) for the last 15% of steps. Model learns to be robust to INT8 rounding before we actually quantize. Top submissions use `late_qat_threshold=0.15`.

**Novel ideas to try:**

- **q_gain schedule** — start at q_gain=1.5 (soft attention, stable), ramp to 5.25 (sharp attention, precise) over training. Could use cosine schedule or step function.

- **Progressive recurrence** — start with 0 recurring layers, add one every 500 steps. Layer 3 recurs at step 500, layer 4 recurs at step 1000, etc.

- **LR per layer group** — different learning rates for early vs late layers. Early layers converge faster (simpler features), late layers need more steps.

**Priority:** Recurrence activation mid-training is the highest-impact and easiest to implement. One env var + a few lines of code.

---

## 5. Early Training Acceleration

The 600s wall means every wasted step costs bpb. If we can make early training converge faster, we get more "useful" steps before the wall.

**Approaches:**

- **Higher LR early** — push matrix_lr higher for first 1000 steps, then decay. QK-norm provides stability safety net. Try MATRIX_LR=0.06 or 0.08 with earlier decay.

- **Shorter Muon warmup** — default `muon_momentum_warmup_steps=500`. Top submission uses 1500 (slower). Try 200 (faster) — get to full momentum sooner, risk instability.

- **Progressive depth** — start with fewer layers (5L), train 1000 steps, then grow to 9L. Early layers already warmed up, new layers init from zero. Related to staged hyperparameters (idea #4). More complex to implement.

- **Larger batch early** — bigger batches = more stable gradients = can use higher LR. Reduce batch later for finer updates. Would need dynamic `train_batch_tokens`.

- **Curriculum learning** — train on shorter sequences first (256 tokens), increase to 1024 mid-training. Shorter sequences = faster steps = more steps early.

**Quick wins to try:** shorter Muon warmup (env var only) and higher early LR (env var only). Progressive depth needs code changes.

---

## 6. Parallel Residuals (GPT-J style)

Instead of sequential attention → MLP, both read from the same input and outputs are summed. Proven in top submissions (+0.004 bpb). The top submission activates this from layer 7 onward.

**Implementation:** In Block.forward, instead of `x = x + attn(norm(x)); x = x + mlp(norm(x))`, do `x = x + attn(norm(x)) + mlp(norm(x))`. ~5 lines of code. Slight speed improvement too since attention and MLP can partially overlap on GPU.

**Priority:** High — proven, simple, stacks with recurrence.

---

## 7. EMA (Exponential Moving Average) of Weights

Keep a running average of model weights during training. At eval time, use the EMA weights instead of the final weights. Smooths out noise from the last few training steps.

**Implementation:** After each optimizer step, update `ema_weights = decay * ema_weights + (1 - decay) * weights`. Top submissions use `decay=0.9965`. At save time, use EMA weights for quantization.

**Cost:** ~2× memory (storing two copies of weights) but no extra compute. Free bpb improvement, typically 0.002-0.005.

**Priority:** High — almost free, every top submission uses it.

---

## 8. Sliding Window Eval at Longer Context

Training at seq_len=1024 but evaluating at seq_len=2048 or 4096 with sliding window. RoPE supports arbitrary lengths (no block_size ceiling), so longer eval context should improve val_bpb without changing training at all.

**Implementation:** Just change `EVAL_SEQ_LEN=2048` at eval time. The baseline already supports this via `eval_seq_len` if the hyperparameter exists, or we add it.

**Priority:** Medium — free improvement at eval time, no training cost. But needs code check to confirm eval supports different seq_len.

---

## 9. Weight Decay Tuning

Default weight decay is implicit in the optimizer config. Top submissions tune `muon_wd=0.04` and `adam_wd=0.04`. Some use `muon_wd=0.095` (much higher).

Weight decay acts as regularization — prevents weights from growing too large. Higher WD = smaller weights = potentially better generalization but worse training loss. The optimal value depends on model size and training length.

**Experiments to try:**
- MUON_WD=0.02 (less regularization)
- MUON_WD=0.06 (more)
- MUON_WD=0.095 (top submission value)

**Priority:** Medium — easy env var change, potentially significant. The jump from 0.04 to 0.095 in top submissions is huge.

---

## 10. Gradient Clipping

Default `grad_clip_norm=0.0` (disabled). Top submissions use `grad_clip_norm=0.3` or `1.0`. Clipping prevents occasional large gradients from destabilizing training.

With Muon + QK-norm we already have stability, but gradient clipping is an additional safety net that lets you push LR higher. The interaction between grad clipping and Muon is worth exploring.

**Try:** GRAD_CLIP_NORM=0.3 (moderate), GRAD_CLIP_NORM=1.0 (light).

**Priority:** Low-medium — easy to try, effect may be small with existing stability mechanisms.

---

## 11. LeakyReLU Instead of ReLU²

Top submissions switched from `relu(x).square()` to `leaky_relu(x, negative_slope=0.5).square()`. LeakyReLU allows small negative values through (scaled by 0.5) instead of hard-zeroing them.

**Why it might help:** ReLU² is very aggressive at killing small activations. LeakyReLU² preserves some gradient flow through negative values, potentially helping early training convergence (ties into idea #5).

**Implementation:** Code change in MLP.forward — `torch.relu` → `F.leaky_relu(x, negative_slope=0.5)`. The top submission uses exactly `negative_slope=0.5`.

**Priority:** Medium — code change needed but simple. Top submission uses it, so it's proven.

---

## 12. Per-Layer Configuration (different methods at different depths)

Different layers do different things — early layers handle simple patterns (syntax, token copying), middle layers store knowledge, late layers do complex reasoning. So why treat them all the same?

**Already proven in top submissions:**
- Parallel residuals only from layer 7+ (early layers stay sequential)
- Recurrence only on middle layers (3,4,5) — not early or late
- Different learning rates for different layer groups

**Ideas to explore:**
- Different activation functions per layer (ReLU² early, LeakyReLU late?)
- Different mlp_mult per layer (wider MLP in middle layers where knowledge is stored?)
- Different num_kv_heads per layer (more KV heads in early layers, fewer late?)
- Gradually increasing q_gain through layers (soft attention early, sharp late)
- Different RoPE base per layer (different position sensitivity at different depths)

**Why this is interesting:** The transformer isn't a homogeneous stack — each layer learns different representations. Treating them identically is leaving performance on the table. The per-layer configuration space is huge and largely unexplored.

**Implementation:** Add per-layer config lists instead of single values. E.g., `MLP_MULT_PER_LAYER=2,2,2,4,4,4,2,2,2`.

---

*(More ideas to be added)*
