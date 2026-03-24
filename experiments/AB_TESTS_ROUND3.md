# A/B Tests Round 3 — Post-1.1130 Optimization

## Current Best

**1.1130 val_bpb** | stride=76 | per-window SGD TTT | 14L | GPTQ int6 | EMA 0.997 | 575s eval | 15.87MB

## Best Clean PRs for Reference


| PR   | BPB    | Key technique                         |
| ---- | ------ | ------------------------------------- |
| #569 | 1.1175 | VRL + LeakyReLU² + Full GPTQ (no TTT) |
| #545 | 1.1179 | Int5 GPTQ + 33.6M model               |
| #589 | 1.1178 | Soft-Round QAT + Backward-Looking TTT |
| #505 | 1.1181 | SwiGLU + VE128 (no TTT)               |


---

## Experiments Queue

### 1. Value Residual Learning (VRL)

**Status: RUNNING (exp202_vrl)**

- Layer 0's V output blended into all subsequent layers via per-layer sigmoid gates
- Source: arxiv:2410.17897, PR #569 (1.1175 without TTT)
- Expected: -0.001 to -0.003 BPB
- Control: exp202_novrl (identical config, VRL_ENABLED=0)
- Extra params: 13 scalars (negligible)

### 2. QAT-Export Alignment

**Status: READY TO IMPLEMENT**

- Problem: STE fake-quant during training may use different clipping than GPTQ at export
- Fix: Match STE clip percentile to GPTQ's percentile (0.9995 from PR #569)
- Source: PR #569 explicitly aligns these
- Expected: -0.001 to -0.002 BPB (reduces quant gap)
- Risk: Low — one constant change

### 3. Soft-Round QAT

**Status: NEEDS IMPLEMENTATION**

- Replace hard `round()` in STE with temperature-controlled smooth approximation
- Source: PR #589 "Late Soft-Round QAT"
- Gives optimizer gradient signal near quantization bin boundaries
- Expected: -0.001 to -0.002 BPB
- Risk: Medium — new forward pass math, could destabilize training if temp schedule is wrong
- Implementation: `soft_round(x, temp) = x + (1/pi) * arctan(sin(2*pi*x) / temp)` or similar

### 4. Int5 GPTQ (fit bigger model)

**Status: NEEDS INVESTIGATION**

- Quantize to int5 (clip_range=15) instead of int6 (clip_range=31)
- Saves ~17% per weight → could fit 15th layer or wider MLP
- Source: PR #545 (33.6M params in 15.5MB with int5)
- Expected: depends on what we do with the space savings
- Risk: High — int5 quality loss might outweigh capacity gain at 14L
- A/B: int5 14L vs int6 14L (same architecture, just quant precision)

### 5. Backward-Looking Chunk TTT

**Status: IMPLEMENTED (exp201 chunked), already tested**

- Score chunk N, train on chunks 0..N-1 (not on N itself)
- Result: 1.1175 BPB in 138s (faster but 0.005 worse than per-window)
- Verdict: Per-window SGD is better for us. Could revisit with AdamW + more epochs.

### 6. EVAL_STRIDE=32

**Status: NOT VIABLE (time budget)**

- PR #545 uses stride=32 for finer overlap
- Our stride=76 at 575s leaves no room for stride=32 (~1300s est)
- Only viable if combined with chunked TTT (~138s base) + stride=32 scoring
- A/B: chunked TTT stride=32 vs per-window SGD stride=76

### 7. MHA 8/8 (drop GQA)

**Status: NEEDS TESTING**

- PR #545 uses full multi-head attention (8/8) vs our GQA (8/4)
- More attention capacity but more params per layer
- At 14L this might push artifact over 16MB unless combined with int5
- A/B: 14L 8/8 MHA vs 14L 8/4 GQA

### 8. Early QAT (threshold 0.5)

**Status: NEEDS TESTING**

- PR #545 starts QAT at 50% through warmdown (threshold=0.5)
- Our current: no QAT during training
- More QAT steps = model better adapted to quantization noise
- Source: PR #414 uses 0.15, PR #545 uses 0.5
- A/B: QAT threshold 0.5 vs 0.15 vs 0 (current)

### 9. Magnitude Pruning Post-GPTQ

**Status: NEEDS TESTING**

- PR #569 uses 2% magnitude pruning AFTER quantization for compression
- We disabled pruning (PRUNE_FRAC=0)
- Could help compression → smaller artifact → room for larger model
- A/B: PRUNE_FRAC=0.02 vs 0 (current)

### 10. Temperature Search (post-TTT)

**Status: TESTED — T=0.98 HURT (+0.0002)**

- Verdict: Not useful for our SGD TTT setup

---

## Completed Results


| Experiment               | BPB        | Time     | Delta vs baseline | Notes                    |
| ------------------------ | ---------- | -------- | ----------------- | ------------------------ |
| stride=64 per-window SGD | 1.1126     | 654s     | —                 | Over time budget         |
| stride=68                | 1.1129     | 640s     | +0.0003           | Over budget              |
| stride=72                | 1.1129     | 604s     | +0.0003           | 4s over                  |
| **stride=76**            | **1.1130** | **575s** | **+0.0004**       | **Submission candidate** |
| stride=80                | 1.1130     | 547s     | +0.0004           | Safe margin              |
| stride=96                | 1.1131     | 458s     | +0.0005           | Very safe                |
| stride=128               | 1.1133     | 347s     | +0.0007           | Very safe                |
| stride=76 T=0.98         | 1.1132     | 566s     | +0.0006           | Temp hurts               |
| chunked AdamW 1ep        | 1.1175     | 138s     | +0.0049           | Fast but worse           |
| VRL (exp202)             | pending    | ~575s    | ?                 | Running                  |
| no-VRL control           | queued     | ~575s    | ?                 | After VRL                |


---

## NEW from research (March 24)

### 11. Multi-Pass Score-First TTT

**Status: NEEDS INVESTIGATION**

- PR #573 uses 3 independent adaptation trajectories with shifted data orderings
- For each token, take min(NLL) across passes — best-of-3 scoring
- Claims 1.0523 BPB (need to verify legality)
- Expected: significant gain if legal
- Risk: 3x eval time — would need stride increase to compensate

### 12. Full GPTQ (not lite)

**Status: NEEDS TESTING**

- PR #593 claims full GPTQ improves post-quant BPB by 0.0048 vs GPTQ-lite
- We already use GPTQ but need to verify it's the full Hessian version with Cholesky error compensation
- Check: does our GPTQ_ENABLED=1 do full Cholesky or simplified?

### 13. EMA Decay Tuning

**Status: NEEDS TESTING**

- Our 14L model is deeper than typical 11L — may need different EMA dynamics
- Test: EMA(0.995), EMA(0.9975), EMA(0.9985) vs current EMA(0.997)
- Quick eval-only A/B if we save pre-EMA checkpoints

### 14. Weight Sharing (Recursive Depth)

**Status: HIGH RISK, INVESTIGATE LATER**

- PR #579: 6 unique blocks × 2 loops = 12 effective depth, wider MLP
- GPTQ catastrophically fails at 3+ loops — only 2 loops viable
- Would require major architecture rework
- Potential: large gain but untested at our scale

### 15. Cosine-Annealed TTT LR

**Status: EASY TO TEST**

- PR #581, #589: cosine LR during TTT (not flat) improves by 0.003+
- We use flat lr=0.002 — easy to add cosine decay across batches
- Can test with saved model (eval-only mode)

---

## Priority Order (Updated)

1. **VRL** (running) — architecture change, proven in #569
2. **QAT-export alignment** — low effort, directly attacks quant gap
3. **Cosine-annealed TTT LR** — easy eval-only test, proven in #581/#589
4. **Early QAT threshold 0.5** — proven in multiple PRs
5. **Soft-round QAT** — novel, higher effort but targets quant gap
6. **Full GPTQ verification** — make sure we're doing full Hessian, not lite
7. **Int5 + wider MLP** — risky but could unlock more capacity
8. **Multi-pass TTT** — need to verify legality and time budget
9. **EMA decay tuning** — quick eval-only if checkpoints available
10. **Pruning 2% post-GPTQ** — easy test, helps compression

---

## Deep Analysis: Where Our 14L Model Has Specific Inefficiencies

### Our unique situation
We're the ONLY clean submission at 14 layers. Everyone else is at 11L. Our depth advantage gives us ~0.012 BPB over 11L baselines, but it comes with costs:
- **~105ms/step vs ~85ms/step** (fewer training steps: 5700 vs 7000)
- **More params to quantize** (more quantization error compounding across layers)
- **Deeper gradient paths** (vanishing/exploding gradient risk)
- **Less room in 16MB** (14 layers of weights vs 11)

### Specific inefficiencies to target

**A. Quantization gap is our biggest loss (0.0149 BPB)**
- Pre-quant: 1.1268, Post-quant: 1.1417, gap = 0.0149
- Top PRs have gaps of 0.007-0.008 (half ours!)
- Our 14 layers mean more weight matrices to quantize → error compounds layer by layer
- GPTQ helps but we're still leaving ~0.007 BPP on the table vs best practices
- **Fix: QAT-export alignment + Early QAT + possibly Soft-Round QAT**
- **Fix: MLWQ-style per-layer bit allocation — give critical layers more bits**

**B. Attention may be under-utilizing our depth**
- 14 layers of identical attention with no cross-layer communication (except skip connections)
- VRL (testing now) adds one form of cross-layer info (V0 residual)
- **Differential Attention** (ICLR 2025): dual softmax branch, subtract noise → amplify signal. At 14 layers, attention noise compounds more than at 11L. Diff attention specifically helps deeper models.
- **Gated Attention** (arxiv 2505.06708): sigmoid gate after softmax, dynamic sparsity. Could help our deeper model prune irrelevant attention at later layers where patterns are more abstract.

**C. MLP activation may not be optimal for our depth**
- leaky_relu(0.5)² is proven at 11L but untested at 14L
- At 14 layers, the squared activation compounds gradient magnitudes more
- **PolyGLU** (arxiv 2603.13347): state-conditional activation routing — different activation per token based on hidden state. Zero extra params, just routes between existing activation functions.
- SwiGLU (PR #505) needs 3 weight matrices vs our 2 → doesn't fit at 14L without shrinking MLP

**D. BigramHash dim=64 may be undersized**
- We use BigramHash(8192, dim=64) then project to dim=512
- The projection from 64→512 is a bottleneck — 64 dims can't capture much
- PR #505 uses dim=128. Our earlier test (exp186) with dim=128 went over 16MB
- **Fix: Try dim=80 or dim=96 — halfway, might fit**

**E. Skip connections may need tuning for 14L**
- U-Net skip: 7 encoder layers, 7 decoder layers
- Skip weights are learned scalars per dim — initialized to 1.0
- At 14L, the encoder/decoder split is deeper than 11L's 5/6 split
- **Fix: Skip gating** (PR #569 uses learned gating on skips) — sigmoid gate instead of scalar multiply, adapts during training to route information more selectively

**F. Our RoPE base=50000 was tuned for earlier configs**
- Default 10000, we use 50000 — tuned for shorter training at an earlier layer count
- At 14L with ~5700 steps (fewer than 11L's ~7000), the RoPE dynamics differ
- **May be worth retesting** base=10000, 30000, 100000

---

## Novel Ideas Specific to Our 14L Architecture

### N1. Per-Layer Quantization Precision (MLWQ-inspired)
- Instead of uniform int6 everywhere, allocate bits per layer based on loss sensitivity
- First and last layers are most sensitive → give them int7 or int8
- Middle layers are more redundant → could handle int5
- Net: same average bits, much less quantization error where it matters
- **This is the #1 thing that could close our 0.015 quant gap**
- Implementation: run GPTQ with different clip_ranges per layer based on Fisher information or gradient magnitude

### N2. Differential Attention (lightweight version)
- Add a second, smaller attention branch (maybe 2 KV heads) that gets subtracted
- Paper shows it specifically helps deeper models by canceling accumulated noise
- Extra cost: ~2 extra KV heads per layer × 14 layers = modest param increase
- Could implement as: `y = y_main - lambda * y_noise` with learned lambda per layer
- **Particularly relevant for us** because 14L accumulates more attention noise than 11L

### N3. Layer-Wise LR Scaling for TTT
- Our per-window SGD TTT uses same LR for all unfrozen layers
- PR #545 uses per-layer LR groups: `lr * (0.5 + 0.5 * layer_idx / (num_layers - 1))`
- Later layers need more adaptation (closer to output), earlier layers less
- **Easy to implement** in the TTT optimizer setup, eval-only testable

### N4. Asymmetric U-Net (more encoder than decoder)
- Currently 7 encoder + 7 decoder (symmetric)
- Research suggests asymmetric splits (e.g., 9 encoder + 5 decoder) can be better
- Encoder layers are "cheaper" (no skip addition) → slightly faster per step
- More encoder layers = richer skip representations for decoder
- **Zero extra params, just changes the split point**

---

## Updated Priority Order (with novel ideas integrated)

### Tier 1: Highest confidence, lowest effort
1. **VRL** (running) — proven in #569
2. **QAT-export alignment** — one constant, attacks 0.015 quant gap
3. **Cosine-annealed TTT LR** — eval-only test, proven in #581/#589
4. **Per-layer TTT LR** — easy, specifically helps 14L
5. **Early QAT threshold 0.15-0.5** — proven in multiple PRs

### Tier 2: Medium effort, novel/high potential
6. **Per-layer quant precision (MLWQ)** — directly targets our #1 loss (quant gap)
7. **Soft-round QAT** — novel, bin-aware gradients
8. **Asymmetric U-Net split** — zero params, might unlock free BPB
9. **Skip gating (sigmoid)** — better than scalar for 14L depth

### Tier 3: Higher effort, uncertain
10. **Int5 + 15th layer** — risky, need int5 GPTQ quality to be good enough
11. **Differential attention (lightweight)** — novel, param cost
12. **BigramHash dim=80-96** — quick if artifact has room
13. **Full GPTQ verification** — verify Cholesky, not lite
14. **Multi-pass TTT** — legality + time budget questions

### Tier 4: Research directions
15. **PolyGLU activation routing** — zero params, novel
16. **RoPE base re-tuning** — may have drifted from 14L optimum
17. **EMA decay tuning for 14L** — deeper model may want different decay

---

## CRITICAL RESEARCH: What's Insanely Applicable to Our 14L Case

### The #1 Insight: Our quantization gap compounds across 14 layers

From EPTQ/HAWQ research: quantization error doesn't just add across layers — it **multiplies**. Each layer's output has small perturbations from quantization, and the next layer amplifies those perturbations through its nonlinear activations. With 14 layers vs 11, we have 27% more layers for error to compound through.

**This explains why our quant gap (0.015) is 2x the 11L PRs (0.007).**

It's not that our GPTQ is worse — it's that we have more layers for error to propagate. The fix isn't better GPTQ — it's reducing error propagation.

### R1. Hessian-Weighted Per-Layer Bit Allocation (HAWQ-style)
**Why it's perfect for us:**
- Our 14 layers have different sensitivities to quantization
- First layer (embedding projection) and last layer (pre-logit) are most sensitive
- Middle U-Net layers are least sensitive (redundancy from skip connections)
- **Concrete plan:** Run one forward pass with calibration data, compute per-layer Hessian trace (cheap via Hutchinson's estimator), then:
  - Layers with top 3 Hessian traces → int8 (256 levels)
  - Layers with bottom 3 Hessian traces → int5 (32 levels)
  - Rest stay int6 (64 levels)
  - Net: same average bits, dramatically less total quantization error
- **Expected gain: 0.003-0.007 BPB** (could halve our quant gap)
- **Effort: Medium** — need to add Hessian computation + mixed-bit GPTQ

### R2. Block-Wise Error Compensation Order
**Why it matters for 14L:**
- Standard GPTQ quantizes layers in order (0, 1, 2, ..., 13)
- Error from layer 0 propagates to layer 1's calibration data, corrupting it
- By layer 13, the calibration data has been corrupted 13 times
- **Fix:** Quantize in sensitivity order (most sensitive first, while calibration data is cleanest)
- Or: quantize from both ends inward (0, 13, 1, 12, 2, 11, ...)
- **Expected gain: 0.001-0.003 BPP**
- **Effort: Low** — just reorder the GPTQ loop

### R3. Activation Smoothing Before Quantization (SmoothQuant-style)
**Why it's perfect for deep models:**
- Deep models develop activation outliers that make quantization harder
- 14 layers = more outlier buildup than 11
- SmoothQuant migrates the quantization difficulty from activations to weights via per-channel scaling
- This makes GPTQ's job easier on every layer
- **Concrete plan:** Before GPTQ, compute per-channel activation scales from calibration data, fold them into weights
- **Expected gain: 0.002-0.004 BPB**
- **Effort: Medium** — add smoothing pass before GPTQ

### R4. Residual Quantization (two-pass GPTQ)
**Why it fits our constraint:**
- After first GPTQ pass, compute residual errors
- Quantize the residuals with a second, smaller codebook
- Store both in the same 16MB budget (residual codebook is tiny)
- This is essentially free error correction
- **Expected gain: 0.001-0.002 BPB**
- **Effort: Medium** — add residual computation + storage

### R5. Knowledge Distillation from Pre-Quant Model During TTT
**Why it's uniquely applicable to our setup:**
- We save the pre-quant model state (EMA weights before GPTQ)
- During TTT, we could use the pre-quant model as a teacher
- TTT adaptation target: match pre-quant model's output distribution, not just next-token loss
- This directly reverses quantization damage during eval
- **Expected gain: 0.002-0.005 BPB** (directly attacks quant gap during TTT)
- **Effort: High** — need to load both models, compute KL divergence
- **Risk: May not fit in GPU memory** (two 14L models)

### TLDR: If we only do ONE thing

**R1 (Hessian-weighted per-layer bit allocation)** is the single highest-impact change for our specific case. Our 14L model is uniquely punished by uniform int6 quantization because error compounds over 27% more layers than competing 11L models. Giving sensitive layers int8 and insensitive layers int5 (same average bits) could recover 0.003-0.007 BPB — more than any architecture change we've tested.

---

## Sources & References

### Quantization
- **HAWQ-V3** (Hessian-aware per-layer bit allocation): https://assets.amazon.science/a5/a5/bc16183e477aabdb282bfbeea260/hawq-v3-dyadic-neural-network-quantization.pdf
- **FlexQ** (INT6 with flexible group sizes): https://arxiv.org/abs/2508.04405
- **EPTQ** (Hessian-guided block reconstruction): https://arxiv.org/abs/2309.11531, code: https://github.com/ssi-research/eptq-sla
- **APTQ** (Attention-aware mixed precision): https://arxiv.org/abs/2402.14866
- **MLWQ** (Multi-level weight quantization for SLMs): https://aclanthology.org/2025.emnlp-main.408
- **SmoothQuant** (Activation smoothing before quant): https://arxiv.org/abs/2211.10438
- **Soft-Round QAT** (PR #589 in parameter-golf): temperature-controlled smooth rounding surrogate

### Architecture
- **VRL / ResFormer** (Value Residual Learning): https://arxiv.org/abs/2410.17897
- **Differential Attention**: https://proceedings.iclr.cc/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html
- **Gated Attention**: https://arxiv.org/abs/2505.06708
- **PolyGLU** (State-conditional activation routing): https://arxiv.org/abs/2603.13347
- **Autoregressive U-Net** (gated skip connections): https://arxiv.org/abs/2506.14761
- **Depth-Width Tradeoff**: https://arxiv.org/abs/2503.01805

### Training & Optimization
- **EMA dynamics in deep learning**: https://arxiv.org/abs/2411.18704
- **Muon optimizer**: https://github.com/KellerJordan/Muon, https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- **Compute-optimal scaling** (OptiBERT): https://aclanthology.org/2025.emnlp-main.1804

### Competition PRs (verified clean)
- **PR #569** (1.1175, VRL + GPTQ, no TTT): VRL + LeakyReLU² + Full GPTQ + QAT-export alignment
- **PR #545** (1.1179, int5 GPTQ): 33.6M params, int5 per-row GPTQ, Early QAT 0.5
- **PR #589** (1.1178, Soft-Round QAT): Late Soft-Round QAT + Backward-Looking TTT
- **PR #505** (1.1181, SwiGLU + VE128): SwiGLU + VE128 + no TTT
- **PR #414** (1.1233, EMA + GPTQ-lite): EMA + GPTQ-lite + QAT@0.15

---

## HYPERSPECIFIC RESEARCH: Problems Unique to Our 14L/512d/GQA/leaky²/GPTQ Model

### Problem 1: leaky_relu(0.5)² gradient compounding at 14 layers

**The issue:** Our activation is `leaky_relu(x, 0.5).square()`. The squared operation means gradients scale as `2 * leaky_relu(x)` in the backward pass. Over 14 layers of MLP, this creates a multiplicative gradient factor. At 11 layers this is manageable; at 14 layers the gradient magnitude compounds ~27% more.

**Research says** (arxiv:2402.03804, zdtech.substack.com): "Squared ReLU exacerbates vanishing gradients in deep networks because the derivative is proportional to 2x for x>0, leading to rapidly diminishing gradients across many layers."

**Our leaky(0.5) mitigation helps** — the 0.5 negative slope prevents dead neurons. But the squaring still compounds. At 14 layers we may be hitting the limit.

**Potential fixes:**
- **R6. Replace square with abs** — `leaky_relu(x, 0.5).abs()` has gradient magnitude 1 everywhere, no compounding. Zero extra params. But changes the activation landscape.
- **R7. Learnable activation power** — `leaky_relu(x, 0.5).pow(p)` where p is learned per-layer, initialized at 2.0. Lets later layers reduce squaring intensity. 14 extra scalar params.
- **R8. Layer-wise activation scaling** — multiply activation output by `1/sqrt(layer_idx+1)` (like LN Scale but for MLP). Prevents gradient explosion from squaring. We tested LN Scale on attention norms and it hurt (+0.028) — but on MLP activations it's different.
- Source: https://arxiv.org/abs/2402.03804

### Problem 2: GQA 8H/4KV wastes attention capacity at 14 layers

**The issue:** With GQA, each KV head serves 2 query heads. At dim=512, each head is 64d. With only 4 KV heads × 64d = 256d of unique key/value information, we're projecting 512d inputs to 256d keys/values — 50% information bottleneck in every layer.

**At 14 layers, this bottleneck compounds:** later layers receive increasingly processed representations but still squeeze them through the same 256d KV bottleneck. 11L models have fewer layers so the bottleneck matters less.

**Research says** (Cost-Optimal GQA, arxiv:2503.09579): "Decoupling total attention head dimensions from model hidden size to flexibly control inference FLOPs." Also: "Weighted Grouped-Query Attention introduces learnable weights for aggregating key and value heads."

**Potential fixes:**
- **R9. Increase KV heads from 4 to 6** — more unique KV information (384d vs 256d). Costs ~33% more attention params but may help deep layers. Need to check if artifact still fits 16MB.
- **R10. Weighted GQA** — add a learned scalar per (query_group, kv_head) pair that weights the KV aggregation. 8 extra params per layer × 14 layers = 112 params total. Near-zero overhead.
- Source: https://arxiv.org/abs/2503.09579

### Problem 3: Our RoPE base=50000 was tuned at fewer layers

**The issue:** We set ROPE_BASE=50000 during earlier experiments (likely at 9-12 layers). RoPE base frequency determines how quickly position information decays across sequence length. Higher base = slower decay = longer effective context. But at 14 layers, the model processes positions differently — deeper models build hierarchical position representations that interact with RoPE frequencies.

**Research says** (arxiv:2503.04355): "Layer-Specific Scaling of Positional Encodings for Superior Long-Context Modeling" — different layers should have different RoPE scaling. Early layers need fine-grained position info, deep layers need coarse position info.

**Potential fixes:**
- **R11. Per-layer RoPE base** — layers 0-4 use base=10000 (fine position), layers 5-9 use 50000 (medium), layers 10-13 use 200000 (coarse). Zero extra params, just changes the precomputed cos/sin tables.
- **R12. Sweep RoPE base** — just test 10000, 30000, 50000, 100000 as a hyperparameter. Quick A/B with eval-only if we save checkpoints at different bases (would need retraining).
- Source: https://arxiv.org/abs/2503.04355

### Problem 4: Entropy of our quantized weights could be lower for better compression

**The issue:** Our artifact is 15.83MB with brotli-11. We have only 170KB headroom. If we could improve compression by even 200KB, we could try BigramHash dim=96 or wider MLP.

**Research says** (EntroLLM, arxiv:2505.02380): Tensor-level quantization (one scale per entire tensor, not per-row) produces "spiky" weight distributions with much lower entropy → dramatically better compression. Per-row quantization spreads the distribution out.

**The tradeoff:** tensor-level quantization is less precise than per-row, so model quality drops. But if the compression savings let us fit a bigger model, net effect could be positive.

**Potential fixes:**
- **R13. Hybrid quantization** — use tensor-level quantization for the middle (least sensitive) layers and per-row for first/last layers. Better compression on middle layers, preserved quality on critical layers.
- **R14. Huffman coding after quantization** — replace brotli with Huffman coding on the int6 weight values. Since int6 has only 64 possible values, the Huffman tree is tiny and compression is near-optimal. May beat brotli on structured quantized data.
- Source: https://arxiv.org/abs/2505.02380

### Problem 5: Our TTT only updates last 12 layers (freeze 2), but research says update only last 25%

**The issue:** We freeze first 2 of 14 layers during TTT (85% unfrozen). Research on efficient TTT says updating only the last 25% of layers (i.e., last 3-4 of 14) maintains performance while being much faster.

**Research says** (test-time-training.github.io, NanoAdapt IJCAI 2024): "Constraining updates to a subset of parameters (e.g., only MLP weights in the last 25% of transformer blocks) maintains performance while lowering computation."

**Potential fixes:**
- **R15. Freeze first 10 layers** (only update layers 10-13). Cuts backward computation by ~70%. Each TTT step is 3x faster → either faster eval or more steps in same budget.
- **R16. Only update MLP weights** during TTT (freeze all attention). MLPs are where most of the model's distribution knowledge lives. Attention patterns are more structural and shouldn't change for domain adaptation.
- Source: https://test-time-training.github.io, https://www.ijcai.org/proceedings/2024/0616.pdf

