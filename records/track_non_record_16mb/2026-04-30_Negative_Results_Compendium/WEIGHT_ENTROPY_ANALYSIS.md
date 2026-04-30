# Weight Entropy, Gaussian Expressiveness, and the Compression Ceiling

**The fundamental tension:** L2 regularization makes weights Gaussian. Gaussian is the maximum-entropy distribution at fixed variance. This makes weights maximally expressive for the model but maximally hard to compress — and we prove that simple training-time regularizers cannot productively resolve this tension.

---

## 1. The Compression Pipeline

The Parameter Golf artifact budget is governed by:

```
artifact bytes ≈ Brotli(quantized_int_stream) + scales + metadata + code
```

For the PR-1787 stack:
- ~31.7M weight values quantized to int6 (values in [-31, 31])
- Per-row fp16 scale: `scale = k × std(row) / 31`
- Full-Hessian GPTQ Cholesky error compensation
- Byte-shuffle + Brotli-11 compression

The Shannon lower bound for the compressed payload is:

```
minimum bits = N × H(Q)
H(Q) = -Σ p(q) log₂ p(q)
```

where `p(q)` is the probability of each integer bucket in the quantized stream.

**Brotli already achieves 101.3% of Shannon** on our integer streams (slightly exceeds the order-0 bound through context modeling). There is effectively no entropy-coder headroom.

---

## 2. The Gaussian Prediction

If weights within each row are i.i.d. Gaussian (the theoretical outcome of L2 regularization), the quantized bucket distribution can be predicted analytically.

For a Gaussian with mean 0 and variance σ², quantized with per-row SDClip at k sigmas and clip range R:

```
step_size = k × σ / R
P(q = 0) = erf(0.5 × step_size / (σ × √2)) × 2  [≈ 2 × Φ(0.5/k × R) - 1]
P(q = j) = Φ((j + 0.5) × k/R) - Φ((j - 0.5) × k/R)  for |j| < R
P(q = ±R) = 1 - Φ((R - 0.5) × k/R)  [clipped tail mass]
```

For k=12, R=31 (stock MLP):

| Metric | Gaussian prediction | Empirical PR-1787 MLP |
|--------|--------------------:|----------------------:|
| H(Q) entropy | 3.425 bits/symbol | 3.427 bits/symbol |
| P(q=0) | 15.3% | 15.5% |
| P(\|q\|≤2) | 66.7% | 67.0% |
| P(\|q\|=4..6) | 16.2% | 16.1% |
| P(\|q\|≥8) | 0.4% | 0.4% |

**The match is near-perfect.** The trained weights are empirically Gaussian within each row to the precision that matters for compression.

This is not surprising — L2 weight decay applies a Gaussian prior, and with thousands of gradient steps the posterior concentrates around the prior's shape. The kurtosis of individual rows is 0.05–0.3 (excess over Gaussian's 3.0) for most tensors, with only late decoder attention showing heavier tails (kurtosis 1.4–2.3).

---

## 3. The Maximum Entropy Theorem

**Theorem (Shannon):** Among all distributions with a given mean and variance, the Gaussian has maximum differential entropy.

For our discrete quantized buckets, this means: at fixed SDClip parameters (k, R) and fixed row variance, the Gaussian weight distribution produces the **highest possible entropy** in the quantized integer stream.

```
H_Gaussian(Q) ≥ H_any(Q)  at fixed σ² and fixed (k, R)
```

**Consequence:** Any distribution shape that departs from Gaussian at the same variance will have **lower** quantized-stream entropy and therefore **better** Brotli compression. This is the theoretical motivation for our regularizer experiments.

---

## 4. The Expressiveness Counter-Argument

The same maximum-entropy property has a dual interpretation in information theory:

**A Gaussian distribution has maximum information capacity per parameter at fixed power (variance).**

In the context of neural network weights:
- Each weight acts as a channel between the loss landscape and the model's function
- Higher entropy = more distinguishable weight configurations = more functions the model can represent
- Gaussian weights at a given scale are the **most expressive** choice

This creates the fundamental tension:

```
Compression wants: low entropy  → peaked, structured distributions
Expressiveness wants: high entropy → spread, unstructured distributions
The Gaussian is: maximum entropy = worst compression = best expressiveness
```

L2 regularization is not accidentally making weights hard to compress. It is pushing them toward the maximally expressive distribution at their given scale. **Any departure from Gaussian that helps compression necessarily reduces the model's representational capacity.**

---

## 5. Experimental Validation

We ran 6+ full 8xH100 training runs testing whether training-time distribution shaping could productively resolve this tension:

### Experiment 1: Kurtosis penalty (target < 3.0)

Push weights toward sub-Gaussian kurtosis to create a more uniform (lower-entropy) central distribution.

**Result:** Kurtosis moved as intended, but quantized-bucket entropy was unchanged. The continuous distribution shape change didn't survive the discretization through SDClip + GPTQ. The compressor doesn't see kurtosis — it sees integer buckets.

### Experiment 2: Shoulder penalty (|q|=4..6 → |q|≤3)

Directly penalize values in the information-theoretically wasteful shoulder region.

**Best result:** Shoulder λ=1e-2 with late WD removal: **-0.76 mBPB** with +3 KB artifact. But exact GPTQ bucket audit showed the entropy change was +0.0008 MB (worsened slightly). The BPB win came from better pre-quant quality (the WD removal), not from entropy shaping.

### Experiment 3: Q-center (strong central pressure)

Aggressively push all q-coordinates toward small absolute values.

**Result:** Saved 259 KB (real entropy win!) but cost +1.64 mBPB. The zero spike thinned and tail mass grew — the model redistributed weight magnitude to preserve function, paying quality to satisfy the regularizer. Classic Goodhart's Law.

### Experiment 4: Q-transport (constrained shoulder → center transport)

The most sophisticated attempt: move only the |q|=5/6 shoulder toward |q|=3/4 while protecting the zero spike and capping tails.

**Conservative result:** Saved 105 KB for +0.185 mBPB. Best training-time entropy/BPB exchange rate achieved.

**Strong result:** Saved 324 KB but cost +2.11 mBPB. The model gamed the loss — values moved below the shoulder mask boundary to q=4 instead of q=3, and |q|≥8 grew because the tail budget was a lump constraint that didn't separately cap each tail bin.

### What the experiments prove

The q-transport experiments are the most informative. They show that:

1. **Training-time entropy shaping CAN move the final quantized stream.** This is not trivial — GPTQ's error compensation partially rewrites the integer stream, so training-time changes might be washed out. They aren't fully washed out, but the magnitude is small.

2. **The exchange rate is poor.** Conservative q-transport achieves ~0.57 MB saved per mBPB cost. Compare with simple k-inflation: ~0.16 MB saved per mBPB cost. Q-transport is ~3.5x more efficient, but the absolute effect is small because the stock distribution is already near-optimal.

3. **The model games the objective.** Any local bucket penalty creates escape routes to neighboring unpena­lized buckets. The strong q-transport run proved this: intended q=5→q=3 migration became q=5→q=4 accumulation plus |q|≥8 growth.

4. **The best achievable exchange rate still can't beat allocation.** Loop-aware k scheduling (giving more precision to recurrent layers 3,4,5) achieves -0.45 mBPB at +91 KB — a better Pareto point than any training-time entropy method.

---

## 6. The Stock Distribution Is Near-Optimal

Current PR-1787 MLP bucket fractions:

```
q = 0:        15.5%    ← large zero spike
|q| = 1:      26.6%    ← 
|q| = 2:      24.9%    ├── 82.6% in |q| ≤ 3
|q| = 3:      15.6%    ←
|q| = 4..6:   16.1%    ← the "shoulder" (target for compression)
|q| = 7:       0.9%    ← sparse
|q| ≥ 8:       0.4%    ← very sparse
```

Shannon entropy: 3.427 bits/symbol → 13.56 MB for 31.7M values → Brotli achieves 13.37 MB.

The tail (|q|≥7) contains only 1.3% of values. Even collapsing all tails into ±7 saves at most ~0.07 MB before BPB damage. The theoretical ceiling for distribution shaping without quality loss is:

```
Gaussian entropy:          3.425 bits/symbol
Uniform bounded entropy:   3.170 bits/symbol  (maximum possible compression)
Difference:                0.255 bits/symbol   = ~1.01 MB saved
```

But approaching uniform requires destroying the peaked structure that makes the model work. In practice, the achievable compression from distribution shaping is 0.1–0.3 MB — real but not game-changing.

---

## 7. The Complete Rate-Distortion Picture

Synthesizing all quantization experiments, the available post-training compression levers and their exchange rates:

| Method | MB saved per mBPB cost | Mechanism | Reliable? |
|--------|----------------------:|-----------|-----------|
| k-inflation (12.85→19.3) | 0.161 | Wider bins, peakier distribution | Yes |
| SparseGPT (t=2) | 0.275 | Zero-or-quantize per column | Yes, but tiny absolute effect |
| Q-transport (conservative) | 0.568 | Training-time shoulder compression | Fragile |
| Loop-aware k allocation | ∞ (improves both) | More precision where needed | **Yes, best lever** |
| LQER top-1 | ∞ (improves both) | Low-rank residual on tok_emb | **Yes, best lever** |

The best strategy is **not** to lower entropy everywhere. It is to spend bytes where sensitivity is high and save bytes where sensitivity is low. This is allocation, not compression.

---

## 8. Implications for the Competition

1. **Don't try to make weights more compressible.** The Gaussian distribution is already near the Pareto frontier of expressiveness vs. compressibility at this model size. The artifact size is a hard constraint, not a soft objective.

2. **Do try to make bytes count more.** Loop-aware precision, LQER on the most sensitive tensor (tok_emb), and central-bucket GPTQ polish are all small reliable wins that compose additively.

3. **The biggest BPB gains come from pre-quant quality, not post-quant tricks.** Every 1 mBPB improvement in pre-quant quality flows through quantization and TTT roughly 1:1. Architecture and optimizer improvements dominate quantization improvements.

4. **Brotli is near-optimal.** Switching entropy coders (Huffman, ANS, LZMA) produces identical or worse results on this payload. The compressed artifact size is governed by the integer-stream entropy, which is governed by the weight distribution, which is governed by the training dynamics. There are no shortcuts.

---

## 9. Key Equations Summary

**SDClip quantization (the mathematical object being compressed):**
```
scale(row) = k × std(row) / R
q_i = clamp(round(w_i / scale(row)), -R, R)
```

**Scale invariance (why L2 WD doesn't help compression):**
```
Q(γW) = clamp(round(γw / (k × γσ / R)), -R, R) = Q(W)
```

**Shannon entropy of quantized stream (what determines artifact size):**
```
H(Q) = -Σ p(q) log₂(p(q))
artifact_bits ≈ N × H(Q)
```

**Gaussian maximum entropy (the fundamental theorem):**
```
H_Gaussian ≥ H_any  at fixed variance
⟹ Gaussian weights produce maximum H(Q) at fixed (k, R)
⟹ L2-trained weights are maximally hard to compress
```

**The rate-distortion tradeoff (what we're actually optimizing):**
```
minimize post_quant_BPB(Q(W))
subject to N × H(Q(W)) / 8 + overhead ≤ 16,000,000 bytes
```

This is a constrained optimization where the objective and the constraint share the same variable (the weight distribution). Making the constraint easier (lower entropy) necessarily makes the objective harder (worse BPB). There is no free lunch.
