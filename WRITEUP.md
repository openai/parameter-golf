# Parameter Golf Submission — val_bpb 1.1224 (3-seed mean)

## Result
**3-seed mean: 1.1224 BPB** | Artifact: 15.6-15.9MB | Train: 600s on 8xH100 | Eval: 74s

## What We Tried (40+ experiments)

**What worked:**
- FA3 Hopper attention (76ms/step, +47% more training steps)
- Fixing 4 training bugs found by deep-diffing against PR#414 (dead bigram weights in optimizer, Muon weight decay order, STE quantizer range mismatch, YaRN RoPE frequency extension)
- BigramHash vocab 3072 (optimized for 16MB budget — 2048 too small, 4096 too big)
- TTT Burst: replaying last 100 training batches at 10% LR before EMA finalization
- **Soft-to-hard quantizer with late temperature annealing** (novel, described below)

**What failed (and why):**
- Looped transformer 8Lx2 (+40% step cost kills training budget)
- MoE with 8 experts (8x params — wrong tradeoff for parameter-constrained setting)
- Focal loss (distorts CE objective; model gets overconfident on easy tokens)
- Entropy regularization on weights (great compression 13.3MB! but 2.5x slower per step)
- Cosine warmdown (worse compression AND worse quality)
- Curriculum seq length 1024->2048 (massive quantization damage)
- 12L architecture (doesn't fit 16MB with 3x MLP)
- int5 MLP quantization (+0.035 BPB damage — too aggressive)
- Star-ReLU, orthogonal init, eval at 4096 — all neutral

## Novel Contribution: Soft-to-Hard Quantizer

**The idea:** Replace hard STE rounding in QAT with temperature-controlled soft rounding. During the final 2% of training (scale < 0.02), the quantizer switches from hard round to sigmoid-interpolated soft round. This gives weight gradients a differentiable signal toward the nearest quantization grid point, nudging weights to "snap" to int6 levels right before EMA/SWA finalizes them.

**Why it works:** Standard STE has zero gradient information about quantization bin assignment — round() has zero derivative everywhere. By using `sigmoid((frac - 0.5) / tau)` as a soft surrogate in the backward pass, the optimizer receives non-zero gradients that push weights toward grid centers. Applying this only in the final phase (tau=0.1) avoids slowing down early training while getting the compression benefit when it matters most.

**Evidence:** Full soft quantizer (every step) compresses to 15.8MB (vs 16.0MB baseline) but costs 14% step overhead. Late-only application (last 2%) achieves the same compression improvement at zero overhead. Combined with bigram 3072 and TTT Burst, the submission achieves 1.1224 mean BPB — beating the prior SOTA of 1.1232.

**Connection to literature:** This is a lightweight instance of the Differentiable Soft Quantization (DSQ) and soft-to-hard vector quantization family, adapted for the parameter golf setting where training budget is tight and the target is a compressed artifact.
