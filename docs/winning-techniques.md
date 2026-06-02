# Winning Techniques Analysis — What Actually Works

> Focus: What separates the top 5 from the middle of the leaderboard? What techniques compound well together?

---

## The Current SOTA Stack (1.1147 BPB)

The top entry is not a single breakthrough — it is **11 validated techniques stacked** over 5 weeks:

```
Full Hessian GPTQ (AR self-gen calib)  ← biggest single contribution in final stack
XSA on ALL 11 layers                   ← free: zero new params
BigramHash 3072 × dim=112              ← wider than predecessors
Partial RoPE (16/64 dims)              ← small positional regularization
LN Scale 1/√(layer+1)                 ← layer-depth normalization
VE128 on layers 9–10                   ← local width boost
SmearGate                              ← positional mixing gate
U-Net skip connections                 ← depth-to-depth residuals
EMA(0.997) + SWA(every 50)            ← weight averaging
Parallel Muon + Warmdown 4000          ← optimizer + schedule
LZMA preset=9                          ← compression
```

**Nothing here is "magic".** Each technique was PRed separately, measured, and then combined.

---

## Tier 1: Must-Have (Every Competitive Run)

These are table stakes. Skipping any of these costs ≥ 0.01 BPB.

| Technique | BPB cost if skipped | Notes |
|-----------|-------------------|-------|
| Sliding Window Eval | −0.019 | Pure eval trick, zero training cost |
| Int6 QAT | ~−0.01 | Enables more params in budget |
| MLP 3× expansion | ~−0.008 | More capacity at same layer count |
| Muon WD + momentum 0.99 | ~−0.005 | Better optimizer signal |
| Warmdown 3500–4000 | ~−0.003 | Free: just change one number |

**Start here. These should be in your baseline before adding anything else.**

---

## Tier 2: High-Impact Techniques (3–8 BPB points each)

| Technique | Δ BPB | Complexity | Compounds with |
|-----------|-------|-----------|----------------|
| Full Hessian GPTQ | −0.008 | High | Int6, LZMA |
| XSA (all layers) | −0.005 | Medium | Any architecture |
| BigramHash (3072) | −0.005 | Medium | Tied embeddings |
| EMA + SWA | −0.003 | Low | Any |
| Partial RoPE | −0.002 | Low | Any |

---

## Tier 3: Small Gains (1–3 BPB points each)

| Technique | Δ BPB | Notes |
|-----------|-------|-------|
| LeakyReLU² | −0.002 | Free activation swap |
| LN Scale | −0.001 | One-line change |
| LZMA over zlib | −0.001 | 30s artifact-size saving |
| Selective ±1 pruning | −0.001 | Post-GPTQ cleanup |
| OrthoInit | −0.001 | Init change only |

---

## Why TTT Was Dropped from SOTA

The top record explicitly dropped TTT despite it contributing −0.0025 BPB in the previous SOTA (PR #549).
After 25 failed integration attempts across two stacks, they concluded:

> "TTT is neutral or negative on this stack. The Full GPTQ improvement more than compensates."

**Lesson**: Techniques interact non-linearly. A technique that helps in one stack may hurt in another. Always ablate.

---

## Quantization: The Core Leverage Point

The 16MB constraint makes quantization the **highest-leverage dimension**:

| Quantization scheme | Effective params in budget | Notes |
|--------------------|--------------------------|-------|
| Int8 + zlib (baseline) | ~50M | Default |
| Int6 STE QAT + zstd | ~65M | Top mid-range runs |
| Int6 Full GPTQ + LZMA | ~70M | Current SOTA approach |
| Int5 mixed | ~75M | Accuracy degrades |
| Ternary (1.57 bits) | ~100M | 1.157 BPB; doesn't beat int6 GPTQ yet |
| Binary (1 bit) | ~200M | 1.123 BPB over 2h; shows future potential |

**Key insight**: The binary/ternary runs show that with enough parameters, lower precision can win. But getting there under 10 minutes is the challenge.

---

## What the Leaderboard Tells Us About Architecture

The SOTA stack has converged on:
- **11 transformer layers** (not 9 or 10, not 12)
- **512 model dim** (changing this breaks the artifact budget)
- **3× MLP expansion** (not 2× or 4×)
- **GQA with 8Q / 4KV heads** (unchanged from baseline)
- **Tied embeddings** (always; saves ~10% of params)

The architectural exploration space is largely exhausted at this param budget. **The frontier is now quantization quality and evaluation tricks**.

---

## Statistical Significance Requirements

The challenge requires `p < 0.01` (Welch's t-test) to claim a new record.
For the current SOTA (std ≈ 0.0004 BPB), beating by 0.005 BPB requires:

| Seeds | Power to detect 0.005 BPB improvement |
|-------|--------------------------------------|
| 3 seeds | Sufficient if std < 0.001 |
| 5 seeds | Safer; use if technique is noisy |

**Run 3 seeds minimum. Report exact per-seed scores, not just mean.**

---

## My Recommended Attack Vector

Given the current state (April 2026):

1. **Non-record track**: Implement one of the open ideas (JEPA, text diffusion, SSM). These will almost certainly not beat SOTA but will be accepted as non-record submissions and are scientifically interesting.

2. **Record track**: The most likely path to beating 1.1147 is:
   - Better calibration for GPTQ (e.g. curriculum-based generation instead of uniform temperature)
   - Pushing BigramHash wider (4096+ buckets) without exceeding artifact budget
   - A more aggressive compression scheme (e.g. model-specific entropy coding)
   - Better EMA/SWA tuning

3. **Stretch goal**: Megakernels that make 8192 seq_len feasible in 600s → combines long context with quantization gains.

---

## Things That Didn't Work (Negative Results Worth Knowing)

From PR #670 (30+ failed experiments by the top team):
- TTT on the GPTQ stack: neutral to negative
- GPTQ-lite vs Full GPTQ: Full wins but calibration data matters enormously
- Wider BigramHash beyond 3072: artifact budget hit before break-even point
- Lower quantization (int4) at 11L: accuracy too degraded

**Don't repeat these experiments.** Start from where the SOTA team left off.
