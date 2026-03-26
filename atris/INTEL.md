# Competitive Intelligence — Updated 2026-03-20 (Cycle 9)

## OFFICIAL LEADERBOARD (14 merged entries!)

| Rank | BPB | Author | Key Techniques | PR |
|------|-----|--------|----------------|----|
| **1** | **1.1428** | thwu1 | Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04 | #180 |
| 2 | 1.1458 | Raahil Shah | SmearGate + BigramHash + MLP3x + OrthoInit + MuonWD + SWA | #162 |
| 3 | 1.1502 | aruniyer | 11L MLP3x + WD=0.04 + zstd-22 + int6 QAT | #86 |
| 4 | 1.1556 | aquariouseworkman | SmearGate + BigramHash + MLP3x + int6 STE QAT | #65 |
| 5 | 1.1586 | yahya010 | 10L int6 QAT + zstd-22 + MLP2.6x + Muon0.99 | #63 |
| 6 | 1.1630 | aquariouseworkman | Int6 blocks + int8 embed + MLP3x + sliding window | #65 |
| 7 | 1.1748 | notapplica | Spectral embed + residual mixing + sliding window | #60 |
| 8 | 1.1925 | Matthew Li | Sliding window eval stride=64 (zero training changes!) | #50 |
| 9 | 1.1928 | samacqua | Sliding window + LoRA TTT (test-time training) | #77 |
| 10 | 1.2014 | Spokane Way | 4k seq length + better hyperparams | #52 |
| 11 | 1.2060 | Spokane Way | 2048 seq length | #49 |
| 12 | 1.2147 | Nan Liu | 10L mixed int8/int6 | #39 |
| 13 | 1.2197 | Renier Velazco | FP16 tied embed + warmdown tuning | #42 |
| 14 | 1.2244 | Baseline | 9L 512d MLP2x 1024vocab | — |

## WINNING TECHNIQUES WE'RE MISSING

### 1. BigramHash (CRITICAL — used by #1 and #2)
- Hash consecutive token pairs → lookup in 4096-10240 bucket embedding table
- 128-dim bigram embeddings projected to model_dim
- Captures local bigram context (~524K params for 4096 buckets)
- Implementation: XOR hash with coprime multipliers
- **Impact: ~0.003 BPB improvement**

### 2. SmearGate (used by #2, #4)
- Per-dimension learned gate blending current token with previous token embedding
- Applied after embedding normalization
- Only ~512 params (one gate vector per dim)
- Captures temporal continuity
- **Impact: ~0.002 BPB improvement**

### 3. SWA — Stochastic Weight Averaging (used by #1, #2)
- Collect checkpoints every 50 steps during warmdown
- Average them at the end (24+ snapshots)
- Start at 40% through training (#1) or 50% (#2)
- Zero artifact cost — just averages weights
- **Impact: ~0.002-0.003 BPB improvement**

### 4. Weight Decay (used by #1, #2, #3)
- WD=0.04 for Muon (decoupled: `p.data.mul_(1 - lr * wd)`)
- WD=0.01-0.04 for AdamW on embeddings/scalars
- Not in baseline at all
- **Impact: ~0.002 BPB improvement**

### 5. Int5 Quantization (used by #1)
- MLP weights at Int5 [-16,15]: 3 zero high bits per byte
- zstd-22 compresses Int5 at 1.88x (vs 1.51x for Int6)
- Saves ~1.86MB → funds 10th layer
- **Impact: enables more params within 16MB budget**

### 6. zstd-22 instead of zlib (used by #1, #3, #5)
- Better compression ratio than zlib
- More room for parameters
- **Impact: ~0.5-1MB saved → more model capacity**

### 7. OrthoInit + muP (used by #2, #4)
- Orthogonal weight initialization
- Output projections scaled by 1/√(2·num_layers)
- Better gradient flow
- **Impact: ~0.001 BPB improvement**

### 8. Gradient Clipping (used by #1)
- grad_clip_norm=0.3 (baseline: 0.0 = disabled)
- Stabilizes training, especially with higher LR/WD

## WHAT WE HAVE vs WHAT WE NEED

| Technique | We Have? | They Have? | Gap? |
|-----------|----------|------------|------|
| 10 layers | ✅ | ✅ | — |
| Lower LR 0.02 | ✅ | ✅ | — |
| INT6 QAT | ✅ | ✅ | — |
| Sliding window eval | ✅ | ✅ | — |
| Muon 0.99 | ✅ | ✅ | — |
| Weight sharing | ✅ | ❌ | We have extra |
| MLP 3x | ✅ (config) | ✅ | — |
| **BigramHash** | ❌ | ✅ (#1,#2) | **MISSING** |
| **SmearGate** | ❌ | ✅ (#2,#4) | **MISSING** |
| **SWA** | ❌ | ✅ (#1,#2) | **MISSING** |
| **Weight Decay** | ❌ | ✅ (#1,#2,#3) | **MISSING** |
| **Int5 quant** | ❌ | ✅ (#1) | **MISSING** |
| **zstd compression** | ❌ | ✅ (#1,#3,#5) | **MISSING** |
| **OrthoInit** | ❌ | ✅ (#2,#4) | **MISSING** |
| **Gradient clip** | ❌ | ✅ (#1) | **MISSING** |

## PRIORITY IMPLEMENTATION ORDER

1. **SWA** — Zero cost, average checkpoints during warmdown. Easiest win.
2. **Weight Decay** — Add WD=0.04 to Muon, WD=0.01 to Adam. One-line changes.
3. **Gradient clipping** — Set GRAD_CLIP_NORM=0.3. Already an env var!
4. **zstd-22** — Replace zlib.compress with zstd. Small code change.
5. **BigramHash** — Need to implement hash table + projection. ~50 lines.
6. **SmearGate** — Learned gate after embeddings. ~20 lines.
7. **Int5 quantization** — Extend our INT6 to INT5 for MLP layers. ~30 lines.
8. **OrthoInit** — Change weight initialization. ~10 lines.

## REALISTIC TARGET

If we stack ALL techniques the top 3 are using:
- Base (our v5): ~1.19 BPB (sliding window + 10L + QAT + lower LR)
- + SWA: ~1.187
- + WD: ~1.185
- + BigramHash: ~1.182
- + SmearGate: ~1.180
- + Int5 + zstd: ~1.175 (more room for params)
- + OrthoInit: ~1.173

**Realistic target: 1.14-1.15 BPB** (competitive with top 3)
**To beat #1 (1.1428): need novel technique or better hyperparameter tuning**
