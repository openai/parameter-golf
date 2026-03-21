# Proven Techniques — What We Know Works (and What Doesn't)

**Base: PR198 script → 12.71MB artifact, ~1.18 BPP sliding, 9L**
**Goal: Start from PR198, add only proven wins, A/B test each addition**

---

## PROVEN WINS (keep these)

### Training Improvements
| Technique | BPP gain | Artifact impact | Source | Confidence |
|-----------|----------|----------------|--------|------------|
| **NorMuon** (vs plain Muon) | +0.004 BPP | Worse compression (~+2MB) | exp058 vs 057 | HIGH — but compression tradeoff! |
| **leaky_relu(0.5)²** (vs relu²) | +0.0014 BPP | Neutral | exp084 vs 081 | HIGH |
| **WD=0.04** (vs WD=0.01) | +0.004 BPP | Slightly better compression | exp072 vs 068 | HIGH |
| **BigramHash** (learned, int6 quantized) | +0.003 BPP | +500KB artifact | exp075 vs 071 | HIGH |
| **SmearGate** | +0.001 BPP | Tiny | PR135 | MEDIUM |
| **OrthoInit + muP scaling** | +0.001 BPP | None | PR135 | MEDIUM |
| **seq_len=2048** (vs 1024) | Better sliding eval | None | exp055 | HIGH |
| **batch_tokens=786432** | Pairs with seq2048 | None | PR135 | HIGH |
| **warmdown_iters=3000** | Standard schedule | None | PR198 | HIGH |
| **grad_clip=0.3** | Stability | None | PR135 | HIGH |

### Post-Training Improvements
| Technique | BPP gain | Artifact impact | Source | Confidence |
|-----------|----------|----------------|--------|------------|
| **Sliding window eval stride=64** | +0.02 BPP (free!) | None | PR50 | HIGH |
| **Int5 MLP post-quant** | None (post-quant) | Saves ~1-2MB | andrewgcodes PR4 | MEDIUM (need to verify) |
| **2% magnitude pruning** | None | Saves ~50-100KB | andrewgcodes PR4 | MEDIUM (not yet tested by us) |
| **TTT (Test-Time Training)** | +0.003-0.006 BPP (free!) | None | andrewgcodes PR4 | MEDIUM (not yet tested by us) |

### Architecture
| Technique | BPP gain | Artifact impact | Source | Confidence |
|-----------|----------|----------------|--------|------------|
| **10 layers** (vs 9) | +0.004 BPP | +~2MB | exp087 vs 084 | HIGH |
| **11 layers** (vs 9) | +0.012 BPP | +~5MB | exp076 vs 075 | HIGH (if fits) |
| **MLP 3x expansion** (h=1536) | Standard | Standard | PR128/135 | HIGH |
| **relu² activation** (for MLP) | Baseline | Standard | PR128 | HIGH |

---

## PROVEN NEUTRAL (don't bother)

| Technique | Result | Source |
|-----------|--------|--------|
| ROPE_BASE=200K | No effect | exp066 |
| SWA with WD ≤ 0.04 | Neutral to slightly worse | exp074, exp011 |
| Sinusoidal/fixed bigram | Worse than learned bigram | exp073 |
| Seed changes | Noise (~0.003 variance) | multiple runs |

---

## PROVEN HURTS (avoid these)

| Technique | How much it hurts | Why | Source |
|-----------|------------------|-----|--------|
| **Int5 QAT during training** | -0.008 BPP, +360KB artifact | Constrains MLP weights too aggressively | exp088 |
| **Bit-packing int6** | Worse compression | Removes redundant zero bits that zstd exploits | exp059 |
| **Outlier splitting** | Bigger artifact | Extra tensors overhead | exp065 |
| **abs² activation** | -0.004 BPP | relu²'s zero-gating helps sparsity | exp083 |
| **Partial RoPE 25%** | -0.006 BPP | Model needs position info on all dims | exp085 |
| **Aggressive warmdown (6000 iters)** | -0.002 BPP | Too long cooldown | exp062 |
| **SwiGLU** (at this scale with int6) | -0.004 BPP vs relu² 3x | relu² compresses better | exp053 |
| **BitNet 1.58-bit** | Terrible quality | Way too aggressive quantization | PR139 |
| **LAWA/SWA without WD** | Hurts BPP | Weight averaging without regularization is noisy | exp011, 015, 043 |

---

## UNTESTED BUT PROMISING (from andrewgcodes PR4)

| Technique | Expected gain | Priority | Notes |
|-----------|--------------|----------|-------|
| **WD=0.08** (double ours) | +0.002 BPP + better compression | HIGH | They use this in winning config |
| **SWA with WD=0.08** | +0.002 BPP | HIGH | SWA works when WD is high enough |
| **Attention gate** (sigmoid per-head) | Unknown | MEDIUM | 96 params/layer, zero-init |
| **Bigram 16384x64** (vs our 4096x128) | +0.005 BPP | HIGH | Bigger table, smaller dim |
| **Muon momentum=0.99** | +0.001 BPP | MEDIUM | Higher than our 0.95 |
| **No QAT** (STE disabled) | Unknown | MEDIUM | Andrewgcodes' winner has STE OFF |

---

## KEY INSIGHT: COMPRESSION vs QUALITY TRADEOFF

NorMuon gives +0.004 BPP but costs ~2MB in artifact size (weights have higher entropy → compress worse).
At 9 layers this is fine (15.76MB < 16MB). But at 10+ layers, the extra 2MB pushes us over budget.

**Options:**
1. Keep NorMuon, use int5 MLP + pruning to recover compression → need to verify
2. Drop NorMuon, use plain Muon like PR198 → lose 0.004 BPP but gain ~2MB headroom
3. Use WD=0.08 to force weight sparsity → may recover compression while keeping NorMuon
4. Accept 9 layers with NorMuon + all techniques → safe but limited BPP

**The andrewgcodes approach**: No NorMuon, no QAT, but WD=0.08 + SWA + TTT + big bigram → gets 1.1385 with 10L at 15.87MB.

---

## A/B TEST PLAN (starting from PR198 base)

Each test changes ONE variable from the previous best. Run on 8xH100, full 10 min.

### Round 1: Core training improvements (9L, using PR198 script)
- **A**: PR198 baseline (plain Muon, relu², WD=0.04) → establishes our baseline on this script
- **B**: A + leaky_relu(0.5)² → test activation improvement
- **C**: A + NorMuon → test optimizer improvement (watch artifact size!)
- **D**: A + WD=0.08 → test high weight decay
- **E**: B + D (leaky2 + WD=0.08) → combine if both help

### Round 2: Post-training techniques (on Round 1 winner)
- **F**: Winner + int5 MLP post-quant → test compression savings
- **G**: Winner + 2% pruning → test compression
- **H**: Winner + TTT 5 epochs → test free BPP gain
- **I**: Winner + F + G + H → combine all post-training wins

### Round 3: Architecture scaling (on Round 2 winner)
- **J**: Winner with 10 layers → test if it fits under 16MB
- **K**: Winner with 11 layers → test if it fits
- **L**: Winner + BigramHash 16384x64 → test bigram

### Round 4: Advanced techniques
- **M**: Winner + SWA (30%/20 steps) → test with high WD
- **N**: Winner + attention gate → test per-head gating
- **O**: Winner + Muon momentum=0.99 → test higher momentum

---

## ARTIFACT SIZE ROOT CAUSES (SOLVED)

Our script produces 15.78MB vs PR198's 12.71MB for identical 9L model. Five causes found:

| Cause | Raw overhead | Compressed overhead | Status |
|-------|-------------|-------------------|--------|
| **FP16_KEEP_NAME_PATTERNS** | 764KB | **1.0-1.5MB** | ✅ Fixed (set to "") |
| **OUTLIER_QUANT=1** (default!) | 129KB | **0.2-0.4MB** | ❌ MUST set to 0! |
| **Code size** (95KB vs 65KB) | 31KB | 31KB | ❌ Need to strip unused code |
| **Compression ratio drag** | 0 | **0.5-1.0MB** | ✅ Fixed with FP16_KEEP removal |
| **NorMuon weight entropy** | 0 | **~0.5MB** (estimated) | Tradeoff — gives +0.004 BPP |

**Key insight**: FP16 data compresses at ~1.0x while int8 compresses at ~1.8x. Each MB of fp16 in the artifact costs almost 1MB compressed, vs 0.56MB for the int8 it replaces. This is why FP16_KEEP was so expensive.

**CRITICAL FIX for all future runs**: Always set `OUTLIER_QUANT=0` and `FP16_KEEP_NAME_PATTERNS=""`

---

## CONCRETE EXPERIMENTS (starting from PR198 base)

### Immediate priority: Build clean PR198-based script
1. Take PR198's train_gpt.py (proven 12.71MB)
2. Add ONLY these proven features (one at a time, verify each):
   - leaky_relu(0.5)² activation (+0.0014 BPP, neutral artifact)
   - int5 MLP post-quant (saves ~1-2MB, no QAT needed)
   - 2% magnitude pruning (saves ~50-100KB)
   - TTT 5 epochs post-quant (free +0.003-0.006 BPP)
   - Sliding window eval stride=64 (free +0.02 BPP)
3. Keep OUTLIER_QUANT=0, FP16_KEEP=""
4. Use simple torch.save + zstd22 (PR198's format)

### PR198 EXACT SUBMISSION CONFIG (from their PR description)
```
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048
MUON_WD=0.04 ADAM_WD=0.04
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000
ITERATIONS=9000 EVAL_STRIDE=64
```
PR198 result: **1.1318 BPP sliding, 15.7MB, 7412 steps @ 81ms/step**

### RESULTS SO FAR (clean_train.py based on PR198)
| Exp | Config | Sliding BPP | Artifact | Fits? | Notes |
|-----|--------|-------------|----------|-------|-------|
| **090** | 9L PR198 defaults (not tuned) | **1.1586** | **13.14MB** ✅ | ✅ | 78ms/step, 7660 steps. |
| **091** | 11L + WD=0.04 + leaky2 (missing momentum!) | **1.1634** | **12.25MB** ✅ | ✅ | Missing momentum=0.99! |
| **093** | PR198 EXACT + leaky2 + int5 + prune | **1.1549** | **13.71MB** ✅ | ✅ | Int5 HURT BPP by ~0.017! |
| **094** | PR198 EXACT (no changes) — BASELINE | **1.1374** | **15.69MB** ✅ | ✅ | 95ms/step, 6343 steps. Matches PR198! |
| **095** | PR198 EXACT + leaky2 ONLY | **1.1355** | **15.71MB** ✅ | ✅ | **leaky2 CONFIRMED +0.002 vs 094!** |
| **096** | PR198 EXACT + leaky2 + TTT | **1.1312** | **16.01MB** ❌ | ❌ | 7.8KB over! TTT=361s (slow). BPB great. |
| **097** | PR198 EXACT + leaky2 + TTT + prune 2% | **1.1314** | **15.72MB** ✅ | ✅ | **SUBMISSION READY! Prune didn't hurt.** |
| **098** | PR198 EXACT + leaky2 + TTT + FLAT serial | **1.1312** | **15.75MB** ✅ | ✅ | FLAT was BIGGER, fell back to torch.save. |
| **100** | PR198 EXACT + leaky2 + TTT + prune + **WD=0.08** | **1.1371** | **13.47MB** ✅ | ✅ | WD=0.08 HURTS val_bpb by 0.006 but great compression. |

### KEY LEARNINGS FROM A/B TESTS
- **leaky2: CONFIRMED +0.002** — exp095 (1.1355) vs exp094 (1.1374). Clean win, no artifact cost.
- **Int5 MLP post-quant: RULED OUT** — costs 0.017 BPP for 2MB savings. Not worth it at 11L (15.69MB fits fine).
- **PR198 reproduction: SUCCESS** — 1.1374 vs their 1.1318. Gap is ~0.006 from ~1000 fewer steps (no FA3).
- **Artifact size: SOLVED** — 15.69-15.71MB matches PR198's 15.7MB perfectly.
- **PR254 (new record 1.1303)**: Just PR198 + TTT (lr=0.002, 3ep, freeze 2). Simple wins.

### Planned experiments (properly based on PR198 exact config)
| Exp | Change from PR198 exact | Expected BPP | Priority |
|-----|------------------------|-------------|----------|
| **093** | + leaky2 + int5 post-quant + 2% prune | ~1.13? | RUNNING |
| **094** | PR198 exact (NO changes) — pure reproduction | ~1.13 | NEXT (baseline) |
| **095** | 093 + TTT 5ep | ~1.125 | HIGH |
| **096** | 093 + WD=0.08 (vs PR198's 0.04) | ~1.128 | HIGH |
| **097** | Best winner + BigBigram 16384x64 | ~1.12 | HIGH |
| **098** | Best + attention gate | ~1.12? | MEDIUM |
| **099** | Best + NorMuon | ~1.116? | MEDIUM (watch size!) |
| **095** | 094 + BigBigram 16384x64 | ~1.125 | +0.5-1MB | HIGH |
| **096** | 095 + SWA (30%/20) | ~1.123 | same | HIGH |
| **097** | 096 + attention gate | ~1.12? | +tiny | MEDIUM |

### Fallback: 9L quick wins (if 11L doesn't work)
| Exp | Change from PR198 | Expected BPP | Expected size | Priority |
|-----|-------------------|-------------|--------------|----------|
| **098** | 9L + leaky2 + WD=0.08 + int5 + prune | ~1.15 | ~11.5MB | backup |
| **099** | + TTT 5ep | ~1.145 | same | backup |

### Round 2: Scale up layers (on Round 1 winner)
| Exp | Change | Expected BPP | Expected size | Priority |
|-----|--------|-------------|--------------|----------|
| **100** | Try 12 or 13 layers (if 11L artifact is small enough) | ~1.11? | ~15-16MB | stretch |
| **101** | + BigramHash 16384x64 | ~1.105? | +0.5-1MB | stretch |

### Round 3: Advanced (on Round 2 winner)
| Exp | Change | Expected BPP | Notes |
|-----|--------|-------------|-------|
| **100** | + SWA (30%/20) | ~1.123 | Needs WD=0.08 |
| **101** | + attention gate | ~1.12? | 96 params/layer |
| **102** | + Muon momentum=0.99 | ~1.12? | Longer warmup 1500 steps |
| **103** | + NorMuon | ~1.116? | Watch artifact size! |

### Things to ALSO test (lower priority)
| Exp | What | Why |
|-----|------|-----|
| **104** | No QAT at all (like andrewgcodes winner) | Their best has STE OFF |
| **105** | stride=256 eval (instead of 64) | PR114 showed it might be better |
| **106** | BigramHash 4096x128 vs 16384x64 | Different size/dim tradeoff |
| **107** | MLP_HIDDEN=1280 with 11L | Trade width for depth |

---

## SIMPLICITY CRITERION

All else being equal, simpler is better.
- Small improvement + ugly complexity? Not worth it.
- Removing something and getting equal/better results? Great — simplification win.
- ~0 improvement but much simpler code? Keep the simpler version.
- 0.001 BPP from 20 lines of hacky code? Probably not worth it.
- 0.001 BPP from deleting code? Definitely keep.
