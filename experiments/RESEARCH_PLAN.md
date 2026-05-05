# Research Plan — Continuous Experiments

## Current Best SUBMISSION-READY: exp084 — 1.1427 BPP sliding, 15.76MB ✅ (9L + leaky_relu(0.5)²)
## Best BPP overall: exp076 — 1.1304 (11L all features, 18.95MB ❌)
## Best 10L: exp087 — 1.1391 sliding, 16.25MB ❌ (252KB over — INT5_MLP should fix!)
## Target: ≤ 1.13 BPP + ≤ 16MB artifact (beat andrewgcodes PR#4's 1.1385)
## KEY DISCOVERY: FP16_KEEP_NAME_PATTERNS was the cause of our artifact bloat, NOT hardware.
## INT5 MLP quantization (PR219/andrewgcodes) should save ~2MB → unlocks 10+ layers!

---

## STATUS SUMMARY (updated Mar 20, 2026)
- Best submission-ready: **exp084 — 1.1427 BPP** (9L, leaky2, 15.76MB ✅)
- Best BPP ever: **exp076 — 1.1304** (11L, 18.95MB ❌)
- Best 10L: **exp087 — 1.1391** (16.25MB ❌, 252KB over)
- **exp088 DONE**: 10L + INT5_MLP (with int5 QAT) → 1.1468 sliding, 16.61MB ❌ — INT5 QAT HURT both BPP and compression!
- **exp089 RUNNING**: 10L + INT5 post-quant only + TTT + 2% pruning, 8×H100
- Competition target: **andrewgcodes PR#4 at 1.1385 BPP, 15.87MB ✅**
- ROOT CAUSE FOUND: FP16_KEEP was part of our artifact bloat, but there's still a ~2MB gap vs PR198's serialization.
- **CRITICAL INVESTIGATION**: PR198 exact script → 12.71MB on our machine. Our script → 15.78MB for same params. 3MB gap from OUR CODE. Threads investigating.

## ACTIVE EXPERIMENT QUEUE (priority order)
1. **089 (RUNNING)**: 10L + INT5 post-quant only (no int5 QAT!) + TTT 5ep + 2% pruning + leaky2
2. **INVESTIGATE ARTIFACT BLOAT** ← threads running: why is our artifact 3MB bigger than PR198?
3. **090**: Fix artifact bloat (adopt PR198's serialization?) + 10L + all techniques
4. **091**: 090 + **BigramHash 16384×64** (andrewgcodes size) → ~0.006 BPP
5. **092**: 091 + **WD=0.08** + **SWA** (start=30%, every=20 steps)
6. **093**: Best config + **11 layers** (if artifact bloat is fixed, 11L should easily fit)
7. **094**: Best config + **attention gate** (per-head sigmoid, 96 params/layer)
8. **095**: Best config + **Muon momentum=0.99** (warmup 0.92→0.99 over 1500 steps)

## EXP088 KEY LEARNING: INT5 QAT HURTS
- Int5 fake quantization DURING TRAINING ([-16,15] range for MLP) degraded both BPP (-0.008) and compression (+360KB)
- Andrewgcodes likely uses int6 QAT for all layers during training, int5 only at post-training serialization
- **Thread investigating their exact code** → andrewg-int5 thread

## ARTIFACT SIZE MYSTERY (3MB gap — threads investigating)
| Config | PR198 artifact | Our artifact | Gap |
|--------|---------------|-------------|-----|
| 9L, same params | 12.71MB | 15.78MB | **3.07MB** |
| Removing FP16_KEEP | ~12.71MB | ~15.0MB | **~2.3MB** |
Possible causes being investigated:
- Our serialization format overhead (manual JSON headers, extra metadata)
- Extra tensor storage (outlier/blockwise quant structures even when disabled)
- Different handling of small/non-matrix tensors
- Code size difference (our script is bigger = more code bytes in artifact)
**SOLUTION**: Possibly adopt PR198's simpler serialization directly

## NEW TECHNIQUES FROM ANDREWGCODES PR#4 (see ANDREWG_ALL_PRS_ANALYSIS.md)
| Technique | Expected BPP gain | Implemented? | Tested? |
|-----------|-------------------|-------------|---------|
| **TTT (Test-Time Training)** | ~0.006 | ✅ in script | exp089 RUNNING |
| **INT5 MLP post-quant** | enables 10L (saves ~2MB) | ✅ in script | exp089 RUNNING |
| **2% magnitude pruning** | better compression | ✅ in script | exp089 RUNNING |
| **INT5 MLP QAT** | — | ✅ but HURTS | exp088 ❌ RULED OUT |
| WD=0.08 + SWA(30%/20) | ~0.002 | partially | NOT YET |
| Attention gate | unknown | NOT YET | NOT YET |
| Bigram 16384×64 | ~0.006 | NOT YET (need artifact fix first) | NOT YET |
| Muon momentum 0.99 | ~0.001-0.002 | env var change | NOT YET |

---

## COMPRESSION TRACK — Must fit ≤16MB

### Tested Compression Methods (exp063-066)
| Method | Size | vs torch.save | Verdict |
|--------|------|---------------|---------|
| torch.save + zstd-22 | 18.55MB | baseline | — |
| Manual serialization + zstd | 17.56MB | -1.0MB | ✓ helps |
| Manual + dtype grouping | 17.50MB | -1.05MB | ✓ marginal |
| Outlier splitting + manual + zstd | 18.85MB | WORSE | ✗ hurts! |
| **FLAT concat + zstd** | **17.73MB** | **-0.82MB** | **✓ BEST** |
| FLAT + LZMA | 17.88MB | -0.67MB | ✓ close |
| Manual + LZMA | 17.88MB | -0.67MB | ✓ |
| Bit-packing int6 | 18.64MB | WORSE | ✗ hurts! |

### Untested Compression Methods (from ChatGPT + Grok)
| Method | Expected impact | Priority | Status |
|--------|----------------|----------|--------|
| **Blockwise int6 quant (block=64)** | Unknown — more scales but tighter quant | HIGH | In script, env BLOCKWISE_QUANT=1 |
| **uint8 scales** | Save ~50% on scale storage | HIGH | In benchmark script |
| **Reduce model params (MLP_HIDDEN=1344)** | -1.77MB raw → fits budget | HIGH | exp067 RUNNING |
| Layer-type tensor grouping (MLP then attn) | Small improvement | LOW | — |
| Mixed compressor (LZMA for weights, zstd for scales) | Unknown | LOW | — |
| Test on Runpod | May solve problem entirely | **CRITICAL** | Not done yet |

### Compression Experiment Queue
- **067**: MLP_HIDDEN=1344, no BigramHash → RUNNING (tests if smaller model fits)
- **068**: Run compression_benchmark.py on GPU with all schemes (blockwise, uint8 scales, etc.)
- **069**: Test on Runpod to see actual submission artifact size

---

## MODEL TRACK — Lower BPB

### Confirmed Techniques (in our best config exp060)
| Technique | BPB impact | Source |
|-----------|-----------|--------|
| relu² 3x MLP (h=1536) | baseline | PR128/135 |
| NorMuon optimizer | +0.004 vs plain Muon | exp058 vs 057 |
| int6 STE QAT | +0.002 (zero overhead!) | exp060 vs 058 |
| OrthoInit + muP scaling | +0.001 | PR135 |
| SmearGate | +0.001 | PR135 |
| BigramHash | +0.001 | PR135 |
| seq2048 + batch786K | best for sliding eval | PR114/135 |
| grad_clip=0.3 | stability | PR114/135 |
| weight_decay=0.01 | helps quantization | PR135 |
| Sliding window eval stride=64 | +0.02 vs standard | PR50 |

### Untested Model Techniques
| Technique | Source | Expected BPB | Priority | Status |
|-----------|--------|-------------|----------|--------|
| **11 LAYERS + WD=0.038 + LR=0.025** | **PR179 (NEW SOTA 1.1472!)** | +0.005-0.01 | **CRITICAL** | 11 layers fits under 16MB with int6+WD! Must test ASAP! |
| **Stride=256 eval** | PR114, PR164 | +0.000-0.005 | **HIGHEST** | FREE — no retraining, just eval change |
| **Partial RoPE (10-25% head dim)** | Grok, nanoGPT speedruns | +0.000-0.003 | HIGH | Easy 8-line change, proven in speedruns |
| **ALiBi (Attention Linear Biases)** | Grok, BLOOM/MPT | +0.005-0.02? | HIGH | Use attn_mask in SDPA or manual QK^T+bias. Biggest potential. |
| **DroPE (drop RoPE at end of training)** | Grok, Sakana AI paper 2512.12167 | +0.000-0.003 | HIGH | Drop PE after 90% training, recalibrate 200 steps |
| **ALiBi (Attention Linear Biases)** | Grok, BLOOM/MPT papers | +0.005-0.02? | HIGH | Need custom attn path (can't use with enable_gqa in SDPA). Biggest potential upside. |
| **PoPE (Polar Position Embedding)** | Grok, arxiv 2509.10534 | Unknown | MEDIUM | OpenAI co-author paper, 10x extrapolation claim. Same param count as RoPE. |
| **Stride=256 eval (instead of 64)** | PR114, PR164 | +0.000-0.005 | HIGH | PR114: stride=256 BETTER than 64 (1.1574 vs 1.1579). PR164 uses it. UNTESTED by us! |
| **FA3 (FlashAttention 3)** | PR122/164 | +5-10ms faster → more steps | HIGH | PR164 gets 68ms with FA3. Need to install on animal machine. |
| **ROPE_BASE=200K** | andrewgcodes PR3 | NEUTRAL | DONE | exp066: no effect |
| **Weight decay 0.02** | PR155 | Unknown | LOW | Not tested |
| **Aggressive warmdown WD=6000** | PR145 | HURT (-0.002) | DONE | exp062: worse |
| **10 layers (reduce MLP to fit)** | PR155 | Unknown | MEDIUM | More depth, less width. Could help with 16MB budget. |

### Model Experiment Queue (prioritized — ACTIVE)

#### PHASE 1: WD sweep + bigram + SWA — RESULTS SO FAR
- **068**: NorMuon + QAT + WD=0.01, no BigramHash — ✅ 1.1513, 15.33MB ✅ (first submission-ready)
- **069**: + WD=0.02 — ✅ 1.1495, 15.34MB ✅ (+0.002 from WD)
- **070**: + learned BigramHash — ✅ 1.1458 BPB (ALL-TIME BEST!) but 17.38MB ❌ (bigram embed too big)
- **071**: + WD=0.03 — ✅ **1.1480, 15.33MB ✅** (WD keeps helping)
- **072**: + WD=0.04 — ✅ 1.1477, 15.32MB ✅ (diminishing returns — WD=0.03 is sweet spot)
- **073**: Sinusoidal bigram (fixed table) + WD=0.03 — ✅ 1.1495 ❌ WORSE than no bigram. RULED OUT.
- **074**: SWA + NorMuon + WD=0.03 — ✅ 1.1482 (SWA NEUTRAL with NorMuon+QAT. Don't use.)

#### PHASE 1b: Bigram + LR tuning (from PR198)
- **075**: Learned BigramHash + int6 quant — ✅ **1.1448, 15.80MB ✅ (BEST 9L submission-ready!)**
- **080**: BIGRAM_VOCAB_SIZE=2048 (vs 4096) — PR198 says "negligible BPB cost", saves 300KB
- **081**: WD=0.04 on BOTH Muon and AdamW (PR198 uses this)
- **082**: Higher LRs: SCALAR_LR=0.025, TIED_EMBED_LR=0.035 (PR198's tuning)
  - Expected: bigram embed compresses 25% better at int6 vs int8 → might fit under 16MB
- **076**: Outer product bigram (CastedLinear projection, inherently int6) — BACKUP

#### PHASE 2: Layer depth sweep (PR179 inspired)
- **076**: 11L + all features → ✅ 1.1304 BPP BEST EVER but 18.95MB ❌ (27.1M params too many)
- **077**: 11L no BigramHash → ✅ 1.1335 but 18.48MB ❌ (26.5M still too many)
- **078**: 11L MLP=1250 → ✅ 1.1452 but 16.38MB ❌ (380KB over — reduced MLP killed depth advantage)
- **079**: 10L MLP=1536 no BigramHash → ✅ 1.1398 but 16.93MB ❌
- **080**: 10L PR198 hparams → ✅ 1.1412 but 16.89MB ❌ (WD=0.04 barely helped compression)
- **081**: 9L PR198 hparams + BigramHash → ✅ **1.1441, 15.78MB ✅ (BEST SUBMISSION-READY)**
- **082**: 10L + clone fix → ✅ 1.1415 but 16.89MB ❌ (clone fix made ZERO difference — confirmed hardware issue)
- **083**: 9L abs² activation → ✅ 1.1480 ❌ WORSE than relu². RULED OUT.
- **084**: 9L leaky_relu(0.5)² → ✅ **1.1427, 15.76MB ✅ (NEW BEST! +0.0014 vs relu²)**
- **085**: 9L partial RoPE 25% + leaky2 → RUNNING NOW
- **086**: **INT5 MLP QUANTIZATION + 10 LAYERS** → NEXT (from PR219 — saves ~2MB, fits 10L under 16MB!)
- **087**: BIGRAM_VOCAB_SIZE=2048 + leaky2 + best config → saves ~200KB
- **088**: Int5 MLP + 11 layers (if 086 works)

### CRITICAL NEW TECHNIQUE: Int5 MLP Quantization (from PR219)
PR219 uses int5 [-16,15] for MLP weights, int6 for attention. MLP weights are less precision-sensitive.
Int5 stored in int8 has 3 zero high bits → zstd compresses at 1.88x vs 1.51x for int6.
Saves ~2MB for 10L model → 16.93MB - 2.0MB = **~14.9MB → FITS under 16MB!**
This could unlock 10 or even 11 layers on our platform!
Need to: add INT5_MLP quantization to our quantize code + modify QAT to train at int5 range for MLP.

### Layer sweep summary so far:
| Layers | MLP | BigramHash | Sliding BPP | Artifact | Fits? |
|--------|-----|-----------|-------------|----------|-------|
| 9 | 1536 | int6 (075) | **1.1448** | 15.80MB | ✅ |
| 9 | 1536 | int6 (075) | **1.1448** | 15.80MB | ✅ ← BEST 9L |
| 9 | 1536 | none (071) | 1.1480 | 15.33MB | ✅ |
| 10 | 1536 | none (079) | **1.1398** | 16.93MB | ❌ (930KB over) |
| 10 | 1536 | none (080) | *running — PR198 hparams (WD=0.04+LR=0.025)* | | ? |
| 11 | 1536 | all (076) | **1.1304** | 18.95MB | ❌ |
| 11 | 1536 | none (077) | 1.1335 | 18.48MB | ❌ |
| 11 | 1250 | none (078) | 1.1452 | 16.38MB | ❌ |

#### PHASE 3: Eval improvements — FREE, no retraining
- **080**: Stride=256 eval on best checkpoint

#### PHASE 4: Position encoding experiments
- **081**: Partial RoPE 10% head dim
- **082**: ALiBi implementation
- **083**: DroPE
- **084**: PoPE

#### PHASE 5: Architecture + speed
- **085**: FA3 installation + test

#### PHASE 6: Advanced techniques (from golf.agustif.com research garden)
- **086**: NuMuon (nuclear-norm-constrained training) — makes weights more compressible during training
- **087**: Recursive layer sharing with layer-wise LoRA — share layers but differentiate with tiny LoRA adapters (~1-2KB per layer)
- **088**: Rotational symmetry "free bits back" — exploit weight matrix rotational symmetries for better compression
- **089**: BackSlash (rate-constrained optimized training) — train with explicit artifact size constraint
- **090**: Byte allocation optimization — instead of uniform int6, allocate more bits to sensitive tensors and fewer to others

### Key insight from golf.agustif.com:
The design space "bends hard" because the cap is on the FINAL ARTIFACT. This rewards:
1. Reducing UNIQUE weights (not just nominal params) — recursive sharing!
2. Making weights EASIER to quantize and compress — NuMuon, compressible training
3. Spending more capability on COMPUTE and less on BYTES — depth recurrence, test-time compute
4. Controlling vocabulary and output-head overhead
5. Preserving quality AFTER serialization and decompression

Most promising unexplored direction: **recursive layer sharing with LoRA adapters**.
Our early recurrence experiments (exp002-020) failed because weight-sharing without differentiation
hurt quality. But LoRA adapters (~1-2KB per unique layer) could fix this — each "virtual layer"
shares the base weights but has a tiny unique LoRA adapter. This could give us 15+ effective layers
while storing only 9 unique layer sets + tiny LoRA deltas.
- **080**: 10 layers + MLP_HIDDEN=1280 (more depth, less width)

#### PHASE 5: Submission
- **081**: Best config with FLAT+zstd serialization → submission candidate
- **082-084**: 3 seed runs for statistical significance (p<0.01)
- **085**: Final Runpod submission run

### Hypothesis for each:
- **070 (BigramHash back)**: ✅ DONE — 1.1458 BPB but 17.38MB ❌. BigramHash embed (fp16) doesn't compress well.
- **071 (WD=0.03)**: WD=0.01→0.02 helped +0.002. Diminishing returns expected but worth testing.
- **072 (sinusoidal bigram)**: Table generated from seed, persistent=False → NOT in state_dict → zero artifact cost. The projection (CastedLinear 128→512) IS stored but gets int6 quantized → compresses well unlike fp16 embedding. Expected: ~1.148 BPB, ~15.4MB ✅.
- **073 (SWA + NorMuon + WD=0.02)**: Our LAWA/SWA failures were without WD. PR162 shows SWA works WITH WD=0.02. Maybe WD regularizes weights enough that SWA averaging helps instead of hurting.
- **074 (PR162 exact + QAT)**: Plain Muon + WD=0.02 + SWA got 1.1483. Adding our QAT (+0.002) could push to 1.146.
- **075 (outer product bigram)**: Uses CastedLinear(1024,512) as projection — gets int6 quantized → compresses MUCH better than the learned embedding table (fp16). May fit under 16MB while keeping bigram signal.
- **076 (stride=256)**: PR114 found stride=256 gives 0.005 BPB BETTER than stride=64. Free.
- **077 (partial RoPE)**: Only rotating 10% of head dims. Faster steps + potentially better quality.
- **078 (ALiBi)**: Linear bias for position → better sliding window extrapolation. Biggest potential but most work.
- **079 (DroPE)**: Position-free at eval = better sliding window generalization.
- **080 (PoPE)**: Polar RoPE fixes what/where entanglement.
- **081 (FA3)**: ~10ms/step faster → ~900 more steps in 10 min → ~0.003 BPB improvement.

### Key insight on bigram compression:
The learned BigramHash uses nn.Embedding(4096,128) = 524K params. Previously stored as int8 (classified as "other") + bigram.scale as fp32 control tensor.
**FIX IMPLEMENTED**: Changed `_classify_param` to classify `bigram.embed` as "mlp" → gets int6 quantization.
Changed CONTROL_TENSOR_NAME_PATTERNS: "bigram" → "bigram.scale" so only the scale scalar gets fp32 passthrough.
This should reduce bigram artifact cost from ~400KB to ~300KB (int6 vs int8).
**EXPERIMENT NEEDED**: Run with learned BigramHash + int6 quantization of bigram embed table.
- **076**: Learned BigramHash + int6 bigram embed quant + WD=0.03 → test if it fits under 16MB now

### Other bigram findings:
- Sinusoidal/fixed bigram (exp073): WORSE than no bigram. Random table adds noise. RULED OUT.
- Outer product bigram: Uses CastedLinear(1024→512) — already int6 quantized. Could work but untested.

---

## COMBINED EXPERIMENTS (compression + model)
- **076**: Best compression (FLAT+zstd or blockwise) + best model config → submission candidate
- **077-079**: 3 seed runs for statistical significance (p<0.01)
- **080**: Final Runpod submission run

---

## COMPETITION LANDSCAPE (updated Mar 20 02:30 AM EDT)
| PR | BPB (sliding) | Key techniques | Artifact |
|----|---------------|----------------|----------|
| **PR162** | **1.1483 (mean 3 seeds)** | Muon WD=0.02 + SWA + OrthoInit+SmearGate+BigramHash | 15.92MB ✅ |
| PR164 | 1.1524 | OrthoInit + SmearGate + BigramHash + int6 + FA3 + stride=256 | 15.4MB ✅ |
| PR135 | 1.1539 | OrthoInit + SmearGate + BigramHash + int6 + no QAT | 15.2MB ✅ |

### OUR BEST RESULTS
| Exp | BPB (sliding) | Config | Artifact | Status |
|-----|---------------|--------|----------|--------|
| **070** | **1.1458** | NorMuon+QAT+WD=0.02+BigramHash | 17.38MB ❌ | Best BPB, over budget |
| **072** | **1.1477** | NorMuon+QAT+WD=0.04+no bigram | 15.32MB ✅ | Near-best, fits |
| **071** | **1.1480** | NorMuon+QAT+WD=0.03+no bigram | 15.33MB ✅ | Best bang/buck |
| 069 | 1.1495 | NorMuon+QAT+WD=0.02+no bigram | 15.34MB ✅ | |
| 068 | 1.1513 | NorMuon+QAT+WD=0.01+no bigram | 15.33MB ✅ | First submission-ready |

**We BEAT PR162 (1.1483) with 071 (1.1480) and 072 (1.1477) — both fit under 16MB!**
**If we can fit BigramHash (exp075 with int6 bigram quant), we could reach 1.145 AND fit.**

**068 is our first SUBMISSION-READY config: 1.1513 BPB, 15.33MB ✅**
**PR162 is new target at 1.1483 — uses Muon WD=0.02 + SWA**
**069 testing Muon WD=0.02 to match/beat PR162**

---

## COMPLETED EXPERIMENTS (038-067)
| Exp | Config | Sliding BPB | Artifact | Key finding |
|-----|--------|-------------|----------|-------------|
| 038 | SwiGLU h=668, Muon, seq1024 | 1.1935 | 15.95MB ✅ | First good result |
| 048 | SwiGLU h=668, NorMuon+int6 | 1.1990 | 12.74MB ✅ | NorMuon worse for SwiGLU |
| 049 | SwiGLU h=1024, NorMuon+int6, seq4096 | 1.1685 | 16.29MB ❌ | Big model helps |
| 051 | SwiGLU h=1024, vocab2048, 8L | 1.1739 | 15.23MB ✅ | Losing layer hurt |
| 053 | relu² 3x, NorMuon, seq4096 | 1.1639 | 16.17MB ❌ | relu² beats SwiGLU |
| 054 | + OrthoInit+SmearGate+BigramHash | 1.1622 | 16.76MB ❌ | PR135 features help |
| 055 | seq2048+batch786K+noQAT | 1.1619 | 16.79MB ❌ | seq2048≈seq4096 sliding |
| 056 | MLP_HIDDEN=1400 | 1.1951* | 13.43MB ✅ | Too aggressive trim |
| 057 | PR135 exact reproduction | 1.1535 | 17.76MB ❌ | Reproduced! Platform size diff |
| 058 | PR135 + NorMuon | 1.1494 | 17.85MB ❌ | NorMuon helps +0.004 |
| 059 | + bit-packing | 1.1489 | 18.64MB ❌ | Bit-pack HURTS compression |
| **060** | **PR135+NorMuon+QAT** | **1.1474** | **18.05MB ❌** | **BEST BPB** |
| 061 | seq4096 variant | 1.1518 | 17.90MB ❌ | seq2048 better for sliding |
| 062 | WD=6000 | 1.1497 | 17.78MB ❌ | Aggressive WD hurt |
| 063 | + manual serialization | 1.1477 | 17.56MB ❌ | Manual saves 400KB |
| 064 | + dtype grouped | 1.1477* | 17.50MB ❌ | Grouping marginal |
| 065 | + outlier splitting (all methods) | — | 17.73MB ❌ | Outlier split HURTS; FLAT best |
| 066 | + ROPE_BASE=200K | 1.1469 | 17.73MB ❌ | RoPE base neutral |
| 067 | MLP=1344, no BigramHash | *running* | *pending* | Budget fit test |

## KEY LEARNINGS
1. relu² 3x > SwiGLU at 21M+ params with int6
2. NorMuon > plain Muon for relu² (+0.004 BPB)
3. int6 QAT is FREE (zero step overhead, +0.002 BPB)
4. LAWA/SWA always hurt
5. seq2048 best for sliding eval (more sliding boost than seq4096)
6. Outlier splitting HURTS compression (extra tensors overhead)
7. Bit-packing HURTS compression (less zstd-exploitable redundancy)
8. FLAT+zstd is best serialization format
9. ROPE_BASE changes are neutral
10. Artifact size gap (~1.7MB) is platform-specific — MUST test on Runpod
11. BigramHash adds ~0.001 BPB but ~760KB artifact cost — may not be worth it

## RULED OUT TECHNIQUES (tested or assessed, not pursuing)
| Technique | Why ruled out |
|-----------|--------------|
| SwiGLU | relu² 3x beats it at 21M+ params with int6 (exp053) |
| LAWA/SWA (without WD) | Hurts without weight decay (exp011, 015, 043, 048). BUT PR162 shows SWA works WITH WD=0.02 — retesting! |
| Bit-packing int6 | HURTS compression — zstd exploits zero'd high bits (exp059) |
| Outlier splitting | HURTS compression — extra tensor overhead (exp065/066) |
| Aggressive warmdown (WD=6000) | Hurts BPB (exp062) |
| ROPE_BASE=200K | Neutral — no effect (exp066) |
| Vocab 2048 + 8 layers | Losing 9th layer hurts more than vocab helps (exp051) |
| BitNet 1.58-bit | Way too much quality loss — PR139 got 1.2029, PR126 got 1.7510 |
| Plain Muon (for relu²) | NorMuon is better for relu² by +0.004 (exp053 vs 052) |
| xPos/Kerple | No nano-scale wins reported; extra compute (Grok assessment) |
| Learned absolute position embeddings | Eats params, hurts extrapolation (Grok assessment) |
| GRAPE | Adds learned group matrix → extra bytes (Grok assessment) |

## LAST PR CHECK: 2:27 AM EDT (Mar 20) — no new PRs beating us
## NEXT PR CHECK: ~3:27 AM EDT
