# Parameter Golf — Official Leaderboard & Technique Evolution

*Fetched from openai/parameter-golf main README, 2026-05-05. Ordered best→worst BPB.*

---

## Official Leaderboard (10-min / 16 MB track)

| # | PR | BPB | Author | Date | Key Techniques |
|---|-----|-----|--------|------|----------------|
| 1 | #2135 | **1.0565** | codemath3000 | 2026-05-01 | Calib32 token-only n-gram + AsymLogit stack |
| 2 | #2014 | **1.0576** | simonbissonnette | 2026-04-30 | Progressive 1k→2k→3k context, short-doc TTT chunks |
| 3 | #1953 | **1.0586** | andrewbaggio1 | 2026-04-30 | EVAL_SEQ_LEN=2560, no-Q/V TTT mask, TTT LR 0.75, QK_GAIN=5.25 |
| 4 | #1945 | **1.0594** | alertcat | 2026-04-29 | AWQ-lite GPTQ + asymmetric logit rescaling |
| 5 | #1855 | **1.0611** | codemath3000 | 2026-04-27 | LQER + SparseAttnGate + per-group lrzip + 9 greedy hparam overrides |
| 6 | #1851/#1868 | **1.0614** | aquariouseworkman | 2026-04-27 | BOS-fixed SmearGate + LQER asymmetric + SparseAttnGate + phased TTT |
| 7 | #1787 | **1.0634** | nprime06 | 2026-04-23 | Polar Express NS, MIN_LR=0.1, SparseAttnGate, fused CE |
| 8 | #1769 | **1.0645** | dexhunter | 2026-04-22 | MLPClip σ=12, SmearGate + LoRA-TTT refinements |
| 9 | #1736 | **1.0655** | dexhunter | 2026-04-19 | CaseOps + GatedAttn + QuantGate + Loop45 + Phased TTT |
| 10 | #1729 | **1.0678** | romeerp | 2026-04-18 | CaseOps tokenizer + tapered WD + phased TTT |
| 11 | #1667 | **1.0714** | MarioPaerle | 2026-04-16 | SmearGate + attention output gate + score-first TTT |
| 12 | #1626 | **1.0719** | dexhunter | 2026-04-14 | VarLen attn, fused MLP, multi-phase global SGD TTT, int7 embeddings |
| 13 | #1610 | **1.0728** | romeerp | 2026-04-13 | Phased TTT (first appearance) |
| 14 | #1530 | **1.0734** | samacqua | 2026-04-11 | VarLen FA3 attn, fused Triton MLP, doc-independent LoRA TTT |
| 15 | #1529 | **1.0758** | msisovic | 2026-04-11 | Parallel residuals PARALLEL_START=8, CUTLASS EVT/Triton kernels |
| 16 | #1514 | **1.0798** | dexhunter | 2026-04-09 | SP8192 + Muon 0.97 + legal score-first TTT |
| 17 | #1493 | **1.0810** | bigbag | 2026-04-09 | 3-layer recurrence + parallel residuals + QK-Gain 5.25 + TTT |
| 18 | #1477 | **1.0822** | aryanbhosale | 2026-04-08 | Parallel residuals + score-first TTT |
| 19 | #1413 | **1.0828** | dexhunter | 2026-04-06 | QK-Gain 5.0 + legal score-first TTT on SP8192 |
| 20 | #1412 | **1.0835** | Robby Sneiderman | 2026-04-06 | Parallel residuals + Hessian-aware SDClip |
| 21 | #1394 | **1.0856** | Kevin Clark | 2026-04-05 | SP8192 + GPTQ embeddings + Loop4-5 + MuonEq-R + SDClip |
| 22 | #1334 | **1.0897** | aryanbhosale | 2026-04-04 | SP4096 + depth recurrence + parallel residuals + MuonEq-R |
| 23 | #1285 | **1.0912** | dexhunter | 2026-04-03 | MuonEq-R + Loop4-5 + WD=0.090 + all-int6 GPTQ |
| 24 | #1218 | **1.0979** | Kevin Clark | 2026-04-01 | SP4096 + 4× MLP + high WD (stripped TTT, hash embeddings, SmearGate) |
| 25 | #1204 | **1.1063** | msisovic | 2026-03-31 | Mini recurrence layers 4-5 + two-lane parallel residuals (first appearance) |
| 26 | #1120 | **1.1099** | newjordan | 2026-03-30 | XSA-all + Parallel Muon + coprime loader + Bigram2048/RoPE16 |
| 27 | #1060 | **1.1122** | dexhunter | 2026-03-29 | Coprime loader + full Hessian GPTQ + XSA all 11 layers |
| 28 | #1019 | **1.1147** | abaybektursun | 2026-03-25 | AR self-gen GPTQ calibration + all-layer XSA |
| 29 | #549 | **1.1194** | abaybektursun | 2026-03-23 | LeakyReLU² + legal score-first TTT (first legal TTT) + Parallel Muon |
| 30 | #374/#414 | **1.1228** | signalrush | 2026-03-22 | GPTQ-lite clip search + EMA + QAT@0.15 |
| 31 | #315 | **1.1248** | jfprincz | 2026-03-21 | Partial RoPE (16/64 dims) + LN scale + EMA + XSA on 4 layers |
| 32 | #287 | **1.1271** | jfprincz | 2026-03-20 | XSA on last 4 layers + EMA replacing SWA |
| 33 | #265 | **1.1307** | unnir | 2026-03-20 | Partial XSA on deepest 3 layers (first XSA) |
| 34 | #180 | **1.1428** | thwu1 | 2026-03-20 | Mixed int5/int6, BigramHash(10240), SWA(0.4) |
| 35 | #162 | **1.1458** | raahilshah | 2026-03-20 | 3× MLP + SmearGate + BigramHash + OrthoInit (first SmearGate) |
| 36 | #86 | **1.1502** | aruniyer | 2026-03-20 | 11 layers + 3× MLP + int6 QAT (first 11L) |
| 37 | #65 | **1.1556** | aquariouseworkman | 2026-03-19 | SmearGate + BigramHash + 3× MLP + int6 STE QAT |
| 38 | #640 | **1.1570** | CiprianFlorin-Ifrim | 2026-03-24 | 73.7M ternary quant + U-Net + SP8192 + YaRN |
| 39 | #63 | **1.1586** | yahya010 | 2026-03-19 | 10L + int6 QAT + zstd-22 |
| 40 | #60 | **1.1748** | notapplica | 2026-03-19 | Sliding window + FP16 embed + 10L + Muon WD |
| 41 | #50 | **1.1925** | mattqlf | 2026-03-19 | Sliding window eval stride=64 (first sliding window) |
| 42 | #77 | **1.1928** | samacqua | 2026-03-19 | First TTT (LoRA, non-legal) |
| 43 | #52 | **1.2014** | Spokane Way | 2026-03-19 | 4k seq length |
| 44 | #49 | **1.2060** | Spokane Way | 2026-03-18 | 2048 seq length |
| 45 | #39 | **1.2147** | Nan Liu | 2026-03-18 | Mixed int8/int6 quantization |
| 46 | #42 | **1.2197** | Renier Velazco | 2026-03-18 | FP16 tied embedding |
| 47 | Baseline | **1.2244** | OpenAI | 2026-03-17 | 9L/512d/SP1024, tied embeddings, 4 KV heads |

---

## Technique Evolution by Phase

### Phase 1 — First 48 Hours (March 18–20)
Rapid parallel exploration of obvious levers. Multiple competitors independently found:
- Sequence length: 1024 → 2048 → 4096 (Δ ≈ −0.02 BPB). Persisted to final SOTA (eventually 3072 eval).
- Sliding window eval (Δ ≈ −0.015). Permanent fixture.
- 10→11 layers. 11L became standard from #86 onward.
- 3× MLP width. Standard until 4× in #1218.
- SmearGate (#162, first appearance). Persisted all the way to final SOTA (BOS-fixed in #1851).
- int6 quantization (#39). Evolved into GPTQ, persists to final SOTA.
- FP16 embeddings. Eventually replaced by int7 GPTQ embeddings (#1626).

### Phase 2 — XSA and GPTQ Maturation (March 20–25)
- XSA introduced (#265), expanded to 4 layers (#287), eventually all 11 (#1060). Permanent.
- EMA replacing SWA (#287). EMA_DECAY=0.9965 frozen since introduction — never revisited.
- Partial RoPE, LN Scale (#315). Permanent.
- GPTQ-lite (#374). Evolved through full Hessian GPTQ (#1060) → GPTQ embeddings (#1394) → LQER (#1851). Permanent.
- Legal score-first TTT (#549). Paradigm shift. TTT became mandatory in all top submissions.
- LeakyReLU² activation (#549, #493). Still in final SOTA.

### Phase 3 — Tokenizer Leap and System Work (March 29 – April 1)
- Full Hessian GPTQ (#1060). Permanent.
- SP4096 (#1218, Kevin Clark). First vocabulary leap. 4× MLP, higher WD.
- Parallel residuals, first depth recurrence (#1204). Both permanent.

### Phase 4 — SP8192 Era (April 3–11)
- SP8192 (#1394, Kevin Clark). Second vocabulary leap. Largest single-step architectural gain.
- GPTQ embeddings, SDClip (#1394). Permanent.
- QK-Gain 5.0→5.25 (#1413, #1953). Permanent.
- VarLen attention + fused Triton MLP (#1530). Permanent.
- Phased TTT (#1610, #1626). Became standard; PHASED_TTT_NUM_PHASES=3.

### Phase 5 — CaseOps and Fine Stacking (April 13–May 1)
- CaseOps tokenizer (#1729). Paradigm shift. All final SOTA uses CaseOps.
- SmearGate BOS fix (#1851). Cleaned up a longstanding bug.
- LQER asymmetric rank-4 (#1851). Permanent.
- SparseAttnGate, Polar Express NS, MIN_LR=0.1 (#1787). Permanent.
- AWQ-lite, AsymLogit rescaling (#1945). In final SOTA.
- Progressive context 1k→2k→3k (#2014). In final SOTA.
- No-Q/V TTT mask, EVAL_SEQ_LEN=2560 (#1953). In final SOTA.
- GPTQ_CALIBRATION_BATCHES=32 (#2135). Final accepted SOTA (one hyperparameter change).

---

## What Persisted vs What Was Dropped

### Persisted to final SOTA (#2135)
- 11 layers (#86)
- Sliding window / VarLen eval (#50 → #1530)
- GPTQ int6 full Hessian (#374 → #1060)
- EMA decay=0.9965 (#287) — frozen since introduction
- Partial RoPE (#315)
- SmearGate BOS-fixed (#162 → #1851)
- XSA all layers (#265 → #1060)
- Depth recurrence Loop3-5 (#1204/#1285)
- Parallel residuals from layer 8 (#1204 → #1529)
- SP8192 + CaseOps (#1394 + #1729)
- LQER asymmetric (#1851)
- SparseAttnGate (#1787)
- Phased score-first TTT (#1610)
- LeakyReLU² MLP activation (#549/#493)
- MIN_LR=0.1 warmdown floor (#1787)

### Introduced but dropped
- SWA → replaced by EMA (#287)
- zstd compression → replaced by per-group lrzip+brotli
- OrthoInit — mentioned early, absent from April stack
- Value residuals — Kevin Clark explicitly removed in #1218, got a gain
- BigramHash — popular March, absent April; superseded by larger vocab
- AR self-gen GPTQ calibration → replaced by training-shard Hessian + calib tuning
- MuonEq-R → replaced by Muon 0.97 (#1514)
- Illegal LoRA TTT (#77) → replaced by legal score-first TTT

---

## Three Paradigm Shifts

1. **Legal TTT** (~March 23, #549): Opened eval-time compute as a new optimization axis.
2. **SP4096→SP8192** (~April 1–5): Vocabulary as primary architecture decision. Freed artifact bytes from embedding tables; biggest single architectural jump.
3. **CaseOps** (~April 18–19, #1729): Lossless tokenizer transform. All top submissions converged on SP8192+CaseOps.

---

## Approximate Single-Step BPB Gains (largest to smallest)
1. SP8192 vocabulary jump: ~−0.028 BPB
2. Legal TTT (cumulative over TTT steps): ~−0.025 BPB
3. XSA all layers (cumulative): ~−0.025 BPB
4. SP4096 vocabulary jump: ~−0.013 BPB
5. Sliding window eval: ~−0.015 BPB
6. Depth recurrence: ~−0.010–0.020 BPB
7. Parallel residuals: ~−0.005–0.010 BPB
8. CaseOps: ~−0.002 BPB direct (enables further composition)
9. LQER asymmetric: ~−0.003–0.005 BPB
10. GPTQ_CALIBRATION_BATCHES=32: ~−0.001 BPB (final SOTA step)
