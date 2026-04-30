# Parameter Golf: Meta-Game Analysis & Submission DAG

*Generated 2026-04-05. Updated 2026-04-06 with full sweep of ~1426 PRs across 9 review agents.*
*Covers 22 merged records, 4 non-records, ~1426 cataloged PRs, and the full frontier.*

---

## 1. Executive Summary

Parameter Golf is a competition to minimize validation bits-per-byte (BPB) on a language modeling task, constrained to a 16MB artifact and 600s training budget. The baseline starts at 1.2244 BPB.

**Current state (April 6, 2026):**

- **Merged SOTA:** 1.1147 BPB (PR #1019, abaybektursun) -- stale, the unmerged frontier is far ahead.
- **Clean frontier:** 1.0800 BPB (PR #1408, abaybektursun) via dTTT + BigramHash on SP8192 stack, followed by PR #1420 at 1.08014 (triple loop + n-gram tilt) and PR #1413 at 1.08279 (SP8192 + legal TTT).
- **SLOT track:** Effectively dead. PR #1240 proved 100% causal violation rate via flip test. Every SLOT submission has been self-closed or contested.
- **Pre-quant TTT:** Contested, leaning illegal. Would remove ~0.034 BPB from some frontier claims.
- **Bandit/n-gram track:** 0.4162 BPB (PR #1379, causal n-gram mixer) -- legitimate but separate category.

The competition has fractured into three distinct tracks: clean standard (~1.080-1.085), SLOT (dead), and bandit/n-gram (~0.42-0.59). The clean frontier is where serious competition remains.

**The standard stack has evolved through 6 generations:**
1. Baseline (1.2244) -- int8+zlib, 9L, SP1024
2. Day 2 revolution (1.15-1.19) -- int6 QAT, MLP3x, sliding window, SmearGate
3. 11L+XSA (1.12-1.13) -- XSA, EMA, partial RoPE
4. TTT+LeakyReLU (1.11-1.12) -- legal TTT, Full GPTQ, parallel Muon
5. Depth recurrence era (1.085-1.10) -- depth recurrence, SP4096/8192, GPTQ embeddings
6. Current frontier (1.079-1.085) -- SP8192, SDClip, GPTQ embeddings, linear LR warmdown, Polar Express NS, legal TTT reintegrated

---

## 2. Complete DAG / Lineage

Each node is a submission. Edges mean "built on top of." Scores are post-quant val_bpb (lower = better).

```
                            +-----------------------------------------------------+
                            |           NAIVE BASELINE (1.2244)                    |
                            |  9L 512d SP1024 int8+zlib ReLU^2 Muon               |
                            +--+------+------+------+------+------+------+--------+
                               |      |      |      |      |      |      |
                +--------------+      |      |      |      |      |      +------------------+
                |                     |      v      v      v     v                          |
        +-------v-------+    +-------v--+  Seq2048 Sliding LoRA  Warmdown-          +-------v--------+
        | Lower LR      |    | FP16     |  (1.206) Window  TTT   Quant              | 10L Mixed      |
        | (1.2230)      |    | Embed    |  spokane (1.193) (1.19)(1.215)            | Precision      |
        | nanlliu       |    | (1.2197) |  -way   mattqlf samacq samuellarson       | (1.2147)       |
        +-------+-------+    | chonchiog|    |                                      | nanlliu        |
                |             +----+-----+   +----v-----+                           +-------+--------+
                |                  |         | Seq4096  |                                    |
                |                  |         | (1.2014) |                                    |
                |                  |         | spokane  |                                    |
                |                  |         +----------+                                    |
                +--------+---------+----------------------------------------------+---------+
                         |                                                        |
                         |  (These streams MERGE here)                            |
                         v                                                        v
              +--------------------------------------------------------------+
              |  CONVERGENCE POINT: "The 10L/11L Stack" (March 19-20)        |
              |  FP16 embed + 10L + int6 STE QAT + MLP3x + zstd-22 +       |
              |  sliding window + Muon WD + momentum 0.99                    |
              |                                                              |
              |  Multiple parallel submissions hit ~1.15-1.16:               |
              |  - SmearGate+OrthoInit+MuonWD (1.1556) PR#65                |
              |  - Int6 QAT + Zstd (1.1586) yahya010                        |
              |  - Mixed Quant Int6/Int8 (1.1630) aquariouseworkman          |
              +--+---------------------------+-------------------------------+
                 |                           |
     +-----------v------------+   +----------v--------------+
     | 11L MLP3x+Int6 QAT    |   | Int6 MLP3x+SmearGate    |
     | (1.1502) PR#198?       |   | +BigramHash (1.1458)    |
     | aruniyer               |   | raahilshah PR#162       |
     +-----------+------------+   +----------+--------------+
                 |                           |
     +-----------v------------+              |
     | 10L Int5-MLP           |              |
     | +BigramHash 10240      |              |
     | (1.1428) thwu1         |              |
     +------------------------+              |
                                             |
              +------------------------------+
              |
              v
   +-----------------------------+
   | 11L Efficient Partial XSA   |  <-- XSA introduced (arXiv:2603.09078)
   | (1.1307) PR#198             |
   | unnir                       |
   +----------+------------------+
              |
              v
   +-----------------------------+
   | 11L XSA4 + EMA             |  <-- EMA replaces SWA
   | (1.1271) PR#287             |
   | jfprincz                    |
   +----------+------------------+
              |
              v
   +-----------------------------+
   | 11L Partial RoPE + LN Scale |  <-- Zero-param tricks
   | (1.1248) PR#315             |
   | jfprincz                    |
   +----------+------------------+
              |
              v
   +-----------------------------+        +-------------------------+
   | 11L EMA + GPTQ-lite        |<-------| PR#374 (unnir)          |
   | + warmdown3500 (1.1228)    |        | VE128 shared value emb  |
   | PR#414 signalrush          |        +-------------------------+
   +----------+------------------+
              |
              |  + PR#493 (parinzee: LeakyReLU^2)
              |  + PR#461 (Christopher-Lee-McClendon: TTT recipe)
              |  + PR#399 (abaybektursun: Parallel Muon)
              v
   +-----------------------------+
   | LeakyReLU^2 + Legal TTT    |
   | + Parallel Muon (1.1194)   |
   | PR#549 abaybektursun       |
   +----------+------------------+
              |
              |  + PR#478 (gowtham0992: XSA-all)
              |  + PR#609 (saml212: selective pruning)
              |  + PR#535 (raahilshah: Full GPTQ)
              |  + PR#160 (ChaseWNorton: LZMA)
              v
   +----------------------------------------------+
   | AR Self-Gen GPTQ + XSA-all + BigramHash 3072 |
   | (1.1147) PR#1019 abaybektursun                |  <-- MERGED SOTA
   +----------------------------------------------+


   === UNMERGED FRONTIER (clean track, 1.079-1.086) ===

   +--------------------------------------------------------+
   |  The merged SOTA (1.1147) is STALE.                    |
   |  Open PRs have pushed to ~1.079-1.086.                 |
   |  Key additions: depth recurrence, SP8192, SDClip,      |
   |  GPTQ embeddings, linear LR warmdown, Polar Express NS |
   +--+----------+----------+----------+----------+---------+
      |          |          |          |          |
      v          v          v          v          v
  PR#1408    PR#1420     PR#1413    PR#1394    PR#1296
  (1.0800)   (1.08014)  (1.08279)  (1.08563)  (1.0897)
  abaybekt.  abaybekt.  clarkkev   clarkkev   aryanbhosale
  dTTT+      Triple     SP8192     SP8192     SP4096
  BigramHash loop+      +Legal     +GPTQ Emb  +Depth Rec
  (cleanish) n-gram     TTT        +SDClip    +Par Resid
             tilt       (clean)    +MuonEq-R  +MuonEq-R
             (clean)               5-seed     3-seed


   === INDEPENDENT LINEAGE ===

   +------------------------------------------------------+
   | TERNARY / BINARY BRANCH (Ciprian-Florin Ifrim)        |
   |                                                       |
   |  73.7M Ternary U-Net (1.1570) -- 10L 768d 8192BPE   |
   |  BitNet b1.58, NeoMuon, 4xMLP, YaRN, FP8 QAT       |
   |  Base-3 LZMA, stride-16 eval, T=0.90                 |
   |                                                       |
   |  106M Binary (1.1239 non-record, 2hr) -- 15L         |
   |  Asymmetric 1-bit, same arch extended                 |
   +------------------------------------------------------+


   === BANDIT / N-GRAM TRACK (separate category) ===

   PR#1379 (0.4162) causal n-gram mixer -- legitimate, clean
   PR#1083 (0.4961) newjordan -- ClownCar Crawler + 9-gram cache (self-closed)
   PR#876  (0.5863) Bortlesboat -- 10L + order-11 n-gram backoff
   PR#770  (0.6672) minh-stakc -- 11L + multi-order n-gram backoff
```

---

## 3. Technique Taxonomy

### A. QUANTIZATION

| Technique | BPB Impact | Key PRs | Status |
|-----------|-----------|---------|--------|
| int8 + zlib (baseline) | ref | Baseline | Superseded |
| int6 STE QAT | -0.03 to -0.05 total | PR#65+ | **Table stakes.** Zero quant gap. Enables MLP3x and 11L. |
| Mixed int6/int8 (int6 blocks, int8 embed) | -0.015 vs uniform int6 | aquariouseworkman | Embedding 32x more sensitive to quant |
| int5 MLP | -0.003 | thwu1 | 32 levels on MLP only (least sensitive) |
| Gradient-guided per-tensor bit-width (int5/6/7) | unexplored | PR#422 | **Unexplored opportunity** |
| zstd-22 | saves ~1.5MB vs zlib | PR#198+ | Standard for int6 |
| LZMA preset=9 | slightly > zstd in some configs | PR#160 | Used in merged SOTA |
| GPTQ-lite (clip search) | -0.0006 | PR#414 | Superseded by Full Hessian GPTQ |
| Full Hessian GPTQ (H=X^TX, Cholesky) | -0.002 to -0.005 | PR#535, #1019 | **Standard.** 31% error reduction over GPTQ-lite. |
| AR self-gen calibration | legality fix | PR#1019 | Model generates own calibration. Closes 84% val-vs-random gap. |
| SDClip (entropy-based clip widths) | better GPTQ | PR#1394 | **Frontier.** Derives clip from entropy: GPTQ runs once per matrix. |
| GPTQ on embeddings | enables SP4096/8192 | PR#1394 | **Key enabler.** Quantize embedding table to fit larger vocab. |
| Linear LR warmdown (pre-GPTQ) | -61% quant gap | PR#1395 | **Major finding.** 0.038 to 0.014 gap. Single-line schedule change. |
| BitNet b1.58 (ternary) | alternative path | PR#126, #139 | Huge quant gap initially (+0.264); improved to near-zero (0.002) in PR#139. |
| Base-3 + LZMA packing | -39% vs int8+zlib | Ciprian-Florin Ifrim | For ternary weights only |
| FP8 QAT (e4m3) | saves ~2.5MB on fp params | same | Halves floating-point parameter cost, 0.002 BPB penalty |
| EGGROLL (zeroth-order post-GPTQ int6 bin refinement) | marginal | PR#1156 | Novel but minor impact |
| Codebook+Huffman | 21% artifact savings | PR#532 | Closed for TTT compliance issues |
| True Int4 bit-packing (int4 MLP, int6 attn) | enables 14L | PR#1426 | **Results pending.** Could unlock significantly more layers. |
| Online Hessian GPTQ | net negative | PR#1251 | 17ms/step overhead; not worth it |

### B. ARCHITECTURE

| Technique | BPB Impact | Key PRs | Status |
|-----------|-----------|---------|--------|
| 9L to 10L to 11L | -0.005 per layer | progression | 11L standard for SP1024; frontier uses depth recurrence instead |
| MLP 3x expansion | -0.01 to -0.02 | PR#65+ | Standard |
| MLP 4x expansion | further gains | PR#1291, frontier | With larger vocab budgets |
| Depth Recurrence (2-3x layer loops) | **-0.01 to -0.02** | PR#1204, #1331, #1394, #1296 | **THE key frontier technique.** Shared layers looped 2-3x for 18-33 effective depth at same param cost. |
| Parallel Residuals | -0.002 | PR#1204, #1296 | Modified residual connections for depth recurrence compatibility |
| Mini Depth Recurrence | originated here | PR#1204 | Origin PR for both parallel residuals and depth recurrence |
| XSA (Exclusive Self-Attention) | -0.002 (partial), -0.005 (all) | PR#198, #478 | All 11 layers standard (PR#478 proved XSA-all > XSA-4) |
| SmearGate | ~-0.002 | PR#65 | **Conflicts with depth recurrence** (PR#363). Replaced by ConsensusWindowBypass (PR#917). |
| ConsensusWindowBypass | replaces SmearGate | PR#917 | Better compatibility with looped architectures |
| BigramHash | -0.005 to -0.010 | PR#162+ | Scales: 2048 to 4096 to 8192 to 3072x112 |
| U-Net skip connections | ~-0.003 | PR#65+ | Encoder/decoder split with learned skip weights |
| Partial RoPE (16/64) | -0.001 to -0.002 | PR#315 | Zero params. Only rotate 16 of 64 head dims. |
| LN Scale 1/sqrt(l+1) | ~-0.001 | PR#315 | Damps deeper layer contributions. Zero params. |
| LeakyReLU(0.5)^2 | -0.002 to -0.003 | PR#493 | Preserves negative gradient flow. One-line change. |
| VE128 (Shared Value Embedding) | ~-0.001 | PR#374 | 128-dim shared value vectors |
| Cross-repeat skip connections | depth recurrence variant | PR#148 | Early exploration of skip connections across repeated layers |
| Catalytic Residuals | ~11K params | PR#450 | Learned per-dim residual scaling |
| Memory Tokens | -0.014 | PR#352 | 64 learnable vectors as global context. Not adopted. |
| BankLinear (shared weight bank for QKV) | param savings | PR#812 | Shared weight bank approach |
| Error Correction Table | -0.079 BPB | PR#108 | Stores compressed answer key in 2.87MB. Novel but niche. |
| Factored Embedding | enables large vocab | Ciprian-Florin Ifrim | 8192x254 bottleneck |
| Gated DeltaNet (GDN) | **INVALID** | PR#875, #969 | 5 confirmed bugs in PR#875. SSMs structurally disadvantaged (worse compression ratios). |

### C. TRAINING OPTIMIZATION

| Technique | BPB Impact | Key PRs | Status |
|-----------|-----------|---------|--------|
| Muon + AdamW | baseline | All | Muon for matrices, AdamW for scalars/embeddings |
| MuonEq-R (row-normalized Muon) | -0.001 | PR#1217, #1394, #1296 | **Frontier standard.** Cross-window gradient leakage found in PR#1217. |
| Parallel Muon + Parameter Banking | +190 steps | PR#399 | 66 matrices to 4 3D tensors. 15x faster NS via bmm. |
| Polar Express NS (4 minimax-optimal steps) | faster NS | PR#1344 | **Frontier.** Alternative to standard 5-step NS. Clean at 1.0923. |
| Muon momentum 0.99 | -0.002 | PR#65+ | With warmup from 0.92 over 1500 steps |
| Muon WD=0.04 to 0.090 | -0.003 to -0.005 | PR#162+, #1218 | Higher WD in frontier (0.085-0.090). Tighter distributions for better quant. |
| EMA (0.997) | -0.001 vs SWA | PR#287 | Continuous averaging, smoother than periodic SWA |
| OrthoInit | ~-0.001 | PR#65 | orthogonal_() with muP output scaling |
| Warmdown 3000 to 3500 to 4000 | -0.001 per bump | progression | Longer warmdown = less quant damage |
| MATRIX_LR=0.03 | ~-0.006 | PR#530 | Specific LR for matrix parameters |
| 8-bit Muon momentum | 62% memory reduction | PR#703 | **Unexplored opportunity.** |
| MiLe loss (entropy-weighted CE) | novel | PR#703 | Decays to standard CE. **Unexplored.** |
| MUD optimizer (triangular Gram) | within 0.056 BPB of SOTA in 4x fewer steps | PR#510 | Novel preconditioning. arxiv:2603.17970. |
| Progressive depth training (2 to 3 to 4 repeats) | +36% more steps | PR#835 | Start shallow, gradually deepen |
| Seq2048/4096 training | -0.005 to -0.010 | spokane-way, PR#505 | Seq2048 > seq1024 by -0.008 BPB |
| Grad clip 0.3 | stability | PR#162+ | Prevents occasional spikes |
| NorMuon + Int6 STE + SWA | subsumed | PR#156 | Subsumed by merged PR#162 |
| QK-Gain=4.0 | established | PR#1125 | 45-experiment ablation. Some frontier PRs use 5.0. |
| Residual lambdas + 12-seed validation | strongest methodology | PR#1130 | Gold standard for ablation rigor |

### D. EVALUATION & TEST-TIME TECHNIQUES

| Technique | BPB Impact | Key PRs | Status |
|-----------|-----------|---------|--------|
| Sliding window eval (stride=64) | **-0.032** | mattqlf | **Biggest single-technique gain.** Free, no training change. Universal. |
| Sliding window (stride=16) | -0.025 vs chunked | Ciprian-Florin Ifrim | Tighter, ~3x slower |
| Legal Score-First TTT | **-0.015 to -0.020** | PR#461, #549 | Score chunk first, THEN train. 3 epochs SGD. ~410s. |
| Pre-Quant TTT (before GPTQ) | -0.0044 | PR#1399 | **Contested.** Leaning illegal. Adapt EMA model on val BEFORE quantization. |
| Discriminative TTT (dTTT) | novel | PR#1408 | Used in current frontier leader. "Cleanish." |
| ETLB (Eval-Time Logit Bias) | -0.002 | PR#1399 | **Contested.** Warm-started vocab bias across sliding windows. |
| SLOT (sample-specific delta) | -0.0008 to -0.007 | PR#1176, #1291 | **DEAD.** PR#1240 proved 100% causal violation via flip test. |
| Batched LoRA TTT | -0.0316 | PR#557 | 64 docs parallel, 8.3min eval. Single seed. |
| E2E TTT / MAML-style | 1.0467 | PR#873 | Meta-learning approach. Interesting but not competitive. |
| N-gram backoff cache | -0.3 to -0.7 | "Bandit" PRs | Separate category. |
| Causal n-gram mixer | 0.4162 | PR#1379 | Legitimate score. Separate category. |
| Online n-gram agreement + split-LR | 1.1079 | PR#1302 | Clean, novel hybrid approach |
| Temperature scaling T=0.90 | -0.001 to -0.002 | Ciprian-Florin Ifrim | For relu^2-based models |

### E. TOKENIZATION / VOCABULARY

| Technique | BPB Impact | Key PRs | Status |
|-----------|-----------|---------|--------|
| SP-1024 (baseline) | ref | Most early submissions | |
| SP-4096 | **-0.01 to -0.02** | PR#200, #1218, #1296 | Needs GPTQ embeddings or factored embedding |
| SP-8192 | further gains | PR#1394, #1413, Ciprian-Florin Ifrim | **Frontier standard.** Best when combined with GPTQ embedding. |
| GPTQ embedding quantization | enables large vocab | PR#1394 | Quantize embedding table to fit SP8192 in 16MB |
| Factored embedding (bottleneck) | enables large vocab | Ciprian-Florin Ifrim | 8192x254 bottleneck projection |
| Scylla tokenizer | **INVALID** | PR#1143, #1184, #1242, #1271 | 93% of "gap" was byte accounting error. Corrected scores much worse. |

### F. COMPRESSION

| Technique | Savings | Key PRs | Status |
|-----------|---------|---------|--------|
| zlib (baseline) | ref | Baseline | Superseded |
| zstd-22 | ~1.5MB vs zlib | PR#198+ | Standard for int6 |
| LZMA preset=9 | slightly > zstd in some cases | PR#160, #1019 | |
| Brotli | frontier | PR#1296, #1396 | Used with depth recurrence |
| Base-3 LZMA | -39% vs int8+zlib | Ciprian-Florin Ifrim | For ternary weights |
| Selective +/-1 pruning | marginal | PR#609 | Prune int6 weights by reconstruction error |
| Compressor-Aware Training (CAT) | novel | PR#1385 | **Unexplored.** Differentiable proxies for LZ77. |

---

## 4. Clean Frontier Assessment

Ranked by BPB. "Clean" means no SLOT, no contested pre-quant TTT, no n-gram cache.

| Rank | PR | Author | BPB | Seeds | Technique Stack | Compliance | Status |
|------|-----|--------|-----|-------|----------------|------------|--------|
| 1 | #1408 | abaybektursun | **1.0800** | 3 | dTTT + BigramHash + SP8192 stack | Cleanish (dTTT novel) | Open |
| 2 | #1420 | abaybektursun | **1.08014** | 3 | Triple loop + n-gram tilt | Clean | Open |
| 3 | #1413 | clarkkev | **1.08279** | 3 | SP8192 + legal TTT | Clean | Open |
| 4 | #1394 | clarkkev | **1.08563** | 5 | SP8192 + SDClip + GPTQ Emb + depth rec + MuonEq-R | Clean (no TTT, no SLOT) | Open |
| 5 | #1296 | aryanbhosale | **1.0897** | 3 | SP4096 + depth rec + parallel residuals + MuonEq-R | Clean | Open |
| 6 | #1331 | dexhunter | **1.0900** | 3 | 3-layer depth recurrence | Clean | Open |
| 7 | #1285 | dexhunter | **1.0912** | 3 | MuonEq-R + depth rec + all-int6 | Clean | Open |
| 8 | #1344 | Omrigotlieb | **1.0923** | 3 | Polar Express NS | Clean | Open |
| 9 | #1395 | dttdrv | **1.0924** | 3 | Linear LR warmdown + SP4096 + depth rec | Clean | Open |
| 10 | #1260 | dexhunter | **1.0929** | 3 | MuonEq-R + depth rec + mixed int5/int6 | Clean | Open |
| -- | #1019 | abaybektursun | **1.1147** | 3 | AR Self-Gen GPTQ + XSA-all + BigramHash 3072 | Clean | **Merged SOTA** |

### Disputed / Pending Submissions

| PR | Author | BPB | Issue |
|----|--------|-----|-------|
| #1416 | -- | 1.07948 | Withdrawn pending compliance review |
| #1423 | -- | 1.0791 | Flagged for val data leakage |
| #1399 | AnubhavBharadwaaj | 1.0898 | ETLB contested + artifact oversized (16,084,685-16,092,287 bytes vs 16,000,000 cap) |
| #1415 | -- | 1.0913 | SP4096 + 3-layer recurrence + ETLB (ETLB contested) |

### The clean frontier to beat: **1.0800** (PR #1408) or **1.08014** (PR #1420) depending on dTTT ruling.

---

## 5. Contested/Invalid Tracks

### A. SLOT (Dead)

SLOT (Sample-specific Learned Output Token-bias) optimizes a per-batch delta vector in hidden space. It spawned an entire sub-competition before being invalidated.

**PR #1240 proved 100% causal violation rate via flip test.** The technique allows future information to influence predictions through attention in subsequent layers. Every SLOT submission has been self-closed or remains contested.

| PR | Variant | BPB | Status |
|----|---------|-----|--------|
| #1128 | First SLOT introduction | -- | Origin |
| #1176 | Original SLOT | 1.0914 | Contested |
| #1291 | SLOT + SP4096 | 1.0925 | Only -0.0002 from SLOT itself |
| #1318 | L-BFGS SLOT | 1.00955 | Contested |
| #1329 | Per-Sample SLOT | 0.6361 | Heavily contested, likely illegal |
| #1333 | Causal SLOT | 1.0766 | Pending ruling |
| #1350 | L-BFGS SLOT v2 | 1.0046 | Self-closed (minibatch leakage) |
| #1376 | SLOT-24 + pre-quant TTT | 0.7094 | "Both blocked" |
| #1306 | Causal SLOT + pre-quant TTT | -- | Closed |

**Conclusion:** SLOT is a dead end. Even "causal" variants have unresolved information leakage concerns.

### B. Pre-Quant TTT (Contested, Leaning Illegal)

Pre-quantization TTT adapts the full-precision EMA model on validation data BEFORE GPTQ quantization. PR #1222 demonstrated that TTT benefits were massively overstated -- compensating for missing context, not genuine adaptation. PR #1341 confirmed TTT+GPTQ are fundamentally incompatible (post-GPTQ TTT is definitely negative).

Pre-quant TTT remains contested. If ruled illegal, it removes ~0.034 BPB from frontier claims like PR #1399 (1.0898) and PR #1416 (1.07948).

### C. N-Gram Caches (Invalid via Mathematical Proof)

PR #1147 provided a mathematical proof that hashed n-gram caches are invalid: the partition function is ~1000, meaning the cache implicitly conditions on future tokens. Mass disqualifications occurred in PRs #659-#887.

The legitimate bandit track (backward-looking only, no oracle) remains valid as a separate competition:
- PR #1379: Causal n-gram mixer at 0.4162 (legitimate)
- PR #1105: True SLOT-free frontier at 1.0962 (fused Triton MLP + causal n-gram tilt, SP4608)
- PR #1145: Legal causal n-gram augment at 1.1109

### D. TTT Legality Crisis (Historical)

Issue #402 triggered a crisis: PRs #417, #442 were ruled invalid for val data leakage. Subsequently 6+ PRs closed for multi-epoch TTT, 4+ for GPTQ calibration timing violations. The "score-first" protocol (PR #461) resolved the legality issue for standard TTT.

### E. Scylla Tokenizer (Measurement Artifact)

PR #1242 confirmed the Scylla byte accounting bug. PR #1271 showed 93% of the claimed "gap" was byte accounting error. Corrected scores are much worse than claimed.

### F. GDN / PR #875 (Invalid)

5 confirmed bugs: wrong BPB constant, single-shard training/eval, broken DeltaNet state passing, no byte-level accounting. The 1.0226 claim is meaningless. SSMs in general are structurally disadvantaged for this competition (PR #969, #1268).

---

## 6. Key Negative Results

These are confirmed dead ends -- do not pursue:

| Finding | Source | Detail |
|---------|--------|--------|
| SmearGate hurts depth recurrence | PR#363 | SmearGate conflates with looped architectures. Remove it when using depth recurrence. |
| SWA sabotages QAT | PR#989 | -3.64 mBPB. Do not combine SWA with quantization-aware training. |
| TTT+GPTQ fundamentally incompatible | PR#1341, #1019 | Full GPTQ captures what TTT provides. Post-GPTQ TTT is neutral/negative. |
| MuonEq-R + Parallel Muon = 40% regression | PR#1418 | These optimizers conflict. Do not combine. |
| Hadamard rotation marginal | PR#1418 | From 100+ experiment negative results compilation. |
| Online Hessian GPTQ net negative | PR#1251 | 17ms/step overhead not worth the improvement. |
| Scylla tokenizer gains are measurement artifact | PR#1271 | 93% of gap was byte accounting error. |
| N-gram caches invalid (partition function ~1000) | PR#1147 | Mathematical proof of future-token conditioning. |
| JEPA collapses in causal LMs | PR#1330 | Same-sequence prediction causes representational collapse. |
| TTT benefits massively overstated | PR#1222 | Compensating for missing context, not genuine adaptation. |
| TrigramHash hurts compression | PR#609 | Improves raw BPB but worse net (larger artifact). |
| Extended context eval catastrophic | PR#609 | RoPE breaks past training length. |
| VRL conflicts with VE128 | PR#609 | Negative interaction. |
| SwiGLU 45% slower per step | PR#609 | Net negative in 10 min budget. |
| SLOT 100% causal violation | PR#1240 | Proven via flip test. |
| PR#1396 regression vs sources | reviewer | Combined PR worse than either source PR (#1344: 1.0923, #1392: 1.1020, combined: 1.1067). |
| EMA incompatible with BitNet ternary | PR#760 | +0.52 BPB, training 2.5x slower. |
| Universal Transformer + ACT underperforms | PR#1293 | Compute overhead eats training budget (7,392 vs 13,780 baseline steps). |
| "Paid prefix" (storing val tokens in artifact) | PR#168, #275 | Ruled out-of-scope. "Guys, come on" -- 0hq. |

---

## 7. Unexplored Opportunities

High-potential directions with no or minimal exploration:

| Opportunity | Evidence | Risk | Potential |
|-------------|----------|------|-----------|
| **Per-group 4-bit quant (NF4, MXFP4, non-uniform grids)** | Nobody tried. Major blind spot. | Medium | High -- could unlock 14L+ models |
| **True Int4 bit-packing (int4 MLP, int6 attn)** | PR#1426 in progress | Medium | High -- enables 14L configurations |
| **CROWN-Q (quant-variance penalty during warmdown)** | PR#692, valid technique, closed for doc issues | Low | Medium -- train weights to be quant-friendly |
| **MiLe loss (entropy-weighted CE)** | PR#703, never adopted | Low | Medium -- easy to integrate |
| **8-bit Muon momentum** | PR#703, 62% memory reduction | Low | Medium -- frees memory for larger models |
| **Compressor-Aware Training (CAT)** | PR#1385, differentiable LZ77 proxies | Medium | Medium -- optimize for artifact size directly |
| **Error Correction Tables** | PR#108, -0.079 BPB at 2.87MB | Low | Medium -- orthogonal to model quality |
| **Low-rank Q factorization (rank=192)** | PR#215, 22% speedup | Low | Low-Medium -- more training steps |
| **Gradient-guided per-tensor bit-width (int5/6/7)** | PR#422 | Medium | Medium -- adaptive precision allocation |
| **Weight decay reduces quant gap** | PR#98 (draft, never merged) | Low | Low -- systematic evidence exists |
| **Step-1000 BPB correlates 0.86 with final** | PR#1162 | n/a | Meta: enables rapid hyperparameter search |
| **Extended 50K step scaling** | PR#1424 | n/a | Meta: understanding scaling behavior |

---

## 8. Recommended Competition Stack

Based on the full analysis, the recommended stack for a new competitor entering April 6:

### Tier 1: Non-Negotiable Foundation
```
Architecture:    11L, 512d, 8H/4KV, MLP 3-4x with LeakyReLU(0.5)^2
Depth:           3-layer depth recurrence (shared layers looped 3x)
Residuals:       Parallel residuals (from layer 7)
Attention:       XSA on all 11 layers + Partial RoPE 16/64 + LN Scale 1/sqrt(l+1)
Vocabulary:      SP8192 with GPTQ-quantized embeddings
Quantization:    int6 STE QAT + Full Hessian GPTQ (AR self-gen calibration)
                 + SDClip (entropy-based clip widths)
Clipping:        Linear LR warmdown before GPTQ (-61% quant gap)
Optimizer:       MuonEq-R + AdamW (NOT Parallel Muon -- they conflict)
                 + Polar Express NS (4 minimax-optimal steps)
Training:        WD=0.085-0.090, momentum 0.99, grad clip 0.3
                 MATRIX_LR=0.03, QK-Gain=4.0
EMA:             0.997 (not SWA -- SWA sabotages QAT)
Eval:            Sliding window stride=64
Compression:     LZMA or Brotli
```

### Tier 2: Likely Gains (test these)
```
Legal Score-First TTT              (-0.015 to -0.020 BPB, +410s eval)
BigramHash 3072x112                (-0.005 to -0.010)
dTTT (discriminative TTT)          (used by frontier leader at 1.0800)
N-gram tilt (causal, legal)        (used by PR#1420 at 1.08014)
ETLB (if ruled legal)              (-0.002)
```

### Tier 3: High-Upside Experiments
```
True Int4 bit-packing for MLP      (enables 14L -- PR#1426 pending)
CROWN-Q (quant-variance penalty)   (train for quant-friendliness)
Compressor-Aware Training (CAT)    (optimize for artifact size)
MiLe loss                          (entropy-weighted CE)
8-bit Muon momentum                (62% memory savings)
Per-group 4-bit quant (NF4)        (nobody has tried this)
```

### What NOT to do
```
- Do NOT use SLOT (proven causal violation)
- Do NOT use SmearGate with depth recurrence (conflicts)
- Do NOT combine SWA with QAT (sabotage)
- Do NOT use post-GPTQ TTT (incompatible)
- Do NOT combine MuonEq-R with Parallel Muon (40% regression)
- Do NOT use n-gram caches with hashed lookups (invalid)
- Do NOT use Scylla tokenizer (measurement artifact)
- Do NOT trust GDN/PR#875 numbers (5 confirmed bugs)
```

---

## 9. Timeline of Key Events

```
Day 1  (Mar 18):  Baseline 1.2244. FP16 embed, lower LR, longer seq.
Day 2  (Mar 19):  MASSIVE convergent innovation. Sliding window (-0.032 FREE),
                  int6 QAT, SmearGate, BigramHash, 10L, MLP3x, Muon WD, zstd-22.
                  PR#44: TTT single-pass rule established (multi-pass disqualified).
Day 3  (Mar 20):  11L, XSA (partial), EMA, int5 MLP, BigramHash scaling.
Day 4  (Mar 21):  Partial RoPE, LN Scale (zero-param wins).
Day 5  (Mar 22):  GPTQ-lite (PR#379 origin), warmdown tuning.
Day 6  (Mar 23):  LeakyReLU^2 (PR#493), Legal TTT (PR#461), Parallel Muon (PR#399).
Day 7  (Mar 24):  Ternary/Binary independent branch.
Day 8  (Mar 25):  AR Self-Gen GPTQ, XSA-all -> 1.1147 merged SOTA (PR#1019).
                  VRL (PR#569) and MUD optimizer (PR#510) introduced.
Day 9+:           TTT legality crisis (issue #402). 6+ PRs closed for multi-epoch TTT.
                  N-gram cache crisis: mass disqualifications (PRs #659-#887).
                  PR#1147: Mathematical proof n-gram caches invalid.
                  SLOT introduced (PR#1128) and explodes in popularity.
                  PR#1240: SLOT proven 100% causal violation via flip test.
                  Depth recurrence emerges (PR#1204) as dominant architecture technique.
                  SP4096/8192 + GPTQ embeddings unlock larger vocabularies.
                  PR#1394: clarkkev reaches 1.08563 (5-seed, fully clean).
                  PR#1408: abaybektursun reaches 1.0800 with dTTT + BigramHash.
                  PR#1420: Triple loop + n-gram tilt at 1.08014 (clean).
                  PR#1426: True Int4 bit-packing proposed (results pending).
                  Scylla tokenizer invalidated (93% measurement error, PR#1271).
                  Pre-quant TTT contested, leaning illegal.
                  Competition deadline: April 30, 2026.
```

### The Meta-Game Pattern

```
1. Someone finds a technique that saves space or compute
2. That space is immediately spent on more parameters/layers/width
3. The community converges on a "standard stack" within 24-48 hours
4. The next breakthrough comes from a NEW dimension
5. Repeat

The cycle: quant budget (int6) -> architecture (11L, MLP3x, XSA) ->
           eval tricks (sliding window, TTT) -> training (EMA, warmdown, GPTQ) ->
           vocab (SP4096/8192) -> recurrence (depth loops) ->
           optimizer (MuonEq-R, Polar Express) -> ??? (Int4? CAT? novel arch?)
```

---

*Updated 2026-04-06. ~1426 PRs cataloged across 9 review agents. The competition runs until April 30.*
