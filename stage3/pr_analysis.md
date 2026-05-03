# PR Analysis Summary — Parameter Golf Competition

**Date**: 2026-04-06
**Our current score**: 1.1631 BPB (rank ~20)
**Merged SOTA**: 1.1147 (PR #1019)
**Open no-TTT frontier**: 1.0856 (PR #1394)
**Open TTT frontier**: 1.0795 (PR #1416)
**Open SLOT frontier**: 0.7094 (PR #1376)

---

## Merged Records (chronological)

| PR | Score | Date | Author | Stack |
|----|-------|------|--------|-------|
| #1019 | 1.1147 | 03-30 | abaybektursun | AR Self-Gen GPTQ, XSA-all, BigramHash 3072×112, Parallel Muon, LeakyReLU², EMA |
| #549 | 1.1194 | 03-24 | (LeakyReLU² + Legal Score-First TTT + Parallel Muon) |
| #414 | 1.1233 | 03-23 | signalrush | 11L, GPTQ-lite, EMA, XSA4, Partial RoPE, LN Scale, VE128, Late QAT@0.15 |
| #315 | 1.1248 | 03-23 | jfprincz | 11L, Partial RoPE 16/64, LN Scale, EMA, XSA4 |
| #287 | 1.1271 | 03-23 | jfprincz | 11L, XSA4, EMA, Int6 MLP3x, WD=0.04 |
| #265 | 1.1307 | 03-23 | unnir | 11L, Efficient Partial XSA (3 layers), FA3, SWA120 |
| #363 | N/A | 03-25 | evangelinehelsinki | Non-record: Depth Recurrence research — what works, what doesn't |
| #180 | 1.1428 | 03-20 | thwu1 | 10L, Mixed int5/int6, BigramHash(10240), SWA(0.4), WD=0.04 |
| #162 | 1.1458 | 03-20 | raahilshah | Int6 MLP3x, SmearGate, BigramHash, OrthoInit, MuonWD, SWA |

---

## Open PRs — Tier A (sub-1.09, real contenders)

### PR #1420 — Triple Loop + Fused Kernels + Parallel Residuals (1.0801, 5-seed)
**Author**: abaybektursun | **TTT**: Yes (Pre-Quant) | **SLOT**: No
**What's new**:
- Triple loop (NUM_LOOPS=3) on layers 4-5 → 17 virtual layers from 11 physical
- Activate looping at 0.35 instead of 0.50 (earlier helps)
- Fused MLP kernels: Triton TMA forward + CUTLASS EVT backward → +10% throughput, +127 steps
- Parallel residuals (GPT-J style) on layers 7-10 → +68 steps from faster forward
- N-gram Tilt (details in appendix)
**Takeaway**: Engineering depth. Fused kernels are high-effort but high-reward. Triple loop > double loop > no loop.

### PR #1416 — SP8192 + Pre-Quant TTT (1.0795, 3-seed)
**Author**: erichroepke | **TTT**: Yes (Pre-Quant) | **SLOT**: No
**What's new**:
- Simple combo: PR #1394 base + PR #1364 pre-quant TTT
- TTT: 6 epochs AdamW on EMA model before GPTQ, freeze first 2 blocks
- TTT gives -0.034 BPB (post-EMA 1.1019 → post-TTT 1.0682)
- Built by a filmmaker with Claude Opus 4.6
**Takeaway**: Pre-quant TTT is orthogonal to base architecture. Just bolt on top.

### PR #1423 — SP8192 + Pre-Quant TTT + QK-Gain 5.0 (1.0791, 3-seed)
**Author**: aryanbhosale | **TTT**: Yes (Pre-Quant) | **SLOT**: No
**What's new**:
- Same as #1416 but QK-Gain 5.0 (up from 4.0)
- One hyperparameter change: -0.0004 BPB
**Takeaway**: QK-Gain 5.0 is free and helps.

### PR #1394 — SP8192 + GPTQ Embeddings + Depth Recurrence + MuonEq-R + SDClip (1.0856, 5-seed)
**Author**: clarkkev | **TTT**: No | **SLOT**: No
**What's new**:
- SP8192 vocab (up from SP4096)
- GPTQ on embedding matrix (int8) — saves space
- Depth recurrence: loop layers 4-5 twice
- SDClip: std-dev based clip threshold instead of percentile search
- Removed value embeddings (not needed with SP8192)
- ShuffledSequenceLoader replaces coprime-stride loader
- Row-normalized Muon (MuonEq-R)
- Brotli compression
**Takeaway**: The strongest no-TTT base. This is the template to adopt.

### PR #1408 — dTTT + BigramHash 3072×112 (1.0800, 3-seed)
**Author**: aamodbhatt | **TTT**: Yes (dTTT) | **SLOT**: No
**What's new**:
- Discriminative TTT: per-block adaptive LR (0.3×-1.0×)
- BigramHash 3072×112 (up from 2048×128)
- 10 epochs dTTT, AdamW LR=0.0005, GPTQ damp=0.005
- QK-Gain 5.0, WARMDOWN=4000, XSA all-layers
**Takeaway**: dTTT (discriminative TTT) with per-block LR groups is the refined TTT approach.

### PR #1415 — SP4096 + 3-Layer Recurrence + GPTQ Embeddings + SDClip + ETLB (1.0913, 3-seed)
**Author**: bigbag | **TTT**: No | **SLOT**: No
**What's new**:
- SP4096 (not SP8192)
- 3-layer depth recurrence (layers 3,4,5)
- ETLB: Eval-Time Logit Bias — optimizes vocab bias during eval
- GPTQ on embeddings
- SDClip
- QK-Gain 5.0
- WD=0.095, MLR=0.022
- LZMA code wrapper (18KB code saves ~40KB artifact)
**Takeaway**: ETLB is a cheap eval-time trick worth ~-0.001 BPB. Higher WD=0.095 works with MuonEq-R.

### PR #1399 — Pre-Quant TTT + ETLB (1.0898, 3-seed)
**Author**: AnubhavBharadwaaj | **TTT**: Yes (Pre-Quant) | **SLOT**: No
**What's new**:
- ETLB: bias vector b ∈ R^vocab optimized during sliding window eval
- Pre-quant TTT: freeze first 9 of 11 blocks, adapt last 2
- ETLB gives ~-0.002 BPB on top of sliding window
**Takeaway**: ETLB is a free add-on for eval. Stacks with everything.

### PR #1392 — SP4096 + Depth Recurrence + Parallel Residuals + Brotli (1.1020, 3-seed)
**Author**: Its-Just-Crump | **TTT**: No | **SLOT**: No
**What's new**:
- SP4096 native architecture
- MLP 4x (wider with SP4096)
- Depth recurrence + parallel residuals
- QK-Gain (higher than default)
- Brotli compression
- Replaces BigramHash with "SP4096 gets bigram info natively from larger vocab"
**Takeaway**: SP4096 + no BigramHash shows vocab size can substitute for explicit bigram features.

---

## Open PRs — Tier B (1.09-1.12, interesting techniques)

### PR #1410 — LatentMask TTT + Product-Key Bigram + Brotli (1.1158, 3-seed)
**Author**: izlley | **TTT**: Yes (LatentMask) | **SLOT**: No
**What's new**:
- LatentMask TTT: per-channel sigmoid masks + biases on MLP/attention outputs, trained per-chunk with sign-based Muon-lite optimizer
- Product-Key Bigram: factored `embed_prev(1024,512) * embed_cur(1024,512)`, zero hash collisions
- Alternating GatedAttention (every other layer)
- Brotli-11 + uint8 log-scale serialization
**Takeaway**: Product-Key Bigram is cleaner than hash-based BigramHash. LatentMask TTT is lightweight (~-0.002 BPB).

### PR #1384 — Progressive Depth + Hedge Mixer (1.1441, 3-seed)
**Author**: iverbovoy | **TTT**: No | **SLOT**: No
- Progressive depth training with hedge-based layer mixing

### PR #1421 — Depth Recurrence + EMA Tuning 0.9965 (1.0925)
**Author**: X-Abhishek-X | **TTT**: Implied | **SLOT**: No
- Higher EMA decay (0.9965 vs 0.997) — longer averaging window

### PR #1422 — Depth Recurrence + GPTQ + SGD TTT on 1xH100 (1.1172)
**Author**: swapp1990 | **TTT**: Yes (SGD) | **SLOT**: No
- Shows depth recurrence + GPTQ works on single H100 too

---

## Open PRs — Tier S (SLOT / exotic, sub-0.80)

### PR #1376 — SLOT-24 + Pre-quant TTT (0.7094, 3-seed)
**Author**: stukenov | **TTT**: Yes | **SLOT**: Yes
**What's new**:
- SLOT (Sparse Latent Optimization at Test-time, arXiv:2505.12392v2)
- Per-sample delta [bsz,1,512] + logit_bias [bsz,1,1024]
- 24 AdamW steps per sample (cosine LR 0.024→0.001, stride=96)
- Model weights frozen during SLOT — only delta+bias optimized
- Combined with pre-quant TTT (6 epochs on EMA before GPTQ)
**Takeaway**: SLOT is a game-changer (-0.37 BPB!). But fundamentally different paradigm. Competition may split.

---

## Merged Research — Key Papers

### PR #363 — Depth Recurrence: What Works, What Doesn't (merged non-record)
**Author**: evangelinehelsinki | **35 runs across 8xH100**
**Key findings**:
1. "Flat 11L 512d: 1.1648 vs Looped 3x3 640d: 1.1894" — looped was WORSE initially
2. **Noisy QAT** collapses recurrence quantization gap from 0.37 to 0.002 BPB
3. 3x3 > 2x5 loops (more unique blocks with fewer repeats)
4. Quantization error compounds through N repeats superlinearly
5. **12 negative results** documented: XSA all layers (+0.001 worse), cyclic momentum (catastrophic), QuadgramHash (unclear), factored embeddings (worse), Value Residual (+0.14 catastrophic), progressive unrolling (DDP crash)

**Why later PRs succeeded**: Better quantization (GPTQ+SDClip instead of simple int6) neutralizes the quantization compounding problem.

---

## Technique Evolution Timeline

```
Era 1 (Mar 18-19): Baseline + env vars
  fp16 embed, sliding window, 10L, MuonWD → 1.17-1.16

Era 2 (Mar 19-20): Module additions
  SmearGate, BigramHash, OrthoInit, SWA, int5/int6 → 1.15-1.14

Era 3 (Mar 20-22): Architecture + quantization
  XSA, EMA, Partial RoPE, LN Scale, VE128, FA3, GPTQ-lite → 1.13-1.12

Era 4 (Mar 22-24): Full GPTQ + LeakyReLU²
  Full Hessian GPTQ, LeakyReLU², Parallel Muon, VRL, TTT → 1.12-1.11

Era 5 (Mar 25-30): Depth recurrence + bigger vocab
  Depth recurrence, MuonEq-R, AR Self-Gen GPTQ, XSA-all → 1.11

Era 6 (Mar 30 - Apr 6): SP8192 + compression-aware quant
  SP4096/SP8192, SDClip, GPTQ embeddings, Brotli → 1.09-1.08
  Pre-Quant TTT, dTTT, ETLB, Parallel Residuals → 1.08
  SLOT → 0.71
```

We are in Era 2-3. The frontier is in Era 6.

---

## What We Need (ordered by impact)

### Must-Have (each is >0.01 BPB)
1. **SP4096 or SP8192 tokenizer** — SP1024 is ~0.02-0.03 BPB behind
2. **Full Hessian GPTQ** — ~0.02 BPB from quantization improvement
3. **Depth Recurrence** — ~0.015 BPB from virtual depth
4. **11 Layers** — ~0.005-0.01 BPB

### High Priority (each is 0.003-0.005 BPB)
5. **MuonEq-R** (row-normalized Muon)
6. **XSA on all 11 layers**
7. **LeakyReLU(0.5)²**
8. **EMA(0.997)**
9. **SDClip** (replaces percentile clip)
10. **GPTQ on embeddings**
11. **Brotli compression** (replaces zstd/lzma)

### Medium Priority (each is 0.001-0.003 BPB)
12. **QK-Gain 5.0**
13. **BigramHash 3072×112** (or Product-Key Bigram)
14. **Parallel Residuals** (GPT-J style, deep layers)
15. **MuonWD=0.04** (or higher: 0.095 in PR #1415)
16. **Warmdown 3500-4000**
17. **ETLB** (eval-time logit bias)
18. **Partial RoPE 16/64**
19. **LN Scale**

### TTT Lane (if pursuing)
20. **Pre-Quant TTT** (6 epochs, freeze first 2 blocks) — ~-0.034 BPB
21. **dTTT** (discriminative, per-block adaptive LR)
22. **SLOT** (24 steps per sample) — ~-0.37 BPB

### Estimated Stack Composite
```
Our current:       1.1631
+ SP4096/8192:     ~1.14   (tokenizer alone gives ~0.02)
+ 11L:             ~1.13
+ MuonWD + EMA:    ~1.12
+ GPTQ + SDClip:   ~1.10   (quantization is massive)
+ Depth Recurrence: ~1.085
+ XSA-all + LeakyReLU² + MuonEq-R: ~1.080
+ Pre-Quant TTT:   ~1.050
```

---

## Lineage of Key PRs

```
PR #162 (SmearGate/BigramHash/MuonWD/SWA, 1.1458)
  └→ PR #265 (XSA, FA3, 1.1307)
     └→ PR #287 (EMA, 1.1271)
        └→ PR #315 (Partial RoPE, LN Scale, Late QAT, 1.1248)
           └→ PR #414 (GPTQ-lite, VE128, warmdown3500, 1.1233)
              └→ PR #549 (Parallel Muon, Score-First TTT, 1.1194)
                 └→ PR #1019 (AR Self-Gen GPTQ, XSA-all, BH3072, 1.1147) ← CURRENT MERGED SOTA

PR #593 (Full GPTQ, Parameter Banking, LeakyReLU², 1.1171) [independent line]

PR #1204 (Depth Recurrence, omrigotlieb) [new technique]
  └→ PR #1217 (MuonEq-R, QK-Gain, bigbag)
     └→ PR #1394 (SP8192, SDClip, GPTQ embed, clarkkev, 1.0856)
        └→ PR #1416 (+ Pre-Quant TTT, erichroepke, 1.0795)
           └→ PR #1420 (Triple Loop + Fused Kernels, abaybektursun, 1.0801)
              └→ PR #1423 (+ QK-Gain 5.0, aryanbhosale, 1.0791)

PR #1364 (Pre-Quant TTT, stukenov) [new technique]
  └→ PR #1376 (+ SLOT-24, stukenov, 0.7094)
```

---

## PR #1019 Code-Level Diff vs Our Baseline

Full diff of `train_gpt.py`: 2135 lines (PR #1019) vs 1372 lines (baseline). Every subsystem was rewritten.

### Architecture Changes

| Component | Baseline | PR #1019 |
|-----------|----------|----------|
| Layers | 9 | **11** |
| MLP activation | relu² with 2× width | **LeakyReLU(0.5)²** with **3× width** |
| Seq len | 1024 | **2048** |
| Batch tokens | 524K | **786K** |
| Attention backend | PyTorch SDPA | **Flash Attention 3** (Hopper kernels) |
| RoPE | Full (all head dims) | **Partial RoPE** (16 of 64 dims) |
| LN Scale | None | **1/√(layer+1)** per-layer scaling on norm outputs |
| XSA | None | **All 11 layers** — subtracts self-value projection: `y_out = y - (y·v̂)×v̂` |
| SmearGate | None | **Yes** — sigmoid gate mixing current + previous position |
| BigramHash | None | **3072×112** — hash-based bigram embeddings (+ optional trigram) |
| Value Embedding | None | **VE128** on layers 9,10 — shared embedding table + per-layer scale |
| U-Net skips | Yes | Yes (unchanged) |

### Optimizer Changes

| Component | Baseline | PR #1019 |
|-----------|----------|----------|
| Muon | Standard all-reduce flat | **Parallel Muon** — async reduce-scatter → local NS5 on shard → async all-gather. No DDP wrapper. |
| Weight decay | 0.0 | **MuonWD=0.04, AdamWD=0.04** |
| NS5 steps | 10 | **5** (but batched 3D via bank tensors) |
| Momentum | 0.95, warmup 500 steps | **0.99**, warmup from **0.92** over **1500** steps |
| Learning rates | matrix=0.04, scalar=0.04 | matrix=**0.025**, scalar=**0.025** |
| Grad clip | 0.0 (off) | **0.3** |
| Warmdown | 1200 iters | **3500** iters |
| EMA | None | **EMA(0.997)** — shadow state_dict updated every step, applied before export |
| SWA | None | **SWA every 50 steps** during warmdown (scale < 0.2) |
| Late QAT | None | **STE quantization-aware training** activated when LR scale < 0.15 |
| Parameter banking | None | **3D bank tensors** — `qo_bank[2*N, dim, dim]`, `kv_bank[2*N, kv_dim, dim]`, `mlp_up_bank[N, mlp_dim, dim]`, `mlp_down_bank[N, dim, mlp_dim]`. All block matrix weights stacked into contiguous 3D tensors for batched Muon NS5. |

### Quantization & Export Changes

| Component | Baseline | PR #1019 |
|-----------|----------|----------|
| Quant format | int8 per-row (percentile clip) | **int6** per-row for attn+MLP matrices, int8 for embed/small tensors |
| Quant method | Simple percentile clip | **Full Hessian GPTQ** — Cholesky inverse of H=X^TX, block-wise error propagation (block_size=128), actorder column reordering |
| Calibration data | N/A | **AR self-gen** — model generates 64 sequences × 2048 tokens at temp=0.8, no external data |
| Hessian collection | N/A | Separate `_HessianGPT` model (non-banked copy with CastedLinear layers for forward hooks) |
| Compression | zlib level=9 | **LZMA preset=9** |
| Size fitting | None | **Selective ±1 pruning** — sort all ±1 quantized values by reconstruction error (scale²), binary search prune count to fit TARGET_MB |
| Unbanking | N/A | 3D bank tensors split into individual 2D `blocks.{i}.attn.c_q.weight` etc. before quantization, re-banked after dequantization |

### Eval Changes

| Component | Baseline | PR #1019 |
|-----------|----------|----------|
| Eval method | Standard full-sequence | **Sliding window** — stride=64, each token scored with max context |
| Eval implementation | model(x, y) | **compiled `forward_logits`** — separate method returning logits without loss, torch.compiled |
| Eval seq len | =train_seq_len (1024) | **2048** (separate `EVAL_SEQ_LEN` parameter) |
| Post-EMA diagnostic | None | **Yes** — runs eval on EMA weights before quantization to measure pre-quant baseline |

### Key Implementation Details

**Parallel Muon pipeline** (3-phase overlapped):
1. After backward: launch async reduce-scatter for all banks (biggest first)
2. While RS in-flight: all-reduce non-bank grads + step Adam on small params
3. Wait for each RS, run local NS5 on shard, launch async all-gather (overlaps with next bank's NS5)

**XSA implementation** (GQA-aware, no repeat_interleave):
```python
# y: [B,T,H,D], v: [B,T,Hkv,D]
y_g = y.reshape(B, T, Hkv, group, D)
vn = F.normalize(v, dim=-1).unsqueeze(-2)
proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
return (y_g - proj).reshape(B, T, H, D)
```

**GPTQ column reordering** (actorder):
```python
perm = torch.argsort(torch.diag(H), descending=True)  # most-sensitive columns first
W = t32[:, perm].clone()
H = H[perm][:, perm]
Hinv = torch.linalg.cholesky(H)
Hinv = torch.cholesky_inverse(Hinv)
Hinv = torch.linalg.cholesky(Hinv, upper=True)
# block-wise error propagation with Cholesky inverse
```

**Quantization gap**: Pre-quant 1.1354 → Post-GPTQ 1.1377 (+0.0023) → Sliding window **1.1147** (sliding window recovers -0.023 BPB)

---

## UPDATE 2026-04-09 — New Frontier PRs

### Leaderboard Snapshot

| Tier | PR | BPB | Author | Key Stack | TTT | SLOT |
|------|----|-----|--------|-----------|-----|------|
| **S** | #1507 | **0.2282** | ChideraIbe123 | L-BFGS SLOT + Entropy-Adaptive N-gram Mixer (order-12) | No | Yes |
| **S** | #1488 | **0.8265** | ndokutovich | SP1024 + SLOT-24 + QK5.25 + Pre-Quant TTT 10ep | Yes | Yes |
| **A** | #1487 | **1.0600** | ndokutovich | SP8192 + Recur345 + Par7 + EMA + QK5.25 + Pre-Quant TTT 10ep | Yes | No |
| **A** | #1485 | **1.0679** | ndokutovich | SP8192 + Recur345 + Par7 + EMA + QK5.0 + Pre-Quant TTT 6ep | Yes | No |
| **A** | #1489 | **1.0736** | joshkmartinez | SP1024 + MLP4x + Looping 4-5 + Par7 + Pre-Quant TTT + ETLB | Yes | No |
| **A** | #1482 | **1.0787** | aamodbhatt | SP8192 + Recur + QK5.25 + Pre-Quant TTT 8ep freeze-1 | Yes | No |
| **A** | #1493 | **1.0810** | bigbag | SP8192 + Recur345 + Par7 + QK5.25 + Legal Score-First TTT | Yes | No |
| **A** | #1477 | **1.0822** | aryanbhosale | SP8192 + Par7 + Score-First TTT 3ep | Yes | No |
| **A** | #1460 | **1.0827** | resouer | SP8192 + Score-First TTT + Eval-Time Hash Embedding | Yes | No |
| **A** | #1450 | **1.0848** | andrewbaggio1 | TMA Megakernel + Triple Loop + Par7 (no TTT) | No | No |
| **A** | #1471 | **1.0866** | X-Abhishek-X | SP8192 + SDClip + Recur345 + EMA 0.9965 (no TTT) | No | No |
| **B** | #1458 | **1.1060** | newjordan | 12L + Mixed int5/6 + Brotli (no TTT, pure neural) | No | No |
| **B** | #1508 | **1.1135** | jpfeiffe | SP4096 + Compressibility Regularization + Brotli | No | No |
| Merged | #1019 | 1.1147 | abaybektursun | AR Self-Gen GPTQ + XSA-all + BH3072 | No | No |

### New Techniques Since Last Update

#### N1: L-BFGS SLOT (PR #1507, 0.2282 BPB)
Replaces AdamW with L-BFGS for the SLOT optimization (per-window delta + logit_bias). L-BFGS uses curvature information via gradient history — 6 outer steps with strong Wolfe line search + history_size=10. Second-order methods converge in O(n) steps for the 1,536-parameter SLOT problem. Combined with **entropy-adaptive order-12 n-gram mixer** (4M hash buckets, backoff). Mixing alpha depends on neural model entropy, match order, and context count — all target-independent. Scaling: AdamW SLOT 0.8637 → L-BFGS SLOT 0.5793 → + n-gram 0.2968 → + order/count features **0.2282**.

#### N2: Pre-Quant TTT Tuning (PR #1487, 1.0600 — current overall no-SLOT frontier)
Same code as PR #1485 but tuned env vars: QK_GAIN_INIT 5.0→**5.25**, TTT_EPOCHS 6→**10**, TTT_FREEZE_BLOCKS 2→**1** (freeze fewer = adapt more), TTT_LR 0.0005→**0.00045**. Delta: -0.0079 BPB from hyperparameter tuning alone. Shows the TTT response surface is steep — small changes in epochs/LR/freeze matter.

#### N3: SP1024 + Pre-Quant TTT + MLP 4x (PR #1489, 1.0736)
Surprises: achieves 1.0736 with SP1024 (not SP8192). Key insight: smaller vocab frees ~4M params for wider transformer (MLP 4x instead of 3x). Pre-quant TTT is the workhorse (-0.034 BPB). Uses ETLB at eval time. Looping on layers 4-5 only (2 loops, not 3). Shows SP1024 is not dead — it's a different tradeoff.

#### N4: SLOT + Pre-Quant TTT combo (PR #1488, 0.8265)
First combination of pre-quant TTT (weight-level adaptation, baked into artifact) with SLOT (hidden-state optimization, eval-time). They're complementary: TTT improves the base sliding score from ~1.12 to 1.088, then SLOT pushes from that better base to 0.8265. Delta vs prior SLOT SOTA: -0.037 BPB.

#### N5: Eval-Time Hash Embedding (PR #1460, 1.0827)
Novel: zero-initialized `nn.Embedding(16384, 512)` created at eval time, trained through the score-first TTT loop. Bigram hash `h = (prev * 2039 + curr) % 16384` looks up residual vector added to tok_emb(x) before RMSNorm. The hash embedding learns document-local bigram patterns. Measured delta: -0.0004 BPB (small but legal and free).

#### N6: TMA Megakernel (PR #1450, 1.0848)
Triton TMA fused MLP forward: fuses `fc → leaky_relu(0.5) → square` into a single Hopper TMA kernel. Uses TensorDescriptor for async global-to-shared transfers, persistent scheduling (one program per SM), 128×256×64 block tiling. Avoids materializing ~384MB intermediate per forward pass. +10.5% throughput → +127 extra steps in 600s. Triple loop (NUM_LOOPS=3, 17 virtual layers). No TTT — pure training quality.

#### N7: 12 Layers via Mixed Quantization (PR #1458, 1.1060)
Fits 12 layers (up from 11) by using mixed-int quantization (attn=int5, mlp=int6, aux=int6, embed=int8) + Brotli. Pure neural submission: no TTT, no SLOT, no eval tricks. Shows that squeezing in an extra layer is worth ~0.01 BPB.

#### N8: Compressibility Regularization (PR #1508, 1.1135)
SP4096 + ramped weight decay during warmdown (WARMDOWN_WD_MULT=2.0). The idea: increasing WD in warmdown pushes weights toward zero, improving compressibility. Combined with brotli-11 (selected when smaller than lzma). Result: zero pruning needed across all 6 seeds. Small but statistically significant delta vs merged SOTA: -0.00125 BPB.

#### N9: 3-Layer Recurrence is Now Standard
Layers 3,4,5 looped → 14 virtual layers from 11 physical. Present in PRs #1471, #1477, #1482, #1485, #1487, #1493. Activation at step 2000 (or frac 0.35). This is no longer novel — it's table stakes for the frontier.

### Revised Priority Stack (as of 2026-04-09)

The no-SLOT, no-TTT frontier is **1.0848** (PR #1450, TMA Megakernel). The no-SLOT, with-TTT frontier is **1.0600** (PR #1487). The SLOT frontier is **0.2282** (PR #1507).

**Tier 0 — Table stakes (all frontier PRs have these)**:
- SP8192 tokenizer
- 11 layers, 512d, 8/4 GQA, MLP 3x (or 4x with SP1024)
- 3-layer depth recurrence (L3-5)
- Parallel residuals from layer 7
- XSA all layers
- LeakyReLU(0.5)²
- QK-Gain 5.0-5.25
- EMA 0.9965
- Full Hessian GPTQ int6 + SDClip + brotli
- Sliding window eval (stride=64)
- Skip gates, SmearGate, partial RoPE 16/64, LN Scale

**Tier 1 — Pre-Quant TTT (worth -0.02 to -0.034 BPB)**:
- AdamW on val data, 6-10 epochs, lr=0.00045, freeze 1-2 blocks, cosine decay
- Applied after EMA, before GPTQ
- Baked into artifact — deterministic at eval time
- Tuning matters: epochs 6→10 and freeze 2→1 gave -0.008 BPB

**Tier 2 — Engineering throughput**:
- TMA Megakernel fused MLP (+10.5% throughput, +127 steps)
- Triple loop (NUM_LOOPS=3, 17 virtual from 11)
- These give ~-0.005 BPB from more steps at same quality

**Tier 3 — Eval-time adaptation (if rules allow)**:
- Score-First TTT (SGD, 3-5 epochs per chunk): -0.002 to -0.005 BPB
- ETLB (eval-time logit bias): -0.001 BPB
- Eval-Time Hash Embedding: -0.0004 BPB
- SLOT (L-BFGS, 6 steps): -0.25+ BPB (game-changer but different paradigm)
- N-gram mixer (order-12, entropy-adaptive): -0.35 BPB on top of SLOT

**Tier 4 — Minor but free**:
- QK-Gain 5.0→5.25: -0.0004 BPB
- Compressibility regularization (WARMDOWN_WD_MULT=2.0): -0.001 BPB
- Brotli vs LZMA: depends on checkpoint, pick whichever is smaller
- 12 layers via mixed int5/int6: -0.01 BPB (if it fits)

### Key Observations

1. **ndokutovich is running the table.** PRs #1485, #1487, #1488 — three submissions in rapid succession, each strictly better than the last. The jump from 1.0679 to 1.0600 was pure hyperparameter tuning (TTT epochs, LR, freeze depth). This player understands the response surface.

2. **Pre-quant TTT is the biggest single technique.** -0.034 BPB standalone. Every sub-1.08 submission uses it. The tuning matters more than people realize: freeze 1 vs 2 blocks, 6 vs 10 epochs, lr 0.0005 vs 0.00045.

3. **SP1024 is not dead.** PR #1489 achieves 1.0736 with SP1024 by going MLP 4x. The freed embedding params pay for a wider model. This is an alternative lane if SP8192 data prep is a bottleneck.

4. **The SLOT paradigm has separated from everything else.** 0.2282 BPB with L-BFGS SLOT + n-gram is a completely different game. The competition may need to split tracks. The n-gram mixer alone (without SLOT) would be interesting to test on the standard stack.

5. **TMA megakernels are pure engineering alpha.** +10.5% throughput is real. But writing Triton TMA kernels is hard. The payoff is ~-0.005 BPB from extra steps.

6. **3-layer recurrence + parallel residuals are now universal.** Every frontier PR uses them. Not debatable anymore.

### Updated Lineage Tree

```
Merged SOTA:
  PR #1019 (1.1147) — AR Self-Gen GPTQ, XSA-all, BH3072

No-TTT frontier:
  PR #1394 (1.0856) — SP8192, SDClip, Recur, MuonEq-R
    └→ PR #1471 (1.0866) — + 3-layer recur (L345), EMA 0.9965
       └→ PR #1450 (1.0848) — + TMA Megakernel, triple loop, +127 steps

TTT frontier:
  PR #1394 base
    └→ PR #1471/1445 tuned hyperparams (WD=0.095, MLR=0.022, warmdown=0.72)
       └→ PR #1482 (1.0787) — + Pre-Quant TTT 8ep, QK5.25, freeze-1
          └→ PR #1485 (1.0679) — + 3-layer recur + par7 + full stack TTT 6ep
             └→ PR #1487 (1.0600) — + TTT tuning: 10ep, lr=0.00045, freeze-1 ← FRONTIER

SP1024 lane:
  PR #1019 base
    └→ PR #1489 (1.0736) — SP1024, MLP 4x, loop 4-5, par7, pre-quant TTT, ETLB

SLOT frontier:
  PR #1313 (0.8637) — AdamW SLOT-24
    └→ PR #1488 (0.8265) — + Pre-Quant TTT + QK5.25
    └→ PR #1507 (0.2282) — L-BFGS SLOT + entropy-adaptive n-gram mixer ← OVERALL BEST
```
