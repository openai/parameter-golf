# Parameter Golf: What's Constant vs. What Changes and Helps

*Empirical analysis of hyperparameters across 23 merged records + 8 frontier PRs.*
*Every value below was extracted from actual training scripts, not summaries.*

---

## TL;DR

The competition looks like a 50-dimensional optimization problem, but it's really ~12 knobs on top of a completely frozen skeleton. Most of the "design space" was inherited from the baseline and never questioned.

---

## 1. THE IMMOVABLE SKELETON (Never Changed in Any Submission)

These values appear identically in every single training script from the naive baseline (1.2244) through the merged SOTA (1.1147) and into the frontier (1.080):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model_dim` | 512 | Not 384, not 768. Nobody tried. |
| `num_heads` | 8 | — |
| `num_kv_heads` | 4 | GQA 2:1 ratio. Nobody tried MHA or different ratios. |
| `head_dim` | 64 | = 512/8. Clean power of 2, aligns with int6 packing. |
| `embed_lr` | 0.6 | AdamW LR for embedding table. **Never tuned.** |
| `head_lr` | 0.008 | AdamW LR for output head. **Never tuned.** |
| `warmup_steps` | 20 | Negligible warmup. Nobody tried more. |
| `rope_base` | 10000 | Standard RoPE base. NTK-aware variants tried but abandoned. |
| `NS_coeffs` | (3.4445, -4.7750, 2.0315) | Muon Newton-Schulz polynomial. Copy-pasted verbatim everywhere. |
| Normalization | RMSNorm | No LayerNorm anywhere, ever. |
| Embeddings | Tied (lm_head = tok_emb) | Universal. |
| Optimizer split | Muon (2D weights) + AdamW (rest) | No submission deviates. |
| Int6 STE formula | `w + (w_q - w).detach()` | Exact same code in every QAT submission. |
| Int6 range | [-32, 31], scale = max(\|w\|)/31 | Per-row. Never per-tensor, per-column, or per-group. |
| Int6 storage | int8 containers | Nobody packs 6-bit into 6-bit. Rely on entropy for compression. |

**What this means**: The entire competition optimized a thin layer of techniques *around* this frozen core. Whether these values are truly optimal or just "good enough that nobody bothered" is the biggest open question.

---

## 2. THE PHASE TRANSITIONS (Changed Once, Then Locked)

These parameters changed at specific moments and were never changed back. Each transition corresponds to a permanent BPB improvement.

### 2.1 Architecture Phase Transitions

| Parameter | Baseline → Final | When | BPB Impact | Reverted? |
|-----------|-----------------|------|------------|-----------|
| `num_layers` | 9 → 10 → **11** | Day 2 (Mar 19-20) | ~-0.015 each step | Never |
| `mlp_mult` | 2 → **3** | Day 2 (Mar 19) | ~-0.010 | Never (4x at frontier) |
| `train_seq_len` | 1024 → **2048** | Day 2 (Mar 18-19) | ~-0.020 | Never |
| `activation` | relu² → **leaky_relu(0.5)²** | Day 6 (Mar 23) | -0.003 | Never |
| `vocab_size` | 1024 → 4096 → **8192** | Frontier (PR #1218, #1394) | -0.015–0.020 | Never |
| Sliding window eval | absent → **stride=64** | Day 2 (Mar 19) | **-0.032 FREE** | Never |
| XSA | absent → last 3 → last 4 → **all 11** | Day 3→8 (Mar 20→25) | -0.006 total | Never |
| BigramHash | absent → **4096×128** → 10240×128 → 2048×128 → 3072×112 | Day 2 (Mar 19) | -0.005–0.008 | Never removed |
| Partial RoPE | full → **16/64 dims** | Day 4 (Mar 21) | -0.001–0.002 | Never |
| LN Scale | absent → **1/√(layer+1)** | Day 4 (Mar 21) | -0.001 | Never |
| VE128 | absent → **dim=128, layers 9-10** | Day 5 (Mar 22) | -0.001 | Never |

### 2.2 Optimizer Phase Transitions

| Parameter | Baseline → Final | When | Reverted? |
|-----------|-----------------|------|-----------|
| `muon_momentum` | 0.95 → **0.99** | Day 2 (Mar 19) | Never |
| `momentum_warmup` | 0.85/500 → **0.92/1500** | Day 2 (Mar 19) | Never |
| `grad_clip_norm` | 0.0 → **0.3** | Day 2-3 (Mar 19-20) | Never |
| `batch_tokens` | 524,288 → **786,432** | Day 3 (Mar 20) | Never |
| `warmdown_iters` | 1200 → 2500 → 3000 → **3500** | Day 2→5 (Mar 19→22) | Never shortened |
| EMA | absent → **decay=0.997** | Day 3 (Mar 20) | Never |
| QAT threshold | absent → **lr_scale < 0.15** | Day 4-5 (Mar 21-22) | Never |
| GPTQ | absent → lite → **Full Hessian** | Day 5→8 (Mar 22→25) | Never downgraded |
| Compression | zlib → **zstd-22** → LZMA/Brotli | Day 2→8 | Never went back to zlib |

### 2.3 Frontier-Only Transitions (Post-Merge)

| Parameter | Standard → Frontier | PRs | Impact |
|-----------|-------------------|-----|--------|
| Depth recurrence | absent → **layers 4-5, 2-3 loops** | #1204, #1285+ | -0.008–0.012 |
| Parallel residuals | absent → **layers 7+** | #1204, #1296+ | -0.002–0.005 |
| MuonEq-R | Muon → **row-normalized Muon** | #1260+ | -0.001–0.002 |
| Polar Express NS | 5 fixed steps → **4 minimax-optimal** | #1344 | Saves compute (~180 extra steps) |
| SDClip | absmax clip → **k·σ(row), k=12.85** | #1394 | -0.003–0.005 |
| GPTQ Embeddings | embed unquantized → **int8 GPTQ** | #1394 | Size savings |
| Linear LR warmdown | cosine-like → **linear to 0** | #1395 | 61% quant gap reduction |
| SmearGate | present → **removed** | #1204+ | Required for depth recurrence compatibility |

---

## 3. THE KNOBS PEOPLE ACTUALLY TUNE (Varies Across Good Submissions)

These are the parameters that competitors actively optimize. Unlike the frozen skeleton, these differ between submissions and correlate with BPB improvement.

### 3.1 `matrix_lr` (Muon learning rate for weight matrices)

| Submission Era | Value | BPB Range |
|---------------|-------|-----------|
| Baseline (Mar 17) | 0.04 | 1.22 |
| Day 2 experiments | 0.02–0.032 | 1.15–1.20 |
| Merged SOTA (Mar 25) | **0.025** | 1.1147 |
| Frontier PRs | 0.020–0.022 | 1.08–1.09 |

**Trend**: Decreasing over time. Started at 0.04, settled to 0.025 for records, frontier pushing to 0.020. Not a clean monotonic relationship — some 0.02 submissions beat 0.04 ones, but 0.04 also appears in good early submissions.

### 3.2 `weight_decay` (The Stealth Compression Lever)

| Submission Era | Muon WD | Adam WD | BPB |
|---------------|---------|---------|-----|
| Baseline | 0.0 | 0.0 | 1.22 |
| Mar 19 (SmearGate) | 0.01 | — | 1.156 |
| Mar 20 (11L XSA) | 0.02 | 0.01 | 1.127 |
| Mar 22 (GPTQ-lite) | **0.04** | **0.04** | 1.123 |
| Merged SOTA | **0.04** | **0.04** | 1.115 |
| Frontier | **0.085–0.105** | 0.020–0.105 | 1.08–1.10 |

**Trend**: Strongly increasing. The frontier uses 2x–2.5x the SOTA's weight decay. This is the most actively tuned parameter at the frontier. The mechanism: higher WD → smaller weight magnitudes → lower entropy → better compression → more room for parameters within 16MB. PR #1218 found R²=0.99 correlation between weight RMS and compressed artifact size.

### 3.3 `warmdown_iters` (How long to anneal LR)

| Era | Value | BPB |
|-----|-------|-----|
| Baseline | 1200 | 1.22 |
| Day 2-3 | 2500–3000 | 1.13–1.16 |
| SOTA+ | **3500** | 1.11–1.12 |
| Frontier | **3500–4000** | 1.08–1.10 |

**Trend**: Longer is consistently better. The transition from 1200→3000 was worth ~0.02 BPB. Further extension to 3500+ gives diminishing but real gains. The frontier is testing 4000.

### 3.4 `BigramHash` configuration

| Era | Config | BPB |
|-----|--------|-----|
| PR #65 | 4096×128 | 1.156 |
| PR #180 | 10240×128 | 1.143 |
| PR #414+ | 2048×128 | 1.123 |
| SOTA | 2048×128 | 1.115 |
| Frontier | 3072×112, or **removed** (SP8192) | 1.08 |

**Trend**: Non-monotonic! Buckets went up (4096→10240) then down (→2048) then up again (→3072). The dimension shrank (128→112). At the SP8192 frontier, BigramHash may become unnecessary — the larger vocab captures more bigram information natively. PR #1420 (1.08014) uses n-gram tilt instead of BigramHash.

### 3.5 `QK_GAIN` (Query-Key scaling)

| Era | Value | Context |
|-----|-------|---------|
| Pre-PR#1125 | 1.0 (default) | Standard |
| PR #1125 ablation | **4.0** | 45-experiment validation |
| Frontier | **4.0–5.0** | Unsettled |

**Observation**: Introduced relatively late, jumped straight to 4.0 based on one large ablation study. Some frontier PRs push to 5.0 but PR #1418 reported QK-Gain=5.0 *degraded* in their setup. Hardware/config sensitive.

---

## 4. THINGS THAT DON'T MATTER (Wide Variation, No Signal)

These parameters vary across submissions but show no clear correlation with BPB:

| Parameter | Range Observed | Correlation with BPB |
|-----------|---------------|---------------------|
| Compression codec (zstd vs LZMA vs Brotli) | All three | Within ~0.1MB of each other |
| SmearGate (present/absent) | Both in records | Helps without recurrence, hurts with it |
| U-Net skip connections | Sometimes present | Absent from SOTA and frontier |
| VE128 (Value Embeddings) | Optional | ~-0.001 BPB, inconsistently used |
| MTP heads | Always disabled | Coded but never enabled |
| GPTQ clip candidates | 5 percentiles standard | Grid not explored |
| SWA configuration | Varies wildly | Conflated with EMA; SWA sabotages QAT |
| Number of GPTQ calibration batches | 128–256 | No clear pattern |

---

## 5. THE EVOLUTION VISUALIZED

```
BPB   Technique Added                                        Date
1.224 ┤ BASELINE (9L, int8, relu², seq1024, zlib)            Mar 17
      │
1.206 ┤ +seq2048                                             Mar 18
      │
1.193 ┤ +sliding window eval (FREE -0.032)                   Mar 19
1.163 ┤ +int6, +MLP3x                                       Mar 19
1.156 ┤ +SmearGate, +BigramHash, +Muon WD                   Mar 19
      │
1.143 ┤ +10L, +BigramHash 10240                             Mar 20
1.131 ┤ +11L, +XSA (last 3), +FA3                           Mar 20
1.127 ┤ +EMA (0.997), +XSA (last 4)                         Mar 20
      │
1.125 ┤ +Partial RoPE, +LN Scale                            Mar 21
1.123 ┤ +GPTQ-lite, +warmdown 3500, +Late QAT               Mar 22
1.119 ┤ +LeakyReLU², +Legal TTT, +Parallel Muon             Mar 23
      │
1.115 ┤ +Full GPTQ, +XSA-all, +AR calibration     SOTA     Mar 25
      │                                          ─────────
      │  ~~~~ merging stopped ~~~~
      │
1.098 ┤ +SP4096, +MLP4x, +WD=0.085                          Mar 28
1.093 ┤ +MuonEq-R, +depth recurrence                        Mar 29
1.090 ┤ +parallel residuals, +3-layer recurrence             Mar 30
1.086 ┤ +SP8192, +SDClip, +GPTQ embed                       Apr 1
1.083 ┤ +Legal TTT reintegrated, +QK-Gain 5                 Apr 3
1.080 ┤ +n-gram tilt / +dTTT / +triple loop      FRONTIER   Apr 5
      │
```

---

## 6. WHAT THIS MEANS FOR A COMPETITOR

### The "Free Points" You Must Have (non-negotiable)
1. Sliding window eval stride=64 (-0.032 BPB)
2. 11 layers, 512d, MLP 3x+, seq2048+
3. Int6 STE QAT with EMA(0.997)
4. Full Hessian GPTQ
5. LeakyReLU(0.5)²
6. XSA on all layers
7. Partial RoPE (16/64), LN Scale
8. grad_clip=0.3, batch_tokens=786432, warmdown≥3500

### The Knobs Worth Tuning
1. **Weight decay** (0.04→0.10+ range) — biggest single lever at frontier
2. **Tokenizer** (SP4096 vs SP8192) — SP8192 is winning
3. **Depth recurrence** config (which layers, how many loops, activation threshold)
4. **Parallel residuals** (which layers to start)
5. **warmdown_iters** (3500→4000)
6. **matrix_lr** (0.020–0.025 range)

### The Untouched Design Space (Potential Edge)
1. **model_dim**: Nobody tried anything other than 512
2. **GQA ratio**: Nobody tried anything other than 2:1
3. **embed_lr / head_lr**: Never tuned (0.6 / 0.008 from baseline)
4. **NS coefficients**: Never modified
5. **Quantization format**: Always int6 per-row uniform. No per-group, no NF4, no non-uniform grids.
6. **Int4 for MLP**: PR #1426 is the first attempt, results pending
7. **head_dim**: Always 64. What about 32 or 128?

### The Graveyard (Confirmed Dead Ends)
- SWA + QAT together (sabotage)
- SmearGate + depth recurrence (incompatible)
- Post-GPTQ TTT (fundamentally incompatible)
- SLOT in any form (causal violation)
- SSM/Mamba architectures (worse compression ratios)
- Scylla/custom tokenizers (byte accounting traps)
- n-gram caches with hash tables (invalid distributions)
- Pre-quant TTT on val tokens (likely illegal)

---

## 7. THE BOTTOM LINE

**90% of the BPB improvement came from ~12 discrete technique additions**, each of which was discovered once, validated, and permanently locked in. The remaining 10% comes from tuning weight decay, warmdown length, tokenizer size, and depth recurrence config.

The skeleton (512d/8H/4KV, embed_lr=0.6, NS coefficients) has **never been questioned**. This is either because it's optimal, or because the competition converged on it too early and nobody had the compute budget to re-explore from scratch. For a new competitor, challenging these assumptions — especially the quantization format and model dimensions — represents the highest-risk, highest-reward strategy.
