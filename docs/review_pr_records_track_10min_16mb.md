# Open PR Analysis: Parameter Golf 10min_16mb Track

## Summary Table (sorted by val_bpb, lower is better)

| Rank | PR | Author | val_bpb | Seeds | Layers | Params | Artifact | Key Techniques |
|------|-----|--------|---------|-------|--------|--------|----------|----------------|
| 1 | #315 | jfprincz | **1.1248** | 3 (mean 1.1250) | 11 | 26.8M | 15.6 MB | Partial RoPE + LN Scale + Late QAT + EMA + XSA4 |
| 2 | #338 | alertcat | **1.1254** | 3 (mean 1.1256) | 11 | 26.8M | 15.55 MB | XSA + EMA + TTT (SGD) |
| 3 | #287 | jfprincz | **1.1271** | 3 (mean 1.1280) | 11 | 26.8M | 15.5 MB | XSA4 + EMA replacing SWA |
| 4 | #254 | timowhite88 | **1.1303** | 1? | 11 | ~27M | ~15.5 MB | TTT + 11L Int6 MLP3x (FarnsworthEngine) |
| 5 | #265 | unnir | **1.1307** | 1 | 11 | 26.8M | 15.9 MB | Efficient Partial XSA (GQA-aware) |
| 6 | #332 | saml212 | **1.1320** | 3 (mean 1.1320) | 12 | 27.6M | 15.7 MB | Gradient-Guided Quant + 12L + Partial RoPE + LN Scale |
| 7 | #290 | ibarrajo | **1.1354** | 1 | 11 | ~27M | 15.85 MB | Partial XSA + TTT + BatchOpt |
| 8 | #307 | dennisimoo | **1.1357** | 1 | 11 | 26.8M | 15.67 MB | XSA4 + EMA + Batch524K + zstd fallback |
| 9 | #339 | sheeki03 | **1.1364** | 1 | 11 | ~27M | 16.17 MB* | Backout Connection (learned residual subtraction) |
| 10 | #236 | saml212 | **1.1400** | 3 (mean 1.1400) | 11 | 26.5M | 15.7 MB | Int6-all + Batch524K optimization |
| 11 | #274 | haikosys | **1.1403** | 3 (mean 1.1403) | 10 | ~22M | 15.7 MB | Stride-32 + Warmdown/Muon tuning |
| 12 | #267 | andrewgcodes | **1.1402** | 3 | ~11 | ~27M | ~15.5 MB | Causal TTT during eval |
| 13 | #317 | chris-buckley | **1.1442** | 2 (mean 1.1442) | 11 | ~27M | 15.6 MB | XSA4 + EMA + TTT + Int6 MLP3x |
| 14 | #327 | Ananddna | **1.1450** | 2 (mean 1.1450) | 10 | ~22M | ~15 MB | TrigramHash + Partial RoPE + HeadTemp |
| 15 | #264 | stukenov | **1.1455** | 1 | 11 | 26.7M | 15.94 MB | Int5-MLP + TTT-SGD + SmearGate |
| 16 | #295 | gowtham0992 | **1.1477** | 1 | ~10 | ~22M | 15.94 MB | QAT Int5/Int6 + Backout + U-Net Skips |
| 17 | #331 | Rhodrium | **1.1487** | 3 (mean 1.1487) | 10 | ~22M | 14.9 MB | MLP3x + BigramHash + SWA + Stride-32 |
| 18 | #289 | integrate-your-mind | **1.1518** | 1 | 11 | 26.8M | 15.2 MB | SmearGate + BigramHash + U-Net Skips |
| 19 | #230 | MatthewHRockwell | **1.1541** | 1 | ~9 | ~22M | 15.99 MB | NorMuon + SmearGate + BigramHash |
| 20 | #333 | mahsumaktas | **1.1565** | 3 (mean 1.1565) | 11 | ~27M | 15.9 MB | XSA4 + SmearGate + BigramHash + SWA + RoPE50K |

*PR #339 is over the 16MB cap (16.17 MB) -- needs int5 MLP to fix.

---

## Detailed Per-PR Analysis

### #1 -- PR #315: val_bpb = 1.1248 (CURRENT BEST)
- **Author**: jfprincz
- **Title**: Record: 11L Partial RoPE + LN Scale + EMA + Late QAT + XSA4
- **Architecture**: 11L, 512d, 8H/4KV, MLP 3x, relu-squared, 26.8M params
- **Artifact**: 15,612,308 bytes (int6 + zstd-22)
- **Seeds**: 3 seeds tested (2025: 1.1248, 42: 1.1250, 1337: 1.1253), mean 1.1250, std 0.0005
- **Key techniques**:
  - **Partial RoPE (16 of 64 dims)**: Only 25% of head dims get rotary embeddings; rest are position-free. Zero new params, improves generalization.
  - **LN Scale**: RMSNorm output scaled by 1/sqrt(layer_idx+1), damping deeper layers.
  - **Late QAT**: STE int6 fake-quantization only in final ~4% of training (lr_scale < 0.1). Cuts int6 degradation 3x.
  - XSA on last 4 layers, EMA (0.997), SmearGate, BigramHash(2048), OrthoInit, WD=0.04, FA3
- **Novel contributions**: Three zero-param techniques (Partial RoPE, LN Scale, Late QAT timing) that stack cleanly.
- **Convergence**: 7,051 steps at 85ms/step. Pre-quant 1.1418, post-quant roundtrip 1.1485, sliding 1.1248.
- **Run command**: Provided (see PR body above).

### #2 -- PR #338: val_bpb = 1.1254
- **Author**: alertcat
- **Title**: Record: 11L XSA+EMA+TTT, sliding val_bpb=1.1254
- **Architecture**: 11L, 512d, 8H/4KV, MLP 3x, 26.8M params
- **Artifact**: 15.55 MB
- **Seeds**: 3 seeds (42: 1.1254, 1337: 1.1258, 2024: 1.1256), mean 1.1256, std 0.0002
- **Key techniques**:
  - **TTT (Test-Time Training)**: 3 epochs SGD on validation tokens (lr=0.002, momentum=0.9), first 2 blocks frozen. ~47s on 8xH100. Improves post-quant BPB by ~0.002.
  - Built on PR #315's stack (XSA4, EMA, SmearGate, BigramHash, OrthoInit)
- **Novel contributions**: First submission combining XSA + EMA + TTT. Shows TTT is orthogonal to the XSA/EMA stack.
- **Eval timing**: Training 600s + TTT 47s + Sliding eval 73s = ~120s total eval.

### #3 -- PR #287: val_bpb = 1.1271
- **Author**: jfprincz
- **Title**: Record: 11L XSA + EMA + Int6 MLP3x + WD=0.04
- **Architecture**: 11L, 512d, 8H/4KV, MLP 3x, 26.8M params
- **Artifact**: 15,534,645 bytes
- **Seeds**: 3 seeds (1337: 1.1271, 42: 1.1286, 2025: 1.1284), mean 1.1280
- **Key techniques**:
  - **Exclusive Self Attention (XSA)** on last 4 layers: removes self-value bias via orthogonal projection. Zero new params.
  - **EMA replacing SWA** (decay=0.997): smoother weight averaging, better generalization and compression.
- **Novel contributions**: Introduced XSA + EMA combo to this competition.

### #4 -- PR #254: val_bpb = 1.1303
- **Author**: timowhite88 (Fawnsworth)
- **Title**: Record: FarnsworthEngine v1 -- TTT + 11L Int6 MLP3x
- **Body**: Empty (no description provided)
- **Seeds**: Unknown
- **Notes**: Score only from title. No details available in PR body.

### #5 -- PR #265: val_bpb = 1.1307
- **Author**: unnir (Vadim Borisov)
- **Title**: Record: 11L + Efficient Partial XSA
- **Architecture**: 11L, 512d, 8H/4KV, MLP 3x, U-Net skips, 26.8M params
- **Artifact**: 15,892,986 bytes
- **Seeds**: 1 (seed 1337)
- **Key techniques**:
  - **Efficient GQA-Aware XSA**: Rewrites XSA to use free reshape + broadcasting instead of expensive repeat_interleave. Reduces XSA overhead from ~7ms/step to ~2ms/step.
  - **Partial XSA**: Applied only to last 3 layers (of 11), targeting highest self-attention bias.
  - NTK-aware RoPE, SWA every 120 steps (13 checkpoints)
- **Novel contributions**: Memory-efficient XSA implementation for GQA architectures. Reference: arXiv:2603.09078.

### #6 -- PR #332: val_bpb = 1.1320
- **Author**: saml212 (Sam Larson)
- **Title**: Record: 12L Gradient-Guided Quant + Partial RoPE + LN Scale + EMA + XSA4
- **Architecture**: **12 layers** (unique!), 512d, MLP hidden=1408 (narrower), 27.6M params
- **Artifact**: 15,652,352 bytes
- **Seeds**: 3 seeds (1337: 1.1321, 1338: 1.1321, 1339: 1.1318), mean 1.1320, std 0.0002
- **Key techniques**:
  - **Gradient-Guided Adaptive Quantization**: Accumulates per-tensor squared gradient magnitudes during last 10% of warmdown. Top 10% sensitivity -> int7, middle 70% -> int6, bottom 20% -> int5. Saves ~1 MB vs uniform int6, funding the 12th layer.
  - **12 layers**: Extra depth from compression headroom. MLP narrowed to 1408 from 1536.
  - Batch=524K for more steps, Partial RoPE, LN Scale, XSA4, EMA
- **Novel contributions**: First 12-layer submission. Gradient-guided mixed-precision quantization is genuinely novel.
- **Negative finding**: Late QAT hurts at 12 layers (step overhead too high).

### #7 -- PR #290: val_bpb = 1.1354
- **Author**: ibarrajo (Alex Ibarra)
- **Title**: Record: 11L + Partial XSA + TTT + BatchOpt
- **Architecture**: 11L, standard stack
- **Artifact**: 15,851,371 bytes
- **Seeds**: 1 (budget constrained)
- **Key techniques**: Partial XSA (last 3 layers), TTT (3-epoch SGD), Batch=524K, RoPE base 50K
- **Notes**: Uses SDPA fallback (no FA3). With FA3, would get ~600 more steps.

### #8 -- PR #307: val_bpb = 1.1357
- **Author**: dennisimoo (Dennis Khylkouski)
- **Title**: Record: 11L XSA4 + EMA + Batch524K + zstd fallback
- **Architecture**: 11L, 512d, 26.8M params
- **Artifact**: 15,669,953 bytes
- **Seeds**: 1
- **Key techniques**: Batch=524K (more steps), SDPA fallback, torch.compile behind env flag, zstd Python-or-CLI fallback
- **Convergence**: 8,202 steps at 73ms/step

### #9 -- PR #339: val_bpb = 1.1364 (OVER SIZE LIMIT)
- **Author**: sheeki03 (Sheeki)
- **Title**: Record: 11L Backout + Int6 + SWA
- **Architecture**: 11L, 512d, BigramHash(4096)
- **Artifact**: 16,170,051 bytes -- **170KB over the 16MB cap**
- **Seeds**: 1
- **Key techniques**:
  - **Backout Connection**: Learned residual subtraction from mid-network hidden state. `lambda * h_mid` subtracted from final representation. Lambda is a learned scalar (init 0.2). Zero matrix params.
  - Controlled comparison shows -0.0071 BPB improvement over baseline.
- **Novel contributions**: Backout connection is a genuinely new idea. Needs INT5_MLP to bring under cap.

### #10 -- PR #236: val_bpb = 1.1400
- **Author**: saml212 (Sam Larson)
- **Title**: 4 the Leaderboard: 11L Int6 + SmearGate + Batch Optimization
- **Architecture**: 11L, 512d, MLP 3x, 26.5M params
- **Artifact**: 15.7 MB
- **Seeds**: 3 (1337: 1.1411, 1338: 1.1381, 1339: 1.1408), mean 1.1400
- **Key techniques**:
  - **Batch size finding**: 524K beats 786K. 22% more gradient updates outweigh 17% fewer total tokens.
  - **Int6-all beats int5-MLP**: quant penalty 0.010 vs 0.029 BPB
  - **Int8 tok_emb instead of fp16**: ~250KB savings with <0.001 BPB cost
  - **WD/artifact tradeoff**: WD directly controls artifact size through weight magnitude
- **Novel contributions**: Excellent empirical analysis of batch size, quantization, and WD tradeoffs.

### #11 -- PR #274: val_bpb = 1.1403
- **Author**: haikosys
- **Title**: Stride-32 + Warmdown/Muon Tuning on SOTA #1
- **Architecture**: 10L baseline
- **Artifact**: ~15.7 MB
- **Seeds**: 3 (42: 1.1392, 1337: 1.1417, 7: 1.1400), mean 1.1403
- **Key techniques**: EVAL_STRIDE=32 (2x context overlap), extended warmdown (5000 from 3000), lower Muon momentum (0.95 from 0.99), batch 524K
- **Novel contributions**: Pure hyperparameter tuning on env vars only -- no code changes.

### #12 -- PR #267: val_bpb = 1.1402
- **Author**: andrewgcodes (Andrew Kean Gao)
- **Title**: Record: val_bpb: 1.14020 [tested 3x on 8xh100]
- **Seeds**: 3
- **Key techniques**: Causal TTT during validation (each chunk evaluated first, then trained on -- tokens scored exactly once)
- **Novel contributions**: Clarified TTT compliance rules (no re-evaluation, no multiple passes).

### #13 -- PR #317: val_bpb = 1.1442
- **Author**: chris-buckley
- **Title**: Record: 11L XSA4 + EMA + TTT + Int6 MLP3x
- **Architecture**: 11L, 512d, standard stack
- **Artifact**: ~15.6 MB
- **Seeds**: 2 (1337: 1.1419, 1338: 1.1464), mean 1.1442
- **Key techniques**: XSA4, EMA, TTT (3 epochs SGD), SDPA fallback, zstd-22
- **Notes**: Different nodes caused different step times (109ms vs 132ms).

### #14 -- PR #327: val_bpb = 1.1450
- **Author**: Ananddna (Anand Rajasekaran)
- **Title**: TrigramHash + PartialRoPE + HeadTemp + stride32
- **Architecture**: 10L, 512d, BigramHash(10240) + TrigramHash(8192)
- **Seeds**: 2 (42: 1.1449, 1337: 1.1451), mean 1.1450, std 0.0001
- **Key techniques**:
  - **TrigramHashEmbedding**: Hash consecutive token triplets into 8192-bucket embeddings (dim=64). Captures 3-word patterns.
  - **Partial RoPE (50%)**: Position-free attention on half the dims.
  - **Per-Head Temperature Scaling**: Each head learns its own temperature.
- **Novel contributions**: TrigramHash is new. Per-head temperature is simple but unexplored here.

### #15 -- PR #264: val_bpb = 1.1455
- **Author**: stukenov (Saken Tukenov)
- **Title**: 11L Int5-MLP + TTT-SGD + SmearGate + SWA
- **Architecture**: 11L, 512d, MLP 3x, 26.7M params
- **Artifact**: 15.94 MB
- **Seeds**: 1 (3-seed in progress)
- **Key techniques**: Int5 MLP + Int6 attn, full-model SGD TTT (2 epochs), SWA (30 checkpoints)
- **Eval timing**: TTT 422s + sliding eval 273s = 696s (tight on 600s eval limit)

---

## Cross-Cutting Observations

### Dominant Architecture
Nearly all top submissions converge on:
- **11 layers, 512 dim, 8 heads, 4 KV heads (GQA), MLP 3x (1536 hidden), relu-squared**
- ~26.8M parameters
- Exception: PR #332 uses 12 layers with narrower MLP (1408) funded by gradient-guided mixed quantization

### Technique Adoption Across Top 20

| Technique | Adoption | Impact |
|-----------|----------|--------|
| Int6 quantization + zstd-22 | 19/20 | Standard. ~1MB savings over int8+zlib |
| SmearGate | 18/20 | Learned gate for token-predecessor blending |
| BigramHash | 17/20 | 2048-10240 bucket hash table for bigram context |
| Sliding window eval (stride 32-64) | 20/20 | ~0.02-0.04 BPB improvement over standard eval |
| OrthoInit + muP scaling | 15/20 | Stabilizes deeper models |
| XSA (Exclusive Self Attention) | 12/20 | Zero-param debiasing, ~0.002-0.005 BPB |
| EMA (replacing SWA) | 8/20 | Smoother averaging, better compression |
| TTT (Test-Time Training) | 7/20 | ~0.002-0.005 BPB at eval time |
| Partial RoPE | 3/20 | Zero-param, improves generalization |
| LN Scale | 2/20 | Zero-param, stabilizes deep models |
| Late QAT | 3/20 | Reduces quant gap, but costs training steps |
| Batch=524K (vs 786K) | 8/20 | More gradient steps in fixed time |

### Novel Techniques Worth Watching
1. **Gradient-Guided Adaptive Quantization** (PR #332): Per-tensor sensitivity-based int5/int6/int7 allocation
2. **Backout Connection** (PR #339): Learned negative residual from mid-network
3. **TrigramHash** (PR #327): Token triplet hashing
4. **Per-Head Temperature** (PR #327): Learned per-head attention scaling
5. **Error Correction Table** (PR #232): Pre-computed position-based logit boosting (likely rule-violating)
6. **Canon ACD Layers** (PR #312): From "Physics of Language Models" paper
7. **Efficient GQA-Aware XSA** (PR #265): Zero-allocation XSA via reshape + broadcasting

### Frontier: What Gets You Below 1.13?
The top 2 PRs (1.1248 and 1.1254) combine:
- 11L/512d/MLP3x base
- XSA on last 4 layers
- EMA (decay=0.997)
- SmearGate + BigramHash(2048)
- OrthoInit + muP + WD=0.04
- Int6 + zstd-22
- Sliding window eval stride=64
- Plus either: Partial RoPE + LN Scale + Late QAT (#315) or TTT (#338)

The next frontier likely requires combining BOTH Partial RoPE/LN Scale/Late QAT AND TTT, plus potentially gradient-guided quantization to fund a 12th layer.

### Reproducibility Assessment
- **Strong (3+ seeds, low variance)**: PRs #315, #338, #332, #236, #274, #331
- **Moderate (2 seeds or 3 seeds with higher variance)**: PRs #287, #317, #327, #333
- **Weak (single seed or pending compute)**: PRs #254, #265, #290, #307, #339, #264, #295, #289, #230
- **Pending compute (no actual results)**: PRs #268, #322, #314
