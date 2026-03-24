# Community PR Survey — openai/parameter-golf

Surveyed 2026-03-20. 164 total PRs reviewed from the original repo.

## Leaderboard (All Submissions with Scores)

| Rank | PR | BPB | Author | Key Stack |
|------|-----|------|--------|-----------|
| 1 | #180 | 1.1453 | thwu1 | Mixed int5/int6, 10L, MuonWD=0.04, SWA/50, SmearGate, BigramHash |
| 2 | #179 | 1.1472 | devin-cog (Silas Alberti) | 11L, MuonWD=0.038, LR=0.025, int6+zstd, stride=64 |
| 3 | #162 | 1.1483 | raahilshah | SmearGate + BigramHash + OrthoInit + MuonWD=0.02 + SWA/200, 3-seed mean |
| 4 | #164 | 1.1524 | jfprincz | OrthoInit + SmearGate + BigramHash + FA3 + Muon 0.99, seq2048 |
| 5 | #173 | 1.1532 | tamoghnokandar | NorMuon + FA3 on top of #114 stack |
| 6 | #135 | 1.1539 | unnir | OrthoInit + SmearGate + BigramHash (stride=64) |
| 7 | #65 | 1.1556 | aquariouseworkman | SmearGate + OrthoInit + MuonWD + Int6 STE QAT + MLP 3x + U-Net skips |
| 8 | #114 | 1.1574 | saml212 | Int6 + MLP 3x + selective precision + seq2048 (current merged SOTA) |
| 9 | #122 | 1.1585 | mtybadger | 2048 vocab, NorMuon, SWA, FA3 (8L, bigger vocab) |
| 10 | #150 | 1.1593 | yahya010 | Int6 QAT + BigramHash + 10L + MLP 1344 |
| 11 | #128 | 1.1594 | rsavitt | STE fake int6 QAT + MLP 3x + sliding window |
| 12 | #63 | 1.1598 | yahya010 | 10L Int6 QAT + zstd + MLP 1344 + warmdown 3600 + stride=64 |
| 13 | #156 | 1.1602 | dexhunter | Int6 STE + NorMuon + SWA + U-Net skip connections, seq1024 |
| 14 | #88 | 1.1605 | seanward | MTP auxiliary head (training-only) + int6 + seq4096 |
| 15 | #99 | 1.1605 | takhir-iota | Int6 MLP3x + Late-K fp16 passthrough + stride=64 |
| 16 | #102 | 1.1618 | unnir | Int6 MLP3x + SmearGate + tuned LR + stride=64 |
| 17 | #89 | 1.1622 | vmfunc | NorMuon + int6 STE + SWA, seq1024 |
| 18 | #160 | 1.1623 | ChaseWNorton | MLP3x + Int8 Tok Emb + Grouped LZMA + sliding window |
| 19 | **#147** | **1.1631** | **ours** | Smaller batch (524K) SOTA stack |
| 20 | #66 | 1.1632 | arjun-krishna1 | AutoResearch agent composing community PRs. MLP 3x + STE int6 QAT + seq4096 |
| 21 | #107 | 1.1648 | m0at | Int6+zstd MLP1488 + QAT + tuned LR + stride=64 |
| 22 | #86 | 1.1652 | aruniyer | MLP 3x + QAT + Int6 + sliding window |
| 23 | #70 | 1.1659 | jfprincz | Wider MLP 3x + int6 + sliding window (earlier version of #164) |
| 24 | #137 | 1.1666 | abhishekgahlot2 | Int6 + MLP 3x + STE QAT + NorMuon + sliding window |
| 25 | #178 | 1.1667 | timowhite88 | "Nuclear Stack" |
| 26 | #170 | 1.1669 | baudrillardsgh0st | Int6 QAT + SmearGate + MuonWD=0.01 |
| 27 | #81 | 1.1670 | polarizedfortnite-cpu | SwiGLU + MLP 3x + Int6 + LoRA TTT |
| 28 | #117 | 1.1702 | trovatochris | Int6 MLP3x + QAT + sliding window |
| 29 | #69 | 1.1708 | TevBenji | Int6 QAT + MLP 3x + SWA + sliding window |
| 30 | #176 | 1.1732 | GLDRoger | 10L slide64 mid6 |
| 31 | #60 | 1.1748 | notapplica | Sliding window + FP16 embed + 10L + MuonWD + overtone init (prev merged SOTA) |
| 32 | #96 | 1.1764 | saml212 | Sliding window + long-context training (seq2048) |
| 33 | #152 | 1.1767 | timowhite88 | TTT (Test-Time Training) |
| 34 | #155 | 1.1876 | peytontolbert | 10L + MuonWD + overtone init + phase-transition resid_mix |
| 35 | #142 | 1.1925 | **ours** | Quant Quality (seq4096 trunk + clip percentile) |
| 36 | #77 | 1.1950 | samacqua | LoRA TTT + sliding window (merged record, planted TTT seed) |
| 37 | #161 | 1.1957 | santosh5541 | TTT-LoRA 512d |
| 38 | #169 | 1.1973 | beee003 | Sliding window + Muon6 |
| 39 | #139 | 1.2029 | ksang123 | BitNet b1.58 — 65M ternary params |
| 40 | #151 | 1.2045 | mrdavtan | FP16 embed + WD20k + seq2048 + doc-isolated sliding |
| 41 | #141 | 1.2075 | nglain | Systematic hyperparameter search |
| 42 | #163 | 1.2091 | Focus2321 | SwiGLU dim=576 + sliding window + MuonWD |
| 43 | #136 | 1.2101 | ibarrajo | Seq2048 training + eval |
| 44 | #39 | 1.2147 | nanlliu | 10L mixed precision (int6 middle layers) |
| 45 | #61 | 1.2154 | saml212 | Warmdown-quantization discovery (WD=20000) |
| 46 | #92 | 1.1940 | saikrishnarallabandi | 8192 vocab + sliding window + selective quant |
| 47 | #78 | 1.1860 | mtybadger | 8192 vocab + NorMuon + selective quant |
| 48 | #123 | 1.1642 | saikrishnarallabandi | Vocab 4096 + MLP 3x + sliding window, 3 seeds |
| 49 | #181 | 1.2194 | manfromnowhere143 | "Aweb Optimized Baseline" |
| 50 | #95 | 1.2253 | MatoTeziTanka | PROTEUS EMA |
| 51 | #146 | 1.2987 | swapp1990 | Warmdown tuning on 1xRTX 5090 |
| 52 | #104 | 1.3360 | gwelinder | Stacked hyperparam tuning (RTX 5090) |

### Questionable / Non-Record

| PR | BPB | Note |
|----|------|------|
| #168 | 1.0238 | "Paid prefix" — 8.75MB blob stored in artifact. Likely compressed val data. |
| #179 (val-only) | 0.9274 | Trains on val data. Marked non-record by author. |
| #120 | 0.9588 | Val-only. Closed. |
| #64 | 0.9695 | Val-only submission. |
| #44 | 1.1111 | Val-only. Closed. |

### Non-Scoring / WIP / Tooling / Infra

PRs #153 (eval pipeline tooling), #133 (MLX harness), #131 (WIP), #130 (7 toggleable improvements, no final score), #127 (depth recurrence non-record), #126 (BitNet 1.75 BPB), #125 (non-record), #119 (runtime replay), #118 (4090 smoke), #115 (no final score), #113 (pending eval), #112/#109 (depth recurrence 5x3), #111 (1-hour 1xH100), #110 (pending eval), #108 (MLX Apple Silicon), #103 (looped transformer non-record), #100 (MLX fix, merged), #98 (WD experiment WIP), #97 (deeper compact), #94/#93 (non-record warmdown), #91 (depth recurrence 1.589), #90 (int7 non-record), #85 (92-experiment autoresearch pending), #84 (MLX non-record), #80 (Aria tooling), #79 (depth recurrence pending H100), #76 (WIP ~1.160), #75/#74 (seq4096 fp16 tok coarsen), #73 (SwiGLU 5090), #72 (tooling), #71 (12x384), #68 (M3 Pro), #62 (WIP adaptive eval), #58 (WIP depth recurrence), #56 (MLX), #55 (scaffold), #54 (depth recurrence), #53 (SP-4096 1.1888), #51 (WIP QAT), #48/#47/#46/#45 (closed early attempts), #40 (shared-recurrent 3.23), #38/#37 (WIP closed), #36 (MLX docs), #34 (dispersion loss), #33 (WIP), #31/#30/#29 (depth recurrence closed), #24/#23/#22 (harness/hooks), #21/#20 (early closed), #19/#18/#16/#15/#14/#13/#11/#10/#9/#8/#7/#6/#5/#4/#2/#1 (misc early/infra/closed).

---

## Technique Inventory

### Training Dynamics

| Technique | PRs | Est. Impact | Description |
|-----------|-----|-------------|-------------|
| **NorMuon** | #89, #122, #137, #156, #173 | 0.005-0.01 | Row-normalized Newton-Schulz on top of Muon. Drop-in from modded-nanogpt. |
| **Muon Weight Decay** | #60, #162, #170, #179, #180 | 0.003-0.01 | WD on Muon-managed params (p.mul_(1 - wd*lr)). Reduces weight norms, improves quant friendliness. Best: 0.02-0.04. #179 found 0.038 optimal for 11L. |
| **SWA** | #69, #89, #122, #156, #162, #180 | 0.002-0.005 | Stochastic Weight Averaging during warmdown. Checkpoint every 50-200 steps, average at end. Smoother weights, better quant. #180 found SWA/50 (~29 ckpts) best. |
| **FlashAttention 3** | #122, #164, #173 | ~10ms/step | Direct FA3 kernel calls. ~5% step time reduction vs SDPA. Free lunch on H100. |
| **OrthoInit + muP** | #65, #135, #162, #164 | ~0.005 | Orthogonal init on all large matrices. Output projections scaled by 1/sqrt(2*num_layers). Faster early convergence. |
| **MTP (Multi-Token Prediction)** | #88 | ~0.003 | Auxiliary next-token head during training only, excluded from artifact. Free gradient signal. |
| **Grad Clip 0.3** | #63, #114 | ~0.003 | Tight gradient clipping for seq2048+ stability. Swept 0.0-1.0, optimum at 0.3. |
| **Momentum warmup** | #114, #162, #164 | standard | Muon momentum 0.92 -> 0.99 over 1500 steps. Nearly universal in top submissions. |
| **Higher LR** | #179 | ~0.002 | matrix_lr=0.025 (vs 0.02) when paired with 11L + WD=0.038. |
| **Longer warmdown** | #63 | small | warmdown=3600 (vs 3000). Marginal, regime-dependent. |

### Architecture

| Technique | PRs | Est. Impact | Description |
|-----------|-----|-------------|-------------|
| **10-11 layers** | #60, #63, #150, #155, #176, #179, #180 | 0.005-0.01 | Extra depth funded by int5/WD compression savings. 10L is well-proven, 11L is frontier (#179). |
| **SmearGate** | #65, #102, #135, #162, #164, #170, #180 | ~0.005 | Learned sigmoid gate blending each token embedding with previous token's. ~512 params. gate = sigmoid(w), output = gate*cur + (1-gate)*prev. |
| **BigramHash** | #135, #150, #162, #164, #180 | ~0.005 | 4096-bucket hash table (dim=128, projected to 512) for token-pair context. ~524K params. |
| **U-Net skip connections** | #65, #69, #156, #160 | small | Encoder-decoder structure with learnable skip weights. Already in baseline for some. |
| **Overtone spectral init** | #60, #155 | small | SVD power-law spectrum shaping for embedding init. |
| **Phase-transition resid_mix** | #60, #155 | small | Sigmoid-scheduled residual mixing initialization. |
| **2048/4096/8192 vocab** | #78, #92, #122, #123 | mixed | Bigger vocab = fewer layers to fit budget. #122 (2048 vocab, 8L) got 1.1585. Tradeoff unclear. |
| **Depth recurrence** | #91, #112, #148, #154, #167 | negative-to-neutral | 3 unique layers x 3 loops. Saves params massively but quality suffers. Best: 1.26-1.59. Not competitive yet. |
| **SwiGLU** | #81, #163 | neutral-negative | No clear win over relu^2 in this regime. #81 got 1.167 but that's explained by other techniques. |
| **Logit soft-capping** | #69, #156 | small | tanh(logits/cap)*cap with cap=30. Minor stability aid. |

### Export / Quantization (Lane B)

| Technique | PRs | Est. Impact | Description |
|-----------|-----|-------------|-------------|
| **STE Int6 QAT** | #63, #65, #69, #86, #89, #107, #128, #137, #150, #156 | ~0.002 gap reduction | Fake int6 per-row quantization during forward pass, straight-through gradient. Model learns to survive 6-bit. Gap drops from ~0.015 to ~0.002. Very widely adopted. |
| **Mixed int5/int6** | #180 | 1.86MB savings | Int5 [-16,15] for MLP (3 zero high bits -> zstd 1.88x), int6 for attention. Funds extra layer. Only #180 uses this. |
| **zstd-22** | most top PRs | ~0.5MB vs zlib | Better compression ratio than zlib-9. Standard practice. |
| **Grouped LZMA** | #160 | similar to zstd | Alternative compressor. Comparable results. |
| **fp16 tied embedding** | standard | ~0.01 | Never quantize the embedding tensor. Most quant-sensitive. Universal in top submissions. |
| **Late-K fp16 passthrough** | #99, #114 | small | Last 2 layers' key projections kept in fp16. |
| **Int6-in-Int8 containers** | #170 | compression trick | Store int6 values in int8 bytes — zstd compresses the restricted range ~35%. Better than bit-packing which destroys byte alignment. |
| **Post-training QAT** | #107 | marginal | 30s of STE fine-tuning after main training. Less effective than full STE QAT. |

### Eval Policy (Lane C)

| Technique | PRs | Est. Impact | Description |
|-----------|-----|-------------|-------------|
| **Sliding window stride=64** | #50, #65, #89, #102, #135, #156, #162, #180 | ~0.020-0.033 | Every token gets ~960 tokens context at seq1024. Most top submissions use 64. |
| **Sliding window stride=256** | #114, #164 | ~0.018 | #114 argued 256 slightly better than 64 (1.1574 vs 1.1579). Minority view. |
| **Sliding window stride=512** | #88 | ~0.015 | Coarser but much faster eval. |
| **LoRA TTT** | #77, #81, #152, #161, #175 | ~0.003 | Per-document rank-8 LoRA adaptation at eval time on Q/V + LM head. Batched, doc-isolated. Uses ~1/10 of eval budget. |
| **Doc-isolated eval** | #77 | ~0.011 | Reset context between documents. Significant standalone. |

---

## Pattern Analysis

### What separates 1.145 from 1.163

The top 3 submissions (#180, #179, #162) all share:
- Muon Weight Decay (0.02-0.04)
- SWA (checkpoint averaging during warmdown)
- Either 10-11 layers OR SmearGate+BigramHash (both add effective capacity)

The key differentiator of #180 (best at 1.1453):
- **Mixed int5/int6** saves 1.86MB → funds 10th layer
- **MuonWD=0.04** (higher than others' 0.02) → better quant compression
- **SWA every 50 steps** (more frequent than others' 200) → smoother average
- Stacks SmearGate + BigramHash on top of 10L

### Common stack of all submissions beating us

Every submission scoring < 1.163 uses at least 3 of these 5:
1. STE Int6 QAT (fake quant during training)
2. SmearGate or BigramHash (cheap bigram features)
3. NorMuon or Muon WD (optimizer improvement)
4. SWA (checkpoint averaging)
5. OrthoInit (weight initialization)

We use **zero** of these five.

### Techniques that didn't work

| Technique | PRs | Result |
|-----------|-----|--------|
| Depth recurrence | #91, #112, #148, #154, #167 | 1.26-1.59 BPB. Not competitive. Saves params but quality collapses. |
| BitNet b1.58 | #126, #139 | 1.20-1.75. Ternary weights too lossy for this regime. |
| SwiGLU | #81, #163 | No improvement over relu^2. |
| 8192 vocab | #92 | 1.194. Loses too many layers. |
| Mirror/looped recurrence | #84, #103 | Non-record. Not competitive. |
| Dispersion loss | #34 | Closed, no results. |

### AutoResearch / agent approaches

- **#66** (arjun-krishna1): Built an agent that read all PRs, bucketed by expected impact, and composed the "High" bucket. Got 1.1632 — similar to our manual result.
- **#85** (hydeh3r3): 92-experiment autoresearch pipeline. Pending eval, pre-quant 1.2156.
- **#128** (rsavitt): Claude Code agent. Got 1.1594.
- **#175** (anthony-maio): Claude Code generated. Stacks TTT LoRA + SOTA training. Pending eval.

---

## Gap Analysis: Us (1.163) vs Frontier (1.145)

Gap: ~0.018 BPB. Techniques we're missing:

| Technique | Est. Gain | Code Change | Difficulty |
|-----------|-----------|-------------|------------|
| STE Int6 QAT | 0.002 | Yes — fake quant in forward pass | Easy |
| SmearGate + BigramHash | 0.005-0.01 | Yes — architecture change | Medium |
| MuonWD 0.04 | 0.003 | Yes if not wired, env var if wired | Easy |
| SWA | 0.003 | Yes — checkpoint averaging | Easy |
| OrthoInit + muP | 0.003 | Yes — init change | Easy |
| NorMuon | 0.005 | Yes — optimizer swap | Medium |
| Mixed int5/int6 + 10L | 0.005 | Yes — export change | Medium |
| Stride=64 (vs our 256) | 0.002 | Env var | Trivial |
| FA3 | ~0.002 (via more steps) | Yes — attention kernel | Easy |
| **Total estimated** | **~0.025-0.030** | | |

All require code changes to train_gpt.py except stride.

## Priority Stack for Next Session

1. **Env-var-only**: stride=64 — test immediately
2. **Easy code wins**: STE QAT, SWA, OrthoInit, MuonWD wiring — low-risk, well-documented by 10+ PRs
3. **Medium code wins**: NorMuon, SmearGate+BigramHash — bigger lift, need careful integration
4. **Advanced**: Mixed int5/int6 + 10L, MTP, LoRA TTT — highest risk/reward
5. **Needs investigation**: Whether these compose additively or have diminishing returns when stacked
