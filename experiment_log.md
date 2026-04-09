# Parameter Golf — Full Experiment Log

**Pod:** RTX 4000 Ada ($0.20/hr) on RunPod
**Baseline:** Cosine LR, 240 steps (600s wallclock cap), 9L/512D model = **1.6117 BPB**
**Competition SOTA:** ~1.1147 BPB (8xH100)
**Best Result:** 1.5207 BPB (H1: PureDecoder+GQA2+Untied+MatLR0.11+Parallel+SiLU²) = **-0.091 vs baseline**
**Artifact Size:** ~14MB int8+zlib (2MB headroom under 16MB budget)
**Total Experiments:** 131

---

## Phase 1: Novel Regularization Techniques (PR #1380)
*Ran on cosine LR baseline, 240 steps*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
| 1 | Z-loss 1e-4 | PaLM-style log(Z)² regularizer on logits | 1.6282 | +0.017 | Worse |
| 2 | Logit Penalty 1e-5 | L2 penalty on logit magnitudes | 1.6117 | 0.000 | Neutral |
| 3 | Token Dropout 5% | Randomly drop 5% of tokens from loss | 1.6145 | +0.003 | Worse |
| 4 | Embed Mixup 0.1 | Interpolate embedding vectors with random pairs | 1.6157 | +0.004 | Worse |
| 5 | Cosine 2-cycle | Two cosine decay cycles instead of one | 1.7383 | +0.127 | Much worse |
| 6 | Combo (Z+Logit+Drop) | Stack of Z-loss + Logit Penalty + Token Drop | 1.6171 | +0.005 | Worse |

**Conclusion:** All novel regularization techniques hurt or were neutral. Standard cross-entropy is already well-tuned.

---

## Phase 2: Nature-Inspired v1
*Ran on cosine LR baseline, 240 steps*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
| N1 | Controlled Burn | Prune smallest 5% of weights every 50 steps | 1.6109 | -0.001 | Neutral |
| N2 | **Tidal LR** | **Golden ratio warmup (38.2% warmup, 61.8% cosine decay)** | **1.5906** | **-0.021** | **Winner** |
| N3 | Head Diversity | Quorum sensing — penalize cosine similarity between attention head projections | 1.6100 | -0.002 | Slight help |
| N4 | Weight Perturbation | Add random noise to weights every 100 steps (viral mutation) | 1.6100 | -0.002 | Slight help |
| N5 | Golden Ratio min_lr | Set min LR fraction to 0.382 (golden ratio) | 4.1069 | — | BROKEN |
| N6 | Metamorphosis | Decay dropout from 10% to 0% over first half | 4.1069 | — | BROKEN |

**Conclusion:** Tidal LR is a clear winner. Extended warmup (38.2% of training) before cosine decay significantly helps. N5/N6 broken due to indentation bug from patching.

---

## Phase 3: Unconventional Architecture (ALL BROKEN)
*torch.compile incompatible — all crashed or produced no training*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
| U1 | Layer Dropout | Randomly skip layers with 20% probability | — | — | CRASH: torch.compile |
| U2 | Weight Sharing | Share weights between layers in groups of 3 | — | — | CRASH: AttributeError |
| U3 | Progressive Growing | Start with 3 layers, add one every 60 steps | 4.1069 | — | BROKEN: no training |
| U4 | Attention Recycling | Run attention twice per layer | — | — | CRASH: torch.compile |
| U5 | Mirror Training | Reverse 50% of input sequences | — | — | Not reached |
| U6 | Head Pruning | Start 16 heads, prune to 8 at step 100 | — | — | Not reached |
| U7 | 12L Asymmetric | 12 layers, 448D, 1 encoder layer | — | — | Not reached |
| U8 | Wide MQA | 640D, 7 layers, 1 KV head | — | — | Not reached |
| U9 | Softcap+RoPE combo | Softcap 15, RoPE base 1000 | — | — | Not reached |
| U10 | Aggressive LR | embed=0.8, matrix=0.06, muon=0.98 | — | — | Not reached |

**Conclusion:** Dynamic model structure changes are incompatible with `torch.compile(fullgraph=True)`. Complete waste of compute. Lesson: only modify training dynamics, not model architecture.

---

## Phase 4: Nature-Inspired LR Schedules (Wave 1 — NF batch)
*All ran independently, 240 steps*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
| NF1 | **Breathing LR** | **4-7-8 pattern: 21% warmup, 37% steady, 42% decay** | **1.5961** | **-0.016** | **Good** |
| NF2 | Whale Dive LR | 3 dive cycles with sharp recoveries | 1.6487 | +0.037 | Worse |
| NF3 | Circadian LR | Day/night sine oscillation overlaid on cosine | 1.6141 | +0.002 | Neutral |
| NF4 | Cosmological Cooling | lr ~ 1/sqrt(1 + alpha*step), Big Bang cooling law | 1.6129 | +0.001 | Neutral |
| NF5 | Synaptic Scaling | Normalize weight norms back to initial values each step | 1.7239 | +0.112 | Very bad |
| NF6 | Mutation Decay | Exponentially decaying random weight noise | 1.6132 | +0.002 | Neutral |

**Conclusion:** Breathing LR confirms the insight: extended high-LR phase helps. Whale Dive's cycling hurts. Synaptic scaling catastrophically bad.

---

## Phase 5: Tidal LR Combinations (NF batch continued)
*Tidal LR + various techniques, 240 steps*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
| NF7 | Tidal + Controlled Burn | Tidal LR + prune 5% every 50 steps | 1.5915 | -0.020 | ~Same as Tidal |
| NF8 | **Tidal + Head Diversity** | **Tidal + quorum sensing head diversity loss (1e-4)** | **1.5867** | **-0.025** | **Better than Tidal** |
| NF9 | Tidal + Mutation Decay | Tidal + exponentially decaying noise (0.001) | 1.5902 | -0.022 | ~Same |
| NF10 | Tidal + Weight Perturb | Tidal + periodic weight perturbation (1e-4) | 1.5901 | -0.022 | ~Same |

**Conclusion:** Head Diversity stacks well with Tidal (+0.004 on top). Other tricks are neutral on top of Tidal.

---

## Phase 6: Architecture Sweeps with Tidal (NF batch continued)
*Tidal LR + architecture parameter changes, 240 steps*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
| NF11 | Deeper 12L/448D | 12 layers, 448 dim (fewer steps: 197) | 1.6620 | +0.050 | Worse |
| NF12 | Wider 640D/7L/MQA | 640 dim, 7 layers, 1 KV head (234 steps) | 1.6013 | -0.010 | Slightly better |
| NF13 | **Aggressive LR** | **embed_lr=0.8, matrix_lr=0.06** | **1.5877** | **-0.024** | **Good** |
| NF14 | **Softcap 20 + RoPE 5000** | **Logit softcap 30→20, RoPE base 10000→5000** | **1.5765** | **-0.035** | **Breakthrough** |

**Conclusion:** Softcap 20 + RoPE 5000 is the biggest single finding. Deeper/wider models hurt because they get fewer steps in the 600s wallclock cap.

---

## Phase 7: 3000-Step Verification (INVALID)
*Wallclock cap still limits to 240 steps regardless of ITERATIONS setting*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
| NF15 | Tidal @ 3000 steps | Tidal LR with ITERATIONS=3000 | 1.6504 | — | INVALID: only 240 steps ran, LR at 8% progress |
| NF16 | Cosine @ 3000 steps | Cosine LR with ITERATIONS=3000 | 1.6271 | — | INVALID: same issue |

**Conclusion:** Setting ITERATIONS=3000 with 600s wallclock makes the LR schedule think it's at 8% progress. Tidal (38% warmup) hasn't peaked yet. Results meaningless.

---

## Phase 8: Stacking All Winners (W0 batch)
*Combining best techniques, 240 steps*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
| W0a | Tidal+SC20+RoPE5k+HeadDiv | Stack top 3 findings | 1.5811 | -0.031 | Good |
| W0b | Tidal+SC20+RoPE5k+AggrLR | + embed=0.8, matrix=0.06 | 1.5774 | -0.034 | Good |
| W0c | Tidal+SC15+RoPE5k | Tighter softcap (15) | 1.5793 | -0.032 | Slightly worse than SC20 |
| W0d | Tidal+SC20+RoPE3k | Tighter RoPE (3000) | 1.5845 | -0.028 | Worse than RoPE5k |
| W0e | **EVERYTHING** | **Tidal+SC20+RoPE5k+HeadDiv+AggrLR** | **1.5744** | **-0.037** | **Best result** |

**Conclusion:** Stacking all winners gives incremental improvement. Sweet spots confirmed: Softcap=20 (not 15), RoPE=5000 (not 3000). AggressiveLR and HeadDiv each add ~0.002.

---

## Phase 9: Novel Nature Gradient Tricks (W1-W10 batch)
*All catastrophically bad*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
| W1 | Canopy Light | Scale gradients by layer depth (top=1.0x, bottom=0.3x) | 1.8587 | +0.247 | Terrible |
| W2 | Predator-Prey LR | Lotka-Volterra: LR & weight decay oscillate in opposition | 1.7981 | +0.186 | Terrible |
| W3 | Punctuated Equilibrium | Low LR + gaussian bursts at 25/50/75% progress | 1.8581 | +0.246 | Terrible |
| W4 | Mycorrhizal Gradient | Blend gradients between layers separated by 3 | 1.6223 | +0.011 | Worse |
| W5 | Thermal Vent | Random 2x-3x gradient boost to 2 random layers per step | 1.6472 | +0.036 | Worse |
| W6 | Canopy + Mycorrhizal | Rainforest combo | 1.8667 | +0.255 | Terrible |
| W7 | Tidal + Canopy | Tidal LR + depth gradient scaling | 1.5954 | -0.016 | Worse than Tidal alone |
| W8 | Tidal + Mycorrhizal | Tidal + gradient sharing | 1.6009 | -0.011 | Worse than Tidal alone |
| W9 | Tidal + Thermal Vent | Tidal + random layer boosting | 1.6124 | +0.001 | Neutral |
| W10 | Tidal + Predator-Prey WD | Tidal + oscillating weight decay | 1.6371 | +0.025 | Worse |

**Conclusion:** ALL gradient-level nature tricks are harmful. Don't mess with gradient flow — the optimizer (Muon) is already well-tuned. Novel LR schedules work; novel gradient modifications don't.

---

## Phase 10: Hyperparameter Grid (Wave 4 — partially run, killed for Wave 5)
*Only E1 completed before being killed in favor of architecture experiments*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
| E13 | Softcap 18 + RoPE 5k | Slightly tighter softcap | 1.5770 | -0.035 | ~Same as SC20 |
| E1 | Label Smoothing 0.1 | On full best config | 1.6338 | +0.059 | Worse |
| E2-E25 | Various | Killed in favor of Wave 5 | — | — | Not run |

**Conclusion:** Label smoothing hurts on best config. Remaining experiments skipped — architecture experiments (Wave 5) were higher priority.

---

## Phase 11: Architecture Experiments (Wave 5)
*Static arch changes: activation swaps, parallel blocks, sandwich norm, combos, scale*

### Activation Functions (on Tidal+SC20+RoPE5k, no HeadDiv/AggrLR)
| # | Experiment | Description | BPB | Steps | Delta | Status |
|---|-----------|-------------|-----|-------|-------|--------|
| A1 | LeakyReLU² | Leaky negative slope 0.01, then square | 1.5820 | 240 | -0.030 | Worse than relu² |
| A2 | LeakyReLU² (cosine) | On cosine baseline | 1.6132 | 239 | +0.002 | Neutral |
| A3 | SwiGLU | Llama/Mistral standard activation | 1.5798 | 226 | -0.032 | Slower (fewer steps) |
| A4 | SwiGLU (cosine) | On cosine baseline | 1.6053 | 226 | -0.006 | Slight help but slow |
| A5 | GELU² | GELU then square | 1.5779 | 240 | -0.034 | Competitive |
| A6 | **SiLU²** | **SiLU then square** | **1.5743** | **240** | **-0.037** | **Ties prev best** |

### Block Structure
| # | Experiment | Description | BPB | Steps | Delta | Status |
|---|-----------|-------------|-----|-------|-------|--------|
| A7 | **Parallel blocks** | **PaLM-style: attn+MLP in parallel** | **1.5600** | **260** | **-0.052** | **Big win + faster** |
| A8 | Parallel (cosine) | On cosine baseline | 1.5868 | 259 | -0.025 | Confirms parallel helps |
| A9 | Sandwich Norm | Extra RMSNorm after attention | 1.5983 | 239 | -0.013 | Minor help |

### Combos
| # | Experiment | Description | BPB | Steps | Delta | Status |
|---|-----------|-------------|-----|-------|-------|--------|
| A10 | LeakyReLU²+Parallel | Combine activation + structure | 1.5661 | 259 | -0.046 | Worse than plain parallel |
| A11 | SwiGLU+Parallel | — | 1.5648 | 245 | -0.047 | Worse than plain parallel |
| A12 | LeakyReLU²+Best | LeakyReLU² + HeadDiv + AggrLR | 1.5802 | 240 | -0.031 | Activation hurts |
| A13 | SwiGLU+Best | SwiGLU + HeadDiv + AggrLR | 1.5848 | 226 | -0.027 | Activation + slow |
| A14 | **EVERYTHING** | **Parallel+LeakyReLU²+HeadDiv+AggrLR** | **1.5586** | **259** | **-0.053** | **New best** |

### Scale with Architecture
| # | Experiment | Description | BPB | Steps | Delta | Status |
|---|-----------|-------------|-----|-------|-------|--------|
| A15 | LeakyReLU²+MLP3x | Wider MLP hidden layer | 1.6098 | 216 | -0.002 | Too slow |
| A16 | SwiGLU wide 576D/8L | Wider model dim | 1.6090 | 206 | -0.003 | Too slow |
| A17 | LeakyReLU²+GQA(2) | 2 KV heads | 1.5761 | 248 | -0.036 | Decent, faster |
| A18 | LeakyReLU²+MQA(1) | 1 KV head | 1.5740 | 253 | -0.038 | Good, fastest |

**Conclusion:** Parallel blocks are the biggest single finding (-0.052). They're both faster (2315ms/step → 260 steps vs 240) AND better quality. Activation swaps are noise (±0.003). MQA is surprisingly competitive. Wider/deeper models lose to step count.

---

## Phase 12: Focused Parallel Optimization (Wave 6)
*Building on parallel blocks win, 5 experiments*

| # | Experiment | Description | BPB | Steps | Delta | Status |
|---|-----------|-------------|-----|-------|-------|--------|
| B4 | **Parallel+SiLU²+HD+AggrLR** | **Best activation + parallel** | **1.5527** | **259** | **-0.059** | **New best** |
| B5 | Parallel+QK2.0+HD+AggrLR | Higher QK gain | 1.5535 | 260 | -0.058 | Close second |
| B1 | Parallel+relu²+HD+AggrLR | Clean combo (no LeakyReLU²) | 1.5561 | 260 | -0.056 | Good |
| B3 | Parallel+10L+HD+AggrLR | Stack extras on 10L | 1.5885 | 234 | -0.023 | 10L hurts |
| B2 | Parallel+10L | Use speed budget for extra layer | 1.5979 | 233 | -0.014 | 10L hurts |

**Conclusion:** SiLU² beats relu² with parallel (+0.003). 10 layers lose too many steps. QK gain 2.0 nearly ties best — attention strength matters slightly with parallel.

---

## Phase 13: Hyperparameter Re-tune for Parallel (Wave 7)
*Re-tuning around Parallel+SiLU² architecture, 14 experiments*

### MQA + Parallel
| # | Experiment | Description | BPB | Steps | Delta vs B4 | Status |
|---|-----------|-------------|-----|-------|-------------|--------|
| C1 | MQA+Parallel | 1 KV head + parallel | 1.5588 | 275 | +0.006 | Faster but worse |
| C2 | MQA+Par+SiLU²+Best | Full stack with MQA | 1.5533 | 274 | +0.001 | Close but no win |
| C3 | MQA+16 query heads | More query capacity | 1.6055 | 266 | +0.053 | Broken/terrible |

### Softcap Re-tune
| # | Experiment | Description | BPB | Steps | Delta vs B4 | Status |
|---|-----------|-------------|-----|-------|-------------|--------|
| C4 | SC15 | Tighter softcap | 1.5519 | 259 | -0.001 | Slight help |
| C5 | SC25 | Looser softcap | 1.5602 | 259 | +0.008 | Worse |
| C6 | SC18 | Mid softcap | 1.5519 | 259 | -0.001 | Tied with SC15 |

### RoPE Re-tune
| # | Experiment | Description | BPB | Steps | Delta vs B4 | Status |
|---|-----------|-------------|-----|-------|-------------|--------|
| C7 | RoPE 3000 | Tighter positions | 1.5553 | 259 | +0.003 | Worse |
| C8 | RoPE 7500 | — | 1.5544 | 259 | +0.002 | Worse |

### LR Re-tune
| # | Experiment | Description | BPB | Steps | Delta vs B4 | Status |
|---|-----------|-------------|-----|-------|-------------|--------|
| C9 | Embed LR 1.0 | More aggressive embed | 1.5552 | 259 | +0.003 | Worse |
| C10 | **Matrix LR 0.08** | **More aggressive matrix** | **1.5501** | **259** | **-0.003** | **New best** |
| C11 | Both LR aggressive | Embed 1.0 + Matrix 0.08 | 1.5541 | 259 | +0.001 | Embed LR hurts |

### Schedule & HeadDiv
| # | Experiment | Description | BPB | Steps | Delta vs B4 | Status |
|---|-----------|-------------|-----|-------|-------------|--------|
| C12 | Breathing LR | Alt schedule | 1.5686 | 259 | +0.016 | Tidal still better |
| C13 | HeadDiv 1e-3 | Stronger diversity | 1.5551 | 259 | +0.002 | Too strong |
| C14 | No HeadDiv | Ablation | 1.5542 | 259 | +0.002 | HD barely matters |

**Conclusion:** Matrix LR 0.08 is a real win. SC15/18 might help but within noise. RoPE 5000 still optimal. HeadDiv barely matters with parallel. MQA branch dead. Breathing worse than Tidal.

---

## Phase 14: Asymmetric Splits + Stacking (Wave 8 — in progress)
*Asymmetric encoder/decoder ratios + stacking Wave 7 wins, 10 experiments*

| # | Experiment | Description | BPB | Delta | Status |
|---|-----------|-------------|-----|-------|--------|
### Stacking
| # | Experiment | Description | BPB | Steps | Delta vs C10 | Status |
|---|-----------|-------------|-----|-------|-------------|--------|
| D1 | MatLR0.08+SC18 | Stack two wins | 1.5501 | 259 | 0.000 | Tied |
| D2 | MatLR0.08+SC15 | Stack two wins | 1.5493 | 259 | -0.001 | Slight help |
| D3 | MatLR0.08+NoHD | Simplify config | 1.5541 | 259 | +0.004 | HD still helps |

### Asymmetric Encoder/Decoder Splits
| # | Experiment | Description | BPB | Steps | Delta vs baseline | Status |
|---|-----------|-------------|-----|-------|-------------------|--------|
| D6 | **Asym 1/8** | **1 encoder, 8 decoder** | **1.5377** | **266** | **-0.074** | **New best** |
| D5 | Asym 2/7 | 2 encoder, 7 decoder | 1.5412 | 263 | -0.071 | Great |
| D4 | Asym 3/6 | 3 encoder, 6 decoder | 1.5439 | 262 | -0.068 | Good |
| D7 | Asym 5/4 | Control: more encoder | 1.5568 | 257 | -0.055 | Worse |

### Stacking Asymmetric + Softcap
| # | Experiment | Description | BPB | Steps | Delta vs baseline | Status |
|---|-----------|-------------|-----|-------|-------------------|--------|
| D9 | Asym 2/7+SC18 | Stack best | 1.5440 | 263 | -0.068 | SC18 didn't help |
| D8 | Asym 3/6+SC18 | Stack best | 1.5461 | 261 | -0.066 | SC18 didn't help |
| D10 | C10 Rerun | Confirm 1.5501 | 1.5515 | 259 | -0.060 | ~0.001 noise |

**Conclusion:** Asymmetric splits are the biggest Wave 8 finding. Monotonic: fewer encoder layers = better BPB AND faster. 1/8 split gives -0.074 total improvement. SC18 doesn't compound with asymmetric. Measurement noise is ~0.001-0.002.

---

## Phase 15: Optimize 1/8 Split (Wave 9 — COMPLETE)
*Softcap, LR, activation, QK tuning on 1/8 split, 10 experiments*

| # | Experiment | Description | BPB | Steps | Delta vs baseline | Status |
|---|-----------|-------------|-----|-------|-------------------|--------|
| E1 | SC15 on 1/8 | Softcap 15 (was 20) | 1.5387 | 265 | -0.073 | Good |
| E2 | SC18 on 1/8 | Softcap 18 | 1.5398 | 265 | -0.072 | Slightly worse |
| E3 | SC12 on 1/8 | Softcap 12 | 1.5378 | 265 | -0.074 | ~Same as D6 |
| E4 | MatLR0.10 on 1/8 | Matrix LR 0.10 (was 0.08) | 1.5392 | 265 | -0.073 | Good alone |
| E5 | MatLR0.12 on 1/8 | Matrix LR 0.12 | 1.5391 | 265 | -0.073 | Good alone |
| E6 | ReLU² on 1/8 | Swap SiLU² → ReLU² | 1.5413 | 267 | -0.070 | SiLU² better |
| E7 | QK Gain 2.0 on 1/8 | Larger QK init | 1.5400 | 265 | -0.072 | No help |
| **E8** | **SC15+MatLR0.10** | **Stack: SC15 + MatLR0.10** | **1.5354** | **265** | **-0.076** | **👑 BEST** |
| E9 | SC15+QK2 | Stack: SC15 + QK Gain 2.0 | 1.5374 | 265 | -0.074 | Worse than E8 |
| E10 | D5 Rerun | Confidence rerun (D5 base) | 1.5395 | 265 | -0.072 | Confirms noise |

**Conclusion:** SC15+MatLR0.10 stacking gives new best: 1.5354 (-0.076). Individual effects are small (~0.001) but compound well. QK gain doesn't help. ReLU² confirmed worse than SiLU² on asymmetric. Measurement noise ~0.002. Artifact = 12.9MB int8+zlib (3.1MB headroom).

---

## Phase 16: Fine-grained Tuning (Wave 10 — COMPLETE)
*Softcap step-of-1 sweep, LR fine-tune, GQA, Tidal warmup on 1/8, base: SC15+MatLR0.10*

| # | Experiment | Description | BPB | Steps | Delta vs baseline | Status |
|---|-----------|-------------|-----|-------|-------------------|--------|
| F1 | SC13+MatLR0.10 | Softcap 13 | 1.5376 | 265 | -0.074 | Worse than E8 |
| F2 | SC14+MatLR0.10 | Softcap 14 | 1.5363 | 265 | -0.075 | Neutral |
| F3 | SC16+MatLR0.10 | Softcap 16 | 1.5359 | 266 | -0.076 | Neutral |
| F4 | SC17+MatLR0.10 | Softcap 17 | 1.5356 | 266 | -0.076 | Neutral |
| F5 | SC15+MatLR0.09 | Matrix LR 0.09 | 1.5377 | 266 | -0.074 | Worse |
| F6 | **SC15+MatLR0.11** | **Matrix LR 0.11** | **1.5341** | **266** | **-0.078** | **New best** |
| F7 | **GQA 2KV** | **2 KV heads (grouped query attention)** | **1.5329** | **276** | **-0.079** | **New best + faster** |
| F8 | Tidal 30% warmup | Shorter warmup (30% vs 38.2%) | 1.5355 | 265 | -0.076 | Neutral |
| F9 | QK Gain 2.0 stack | QK init 2.0 on E8 | 1.5401 | 265 | -0.072 | Worse |
| F10 | E8 Rerun | Confidence check | 1.5365 | 265 | -0.075 | Variance ~0.001 |

**Conclusion:** Two wins — MatLR=0.11 and GQA with 2 KV heads. GQA is a double win: faster (2181ms/step → 276 steps vs 265) AND better quality. Softcap sweep confirms SC15 optimal but SC13-17 all within noise. Tidal 30% warmup neutral. QK gain confirmed dead.

---

## Phase 17: Novel Ideas (Wave 11 — COMPLETE)
*Trimmed to 2 key experiments: untied embeddings, WD schedule*

| # | Experiment | Description | BPB | Steps | Delta vs baseline | Status |
|---|-----------|-------------|-----|-------|-------------------|--------|
| G1 | **Untied Embeddings** | **Separate input/output embeddings (TIE_EMBEDDINGS=0)** | **1.5211** | **266** | **-0.091** | **Huge win!** |
| G3 | WD Schedule 0.01 | Ramp weight decay 0→0.01 | 1.5603 | 265 | -0.051 | Much worse |

**Conclusion:** Untied embeddings is the biggest single improvement since parallel blocks (-0.012 on top of F7). WD schedule is catastrophically harmful — do not use. Artifact size with untied embeddings: ~14MB (still under 16MB budget).

---

## Phase 18: Aggressive Experiments (Wave 12 — COMPLETE)
*Pure decoder, wider MLP, WD schedule, stacking — all with GQA2+Untied+MatLR0.11*

| # | Experiment | Description | BPB | Steps | Delta vs baseline | Status |
|---|-----------|-------------|-----|-------|-------------------|--------|
| **H1** | **Pure Decoder** | **ENCODER_LAYERS=0, all 9 layers as decoder** | **1.5207** | **276** | **-0.091** | **Best overall** |
| H2 | MLP 3x width | MLP hidden 1024 (vs 682) | 1.5481 | 244 | -0.064 | Too slow |
| H3 | MLP 3x + Pure Decoder | 3x MLP + ENCODER_LAYERS=0 | 1.5474 | 244 | -0.064 | Too slow |
| H4 | WD Schedule 0.04 | Ramp weight decay 0→0.04 | 1.6569 | 276 | -0.045 | Catastrophic |
| H5 | All Stacked | Pure decoder + 3x MLP + WD 0.04 | 1.6974 | 244 | +0.086 | Terrible |
| H6 | Best Rerun | H1 config confidence run | 1.5214 | 276 | -0.090 | Confirms H1 |

**Conclusion:** Pure decoder (0 encoder layers) gives a slight edge over 1/8 split. 3x MLP is too slow on RTX 4000 Ada (2462ms/step → only 244 steps, losing quality). WD schedule catastrophically bad in all forms. H6 confirms H1 is reproducible (1.5207 vs 1.5214, noise ~0.0007).

---

## Summary of Key Findings (131 experiments)

### What works (ranked by impact):
1. **Parallel blocks (PaLM-style)** — -0.052 alone, faster + better quality
2. **Untied embeddings** — -0.012 on top of best config, biggest late-stage win
3. **Pure decoder (ENCODER_LAYERS=0)** — monotonically better with fewer encoder layers
4. **GQA with 2 KV heads** — faster (2181ms vs 2264ms/step) AND better quality
5. **SiLU² activation** — best activation with parallel blocks
6. **Logit Softcap 15** (default 30) — tighter logit distribution
7. **Matrix LR 0.11** (default 0.06) — more aggressive matrix learning rate
8. **Tidal LR (golden ratio warmup)** — -0.021, 38.2% warmup before cosine decay
9. **RoPE base 5000** (default 10000) — sharper positional attention
10. **Head Diversity loss** (1e-4) — marginal but real

### What doesn't work:
- All gradient-level tricks (Canopy, Mycorrhizal, Thermal Vent, Predator-Prey)
- All regularization (Z-loss, token dropout, embed mixup, synaptic scaling, label smoothing)
- LR cycling (Whale Dive, Cosine 2-cycle, Breathing with parallel)
- Weight decay schedule (catastrophically bad in all forms)
- 3x MLP width (too slow on single GPU — loses steps)
- Wider/deeper models at 240-step wallclock (fewer steps kills gains)
- Weight perturbation/mutation (neutral at best)
- 10 layers (even with parallel speed savings, too slow)
- MQA with 1 KV head (faster but loses quality vs GQA with 2)
- More encoder layers (5/4 worse than 4/5 worse than 1/8 worse than 0/9)
- SC18 on asymmetric splits (doesn't compound)
- Embed LR 1.0 (too aggressive)
- QK Gain 2.0 (no help on asymmetric/pure decoder)
- ReLU² on asymmetric (SiLU² better)

### What was broken (wasted compute):
- N5, N6: Indentation bug from patching — train_loss=0.0000
- U1-U10: torch.compile incompatible architecture changes
- E1-E12 (first run): $BEST variable not expanding as env vars in bash
- NF15-NF16: 3000-step verification invalid due to wallclock cap

### Best Config (H1):
```bash
TIDAL_LR=1 LOGIT_SOFTCAP=15.0 ROPE_BASE=5000 PARALLEL_BLOCK=1 MLP_ACT=silu2 HEAD_DIVERSITY=1e-4 EMBED_LR=0.8 MATRIX_LR=0.11 ENCODER_LAYERS=0 NUM_KV_HEADS=2 TIE_EMBEDDINGS=0
```
**Result: 1.5207 BPB (-0.091 vs 1.6117 baseline)**
**Artifact: ~14MB int8+zlib (2MB headroom under 16MB)**
**Hardware: Single RTX 4000 Ada, 276 steps in 600s wallclock**

### Progression:
```
Baseline (cosine):     1.6117
+ Tidal LR:            1.5906  (-0.021)
+ Head Diversity:      1.5867  (-0.025)
+ SC20 + RoPE 5k:      1.5744  (-0.037)
+ Parallel blocks:     1.5600  (-0.052)
+ SiLU²:               1.5527  (-0.059)
+ Matrix LR 0.08:      1.5501  (-0.062)
+ Asymmetric 1/8:      1.5377  (-0.074)
+ SC15 + MatLR 0.10:   1.5354  (-0.076)
+ MatLR 0.11:          1.5341  (-0.078)
+ GQA 2KV:             1.5329  (-0.079)
+ Untied embeddings:   1.5211  (-0.091)
+ Pure decoder (0 enc): 1.5207  (-0.091)
```

### Gap to Competition:
- Our best: 1.5207 (single RTX 4000 Ada, 276 steps)
- Competition baseline: 1.2244 (8xH100, ~3500 steps)
- Competition SOTA: 1.1147 (8xH100, int6 QAT + TTT + XSA + 11 layers)
- Key techniques we lack: int6 QAT, SWA/EMA, sliding window eval, 10-11 layers, BigramHash
