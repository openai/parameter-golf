# Parameter Golf — Our Results & Strategy
_Last updated: 2026-03-20 4:04pm EDT_

## Leaderboard Context (updated 2026-03-20 4:11pm EDT)
| # | bpb | Author | Key Technique | PR |
|---|-----|--------|---------------|-----|
| **1** | **1.0539** | **ibarrajo** | **Paid Prefix exploit — 10% val tokens stored as LZMA blob** | **#262** ⚠️ |
| 2 | 1.1318 | jfprincz | Int6+SWA+11L+SmearGate+BigramHash (REAL SOTA BASE) | #198 |
| 3 | 1.1455 | stukenov | Int5 MLP + Int6 attn, full-model SGD TTT | #264 |
| 4 | 1.1719 | KitKatMarty | WWW architecture (ternary VQ) | #211 |
| 5 | 1.1748 | notapplica | Muon WD + spectral embed + 10L | — |

## Our 4090 Results (340 steps, ~10 min)

### Completed Code-Patch Experiments (aurora_autoresearch)
| Experiment | int8+zlib bpb | sliding bpb | TTT LoRA bpb | vs baseline | Status |
|---|---|---|---|---|---|
| **baseline** | 1.6353 | — | — | — | reference |
| **qk_gain_init_12** | **1.6133** | — | **1.6031** | **-0.022** ✅ | done |
| fp16_embed_export | 1.6184 | — | 1.6070 | -0.017 ✅ | done |
| spectral_embed_init | 1.6219 | — | 1.6123 | -0.013 ✅ | done |
| **sliding_window_eval** | 1.6144 | **1.5842** | — | **-0.051** ✅ | done |
| muon_weight_decay_002 | 1.7758 | — | — | +0.140 ❌ | done (WORSE) |
| lr_tuning_032 | 1.6474 | — | 1.6352 | -0.001 | done |
| tied_embed_lr_010 | 1.6397 | — | 1.6404 | worse | done |
| warmdown_2500 | 1.8515 | — | 1.8227 | way worse | done |

### Completed Env-Var Experiments (smart_autoresearch)
| Experiment | int8+zlib bpb | Status |
|---|---|---|
| baseline | 1.6353 | done |
| NUM_LAYERS=10 | 1.6964 | worse (needs more steps) |
| MATRIX_LR=0.06 + WARMDOWN=3600 | 1.9132 | way worse |
| MATRIX_LR=0.032 + SCALAR_LR=0.032 | 1.6756 | worse |
| MATRIX_LR=0.02 + SCALAR_LR=0.02 | 1.8430 | way worse |
| WARMDOWN=3600 | 2.0771 | terrible |
| 10L + low LR | 1.9128 | worse |
| 10L + high LR + WARMDOWN | 1.9889 | worse |
| SEQ_LEN 2048 | crash (OOM) | — |
| SEQ_LEN 4096 | crash (OOM) | — |

### Running Now (aurora_autoresearch round 2)
| Experiment | Description | Status |
|---|---|---|
| sliding_window_eval | Eval-only stride=64 | running |
| muon_weight_decay_002 | Muon WD=0.02 | queued |
| stack_eval_tricks | sliding + FP16 embed + spectral | queued |
| stack_full_v1 | above + muon WD | queued |
| stack_full_v2 | above + tied_embed_lr=0.10 | queued |

### Earlier Manual Tests
| Config | Raw bpb | Notes |
|---|---|---|
| Sliding window eval (manual) | 1.5879 | eval-only, free gain |
| Seq len 2048 (manual) | 1.6372 | barely moved |
| 10-layer notapplica config | 2.0866 | needs way more steps |

## Key Insights

### What works on 4090 (~340 steps)
- **QK gain init 1.2** — best single change (-0.022 bpb)
- **FP16 embed export** — keeps embeddings in FP16 during int8 quant (-0.017)
- **Spectral embed init** — SVD power-law shaping (-0.013)
- **Sliding window eval** — free eval-only gain (~0.03 raw bpb)

### What doesn't work on 4090
- **10 layers** — needs 10k+ steps to converge, only get 340
- **Longer sequences (2048/4096)** — OOM on 4090 24GB
- **Higher warmdown iters** — 3600 too aggressive for 340 steps
- **Aggressive LR changes** — either way too high or too low for short runs

### 4090 vs H100 Transfer Gap
- 4090: ~340 steps in 10 min (~1.7s/step, single GPU)
- H100 x8: ~13,000+ steps in 10 min (distributed training)
- **Architectural improvements transfer** (init, eval tricks, quant strategies)
- **Hyperparameters DON'T transfer** (LR schedules, warmdown timing need re-tuning)
- 10-layer models: terrible on 4090, #1 on H100 (just needs more steps)

## Competitive Intelligence (from PR #198 + #211 analysis, 2026-03-20)

### PR #198 — REAL SOTA (1.1318 bpb) by jfprincz
- **11 layers**, 512d, MLP 3x, SmearGate, BigramHash, OrthoInit
- **Int6 quantization + zstd-22** compression
- **WD=0.04**, FA3 (Flash Attention 3), SWA
- relu² activation
- NTK RoPE
- This is THE config everyone is forking from

### PR #211 — WWW (#1 leaderboard, 1.1719) by KitKatMarty/dubthecat
- **Wavelet Weighted Widenet** — 12-layer ternary MLP transformer
- **VQ compression (~1 bit/param)** — very different approach
- Ternary weights = massive parameter count in 16MB budget

### Emerging techniques from PR #198 forks:
- **Int5 for MLP, Int6 for attention** — mixed precision saves ~1.8MB, funds a 12th layer (alertcat)
- **RoPE base 50K** — smoother position interpolation at seq2048
- **LAWA-EMA** replacing periodic SWA — continuous exponential moving average (machdragon)
- **Context-length curriculum** — seq1024 early (60% more steps), seq2048 late
- **Full-model SGD TTT** — 1 epoch lr=3e-4 on val data (not just LoRA)
- **Sigmoid gate after attention** — elementwise gate, zero-init (mattqlf, just 3 lines)
- **12 layers via Int5 budget savings** — trade quant precision for depth

### Key Competitive Insight
Everyone serious is forking PR #198 and stacking improvements. We should do the same.
Our current approach (tweaking baseline env vars) is in a different league.
**To be competitive: start from PR #198's train_gpt.py as our base.**

## Unexplored High-Value Techniques
1. **Fork PR #198 as new baseline** — 11L, Int6+zstd, SmearGate, BigramHash, WD=0.04, FA3
2. **Int6 QAT** — we built `train_gpt_int6_swa.py` (1504 lines) but haven't tested
3. **Add sigmoid attention gate** — 3 lines, potential free win on top of #198
4. **LAWA-EMA** — replace SWA with continuous EMA (machdragon approach)
5. **Mixed Int5/Int6** — Int5 MLP + Int6 attention, fund 12th layer
6. **Custom tokenizer** — greedy+Capital reportedly a huge lever (TScale2)
7. **BitNet b1.58** — ternary quantization, 60M+ params in 16MB (ksang: 1.2029)

## Submission Status
- **PR #259** opened on openai/parameter-golf
- Config: QK Gain 1.2 + Sliding Window Eval
- Waiting for CI to run on 8xH100

## Discord Intel (2026-03-20, #parameter-golf-discussions)

### Key Discussion: TTT Exploit Potential (sam_acqua / Larry / will depue)
- **sam_acqua** (wrote the TTT code in main) flagged: if you train on ALL tokens across sequences (not just within-document), increasing val set size artificially decreases val loss
- Current implementation only trains within-document — but cross-document TTT could be an exploit
- **Larry's insight:** "The trained model is only saturated under the 16mb limit. If you can add parameters during eval, then you'd want to do extended pre-training on Val data while adding more parameters. It's not rly TTT, just working around the 16mb bottleneck."
- **will depue (OAI):** confirmed you can't "train" on validation set, but TTT is allowed — the distinction is you predict first, then learn from what you predicted
- **Implication for us:** Our TTT LoRA approach is legitimate and everyone uses it. The winning edge is optimizing HOW TTT is done (cross-doc vs within-doc, number of gradient steps, LR scheduling during TTT)

### Other Notes
- Competition runs until **April 30th** — plenty of time
- Every new technique gets repeated 5x in PRs — lots of copycats, original ideas matter
- Most participants are using Claude/Cursor to generate submissions (visible in commit messages)
- Many PRs are just forks of #198 with minor tweaks

## Next Steps After Current Batch
1. Update PR #259 with stacked results if they beat current submission
2. Test Int6 QAT script on 4090
3. Research WWW architecture (#1 on leaderboard)
4. When RunPod credits arrive: blast best config on H100
