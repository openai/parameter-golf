# A/B Test Results — March 22, 2026 (Online TTT Baseline)

**Baseline: 1.1201 sliding BPB (exp146, online TTT, seed=1337)**
**Script base: `clean_train_113_online_ttt.py`**
**Goal:** Find any single change that beats 1.1201 AND fits 16MB.

## Completed Tests


| Exp  | Change                            | Sliding BPB                         | Δ vs 1.1201 | Artifact MB | Fits?              | Verdict                                                                                            |
| ---- | --------------------------------- | ----------------------------------- | ----------- | ----------- | ------------------ | -------------------------------------------------------------------------------------------------- |
| 147a | XSA last 4 (GQA-aware orthogonal) | ~1.125                              | +0.005      | 15.97       | Yes                | **WORSE** — verified correct vs paper (arxiv 2603.09078) but hurts at 12L dim=512                  |
| 148  | Late QAT threshold=0.1            | 1.1207                              | +0.001      | 16.27       | **No**             | **WORSE** — QAT makes weights less compressible                                                    |
| 149  | GPTQ-lite (clip search)           | 1.1203                              | +0.000      | 15.97       | Yes                | **NEUTRAL** — no benefit from clip search at our scale                                             |
| 150  | Backout Connection (lambda=0.2)   | 1.1211                              | +0.001      | 15.96       | Yes                | **WORSE** — learned mid-layer subtraction doesn't help                                             |
| 151  | Adaptive CE ((1-p)*CE)            | 1.1251                              | +0.005      | 15.94       | Yes                | **MUCH WORSE** — weighted loss destabilizes training                                               |
| 152  | ADAM_WD=0.02                      | **1.1199**                          | **-0.0002** | 16.00       | Yes                | **MARGINAL WIN** — lower Adam WD helps embeddings slightly                                         |
| 153  | MUON_WD=0.04                      | **1.1167**                          | **-0.003**  | 17.49       | **No**             | **BEST BPP but over limit** — lower WD = better quality                                            |
| 155  | MUON_WD=0.05                      | **1.1180**                          | **-0.002**  | 16.63       | **No**             | **GOOD BPP but 628KB over**                                                                        |
| 156  | WD=0.05 + INT5_MLP                | ~1.15+                              | +0.03+      | 14.22       | Yes                | **MUCH WORSE** — INT5 quant gap too large (0.035 BPP)                                              |
| 157  | LN Scale (1/sqrt(layer+1))        | [killed@1.336](mailto:killed@1.336) | +0.028      | -           | -                  | **TERRIBLE** — killed early, LN Scale hurts at our scale                                           |
| 158c | Wanda pruning (2%)                | 1.1203                              | +0.000      | 15.96       | Yes                | **NEUTRAL** — activation-weighted pruning = magnitude at 2%                                        |
| 159  | Tight SWA (scale<0.2, every 50)   | **1.1193**                          | **-0.0008** | 16.16       | **No**             | **GOOD but 164KB over**                                                                            |
| 160  | Simple XSA (y-=y.mean)            | [killed@0.91](mailto:killed@0.91)   | N/A         | -           | -                  | **BUGGY** — wrong dim, not actual XSA. Killed.                                                     |
| 160b | EMA (decay=0.997, from step 0)    | **1.1181**                          | **-0.002**  | 16.08       | **No (84KB over)** | **WIN but 84KB over!** EMA smooths weights → better quant → better TTT. Need compression headroom. |


## Key Insights

1. **WD is the strongest lever** — MUON_WD=0.04 gives -0.003 BPP but doesn't fit 16MB
2. **Everything that helps BPP also hurts compression** — lower WD, EMA, Tight SWA all go over 16MB
3. **Architectural changes (XSA, Backout, LN Scale, Adaptive CE) all hurt** at our 12L dim=512 scale
4. **GPTQ-lite and Wanda are neutral** — no benefit from smarter quantization/pruning at 2% sparsity
5. **XSA implementation was correct** — verified against arxiv 2603.09078, but GQA-aware version genuinely doesn't help at our scale
6. **Grok's "simple XSA" (y.mean) was WRONG** — not actual XSA, just sequence-mean subtraction

## Compression Problem

The core challenge: techniques that improve BPP also make weights less compressible:

- WD=0.04: -0.003 BPP but +1.5MB artifact
- WD=0.05: -0.002 BPP but +0.6MB artifact
- EMA: -0.004 roundtrip but +0.08MB artifact
- Tight SWA: -0.001 BPP but +0.16MB artifact

Need to find compression headroom WITHOUT hurting BPP. Options:

- Smaller code size (strip dead code)
- Better compression algo
- Trade some capacity (fewer bigram buckets, smaller bigram dim)

## Recently Completed (EMA sweep + TTT tests)


| Exp         | Change                                  | Sliding BPB                         | Δ vs 1.1201 | Artifact | Verdict                                                                            |
| ----------- | --------------------------------------- | ----------------------------------- | ----------- | -------- | ---------------------------------------------------------------------------------- |
| 161         | EMA + WD=0.06 + no prune                | **1.1175**                          | **-0.003**  | 16.07MB  | Pruning hurts EMA                                                                  |
| 162         | EMA + WD=0.055 + no prune               | **1.1167**                          | **-0.003**  | 16.39MB  | Lower WD helps                                                                     |
| **163**     | **EMA + WD=0.05 + no prune**            | **1.1161**                          | **-0.004**  | 16.74MB  | **NEW BEST!**                                                                      |
| 164         | EMA + WD=0.05 + ADAM_WD=0.02 + no prune | **1.1159**                          | **-0.004**  | 16.78MB  | Marginal over 163                                                                  |
| 165b        | Seed projection (all MLP down-proj)     | [killed@1.354](mailto:killed@1.354) | +0.046      | -        | **TERRIBLE** — random projections too aggressive at dim=512                        |
| 166         | Untied embeddings + HEAD_LR=0.008       | [killed@1.261](mailto:killed@1.261) | +0.008      | -        | **WORSE** — separate head hurts at our scale                                       |
| TTT-5ep     | Online TTT 5 epochs (vs 3)              | 1.1204                              | +0.000      | -        | **NEUTRAL** — model already converges in 3 epochs/window                           |
| TTT-freeze0 | Online TTT freeze=0 (all layers)        | 1.1201                              | +0.000      | -        | **NEUTRAL** — unfreezing all doesn't help                                          |
| TTT-lr003   | Online TTT LR=0.003 (vs 0.002)          | 1.1201                              | +0.000      | -        | **NEUTRAL** — higher LR same as default                                            |
| 168         | Canon v1 (conv inside MLP)              | [killed@0.001](mailto:killed@0.001) | N/A         | -        | **BROKEN** — degenerate loss from conv+square interaction                          |
| 168b        | Canon v2 (Canon-C, conv before MLP)     | [killed@1.348](mailto:killed@1.348) | +0.040      | -        | **MUCH WORSE** — 35% slower (122ms) + terrible quality even with correct placement |
| 169         | Warmdown 3500 (vs 3000)                 | **1.1199**                          | **-0.0002** | 15.82    | Yes                                                                                |


## Recently Completed (Partial RoPE, Simple XSA)


| Exp  | Change                                   | Sliding BPB | Δ vs 1.1201 | Artifact | Verdict                                                                             |
| ---- | ---------------------------------------- | ----------- | ----------- | -------- | ----------------------------------------------------------------------------------- |
| 170  | Value Embeddings (dim=128, layers 10,11) | 1.1203      | +0.000      | 16.22MB  | **NEUTRAL** — pre-quant was -0.002 better but quant gap ate it. Over 16MB.          |
| 171b | Partial RoPE (32/64 dims)                | 1.1217      | +0.002      | 16.01MB  | **WORSE** — partial rotation hurts at dim=512, larger quant gap eats pre-quant gain |


| 172 | Simple XSA (y -= y.mean(dim=1), last 4) | [killed@0.4876](mailto:killed@0.4876) | N/A | - | **INVALID** — mean over seq dim leaks future info (non-causal). BPB 0.49 at step 4000 = impossibly good = info leak. PR#414 uses orthogonal projection XSA, NOT mean-subtract. |

## Currently Running


| Exp | Change                                                | Sliding BPB | Δ vs 1.1201 | Artifact    | Verdict                                                                          |
| --- | ----------------------------------------------------- | ----------- | ----------- | ----------- | -------------------------------------------------------------------------------- |
| 173 | EMA(0.997) + BigramVocab=8192 + no prune              | **1.1199**  | **-0.0002** | **15.57MB** | **MARGINAL WIN** — fits with 425KB headroom but bigram reduction costs 0.002 BPP |
| 174 | EMA(0.997) + BigramVocab=16384 + BigDim=48 + no prune | 1.1223      | +0.002      | **14.99MB** | **WORSE** — dim=48 hurts BPP more than expected, but 1MB headroom!               |
| 175 | ~~EMA + full bigram rerun~~                           | killed      | -           | -           | Same compression = same 71KB over                                                |
| 176 | EMA + bigram8k + ADAM_WD=0.02 + Warmdown3500          | **1.1189**  | **-0.0012** | **15.43MB** | **NEW BEST** — stacked wins work! 570KB headroom                                 |


| Exp | Change | Sliding BPB | Δ vs 1.1201 | Artifact | Verdict |
| --- | ------ | ----------- | ----------- | -------- | ------- |
| 177 | PRP MLP last 4 layers (arxiv 2512.13480) | killed@1.2792(2k) | +0.027 | - | **WORSE** — random projection with diagonal modulation can't match learned dense at dim=512 |
| 178 | Butterfly attn proj last 4 (FFT factorized) | killed@step200 | +0.027@1k | - | **KILLED** — 2.6x slower (239ms vs 90ms). Reshape+einsum loop can't compete with GEMM at dim=512 |
| 179 | EMA + bigram8k + **WD=0.055** + ADAM_WD=0.02 + WD3500 | **1.1182** | **-0.0019** | **15.74MB** | **NEW BEST** — lower WD gives 0.001 more BPP, still fits with 258KB headroom |
| 180 | EMA + bigram8k + **WD=0.05** + ADAM_WD=0.02 + WD3500 | **1.1176** | **-0.0025** | 16.09MB | 86KB over with zstd — BPP is best yet |
| 181 | Same as 180 + BIGRAM_DIM=56 | killed | - | 15.52MB | Killed — researched compression instead of degrading model |
| **182b** | **Same as 180 + BROTLI compression** | **1.1173** | **-0.0028** | **15.92MB** | **ALL-TIME BEST** — brotli saves 364KB vs zstd, full quality fits! Checkpointed. |
| 183 | WD=0.045 + brotli | 1.1170 | -0.0031 | 16.33MB | ❌ Over limit — WD=0.05 confirmed as sweet spot with brotli |
| 184 | TTT_EPOCHS=4 (was 3) | 1.1175 | -0.0026 | 15.93MB | **NEUTRAL** — extra epoch slightly overfits, 3 epochs optimal |
| 185 | EMA_DECAY=0.998 (was 0.997) | 1.1173 | -0.0028 | 15.98MB | **NEUTRAL** — matches 0.997 exactly, no improvement |
| 186 | Full bigram 16384x64 + WD=0.05 + brotli | ~1.1155 est | -0.0046 est | 16.42MB | ❌ Over limit — full bigram too big even with brotli |
| 187 | 13L + WD=0.07 + EMA + brotli | 1.1158 | -0.0043 | 15.80MB | +1 layer helps. Checkpointed. |
| **188** | **14L + WD=0.09 + EMA + brotli** | **1.1155** | **-0.0046** | **15.83MB** | **🏆🏆🏆 ALL-TIME BEST** — 14L optimal depth. Checkpointed. |
| 189 | 15L + WD=0.11 + EMA + brotli | 1.1171 | -0.0030 | 15.86MB | **WORSE** — 15th layer doesn't compensate for WD=0.11 penalty + fewer steps. 14L is optimal. |

## Planned Tests (Priority Order)

### Still To Test


| Exp | Change                                           | Expected Δ | Rationale                                                          |
| --- | ------------------------------------------------ | ---------- | ------------------------------------------------------------------ |
| 170 | Value Embeddings (shared dim=128, last 2 layers) | ±0.003     | In PR#374, #401, #414 — adds learned embedding to attention values |
| 171 | ~~Partial RoPE (32/64 dims)~~                    | **+0.002** | **DONE — WORSE**                                                   |
| 172 | Online TTT with LoRA adaptation                  | unknown    | Faster per-step → more steps in eval budget                        |
| 173 | Online TTT LR cosine decay per window            | ±0.001     | Better per-window learning schedule                                |
| 174 | Element-wise modulated random matrix             | unknown    | Fixed random matrix × learned diagonal (N not N²) — Grok #1        |
| 175 | Curriculum recurrence (ramp loops 2→16)          | unknown    | Progressive depth — Grok #2                                        |
| 176 | Noble (NOBLELinear rank=4)                       | unknown    | Nonlinear low-rank branch — needs very low rank to fit             |


### Dead / Not Viable


| Idea                         | Reason                                                                                                    |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| Seed projection (all layers) | +0.046 BPP at step 1000. Too aggressive at dim=512.                                                       |
| Canon (both placements)      | v1: degenerate collapse. v2: +0.040 worse + 35% slower. Paper results are on synthetic tasks, not LM BPP. |
| Noble (rank=32)              | +3.8M params = +2.9MB. Doesn't fit 16MB.                                                                  |
| FFT butterfly projection     | 2.6x slower (239ms vs 90ms). Reshape+einsum loop can't compete with GEMM at dim=512. Killed.             |
| PRP (arxiv 2512.13480)       | +0.027 BPP worse. Random proj + diagonal modulation fundamentally less expressive than dense at dim=512.   |
| Untied embeddings            | +0.008 worse — separate head hurts at our scale                                                           |
| Simple XSA (y.mean(dim=1))   | Leaks future info through non-causal mean. BPB 0.49 at step 4000 = info leak, not real improvement.       |
| Late QAT                     | Makes weights less compressible, over 16MB                                                                |
| LN Scale                     | +0.028 at step 1000. Terrible.                                                                            |
| INT5 MLP                     | 0.035 BPP quant gap. Way too aggressive.                                                                  |


### Compression-Focused Tests (do when we have a winning BPP config)


| Exp | Change                             | Expected Δ | Rationale                |
| --- | ---------------------------------- | ---------- | ------------------------ |
| 177 | Strip dead code from script        | 0 BPP      | Save ~10-20KB code_bytes |
| 178 | Smaller BigramHash (8192 vs 16384) | ±0.002     | Frees ~32KB              |
| 179 | BigramHash dim=32 (vs 64)          | ±0.001     | Halves bigram params     |


