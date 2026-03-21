# Idea Bank — Parameter Golf Competition

## Leverage Table — Where Are the Outsized Returns?

| Lever | Current | Max | Gap | Status |
|-------|---------|-----|-----|--------|
| **Training time** | 600s | 600s | 0 | Maxed out |
| **Eval time** | ~200s | 600s | **400s idle** | Biggest untapped resource. Model ensemble? |
| **Artifact bytes** | 15.5MB | 16.0MB | **500KB** | ~800K more params possible |
| **Model architecture** | Transformer + XSA | Novel ops | **Open** | Partial RoPE, LN Scale being tested |
| **Training data ordering** | Sequential shards | Curriculum/shuffle | **Fully untapped** | Nobody has tried this |
| **Compression scheme** | Uniform int6 | Adaptive int5/6/7 | **Partially tested** | Gradient-guided quant (novel) |
| **Weight averaging** | EMA (0.997) | — | Small | EMA seems near-optimal |
| **Quantization timing** | Post-training | Late QAT (last 4%) | **Testing now** | 3x less quant degradation |
| **Eval methodology** | Sliding stride=64 | — | Small | Near optimal for non-TTT |
| **Optimizer config** | Muon + AdamW | — | Explored | WD/LR tuned |

**Strategy**: Find outsized returns OUTSIDE the training loop. Training is near-maxed. The 400s of idle eval time and the compression scheme are the biggest opportunities for differentiation.

## NOVELTY CHECK — Before Every Novel Idea

Before implementing ANY idea from "Novel" sections, verify:
```bash
gh pr list --repo openai/parameter-golf --state open --limit 50 --json number,title,body | grep -i "<keyword>"
```
If someone submitted it, read their PR, learn from it, then figure out how to do it BETTER.

Tags:
- [NOVEL] — nobody has tried this (verified via PR search)
- [ADAPTED] — someone tried similar, we're improving it
- [COMMODITY] — table stakes, everyone does this

## Priority 1: Proven Commodity Techniques (Get Competitive)
- [x] OrthoInit + MuonWD=0.02 → 1.1536 (proven +0.0038)
- [ ] [COMMODITY] 10 layers with int5-MLP — PR #180's core. Cold cache gave 1.1758. Need warm cache run
- [ ] [COMMODITY] SmearGate — PR #162/#180/#194 all use it. 512 params, trivial
- [ ] [COMMODITY] BigramHash — PR #162/#180. 4096 buckets, dim=128, ~524K params
- [ ] [COMMODITY] SWA every 50 steps — PR #180/#194 both use it
- [ ] [COMMODITY] Int6 QAT with STE — PR #194 gets 11L at 74ms/step. Fake quant during training

## Priority 2: Scaling What Works
- [ ] [ADAPTED] Int4 for MLP weights ([-8,7], 16 levels) — push int5 harder. SWA should smooth enough
- [ ] [ADAPTED] 4x MLP expansion (MLP_HIDDEN=2048) with int4 — more capacity than anyone
- [ ] [ADAPTED] 12 layers with int4+zstd — push depth past PR #179's 11L
- [ ] [ADAPTED] 13 layers with int4+zstd — keep going until it stops
- [ ] [ADAPTED] Double BigramHash to 8192 buckets — fewer collisions
- [ ] [NOVEL] TrigramHash — (t, t-1, t-2) triplets to embeddings. Richer than bigram
- [ ] [ADAPTED] SWA every 25 or every 10 steps — more frequent averaging
- [ ] [ADAPTED] MuonWD sweep to 0.06, 0.08 — more regularization for quant
- [ ] [ADAPTED] stride=128 sliding window — between our 256 and PR #180's 64

## Priority 3: Novel Eval Ideas
- [ ] [NOVEL] Multi-resolution sliding window ensemble — 3 passes at 512/64, 1024/128, 2048/256. Average per-position BPP. Room in 600s eval budget
- [ ] [NOVEL] Bidirectional sliding window — forward + backward (right-to-left) pass, average predictions. Scoring not generating
- [ ] [NOVEL] Attention loss weighting — weight later positions (more context) higher in sliding window
- [ ] [NOVEL] Test-Time Training — 1-3 gradient steps on each eval window before scoring. Reset between windows

## Priority 4: Novel Training Ideas
- [ ] [NOVEL] Future-token noise injection — replace 5-10% of input tokens with t+2. Robust representations
- [ ] [NOVEL] Progressive quantization curriculum — no quant → int8 STE → int6 → int5 over training
- [ ] [NOVEL] Data shard ordering — shuffle/reverse/sort shards. Curriculum effect
- [ ] [NOVEL] Cosine warmdown instead of linear — more time at moderate LR

## Priority 5: Novel Architecture Ideas
- [ ] [NOVEL] Progressive precision gradient — int7 outer layers, int5 inner, int4 innermost
- [ ] [NOVEL] Learned per-row bit-width — adaptive int4/5/6 per row via sensitivity
- [ ] [NOVEL] Token-frequency-weighted embedding precision — common tokens get more bits
- [ ] [NOVEL] Asymmetric encoder/decoder — 7E+3D or 8+3 instead of equal halves

## Priority 6: Tokenizer Angle
- [ ] [NOVEL] SP4096 + int5 + wider MLP — tokens_per_byte 0.41→0.31. May now fit at dim=512
- [ ] [NOVEL] SP8192 at dim=448 — aggressive vocab angle

## Priority 7: Throughput
- [ ] [ADAPTED] FlashAttention 3 — PR #164 got 68ms/step. Check availability
- [ ] [ADAPTED] torch.compile mode="max-autotune"
- [ ] [ADAPTED] TRAIN_SEQ_LEN=1024 with sliding window — faster attention, more steps
- [ ] [ADAPTED] MUON_BACKEND_STEPS=3 — saves ~2ms/step

## Tried (Record Results Here)
| Idea | Result | Takeaway | Tag |
|------|--------|----------|-----|
| 10L int5-MLP (cold cache) | 1.1758 | Cold cache 111ms/step killed it | COMMODITY |
| SmearGate + BigramHash | 1.1644 | +0.011 improvement on 10L | COMMODITY |
| QAT int6 STE | 1.1755 | NET LOSS — 115ms/step overhead not worth quant improvement | COMMODITY |
| Int6-all (not int5-MLP) | 1.1465 | Quant penalty 0.010 vs 0.029. Much better | ADAPTED |
| Batch=524K (from 786K) | 1.1465 | 63ms/step, 9400 steps. Throughput win | ADAPTED |
| Batched sliding eval stride=64 | ~0.004 improvement | Batching 32 windows makes stride=64 feasible in 172s | ADAPTED |
| 11L MLP=1280 | 1.1480 | Too narrow — worse than 10L/1536 | COMMODITY |
| **11L MLP=1408** | **1.1444 (3-seed)** | **BEATS LEADER.** Sweet spot depth×width | COMMODITY |
| FlashAttention 2.8.3 | 1.1429 | 66ms vs 68ms, ~200 more steps | ADAPTED |

## Principles
1. If something works, try MORE of it until it stops working
2. Optimize outside the constraint — squeeze more into 16MB, get more steps per second
3. Push every dimension: compression, MLP width, layer count, eval methodology, quant granularity
4. The winning submission has one or two [NOVEL] techniques that actually work
5. Before declaring an idea dead, check if cold cache was the problem

---
Maintain at least 10 untried ideas at all times. If below 10, brainstorm before continuing.
Every 10 experiments, refresh novelty tags by checking latest PRs.
