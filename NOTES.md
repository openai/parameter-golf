# Experiment Notes

## Key Competitor PRs (as of 2026-04-08)

| PR | BPB | Vocab | Key Technique |
|----|-----|-------|--------------|
| [#1450](https://github.com/openai/parameter-golf/pull/1450) | 1.08480 | SP8192 | TMA Megakernel (+10.5% throughput, fused Triton MLP) |
| [#1437](https://github.com/openai/parameter-golf/pull/1437) | 1.08091 | SP8192 | N-gram Tilt (`p *= exp(beta * 1[t==bigram_hint]) / Z`) |
| [#1460](https://github.com/openai/parameter-golf/pull/1460) | 1.08269 | SP8192 | Score-first TTT + Eval-Time Hash Embedding |

All top PRs use **SP8192** (8192 BPE vocab) vs our **SP1024** — this is the biggest gap.

---

## sota_16 Changes (from sota_15)

### Eval-time only (no training change)

**1. N-gram Tilt** (from PR #1437)
- Bigram count table `bg_counts[vocab, vocab]`, add-1 smoothed
- At scoring: `lf += beta * one_hot(argmax(bg_counts[prev_tok]))`
- Table updated **AFTER** scoring each chunk (causal, score-first)
- `NGRAM_BETA=0.5`, expected gain ~0.010–0.015 BPB

**2. Eval-Time Hash Embedding** (from PR #1460)
- `nn.Embedding(16384, 512)`, zero-init, created fresh at eval
- `h = (prev_token * 2039 + curr_token) % 16384`
- Added as residual to `tok_emb` via `register_forward_hook`
- Trained in TTT SGD alongside model weights
- `HASH_EMB_SIZE=16384`, expected gain ~0.0004 BPB

**3. TTT LR fix** (2026-04-08, after comparing PR #1460)
- LR: `0.001 → 0.005` (5× increase, matched to PR #1460)
- Added **cosine LR decay** within each chunk's TTT steps
  - `cos_lr = ttt_lr * 0.5 * (1 + cos(π * step / total_steps))`
  - Starts at full LR, decays to 0 by end of each chunk

---

## sota_15 Changes (from sota_12)

- **DyT** replaces all 6 `RMSNorm` sites: `forward = tanh(alpha * x)`, `alpha` init=0.5
- **JEPA** auxiliary loss: `JEPAPredictor(512 → 64 → 512)`, weight=0.1
  - Predicts `h[t+1]` from `h[t]` with cosine loss + stop-gradient target
  - Training only, zero parameter overhead at eval

---

## Architecture Baseline (sota_12)

- 11L / 512d / 8H / 4KV GQA
- XSA all layers
- Full Hessian GPTQ int6
- Legal score-first TTT
- MTP (2 heads, weight 0.1)
- Depth recurrence (L2,3,4,5, starts step 1500)
- Parallel residuals (L5+)
- Trigram + VE (L8,9,10)
- Warmdown 5500 iters

---

## TTT Tips

- **LR**: 0.005 works better than 0.001 (PR #1460 uses 0.005)
- **Cosine decay** within chunk: start full LR → 0 over all steps in chunk
- **Momentum**: 0.9 SGD
- **Epochs**: 3 per chunk
- **Chunk size**: 32768 tokens
- **Score-first**: always `inference_mode` score before any `backward`

---

## Todo / Ideas

- [ ] SP8192 tokenizer + dataset (biggest unlock, ~0.01-0.02 BPB)
- [ ] TMA Megakernel (Triton, H100 TMA, +10.5% steps = ~700 extra iters)
- [ ] Tune `NGRAM_BETA` in {0.3, 0.5, 0.8, 1.0} if sota_16 underperforms
- [ ] Try trigram tilt (not just bigram)
- [ ] Larger hash embedding size (32768, 65536)
