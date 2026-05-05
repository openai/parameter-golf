# Tokenizer Optimization Analysis for Parameter Golf

## Executive Summary

**Verdict: vocab=2048 is PLAUSIBLE but RISKY. Not recommended as a primary strategy, but worth trying as a secondary optimization IF combined with depth recurrence.**

The math shows vocab=2048 could improve BPB by 0.01-0.05 in the optimistic case, but could also hurt by 0.01-0.02 in the pessimistic case. The wide confidence interval makes this a gamble. Architecture improvements (depth recurrence at 7×2@672, which already beats baseline by 0.018 BPB) are a much safer bet.

---

## 1. How BPB Works

```
BPB = (val_loss / ln(2)) × (tokens_per_byte)
```

- **val_loss**: cross-entropy in nats per token (higher = worse per-token prediction)
- **tokens_per_byte**: how many tokens the text gets split into per byte (lower = more efficient tokenizer)
- **BPB**: bits per byte (lower = better compression = better model)

**Key insight**: Larger vocab → fewer tokens per byte (good!) but harder per-token prediction (bad!). The net effect depends on model capacity.

## 2. Actual Measurements for sp1024

From our experimental data (step 500: val_loss=2.4997, val_bpb=1.4805):

| Metric | Value |
|--------|-------|
| tokens_per_byte (sp1024) | **0.4105** |
| bytes_per_token (sp1024) | **2.44** |
| Implied val_loss at BPB=1.2244 | **2.067 nats** |

Each sp1024 token covers ~2.44 bytes of UTF-8 text on average.

## 3. Parameter Budget Impact

### Embedding table growth

| Vocab | Embedding Params | Embedding Size (int8) | % of 16MB Budget |
|------:|------------------:|----------------------:|------------------:|
| 768 | 393,216 | 0.39 MB | 2.4% |
| 1024 | 524,288 | 0.52 MB | 3.3% |
| 1536 | 786,432 | 0.79 MB | 4.9% |
| 2048 | 1,048,576 | 1.05 MB | 6.5% |
| 4096 | 2,097,152 | 2.10 MB | 13.1% |

Going 1024→2048 adds **524,288 parameters** (~0.52 MB in int8).

### Fitting in 16MB

Current baseline uses ~15.86 MB (only ~140 KB headroom).

To fit vocab=2048, we must shrink the model. Options:

| Config | Total Params | vs Baseline | Feasible? |
|--------|-------------:|------------:|-----------|
| 9L d504 v2048 | 17,386,488 | +322K (over!) | ❌ Needs more compression |
| 9L d496 v2048 | 16,539,120 | -525K | ✅ Fits comfortably |
| 8L d512 v2048 | 15,751,168 | -1.3M | ✅ Fits but loses a layer |
| 7x2@640 v2048 (recurrent) | ~15.5M | ~same | ✅ Best option |

**Best fit**: 9L d496 v2048 — only 3% width reduction, stays well under 16MB.

## 4. Break-Even Analysis

For vocab=2048 to match baseline BPB=1.2244, the model can absorb this much val_loss increase:

| tokens/byte improvement | Break-even val_loss increase | In % of baseline |
|:-----------------------:|:---------------------------:|:----------------:|
| 10% | +0.230 nats | +11.1% |
| 15% | +0.365 nats | +17.7% |
| 20% | +0.517 nats | +25.0% |

**The margin is GENEROUS.** Even a 10% tokens/byte improvement allows up to +0.23 nats val_loss increase before BPB becomes worse.

### Expected val_loss increase from doubling vocab

| Factor | Expected Impact |
|--------|:---------------:|
| 2× softmax classes (harder prediction) | +0.00 to +0.10 nats |
| 2× embedding entries to learn | +0.01 to +0.05 nats |
| Model shrink (dim 512→496) | +0.01 to +0.03 nats |
| Fewer prediction positions (benefit) | -0.00 to -0.05 nats |
| **Net expected** | **+0.02 to +0.13 nats** |

This is comfortably within the break-even range for 10-15% tokens/byte improvement.

**Expected net BPB change: -0.05 to +0.02** (wide confidence interval).

## 5. Data Availability

### Can we download sp2048 data?

The `cached_challenge_fineweb.py` accepts `--variant sp2048`, but this requires the HuggingFace repo (`willdepueoai/parameter-golf`) to have pre-tokenized sp2048 data in its manifest. The help text mentions sp4096 as an example variant, suggesting multiple variants may exist.

**To check**: Run `python3 data/cached_challenge_fineweb.py --variant sp2048` — if the manifest has it, download is straightforward. If not, we need to retokenize from scratch.

### Retokenization cost (if sp2048 not pre-cached)

1. Download `docs_selected.jsonl` (~10-50 GB raw text)
2. Train SentencePiece BPE at vocab=2048 (minutes)
3. Retokenize 10B tokens (hours of CPU time)
4. Write shard files

This is a significant overhead but a one-time cost.

## 6. Competition Considerations

From the README:
> "Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score."

- Custom tokenizer submissions face **extra scrutiny**
- Must **prove** BPB is correctly calculated
- The byte-counting LUT (`build_sentencepiece_luts`) handles any SentencePiece model automatically — no code changes needed for BPB correctness
- No existing submissions use vocab > 1024 (we'd be first)

## 7. Concrete Recommendations

### Priority 1: DO NOT PURSUE VOCAB CHANGE AS PRIMARY STRATEGY

Architecture improvements are a bigger, safer win:
- **7×2@672 recurrent** already beats baseline by 0.018 BPB at 2K steps
- Depth recurrence + speed optimization should be the main focus
- Tokenizer change adds complexity, data pipeline work, and regulatory risk

### Priority 2: TRY VOCAB=2048 AS A SECONDARY EXPERIMENT

**If and only if** we have spare engineering bandwidth:

1. Check if `python3 data/cached_challenge_fineweb.py --variant sp2048` works
2. If pre-tokenized data exists, run a 2K-step screening on 1×H100:
   ```bash
   VOCAB_SIZE=2048 MODEL_DIM=496 NUM_LAYERS=9 \
   DATA_PATH=./data/datasets/fineweb10B_sp2048 \
   TOKENIZER_PATH=./data/tokenizers/fineweb_2048_bpe.model \
   ITERATIONS=2000 \
   torchrun --standalone --nproc_per_node=1 train_gpt.py
   ```
3. Compare val_bpb at step 500 and 2000 against baseline (1.4805 / 1.2963)

### Priority 3: BEST COMBINED CONFIG (if vocab=2048 works)

Combine with depth recurrence:
```
7 unique blocks × 2 loops, dim=640, vocab=2048
```
- Weight sharing saves enough params to absorb larger embedding
- Wider model (640 vs 512) compensates for vocab overhead
- 14 effective layers maintain depth

### What NOT to try

- **vocab=4096**: Too expensive (2.1MB embedding in int8). Would require dim≈450, massive capacity loss.
- **vocab=768 or smaller**: Increases tokens_per_byte, which hurts BPB even if per-token loss improves slightly.
- **byte-level (vocab=260)**: tokens_per_byte=1.0, would need val_loss < 0.849 nats — impossible at this model scale.

## 8. Key Numbers Summary

| Parameter | sp1024 (baseline) | sp2048 (estimated) |
|-----------|-------------------:|-------------------:|
| tokens_per_byte | 0.4105 | ~0.35 (est.) |
| bytes_per_token | 2.44 | ~2.85 (est.) |
| Embedding params | 524K | 1,049K |
| Embedding size (int8) | 0.52 MB | 1.05 MB |
| Extra budget needed | — | ~0.53 MB |
| Dim required (9L) | 512 | ~496 |
| Expected BPB impact | — | -0.05 to +0.02 |
| Data availability | ✅ Downloaded | ❓ Check HF repo |
| Implementation effort | — | Low (env vars only) |
| Competition risk | Low | Medium (extra scrutiny) |
