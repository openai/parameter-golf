# Experiment 4: How Depth Recurrence Beat SOTA by 0.017 BPB

## The Result

We submitted **PR #1435** to openai/parameter-golf with a **1.0980 BPB** score (3-seed mean), beating the merged SOTA of 1.1147 by **0.0167 BPB**. The artifact is only 14.6 MB out of the 16 MB limit. This is a legitimate record-tier submission.

## What Is Depth Recurrence?

Normal transformers have N layers, each with unique weights. In depth recurrence, some layers are **reused** — their weights are shared across multiple positions in the forward pass. Our model has 11 physical layers but layers 4 and 5 run twice:

```
Physical layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  (11 layers)
Virtual layers:  [0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9, 10]  (13 layers)
                                    ^^^^
                              these repeat once
```

The model gets 13 layers of effective depth while only storing 11 layers of parameters. That's 2 free layers of computation.

## Why This Was Supposed to Be Impossible

A detailed non-record submission (PR #363) spent 4 days and $200 proving depth recurrence **doesn't work**. The author found two fundamental problems:

1. **Quantization compounding**: Shared weights get quantized once but errors propagate through every repeat. With 3 repeats, quantization error amplified ~900x.
2. **Step time overhead**: More forward passes = slower steps = fewer training steps in the 10-minute budget. A 3x3 loop config lost 22% of training steps.

Their controlled test: flat 11L scored 1.1648, looped 3x3 scored 1.1894. Recurrence was **0.025 BPB worse**.

Three independent researchers (Evangeline, Frosty40, Ciprian-Florin) all concluded recurrence was a dead end.

## What Changed

PR #1334/#1421 found the sweet spot that avoids both taxes:

| Problem | Failed Approach (PR #363) | Our Approach (PR #1421 base) |
|---------|--------------------------|------------------------------|
| Loop config | 3 layers x 3 repeats (9 virtual layers added) | **2 layers x 1 repeat (2 virtual layers added)** |
| Quantization compounding | 900x error amplification through 3 loops | Minimal — only 1 extra pass, GPTQ int6 handles it |
| Step overhead | 32ms/step slower = 22% fewer steps | **~12% throughput drop**, only active for last 45% of training |
| Activation timing | From step 0 | **From step 3000** (trains flat for first 55%) |
| Model width | 640d (wider to use saved params) | **512d (same as flat)** |

The key insight: **don't try to save parameters with recurrence — use it for free depth.** Keep all 11 layers at 512d, just loop 2 of them once. The overhead is tiny and the model gets 2 extra layers of representation for free.

## What We Added: BigramHash

On top of the PR #1421 base, we added BigramHash(1536, dim 112) — a hash-based bigram embedding that gives the model character n-gram features. This was part of the previous SOTA stack but wasn't in PR #1421.

| Variant | BPB (seed 1337) | Artifact |
|---------|----------------|----------|
| Vanilla (PR #1421 base) | 1.0999 | 14.3 MB |
| **+ BigramHash** | **1.0989** | 14.6 MB |

Small but consistent improvement. BigramHash adds ~230K parameters and ~270KB to the artifact.

## The Architecture Stack

Everything that makes this work:

- **Depth recurrence** (layers 4,5 repeat once, activated step 3000)
- **BigramHash** (1536 buckets, dim 112) + SmearGate
- **U-Net skip connections** with learnable sigmoid gating (skip gates)
- **Parallel residuals** (layers 7+: attention and MLP run in separate lanes)
- **MuonEq-R optimizer** (row-normalized Muon with NS5)
- **EMA 0.9965** (key tuning — slightly more weight on recent steps)
- **GPTQ int6 + Brotli** (quantization + compression for 16MB limit)
- **XSA on all 11 layers** (exclusive self-attention)
- **Value Embedding** (dim 128, layers 9,10)
- **GQA** (8 heads, 4 KV heads)

## What We Left On the Table

SP4096 tokenizer wasn't available in the public data manifest, so we used SP1024. Every top open PR uses SP4096 — it's worth ~0.01 BPB. With SP4096, our result would likely be **~1.088 BPB**.

The open PR frontier (PRs #1406, #1421) reaches 1.089-1.093 with SP4096. Our 1.098 with SP1024 is competitive despite the tokenizer handicap.

## Cost

4 training runs (vanilla s1337, bigram s1337, bigram s42, bigram s2024) on an 8xH100 SXM SECURE pod. ~4 hours total runtime at $21.52/hr = **~$86**.

## PR Link

**https://github.com/openai/parameter-golf/pull/1435**
