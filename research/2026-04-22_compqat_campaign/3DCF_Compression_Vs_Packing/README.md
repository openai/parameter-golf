> This is a supporting research note, not a Parameter Golf leaderboard submission.  
> It explores compression-vs-packing observations from the campaign and is included for context.

# Non-Record Research Note: Compression Beats Packing — Bit-Packed int6 Weights Compress Worse Than int8-Stored int6

**Author:** yevh ([@yevh](https://github.com/yevh)) | **Contribution type:** research note + negative result | **Track:** non-record

---

> **Context.** This is a research note distilled from a multi-week exploration of dense quantization strategies for competition-constrained models. The exploration did not produce a leaderboard result; the negative result itself is the contribution. The same lesson, reframed positively, seeded a separate submission (compression-aware QAT) that does land on the leaderboard — see `../2026-04-22_SP8192_CompQAT_PR1493/`.
>
> **Full research narrative:** [Blog post — OpenAI Parameter Golf: what I built, what worked, what didn't](./article.md).
>
> **Rust quantization crate used for the measurements:** [github.com/3DCF-Labs/model-compress](https://github.com/3DCF-Labs/model-compress), MIT-licensed.

---

## Finding

Bit-packing int6 weights to 6 bits per value (0.75 bytes/weight, the naive 25%-density-saving) produces *larger* post-compression artifacts than storing the same int6 values one-per-int8-byte (1.0 bytes/weight, 25% "wasted" space).

Measured across three model sizes, competition-scale training (2×H100 NVL, 600s, SP8192, all else held equal):

| Config | Params | int6 stored as int8 + zstd | Bit-packed int6 + zstd | Packed/int8 |
|---|---:|---:|---:|---:|
| 11 layers | 26.5M | **17.5 MB** | 19.4 MB | **+11%** |
| 14 layers | 33.6M | **20.5 MB** | 24.1 MB | **+18%** |
| 15 layers | 35.9M | **21.3 MB** | 25.5 MB | **+20%** |

The bit-packed artifacts are 11–20% *larger* than the int8-stored artifacts, and the gap grows with model size. This contradicts the naive optimization intuition that "less raw data = less compressed data."

---

## Why

zstd (and brotli, and every practical entropy coder) exploits *byte-level redundancy*. Its compression ratio is determined by the entropy of the byte stream it sees, not by the information content of the values those bytes encode.

**int6 stored as int8 bytes:** only 63 of the 256 possible byte values are ever used (−31 to +31 in two's complement), and those values cluster tightly near zero for a trained network. This is a highly redundant byte stream. zstd achieves a typical ratio of **0.65** — compressed size ≈ 65% of raw.

**int6 bit-packed:** the 6-bit values are packed densely, so the resulting byte stream uses all 256 byte values roughly uniformly. The byte distribution is close to maximum entropy, and there's nothing for zstd to exploit. Typical ratio: **0.97** — almost no compression.

The 25% raw-byte savings from packing is more than eaten by the 32% degradation in compression ratio (0.97/0.65 − 1). Net: bigger artifact.

Additional penalty: when I tried to use the "freed" space for a larger model (14L or 15L instead of 11L), the bigger model trains 20–30% fewer steps in the same wallclock budget, so BPB is also worse. Both axes point the wrong direction.

---

## The general principle

**Byte-level redundancy that an entropy coder can exploit beats packing density.**

For any dense entropy-coded byte payload, denser representations are in tension with post-compression size. Quantization compactness (how many bits per value) and post-compression artifact size (what the shipped `.ptz` weighs) are not the same metric, and they often point in opposite directions.

Corollary for Parameter Golf specifically: any submission that reports "raw quantized weight size" is optimizing the wrong metric. The actual cap is `Total submission size quantized+brotli` (the value [PR #1493](https://github.com/openai/parameter-golf/pull/1493)'s `serialize()` function prints at the end of a run). Submissions optimizing for raw density without checking post-compression size are probably leaving bytes on the table in the wrong direction.

---

## What I also tested (and what also didn't work)

While exploring the same "denser is better" direction, I implemented and measured several alternative quantization methods in the Rust crate. Each has its own negative result worth recording.

### Codebook methods: MSE is not BPB

| Method | MSE vs original weights | vs scalar int8 | BPB regression vs int8 |
|---|---:|---:|---:|
| Residual Quantization (6+2 codebooks) | 0.0000010 | **2× better** | **+1.76** (catastrophic) |
| Lloyd-Max | 0.0000028 | 1.4× worse | +1.76 (same regime) |
| Product Quantization (2×256) | 0.0000134 | 6.8× worse | even worse |
| Scalar int8 (baseline) | 0.0000020 | — | 0 |
| Scalar int6 (per-row symmetric) | 0.0000054 | 2.7× worse | +0.002 (acceptable) |

Residual Quantization has **half the MSE of int8** — by standard quantization-quality metrics, it's strictly better. But when that RQ-quantized model is evaluated end-to-end on a language-modeling task, the BPB is 3.35 while int8's is 1.59. A catastrophic regression despite better MSE.

The mechanism: codebook methods produce **correlated quantization errors**. Two weights with originally different values can both map to the same codebook entry. From the network's perspective this is as if the two weights were always identical — and language models' in-context computation depends on being able to make different weights do different things. Small uncorrelated scalar noise (int8 or int6 jitter) is absorbable. Structured correlated noise is not.

**Takeaway: in language-model quantization, MSE is a misleading metric. You need the end-to-end BPB measurement.** This result was confirmed independently by issue #140 in the Parameter Golf tracker.

### The QAT STE mismatch (a bug story with a lesson)

Initial 3DCF int6 results at competition scale (Run 18) showed a +0.020 BPB gap vs scalar int8 — a meaningful regression. Running a full debugging cycle revealed the cause: my QAT straight-through estimator was simulating quantization with a sqrt-companding asymmetric mapping, while the actual 3DCF export was using uniform symmetric int6 in [−31, +31]. The model was being trained against one quantization and then shipped with a different one.

After fixing the STE to match the export exactly (Run 19), the gap collapsed from +0.020 to +0.002. The QAT mechanism works when the STE faithfully simulates what ships.

**Takeaway: verify QAT STE and export quantization are bit-identical. This seems obvious in retrospect, but it took a full 600s cloud run + a full debugging cycle to catch.**

---

## What this research produced

Two clean takeaways worth adopting in any future Parameter Golf submission:

1. **Don't pack bits for size savings.** Store int6 in int8 bytes and let the entropy coder do the work. Every competition record to date follows this pattern; now there's a measurement that says why.

2. **Measure BPB, not MSE.** The two don't track each other for language-model quantization. Codebook-based methods that look great on MSE can destroy BPB in a single run.

A third takeaway, more subtle, became the starting point of the compression-aware QAT submission (`../2026-04-22_SP8192_CompQAT_PR1493/`):

3. **If packing is the wrong lever, training is.** Bit-packing is trying to feed the compressor less data. The opposite direction — feed the compressor *more* of what it wants — is to train so the post-quantization byte distribution is intrinsically more compressible. Concentrate weights near a small number of int6 grid centers during training, and brotli has more redundancy to exploit on the shipped bytes.

That third takeaway is what the compression-aware QAT submission measures empirically. It shrinks the artifact by 672 bytes vs [PR #1493](https://github.com/openai/parameter-golf/pull/1493) at the same training budget — validating the principle that drove this research, even though the direct 3DCF packing approach failed.

---

## Reproduction

The Rust quantization crate used for all measurements in this note is at [github.com/3DCF-Labs/model-compress](https://github.com/3DCF-Labs/model-compress). Modules relevant to this note:

- `quantize.rs` — symmetric int6/int8 scalar quantization with per-row scales (the baseline)
- `compress.rs` — bit-packing + zstd wrapper (the failed denser-packing experiment)
- `lloyd_max.rs`, `residual_quant.rs`, `product_quant.rs` — codebook methods (the failed "better MSE" experiments)
- `gptq.rs` — GPTQ with per-row diagonal Hessian (strictly dominated by [PR #1394](https://github.com/openai/parameter-golf/pull/1394)'s full Hessian + SD-Clip, kept for reference)

The crate is MIT-licensed and includes its own README with build/use instructions. The three competition-scale measurements above (11L/14L/15L at 600s on 2×H100 NVL) are documented in the cloud run logs at the linked research repo.

---

## Scope and what this is *not*

- **Not a record submission.** No BPB improvement over [PR #1493](https://github.com/openai/parameter-golf/pull/1493)'s 1.0810 is claimed here.
- **Not a claim that codebook methods are useless in general.** They have legitimate applications outside autoregressive language models. The claim is specifically that language-model in-context computation does not tolerate correlated quantization errors at the scale of Parameter Golf models.
- **Not a claim that bit-packing is useless in general.** For byte streams that *don't* subsequently pass through an entropy coder, packing is usually the right move. The claim is specifically about the Parameter Golf deployment path (int6 → zstd/brotli → shipped `.ptz`) and analogous pipelines.

---

## Files in this directory

- `README.md` — this document
- `measurements.md` — raw numbers from the bit-packing comparison and codebook MSE comparison, with explicit run configurations for each
