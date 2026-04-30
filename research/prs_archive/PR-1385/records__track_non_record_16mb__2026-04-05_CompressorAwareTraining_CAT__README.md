# Compressor-Aware Training (CAT)

Non-record submission, 1xH100. Submitting for the technique, not the score.

**val_bpb:** 1.4465 (int8+zlib roundtrip) | **Artifact:** 11.48 MB | **Hardware:** 1xH100 80GB, 600s

## Why I did this

I'm a data scientist. I work mostly with Bayesian stats and causal inference, not language models. I entered Parameter Golf because something about the setup bothered me: everyone compresses their artifact, but training is completely indifferent to compression. The model doesn't know or care that its weights are about to be quantized and compressed. It just optimizes for prediction quality, and compression is an afterthought.

That felt like a missed opportunity. The competition is really a two-level compression problem. Your model compresses text (the BPB score). Then the model itself gets compressed (int8 + zlib, has to fit in 16MB). These two things interact, but everyone treats them as separate steps. In Bayesian model selection, this is the Minimum Description Length principle: minimize the total cost of describing the model plus the data given the model. The 16MB cap is the first term. BPB is the second. Nobody's jointly optimizing them.

So I spent a few days (when I had free time) figuring out whether you could actually train a model to produce weights that compress better. Not for any specific compressor, but for the general family of compressors that most tools use (zlib, zstd, brotli all share the same basic structure). Turns out nobody's tried it.

## How compression works (and why training ignores half of it)

Most compressors (zlib, zstd, brotli) use some variant of the same two-stage pipeline:

1. **Dictionary matching (LZ77):** scan through the bytes looking for repeated sequences. When you find one, replace it with a pointer back to the first occurrence. More and longer repeats = smaller output.

2. **Entropy coding (Huffman / FSE):** take whatever's left and assign shorter codes to byte values that appear more often. Concentrated value distributions = fewer bits.

Every paper I found on compression-aware training (Wiedemann 2018, CERWU 2025, Deep Compression 2015) uses Shannon entropy as the proxy for "how compressible are these weights." Shannon entropy is a good proxy for the entropy coding stage. It tells you nothing about dictionary matching.

Here's what convinced me this matters: I made two arrays with the exact same value distribution, identical histogram, identical Shannon entropy. One was a smooth wave, the other was the same values shuffled randomly. zlib compressed the smooth one 3x smaller. Same entropy, totally different compressed size. Dictionary matching was doing most of the work, and nobody was accounting for it.

## What I built

Two differentiable loss terms that approximate what a typical two-stage compressor does:

```
L_total = L_language_model + lambda_lz * L_dictionary_match + lambda_h * L_entropy
```

**The dictionary matching proxy** measures how similar nearby bytes are in the serialized weight stream. I compute a soft match score at power-of-2 lag distances (1, 2, 4, ... 512 bytes apart):

```python
for lag in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    diff_sq = (byte_stream[lag:] - byte_stream[:-lag]).square()
    match_score += torch.exp(-diff_sq / temperature).mean()
```

`exp(-diff^2/T)` is ~1 when two bytes are the same, ~0 when they're different. It's a smooth, differentiable version of "do these bytes match?" The gradient tells each weight which direction to move to create more repeated patterns. This works for any LZ-family compressor, not just zlib.

**The entropy proxy** builds a soft histogram of byte values using a Gaussian kernel, then computes Shannon entropy of that histogram. This is a standard technique. I included it because it covers the entropy coding half.

Both losses backpropagate through the quantization step using the straight-through estimator (STE). The STE is a trick where the forward pass does real int8 rounding, but the backward pass pretends the rounding didn't happen so gradients can flow through.

## What happened

### Debugging the dictionary matching proxy

The first version didn't work at all. I normalized byte values to [0, 1] before computing soft matches, which made all differences tiny. With temperature=1.0, even random bytes scored 0.89 similarity. The proxy couldn't tell structured data from noise.

Fixed it by using raw byte values [0-255] and temperature=50. At T=50, structured data scores 0.99 while random data scores 0.05. 21x discrimination, enough to give a useful gradient signal.

### Another bug on the GPU

First 8xH100 run crashed silently. torchrun swallowed the traceback. Had to run it with plain `python3` to see the actual error: `torch.Generator` on CUDA can't be used with `torch.randint`. Cost about $14 in wasted GPU time before I figured it out.

### 1xH100 results (5 runs, 600s each)

| Run | Config | BPB | Artifact | vs Control |
|-----|--------|-----|----------|-----------|
| Control | no CAT | 1.4374 | 12.32 MB | -- |
| Dict. match only | lz=0.01 | 1.4463 | 12.15 MB | -173 KB |
| Entropy only | h=0.1 | 1.4465 | 11.52 MB | -808 KB |
| Combined | lz=0.01, h=0.1 | 1.4465 | 11.48 MB | -842 KB |
| Entropy strong | h=1.0 | 1.5044 | 9.81 MB | -2.52 MB |

The entropy proxy does most of the work. At lambda=0.1 it saves 808 KB (6.6%) for only +0.009 BPB. The dictionary matching proxy adds 173 KB on its own and 34 KB on top of entropy. Smaller effect, but real.

At lambda=1.0 the entropy proxy saves 2.52 MB (20%) but BPB takes a bigger hit (+0.067). There's a tradeoff you can dial wherever you want.

## What this means

The control run produced a 12.32 MB artifact. CAT combined brought that down to 11.48 MB. That's 842 KB freed up.

842 KB is roughly 842K extra parameters in int8. For a dim=896 model, that's about one extra attention layer's worth of capacity. Whether that extra capacity offsets the 0.009 BPB cost is the open question. I didn't have enough compute budget to test the "reinvest the saved bytes into a wider model" experiment.

The entropy-strong run is more dramatic: 9.81 MB leaves 6.2 MB of headroom under the 16MB limit. That's a lot of room for a bigger model.

## What's new here

I searched for prior work on training neural network weights to be friendly to compression. Specifically for dictionary matching (the spatial pattern part, not just value distributions).

I didn't find anything. The closest things:

- Wiedemann 2018, CERWU 2025: use Shannon entropy (covers entropy coding only, misses dictionary matching)
- Deep Compression 2015: applies Huffman after training, not during
- Sandwiched Compression (Google 2024): differentiable proxy of a fixed codec, but for images going through JPEG, not neural net weights
- NuMuon 2026: nuclear norm constraints happen to help zstd (low rank = repeated patterns), but it's not designed for compression

The dictionary matching proxy via multi-lag autocorrelation is the new part. The entropy proxy is established. The combination, applied to neural network weight compression during training and targeting the LZ-family compressor structure that most tools share, is what I haven't seen before.

## Architecture

4 physical transformer layers looped 3 times (12 effective layers), with per-loop LoRA adapters (rank 16). Based on the Relaxed Recursive Transformers paper (Bae et al. 2024). dim=896, 14 attention heads, 2 KV heads (GQA). QAT fused with LR cooldown.

## Compute

Total spend: ~$18 across two sessions.

- March 18-19: Built the depth-recurrent architecture, ran 28 experiments. ~$15.
- April 4-5: Designed CAT, ran 20+ local experiments on MLX (free) and 6 H100 runs. ~$3.

## What I'd do with more compute

1. Train a wider model (dim=1024+) using the bytes saved by CAT, check if net BPB improves
2. Test CAT with int6 quantization (what the leaders use)
3. Run on 8xH100 for 3000+ steps, the compression effect compounds during training
4. Sweep dictionary matching temperature, maybe sharper matching helps

## References

- RFC 1951: DEFLATE specification
- Wiedemann et al. 2018. Entropy-Constrained Training. arXiv:1812.07520
- Conzelmann & Bamler 2025. CERWU. arXiv:2505.18758
- Han, Mao, Dally 2015. Deep Compression. arXiv:1510.00149
- Ullrich, Meeds, Welling 2017. Soft Weight-Sharing. arXiv:1702.04008
- Google 2024. Sandwiched Compression. arXiv:2402.05887
- Deletang et al. 2024. Language Modeling Is Compression. arXiv:2309.10668
- Bae et al. 2024. Relaxed Recursive Transformers. arXiv:2410.20672
- Bengio et al. 2013. Estimating Gradients Through Stochastic Neurons. arXiv:1308.3432
