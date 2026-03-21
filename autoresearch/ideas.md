Simon, my highest-confidence take is this: **don’t optimize Parameter Golf like “tiny LM pretraining”; optimize it like “online compression with a tiny learned prior.”** The challenge is a 16MB self-contained artifact, trained in under 10 minutes on 8xH100s, scored by tokenizer-agnostic validation bits-per-byte on FineWeb; the repo explicitly points at test-time compute/training, long context, parameter tying, and novel tokenizers as intended directions. The public baseline is 1.2244 val_bpb, and a 4-hour non-record baseline is already at 1.2074, so the starter GPT is clearly not near the frontier. The metric is literally token cross-entropy converted to bits/token and then scaled by tokens/byte, so both **prediction quality** and **representation granularity** matter. ([GitHub][1])

My top three buckets for agent time are: **online context mixing / fast adaptation**, **learned segmentation instead of fixed tokenization**, and **native storage-aware models rather than post-hoc compression**. The winning system may look less like “a better 9-layer GPT” and more like “a tiny base model wrapped in a clever compressor.”

1. **PAQ-ify the evaluator.**
   The biggest near-term EV is to treat the base LM as one expert inside an **online context mixer**: combine LM logits with a token n-gram expert, a byte n-gram expert, a copy/pointer expert, and a cheap adaptive bias head that updates on the validation prefix. Classical PAQ/cmix-style compressors work by mixing many specialized predictors, and the recent neural compressor Nacrith gets its gains from the same pattern: transformer prior + n-gram + online correction. In Parameter Golf this is especially attractive because the artifact cap is code + compressed weights, while evaluation methods are otherwise allowed to be aggressive within the time budget, so **online state built from already-seen validation text is unusually cheap leverage**. ([Matt Mahoney's Home Page][2])

2. **Dynamic evaluation / fast weights.**
   Old dynamic-evaluation papers already showed that updating a model on the test prefix can materially improve compression metrics, including on Hutter Prize-style character benchmarks, and newer test-time-training work reframes this as using small fast weights as transient memory. For this contest, I would not update the whole model; I’d update a tiny subset: output biases, layernorm gains, per-head temperatures, or a rank-1/rank-2 low-rank delta on the LM head. This is one of the cleanest ways to convert extra eval-time compute into lower val_bpb. ([arXiv][3])

3. **Copy, cache, and retrieve from the prefix.**
   Small LMs are especially bad at repeated names, URLs, boilerplate, and long-tail strings; cache and pointer mechanisms are exactly for that. Continuous cache, unbounded cache, pointer-sentinel mixtures, and Memorizing Transformers all show that recent hidden-state memory or copy distributions help language modeling without needing many more stored parameters. On web text, this feels like a very strong fit. ([arXiv][4])

4. **Move from fixed tokenization to learned or dynamic segmentation.**
   A lot of the best recent work is basically saying that static vocabularies are a bad place to freeze compression decisions. BLT uses entropy-based byte patching, H-Net learns dynamic chunking end-to-end, MrT5 learns to merge/delete tokens to shorten sequences, and SpaceByte gets mileage from allocating larger blocks around spaces. For Parameter Golf, the practical version is probably not “full pure byte model everywhere,” but rather **tiny local byte model + learned downsampling/patching + global latent model**. Tokenizer work is high-upside here, but the repo also says tokenizer changes will be scrutinized carefully, so exact bpb accounting matters. ([arXiv][5])

5. **Do tokenizer search against the real objective, not BPE frequency.**
   This one is my own synthesis: standard BPE chooses merges by frequency, but your real objective is **loss drop per added artifact byte**. Each extra token costs embedding/output storage, so candidate merges should be scored by something like `expected Δbpb / added compressed bytes`, not raw count. I would explicitly search for web-structure-heavy merges like URL fragments, HTML delimiters, indentation/newline patterns, and common multi-byte substrings that shorten sequences *and* make the next prediction easier.

6. **Train native compressed models, not just dense models that you squeeze afterward.**
   The baseline script already does a decent post-train trick: int8 quantization with per-row scales, then zlib compression. The next step is native low-bit or codebookized training: BitNet-style 1.58-bit/ternary matrices, shared codebooks across layers, or AQLM-like additive quantization. This is directly relevant because the contest limit is compressed artifact bytes, not nominal parameter count, and there is now evidence that 1.58-bit training can work even on relatively small networks. ([GitHub][6])

7. **Optimize for zlib, not just for loss.**
   Another contest-specific angle: the submission is scored on **actual compressed model bytes**, so learn weights that are easy for the submission codec to compress. Deep Compression and AQLM already point toward weight sharing and codebooks; I’d go further and make the training aware of the export path: shared codebooks across layers, tied scales, delta-coded repeated blocks, row/column reordering, and regularizers on code-index entropy. This is weird, but very on-theme. ([GitHub][1])

8. **Use recurrent depth / parameter sharing aggressively.**
   ALBERT-style sharing, Universal Transformer recurrence, and newer recursive-transformer work are almost tailor-made for a storage-constrained contest: you buy depth with compute instead of bytes. My guess is that a serious contender will have **1–3 unique blocks** reused many times, plus a tiny untied budget for per-depth scalars, norms, or LoRA-style adapters so the model does not collapse into oversharing. This is also explicitly aligned with the challenge README’s “aggressive parameter tying” and “depth recurrence” suggestions. ([OpenReview][7])

9. **Exploit the permissive eval regime with longer context and compressed memory.**
   The FAQ explicitly allows evaluation at any sequence length and encourages aggressive evaluation methods, as long as they fit the eval-time limit. That strongly suggests trying chunkwise recurrent memory, compressed summaries, or retrieval over earlier hidden states even if the model trains at 1k context. Compressive Transformer, Memorizing Transformer, and Infini-attention are all templates for that. ([GitHub][1])

10. **Try hybrid recurrent backbones, especially for byte-ish models.**
    Mamba, RetNet, and Gated Linear Attention all exist because vanilla attention is not the only way to buy long-range modeling, and MambaByte specifically argues that byte-level modeling is a good fit for recurrent/state-space approaches. A hybrid I’d actually test is: cheap recurrent/linear backbone for most layers, sparse/global attention every few layers, plus the cache/pointer machinery above. That feels more promising than a pure tiny GPT if you go closer to bytes. ([arXiv][8])

11. **Adaptive compute over token positions is probably underexploited here.**
    CALM, Mixture-of-Depths, Adaptive Span, and SpaceByte all exploit the same fact: easy tokens do not deserve the same compute as hard tokens. In this contest that can show up as early exit, entropy-gated extra recurrence, larger blocks only around spaces/newlines/high-surprisal regions, or token skipping/merging after a few cheap local layers. It is a nice way to trade eval or train compute for bpb without paying many extra bytes. ([arXiv][9])

12. **Distillation is still a hammer.**
    Even though it’s obvious, I would not ignore it: distillation remains one of the best ways to make small models less stupid. MiniLLM argues that reverse-KL/on-policy KD is better suited to generative LMs than vanilla KD, so a practical contest version is teacher supervision only on hard spans or top-k logits, not full dense targets everywhere. I’d treat this as especially strong for the unlimited-compute / non-record track; for record-track submissions, the README leaves “external compute” intentionally fuzzy and reserves discretion. ([arXiv][10])

A few **high-upside weird bets** I’d absolutely seed:

* **Document-type experts**: route between prose / code / HTML / URL-heavy modes with tiny calibration heads.
* **Compression-aware merge search**: not BPE, but merges picked by estimated bpb gain per artifact byte.
* **Rank-1 online bias adapters**: per-document or sliding-window deltas on the output head.
* **Sparse MoE with shared quantized codebooks**: risky, but recent sparse-scaling work suggests larger sparse models can beat smaller dense ones at fixed compute if you can make the storage work. ([arXiv][11])

If I were wiring agents tonight, I’d run three queues in parallel: **(a) eval-time compressor tricks**—mixtures, cache, fast weights; **(b) representation tricks**—dynamic segmentation, byte-latent hybrids, compression-aware tokenizers; and **(c) storage-native models**—ternary/codebook weights, recurrent shared-depth blocks, zlib-aware export. My guess is the winner comes from **combining one idea from each queue**, not from maxing out any single one.

[1]: https://github.com/openai/parameter-golf "https://github.com/openai/parameter-golf"
[2]: https://mattmahoney.net/dc/paq.html "https://mattmahoney.net/dc/paq.html"
[3]: https://arxiv.org/abs/1709.07432 "https://arxiv.org/abs/1709.07432"
[4]: https://arxiv.org/abs/1612.04426 "https://arxiv.org/abs/1612.04426"
[5]: https://arxiv.org/abs/2412.09871 "https://arxiv.org/abs/2412.09871"
[6]: https://raw.githubusercontent.com/openai/parameter-golf/main/train_gpt.py "https://raw.githubusercontent.com/openai/parameter-golf/main/train_gpt.py"
[7]: https://openreview.net/pdf?id=H1eA7AEtvS "https://openreview.net/pdf?id=H1eA7AEtvS"
[8]: https://arxiv.org/abs/2312.00752 "https://arxiv.org/abs/2312.00752"
[9]: https://arxiv.org/abs/2207.07061 "https://arxiv.org/abs/2207.07061"
[10]: https://arxiv.org/abs/1503.02531 "https://arxiv.org/abs/1503.02531"
[11]: https://arxiv.org/html/2501.12370v2 "https://arxiv.org/html/2501.12370v2"
