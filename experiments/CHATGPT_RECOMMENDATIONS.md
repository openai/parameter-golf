Yes — on the **compression** side specifically, there are still a few paper-backed moves beyond your current `int6 + manual + zstd` stack.

The main research takeaway is that the next wins usually come from **making the byte stream more compressible**, not just lowering nominal bitwidth. Surveys of low-bit LLM compression consistently point to three levers that matter after basic quantization: **outlier handling, finer-grained/blockwise scaling, and vector/codebook quantization**.  [oai_citation:0‡arXiv](https://arxiv.org/html/2409.16694v2?utm_source=chatgpt.com)

For **your exact pipeline**, I’d rank the remaining ideas like this:

**1. Outlier splitting is the highest-ROI next step.**  
OWQ-style methods keep a tiny set of quantization-sensitive weights in higher precision and quantize the rest more aggressively; the whole point is that a small subset of “difficult” weights causes a disproportionate amount of quality loss under low-bit compression.  [oai_citation:1‡arXiv](https://arxiv.org/abs/2306.02272?utm_source=chatgpt.com)  
For you, that means: for each large matrix, store the top-magnitude or top-error weights separately in fp16, zero them in the bulk tensor, then quantize/compress the residual. This often helps **both** quality and compression because the residual matrix becomes less heavy-tailed and more regular, which is friendlier to downstream compression. That outlier-heavy failure mode is repeatedly highlighted in low-bit LLM literature.  [oai_citation:2‡arXiv](https://arxiv.org/html/2306.02272v4?utm_source=chatgpt.com)

**2. Move from per-row scales to blockwise scales.**  
Your current int6 path is per-row. Recent work on fine-grained/blockwise formats argues that a single row scale is often too coarse because distributions vary within a row; smaller blocks capture local statistics better and reduce quantization error. BlockDialect is a recent example centered on fine-grained mixed-format, block-level handling for exactly this reason.  [oai_citation:3‡arXiv](https://arxiv.org/pdf/2501.01144?utm_source=chatgpt.com)  
Practically: instead of one scale per row, try 32- or 64-element blocks. That can improve quality at the same nominal size, and it can also help compression if you serialize blocks in a regular layout.

**3. Reorder tensors by type before compression.**  
This is not a brand-new research novelty, but it follows directly from compression principles and from what the papers are exploiting with structured representations: more homogeneous streams compress better. Your current manual serializer sorts by key; that is deterministic, but not necessarily compression-optimal. A better order is all quantized payloads first, grouped by dtype/bitpacking scheme, then scales, then passthrough tensors. This is consistent with why additive/codebook and block methods do well: they create more structure in the representation.  [oai_citation:4‡arXiv](https://arxiv.org/abs/2401.06118?utm_source=chatgpt.com)  
This is one of the easiest things to try because it does not change model quality at all.

**4. Codebook / additive quantization is the most promising “bigger swing,” but it’s more invasive.**  
AQLM is the clearest paper here: it uses additive quantization with learned codebooks and is specifically motivated by extremely low-bit compression, around the 2–3 bits-per-parameter regime. The core idea is to encode groups of weights using combinations of codebook entries instead of scalar rounding each weight independently.  [oai_citation:5‡arXiv](https://arxiv.org/abs/2401.06118?utm_source=chatgpt.com)  
Why this matters for you: scalar int6 is simple, but codebooks can beat scalar quantization when you are really squeezing size. The downside is engineering complexity and decode cost. Under your constraints, I would not jump straight to full AQLM unless you are willing to change the export/load format more substantially.

**5. Mixed precision should be sensitivity-driven, not category-driven.**  
Right now your rule is basically “attn and mlp go int6, some named tensors stay fp16.” The literature keeps coming back to the fact that different layers and channels have very different sensitivity to compression, and that treating them unequally improves the size/quality tradeoff. Surveys and recent evaluations both emphasize that aggressive low-bit compression benefits from identifying the sensitive pieces rather than using one global rule.  [oai_citation:6‡arXiv](https://arxiv.org/html/2409.16694v2?utm_source=chatgpt.com)  
So instead of only name-pattern exemptions, use a cheap measured proxy: per-tensor reconstruction error, Hessian-ish proxy if you can afford it, or simply validation degradation from temporarily quantizing one tensor at a time.

What I would **not** prioritize for your setup:

- **Replacing zstd.** Your manual contiguous format plus zstd is already the right general direction; the remaining gains are more likely to come from making the serialized content easier for zstd to compress than from swapping compressors. The papers that win big are changing representation, not just changing containers.  [oai_citation:7‡arXiv](https://arxiv.org/abs/2401.06118?utm_source=chatgpt.com)  
- **Pure pruning/sparsity unless your runtime can exploit it cleanly.** Compression surveys cover pruning as a major family, but if you have to store indices and then densify at load time, the net artifact gain can be smaller than it first looks, especially for models already pushed into low-bit formats.  [oai_citation:8‡arXiv](https://arxiv.org/html/2308.07633v4?utm_source=chatgpt.com)

So, in blunt priority order for **compression**, I’d do this:

1. **Outlier splitting**
2. **Blockwise int6 scales**
3. **Serializer reordering by payload type**
4. **Sensitivity-based fp16/int8/int6 assignment**
5. **Only then** consider codebook/additive quantization

If you want the fastest next experiment, do **outlier splitting on your largest MLP/attention matrices plus grouping all `.q` blobs together before zstd**. That’s the most plausible “real gain without rewriting everything” move based on the literature.