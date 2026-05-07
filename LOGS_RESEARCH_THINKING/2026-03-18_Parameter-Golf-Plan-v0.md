# Winning parameter golf: information theory, literature, and roadmap

**The OpenAI Parameter Golf competition is a near-perfect instantiation of the rate-distortion tradeoff in information theory.** Given a rate budget of **128 million bits** (16,000,000 bytes), competitors must minimize distortion measured as bits-per-byte (BPB) on FineWeb validation data. The baseline — a 9-layer, 512-dim GPT with 1024 BPE vocabulary and tied embeddings — scores **1.2244 BPB**. A 4-hour extended run reached only 1.2074 BPB, confirming that model capacity, not training time, is the binding constraint. Neural scaling laws predict ~1.27 BPB for this parameter regime, so the baseline is already well-optimized for a vanilla architecture. Beating it substantially requires novel compression, architecture design, and test-time compute — the three axes where information-theoretic reasoning provides sharp guidance.

---

## Part 1: Formal problem statement

### The competition as a rate-distortion problem

Shannon's rate-distortion theory (Shannon 1959, "Coding Theorems for a Discrete Source with a Fidelity Criterion") defines the minimum rate needed to describe a source within a given distortion:

$$R(D) = \min_{p(\hat{x}|x): \mathbb{E}[d(x,\hat{x})] \leq D} I(X;\hat{X})$$

In Parameter Golf, the **rate** R is the artifact size: 16,000,000 bytes = **1.28 × 10⁸ bits**. This budget must encode everything — Python code, tokenizer specification, and zlib-compressed int8 model weights. The **distortion** D is the BPB on the validation set, measuring how well the artifact's implied probability model reconstructs the data distribution. The competition objective is thus:

$$\min_{M: |M| \leq R} \; D(M) = -\frac{1}{n}\sum_{i=1}^n \log_2 p_M(x_i | x_{<i})$$

where $M$ is the artifact (code + weights), $|M|$ is its byte count, $p_M$ is the probability distribution induced by the model, and $n$ is the number of bytes in the validation set. This is exactly the rate-distortion optimization: find the best "reconstruction" (predictive distribution) given a rate constraint (artifact size).

The rate-distortion function for a Gaussian source gives $D(R) = \sigma^2 \cdot 2^{-2R}$, implying exponentially diminishing returns from additional bits. For language modeling, the analogous relationship is Kaplan's scaling law $L(N) = (N_c/N)^{\alpha_N}$, which predicts power-law diminishing returns from additional parameters — consistent with the observation that 24× more training compute (4-hour vs. 10-minute baseline) improved BPB by only 0.017.

### MDL principle: the artifact IS the description

The Minimum Description Length principle (Rissanen 1978, "Modeling by Shortest Data Description") formalizes model selection as minimizing total code length:

$$L_{\text{total}} = L(\text{model}) + L(\text{data} | \text{model})$$

Parameter Golf is a **pure MDL problem** with the constraint $L(\text{model}) \leq 1.28 \times 10^8$ bits. The BPB metric is $L(\text{data}|\text{model})/n$, the per-byte cost of encoding data given the model. Competitors minimize $L(\text{data}|\text{model})$ subject to the model description length constraint. Grünwald's textbook treatment (2007, *The Minimum Description Length Principle*, MIT Press) and Barron, Rissanen & Yu (1998, "The Minimum Description Length Principle in Coding and Modeling," *IEEE Trans. Inf. Theory*) establish that this two-part code is asymptotically equivalent to Bayesian model averaging, normalized maximum likelihood, and prequential coding — all of which converge to the same effective model complexity penalty.

The key insight from MDL theory is that **every bit spent on the model description must earn its keep** by reducing data description length by more than one bit. A weight that occupies 8 bits in the artifact but improves predictions by less than 8 bits total across the validation set is wasted. This creates strong pressure toward aggressive quantization and parameter sharing.

### Prediction equals compression

Shannon's source coding theorem (1948, "A Mathematical Theory of Communication") establishes that a probabilistic model assigning $p(x_i | x_{<i})$ to the next symbol achieves code length $-\log_2 p(x_i | x_{<i})$ bits via arithmetic coding. **BPB is literally the compression rate achievable by the model.** Delétang et al. (2024, "Language Modeling Is Compression," ICLR) demonstrated this rigorously, showing that Chinchilla 70B compresses ImageNet to 43.4% of raw size, beating PNG at 58.5%.

Solomonoff (1964, "A Formal Theory of Inductive Inference") showed that the theoretically optimal predictor is the universal prior $m(x) = \sum_p 2^{-\ell(p)}$, summing over all programs producing $x$. The Parameter Golf artifact is literally a program (Python code + weights) whose description length upper-bounds the Kolmogorov complexity $K(\text{model})$. By Levin's coding theorem, $K(x) \approx -\log_2 m(x)$, directly linking program length to predictive probability. The competition thus asks: **what is the shortest program that best predicts web text?**

### Neural scaling laws at fixed N

Kaplan et al. (2020, "Scaling Laws for Neural Language Models," arXiv:2001.08361) established:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}$$

For ~16M non-embedding parameters, this predicts $L(N) \approx 3.25$ nats/token. Converting to BPB using ~3.7 bytes/token for a 1024-BPE tokenizer: BPB $\approx 3.25 / (3.7 \times \ln 2) \approx$ **1.27 bits/byte**. The baseline's 1.2244 BPB slightly outperforms this prediction, suggesting the baseline already exploits some efficiency gains (e.g., tied embeddings, GQA with 4 KV heads).

Hoffmann et al. (2022, "Training Compute-Optimal Large Language Models," NeurIPS — the Chinchilla paper) established that compute-optimal training uses ~20 tokens per parameter. For a 28M-parameter model, this gives an optimal training set of ~560M tokens — far below the ~14B tokens processable in 10 minutes on 8×H100. The model is **deep in the data-unlimited regime**, where capacity $N$ is the sole binding constraint. This means the competition is purely about **L(N) optimization**: maximizing predictive quality per bit of model description.

### Information-theoretic lower bounds on achievable BPB

The entropy rate of English provides the absolute floor below which no model — regardless of size — can achieve:

| Study | Method | Estimate |
|-------|--------|----------|
| Shannon (1951) | Human guessing game | **0.6–1.3 bpc** |
| Cover & King (1978) | Gambling estimate | **~1.25 bpc** |
| Brown et al. (1992) | Word trigram model | **1.75 bpc** (upper bound) |
| Takahashi & Tanaka-Ishii (2018) | Neural LM extrapolation | **1.12 bpc** |
| LLaMA-7B + arithmetic coding | LLMZip (Valmeekam et al. 2023) | **0.71 bpc** |

For ASCII-encoded English, 1 character = 1 byte, so bpc ≈ bpb. FineWeb contains web text with URLs, code snippets, and structural content that add entropy beyond clean prose. The entropy rate of FineWeb-like web text for an ideal infinite-capacity model is estimated at **0.8–1.0 bits/byte**. State-of-the-art frontier LLMs (>100B parameters) likely achieve **~0.5–0.7 BPB** on clean English text when model size overhead is ignored.

The gap between the baseline's **1.22 BPB** and the entropy floor of **~0.8–1.0 BPB** is the "capacity tax" — the information that a 16MB model simply cannot capture. Closing this gap requires packing more effective parameters into 16MB (via better compression) or extracting more predictive power per parameter (via better architectures).

---

## Part 2: Relevant literature and techniques

### Classic information theory foundations

**Shannon source coding theorem** (Shannon 1948): Expected code length $\geq H(X)$, with equality achievable via arithmetic coding. Any language model IS a compressor; BPB measures compression efficiency directly. **Cover & Thomas** (*Elements of Information Theory*, 2006) provide the canonical textbook treatment of rate-distortion theory, source coding, and entropy.

**MDL principle**: Rissanen (1978) introduced two-part codes; Barron, Rissanen & Yu (1998) connected MDL to universal coding and minimax risk; Grünwald (2007) unified MDL with normalized maximum likelihood. The competition's 16MB constraint is literally the "description length" budget.

**Kolmogorov complexity**: Kolmogorov (1965, "Three Approaches to the Quantitative Definition of Information") defined $K(x)$ as the length of the shortest program producing $x$. Li & Vitányi (2008, *An Introduction to Kolmogorov Complexity and Its Applications*, 3rd ed.) show that $\mathbb{E}[K(X)] \approx H(X)$ for ergodic sources. Grünwald & Vitányi (2004) established that every Shannon information inequality is also valid in Kolmogorov's theory.

### Optimal quantization theory

**Lloyd-Max quantizer** (Lloyd 1957/1982; Max 1960) is the optimal scalar quantizer minimizing MSE, using alternating centroid and midpoint conditions. For Gaussian sources, the distortion-rate function is $D(R) = \sigma^2 \cdot 2^{-2R}$ — each additional bit reduces MSE by ~6 dB.

**Vector quantization** (Gersho & Gray 1992) achieves gains over scalar quantization through shaping and space-filling. The E₈ lattice in 8 dimensions provides optimal sphere packing, exploited by QuIP# for weight quantization.

**Entropy-coded quantization** uses variable-length codes after quantization to approach the rate-distortion bound. zlib's DEFLATE (LZ77 + Huffman) naturally provides this for the competition's int8 format — weights with few distinct values or peaked distributions compress dramatically.

### Extreme quantization of neural networks

**BitNet** (Wang et al. 2023, "Scaling 1-Bit Transformers for Large Language Models," Microsoft): Binary weights {-1, +1} with 8-bit activations. Follows scaling laws similar to FP16 transformers; quality gap narrows with model size.

**BitNet b1.58** (Ma et al. 2024, "The Era of 1-bit LLMs"): Ternary weights {-1, 0, +1} requiring log₂(3) ≈ **1.58 bits per weight**. At 3B parameters, matches LLaMA in perplexity. At 70B: 4.1× faster, 7.2× less memory. **Critical finding**: 1.58-bit models match full-precision when hidden dimension is ~2× larger (Nielsen et al. 2024), suggesting a net ~2.5× gain in effective parameters per bit of description.

**GPTQ** (Frantar et al. 2023): Post-training quantization using approximate Hessian information. Quantizes weights column-by-column with error compensation: $\Delta W \propto H^{-1} \cdot \text{error}$. Achieves 3-4 bit quantization with negligible quality loss.

**AWQ** (Lin et al. 2024, MLSys): Activation-aware weight quantization. Key insight: <1% of weights are "salient" (connected to high-magnitude activations). Scales weights to protect salient channels. INT4 LLaMA-2-7B: perplexity 5.60 vs. 5.47 FP16.

**SqueezeLLM** (Kim et al. 2024, ICML): Sensitivity-based non-uniform quantization via weighted k-means clustering. Achieves lossless compression at 3-bit precision.

**QuIP#** (Tseng et al. 2024, NeurIPS): Randomized Hadamard Transform for incoherence processing + E₈ lattice codebooks for vector quantization. State-of-the-art at ≤4 bits/weight. **QTIP** (Tseng et al. 2024): Trellis-coded quantization approaches the rate-distortion bound — for Gaussian k=2, Lloyd-Max MSE=0.118, QuIP# E₈=0.089, QTIP 256D=0.069, theoretical minimum $D_R$=0.063.

**Deep Compression** (Han, Mao & Dally 2016, ICLR Best Paper): Prune → quantize → Huffman encode pipeline achieving **35-49× compression** with no accuracy loss on AlexNet/VGG-16. The zlib compression in Parameter Golf mirrors this quantize-then-entropy-code paradigm.

### Parameter-efficient architectures

**Weight tying** (Press & Wolf 2017, EACL, "Using the Output Embedding to Improve Language Models"): Sharing input/output embeddings reduces parameters by ~15% for small models while improving perplexity. Now standard practice.

**ALBERT** (Lan et al. 2020, ICLR): Cross-layer parameter sharing reduces transformer parameters by L-fold (where L = number of layers). ALBERT-large has **18× fewer parameters than BERT-large** (18M vs. 334M) while achieving competitive performance. Factorized embedding parameterization decomposes $V \times H$ into $V \times E + E \times H$ matrices.

**Universal Transformers** (Dehghani et al. 2019, ICLR): Depth recurrence — applying the same transformer block T times. With adaptive computation time (ACT, Graves 2016), different tokens receive different computational depth. Turing-complete, unlike fixed-depth transformers. **Huginn-3.5B** (Geiping et al. 2025) scales this idea: 2 prelude + 4 recurrent + 2 coda blocks; at R=128 recurrence passes, a 3.5B-parameter model achieves performance equivalent to **~50B parameters** of compute on reasoning benchmarks.

**Low-rank factorizations**: Li et al. (2018, ICLR, "Measuring the Intrinsic Dimension of Objective Landscapes") showed that neural network objective landscapes have intrinsic dimensions far below nominal parameter count — MNIST needs only ~750 dimensions for a 200K-parameter network. LoRA (Hu et al. 2022, ICLR) exploits this: rank r=1 or r=2 suffices for fine-tuning GPT-3 175B.

**Lottery Ticket Hypothesis** (Frankle & Carlin 2019, ICLR): Dense networks contain sparse subnetworks at 10-20% of original size that train to full accuracy. Arora et al. (2018, ICML, "Stronger Generalization Bounds for Deep Nets via a Compression Approach") proved that compressed networks generalize well, with effective parameters orders of magnitude below nominal count.

### Tokenizer design and BPB efficiency

BPB decomposes as: BPB = (tokens per byte) × (cross-entropy per token) / ln(2). Larger vocabularies reduce tokens per byte but make each token prediction harder (higher entropy over more classes). **Tao et al. (2024, NeurIPS, "Scaling Laws with Vocabulary")** showed that optimal vocabulary size depends on model size — most LLMs use too-small vocabularies. For a ~16M parameter model, the optimal range is **512–2048 tokens**. The baseline's 1024 BPE vocab appears near-optimal: the embedding matrix (1024 × 512 ≈ 524K params with tied weights) is only ~2% of total parameters, leaving the budget for transformer layers.

### Training efficiency

**Chinchilla** (Hoffmann et al. 2022): Optimal ratio is ~20 tokens per parameter. A 28M-parameter model needs only ~560M tokens, but the 10-minute 8×H100 budget allows ~14B tokens — **25× the Chinchilla-optimal amount**. This extreme data surplus means the model can train to near-convergence; the bottleneck is purely capacity.

**muP** (Yang et al., "Tensor Programs V"): Maximal update parameterization enables zero-shot hyperparameter transfer across scales. Cerebras/EleutherAI demonstrated: 111M proxy → 3B target achieved 7B-comparable performance with 3.3× less FLOPS.

**WSD schedule** (Wen et al. 2024, arXiv:2410.05192): Warmup-Stable-Decay. Unlike cosine, doesn't require pre-specifying total compute. The "river valley" theory explains that high LR propels the optimizer along the loss valley; rapid decay quells oscillations. 20% cooldown fraction is effective (Hägele et al. 2024).

**Muon optimizer**: A variant of SGD with momentum using Newton-Schulz orthogonalization, central to the NanoGPT speedrun records. Massive speedup over AdamW for transformer hidden layers. Combined with heterogeneous batch sizes (different update frequencies per parameter group).

### NanoGPT speedrunning and related work

The modded-nanogpt challenge (github.com/KellerJordan/modded-nanogpt) trains a 124M-parameter model to 3.28 cross-entropy in under **90 seconds** on 8×H100 using: Muon optimizer, partial RoPE, QK-norm, ReLU², sigmoid-gated skip connections, document-aligned batching, progressive attention window expansion, and FP8 compute. Key architectural innovations include the **smear module** (cheap bigram inductive bias: token + 0.07×prior_token), **sparse attention gate** (sigmoid gate per head), and **backout module** (model can "back out" 50% of early-layer contributions before prediction). These techniques represent state-of-the-art training efficiency for small transformers.

---

## Part 3: Implementation roadmap

### Tier 1 — highest ROI modifications (expected 0.02–0.05 BPB improvement each)

**1. Aggressive quantization-aware training (QAT) to maximize effective parameters.** The baseline stores ~28M int8 parameters compressed under 16MB. The single highest-leverage move is switching to **4-bit or ternary QAT** to roughly double or triple the effective parameter count within the same 16MB. At 4-bit effective precision stored as int8, zlib compresses the 16 distinct values efficiently, allowing ~35-40M parameters. At ternary (1.58-bit, BitNet b1.58 style), three values {-1, 0, 1} stored as int8 compress ~4-5× under zlib, allowing ~50-70M parameters. However, ternary models need ~2× wider hidden dimensions to match full-precision quality, yielding a net ~2.5× effective gain. **Recommended approach**: train with 4-bit QAT using straight-through estimator, targeting ~35M parameters. Design weight distributions explicitly for zlib compressibility — peaked, few distinct values, structured patterns that LZ77 exploits.

**2. Cross-layer parameter sharing (ALBERT-style depth recurrence).** Share transformer block parameters across all layers, then iterate the block L times. This converts the parameter budget from $N = 12L \cdot d^2$ (L unique layers) to $N = 12 \cdot d^2$ (one shared layer iterated L times). For the same 16MB, this enables either a much wider model (larger d) or the same model with dramatically more effective depth. Huginn-3.5B showed that a recurrent-depth 3.5B model performs like a 50B model — an **~14× amplification** of effective parameters. For Parameter Golf, sharing all layers and iterating 9-18 times should significantly outperform 9 unique layers. ALBERT showed only ~1.5-point average degradation from full sharing. The key is adding **per-iteration features** (e.g., iteration-dependent position encodings, learnable iteration gates) so the model can differentiate its behavior across depths without unique parameters per layer.

**3. NanoGPT speedrun architecture improvements.** Adopt the proven techniques from the modded-nanogpt record: **Muon optimizer** for hidden layers (with AdamW for embeddings), **partial RoPE** on 50% of head dimensions, **QK-norm** for training stability, **ReLU²** activation, **smear module** for cheap bigram context, **document-aligned batching** for lower gradient variance, and **WSD learning rate schedule** with 20% linear cooldown. These are battle-tested for small GPTs and can be implemented with relatively modest engineering effort.

### Tier 2 — moderate ROI (expected 0.01–0.03 BPB each)

**4. Optimal vocabulary size tuning.** The baseline uses 1024 BPE tokens. Systematically sweep 512, 1024, 2048, and 4096 with the same total parameter budget to find the BPB-optimal vocabulary. For a ~16M non-embedding model, theoretical analysis suggests 1024–2048 is near-optimal, but the optimum depends sensitively on the compression scheme (larger vocab = larger embedding matrix = more bytes in the artifact). Consider **byte-level modeling** (V=256) with a deeper/wider transformer — this eliminates tokenizer overhead entirely and is "tokenizer-agnostic" by definition.

**5. Factorized embeddings and attention.** If vocabulary size increases beyond 1024, factorize the embedding matrix: $V \times d = V \times E + E \times d$ with $E \ll d$. This was critical for ALBERT with V=30K but less impactful at V=1024. More impactful at this scale: **grouped query attention** (GQA) with aggressive head reduction (the baseline already uses 4 KV heads). Consider **multi-head latent attention** (DeepSeek-V2 style) where keys and values are projected through a low-rank bottleneck.

**6. Training data and schedule optimization.** With ~14B tokens processable in 10 minutes, the model sees each of the 10B training tokens ~1.4 times. Strategies: (a) curate a smaller, higher-quality subset and train multiple epochs; (b) use curriculum learning with progressively harder examples; (c) increase sequence length during training (start short, end long) to maximize tokens/second early while building long-range capacity later; (d) heterogeneous batch sizes per parameter group (larger batches for embedding updates, smaller for hidden layers).

### Tier 3 — exploratory and high-variance (potentially large improvements)

**7. Test-time compute via depth recurrence.** The competition FAQ states evaluation must complete in <10 minutes on 8×H100 but "you're free to evaluate however." This opens the door to **test-time depth recurrence**: run the shared transformer block many more times at inference than at training. A model trained with 9 iterations could be evaluated with 18-36 iterations, spending extra FLOPS for better predictions. The key requirement is that the recurrent architecture must be trained to benefit from additional iterations — using ACT (adaptive computation time) or simply training with variable iteration counts.

**8. Test-time training (TTT).** Adapt the model on the validation sequence itself using self-supervised learning (next-token prediction on preceding context). Sun et al. (2024) showed TTT layers can match models 10× larger with quality neighbors. The competition allows arbitrary evaluation methods as long as no training data is accessed — TTT on the validation context is arguably legal since you're learning from the test data itself, but this must be verified against competition rules. If allowed, TTT could provide substantial BPB improvement by specializing the model to the specific validation distribution.

**9. Non-transformer architectures.** Mamba (Gu & Dao 2023) showed that state-space models match transformers 2× their size on language modeling — suggesting a ~2× parameter efficiency advantage. However, at small scales (<30M params) the advantage is less clear, and the tooling ecosystem is less mature. RWKV offers similar benefits with better training parallelism. A **hybrid architecture** (attention for short-range, SSM for long-range) could capture the best of both worlds but adds implementation complexity. Recommended only for experienced practitioners.

**10. Novel weight encoding beyond int8.** Instead of storing int8 values and relying on zlib, consider custom encoding schemes within the artifact's code: (a) **codebook quantization** — store a codebook of 16-256 centroids and index weights with 4-8 bit codes; (b) **arithmetic coding** of weight distributions (more compact than zlib's DEFLATE for known distributions); (c) **weight generation** — use a small neural network or hash function to generate some weights deterministically from a seed, reducing stored parameters. The code bytes are part of the 16MB budget, so the encoding/decoding logic must be compact.

### Quantitative parameter budget analysis

| Strategy | Bits/weight | zlib ratio | Effective params in 16MB | Expected BPB |
|----------|------------|------------|--------------------------|--------------|
| Baseline (int8) | 8 | ~1.3× | ~25-28M | ~1.22 |
| 4-bit QAT | 4 | ~2× | ~35-40M | ~1.16–1.19 |
| Ternary QAT | 1.58 | ~4-5× | ~50-70M (but 2× wider needed) | ~1.12–1.18 |
| Layer sharing + 4-bit | 4 | ~2× | ~35M (×9-18 iterations) | ~1.10–1.15 |
| Layer sharing + ternary | 1.58 | ~4-5× | ~60M (×9 iterations) | ~1.05–1.12 |

### Optimal architecture sketch

Based on the literature, the most promising competition entry combines:

**Architecture**: A single transformer block with d_model=384-512, iterated 12-18 times (depth recurrence with learned iteration embeddings). SwiGLU FFN with d_ff = 2.67×d_model. 4-8 attention heads with 2-4 KV heads (GQA). RoPE on 50% of head dimensions. QK-norm and RMSNorm. Smear module for bigram context. The block has ~1.5-2M unique parameters, but when iterated 12-18 times, provides effective depth equivalent to a 12-18 layer model.

**Quantization**: 4-bit QAT with straight-through estimator. 16 non-uniform quantization levels determined by weighted k-means on trained weight distribution. Store as int8 values in {0,...,15} for maximum zlib compressibility. Alternatively, ternary {-1, 0, 1} with 2× wider model if quality holds.

**Tokenizer**: 1024-2048 BPE. With tied embeddings, the vocabulary cost is manageable: 2048 × 512 = 1M parameters = 1MB at int8 (well under budget).

**Training**: Muon optimizer for transformer weights, AdamW for embeddings. WSD schedule: 5% warmup, 75% stable, 20% linear cooldown. FP8 training for ~2× throughput. Document-aligned batching. Process ~10-20B tokens in 10 minutes on 8×H100.

**Evaluation**: Iterate the recurrent block 2-3× more times than during training (e.g., train at 12 iterations, evaluate at 24-36). If allowed, apply 1-step TTT on each validation context window.

### Theoretical limits and what to expect

The Kaplan scaling law predicts ~1.27 BPB for the baseline's parameter count. The baseline already beats this at 1.2244, suggesting the tied embeddings and GQA provide meaningful efficiency gains. With the aggressive compression + architecture strategy outlined above, the effective parameter count could reach 50-100M equivalent, which scaling laws predict would achieve **~1.10–1.15 BPB**. Adding test-time compute (depth recurrence at inference) could push toward **1.05–1.10 BPB**. The hard floor set by the entropy rate of web text (~0.8–1.0 BPB) and the 16MB rate constraint together suggest that **~1.0 BPB is the approximate limit** of what's achievable in this competition, with diminishing returns below ~1.10.

The winning strategy is not any single technique but the **multiplicative composition** of compression efficiency (more parameters per bit), architectural efficiency (more predictive power per parameter), training efficiency (more learning per FLOP), and test-time compute (more inference per prediction). Each axis offers 5-15% improvement; composed, they could yield 20-40% BPB reduction from the baseline, targeting a final score in the **1.05–1.15 BPB** range.