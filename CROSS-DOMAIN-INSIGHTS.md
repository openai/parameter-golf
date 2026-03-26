# Cross-Domain Insights for Building Better Small Language Models

**Research Date:** 2026-03-24
**Purpose:** What can fields OUTSIDE of AI teach us about building more efficient small language models?

---

## 1. Information Theory & Compression — The Theoretical Floor

### The Core Idea
Claude Shannon's 1951 paper estimated English text has **between 0.6 and 1.3 bits of entropy per character** (bpc). This is the theoretical minimum — the absolute least information needed to perfectly predict/represent English. Current large language models achieve roughly **0.8–1.0 bpc** on benchmark text (GPT-2 era models hit ~0.9 bpc; modern LLMs push closer to 0.8 bpc).

### Why This Matters for Small Models
- **The gap is small.** We're already within striking distance of Shannon's upper bound (~1.3 bpc). For a small model, chasing the last 0.1–0.2 bpc is enormously expensive in parameters. This tells us: **diminishing returns are real and quantified**.
- **Per-character entropy is the wrong unit for practicality.** Shannon's estimate treats all characters equally. Real LMs tokenize, and token-level entropy depends heavily on vocabulary size. A small model with a good tokenizer can match a large model's effective compression without matching character-level entropy.
- **The gap between 1.3 and 0.6 bpc is "world knowledge."** The lower bound (0.6) assumes perfect prediction given unlimited context and knowledge. The upper bound (1.3) assumes limited context. A small model will naturally sit closer to the upper bound — and that's fine.

### Actionable Techniques
- **Use compression (bits per byte) as your primary training metric**, not just loss. This connects directly to Shannon limits and gives you a universal benchmark.
- **Measure your model against PPM compressors** as a baseline. If your model can't beat gzip (~2.5 bpc English), it's not learning real language structure.
- **Design your tokenizer to minimize byte-level entropy** — byte-pair encoding variants that target the Shannon limit directly.

---

## 2. Signal Processing — Sparse Representations & Multi-Scale Analysis

### The Core Idea
Signal processing has spent 70+ years solving one problem: **how to represent information efficiently**. Three key ideas transfer directly:

- **Sparse Coding:** Represent signals using a small number of active elements from a large dictionary. The brain does this. Compression algorithms do this. Neural networks mostly don't.
- **Wavelets:** Represent signals at multiple scales simultaneously. A wavelet decomposition lets you see both the forest AND the trees without paying for a full-resolution representation of both.
- **Compressed Sensing (Donoho 2006):** You can recover a signal from far fewer measurements than traditional theory says, IF the signal is sparse in some basis. The math: if a signal has only k non-zero components in some basis, you need only O(k log n) measurements, not n.

### Why This Matters for Small Models
- **Transformers use dense representations everywhere.** Every activation in every layer is a dense float vector. Signal processing says: **this is wasteful**. Most information in a hidden state is concentrated in a few dimensions.
- **LISTA (Learned ISTA, Gregor & LeCun 2010)** unrolled sparse coding into a neural network and showed you can learn sparse representations end-to-end. This is directly applicable to transformer hidden states.
- **Multi-scale analysis is exactly what attention should do**, but in a more principled way. Wavelet-style hierarchical decomposition could replace some attention heads.

### Actionable Techniques
- **Enforce sparsity in hidden states** — add an L1 penalty or use top-k activation in FFN layers. A model with 50% sparse activations runs 2x faster with minimal quality loss.
- **Experiment with wavelet-style positional encodings** that capture both local and global structure at different scales, instead of standard sinusoidal/RoPE embeddings.
- **Use compressed sensing for weight pruning** — don't just prune to zero; learn a sparse basis where pruned weights can be reconstructed from remaining ones.

---

## 3. Neuroscience — How the Brain Runs Language on 20 Watts

### The Core Idea
The human brain processes language with roughly **20 watts of power** — less than a light bulb. A modern GPU running a 70B parameter model uses 300-700 watts. The efficiency gap is staggering. Neuroscience offers three key insights:

- **Predictive Coding (Rao & Ballard 1999, Friston 2010):** The brain doesn't process raw sensory input bottom-up. Instead, higher cortical areas generate predictions, and lower areas only transmit the **prediction errors** — the surprise. This means most neural activity is about what's ALREADY EXPECTED, not new information.
- **Sparse Distributed Representations (SDRs):** At any moment, only ~1-4% of cortical neurons are active. Information is encoded not in which neurons are active, but in the **pattern of sparse co-activation**. This is maximally efficient for both energy and storage.
- **Neuromorphic Timing:** Biological neurons communicate via spikes — events, not continuous values. A neuron that fires 5 Hz processes information with 5 discrete events per second. This is inherently sparse in time.

### Why This Matters for Small Models
- **Transformers do the opposite of predictive coding.** A transformer processes full input through all layers. The brain only processes the **residual** — what it didn't predict. A small model that only computes "surprise" tokens could be dramatically more efficient.
- **Sparse activations mean fewer FLOPs.** If only 5% of neurons fire, you do 20x fewer multiplications. For a small model, this is the difference between running on a phone and requiring a GPU.
- **The brain's language areas (Broca's, Wernicke's) use < 5% of cortex.** Language is modular in the brain — not because it was designed that way, but because specialization is efficient. Small models should be similarly specialized.

### Actionable Techniques
- **Implement a "prediction residual" architecture** — use a cheap baseline predictor (like a small RNN or n-gram) to generate easy predictions, and have the transformer only refine the hard tokens. This is essentially what Mixture-of-Experts does for capacity, but for compute.
- **Use Top-K or Top-P sparsity in every layer** — force activations to be sparse during training, not just inference. Train with straight-through estimators for hard sparsity.
- **Add a "surprise gate"** — compute a per-token difficulty score (using a small auxiliary network), and only route "surprising" tokens through expensive layers. Cheap tokens skip ahead.

---

## 4. Computational Linguistics Pre-Deep-Learning — What We Forgot

### The Core Idea
Before transformers, NLP used techniques that were computationally cheaper and sometimes more interpretable:

- **N-gram models with Kneser-Ney smoothing:** Statistical language models from the 1990s achieved remarkable performance with essentially a lookup table plus clever interpolation. Kneser-Ney smoothing handles the "I saw a red..." problem (red what?) elegantly.
- **Hidden Markov Models (HMMs):** The workhorse of speech recognition and MT until ~2012. They're linear-time, interpretable, and surprisingly competitive on narrow domains.
- **IBM Models for Statistical MT:** Word alignment models that decomposed translation into fertility, translation probability, and distortion — all with closed-form training (EM algorithm).
- **PPM (Prediction by Partial Matching):** A compression algorithm that achieves near-Shannon-limit compression using variable-order context modeling. PPM achieves ~2.0 bpc on English — competitive with early neural LMs.

### Why This Matters for Small Models
- **Classical models had inductive biases that small data loves.** N-grams assume locality. HMMs assume hidden states. These biases REDUCE the amount of data needed. Transformers have almost no inductive bias — they must learn everything from data.
- **Hybrid architectures could combine classical efficiency with neural flexibility.** A small model doesn't need to do everything with attention. Use an n-gram or HMM for the easy local stuff, and save neural compute for the hard long-range stuff.
- **PPM is basically a language model that runs in O(n) time and O(context) memory.** It's worth benchmarking against.

### Actionable Techniques
- **Build a two-stage system:** A fast classical model (n-gram with Kneser-Ney or PPM) makes first-pass predictions. A small transformer only processes tokens where the classical model is uncertain. This is a form of **adaptive compute**.
- **Use classical word alignment models** for bilingual data preparation — they're faster and often better than neural aligners for phrase-level alignment.
- **Borrow the EM algorithm's philosophy:** unsupervised structure discovery. Use EM to find latent topics or word classes before training your model, giving it a head start.
- **N-gram features as additional input channels** — feed character n-gram embeddings alongside subword token embeddings. This gives the model cheap access to morphological patterns without needing to learn them from scratch.

---

## 5. Lossless Compression Algorithms — The Prediction-Compression Duality

### The Core Idea
There's a fundamental theorem: **compression = prediction**. A perfect compressor is a perfect predictor, and vice versa. Every lossless compression algorithm is secretly a language model:

- **LZ77/LZMA (gzip, 7-zip):** Dictionary-based. "I saw this substring 200 bytes ago." Good for repetitive text, bad for novel sequences. ~2.5 bpc on English.
- **BWT (bzip2):** Sorts the input to group similar contexts together, then applies move-to-front coding + Huffman. ~2.3 bpc on English.
- **PPM (paq, ccm):** Variable-order context modeling with exclusion. The gold standard for compression quality. ~1.7 bpc on English. Essentially a variable-n-gram model.
- **ANS (Asymmetric Numeral Systems):** The modern workhorse (used in zstd, Lizard). Combines the compression efficiency of arithmetic coding with the speed of table-based decoding. This is how neural network probability distributions can be converted into bitstreams.
- **Brotli:** Google's format using a combination of LZ77, Huffman coding, and 2nd-order context modeling. ~2.2 bpc.

### Why This Matters for Small Models
- **PPM at 1.7 bpc vs GPT-2 at 0.9 bpc — the neural advantage is real but not free.** Neural models achieve ~0.8 bpc improvement over classical methods, but at enormous computational cost. A small model should aim to close this gap efficiently.
- **ANS is how you connect neural predictions to actual compression.** If you can output ANS-encoded tokens from your model's probability distributions, you get both generation AND compression in one system.
- **The hierarchy of compressors tells you where the easy wins are.** Going from gzip (2.5) to PPM (1.7) is "free" — it's just better algorithms. Going from PPM (1.7) to neural (0.9) costs parameters. A small model should at minimum beat PPM.

### Actionable Techniques
- **Use zstd-style preprocessing** on your training data. BWT-transform or delta-encode text before tokenization to reduce the "easy" redundancy, letting your model focus on the "hard" semantic compression.
- **Implement ANS encoding as your generation head.** Instead of softmax → sample, do softmax → ANS encode. This gives you a provably optimal (in the information-theoretic sense) way to convert probabilities to bits.
- **Benchmark your model as a compressor.** Feed it text, extract the cross-entropy as a bitstream via ANS, and compare directly to zstd, PPM, and gzip. If your model can't beat zstd on domain-specific text, it's not learning useful representations.
- **Learn from LZMA's "optimal parsing"** — when generating text, look ahead to find the best continuation, not just the most likely next token. This is beam search, but LZMA has been optimizing this for decades.

---

## 6. Complexity Theory — How Big Does a Model Need to Be?

### The Core Idea
Statistical learning theory tries to answer: **what's the minimum model size (parameters, architecture) needed to learn a given task?**

- **VC Dimension:** Measures the "capacity" of a model class — how many data points it can shatter (perfectly classify). A linear classifier in d dimensions has VC dimension d+1. A neural network with W weights has VC dimension O(W log W). Key insight: **the VC dimension tells you how much data you need to train, not how good the model will be**.
- **Rademacher Complexity:** A data-dependent measure of model capacity. How well can your model fit random noise? If Rademacher complexity is high, your model can memorize anything — which means it can overfit. The generalization bound: generalization error ≤ training error + O(Rademacher complexity / √n).
- **PAC Learning (Probably Approximately Correct):** To learn a concept class of VC dimension d to accuracy ε with probability 1-δ, you need O(d/ε · log(1/ε) + log(1/δ)/ε) examples. For a neural network with W parameters, this means you need O(W log W) training examples.

### Why This Matters for Small Models
- **These bounds are LOOSE for neural networks.** The PAC/VC bounds say you need millions of examples for a million-parameter model. In practice, transformers generalize far better than theory predicts. This is the "double descent" mystery — but it means **you can go smaller than theory suggests**.
- **The bounds DO tell you the direction of improvement:** more parameters → more data needed → more compute. A small model with good inductive biases has effectively lower VC dimension, meaning it needs less data and generalizes better.
- **Rademacher complexity gives you a principled regularization strategy.** Instead of arbitrary dropout rates, measure your model's ability to fit noise and regularize to match.

### Actionable Techniques
- **Use Rademacher complexity as a diagnostic.** During training, measure how well your model fits shuffled labels. If it fits them too well, increase regularization — you're over-parameterized for your data.
- **Design architectures that are "capacity-efficient"** — architectures where each parameter contributes maximally to the VC dimension. Skip connections, weight sharing, and shared embeddings all reduce effective capacity without reducing expressiveness.
- **Apply the "compression approach to generalization"** (Hinton, 1993): a model that can both fit the data AND be compressed (via quantization, pruning) is likely generalizing well. Use compressibility as a training regularizer.
- **Match model size to data size using the PAC framework.** If you have N examples, target O(√N) parameters for a transformer (based on empirical scaling laws refined by theory).

---

## 7. Game Theory & Mechanism Design — Beyond Cross-Entropy

### The Core Idea
Game theory studies strategic interactions between agents. Two key ideas transfer to model design:

- **Minimax / Adversarial Training:** In a zero-sum game, both players optimize against each other's best response. Generative Adversarial Networks (GANs) are the obvious example, but the deeper insight is about **robustness**: a minimax-optimal model performs well even against worst-case inputs.
- **Mechanism Design:** Designing the rules of a game so that self-interested agents produce a desired outcome. Applied to neural networks: design the training objective so that individual components (experts, heads, layers) self-organize into an efficient system.
- **Mixture-of-Experts as a Routing Game:** Each expert wants to handle the tokens it's best at. The router wants to distribute load evenly. This is a game-theoretic optimization problem. Google's Expert Choice Routing (Zhou et al. 2022) flipped the script: instead of tokens choosing experts, experts choose tokens. This improved training convergence by 2x.

### Why This Matters for Small Models
- **Cross-entropy is a single-player game.** The model optimizes against a fixed data distribution. But real language use is adversarial — speakers say unexpected things, opponents argue against you, questions are designed to trick. A model trained with minimax objectives is more robust per parameter.
- **The "expert routing" game is under-explored for small models.** MoE is usually used to make LARGE models more efficient. But the same principles can make a SMALL model more efficient — have specialized sub-networks for different input types.
- **Nash equilibrium concepts apply to multi-objective training.** A small model needs to balance quality, speed, and memory. These are competing objectives, and game theory provides the math for finding optimal tradeoffs.

### Actionable Techniques
- **Implement adversarial training with a small perturbation network.** Train a tiny "adversary" that generates the hardest possible inputs for your model (within some norm ball). Train your model against these adversarial examples. This makes every parameter count more.
- **Use Expert Choice routing in a small MoE.** Instead of a standard top-1 router, let experts bid on tokens. For a 100M model, you might have 4 experts at 25M each, with only 2 active per token — same compute, 4x capacity.
- **Train with a minimax objective over token difficulty.** Sample hard tokens more aggressively during training (this is what curriculum learning does, but frame it as a game: the "difficulty player" maximizes loss, the model minimizes it).
- **Design a "mechanism" for layer skipping.** Each layer should "bid" on whether it adds value to the current token. Layers that consistently don't add value get skipped. This is mechanism design: design the rules so the network self-prunes.

---

## 8. Thermodynamics of Computation — Energy as Information

### The Core Idea
Thermodynamics sets absolute physical limits on computation:

- **Landauer's Principle (1961):** Erasing one bit of information dissipates at minimum kT ln(2) ≈ 3 × 10⁻²¹ joules of heat (at room temperature). This is the **thermodynamic minimum** for irreversible computation.
- **Reversible Computing:** If you never erase information — if every computation is bijective (reversible) — you can theoretically compute with zero energy dissipation. Bennett (1973) proved this is possible for any computation.
- **Entropy of Computation:** A computation that maps many inputs to one output (like a hash, or a ReLU activation where many values map to zero) is thermodynamically irreversible. A computation that preserves information (like a permutation) is reversible and thermodynamically free.

### Why This Matters for Small Models
- **Every ReLU that zeros out an activation is a Landauer violation.** It erases information and generates heat. For small models running on edge devices, this matters. Using GELU or Swish instead of ReLU is marginally more reversible.
- **Batch normalization and dropout erase information.** They map multiple inputs to the same output. A small model that avoids these operations (or uses reversible alternatives) is thermodynamically more efficient.
- **The reversible residual (RevNet) architecture** is literally reversible computing applied to neural networks. RevNets can be run forward and backward with the same computation graph, meaning they don't need to store activations for backpropagation. This halves memory during training.
- **The thermodynamic bound on inference energy** is shockingly small. A forward pass through a 100M parameter model using 10¹⁶ multiply-accumulates would dissipate ~10⁻⁵ joules if reversible. Current hardware dissipates ~1-10 joules. There's a 5-6 order of magnitude gap.

### Actionable Techniques
- **Use RevNet-style reversible residual blocks.** Each block should be invertible: x → f(x,y), y → g(x,y) where you can recover (x,y) from (f,g). This eliminates the need to store activations during training, cutting memory in half.
- **Replace ReLU with "information-preserving" activations.** Use activations that are injective (one-to-one): GELU, Swish, or even learned monotonic activations. Every zeroed-out ReLU is wasted thermodynamic capacity.
- **Consider reversible attention mechanisms.** Standard attention is not reversible (it uses softmax which many-to-one maps). An attention mechanism based on Householder reflections or orthogonal transformations would be reversible.
- **Train with quantization-aware methods** that treat bit-erasure as a cost. Each quantization step (e.g., fp32 → int8) erases bits. Design training to minimize the total bit-erasure, which correlates with both energy efficiency and model compactness.
- **Use "conservative forces" in weight updates.** Think of gradient descent as a physical system. Momentum and Nesterov acceleration are literally Newtonian mechanics — they preserve "energy" (information) through the optimization trajectory. Adam's adaptive learning rates waste less "energy" than vanilla SGD.

---

## Summary: The Top 10 Techniques to Implement

| Priority | Technique | Domain | Expected Impact |
|----------|-----------|--------|----------------|
| 1 | Compression-based training metric (bits/byte) | Information Theory | Better measurement → better decisions |
| 2 | Sparse activations (Top-K in every layer) | Neuroscience + Signal Processing | 3-5x compute reduction |
| 3 | Two-stage prediction (classical + neural) | Computational Linguistics | Handle easy tokens cheaply |
| 4 | Expert Choice MoE routing | Game Theory | 2x training efficiency |
| 5 | RevNet-style reversible blocks | Thermodynamics | 2x memory reduction in training |
| 6 | Adversarial training for robustness | Game Theory | More per-parameter quality |
| 7 | Wavelet-inspired multi-scale embeddings | Signal Processing | Better long-range with fewer params |
| 8 | ANS-based generation head | Compression | Provably optimal bit generation |
| 9 | Prediction-residual architecture | Neuroscience | Only compute "surprise" |
| 10 | Information-preserving activations | Thermodynamics | Small efficiency gains per layer, compound |

---

## Key Takeaway

The biggest insight across all 8 domains: **efficiency comes from knowing what NOT to compute.** The brain doesn't process everything — it processes errors. Compressors don't store everything — they store differences. Signal processing doesn't measure everything — it exploits sparsity. Thermodynamics says erasure costs energy — don't erase.

A small language model should be designed around the principle of **selective computation**: identify what's easy/expected and skip it, identify what's hard/surprising and invest in it, and represent everything as sparsely as the information content allows.

The current paradigm of "same architecture for every token" is like broadcasting TV in the 1950s — everything at full power, regardless of the audience. A small model that learns *what to compute* will punch far above its parameter count.
