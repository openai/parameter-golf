# Category 7: Kolmogorov-Arnold Networks (KAN)

**Research Date:** 2026-03-24  
**Focus:** Parameter efficiency, learnable activation functions, applicability to training a better model in 16MB / 10 minutes

---

## 1. What Are KANs?

Kolmogorov-Arnold Networks (KANs) are a neural network architecture inspired by the **Kolmogorov-Arnold Representation Theorem** (KART), which states that any multivariate continuous function can be decomposed into a finite composition of univariate continuous functions and the binary operation of addition.

Formally, for a multivariate function $f(x_1,...,x_n)$:

$$f(\mathbf{x}) = \sum_{q=0}^{2n} \Phi_q \left( \sum_{p=1}^{n} \phi_{q,p}(x_p) \right)$$

Where $\Phi_q$ and $\phi_{q,p}$ are univariate functions.

### KAN vs. MLP: The Core Distinction

| Feature | MLP (Multi-Layer Perceptron) | KAN |
|--------|------------------------------|-----|
| Activation functions | Fixed (ReLU, sigmoid), placed on **nodes** | **Learnable splines**, placed on **edges** |
| Linear weights | Yes, weight matrices between nodes | **No fixed linear weights** — replaced entirely by univariate functions |
| Theoretical basis | Universal Approximation Theorem | Kolmogorov-Arnold Representation Theorem |
| Interpretability | Low (black box) | High (each edge function is visualizable) |
| Parameter scaling | ~O(N²) per layer | ~O(N × G) where G = grid resolution |

In KANs, **every "weight" is itself a learnable spline function** (typically B-splines) parametrized by a grid. This means each connection between neurons is not a scalar but a small function that adapts its shape during training.

---

## 2. Who's Building It

### Original KAN (MIT — Ziming Liu et al.)
- **Lead:** Ziming Liu (MIT), Max Tegmark (MIT), and collaborators
- **Paper:** "KAN: Kolmogorov-Arnold Networks" — arXiv:2404.19756 (April 2024); **Accepted at ICLR 2025** (v5: Feb 2025)
- **Code:** https://github.com/KindXiaoming/pykan (`pip install pykan`)
- **Docs:** https://kindxiaoming.github.io/pykan/

### KAN 2.0 (MIT)
- **Paper:** "KAN 2.0: Kolmogorov-Arnold Networks Meet Science" — arXiv:2408.10205 (Aug 2024)
- Adds: MultKAN (multiplication nodes), kanpiler (KAN compiler for symbolic formulas), tree converter
- Focused on scientific discovery: finding conserved quantities, Lagrangians, symmetries

### Kolmogorov-Arnold Transformer (KAT) — UC Santa Cruz + Others
- **Paper:** "Kolmogorov-Arnold Transformer" — arXiv:2409.10594 (Sep 2024)
- **Key innovation:** Replaces MLP layers in Transformers with KAN layers (Group-Rational KAN / GR-KAN)
- **Three solutions to scaling:** Rational basis (replaces B-splines with GPU-friendly rational functions), Group KAN (shared activation weights), Variance-preserving initialization
- **Code:** https://github.com/Adamdad/kat
- Reviewed at ICLR 2025: https://openreview.net/forum?id=BCeock53nt

### FlashKAT (Oregon State University STARLAB — AAAI 2026)
- **Paper:** "FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer" — arXiv:2505.13813 (May 2025, AAAI 2026)
- **Achievement:** **86.5x training speedup** over KAT by addressing memory stalls in gradient accumulation
- **Code:** https://github.com/OSU-STARLAB/FlashKAT
- Key finding: KAT was 123x slower than MLP-Transformers despite comparable FLOPs — bottleneck was memory, not compute

### FastKAN (Independent — Ziyao Li)
- **Paper:** "Kolmogorov-Arnold Networks are Radial Basis Function Networks" — arXiv:2405.06721 (May 2024)
- Proves B-splines can be well-approximated by Gaussian RBFs → creates **FastKAN**, much faster to compute
- Enables KAN on CPU without major slowdown

### Wav-KAN (Wavelet-based)
- **Paper:** "Wav-KAN: Wavelet Kolmogorov-Arnold Networks" — arXiv:2405.12832 (May 2024)
- Replaces splines with wavelet functions (both CWT and DWT)
- Captures both high-frequency and low-frequency components
- Claims: "enhanced accuracy, faster training speeds, and increased robustness compared to Spl-KAN and MLPs"
- Code: https://github.com/zavareh1/Wav-KAN

### Chebyshev KAN
- **Paper:** "Chebyshev Polynomial-Based Kolmogorov-Arnold Networks" — arXiv:2405.07200 (May 2024)
- Replaces B-splines with Chebyshev polynomials
- Demonstrates parameter efficiency and interpretability gains over MLPs on digit classification and function approximation

### KAF (Kolmogorov-Arnold Fourier Networks) — Feb 2025
- **Paper:** "Kolmogorov-Arnold Fourier Networks" — arXiv:2502.06018 (Feb 2025)
- Integrates trainable Random Fourier Features + hybrid GELU-Fourier activation
- Addresses parameter explosion in high-dimensional tasks
- Evaluated across vision, NLP, audio, and PDE-solving
- Code: https://github.com/kolmogorovArnoldFourierNetwork/KAF

### Polynomial KAN Variants
- **Paper:** "Exploring the Potential of Polynomial Basis Functions in KANs" — arXiv:2406.02583 (May 2024)
- Tests 18 different polynomial families (Gottlieb, Chebyshev, Legendre, Fibonacci-related, etc.)
- Gottlieb polynomials achieved best MNIST performance

### KANtize (March 2026)
- **Paper:** "KANtize: Exploring Low-bit Quantization of Kolmogorov-Arnold Networks for Efficient Inference" — arXiv (March 2026)
- Quantization of KANs for deployment — bridges the gap between research accuracy and inference efficiency
- **Key for 16MB constraint**: Low-bit quantization of spline parameters could compress KAN models dramatically

### KALLM Project (UC Santa Cruz OSPO — 2025)
- **URL:** https://ucsc-ospo.github.io/project/osre25/unl/kallm/
- Implementing KAT into SmolLM2 (Hugging Face's compact LLM)
- Open-source KAN-based Transformer implementations for LLMs
- Uses KAT (from ICLR 2025) as the core architecture

### KANElectra
- GitHub: https://github.com/Klassikcat/KANElectra
- Replaces fully connected layers in Transformer Encoder with KAN layers
- Tests on BERT/Electra-style architectures

### ET-KAN (Energy-based Transformer + KAN)
- Published in Complex & Intelligent Systems (Springer Nature), ~3 weeks before March 24, 2026
- Combines Energy Transformer with KAN for image reconstruction
- Demonstrates parameter efficiency without sacrificing performance

### KaCGM (Causal Generative Models with KAN) — March 2026
- **Paper:** "Kolmogorov-Arnold causal generative models" — arXiv (March 20, 2026)
- Uses KAN to parametrize structural equations for interpretable causal inference on tabular data

### SINDy-KANs — March 2026
- **Paper:** "SINDy-KANs: Sparse identification of non-linear dynamics through KANs" — arXiv (March 19, 2026)
- Combines sparse identification with KAN for interpretable dynamics
- Collaboration: Amanda Howard, Steven Brunton, Pacific Northwest National Lab

---

## 3. The Mathematical Foundation

### Kolmogorov-Arnold Representation Theorem (1957)
Originally proven by A.N. Kolmogorov (1957) and Vladimir Arnold (1957-1958), the theorem shows any multivariate continuous function can be represented as a superposition of continuous functions of a single variable.

### From Theorem to Network
- MLPs interpret "depth" via universal approximation — any function with enough neurons
- KANs interpret "width" and "depth" differently: each layer is a matrix of 1D functions
- The key innovation: making those 1D functions **learnable** (B-splines, Chebyshev polynomials, wavelets, etc.)

### Scaling Laws
The original KAN paper claims KANs obey **faster neural scaling laws** than MLPs:
- KAN: test RMSE ~ N^(-4) where k=3 (3rd-order splines)
- MLP: scales slowly and plateaus quickly
- This means: for the same number of parameters, KANs can achieve better function approximation on smooth/structured data

---

## 4. What the Research Shows: Honest Assessment

### Where KANs Win
1. **Symbolic regression and function approximation** — Consistent wins over MLP
2. **PDE solving** — Physics-informed problems where the function structure is compositional
3. **Scientific discovery** — Rediscovering mathematical laws, conservation laws, Lagrangians
4. **Interpretability** — Edge functions are visualizable; symbolic formulas can be extracted
5. **Small-data, structured-function regimes** — Where the Kolmogorov decomposition is natural

### Where KANs Lose (Critical Assessment)
A systematic evaluation paper (arXiv:2407.11075, v8 updated Sep 2025) found:
- **KANs underperform MLPs** on: machine learning benchmarks, computer vision, NLP, audio processing
- **1.36x to 100x slower** training than MLPs when using B-spline KANs
- Claims about "breaking the curse of dimensionality" lack rigorous foundation
- The main advantage (B-spline activations) could largely be replicated by adding B-spline activations to MLPs

### "KAN or MLP: A Fairer Comparison" (arXiv:2407.16674)
Under same parameter/FLOP budget:
- KAN outperforms MLP **only in symbolic formula representation**
- KAN **inferior to MLP** on ML, computer vision, NLP, audio
- KAN's advantage largely stems from B-spline activation functions, not architecture

### The Speed Problem (Critical for 10-minute constraint)
- Original KAN (B-spline): up to **100x slower** than MLP
- KAT (Rational KAN): comparable FLOPs to MLP-Transformer, but **123x slower** in practice
- FlashKAT (2026): resolves the bottleneck → now only ~1.5x slower than baseline MLP-Transformer

---

## 5. How KAN Could Help Train a Better Model in 16MB / 10 Minutes

### Scenario A: Replace MLP Layers with FastKAN/WavKAN (Conservative)
**What:** Swap the FFN/MLP layers in a small transformer with RBF-based FastKAN or WavKAN layers  
**Why it might help:**
- For structured, compositional tasks (e.g., reasoning, structured prediction), KAN layers may require **fewer parameters for same accuracy**
- WavKAN adaptively learns both high and low-frequency patterns — potentially better sample efficiency
- **Concrete parameter saving:** A KAN layer with width $N$ and grid $G$ has ~$N^2 \times G$ parameters. With small G=3 (small grids), overhead is minimal

**Constraints for 16MB:**
- A float32 parameter is 4 bytes → 16MB = ~4M parameters
- B-spline KAN with G=3 uses ~3x the parameters of equivalent MLP for same width
- **FastKAN (RBF)** or **Chebyshev KAN** use polynomial order k instead of grid G — at k=3, parameter overhead is manageable
- WavKAN can be more parameter-efficient by sharing wavelet basis

**For 10 minutes:**
- FastKAN/RBF variant is critical — avoids the B-spline speed penalty
- FlashKAT optimization techniques (AAAI 2026) bring training speed to near-MLP levels
- Use `model.speed()` in PyKAN before training (disables symbolic branch, removes 10-100x overhead)

### Scenario B: KAN as a Hybrid Activation Drop-In (Most Practical)
**What:** Keep MLP structure but replace fixed activation functions with learnable B-spline/Chebyshev/wavelet activations on selected layers  
**Why it's useful:**
- Much less disruptive than full KAN architecture
- Can be applied to any existing architecture (e.g., small GPT, SmolLM)
- The "fairer comparison" paper confirms: KAN's advantage comes mostly from **the learnable activation, not the full architecture**
- Practically: `F.silu(x)` → `ChebyshevActivation(x)` (trainable, k=3 Chebyshev)

**Parameter overhead:** ~k additional scalars per activation per position — near-zero cost for k=3

**Training speed:** Chebyshev polynomial activations are fast (can use standard matrix ops, no spline grid bookkeeping)

### Scenario C: KAN for the Final Prediction Head
**What:** Use a small KAN (2-3 layers) as the output head / task-specific decoder while the main model remains a standard transformer  
**Why:**
- If the task has compositional structure, a KAN head learns the mapping more efficiently
- The KAN can be interpretable — useful for debugging
- Parameter cost of a small KAN head: e.g., KAN([64, 32, vocab_size]) with G=3 ≈ 64×32×3 + 32×vocab_size×3 parameters — workable if vocab_size is small

### Scenario D: KAN for Hypernetwork / Adapter Layers
**What:** KAN-based adapter or LoRA-style fine-tuning layers  
**Why:**
- Learnable activation adapters can capture domain shift with very few parameters
- A KAN with width [d_model, 16, d_model] at k=3 is ~d_model×16×3×2 parameters — tiny
- Analogous to KAN's strength in scientific discovery: if fine-tuning domain has structured/smooth structure, KAN adapter is more parameter-efficient

### Scenario E: KAN for Tokenizer/Embedding Compression
**Why potentially useful:**
- If token features have compositional structure (e.g., subword morphology), KAN can represent the embedding transformation more efficiently
- In 16MB regime, embedding tables are expensive — a KAN-based learned hash/compression could reduce embedding size
- Speculative but theoretically sound

---

## 6. Practical Implementation for 16MB / 10 Minutes

### Recommended Architecture: Hybrid KAN-Transformer

```
Input → Embedding → [Attention + KAN-FFN] × N layers → KAN head → Output
```

Where KAN-FFN uses **Chebyshev KAN** (k=3, no grid, fast matrix ops):

**Specific design:**
- Main model: ~2-4M parameters (transformer backbone)
- KAN degree: k=3 (cubic Chebyshev, ~3x parameter overhead of standard linear, manageable)
- Use **Group KAN** (from KAT) — share activation weights across neuron groups → 8-16x parameter reduction
- Use **RBF/Gaussian FastKAN** basis instead of B-splines → GPU-optimized
- Disable symbolic computation branch (major speedup)

### Parameter Budget in 16MB (4M float32 params):

| Component | Params | Notes |
|-----------|--------|-------|
| Embedding table | ~500K | vocab=8K, dim=64 |
| 4-layer KAN-Transformer | ~2.5M | 4 layers, d=64, heads=4 |
| KAN-FFN per layer | ~200K | width=[64,128,64], Cheby k=3, Group KAN |
| KAN output head | ~100K | [64, vocab_size] |
| Total | ~3.3M | ✅ fits in 16MB |

### Training Speed for 10 Minutes:
- FastKAN/Chebyshev basis: ~1.5-2x MLP training speed (acceptable)
- With FlashKAT-style kernel optimizations: near-MLP speed
- For 10-minute run: target ~1B tokens/parameter-equivalent, batch size 512-1024, short sequences (256 tokens)
- Critical: call `model.speed()` equivalent — disable interpretability features during training

### When NOT to Use KAN for This Task:
- If training data is unstructured / natural language with no clear compositional structure → MLP likely better
- If memory bandwidth is the bottleneck (spline lookups are cache-unfriendly) → use FastKAN RBF
- If training time is the primary constraint → stick with MLP + cosine activations

---

## 7. Key Insights Summary

### KAN's Parameter Efficiency Claims — Verified With Caveats
✅ **True for smooth/compositional functions** — KANs achieve same accuracy with fewer parameters than MLPs  
✅ **True for scientific tasks** — PDE solving, symbolic regression, law discovery  
❌ **Not true for NLP/vision/audio at scale** — MLPs win or tie under fair comparison  
⚠️ **Speed is a major concern** — B-spline KANs are 10-100x slower; FastKAN/FlashKAT bring this to 1.5-2x

### For Parameter Golf (16MB / 10 minutes):
1. **Best bet:** Learnable activation functions (Chebyshev k=3) as drop-in for MLP activations — near-zero overhead, potential accuracy gain
2. **Second bet:** FastKAN (RBF) for FFN layers in transformer — small parameter overhead, GPU-compatible
3. **Third bet:** Group KAN (from KAT) for output head — shared activations reduce parameter count dramatically
4. **Avoid:** Full B-spline KAN for main architecture — too slow for 10-minute window
5. **Quantize:** KANtize (2026) research shows KANs can be quantized to low-bit — apply INT8 or INT4 to spline coefficients to cut 16MB budget by 4-8x

---

## 8. Key Papers Reference Table

| Paper | arXiv | Date | Relevance |
|-------|-------|------|-----------|
| KAN: Kolmogorov-Arnold Networks | [2404.19756](https://arxiv.org/abs/2404.19756) | Apr 2024 (ICLR 2025) | Foundation |
| KAN 2.0: Meet Science | [2408.10205](https://arxiv.org/abs/2408.10205) | Aug 2024 | MultKAN, symbolic tools |
| KAN Critical Assessment | [2407.11075](https://arxiv.org/abs/2407.11075) | Jul 2024 (v8 Sep 2025) | Honest limitations |
| KAN or MLP: A Fairer Comparison | [2407.16674](https://arxiv.org/abs/2407.16674) | Jul 2024 | MLP wins on NLP/vision |
| FastKAN (RBF-based) | [2405.06721](https://arxiv.org/abs/2405.06721) | May 2024 | Speed fix |
| Wav-KAN (Wavelet) | [2405.12832](https://arxiv.org/abs/2405.12832) | May 2024 | Frequency adaptivity |
| Chebyshev KAN | [2405.07200](https://arxiv.org/abs/2405.07200) | May 2024 | Parameter efficiency |
| Polynomial KAN Survey | [2406.02583](https://arxiv.org/abs/2406.02583) | May 2024 | 18 basis families |
| KAT (Kolmogorov-Arnold Transformer) | [2409.10594](https://arxiv.org/abs/2409.10594) | Sep 2024 | KAN in Transformers |
| KAF (Fourier KAN) | [2502.06018](https://arxiv.org/abs/2502.06018) | Feb 2025 | High-dim efficiency |
| FlashKAT | [2505.13813](https://arxiv.org/abs/2505.13813) | May 2025 (AAAI 2026) | 86.5x speedup |
| KANtize (Quantization) | arXiv Mar 2026 | Mar 2026 | Low-bit inference |
| DropKAN (Regularization) | [2407.13044](https://arxiv.org/abs/2407.13044) | Jul 2024 | Better generalization |

---

## 9. GitHub Repositories

| Repo | Stars | Purpose |
|------|-------|---------|
| [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan) | ~19K | Official KAN library |
| [Adamdad/kat](https://github.com/Adamdad/kat) | ~2K | KAT (KAN Transformer) |
| [OSU-STARLAB/FlashKAT](https://github.com/OSU-STARLAB/FlashKAT) | Growing | Fast KAT training |
| [zavareh1/Wav-KAN](https://github.com/zavareh1/Wav-KAN) | ~1K | Wavelet KAN |
| [mintisan/awesome-kan](https://github.com/mintisan/awesome-kan) | ~5K | KAN resource hub |
| [seydi1370/Basis_Functions](https://github.com/seydi1370/Basis_Functions) | ~500 | Polynomial KAN variants |
| [kolmogorovArnoldFourierNetwork/KAF](https://github.com/kolmogorovArnoldFourierNetwork/KAF) | Growing | KAF implementation |
| [saisumanv/KALLM](https://github.com/saisumanv/KALLM) | New | KAN in SmolLM2 (2025) |
| [Klassikcat/KANElectra](https://github.com/Klassikcat/KANElectra) | ~200 | KAN Electra transformer |

---

## 10. Bottom Line: Should You Use KAN for Parameter Golf?

### The honest answer: **Partially yes, strategically**

**✅ Use:** Learnable Chebyshev/RBF activations as drop-in replacements for fixed activations (ReLU, GELU) in selected layers. This captures most of KAN's expressiveness gain with near-zero speed/memory overhead. This is the "KAN as better activation" play.

**✅ Use:** Group KAN (from KAT paper) for the final prediction head or small connector modules where interpretability helps debugging and parameter sharing across neuron groups reduces cost.

**✅ Use:** FastKAN (Gaussian RBF basis) if you want a full KAN-based FFN layer — GPU-friendly, avoids spline bookkeeping, k=3 is highly compatible with 16MB budget.

**❌ Avoid:** Full pykan B-spline KAN for the main training loop — the training speed penalty (10-100x) is fatal for a 10-minute budget unless you use FlashKAT kernels.

**❌ Avoid:** KAN for standard NLP token-level tasks where MLP already provides good coverage — empirical evidence (2024-2025) consistently shows MLP wins or ties here.

**🔥 Best single move:** Replace all `nn.Linear` + `F.gelu()` combinations in FFN blocks with `ChebyshevKAN(k=3)` — learnable polynomial activation, same parameter count, potentially better expressiveness. This requires ~5 lines of code change, zero architecture redesign, and may yield 5-15% accuracy improvement on structured reasoning tasks.

---

*Research compiled March 2026. KAN is a rapidly evolving field — new variants appearing monthly on arXiv. For the latest, monitor: https://arxiv.org/search/?query=kolmogorov-arnold&searchtype=all and https://github.com/mintisan/awesome-kan*
