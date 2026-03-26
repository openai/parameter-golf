# Category 4: Energy-Based Models & Boltzmann Machines
**Research Date:** March 24, 2026  
**Focus:** EBM revival, diffusion models as energy models, score matching, efficiency for small models  
**Parameter Golf Context:** 16MB model / 10-minute training constraint

---

## Executive Summary

Energy-Based Models (EBMs) have undergone a significant revival since 2020 and continue to produce important research in 2025–2026. The core insight is that EBMs provide a *unified probabilistic framework* that avoids many of the constraints of other generative models (VAEs, GANs, normalizing flows). The 2025 convergence between EBMs, flow matching, and diffusion models has produced new training methods that are MCMC-free or drastically reduce sampling cost — critical for small, fast-training scenarios.

**Verdict for Parameter Golf (16MB/10min):** EBMs are *not* the primary training approach to use directly — MCMC sampling overhead is prohibitive. But several EBM-derived techniques are directly applicable:
1. **Score matching** (the training objective behind diffusion) can train models without the partition function
2. **Noise Contrastive Estimation (NCE)** can replace MCMC entirely
3. **EBRM-style reward models** can improve output quality post-hoc without retraining
4. **Joint EBM (JEM)** lets a classifier simultaneously be a generative model — doubles utility per parameter
5. **Energy Matching** (NeurIPS 2025) provides flow-matching efficiency + EBM flexibility

---

## 1. What Are Energy-Based Models (EBMs)?

### Core Concept
An EBM defines a probability distribution over data via an *energy function* E_φ(x) where lower energy = higher probability:

```
p_φ(x) = exp(-E_φ(x)) / Z
```

`Z` is the *partition function* (normalizing constant) — a sum over all possible inputs. This is **intractable for high-dimensional data**, which is the central challenge of EBMs.

The model learns to assign:
- **Low energy** → data that looks like training data (probable)
- **High energy** → everything else (improbable)

### Historical Roots
- **Boltzmann Machines (1985, Hinton & Sejnowski):** First practical EBMs. Stochastic, biologically inspired, learn via contrastive Hebbian learning.
- **Restricted Boltzmann Machines (RBMs, 1986+):** Simplified structure (no hidden-hidden connections), tractable with block Gibbs sampling. Core of deep belief networks.
- **Term "Energy-Based Models" coined (2003 JMLR):** LeCun, Chopra, Hadsell formalized the framework.
- **Deep EBM revival (2019–present):** Grathwohl et al.'s JEM, Du & Mordatch's implicit generation, Langevin dynamics as sampling.

### The Training Challenge
Maximum likelihood training requires:
```
∇_θ log p_θ(x) = -∇_θ E_θ(x) + E_{p_θ}[∇_θ E_θ(x)]
```
The second term is an **expectation under the model** — requires sampling from the current model, typically via MCMC. This is the fundamental bottleneck.

**References:**
- Wikipedia: Energy-Based Model — https://en.wikipedia.org/wiki/Energy-based_model
- "How to Train Your Energy-Based Models" (Song & Kingma, 2021) — https://arxiv.org/abs/2101.03288

---

## 2. The Main Training Approaches for EBMs

### 2.1 MCMC-Based Maximum Likelihood (Classic)

**What it is:** Sample from the current model using Markov Chain Monte Carlo (Langevin dynamics, HMC, Gibbs sampling), then update parameters to push down data energy and push up sample energy.

**Stochastic Gradient Langevin Dynamics (SGLD):**
```
x_{t+1} = x_t - (ε/2)∇_x E_θ(x_t) + η, η ~ N(0, ε)
```
Use a *replay buffer* of past MCMC samples for stability (Grathwohl et al., 2019).

**Why it matters for small models:** SGLD is computationally expensive and unstable for complex data. Small models may converge faster but still require hundreds of MCMC steps per update.

**Key problem:** Short-run MCMC introduces bias; long-run MCMC is too slow for 10-minute training.

### 2.2 Contrastive Divergence (CD) and Its Descendants

**What it is:** A practical approximation — run MCMC for only k steps (k=1 to 20) rather than convergence. Biased but fast.

**Diffusion Contrastive Divergences (DCD, 2023):**
Replace Langevin dynamics in CD with diffusion processes (parameter-free). Faster and more stable than CD:
> "By replacing the Langevin dynamic used in CD with other EBM-parameter-free diffusion processes, we propose a more efficient divergence."
— https://arxiv.org/abs/2307.01668

**Wasserstein Gradient Flow (EBM-WGF, 2025):**
New approach (ScienceDirect, Neural Networks 2025) avoids MCMC entirely by training via Wasserstein gradient flow of KL divergence. Equivalent to Langevin dynamics but without explicit sampling cost.
— https://www.sciencedirect.com/science/article/abs/pii/S0893608025001790

### 2.3 Score Matching (SM) — MCMC-Free ⭐

**What it is:** Instead of estimating the full probability, match the *score function* (gradient of log probability with respect to data):
```
s_θ(x) = -∇_x E_θ(x) ≈ ∇_x log p_data(x)
```
Score matching **completely avoids the partition function Z** and MCMC sampling.

**Denoising Score Matching (DSM):** Practical approximation — add noise to data, train network to remove noise. This is the foundation of diffusion models.

**Why it matters for small models:**
- Train on denoising task only — no partition function needed
- Works with any network size including tiny ones
- The training objective is a simple MSE loss: `E[||s_θ(x̃) - ∇_x̃ log q(x̃|x)||²]`
- No MCMC chains needed during training

**2025 Extensions:**
- **Nonlinear Denoising Score Matching (NDSM):** Handles structured distributions; demonstrated improved learning from scarce data. Latent version (LNDSM, Dec 2025) combines with VAE latent space for computational efficiency.
  — https://arxiv.org/html/2512.06615

**References:**
- Denoising Score Matching explanation — https://johfischer.com/2022/09/18/denoising-score-matching/
- Vizuara AI breakdown — https://vizuara.substack.com/p/energy-based-models-score-matching

### 2.4 Noise Contrastive Estimation (NCE) — MCMC-Free ⭐

**What it is:** Train by distinguishing real data from "noise" samples drawn from a known distribution. Converts density estimation into binary classification.

```
p_θ(x) vs. q(x)  →  logistic regression problem
```

**Why it matters for small models:**
- No partition function needed at training time
- Can train on standard cross-entropy loss — familiar, stable
- Used in EDLM (NVIDIA, 2025) to fine-tune bidirectional transformers as EBMs on top of pretrained models
- Used in EBRM (Purdue, 2025) for lightweight post-hoc reward refinement

**Self-Adapting NCE (2022, arXiv):** Uses past model snapshots as noise distribution — avoids need for separate noise model design.
— https://arxiv.org/abs/2211.02650

---

## 3. Major 2025–2026 Developments

### 3.1 Energy-Based Diffusion Language Models (EDLM) — NVIDIA/Stanford, January 2025 ⭐⭐

**What it is:** Combines EBMs with discrete diffusion models for text generation. An EBM operates at the *full sequence level* at each diffusion step, correcting approximation errors in vanilla diffusion models.

**Who's building it:** Minkai Xu (Stanford), Yilun Xu (NVIDIA), Jure Leskovec (Stanford), Stefano Ermon (Stanford).

**Key results:**
- Outperforms SOTA diffusion models on language modeling benchmarks
- Approaches autoregressive model perplexity (major result)
- **1.3x sampling speedup** over existing diffusion models via parallel importance sampling
- EBM in residual form — parameters obtained by finetuning a bidirectional transformer via NCE

**How it could help 16MB/10min:**
- The "EBM in residual form" idea: add a lightweight energy correction on top of a pretrained base model. This correction can be tiny — a small MLP or shallow transformer head.
- If you train a base 16MB model normally, then add an EBM residual correction layer (trained via NCE, not MCMC), you can improve output quality without adding much size.
- The NCE training is fast and doesn't require MCMC sampling during training.

**URL:** https://research.nvidia.com/publication/2025-01_energy-based-diffusion-language-models-text-generation  
**Code:** https://github.com/MinkaiXu/Energy-Diffusion-LLM

---

### 3.2 Energy Matching: Unifying Flow Matching and EBMs — NeurIPS 2025 ⭐⭐

**What it is:** A framework that endows flow-based generative models with EBM flexibility. Uses a *single time-independent scalar field* (the energy function) as both generator and prior. Far from data manifold: optimal transport flow paths. Near data manifold: Boltzmann equilibrium via entropic energy term.

**Who's building it:** Michal Balcerak, Tamaz Amiranashvili, et al. (multiple institutions).

**Key results:**
- Substantially outperforms existing EBMs on CIFAR-10 and ImageNet
- Simulation-free training (no MCMC!) away from the data manifold
- One network, one scalar potential — no auxiliary generators, no time conditioning
- Works for protein generation (demonstrates flexibility)

**How it could help 16MB/10min:**
- **Simulation-free training** is the key. Flow matching does not require running the model forward to sample during training — just fit transport paths from noise to data.
- The energy field is a *single scalar network* — can be compact (a few MLP layers).
- If you parameterize the 16MB model as an energy/flow field, you get both generative capability and explicit likelihood information — more capability per parameter.
- NeurIPS 2025 acceptance validates this direction.

**URL:** https://arxiv.org/abs/2504.10612  
**Code:** https://github.com/m1balcerak/EnergyMatching

---

### 3.3 Autoregressive LMs as Secretly EBMs — December 2025 / January 2026 ⭐⭐⭐

**What it is:** Theoretical paper showing a *bijection* between autoregressive language models (ARMs) and EBMs in function space. The bijection corresponds to a special case of the soft Bellman equation in maximum entropy RL. Key insight: ARMs and EBMs are equivalent when trained via supervised learning. Also derives error bounds for distilling EBMs into ARMs.

**Who's building it:** Mathieu Blondel et al. (Google DeepMind), arXiv Dec 2025 / updated Jan 2026.

**Why it matters:**
- Every autoregressive LM you train is already implicitly an EBM
- Explains how ARMs can "plan ahead" despite being next-token predictors
- Theoretically unifies RLHF alignment (EBMs naturally characterize the optimal policy)
- Suggests EBM training objectives could be used *directly* to train or fine-tune ARMs

**How it could help 16MB/10min:**
- **Theoretical grounding for treating your model as an EBM.** Train the 16MB model normally with autoregressive loss. The model *is* an EBM — you can do EBM-style inference (Langevin sampling, best-of-N with energy guidance) at test time.
- If fine-tuning for alignment on limited data, EBM-style objectives (score matching, NCE) may converge faster than RLHF's sampling-heavy approach.
- Energy-guided decoding: once trained, use the energy score to rescore candidates at inference time — no extra training cost.

**URL:** https://arxiv.org/abs/2512.15605

---

### 3.4 Scalable EBMs via Adversarial Training (SGLD-Free JEM) — October 2025 ⭐⭐

**What it is:** New framework for Joint Energy-Based Models (JEM) that replaces unstable SGLD-based sampling with adversarial training. Uses Binary Cross-Entropy loss with Projected Gradient Descent (PGD) for contrastive samples. First EBM-based hybrid to scale to ImageNet 256×256.

**Who's building it:** Xuwang Yin (Independent), Claire Zhang, Tony Wang (MIT), Nir Shavit (MIT).

**Key results:**
- First to scale JEM to high-resolution data with stability
- Matches autoregressive models, surpasses diffusion models as standalone generative model
- Combines generative quality with adversarial robustness (faithful counterfactual explanations)

**How it could help 16MB/10min:**
- JEM's core insight: **your classifier IS an energy model**. A 16MB classifier trained for next-token prediction can simultaneously be used as a generative model without separate training.
- Replacing SGLD with PGD-based adversarial training means stable training without MCMC chains — compatible with a 10-minute budget.
- "Two-stage training" can leverage pretrained models — if you have a pretrained 16MB base, the EBM head trains quickly.

**URL:** https://arxiv.org/html/2510.13872

---

### 3.5 Energy-Based Reward Models (EBRM) — Purdue, April 2025 ⭐

**What it is:** Lightweight post-hoc framework that wraps an existing reward model with an EBM layer. The EBM models the *distribution* of reward values rather than a scalar, capturing uncertainty in human preferences. Uses NCE training — avoids partition function computation. Adds conflict-aware data filtering and label-noise handling.

**Who's building it:** Anamika Lochab, Ruqi Zhang (Purdue University).

**Key results:**
- 5.97% improvement in safety-critical alignment tasks vs. standard RMs
- Works without retraining the base model
- Training: ~465 seconds for 5 epochs over 70M embeddings on single GPU
- Inference: ≤50 gradient steps per example

**How it could help 16MB/10min:**
- **Post-hoc enhancement.** Train your 16MB model in 10 minutes normally. Then add an EBRM wrapper (tiny, trains in seconds) to improve reward quality or alignment.
- The EBRM head itself is very small — just an energy function over (embedding, reward) pairs.
- NCE training means no MCMC during EBRM training — fast and stable.
- Directly addresses reward hacking — critical if you're doing any RLHF-style fine-tuning on small models.

**URL:** https://arxiv.org/abs/2504.13134  
**Code:** https://github.com/AnamikaLochab/EBRM

---

### 3.6 Riemannian Metrics from EBMs — NeurIPS 2025 ⭐

**What it is:** Derives Riemannian metrics directly from pretrained EBMs. These metrics define spatially varying distances, enabling geodesics that follow the data manifold's intrinsic geometry. EBM-derived metrics outperform established baselines, especially in high-dimensional settings.

**Who's building it:** Victor Boutin, Louis Béthune, Yilun Du, Rufin VanRullen, Thomas Serre (Brown/CNRS/etc.).

**Why it matters:** EBMs can be used as geometry engines — understanding the structure of data spaces. Applications in representation learning, compression.

**How it could help 16MB/10min:**
- Less direct for training, but relevant for architecture design: the energy landscape reveals which parts of the parameter space matter.
- Could guide efficient initialization or pruning — start parameters near low-energy regions.

**URL:** https://arxiv.org/abs/2505.18230

---

### 3.7 Unsupervised Ensemble via EBMs — AISTATS 2026 ⭐

**What it is:** Novel deep EBM method for building meta-learners from multiple model predictions without labeled data or retraining. Works in data-scarce settings.

**Who's building it:** Ariel Maymon, Yanir Buznah, Uri Shaham (Hebrew University).

**How it could help 16MB/10min:**
- If you train multiple small 16MB models (with different seeds/data subsets) in 10 minutes each, EBRM ensemble can combine them without additional labeled data.
- Enables cheap ensembling that approaches larger model performance.

**URL:** https://arxiv.org/abs/2601.20556

---

## 4. Boltzmann Machine Revival

### 4.1 Classical Restricted Boltzmann Machines (RBMs)

**What they are:** Two-layer undirected graphical models with visible units (data) and hidden units (features). Energy:
```
E(v, h) = -v^T W h - b^T v - c^T h
```
Training via contrastive divergence (approximate MLE). Famous for deep belief network pre-training (Hinton, 2006).

**Current status:** Largely superseded by VAEs, GANs, and diffusion models for generation. Still used in physics simulation (quantum state representation) and recommender systems.

### 4.2 Semi-Quantum RBM (sqRBM) — February 2025 / Communications Physics ⭐

**What it is:** Classical RBMs require 3x more hidden units than sqRBMs to represent the same distribution. The sqRBM has a commuting Hamiltonian in visible space but non-commuting in hidden space — enabling closed-form gradients while exploiting quantum expressivity.

**Who's building it:** Maria Demidik et al. (multiple institutions).

**Key result:** To learn a given distribution, classical RBM needs 3× more hidden units than sqRBM — with the *same total parameter count*.

**How it could help 16MB/10min:**
- Primarily a quantum computing direction (near-term NISQ devices).
- The **implication for classical models**: it demonstrates that non-commuting hidden representations can be more expressive per parameter. This suggests that complex-valued or entangled hidden representations in classical networks (similar to quaternion networks) might achieve higher expressivity per parameter count.
- Indirect: if 3× parameter efficiency were achievable in classical networks via this principle, a 16MB model could match a 48MB model in expressivity.

**URL:** https://arxiv.org/abs/2502.17562  
**Published in:** Communications Physics (Nature), October 2025

---

## 5. Score Matching Deep Dive

### What Score Matching Is

Score matching (Hyvärinen, 2005) trains by matching the *score* (gradient of log density w.r.t. data) instead of the density itself:

```
minimize: E_data[||s_θ(x) - ∇_x log p_data(x)||²]
```

Key: This can be rewritten as:
```
minimize: E_data[tr(∇_x s_θ(x)) + (1/2)||s_θ(x)||²]
```
— computable without knowing p_data or Z.

### Denoising Score Matching (DSM)

Practical variant: corrupt data with Gaussian noise, train network to predict the noise (or the clean signal). This is exactly what diffusion models do (DDPM, Score SDE).

**Why this is the bridge to diffusion models:** Diffusion models ARE score-based EBMs trained via DSM. The score network s_θ(x, t) = -∇_x E_θ(x, t) is a time-conditioned energy function trained without any partition function computation.

### For Small Models

Score matching (especially DSM) is **the most practical EBM training approach for constrained budgets**:

1. **No MCMC during training** — just a denoising loss
2. **Scales to any model size** — works with tiny networks  
3. **Well-understood loss landscape** — stable training
4. **Directly connects to diffusion** — large literature to draw on

**Practical recipe for 16MB/10min:**
```python
# Core denoising score matching loss (runs in any time budget)
noise = torch.randn_like(x)
t = torch.rand(batch) * T_max
x_noisy = sqrt_alphas[t] * x + sqrt_one_minus_alphas[t] * noise
predicted_noise = model(x_noisy, t)
loss = F.mse_loss(predicted_noise, noise)
```

---

## 6. Latent Space EBMs

### Why Latent Space Matters

Rather than modeling p(x) in pixel/token space, model p(z) where z is a latent representation from a VAE or encoder. This dramatically reduces dimensionality and makes MCMC/sampling tractable.

**Latent EBM approach:**
1. Train a VAE to get latent codes z
2. Train an EBM on z (low-dimensional, much faster sampling)
3. Use VAE decoder at inference

**Why this helps 16MB/10min:**
- Even a 16MB model can learn a good energy function in 32-128 dim latent space
- MCMC in latent space is fast (low-dim = fewer steps to convergence)
- Total model = VAE encoder + tiny latent EBM + VAE decoder — each piece is small

**2025 work:** Latent Nonlinear DSM (LNDSM, Dec 2025, Georgia Tech) shows improved structured distribution learning in VAE latent space with better computational efficiency vs. pixel-space score matching.
— https://arxiv.org/html/2512.06615

---

## 7. EBM Tooling (2025)

### TorchEBM
"A high-performance PyTorch library that makes Energy-Based Models accessible and efficient for researchers and practitioners alike."
- **URL:** https://github.com/soran-ghaderi/torchebm  
- **Released:** 2025
- Includes Langevin dynamics, contrastive divergence, score matching, visualizations
- Good starting point for quick experiments

### mini-ebm
"Minimalist, educational implementation of Energy-Based Models in PyTorch."
- **URL:** https://github.com/yataobian/mini-ebm  
- **Released:** 2025 (ongoing)
- Educational — understand EBM training before optimizing it

### awesome-ebm (Curated Resource List)
- **URL:** https://github.com/yataobian/awesome-ebm  
- Comprehensive papers list through 2026, libraries, tutorials

---

## 8. Specific Recommendations for 16MB / 10-Minute Training

### What DOESN'T Work
- **Full MCMC training:** SGLD/HMC/Gibbs with thousands of steps per update — way too slow for 10-minute budget
- **Large EBMs as primary models:** The partition function problem is worse with large networks; requires long MCMC chains

### What DOES Work

#### Option A: Train as Score Model (Diffusion / DSM)
If your 16MB model IS a score/diffusion model:
- You're already using EBM training (DSM) — just don't call it that
- Denoising loss: `MSE(predicted_noise, actual_noise)` — fast, stable, no partition function
- 10 minutes is enough for meaningful training on modest datasets with DSM loss
- **Relevance:** High. This reframes the constraint positively — if you use diffusion-style objectives, you're getting EBM benefits for free.

#### Option B: NCE Training for Fast Non-MCMC EBMs
- Define an energy function: a small MLP or transformer-based scorer
- Draw noise samples from a known distribution (Gaussian, uniform)
- Train with NCE: binary cross-entropy to distinguish real vs. noise
- This trains the EBM without any MCMC — compatible with 10-minute budget
- **Relevance:** Medium-high. Works well if you have a small, structured output space.

#### Option C: EBM as Post-Hoc Correction Layer
Inspired by EDLM (NVIDIA, 2025) and EBRM (Purdue, 2025):
1. Train your 16MB model normally (autoregressive, 10 min)
2. Add a tiny EBM residual correction (1-5% of total params, e.g. 160K–800K params)
3. Train the correction layer via NCE in 1-2 minutes
4. At inference: use model output + EBM correction for better distribution
- **Relevance:** High. Maximizes training time for the main model, adds EBM benefits at minimal cost.

#### Option D: Joint Discriminative-Generative (JEM-style)
- Train 16MB model for next-token prediction
- Simultaneously use the logits as an energy function (JEM framework)
- This is **free** — no extra training, same model does both
- Use energy scores for: best-of-N sampling, reranking, OOD detection
- **Relevance:** Very high. Zero-cost EBM capability from any trained classifier/LM.

#### Option E: Latent EBM
- Train small VAE encoder (4MB) + decode r (4MB) + latent energy model (8MB)
- Model p(z) in 64-256 dim latent space
- MCMC/sampling in latent space: fast (100 steps in ~1 second vs. days in pixel space)
- **Relevance:** Medium. More complex setup but better parameter efficiency for generative modeling.

---

## 9. Key Papers Reference Table

| Paper | Year | Key Innovation | EBM Relevance | URL |
|-------|------|----------------|---------------|-----|
| Energy-Based Diffusion LM (EDLM) | Jan 2025 | EBM residual on top of diffusion; NCE training; 1.3x speedup | ⭐⭐⭐ High | [NVIDIA Research](https://research.nvidia.com/publication/2025-01_energy-based-diffusion-language-models-text-generation) |
| ARMs are secretly EBMs | Dec 2025 | Bijection between autoregressive models and EBMs | ⭐⭐⭐ High | [arXiv 2512.15605](https://arxiv.org/abs/2512.15605) |
| Energy Matching | Apr–Oct 2025 | Unified flow+EBM, simulation-free, single scalar | ⭐⭐⭐ High | [arXiv 2504.10612](https://arxiv.org/abs/2504.10612) |
| Scalable JEM via Adversarial Training | Oct 2025 | SGLD-free JEM, scales to ImageNet 256, adversarial stability | ⭐⭐ Med-High | [arXiv 2510.13872](https://arxiv.org/html/2510.13872) |
| Energy-Based Reward Models (EBRM) | Apr 2025 | Lightweight post-hoc RM refinement, NCE, no retraining | ⭐⭐ Med-High | [arXiv 2504.13134](https://arxiv.org/abs/2504.13134) |
| sqRBM (Quantum RBM) | Feb 2025 | 3x fewer hidden units needed vs classical RBM | ⭐ Indirect | [arXiv 2502.17562](https://arxiv.org/abs/2502.17562) |
| EBM-WGF (Wasserstein Gradient Flow) | Mar 2025 | MCMC-free EBM training via WGF | ⭐⭐ Med | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608025001790) |
| Riemannian Metrics from EBMs | May 2025 | EBMs define data manifold geometry | ⭐ Indirect | [arXiv 2505.18230](https://arxiv.org/abs/2505.18230) |
| Latent NDSM | Dec 2025 | Structured DSM in VAE latent space, efficient | ⭐⭐ Med | [arXiv 2512.06615](https://arxiv.org/html/2512.06615) |
| Diffusion Contrastive Divergence | Jul 2023 | Replace Langevin in CD with diffusion processes | ⭐⭐ Med | [arXiv 2307.01668](https://arxiv.org/abs/2307.01668) |
| Unsupervised Ensemble EBM | Jan 2026 | EBM meta-learner, no labels, data-scarce | ⭐ Med | [arXiv 2601.20556](https://arxiv.org/abs/2601.20556) |
| How to Train Your EBM (tutorial) | 2021 | Comprehensive review: MCMC, SM, NCE | ⭐⭐ Reference | [arXiv 2101.03288](https://arxiv.org/abs/2101.03288) |

---

## 10. The Bottom Line

### EBMs in 2025–2026: The Convergence Era

The key meta-trend: the boundaries between EBMs, diffusion models, flow matching, and autoregressive models are dissolving:
- **Diffusion = score-based EBM** (DSM training)
- **Flow matching = simulation-free optimal transport** (can add EBM energy terms)
- **Autoregressive LMs = EBMs** (bijection proven Dec 2025)
- **RLHF alignment = EBM inference** (optimal policy = energy normalization)

This convergence means EBM techniques are *already inside* the most efficient modern architectures. You don't need to choose "EBM or autoregressive" — they're the same thing at a mathematical level.

### For Parameter Golf Specifically

**Top 3 actionable techniques:**

1. **Treat your autoregressive model as an EBM at inference** (zero cost, immediate gain from the Dec 2025 bijection paper). Use energy scoring (log-prob of full sequences) for reranking at test time.

2. **Use denoising score matching (DSM) if doing any diffusion-style training** — this is EBM training without the partition function hell. Completely compatible with 10-minute budgets.

3. **Add an EBRM residual correction layer** post-training (inspired by EDLM). Train a 160KB energy head via NCE in ~2 minutes on top of your 16MB base model. Gets you EBM-quality distributions on top of a fast AR-trained base.

---

*Research compiled March 24, 2026. Sources: arXiv, NVIDIA Research, Nature/Communications Physics, NeurIPS 2025, AISTATS 2026, emergentmind.com.*
