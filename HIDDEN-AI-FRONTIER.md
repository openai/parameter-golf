# Hidden AI Frontier - Stealth Startups, Labs, and Contrarian Thinkers

*Research conducted on 2026-03-24*
*Target: Identify under-the-radar AI companies, labs, and approaches relevant to building a 16MB model*

---

## 1. Stealth AI Startups 2025-2026

*Search queries: "stealth AI startup funding 2025", "new AI company launch 2026"*

### Initial Findings:

**Notable stealth/emerging companies mentioned:**

1. **Project Prometheus** - Jeff Bezos' new well-funded startup (mentioned in CRN article)
2. **Aurascape** - Launched out of stealth with AI-native security platform
3. **Yann LeCun's new company** - Announced Dec 2025 departure from Meta, focusing on "advanced machine intelligence (AMI)" and world models
4. **xAI** - Elon Musk's company, raised $20B Series E in Jan 2026
5. **Anthropic** - Raising $20B at $350B valuation as of Feb 2026

**Key insight:** Major AI pioneers (LeCun, Bezos) are launching new ventures in 2025-2026 focused on next-generation AI beyond current transformer architectures.

**Relevance to 16MB models:** LeCun's focus on "world models" that understand physical reality could lead to more efficient architectures than transformers. Smaller models that understand causality and physics could outperform larger transformer-based models.

---

## 2. Non-Western AI Labs

### Japan

**RIKEN Center for Advanced Intelligence Project (AIP):**
- **What:** Government research institute with 29 papers accepted at ICLR 2025
- **Key research:** Flow matching, tree-sliced Wasserstein distance, collaboration with Preferred Networks
- **Hardware:** Collaborating with NVIDIA on FugakuNEXT supercomputer (launch ~2030), 256-qubit quantum computer (April 2025), 1000-qubit planned for 2026
- **Why matters for 16MB models:** Access to cutting-edge HPC and quantum infrastructure for training/evaluating small models
- **Maturity:** Established (since 2016), government-backed
- **Links:** https://aip.riken.jp/, https://www.riken.jp/en/research/labs/aip/

**Preferred Networks:**
- **What:** Industrial ML/AI startup, collaborates with RIKEN
- **Key people:** Kenji Fukumizu, Masanori Koyama
- **Research focus:** Flow matching, statistical machine learning
- **Why matters:** Industrial applications focus could yield practical efficiency insights
- **Maturity:** Established startup

**Sakana AI:** (Need more recent info - search limited by rate limits)

### Korea

**KAIST (Korea Advanced Institute of Science and Technology):**
- **What:** Premier Korean research university
- **Key research areas:**
  - **KAIST Visual AI Group:** Discrete Flow Models (DFMs), PairFlow for few-step generation (ICLR 2026), inference-time scaling for flow models (NeurIPS 2025)
  - **DSAIL @ KAIST:** 6 papers accepted at ICLR 2026 workshops (AI & PDEs, AI4Mat, recursive self-improvement, biomolecular design)
  - **Cost reduction:** Technology that cuts AI service costs by 67% using consumer GPUs (Dec 2025)
  - **Factory automation:** Generative AI optimizing injection molding processes (Journal of Manufacturing Systems, 2025)
- **Why matters for 16MB models:** Discrete Flow Models and efficient inference techniques directly relevant to small model deployment. Cost reduction research enables cheaper experimentation.
- **Maturity:** Established research institution
- **Links:** https://visualai.kaist.ac.kr/, https://dsail.kaist.ac.kr/

**Samsung AI:** (Search limited by rate limits)
**Naver:** (Search limited by rate limits)

### India

**AI4Bharat (IIT Madras):**
- **What:** Research lab dedicated to AI for Indian languages
- **Key initiatives:**
  - **Indic LLM Arena:** Open-source, crowd-sourced benchmarking platform for evaluating LLMs for Indian users (launched Nov 2025)
  - **Voice of India benchmark:** Tests speech models across 15 Indian languages/dialects (Feb 2026)
  - **Open-source datasets and models** for 22 scheduled Indian languages
- **Funding:** Received funding round in June 2025; part of India's sovereign AI initiative
- **Why matters for 16MB models:** Multilingual efficiency - models that work well across many languages with limited parameters. Benchmarking infrastructure for small model evaluation.
- **Maturity:** Established research lab with Microsoft support
- **Links:** https://ai4bharat.iitm.ac.in/, https://huggingface.co/ai4bharat

**Sarvam AI:**
- **What:** Indian AI company focused on sovereign AI
- **Plans:** Launching a device in May 2026
- **Role:** Selected for India's sovereign AI model initiative under IndiaAI Mission (2025)
- **Why matters:** Sovereign AI initiatives may prioritize efficiency over scale
- **Maturity:** Startup, government-backed

### Israel

**AI21 Labs:**
- **What:** Israeli LLM startup
- **Recent developments:** 
  - Raised $300M Series D in May 2025 (total $636M)
  - Reportedly in acquisition talks with NVIDIA for $2-3B (Dec 2025)
  - Research publications through 2026
- **Key insight:** Believes "SLMs will play an essential part in AI systems, and that role is expected to deepen into 2026" (Dec 2025 blog)
- **Why matters for 16MB models:** Explicit focus on small language models (SLMs) as essential component of future AI systems
- **Maturity:** Established startup (~200 employees), potential acquisition target
- **Links:** https://www.ai21.com/research/

**Run:ai:** (Search limited by rate limits)

### Middle East

**MBZUAI (Mohamed bin Zayed University of Artificial Intelligence):**
- **What:** World's first AI university (Abu Dhabi)
- **Research:** Broad AI research across disciplines
- **Why matters:** Access to funding and compute resources in oil-rich region
- **Maturity:** Established 2019

**TII (Technology Innovation Institute):**
- **What:** Abu Dhabi's advanced technology research center
- **Falcon LLM:** Open-source large language model family
- **Why matters:** Sovereign AI development with focus on Arabic language
- **Maturity:** Established

---

## 3. Academic Breakthroughs 2025-2026

### NeurIPS 2025 Best Papers (November 2025):

**Four Best Papers:**

1. **"Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)"**
   - **Authors:** Liwei Jiang, Yuanjun Chai, Margaret Li, et al. (University of Washington, CMU, Allen Institute)
   - **Key finding:** LLMs show "Artificial Hivemind" effect - intra-model repetition and inter-model homogeneity in open-ended generation
   - **Dataset:** Infinity-Chat (26K diverse open-ended queries, 31K human annotations)
   - **Why matters for 16MB models:** Highlights need for diversity in small models; homogenization risk increases with model size constraints

2. **"Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"**
   - **Authors:** Zihan Qiu et al. (Alibaba Qwen team)
   - **Key finding:** Adding head-specific sigmoid gate after Scaled Dot-Product Attention improves performance, stability, long-context extrapolation
   - **Experiments:** 30+ variants of 15B MoE and 1.7B dense models on 3.5T tokens
   - **Why matters for 16MB models:** Attention optimization crucial for small models; gating reduces "attention sink" phenomenon
   - **Implementation:** Used in Qwen3-Next models

3. **"1000 Layer Networks for Self-Supervised RL: Scaling Depth Can Enable New Goal-Reaching Capabilities"**
   - **Authors:** Kevin Wang et al.
   - **Key finding:** Scaling RL networks to 1024 layers (vs typical 2-5 layers) unlocks major gains in self-supervised RL
   - **Why matters:** Challenges assumption that RL can't guide deep networks; suggests RL can scale with depth

4. **"Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training"**
   - **Authors:** Tony Bonnaire et al.
   - **Key finding:** Diffusion models have early generalization phase (constant) and later memorization phase (linear with dataset size)
   - **Why matters:** Understanding generalization in generative models

**Three Runner-Up Papers:**

5. **"Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"**
   - **Finding:** RLVR (RL with Verifiable Rewards) improves sampling efficiency but doesn't create new reasoning patterns beyond base model capacity

6. **"Optimal Mistake Bounds for Transductive Online Learning"**
   - **Finding:** Resolves 30-year open problem; transductive online learning has Ω(√d) mistake bound (quadratic gap vs standard online learning)

7. **"Superposition Yields Robust Neural Scaling"**
   - **Finding:** Representation superposition (more features than dimensions) is key driver of neural scaling laws
   - **Why matters for 16MB models:** Understanding how small models can represent many features through superposition

### ICLR 2026 (Search limited by rate limits)

### ICML 2025 (Search limited by rate limits)

---

## 4. Contrarian AI Thinkers

**Key contrarian voices (2025-2026):**

1. **Co-author of "Attention Is All You Need" paper (likely Ashish Vaswani or others):**
   - **Position:** Reportedly "absolutely sick of transformers" (Oct 2025)
   - **Argument:** Need to fund multiple research avenues beyond just scaling transformers

2. **Zhang Xiangyu (chief scientist at Jiyue Xingchen, Tencent conference):**
   - **Position:** Current transformer architectures may not be sufficient for next-generation intelligent agents (Dec 2025)
   - **Context:** Chinese AI researcher questioning transformer dominance

3. **Nature npj Robotics article (May 2025):**
   - **Position:** Questions utility of GPTs for robotics vs. insect brains
   - **Argument:** GPTs demand enormous compute, excessive training times; insect brains achieve robust autonomy with none of these constraints

4. **Research community sentiment (Reddit discussions, Jan 2026):**
   - **Observation:** Transformers can work on modest devices but not smallest edge devices
   - **Implication:** Alternative architectures needed for extreme edge computing

**Why matters for 16MB models:** If transformers have fundamental limitations for edge/robotics applications, alternative architectures (RNNs, SSMs, etc.) may be better suited for small models.

---

## 5. Cross-Domain Techniques

### Physics-Informed Neural Networks (PINNs)

**Key developments (2025-2026):**

1. **NVIDIA PhysicsNeMo:**
   - **What:** NVIDIA's physics-ML framework with curated model architectures
   - **Includes:** Fourier feature networks, Fourier neural operators, GNNs, point cloud and diffusion models
   - **Why matters:** Industry-standard implementation for physics-aware ML

2. **Academic research:**
   - **Oxford course:** "Physics Informed Neural Networks: 2025-2026" (CS department)
   - **Review papers:** Multiple comprehensive reviews in 2025-2026 (Tsinghua Science and Technology, Nature Communications, Scientific Reports)
   - **Applications:** Power systems, physiological signal processing, hyperbolic conservation laws

3. **Key insight:** PINNs enable solving complex problems with scarce data by embedding physical laws

**Why matters for 16MB models:** Physics constraints reduce parameter search space, potentially allowing smaller models to learn complex physical systems.

### Neuroscience-Inspired AI

**Key developments (2025-2026):**

1. **AI-NeurIPS 2026 Conference:**
   - **What:** Premier virtual event bringing together AI researchers and neuroscientists
   - **Focus:** Neural networks and brain-inspired AI

2. **Nature Neuroscience Perspective (Dec 2025):**
   - **Title:** "Leveraging insights from neuroscience to build adaptive artificial intelligence"
   - **Key concept:** "Adaptive intelligence" - AI that learns online, generalizes and adapts quickly like animals
   - **Approach:** Brain-inspired strategies for flexible, adaptive AI algorithms

3. **Georgia Tech research (June 2025):**
   - **Breakthrough:** Brain-inspired AI systems spotlighted at global conference
   - **Researchers:** Murty and Deb continuing to refine brain-inspired AI systems

4. **Empire AI Consortium (2025):**
   - **What:** New York State consortium bridging researchers, public interest organizations, and small companies
   - **Members:** University of Rochester and other institutions
   - **Goal:** Accelerate neuroscience-AI convergence

**Why matters for 16MB models:** Brain-inspired architectures (spiking neural networks, etc.) may be more parameter-efficient than transformers for certain tasks.

### Other Cross-Domain Approaches

- **Signal processing:** Traditional signal processing techniques combined with ML
- **Control theory:** Dynamical systems approaches to AI
- **Dynamical systems:** Mathematical frameworks for modeling complex systems

---

## 6. Underrated Open Source Projects

### Notable GitHub Projects (2025-2026):

1. **RuVector (ruvnet/RuVector):**
   - **What:** High performance, real-time, self-learning vector graph neural network and database in Rust
   - **Architecture:** Hybrid combining Spiking Neural Networks (SNN), SIMD-optimized vector operations, and 5 attention mechanisms with meta-cognitive self-discovery
   - **Why matters:** Novel architecture combining multiple approaches; Rust implementation for performance
   - **Maturity:** Active development

2. **AIKA (Artificial Intelligence for Knowledge Acquisition):**
   - **What:** Innovative neural network design diverging from traditional matrix/vector operations
   - **Approach:** Different architectural paradigm
   - **Why matters:** Alternative to mainstream architectures

3. **NNabla-NAS (sony/nnabla-nas):**
   - **What:** Neural Architecture Search for Sony's Neural Network Libraries
   - **Focus:** Neural hardware aware NAS
   - **Why matters:** Hardware-aware architecture search could find optimal small architectures

4. **Gated Attention implementation (from NeurIPS 2025 paper):**
   - **GitHub:** qiuzh20/gated_attention
   - **Hugging Face:** QwQZh/gated_attention
   - **What:** Implementation of gated attention mechanism that won NeurIPS 2025 Best Paper
   - **Why matters:** State-of-the-art attention optimization

5. **Neural Networks for microcontrollers (GiorgosXou/NeuralNetworks):**
   - **What:** Header-only neural network library for microcontrollers
   - **Support:** Partial bare-metal & native-OS support
   - **Why matters for 16MB models:** Extreme edge computing libraries

### Search Limitations:
- GitHub search rate limited; many innovative projects likely exist but not surfaced in initial search
- "alternative to transformer github" search blocked by rate limits

---

## 7. AI Efficiency Competitions

### MLPerf (Industry Standard)

**Current status (2025-2026):**

1. **MLPerf Inference v5.1 (Sept 2025):**
   - **Latest release:** September 2025
   - **Purpose:** Measure how quickly systems can run AI models across workloads
   - **Characteristics:** Architecture-neutral, representative, reproducible
   - **Participants:** AMD, NVIDIA, Lenovo, Oracle, etc.

2. **MLPerf Training v5.1 (Nov 2025):**
   - **Latest release:** November 2025
   - **Focus:** Full system tests stressing models, software, hardware
   - **Notable result:** AMD Instinct MI325X outperforms NVIDIA H200 by up to 8% when fine-tuning Llama 2-70B-LoRA

3. **MLPerf Inference v6.0 (Feb 2026):**
   - **Call for submissions:** February 2026
   - **New benchmark:** Qwen3 VL MoE (Vision-Language Mixture of Experts)
   - **Submission deadline:** February 13, 2026

**Why matters for 16MB models:** MLPerf provides standardized benchmarks for comparing efficiency across hardware/software stacks.

### Other Competitions:

- **NanoGPT Speedrun:** (Search limited by rate limits)
- **Chinese/Japanese/Korean competitions:** (Search limited by rate limits)

---

## 8. Radical Hardware

### Cerebras (Wafer-Scale)

**Current status (2025-2026):**

1. **Wafer-Scale Engine 3 (WSE-3):**
   - **Specs:** 4 trillion transistors, 125 petaflops, single silicon wafer
   - **Performance:** 100x higher fault tolerance across 300mm wafer vs. smaller GPU
   - **Inference speed:** 2,100-2,500 tokens/sec (Llama 4 Maverick), 2x+ faster than NVIDIA DGX B200 Blackwell

2. **Business developments:**
   - **OpenAI deal (Jan 2026):** $10B+ deal to deliver 750MW computing power through 2028
   - **Funding:** Raised $1B more in Feb 2026
   - **IPO plans:** Targeting 2026 IPO to challenge NVIDIA
   - **Revenue:** Estimated >$1B in 2025

3. **Condor Galaxy supercomputer network:**
   - **Projection:** 36 exaflops of AI compute by end of 2026

**Why matters for 16MB models:** Wafer-scale architecture enables running many small models simultaneously without data movement bottlenecks.

### Groq LPU (Language Processing Unit)

*(Search limited by rate limits)*

### Mythic Analog

*(Search limited by rate limits)*

### Photonic Computing

*(Search limited by rate limits)*

### In-Memory Computing

*(Search limited by rate limits)*

---

## Summary & Strategic Implications for 16MB Models

### Key Findings:

1. **Architectural diversity is increasing:** Contrarian voices questioning transformers, new architectures emerging (RuVector, AIKA, brain-inspired designs)

2. **Efficiency focus growing:** MLPerf benchmarks, hardware innovations (Cerebras), academic research on small models

3. **Geographic diversification:** Non-Western labs (Japan, Korea, India, Israel, Middle East) pursuing unique approaches

4. **Cross-disciplinary convergence:** Physics-informed ML, neuroscience-inspired AI, control theory approaches

5. **Open source innovation:** Underrated GitHub projects exploring alternative architectures

### Recommendations for Parameter Golf:

1. **Explore alternative architectures:** Beyond transformers - consider SSMs, SNNs, physics-informed networks
2. **Leverage cross-domain techniques:** Physics constraints, brain-inspired designs for parameter efficiency
3. **Monitor non-Western research:** Japanese/Korean/Indian labs may have efficiency insights not in Western literature
4. **Test on radical hardware:** Cerebras wafer-scale could enable novel training approaches for small models
5. **Participate in efficiency competitions:** MLPerf and others provide benchmarking and visibility

### Most Promising Areas for 16MB Models:

1. **Gated Attention** (NeurIPS 2025 Best Paper) - attention optimization
2. **Physics-Informed Neural Networks** - constraints reduce parameter search
3. **Brain-inspired architectures** - potentially more parameter-efficient
4. **Discrete Flow Models** (KAIST research) - efficient inference techniques
5. **Hardware-aware NAS** (Sony NNabla-NAS) - find optimal small architectures

---

*Research conducted via web search with rate limitations. Some areas (Groq, Mythic, photonic computing, Asian competitions) require follow-up searches.*

