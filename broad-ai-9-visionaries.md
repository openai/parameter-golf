# Category 9: AI Visionaries Pushing Boundaries
## What the Frontier Minds Say About the NEXT Paradigm

*Research compiled: 2026-03-24 | Parameter Golf project*

---

## Overview

Eight thinkers. Eight distinct bets on what comes after the current LLM scaling era. Each has a radically different answer — and that divergence is the signal. When the top minds disagree this sharply, it means the next paradigm hasn't been won yet. That's exactly where a constraint-driven project like 16MB/10min training can punch above its weight.

---

## 1. Yann LeCun — JEPA / World Models (AMI Labs)

### What He's Saying
LeCun left Meta in early 2026 to found **Advanced Machine Intelligence (AMI Labs)**, launching with a **$1.03 billion seed round** at a $3.5B pre-money valuation — likely the largest seed round ever. His thesis: LLMs are fundamentally the wrong approach to intelligence. The next paradigm is **world models** built on his **Joint-Embedding Predictive Architecture (JEPA)**.

**Core argument:** LLMs predict tokens. JEPA predicts *representations* in latent space — not pixels, not words, but abstract embeddings of what's likely to happen next. This is how the brain works. No hallucinations from reconstructing every detail; instead, learning causal, physical structure of the world.

**Quote (MIT Technology Review, Jan 2026):**
> "There's certainly a huge demand from the industry and governments for a credible frontier AI company that is neither Chinese nor American... AI is going to become a platform, and most platforms tend to become open-source."

**The AMI thesis:**
- LLMs cannot scale to human-level intelligence because they're trained on text, not reality
- World models learn from sensor data (video, robotics, interaction) — the same data humans learn from
- JEPA: predict abstract representations of masked patches/blocks in latent space, not reconstructed pixels
- The masking strategy is crucial — large-scale semantic targets force the model to learn *meaning*, not noise

**Status (March 2026):** AMI is hiring world-model researchers, representation learning specialists, scaling experts. Research is in Paris and Zürich. CEO is Alexandre LeBrun. Founding team includes Saining Xie (CSO), Pascale Fung.

### Relevance to 16MB / 10-Minute Training

**High relevance.** JEPA is explicitly designed for *data efficiency* — the model must learn rich representations from partial information. Key insights:

1. **Latent-space prediction > pixel/token reconstruction**: Training a tiny model to predict abstract embeddings rather than raw outputs dramatically reduces the information it needs to process. For 16MB, this could mean predicting *semantic embeddings* of outputs rather than tokens directly.

2. **Masking strategy as a training signal**: JEPA's masking forces the model to fill in semantic gaps, not noise. In 10 minutes, a well-designed masking curriculum could force a small model to learn structure faster than standard pretraining.

3. **No need for data augmentation**: I-JEPA (image JEPA) trained ViT-Huge on ImageNet in 72 hours without hand-crafted augmentations. The architecture itself is the curriculum. For constrained training, removing augmentation overhead matters.

**Concrete application:** Use JEPA-style latent-predictive pretraining as a replacement for or complement to next-token prediction. Instead of "predict the next word," the loss is "predict the abstract representation of what comes next in a causal world-model sense."

### URLs
- AMI Labs announcement: https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/
- MIT Tech Review interview: https://www.technologyreview.com/2026/01/22/1131661/yann-lecuns-new-venture-ami-labs/
- Latent Space AINEWS: https://www.latent.space/p/ainews-yann-lecuns-ami-labs-launches
- I-JEPA paper: https://arxiv.org/abs/2301.08243
- WIRED profile: https://www.wired.com/story/yann-lecun-raises-dollar1-billion-to-build-ai-that-understands-the-physical-world/

---

## 2. Ilya Sutskever — Continual Learning / Safe Superintelligence (SSI)

### What He's Saying
Sutskever left OpenAI in 2024, founded **Safe Superintelligence Inc. (SSI)** with $3B raised, valued at **$32 billion** (April 2025). One product, one goal: safe superintelligence. No commercial products planned yet — straight research.

**The paradigm shift he's betting on:** Scaling is over. The next breakthroughs come from **new learning methods**, specifically the ability to *generalize from fewer examples* and *learn continually*.

**Key claims (Dwarkesh Patel interview, late 2025):**
- "It is back to the age of research again, just with big computers."
- Pre-training has consumed nearly all high-quality internet text — the pretraining data wall is real
- The next breakthroughs will not come from 10 trillion parameter models but from new training algorithms
- **AGI will emerge as a "superintelligent learner"** — not an all-knowing oracle, but a system that can learn any job extremely fast through deployment
- Human-level learning systems: 5–20 year timeline
- **Continual learning** is the core missing piece: models that update their weights from experience, not just in context

**Quote:**
> "AGI will start as a superintelligent learner, not an all-knowing oracle. A system that can learn every job extremely fast becomes superintelligent through deployment."

**SSI research directions (2026 targets):** Specific AI training and safety protocol advancements. The company is in stealth — no products, no demos, just research.

### Relevance to 16MB / 10-Minute Training

**Critical relevance.** Sutskever is explicitly saying the next era is about *learning efficiency*, not scale.

1. **"Generalization from fewer examples"**: The core problem for 16MB models. If SSI's thesis is correct, there exist training methods that dramatically improve sample efficiency. The question is what those methods are — RLVR (Karpathy's framing) is the current best guess.

2. **Continual learning / weight updates**: For a 16MB model, the ability to update weights from a 10-minute training run is *the entire game*. Sutskever's bet that frozen-weight models are a dead end is directly relevant — the value of a tiny model increases massively if it can learn on-device.

3. **New learning algorithms > more data**: In a 10-minute window, you can't feed a model billions of tokens. New algorithms (RLVR, synthetic data generation, curriculum learning) that extract more signal per token are the path forward.

**Concrete application:** Design training curriculum around *verifiable tasks with automated rewards* (RLVR) rather than pure next-token prediction. Even in 10 minutes, a model that gets reward signal from solving small, checkable problems will develop stronger reasoning than one that sees equivalent text.

### URLs
- SSI: https://ssi.inc/
- Inc. profile (no products): https://www.inc.com/ben-sherry/openai-co-founder-ilya-sutskever-safe-superintelligence-3-billion-no-product/91271937
- $32B valuation: https://techcrunch.com/2025/04/12/openai-co-founder-ilya-sutskevers-safe-superintelligence-reportedly-valued-at-32b/
- "End of scaling" analysis: https://www.the-ai-corner.com/p/ilya-sutskever-safe-superintelligence-agi-2025

---

## 3. Dario Amodei — Constitutional AI / "Machines of Loving Grace" / AI-as-Scientist

### What He's Saying
Amodei (Anthropic CEO) published **"Machines of Loving Grace"** (October 2024), a 15,000-word manifesto about what AI could do if everything goes right. His 2026 trajectory is increasingly focused on **AI as a scientific discovery engine** — specifically biology and neuroscience.

**The paradigm he's betting on:** AI-as-scientist. Not AI that summarizes papers, but AI that *designs experiments, tests hypotheses, and iterates autonomously* on scientific problems. The frontier is not smarter chatbots — it's compressing decades of scientific progress into years.

**Five areas he identifies as most transformative:**
1. Biology and physical health — "medicine won't look like it does today in 10 years"
2. Neuroscience and mental health
3. Economic development and poverty
4. Peace and governance
5. Work and meaning

**The tension (2026):** Fortune reported in February 2026 that Amodei "admits his company struggles to balance safety with commercial pressure." The Atlantic ran a piece called "The Dissonance of Anthropic CEO Dario Amodei" (March 2026), covering his standoff with the Pentagon on AI for defense.

**Constitutional AI (the technical bet):** Train models with an explicit constitution of values — AI critique and revision of its own outputs against stated principles. This replaced RLHF as Anthropic's core alignment method and is now integrated into Claude's training pipeline.

**Key technical thesis for the next paradigm:**
- Reinforcement learning from verifiable science is the path (analogous to Karpathy's RLVR framing, but for scientific tasks)
- Models need to learn to *generate hypotheses and verify them* — not just answer questions
- Safety and capability are not in opposition: safe models that can reason about their own behavior will be more capable, not less

### Relevance to 16MB / 10-Minute Training

**Moderate-high relevance.**

1. **Constitutional AI as a loss function**: For a tiny model in 10 minutes, you can't do RLHF (no human raters). But you *can* implement a lightweight constitutional critique step — have the model evaluate its own outputs against a small set of explicit principles and use that as a training signal. Zero human labor, fully automated.

2. **AI-as-scientist framing**: Even at 16MB, if the training task is structured as "generate hypothesis → verify → update," the model develops reasoning patterns that transfer. The *task structure* matters more than model size.

3. **Verifiable rewards in science**: Amodei's bet that AI will compress scientific progress implies that *correctness-checkable tasks* (math, code, factual claims that can be verified) are the right training substrate. This aligns with RLVR — train on tasks where the answer is checkable.

**Concrete application:** Add a constitutional critique layer to the training loop: after each generation, run a lightweight self-evaluation pass against 3-5 explicit principles (accuracy, helpfulness, reasoning quality) and use the critique as a training signal. Takes minutes, adds no human labor.

### URLs
- Machines of Loving Grace essay: https://darioamodei.com/essay/machines-of-loving-grace
- Atlantic profile (March 2026): https://www.theatlantic.com/technology/2026/03/anthropic-dod-ai-utopianism/686327/
- Safety paradox analysis: https://digidai.github.io/2026/03/06/dario-amodei-anthropic-ai-safety-evangelist-business-path-deep-investigation/
- CFR speaker series: https://www.cfr.org/event/ceo-speaker-series-dario-amodei-anthropic

---

## 4. Demis Hassabis — AI-as-Scientific-Engine / World Models / Automated Labs

### What He's Saying
Hassabis (Google DeepMind CEO, 2024 Nobel Laureate in Chemistry for AlphaFold 2) is betting on **AI as a closed-loop scientific discovery system**. His December 2025 podcast crystallized his two-part thesis for AGI:

**Two prerequisites for AGI (December 2025 podcast):**
1. **World models** — AI that truly understands physics, space, and causality (not just text patterns)
2. **Automated experimentation** — AI that can set up, run, and interpret experiments without human intervention

**Quote (Fortune, February 2026):**
> "In 10, 15 years' time, we'll be in a kind of new golden era of discovery that is a kind of new renaissance."

**The products building toward this:**
- **Veo**: Video model learning motion, liquid flow, light physics
- **Genie**: Generates interactive game worlds with physical structure → AI training ground
- **Sima**: AI avatar that acts in virtual environments, developing perception-action chains
- **Genie + Sima loop**: Genie generates the world, Sima explores it → self-supervised training loop without human intervention
- **Isomorphic Labs**: Applying AlphaFold to drug discovery — "1,000× more efficient" drug development through simulation

**Key 2026 development:** DeepMind signed a cooperation agreement with the UK government (December 2025) to establish its **first fully automated scientific laboratory** in 2026 — AI that runs physical experiments.

**His diagnosis of current LLM failures:**
> "They can win gold medals in the International Mathematical Olympiad but may make mistakes in primary school geometry problems."

LLMs lack world models. They've read books but never touched the physical world. Sensor data, motor control, tactile feedback — all absent from text pretraining.

### Relevance to 16MB / 10-Minute Training

**High relevance on the architecture side.**

1. **Genie + Sima self-supervised loop**: For tiny model training, the concept of a *self-generating training environment* is crucial. If you can build a small synthetic world in which the model must act and verify results, you get dense reward signal without needing massive labeled datasets.

2. **Physical simulation as cheap training data**: Rather than scraping internet text, generate *simulated physical scenarios* (gravity, motion, containment) with verifiable outcomes. A 16MB model trained on physics puzzles with ground-truth answers will generalize better than one trained on text prediction.

3. **The "truly understand" benchmark**: Hassabis's test — "if you can simulate the world, you understand it" — suggests a training objective. Train the model to predict physical state transitions (even toy examples). A model that can answer "what happens if you drop a cup?" correctly has learned something more transferable than a model that has pattern-matched the question.

**Concrete application:** Supplement text pretraining with a small physics/causal puzzle dataset where correct answers are checkable. Generate with Python scripts: motion, containment, counting, spatial reasoning. 1,000 examples in 10 minutes = a powerful signal for grounding.

### URLs
- Fortune profile (February 2026): https://fortune.com/2026/02/11/demis-hassabis-nobel-google-deepmind-predicts-ai-renaissance-radical-abundance/
- Two steps to AGI analysis: https://eu.36kr.com/en/p/3598888503902981
- Nature deep dive: https://www.nature.com/articles/d41586-025-03713-1
- Nobel 2024 win context: https://markets.financialcontent.com/wral/article/tokenring-2026-1-2-the-year-ai-conquered-the-nobel-how-2024-redefined-the-boundaries-of-science

---

## 5. Jeff Dean — Virtual Engineers / Specialized Compute / Organic AI Systems

### What He's Saying
Dean (Google Chief Scientist, Alphabet) gave his clearest statement of the next paradigm at **Sequoia's AI Ascent 2025**:

**Bold prediction (2025):** "We will have AI systems operating at the level of junior engineers within a year."

**The paradigm he's identifying:** Transition from "AI as tool" to **AI as a participant in the engineering workflow** — not just answering questions, but reasoning across long chains of steps, using tools, executing code, and iterating.

**Key themes from the Sequoia interview:**
1. **"Bigger model, more data, better results" is mostly still true, but...**  — the biggest models will be limited to a handful of well-resourced players. For everyone else, the game is **distillation and specialization**.

2. **Model distillation enables startups**: Large models compress knowledge into smaller, specialized models that maintain strong performance. This is the key path for non-hyperscale players.

3. **Multimodality and agents are major growth vectors**: Seamlessly working across text, code, audio, video, and images. Agents: "there's a clear path to rapid improvement through reinforcement learning and use of simulated environments."

4. **Education and productivity tools are immediate opportunities**: AI creating interactive educational experiences. Specific workflows where AI can dramatically improve efficiency.

5. **Future AI systems: "more organic, flexible"** — varying levels of compute intensity, specialized components, continuous learning and adaptation. Not one giant model but a heterogeneous system.

**His historical framing:**
> "Starting in 2012, the same algorithmic approach would work for vision, speech, and language. That was remarkable — and the same sort of scaling worked. 'Bigger model, more data, better results' has been relatively true for 12–15 years."

**The shift:** Models are now capable of interesting things but "can't solve every problem." The era of capability comes from combining scaling with new training methods, multimodality, and agent loops.

### Relevance to 16MB / 10-Minute Training

**Direct relevance on the distillation angle.**

1. **Distillation is the explicit path for non-hyperscale**: Dean names this directly. For 16MB, knowledge distillation from a large teacher model is the single highest-leverage technique available. Take a 70B model's output distributions on a targeted domain, use those as training targets for the tiny model. You get 70B-quality "soft labels" in 10 minutes.

2. **Simulated environments for agent training**: Dean's point about agents improving through "reinforcement learning and simulated environments" applies at tiny scale. A small model that's trained in a verifiable simulation environment (even simple text-based puzzles) develops better reasoning.

3. **Specialization > generalization at small scale**: Don't try to build a general 16MB model. Build a specialized one. Dean's framing supports this: the opportunity for small players is in specific verticals and specific workflows.

**Concrete application:** Use a large model as a teacher to generate high-quality training examples specifically targeted at your use case. Run the large model on your domain, collect its outputs (including chain-of-thought), and use those as training data for the 16MB student. Knowledge distillation from inference, not training.

### URLs
- Sequoia AI Ascent interview: https://sequoiacap.com/podcast/training-data-jeff-dean/
- Google Research profile: https://research.google/people/jeff/
- Time 100 AI profile: https://time.com/collections/time100-ai-2025/7305831/jeffrey-dean/
- Pathways blog post: https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/

---

## 6. Andrej Karpathy — RLVR / "LLM OS" / Ghosts vs. Animals / Vibe Coding

### What He's Saying
Karpathy published his **2025 LLM Year in Review** (December 19, 2025), one of the most insightful documents of the year. He runs Eureka Labs, focused on AI education.

**Paradigm 1: Reinforcement Learning from Verifiable Rewards (RLVR)**

This is the biggest shift of 2025 according to Karpathy. The standard training stack (Pretraining → SFT → RLHF) now has a fourth stage: **RLVR** — training against automatically verifiable reward functions (math, code, puzzles).

Key insight: By training against *non-gameable* objective rewards, LLMs spontaneously develop reasoning strategies — they learn to break down problems, backtrack, and iterate. "These strategies would have been very difficult to achieve in the previous paradigms because it's not clear what the optimal reasoning traces look like — it has to find what works for it."

**Key RLVR properties:**
- Longer optimization than SFT/RLHF (months of RL, not days of finetuning)
- Creates *scaling law for test-time compute* — longer thinking = better answers
- New "thinking time" knob: control capability vs. compute
- Gobbled up compute originally intended for pretraining — more efficient than scaling

**Paradigm 2: Ghosts vs. Animals / Jagged Intelligence**

LLMs are not evolving animals. They are "summoned ghosts" — entities optimized for imitating humanity's text, not for survival. They display *jagged intelligence*: genius polymath and confused grade-schooler simultaneously.

**Implications:** Standard benchmarks are useless because RLVR + synthetic data can make any benchmark look saturated while actual capability remains gaps-ridden. The real measure is task performance in novel, unanticipated situations.

**Paradigm 3: Claude Code / AI that lives on your computer**

Karpathy identifies Claude Code (CC) as "the first convincing demonstration of what an LLM Agent looks like — something that in a loopy way strings together tool use and reasoning for extended problem solving." Key: runs on localhost with your private context. Not cloud. The shift from "website you go to" to "little spirit that lives on your computer."

**Paradigm 4: Vibe coding**

Natural language as the primary programming interface. Not a gimmick — a new interaction layer that changes who can build software.

**The LLM OS framing (from earlier work):** LLMs are the kernel of a new OS. Peripheral devices = context window (RAM), long-term storage (disk), tools (I/O). Applications = prompts and agent loops. This framing unifies the architecture.

### Relevance to 16MB / 10-Minute Training

**Extremely high relevance — most directly applicable of all eight.**

1. **RLVR is the path**: Training for 10 minutes with verifiable rewards is more efficient than training for 10 minutes on text prediction. Even 100 RLVR steps on math/code problems will produce a model that *reasons*, not just pattern-matches. This is the single most actionable insight.

2. **Jagged intelligence as a design target**: For 16MB, you want *sharp jaggies*, not uniform mediocrity. A model with excellent performance on one specific task type and poor performance elsewhere is more useful than a uniformly mediocre generalist. RLVR naturally creates jagged specialization.

3. **Thinking time = quality**: Even a tiny model can use "think longer" at inference. Training with chain-of-thought targets (from RLVR) builds this capability into the weights.

4. **Synthetic data from verifiable domains**: Math and code have ground truth. Generate thousands of problems with solutions in a Python script. Feed them to the 16MB model as RLVR targets. In 10 minutes, you've given the model more *learning signal* than equivalent time on text prediction.

**Concrete application:** Build a curriculum of 5,000 verifiable short-form tasks (arithmetic, simple code, factual lookup, logical inference). Train the model to generate answers and receive reward signal from an automated verifier. This is doable in 10 minutes on a consumer GPU with a 16MB model.

### URLs
- 2025 Year in Review: https://karpathy.bearblog.dev/year-in-review-2025/
- Animals vs. Ghosts: https://karpathy.bearblog.dev/animals-vs-ghosts/
- Verifiability post: https://karpathy.bearblog.dev/verifiability/
- YC Talk transcript: https://www.donnamagi.com/articles/karpathy-yc-talk
- LLM Year in Review summary: https://mlops.substack.com/p/2025-llm-year-in-review-from-andrej

---

## 7. George Hotz — Tinygrad / Local Learning / Anti-Cloud Thesis

### What He's Saying
Hotz (geohot) left comma.ai in November 2025 and is now fully focused on **Tiny Corp** — building tinygrad (minimal ML framework) and the Tinybox hardware line. He's the most contrarian voice here, and his positions have clarified significantly in early 2026.

**The paradigm he's betting on: Local models that actually learn.**

**From "tiny corp's product — a training box" (February 15, 2026):**
> "Every month, we see these LLMs become more and more human. However, there's a major difference. They do not learn. Everyone has the same Claude/Codex/Kimi, with the same weights, the same desires, and the same biases."
> 
> "The only way local models win is if there's some value in full on learning per user or organization. At that point, with entirely different compute needing to run per user, local will beat out cloud."
> 
> "Not API keyed SaaS clones. Something that lives in your house and **learns your values**. Your child."

His product vision: **Tinybox** (local GPU cluster, consumer-grade, offline) that runs models which update their weights from interaction. The anti-thesis of cloud AI: personalized, learning, private.

**The software angle:**
- **Tinygrad**: Minimal deep learning framework with zero dependencies, LLVM removal target, pure Python GPU driving
- Goal: Software sovereignty — AMD/NVIDIA-independent training stack
- Philosophy: "The path to competing with NVIDIA is through software sovereignty, not hardware"

**From "running 69 agents" (March 11, 2026):**
> "AI is not a magical game changer, it's simply the continuation of the exponential of progress we have been on for a long time."
> 
> "People see 'AI' and they attribute some sci-fi thing to it when it's just search and optimization. Always has been."

Hotz is aggressively contrarian on AI hype — but his product thesis (local, learning, personal) is potentially the most distinctive.

**The Tinybox product:**
- On-premises, purpose-built AI compute
- Consumer GPU hardware, shipped to your door
- Runs and *trains* AI models locally, no cloud dependency
- Target: researchers, hobbyists, compliance-sensitive orgs (healthcare, legal, finance)
- Key differentiator: full control over weights, data, and software stack

### Relevance to 16MB / 10-Minute Training

**Very high relevance — this is the exact target market.**

1. **16MB / 10-minute training IS the Hotz vision**: A model small enough to train locally, in minutes, on consumer hardware. This is exactly what Hotz is building infrastructure for. The parameter-golf project is the model side of what Tiny Corp is building the hardware side for.

2. **Tinygrad as a training substrate**: Tinygrad is explicitly designed for running and training small-to-medium models on non-NVIDIA hardware. If you want to run 10-minute training loops on AMD GPUs or exotic hardware, tinygrad is the most relevant framework. Zero corporate dependencies.

3. **Weight updates from interaction**: Hotz's key bet — models that learn from each interaction, updating weights based on user feedback — requires exactly the kind of fast, efficient training infrastructure that 16MB/10-min optimizes for. The model has to be *small enough to train* on a Tinybox.

4. **Anti-benchmark philosophy**: "AI is just search and optimization." Hotz's framing implies: don't optimize for benchmark scores. Optimize for the actual task. This is directly applicable — don't train the 16MB model to score well on standard benchmarks. Train it to be excellent at one specific, verifiable, useful thing.

**Concrete application:** Use tinygrad (https://tinygrad.org) as the training framework. It's minimal, well-optimized for non-NVIDIA hardware, and actively maintained. The framework itself is designed around the constraint of efficiency — Hotz has been optimizing for exactly the 16MB/10-min regime.

### URLs
- Training box blog post: https://geohot.github.io/blog/jekyll/update/2026/02/15/tiny-corp-product.html
- Running 69 agents: https://geohot.github.io/blog/jekyll/update/2026/03/11/running-69-agents.html
- Blog index: https://geohot.github.io/blog/
- Tinygrad: https://tinygrad.org
- PrismNews on Tinybox: https://www.prismnews.com/news/george-hotzs-tiny-corp-brought-high-end-ai-hardware-to-your
- Semperfly analysis: https://semperfly.substack.com/p/george-hotz-comma-ai-and-the-rise

---

## 8. Jim Keller — Open Chiplet Architecture / RISC-V / Software-Defined Hardware

### What He's Saying
Keller (CEO of Tenstorrent, legendary chip architect behind AMD Zen, Apple A4/A5, Tesla Autopilot FSD chip) is focused on the **infrastructure layer**: how AI hardware and software must co-evolve for the next generation of AI.

**The paradigm he's betting on: Open, modular, software-defined AI compute.**

**From the Tech Threads interview (November 2025):**

**Core thesis 1: Modularity enables complexity**
> "By making everything have the right abstraction layers and modularity, we can actually build really complicated things out of simpler components."

The chiplet revolution: deconstruct monolithic chips into smaller, specialized components. This enables adaptive processor integration, independent development, easier scaling and customization.

**Core thesis 2: RISC-V for open AI integration**
> "We picked RISC-V so we can build our own CPU and connect it with AI in a really flexible, novel way. The way that the processor integrates with the AI is novel, and I think it's really going to unlock a lot of creativity on how to build future software stacks."

**The Open Chiplet Atlas architecture**: Tenstorrent's framework for chiplet design with published support IP, verification environments, performance testing frameworks, and flexible interface configurations.

**Ascalon RISC-V IP** (December 2025): Tenstorrent unleashing RISC-V processor IP to disrupt data centers — "Ascalon-X" targeting LG, Hyundai, and others for 2026 SoC deployments.

**Key Keller philosophy:**
- "Committees are good for standards that stabilize, not for speed. You can't meet once a quarter to solve a thousand problems."
- Software-defined hardware: AI systems need dynamic, adaptable software layers
- Markets "not well served by NVIDIA" — the non-hyperscale, non-datacenter AI compute market
- AMD, Amazon, Tesla, Groq have all made good chips but failed at training because they lack the software stack
- **Software sovereignty** (same as Hotz, different path) is the competitive moat

**2026 roadmap:** First wave of Ascalon-powered SoCs from Tenstorrent customers shipping in 2026. Open Chiplet Initiative gaining industry traction.

### Relevance to 16MB / 10-Minute Training

**High relevance on the infrastructure side.**

1. **Non-NVIDIA training paths**: Keller is explicitly building AI hardware and software stacks for markets that can't afford NVIDIA. For 10-minute training loops on constrained budgets, Tenstorrent + tinygrad represents a plausible stack.

2. **Software-defined hardware**: Keller's insight that "make invisible system-level dynamics visible to the architect" applies to model architecture. For 16MB, making the model's internal compute patterns (attention sparsity, layer utilization, memory access patterns) visible and tunable gives more optimization handles.

3. **Open standards enable innovation**: The chiplet/RISC-V movement is creating a hardware ecosystem where small teams can build custom inference/training hardware. 16MB models that need to run in specialized environments (edge devices, embedded systems) benefit from this ecosystem.

4. **"You can't design complicated things" / simplicity principle**: Keller's Sysbibit rule — build simple components that compose cleanly — applies directly to model architecture. For 16MB, simplicity of architecture (fewer types of operations, cleaner abstractions) may outperform complex architectures at the same parameter count.

**Concrete application:** Explore Tenstorrent's Blackhole hardware (p150 cards, now 120 tensor cores) as a training substrate for 10-minute runs. The combination of Tenstorrent hardware + tinygrad framework represents the most cost-efficient non-NVIDIA path for this use case.

### URLs
- Tech Threads interview: https://semiengineering.com/shaping-the-future-of-ai-processors-a-tech-threads-conversation-with-jim-keller/
- Ascalon RISC-V IP: https://markets.financialcontent.com/wral/article/tokenring-2025-12-25-the-arm-killer-jim-kellers-tenstorrent-unleashes-ascalon-risc-v-ip-to-disrupt-the-data-center
- Open chiplet initiative: https://www.sdxcentral.com/news/jim-kellers-tenstorrent-launches-initiative-targeting-open-chiplet-design/
- Tom's Hardware profile: https://www.tomshardware.com/tech-industry/artificial-intelligence/chip-design-legend-jim-keller-aims-for-tenstorrent-wins-in-markets-not-well-served-by-nvidia

---

## Cross-Cutting Synthesis: What They All Agree On

Despite dramatic disagreements on architecture and approach, five themes emerge across all eight:

### 1. Scaling is NOT dead, but it's no longer sufficient
Every visionary here (even LeCun, who hates LLMs) acknowledges that scaling still works at the top. The shift is: *small players can no longer win by scaling*. The opportunities are elsewhere.

**16MB implication:** Stop competing on raw capacity. Compete on efficiency, specialization, and the ability to train quickly.

### 2. Verifiable rewards > human feedback
Karpathy names it most clearly (RLVR), but Sutskever, Amodei, Hassabis, and Dean all point in the same direction: training signals from *correctness-checking* (not human preference) is the paradigm shift of 2025–2026.

**16MB implication:** Build your training loop around verifiable tasks. Even simple math/code/logic with automated grading gives stronger learning signal than text prediction in the same wall-clock time.

### 3. Generalization from fewer examples is unsolved
Sutskever identifies this as the core frontier. Current models generalize dramatically worse than humans from small example counts.

**16MB implication:** This is both a problem and an opportunity. A 16MB model that generalizes well from few examples is more useful than one that achieves narrow benchmark scores. Curriculum design, RLVR, and JEPA-style self-supervised learning all target this.

### 4. Local, specialized models have a real future
Hotz, Keller, Dean, and LeCun all point toward the same structural outcome: giant models for hyperscalers, specialized/local models for everyone else.

**16MB implication:** The market you're targeting with parameter golf is *real* and being validated from multiple directions simultaneously.

### 5. Architecture matters more at small scale
JEPA (LeCun), chiplet modularity (Keller), tinygrad minimalism (Hotz) — all emphasize that architectural choices dominate at constrained compute/parameter budgets. The right architecture can outperform a poorly designed model many times larger.

**16MB implication:** Don't default to transformer architecture just because it's dominant. Explore hybrid architectures: JEPA-style latent prediction heads, sparse attention, state-space models. The parameter budget forces the question.

---

## Priority Action List: What to Try in 16MB / 10 Minutes

Based on all eight visionaries, ranked by expected signal-to-effort ratio:

| # | Action | Source | Effort | Expected Impact |
|---|--------|---------|--------|-----------------|
| 1 | **Add RLVR to training loop** — verifiable math/code rewards | Karpathy + Sutskever | Medium | Very High |
| 2 | **Knowledge distillation from 70B teacher** — use large model outputs as training targets | Dean | Low | Very High |
| 3 | **JEPA-style masking loss** — predict latent embeddings, not tokens | LeCun | High | High |
| 4 | **Constitutional self-critique step** — model reviews its own outputs against 3-5 principles | Amodei | Low | Medium-High |
| 5 | **Synthetic verifiable dataset** — 5,000 checkable problems auto-generated | Karpathy + Hassabis | Low | High |
| 6 | **Try tinygrad as framework** — minimalist, efficient, non-NVIDIA | Hotz | Medium | Medium |
| 7 | **Causal physics puzzles** — simulate state transitions with ground truth | Hassabis | Medium | Medium |
| 8 | **Specialize, don't generalize** — pick ONE task type, train only on that | Dean + Keller | Low | High |

---

## The Bet They're All Making That Parameter Golf Can Exploit

Every single person on this list is betting — in different ways — that **the next decade of AI value is not in bigger models, but in smarter training**. 

The specific insight: a 16MB model trained with RLVR, distilled from a 70B teacher, trained for 10 minutes on a curated verifiable dataset, with a JEPA-style prediction head and a constitutional self-critique layer — that model is *playing the next paradigm*, not the current one.

It's not trying to out-scale GPT-4. It's trying to out-learn it.

---

*File written by research subagent | Sources: live web research, March 2026 | All URLs checked*
