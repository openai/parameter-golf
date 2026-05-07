# Hunter Alpha — Parameter Golf Research Handoff

## What This Is

You're helping us compete in **OpenAI's Parameter Golf** — a live competition to train the best language model that fits in **16MB** and trains in **10 minutes on 8×H100 GPUs**. Scored by bits-per-byte (bpb) on English text. Lower = better. Deadline: April 30, 2026.

**Current #1:** 1.1228 bpb (transformer with int6 quantization, 11 layers, 512-dim)
**Pending PRs:** As low as 1.0523 bpb (using test-time training)
**Projected winner by April 30:** 1.02-1.04 bpb

## The Problem

Every competitor is doing the same thing: optimizing transformers incrementally. Same architecture, same techniques, reading each other's PRs. The leaderboard is a pile-on.

We want to find approaches **nobody else is using** that could leapfrog the competition.

## What We've Already Researched

Our team has produced 13+ research files covering:
- Standard LLM optimization (quantization, attention tricks, training schedules)
- Post-transformer architectures (Mamba, RWKV, xLSTM, liquid NNs)
- Chinese ML ecosystem (DeepSeek MLA, MiMo, Kimi)
- Quantum/emerging compute (tensor networks, photonic, neuromorphic)
- Cross-domain insights (information theory, neuroscience, signal processing, compression theory)
- First-principles constraint analysis
- Training paradigm shifts (evolutionary search, distillation, meta-learning, curriculum learning)
- Broad AI landscape (50+ architectures across 10 categories)
- Competition social intelligence

## What We HAVEN'T Researched (Your Mission)

We've been thinking inside AI. Think outside it. Explore these new pillars:

### New Research Pillars
1. **Cognitive science** — How do children learn language so efficiently with so little data? What does developmental psychology suggest about learning architecture?
2. **Linguistics** — Chomsky's universal grammar, construction grammar, usage-based theory. Do linguistic theories about how language WORKS suggest better model architectures?
3. **Music/audio ML** — Audio models (WaveNet, spectrograms, mel filters) are extremely parameter-efficient. Can audio processing techniques improve text models?
4. **Game AI** — AlphaGo/AlphaZero learned superhuman performance through self-play. Could self-play or game-theoretic training improve language models?
5. **Robotics control** — Real-time control systems optimize under extreme latency and compute constraints. What techniques do they use that we could steal?
6. **Ecology & evolution** — Fitness landscapes, niche specialization, co-evolution, genetic drift. Nature solved optimization over billions of years.
7. **Mathematics** — Category theory, algebraic topology, abstract algebra applied to neural networks. Any structural insights?
8. **Economics** — Market microstructure, mechanism design, auction theory. Could economic models improve expert routing in Mixture-of-Experts?
9. **Materials science** — Metamaterials, self-organizing systems, crystallography. Patterns in physical structures that could inspire neural architectures?
10. **Compression science beyond ML** — Video codecs (H.265, AV1), image compression (JPEG XL), genomic compression. These fields compress data incredibly efficiently — what can we learn?

### Strategic Synthesis Questions
After exploring new pillars, answer these:
- What combination of techniques gives the highest probability of a **top-3 finish**?
- What's the **single most contrarian bet** we could make?
- If you had to bet $10,000 on **ONE approach** beating the current leader, what would it be and why?
- What would a submission look like that makes the judges say **"we've never seen this before"**?

## The Hard Constraints
- **16,000,000 bytes** total (code + compressed model weights)
- **10 minutes** training on 8×H100 SXM GPUs
- **10 minutes** evaluation on 8×H100 SXM GPUs
- Must fit in one file: `train_gpt.py`
- Scored on FineWeb validation set (internet text)
- No external downloads during evaluation
- Test-time training IS allowed on already-evaluated tokens
- Custom tokenizers allowed (examined more carefully)
- AI assistance explicitly encouraged

## What We Need From You
- **Be opinionated.** We don't want a literature review. We want your judgment on what WINS.
- **Be specific.** Name techniques, papers, repos, people.
- **Be contrarian.** The obvious path is "stack more transformer tricks." Tell us what the non-obvious path is.
- **Write as you go.** Don't wait until the end to write your findings.

## Output
Write your findings and strategic synthesis to a document. Cover the new pillars and end with your strategic recommendations.
