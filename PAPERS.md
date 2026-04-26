# Curated arxiv reading list

## State Space Models (current focus)

### Foundational
- HiPPO: Recurrent Memory with Optimal Polynomial Projections — arxiv:2008.07669 (Voelker, Gu, Re)
- S4: Efficiently Modeling Long Sequences with Structured State Spaces — arxiv:2111.00396 (Gu, Goel, Re)
- DSS: Diagonal State Spaces are as Effective as Structured State Spaces — arxiv:2203.14343 (Gupta, Gu, Berant)
- S4D: On the Parameterization and Initialization of Diagonal State Space Models — arxiv:2206.11893 (Gu, Goel, Gupta, Re)

### Selective and gated
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces — arxiv:2312.00752 (Gu, Dao)
- Mamba-2 / SSD: Transformers are SSMs (state space duality) — arxiv:2405.21060 (Dao, Gu)
- GLA: Gated Linear Attention — arxiv:2312.06635 (Yang et al.) — data-dep diagonal gate, chunkwise matmul-friendly
- Hyena: Towards Convolutional Language Models — arxiv:2302.10866 (Poli et al.) — implicit long conv via small FFN, FFT-conv
- RetNet — arxiv:2307.08621 (Sun et al.) — special case of SSD with constant scalar decay
- GateLoop — arxiv:2311.01927 (Katsch) — concurrent with SSD

### Mamba-3 (bleeding-edge, paper not stable)
- Mamba-3 Part 1 (Tri Dao, 2026): https://tridao.me/blog/2026/mamba3-part1/
- Mamba-3 Part 2: https://tridao.me/blog/2026/mamba3-part2/
- Goomba Lab Mamba-3 cross-post: https://goombalab.github.io/blog/2026/mamba3-part1/

### Hybrids
- Jamba: A Hybrid Transformer-Mamba — arxiv:2403.19887
- Zamba: Compact Mamba-Based Hybrid — arxiv:2405.16712
- Samba — arxiv:2406.07522
- Hymba: Hybrid-head Architecture — arxiv:2411.13676

### Quantization
- Quamba: Post-Training Quantization for SSMs — arxiv:2410.13229
- Quamba2 W4A8 — arxiv:2503.22879

### Limitations / weaknesses
- Zoology: Measuring and Improving Recall in Efficient Language Models — arxiv:2312.04927
- The Illusion of State in State-Space Models — arxiv:2404.08819

### Surveys
- A Survey on State Space Models — arxiv:2404.09516

## Transformer/optimizer techniques retained (for hybrid composition)

Reference list. Read selectively when grounding a hypothesis. Fetch with `curl https://arxiv.org/pdf/<id>` only when you specifically need it. **Do not browse the web freely.**

## Optimization

- Loshchilov & Hutter, AdamW — `arxiv:1711.05101`
- Keller Jordan et al., Muon optimizer — `https://kellerjordan.github.io/posts/muon/`
- Liu et al., Sophia — `arxiv:2305.14342`
- Zhao et al., Cautious Optimizers — `arxiv:2411.16085`

## Parameter efficiency / weight tying / sharing

- Press & Wolf, Tied embeddings — `arxiv:1608.05859`
- Lan et al., ALBERT (cross-layer sharing) — `arxiv:1909.11942`
- Dehghani et al., Universal Transformers — `arxiv:1807.03819`
- Bae et al., Relaxed Recursive Transformers — `arxiv:2410.20672`

## Architecture

- Su et al., RoPE — `arxiv:2104.09864`
- Ainslie et al., GQA — `arxiv:2305.13245`
- DeepSeek-AI, MLA in DeepSeek-V2 — `arxiv:2405.04434`
- Shazeer, GLU variants — `arxiv:2002.05202`
- Loshchilov et al., nGPT (normalized transformer) — `arxiv:2410.01131`
- Zhu et al., ResFormer / value residuals — `arxiv:2410.17897`

## Quantization

- Frantar et al., GPTQ — `arxiv:2210.17323`
- Esser et al., LSQ (learned step size QAT) — `arxiv:1902.08153`
- Bhalgat et al., LSQ+ — `arxiv:2004.09576`
- Ma et al., BitNet b1.58 (ternary) — `arxiv:2402.17764`
- Dettmers et al., LLM.int8() — `arxiv:2208.07339`

## Scaling laws / training dynamics

- Kaplan et al., Neural scaling laws — `arxiv:2001.08361`
- Hoffmann et al., Chinchilla — `arxiv:2203.15556`
- Bahri et al., Explaining scaling laws — `arxiv:2102.06701`

## Test-time compute / TTT

- Sun et al., Test-time training (general) — `arxiv:1909.13231`
- Akyürek et al., Surprising effectiveness of test-time training — `arxiv:2411.07279`

## Tokenization

- Kudo & Richardson, SentencePiece — `arxiv:1808.06226`
- Schmidt et al., H-Net (no tokenizer) — `arxiv:2509.18781` (verify the ID before fetching; this one is recent and less certain)

## Reference codebases (not papers, read selectively)

- modded-nanogpt — github.com/KellerJordan/modded-nanogpt
- nanoGPT — github.com/karpathy/nanoGPT
- autoresearch-macos — github.com/karpathy/autoresearch-macos (the workflow pattern this harness is modeled on)
