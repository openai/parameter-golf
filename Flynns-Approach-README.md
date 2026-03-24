# Flynn's Approach: Skill Forge — Autonomous ML Experimentation

**Author**: Flynn Cruse ([@FlynnCruse](https://github.com/FlynnCruse))

## Philosophy

Rather than manually tuning one experiment at a time, I built an **autonomous experimentation system** that runs overnight — modifying code, training, measuring, keeping improvements, discarding regressions, and evolving its own strategies based on what works.

The core insight: **research knowledge compounds faster when the system that generates experiments can also learn from their outcomes.** A human researcher carries intuition between runs. My system encodes that intuition into skills that crystallize over time — starting as broad heuristics and hardening into specific playbooks as patterns prove themselves.

## The System: Skill Forge

Skill Forge is a three-tier autonomous experiment system built as Claude Code skills:

```
/skill-forge                      ← Single entry point
│
├── Experiment Loop (NEVER STOP)
│   ├── Exploration Policy        ← Explore vs exploit, skill selection
│   ├── Results Analyzer          ← Pattern detection across history
│   └── Domain Skills             ← Guide what to modify:
│       ├── Architecture Search   ← Depth, width, attention, skip connections
│       ├── Optimizer Tuning      ← Muon, EMA/SWA, warmdown, LR schedules
│       ├── Compression Eng       ← Quantization, GPTQ-lite, zstd, artifact budget
│       ├── Eval Strategy         ← Sliding window, stride, TTT
│       ├── Init Strategy         ← OrthoInit, muP, embedding init
│       └── Training Dynamics     ← Batch size, throughput, data efficiency
│
└── Meta Layer (every 5 experiments)
    ├── Skill Evaluator           ← Score which skills drive gains
    └── Skill Improver            ← Crystallize heuristics → playbooks
```

### How It Works

1. **Explore**: The system selects a domain skill and proposes a hypothesis
2. **Modify**: One focused change to `train_gpt.py` per experiment
3. **Train**: Run training with the modification
4. **Measure**: Parse val_bpb from the training log
5. **Keep/Discard**: If BPB improved and artifact < 16MB, keep. Otherwise revert.
6. **Evolve**: Every 5 experiments, evaluate which skills are working and improve them

### Skill Crystallization

Skills start as broad principles:
> "More layers > wider layers for parameter-constrained models"

As experiments accumulate evidence, the meta-layer crystallizes them into specific playbooks:
> "**PROVEN** (7/10 experiments): 11 layers at 512dim is optimal. Each layer beyond 11 requires int5 MLP to fit 16MB. Step 1: Set num_layers=11. Step 2: If artifact > 15.5MB, reduce MLP before reducing layers."

This mirrors how human research knowledge matures — from hypothesis to validated protocol.

## Inspirations

- **[Karpathy's autoresearch](https://github.com/karpathy/autoresearch)**: The core modify→train→measure→keep/discard loop. My system extends this with skill-guided hypothesis generation and meta-learning.
- **[DeepMind's AlphaEvolve](https://arxiv.org/abs/...)**: Evolutionary code search with LLM mutation. The island-based population management and heuristic discovery patterns informed the exploration/exploitation policy.
- **[Claude Code skill-creator](https://github.com/anthropics/claude-code)**: The skill file format, progressive disclosure, and self-improving evaluation patterns. Each domain skill uses binary eval criteria that enable automated optimization.

## Techniques

The domain skills encode research-backed strategies seeded from competitive analysis of the leaderboard (baseline 1.2244 → SOTA 1.1228 BPB) and deep literature review (13+ arXiv papers from Feb-March 2026):

### Architecture (drove ~80% of competitive gains)
- 11L/512d depth-first scaling with U-Net skip connections
- XSA (Exclusive Self Attention) on deepest layers [arXiv 2603.09078]
- 3x MLP expansion with relu² (more parameter-efficient than SwiGLU at this scale)
- Partial RoPE (16/64 dims), SmearGate + BigramHash for small-vocab augmentation

### Optimization (drove ~15% of gains)
- Muon optimizer with validated hyperparameters (lr=0.025, momentum=0.99, WD=0.04)
- EMA weight averaging (decay=0.997) stacked with Tight SWA
- Wallclock-based linear warmdown over 3500 iterations
- Six Muon variants explored: Muon+, NorMuon, MUD, RMNP, Mousse, AdEMAMix

### Compression (enables larger models)
- Late QAT (STE int6) at LR scale < 0.15 — quantization gap reducible to 0.0000 BPB
- GPTQ-lite per-row clip percentile search (5 candidates, zero training cost)
- Mixed precision: int5 MLP (1.88x zstd ratio), int6 attention, FP16 embeddings
- zstd level 22 final compression

## Build Process

The system was built in a single session:
1. **Research**: Explored Karpathy autoresearch, DeepMind AlphaEvolve/FunSearch, Claude Code skill-creator patterns
2. **Design**: Brainstormed the three-tier architecture (meta → infrastructure → domain)
3. **Implement**: 12 skills with references, agents, eval-criteria, frontiers.json — all created in parallel
4. **Enrich**: Deep online research (architecture, optimization, compression) integrated into domain skills
5. **Local Setup**: RTX 4070 Laptop GPU profiling, VRAM ceiling finding, local training scripts

## Results

*Results from competition hardware runs will be added here.*

| Run | val_bpb | Artifact | Key Technique | Date |
|-----|---------|----------|---------------|------|
| Local baseline | 1.8990 | N/A (local) | 7L/512d unmodified, RTX 4070, 4min | 2026-03-24 |

## Reproducibility

```bash
# Setup
git clone https://github.com/FlynnCruse/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Competition run (8×H100)
RUN_ID=submission torchrun --standalone --nproc_per_node=8 train_gpt.py

# Local iteration (RTX 4070 or similar)
bash scripts/local_train.sh my_experiment
```
