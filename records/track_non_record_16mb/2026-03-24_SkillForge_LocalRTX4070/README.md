# Non-Record Submission: Skill Forge — Autonomous ML Experimentation System

**val_bpb: 1.8990** (local RTX 4070, 7L/512d, seq512, 4 min) | **Non-record: approach demonstration**

## What This Is

This is a **non-record submission** demonstrating an autonomous ML experimentation system called **Skill Forge**, built to compete in Parameter Golf. The system is designed to run overnight autoresearch-style loops, modifying `train_gpt.py`, training, measuring `val_bpb`, keeping improvements, and discarding regressions — while evolving its own optimization strategies based on accumulated evidence.

This submission includes local validation results from a consumer laptop GPU. Competition-grade results on 8×H100 are planned after compute grant award.

## Compute Constraints & What I Did About Them

I don't currently have access to 8×H100 GPUs. Instead of waiting, I built the full experimentation pipeline and validated it locally:

| Constraint | Solution |
|-----------|----------|
| RTX 4070 Laptop (8GB VRAM) vs H100 (80GB) | Scaled model to 7L/512d (competition uses 11L/512d). Same weight matrices and attention patterns at 512d width — techniques transfer directly. |
| No FlashAttention 3 (sm_89 vs sm_90) | Math SDPA backend. Slower but produces identical model weights. |
| No torch.compile kernels | Disabled via env var. ~30% slower throughput, irrelevant for technique ranking. |
| Post-training quantize+eval OOMs at 8GB | Added `SKIP_QUANT_EVAL=1` — clean exit after training. Training val_bpb used for technique ranking. |
| 4-minute local budget vs 10-minute competition | ~780 steps locally vs ~7000 on H100. Sufficient for relative technique comparison. |

**Key insight**: At 512d model width, the weight matrices, attention patterns, optimizer dynamics, and initialization strategies are identical to competition config. A technique that improves BPB at 7L/512d locally will also improve at 11L/512d on H100 — the absolute numbers differ but the relative ranking transfers.

## The System: Skill Forge

An autonomous three-tier experiment system built as Claude Code skills:

### Architecture
```
/skill-forge                      ← Single entry point
├── Experiment Loop (NEVER STOP)
│   ├── Exploration Policy        ← Explore vs exploit balance
│   ├── Results Analyzer          ← Pattern detection
│   └── 6 Domain Skills           ← Seeded from competitive analysis + 2026 research
│       ├── Architecture Search   ← XSA, partial RoPE, SmearGate, U-Net skips
│       ├── Optimizer Tuning      ← Muon variants, EMA, warmdown, 6 arxiv papers
│       ├── Compression Eng       ← QAT, GPTQ-lite, mixed precision, 7 arxiv papers
│       ├── Eval Strategy         ← Sliding window, TTT
│       ├── Init Strategy         ← OrthoInit, muP scaling
│       └── Training Dynamics     ← Batch size, throughput, curriculum
└── Meta Layer (every 5 experiments)
    ├── Skill Evaluator           ← Score which strategies work
    └── Skill Improver            ← Crystallize heuristics → playbooks
```

### How Skills Evolve

Skills start as broad heuristics seeded from competitive analysis and deep research:
> "More layers > wider layers for parameter-constrained models"

As experiments accumulate evidence, the meta-layer **crystallizes** them into specific playbooks:
> "PROVEN (7/10 experiments): 11L at 512dim is optimal. If artifact > 15.5MB, reduce MLP before layers."

Each skill also has:
- `config/eval-criteria.json` — Binary evaluation criteria for automated optimization
- `config/frontiers.json` — Machine-readable exploration frontiers with risk/impact estimates
- `agents/hypothesis-generator.md` — Subagent prompt for structured JSON recommendations
- `references/` — Deep knowledge: proven patterns, parameter budgets, implementation code

### Research Integrated

Domain skills are seeded from 13+ arXiv papers (Feb-March 2026) and direct analysis of all 21 competition submissions:

- **Architecture**: XSA [2603.09078], relu² vs SwiGLU analysis, SmearGate implementation, MTP auxiliary loss
- **Optimization**: 6 Muon variants (Muon+, NorMuon, MUD, RMNP, Mousse, AdEMAMix), exact SOTA hyperparameters, LR sweep data
- **Compression**: GPTQ-lite implementation code, torch.compile QAT constant-folding bug, quantization gap tables, 7 frontier papers (Astro, ScaleBITS, RAMP, pQuant, Sparse-BitNet)

## Local Results

| Metric | Value |
|--------|-------|
| Config | 7L, 512d, 8 heads, 2x MLP, seq_len=512, 16K batch |
| val_bpb | 1.8990 (step 779, wallclock 240s) |
| Steps | 779 in 240 seconds (308ms/step) |
| Peak VRAM | 3404 MiB |
| GPU | NVIDIA RTX 4070 Laptop (8GB) |

**Note**: This val_bpb is not directly comparable to competition results. The competition baseline (9L/512d/seq1024 on H100) achieves 1.2244 BPB. The difference is due to fewer layers (7 vs 9+), shorter sequence length (512 vs 1024+), and fewer training steps (779 vs 7000+).

## Planned Competition Run (After Compute Grant)

With 8×H100 SXM access, the plan is:

1. **Validate baseline**: Run unmodified `train_gpt.py` on 8×H100 to establish reference BPB
2. **Apply winning local techniques**: Transfer architecture, optimizer, and compression improvements proven locally
3. **Run Skill Forge overnight**: ~48 experiments in 8 hours at competition dimensions
4. **3-seed submission**: Statistical significance validation with seeds 1337, 42, 2024

Expected improvements based on competitive analysis:
- Architecture scaling (9L→11L, 3x MLP): -0.035 to -0.050 BPB
- Optimizer tuning (EMA, WD=0.04, Muon momentum): -0.010 to -0.015 BPB
- Compression (int6 QAT, GPTQ-lite): enables larger model within 16MB

## Inspirations

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — The core modify→train→measure→keep/discard loop
- [DeepMind's AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve/) — Evolutionary code search with population management
- [Claude Code skill-creator](https://github.com/anthropics/claude-code) — Self-improving skill patterns with binary evaluation criteria

## Reproducibility

```bash
# Clone and setup
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Local run (RTX 4070 or similar, 8GB+ VRAM)
bash scripts/local_train.sh my_experiment

# Competition run (8×H100)
RUN_ID=submission torchrun --standalone --nproc_per_node=8 train_gpt.py
```
