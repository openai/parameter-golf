# 🔬 Autoresearch × Parameter Golf — Integration Plan

> **Goal**: Use Karpathy's Autoresearch framework to autonomously optimize our Parameter Golf submission overnight, leveraging an AI agent that iterates on `train_gpt.py` while we sleep.

---

## 1. What Is Autoresearch?

Autoresearch (by @karpathy) is an **autonomous AI research framework** where:

1. An AI agent (Claude/Codex) reads a `program.md` "skill file"
2. It modifies a single `train.py` file — tweaking architecture, hyperparameters, optimizer, etc.
3. It runs a training experiment (fixed 5-min time budget per run)
4. It reads the output metric (`val_bpb`)
5. If the result improved → **keep** the change (advance the Git branch)
6. If the result worsened → **discard** (git reset)
7. **Loop forever** — ~12 experiments/hour, ~100 experiments while you sleep

### Autoresearch Architecture

| File | Role | Who Edits |
|---|---|---|
| `prepare.py` | Fixed: data download, tokenizer, dataloader, eval metric | **Nobody** (read-only) |
| `train.py` | Model architecture, optimizer, hyperparams, training loop | **The AI agent** |
| `program.md` | Instructions/strategy for the agent | **The human (us)** |
| `results.tsv` | Experiment log (commit, val_bpb, memory, status, description) | **The AI agent** |

### Key Design Principles
- **Single file to modify** → clean diffs, no cross-file coordination
- **Fixed 5-min time budget** → experiments are directly comparable regardless of changes
- **Git branch per session** → full experiment history, easy rollback
- **Autonomous = no human intervention** → agent runs indefinitely until manually stopped

---

## 2. How Autoresearch Maps to Parameter Golf

The two systems are **remarkably similar** — both are GPT training setups with the Muon optimizer and fixed time budgets. Here's the mapping:

| Aspect | Autoresearch | Parameter Golf |
|---|---|---|
| **Model** | GPT (nanochat-derived) | GPT (modded-nanogpt-derived) |
| **Optimizer** | MuonAdamW (combined) | Muon + separate AdamW |
| **Activation** | ReLU² | ReLU² |
| **Attention** | Flash Attention 3 + sliding window | Flash SDP + GQA |
| **Time Budget** | 5 minutes (1 GPU) | 10 minutes (8 GPUs) |
| **Metric** | val_bpb (bits per byte) | val_bpb (bits per byte) |
| **Data** | ClimbMix-400B | FineWeb-10B |
| **Vocab Size** | 8192 | 1024 |
| **Seq Len** | 2048 | 1024 (baseline), 2048+ (optimized) |
| **GPUs** | 1 GPU | 1 GPU (dev) → 8 GPUs (final) |
| **File Edited** | `train.py` | `train_gpt.py` |

### Critical Differences to Handle
1. **Different datasets**: Autoresearch uses ClimbMix-400B, Parameter Golf uses FineWeb-10B
2. **Different tokenizers**: 8192 BPE vs 1024 SentencePiece BPE
3. **Different eval functions**: Autoresearch's `evaluate_bpb()` in `prepare.py` vs Parameter Golf's `eval_val()` in `train_gpt.py`
4. **Quantization**: Parameter Golf has a 16MB artifact cap — the AI agent must also optimize for post-quantization quality
5. **Multi-GPU**: Parameter Golf's final submission needs `torchrun --nproc_per_node=8`, but we can iterate on 1 GPU

---

## 3. The Plan: Step-by-Step

### Phase 0: Environment Setup (30 min)

1. **Rent a 1×H100 on DigitalOcean** ($3.39/hr)
2. **SSH into the machine** and set up:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone our parameter-golf repo
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Download FineWeb data
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Install deps
pip install -r requirements.txt
pip install zstandard

# Verify baseline works (quick smoke test)
RUN_ID=smoke \
ITERATIONS=50 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Phase 1: Create the Autoresearch Adapter (1-2 hours, done locally by us)

We need to create **3 files** inside `parameter-golf/` to wire up the autoresearch loop:

#### File 1: `prepare_pgolf.py` (The Fixed Evaluation Harness)

This wraps Parameter Golf's data loading and evaluation into Autoresearch's expected interface. It is **read-only** for the agent.

```python
"""
Fixed evaluation harness for Parameter Golf autoresearch.
DO NOT MODIFY. The agent modifies train_pgolf.py only.
"""
import os, glob, math, time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch

# --- Fixed Constants ---
TIME_BUDGET = 300          # 5 minutes per experiment (1 GPU dev mode)
DATA_PATH = "./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
VOCAB_SIZE = 1024
TRAIN_SEQ_LEN = 1024       # Default, agent can override in train_pgolf.py
VAL_BATCH_SIZE = 524_288

# ... (wraps load_validation_tokens, eval_val, 
#      quantize_state_dict, compress, measure artifact size)

def evaluate_submission(model, args):
    """
    Full evaluation pipeline:
    1. Compute val_bpb (with sliding window if enabled)
    2. Quantize model (int8/int6/int5 as configured)
    3. Compress with zstd/zlib
    4. Report: val_bpb, artifact_size, quant_gap
    """
    ...
```

#### File 2: `train_pgolf.py` (The File the Agent Modifies)

This is a copy of the current SOTA `train_gpt.py` **reformatted to match autoresearch conventions**:
- All hyperparameters at the top as simple constants
- Clear section markers
- Fixed 5-min time budget for 1-GPU iteration
- End-of-training calls `evaluate_submission()` from `prepare_pgolf.py`
- Prints the standard autoresearch output format:

```
---
val_bpb:          1.1428
artifact_bytes:   15965978
quant_gap:        0.0012
training_seconds: 300.1
total_seconds:    385.2
peak_vram_mb:     45060.2
num_steps:        3450
num_params_M:     22.4
```

#### File 3: `program_pgolf.md` (Our Research Strategy for the Agent)

This is the **brain** — the instructions we give to the AI agent. This is where our strategic advantage lives:

```markdown
# Parameter Golf Autoresearch Program

## Context
You are optimizing a GPT language model for the OpenAI Parameter Golf 
challenge. The goal is to minimize val_bpb (bits per byte) on FineWeb 
validation data, subject to these HARD constraints:

1. Compressed artifact (code + quantized model) must be ≤ 16,000,000 bytes
2. Training must complete within the time budget (5 min on 1 GPU for dev)
3. val_bpb is the ONLY metric that matters (lower is better)

## Current SOTA Techniques (already in the baseline)
- 10 layers, 512 dim, MLP 3×, GQA, ReLU²
- SmearGate + BigramHash(10240)
- Int5 MLP / Int6 attention / FP16 embedding quantization
- Muon WD=0.04, momentum=0.99, orthogonal init
- SWA (start_frac=0.4, every=50 steps)
- Sliding window eval (stride=64)
- zstd-22 compression

## What to Try (Priority Order)
1. QUANTIZATION: Try int4 for MLP weights. Try mixed int3/int4/int5.
2. ARCHITECTURE: Try 11-12 layers (if quant savings allow). Try MLP 4×.
3. BIGRAM HASH: Try 16384 or 20480 buckets. Try trigram hash.
4. OPTIMIZER: Sweep WD (0.02-0.06), LR, momentum. Try cosine schedule.
5. NOVEL: Try MoE with 2-4 experts. Try multi-token prediction head.
6. EVALUATION: Better sliding window stride. Try LoRA TTT.

## What NOT to Try (Known Failures)
- SwiGLU (45% slower per step, net negative)
- Layer recurrence (catastrophic on short training)
- LZMA compression (worse than zlib for weight data)

## CRITICAL: Artifact Size Check
After EVERY experiment, verify artifact_bytes ≤ 16,000,000.
If artifact is too large, the experiment is INVALID regardless of val_bpb.

## The Experiment Loop
[Standard autoresearch loop from program.md]
...
```

### Phase 2: Start the Autonomous Agent (5 min setup, then hands-off)

1. **SSH into the H100 machine**
2. **Create Git branch**: `git checkout -b autoresearch/pgolf-mar22`
3. **Launch the AI agent** (Claude Code / Codex / Gemini CLI — whichever is available):

```bash
# Example with Claude Code:
claude --project /workspace/parameter-golf

# Then prompt:
# "Read program_pgolf.md and kick off a new experiment session. 
#  Start by running the baseline train_pgolf.py as-is."
```

4. **Walk away** — the agent loops autonomously:
   - Each experiment: ~5 min training + ~2 min eval/overhead ≈ **~7 min total**
   - **~8-9 experiments per hour**
   - **~70 experiments overnight (8 hours)**

### Phase 3: Review Results in the Morning (30 min)

1. **SSH back in** and check `results.tsv`:

```bash
cat results.tsv | column -t -s $'\t'
```

2. **Analyze the winning experiments**:
   - Which changes gave the biggest improvements?
   - What's the best val_bpb achieved?
   - What's the artifact size?

3. **Take the best `train_pgolf.py`** and convert it back to a proper `train_gpt.py` for the 8×H100 submission

### Phase 4: Scale to 8×H100 for Final Submission (1 hour)

1. **Spin up the 8×H100 instance** on DigitalOcean ($23.92/hr)
2. **Adapt the best train_pgolf.py** for multi-GPU (add torchrun/DDP)
3. **Run 3 seeds** to verify statistical significance
4. **Create the PR** to the parameter-golf repo with our submission

---

## 4. Cost Estimate

| Phase | GPU | Duration | Cost |
|---|---|---|---|
| **Setup + Baseline** | Local RTX 5060 | 1 hour | $0.00 |
| **Overnight Autoresearch** | Local RTX 5060 | 8 hours | $0.00 (electricity only) |
| **Final 8×H100 Runs** | 8×H100 (OpenAI credit) | 1 hour | $0.00 (covered by $25 credit) |
| **Total** | | **~10 hours** | **~$0.00** |

> Using the local RTX 5060 for all development + OpenAI's $25 credit for final submission runs.

---

## 5. What the Agent Will Optimize

The AI agent iterates autonomously, trying changes like:

### Iteration Examples (what a typical overnight session looks like)

```
Exp 001: Baseline                           → val_bpb: 1.1850 (1-GPU baseline)   KEEP
Exp 002: Increase bigram_hash to 16384      → val_bpb: 1.1842 (−0.0008)          KEEP
Exp 003: Try int4 MLP quantization          → val_bpb: 1.1835 (−0.0007)          KEEP
Exp 004: Add 11th layer                     → CRASH (artifact > 16MB)             DISCARD
Exp 005: 11th layer + int4 MLP             → val_bpb: 1.1810 (−0.0025)          KEEP
Exp 006: Try trigram hash (8192 buckets)    → val_bpb: 1.1805 (−0.0005)          KEEP
Exp 007: MLP 3.5× expansion               → val_bpb: 1.1790 (−0.0015)          KEEP
Exp 008: WD=0.05 (from 0.04)              → val_bpb: 1.1795 (+0.0005)           DISCARD
Exp 009: WD=0.03                           → val_bpb: 1.1788 (−0.0002)          KEEP
Exp 010: Try MoE with 2 experts            → val_bpb: 1.1760 (−0.0028)          KEEP
...continues for 70+ experiments...
```

### Why This Works for Parameter Golf

1. **The metric is the same** — both systems optimize val_bpb
2. **The time budget enforces fair comparison** — every experiment gets exactly 5 minutes
3. **The 16MB constraint is enforceable** — the eval harness checks artifact size
4. **Git history = experiment log** — every change is traceable
5. **1-GPU results transfer to 8-GPU** — the architecture improvements carry over; only throughput scales linearly

---

## 6. Key Adaptations from Vanilla Autoresearch

### 6.1 Artifact Size Constraint
Vanilla autoresearch has no model size constraint. We add:
- Post-training quantization + compression pipeline in `prepare_pgolf.py`
- Artifact size check after every experiment
- Agent instructed to DISCARD any experiment exceeding 16MB

### 6.2 Quantization-Aware Metrics
The agent must track **post-quantization val_bpb**, not just raw training loss. We add:
- Automatic quantization → decompress → re-evaluate roundtrip
- Both pre-quant and post-quant val_bpb reported
- Quant gap tracked as a separate metric

### 6.3 1-GPU → 8-GPU Transfer
Experiments run on 1×H100 for speed. The `train_pgolf.py`:
- Uses `torchrun --nproc_per_node=1` during experimentation
- The final winning script gets adapted for `nproc_per_node=8`
- Architecture/hyperparameter discoveries are GPU-count-agnostic
- Only batch size and gradient accumulation change for multi-GPU

### 6.4 Customized program.md
Our `program_pgolf.md` is much more targeted than the generic one:
- Encodes all knowledge from our research.md (techniques, ablations, negative results)
- Prioritized list of what to try
- Known failures to avoid
- Specific instructions for the artifact size constraint

---

## 7. Files to Create

```
parameter-golf/
├── prepare_pgolf.py       # Fixed eval harness (wraps PGolf's eval + quant)
├── train_pgolf.py         # The one file the agent modifies (starts from SOTA)
├── program_pgolf.md       # Agent instructions (our strategy encoded)
├── results.tsv            # Experiment log (created by agent)
├── research.md            # Our original deep research (already exists)
├── plan.md                # THIS FILE
├── train_gpt.py           # Original baseline (untouched)
└── data/                  # FineWeb dataset + tokenizer
```

---

## 8. Timeline

| Time | Action |
|---|---|
| **T+0h** | Rent 1×H100, SSH in, setup environment |
| **T+0.5h** | Run baseline, verify everything works |
| **T+1h** | Create `prepare_pgolf.py`, `train_pgolf.py`, `program_pgolf.md` |
| **T+1.5h** | Launch AI agent, verify first experiment completes |
| **T+2h** | Walk away — agent runs autonomously |
| **T+10h** | Come back, review ~70 experiments in `results.tsv` |
| **T+10.5h** | Take best config, adapt for 8×H100 |
| **T+11h** | Spin up 8×H100, run 3-seed final evaluation |
| **T+12h** | Submit PR to parameter-golf repo 🏆 |

---

## 9. Risk Mitigation

| Risk | Mitigation |
|---|---|
| Agent gets stuck in a loop | `program_pgolf.md` instructs diverse exploration strategies |
| Agent breaks the code badly | Git branch + rollback on every failed experiment |
| OOM on 1×H100 | Start from SOTA which uses ~11GB; 1×H100 has 80GB headroom |
| 1-GPU results don't transfer to 8-GPU | Architecture improvements are GPU-agnostic; only batch/grad_accum changes |
| Agent exceeds $25 credit | 1×H100 × 8 hours = $27; we stay within DigitalOcean quota |
| Agent runs out of ideas | program.md has a large prioritized backlog of techniques from research.md |

---

> [!TIP]
> **The key insight**: Autoresearch turns our human bottleneck (we can try ~3-4 experiments manually per hour) into an autonomous pipeline (~8-9 experiments per hour, running 24/7). Over an 8-hour overnight session, we get **~70 experiments** — equivalent to several days of manual work.

Let's build this. 🚀
