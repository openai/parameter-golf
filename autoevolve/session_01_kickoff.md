# Session 01 — Project Kickoff

---

## User

There is the autoresearch repo from Andrej Karpathy, in the autoresearch folder. You can scan the repo to understand what the code is doing and how the auto research and auto evolve is achieved.

Now read the readme file about this openai challenge carefully to understand the requirements. I think the readme should provide everything you need, there is also a FAQ section for your info.

Now we will essentially leverage the idea from the autoresearch to ace this openai challenge. We will build up auto self-evolve scripts to build an agent and let it enhance its solutions or ideas, learn from the training results and mistakes, and push as hard as it can towards the most powerful solution and training results to ace the competition. I have added my openai api key in the .env file.

You can ask me any clarifying question if you have any. Let us now begin.

---

## Assistant

After reading the README and thoroughly exploring the autoresearch repo, here is a full picture of the challenge and the approach, followed by what was built.

---

### Challenge: OpenAI Parameter Golf

Train the best language model under these hard constraints:

- **Artifact ≤ 16,000,000 bytes** (code + zlib-compressed INT8 model weights)
- **Training time ≤ 10 minutes** on 8×H100 SXM GPUs
- **Metric: val_bpb** (bits per byte) on FineWeb validation set — lower is better
- Tokenizer-agnostic evaluation (BPB, not cross-entropy loss)
- Fully self-contained — no network calls during evaluation

**Current leaderboard SOTA: 1.1748 BPB** (notapplica, 2026-03-19)

Key techniques in the SOTA submission:
- 10 transformer layers with U-Net skip connections
- 512 model dim, 8 heads, 4 KV heads (GQA), relu² MLP
- 1024 vocab (SentencePiece BPE), 1024 training sequence length
- Tied embeddings kept in FP16 (not INT8) to halve quantization degradation
- Overtone spectral embedding initialization (power-law SVD reshape)
- Phase-transition residual mixing (sigmoid schedule across layers)
- Muon optimizer for matrix params + AdamW for embeddings/scalars
- Decoupled weight decay (0.02) on Muon matrix params
- Sliding window evaluation (stride=64, seq_len=1024)
- INT8 per-row quantization + zlib compression

---

### Autoresearch Approach (Karpathy)

The key insight: give an AI agent a single modifiable file and let it run an autonomous experiment loop overnight.

```
LOOP FOREVER:
  1. Read current train.py + results history
  2. LLM proposes a modification
  3. Apply change, git commit
  4. Run experiment
  5. Parse val_bpb
  6. If improved → keep (advance branch)
     If worse   → discard (git reset)
  7. Log to results.tsv
```

Core design principles:
- **One file to modify** — isolated, reviewable
- **Fixed time budget** — all experiments comparable
- **Binary keep/discard** — enforced via git
- **BPB metric** — vocab-size-independent
- **Agent loops indefinitely** — assumes human may be asleep

---

### What Was Built

**`autoevolve/evolve.py`** — Main orchestration script

- Calls OpenAI API (default: o3) to propose modifications
- Uses **search/replace diff blocks** (not full file replacement) — more reliable, fewer tokens
- LLM performs structured reasoning: Diagnose → Hypothesize → Estimate → Implement
- Validates syntax and structural integrity before running
- Runs experiments via `torchrun`, parses val_bpb from output
- Keeps improvements, reverts regressions (git-based)
- Tracks all experiments in `results.tsv` including full reasoning

**`autoevolve/program.md`** — Agent instructions and ideas bank

- Current SOTA analysis (all techniques already in the script)
- 20 ranked ideas to explore: SwiGLU, depth recurrence, QAT, MoE, TTT, etc.
- Common pitfalls and reasoning framework

**`autoevolve/train_gpt.py`** — Working copy (starts from SOTA 1.1748)

**`autoevolve/best_train_gpt.py`** — Best version found so far

**`autoevolve/results.tsv`** — Experiment log (iteration, val_bpb, status, reasoning)

---

### How to Run

```bash
# Dry run (propose + validate, no GPU needed):
conda run -n parameter-golf python autoevolve/evolve.py --dry-run --model o3

# On 1×H100 (cheaper iteration):
conda run -n parameter-golf python autoevolve/evolve.py --nproc 1 --model o3

# On 8×H100 (competition setup):
conda run -n parameter-golf python autoevolve/evolve.py --nproc 8 --model o3
```

---

### First Dry-Run Result

o3 proposed: **Replace ReLU² MLP with parameter-balanced SwiGLU**

> **Diagnosis:** The relu² MLP is an outdated activation that wastes parameters and limits representation power. Modern small-scale LMs (LLaMA, Gemma, Mistral) all replaced ReLU with SwiGLU.
>
> **Hypothesis:** Switch to SwiGLU while keeping total parameter count constant (hidden = 2/3 · mlp_mult · d). This increases multiplicative expressivity and typically yields ~0.5–1.0 ppl improvement.
>
> **Expected delta:** −0.004 BPB
>
> **Risk:** Minimal. Negligible runtime cost (<1 ms/step). No CUDA changes needed.

Syntax validated. Ready to run on GPU.
