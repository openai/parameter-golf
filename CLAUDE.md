# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a private fork of OpenAI's **Parameter Golf** challenge: train the best language model fitting in a **16MB artifact** (compressed model + code) in **under 10 minutes on 8xH100s**, evaluated by **bits-per-byte (BPB)** on the FineWeb validation set (lower is better).

Our differentiating strategy is **PolyGLU** (Polychromatic Gated Linear Unit) — a per-neuron activation routing mechanism integrated into the `PolyMLP` class in `train_gpt.py`. Each neuron dynamically selects among K=4 candidate activations (relu², tanh, SiLU, GELU) via Gumbel-Softmax routing. See `private_fork_references/polyglu_reference/` for the paper and design rationale.

**Origin remote**: `https://github.com/danielxmed/parameter-golf.git` (private fork — never push to `openai/parameter-golf`)

## RunPod Deployment (8xH100)

This is the primary workflow. Clone the **private fork**, set up, and run.

### Step 0: Clone and Setup
```bash
cd /workspace
git clone https://github.com/danielxmed/parameter-golf.git
cd parameter-golf

# Dependencies are pre-installed in the RunPod template image.
# If not, or if running outside the template:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Download Data
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### Step 2: Quick Smoke Test (< 2 min, single GPU)
```bash
ITERATIONS=10 VAL_LOSS_EVERY=0 RUN_ID=smoke \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```
Confirms torch.compile(fullgraph=True) works with PolyMLP. Must complete without errors.

### Step 3: Full PolyGLU Training Run (10 min, 8xH100)
```bash
NUM_LAYERS=11 MLP_MULT=3 RUN_ID=polyglu_seed1337 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Step 4: Verify Artifact Size
```bash
# Artifact limit is 16,000,000 bytes (decimal 16MB, NOT 16 MiB)
CODE_BYTES=$(wc -c < train_gpt.py)
MODEL_BYTES=$(wc -c < final_model.int8.ptz)
TOTAL=$((CODE_BYTES + MODEL_BYTES))
echo "code=${CODE_BYTES} model=${MODEL_BYTES} total=${TOTAL} limit=16000000"
[ "$TOTAL" -lt 16000000 ] && echo "OK: under limit" || echo "FAIL: over limit"
```

### Step 5: Multi-Seed Validation (for submission)
Submissions require p<0.01 statistical significance. Run 3 seeds:
```bash
NUM_LAYERS=11 MLP_MULT=3 RUN_ID=polyglu_seed1337 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

NUM_LAYERS=11 MLP_MULT=3 RUN_ID=polyglu_seed42 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

NUM_LAYERS=11 MLP_MULT=3 RUN_ID=polyglu_seed2025 SEED=2025 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Step 6: Baseline Comparison (optional, for ablation)
```bash
POLYGLU_ENABLED=0 NUM_LAYERS=11 MLP_MULT=3 RUN_ID=baseline_seed1337 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Step 7: Check Results
```bash
grep "val_bpb" logs/polyglu_seed1337.txt
grep "val_bpb" logs/polyglu_seed42.txt
grep "val_bpb" logs/polyglu_seed2025.txt
grep "tau:" logs/polyglu_seed1337.txt | tail -3   # confirm tau annealed to ~0.1
```

## Submission Preparation

If results are competitive (beat SOTA by >=0.005 nats), prepare a submission PR **to the upstream openai/parameter-golf repo** (NOT our private fork).

### Required Files

Create `records/track_10min_16mb/YYYY-MM-DD_PolyGLU/` with:

1. **`train_gpt.py`** — copy of the training script (must compile and run standalone within the records folder)
2. **`README.md`** — explain the submission:
   - Results table (3 seeds: seed, step_avg, steps, val_bpb, artifact size)
   - Key Innovation section explaining PolyGLU
   - Run command
   - Ablation vs baseline (PolyGLU enabled vs disabled)
   - Credits (cite arXiv:2603.13347v1)
3. **`submission.json`** — metadata:
   ```json
   {
     "name": "PolyGLU: Per-Neuron Activation Routing",
     "val_bpb": <3-seed mean>,
     "bytes_total": <largest artifact across seeds>,
     "blurb": "<one-paragraph summary>",
     "author": "danielxmed",
     "github_id": "danielxmed",
     "date": "YYYY-MM-DD"
   }
   ```
4. **Training logs** — `train_seed1337.log`, `train_seed42.log`, `train_seed2025.log` (copied from `logs/`)

### Submission Criteria
- Beat SOTA by **>=0.005 nats** with **p<0.01** across 3+ seeds
- Artifact (model + code) **< 16,000,000 bytes**
- Training **< 10 minutes** on 8xH100 SXM
- Evaluation **< 10 minutes additional**

## Commands

### Key Env Vars
- `MAX_WALLCLOCK_SECONDS=600` — 10-minute cap (default, set to `0` to disable)
- `VAL_LOSS_EVERY=200` — periodic validation (0 = only at end)
- `ITERATIONS=20000` — max steps (usually wall-clock limited)
- `TRAIN_SEQ_LEN`, `TRAIN_BATCH_TOKENS`, `NUM_LAYERS`, `MODEL_DIM`, `MLP_MULT`, `VOCAB_SIZE`
- `POLYGLU_ENABLED=1` — enable PolyGLU (default on). Set to `0` for baseline relu² MLP
- `POLYGLU_GATE_DIM=16` — gate network bottleneck dimension
- `POLYGLU_TAU_MIN=0.1` — final Gumbel-Softmax temperature (anneals from 1.0 to this)

### Training (Mac/MLX local smoke test)
```bash
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```
Note: MLX script does NOT include PolyGLU. Use CUDA for PolyGLU training.

## Architecture

### train_gpt.py (single-file, ~1197 lines)

The entire model, optimizer, data loading, quantization, and training loop live in one file. Key sections:

1. **Hyperparameters** (line ~39): All config via env vars, class-level defaults (including PolyGLU config at ~89)
2. **Muon optimizer** (line ~96): Newton-Schulz orthogonalization for 2D matrix params
3. **BPB evaluation** (line ~180): Tokenizer-agnostic validation using SentencePiece byte LUTs
4. **Int8 quantization** (line ~293): Per-row int8 for 2D floats, fp32 passthrough for control/routing params
5. **Data loading** (line ~435): Sequential shard streaming with DistributedTokenLoader
6. **Model** (line ~505):
   - `CastedLinear` — fp32 weights, bf16 compute
   - `CausalSelfAttention` — GQA + RoPE + QK-norm + flash attention
   - `MLP` — original relu² (used when `POLYGLU_ENABLED=0`)
   - `PolyMLP` (line ~625) — **PolyGLU activation routing** (K=4: relu², tanh, SiLU, GELU)
   - `Block` — pre-norm residual with learned `attn_scale`, `mlp_scale`, `resid_mix`
   - `GPT` — U-Net skip connections, tied embeddings, logit softcap
7. **Optimizer setup** (line ~910): Split into Muon (2D matrix params) and Adam (embeddings, scalars, routing params)
8. **Training loop** (line ~1030): Warmdown LR, wall-clock cap, **tau annealing** (line ~1067), gradient accumulation
9. **Serialization** (line ~1130): int8 quantization + zlib compression → `final_model.int8.ptz`, then roundtrip eval

### PolyMLP: How Activation Routing Works

The `PolyMLP` class (line ~625) replaces the fixed relu² with per-neuron activation routing:

1. **Mean-pool** input over sequence dim → `[B, dim]`
2. **Gate network** (`dim → 16 → 4`) produces dynamic routing signal → `[B, 4]`
3. **Routing logits** = `routing_alpha[hidden, 4]` + `routing_beta[4]` × gate_out → `[B, hidden, 4]`
4. **Gumbel-Softmax** (train) or plain softmax (eval) → routing weights `g`
5. **Weighted activation**: `g[0]*relu²(h) + g[1]*tanh(h) + g[2]*silu(h) + g[3]*gelu(h)`

Temperature anneals linearly from 1.0→0.1 over training (wall-clock aware). Routing converges to near-deterministic selections purely from language modeling loss — no auxiliary losses.

### Optimizer Split

- **Muon**: All 2D parameters in transformer blocks that are NOT in `CONTROL_TENSOR_NAME_PATTERNS`
- **Adam (scalar)**: All 1D params, skip_weights, **routing params** (`routing_alpha`, `routing_beta`, `gate_w1`, `gate_w2`), and other control params
- **Adam (embedding)**: `tok_emb.weight` with its own LR

`CONTROL_TENSOR_NAME_PATTERNS` (line ~293) includes `routing_alpha,routing_beta,gate_w1,gate_w2` to ensure routing params go to Adam (not Muon), stay fp32, and get fp32 passthrough in quantization.

### Quantization Pipeline

- 2D float tensors > 65536 elements → per-row int8 with clipping at 99.99984th percentile
- Small float tensors ≤ 65536 elements → fp32 passthrough if name matches CONTROL patterns, else fp16 passthrough
- All PolyGLU routing params are < 65536 elements → **zero quantization loss**
- Final artifact: `torch.save` → `zlib.compress(level=9)` → must be < 16,000,000 bytes total with code

## PolyGLU Implementation Notes

### Known Issues in Reference Docs — ALL RESOLVED

1. **Generic name patterns** — RESOLVED: Params named `routing_alpha`, `routing_beta` (not generic `alpha`/`beta`). Added to `CONTROL_TENSOR_NAME_PATTERNS` with no substring collisions.

2. **Gumbel noise at eval time** — RESOLVED: `self.training` branch in `PolyMLP.forward()`. Training uses manual Gumbel-Softmax (float32 precision). Eval uses plain `F.softmax(logits / tau)` with no random noise. `torch.compile` creates separate compiled graphs for each mode.

3. **Inconsistent parameter counts** — RESOLVED: Actual overhead is ~14.4K params/layer × 11 layers = ~158K total (~635KB fp32). Negligible vs 16MB budget.

4. **torch.compile compatibility** — RESOLVED: All 4 activations computed explicitly (no lambdas). Additive accumulation pattern (no `torch.stack` of `[B,seq,hidden,4]`). Temperature stored as `register_buffer('_tau', ...)` not a Python float (which torch.compile would bake as a constant).

5. **Mean pooling over sequence dim** — Accepted: `x.mean(dim=1)` computes one routing decision per sequence. Same as the original paper. Fine for the challenge.

### Key Design Decisions (do NOT change without understanding)

- **`_tau` is a buffer, not a Python float**: `torch.compile(fullgraph=True)` bakes Python floats at trace time. The buffer is a graph input, updated via `fill_()` between forward calls.
- **Gumbel noise in float32**: Manual implementation avoids bf16 precision issues with `log(log(u))` where 1e-10 epsilon rounds to zero in bf16.
- **Additive accumulation**: `g[...,0]*a0 + g[...,1]*a1 + ...` instead of `(g * torch.stack([a0,a1,...], dim=-1)).sum(-1)`. Avoids allocating a 4x-sized intermediate tensor (~800MB/layer → ~200MB/layer).
- **No weight decay on routing params**: The paper found weight decay suppresses routing specialization. Current code uses no weight_decay anywhere — if future changes add it, routing params MUST be exempt.
- **No auxiliary losses**: Routing converges purely from the LM loss. Do NOT add sparsity/entropy penalties.

### Tuning Knobs for Iteration

If PolyGLU doesn't improve BPB on first run, try:
- `POLYGLU_TAU_MIN=0.05` — harder routing commitment
- `POLYGLU_GATE_DIM=8` or `32` — smaller/larger gate network
- Swap activation candidates in `PolyMLP.forward()` (e.g., replace `relu²` with `leaky_relu(0.5)²`)

## Current Leaderboard Context (as of March 23, 2026)

Best score: **1.1194 BPB** (LeakyReLU² + TTT + Parallel Muon). To submit a new SOTA, must beat by >=0.005 nats with p<0.01 statistical significance over multiple runs.

Key techniques in the winning stack: 11 layers, 3x MLP, LeakyReLU(0.5)², Partial RoPE, XSA, EMA, int6 QAT, GPTQ-lite, zstd, sliding window eval, BigramHash, SmearGate, Muon WD, TTT with LoRAs. PolyGLU is orthogonal to all of these.

## Constraints

- **16,000,000 byte artifact limit** (decimal 16MB) = compressed model bytes + code bytes (code = the `train_gpt.py` file)
- **10 minutes on 8xH100 SXM** for training
- **10 minutes additional** allowed for evaluation
- No network access during eval; artifact must be self-contained
- Cannot train on validation data (TTT only on already-evaluated tokens)
