# Parameter Golf: Autonomous Code Research

This is an autonomous LLM research program following the Karpathy autoresearch pattern.
You are an autonomous researcher. Your job is to modify code, train, measure, keep or discard.

## Goal

Minimize **val_bpb** (validation bits per byte) on the FineWeb validation set.

**Current best:** ~3.93 BPB (SOTA smoke test, 20 steps only — full 80-min run expected to reach ~1.12 BPB range)

**Community SOTA:** 1.1194 BPB (8×H100, 10 min) — our target is to match or beat this on single GB10 in 80 min.

## Constraints

1. **Artifact size** must be under **16,000,000 bytes** (code + int6+lzma compressed model). Exceeding this disqualifies the run.
2. **Time budget**: Each experiment runs for **80 minutes** (MAX_WALLCLOCK_SECONDS=4800) on a single NVIDIA GB10 Blackwell GPU (~7.4s/step → ~650 steps in 80 min).
3. **Single GPU** only. No distributed training (grad_accum_steps=8 compensates).

## Hardware

- NVIDIA DGX Spark: ARM64 (aarch64), GB10 Blackwell GPU, compute capability sm_121
- 128 GB unified memory (UMA — shared CPU+GPU)
- Running inside NGC PyTorch 25.12 container with native Blackwell support
- torch.compile works and is critical
- Flash Attention 3 NOT available — using F.scaled_dot_product_attention with GQA head expansion
- SDP backends: flash=False, mem_efficient=True, math=True

## Files

### What you modify

- **`train_gpt_spark_sota.py`** — the single file you edit. This is the community SOTA (PR #549 / 2026-03-23 LeakyReLU² + Legal TTT + Parallel Muon) adapted for DGX Spark. Contains:
  - 11-layer GPT with 512d, 8 query heads, 4 KV heads (GQA)
  - LeakyReLU(0.5)² activation in MLP (3x width)
  - BigramHash embeddings (2048)
  - XSA (cross-sequence attention) on last 4 layers
  - Partial RoPE (16/64 dims)
  - Per-layer LN scale (1/sqrt(layer+1))
  - VE128 on layers 9-10
  - EMA(0.997) + Tight SWA(every 50) weight averaging
  - GPTQ-lite int6 + lzma quantization
  - Parameter Banking + Parallel Muon optimizer (batched Newton-Schulz)
  - Legal TTT (test-time training) protocol
  - Sliding window evaluation with stride 64
  - Flash Attention 3 replaced with F.scaled_dot_product_attention + GQA head expansion

### What you do NOT modify

- `start_sota.sh` — the training launcher
- `run_tracker.py` — experiment tracking
- `data/` — training data
- `prepare.py`, `program.md` — these are read-only context

### How to run an experiment

```bash
RUN_ID=exp_name \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=4800 \
VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=10 \
bash start_sota.sh
```

### Key env vars (all have sensible defaults in the script)

```
NUM_LAYERS=11        MODEL_DIM=512       NUM_HEADS=8         NUM_KV_HEADS=4
MLP_MULT=3.0         BIGRAM_VOCAB_SIZE=2048
XSA_LAST_N=4         ROPE_DIMS=16        LN_SCALE=1
VE_ENABLED=1         VE_DIM=128          VE_LAYERS=9,10
EMA_DECAY=0.997      SWA_EVERY=50
MATRIX_LR=0.025      SCALAR_LR=0.025     TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99   MUON_WD=0.04        ADAM_WD=0.04
WARMDOWN_ITERS=3500   WARMUP_STEPS=20
EVAL_STRIDE=64       LOGIT_SOFTCAP=30.0
LATE_QAT=1           LATE_QAT_THRESHOLD=0.15
TTT_ENABLED=0        TTT_LR=0.002        TTT_EPOCHS=3
TTT_CHUNK_TOKENS=32768  TTT_FREEZE_BLOCKS=0
TRAIN_SEQ_LEN=2048   EVAL_SEQ_LEN=2048   VOCAB_SIZE=1024
```

### How to read results

From the log file `logs/<RUN_ID>.txt`:
- Final score: `grep "final_int6_roundtrip_exact" logs/<RUN_ID>.txt`
- Sliding window score: `grep "final_int6_sliding_window_s64_exact" logs/<RUN_ID>.txt`
- Artifact size: `grep "Total submission size" logs/<RUN_ID>.txt`
- Training steps: `grep "step:" logs/<RUN_ID>.txt | tail -1`

## What to focus on

The SOTA techniques are **already implemented** in the code. Don't re-implement them. Instead focus on:

1. **Hyperparameter tuning** within the SOTA architecture — learning rates, warmdown schedule, momentum, weight decay, EMA decay, SWA frequency, etc.
2. **Novel improvements NOT yet on the leaderboard** — new activation functions, attention patterns, normalization schemes, etc.
3. **Quantization improvements** — int5? Better compression codecs? Mixed-precision quantization? The artifact budget is 16MB.
4. **TTT improvements** — different optimizers (Adam instead of SGD?), learning rate schedules, chunk sizes, more epochs. Note: TTT adds significant eval time so budget carefully.
5. **Spark-specific optimizations** — the GB10 has different characteristics than H100 (slower compute, larger memory). Exploit this.
6. **Creative ideas** — anything you can think of that might help. The community has explored many ideas (see records/ directory for inspiration).

## The experiment loop

**IMPORTANT: Always git commit before modifying code, so you can revert.**

LOOP FOREVER:

1. `git add train_gpt_spark_sota.py && git commit -m "checkpoint before experiment"`
2. Make ONE focused change to `train_gpt_spark_sota.py`
3. `git add train_gpt_spark_sota.py && git commit -m "experiment: <description>"`
4. Run the experiment:
   ```bash
   RUN_ID=exp_name ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=4800 \
   VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=10 \
   bash start_sota.sh 2>&1 | tee logs/latest_experiment.txt
   ```
5. Read the result lines from the output
6. If val_bpb **improved** AND artifact_bytes < 16000000:
   - **KEEP** — this is the new baseline. Update the "current best" in your notes.
7. If val_bpb did NOT improve or artifact is too large:
   - **DISCARD** — `git revert HEAD --no-edit`
8. Log the result (what you tried, the score, keep/discard)
9. Go to step 1

**If a run crashes**, read the error from the log. If it's a simple bug (typo, shape mismatch), fix and re-run. If the idea is fundamentally broken, discard and move on.

**NEVER STOP.** The human may be asleep. Keep running experiments until manually interrupted. If you run out of ideas, re-read this file, re-read train_gpt_spark_sota.py for new angles, try combining previous near-misses, or try more radical changes.

## What NOT to do

- Do NOT modify any file except `train_gpt_spark_sota.py`
- Do NOT disable `torch.compile` — it's critical for performance
- Do NOT change `VOCAB_SIZE` (1024) — fixed by the competition
- Do NOT install new packages — only use what's in the container
- Do NOT use more than 1 GPU
- Do NOT use Flash Attention 3 (not available on Blackwell sm_121 — the SDP replacement is already in the code)
- Do NOT change the evaluation or scoring logic — only change the model, optimizer, and training loop
- Do NOT re-implement techniques that are already in the code — tune them or add new ones
