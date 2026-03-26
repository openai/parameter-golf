# Parameter Golf Autoresearch Program

This is an autonomous AI research experiment optimizing for the OpenAI Parameter Golf challenge.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current main branch.
3. **Read the in-scope files**: The relevant files are:
   - `research.md` — comprehensive analysis of the challenge, all known techniques, leaderboard history, and strategy.
   - `prepare_pgolf.py` — fixed constants, data loading, evaluation, quantization. **Do not modify.**
   - `train_pgolf.py` — the file you modify. Model architecture, optimizer, hyperparameters, training loop.
4. **Verify data exists**: Check that `data/datasets/fineweb10B_sp1024/` contains `.bin` files and `data/tokenizers/fineweb_1024_bpe.model` exists. If not, tell the human to run `python prepare_pgolf.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Context: What Is Parameter Golf?

You are optimizing a GPT language model for the OpenAI Parameter Golf challenge. The scoring metric is **post-quantization val_bpb** (bits per byte) on the FineWeb validation set — **lower is better**.

### HARD CONSTRAINTS (violations = invalid experiment)
1. **Artifact size**: Compressed model + code ≤ **16,000,000 bytes**. Check `artifact_ok` in the output.
2. **Training time**: Must complete within the **5-minute time budget** (enforced by `prepare_pgolf.py`).

### Current Leaderboard SOTA
- **1.14276 val_bpb** (thwu1, March 20 2026)
- Techniques: 10L, MLP 3×, Int5 MLP/Int6 attn quantization, BigramHash(10240), SWA, SmearGate

## Experimentation

Each experiment runs on a **single GPU** (RTX 5060, 16GB VRAM). Training runs for a **fixed time budget of 5 minutes** (wall clock). You launch it as:

```
python train_pgolf.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train_pgolf.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization strategy, etc.

**What you CANNOT do:**
- Modify `prepare_pgolf.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness or BPB calculation.

**The goal: minimize post-quantization val_bpb** while keeping `artifact_ok: True`.

**VRAM constraint**: The RTX 5060 has only **16GB VRAM**. You MUST be conservative with batch sizes and model sizes. If you OOM, reduce `DEVICE_BATCH_SIZE` or enable activation checkpointing.

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 val_bpb improvement that adds 20 lines of ugly code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep.

## What To Try (Priority Order)

### Tier 1: High Impact, Low Risk
1. **MLP 3× expansion** (MLP_MULT=3): Largest single improvement historically (~−0.029 bpb). Will increase artifact size — check it fits.
2. **Weight Decay for Muon** (WEIGHT_DECAY=0.04): Helps both generalization and quantization robustness.
3. **Tune MATRIX_LR**: Try 0.015, 0.02, 0.025. Default may be suboptimal.
4. **Increase model depth**: Try 10 layers. Costs ~1.5MB of artifact but gives ~−0.005 bpb.
5. **Tune WARMDOWN_ITERS**: Try 2000, 3000, 5000. Critical for convergence in short runs.
6. **Sliding window eval**: Set EVAL_STRIDE=64 for free ~−0.032 bpb improvement (costs ~60s eval time).

### Tier 2: Medium Impact, Medium Risk
7. **BigramHash embedding**: Add a hash table for token-pair features (prev_token, curr_token). 4096-10240 buckets, dim=128.
8. **SmearGate**: Learned gate blending current + previous token embedding. ~512 extra params.
9. **Orthogonal initialization**: Initialize matrix weights using QR decomposition.
10. **Muon momentum 0.99**: Increase from 0.95 after warmup.
11. **Sequence length 2048**: Double the training context. Will halve steps but improve quality. Must adjust batch size.
12. **Gradient clipping 0.3**: Tighter clipping may help.

### Tier 3: Novel/Experimental
13. **Int6 QAT (Quantization-Aware Training)**: Fake-quantize weights during forward pass with STE.
14. **Multi-token prediction**: Add auxiliary head predicting token+2 for extra gradient signal.
15. **Mixture of Experts**: Replace MLP with MoE (2-4 experts, top-1 routing).
16. **LoRA-style Test-Time Training**: Adapt small parameters during eval for per-document improvement.

### KNOWN FAILURES — Do NOT Try
- **SwiGLU activation**: 45% slower per step, net negative on consumer GPUs.
- **Layer recurrence / weight sharing across layers**: Catastrophic (−0.051 bpb).
- **LZMA compression**: Worse than zlib for this weight distribution.
- **Very high LR (>0.04 for Muon)**: Destabilizes training.

## Output Format

Once the script finishes, it prints a summary:

```
---
val_bpb:          1.220000
val_bpb_prequant: 1.210000
quant_gap:        0.010000
artifact_bytes:   14500000
artifact_ok:      True
training_seconds: 300.1
eval_seconds:     15.2
total_seconds:    325.9
peak_vram_mb:     12500.0
num_steps:        3450
num_params_M:     17.1
depth:            9
```

Extract key metrics: `Select-String "^val_bpb:|^artifact_bytes:|^artifact_ok:|^peak_vram_mb:" run.log`

On Bash/Linux: `grep "^val_bpb:\|^artifact_bytes:\|^artifact_ok:\|^peak_vram_mb:" run.log`

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	val_bpb	quant_gap	artifact_mb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (post-quant) — use 0.000000 for crashes
3. quant_gap (post_quant - pre_quant bpb) — use 0.000000 for crashes
4. artifact size in MB, round to .1f — use 0.0 for crashes
5. peak memory in GB, round to .1f — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:
```
commit	val_bpb	quant_gap	artifact_mb	memory_gb	status	description
a1b2c3d	1.220000	0.010000	14.5	12.3	keep	baseline
b2c3d4e	1.195000	0.008000	14.8	12.5	keep	MLP 3x expansion
c3d4e5f	1.230000	0.015000	16.2	13.0	discard	artifact too large
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar22`).

LOOP FOREVER:

1. Look at the git state and results.tsv to understand where you are.
2. Choose an idea from the priority list, considering what has and hasn't been tried yet.
3. Edit `train_pgolf.py` with the experimental change.
4. `git add -A && git commit -m "experiment: <short description>"`
5. Run the experiment: `python train_pgolf.py > run.log 2>&1` (redirect everything)
6. Read the results: `Select-String "^val_bpb:|^artifact_bytes:|^artifact_ok:|^peak_vram_mb:" run.log`
7. If empty → crash. Read `Get-Content run.log -Tail 50` (PowerShell) for stack trace.
8. Record in results.tsv (do NOT commit results.tsv).
9. **Keep/Discard logic:**
   - If `artifact_ok: False` → ALWAYS discard (invalid experiment).
   - If val_bpb improved (lower) AND artifact_ok → KEEP (advance branch).
   - If val_bpb same or worse → DISCARD (`git reset --hard HEAD~1`).
10. Go to step 1.

**Timeout**: Each experiment should take ~5 minutes training + ~1-3 minutes eval. If > 15 minutes, kill and discard.

**Crashes**: If it's a typo, fix and retry. If the idea is broken, skip it, log "crash", move on.

**NEVER STOP**: You are autonomous. Do NOT pause to ask the human anything. If you run out of ideas, re-read `research.md` for new angles, try combining previous near-misses, try more radical changes. The loop runs until you are manually stopped.

**CRITICAL**: After EVERY experiment, verify `artifact_ok: True` in the output. An experiment with val_bpb=0.5 but artifact_ok=False is WORTHLESS.
