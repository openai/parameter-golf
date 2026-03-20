# Parameter Golf Autoresearch

Autonomous research loop for the OpenAI Parameter Golf challenge.

## Challenge Constraints

- **Artifact limit**: code_bytes + compressed_model_bytes <= 16,000,000 bytes
- **Training time**: 5 minutes per experiment (we use 5 min for fast iteration; final submission uses 10 min on 8xH100)
- **Hardware**: 1xH100 SXM 80GB (iteration); 8xH100 SXM (final)
- **Dataset**: Fixed FineWeb with SentencePiece 1024-token vocab
- **Metric**: val_bpb (bits per byte), lower is better
- **Current SOTA on leaderboard**: 1.1748 val_bpb

## What the leaderboard winners use

The top submissions all share these techniques:
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- Muon optimizer for matrix params, AdamW for scalars/embeddings
- FP16 tied embeddings (eliminates quantization gap)
- Sliding window eval with stride=64 (~0.03 bpb free improvement)
- Spectral/overtone embedding init (SVD power-law spectrum shaping)
- Residual mixing with sigmoid phase scheduling
- relu^2 MLP activation
- Logit softcap (30.0)
- U-net skip connections (encoder-decoder style)
- int8 per-row quantization + zlib compression
- ~11-12MB artifact size (4-5MB headroom under 16MB cap)

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - This file (`program.md`) -- experiment instructions
   - `prepare.py` -- fixed constants, data loading, evaluation, quantization. **Do not modify.**
   - `train.py` -- the file you modify. Architecture, optimizer, hyperparameters.
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains data shards. If not, run `python ../data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.
5. **Initialize results.tsv**: Create with header row only.
6. **Confirm and go**.

## Experimentation

Each experiment runs on a single H100 GPU. Training runs for a **fixed 5-minute wall clock budget**. Launch: `python train.py > run.log 2>&1`

**What you CAN modify:**
- `train.py` -- everything is fair game: architecture, optimizer, hyperparameters, quantization, depth, width, activation functions, attention patterns, initialization, etc.

**What you CANNOT modify:**
- `prepare.py` -- contains fixed evaluation, data loading, quantization, and size checking.
- Do not install new packages beyond what's in requirements.txt.

**The goal: get the lowest val_bpb while staying under the 16MB artifact limit.**

Key metrics printed at the end of each run:
```
val_bpb: X.XXXX
artifact_bytes: XXXXXXX (must be <= 16000000)
compressed_model_bytes: XXXXXXX
code_bytes: XXXXXXX
```

If artifact_bytes > 16,000,000, the run is INVALID regardless of val_bpb. You must reduce model size.

## Research Strategy

### Primary thesis: "The Depth Frontier"

Everyone uses int8 (8 bits/param) = ~19M params in 16MB. But int8 wastes bits -- trained weights have ~3-5 bits of actual information. The key insight: **use fewer bits per weight to fit MORE unique layers.**

| Bits/weight | Params in 16MB | Layers at dim=512 | Status |
|---|---|---|---|
| int8 | ~19M | 10 | Current SOTA |
| int6 | ~25M | 13 | Partially explored |
| **int4** | **~38M** | **~20** | **Unexplored -- highest priority** |
| int3 | ~50M | ~26 | High risk |

**The sweet spot is likely int4-int5 with 15-20 layers.** Nobody on the leaderboard has explored this.

### Experiment sequence (in priority order)

**Phase 1 -- Establish baseline and quick wins (experiments 1-5):**
1. Run baseline as-is to establish val_bpb reference
2. Try matrix_lr=0.03 (SOTA uses slightly lower than 0.04)
3. Try 11 layers (we have ~5MB headroom)
4. Try 12 layers
5. Try warmdown_iters=3000 (longer LR decay)

**Phase 2 -- Quantization-Aware Training (experiments 6-15):**
6. Add STE fake quantization during last 20% of training (warmdown period)
7. Try int6 QAT on all layers
8. Try int6 QAT on middle layers only, int8 on first/last 2
9. Try int5 QAT
10. Try int4 QAT
11. With best QAT: increase to 13 layers
12. With best QAT: increase to 15 layers
13. With best QAT: increase to 18 layers
14. With best QAT: increase to 20 layers
15. Find the optimal depth for the best QAT precision

**Phase 3 -- Architecture optimization at best depth (experiments 16-25):**
16. Tune MLP expansion ratio (try 3x instead of 2x, or 1.5x)
17. Try different GQA ratios (2 KV heads vs 4)
18. Try head_dim=48 or head_dim=80 instead of 64
19. Try SwiGLU MLP (may be worth it at higher depth)
20. Tune learning rates for the deeper model
21. Try longer sequences (2048) if step time allows
22. Try different warmdown schedules
23. Try different initialization strategies
24. Combine best findings
25. Final tuning pass

**Phase 4 -- Advanced compression (experiments 26+):**
26. NF4 quantization instead of uniform int4
27. Mixed precision per layer (sensitivity analysis)
28. Product quantization for embedding table
29. Learned codebook quantization

### Approaches that DON'T work (skip these)

- **Weight sharing / ALBERT-style**: Tested extensively. Fails because unique params matter more than recycled params at this scale (~19M params). Reusing weights does NOT create new information.
- **Hypernetworks**: Too complex to train in 10 min.
- **Byte-level tokenizer**: The evaluation uses the fixed SentencePiece tokenizer. Can't change it.
- **Mamba/SSM**: Requires custom CUDA kernels not available in the environment.

## Output Format

The training script prints a summary:
```
---
val_bpb:              1.XXXX
pre_quant_val_bpb:    1.XXXX
artifact_bytes:       XXXXXXX
compressed_model_bytes: XXXXXXX
code_bytes:           XXXXXXX
model_params:         XXXXXXX
num_steps:            XXXX
step_avg_ms:          XXX.XX
training_time_ms:     XXXXXX
peak_memory_mb:       XXXXX
```

Extract key metrics: `grep "^val_bpb:\|^artifact_bytes:\|^num_steps:" run.log`

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	val_bpb	artifact_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved -- use 0.000000 for crashes
3. artifact size in MB, round to .1f -- use 0.0 for crashes
4. status: `keep`, `discard`, `crash`, or `invalid` (over 16MB)
5. short text description of what was tried

Example:
```
commit	val_bpb	artifact_mb	status	description
a1b2c3d	1.4050	11.2	keep	baseline 10L 512dim int8
b2c3d4e	1.3900	11.2	keep	matrix_lr=0.03
c3d4e5f	1.3700	13.4	keep	12 layers
d4e5f6g	1.3500	11.8	keep	12L + int6 QAT middle layers
e5f6g7h	1.3200	12.5	keep	15L + int5 QAT
f6g7h8i	0.0000	0.0	crash	int4 QAT numerical instability
g7h8i9j	1.3100	17.2	invalid	20L int8 exceeds 16MB limit
```

## The Experiment Loop

LOOP FOREVER:

1. Look at git state and results.tsv to understand current best and what's been tried.
2. Decide what to try next. Follow the experiment sequence above, but adapt based on results.
3. Modify `train.py` with the experimental change.
4. `git add train.py && git commit -m "<short description>"`
5. Run: `python train.py > run.log 2>&1`
6. Extract results: `grep "^val_bpb:\|^artifact_bytes:\|^num_steps:\|^peak_memory_mb:" run.log`
7. If grep is empty, run crashed. `tail -n 50 run.log` to diagnose. Fix if simple, skip if fundamental.
8. Log to results.tsv.
9. If val_bpb improved AND artifact_bytes <= 16000000: keep (advance branch).
10. If val_bpb worse or artifact invalid: `git reset --hard HEAD~1` (discard).

**Timeout**: Each run should take ~5-7 minutes total (5 min training + eval overhead). If >10 minutes, kill and treat as failure.

**NEVER STOP**: Once started, do NOT pause to ask the human. Run indefinitely until manually stopped. If stuck, think harder -- re-read the code, try combinations, try radical changes. The human might be asleep.

**Key principle**: Each experiment should test ONE hypothesis. Don't change 5 things at once -- you won't know what helped.

**Before each experiment**: Before running, kill any stale GPU processes: `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true`

**If OOM**: Reduce TRAIN_BATCH_TOKENS or grad_accum_steps. Don't give up on the architecture -- just find a batch size that fits.
