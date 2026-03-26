# Autoresearch Experiment Notes — Parameter Golf

> **Branch**: `autoresearch/runpod`
> **GPU**: NVIDIA H100 80GB HBM3
> **Time budget**: 5 minutes per experiment (single GPU)
> **Started**: 2026-03-26

---

## Current Best

| Metric | Value |
|--------|-------|
| **val_bpb** | **1.527175** |
| **artifact_bytes** | 14,155,978 (14.2 MB) — **under 16 MB, valid submission** |
| **quant_gap** | 0.000949 |
| **config** | 6L, dim=512, MLP 3x, GQA 8/4, compile, WD=0.04, Muon mom=0.99 |
| **commit** | `aa4b189` |
| **steps** | 4087 in 300s (~73ms/step compiled) |

Note: This uses **standard eval** (not sliding window). Sliding window would improve bpb by ~0.03 but takes 10+ minutes on this validation set (62M tokens). For final submission, sliding window should be enabled.

---

## Key Findings

### What Worked (Keep These)

1. **torch.compile** — 2.4x speedup (220ms → 73ms/step). Single biggest infrastructure win.
2. **MLP 3x expansion** — Better than deeper networks. 6L MLP3x (1.527) beats 8L MLP2x (1.549).
3. **Weight decay 0.04** — Helps both generalization and quantization robustness.
4. **Muon momentum 0.99** (warmup from 0.92 over 500 steps) — Noticeable improvement.
5. **Batch size 131K tokens** — Sweet spot for this model. Gives ~4000 steps in 5 min.
6. **WARMDOWN_ITERS=700** — With ~4000 steps, warmdown starts at ~83% through training.

### What Failed (Don't Retry)

1. **MLP 3x + 8 layers** — Artifact = 18.9MB, exceeds 16MB with int8+zlib. Need int6/int5 quant to fit.
2. **7L MLP 2.5x** (1.568) — Worse than 6L MLP3x (1.527). Width > depth at this scale.
3. **SEQ_LEN=2048** (1.589) — Slower steps → fewer total steps → worse. Not worth it on single GPU.
4. **Batch 65K** (1.608) — Too noisy, more steps don't compensate.
5. **Batch 262K** (1.715) — Too few steps (2316), much worse.
6. **Sliding window eval with EVAL_BATCH_SEQS=4** — Takes 10+ minutes. Need EVAL_BATCH_SEQS=64+ or skip.
7. **WARMDOWN_ITERS=3000 with 131K batch** — Warmdown was active from step 1 (LR never reached full). Must match warmdown to expected step count.

### Known Issues

- **Original baseline (9L dim512 MLP2x) = 17.3MB** — already over 16MB with int8+zlib. Had to reduce to 8L.
- **Sliding window eval** is very slow on 62M token val set. Standard eval takes ~18s, sliding takes 10+ min.
- **results.tsv resets on git reset --hard** — need to keep it out of git or rebuild after resets.

---

## Architecture Budget Analysis

With int8 quantization + zlib compression, the approximate artifact budget:

| Config | Params | Artifact (approx) | Fits? |
|--------|--------|-------------------|-------|
| 6L dim512 MLP3x | 14.7M | 14.2 MB | Yes (1.8MB headroom) |
| 7L dim512 MLP2.5x | 15.2M | 15.0 MB | Yes (1.0MB headroom) |
| 8L dim512 MLP2x | 15.2M | 14.9 MB | Yes (1.1MB headroom) |
| 8L dim512 MLP3x | 19.4M | 18.9 MB | NO |
| 9L dim512 MLP2x | 17.0M | 17.4 MB | NO |

**To fit larger models**: Need int6/int5 quantization (not in prepare_pgolf.py eval harness).

---

## Next Experiments to Try

### High Priority
- [ ] WARMDOWN_ITERS=2000 (current experiment, pending)
- [ ] MATRIX_LR sweep: 0.015, 0.025, 0.03
- [ ] EMBED_LR sweep: 0.04, 0.08
- [ ] Warmup steps: try 20-50 steps
- [ ] Grad clip 0.3 (tighter)

### Medium Priority
- [ ] Model dim 576 with 5L MLP3x (different width/depth trade)
- [ ] Model dim 448 with 8L MLP3x (more depth, narrower)
- [ ] Orthogonal initialization for weights
- [ ] Different activation (GELU instead of ReLU²?)

### Low Priority / Needs Research
- [ ] BigramHash embedding (adds params, needs artifact check)
- [ ] SmearGate (tiny param cost, ~512 params)
- [ ] Int6 quantization in training (QAT) — would need modifying prepare_pgolf.py or doing it in train_pgolf.py
- [ ] SWA (Stochastic Weight Averaging)

---

## Leaderboard Context

| Entry | val_bpb | Notes |
|-------|---------|-------|
| SOTA (thwu1) | 1.1428 | 10L, Int5 MLP, BigramHash, SWA, 8xH100 10min |
| Our best | **1.527** | 6L MLP3x, 1xH100 5min, standard eval, int8+zlib |

Gap to SOTA: ~0.385 bpb. Major reasons:
1. We use 1 GPU / 5 min vs 8 GPU / 10 min (16x less compute)
2. We use int8+zlib vs int5/int6+zstd (worse compression = fewer params)
3. No BigramHash, SmearGate, SWA, orthogonal init, sliding window eval
4. No QAT (Quantization-Aware Training)

---

## How to Run

```bash
# Run a full experiment (5 min training + ~20s eval)
python train_pgolf.py > run.log 2>&1

# Extract key metrics
grep "^val_bpb:\|^artifact_bytes:\|^artifact_ok:\|^peak_vram_mb:\|^num_steps:" run.log

# Smoke test (30s)
python train_pgolf.py --smoke-test
```
