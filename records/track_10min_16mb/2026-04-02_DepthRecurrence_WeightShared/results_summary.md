# Depth Recurrence — Results Summary

**Project:** Parameter Golf Challenge (OpenAI)
**PR:** openai/parameter-golf#1278
**Date:** April 3, 2026
**Author:** GitGeeks (milhouse)

---

## What We're Doing

Weight-shared depth recurrence: instead of N unique transformer blocks, share K blocks and iterate them multiple times. This gives more effective depth per parameter, directly optimizing L(N).

This is listed as an OpenAI "Request for PR" technique. No prior submission has made it work.

---

## Phase 1 Results: GPU Throughput (H100 SXM)

**Verdict: GREEN LIGHT — recurrence is essentially free on H100.**

| Config | Unique Layers | Repeats | Eff Depth | Params | step_avg | Overhead |
|--------|--------------|---------|-----------|--------|----------|----------|
| Baseline | 9 | 1 | 9 | 17.1M | 518ms | 1.00x |
| 3x3 | 3 | 3 | 9 | 6.0M | 508ms | **0.98x** |
| 3x5 | 3 | 5 | 15 | 6.1M | 550ms | **1.06x** |
| 4x5 | 4 | 5 | 20 | 7.9M | 693ms | **1.34x** |

Key finding: 3x3 is actually **faster** than baseline because fewer parameters = less memory bandwidth. Even 4x5 (20 effective layers) is only 1.34x overhead.

---

## MLX Validation Results (Local)

| Config | Params | Eff Depth | val_bpb | Compressed Size |
|--------|--------|-----------|---------|-----------------|
| Baseline (9 unique layers) | 17.1M | 9 | 3.2273 | 5.10 MB |
| Recurrence 3L x 3R | 6.0M | 9 | 3.2264 | 1.87 MB |
| **Recurrence 3L x 7R** | **6.1M** | **21** | **3.2134** | **1.89 MB** |

- 3x3 matches baseline at 2.8x fewer params
- 3x7 **beats baseline by 0.014 BPB** at same param count, 2.3x deeper
- Deeper recurrence converges faster at every step count

---

## Parameter Budget Analysis (16MB artifact limit)

| Config | Params | Estimated Artifact | Eff Depth | Notes |
|--------|--------|-------------------|-----------|-------|
| Current SOTA (11L, 512d) | 27M | ~15.9 MB | 11 | Near ceiling |
| 4 unique x 5 reps, 768d | 22.6M | ~13.3 MB | 20 | 1.5x wider, 1.8x deeper |
| 3 unique x 7 reps, 768d | 17.3M | ~10.2 MB | 21 | Most depth-efficient |

Both recurrence configs fit within 16MB while delivering ~2x the effective depth at greater width.

---

## Wallclock Projections (10 min on 8xH100)

| Config | step_avg (1xH100) | Est. steps in 600s | Effective depth |
|--------|-------------------|-------------------|-----------------|
| Baseline 9L | 518ms | ~1158 | 9 |
| Rec 3x3 | 508ms | ~1181 | 9 |
| Rec 3x5 | 550ms | ~1091 | 15 |
| Rec 4x5 | 693ms | ~866 | 20 |

Even at 1.34x overhead, the 4x5 config gets 866 steps with 20 effective layers. The deeper model converges faster per step, so fewer total steps are needed.

---

## Implementation

### Per-iteration conditioning
```python
# Before each block call at effective layer i:
gate = sigmoid(iter_gate[i])      # starts near 0 (init: -2.0)
x = x + gate * iter_embed[i]      # additive conditioning
x = blocks[i % num_unique](x, x0) # shared block with cycling
```

### U-Net skip connections
Adapted for effective depth: encoder = first half of effective layers, decoder = second half with reversed skips.

### Files
- `train_gpt_mlx_recurrence.py` — MLX prototype (validated)
- `train_gpt_recurrence.py` — CUDA port (Phase 1 tested on H100)

---

## What's Next

| Phase | Description | Cost | Status |
|-------|-------------|------|--------|
| 1 | GPU throughput gate | ~$1.50 | **DONE** |
| 2 | Architecture search (wider/deeper configs) | ~$20 | Next |
| 3 | SOTA stack integration (GPTQ+XSA+BigramHash) | ~$80 | Planned |
| 4 | Final submission (3-seed validation) | ~$60 | Planned |

**Budget:** ~$163 estimated of $500 grant

---

## Competition Context

- **Current SOTA:** 1.1147 BPB (PR #1019, abaybektursun)
- **Baseline:** 1.2244 BPB
- **Challenge:** Lowest val_bpb within 16MB artifact, 10 min on 8xH100
- **Deadline:** April 30, 2026
