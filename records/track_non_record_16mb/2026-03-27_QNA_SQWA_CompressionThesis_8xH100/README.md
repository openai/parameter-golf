# Non-Record Submission: Quantization Noise Annealing + Stochastic Quantized Weight Averaging

**Best val_bpb: 1.1216** (Run 1 baseline, int6 sliding window stride=64, post-quantization)

This submission's best score is the baseline run (1.1216). Runs 2 and 3 are ablations included to document the compression-thesis result.

This is a non-record submission documenting a controlled 3-run ablation on **8xH100 SXM** testing whether training for the quantized artifact directly improves post-quantization BPB. The core thesis: if the 16MB artifact is an int6-quantized, LZMA-compressed model, the training loop should optimize for that deployed form rather than the float checkpoint.

**Result: interesting negative.** Both techniques work mechanistically (quant gap shrinks dramatically) but neither improves the leaderboard metric. The bottleneck in current SOTA is float model quality, not quantization error.

## Thesis

Parameter Golf scores the int6-quantized artifact, not the float model. The gap between float and quantized BPB is wasted capacity. Two complementary techniques aim to close it:

1. **Quantization Noise Annealing (QNA)**: During training, inject uniform noise matching the int6 quantization error into `CastedLinear` forward passes. Scale noise proportionally to LR (high early, zero at convergence). Disabled automatically when late QAT activates. This teaches the model to be robust to quantization noise throughout training, not just during the final QAT phase.

2. **Stochastic Quantized Weight Averaging (SQWA)**: During warmdown, periodically snapshot the EMA weights, quantize to int6, dequantize back to float, and accumulate. At consolidation, average these quantize-dequantized snapshots. This ensures the final averaged model lives in the quantization-friendly subspace rather than averaging float weights that may individually quantize well but average poorly.

**Kill criteria:**
- Run 2 (QNA) worse than Run 1 by >0.003 BPB → try lower magnitude
- Run 3 (QNA+SQWA) worse than Run 2 by >0.002 BPB → drop SQWA
- Submission size >16MB → reduce model or compression

## What Changed

| Area | SOTA Baseline | QNA (Run 2) | QNA+SQWA (Run 3) | Why |
|------|---------------|-------------|-------------------|-----|
| Training noise | None | Uniform noise ~ int6 error, scaled by LR | Same as QNA | Teach robustness to quant error throughout training |
| Weight consolidation | EMA only | EMA only | SQWA: average quantize-dequantized EMA snapshots | Ensure averaged weights stay in quant-friendly subspace |
| Late QAT interaction | QAT at scale<0.15 | QNA disabled when QAT activates | Same | QNA and QAT serve complementary roles: QNA for bulk training, QAT for fine-tuning |

All changes toggleable via env vars (`QNA_ENABLED`, `SQWA_ENABLED`). Defaults are off — a bare run reproduces SOTA.

## Parent Baseline

- **Base script:** merged SOTA by abaybektursun (2026-03-23), 1.1194 BPB
- **Techniques already in base:** 11L, 512d, GQA(8Q/4KV), LeakyReLU(0.5)², XSA4 (last 4 layers), Partial RoPE (16/64 dims), LN Scale, BigramHash(2048, dim=128), SmearGate, Muon(WD=0.04), EMA(0.997), late QAT(scale<0.15), int6+LZMA, sliding window eval(stride=64)
- **Our reproduction:** 1.1216 BPB (Run 1) — within 0.002 of published result

## Run Commands

All runs on 8xH100 SXM, Secure Cloud, 600s wallclock, identical except for env overrides.

```bash
# Run 1: Baseline reproduction
RUN_ID=run1_base PYTHONUNBUFFERED=1 SEED=42 \
torchrun --nproc_per_node=8 train_gpt.py

# Run 2: QNA only
RUN_ID=run2_qna PYTHONUNBUFFERED=1 QNA_ENABLED=1 QNA_MAGNITUDE=1.0 SEED=42 \
torchrun --nproc_per_node=8 train_gpt.py

# Run 3: QNA + SQWA
RUN_ID=run3_qna_sqwa PYTHONUNBUFFERED=1 QNA_ENABLED=1 QNA_MAGNITUDE=1.0 SQWA_ENABLED=1 SEED=42 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Results

### Ablation Table

| Run | Config | Float val_bpb | Int6 Roundtrip val_bpb | Sliding Window val_bpb | Quant Gap | Artifact Size | Steps |
|-----|--------|---------------|------------------------|------------------------|-----------|---------------|-------|
| 1 | Baseline | 1.1369 | 1.1450 | **1.1216** | 0.0081 | 15.93 MB | 7173 |
| 2 | +QNA | 1.1382 | 1.1457 | 1.1222 | 0.0075 | 15.95 MB | 7167 |
| 3 | +QNA+SQWA | 1.1464 | 1.1492 | 1.1258 | **0.0028** | 16.15 MB | 6736 |

### Interpretation

**QNA (Run 2 vs Run 1):** Quant gap shrank from 0.0081 to 0.0075 (7% reduction). But the float model degraded slightly (1.1369 → 1.1382), and the net effect on the leaderboard metric was +0.0006 worse. QNA works but the gain is smaller than the cost.

**QNA+SQWA (Run 3 vs Run 1):** Quant gap collapsed from 0.0081 to 0.0028 (65% reduction). SQWA dramatically improved quantization alignment. But:
- Float model degraded more (1.1369 → 1.1464, +0.0095 worse)
- SQWA overhead cost ~440 training steps (6736 vs 7173) — ~6% fewer steps due to periodic quantize-dequantize snapshots
- Run 3 artifact size was 16.15 MB (over the 16MB cap), partly due to code growth from the SQWA distributed broadcast fix
- Net effect on leaderboard metric: +0.0042 worse

**The fundamental issue:** With a quant gap of only 0.008 BPB at baseline, there's very little headroom to exploit. The SOTA's existing late QAT already handles quantization well enough. The bottleneck is the float model itself, not the quantization step.

## What We Learned

- **Mechanistic lesson:** QNA and SQWA both work as designed. QNA makes weights robust to quantization noise. SQWA ensures averaged weights stay in the quantization-friendly subspace. The 65% quant gap reduction from SQWA is a real, large effect.

- **Strategic lesson:** The current SOTA's quant gap (~0.008 BPB) is already small enough that optimizing it further yields diminishing returns. Float model quality dominates. Future work should focus on architecture, training recipe, or evaluation-time techniques rather than compression alignment.

- **Systems lesson:** SQWA accumulates state only on rank 0 (master_process). At consolidation, this causes a NCCL deadlock if the broadcast path is only entered by rank 0. Fix: broadcast the `use_sqwa` decision to all ranks before entering the branch. All ranks must participate in the same collective operations.

- **Byte-budget lesson:** SQWA's periodic quantize-dequantize loop adds ~0.6ms per snapshot step. Over ~280 warmdown steps with 6 snapshots collected, the overhead is negligible per step. However, Run 3's artifact came in at 16.15 MB (over the 16MB cap), partly due to code growth from the SQWA broadcast fix. Code size matters at the margin.

- **Eval lesson:** Sliding window eval (stride=64) takes ~75-95s on 8xH100 and is not included in the 600s wallclock cap. Total wall time per run is ~12 minutes (10 min training + 2 min eval/serialization).

## Implementation Details

### QNA (Quantization Noise Annealing)

```python
# In CastedLinear.forward(), when QNA is enabled:
if self.training and qna_scale > 0:
    # Compute int6 quantization step size for this weight
    clip_range = 31  # int6 for attention layers
    max_val = weight.abs().amax()
    step_size = max_val / clip_range
    # Add uniform noise scaled by LR (via qna_scale tensor)
    noise = (torch.rand_like(weight) - 0.5) * step_size
    weight = weight + noise * qna_scale
```

QNA scale is tied to the LR schedule — high noise early (when LR is high and the model can adapt), zero noise at convergence. Automatically disabled when late QAT activates (QAT provides its own gradient-based quantization alignment).

### SQWA (Stochastic Quantized Weight Averaging)

```python
# During warmdown (scale < 0.2), every swa_every steps, on rank 0:
snap_sd = {k: v.clone().cpu() for k, v in ema_state.items()}
snap_unbanked = _unbank_state_dict(snap_sd)
q_result, q_meta = mixed_quantize_int6(snap_unbanked)
deq_unbanked = dequantize_mixed_int6(q_result, q_meta)
deq_rebanked = _rebank_state_dict(deq_unbanked)
# Accumulate
sqwa_state[k] += deq_rebanked[k].float()
sqwa_count += 1

# At consolidation: average and broadcast to all ranks
avg_state = {k: sqwa_state[k] / sqwa_count for k in sqwa_state}
for name in current_state:
    dist.broadcast(avg_state[name].to(device), src=0)
```

SQWA collected 6 snapshots over the last ~280 steps of warmdown. Each snapshot goes through the full int6 quantize-dequantize roundtrip before accumulation, ensuring the averaged model is centered in the quantization grid.

## Configuration (All Runs)

```
VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_MULT=3.0 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048
MAX_WALLCLOCK_SECONDS=600 WARMUP_STEPS=20
MATRIX_LR=0.025 SCALAR_LR=0.025 EMBED_LR=0.035
ROPE_DIMS=16 XSA_LAST_N=4 LN_SCALE=1
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128
EMA_DECAY=0.997 LATE_QAT_THRESHOLD=0.15
SWA_START_FRAC=0.2 SWA_EVERY=50
```

- Model params: 26,993,756
- Peak memory: 21,474 MiB per GPU
- Step time: ~83.7ms (baseline), ~84.3ms (QNA+SQWA)

## Included Files

- `train_gpt.py` — full training script with QNA and SQWA (env-var toggleable, defaults off)
- `run1_base.log` — baseline reproduction (SEED=42, val_bpb=1.1216)
- `run2_qna.log` — QNA ablation (SEED=42, val_bpb=1.1222)
- `run3_qna_sqwa.log` — QNA+SQWA full thesis (SEED=42, val_bpb=1.1258)
- `submission.json` — leaderboard metadata
