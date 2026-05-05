# Experiment 003: Depth Recurrence 3x3 at dim=720, 2K Steps

## Status: RUNNING (wandb: parameter-golf / exp003_3x3_d720_2k)

## Hypothesis
**Based on**: Exp 002 failure (eval bug) + Looped Transformers (ICLR 2024), MoEUT (NeurIPS 2024),
Relaxed Recursive Transformers (Google DeepMind 2024)

A 3x3@720 config gives 9 effective layers (same depth as baseline) but wider (720 vs 512).
With weight sharing, this uses only ~11.7M params vs baseline 17M. The extra parameter budget
is spent on per-iteration scalars that break symmetry.

Key changes from Exp 002:
- Fixed eval bug: eval now uses same loop count as training (no graph mismatch)
- Reduced loops 5->3 to match baseline effective depth and improve throughput
- wandb logging enabled

**Prediction**: val_bpb < 1.28 at 2000 steps (vs baseline 1.2963).

## Configuration
- **Architecture**: 3 unique blocks x 3 loops = 9 effective layers, dim=720, 10 heads, 5 KV heads
- **Innovations**: QAT (STE), logit_softcap=15, adam_eps=1e-10
- **Training**: 2000 iters, 524K tokens/step, warmdown=1200
- **Parameters**: ~11.7M (vs baseline 17M)
- **wandb run**: `parameter-golf / exp003_3x3_d720_2k`

## Literature Grounding
- **Looped Transformers (ICLR 2024)**: "<10% params for comparable ICL" — we use 69% of baseline params
- **MoEUT (NeurIPS 2024)**: "Per-iteration layer norms essential for symmetry breaking"
- **Relaxed Recursive Transformers (DeepMind 2024)**: "Layer-wise LoRA for per-iteration specialization" — our per-iteration scalars serve same purpose but simpler
- **Key insight from DeepMind**: "Careful initialization and minimal uptraining = strong performance faster"

## Results (partial at step 500)

| Step | train_loss | val_loss | val_bpb | step_avg |
|------|-----------|----------|---------|----------|
| 100  | 3.15      | -        | -       | 221ms    |
| 200  | 2.76      | -        | -       | 502ms    |
| 500  | 2.55      | 2.55     | **1.5097** | 701ms |

**val_bpb = 1.5097 vs baseline 1.4805 — 0.03 BPB WORSE.**

### Analysis
The recurrent model with 3x3@720 (9 effective layers, wider) underperforms the baseline's
9 unique layers at dim=512. This suggests:
1. Unique layer weights > shared weights + wider dim for this parameter count
2. The per-iteration scalars are not enough to differentiate the shared layers
3. QAT + softcap + eps changes confound the comparison (need ablation)
4. DeepMind's Relaxed Recursive paper used per-iteration LoRA, which is much richer

### Next Steps
- **Exp 004**: Baseline + only training improvements (softcap 15, eps 1e-10, 60% warmdown) — isolate training tricks from architecture changes
- Consider adding per-iteration LoRA to shared blocks (from DeepMind paper)
