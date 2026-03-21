# Experiment 004: Baseline Architecture + Training Improvements Only

## Status: COMPLETE (wandb: parameter-golf / exp004_softcap15_eps10)

## Hypothesis
**Based on**: modded-nanogpt record #77, Exp 003 showing weight sharing doesn't help

Apply only the proven training improvements to the baseline architecture (no weight sharing):
1. Logit softcap 30 -> 15 (modded-nanogpt: saved ~100 steps equivalent)
2. Adam eps 1e-8 -> 1e-10 (modded-nanogpt: improved loss by 0.0014)

**Prediction**: val_bpb < 1.285 at 2000 steps (vs baseline 1.2963). WRONG — improvement vanishes.

## Configuration
- **Architecture**: SAME as baseline (9 blocks, dim=512, 8 heads, 4 KV heads)
- **Changes**: LOGIT_SOFTCAP=15, ADAM_EPS=1e-10
- **Script**: train_gpt.py (baseline, with wandb added)
- **wandb run**: `exp004_softcap15_eps10`
- **Parameters**: ~17M (same as baseline)

## Results

| Step | train_loss | val_loss | val_bpb | step_avg | vs baseline |
|------|-----------|----------|---------|----------|-------------|
| 0    | 6.94      | 6.94     | 4.1076  | -        | same        |
| 100  | 3.21      | -        | -       | 216ms    | -0.11 loss  |
| 500  | 2.49      | 2.49     | 1.4753  | 422ms    | **-0.005 BPB** |
| 1000 | 2.32      | 2.32     | 1.3759  | 480ms    | +0.000 BPB  |
| 1500 | 2.24      | 2.24     | 1.3265  | 480ms    | +0.000 BPB  |
| 2000 | 2.19      | 2.19     | 1.2969  | 480ms    | +0.001 BPB  |

**Final (post-quant int8+zlib): val_bpb = 1.3001** vs baseline 1.2978

## Key Findings
1. **Softcap=15 + eps=1e-10 help early training** (0.005 BPB better at step 500)
2. **But the advantage VANISHES by step 2000** — final BPB is essentially same as baseline
3. At 2K steps, the model hasn't converged enough for these to matter at the end
4. At 20K steps (full training), the effect might be different — modded-nanogpt saw gains there
5. The softcap change dramatically improves early loss (5.42 vs 5.99 at step 10) but this washes out

## Lesson
These training tricks are NOT a shortcut at 2K steps. Need to test at full 20K scale or find
bigger architectural wins. The 0.005 BPB gap at step 500 suggests they DO help training dynamics
— they just need more steps to compound.
