# Experiment 030: SwiGLU + Lower LR + Default Softcap — NEW BEST! ⭐⭐⭐⭐

## Status: RUNNING (wandb: parameter-golf / exp030_swiglu_lowlr_sc30)

## Step 500: val_bpb = 1.4512 (baseline 1.4805 = 0.029 BETTER!)
## Step avg: 498ms (baseline 440ms = 1.13x = very close!)
## Config: SwiGLU(h=672) + MATRIX_LR=0.02 + SCALAR_LR=0.02 + TIED_EMBED_LR=0.03 + softcap=30 (DEFAULT)

## KEY FINDING: Softcap=30 is BETTER than 15 for SwiGLU!
## The softcap=15 trick that helped relu² does NOT help SwiGLU.
## SwiGLU's sigmoid gating already provides its own output regulation.

## Projected full-scale performance
## At 498ms/step on 1GPU → ~55ms on 8xH100 → ~10,900 steps in 10 min
## With 0.029 per-step advantage, could reach val_bpb ~1.19-1.20
## Beats current leaderboard leader (1.2230 from PR #39)!
