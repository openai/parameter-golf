# Stage 3.5 Rebase Hypotheses

Date: 2026-04-09

`stage3_5` survives, but its branch tournament should now operate on an Era 6 shared trunk:

- `SP4096` or `SP8192`
- `depth recurrence`
- `full GPTQ + SDClip`
- `GPTQ embeddings`
- `MuonEq-R`

This stage is no longer about proving that late branching matters at all.
It is about using late branching to choose among **multiple strong deployment programs**.

## New Branch Dimensions

The old branch set was:

- EMA-heavy
- deploy-QAT
- family-split warmdown

The new branch set should be built from:

- pre-quant TTT variants
- dTTT variants
- ETLB / eval-time bias variants
- raw vs EMA vs post-TTT export states
- aggressive recurrent-block late quant shaping

## B501 Pre-Quant TTT Tournament

- Mechanism:
  - shared trunk, then branch into multiple pre-quant TTT recipes
  - vary freeze depth, epoch count, and optimizer recipe
- Why:
  - [pr_analysis.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/pr_analysis.md#L40) shows pre-quant TTT is now one of the largest clean gains
- Dominant lane:
  - deploy / TTT
- Expected impact:
  - large
  - `0.010 - 0.030 BPB`
- Failure mode:
  - branch budget is too small to differentiate TTT recipes meaningfully

## B502 dTTT Branch Duel

- Mechanism:
  - branch between plain pre-quant TTT and discriminative TTT with per-block LR groups
- Why:
  - the latest TTT story is no longer just “do TTT”; it is “which TTT law wins”
- Dominant lane:
  - deploy / TTT
- Expected impact:
  - medium to large
  - `0.005 - 0.015 BPB`
- Failure mode:
  - dTTT complexity is too high for the branch budget and the simpler recipe dominates

## B503 Export-State + ETLB Tournament

- Mechanism:
  - branch over:
    - raw export
    - EMA export
    - post-TTT export
    - ETLB-on vs ETLB-off
- Why:
  - export state and eval-time bias are now both clearly real, but small enough that a tournament can choose them cheaply
- Dominant lane:
  - export + eval
- Expected impact:
  - medium
  - `0.003 - 0.010 BPB`
- Failure mode:
  - ETLB remains too additive and the branch mostly reduces to plain export-state choice

## B504 Aggressive Recurrent Deploy Branch

- Mechanism:
  - one branch specializes recurrent/looped layers aggressively for late quant robustness
  - another stays conservative
- Why:
  - recurrence is only strong when quantization is handled correctly; branching is a natural way to test harder late shaping without committing globally
- Dominant lane:
  - architecture + export
- Expected impact:
  - medium
  - `0.004 - 0.012 BPB`
- Failure mode:
  - aggressive recurrent shaping destabilizes the shared trunk or yields no extra gain over strong GPTQ

## B505 Hybrid TTT + ETLB Branch

- Mechanism:
  - branch one path to plain pre-quant TTT
  - branch another to smaller TTT plus ETLB
- Why:
  - this tests whether some of the TTT gain can be traded for a cheaper eval-time head correction
- Dominant lane:
  - deploy / eval
- Expected impact:
  - medium
  - `0.003 - 0.012 BPB`
- Failure mode:
  - ETLB remains too small to substitute for stronger TTT

## B506 Failsafe Branch Policy On New Base

- Mechanism:
  - keep the event-trigger/failsafe logic, but the actual branch programs become TTT/export/eval recipes instead of old late training laws
- Why:
  - trigger logic is still useful, but only if it chooses among branch programs that matter now
- Dominant lane:
  - process split
- Expected impact:
  - medium
  - `0.003 - 0.009 BPB`
- Failure mode:
  - adaptive trigger adds no value over a fixed late tournament point

## New Role Of Stage 3.5

`stage3_5` should now answer:

- which late deployment program wins on a strong shared trunk?
- which export/eval state combination is actually best?
- when is it worth paying for stronger TTT?

It should not answer:

- whether SP1024 late branching beats the modern frontier
