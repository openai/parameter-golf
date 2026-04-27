# Crawler Track — Agent Protocol

## You are in: CRAWLER SOTA (Bandit_Wagon lineage)
Goal: best compression/quality ratio. Score: int6 sliding-window BPB + artifact size.
Lower BPB is better. Smaller artifact is better. Both matter.

## Current leader
```
cat crawler/LEADER.md
```
Source: `crawler/2026-03-29_BW5/train_gpt.py`
Best: 1.18672385 BPB, 8.61MB (seed 444)

## Leg structure
```
crawler/YYYY-MM-DD_name/
  hypothesis.md   ← what ONE thing changed, and why
  train_gpt.py    ← copy from leader, then modify
  gate.sh         ← 1-GPU 2000-step gate
  run.sh          ← 8×H100 full run (only after gate passes)
  gate_seed444.log / run results after runs complete
```

## Rules specific to this track
- Base for new legs: `crawler/2026-03-29_BW5/train_gpt.py`
- COMPILE_FULLGRAPH=1 is baseline — do not disable.
- CRAWLER_MLP_CHOKE_DIM=0 is baseline — no choke.
- CRAWLER_LOOP_ROPE_SCALES=9,1,1 is baseline.
- SKIP_GPTQ=1 is baseline for this track.
- Artifact size must stay under 16MB. Check before promoting.
- step_avg target: ~74.68ms on 8×H100. Higher = wrong stack or regression.

## Promotion gate
Beat `1.18672385` BPB (seed 444) AND artifact ≤ 16MB
→ confirm on seed 300 (seed 300 may be slightly worse — check mean)
→ update `crawler/LEADER.md`

## Note on seed 300
Current seed 300 (1.18758) is +0.00012 worse than Leg 3 seed 300.
Mean is still better. A new leg should improve both seeds to fully confirm.

## Never
- Run without committing and pushing the script first
- Mix crawler code into neural/ or vice versa
- Change more than one variable vs the parent leg
- Use seed 1337
