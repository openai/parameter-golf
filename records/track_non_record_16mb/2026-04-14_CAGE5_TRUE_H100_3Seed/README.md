# CAGE5 TRUE H100 3-seed proxy validation (non-record 16MB)

This folder captures the true locked H100 3-seed proxy validation for CAGE5.

It validates a strictly causal hashed 5-gram mixer stacked with legal score-first TTT on 1x H100. This is not a final 8xH100 leaderboard submission; it is the stable proof-of-method checkpoint for the CAGE5 stack before frontier-port work.

## Hardware
- 1x NVIDIA H100 80GB HBM3
- BF16 enabled

## Core idea
- legal score-first per-chunk TTT
- strictly causal hashed 5-gram interpolation inside the scoring path

## Included logs
- `train.log` = seed 2026
- `seed314.log`
- `seed999.log`

## TRUE H100 3-seed summary
legal_ttt_exact seeds:
- 2026: 2.56061156
- 314: 2.55201188
- 999: 2.54242330

Mean: 2.55168225
Std: 0.00742898

sliding exact seeds:
- 2026: 2.56936489
- 314: 2.56028065
- 999: 2.55063981

Mean: 2.56009512
Std: 0.00764561

Artifact bytes:
- 1320479
- 1325743
- 1320587

Max artifact size: 1325743 bytes

## Notes
- This is the true locked H100 3-seed result for the real CAGE5 script.
- Earlier H100 exploratory runs and intermediate proxy folders are not used as the headline result.
