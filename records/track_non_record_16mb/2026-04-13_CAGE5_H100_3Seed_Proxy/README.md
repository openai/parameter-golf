# CAGE5 H100 3-seed proxy validation (non-record 16MB)

This folder captures a non-record H100 proxy validation for Parameter Golf.

It validates a strictly causal hashed 5-gram mixer stacked with legal score-first TTT on 1x H100. This is not a final 8xH100 leaderboard submission; it is a stronger proxy validation of the method before frontier-port work.

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

## H100 3-seed summary
legal_ttt_exact seeds:
- 2026: 2.47893474
- 314: 2.58368013
- 999: 2.50894781

Mean: 2.52385423
Std: 0.04404203

sliding exact seeds:
- 2026: 2.48587535
- 314: 2.59245887
- 999: 2.51649468

Mean: 2.53160963
Std: 0.04480594

Artifact bytes:
- 1339623
- 1313467
- 1334507

Max artifact size: 1339623 bytes

## Notes
- This is a stable 3-seed H100 proxy result for the current PR code path.
- An earlier exploratory H100 run reached a lower BPB, but its exact code provenance was not locked, so it is not used as the headline result here.
