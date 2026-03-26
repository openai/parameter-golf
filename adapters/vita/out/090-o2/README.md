# VITA submission package: 090-o2

This package mirrors parameter-golf record style for VITA objective-first optimization.

## Leaderboard-style summary
- Run: VITA-090 Objective-B package (090-o2)
- Winner config: C2
- Objective: B (accuracy floor 74.5 on cifar10)
- Score tuple: max_ratio=0.7, acc_at_max=87.27, mean_band=87.3225
- Default operating point: prune_ratio=0.5

## Evidence chain
- O1 ranking source: /Users/ever/Documents/GitHub/vita-autoresearch/pass2/targeted/optimization/o1/reports/o1_results.json
- O2 ranking source: /Users/ever/Documents/GitHub/vita-autoresearch/pass2/targeted/optimization/o2/reports/o2_results.json
- O2 reported winner: C2
- Winner train accuracy: 87.28
- Winner confirm@0.5: 87.37
- Confirm matches sweep: True

## Files in this folder
- submission.json
- leaderboard_row.json
- results.tsv
- README.md

## Repro note
- Repo id/name: 090 / SFW Once-for-All Pruning
- This is a packaging adapter; source training code and logs remain in vita-autoresearch artifacts.
