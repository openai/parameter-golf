# Auto Precision Budget 10L (1xH100 Exploratory)

This folder records a non-record exploratory submission built on top of the public `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` recipe.

The main idea is to replace the base recipe's hand-authored mixed-precision export exceptions with a calibration-driven precision allocator:
- start from the same default export policy
- run a short post-training calibration pass after SWA and pruning
- evaluate candidate tensor promotions under the real compressed artifact size
- greedily spend bytes on the promotions that improve calibration `val_bpb` the most per added byte

This is aligned with a compression-first research direction: if the challenge is byte-limited, the next step after architecture tuning is to spend the precision budget where quantization hurts the most rather than relying only on fixed heuristics.

## What Changed

The submitted `train_gpt.py` keeps the underlying 10-layer recipe intact and changes the export path:
- adds `AUTO_PRECISION_POLICY`
- adds short calibration on exported roundtrip weights
- evaluates candidate promotions such as `tok_emb.weight`, `bigram.*`, and late-layer attention projections
- uses a greedy byte allocator to choose promotions under the 16,000,000-byte cap

For distributed runs, calibration can be done rank-0-only with file-based policy sync. This keeps the idea compatible with future 8xH100 runs without requiring repeated distributed calibration collectives.

## Submitted Run

This is not intended to satisfy the main 10-minute leaderboard requirements. It is a cheap 1xH100 Modal smoke run submitted as an in-progress non-record experiment.

Run configuration:
- GPU: `1xH100`
- Train data: `--train-shards 1`
- `MAX_WALLCLOCK_SECONDS=60`
- `ITERATIONS=150`
- `AUTO_CALIBRATION_WINDOWS=16`
- `FINAL_EVAL_MAX_WINDOWS=16`

Because the final eval is intentionally capped to 16 sliding windows for cost, the reported metric is a smoke metric rather than a full-validation leaderboard metric. The point of this submission is to document the method and show that it runs successfully under the byte cap with a concrete sensitivity-driven policy search.

## Results

From `train.log`:
- training stopped at `87/150` steps due to the 60-second wallclock cap
- final exact metric: `val_loss:5.53668879`, `val_bpb:3.08435975`
- compressed model bytes: `15,771,560`
- total submission bytes: `15,836,818`
- selected promotions: `blocks.9.attn.c_k.weight`, `blocks.9.attn.c_v.weight`

During earlier paired smoke testing on the same idea family, the auto-precision allocator also showed a small improvement over a fixed export policy, which is why this branch is worth keeping alive for longer 8xH100 experiments.

## Why This Is Worth Exploring

Even if the gains remain small, this is a useful long-term direction:
- the current strongest recipes already depend on hand-tuned mixed precision
- a sensitivity-driven allocator can generalize across future architecture changes
- negative results still tell us which tensors are and are not the real quantization bottlenecks

Included files:
- `train_gpt.py`
- `train.log`
- `submission.json`
