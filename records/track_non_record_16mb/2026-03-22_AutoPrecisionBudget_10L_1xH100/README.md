# Auto Precision Budget 10L (1xH100 exploratory)

This folder is a small non-record experiment built on top of the `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` recipe.

The motivation is simple: the current best compressed models already rely on a hand-designed precision budget, with a few tensors kept at higher precision and the rest quantized more aggressively. That works, but it is still mostly heuristic. I wanted to try a more systematic version of the same idea by measuring which tensors are actually most sensitive to quantization and then spending bytes there.

The core change in `train_gpt.py` is a calibration-driven precision allocator. After training, SWA, and pruning, the script:
- starts from the default mixed-precision export policy,
- evaluates a small set of candidate tensor promotions,
- measures the calibration impact after roundtripping through the compressed export path, and
- greedily accepts promotions that give the best improvement per added byte while staying under the 16,000,000-byte limit.

The current candidate set includes:
- `tok_emb.weight`
- `bigram.proj.weight`
- `bigram.embed.weight`
- late-layer attention `c_k`, `c_v`, and `c_proj` weights

I also added a rank-0-only calibration path for distributed runs so the same method can be reused in future 8xH100 experiments without repeatedly doing distributed calibration collectives.

## Submitted run

This submission is intentionally modest. It is a cheap 1xH100 Modal smoke run, not a leaderboard attempt.

Configuration:
- GPU: `1xH100`
- Training data: `--train-shards 1`
- `MAX_WALLCLOCK_SECONDS=60`
- `ITERATIONS=150`
- `AUTO_CALIBRATION_WINDOWS=16`
- `FINAL_EVAL_MAX_WINDOWS=16`

Because the final evaluation is capped to 16 sliding windows, the reported number is a smoke metric rather than a full-validation metric. I am submitting it as a concrete, working non-record experiment that motivates further runs, not as a strong score claim.

## Result

From `train.log`:
- training stopped at `87/150` steps because of the 60-second wallclock cap
- final exact metric: `val_loss:5.53668879`, `val_bpb:3.08435975`
- compressed model bytes: `15,771,560`
- total submission bytes: `15,836,818`
- selected promotions: `blocks.9.attn.c_k.weight`, `blocks.9.attn.c_v.weight`

In earlier paired smoke testing on the same idea, the allocator also showed a small improvement over a fixed export policy, which is why I think this direction is still worth pursuing.

## Why I think this is worth exploring

Even if the gain here is small, I think the idea is useful for longer-term work:
- strong current recipes already depend on manual mixed-precision choices,
- a sensitivity-driven allocator should transfer better across future architecture changes,
- and even negative results help identify which tensors are real quantization bottlenecks versus harmless heuristics.

Included files:
- `train_gpt.py`
- `train.log`
- `submission.json`
