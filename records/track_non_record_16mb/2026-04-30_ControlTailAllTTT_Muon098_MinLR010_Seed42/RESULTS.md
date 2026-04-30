# Control+Tail TTT Experiment Log

**Author:** zhenyi-ji ([@Gotnhub](https://github.com/Gotnhub))

This file summarizes the experiment path behind the non-record submission in
this folder. It is intentionally compact: the goal is to make the mechanism,
evidence, and limitations easy for reviewers to audit.

## Hypothesis

Full-model score-first TTT is powerful but indiscriminate. Control-only TTT is
auditable and cheap, but may not have enough capacity to meaningfully adapt the
model. A better middle point is to adapt:

- global control/gating parameters everywhere
- all weights in the final few transformer blocks

This makes the selected weights act like an implicit adapter without adding
LoRA modules or changing the validation order.

## Completed Runs

| Run | Seed | BPB | Artifact | Verdict |
|---|---:|---:|---:|---|
| `repro-top1-seed42` | 42 | 1.07960392 | 15,991,264 | Local accepted-base anchor |
| `repro-top1-muon098-seed42` | 42 | 1.07898847 | 15,988,757 | Weak keep |
| `top1-control-ttt-muon098-minlr010-seed42` | 42 | 1.08009438 | 15,989,187 | Control-only TTT was too weak |
| `top1-control-tailall-muon098-minlr010-seed42` | 42 | **1.07734522** | 15,990,737 | Best completed run |
| `top1-control-tailall4-anchor002-muon098-minlr010-seed42` | 42 | 1.07944810 | 15,989,483 | More capacity plus anchor regressed |

The submitted run is:

```text
top1-control-tailall-muon098-minlr010-seed42
```

## Submitted Run Details

| Metric | Value |
|---|---:|
| Train steps | 4,787 |
| Train time | 588.060s |
| Pre-quant BPB | 1.08399328 |
| Quantized BPB | 1.09540727 |
| Sliding BPB | 1.07870003 |
| Quantized TTT BPB | 1.07734522 |
| TTT eval time | 776.046s |
| TTT parameter count | 8,681,560 |
| Artifact bytes | 15,990,737 |

The TTT gain over sliding is 0.00135481 BPB on seed 42.

## What Did Not Work

### Control-only TTT

The first original variant adapted only existing scalar/control parameters.
This was attractive for auditability, but the TTT subspace was too small:

```text
top1-control-ttt-muon098-minlr010-seed42: 1.08009438 BPB
```

### Last 4 Blocks Plus Anchor

Increasing the tail window to 4 blocks and adding anchor decay made the
mechanism more conservative but hurt the result:

```text
top1-control-tailall4-anchor002-muon098-minlr010-seed42: 1.07944810 BPB
```

The best observed point was therefore last 3 blocks with no anchor decay.

### Late PR1926 / PR1934 Routes

Fresh PR1926/PR1934-inspired directions were explored as mechanism references,
but they were not used for this submission. Those attempts either lacked a
final TTT metric or failed before training in the local Modal environment. They
are excluded from the submitted score.

## Legal Evaluation Notes

The submitted TTT path keeps the accepted base's score-first order:

1. score the validation chunk under `torch.no_grad()`
2. train on tokens that have already been scored
3. move to the next chunk

It does not use SLOT, pre-quant validation TTT, ETLB, n-gram caches, or logit
biasing. It does not alter the tokenizer or dataset.

## Limitations

- Only seed 42 was run.
- The final TTT eval time is 776.046s, above the 10-minute eval cutoff.
- The result is not competitive with the current official SOTA.
- This should be evaluated as a non-record mechanism contribution, not a
  leaderboard record.
