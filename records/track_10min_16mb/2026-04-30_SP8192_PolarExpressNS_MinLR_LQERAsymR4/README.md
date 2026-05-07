# SP8192 PolarExpressNS MinLR LQERAsymR4 Baseline

Baseline run establishing the current score for `SP8192` with `PolarExpressNS`, `MIN_LR`, and `LQERAsymR4`. 

This baseline implements a 3-layer depth recurrence mechanism, symmetric row/col normalization in the Muon optimizer, a causal token-only n-gram tilt, parallel residuals in layers 7-10, and split learning rates for early/late parameter layers, alongside targeted fixes for proper MLP normalization independence and PyTorch `no_grad()` TTT evaluation. Soft-Round QAT was stripped to provide a clean baseline for these architectural techniques.

## Validation Performance
The submission yields a 3-seed mean `val_bpb` of **1.07302**.

| Seed | `val_bpb` | Artifact Size (bytes) |
|------|-----------|-----------------------|
| 1337 | 1.07294458| 15,955,411            |
| 42   | 1.07270429| 15,953,179            |
| 2025 | 1.07342419| 15,951,874            |
| **Mean** | **1.07302** | **15,953,488** |

## Compliance & Evaluation
- `train_under_600s`: True
- `artifact_under_16mb`: True
- `eval_under_600s`: True
- `no_pre_quant_ttt_on_val`: True
- `score_first_ttt`: True
- `three_seeds`: True

## Contents
- `train_gpt.py`: Main training script with all modifications.
- `submission.json`: Official track evaluation parameters and scores.
- `train_seed*.log`: Detailed logs for all three seed evaluations.
