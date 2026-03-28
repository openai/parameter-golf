# Non-Record: MC Dropout Ensembling Is Negative for Small LMs

**Author:** abaybektursun | **Date:** 2026-03-28 | **Track:** Non-record study

MC Dropout — train with dropout, keep it on at eval, average K softmax distributions — does not improve BPB at the 17M-parameter scale. The deterministic single pass always wins.

## Method

1. Add `nn.Dropout(p)` after attention and MLP outputs in each transformer block (2 lines changed in `Block.forward`)
2. Train normally under the 600s wallclock cap
3. At eval: `model.train()` + `torch.no_grad()`, run K=16 forward passes, average softmax probabilities

Each softmax distribution sums to 1. Their average is a convex combination — also sums to 1. No normalization needed, no probability inflation.

## Results

| dropout | Baseline BPB (dropout OFF) | MC K=16 BPB | Delta |
|---------|---------------------------|-------------|-------|
| 0.30 | 1.3708 | 1.3756 | +0.0049 |
| 0.05 | 1.3250 | 1.3269 | +0.0019 |

- Higher dropout hurts training more than ensembling recovers (+0.005 BPB)
- Lower dropout trains better but ensemble gain is still negative (+0.002 BPB)
- The deterministic pass (dropout OFF) is strictly better in both cases

## Why It Fails

MC Dropout approximates a Bayesian ensemble by sampling sub-networks via dropout masks. For this to help, the sub-networks need to learn complementary features that average into a better predictor. At 17M parameters with 1620 training steps, the model has neither the capacity nor the training duration for dropout masks to induce meaningful diversity. The sub-networks are too similar — averaging them just adds noise.

## Hardware

- dropout=0.30: 1×H100 80GB SXM, 1657 steps in 600s
- dropout=0.05: 1×H200 NVL 141GB, 1620 steps in 600s

## Code

- Training: `experiments/mc_dropout/train.py` (baseline + 13 lines: `nn.Dropout` in Block, `dropout_rate` hyperparam)
- Eval: `experiments/mc_dropout/eval.py` (K-pass averaging with probability sum assertion)
