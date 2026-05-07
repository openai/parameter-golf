# Confidence-Weighted Legal TTT on PR #1493 stack
val_bpb: TBD | 8xH100 SXM

## Single targeted change vs PR #1493

Modify the legal score-first TTT update to weight per-token cross-entropy by the
**per-token NLL recorded during the SCORE phase**, normalized to chunk mean and
clipped to [1/clip, clip]. Tokens with high score-time NLL get more gradient;
already-easy tokens contribute less.

```
weight[t] = clip( nll_score[t] / mean(nll_score), 1/C, C )
loss = mean( ce_per_token * (1-alpha + alpha*weight) )
```

Defaults: alpha=1.0, clip=3.0. Toggle via TTT_CONF_WEIGHTED.

## Theoretical motivation

The legal TTT update direction at each chunk is essentially a stochastic gradient
of NLL on the chunk's tokens. Uniform weighting treats all chunk tokens as
equally informative for the val-distribution shift. But the val-distribution
shift is most strongly expressed in tokens the (already-trained) model fails on:
those tokens carry the most novel signal about the validation distribution.
Weighting by score-time NLL is a local importance-sampling correction that
focuses TTT on hard tokens, analogous to focal loss / hard-example mining.

Compliance: weights derive only from the score-pass NLL of the *same chunk*,
which is computed under `torch.no_grad()` strictly before any TTT update for
that chunk. This preserves the score-before-update legality of PR #549/#1413/#1493.

## Run

```
SEED=42 TTT_ENABLED=1 TTT_CONF_WEIGHTED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Base credit

Built on PR #1493 (clarkkev / dexhunter / abaybektursun / Robby955 / msisovic /
X-Abhishek-X) — SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25
+ Legal TTT.
