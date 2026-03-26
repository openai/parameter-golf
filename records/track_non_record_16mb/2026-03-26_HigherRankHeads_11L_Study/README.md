# Approach Study: Higher-Rank Output Heads on a Frontier 11L Baseline

This folder documents a non-record study of higher-rank output heads on top of a fixed frontier-aligned 11-layer baseline.

Research question:
- On a strong fixed 11L control, can higher-rank output heads outperform the standard tied softmax head under the 10-minute training budget?

## Summary

- Control: the standard tied head reached `1.1734 val_bpb` in `600s`.
- Result: every tested higher-rank head variant underperformed the control, often by a large margin.
- Artifact impact: mixture heads increased artifact size, while the simplex head reduced artifact size substantially but collapsed score.
- Main finding: on this frontier-aligned small-budget regime, the standard tied head remained the strongest option. Extra output-head structure behaved as an optimization burden rather than a compression win.

## Fixed Baseline

All runs used the same training and evaluation stack:
- 11 layers, `d_model=512`, `8` query heads, `4` KV heads
- `MLP_MULT=3`
- EMA from init (`alpha=0.997`)
- XSA on the last `4` layers
- SmearGate enabled
- BigramHash enabled (`2048` buckets, `128` dim)
- partial RoPE (`16` rotary dims) with NTK-aware scaling
- LN Scale enabled
- VE128 enabled on layers `9,10`
- Late QAT enabled at `lr_scale < 0.15`
- `seq2048`, `786432` train tokens/step
- sliding evaluation (`stride=64`)
- `8xH100`, `600s` wallclock cap
- Hopper FA3, compiled training, and the real quantization/artifact path

Only one family parameter changed across the study:
- output-head type and its local bottleneck settings

## Variants

Tested head family:
- `H0`: standard tied head
- `H1`: factorized head, rank `64`
- `H2`: factorized head, rank `128`
- `H3`: mixture-softmax, `K=2`, rank `64`
- `H4`: mixture-softmax, `K=4`, rank `64`
- `H5`: mixture-softmax, `K=4`, rank `128`
- `H6`: simplex head, bottleneck `128`

## Results

| Run | Head Variant | `val_bpb` | Δ vs `H0` | Steps | Artifact bytes | Notes |
|-----|--------------|----------:|----------:|------:|---------------:|-------|
| `H0` | standard tied head | `1.1734` | `0.0000` | `4415` | `16826913` | control |
| `H1` | factorized `r=64` | `2.4396` | `+1.2662` | `4451` | `16729834` | severe degradation |
| `H2` | factorized `r=128` | `1.9227` | `+0.7494` | `4425` | `16918260` | still far worse than control |
| `H3` | MoS `K=2`, `r=64` | `2.6167` | `+1.4434` | `4428` | `16565348` | severe degradation |
| `H4` | MoS `K=4`, `r=64` | `2.7112` | `+1.5379` | `4149` | `17172588` | worst mixture result |
| `H5` | MoS `K=4`, `r=128` | `2.0898` | `+0.9165` | `4160` | `17943057` | worse score and larger artifact |
| `H6` | simplex `128` | `4.1069` | `+2.9336` | `4241` | `10950817` | smallest artifact, unusable score |

The result is unambiguous: none of the tested higher-rank heads improved the frontier-aligned control, and several failed catastrophically.

## Interpretation

This study does not show that higher-rank output heads are useless in general. It shows something narrower and still useful:
- on this specific frontier-aligned 11L budgeted regime,
- with a strong tied-head baseline already in place,
- extra output-head structure was harder to optimize than the standard head,
- and the added expressivity did not translate into better compression.

The negative result is still useful for future work:
- if this family is revisited, it likely needs a different training regime rather than a direct swap on top of a tuned small-budget control
- the simplex head is notable as an artifact-size reduction idea, but not as a quality-preserving one in this form
- the mixture-head variants were the clearest failure mode: more parameters in the output head did not buy better compression here

## Why There Is No Separate Confirmatory Matrix

Unlike the semantic-tube study, this family sweep was already run on the intended fast path:
- compiled training
- Hopper FA3
- full `80` training shards
- sliding evaluation
- real quantization and artifact generation

So the family sweep itself already serves as the authoritative result set for this study.

## Included Files

Included here:
- `family_heads.jsonl`: raw study results
- `family_heads_review.md`: compact study summary
- `train_gpt.py`: self-contained study-local training script
- `install_flash_attn_hopper.sh`: Hopper-only FA3 installer used by the study runner
- `run_higher_rank_heads_study.sh`: self-contained family runner
- `REPRODUCE.md`: reproduction commands
