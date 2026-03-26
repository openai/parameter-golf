# Approach Study: Higher-Rank Output Heads on a Frontier 11L Baseline

This folder documents a non-record study of higher-rank output heads on top of a fixed frontier-style 11-layer baseline.

The study asks one narrow question:
- On a strong fixed 11L control, can higher-rank output heads outperform the standard tied softmax head under the 10-minute training budget?

## Summary

- Control: the standard tied output head reached `1.1734` `val_bpb` in `600s`.
- Result: every tested higher-rank head variant underperformed the control, often by a large margin.
- Throughput: more expressive heads generally trained slightly slower and, for mixture heads, also increased artifact size.
- Main finding: on this frontier-style small-budget regime, the standard tied head is difficult to beat; adding head expressivity here behaves more like an optimization burden than a win.

## Setup

Fixed backbone across the study:
- 11 layers, `d_model=512`, `8` heads, `4` KV heads
- `MLP_MULT=3`
- EMA from init (`alpha=0.997`)
- XSA on the last 4 layers
- SmearGate enabled
- BigramHash enabled (`2048` buckets, `128` dim)
- NTK-aware RoPE enabled with partial rotary dims (`16`)
- LN Scale enabled
- VE128 enabled on layers `9,10`
- Late QAT enabled at `lr_scale < 0.15`
- `seq2048`, `786432` train tokens/step
- sliding eval (`stride=64`)
- `8xH100`, `600s` wallclock cap
- Hopper FA3, compiled training, real quantization/artifact path

Only one family parameter changes within the study:
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

| Run | Head Variant | `val_bpb` | Δ vs `H0` | Steps | `step_avg` | Artifact | Notes |
|-----|--------------|----------:|----------:|------:|-----------:|---------:|-------|
| `H0` | standard tied head | `1.1734` | `0.0000` | `4415` | `135.91 ms` | `16.83 MB` | control |
| `H1` | factorized `r=64` | `2.4396` | `+1.2662` | `4451` | `135.85 ms` | `16.73 MB` | severe degradation |
| `H2` | factorized `r=128` | `1.9227` | `+0.7494` | `4425` | `135.59 ms` | `16.92 MB` | still far worse than control |
| `H3` | MoS `K=2`, `r=64` | `2.6167` | `+1.4434` | `4428` | `135.50 ms` | `16.57 MB` | severe degradation |
| `H4` | MoS `K=4`, `r=64` | `2.7112` | `+1.5379` | `4149` | `144.61 ms` | `17.17 MB` | worst mixture result |
| `H5` | MoS `K=4`, `r=128` | `2.0898` | `+0.9165` | `4160` | `144.23 ms` | `17.94 MB` | slower and larger |
| `H6` | simplex `128` | `4.1069` | `+2.9336` | `4241` | `141.47 ms` | `10.95 MB` | smallest artifact, unusable score |

The result is unambiguous: none of the tested higher-rank heads improved the frontier-aligned control, and several failed catastrophically.

## Interpretation

This study does not show that higher-rank output heads are useless in general. It shows something narrower and still useful:
- on this specific frontier-style 11L budgeted regime,
- with a strong tied-head baseline already in place,
- extra output-head structure was harder to optimize than the standard head,
- and the added expressivity did not translate into better compression.

The negative result is still informative for future work:
- if this family is revisited, it likely needs a different training regime rather than a direct swap on top of a tuned small-budget control
- the simplex head is particularly notable as an artifact-size reduction idea, but not as a quality-preserving one in this form

## Files

Included here:
- `family_heads.jsonl`: raw study results
- `family_heads_review.md`: auto-generated family summary
- `logs/`: exact per-run logs for `H0-H6`
- `train_gpt.py`: self-contained study-local training script
- `install_flash_attn_hopper.sh`: Hopper-only FA3 installer used by the study runner
- `run_higher_rank_heads_study.sh`: self-contained family runner
- `REPRODUCE.md`: reproduction commands
