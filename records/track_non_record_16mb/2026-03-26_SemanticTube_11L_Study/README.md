# Approach Study: Semantic Tube Regularization on a Fixed 11L Baseline Across Seq1024-Seq2048

This folder documents a non-record study of semantic tube regularization on top of a fixed 11-layer baseline.

The study asks two narrow questions:
- On a fixed `seq1024` baseline, does semantic tube regularization improve `val_bpb` over a matched no-tube control?
- Does the same regularizer setting still help when the same baseline is extended to `seq1536` and `seq2048`?

## Setup

Fixed backbone across the study:
- 11 layers, `d_model=512`, `8` heads, `4` KV heads
- `MLP_MULT=3`
- EMA from init (`alpha=0.997`)
- XSA on the last 4 layers
- SmearGate enabled
- NTK-aware RoPE enabled
- Muon weight decay `0.04`
- `8xH100`, `600s` wallclock cap

Only one family parameter changes within each matched comparison:
- `LAMBDA_TUBE`

The semantic tube loss is a second-difference penalty on hidden-state trajectories along sequence positions:

```text
v_t = h_t - h_{t-1}
a_t = v_t - v_{t-1}
L_tube = lambda * mean(||a_t||^2)
```

The `seq_len` extension is treated as a separate axis, not as part of the definition of the technique. Each longer-context result is paired with its own matched no-tube control:
- `T0` vs `T4` at `seq1024`
- `S0` vs `S1` at `seq1536`
- `S2` vs `S3` at `seq2048`

## Summary

- Discovery sweep: semantic tube regularization improved a fixed `seq1024` baseline in a cheaper proxy regime and remained positive in matched `seq1536`/`seq2048` extensions.
- Best discovery result: `S3` at `seq2048`, `lambda=5e-4`.
- Main geometric effect: the regularizer sharply reduced hidden-state trajectory curvature without causing representation collapse.

## Discovery Sweep

### Core Family (`seq1024`)

| Run | `LAMBDA_TUBE` | `val_bpb` | Steps | `step_avg` | Notes |
|-----|--------------:|----------:|------:|-----------:|-------|
| `T0` | `0` | `1.2595` | `1692` | `354.61 ms` | no-tube control |
| `T1` | `5e-5` | `1.2559` | `1676` | `358.08 ms` | monotonic gain vs control |
| `T2` | `1e-4` | `1.2558` | `1680` | `357.25 ms` | similar to `T1` |
| `T3` | `2e-4` | `1.2553` | `1671` | `359.03 ms` | improved again |
| `T4` | `5e-4` | `1.2549` | `1668` | `359.70 ms` | best `seq1024` discovery run |
| `T5` | `1e-3` | `1.2555` | `1673` | `358.76 ms` | slightly worse than `T4` |
| `T6` | `2e-3` | `1.2560` | `1679` | `357.48 ms` | continued the regression past `T5` |

### `seq_len` Extension

| Run | `TRAIN_SEQ_LEN` | `LAMBDA_TUBE` | `val_bpb` | Steps | `step_avg` | Notes |
|-----|----------------:|--------------:|----------:|------:|-----------:|-------|
| `S0` | `1536` | `0` | `1.2316` | `2189` | `274.17 ms` | seq1536 no-tube control |
| `S1` | `1536` | `5e-4` | `1.2293` | `2166` | `277.05 ms` | improves over `S0` by `0.0023` bpb |
| `S2` | `2048` | `0` | `1.2207` | `2406` | `249.39 ms` | seq2048 no-tube control |
| `S3` | `2048` | `5e-4` | `1.2177` | `2380` | `252.15 ms` | improves over `S2` by `0.0030` bpb |

## Geometry Diagnostics

| Run | drift_cos | curvature | isotropy |
|-----|----------:|----------:|---------:|
| `T0` | `0.7361` | `20.576153` | `0.0135` |
| `T1` | `0.7684` | `3.677600` | `0.0137` |
| `T2` | `0.7762` | `2.337052` | `0.0130` |
| `T3` | `0.7770` | `1.518946` | `0.0143` |
| `T4` | `0.7730` | `0.790154` | `0.0141` |
| `T5` | `0.7843` | `0.429305` | `0.0133` |
| `T6` | `0.7999` | `0.232065` | `0.0148` |
| `S0` | `0.7362` | `37.779900` | `0.0107` |
| `S1` | `0.7947` | `0.737088` | `0.0148` |
| `S2` | `0.8514` | `33.248093` | `0.0076` |
| `S3` | `0.8800` | `0.618301` | `0.0094` |

## Files

Included here:
- `family_tube.jsonl`: raw discovery-sweep results
- `family_tube_review.md`: auto-generated discovery summary
- `logs/`: exact per-run logs for `T0-T6` and `S0-S3`
