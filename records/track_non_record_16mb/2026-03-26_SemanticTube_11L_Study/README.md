# Approach Study: Semantic Tube Regularization on a Fixed 11L Baseline Across Seq1024-Seq2048

This folder documents a non-record study of semantic tube regularization on top of a fixed 11-layer baseline.

The study asks two narrow questions:
- On a fixed `seq1024` baseline, does semantic tube regularization improve `val_bpb` over a matched no-tube control?
- Does the same regularizer setting still help when the same baseline is extended to `seq1536` and `seq2048`?

## Summary

- Discovery sweep: semantic tube regularization improved a fixed `seq1024` baseline in a cheaper proxy regime and remained positive in matched `seq1536`/`seq2048` extensions.
- Fast-path confirm: on compiled Hopper FA3 runs with the real quantization/artifact path, the same `lambda=5e-4` setting did not improve final `val_bpb` at either `seq1024` or `seq2048`.
- Main retained effect: the regularizer sharply reduced hidden-state trajectory curvature without causing representation collapse.

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

## Discovery Sweep

The initial family sweep was run as a comparative discovery pass. It kept the backbone fixed and changed only `LAMBDA_TUBE`.

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

Within this discovery regime, the fixed-seq family has the expected regularizer shape: improvement from `T0` through `T4`, then regression at higher lambda.

### `seq_len` Extension

| Run | `TRAIN_SEQ_LEN` | `LAMBDA_TUBE` | `val_bpb` | Steps | `step_avg` | Notes |
|-----|----------------:|--------------:|----------:|------:|-----------:|-------|
| `S0` | `1536` | `0` | `1.2316` | `2189` | `274.17 ms` | seq1536 no-tube control |
| `S1` | `1536` | `5e-4` | `1.2293` | `2166` | `277.05 ms` | improves over `S0` by `0.0023` bpb |
| `S2` | `2048` | `0` | `1.2207` | `2406` | `249.39 ms` | seq2048 no-tube control |
| `S3` | `2048` | `5e-4` | `1.2177` | `2380` | `252.15 ms` | improves over `S2` by `0.0030` bpb |

In the discovery sweep, the same `lambda=5e-4` setting still helped at both `seq1536` and `seq2048`, and `S3` was the best result in the full private study.

These discovery runs are still useful as within-family evidence, but their absolute throughput and score are not the authoritative numbers for the study. The public confirmatory reruns below were executed on the intended fast path: full `fineweb10B_sp1024` data (`80` train shards), compiled training, Hopper FA3, and the real quantization/artifact path.

## Discovery Geometry Diagnostics

Each run also logs three diagnostics during evaluation:
- `drift_cos`: cosine similarity between successive hidden states
- `curvature`: mean squared second difference of the hidden trajectory
- `isotropy`: covariance-spectrum health score of sampled hidden states

### Core Family Diagnostics

| Run | drift_cos | curvature | isotropy |
|-----|----------:|----------:|---------:|
| `T0` | `0.7361` | `20.576153` | `0.0135` |
| `T1` | `0.7684` | `3.677600` | `0.0137` |
| `T2` | `0.7762` | `2.337052` | `0.0130` |
| `T3` | `0.7770` | `1.518946` | `0.0143` |
| `T4` | `0.7730` | `0.790154` | `0.0141` |
| `T5` | `0.7843` | `0.429305` | `0.0133` |
| `T6` | `0.7999` | `0.232065` | `0.0148` |

### `seq_len` Extension Diagnostics

| Run | drift_cos | curvature | isotropy |
|-----|----------:|----------:|---------:|
| `S0` | `0.7362` | `37.779900` | `0.0107` |
| `S1` | `0.7947` | `0.737088` | `0.0148` |
| `S2` | `0.8514` | `33.248093` | `0.0076` |
| `S3` | `0.8800` | `0.618301` | `0.0094` |

## Public Confirmatory Reruns

These reruns use the included study-local trainer, full `fineweb10B_sp1024` data, compiled mode, Hopper FA3, and the real quantization/artifact path.

| Run | `TRAIN_SEQ_LEN` | `LAMBDA_TUBE` | final `val_bpb` | Steps | `step_avg` | Size | Notes |
|-----|----------------:|--------------:|----------------:|------:|-----------:|-----:|-------|
| `T0` | `1024` | `0` | `1.1821` | `4832` | `124.35 ms` | `21.45 MB` | fixed-seq public control |
| `T4` | `1024` | `5e-4` | `1.1826` | `4914` | `122.10 ms` | `21.31 MB` | smoother geometry, slightly worse score |
| `S2` | `2048` | `0` | `1.1631` | `6676` | `90.21 ms` | `21.39 MB` | longer-context public control |
| `S3` | `2048` | `5e-4` | `1.1637` | `6676` | `90.14 ms` | `21.18 MB` | smoother geometry, slightly worse score |

Matched public deltas:
- `T4 - T0`: `+0.00050` bpb
- `S3 - S2`: `+0.00058` bpb

On the full fast path, semantic tube regularization did not improve final `val_bpb` at either `seq1024` or `seq2048`. The robust effect that remained was geometric: sharply lower trajectory curvature, stronger drift alignment, and no catastrophic representation collapse.

This makes the study most useful as a compute/regularization boundary finding:
- semantic tube regularization was helpful in the cheaper discovery regime
- on the stronger fully optimized path, the same `lambda=5e-4` setting became neutral to slightly harmful on final score

## Public Geometry Diagnostics

| Run | drift_cos | curvature | isotropy |
|-----|----------:|----------:|---------:|
| `T0` | `0.8783` | `109.151741` | `0.0061` |
| `T4` | `0.9099` | `0.672642` | `0.0085` |
| `S2` | `0.8497` | `193.878159` | `0.0042` |
| `S3` | `0.9227` | `0.749701` | `0.0097` |

The geometry effect is not subtle. On both matched pairs, semantic tube regularization dramatically smooths the hidden-state trajectory. What disappeared on the full fast path was the score gain, not the geometric effect.

## Artifact Size

The discovery sweep used `SKIP_QUANT=1`, so the values in `artifact_est_bytes` are only rough proxy estimates from the study harness.

The public confirmatory runs use the real quantization/artifact path, but they still exceed the nominal `16MB` track cap because this study uses the default quantization policy of the included study-local trainer rather than a separately artifact-fitted submission configuration. That artifact-fitting problem is orthogonal to the semantic-tube family claim and should be treated as a separate follow-up.

## Files

Included here:
- `family_tube.jsonl`: raw discovery-sweep results
- `family_tube_review.md`: auto-generated discovery summary
- `logs/`: exact per-run logs for `T0-T6` and `S0-S3`
- `public_logs/`: exact logs for the four fast-path confirmatory reruns
- `train_gpt.py`: self-contained study-local training script used for the public confirmatory reruns
- `install_flash_attn_hopper.sh`: Hopper-only FA3 installer used by the public confirmatory runner
- `run_semantic_tube_study.sh`: single-run confirmatory launcher
- `run_semantic_tube_public_matrix.sh`: matched-pair confirmatory launcher
- `REPRODUCE.md`: reproduction commands for the confirmatory runs
