# MuonPower: GradPower Below 1 for Muon

**Track:** non-record 16MB research submission  
**Author:** Lucas Bryant / PapaFranku4647  
**Date:** 2026-04-16  
**Best H100 run:** `1.28338224` BPB with `MUON_GRAD_POWER=0.9`  
**Matched H100 control:** `1.28691190` BPB with vanilla Muon, `MUON_GRAD_POWER=1.0`  
**H100 delta:** `-0.00352966` BPB on seed `1337`  
**Local 4080 sweep:** `p=0.9` best 3-seed mean, `-0.00190881` BPB versus `p=1.0`

## Summary

This is a small optimizer ablation inspired by the GradPower paper. GradPower transforms a gradient with:

```python
g = sign(g) * abs(g) ** p
```

The paper's default `p=1.2` did not transfer to this Parameter Golf setup. In this Muon-heavy stack, powers below 1 worked better. The best tested value was `p=0.9`.

This is not submitted as a record. It is a non-record research result showing that GradPower can help Muon here, but only with the exponent moved below 1.

## What Changed

The only optimizer change is in Muon. Matrix gradients are powered before Muon momentum and Newton-Schulz orthogonalization:

```python
if grad_power != 1.0:
    g = g.sign() * g.abs().pow(grad_power)
```

The environment variable is:

```bash
MUON_GRAD_POWER=0.9
```

`MUON_GRAD_POWER=1.0` is vanilla Muon.

This is MuonPower, not full AdamPower. Adam is still used for embeddings and scalar/vector/control tensors. The main transformer matrices are optimized by Muon, so this tests the dominant optimizer path.

## H100 Matched Run

Both H100 runs use:

- 1x NVIDIA H100 80GB HBM3
- PyTorch `2.9.1+cu128`
- `fineweb10B_sp1024`
- `train_shards:80`
- `model_params:17059912`
- `train_batch_tokens:524288`
- `train_seq_len:1024`
- `grad_accum_steps:8`
- `matrix_lr:0.04`
- `scalar_lr:0.04`
- `embed_lr:0.05`
- `max_wallclock_seconds:1200`
- seed `1337`

| `MUON_GRAD_POWER` | steps | last val BPB | final int8+zlib BPB | delta vs `p=1.0` | log |
| ---: | ---: | ---: | ---: | ---: | --- |
| **0.9** | 2391 | 1.2818 | **1.28338224** | **-0.00352966** | `logs/h100_1200s/h100_1200s_p0.900_seed1337.txt` |
| 1.0 | 2355 | 1.2852 | 1.28691190 | 0.00000000 | `logs/h100_1200s/h100_1200s_p1.000_seed1337.txt` |

The H100 loss curves separate early and stay separated:

| step | `p=0.9` train loss | `p=1.0` train loss |
| ---: | ---: | ---: |
| 200 | 2.7756 | 2.8018 |
| 400 | 2.3893 | 2.4025 |
| 600 | 2.4928 | 2.5012 |
| 800 | 2.3402 | 2.3505 |
| 1000 | 2.3598 | 2.3653 |
| 1200 | 2.2881 | 2.2951 |
| 1400 | 2.3123 | 2.3177 |
| 1600 | 2.1970 | 2.2007 |
| 1800 | 2.2342 | 2.2393 |
| 2000 | 2.1817 | 2.1866 |
| 2200 | 2.1070 | 2.1102 |

The final H100 artifact sizes were both under the 16MB cap:

| `MUON_GRAD_POWER` | int8+zlib model bytes | total bytes with code |
| ---: | ---: | ---: |
| 0.9 | 15,412,529 | 15,460,600 |
| 1.0 | 15,356,997 | 15,405,068 |

## Local 4080 Sweep

The local sweep was run on 1xRTX 4080 under WSL with the same model family and `max_wallclock_seconds=1200`. Each value below is the final int8+zlib roundtrip BPB over seeds `1337`, `1338`, and `42`.

| `MUON_GRAD_POWER` | mean BPB | sample std | delta vs `p=1.0` |
| ---: | ---: | ---: | ---: |
| 0.850 | 1.36693001 | 0.00266328 | -0.00170810 |
| 0.875 | 1.36682637 | 0.00256659 | -0.00181174 |
| **0.900** | **1.36672930** | **0.00298812** | **-0.00190881** |
| 0.925 | 1.36695266 | 0.00176984 | -0.00168545 |
| 0.950 | 1.36710799 | 0.00162211 | -0.00153012 |
| 0.975 | 1.36778177 | 0.00161212 | -0.00085634 |
| 1.000 | 1.36863811 | 0.00188080 | 0.00000000 |

The matched-seed deltas for `p=0.9` versus vanilla Muon were:

| seed | `p=0.9` BPB | `p=1.0` BPB | improvement |
| ---: | ---: | ---: | ---: |
| 1337 | 1.36574961 | 1.36852546 | -0.00277585 |
| 1338 | 1.36435400 | 1.36681616 | -0.00246216 |
| 42 | 1.37008428 | 1.37057270 | -0.00048842 |

Every tested `p < 1` value beat `p=1.0` on the local 3-seed mean. The local result is a plateau, not a knife edge: `0.85` through `0.925` are all close, with `0.9` best among tested values.

## Negative Result: Do Not Blindly Use `p=1.2`

The first exploratory 4080 sweep tested powers above 1. Those runs were consistently worse:

| `MUON_GRAD_POWER` | seeds | mean final BPB | note |
| ---: | ---: | ---: | --- |
| 1.0 | 3 | 1.50614612 | vanilla Muon |
| 1.05 | 2 | 1.51246020 | worse |
| 1.1 | 3 | 1.51651419 | worse |
| 1.2 | 2 | 1.52743637 | much worse |
| 0.95 | 1 | 1.50200873 | better |
| 0.9 | 1 | 1.50017752 | best exploratory point |
| 0.8 | 1 | 1.50416208 | still below 1, but past the apparent sweet spot |

This is the main practical lesson. The GradPower idea appears useful, but the exponent needs to match the optimizer and noise regime. In this Muon setup, the paper's AdamPower default `p=1.2` is the wrong direction.

## What I Would Run With More Compute

The next funded experiments are straightforward:

1. **Multi-seed H100 confirmation:** run `p=0.9` versus `p=1.0` on seeds `1337`, `1338`, and `42`.
2. **H100 plateau check:** test `p=0.875`, `p=0.9`, and `p=0.925`.
3. **Matrix LR retune:** retune around `MATRIX_LR=0.04` with `p=0.9`, since the powered gradient changes Muon's input geometry.
4. **Transfer to the strongest stack:** apply the best `p < 1` setting to the current SP8192/MuonEq-R/TTT-style leaderboard stack and compare against the exact same stack at `p=1.0`.
5. **Adam-group ablation:** optionally power the Adam groups separately to see whether embeddings/scalars want the same exponent.

Those runs would turn this from a strong non-record ablation into either a leaderboard ingredient or a well-documented optimizer boundary.

## Reproduction

Single H100-style run:

```bash
SEED=1337 \
MUON_GRAD_POWER=0.9 \
MAX_WALLCLOCK_SECONDS=1200 \
python3 train_gpt.py
```

Vanilla control:

```bash
SEED=1337 \
MUON_GRAD_POWER=1.0 \
MAX_WALLCLOCK_SECONDS=1200 \
python3 train_gpt.py
```

Primary local sweep:

```bash
for p in 0.85 0.875 0.9 0.925 0.95 0.975 1.0; do
  for seed in 1337 1338 42; do
    SEED="$seed" MUON_GRAD_POWER="$p" MAX_WALLCLOCK_SECONDS=1200 python3 train_gpt.py
  done
done
```

## Included Files

- `train_gpt.py`: the MuonPower training script
- `submission.json`: non-record metadata
- `logs/h100_1200s/`: matched H100 `p=0.9` and `p=1.0` logs
- `logs/4080_1200s/`: primary local 3-seed sweep logs
- `logs/4080_600s_exploratory/`: exploratory sweep, including the negative `p>1` runs

