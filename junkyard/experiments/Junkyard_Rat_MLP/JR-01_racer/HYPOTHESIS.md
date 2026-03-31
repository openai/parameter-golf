# Junkyard_Rat — Rat Rod Base + Triton + Loader Garage

Date: 2026-03-29

## Mission
Build the next clean base-model garage on top of the strongest honest in-repo anchor, then import the best current external neural ideas without drifting into unstable or rule-fragile territory.

This lane is not an n-gram lane.
This lane is the base-model lane that Bandit and X-WING can later consume.

## Why This Garage Exists

Three facts now matter:

1. **Best honest local base model is still Rat Rod Green v1**
   - `sliding_window val_bpb: 1.1129`
   - `post_ema val_bpb: 1.1364`
   - `6882` steps in `600s`
   - Source: `experiments/Rat_Rod/PROGRESS.md`

2. **Bandit proves compressed architecture headroom matters**
   - `0.4961 BPB` at about `9.2 MB`
   - honest crawler base is only `1.1867` sliding
   - meaning: the crawler family is strategically important, but it is not yet the best pure base-model anchor

3. **The strongest current external neural needles are systems + legality-safe**
   - `#1060`: coprime-stride loader + full Hessian GPTQ + XSA-all, `1.1122`
   - `#1072`: fused Triton LeakyReLU^2 MLP + online Hessian GPTQ, projected `1.117` from a faster training loop

So the right move is:
- keep Bandit under active attention as the compact architecture hedge
- build the next garage on the stronger Rat Rod base
- import the two external ideas most likely to compound with that base

## Anchor Record

### Local best honest base anchor

`Rat Rod Green v1`

| Metric | Value |
|---|---:|
| Sliding BPB | `1.1129` |
| Post-EMA BPB | `1.1364` |
| Steps | `6882` |
| ms/step | `87.20` |
| N-gram legal BPB | `0.4489` |

Config highlights:
- Parallel Muon
- XSA on all 11 layers
- BigramHash `2048`
- RoPE dims `16`
- SWA every `50`
- no GPTQ in this base harness
- no complementary training

### Evidence of 1.10x latent capacity

Archived local notes already claim:
- chunk-51 peak around `1.1104`
- running average around `1.1107`

That is not a final submission metric, but it is enough signal to justify a new garage built around squeezing training throughput and artifact path quality out of the `1.1129` base.

## Imported Hypotheses

### H1: Coprime-stride loader from `#1060`

Claim:
- More diverse batches at effectively zero step-time cost improve the same base stack without changing the model.

Why it fits this garage:
- Rat Rod is already a strong pure-model anchor.
- Zero-overhead data diversity is exactly the kind of change that can move `1.1129` to `1.10x` without making the codebase brittle.

What to import:
- multi-shard block sampling
- coprime strides per shard
- no reduction in throughput tolerance

Success condition:
- equal or better step time than Rat Rod Green v1
- lower cap-BPB and lower final sliding BPB

### H2: Fused Triton LeakyReLU^2 MLP from `#1072`

Claim:
- Fuse `linear -> leaky_relu -> square` into one GPU pass and buy back enough wallclock to materially increase steps.

Why it fits this garage:
- Rat Rod is already compute-efficient but still sits around `87ms/step`.
- The fastest path to more base quality may simply be more updates inside the same 600s.
- If this works on the flat 11L stack, it becomes a direct future lever for the crawler family too.

What to import:
- fused MLP kernel only
- no architectural drift in the first pass
- keep the rest of the Rat Rod stack fixed

Success condition:
- step time improvement that survives full 600s runs
- no regression in numerical stability
- net gain in final sliding BPB, not just prettier throughput

### H3: Online Hessian collection / legal Full GPTQ path

Claim:
- We should stop separating "base lane" and "artifact lane" so hard.
- Collect Hessian information during training or inside a reserved legal budget, then get a stronger artifact path without sacrificing wallclock discipline.

Why it fits this garage:
- Rat Rod v1 deliberately removed GPTQ for base testing.
- That was useful for signal isolation, but the next garage needs an artifact-ready finish.
- `#1060` and `#1072` both point the same direction: stronger quantization quality matters, but it must be integrated legally.

What to import:
- reserve explicit training-budget time for quantization calibration
- prefer online or in-budget Hessian gathering
- test Full GPTQ only after the loader and Triton changes are stable

Success condition:
- artifact-ready run under rules
- base quality preserved as much as possible after roundtrip

## Non-Goals

This garage does **not** start with:
- DeltaNet
- crawler loops
- n-gram mixer changes
- complementary training
- kitchen-sink combinations

Reason:
- we already know the crawler family is strategically important
- we also already know the cleanest current base signal lives in Rat Rod
- first we improve the base, then we feed the crawler/Bandit line with better ideas

## Garage Order

### Phase A: loader-only

Objective:
- port coprime-stride loader onto Rat Rod Green v1 semantics

Keep fixed:
- model
- optimizer
- XSA=11
- BigramHash=2048
- RoPE=16
- no GPTQ

Decision rule:
- if loader hurts throughput or fails to improve directionally, kill it fast

### Phase A result: `JR-01` wins

We now have a clean first A/B on the real 80-shard dataset.

| ID | Variant | Step avg | Post-EMA BPB | Sliding BPB | Decision |
|---|---|---:|---:|---:|---|
| `JR-00` | sequential loader | `87.08ms` | `1.1354` | `1.11184332` | loser |
| `JR-01` | coprime loader | `91.00ms` | `1.1340` | `1.11056240` | winner |

Interpretation:
- coprime costs about `+4.5%` in step time
- but improves sliding BPB by about `-0.00128`
- that is enough to keep as the active base lane

Organizational rule from here:
- the active runner stays in `experiments/Junkyard_Rat/run.sh`
- losing variants move into `experiments/Junkyard_Rat/losers/`

### Phase B: Triton-only

Objective:
- port fused LeakyReLU^2 MLP kernel onto the same base

Keep fixed:
- same loader state as winning Phase A variant, or v1 if loader dies

Decision rule:
- if it does not buy real steps in a 600s run, it is just complexity

### Phase B runner: `JR-02`

`JR-02` is now scaffolded as the next candidate on top of the `JR-01` coprime winner.

What is implemented:
- opt-in compile mode hook in `train_gpt.py`
- dedicated Triton/Inductor candidate runner
- dedicated local bench for the exact LeakyReLU^2 MLP path

Entry points:

```bash
python experiments/Junkyard_Rat/bench_triton.py
```

```bash
bash experiments/Junkyard_Rat/run_triton_candidate.sh
```

Triton work area:

```bash
ls experiments/Junkyard_Rat/triton
```

Interpretation rule:
- if `bench_triton.py` shows meaningful forward+backward speedup, run `JR-02`
- if the full run does not improve either step time or final sliding BPB, move it to `losers/`

### Phase C: artifact path

Objective:
- add legal Full GPTQ / online Hessian path

Keep fixed:
- best Phase A/B winner

Decision rule:
- prioritize honest submission readiness over theoretical quant elegance

### Phase D: hand-off to compact architecture line

Only after a base winner exists:
- test whether winning loader/Triton ideas transfer to Bandit_Wagon or crawler-only rebuild

This is where Bandit stays important:
- not as the strongest pure base today
- but as the compact architecture that can spend newly unlocked quality much more efficiently

## Current Comparative Position

| System | Pure base sliding BPB | Combined BPB | Size | Role |
|---|---:|---:|---:|---|
| Rat Rod Green v1 | `1.1129` | `0.4489` | larger | strongest honest base anchor |
| X-WING Cubric | `1.1199` | `0.4820` | `15.58 MB` | strong flat full-budget stack |
| Bandit | `1.1867` | `0.4961` | `~9.2 MB` | compact architecture hedge |
| Medusa_VII DN=0 | `1.1823` | n/a | `9.08 MB` | honest crawler baseline |

## Concrete Read on Bandit

Bandit should be interpreted as:
- a validated compact architecture
- a proof that there is real value in the crawler family even before spending the full 16 MB
- a headroom engine for later width/depth upgrades

Bandit should **not** be interpreted as:
- proof that the crawler base is already better than Rat Rod

So the right organizational split is:
- `Junkyard_Rat`: best clean base-model garage
- `Bandit_Wagon`: best compact-architecture headroom garage

## Exit Criteria

This garage is successful if it produces any one of these:

1. a reproducible base-model run better than `1.1129` sliding
2. an artifact-ready legal run that closes the base-to-roundtrip gap cleanly
3. a systems improvement that clearly transfers to Bandit_Wagon later

## Immediate Next Ablations

1. `JR-02`: `JR-01` winner + fused Triton LeakyReLU^2 MLP
2. `JR-03`: best loader/systems winner + legal Full GPTQ path
3. transfer winning systems ideas into Pocket_Bandit and Shroud_Crawler later

## References

- `experiments/Rat_Rod/PROGRESS.md`
- `experiments/Bandit_Wagon/HYPOTHESIS.md`
- `experiments/Medusa_VII/HYPOTHESIS.md`
- `records/track_10min_16mb/2026-03-29_Bandit_ClownCar_X_CubricNgram9_8xH100/README.md`
- `records/track_10min_16mb/2026-03-26_XWING_Cubric3D_complementary_8xH100/README.md`
