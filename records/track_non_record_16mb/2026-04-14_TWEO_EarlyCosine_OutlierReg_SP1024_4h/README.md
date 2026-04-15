# Non-record: TWEO Early-Cosine Activation Outlier Regularization

**Status:** draft research submission pending H100 confirmation.

**Current local 4h mean:** baseline `1.23128` BPB -> TWEO `1.22993` BPB (`-0.00134`), 2 matched seeds.

This is a non-record research submission testing TWEO (Transformers Without Extreme Outliers, Liang et al., 2025) on the Parameter Golf SP1024 baseline. TWEO is a training-only activation regularizer:

```text
L_total = L_CE + lambda(t) * (1/L) * sum_l mean((abs(A_l) / (tau + eps))^p)
```

where `A_l` is the post-block residual activation after `x = x + MLP(LN(x))`. I used `tau=5`, `p=4`, and a tiny early cosine decay from `lambda=0.0002` to `0` over the first `3000` steps.

The useful finding is narrow but reproducible in this setup: **always-on TWEO hurts BPB, but a small early TWEO pulse improves final int8+zlib BPB on matched 4h wallclock controls.** This suggests TWEO is acting as an early trajectory regularizer, not as a persistent compression regularizer.

## Results

All runs use the same SP1024 baseline architecture, same data, same validation, same int8+zlib roundtrip, `MAX_WALLCLOCK_SECONDS=14400`, and matched seed pairs.

| Seed | Run | Steps | Checkpoint BPB | Final int8+zlib BPB | Artifact bytes |
|---:|---|---:|---:|---:|---:|
| 42 | baseline | 11102 | 1.2247 | 1.23166876 | 15,875,162 |
| 42 | TWEO early cosine | 11050 | 1.2235 | 1.22994812 | 15,888,223 |
| 314 | baseline | 11100 | 1.2238 | 1.23088231 | 15,875,856 |
| 314 | TWEO early cosine | 11041 | 1.2233 | 1.22991567 | 15,892,526 |
| **Mean** | **baseline** | | | **1.23127554** | **15,875,509** |
| **Mean** | **TWEO early cosine** | | | **1.22993190** | **15,890,375** |

Mean delta: `-0.00134364` BPB. The TWEO runs are slightly slower and finish with fewer steps, so this is not a throughput artifact.

## What Worked

### Early cosine TWEO

Best tested setting:

```bash
TWEO_LAMBDA=0.0002
TWEO_LAMBDA_FINAL=0
TWEO_LAMBDA_SCHEDULE=cosine
TWEO_DECAY_STEPS=3000
TWEO_TAU=5
TWEO_P=4
```

The curve is not immediately better. It falls behind the baseline early, then recovers after the regularizer decays to zero. This pattern repeated on seeds 42 and 314.

### Activation geometry changed

TWEO sharply reduces activation extremes even after lambda reaches zero. Example at seed 314, step 8000:

| Run | Post-block absmax |
|---|---:|
| baseline | 9,240,576 |
| TWEO early cosine | 1,441,792 |

This confirms the intervention affects the mechanism targeted by the paper. The open question is how much that helps Parameter Golf's int8+zlib path.

## What Did Not Work

### Paper-scale / always-on TWEO

Large fixed lambdas looked excellent in very short, undertrained runs, then failed at longer horizons. This was the main trap in the project.

### Fixed tiny TWEO

Fixed `lambda=0.00005`, `tau=5` kept activations small but hurt the LM objective badly:

| Seed | Run | Final int8+zlib BPB |
|---:|---|---:|
| 42 | baseline | 1.23166876 |
| 42 | fixed `lambda=0.00005`, `tau=5` | 1.25002811 |

### Nonzero tail after cosine decay

A tiny tail (`TWEO_LAMBDA_FINAL=2e-6`) was already too much. At seed 42, step 4000:

| Run | Step 4000 BPB |
|---|---:|
| baseline | 1.2862 |
| pure cosine to 0 | 1.2907 |
| cosine with `2e-6` tail | 1.3030 |

This suggests persistent outlier suppression conflicts with the final predictor quality under this setup.

## Method

TWEO is implemented inside the model forward pass, accumulating the scaled Lp penalty over post-block residual outputs. It is only added to the training loss. Evaluation, final int8+zlib roundtrip, and validation do not train or adapt the model.

Implementation details:

- `TWEO_LAMBDA=0` disables the train-time TWEO path and recovers the baseline pipeline.
- `TWEO_LAYER_STRIDE=1` monitors every transformer block.
- `TWEO_ACT_STATS_EVERY` and `TWEO_ACT_STATS_BATCHES` only control no-grad diagnostic logging.
- Cosine schedule uses `TWEO_DECAY_STEPS=3000`; after that, lambda is exactly zero for the rest of training.

## Reproduction

Baseline:

```bash
SEED=42 \
RUN_ID=tweo_wall4h_seed42_base \
MAX_WALLCLOCK_SECONDS=14400 \
WARMDOWN_ITERS=1200 \
VAL_LOSS_EVERY=4000 \
TRAIN_LOG_EVERY=1000 \
TWEO_LAMBDA=0 \
python3 train_gpt.py
```

TWEO early cosine:

```bash
SEED=42 \
RUN_ID=tweo_wall4h_seed42_cosdecay_lam0002_tau5_d3000 \
MAX_WALLCLOCK_SECONDS=14400 \
WARMDOWN_ITERS=1200 \
VAL_LOSS_EVERY=4000 \
TRAIN_LOG_EVERY=1000 \
TWEO_LAMBDA=0.0002 \
TWEO_LAMBDA_FINAL=0 \
TWEO_LAMBDA_SCHEDULE=cosine \
TWEO_DECAY_STEPS=3000 \
TWEO_START_STEP=0 \
TWEO_RAMP_STEPS=0 \
TWEO_TAU=5 \
TWEO_P=4 \
TWEO_ACT_STATS_BATCHES=1 \
TWEO_ACT_STATS_EVERY=4000 \
python3 train_gpt.py
```

## Compliance

- Non-record, unlimited-compute-style experiment; current runs use a 4h wallclock cap.
- Artifact stays under `16,000,000` bytes in all included runs.
- No tokenizer or dataset changes.
- No validation/test-time training.
- No eval-time adaptation, no n-gram cache, no SLOT, no ETLB.
- No network calls or external downloads during evaluation.
- TWEO is a train-time-only loss term and adds no parameters to the artifact.

## Limitations

- This is not a leaderboard record and is far from the current SP8192 + Legal TTT stack.
- The result is only tested on the SP1024 baseline so far.
- The effect size is small: two matched 4h seed pairs show a mean `-0.00134` BPB, but more seeds are needed.
- Artifact size gets slightly worse in the winning setting.
- The paper's strongest claim is about enabling native FP8 training and W8A8 per-tensor static quantization. This submission does not use FP8 training; it tests whether TWEO transfers to Parameter Golf's BF16 training plus int8+zlib roundtrip.

## Planned H100 Confirmation

Before opening this PR, I plan to add H100 logs for matched baseline/TWEO runs and, if compute allows, a seed 999 local/H100 confirmation. The acceptance claim should be either:

- positive: TWEO early cosine improves BPB across 3 matched seeds, or
- negative/weak-positive: TWEO changes activation geometry reliably, but the BPB gain is small and not yet robust enough for a record path.

## Included Files

- `README.md`
- `submission.json`
- `results.tsv`
- `train_gpt.py`
- `train_seed42_base_4h.log`
- `train_seed42_tweo_cosdecay_4h.log`
- `train_seed314_base_4h.log`
- `train_seed314_tweo_cosdecay_4h.log`

## Credits

- TWEO: Guang Liang, Jie Shao, Ningyuan Tang, Xinyao Liu, Jianxin Wu, *Transformers Without Extreme Outliers Enables FP8 Training And Quantization For Dummies*.
- Parameter Golf baseline and challenge infrastructure: OpenAI Model Craft Challenge.
