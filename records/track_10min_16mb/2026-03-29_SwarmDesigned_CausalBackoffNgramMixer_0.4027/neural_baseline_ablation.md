# Neural-only Ablation — Where the 0.3958 BPB comes from

This file decomposes the 0.3958 BPB submission into **(a) the trained neural
model** and **(b) the eval-time Causal BackoffNgramMixer**, using the exact
log lines from the three archived runs that produced `submission.json`.

**TL;DR:** the trained neural model by itself scores ~1.148 BPB. The same
model + `BackoffNgramMixer` at eval time scores 0.3958 BPB. The **~0.75
BPB improvement is entirely an eval-stage compression refinement**; no
training-objective change, no data leakage, no novel optimizer. This is
a direct descendant of already-merged #779 and #803.

## Per-seed ablation (from the archived run logs)

Source: `swarm_submission/run_final_seed{7,1337,2024}.log`, same runs that
populate `submission.json`.

| seed | post-EMA diagnostic<br>(neural, no quant, no mixer) | `final_int6_roundtrip`<br>(neural, int6 point eval) | `final_int6_sliding_window`<br>(neural + mixer, stride=64) |
|---|---|---|---|
| 7    | **1.1394** | **1.1481** | **0.3948** |
| 1337 | **1.1396** | **1.1480** | **0.3957** |
| 2024 | **1.1404** | **1.1492** | **0.3969** |
| **mean** | **1.1398** | **1.1484** | **0.3958** |

- `post-EMA diagnostic` = `train_gpt.py:1483` — the raw trained model's val_bpb on a standard non-sliding-window eval, taken immediately after EMA weight decay, before any quantization. This is the purest "neural only" number.
- `final_int6_roundtrip` = `train_gpt.py:1551` — same weights after int6 GPTQ-lite quantization + LZMA compression roundtrip, still no mixer, still point eval. ~0.009 BPB of quant noise vs the diagnostic.
- `final_int6_sliding_window` = `train_gpt.py:1577` — **same int6 weights**, sliding-window eval at stride=64, **with the mixer enabled**. No further training, no further weight changes.

**Mixer-attributed delta: 1.1484 − 0.3958 = 0.7526 BPB** (mean across seeds).

## Verbatim log excerpts

### seed 7 (`run_final_seed7.log`)
```
step:7024/20000 val_loss:1.9257 val_bpb:1.1405 train_time:600086ms step_avg:85.43ms
stopping_early: wallclock_cap train_time:600086ms step:7024/20000
DIAGNOSTIC post_ema val_loss:1.9239 val_bpb:1.1394 eval_time:1989ms
final_int6_roundtrip val_loss:1.9386 val_bpb:1.1481 eval_time:19276ms
final_int6_sliding_window val_loss:0.6667 val_bpb:0.3948 stride:64 eval_time:582774ms
final_int8_zlib_roundtrip_exact val_loss:0.66665722 val_bpb:0.39483300
```

### seed 1337 (`run_final_seed1337.log`)
```
DIAGNOSTIC post_ema val_loss:1.9241 val_bpb:1.1396 eval_time:1988ms
final_int6_roundtrip val_loss:1.9383 val_bpb:1.1480 eval_time:5946ms
final_int6_sliding_window val_loss:0.6681 val_bpb:0.3957 stride:64 eval_time:593857ms
final_int8_zlib_roundtrip_exact val_loss:0.66811451 val_bpb:0.39569610
```

### seed 2024 (`run_final_seed2024.log`)
```
DIAGNOSTIC post_ema val_loss:1.9254 val_bpb:1.1404 eval_time:2109ms
final_int6_roundtrip val_loss:1.9404 val_bpb:1.1492 eval_time:16040ms
final_int6_sliding_window val_loss:0.6701 val_bpb:0.3969 stride:64 eval_time:595814ms
final_int8_zlib_roundtrip_exact val_loss:0.67013029 val_bpb:0.39688996
```

## Mixer convergence curve (seed 7)

The mixer starts empty and accumulates n-gram counts in strict score-first
order as it walks the val stream. Running BPB across the eval (every ~128K
tokens of 969088 total):

| tokens scored | running bpb |
|---|---|
| 128 / 969088 | 1.175661 |
| 102528 / 969088 | 0.889010 |
| 230528 / 969088 | 0.643985 |
| 358528 / 969088 | 0.538056 |
| 486528 / 969088 | 0.483657 |
| 614528 / 969088 | 0.448113 |
| 742528 / 969088 | 0.423662 |
| 870528 / 969088 | 0.406234 |
| **969088 / 969088** | **0.394833** |

The first scored batch (128 tokens) is at 1.176 BPB — effectively the
neural-only floor since the mixer has no counts yet. As the mixer
accumulates counts from already-scored tokens, BPB drops monotonically
to 0.3948. **At no point does the mixer see a token before it is scored**
(see `train_gpt.py:876-935`, `eval_val_sliding` with mixer).

## Relationship to prior art

- **#779** — original `BackoffNgramMixer`, flat-hash design, entropy-adaptive alpha. Merged.
- **#803** — @pentxayc's Complementary Training + `BackoffNgramMixer` at 0.4416. Merged.
- **#1094 (this PR)** — same mixer family as #803, three orthogonal refinements:
  1. Higher n-gram orders (2–10 vs 2–7)
  2. 4.2M hash buckets per order (vs 1M)
  3. Causal sequential chunk eval (score-first-per-batch, strictly backward-looking — `train_gpt.py:876-935`)

The 0.0458 improvement over #803 is an eval-stage refinement on top of a
legal, merged technique — not a new training method, not a new objective,
not a new dataset.

## Reproducibility

```bash
USE_NGRAM_MIXER=1 NGRAM_ORDER=10 NGRAM_BUCKETS=4194304 \
SEED=7 python train_gpt.py        # expected: 0.3948 ± 0.001 BPB
SEED=1337 python train_gpt.py     # expected: 0.3957 ± 0.001 BPB
SEED=2024 python train_gpt.py     # expected: 0.3969 ± 0.001 BPB
```

3-seed mean 0.3958 BPB, std 0.0011, all under the 16 MB artifact cap
(15,943,009 / 15,940,706 / 15,957,577 bytes) and the 600 s eval cap
(583 / 594 / 596 s). See `submission.json`.
