# Non-Record Submission: Compiled LeakyReLU2 + Warmdown300 (1xRTX4090)

This is a non-record submission for the 16MB track documenting a structured single-GPU search on a **1x RTX 4090**. The best run in this folder is a compiled 300-second proxy run built on top of the stock 9-layer configuration with:

1. `LeakyReLU2` MLP activation
2. quarter batch sizing (`TRAIN_BATCH_TOKENS=131072`)
3. a longer `WARMDOWN_ITERS=300` schedule
4. `torch.compile` enabled

The final post-quant roundtrip score is **1.42394278 val_bpb** with a total submission artifact size of **14,624,248 bytes**, which is under the 16MB cap.

This is **not** a leaderboard record claim. It was run on a single RTX 4090 for **300 seconds**, not the official 8xH100 / 600s setting. The point of this submission is to document a reproducible, interesting non-record result and the search path that led to it.

## Summary of What Helped

The useful changes in this search were:

1. **Compiled execution**: once `torch.compile` was working, the proxy changed substantially. The same baseline class of run improved from roughly `1.85` in the early uncompiled 120s proxy to `1.69` compiled at 120s.
2. **Longer warmdown**: increasing warmdown kept helping on the compiled 300s proxy through `WARMDOWN_ITERS=300`.
3. **LeakyReLU2**: replacing the default ReLU2-style MLP activation with a LeakyReLU-squared variant gave a small but repeatable gain in the early proxy and remained part of the best compiled run.
4. **Quarter batch / more optimizer steps**: cutting `TRAIN_BATCH_TOKENS` to `131072` made compiled runs much more effective on a single 4090 in fixed wall-clock time.

## Best Run

Run ID:

- `baseline_leaky_wd300_compiled_tb131072_uc300`

Key metrics from `train.log`:

- `final_int8_zlib_roundtrip_exact val_bpb`: **1.42394278**
- `final_int8_zlib_roundtrip_exact val_loss`: **2.40426773**
- `step_stop`: **1206**
- `wallclock_seconds`: **300**
- `Total submission size int8+zlib`: **14,624,248 bytes**
- `Serialized model int8+zlib`: **14,569,734 bytes**
- `Code size`: **54,514 bytes**
- peak memory: **2664 MiB allocated / 3352 MiB reserved**

Exact configuration for the best run:

```bash
VOCAB_SIZE=1024
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=2
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=131072
MAX_WALLCLOCK_SECONDS=300
WARMDOWN_ITERS=300
MLP_ACTIVATION=leakyrelu2
LEAKY_SLOPE=0.5
```

Other logged settings of note:

- `tie_embeddings=True`
- `matrix_lr=0.04`
- `scalar_lr=0.04`
- `embed_lr=0.05`
- `qk_gain_init=1.5`
- `EMA_DECAY=0.0`
- `QAT_START_FRACTION=0.0`

## Search Process

This submission comes from a **29-experiment** search loop. The full table is included in `results.tsv`.

Important milestones:

| Run | val_bpb | Note |
| --- | --- | --- |
| `baseline_uc120` | `1.86281828` | initial uncompiled baseline proxy |
| `baseline_leaky_uc120` | `1.84941753` | early gain from LeakyReLU2 |
| `baseline_leaky_compiled_tb131072_uc120` | `1.68955985` | compile + quarter batch changed the proxy materially |
| `baseline_leaky_compiled_tb131072_uc300` | `1.46329953` | longer compiled run |
| `baseline_leaky_wd80_compiled_tb131072_uc300` | `1.43852172` | warmdown clearly helping |
| `baseline_leaky_wd150_compiled_tb131072_uc300` | `1.43197024` | more warmdown still better |
| `baseline_leaky_wd200_compiled_tb131072_uc300` | `1.42924227` | still improving |
| `baseline_leaky_wd250_compiled_tb131072_uc300` | `1.42569532` | still improving |
| `baseline_leaky_wd300_compiled_tb131072_uc300` | **`1.42394278`** | best run |

## Negative Results

Several ideas that looked attractive from leaderboard writeups did **not** help in this proxy search:

- **11-layer stacks**: the compiled 300s 11-layer run improved versus the early uncompiled proxy, but still underperformed the best 9-layer run and exceeded the 16MB cap in the tested form.
- **EMA**: live validation improved slightly, but export-time roundtrip metrics collapsed badly in these runs.
- **Late QAT / fake-QAT**: forward-only fake-QAT behaved better than destructive QAT, but neither beat the best compiled warmdown path here.
- **`TRAIN_SEQ_LEN=2048`**: slightly worse than the `1024` sequence-length winner in the 300s compiled proxy.

These negative results are included because they were informative in narrowing the search direction.

## Hardware Note

This work used a **single RTX 4090** and a **300-second** wall-clock cap. Because of that, it should be interpreted as a reproducible, budget-conscious proxy result, not a claim about final leaderboard rank. The architectural and schedule changes here should be validated on the official 8xH100 / 600s setting before any record claim.

## Included Files

- `train_gpt.py` — self-contained training script snapshot for this submission
- `train.log` — automatically produced log from the best run
- `results.tsv` — full 29-run experiment table used during the search
- `submission.json` — metadata for this non-record submission

