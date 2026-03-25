This directory contains a non-record submission built from the March 23 leader, with the same base model and training recipe but a different legal test-time training protocol.

## Non-Record Submission

This run is intentionally submitted as a non-record result.

The goal is to contribute a clear data point on streaming legal TTT:

- same base model as the March 23 leader
- same `LeakyReLU(0.5)^2` activation
- same 8xH100 / 10-minute training setup
- different legal TTT procedure at evaluation

The hope is that this helps the community separate:

- gains from the underlying trained model
- gains from the exact legal TTT protocol

## What Changed

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, this submission changes the eval-time adaptation path in two ways:

- `TTT_MODE=stream` instead of chunked score-first TTT
- `TTT_PARAM_MODE=late_blocks` with `TTT_LAST_N_BLOCKS=4`

The resulting run adapts online on the newly scored suffix of each sliding-window evaluation batch, and only updates the last 4 transformer blocks during TTT. The candidate code also adds explicit bank-gradient masking so the late-block restriction applies correctly to the parameter-bank tensors.

Training, architecture, export, and activation were otherwise kept aligned with the March 23 leader starting point.

## Why This Is Interesting

The final result was slightly worse than the March 23 leader on the final `legal_ttt_exact` metric, but only by a small margin:

- this run: `1.12082320`
- March 23 leader seed-1337: `1.11922988`

The intermediate metrics were very close:

- this run `DIAGNOSTIC post_ema`: `1.1366`
- leader seed-1337 `DIAGNOSTIC post_ema`: `1.1369`
- this run `final_int6_sliding_window_exact`: `1.12125572`
- leader seed-1337 `final_int6_sliding_window_exact`: `1.12165091`

That makes this a useful negative or near-neutral result: the underlying trained model remained competitive, but the specific switch from chunked all-block TTT to streaming late-block TTT did not beat the incumbent TTT recipe in this full run.

## Exact Run Settings

The promoted run used:

- `TTT_MODE=stream`
- `TTT_PARAM_MODE=late_blocks`
- `TTT_LAST_N_BLOCKS=4`
- `TTT_LR=0.0005`
- `TTT_EPOCHS=1`
- `TTT_LR_SCHEDULE=constant`
- `DATA_ROOT_MODE=tmp`
- `NPROC_PER_NODE=8`

## Results

From `train_seed1337.log`:

- `DIAGNOSTIC post_ema val_bpb: 1.1366`
- `final_int6_roundtrip_exact val_bpb: 1.14466312`
- `final_int6_sliding_window_exact val_bpb: 1.12125572`
- `legal_ttt_exact val_bpb: 1.12082320`
- `Total submission size int6+lzma: 15867004`
- `Code size: 101140`

## Included Files

- `train_gpt.py`
- `train_seed1337.log`
- `runpod_launch.log`
- `preflight.log`
- `submission.json`
