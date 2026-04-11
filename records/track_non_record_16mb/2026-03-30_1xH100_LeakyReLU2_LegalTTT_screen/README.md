# 1xH100 LeakyReLU2 + Legal TTT screening

This is a non-record submission intended as a low-cost 1xH100 screening run of the 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon record family.

## Why submit this
I used this run to test whether this newer frontier branch still looked promising in a cheaper 1xH100 setting before requesting more compute for multi-GPU reproduction.
The result was an interesting negative: the run looked reasonable during training, but performance collapsed after the final compressed artifact and final evaluation path.

## Setup
- Hardware: 1xH100
- Wallclock cap: 180 seconds
- Seed: 42
- Dataset: FineWeb sp1024, train-shards 1
- Script: copied from the original record folder with no algorithmic changes

## Result
- Mid-run validation at wallclock stop:
  - val_bpb: 2.1050
- Final compressed artifact result:
  - final_int6_roundtrip_exact val_bpb: 4.62390226
- Final submission size:
  - 5019370 bytes

## Takeaway
This branch showed signs of life during training but did not survive the full final artifact and evaluation path in this low-budget 1xH100 regime.
I am submitting it as an interesting negative result and screening datapoint, not as a leaderboard attempt.
