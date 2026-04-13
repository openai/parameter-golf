# CAGE5 H100 proxy validation (non-record 16MB)

This folder captures a non-record H100 proxy validation for Parameter Golf.

It validates a strictly causal hashed 5-gram mixer stacked with legal score-first TTT on 1x H100. This is **not** a final 8xH100 leaderboard submission; it is a stronger proxy validation of the method before frontier-port work and 3-seed validation.

## Hardware

- 1x NVIDIA H100 80GB HBM3
- BF16 enabled

## Core idea

- legal score-first per-chunk TTT
- strictly causal hashed 5-gram interpolation inside both sliding-window and legal-TTT scoring paths

## Best result in `train.log`

- `final_int6_roundtrip_exact val_bpb = 2.74196351`
- `final_int6_sliding_window_exact val_bpb = 2.41599902`
- `legal_ttt_exact val_bpb = 2.41025369`
- `Total submission size int6+lzma = 1372587 bytes`

## Notes

- This run uses a reduced proxy configuration and is meant to validate the method on H100 hardware.
- 3-seed validation is in progress.
