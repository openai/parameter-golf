# Non-Record Submission: 1xH100 Budget Seq2048 + LR Warmdown Tune

This is a non-record submission documenting progress under strict credit constraints on a single H100.

## Best Result

- `final_int8_zlib_roundtrip_exact val_bpb`: **1.30294738**
- `final_int8_zlib_roundtrip_exact val_loss`: **2.19997208**
- `Total submission size int8+zlib`: **11,851,989 bytes**
- Wallclock cap: **600s**
- Hardware: **1x H100**

## Configuration

The best run used:

- `TRAIN_SEQ_LEN=2048`
- `MATRIX_LR=0.028`
- `SCALAR_LR=0.028`
- `TIED_EMBED_LR=0.038`
- `WARMDOWN_ITERS=2200`
- `MAX_WALLCLOCK_SECONDS=600`
- `torchrun --standalone --nproc_per_node=1 train_gpt.py`

## Notes

- This run is intentionally budget-focused and intended as a reproducible non-record baseline.
- The artifact remains safely below the 16,000,000-byte cap.

## Included Files

- `train_gpt.py` - training/eval script used for the run
- `train.log` - captured run output excerpt with final metrics
- `submission.json` - metadata for this submission
