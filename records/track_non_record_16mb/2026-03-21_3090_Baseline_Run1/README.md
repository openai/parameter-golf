# Non-record: 1x RTX 3090 baseline run

This is a documented non-record baseline for Parameter Golf.

## Hardware
- Provider: RunPod
- GPU: 1x RTX 3090

## Data
- Dataset: fineweb10B_sp1024
- Tokenizer: fineweb_1024_bpe.model
- Train shards: 1

## Results
- In-run validation: val_loss=2.6198, val_bpb=1.5516
- Final int8+zlib roundtrip: val_loss=2.65578578, val_bpb=1.57290593
- Total submission size int8+zlib: 9283646 bytes
- Peak GPU memory allocated: 10699 MiB

## Command
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
python train_gpt.py

## Notes
This is not an official 8xH100 leaderboard-equivalent submission. It is a non-record baseline meant to document a working setup and support further compute requests.
