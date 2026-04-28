# Non-record: Model Stack BitNet MLP2304 + Overtone Embeddings, SP1024

This is the best Model Stack BitNet-compatible Parameter Golf run from the
current experiment series.

It trains the Parameter Golf transformer directly with Model Stack's
`TrainableBitNetLinear` QAT modules, exports a packed runtime-row BitNet bundle,
and keeps the packed bundle plus submitted code under the 16MB budget.

## Result

| Metric | Value |
|---|---:|
| Final sliding val_loss | 2.0607 |
| Final sliding val_bpb | 1.2205 |
| Standard val_loss at stop | 2.0805 |
| Standard val_bpb at stop | 1.2322 |
| Training steps | 6,466 |
| Training time | 599,020 ms |
| Step average at stop | 92.64 ms |
| Model Stack BitNet bundle bytes | 15,612,895 |
| Code bytes | 80,393 |
| Bundle + code bytes | 15,693,288 |
| Budget | 15.69MB / 16.00MB |

The run improves over the earlier legal Model Stack BitNet MLP2 run:

| Variant | Steps | Step avg | Sliding val_bpb | Bundle + code |
|---|---:|---:|---:|---:|
| MLP2 dense-backward | 5,968 | 100.38 ms | 1.2303 | 14,753,017 |
| MLP2304 + overtone | 6,466 | 92.64 ms | 1.2205 | 15,693,288 |

## Technique

- Model Stack `TrainableBitNetLinear` QAT modules wired into Parameter Golf training.
- Runtime-row packed BitNet export, with 28 packed modules and 51,380,224 packed ternary params.
- Fused QKV projection and FlashAttention backend.
- Dense training backward for both grad-input and grad-weight where the full compiled step is faster than the int8 backward candidates.
- Parallel Muon matrix updates with shape-bucket sharding.
- ReLU2 MLP with `MLP_HIDDEN_DIM=2304`, which uses the legal-size headroom without exceeding 16MB.
- Overtone spectral embedding initialization: QR-orthogonalized token embeddings with a power-law spectrum `S_k ~ k^-0.5`.
- Full validation uses legal sliding-window evaluation with stride 64.

## Configuration

- Tokenizer/data: `sp1024`, cached FineWeb challenge split, 80 train shards.
- Context: `TRAIN_SEQ_LEN=4096`, `YARN_MAX_LEN=4096`.
- Model: `MODEL_DIM=1024`, `NUM_LAYERS=7`, `NUM_HEADS=16`, `NUM_KV_HEADS=4`.
- MLP: `MLP_HIDDEN_DIM=2304`, `ACTIVATION=relu2`.
- Overtone embedding: `OVERTONE_EMBED_INIT=1`, `OVERTONE_EMBED_POWER=0.5`, `OVERTONE_EMBED_SCALE=1.0`.
- Model Stack BitNet: `MODEL_STACK_BITNET_QAT=1`, `MODEL_STACK_BITNET_SCALE_LAYOUT=runtime_row`, `MODEL_STACK_BITNET_GROUP_SIZE=64`.
- Training: `TRAIN_BATCH_TOKENS=524288`, `MAX_WALLCLOCK_SECONDS=599`, `ITERATIONS=20000`.
- Hardware: 8x NVIDIA H100 80GB HBM3.

## Run Command

The exact launcher is included as `run_mlp2304_overtone_8xh100.sh`.

```bash
bash run_mlp2304_overtone_8xh100.sh
```

## Included Files

- `train_gpt.py`: standalone training/evaluation script used for this run.
- `train.log`: full canonical 8xH100 run log.
- `run_mlp2304_overtone_8xh100.sh`: exact launcher for the run.
- `submission.json`: metadata for this non-record submission.

## Notes

This is not a leaderboard record against the SP8192 + legal TTT submissions. It
is intended as the strongest current Model Stack BitNet PR artifact: the training
stack uses Model Stack QAT modules, the exported packed BitNet bundle fits the
track budget, and the run demonstrates a faster and better legal BitNet result
than the earlier MLP2 baseline.
