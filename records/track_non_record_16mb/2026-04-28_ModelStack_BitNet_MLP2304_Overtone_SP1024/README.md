# Non-record: Model Stack BitNet MLP2304 + Overtone Embeddings, SP1024

This is the best Model Stack BitNet-compatible Parameter Golf run from the
current experiment series.

It trains the Parameter Golf transformer directly with Model Stack's
`TrainableBitNetLinear` QAT modules, exports a packed runtime-row BitNet bundle,
and keeps the packed bundle plus submitted code under the 16MB budget.

Important artifact caveat: the standard `final_model.pt` and
`final_model.int8.ptz` artifacts emitted by this script are not under 16MB. The
under-budget artifact is the custom Model Stack BitNet artifact
`final_model.int1.ptz` plus code. This is therefore a non-record Model Stack
packed-export artifact until the restore/evaluation path scores that artifact
directly.

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
| Sliding eval time | 401,147 ms |
| Model Stack BitNet bundle bytes | 15,612,895 |
| Model Stack int1+zlib bytes | 12,184,174 |
| Code bytes | 86,535 |
| Model Stack bundle + code bytes | 15,699,430 |
| Model Stack int1+zlib + code bytes | 12,270,709 |
| Raw `final_model.pt` + code bytes | 207,847,742 |
| Int8+zlib `final_model.int8.ptz` + code bytes | 42,950,518 |
| Model Stack int1+zlib budget | 12.27MB / 16.00MB |

The sliding-window evaluation time is 401.147s, which is below the competition
evaluation cap of 10 minutes on 8xH100. The stock raw and int8+zlib artifacts
are over the 16,000,000-byte artifact cap; only the Model Stack packed BitNet
artifacts are under the size limit. `final_model.int1.ptz` is the compressed
packed BitNet submission artifact.

The run improves over the earlier legal Model Stack BitNet MLP2 run:

| Variant | Steps | Step avg | Sliding val_bpb | Bundle + code |
|---|---:|---:|---:|---:|
| MLP2 dense-backward | 5,968 | 100.38 ms | 1.2303 | 14,753,017 |
| MLP2304 + overtone | 6,466 | 92.64 ms | 1.2205 | 12,265,263 int1+zlib |

## Technique

- Model Stack `TrainableBitNetLinear` QAT modules wired into Parameter Golf training.
- Runtime-row packed BitNet export, with 28 packed modules and 51,380,224 packed ternary params.
- Explicit `final_model.int1.ptz` artifact with packed BitNet runtime metadata, including activation quantization mode/bits.
- Direct `final_model.int1.ptz` restore path that rebuilds `QuantizedLinearBitNet` runtime modules for roundtrip evaluation.
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
- Activation quant export defaults: `MODEL_STACK_BITNET_ACTIVATION_QUANT=none`, `MODEL_STACK_BITNET_ACTIVATION_BITS=8`. Set `MODEL_STACK_BITNET_ACTIVATION_QUANT=dynamic_int8` and `MODEL_STACK_BITNET_ACTIVATION_BITS=4` for a4.8-style W1.58A4 export/eval experiments.
- Training: `TRAIN_BATCH_TOKENS=524288`, `MAX_WALLCLOCK_SECONDS=599`, `ITERATIONS=20000`.
- Hardware: 8x NVIDIA H100 80GB HBM3.

## Run Command

The exact launcher is included as `run_mlp2304_overtone_8xh100.sh`.

```bash
bash run_mlp2304_overtone_8xh100.sh
```

## Included Files

- `train_gpt.py`: standalone training/evaluation script used for this run.
- `final_model.int1.ptz`: generated compressed packed BitNet artifact name.
- `train.log`: full canonical 8xH100 run log.
- `run_mlp2304_overtone_8xh100.sh`: exact launcher for the run.
- `submission.json`: metadata for this non-record submission.

## Notes

This is not a leaderboard record against the SP8192 + legal TTT submissions. It
is intended as the strongest current Model Stack BitNet PR artifact: the training
stack uses Model Stack QAT modules, the exported packed BitNet `int1.ptz` plus
code fits the track budget, and the run demonstrates a faster and better Model
Stack BitNet result than the earlier MLP2 baseline. The script now writes
`final_model.int1.ptz` and includes a direct restore/eval path for that artifact;
the logged canonical run predates that added roundtrip line, so the prior
`final_sliding` score remains the measured score until the same config is
rerun with the direct `int1` roundtrip enabled.
