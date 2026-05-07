This record captures our strongest verified single-GPU H100 frontier-family run to date.

This is a `non-record` submission and is not intended for leaderboard acceptance. It is a development/engineering run used to validate the March 20 frontier family on a real H100 before spending scarce `8xH100` budget on exact frontier reproductions.

The main differences from leaderboard-valid runs are:

- hardware is `1x NVIDIA H100 NVL 94GB`, not `8xH100 SXM`
- training used a longer wallclock cap (`1800s`) for engineering telemetry
- the exact post-quant roundtrip metric completed, but the trailing eval tail was interrupted by `SIGTERM`, so this should not be treated as a leaderboard-comparable sliding-window score

Why submit this anyway:

- it is our first verified H100-side run that reaches the low `1.2x` BPB regime
- the artifact is real and well under the `16,000,000` byte cap
- it gives a reproducible H100 engineering anchor between cheap T4 proxy work and `8xH100` submission-grade runs
- it materially improved over our prior proxy-only captures and justified the current clean-frontier reproduction plan

Configuration:

- Track: `non-record`, single-GPU engineering, still under the `16,000,000` byte artifact cap
- Base family: March 20 frontier `10L Int5-MLP + BigramHash(10240) + SWA + WD=0.04`
- Exact repo commit: `8e8c6d7a23f1bdf26905c0d93e43b5233a45a8ac`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Bigram path: `BIGRAM_VOCAB_SIZE=10240 BIGRAM_DIM=128`
- Batching: `TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048`
- Training cap: `MAX_WALLCLOCK_SECONDS=1800`
- Export/eval: exact `int8+zlib` roundtrip completed at `eval_seq_len:1024`

Command:

```bash
cd /home/warrenjo/src/parameter-golf
git checkout feature/contest-frontier-stack
RUN_ID=h100_frontier_repro \
MAX_WALLCLOCK_SECONDS=1800 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Key metrics from `train.log`:

- final pre-quant eval at stop: `val_loss:2.1260`, `val_bpb:1.2591`
- exact post-quant roundtrip eval: `val_loss:2.13139766`, `val_bpb:1.26233375`
- post-quant gap: `+0.00323375 bpb`
- step stop: `3752`
- train time: `1800135ms` (`step_avg:479.78ms`)
- peak memory: `12234 MiB allocated`, `13294 MiB reserved`
- serialized model: `67224578 bytes`
- code size: `65146 bytes`
- serialized model int8+zlib: `12662104 bytes`
- total submission size int8+zlib: `12727250 bytes`

Important caveat:

- after the exact `final_int8_zlib_roundtrip_exact` metric printed, the process received `SIGTERM`, so the trailing eval tail did not complete. We are publishing this as a transparent engineering record, not as a contest-grade score.

Included files:

- `train_gpt.py`: exact root training script snapshot from commit `8e8c6d7`
- `train.log`: exact Azure H100 training log
- `submission.json`: metadata for this non-record engineering submission
