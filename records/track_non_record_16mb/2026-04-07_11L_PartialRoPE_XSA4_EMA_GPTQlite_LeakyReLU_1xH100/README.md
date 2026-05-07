# Non-Record Submission: 11L PartialRoPE XSA4 EMA GPTQ-lite LeakyReLU on 1xH100

This is a non-record 16MB submission derived from the March 22 GPTQ-lite record, with one targeted architectural change ported from the March 23 record: the MLP activation is changed from `ReLU^2` to `LeakyReLU(0.5)^2`.

This run was used as a fair 1xH100 screening experiment, not as an 8xH100 main-leaderboard attempt. It uses the same frozen 4-shard FineWeb prefix, the same 600-second wallclock cap, the same validation cadence, and the same seed as the matched official-baseline comparison run.

## Summary of Changes

- Base lineage: `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- One-line MLP change from `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
- Keep the proven 11-layer 512d / 8-head / 4-KV-head / tied-embedding / 3x-MLP stack
- Keep Partial RoPE (`16/64`), LN scaling, XSA on last 4 layers, EMA, SWA, VE(128) on layers `9,10`, warmdown `3500`, and the existing int6+zstd GPTQ-lite export path
- Do not add legal TTT, XSA-all, AR self-generated GPTQ, selective pruning, or the larger 3072-wide BigramHash stack

## Run Setup

- Track: `non-record-16mb`
- Hardware: `1xH100 80GB`
- Dataset prefix: first `4` training shards of `fineweb10B_sp1024`
- Validation split: full `fineweb_val_*`
- Wallclock cap: `600.043s`
- Seed: `1337`

Command used for the recorded run:

```bash
RUN_ID=candidate_v1_ab_1gpu_shards4_t600 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
ITERATIONS=3000 \
TRAIN_BATCH_TOKENS=131072 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Key Metrics

Conservative post-quant headline metric used in `submission.json`:

- `final_int6_roundtrip_exact val_bpb: 1.30806928`
- `final_int6_roundtrip_exact val_loss: 2.20862019`

Pre-quant stop metrics:

- `step_stop: 2552`
- `val_bpb: 1.2916`
- `val_loss: 2.1809`

Artifact size:

- `Serialized model int6+zstd: 12821686 bytes`
- `Code size: 67625 bytes`
- `Total submission size int6+zstd: 12889311 bytes`

Matched 1xH100 baseline comparison under the same 4-shard / 600-second setup:

- official baseline `final_int8_zlib_roundtrip_exact val_bpb: 1.33779045`
- this submission `final_int6_roundtrip_exact val_bpb: 1.30806928`

## Note on Additional Sliding-Window Logs

This script also prints:

- `final_int6_sliding_window_exact val_bpb: 1.28388104`

and then reuses the legacy label `final_int8_zlib_roundtrip_exact` for that same sliding-window number. That trailing label is only a logging artifact of this experiment script. The conservative submission metric recorded in `submission.json` is the explicit `final_int6_roundtrip_exact` line above.

## Included Files

- `train_gpt.py`: exact code snapshot used for the recorded run
- `train.log`: exact 1xH100 run log
- `submission.json`: non-record metadata
