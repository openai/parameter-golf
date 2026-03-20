# Int6 MLP3x 11L + SmearGate + BigramHash4096x128 + MuonWD038 + SWA50 + DocSliding

**Single-run legal score:** `val_loss=1.95474571`, `val_bpb=1.15677715`

This folder packages the strongest local dense-lexical `11x512 KV4` checkpoint as a `track_10min_16mb` submission candidate.

As of `2026-03-20`, the README leaderboard leader is the 3-seed `Muon WD + 10 layer` submission at `mean_val_bpb=1.17475315`. This folder is numerically better than that score, but it only includes one recorded `8xH100` training run, so it should be treated as a single-run candidate rather than a statistically demonstrated new record.

## Model

- dense `11x512`, `KV4`, `TRAIN_SEQ_LEN=2048`
- `MLP_MULT=3`
- `SmearGate`
- `BigramHash(4096 x 128)`
- `MUON_WEIGHT_DECAY=0.038`
- `ADAM_WEIGHT_DECAY=0.01`
- `SWA_EVERY=50`, `SWA_START_FRAC=0.50`
- legal export and evaluation: `int6_zstd_core` with `doc_sliding 2048/256`

## Results

- Recorded training run: `step:6038/20000`, `train_time:597185ms`, `swa_updates:59`
- Pre-quant eval at stop from `train.log`: `val_loss=1.9529`, `val_bpb=1.1566`
- Built-in trainer roundtrip from `train.log`: `val_loss=1.98941715`, `val_bpb=1.17824489`, `artifact_bytes=16032236` (over cap)
- Chosen legal eval from `eval_doc2048_256.csv`: `val_loss=1.95474571`, `val_bpb=1.15677715`, `model_bytes=15634707`, `artifact_bytes=15704854`

The legal score in this folder comes from re-exporting the raw checkpoint with `checkpoint_frontier_sweep.py` under `int6_zstd_core`. The stronger integrated exporter used during the recorded training run kept `tok_emb.weight` and bigram tensors in float, which slightly exceeded the decimal `16,000,000` byte limit.

## Notes

- `train.log` is the canonical recorded `8xH100` run and embeds the original trainer source at the top of the file.
- The checked-in `train_gpt.py` is a whitespace-trimmed copy of that source. Only blank lines were removed so the script stays under the repo's `1500`-line limit; behavior is unchanged.
- Because of that trim, the checked-in code size is `70147` bytes instead of the `70319` bytes printed inside the original training log, and the checked-in artifact total is `15704854` bytes instead of `15705026`.

## Files

- `train.log`: canonical `8xH100` training run
- `train_gpt.py`: local training and export script used by this submission folder
- `checkpoint_frontier_sweep.py`: full-val scorer that imports the local `train_gpt.py`
- `eval_doc2048_256.csv`: chosen legal full-val evaluation
- `eval_doc2048_256_sweep.csv`: nearby export comparison sweep
- `run.sh`: reproduction wrapper for the recorded `600s` training run
- `eval_doc2048_256.sh`: reproduction wrapper for the chosen legal evaluation
- `submission.json`: leaderboard metadata

## Reproduction

```bash
bash run.sh
bash eval_doc2048_256.sh final_model.pt
```
