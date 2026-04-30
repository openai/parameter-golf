# Non-Record Submission: Saliency-Guided Local 5090

This record documents a local 1xRTX 5090 saliency-guided sweep around the 9x512 GQA baseline. The run sequence was:

1. the original 5B-token full-eval run on March 26, 2026
2. a 24-hour full-eval continuation on April 8, 2026
3. a 1-hour cosine-schedule proxy run on April 9, 2026

The folder is intentionally normalized to the standard record layout:

- `README.md` for the write-up
- `results.tsv` for the run table
- `submission.json` for the best full-eval run metadata
- `train_gpt.py` for the experiment-local code snapshot

`results.tsv` is transcribed from the local `wandb/` run directories, not from ad hoc notes.

## Best Result

The best full-eval run in this sweep is the 24-hour continuation:

- run: `saliency_24h_30gb_legacy_seed2025`
- started: `2026-04-08 11:21:09Z` / `2026-04-08 20:21 KST`
- pre-quant `val_bpb`: `1.21953852`
- post-quant `val_bpb`: `1.22864374`
- counted artifact bytes: `15,850,915`
- stop step: `44,628`
- GPU: `1xRTX5090`

This is the run described in `submission.json`.

## Sweep Summary

All three recorded runs use the same core saliency recipe:

- 9 layers, width 512, 8 heads, 4 KV heads, MLP mult 2
- SentencePiece vocab 1024, sequence length 1024
- saliency token prior on
- saliency dynamic correction on
- saliency phrase term on
- saliency attention bias on
- saliency bigram off
- `TRAIN_BATCH_TOKENS=262144`

The changes across the sweep were:

- 5B run: fixed-token budget, `50` train shards, legacy flat-plus-tail warmdown schedule
- 24h run: same legacy schedule and saliency settings, but extended to `125` train shards and a 24-hour wallclock cap
- 1h run: same saliency settings, but reduced to `8` train shards and switched to cosine LR with `LR_WARMUP_STEPS=64`, `LR_DECAY_START_FRAC=0.65`, `LR_MIN_SCALE=0.15`

## Run Chronology

| Run | Start | Eval | Key result |
| --- | --- | --- | --- |
| 5B | 2026-03-26 18:01 KST | full | post-quant `1.24582822` |
| 24h | 2026-04-08 20:21 KST | full | post-quant `1.22864374` |
| 1h | 2026-04-09 23:26 KST | proxy (4,194,304 val tokens) | post-quant `1.35156821` |

## Results Source

The values in `results.tsv` were read from these local W&B run folders:

- `wandb/run-20260326_180143-wzspgrg0`
- `wandb/run-20260408_202109-vajvzlaj`
- `wandb/run-20260409_232646-hgygstyg`

The recorded numbers come from `wandb-summary.json`, `config.yaml`, and `output.log` in those directories.

## Included Files

- `train_gpt.py`: experiment-local trainer snapshot kept with this record
- `results.tsv`: normalized run table from the local W&B artifacts
- `submission.json`: metadata for the best full-eval run

## Notes

- This folder date, `2026-04-13`, is the record write-up date, not the original training date.
- The included `train_gpt.py` is the experiment-local code snapshot kept for recordkeeping and reruns. The reported metrics themselves come from the recorded W&B runs listed above.
- The 1-hour run is intentionally marked as proxy-only because it used `VAL_MAX_TOKENS=4194304` with `LOCAL_PROXY_EVAL=1`.
