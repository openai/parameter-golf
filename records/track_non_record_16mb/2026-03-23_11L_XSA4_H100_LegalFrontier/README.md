# Non-Record: 11L + XSA4 H100 Frontier

This PR adds a non-record `16 MB` submission built around an `11`-layer decoder-only transformer with `XSA` applied only in the final `4` layers. The model uses width `512`, `8` query heads, `4` KV heads, tied embeddings, ReLU^2 MLPs, `TRAIN_SEQ_LEN=256`, `TRAIN_BATCH_TOKENS=524288`, `WARMDOWN_ITERS=200`, and checkpoint-frontier saving every `25` steps. Artifacts use custom packed serialization with `packed_zstd`.

This is not a record-track claim. It was developed and validated on single-`H100 80GB` hardware and is submitted as a reproducible non-record technical result.

## Official Legal Result

The official metric in `submission.json` is the best legal checkpoint from a `650`-step H100 frontier:

- checkpoint: `logs/checkpoints/11l_xsa4_h100_scored_step00650.pt`
- selection policy: `--no-default-large-keeps`
- `val_loss: 2.46718130`
- `val_bpb: 1.46120374`
- compressed model bytes: `15,907,290`
- code bytes: `76,313`
- total bytes: `15,983,603`

## Stronger Full-Data Frontier

I also ran the same recipe on the full cached challenge shard set on single H100 hardware. The strongest raw point from that run was:

- `950` steps
- final roundtrip exact `val_bpb: 1.41212874`
- total submission bytes: `17,636,401`

This result is materially stronger in BPB but over the `16,000,000` byte cap, so it is included here as development evidence rather than as the official submission metric.

## Training Command

```bash
RUN_ID=11l_xsa4_h100_scored \
NUM_LAYERS=11 \
XSA_TAIL_LAYERS=4 \
TRAIN_SEQ_LEN=256 \
EVAL_SEQ_LEN=256 \
EVAL_STRIDE=256 \
EVAL_BATCH_SEQS=32 \
TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_ITERS=200 \
ITERATIONS=650 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=0 \
RUN_TTT_EVAL=0 \
SAVE_DENSE_CHECKPOINT_EVERY=25 \
MAX_WALLCLOCK_SECONDS=0 \
python train_gpt.py
```

Checkpoint selection:

```bash
python scripts/eval_quant_candidate.py \
  --state-dict-path logs/checkpoints/11l_xsa4_h100_scored_step00650.pt \
  --no-default-large-keeps \
  --val-batch-size 524288 \
  --num-layers 11 \
  --xsa-tail-layers 4 \
  --train-seq-len 256
```

## Notes

- `train_gpt.py` is the exact code snapshot used for this submission.
- `train.log` contains the corresponding H100 run log for the official legal result.
- The main remaining bottleneck is artifact size, not raw model quality.
- The next obvious follow-up is to combine this recipe with stronger artifact economics or post-quant-aware late training so later checkpoints remain legal.
