# Non-Record Submission: Full80 Recurrence + Step Embed + Seq1536 (20k)

This is an unlimited-compute non-record submission for the 16MB track.

It documents a recurrence-based continuation of the `full80` mixed-lowbit family with:
- fixed middle-block recurrence
- learned step-aware recurrence embeddings
- `TRAIN_SEQ_LEN=1536`
- export rescue with `BIGRAM_EXPORT_BITS=4`
- export rescue with `MAG_PRUNE_PCT=0.033`

This run is not intended to satisfy the 10-minute `8xH100` record-track requirement. It was developed and evaluated as a longer local experiment and is submitted as a non-record result because the architecture and training recipe are meaningfully different from the stock baseline while remaining under the `16,000,000` byte artifact cap.

## Result

- Final roundtrip: `val_loss=1.98616148`, `val_bpb=1.17631839`
- Pre-export quant proxy at stop: `val_loss=1.9815`, `val_bpb=1.1735`
- Artifact size: `15,501,266` bytes
- Compressed model size: `15,402,045` bytes
- Code size: `99,221` bytes
- Step stop: `20,000`
- Wallclock: `39,715.675s`
- GPU: `1xRTX5060`

## Why This Submission Is Interesting

- It turns depth recurrence into a competitive under-cap result rather than just a parameter-sharing curiosity.
- The added step-aware recurrence embedding was important; earlier recurrence variants were promising before export but lost too much on final roundtrip.
- A small export rescue (`BIGRAM_EXPORT_BITS=4`, `MAG_PRUNE_PCT=0.033`) kept the recurrence model under the hard cap without giving up most of the gain.
- The final result beats the older local mixed-lowbit baseline family while staying under the same artifact budget.

## Configuration Summary

- Layout: `NUM_LAYERS=10 MODEL_DIM=576 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3.0`
- Recurrence: `RECURRENT_MODE=fixed RECURRENT_CORE_START=3 RECURRENT_CORE_LEN=2 RECURRENT_STEPS=2`
- Recurrence step embedding: `RECURRENT_STEP_EMBED=1 RECURRENT_STEP_EMBED_INIT_STD=0.01`
- Sequence length: `TRAIN_SEQ_LEN=1536`
- Batch tokens: `TRAIN_BATCH_TOKENS=122880`
- Export rescue: `BIGRAM_EXPORT_BITS=4 MAG_PRUNE_PCT=0.033`
- Continuation: resumed from the saved `10k` checkpoint and extended to `20k`

## Included Files

- `train_gpt.py` — code snapshot used for the submitted run
- `train.log` — training and final evaluation log
- `submission.json` — metadata for the non-record submission
- `requirements.txt` — dependency reference
