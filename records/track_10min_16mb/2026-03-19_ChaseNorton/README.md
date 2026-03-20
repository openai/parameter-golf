# MLP3x + Int8 Tok Emb + Grouped LZMA + Sliding Window

This submission starts from the public 10-minute baseline family and makes three main changes:

1. Increase feedforward capacity from `2x` to `3x`.
2. Train and evaluate at `seq_len=2048`.
3. Use a custom grouped low-bit export plus sliding-window evaluation to improve the final under-cap score.

The model was trained for the official `600s` wallclock limit on `8x H100 SXM`, then repacked into a submission-valid artifact under the `16,000,000` byte limit.

## Model

- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=3`
- `TIE_EMBEDDINGS=1`
- RoPE + RMSNorm + tied input/output embeddings
- U-Net-style skip structure inherited from the baseline

This keeps the backbone close to the baseline while spending more of the parameter budget on the MLP.

## Training Setup

The timed run used:

- `TRAIN_BATCH_TOKENS=786432`
- `TRAIN_SEQ_LEN=2048`
- `ITERATIONS=20000`
- `MAX_WALLCLOCK_SECONDS=600`
- `WARMUP_STEPS=20`
- `WARMDOWN_ITERS=3000`
- `TIED_EMBED_LR=0.03`
- `MATRIX_LR=0.02`
- `SCALAR_LR=0.02`
- `MUON_MOMENTUM=0.99`

Logged optimizer summary:

- `tie_embeddings:True embed_lr:0.03 head_lr:0.0 matrix_lr:0.02 scalar_lr:0.02`

Logged attention summary:

- `attention_mode:gqa num_heads:8 num_kv_heads:4`

The script includes QAT support, but this specific timed run stopped before QAT activation:

- `qat_enabled:True`
- `qat_start_frac:0.500`
- `qat_start_step:10000`
- actual stop step: `7534`

So the final reported result is from post-training repacking of the timed checkpoint rather than from a checkpoint that had entered the QAT phase.

## Timed Training Result

The official training run stopped at the wallclock cap:

- `step:7534/20000`
- `train_time:600120ms`
- `step_avg:79.65ms`

Validation at stop:

- `val_loss=1.9844`
- `val_bpb=1.1753`

Other logged details:

- peak memory allocated: `16738 MiB`
- peak memory reserved: `16944 MiB`
- raw checkpoint size: `86099351` bytes

## Export / Compression

The first export evaluated from the timed run used:

- int6 for most tensors
- `fp16` token embedding passthrough
- `fp16` passthrough for the last two `c_k` weights
- `zlib`

That version evaluated very well but was not submission-valid because it was over the size cap:

- total size: `16639274` bytes
- standard post-export score: `1.18101095`
- sliding-window score: `1.16018011`

The final submission-valid repack uses:

- grouped `QGv3` serialization
- `lzma` compression
- int6 for most tensors
- int8 for `tok_emb.weight`
- no `fp16` passthrough tensors

This change was enough to get back under the limit while preserving almost all of the sliding-window gain.

## Final Submission Artifact

- artifact path: `final_model.mixed_tok8_lzma.ptz`
- compressor: `lzma`
- model bytes: `15845980`
- code bytes: `64924`
- total bytes: `15910904`

This is submission-valid under the `16,000,000` byte cap.

## Final Scores

Exact post-pack scores for the final under-cap artifact:

- standard eval:
  - `val_loss=1.99867543`
  - `val_bpb=1.18372817`

- sliding-window eval with `seq_len=2048`, `stride=256`:
  - `val_loss=1.96250243`
  - `val_bpb=1.16230441`

The submission score is therefore:

## `1.16230441 val_bpb`

## Notes

- The largest quality-preserving export change was keeping the token embedding at `8-bit` while quantizing the rest of the model to `6-bit`.
- Grouped serialization reduced overhead compared with a naive `torch.save` artifact.
- Fresh-load evaluation required keeping low-dimensional floating buffers, including RoPE frequency buffers, in `fp32` to match the trained checkpoint behavior after reload.

## Included Files

- `README.md` - this writeup
- `submission.json` - submission metadata
- `train.log` - exact log from the timed `8x H100 SXM` run
- `train_gpt.py` - code snapshot for the submission artifact and evaluation path
