This folder captures a non-record submission built from a baseline-adjacent mixed-precision export sweep.

This run is not intended to claim a new main-track record. It was selected as the strongest result from our 1xH200, 10-minute screening sweep, where we compared clean baseline, fp16 embedding passthrough, sink tokens, and multiple MTP variants under the same wallclock cap and 16MB artifact constraint.

Why this submission:
- It is the best completed run from our H200 screening sweep.
- It stays under the `16,000,000` byte artifact cap.
- It provides a clean negative-result package for the approaches that did not beat it, especially fixed MTP and curriculum MTP.

Configuration:
- Track: `non-record`, still under the `16,000,000` byte artifact cap
- Hardware: `1xH200` RunPod Parameter Golf template
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Optimizer LRs: `TIED_EMBED_LR=0.05 MATRIX_LR=0.04 SCALAR_LR=0.04`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Export tweak: `INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight`
- Wallclock cap: `MAX_WALLCLOCK_SECONDS=600`

Command (track-relevant params):
```bash
RUN_ID=runpod_baseline_fp16_embed \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `1503/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.2292`, `val_bpb:1.3202`
- Post-quant roundtrip eval: `val_loss:2.2301`, `val_bpb:1.3208`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.32078403`
- Train time: `600243ms` (`step_avg:399.36ms`)
- Peak memory: `10239 MiB allocated`, `10850 MiB reserved`
- Serialized model int8+zlib: `14269846 bytes`
- Code size: `57289 bytes`
- Total submission size int8+zlib: `14327135 bytes`

Sweep context:

| Variant | Exact final val_bpb | Notes |
|---|---:|---|
| Clean baseline | `1.32171904` | Reference 1xH200 run |
| FP16 embed passthrough | `1.32078403` | Best completed branch |
| Sink4 | `1.32107101` | Small positive signal, weaker than fp16 embed |
| MTP1 fixed | `1.33126842` | Worse than baseline |
| MTP2 fixed | `1.32792538` | Better than MTP1, still worse than baseline |
| FP16 embed + MTP2 fixed | `1.32718519` | Best MTP branch, still behind fp16 embed alone |

Interpretation:
- The strongest result in this sweep came from a very small exporter-side change rather than a new training objective.
- Fixed MTP variants were stable after code fixes, but did not outperform the baseline-family fp16 embedding passthrough configuration in this budget.
- This suggests our next serious step should be validating the baseline-adjacent fp16 embedding path on `8xH100`, rather than moving more complexity into the training objective.

Included files:
- `train_gpt.py` (exact code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (metadata for this non-record submission)
