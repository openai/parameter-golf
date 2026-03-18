This record captures an improved baseline for the Parameter Golf challenge.

Key architectural changes from the naive baseline:
- **Depth recurrence**: 5 unique transformer blocks looped 2x = 10 effective layers (vs baseline's 9 unique layers). Weight sharing reduces stored parameter count while maintaining effective model depth.
- **Wider model**: MODEL_DIM=704 (vs baseline 512), enabled by the parameter savings from weight sharing. Gives 88-dim attention heads vs 64.
- **Per-recurrence learned scales**: Each effective layer gets a learned scale vector so shared blocks can distinguish which recurrence pass they're on, allowing the same block to behave differently in encoder vs decoder roles.
- **Encoder-decoder skip connections**: Maintained from baseline, adapted for recurrent depth.

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_UNIQUE_LAYERS=5 NUM_RECURRENCE=2 MODEL_DIM=704 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Effective depth: 10 layers (5 unique x 2 recurrence passes)
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Command (track-relevant params):
```bash
RUN_ID=improved_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Design rationale:
- Depth recurrence with 5 unique blocks x 2 passes keeps per-step compute at ~2.1x baseline (est ~91ms/step vs ~44ms), allowing ~6500 steps / ~3.4B tokens in 10 minutes.
- The wider dimension (704 vs 512) improves per-head representational capacity.
- Learned per-recurrence scales (10 x 704 params) let the model route information differently on each pass through the shared weights — critical for encoder/decoder role differentiation.

Estimated model size:
- ~18.1M parameters (unique)
- Estimated int8+zlib: ~8.9 MB
- Estimated code: ~44.3 KB
- Estimated total: ~9.0 MB (under 16 MB cap)

Included files:
- `train_gpt.py` (code snapshot for the run)
- `submission.json` (leaderboard metadata)
- `README.md` (this file)
