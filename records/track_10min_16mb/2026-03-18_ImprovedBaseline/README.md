This record captures an improved baseline for the Parameter Golf challenge.

Key architectural changes from the naive baseline:
- **Depth recurrence**: 6 unique transformer blocks looped 2x = 12 effective layers (vs baseline's 9 unique layers). This dramatically reduces unique parameter count while maintaining effective model depth.
- **SwiGLU MLP**: Replaces ReLU^2 with SwiGLU activation (gate + fc + proj). More parameter-efficient per quality unit despite having 3 projections vs 2.
- **Wider model**: MODEL_DIM=640 (vs baseline 512), enabled by the parameter savings from weight sharing.
- **QAT noise injection**: Simulated int8 quantization noise during the second half of training to reduce post-quantization BPB degradation.
- **Encoder-decoder skip connections**: Maintained from baseline, adapted for recurrent depth.

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_UNIQUE_LAYERS=6 NUM_RECURRENCE=2 MODEL_DIM=640 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Effective depth: 12 layers (6 unique x 2 recurrence passes)
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- QAT noise: enabled, starts at 50% of training

Command (track-relevant params):
```bash
RUN_ID=improved_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_UNIQUE_LAYERS=6 \
NUM_RECURRENCE=2 \
MODEL_DIM=640 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Design rationale:
- Depth recurrence is the key insight: by sharing weights across 2 passes, we fit ~22.8M effective parameters worth of computation into ~22.8M unique parameters, but the model behaves like a 12-layer network. The weight sharing acts as a strong regularizer.
- SwiGLU was chosen over ReLU^2 because it consistently outperforms at equal parameter count in the literature.
- The wider dimension (640 vs 512) gives each attention head dim=80 (vs 64), improving per-head capacity.
- QAT noise trains the model to be robust to int8 quantization, reducing the typical ~0.007 BPB post-quant degradation.

Estimated model size:
- ~22.8M parameters
- Estimated int8+zlib: ~15.5 MB
- Estimated code: ~45.8 KB
- Estimated total: ~15.6 MB (under 16 MB cap)

Included files:
- `train_gpt.py` (code snapshot for the run)
- `submission.json` (leaderboard metadata)
- `README.md` (this file)
