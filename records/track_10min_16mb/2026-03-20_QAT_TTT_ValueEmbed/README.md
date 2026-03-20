Adds QAT, value embeddings, test-time LoRA, SWA, and temperature search on top of established techniques (10L, sliding window, fp16 embed, Muon WD, spectral init). No GPU runs yet - submitting as non-record pending compute.

## what changed

**QAT**: fake int8 per-row quantization baked into the forward pass with straight-through estimator. Kicks in at 15% of training. The model sees quantization noise during training instead of getting hit with it at export time.

**value embeddings**: 3 learned lookup tables (1024 x 256) shared across layers in a U-net pattern. Indexed by input tokens and added into attention values with sigmoid gates. Basically free - it's just an embedding lookup.

**test-time LoRA**: after quantization, rank-4 LoRA adapters go into all projections (Q/K/V/out + MLP). Model processes val data sequentially, one Adam step per chunk. Then re-eval with sliding window on the adapted model.

**SWA**: average checkpoints during warmdown. Flatter minimum, compresses better.

**temperature search**: sweep [0.92-1.08] on final logits after everything else.

## configuration

```
VOCAB_SIZE=1024  NUM_LAYERS=10  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_MULT=2  TIE_EMBEDDINGS=1  EVAL_STRIDE=64  NUM_VE_TABLES=3
```

## command

```bash
RUN_ID=qat_ttt_ve \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=10 \
EVAL_STRIDE=64 \
QAT_ENABLED=1 \
SWA_ENABLED=1 \
TTT_ENABLED=1 \
EVAL_TEMP_SEARCH=1 \
NUM_VE_TABLES=3 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## results

Pending. Need compute to run + ablate.

## files

- `train_gpt.py`
- `submission.json`
