# SOTA + Seq4096

This record combines the current SOTA recipe with seq_len=4096 training.

## Key Techniques

1. **SOTA architecture and eval stack**: 10 layers at width 512, sliding-window evaluation (`EVAL_STRIDE=64`), FP16 tied embedding export, overtone embedding initialization, phase-transition residual mixing, and NTK-aware RoPE scaling for longer eval contexts.

2. **Long-context training**: `TRAIN_SEQ_LEN=4096` with `TRAIN_BATCH_TOKENS=393216` (3/4 of baseline batch) to increase useful context per update.

3. **Seq4096 optimizer profile**: lower LRs (`MATRIX_LR=0.020`, `SCALAR_LR=0.020`, `TIED_EMBED_LR=0.040`) with higher Muon momentum (`0.99`) and extended momentum warmup.

4. **Decoupled Muon weight decay (configurable)**: `MUON_WEIGHT_DECAY` applies post-step decay on Muon matrix params (default `0.02`).

5. **Mixed-bit export support for deeper models**: optional low-bit rounding on selected block weights at export time using `LOWBIT_LAYERS` + `LOWBIT_STEP` (for example `LOWBIT_STEP=4` gives int6-like quantization levels on those layers).

## Configuration (Default)

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Training context: `TRAIN_SEQ_LEN=4096`
- Batching: `TRAIN_BATCH_TOKENS=393216`
- Eval: `EVAL_STRIDE=64 EVAL_SEQ_LEN=0` (0 means same as train seq len)
- Optimizer: `MATRIX_LR=0.020 SCALAR_LR=0.020 TIED_EMBED_LR=0.040 MUON_MOMENTUM=0.99 MUON_WEIGHT_DECAY=0.02`
- Export precision defaults: `LOWBIT_STEP=1` (disabled unless set > 1)

## Command

```bash
RUN_ID=sota_seq4096_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_SOTA_Seq4096/train_gpt.py
```

### Example: 11-layer mixed-bit variant

```bash
RUN_ID=sota_seq4096_11l_lowbit \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
LOWBIT_LAYERS=3,4,5,6,7 \
LOWBIT_STEP=4 \
MUON_WEIGHT_DECAY=0.02 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_SOTA_Seq4096/train_gpt.py
```

## Results

Pending run logs. Fill this section with:
- `final_int8_zlib_roundtrip_exact val_loss`
- `final_int8_zlib_roundtrip_exact val_bpb`
- final artifact bytes (`code + final_model.int8.ptz`)
- step count and average step time

## Included Files

- `train_gpt.py`
- `README.md`
