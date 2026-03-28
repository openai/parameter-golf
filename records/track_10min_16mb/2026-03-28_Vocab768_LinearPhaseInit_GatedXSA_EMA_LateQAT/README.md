# 2026-03-28 Vocab768 LinearPhaseInit GatedXSA EMA LateQAT

This record captures the native `train_gpt_v4.py` implementation that paired the
custom `sp768` tokenizer export with the strongest feature combination we found in
the clean fast branch:

- SentencePiece BPE tokenizer with `vocab_size=768`
- linear phase-mix initialization (`ENABLE_PHASE_MIX_INIT=1`)
- gated XSA on the last 2 layers
- EMA during late training
- late QAT with matrix-only fake quant
- FlashAttention 3 backend

## Command

```bash
RUN_ID=v4_vocab768_b768_9x_ctx2048 \
VOCAB_SIZE=768 \
ENABLE_PHASE_MIX_INIT=1 \
XSA_LAST_N=2 \
XSA_MODE=gated \
XSA_ALPHA_INIT=0.1 \
LATE_QAT_THRESHOLD=0.15 \
QAT_SCOPE=matrix_only \
NUM_LAYERS=9 \
MLP_MULT=3 \
EMA_ENABLED=1 \
VALUE_BIAS_LAST_N=2 \
VALUE_BIAS_DIM=128 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=786432 \
VAL_BATCH_SIZE=524288 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Tokenizer / Data

- tokenizer path: `./data/tokenizers/fineweb_768_bpe.model`
- dataset path: `./data/datasets/fineweb10B_sp768`
- train shards: `80`
- validation tokens: `67,778,560`

This tokenizer was rebuilt locally from the published docs cache, capped to the
challenge-style 10B train-token budget, uploaded to Hugging Face, and then
downloaded through the patched manifest-driven loader.

## Run Summary

- model params: `21,778,507`
- world size: `8`
- grad accumulation steps: `1`
- attention backend: `flash_attn_3`
- average step time near convergence: about `64.3 ms`
- EMA start: `step 9094`
- late QAT enable: `step 9132` at `scale 0.1494`
- early stop on wallclock cap: `step 9252`

## Key Metrics

- in-run validation: `val_loss=1.8570`, `val_bpb=1.2019`
- final int6 roundtrip: `val_loss=1.8718`, `val_bpb=1.2115`
- peak memory allocated: `16864 MiB`
- peak memory reserved: `17572 MiB`

## Serialization

- raw model: `86,166,556 bytes`
- code size: `61,461 bytes`
- total submission size: `86,228,017 bytes`
- compressed int6+zstd artifact: `15,021,344 bytes`
- total submission size int6+zstd: `15,082,805 bytes`

This is the first tokenizer-change run in this branch that clearly beat the
previous `sp1024` results while also coming in under the 16 MB compressed
submission limit.
