This record captures the best valid run from the 10-layer sliding-window family so far.

It keeps the stronger Muon crossover schedule and the valid mixed-precision export policy from the prior `mid6` submission, then makes one small training-side adjustment: `TIED_EMBED_LR=0.08` instead of `0.10`.

## Result

- `final_sliding_window_exact val_bpb: 1.17319477`
- `final_quant_zlib_roundtrip_exact val_bpb: 1.20735381`
- `Total submission size quant+zlib: 15846677 bytes`

Compared with the previous valid `mid6` run:
- old `final_sliding_window_exact val_bpb: 1.17334285`
- new `final_sliding_window_exact val_bpb: 1.17319477`
- improvement: `-0.00014808`

## What Changed

This run keeps the same high-level recipe:
- `10` layers at `512` dim with `8` attention heads and `4` KV heads
- tied embeddings
- sliding-window eval at `EVAL_SEQ_LEN=1024`, `EVAL_STRIDE=64`
- fp16 passthrough for the tied embedding/output-head tensor
- int8 by default, with only middle blocks `3,4,5,6` forced to int6
- Muon crossover schedule:
  - `MATRIX_LR=0.02`
  - `SCALAR_LR=0.02`
  - `MUON_MOMENTUM=0.99`
  - `MUON_MOMENTUM_WARMUP_START=0.92`
  - `MUON_MOMENTUM_WARMUP_STEPS=1500`
  - `WARMDOWN_ITERS=3000`
  - `MUON_WEIGHT_DECAY=0.02`
- `SPECTRAL_EMBED_INIT=1`
- `PHASE_RESID_MIX_INIT=1`

The only training change relative to the previous submission is:
- `TIED_EMBED_LR=0.08` instead of `0.10`

## Why This Variant

The prior valid `mid6` run already showed that:
- the crossover schedule was strong enough to beat `1.1748`
- the bottleneck was keeping the artifact under `16,000,000` bytes

That run was valid and strong, but there was still room for a small training-side improvement without changing eval semantics or risking size overflow. Reducing the tied-embedding LR slightly improved the sliding exact score while also shrinking the final compressed artifact by about `13 KB`.

## Command

```bash
NCCL_IB_DISABLE=1 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=10 \
UNIQUE_BLOCKS=10 \
UNIQUE_MLPS=10 \
TIED_EMBED_LR=0.08 \
EVAL_SEQ_LEN=1024 \
EVAL_STRIDE=64 \
VAL_BATCH_SIZE=8388608 \
WEIGHT_BITS=8 \
EMBED_BITS=16 \
LM_HEAD_BITS=16 \
BLOCK_INT6_LAYERS=3,4,5,6 \
MUON_WEIGHT_DECAY=0.02 \
SPECTRAL_EMBED_INIT=1 \
PHASE_RESID_MIX_INIT=1 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=600 \
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 train_gpt.py
```

## Key Metrics

- Timed training stopped at `12428/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0328`, `val_bpb:1.2039`
- Post-quant roundtrip eval: `val_loss:2.03856633`, `val_bpb:1.20735381`
- Sliding exact eval: `val_loss:1.98089037`, `val_bpb:1.17319477`
- Train time: `600018ms` (`step_avg:48.28ms`)
- Peak memory: `19698 MiB allocated`, `36280 MiB reserved`
- Serialized model quant+zlib: `15781009 bytes`
- Code size: `65668 bytes`
- Total submission size quant+zlib: `15846677 bytes`

## Training Volume

- Global batch: `524288` tokens/step
- Total train tokens seen: `6515851264`

## Included Files

- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
