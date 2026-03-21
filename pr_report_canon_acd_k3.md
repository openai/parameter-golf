Submission: Int6 + Canon ACD (K=3) + Muon WD 0.04 + SWA + Sliding Eval (val_bpb=1.1668)

This PR reports a standalone run using Canon ACD (`CANON_SET=ACD`) with `CANON_KERNEL=3` and mixed int6 quantization for `mlp,attn`.

Approach summary:
- Architecture: 9-layer decoder-only Transformer, `model_dim=512`, `num_heads=8`, `num_kv_heads=4`, `MLP_MULT=3.0`.
- MLP nonlinearity: ReLU-squared style MLP as used in this repo.
- Context modules: Bigram hash embedding (`bigram_vocab_size=2048`, `bigram_dim=128`) and SmearGate.
- Quantization: mixed post-training quantization where `mlp/attn` are int6 and remaining large tensors stay int8;
- Optimizer: mixed Muon + Adam setup. Muon handles matrix-like parameters; Adam handles token/scalar/head groups.
- Schedule: momentum warmup (`0.92 -> 0.99`), warmdown tail (`WARMDOWN_ITERS=3000`), and SWA averaging near the end.
- Eval: report both final roundtrip and sliding-window eval (`EVAL_STRIDE=64`), where sliding bpb is the key comparison metric.

Canon details:
- Canon layer is a depthwise causal 1D conv with residual connection.
- `A`: Canon before attention.
- `B`: Canon on concatenated QKV stream (expensive because width is larger).
- `C`: Canon before MLP.
- `D`: Canon in the widened MLP hidden stream.
- This run uses `ACD` to keep most Canon effect while avoiding `B` compute cost.

Configuration highlights:
- 8x GPU (`torchrun --nproc_per_node=8`)
- `TRAIN_BATCH_TOKENS=524288`, `TRAIN_SEQ_LEN=2048`
- `EVAL_SEQ_LEN=2048`, `EVAL_STRIDE=64`, `EVAL_BATCH_SEQS=32`
- `INT6_CATEGORIES=mlp,attn`
- `CANON_SET=ACD`, `CANON_KERNEL=3`, `CANON_RESIDUAL=1`, `CANON_ACTIVATION=0`, `CANON_BIAS=0`
- `MATRIX_LR=0.025`, `SCALAR_LR=0.025`, `TIED_EMBED_LR=0.035`
- `MUON_MOMENTUM=0.99`, `MUON_WEIGHT_DECAY=0.04`, `ADAM_WEIGHT_DECAY=0.04`
- `SWA_ENABLED=1`, `SWA_EVERY=200`, `SWA_START_LRMUL=0.5`
- `ITERATIONS=7200`, `WARMUP_STEPS=20`, `WARMDOWN_ITERS=3000`, `MAX_WALLCLOCK_SECONDS=600`
- `VOCAB_SIZE=1024`, `SEED=1337`

Run output:
- `final_int6_sliding_window val_bpb` (stride=64): **1.16682362**
- `final_model.int6.ptz`: **13,196,032 bytes**
- Code size (`train_gpt.py`): **71,315 bytes**
- Total submission size: **13,267,347 bytes** (under 16,000,000 limit)

Notes:
- Main score to compare is sliding-window bpb: **1.16682362**.
- SWA applied with 8 checkpoints.
- Data loader overhead is low (`data_loading_step_avg=0.64ms`).
- End-of-run metrics:
  - `data_loading_total:4637ms`
  - `final_int6_sliding_window eval_time:246064ms`

Repro command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
env \
  RUN_ID=frontier_canon_acd_k3_8gpu \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 SEED=1337 \
  TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048 \
  EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
  ITERATIONS=7200 WARMUP_STEPS=20 WARMDOWN_ITERS=3000 MAX_WALLCLOCK_SECONDS=600 \
  MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
  MUON_WEIGHT_DECAY=0.04 ADAM_WEIGHT_DECAY=0.04 \
  SWA_ENABLED=1 SWA_EVERY=200 SWA_START_LRMUL=0.5 \
  INT6_CATEGORIES=mlp,attn \
  CANON_SET=ACD CANON_KERNEL=3 CANON_RESIDUAL=1 CANON_ACTIVATION=0 CANON_BIAS=0 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```
