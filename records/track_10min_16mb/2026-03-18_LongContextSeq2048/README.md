This record submission is called `Long Context Seq2048 v2`.

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Sequence length: `TRAIN_SEQ_LEN=2048`
- Batching: `TRAIN_BATCH_TOKENS=524288`
- Learning rates: `TIED_EMBED_LR=0.04 MATRIX_LR=0.032 SCALAR_LR=0.032`

Command:
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=seq2048_sxm28_full_20260319a \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-18_LongContextSeq2048/train_gpt.py
```

Verification environment:
- `8x H100 80GB HBM3`
- all-to-all `NV18` topology
- `torch 2.8.0+cu128`

Key metrics (from `train.log` in this folder, rerun on the target SXM-class box):
- Timed training stopped at `11564/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0269`, `val_bpb:1.2005`
- Post-quant roundtrip eval: `val_loss:2.0359`, `val_bpb:1.2058`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.20576485`
- Train time: `600038ms` (`step_avg:51.89ms`)
- Peak memory: `10247 MiB allocated`, `10488 MiB reserved`
- Serialized model int8+zlib: `15819554 bytes`
- Code size for this standalone record script: `47716 bytes`
- Total submission size int8+zlib: `15867270 bytes`

Additional full-run reproducibility logs included in this folder:
- `train.log`: canonical SXM rerun, `SEED=1337`, `val_bpb=1.20576485`
- `train_seed1338.log`: SXM rerun, `SEED=1338`, `val_bpb=1.20617460`
- `train_seed1339.log`: SXM rerun, `SEED=1339`, `val_bpb=1.20715923`

Record-track significance note:
- The public repo state for this submission has `Naive Baseline` at `1.2243657`.
- The challenge therefore requires beating `1.2193657` to claim a new record.
- All three included SXM full runs clear that threshold:
  - `SEED=1337`: `1.20576485`
  - `SEED=1338`: `1.20617460`
  - `SEED=1339`: `1.20715923`
- Sample mean across the three runs: `1.20636623`
- Sample standard deviation: `0.00071667`
- One-sided one-sample t-test against `1.2193657`: `t=31.42` with `df=2`, which gives `p=0.00051`

Why this folder is standalone:
- `train_gpt.py` compiles from inside this record folder and was used for the canonical rerun whose output is saved as `train.log`.
- No extra Python source files are required for the training path.
- The only inputs expected at runtime are the cached dataset and tokenizer paths described in the main repo README.

Included files:
- `train_gpt.py` (standalone winning recipe with defaults baked in)
- `README.md` (this file)
- `submission.json` (leaderboard metadata)
- `train.log` (canonical full log from the standalone record script)
- `train_seed1338.log`, `train_seed1339.log` (extra full reruns for reproducibility)
- `logs/seq2048_sxm28_*` (raw per-run tee output and trainer text logs from the SXM verification box)
