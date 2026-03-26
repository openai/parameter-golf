This record submission is called `Training Opt Seq4096 v1`.

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Sequence length: `TRAIN_SEQ_LEN=4096`
- Batching: `TRAIN_BATCH_TOKENS=393216` (3/4 batch)
- Learning rates: `TIED_EMBED_LR=0.030 MATRIX_LR=0.020 SCALAR_LR=0.020`
- Muon optimizer: `MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_MOMENTUM_WARMUP_START=0.92`
- Schedule: `WARMDOWN_ITERS=3000`

Command:
```bash
RUN_ID=training_opt_seq4096_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py
```

Key metrics (from the standalone record run):
- Timed training stopped at `8394/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0227`, `val_bpb:1.1980`
- Post-quant roundtrip eval: `val_loss:2.0286`, `val_bpb:1.2014`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.20143417`
- Train time: `599921ms` (`step_avg:71.47ms`)
- Peak memory: `7748 MiB allocated`, `8070 MiB reserved`
- Serialized model int8+zlib: `15820684 bytes`
- Code size for this standalone record script: `47759 bytes`
- Total submission size int8+zlib: `15868326 bytes`

Approach:
This submission combines two independent improvements over the naive baseline:

1. **Longer training context (seq_len=4096):** Each training sequence sees 4x more context than the 1024-token baseline, giving the autoregressive model much better signal per token. This costs ~71ms/step (vs ~43ms at seq_len=1024), but the quality improvement far outweighs the fewer total steps.

2. **Aggressive Muon optimizer tuning:**
   - **Higher momentum (0.99 vs 0.95):** Provides stronger gradient smoothing, leading to better convergence.
   - **Lower learning rates (0.020 vs 0.04):** Dramatically reduces int8 quantization loss (0.0034 BPB quant penalty vs 0.007+ at default LR) while maintaining similar pre-quant quality.
   - **3/4 batch (393K vs 524K tokens):** More optimizer updates per wallclock second.
   - **Extended momentum warmup (1500 steps from 0.92):** Prevents early instability with the higher momentum.
   - **Longer warmdown (3000 steps):** Proportionally longer LR decay for the ~8400-step run.

The net effect is a **0.023 BPB improvement** over the naive baseline (1.2014 vs 1.2244), and a **0.015 BPB improvement** over the previous best entry (Long Context Seq2048 v2 at 1.2162).

Additional full-run reproducibility logs included in this folder:
- `train.log`: canonical standalone run, `SEED=1337`, `val_bpb=1.20143417`
- `train_seed1338.log`: full rerun, `SEED=1338`, `val_bpb=1.19945102`
- `train_seed1339.log`: full rerun, `SEED=1339`, `val_bpb=1.20319508`

Record-track significance note:
- The current SOTA is `Long Context Seq2048 v2` at `1.21613611`.
- The challenge requires beating `1.21113611` (SOTA - 0.005) at p < 0.01.
- All three included full runs clear that threshold:
  - `SEED=1337`: `1.20143417`
  - `SEED=1338`: `1.19945102`
  - `SEED=1339`: `1.20319508`
- Sample mean across the three runs: `1.20136009`
- Sample standard deviation: `0.00187`
- One-sided one-sample t-test against `1.21113611`: `t=9.06` with `df=2`, which gives `p=0.006`

Hardware: 8x NVIDIA H100 80GB HBM3 (SXM, NVLink NV18 all-to-all), PyTorch 2.8.0+cu128.

Why this folder is standalone:
- `train_gpt.py` compiles from inside this record folder and was used for the canonical run whose output is saved as `train.log`.
- No extra Python source files are required for the training path.
- The only inputs expected at runtime are the cached dataset and tokenizer paths described in the main repo README.

Included files:
- `train_gpt.py` (standalone winning recipe with defaults baked in)
- `README.md` (this file)
- `submission.json` (leaderboard metadata)
- `train.log` (canonical full log from the standalone record script)
- `train_seed1338.log`, `train_seed1339.log` (extra full reruns for reproducibility)
