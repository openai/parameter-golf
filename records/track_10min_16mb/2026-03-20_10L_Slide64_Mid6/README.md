This record captures a valid `10-minute / 8xH100 / <16MB` submission from the 10-layer sliding-window family.

The important lesson from this run was that the strong `Muon 0.99` crossover schedule was already good enough to beat `1.1748`, but the first full-fidelity export overflowed the byte cap. A blanket `WEIGHT_BITS=6` shrink fixed size but over-compressed the model. The final valid recipe keeps the same training schedule and sliding-window eval, preserves fp16 tied embeddings, and only applies int6 export to the middle transformer blocks `3,4,5,6`.

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Schedule: `TIED_EMBED_LR=0.10 MATRIX_LR=0.02 SCALAR_LR=0.02 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 MUON_WEIGHT_DECAY=0.02`
- Init: `SPECTRAL_EMBED_INIT=1 PHASE_RESID_MIX_INIT=1`
- Eval: `EVAL_SEQ_LEN=1024 EVAL_STRIDE=64`
- Export: `WEIGHT_BITS=8 EMBED_BITS=16 BLOCK_INT6_LAYERS=3,4,5,6`

Command (track-relevant params):
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=lead_03_10l_slide64_fp16emb_muon099_mid6 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=10 UNIQUE_BLOCKS=10 UNIQUE_MLPS=10 \
TIED_EMBED_LR=0.10 MATRIX_LR=0.02 SCALAR_LR=0.02 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 MUON_WEIGHT_DECAY=0.02 \
SPECTRAL_EMBED_INIT=1 PHASE_RESID_MIX_INIT=1 \
EVAL_SEQ_LEN=1024 EVAL_STRIDE=64 VAL_BATCH_SIZE=8388608 \
WEIGHT_BITS=8 EMBED_BITS=16 BLOCK_INT6_LAYERS=3,4,5,6 \
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29500 \
  /workspace/parameter-golf/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `12422/20000` due to the 600-second wallclock cap.
- Pre-export eval at stop: `val_loss:2.0331`, `val_bpb:1.2041`
- Quantized roundtrip exact metric: `final_quant_zlib_roundtrip_exact val_bpb:1.20752367`
- Sliding-window exact metric: `final_sliding_window_exact val_bpb:1.17334285`
- Train time: `600030ms` (`step_avg:48.30ms`)
- Peak memory: `19699 MiB allocated`, `36412 MiB reserved`
- Quant bits histogram: `{'16': 524288, '8': 11010048, '6': 7340032}`
- Serialized model quant+zlib: `15794032 bytes`
- Code size: `65668 bytes`
- Total submission size quant+zlib: `15859700 bytes`

Why this variant:
- A higher-fidelity crossover run reached `1.17119058`, but was invalid at `17210659` bytes.
- An all-`w6` shrink run was valid at `13909385` bytes, but regressed to `1.17922524`.
- This middle-block int6 policy recovered most of the score while staying under the cap.

Included files:
- `train_gpt.py` (exact code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
