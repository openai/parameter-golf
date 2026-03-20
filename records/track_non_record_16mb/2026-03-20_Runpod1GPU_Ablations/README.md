This folder checkpoints the first Runpod exploratory ablations run on March 20, 2026.

These were not leaderboard-attempt runs. They used `1x H100`, `TRAIN_SHARDS=1`, and a strict `600s` wallclock cap to de-risk code changes and compare low-cost ablations before spending 8-GPU budget.

Best observations:
- Best roundtrip score: `twice_eval2048` at `final_int8_zlib_roundtrip_exact val_bpb: 1.39909070`
- Best LoRA-TTT score: `twice_eval2048_ttt1024` at `final_int8_ttt_lora val_bpb: 1.3960`
- Best balanced profile so far: `twice_eval2048_ttt1024`

Key takeaways:
- `ATTN_TWICE_ALPHA=0.05` helped versus baseline.
- `Z_LOSS_COEF=0.0001` regressed both roundtrip and TTT metrics.
- `EVAL_SEQ_LEN=2048` improved roundtrip scoring but hurt TTT when `TTT_EVAL_SEQ_LEN` also increased.
- Splitting eval settings (`EVAL_SEQ_LEN=2048`, `TTT_EVAL_SEQ_LEN=1024`) recovered the best TTT result seen so far.

Best current training command:
```bash
RUN_ID=twice_eval2048_ttt1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=1000 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
NUM_LAYERS=10 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=2500 \
MUON_MOMENTUM=0.95 \
MUON_BACKEND_STEPS=5 \
MUON_WEIGHT_DECAY=0.02 \
OVERTONE_INIT_POWER=0.5 \
RESID_MIX_INIT_SCALE=3.0 \
Z_LOSS_COEF=0.0 \
ATTN_TWICE_ALPHA=0.05 \
TTT_LORA_RANK=8 \
TTT_LORA_LR=0.01 \
TTT_CHUNK_SIZE=256 \
TTT_EVAL_SEQ_LEN=1024 \
TTT_BATCH_SIZE=64 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Checkpoint note:
- The raw best checkpoint still lives on the Runpod pod at `/workspace/parameter-golf/final_model.pt`.
- Size on pod: `72M`
- SHA256: `292d79fa54a638be348354f09d185f80b69710e7de8f4dfa42b36e43afccdc96`
- Runpod's SSH wrapper blocked automated binary transfer, so this repo checkpoint stores the exact metrics and checkpoint manifest rather than the raw `.pt` file.

Included files:
- `train_gpt.py`
- `submission.json`
- `results.json`
- `base10l.tail.txt`
- `zloss_low.tail.txt`
- `twice_low.tail.txt`
- `twice_eval2048.tail.txt`
- `twice_eval2048_ttt1024.tail.txt`
