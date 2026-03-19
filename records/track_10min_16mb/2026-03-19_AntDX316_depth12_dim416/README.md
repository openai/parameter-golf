This record captures a small-tweak `Parameter Golf` run based on the simple baseline.

Trainer changes in this snapshot:
- current repository `train_gpt.py` snapshot copied into the record folder
- 10-minute wallclock cap on `8xH100`
- final metric taken from the exact printed `final_int8_zlib_roundtrip_exact` line
- architecture changed from the naive baseline to a deeper/narrower layout

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=12 MODEL_DIM=416 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Command (track-relevant params):

RUN_ID=antdx316_depth12_dim416_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=12 \
MODEL_DIM=416 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=20000 \
WARMUP_STEPS=20 \
WARMDOWN_ITERS=1200 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nnodes=1 --nproc_per_node=8 train_gpt.py

Key metrics (from train.log):
- Timed training stopped at 9067/20000 steps due to the wallclock cap.
- Pre-quant eval at stop: val_loss:2.2714, val_bpb:1.3453
- Post-quant roundtrip eval: val_loss:2.2810, val_bpb:1.3509
- Exact printed metric: final_int8_zlib_roundtrip_exact val_bpb:1.35091763
- Exact printed val loss: final_int8_zlib_roundtrip_exact val_loss:2.28096783
- Train time: 600045ms (step_avg:66.18ms)
- Peak memory: 14086 MiB allocated, 14998 MiB reserved
- Serialized model int8+zlib: 14249706 bytes
- Code size: 51856 bytes
- Total submission size int8+zlib: 14301562 bytes

Training volume:
- Global batch: 524288 tokens/step
- Total train tokens seen: 4753723392

Included files:
- train_gpt.py (code snapshot used for the run)
- train.log (exact training log)
- submission.json (leaderboard metadata)
