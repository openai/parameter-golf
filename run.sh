source .venv/bin/activate

RUN_ID=SEED_42 \
SEED=42 \
VOCAB_SIZE=8192 \
CASEOPS_ENABLED=1 \
DATA_DIR=./datasets \
PPM_MIX_ENABLED=1 \
TOKENIZER_PATH=${TOKENIZER_PATH:-./datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model} \
DATA_PATH=${DATA_PATH:-./datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved} \
TRAIN_LOOP_PHASE_DEPTHS=1,3,4 \
TRAIN_LOOP_PREWARM_DEPTHS=3,4 \
COMPRESSOR=lzma \
MAX_WALLCLOCK_SECONDS=588 \
torchrun --nproc_per_node=8 records/track_10min_16mb/record3/train_gpt.py