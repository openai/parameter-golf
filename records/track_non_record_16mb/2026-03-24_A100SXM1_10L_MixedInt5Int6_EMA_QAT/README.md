# Non-Record Submission: 10L + Mixed Int5/Int6 + EMA + QAT(0.15) on 1xA100 SXM

This is a **non-record** submission on **1xA100 SXM** using the current `train_gpt.py` in this repository, with:

- 10 transformer layers (`NUM_LAYERS=10`)
- 3x MLP expansion (`MLP_MULT=3`)
- mixed quantization enabled (`MIXED_QUANT_ENABLED=1`): int5 for MLP weights, int6 for attention/bigram-sensitive weights
- EMA enabled for export-time weights (`EMA_ENABLED=1`, `EMA_DECAY=0.9999`)
- final-fraction QAT (`QAT_ENABLED=1`, `QAT_FINAL_FRAC=0.15`)
- BigramHash enabled (`BIGRAM_VOCAB_SIZE=10240`, `BIGRAM_DIM=128`)

## Run Command

```bash
RUN_ID=pipeline_test \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=400 \
WARMDOWN_ITERS=160 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
NUM_LAYERS=10 \
MLP_MULT=3 \
MATRIX_LR=0.02 \
MUON_MOMENTUM=0.99 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.04 \
WEIGHT_DECAY=0.04 \
MIXED_QUANT_ENABLED=1 \
EMA_ENABLED=1 \
EMA_DECAY=0.9999 \
QAT_ENABLED=1 \
QAT_FINAL_FRAC=0.15 \
BIGRAM_VOCAB_SIZE=10240 \
BIGRAM_DIM=128 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Final Metrics

- `final_int8_zlib_roundtrip_exact val_loss`: **2.40137744**
- `final_int8_zlib_roundtrip_exact val_bpb`: **1.42223098**
- `Total submission size int8+zlib`: **15,576,677 bytes**
- `Serialized model int8+zlib`: **15,528,991 bytes**
- `Code size`: **47,686 bytes**
- Peak memory: **18,768 MiB allocated**, **18,896 MiB reserved**

This artifact is under the 16,000,000-byte cap.

## Included Files

- `train_gpt.py` — exact script used for this run
- `train_seed1337.log` — run log from the command above
- `submission.json` — metadata entry
- `requirements.txt` — dependency list

## Notes

- This run used the `sp1024` tokenizer/dataset with 40 local training shards in this test environment.
- This is submitted to the non-record track and is not claiming 8xH100 record compliance.
