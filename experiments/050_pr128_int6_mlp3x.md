# Experiment 050: PR128 Script — Int6 MLP3x + STE QAT + Sliding Window

## Status: KILLED (user redirected to run our script instead)

## Config (PR128 exact)
- VOCAB_SIZE=1024, NUM_LAYERS=9, MODEL_DIM=512, MLP_MULT=3.0 (relu² 3x h=1536)
- Int6 QAT + zstd-22, fp16 embed, sliding eval stride=64
- MATRIX_LR=0.02, MUON_MOMENTUM=0.99, WARMDOWN_ITERS=3000
- TRAIN_SEQ_LEN=4096, TRAIN_BATCH_TOKENS=393216

## Partial Results (killed at step ~1200)
- 55.6ms/step, train_loss 2.33 at step 1200

## PR128 reported: val_bpb=1.1594 sliding, artifact 15.16MB, 10,535 steps @ 57ms

## Script
experiments/pr128_train_gpt.py
