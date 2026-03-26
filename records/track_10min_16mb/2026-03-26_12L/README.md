# 12L 

This PR experiments the idea of a 12 layers GPT. It takes the script from record `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` and does some modifications. To add an extra layer some parameters are removed from other parts. Mainly, they are removed from bigram and vocab embeddings.

Experiment 1:

* NUM_LAYERS: 11 -> 12
* MLP_MULT: 3.0 -> 2.7
* BIGRAM_VOCAB_SIZE: 2048 -> 1024
* BIGRAM_DIM: 128 -> 128
* VE_DIM -> 128 -> 64

Exceeds 16MB by 0.1MB on seed 1337. Not checked others because of costs.
DIAGNOSTIC post_ema val_bpb:1.1422 vs val_bpb: 1.1379 from previous record

Experiment 2:

* NUM_LAYERS: 11 -> 12
* MLP_MULT: 3.0 -> 2.7
* BIGRAM_VOCAB_SIZE: 2048 -> 2048
* BIGRAM_DIM: 128 -> 32
* VE_DIM -> 128 -> 64

Stays under 16MB by 0.2MB on seed 1337. Not checked others because of costs.
DIAGNOSTIC post_ema val_bpb:1.1430 vs val_bpb: 1.1379 from previous record

Experiment 3:

* NUM_LAYERS: 11 -> 12
* MLP_MULT: 3.0 -> 2.6
* BIGRAM_VOCAB_SIZE: 2048 -> 2048
* BIGRAM_DIM: 128 -> 256
* VE_DIM: 128 -> 256

Stays above 16MB by 0.1MB on seed 1337. Not checked others.
DIAGNOSTIC post_ema val_bpb:1.1407 vs val_bpb: 1.1379 from previous record

Others combinations reducing ML_MULP to 2.5 and mantaining VE_DIM and BIGRAM configuration exceed size or didnt improve at all.

Notes: expected improvement of loss per step for experiments 1 and 3 (0.01 bpb at step 4000) but since model trains slower it does not achieve better bpb.

## Artifacts

Included `.txt` files in this folder capture the exact `train_gpt.py` snapshot and logs used for the runs:

* `log_exp1_1337.txt`: full script snapshot for Experiment 1 (12L, `MLP_MULT=2.7`, `BIGRAM_VOCAB_SIZE=1024`, `BIGRAM_DIM=128`, `VE_DIM=64`)
* `log_exp2_1337.txt`: full script snapshot for Experiment 2 (12L, `MLP_MULT=2.7`, `BIGRAM_VOCAB_SIZE=2048`, `BIGRAM_DIM=32`, `VE_DIM=64`)
* `857e070d-ddf3-4f56-b3c5-bfda3a4c00a2.txt`: script snapshot with `BIGRAM_VOCAB_SIZE=2048`, `BIGRAM_DIM=64`, `VE_DIM=64` (extra variant)
