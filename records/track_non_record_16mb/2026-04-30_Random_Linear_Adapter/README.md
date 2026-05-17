# Notable Non-Record Submission: 1.1971 BPB — Learned Adapters on Random Linear Maps

## Summary
12 transformer layers, random adapter MLPs, mixed int-6/int-8 compression, and sliding eval

## Key Architecture Change
The main idea behind this submission is the idea of learned adapters on random linear maps. In the baseline, within each block's MLP, there are 2 large matrices stored &mdash; fc of (hidden, dim) and proj of (dim, hidden). Instead, the AdapterMLP utilizes a random seed to generate W_fc and W_proj of the same dimensions as fc and proj in the baseline. However, these do not take up space within the artifact, since they are randomly generated and not fixed parameters (random linear map). In order to actually have meaningful computation within the MLP, there are 2 low-rank matrices stored for each W_fc and W_proj, essentially acting as LoRA matrices that facilitate learning (learned adapters). The dimensions of A, B for W_fc are (hidden, rank) and (rank, dim). The dimensions of A, B for W_proj are (dim, rank) and (rank, hidden). 

Instead of 2 * hidden * dim parameters for the MLP within each transformer block, the AdapterMLP architecture requires 2 * (rank * hidden + rank * dim) parameters. This artifact reduction is what allowed me to expand the number of transformer layers and mlp_mult factor &mdash; creating a wider + deeper network. 

Baseline MLP (mlp_mult = 2):
2 * 1024 * 512 = 1,048,576 parameters per block

Adapter MLP (rank = 160, mlp_mult = 3):
2 * (160 * 1536 + 160 * 512) = 655,360 parameters per block

Even with a wider MLP hidden dimension, this architecture saves 37.5% parameters per block.

## Space Savings
With these savings, I mostly utilized the freed artifact budget for:
- increasing the number of transformer layers
- wider MLP &mdash; using rank=160 to maintain expressiveness, while staying within the artifact budget

## Changes from Baseline
1. 12 transformer layers (vs 9 baseline)
2. random adapter MLPs
3. MLP mult 3x (instead of 2)
4. sequence length 2048
4. FP16 tied embedding
5. mixed int-6/int-8 compression (int-6 compression only on 3 blocks, i.e. 5, 6, 7)
6. zstd-22 compression
7. sliding window evaluation: stride=512
8. lower learning rates: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03

## Run
```bash pip install zstandard```

```bash
RUN_ID=train_gpt \
NUM_LAYERS=12 \
MLP_MULT=3 \
MLP_RANK=160 \
INT6_LAYERS=5,6,7 \
INT6_STEP=4 \
TRAIN_SEQ_LEN=2048 \
STRIDE=512 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
ITERATIONS=20000 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 10-minute Wallclock Results (8×H100 SXM)
| seed     | val_bpb  | sliding_val_bpb | submission_size|
|----------|----------|-----------------|----------------|
| 1337     | 1.2184   | 1.1971          | 15418110       |
| 42       | 1.2182   | 1.1969          | 15422121       |
| 2026     | 1.2179   | 1.1967          | 15413210       |