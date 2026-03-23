# Value Residual + DenseFormer DWA + TTT

**val_bpb: TBD** (pending 8xH100 evaluation)

## Run Command

```bash
# Setup (once)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train + evaluate
RUN_ID=vresid_dwa_ttt \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-23_ValueResid_DWA_TTT/train_gpt.py
```

## Key Techniques (3 new additions on top of current SOTA)

### 1. Value Residual (ResFormer) -- 0 extra params
Cache the first layer's value vectors (V_1) and blend them into all subsequent layers:
`V_used = 0.5 * (V_n + V_1)` for layers n >= 2.

From "Value Residual Learning For Alleviating Attention Concentration" (arXiv:2410.17897):
- 10.4% fewer parameters needed for equivalent loss
- Zero parameter overhead -- pure algorithmic improvement
- Prevents attention concentration in deeper layers

### 2. DenseFormer DWA (replaces U-Net skips) -- ~65 scalar params
Each layer's input is a learned weighted average of ALL previous layer outputs, not just the mirror-skip from U-Net. This is strictly more general than U-Net skip connections.

From "DenseFormer: Enhancing Information Flow via Depth Weighted Averaging" (arXiv:2402.02622):
- 48-block DenseFormer matches 72-block Transformer
- Identity-initialized (diagonal=1, off-diagonal=0) so training starts from standard Transformer behavior
- Negligible parameter cost (65 scalars for 10 layers)

### 3. Test-Time Training (TTT) -- 0 storage cost
During evaluation, sequentially process validation windows: SCORE each window first, then TRAIN on the already-scored tokens using AdamW. This adapts the model to the validation distribution without cheating.

- Only trains on tokens that have already been graded
- AdamW optimizer on non-embedding parameters
- Default: 10 epochs per window, lr=1e-4

## Architecture (inherited from SOTA)
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- Mixed int5/int6 quantization + zstd-22
- Muon optimizer + SWA(0.4)
- Sliding window eval stride=64

## New vs Base
Built on the current #1 submission (10L_Int5MLP_MuonWD04_SWA50 by thwu1).
