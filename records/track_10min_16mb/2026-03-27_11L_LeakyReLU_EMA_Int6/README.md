# 11L LeakyReLU(0.5)^2 + EMA + Int6 Quantization

## Summary

Improved baseline combining proven techniques from the top leaderboard submissions:

- **11 transformer layers** (up from 9) with **3x MLP expansion** (up from 2x)
- **LeakyReLU(0.5)^2 activation** replacing ReLU^2 (preserves negative gradient flow, eliminates dead neurons)
- **EMA weight averaging** (decay=0.997) applied every training step
- **Int6 quantization** for weight matrices (6-bit signed, [-31,31]) + zstd-22 compression
- **Sequence length 2048** with NTK-aware RoPE scaling (base=20000)
- **Tuned hyperparameters**: Muon momentum 0.99 (warmup 0.92->0.99 over 1500 steps), matrix LR 0.025, grad clip 0.3, warmdown 3000 iters

## Architecture Details

| Component | Baseline | This Submission |
|-----------|----------|-----------------|
| Layers | 9 | **11** |
| MLP expansion | 2x (1024 hidden) | **3x (1536 hidden)** |
| Activation | ReLU^2 | **LeakyReLU(0.5)^2** |
| Sequence length | 1024 | **2048** |
| RoPE base | 10000 | **20000** (NTK scaling) |
| Weight averaging | None | **EMA (0.997)** |
| Quantization | int8 + zlib | **int6 + zstd-22** |
| Parameters | ~17M | **~26.5M** |
| Muon momentum | 0.95 | **0.99** |
| Matrix LR | 0.04 | **0.025** |
| Warmdown | 1200 | **3000** |
| Grad clip | None | **0.3** |
| Train batch tokens | 524,288 | **786,432** |

## Techniques Explained

### LeakyReLU(0.5)^2
Instead of `relu(x)^2` which kills all negative activations, we use `leaky_relu(x, 0.5)^2` which allows negative values to pass through with slope 0.5 before squaring. This eliminates dead neurons while maintaining the non-negative output bias. Proven -0.002 BPB improvement by the top submission.

### Int6 Quantization
Maps 2D weight matrices to 6-bit signed integers [-31, 31] instead of int8's [-127, 127]. This gives coarser precision per weight but allows ~25% more parameters to fit in the 16MB budget, which more than compensates for the precision loss.

### EMA Weight Averaging
Maintains an exponential moving average of all model parameters throughout training. At the end, we load the EMA weights for evaluation. This smooths out training oscillations and typically improves generalization at zero training cost.

## Running

### Local testing (Apple Silicon with MLX)
```bash
RUN_ID=improved_test \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
EMA_ENABLED=0 \
python3 train_improved_mlx.py
```

### 8xH100 (leaderboard submission)
```bash
RUN_ID=11L_leakyrelu_ema_int6 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

**Status: Awaiting H100 validation.** Local MLX testing confirms architecture trains correctly with decreasing loss.

Expected range based on technique overlap with top submissions: ~1.15-1.18 BPB (pre-TTT).
