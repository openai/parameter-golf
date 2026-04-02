# [Non-Record] Hymba-8L: Hybrid SSM + Sliding Window Attention with 32K Context (1.1470 BPB)

## Summary

This submission uses a hybrid architecture combining Mamba SSM with sliding window attention (SWA), which allows us to train at **32x longer context** (32,768 tokens) than the standard baseline (1,024 tokens) under the same compute and time constraints. Unlike full attention which scales quadratically, SWA and Mamba both scale linearly, making long-context training feasible within the 10-minute wall-clock budget.

Building on our previous Hymba submission (1.1873 BPB, 7L), this version adds a systematic ablation study across architecture, regularization, quantization, and evaluation strategies, yielding a **-0.040 BPB improvement**.

## Results

| Seed | val_bpb | val_loss | Steps | Artifact Size |
|------|---------|----------|-------|---------------|
| 1337 | 1.1474  | 1.9374   | 6,621 | 15.7 MB       |
| 42   | 1.1469  | 1.9366   | 6,620 | 15.6 MB       |
| 7    | 1.1468  | 1.9363   | 6,606 | 15.3 MB       |
| **Mean** | **1.1470 ± 0.0003** | | | |

- Training: 600s on 8xH100 SXM, ~90.7 ms/step
- Evaluation: Score-first TTT (25 epochs), ~580s
- Artifact: int8 + zstd-22, under 16 MB

## Key Improvements Over Previous Submission (1.1873 BPB)

### 1. 8 Layers (up from 7)
Added an 8th layer for more model capacity. Despite slower per-step time (~180 vs ~160 ms on 4xH100), the quality improvement (-0.009 BPB) more than compensates. Artifact still fits under 16 MB with int8 + WD=0.14.

### 2. SWA-1024 (up from SWA-512)
Doubled the sliding window attention window from 512 to 1024 tokens. Each token now attends to 1024 previous tokens via local attention while Mamba handles global context through recurrent state. This yielded another -0.009 BPB with only ~6 ms/step overhead.

### 3. SSM State=4 (down from 8)
Counter-intuitively, reducing the Mamba state dimension from 8 to 4 improved both speed and quality. With SWA-1024 handling local patterns, the SSM needs less recurrent state. This gave ~8 ms/step speedup and -0.002 BPB improvement.

### 4. Untied Embeddings
Separate lm_head instead of tying with tok_emb. Faster training (~159 vs ~176 ms/step with tied) and better BPB. The speed gain alone yields ~200 more steps in 10 minutes.

### 5. High Weight Decay (WD=0.15) + int8 Quantization
Higher WD acts as strong regularization, improving pre-quant BPB monotonically up to WD=0.14. WD=0.15 is used to ensure all seeds fit under 16 MB with a safety margin. Combined with full int8 quantization (not int6), this gives the best post-quant BPB while fitting under 16 MB. Key insight: the model is overfitting at this training duration, so aggressive regularization helps generalization.

### 6. Aggressive Warmdown (7000 iters)
Extended cosine LR warmdown from 3000 to 7000 iterations (wall-clock based). The model benefits greatly from prolonged LR decay, with a large BPB drop during the warmdown phase. This also reduces step time late in training due to smaller weight updates.

### 7. TTT: 25 Epochs, No Freeze
Increased test-time training from 3 epochs with 2 frozen blocks to 25 epochs with all blocks unfrozen. The cosine LR decay in TTT prevents catastrophic forgetting even without freezing. Score-first TTT remains the evaluation strategy.

### 8. GRAD_ACCUM_STEPS=1
Eliminated gradient accumulation overhead by processing the full local batch in a single micro-step per GPU. This saves ~6 ms/step, yielding ~200 more training steps.

## Hymba Hybrid Architecture

Based on the Hymba paper (arXiv:2411.13676), each block runs attention and Mamba **in parallel** within a single layer:
- Attention branch: Q projection + shared KV projection, GQA (8 heads, 4 KV heads), RoPE, QK-norm, SWA-1024
- Mamba branch: Selective scan with causal 1D convolution, gated output, state dim=4
- Learned merge: sigmoid-gated weighted sum of both branches
- Post-merge: output projection + residual with learned scale

Additional: LeakyReLU(0.9)^2 MLP, SmearGate + BigramHash embedding, U-Net skip connections, EMA(0.997).

## Ablation Summary

Over 50 ablation experiments were conducted across two days. Key findings:
- **Architecture**: 8L > 7L, SWA-1024 > 512 > 256, SSM_STATE=4 > 8 > 16, untied embeddings
- **Regularization**: WD=0.14 optimal for int8 under 16MB, WD=0.12 better BPB but over budget
- **Quantization**: int8 with high WD beats mixed int8/int6, GPTQ_LITE=0 works fine
- **Training**: warmdown=7000, GRAD_ACCUM_STEPS=1, 524K batch, EMA_EVERY=10, Muon steps=5
- **TTT**: LR=0.002, 25 epochs no-freeze, cosine decay
- **Not helpful**: XSA, Partial RoPE (16/64), LZMA compression, smaller/larger batch sizes

## Run Command

```bash
SEED=1337 SLIDING_WINDOW=1024 SWA_GLOBAL_LAYERS=none TRAIN_SEQ_LEN=32768 \
NUM_LAYERS=8 MLP_MULT=4 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MATRIX_LR=0.02 SCALAR_LR=0.02 WARMDOWN_ITERS=7000 WARMDOWN_SHAPE=cosine \
EVAL_STRIDE=0 EVAL_BATCH_SEQS=4 GPTQ_LITE=0 QUANT_BITS=8 \
HYMBA_EXPAND=1 HYMBA_SSM_STATE=4 \
USE_SMEARGATE=1 USE_BIGRAM_HASH=1 TIE_EMBEDDINGS=0 LEAKY_RELU_SLOPE=0.9 \
WEIGHT_DECAY=0.15 EMA_EVERY=10 GRAD_ACCUM_STEPS=1 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=25 TTT_CHUNK_TOKENS=524288 \
TTT_FREEZE_BLOCKS=0 TTT_BATCH_SEQS=4 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Dependencies

```bash
export PATH=/usr/local/cuda/bin:$PATH
pip install --no-build-isolation --break-system-packages mamba-ssm causal-conv1d zstandard sentencepiece
```

Requires PyTorch >= 2.5 for flex_attention (sliding window). Tested on PyTorch 2.8.0+cu128.
