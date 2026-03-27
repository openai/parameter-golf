# [Non-Record] Hymba-LongContext: 32K Context Training via Hybrid SSM + Sliding Window Attention (1.1873 BPB)

## Summary

This submission demonstrates that hybrid SSM architectures can train at **32x longer context** (32,768 tokens) than the standard baseline (1,024 tokens) with **near-constant cost as context length increases**. By combining Mamba (selective state space model) with sliding window attention (SWA-512), both branches have constant per-token cost, going from 8K to 64K context adds virtually no overhead (~80-83 ms/step). This enables ultra-long context training within the 10-minute wall-clock budget.

## Results

| Seed | val_bpb | val_loss | Steps | Artifact Size |
|------|---------|----------|-------|---------------|
| 1337 | 1.1866  | 2.0036   | 7,334 | 14.3 MB       |
| 42   | 1.1881  | 2.0061   | 7,139 | 14.6 MB       |
| 7    | 1.1873  | 2.0048   | 7,176 | 14.5 MB       |
| **Mean** | **1.1873 +/- 0.0008** | | | |

- Training: 600s on 8xH100 SXM, ~82-88 ms/step
- Evaluation: Score-first TTT, ~67-74s
- Artifact: int6 + zstd-22, under 16 MB

## Key Innovations

### 1. Ultra-Long Context Training (32K tokens)
The naive transformer baseline trains at 1,024 token context. We train at **32,768 tokens** — a 32x increase. Increasing context from 8K to 64K adds virtually no overhead (~80-83 ms/step across that range). This is because both SWA and Mamba have constant per-token cost as SWA attends to a fixed 512-token window regardless of sequence length, and Mamba's recurrent scan processes each token in O(1). Since the total tokens per batch is fixed (524K), the step time stays roughly constant regardless of how those tokens are divided into sequences.

This is made possible by two architectural choices:
- **Mamba SSM** processes sequences via selective scan with O(1) per-token cost (recurrent state)
- **Sliding Window Attention (SWA-512)** on all layers limits attention to a fixed 512-token local window, also O(1) per token

The Mamba branch handles global context (full sequence memory via recurrent state), while SWA handles local pattern matching.

### 2. Hymba Hybrid Architecture
Based on the Hymba paper (arXiv:2411.13676), each block runs attention and Mamba **in parallel** within a single layer:
- Attention branch: Q projection + shared KV projection, GQA (8 heads, 4 KV heads), RoPE, QK-norm
- Mamba branch: Selective scan with causal 1D convolution, gated output
- Learned merge: sigmoid-gated weighted sum of both branches
- Post-merge: output projection + residual with learned scale

Architecture: 7 layers, 512 dim, MLP 4x with LeakyReLU(0.9)^2, U-Net skip connections, SmearGate + BigramHash embedding, EMA (0.997).

### 3. Score-First Test-Time Training (TTT)
Legal TTT following the PR #461 recipe:
1. **Score** each 524K-token chunk under `inference_mode` (no gradient)
2. **Train** on the already-scored chunk with SGD (lr=0.002, momentum=0.9)
3. First 2 blocks frozen to prevent catastrophic forgetting
4. 3 epochs per chunk with cosine LR decay
5. Total eval time: ~67-74s

TTT improves post-quantization BPB by adapting the quantized model to validation data patterns.

### 4. Context Length Scaling Results
We systematically evaluated training context length while keeping all other hyperparameters fixed:

| Train Seq Len | ms/step | Pre-quant BPB (13,780 steps) |
|---------------|---------|------------------------------|
| 8,192         | 79.0    | 1.1507                       |
| 16,384        | 80.1    | 1.1491                       |
| 32,768        | 80.6    | 1.1478                       |
| 65,536        | 82.8    | 1.1477                       |

Per-step cost is nearly constant from 8K to 64K context because the per-token cost of both SWA and Mamba is independent of sequence length (see above). Quality improves with longer context up to ~32K, then plateaus.

## Why This Matters

With constant per-token cost, our architecture can train and evaluate at long context without the quadratic overhead that full attention would incur. This frees the eval time budget for TTT adaptation rather than expensive sliding window overlap.

The competition README specifically requests "state-space models" and "super long context for evaluation or training" as novel directions. This submission demonstrates both, showing that hybrid SSM architectures naturally enable ultra-long context training regimes.

## Run Command

```bash
SEED=1337 SLIDING_WINDOW=512 SWA_GLOBAL_LAYERS=none TRAIN_SEQ_LEN=32768 \
NUM_LAYERS=7 MLP_MULT=4 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MATRIX_LR=0.02 SCALAR_LR=0.02 WARMDOWN_ITERS=3000 WARMDOWN_SHAPE=cosine \
EVAL_STRIDE=0 EVAL_BATCH_SEQS=4 QUANT_BITS=6 GPTQ_LITE=1 \
HYMBA_EXPAND=1 HYMBA_SSM_STATE=8 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=524288 \
TTT_FREEZE_BLOCKS=2 TTT_BATCH_SEQS=4 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/hymba_long_context/train_gpt.py
```

## Dependencies

```bash
pip install --no-build-isolation mamba-ssm causal-conv1d
pip install zstandard
```

## Setup

Requires PyTorch >= 2.5 for flex_attention (sliding window). Tested on PyTorch 2.8.0+cu128.

```bash
pip install --no-build-isolation --break-system-packages mamba-ssm causal-conv1d
```

Note: `--no-build-isolation` is critical to avoid CUDA version mismatch during the mamba-ssm/causal-conv1d CUDA kernel builds.
