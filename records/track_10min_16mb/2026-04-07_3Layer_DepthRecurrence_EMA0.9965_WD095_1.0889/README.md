## Record: 3-Layer Depth Recurrence + EMA 0.9965 + WD 0.095 (val_bpb: 1.0889)

**val_bpb: 1.0889** (sliding window stride=64, 3-seed mean, std 0.0005) | **~15.89 MB** | 8xH100 SXM, 590s

### 3-Seed Results (8×H100 80GB SXM)

| Seed | Pre-quant BPB | Sliding BPB (s64) | Artifact |
|------|---------------|-------------------|----------|
| 42 | 1.0950 | **1.0885** | 15,890,417 B |
| 1337 | 1.0959 | **1.0894** | — |
| 2024 | 1.0954 | **1.0888** | 15,895,711 B |

**Mean: 1.0889 | Std: 0.0005** | All artifacts under 16,000,000 bytes

Current merged SOTA: **1.1147** (PR #1019). Delta: **−0.0258 BPB**.

### Key Changes

Four refinements stacked on top of PR #1334's depth recurrence architecture:

| Parameter | PR #1334 | This | Source |
|-----------|----------|------|--------|
| **Recurrence layers** | 4,5 (2-layer) | **3,4,5 (3-layer)** | PR #1331 |
| **Weight decay** | 0.090 | **0.095** | PR #1331 |
| **Matrix LR** | 0.020 | **0.022** | PR #1331 |
| **EMA decay** | 0.997 | **0.9965** | PR #1421 (this author) |
| **Recurrence start** | step 3000 | **step 2000** | This work |
| **Warmdown fraction** | 0.667 | **0.72** | This work |

### Why This Combination Works

1. **3-layer recurrence (layers 3,4,5)**: Repeats 3 layers instead of 2, producing 14 virtual layers from 11 physical layers. More compute per forward pass without additional parameters.

2. **WD=0.095 + MLR=0.022**: Higher weight decay compresses weights more aggressively, improving GPTQ quantization. Higher matrix LR compensates for the regularization. Only 134K-186K values pruned (vs 290K+ at WD=0.090).

3. **EMA decay=0.9965**: Assigns slightly more weight to recent training steps, producing a final checkpoint that quantizes more cleanly under GPTQ int6.

4. **Early recurrence (step 2000)**: Activating depth recurrence 1000 steps earlier gives the model more training time with 14 virtual layers, improving final quality.

5. **Extended warmdown (72%)**: Longer learning rate decay allows weights to fully settle before GPTQ quantization, reducing the quant gap.

### Architecture (from PR #1334)

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- **Depth recurrence**: layers 3,4,5 repeat (virtual 14 layers), activated at step 2000
- Skip gates (learnable residual gating)
- Parallel residuals from layer 7
- QK-Gain 5.0
- Shared Value Embedding (dim=128, layers 9,10)
- Tied embeddings, logit softcap=30.0
- SP4096 tokenizer (SentencePiece BPE)

### Training

- FlashAttention 3 (Hopper-optimized)
- Muon optimizer (matrices): lr=0.022, momentum=0.99, WD=0.095, backend_steps=5
- Adam (head): lr=0.008, fused=True
- AdamW (embeddings): lr=0.6, WD=0.095, fused=True
- AdamW (scalars): lr=0.02, WD=0.02, fused=True
- Gradient clip: 0.3, Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 72%, **EMA decay=0.9965**
- Wallclock: 590s effective (10s reserved for GPTQ)

### Quantization

- GPTQ int6 with percdamp=0.05, 64 calibration batches
- Selective pruning (~134K-186K lowest-error ±1 values)
- Brotli compression

### Run Command

```bash
SEED=42 RECUR_START_STEP=2000 WARMDOWN_FRAC=0.72 \
DATA_PATH=./data/datasets/fineweb10B_sp4096/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
VOCAB_SIZE=4096 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Reproducibility

All 3 seeds produce valid artifacts under 16MB with tight variance (std=0.0005 BPB). Training completes in ~590s. The env-var based configuration ensures exact reproducibility.

### Credits

- **Base architecture + depth recurrence**: PR #1334 by @aryanbhosale
- **3-layer recurrence + WD/LR tuning**: PR #1331
- **EMA decay tuning (0.9965)**: PR #1421 by @X-Abhishek-X (this author)
- **Early recurrence + extended warmdown**: This work
