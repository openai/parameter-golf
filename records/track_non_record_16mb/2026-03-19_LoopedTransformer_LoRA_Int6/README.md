# Non-Record: Looped Transformer + Per-Layer LoRA + Int6 + Sliding Window

**Status**: Non-record submission (1xH100). Pending 8xH100 compute for official validation.

## Method

This submission combines **depth recurrence** with **per-virtual-layer adaptation** to maximize model depth per stored byte.

### Architecture: Looped Transformer with LoRA

- **5 unique transformer blocks** looped to create **30 virtual layers**
- Forward pass: `for i in range(30): x = blocks[i % 5](x, x0)`
- Stores 5 blocks but computes 30 layers deep (6x depth multiplier)
- **Per-virtual-layer LoRA adapters** on Q and V projections (rank=4) differentiate each virtual layer
- **Per-virtual-layer learnable scale** parameter for fine-grained control
- Base block weights trained with Muon optimizer; LoRA/scalar params with Adam

The key insight is that naive weight sharing (as shown in PR #31, which got 1.2663 BPB, *worse* than baseline) fails because identical layers cannot specialize. Per-virtual-layer LoRA adapters solve this at minimal parameter cost: 30 pairs of rank-4 adapters add only ~307K parameters (~1.5% of total), but allow each virtual layer to develop distinct attention patterns.

### Quantization: Int6 with FP16 Embedding Passthrough

- Block weights quantized to int6 range [-31, 31], stored as int8 bytes
- Per-row fp16 scales (same approach as int8, but 6-bit range)
- **Token embedding kept in fp16** (most quantization-sensitive tensor)
- **LoRA parameters kept in fp16** (small, sensitive to quantization)
- Zlib compresses the zero high bits in int8-stored int6 values efficiently
- No outlier protection needed (int6 has sufficient dynamic range)

### Evaluation: Sliding Window

- Overlapping windows of 4096 tokens, stride 64
- Each scored token gets nearly full 4096-token context
- Only the rightmost 64 tokens per window are scored
- Zero artifact cost improvement (~0.03-0.04 BPB from PR #77 ablations)

## Configuration

```
MODEL_DIM=768
UNIQUE_LAYERS=5
VIRTUAL_DEPTH=30
NUM_HEADS=12
NUM_KV_HEADS=4
MLP_MULT=2
VOCAB_SIZE=1024
LORA_RANK=4
EXPORT_BITS=6
EVAL_SEQ_LEN=4096
EVAL_STRIDE=64
TIE_EMBEDDINGS=1
TRAIN_BATCH_TOKENS=524288
TRAIN_SEQ_LEN=1024
```

## Results (1xH100, RunPod)

Three development runs were performed on a single H100 80GB (RunPod). The pod was terminated after runs completed to save costs; detailed step logs were not preserved.

### Run 1: Smoke test (d=384, 3 blocks, 9 virtual depth)
- val_bpb: 2.26 (2.28 roundtrip int4) | Artifact: 1.96 MB | 64.6s training

### Run 2: Scaled (d=768, 5 blocks, 25 virtual depth, batch=65K)
- val_bpb: 1.57 (1.78 roundtrip int4) | Artifact: 8.56 MB | 600s, 59M tokens

### Run 3: Improved (d=768, 5 blocks, 25 virtual depth, batch=524K, compile)
- val_bpb: **1.50 pre-quant** (1.60 roundtrip int4) | Artifact: 8.88 MB | 600s, 203M tokens
- Outlier protection cut int4 quantization loss by 53%

### Estimated improvement with int6 + fp16 embed + sliding window
Based on ablation data from PR #65, #66, #77:
- Int6 quant gap: ~+0.025 (vs +0.10 with int4), saving ~0.075 BPB
- FP16 embedding: ~-0.007 BPB
- Sliding window: ~-0.035 BPB
- **Estimated 1xH100 BPB: ~1.48** (vs 1.60 with int4)

### 8xH100 Projection
- 8xH100 would process ~1.77B tokens (vs 203M on 1xH100, 8.7x more)
- BPB improves ~0.09 per data doubling (~3.1 doublings)
- **Projected 8xH100 BPB: ~1.20** (would beat baseline 1.2244)
- With further optimization (wider MLP, QAT, hyperparameter tuning): potentially ~1.17-1.19

## Artifact Size Estimate

| Component | Raw Bytes | Notes |
|-----------|-----------|-------|
| Block weights (int6 as int8) | 19.66 MB | 5 blocks, d=768, MLP 2x |
| Scales (fp16) | 0.04 MB | Per-row scales |
| Embedding (fp16) | 1.57 MB | 1024 vocab, kept full precision |
| LoRA (fp16) | 0.61 MB | 30 virtual layers, rank 4 |
| Scalars (fp32) | 0.03 MB | Norms, gains, layer scales |
| **Raw total** | **21.92 MB** | |
| **After zlib (est. 0.65)** | **~14.3 MB** | |
| Code | 0.039 MB | |
| **Total artifact (est.)** | **~14.3 MB** | Under 16 MB cap |

## Command

```bash
RUN_ID=looped_int6_v1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For 1xH100 development:
```bash
RUN_ID=looped_int6_dev \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Included Files

- `train_gpt.py` — Complete training script with looped architecture, LoRA, int6 export, sliding window eval
- `submission.json` — Leaderboard metadata
- `train_summary.log` — Summary of 1xH100 development runs (detailed logs lost with terminated pod)
