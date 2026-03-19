# Non-Record: Looped Transformer + Per-Layer LoRA + Int6 + Sliding Window

**Status**: Non-record submission (1xH100). Pending 8xH100 compute for official validation.

## Method

This submission combines **depth recurrence** with **per-virtual-layer adaptation** and **encoder-decoder skip connections** to maximize model depth per stored byte.

### Architecture: Looped Transformer with LoRA + Skip Connections

- **5 unique transformer blocks** looped to create **30 virtual layers** (6x depth multiplier)
- **Encoder-decoder structure**: first 15 virtual layers = encoder (stores skip tensors), last 15 = decoder (consumes them in reverse via learned `skip_weights`)
- **Per-virtual-layer LoRA adapters** (rank=4) on Q,V projections differentiate each virtual layer
- **Residual mixing** (`resid_mix`): learned blend of hidden state with original embedding at each layer
- **Per-virtual-layer learnable scale** for fine-grained depth control
- Base block weights trained with NorMuon optimizer; LoRA/scalar params with Adam

The key insight is that naive weight sharing (PR #31: 1.2663 BPB, *worse* than baseline) fails because identical layers cannot specialize. Per-virtual-layer LoRA adapters solve this at minimal cost: 30 pairs of rank-4 adapters add only ~307K params (~1.5% of total).

### Training Improvements

- **NorMuon optimizer**: per-row normalized Newton-Schulz orthogonalization for better gradient conditioning
- **Wallclock-aware warmdown**: LR decay triggers based on remaining wall time, not step count (fixes warmdown never triggering when wallclock cap is hit before max iterations)
- **Stochastic Weight Averaging (SWA)**: averages 7 checkpoints during warmdown for smoother final weights
- **Gradient clipping** (norm=1.0) for stability with 30 virtual layers
- **Tuned LRs**: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03, MUON_MOMENTUM=0.99 with warmup from 0.92 over 1500 steps
- **Training at seq_len=4096** for richer context per token

### Quantization: Int6 with FP16 Embedding Passthrough

- Block weights quantized to int6 range [-31, 31], stored as int8 bytes
- Per-row fp16 scales
- **Token embedding kept in fp16** (most quantization-sensitive tensor)
- **LoRA parameters kept in fp16** (small, sensitive to quantization)
- Zlib compresses the zero high bits in int8-stored int6 values efficiently

### Evaluation: Sliding Window

- Overlapping windows of 4096 tokens, stride 64
- Each scored token gets nearly full 4096-token context
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
TRAIN_SEQ_LEN=4096
MATRIX_LR=0.02
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99
WARMDOWN_ITERS=3000
GRAD_CLIP_NORM=1.0
SWA_CHECKPOINTS=7
```

## Results (1xH100, RunPod — prior version without skip connections/NorMuon/SWA)

Three development runs were performed on a single H100 80GB (RunPod) with an earlier version of this script (int4 quant, no skip connections, basic Muon). Pod terminated after runs; detailed step logs not preserved.

### Run 3 (Best): d=768, 5 blocks, 25 virtual depth, batch=524K
- val_bpb: **1.50 pre-quant** (1.60 roundtrip int4) | Artifact: 8.88 MB | 600s, 203M tokens

### Estimated improvements (cumulative)

| Technique | Est. BPB Impact |
|-----------|----------------|
| Int6 quant (vs int4) | -0.075 |
| FP16 embedding passthrough | -0.007 |
| Sliding window eval | -0.035 |
| Encoder-decoder skip connections | -0.01 to -0.02 |
| Residual mixing | -0.01 |
| NorMuon optimizer | -0.002 to -0.003 |
| Tuned LRs + warmdown fix | -0.01 to -0.02 |
| SWA (7 checkpoints) | -0.003 to -0.005 |
| **Estimated 1xH100 BPB** | **~1.35-1.40** |

### 8xH100 Projection
- 8xH100 processes ~1.77B tokens (8.7x more than 203M on 1xH100)
- ~3.1 data doublings at ~0.09 BPB/doubling = -0.28 BPB from data scaling
- **Projected 8xH100 BPB: ~1.07-1.12** (would beat current SOTA ~1.163)

## Artifact Size Estimate

| Component | Raw Bytes | Notes |
|-----------|-----------|-------|
| Block weights (int6 as int8) | 19.66 MB | 5 blocks, d=768, MLP 2x |
| Scales (fp16) | 0.04 MB | Per-row scales |
| Embedding (fp16) | 1.57 MB | 1024 vocab, full precision |
| LoRA (fp16) | 0.61 MB | 30 virtual layers, rank 4 |
| Skip weights + scalars (fp32) | 0.08 MB | skip_weights, resid_mix, layer_scales |
| **Raw total** | **22.00 MB** | |
| **After zlib (est. 0.65)** | **~14.9 MB** | |
| Code | ~0.04 MB | |
| **Total artifact (est.)** | **~14.9 MB** | Under 16 MB cap |

## Command

```bash
RUN_ID=looped_int6_v2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py` — Complete training script
- `submission.json` — Leaderboard metadata
- `train_summary.log` — Summary of 1xH100 development runs
