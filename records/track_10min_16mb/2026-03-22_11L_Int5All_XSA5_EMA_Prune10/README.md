# 11L Int5-All + XSA5 + EMA + 10% Pruning

**val_bpb: 1.1466** (sliding window stride=64, post int5+zstd quantization roundtrip, seed 42)

## Key Techniques

Built on the XSA+EMA+PartialRoPE stack from PR #315, with two novel modifications:

### 1. Uniform Int5 Quantization (MLP + Attention)
Prior SOTA uses int5 for MLP and int6 for attention. Through systematic post-training quantization search on a saved checkpoint, we found that **attention weights tolerate aggressive quantization as well as MLP weights** — the per-row scaling preserves the critical information even at 5-bit resolution.

Using int5 for both saves ~1MB compressed vs mixed int5/int6, freeing byte budget.

### 2. 10% Magnitude Pruning
After EMA averaging but before quantization, we zero out the smallest 10% of weights (by absolute value) in all large matrices. These near-zero weights contribute minimally to model quality but waste bits after quantization. The zeros compress extremely well under zstd, saving ~500KB.

Combined, these two techniques reduce the artifact from ~15.6MB to **14.8MB** while maintaining competitive quality.

## Architecture

- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA), MLP 3x
- XSA (Exclusive Self Attention) on last 5 layers
- EMA (decay=0.997) instead of SWA
- Partial RoPE (16 of 64 head dims)
- LN Scale (1/sqrt(layer_idx+1))
- SmearGate + BigramHash(4096, dim=128)
- Late QAT: int5 STE fake-quantization in final ~5% of training
- Orthogonal init + muP output scaling

## Training

- Muon optimizer: matrix_lr=0.025, WD from decoupled weight decay
- Momentum: 0.99 (warmup from 0.85 over 500 steps)
- Warmdown: 3000 iters, warmup: 20 steps
- Seq len: 2048, batch: 786K tokens/step
- Grad clip: 0.3

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1504 |
| Post int5+zstd val_bpb (standard) | 1.1703 |
| **Post int5+zstd val_bpb (sliding s64)** | **1.1466** |
| Steps completed | 5,987 |
| Step time | 100ms |
| Artifact size | 14,811,335 bytes |
| Model params | 26,829,913 |

## Run Command

```bash
# 8xH100 SXM, all defaults baked into the script
RUN_ID=run SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Acknowledgments

Built on the XSA+EMA architecture from PR #315 by @alertcat.
