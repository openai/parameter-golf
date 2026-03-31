# Competitive Submission: Depth Recurrence + Widening + QAT + SP4096

## Architecture

- **Depth Recurrence**: 3 physical transformer blocks looped 3 times (9 effective layers)
- **Per-loop LoRA**: Rank-4 LoRA adapters on Q/K/V/O per loop iteration for specialization
- **Model Widening**: dim 768, 12 query heads, 4 KV heads (GQA 3:1), 2x MLP expansion
- **Vocabulary**: SP4096 (4096-token SentencePiece BPE)
- **Factorized Embedding**: 4096 x 128 -> 128 x 768 (ALBERT-style bottleneck), tied output projection

## Training Optimizations

- **QAT**: Fake int8 quantization with STE, enabled as compile-time constant (uses `amax` not `quantile`)
- **Muon Optimizer**: For matrix params with Newton-Schulz orthogonalization
- **Warmdown**: Cosine-style LR decay in final phase

## Evaluation

- **Long-context eval**: Non-overlapping chunks at eval_seq_len=2048 with NTK-aware RoPE scaling
- **Compliant**: Every validation token scored exactly once

## Parameter Budget

| Component | Parameters | Notes |
|-----------|-----------|-------|
| 3 shared blocks (768-dim) | ~11.8M | Q+K+V+O+MLP per block, int8 export |
| LoRA adapters (rank 4, 9 iters) | ~184K | fp16 passthrough (small tensors) |
| Factorized embedding (4096x128 + 128x768) | ~622K | fp16 passthrough |
| Control tensors | ~25K | Skip weights, scales, etc. |
| **Total** | **~12.6M** | Est. ~12.7MB artifact |

## How to Run

```bash
# Download SP4096 data
python data/cached_challenge_fineweb.py --variant sp4096

# Train (8xH100)
torchrun --nproc_per_node=8 records/track_10min_16mb/2026-03-18_CompetitiveSubmission/train_gpt.py

# Train with SP1024 fallback
DATA_PATH=./data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
torchrun --nproc_per_node=8 records/track_10min_16mb/2026-03-18_CompetitiveSubmission/train_gpt.py
```
