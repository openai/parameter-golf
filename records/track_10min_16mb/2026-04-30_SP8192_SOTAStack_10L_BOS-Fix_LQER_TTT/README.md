# SP8192 + BOS-Fix SmearGate + LQER Asym + Phased TTT (10L)

**val_bpb: 1.07171** (single seed 314) | **15.37 MB** | 8xH100 SXM, 596s

Applies the full SOTA stack from PR #1851 (BOS-fixed SmearGate + LQER Asymmetric + Phased TTT + layer looping) with the SP8192 tokenizer instead of SP1024. Uses 10 transformer layers instead of 11 to fit the larger embedding table (8192 vocab) under the 16MB artifact limit with brotli compression.

## Results

| Metric | Value |
|---|---|
| Pre-quant val_bpb | 1.07399 |
| Post-GPTQ val_bpb | 1.08251 |
| Post-TTT val_bpb | 1.07171 |
| Artifact size | 15,373,365 bytes |
| Training steps | 5,218 |
| Training time | 596s |
| Hardware | 8xH100 80GB SXM |
| Seed | 314 |

## Changes vs PR #1851

- **SP8192 tokenizer** instead of SP1024 — better tokenization efficiency, ~0.02 BPB improvement
- **10 layers** instead of 11 — required to fit under 16MB with the larger 8192-vocab embedding table
- All other settings identical to PR #1851: BOS-fixed SmearGate, GPTQ int6 + LQER Asymmetric, Phased TTT (1 phase, 2000 prefix docs), layer looping, SparseAttnGate

## Architecture

| Component | Setting |
|---|---|
| Layers | 10 (512d, 8 heads, 4 KV heads) |
| Vocabulary | SP8192 |
| Layer loop | encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9] |
| SmearGate | BOS-boundary fixed (PR #1851) |
| Quantization | GPTQ int6 + LQER Asymmetric |
| TTT | Phased score-first LoRA TTT (rank 96, 1 phase) |
| Compression | brotli quality=11 |

## Run Command

```bash
TORCHINDUCTOR_CACHE_DIR=/workspace/inductor_cache \
RUN_ID=sota_sp8192_10L SEED=314 VOCAB_SIZE=8192 NUM_LAYERS=10 \
SMEAR_GATE_ENABLED=1 SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 \
MLP_CLIP_SIGMAS=11.5 EMBED_CLIP_SIGMAS=14.0 \
WARMDOWN_FRAC=0.85 MIN_LR=0.1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

```
PR #1851 (1.0614 BPB) — BOS-fix SmearGate + LQER Asym + Phased TTT
    └── This work:
        ├── SP8192 tokenizer (from sproos/parameter-golf-tokenizers)
        └── 10 layers (down from 11 to fit 16MB with larger vocab embedding)
```
