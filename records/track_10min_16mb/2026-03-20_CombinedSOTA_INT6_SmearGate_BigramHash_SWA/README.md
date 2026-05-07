# 11L XSA + SmearGate + BigramHash + SWA + OrthoInit + RoPE50K

**Mean val_bpb: 1.1565 (3 seeds)** | Best: 1.1538 (seed 1337) | Artifact: ~15.9 MB

## Key Techniques

1. **11 transformer layers** (baseline 9) with XSA on last 4 layers
2. **Exclusive Self Attention (XSA)** — removes self-value bias via GQA-compatible subtraction
3. **SmearGate + BigramHash(2048)** — bigram-aware embedding with OrthoInit (critical co-dependency)
4. **INT6 per-row quantization + zstd-22** — FP16 tied embedding + Late-K FP16 (last 2 layers c_k)
5. **SWA (every 50 steps, start at 40%)** — fp32 accumulation (bf16 causes catastrophic loss)
6. **Muon WD=0.04** — decoupled weight decay for quantization-friendly weights
7. **RoPE base 50K** (default 10K) — better long-context modeling
8. **Overtone SVD init + Phase-transition residual mixing** — spectral embedding initialization
9. **MLP 2.75x expansion** (hidden=1408) — sweet spot for 16MB with 11L
10. **Magnitude pruning 2%** before quantization

## Results (3 seeds)

| Seed | Steps | Sliding BPB | Post-quant BPB | Artifact |
|------|-------|-------------|----------------|----------|
| 1337 | 7,910 | **1.1538** | 1.1766 | 15.99 MB |
| 42 | 7,927 | 1.1565 | 1.1790 | 15.87 MB |
| 7 | 7,922 | 1.1593 | 1.1820 | 15.93 MB |
| **Mean** | | **1.1565** | **1.1792** | |

## Configuration

```bash
NUM_LAYERS=11 XSA_LAST_N=4 BIGRAM_VOCAB_SIZE=2048
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 MLP_MULT=2.75
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_WEIGHT_DECAY=0.04
GRAD_CLIP_NORM=0.3 WARMDOWN_ITERS=3000 ROPE_BASE=50000
SWA_ENABLED=1 SWA_EVERY=50 SWA_START_FRAC=0.4
USE_INT6=1 USE_OVERTONE_INIT=1 LATE_K_FP16_LAYERS=2
EVAL_STRIDE=64 COMPILE_FULLGRAPH=0
```

## Training Command

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

## Key Findings from 23 Runs

- **EMA(0.997) destroys quantization** — 0.14 BPB gap vs SWA's 0.02 (contradicts PR #287)
- **11L + MLP 3x doesn't fit** in 16MB with SmearGate+BigramHash
- **SmearGate matters** — removing it to fit MLP 3x loses more than it gains
- **XSA GQA bug** — must use repeat_interleave for v expansion (4 KV → 8 Q heads)
- **Seq curriculum doesn't work** — SWA checkpoint incompatibility across seq lengths
- **Higher LR (0.03) improves BPB** but makes artifact larger (worse compression)
- **Depth recurrence works** but dim=640 too small; dim=768+ exceeds 16MB

## Dependencies

- PyTorch 2.5+ (CUDA 12.1+)
- zstandard, sentencepiece, numpy

Built with Claude Code (Anthropic).
