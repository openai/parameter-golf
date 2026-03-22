# 2026-03-20_Int6_3xMLP

**Mean val_bpb = 1.1724** (5 seeds, std=0.0026, p<0.01 vs baseline)

Int6 per-row quantization with 3x MLP expansion, accompanied by 9 controlled ablations.

### Statistical Validation

| Seed | val_bpb |
|------|---------|
| 31337 | 1.1703 |
| 1337 | 1.1708 |
| 2024 | 1.1712 |
| 42 | 1.1732 |
| 7 | 1.1767 |
| **Mean** | **1.1724** |
| **Std** | **0.0026** |

---

## Approach

Int6 per-row quantization stores weights in 6 bits ([-31, 31]) instead of int8's 8 bits ([-127, 127]). Combined with zstd-22 compression, this saves ~3MB of artifact budget compared to int8+zlib, enabling a 3x MLP expansion (hidden=1536 vs baseline's 1024). The wider MLP adds 4.8M parameters.

### Configuration

```
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_HIDDEN=1536 TIE_EMBEDDINGS=1 FP16_EMBED_EXPORT=1
QUANT_BITS=6 USE_ZSTD=1
WARMDOWN_ITERS=20000 MATRIX_LR=0.06 SCALAR_LR=0.06 TIED_EMBED_LR=0.07
GRAD_CLIP_NORM=1.0 MUON_BACKEND_STEPS=5 MUON_MOMENTUM=0.99
EVAL_STRIDE=64 MAX_WALLCLOCK_SECONDS=600
```

### Key Metrics

| Metric | Naive Baseline | Int6 + 3xMLP |
|--------|---------------|-----------------|
| model_params | 17,059,912 | 21,778,504 |
| Pre-quant val_bpb | 1.2172 | 1.1949 |
| **Post-quant val_bpb** | **1.2244** | **1.1708** |
| Artifact size | 15,863,489 (int8+zlib) | 15,175,136 (int6+zstd) |
| Artifact headroom | 137KB | 825KB |
| Steps | ~13,800 | 12,507 |
| step_avg | 43.5ms | 48.0ms |

---

## Ablations

| # | Technique | val_bpb | vs Control (1.1929) | Verdict |
|---|-----------|---------|--------------------|---------|
| 1 | SWA (weight averaging) | 1.1933 | +0.0004 | **No effect** at WD=1200 |
| 2 | Doc-isolated eval | 1.2015 | +0.0086 | **Hurts** at stride=64 |
| 3 | Curriculum learning | 1.1942 | +0.0013 | **No effect** |
| 4 | Multi-token prediction | 1.1947 | +0.0018 | **No effect** |
| 5 | **Int6 + 3x MLP** | **1.1708** | **-0.0221** | **Best result** |
| 6 | + SmearGate + BigramHash | 1.1739 | -0.0190 | **Hurts** on top of int6 |
| 7 | Depth recurrence + Huginn (skips) | 4.34 | — | **Catastrophic** |
| 8 | Depth recurrence + Huginn (flat) | 5.58 | — | **Catastrophic** |
| 9 | Int8 QAT (PR #145) | 1.2052 | +0.0123 | **Overhead exceeds recovery** |

### Key Negative Findings

**1. Doc-isolated eval hurts at stride=64**

The LoRA TTT entry (#77) found doc-isolation worth +0.011 BPB at stride=256. At stride=64, it costs -0.009 BPB. At stride=64, tokens already have 960+ tokens of context. Removing cross-doc context at document boundaries means start-of-document tokens lose all context, which hurts more than cleaner context helps. There is a crossover stride length between 64 and 256 where doc-isolation flips from harmful to helpful.

**2. SmearGate + BigramHash hurt with int6**

SmearGate + BigramHash have been reported as helpful in other entries, but on the int6+3xMLP base they cost +0.003 BPB. BigramHash adds ~524K params that get int6 quantized and had insufficient training steps. The implementations may differ from the originals, or the gains require interaction with other techniques (OrthoInit, specific SWA schedule).

**3. Huginn eval-time scaling fails at small scale**

Depth recurrence (3 shared blocks × 3 loops = 9 effective layers) with Huginn-style eval scaling (6 loops at eval) produces random output (4.34-5.58 BPB). Tested both with U-Net skips (skips disabled for extra loops) and flat loops (trained without skips). Neither works. The 3 shared blocks at 7.6M params lack sufficient capacity to learn general iterative refinement. Huginn was validated at 3.5B — the technique does not transfer to 7.6M scale.

**4. Int8 QAT overhead exceeds recovery**

Exact INT8_CLIP_Q percentile matching via `torch.quantile` adds ~20% per-step overhead, costing ~2,000 training steps. The lost training tokens hurt more than the ~0.007 BPB quantization gap recovery. QAT likely only pays off with int6 (larger gap to close) using a faster approximate quantile.

### Implementation Bugs Discovered

**SWA bf16 accumulation:** Initial SWA implementation accumulated weights in bf16, producing val_bpb=2.62 after thousands of additions. Fix: accumulate in float32, sample every 50 steps.

**torch.compile graph priming:** Pre-compiling both QAT and non-QAT graphs during warmup caused 50% step time regression for the non-QAT path. Fix: don't pre-prime conditional code paths.

**zstd/zlib decompression mismatch:** Compressing with zstd then decompressing with zlib crashes. Fix: match decompressor to compressor.

---

## Reproduction

```bash
cd /workspace
git clone https://github.com/mrdavtan/parameter-golf.git
cd parameter-golf && git checkout int6-3xMLP-pr
pip install sentencepiece huggingface_hub zstandard
python3 data/cached_challenge_fineweb.py --variant sp1024

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_Int6_3xMLP/train_gpt.py
```

Environment variables listed in Configuration section above.

Hardware: 8×H100 SXM (RunPod Parameter Golf template), PyTorch 2.9.1+cu128

---

## Acknowledgments

- Int6 quantization approach studied from WarmdownQuantization entry by @samuellarson
- Sliding window evaluation from #50 by @mattqlf
- Hyperparameter tuning informed by #65 (@samuellarson) and #128 (@rsavitt)
- SmearGate/BigramHash implementations based on modded-nanogpt community
- Depth recurrence inspired by PR #167 and Huginn (arxiv 2502.05171)
- Doc-isolated eval concept from LoRA TTT entry (#77) by @samacquaviva
