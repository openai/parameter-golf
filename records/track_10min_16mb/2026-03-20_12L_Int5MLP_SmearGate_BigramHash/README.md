# 12L Int5-MLP + Int6-Attn + SmearGate + BigramHash + SWA

**val_bpb: 1.1541** (sliding window stride=64, 3-seed mean) | **~15.9 MB** artifact | 8xH100 SXM, 600s

## Key Innovation: Mixed Int5/Int6 Quantization + 12 Layers

Instead of uniform int6 quantization, we use precision-tiered quantization:
- **Int5 [-16,15]** for MLP weights (largest tensors, most compressible)
- **Int6 [-32,31]** for attention weights (more precision-sensitive)
- **FP16** for tied embeddings

Int5 values stored in int8 have **3 zero high bits** vs 2 for int6. zstd-22 compresses int5 at ~1.88x vs int6 at ~1.51x, saving ~1.8MB. This funds a **12th transformer layer** while staying under 16MB — the deepest model submitted to date.

## Architecture

- **12 transformer layers** (deepest submission), 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu² activation
- SmearGate (learned token blending gate)
- BigramHash (2048 buckets, dim=128)
- U-Net skip connections
- Tied embeddings, vocab 1024, seq_len 2048

## Training Config

| Parameter | Value |
|-----------|-------|
| Layers | **12** |
| Matrix LR | 0.025 |
| Scalar LR | 0.025 |
| Tied Embed LR | 0.035 |
| Muon Momentum | 0.99 |
| Muon WD | 0.04 |
| Adam WD | 0.04 |
| Warmdown | 3000 iters |
| SWA | every 200 steps, ~7 checkpoint avg |
| Eval stride | 64 |
| Batch | 786,432 tokens/step |

## Results (3-seed)

| Seed | Steps | ms/step | Post-Q BPB | Sliding BPB (s64) |
|------|-------|---------|------------|-------------------|
| 1337 | 5,590 | 107.34 | 1.17668 | **1.15402** |
| 42 | 5,588 | 107.37 | 1.17647 | **1.15390** |
| 2024 | 5,589 | 107.35 | 1.17679 | **1.15425** |

**Mean sliding BPB: 1.15406 | Std: 0.00035**

## Ablation: Why 12 Layers + Int5

| Config | Sliding BPB | Artifact | Notes |
|--------|-------------|----------|-------|
| 9L int6 (PR #162 base) | ~1.148 | 15.4 MB | Baseline |
| 11L int6 (PR #198) | **1.1318** | 15.7 MB | Current SOTA |
| **12L int5-MLP + int6-attn** | **1.1541** | ~15.9 MB | This submission |

The 12th layer adds depth but each step is slower (107ms vs 81ms for 11L), yielding ~5,590 steps vs ~7,412. The depth-vs-speed tradeoff doesn't fully pay off at 600s, but demonstrates that int5 MLP quantization is a viable compression strategy for fitting more layers.

## Reproduction

```bash
cd /workspace
git clone https://github.com/alertcat/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
bash run_8xh100.sh
```

## Files

- `train_gpt.py` — full training script with Int5 MLP quantization
- `README.md` — this file
- `submission.json` — structured results
