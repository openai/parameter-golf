# 7L MLP3x 4k Seq LR-Tuned

## Summary

A 7-layer, 512-dim transformer with 3x MLP width (hidden=1536), trained at 4096 sequence length with carefully tuned learning rates. The key insight is that **lower learning rates create smoother weight distributions that quantize far better**, reducing the int8 quantization gap from 0.008 to ~0.001.

## Architecture

- **7 transformer layers** at width 512 (vs baseline 9 layers)
- **MLP expansion 3x** (hidden=1536 vs baseline 2x/hidden=1024)
- 8 attention heads, 4 KV heads (GQA), tied embeddings
- Extra RMSNorm before attention and MLP output projections
- Same total parameter count as baseline (~17M), fewer layers but wider MLP
- Fewer layers = faster step time (39-58ms vs 47ms), more steps in 600s

## Training Recipe

- **Sequence length 4096** (vs baseline 1024) — longer context dramatically improves per-step quality
- `tied_embed_lr=0.01` (vs baseline 0.05) — single biggest improvement, controls quantization gap
- `matrix_lr=0.03` (vs baseline 0.04) — smoother Muon updates
- `logit_softcap=15` (vs baseline 30) — tighter logit capping helps both training and quantization
- `qk_gain_init=1.0` (vs baseline 1.5)
- `warmdown_iters=3000` (vs baseline 1200) — longer gradual LR decay
- Standard int8 + zlib compression, no QAT

## Key Metrics

| Metric | Value |
|--------|-------|
| Post-quant val_bpb | **1.1933** |
| Pre-quant val_bpb | 1.1922 |
| Quantization gap | 0.0011 |
| Artifact size | 15.77 MB |
| Steps completed | ~10,400 |
| Step time | ~58ms |
| Training time | 600s (wallclock cap) |

## Command

```bash
RUN_ID=submission \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are set as defaults in the script. No environment variable overrides needed.

## What Worked (ranked by impact)

1. **Sequence length 4096**: ~0.020 bpb improvement. Longer context gives much better predictions.
2. **tied_embed_lr 0.05→0.01**: ~0.005 bpb + quant gap 0.008→0.001. Lower embedding LR creates smooth weight distributions.
3. **7L mlp_mult=3**: ~0.004 bpb. Same params, 15% faster steps than 9L mlp_mult=2.
4. **warmdown_iters 1200→3000**: ~0.003 bpb. Gradual LR decay helps generalization.
5. **matrix_lr 0.04→0.03**: ~0.001 bpb. Smoother Muon optimizer updates.
6. **logit_softcap 30→15**: ~0.001 bpb. Tighter capping constrains the output distribution.

## What Didn't Work

- SwiGLU (better per-step but slower, net negative under wallclock)
- Depth recurrence / weight sharing (catastrophic failure)
- Cosine LR schedule (worse than linear warmdown)
- Gradient clipping (neutral)
- Changing INT8_CLIP_PERCENTILE (made quant gap worse)
- seq_len=8192 (too slow, fewer steps negated context benefit)

## Statistical Significance (3 runs)

| Seed | val_bpb |
|------|---------|
| 1337 | 1.1933 |
| 42 | 1.1930 |
| 7 | 1.1916 |
| **Mean** | **1.1926** |
| **Std** | 0.0009 |

All 3 runs beat the naive baseline (1.2244) by >0.030 (p << 0.01).

## Included Files

- `train_gpt.py` — code snapshot
- `train.log` — training log (seed 1337)
- `train_seed42.log` — training log (seed 42)
- `train_seed7.log` — training log (seed 7)
- `submission.json` — metadata
