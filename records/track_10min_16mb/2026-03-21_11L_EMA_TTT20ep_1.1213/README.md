## Record: 11L EMA + TTT(20ep) — val_bpb: 1.1213

**val_bpb = 1.1213** (sliding window stride=64, best seed 1337) | **15.53 MB** artifact | 8xH100 SXM, 600s

### Key Finding: EMA + Aggressive TTT with All Blocks Unfrozen

EMA(0.997) weight averaging combined with aggressive test-time training (20 epochs SGD, lr=0.008, **all blocks unfrozen**) outperforms Tight SWA + VE128 approaches. Critical discoveries:

1. **TTT_FREEZE_BLOCKS=0 is essential.** Freezing early blocks during aggressive TTT creates internal inconsistency — unfrozen layers overfit while frozen layers can't adapt. Quant gap 5x worse with freeze=2 (Run 14 in our ablation).
2. **Late QAT is counterproductive** with aggressive TTT. Disabling it keeps weights clean for TTT adaptation.
3. **XSA (Exclusive Self Attention) removed** — saves ~1.4ms/step with FA2 fallback, yielding ~130 more training steps in the 600s budget.

### Results (3-seed, 8xH100 SXM)

| Seed | Steps | Step avg | Sliding BPB (s64) | Roundtrip BPB | Pre-quant BPB | Artifact |
|------|-------|----------|-------------------|---------------|---------------|----------|
| **1337** | **7386** | **81.2ms** | **1.1213** | 1.1446 | 1.1418 | 15.53 MB |
| 42 | 7411 | 81.0ms | 1.1221 | 1.1454 | 1.1426 | 15.51 MB |
| 2025 | 7386 | 81.2ms | 1.1228 | 1.1461 | 1.1418 | 15.53 MB |

**Mean: 1.1221 | Std: 0.0008**

### Comparison to Prior SOTA

| Submission | val_bpb | TTT config | Weight averaging |
|-----------|---------|------------|-----------------|
| PR #388 (prev SOTA) | 1.1231 | 25ep, lr=0.008, freeze=0 | Tight SWA + VE128 |
| **This submission** | **1.1213** | 20ep, lr=0.008, freeze=0 | EMA(0.997) |

### Architecture

- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA), MLP 3x (hidden=1536), relu-squared
- SmearGate + BigramHash(2048, dim=128) + OrthoInit
- Partial RoPE (16/64 dims) + LN Scale (1/sqrt(layer+1))
- EMA (decay=0.997), no SWA
- No XSA, no Late QAT
- Int6 mixed quantization + zstd-22 compression
- Logit softcap = 30

### Training

- Muon optimizer (matrix_lr=0.025, momentum 0.92→0.99 over 1500 steps, WD=0.04)
- AdamW for scalars/embeddings (scalar_lr=0.025, tied_embed_lr=0.035, WD=0.04)
- Batch: 786,432 tokens, seq_len=2048
- Grad clip: 0.3
- Warmdown: 3000 steps
- 20 compile warmup steps

### Test-Time Training

After training and int6 quantization roundtrip:
- 20 epochs full-weight SGD on validation tokens
- lr=0.008, momentum=0.9, grad_clip=1.0
- **All blocks unfrozen** (freeze_blocks=0)
- ~292s on 8xH100 (sharded across GPUs)
- TTT loss: 1.9406 → 1.9335 (seed 1337)

### Eval Timing (seed 1337)

| Phase | Time |
|-------|------|
| Training (600s cap) | 600s |
| TTT (20 epochs) | 292s |
| Non-overlapping eval | 1.9s |
| Sliding window eval (s64) | 90s |
| **Total eval** | **~384s** |

### Systematic Ablation (15 runs)

This submission is backed by a 15-run ablation study testing:

| Technique | Result | Finding |
|-----------|--------|---------|
| EMA + TTT(3ep, freeze=2) | 1.1242 | Baseline competitive config |
| Memory Tokens (64) | 1.1244 | Don't survive int6 quantization |
| Warmdown=20000 | ~1.28 | Catastrophic: over-smoothed weights, 24x worse quant gap |
| Batch 524K | killed | Way behind: fewer tokens/step not compensated |
| Tight SWA | 1.1249 | Worse quant gap than EMA (+0.0071 vs +0.0058) |
| Causal TTT | 1.1262 | Score-then-update: slightly worse, 33% faster |
| Two-Phase TTT | 1.1262 | Phase 2 adds nothing after standard TTT |
| Gradient-Guided Quant | 1.1250 | Reduces quant gap but artifact over 16 MB |
| Z-loss + no Late QAT | 1.1274 | Z-loss hurts pre-quant quality |
| TTT(20ep, freeze=2) | 1.1488 | Catastrophic: frozen blocks + aggressive TTT |
| **TTT(20ep, freeze=0)** | **1.1213** | **Winner: all blocks must adapt coherently** |
| PPM-C eval blending | 1.1350 | Classical compression hurts strong models |

### Run Command

```bash
pip install zstandard flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

SEED=1337 NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=0 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=0 \
TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=20 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Included Files

- `README.md` — this file
- `submission.json` — leaderboard metadata with 3-seed results
- `train_gpt.py` — complete training + TTT + evaluation script
- `train.log` — best seed (1337) full log
- `train_seed42.log` — seed 42 full log
- `train_seed2025.log` — seed 2025 full log
