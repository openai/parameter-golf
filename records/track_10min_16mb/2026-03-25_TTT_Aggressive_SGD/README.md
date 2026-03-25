# Aggressive SGD TTT (val_bpb: 1.1124)

**3-seed mean val_bpb: 1.1124** (std=0.0008) | **15.4 MB artifact** | 8xH100 SXM, 600s training + 591s eval

## Results

| Seed | val_bpb (sliding, s64) | Artifact |
|------|------------------------|----------|
| 1337 | 1.1129 | 15,405,733 |
| 42 | 1.1128 | ~15.4M |
| 2024 | 1.1114 | ~15.4M |
| **Mean ± Std** | **1.1124 ± 0.0008** | |

## Approach

Standard 11L architecture, nothing exotic on the model side. The interesting part is the TTT. The base model trains for 600s, then TTT adapts all weights via SGD for 30 epochs on the validation data (score-first protocol).

The conventional wisdom is TTT at LR=0.002 for 3 epochs. We ran 20+ configurations on 4xH200 and found that cranking the LR to 1.0 and unfreezing every block turns a -0.0025 BPB technique into a -0.041 BPB technique. That's a 16x improvement from the same underlying method. It's like finding out your car has a sport mode you never tried.

## TTT Configuration

I swept this on 4xH200 before validating on 8xH100. The sweep told the whole story.

| Parameter | Our Value | PR #549 (merged SOTA) |
|-----------|-----------|----------------------|
| LR | 1.0 | 0.002 |
| Epochs | 30 | 3 |
| Freeze blocks | 0 (all unfrozen) | 0 |
| Momentum | 0.9 | 0.9 |
| TTT gain | -0.041 BPB | -0.0025 BPB |

### TTT LR Sweep (4xH200, 20 epochs, freeze=2)
| LR | Sliding BPB |
|----|------------|
| 0.01 | 1.1489 |
| 0.02 | 1.1471 |
| 0.05 | 1.1444 |
| 0.1 | 1.1422 |
| 0.2 | 1.1400 |
| 0.5 | 1.1351 |
| **0.7** | **1.1327** |
| 0.8 | 1.1355 |
| 1.0 | 1.1585 (diverged) |

BPB just keeps getting better as LR goes up... until it doesn't. Peak at 0.7 with 2 frozen blocks.

### Unfreezing all blocks (4xH200, 20 epochs)
| LR | freeze=2 | freeze=0 | Delta |
|----|----------|----------|-------|
| 0.7 | 1.1327 | 1.1255 | -0.007 |
| 1.0 | diverged | 1.1183 | — |
| **1.5** | **diverged** | **1.1110** | — |

This was the breakthrough. With 2 frozen blocks, LR=1.0 diverges. Unfreeze everything and it converges fine. The extra capacity from unfreezing absorbs the aggressive learning rate. It also shifts the optimal LR from 0.7 all the way up to 1.5.

### Epoch scaling (4xH200, LR=1.0, freeze=0)
| Epochs | Sliding BPB | TTT time |
|--------|------------|----------|
| 20 | 1.1183 | 569s |
| **30** | **1.1076** | **854s** |

On 8xH100, each TTT epoch runs in ~16.6s (vs 28.5s on 4xH200), so 30 epochs fits within the 10-minute eval budget.

## Architecture

| Component | Detail |
|-----------|--------|
| Layers | 11 |
| Dim | 512 |
| Heads | 8 (4 KV, GQA) |
| MLP | 3x, relu-squared |
| XSA | Last 4 layers |
| EMA | 0.997 |
| Late QAT | Int6 STE when lr_scale < 0.1 |
| Value Embeddings | 128-dim, 5 sets |
| BigramHash | 6144 buckets |
| SmearGate | Learned token blending |
| Warmdown | 1600 iterations |
| Seq length | 2048 (train), 1024 (eval) |
| Sliding window | stride=64 |
| Quantization | Int6 per-row + zstd-22 |

## Training

- Muon optimizer (matrix_lr=0.025, momentum=0.99 with warmup from 0.85)
- AdamW for embeddings/scalars (WD=0.04)
- Flash Attention v3 (Hopper) where available, SDPA fallback
- 6039 steps in 600s on 8xH100 (~99ms/step)

## Evaluation

Three phases, all within the 10-minute eval budget:

1. Int6+zstd quantization roundtrip
2. TTT: SGD(lr=1.0, momentum=0.9), 30 epochs, all blocks unfrozen, score-first
3. Sliding window eval (stride=64, seq_len=1024)

Total eval time: ~591s (TTT 497s + sliding window 92s + roundtrip 2s)

## Run Command

```bash
TTT_ENABLED=1 TTT_LR=1.0 TTT_EPOCHS=30 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 \
VE_ENABLED=1 WARMDOWN_ITERS=1600 NUM_LAYERS=11 XSA_LAST_N=4 \
EMA_ENABLED=1 LATE_QAT=1 BIGRAM_VOCAB_SIZE=6144 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How I Got Here

~20 hours on 4xH200, 54 experiments. Started from the 9L baseline and worked forward:

1. Baseline (9L, no extras): 1.1808
2. +11L, XSA, EMA, QAT: 1.1619
3. +Flash Attention v3: 1.1527
4. +Value Embeddings, warmdown tuning: 1.1521
5. +TTT (LR=0.01, 10ep, freeze=2): 1.1489
6. TTT LR sweep to 0.7: 1.1327
7. Unfreeze all blocks: 1.1255
8. LR=1.5, 20ep: 1.1110
9. 30ep, LR=1.0: 1.1076
10. 8xH100 (more training steps): **1.1124**

Step 7 was where it got fun. Everything before that was incremental hill climbing. Unfreezing all blocks during TTT changed the optimization landscape enough that learning rates that previously diverged started converging, and the whole curve shifted.

## Schrödinger's SOTA

This beats the merged leaderboard (1.1194) by 0.007 BPB. I haven't checked the pending PRs. Until they're merged, this is simultaneously a record and not a record, and I'm choosing to live in that superposition for a bit.

## Credits

Built on the community's collective work, especially PR #414 (signalrush), PR #461 (Christopher-Lee-McClendon), and PR #549 (abaybektursun).
