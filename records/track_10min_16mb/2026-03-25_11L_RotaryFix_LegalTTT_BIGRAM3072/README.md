# 11L Rotary-Fix + Legal TTT + Parallel Muon + BIGRAM-3072

**val_bpb: 1.11869** (3-seed mean, std 0.00024) | **~16.06 MB** | 8×H100 SXM

## Results

| Seed | Post-EMA BPB | int6+lzma BPB | **Legal TTT BPB** |
|------|---|---|---|
| 1337 | 1.1365 | 1.14473 | **1.11877** |
| 42   | 1.1360 | 1.14437 | **1.11836** |
| 2025 | 1.1367 | 1.14493 | **1.11893** |
| **Mean** | 1.1364 | 1.1445 | **1.11869 ± 0.00024** |

TTT gain: −0.0258 BPB (int6 baseline → TTT final)

## Key Contributions vs. Previous Leaderboard Entries

### 1. Rotary NTK-Scaling Bug Fix
Previous entries (including PR #549) hardcode `train_seq_len=1024` in the `Rotary` module while using `train_seq_len=2048` for training. This permanently activates the NTK-aware frequency scaling branch, introducing distorted positional information for all training steps.

```python
# Previous (buggy) — in CausalSelfAttention.__init__ and GPT.__init__
self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)

# This submission (fixed) — train_seq_len propagated dynamically
self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
```

The fix is applied **twice**: in `base_model` (training) and `eval_model` (int6 quantized roundtrip + TTT scoring). The eval_model Rotary fix is a previously unreported bug that affects the quality of the causal TTT scoring window.

### 2. BIGRAM Vocabulary Scaling (3072 vs 1536)
Per ablation from PR #549, BigramHash size 3072 gives an additional −0.0009 BPB. We use this as-is.

### 3. Late QAT Threshold Tuning (0.57 vs 0.15)
Previous entries used `LATE_QAT_THRESHOLD=0.15`, which translates to only ~525 QAT steps before wallclock stop. We set `0.57`, giving ~1700 QAT steps — 3× more training time for quantization-aware adaptation.

## Architecture

| Component | Setting |
|---|---|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | **3072** |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims), **train_seq_len=2048 (fixed)** |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma, **Late QAT threshold 0.57** |
| Optimizer | Parameter Banking + Parallel Muon (lr=0.025, mom=0.99, wd=0.04) |

## Legal TTT Protocol

Backward-looking, score-first TTT (identical to PR #549 framework):

1. Val tokens split into 1,893 non-overlapping 32K-token chunks
2. **For each chunk:**
   - **SCORE:** Sliding window eval under `torch.no_grad()` — no weight mutation
   - **TRAIN:** SGD(lr=0.002, mom=0.9) on already-scored chunk, 3 epochs, all blocks unfrozen
3. Last chunk scored but never trained on

| Parameter | Value |
|---|---|
| Chunk size | 32,768 tokens |
| Optimizer | SGD + momentum(0.9) |
| LR | 0.002 (cosine decay) |
| Epochs/chunk | 3 |
| Frozen blocks | None |
| Gradient clip | 1.0 |
| TTT time | ~425s/seed |

## Run Command

```bash
export TRAIN_SEQ_LEN=2048
export NUM_LAYERS=11
export BIGRAM_VOCAB_SIZE=3072
export XSA_LAST_N=4
export ROPE_DIMS=16
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="9,10"
export EMA_ENABLED=1
export EMA_DECAY=0.997
export SWA_ENABLED=1
export SWA_EVERY=50
export LATE_QAT=1
export LATE_QAT_THRESHOLD=0.57
export TTT_ENABLED=1
export TTT_FREEZE_BLOCKS=0
export TTT_LR=0.002
export TTT_EPOCHS=3
export MUON_WD=0.04
export MATRIX_LR=0.025
export MUON_MOMENTUM=0.99
export ITERATIONS=9000
export WARMDOWN_ITERS=3500
export MAX_WALLCLOCK_SECONDS=600.0
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
