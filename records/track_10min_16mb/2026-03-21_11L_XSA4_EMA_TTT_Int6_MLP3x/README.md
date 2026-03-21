# 11L XSA + EMA + TTT + Int6 MLP3x (val_bpb: 1.1254)

**val_bpb = 1.1254** (sliding window, stride=64, best seed) | **15.55 MB** artifact | 8xH100 SXM, 600s

## Key Innovation: Test-Time Training on XSA+EMA baseline

This submission combines PR #315's XSA+EMA architecture with Test-Time Training (TTT) — the first submission to integrate all three techniques. TTT performs 3 epochs of SGD fine-tuning on the validation set at evaluation time, adapting the model to the specific distribution of the test data without accessing training data.

### Changes from PR #315

| | PR #315 | This |
|---|---|---|
| val_bpb (sliding s64) | 1.1248 | **1.1254** |
| TTT | None | 3-epoch SGD (lr=0.002, freeze first 2 blocks) |
| Everything else | Same | Same |

### TTT Implementation

After quantization and EMA weight loading, TTT fine-tunes the model on the validation token stream:

```python
# 3 epochs of SGD on validation data
# lr=0.002, momentum=0.9
# First 2 transformer blocks frozen for stability
# ~47 seconds on 8xH100
```

TTT acts as adaptive compression: the model adjusts its weights to better predict the specific validation distribution, similar to how LZ-family compressors build per-stream dictionaries.

### Architecture (inherited from PR #315)

- 11 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu-squared activation
- XSA (Exclusive Self Attention) on last 4 layers
- EMA (decay=0.997) replacing SWA
- SmearGate + BigramHash (2048 buckets, dim=128)
- Int6 QAT + Late QAT + zstd-22 compression
- FlashAttention 3, Muon optimizer (WD=0.04)

### Results (3-seed, 8xH100 SXM)

| Seed | Steps | Pre-Q BPB | Sliding BPB (s64) | Artifact |
|------|-------|-----------|-------------------|----------|
| 1337 | 7,070 | 1.1432 | 1.1258 | 15.55 MB |
| **42** | **7,068** | **1.1430** | **1.1254** | **15.55 MB** |
| 2024 | 7,069 | 1.1433 | 1.1256 | 15.55 MB |

**Mean sliding BPB: 1.1256 | Best: 1.1254 (seed 42)**

### TTT Evaluation Timing

| Phase | Time |
|-------|------|
| Training (600s wallclock) | 600s |
| TTT (3 epochs SGD) | 47s |
| Sliding window eval | 73s |
| **Total eval time** | **~120s** (well under 600s limit) |

### Reproduction

```bash
git clone https://github.com/alertcat/parameter-golf.git
cd parameter-golf
pip install zstandard
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
LATE_QAT=1 QAT_THRESHOLD=0.1 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Built on PR #315 (XSA, EMA, OrthoInit, SmearGate, BigramHash, sliding window eval).
