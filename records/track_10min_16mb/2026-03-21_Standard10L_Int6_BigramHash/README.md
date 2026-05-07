This record captures the **Standard 9-Layer Transformer with U-Net Skips, ResidMix, XSA, BigramHash, SWA, and Late QAT**.

## Approach

A 9-layer standard transformer building on the baseline architecture with several additions:

- **U-Net skip connections** — encoder (first 4 layers) stores skip outputs, decoder (last 5 layers) adds them back with learned per-dim `skip_weights`
- **resid_mix (x0 blending)** — every layer mixes the running hidden state with the original post-embedding representation via learned per-dim gates
- **XSA (Exclusive Self-Attention)** on top 3 layers — projects v onto the orthogonal complement of q (GQA-aware), removing self-value bias; zero extra parameters
- **BigramHash(2048, dim=64)** — maps (prev_token, curr_token) pair to a small embedding added before the first layer
- **Late QAT** (activates at 80% of wallclock) — STE fake-quantize for int8 tolerance in the final 20% of training
- **SWA** (Stochastic Weight Averaging, last 30% of training, every 50 steps) — averages 75 snapshots for smoother weight convergence
- **qk_gain_init=1.5** — sharper attention patterns with QK-norm
- **tied_embed_init_std=0.005** — small embedding init for tied weight stability
- **GQA** (4 KV heads, 8 query heads) — reduces KV projection cost

## Configuration

```
NUM_LAYERS=9  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4  MLP_MULT=2
NUM_XSA_LAYERS=3
BIGRAM_BUCKETS=2048  BIGRAM_DIM=64
ROPE_BASE=10000  LOGIT_SOFTCAP=30.0  QK_GAIN_INIT=1.5  TIED_EMBED_INIT_STD=0.005
QAT_ENABLED=1  QAT_START_FRAC=0.80
SWA_ENABLED=1  SWA_START_FRAC=0.3  SWA_INTERVAL=50
EMBED_LR=0.05  MATRIX_LR=0.04  SCALAR_LR=0.04
WARMDOWN_ITERS=1200
TRAIN_BATCH_TOKENS=524288  TRAIN_SEQ_LEN=1024  MAX_WALLCLOCK_SECONDS=600
```

## Command

```bash
torchrun --standalone --nproc_per_node=8 our_train_gpt.py
```

(Run via Modal with 8×H100 SXM5)

## Key metrics

- Timed training stopped due to wallclock cap at 10 min
- Steps completed: 12704 at ~47.23ms/step on 8×H100
- Pre-quant eval: `val_loss:2.0779`, `val_bpb:1.2306`
- Post-quant roundtrip: `val_loss:2.0920`, `val_bpb:1.2390`
- Total submission size: 16,099,282 bytes (16.1 MB)
- Peak memory: ~10295 MiB allocated / 10544 MiB reserved
- SWA averaged 75 snapshots

## Included files

- `train_gpt.py` — training script
- `modal_run.py` — Modal cloud runner
- `submission.json` — leaderboard metadata
- `README.md` — this file
