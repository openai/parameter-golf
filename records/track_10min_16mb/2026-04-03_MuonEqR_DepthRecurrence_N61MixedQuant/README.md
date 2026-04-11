## Record: MuonEq-R + Depth Recurrence + N61 Mixed Int5/Int6 GPTQ (val_bpb: 1.0924)

**val_bpb = 1.0924** (3-seed mean, std 0.0008) | **2.5133 nats** | **~15.98 MB** | 8xH100 SXM, 590s train + ~76s eval | No TTT

Built on [PR #1218](https://github.com/openai/parameter-golf/pull/1218) by @clarkkev (4096-Vocab + 4.0-MLP-mult + 0.085-WD).

Previous: [PR #1019](https://github.com/openai/parameter-golf/pull/1019) (1.1147) -> [PR #1218](https://github.com/openai/parameter-golf/pull/1218) (1.0979) -> [PR #1260](https://github.com/openai/parameter-golf/pull/1260) (1.0929) -> this (1.0924)

### Changes from PR #1218

| | PR #1218 | This |
|---|---|---|
| val_bpb | 1.09785 | **1.09241** |
| Optimizer | Muon | **MuonEq-R** (row-norm before NS5) |
| Depth recurrence | None | **Layers 4,5 repeated** (RECUR_LAYERS=4,5) |
| Recurrence MLP sharing | N/A | **Fully shared** (REPEAT_UNTIE_MLP=none) |
| Mixed quantization | No | **Yes** (61 int6 + 5 int5 via Hessian sensitivity) |
| Recurrence activation | N/A | Step 3000 with 20-step warmup |
| Everything else | Same | Same |

### Changes from PR #1260

| | PR #1260 | This |
|---|---|---|
| val_bpb | 1.09290 | **1.09241** |
| N_INT6_LAYERS | 60 | **61** |
| Seeds | 1337, 42, 0 | **42, 0, 7** |
| Code size | 21,084 | 21,396 |

Key insight: N_INT6=61 (one more int6 layer) improves BPP by ~0.001 per seed with no architecture change. The smaller mini (21,396 bytes vs 87K standalone) creates enough headroom for the extra int6 layer to fit.

### What's New

1. **MuonEq-R** — Row-normalizes gradient matrices before Newton-Schulz orthogonalization in the Muon optimizer. Zero-byte cost, ~0.001 BPB improvement.

2. **Depth Recurrence** — Layers 4 and 5 are repeated once after the initial forward pass (virtual layers 12-13 on top of 11 physical layers). MLP weights are fully shared (REPEAT_UNTIE_MLP=none), adding zero extra parameters. Activated at step 3000 with 20-step warmup. ~0.003 BPB improvement.

3. **N_INT6=61 Mixed Quantization** — Hessian sensitivity ranking: 61 int6 + 5 int5 layers (vs 60+6 in PR #1260). One additional int6 layer improves BPP by ~0.001 with minimal artifact increase. Combined with full GPTQ and brotli-11 compression.

### Carried from PR #1218

- 4096 SentencePiece BPE vocabulary
- 4.0x MLP multiplier with sigmoid-gated activation
- Weight decay 0.085
- Full Hessian GPTQ quantization
- XSA-all-11 attention
- BigramHash embedding (2816x160)
- Sigmoid-gated skip connections + soft-round QAT
- Split-LR training
- Brotli-11 compression with byte shuffle
- EMA (decay 0.997)

### Configuration

```bash
NCCL_NET=Socket \
DATA_DIR=./data \
SEED=42 \
MIXED_QUANT=1 \
N_INT6_LAYERS=61 \
RECUR_LAYERS=4,5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, no TTT)

### Core Results

| Seed | Steps | ms/step | Post-EMA BPB | Sliding BPB | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-------------|-----------------|----------|
| 42 | 5,540 | 106.5 | 1.0985 | 1.0917 | 2.51171 | 15,996,591 |
| 0 | 5,536 | 106.6 | 1.0988 | 1.0923 | 2.51309 | 15,974,481 |
| 7 | 5,538 | 106.6 | 1.0995 | 1.0932 | 2.51522 | 15,982,332 |
| **Mean** | **5,538** | **106.6** | **1.0989** | **1.0924** | **2.51334** | **15,984,468** |

### Supplemental Diagnostics

| Seed | Post-EMA BPB | Roundtrip BPB | Sliding BPB | val_loss (nats) | Code size | Total submission | Train time | Eval time |
|------|--------------|---------------|-------------|-----------------|-----------|------------------|------------|-----------|
| 42 | 1.0985 | 1.1101 | 1.0917 | 2.51171 | 21,396 | 15,996,591 | 590s | 83s |
| 0 | 1.0988 | 1.1108 | 1.0923 | 2.51309 | 21,396 | 15,974,481 | 590s | 83s |
| 7 | 1.0995 | 1.1115 | 1.0932 | 2.51522 | 21,396 | 15,982,332 | 590s | 83s |
| **Mean** | **1.0989** | **1.1108** | **1.0924** | **2.51334** | **21,396** | **15,984,468** | **590s** | **83s** |

### Rule Compliance

- No TTT (no test-time training or adaptation)
- No SLOT (no scored-position lookup table)
- No validation data during training
- No training data during evaluation
- Artifact < 16,000,000 bytes for ALL seeds (max: 15,996,591)
- Train < 600s on 8xH100 SXM (590s)
- Eval < 600s on 8xH100 SXM (~83s)

### Architecture

- 11 layers + 2 virtual (depth recurrence on layers 4,5)
- d_model = 512, MLP 4x (2048), 8 heads, 4 KV heads
- 4096 SentencePiece BPE vocabulary
- BigramHash(2816x160) token embedding
- Sigmoid-gated skip connections with soft-round QAT
- MuonEq-R optimizer with row normalization
- Full Hessian GPTQ with 61 int6 + 5 int5 layers

### Run Command (3-seed loop)

```bash
for SEED in 42 0 7; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  SEED=$SEED \
  MIXED_QUANT=1 \
  N_INT6_LAYERS=61 \
  RECUR_LAYERS=4,5 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee train_seed${SEED}.log
done
```

### Lineage

PR #1019 (1.1147) -> PR #1218 (1.0979) -> PR #1260 (1.0929) -> this (1.0924)

### Credits

- @clarkkev for PR #1218 (4096-Vocab + high-WD architecture)
- @abaybektursun for PR #1019 (GPTQ + XSA + BigramHash baseline)
- @msisovic for PR #1204 (depth recurrence concept)
- @dexhunter for PR #1260 (MuonEq-R + recurrence + mixed quant)

### Included Files

- `train_gpt.py` — full training + quantization + evaluation script (21,396 bytes, self-extracting)
- `train_seed42.log`, `train_seed0.log`, `train_seed7.log` — all seed logs
- `submission.json` — leaderboard metadata
