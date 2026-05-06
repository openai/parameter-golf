## Record: MuonEq-R + Depth Recurrence + WD=0.090 + All-Int6 GPTQ (val_bpb: 1.0912)

**val_bpb = 1.0912** (3-seed mean, std 0.0009) | **2.5106 nats** | **~15.96 MB** | 8xH100 SXM, 590s train + ~76s eval | No TTT

Built on [PR #1218](https://github.com/openai/parameter-golf/pull/1218) by @clarkkev (4096-Vocab + 4.0-MLP-mult).

Previous: [PR #1019](https://github.com/openai/parameter-golf/pull/1019) (1.1147) -> [PR #1218](https://github.com/openai/parameter-golf/pull/1218) (1.0979) -> [PR #1260](https://github.com/openai/parameter-golf/pull/1260) (1.0929) -> this (1.0912)

### Changes from PR #1218

| | PR #1218 | This |
|---|---|---|
| val_bpb | 1.09785 | **1.09124** |
| Optimizer | Muon | **MuonEq-R** (row-norm before NS5) |
| Depth recurrence | None | **Layers 4,5 repeated** |
| Weight decay | 0.085 | **0.090** |
| Mixed quantization | No | **All int6** (66/66 layers) |
| Everything else | Same | Same |

### Key Innovation: WD-Quantization Synergy

The critical insight: **higher weight decay (0.090 vs 0.085) produces smaller weights that compress 5% better under brotli-11**, creating enough artifact headroom to keep **ALL 66 layers at int6 precision** (vs 60-61 int6 in previous PRs). The extra quantization precision more than recovers the BPP cost of higher weight decay:

| Config | WD | N_INT6 | Artifact | BPB (seed 42) |
|--------|-----|--------|----------|---------------|
| PR #1260 | 0.085 | 60 | 15,981K | 1.09217 |
| PR #1279 | 0.085 | 61 | 15,997K | 1.09170 |
| **This** | **0.090** | **66** | **15,967K** | **1.09057** |

### What's New

1. **WD=0.090** — Increased from 0.085. Higher WD reduces weight magnitudes, improving brotli-11 compression by ~5%. This creates ~280K bytes of artifact headroom (vs 3K margin at WD=0.085/N61).

2. **All-Int6 GPTQ** — With the compression headroom from WD=0.090, we can keep ALL 66 weight layers at int6 precision (clip_range=31). No layers need to be demoted to int5. This is the theoretical maximum quantization quality for the given architecture.

3. **MuonEq-R** — Row-normalizes gradient matrices before Newton-Schulz orthogonalization. Zero-byte cost, ~0.001 BPB improvement.

4. **Depth Recurrence** — Layers 4,5 repeated with fully shared MLP (zero extra params). ~0.003 BPB improvement.

### Carried from PR #1218

- 4096 SentencePiece BPE vocabulary
- 4.0x MLP multiplier with sigmoid-gated activation
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
N_INT6_LAYERS=66 \
MUON_WD=0.090 \
EMBED_WD=0.090 \
RECUR_LAYERS=4,5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, no TTT)

### Core Results

| Seed | Steps | ms/step | Post-EMA BPB | Sliding BPB | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-------------|-----------------|----------|
| 42 | 5,540 | 106.5 | 1.0990 | 1.0906 | 2.50910 | 15,967,483 |
| 0 | 5,536 | 106.6 | 1.0992 | 1.0908 | 2.50973 | 15,962,242 |
| 1337 | 5,538 | 106.6 | 1.0998 | 1.0923 | 2.51309 | 15,959,253 |
| **Mean** | **5,538** | **106.6** | **1.0993** | **1.0912** | **2.51064** | **15,962,993** |

### Supplemental Diagnostics

| Seed | Post-EMA BPB | Roundtrip BPB | Sliding BPB | val_loss (nats) | Code size | Total submission | Train time | Eval time |
|------|--------------|---------------|-------------|-----------------|-----------|------------------|------------|-----------|
| 42 | 1.0990 | 1.1081 | 1.0906 | 2.50910 | 21,396 | 15,967,483 | 590s | 83s |
| 0 | 1.0992 | 1.1082 | 1.0908 | 2.50973 | 21,396 | 15,962,242 | 590s | 83s |
| 1337 | 1.0998 | 1.1101 | 1.0923 | 2.51309 | 21,396 | 15,959,253 | 590s | 83s |
| **Mean** | **1.0993** | **1.1088** | **1.0912** | **2.51064** | **21,396** | **15,962,993** | **590s** | **83s** |

### Rule Compliance

- No TTT (no test-time training or adaptation)
- No SLOT (no scored-position lookup table)
- No validation data during training
- No training data during evaluation
- Artifact < 16,000,000 bytes for ALL seeds (max: 15,967,483, min margin: 32,517)
- Train < 600s on 8xH100 SXM (590s)
- Eval < 600s on 8xH100 SXM (~83s)

### Architecture

- 11 layers + 2 virtual (depth recurrence on layers 4,5)
- d_model = 512, MLP 4x (2048), 8 heads, 4 KV heads
- 4096 SentencePiece BPE vocabulary
- BigramHash(2816x160) token embedding
- Sigmoid-gated skip connections with soft-round QAT
- MuonEq-R optimizer with row normalization
- Full Hessian GPTQ — all 66 layers at int6 precision
- Weight decay 0.090 (muon + embed)

### Run Command (3-seed loop)

```bash
for SEED in 42 0 1337; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  SEED=$SEED \
  MIXED_QUANT=1 \
  N_INT6_LAYERS=66 \
  MUON_WD=0.090 \
  EMBED_WD=0.090 \
  RECUR_LAYERS=4,5 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee train_seed${SEED}.log
done
```

### Lineage

PR #1019 (1.1147) -> PR #1218 (1.0979) -> PR #1260 (1.0929) -> this (1.0912)

### Credits

- @clarkkev for PR #1218 (4096-Vocab + high-WD architecture — the WD insight)
- @abaybektursun for PR #1019 (GPTQ + XSA + BigramHash baseline)
- @msisovic for PR #1204 (depth recurrence concept)
- @dexhunter for PR #1260 (MuonEq-R + recurrence + mixed quant)

### Included Files

- `train_gpt.py` — full training + quantization + evaluation script (21,396 bytes, self-extracting)
- `train_seed42.log`, `train_seed0.log`, `train_seed1337.log` — all seed logs
- `submission.json` — leaderboard metadata
