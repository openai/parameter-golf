**val_bpb: 1.1720** | **19.4 MB** (unlimited compute) | 1xA6000, 9500 steps, 14.5hr

# 11L Production: Value Residual + Gated Attention

Non-record submission demonstrating two novel architectural techniques — **Value Residual** and **Gated Attention** — integrated into the full community meta-stack on an 11-layer MLP3x model.

## Novel Contributions

### Value Residual (ResFormer) — -0.015 BPB

From [arXiv:2410.17897](https://arxiv.org/abs/2410.17897) (ACL 2025). Caches the value vectors from layer 0 and mixes them into all subsequent layers via per-layer learnable scalars (22 total params). This creates a "residual highway" for value information, preventing representation collapse in deeper layers.

- Impact: **-0.015 BPB** (ablated on 9L v1024 baseline)
- Parameters added: 22 (one scalar per layer pair)
- Implementation: ~15 lines of code

### Gated Attention — -0.003 BPB

From [arXiv:2505.06708](https://arxiv.org/abs/2505.06708). Adds a per-head sigmoid gate applied after scaled dot-product attention, allowing each head to dynamically suppress uninformative attention patterns (attention sinks). Total added params: 37K.

- Impact: **-0.003 BPB** (ablated on 9L v1024 baseline)
- Parameters added: ~37K
- Implementation: ~10 lines of code

### Combined Impact

Both techniques stack **additively**: -0.0172 BPB combined vs -0.015 (VR) + -0.003 (GA) individually.

## Ablation Table (9L v1024 baseline, 1000 steps, 131K batch, 1x3090)

| Config | val_bpb | Delta |
|--------|---------|-------|
| Control (SG+BH+OI+WD0.04) | 1.4697 | — |
| + Gated Attention | 1.4665 | -0.0032 |
| + Value Residual | 1.4546 | -0.0151 |
| + Value Residual + Gated Attention | 1.4525 | **-0.0172** |

## Production Configuration

Full community meta-stack with VR + GA on 11 layers:

- **Architecture**: 11L, 512d, 8 heads / 4 KV heads, MLP 3x, tied embeddings
- **Novel**: Value Residual + Gated Attention
- **Community stack**: SmearGate, BigramHash(2048, dim=128), OrthoInit, Weight Decay 0.04
- **Advanced**: XSA (last 4 layers), EMA (decay=0.997), Partial RoPE (16/64 dims), LN Scale, Logit Softcap (30.0)
- **Optimizer**: Muon (momentum=0.99, warmup from 0.92 over 1500 steps, backend=5) + AdamW for scalars
- **Training**: 9500 steps, 524K batch tokens, seq_len=1024, warmdown 3000 steps
- **Quantization**: int6 + zstd compression
- **Hardware**: 1x NVIDIA RTX A6000 (48GB), ~14.5 hours

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1710 |
| Post-quant val_bpb (int6+zstd) | **1.1720** |
| Quant gap | 0.0010 |
| Artifact size | 19.4 MB |
| Training time | 52,327s (~14.5hr) |
| Step average | 5508 ms |
| Peak memory | 23,922 MiB |

### Training Progression

| Step | val_bpb |
|------|---------|
| 0 | 4.1028 |
| 1000 | 1.3532 |
| 2000 | 1.3029 |
| 3000 | 1.2792 |
| 4000 | 1.2699 |
| 5000 | 1.2643 |
| 6000 | 1.2641 |
| 7000 | 1.2485 |
| 8000 | 1.2213 |
| 9000 | 1.1878 |
| 9500 | 1.1710 |

## Community Adoption

These techniques have been independently adopted and validated by multiple submissions:

- **TrigramHash + VR + GradQuant + AdamW TTT** — 1.1101 BPB (record-tier, 3-seed mean 1.1132)
- **seq4096 + VRL + XSA + cross-doc TTT** — 1.1839 BPB
- **Catalytic Residuals + VR + GA + BigramHash(10240) + 12L** — 1.1690 BPB
- **11L Frontier Stack + VR + GA** — independent replication

## Reproducibility

```bash
TORCHDYNAMO_DISABLE=1 \
VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=9500 WARMDOWN_ITERS=3000 WARMUP_STEPS=20 MAX_WALLCLOCK_SECONDS=0 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_BACKEND_STEPS=5 GRAD_CLIP_NORM=0.3 \
WEIGHT_DECAY_MUON=0.04 WEIGHT_DECAY_ADAM=0.04 \
SMEAR_GATE=1 BIGRAM_HASH=1 BIGRAM_BUCKETS=2048 BIGRAM_DIM=128 ORTHO_INIT=1 \
XSA_LAYERS=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
PARTIAL_ROPE_DIMS=16 LN_SCALE=1 LOGIT_SOFTCAP=30.0 \
GATED_ATTENTION=1 VALUE_RESIDUAL=1 \
QUANT_BITS=6 \
EVAL_SEQ_LEN=1024 EVAL_STRIDE=128 VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024_full \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee train_log.txt
```

## Files

- `README.md` — this writeup
- `submission.json` — leaderboard metadata
- `train_gpt.py` — training script used for the production run
- `train.log` — complete training log
